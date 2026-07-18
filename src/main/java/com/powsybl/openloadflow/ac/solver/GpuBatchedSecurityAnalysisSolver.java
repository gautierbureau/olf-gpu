/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver;

import com.powsybl.commons.report.ReportNode;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivation;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataCache;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcStructures;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.TargetVector;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfShunt;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Solver for batched N-1 security analysis on the GPU: runs hundreds to thousands of
 * post-contingency AC power flows in one batched Newton-Raphson, with per-scenario
 * convergence retirement (converged scenarios stop filling/solving early). Expected
 * throughput gain: one-to-two orders of magnitude over sequential per-contingency solves,
 * amortizing the device context creation and cuDSS symbolic analysis over the entire
 * contingency set.
 *
 * <p>Scope: single-branch outages that keep the main synchronous component intact
 * (checked per scenario; contingencies that would island or fragment the network are
 * rejected). Distributed slack, reactive limits, and secondary controls are not yet
 * device-resident (scope: plain N-1 with base-case participation factors).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuBatchedSecurityAnalysisSolver {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuBatchedSecurityAnalysisSolver.class);

    /** System property: an explicit GPU batch chunk size (scenarios). When unset, the chunk is auto-sized from free device memory. */
    // MAX_VOLTAGE_CHANGE state-vector scaling for the batched device solve (mirrors the CPU's
    // MaxVoltageChangeStateVectorScaling PER SCENARIO). Set by GpuBatchedAcSecurityAnalysis from the
    // resolved OLF parameters before each run; 0 = raw full Newton steps (the historical behavior).
    private static volatile int batchLsMode;
    private static volatile double batchMaxDv;
    private static volatile double batchMaxDphi;

    public static void setStateVectorScaling(int lsMode, double maxDv, double maxDphi) {
        batchLsMode = lsMode;
        batchMaxDv = maxDv;
        batchMaxDphi = maxDphi;
    }

    // Transformer voltage-control orchestration (AFTER_GENERATOR mode ported per scenario as a
    // Java orchestrator around repeated batched solves). Set by GpuBatchedAcSecurityAnalysis when
    // the resolved outer loops include TransformerVoltageControl and the network is in stage-1
    // scope (no generator controls under the transformer-controlled voltage level — see
    // transformerControlSupported); the guard falls back to CPU otherwise.
    private static volatile boolean batchTransformerControl;

    public static void setTransformerControl(boolean enabled) {
        batchTransformerControl = enabled;
    }

    /** One voltage-controlling transformer, precomputed for the per-scenario orchestrator:
     *  rho row (== r1 col, self-paired), controlled-bus V col, target/deadband, continuous r1
     *  range, discrete tap r1 values, its element index (per-scenario outage check), and the
     *  row-mode/target DELTAS between control-off (base) and control-on activations. */
    record TransformerController(int rhoRow, int vCol, double targetV, double halfDeadband,
                                 double minR1, double maxR1, double[] taps, int elemIndex,
                                 int[] deltaRows, int[] deltaModesOn, double[] deltaTargetsOn) {
        double closestTap(double r1) {
            if (taps.length == 0) {
                return r1;                                   // no tap table: keep the continuous ratio
            }
            double best = taps[0];
            for (double t : taps) {
                if (Math.abs(t - r1) < Math.abs(best - r1)) {
                    best = t;
                }
            }
            return best;
        }
    }

    /** Stage-1 scope check: the CPU loop's GeneratorVoltageControlManager disables generator
     *  voltage controls at buses under the max transformer-controlled nominal voltage. Porting
     *  the frozen-Q flip is stage 2 — until then a network that would NEED it must stay on the
     *  CPU path (the provider's guard calls this). */
    public static boolean transformerControlSupported(LfNetwork network) {
        double maxControlledNominalV = network.<LfBranch>getControllerElements(com.powsybl.openloadflow.network.VoltageControl.Type.TRANSFORMER).stream()
                .flatMap(b -> b.getVoltageControl().stream())
                .map(vc -> vc.getControlledBus().getNominalV())
                .mapToDouble(Double::doubleValue).max().orElse(-1);
        if (maxControlledNominalV < 0) {
            return true;                                     // no transformer controls at all: loop is a no-op
        }
        boolean genUnderThreshold = network.getControlledBuses(com.powsybl.openloadflow.network.VoltageControl.Type.GENERATOR).stream()
                .anyMatch(bus -> bus.getNominalV() <= maxControlledNominalV);
        if (genUnderThreshold) {
            LOGGER.info("Transformer control orchestration out of stage-1 scope: generator voltage controls "
                    + "exist under the transformer-controlled nominal voltage ({} kV) — CPU fallback", maxControlledNominalV);
        }
        return !genUnderThreshold;
    }

    /** Build the orchestrator's controller pack by TOGGLING each control on the shared equation
     *  system and diffing the activation — the ON recipe is exactly what the engine itself would
     *  produce, no hand-crafted modes. The network is restored before returning. */
    private static List<TransformerController> buildTransformerControllers(
            LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
            TargetVector<AcVariableType, AcEquationType> targetVector,
            GpuAcActivation baseActivation, double[] baseTargets, GpuBatchElementMapper elementMapper) {
        List<TransformerController> out = new ArrayList<>();
        java.util.Map<Integer, Integer> busVCols = new java.util.HashMap<>();
        java.util.Map<Integer, Integer> branchR1Cols = new java.util.HashMap<>();
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            if (v.getType() == AcVariableType.BUS_V) {
                busVCols.put(v.getElementNum(), v.getRow());
            } else if (v.getType() == AcVariableType.BRANCH_RHO1) {
                branchR1Cols.put(v.getElementNum(), v.getRow());
            }
        }
        for (LfBranch branch : network.<LfBranch>getControllerElements(com.powsybl.openloadflow.network.VoltageControl.Type.TRANSFORMER)) {
            var vcOpt = branch.getVoltageControl();
            if (vcOpt.isEmpty() || !branch.isConnectedAtBothSides()) {
                continue;
            }
            var vc = vcOpt.orElseThrow();
            Integer r1Col = branchR1Cols.get(branch.getNum());
            Integer vCol = busVCols.get(vc.getControlledBus().getNum());
            int elemIndex = elementMapper.getElementIndex(branch);
            if (r1Col == null || vCol == null || elemIndex < 0) {
                LOGGER.info("Transformer controller {} not representable on the GPU batch — skipped (results follow the base tap)", branch.getId());
                continue;
            }
            double halfDeadband = vc.getTargetDeadband().orElse(0.1 / vc.getControlledBus().getNominalV()) / 2;
            var piModel = branch.getPiModel();
            double[] taps = tapR1Values(piModel);
            boolean wasEnabled = branch.isVoltageControlEnabled();
            branch.setVoltageControlEnabled(true);
            GpuAcActivation onAct = GpuAcActivation.fromEquationSystem(network, es);
            double[] onTargets = onAct.applyTargets(targetVector.getArray());
            branch.setVoltageControlEnabled(wasEnabled);
            List<Integer> dRows = new ArrayList<>();
            for (int r = 0; r < baseTargets.length; r++) {
                if (onAct.rowMode()[r] != baseActivation.rowMode()[r] || onTargets[r] != baseTargets[r]) {
                    dRows.add(r);
                }
            }
            int[] deltaRows = dRows.stream().mapToInt(Integer::intValue).toArray();
            int[] deltaModesOn = new int[deltaRows.length];
            double[] deltaTargetsOn = new double[deltaRows.length];
            for (int k = 0; k < deltaRows.length; k++) {
                deltaModesOn[k] = onAct.rowMode()[deltaRows[k]];
                deltaTargetsOn[k] = onTargets[deltaRows[k]];
            }
            out.add(new TransformerController(r1Col, vCol, vc.getTargetValue(), halfDeadband,
                    piModel.getMinR1(), piModel.getMaxR1(), taps, elemIndex,
                    deltaRows, deltaModesOn, deltaTargetsOn));
        }
        return out;
    }

    /** Discrete tap r1 values, snapshotted read-only (reflection on PiModelArray.models; an
     *  inaccessible or non-array model yields an empty table = keep the continuous ratio). */
    private static double[] tapR1Values(Object piModel) {
        try {
            if (!"PiModelArray".equals(piModel.getClass().getSimpleName())) {
                return new double[0];
            }
            java.lang.reflect.Field f = piModel.getClass().getDeclaredField("models");
            f.setAccessible(true);
            List<?> models = (List<?>) f.get(piModel);
            double[] taps = new double[models.size()];
            for (int i = 0; i < models.size(); i++) {
                taps[i] = (double) models.get(i).getClass().getMethod("getR1").invoke(models.get(i));
            }
            return taps;
        } catch (ReflectiveOperationException | RuntimeException e) {
            LOGGER.info("Transformer tap table not accessible ({}) — continuous ratios will not be rounded", e.toString());
            return new double[0];
        }
    }

    public static final String MAX_BATCH_PROPERTY = "olf.gpu.sa.maxBatch";
    /** Fallback chunk size used when auto-sizing is unavailable (no GPU / memory-query failure). */
    public static final int DEFAULT_MAX_BATCH = 256;
    /**
     * System property: the native batched-solve mode (cuDSS >= 0.8, all compiled in, runtime-selected).
     * 0 = per-scenario refactor+solve loop (memory-light serial fallback; never benchmarked as best);
     * 1 = uniform batched cuDSS API (one batched factorization+solve per iteration — faster per-iteration
     * SOLVE, but its nS-block ANALYSIS costs ~1.6 s cold, the single-pass loser even with the context cache);
     * 2 = block-diagonal single matrix, factorize-once + refactorize-every (the refactor-every strategy;
     * measured ~parity with mode 1 since cuDSS refactorization is not cheaper than factorization);
     * 3 = UBATCH (single per-block matrix + CUDSS_CONFIG_UBATCH_SIZE), the DEFAULT
     * mechanism: a cheap one-block ANALYSIS (~150 ms vs mode 1's ~1.6 s) wins the PRIMARY single-cold-pass
     * use case. The CHUNK SIZE is decisive on cuDSS 0.8: its UBATCH phases carry a per-launch fixed cost
     * that amortizes with the batch (pegase9241, K=4, warm ms/scenario: chunk 32 = 17.8, 125 = 6.35,
     * 500 = 4.87 — the fastest measured configuration on any mode/version; cuDSS 0.7.x is
     * chunk-size-insensitive at ~8.3 instead). Keep mode 3 with LARGE chunks (the memory auto-sizing)
     * on 0.8; mode 1 is the small-chunk alternative there (~6.2 at chunk 32).
     * 4 = UBATCH + iter0-only (factor at each chunk's iter 0, SOLVE-only after) and
     * 5 = UBATCH + refactorize-after: EXPERIMENTAL factor-skip probes, kept for retesting on other
     * grids/formulations. Measured non-beneficial on case9241pegase (mode 5 = mode 3 exactly; mode 4 is
     * ~2x faster per iter but needs ~3x more iters to converge, a net wash) — do NOT use in production.
     */
    public static final String BATCH_MODE_PROPERTY = "olf.gpu.batchMode";

    /** Resolve the native batched-solve mode from {@value #BATCH_MODE_PROPERTY} (default 3, UBATCH). */
    public static int resolveBatchMode() {
        return Integer.getInteger(BATCH_MODE_PROPERTY, 3);
    }

    /**
     * Cached batched native context, reused across {@code solveBatchN1} calls whose extracted
     * {@link GpuAcData}, chunk size and mode are identical. The cuDSS symbolic ANALYSIS (and the
     * uploaded per-element static data) depend only on the network pattern + admittances — NOT on
     * the operating point (injections/targets are passed per solve) — so successive SA runs on the
     * same network reuse the analyzed, allocated context instead of rebuilding it (the per-run
     * extraction/build/analysis + device alloc/free was ~25% of a single-run wall time). The native
     * handle is freed when a non-matching context evicts it, by {@link #clearContextCache()}, or at
     * JVM shutdown. NOT thread-safe across different networks (single-entry cache); guarded so the
     * native handle is never corrupted.
     */
    private static final class CachedBatchContext {
        final long handle;
        final GpuAcData data;
        final int chunkSize;
        final int mode;

        CachedBatchContext(long handle, GpuAcData data, int chunkSize, int mode) {
            this.handle = handle;
            this.data = data;
            this.chunkSize = chunkSize;
            this.mode = mode;
        }
    }

    private static CachedBatchContext cachedContext;

    static {
        Runtime.getRuntime().addShutdownHook(new Thread(GpuBatchedSecurityAnalysisSolver::clearContextCache,
                "olf-gpu-batch-context-cleanup"));
    }

    /** Acquire a batched native context for this (data, chunkSize, mode), reusing the cache when it matches. */
    private static synchronized long acquireBatchContext(GpuAcData data, int chunkSize, int mode, int n) {
        if (cachedContext != null && cachedContext.chunkSize == chunkSize && cachedContext.mode == mode
                && cachedContext.data.sameAs(data)) {
            LOGGER.debug("Batched N-1: reusing cached GPU context (chunk {}, mode {})", chunkSize, mode);
            return cachedContext.handle;
        }
        if (cachedContext != null) {                         // evict a stale context (different network/chunk/mode)
            GpuAcNewtonSolver.destroyBatchContext(cachedContext.handle);
            cachedContext = null;
        }
        // the shared structure comes from the per-extraction cache — when the base load flow ran
        // on the GPU with the same extraction (GpuAcDataCache hand-off), this is a device-side hit
        long handle = GpuAcNewtonSolver.createBatchContext(GpuAcStructures.handle(n, data), chunkSize, mode,
                data.rlVrow(), data.rlMinQ(), data.rlMaxQ(), data.rlLoadQ(), data.rlVtarget(),
                data.dsPhiRow(), data.dsFactor(), data.dsSlackPhiRow(), data.dsSlackTargetP(),
                data.dsBaseTargetP(), data.dsMaxDelta(), data.dsMinDelta());
        cachedContext = new CachedBatchContext(handle, data, chunkSize, mode);
        return handle;
    }

    /** Free the cached batched native context, if any. Safe to call repeatedly. */
    public static synchronized void clearContextCache() {
        if (cachedContext != null) {
            GpuAcNewtonSolver.destroyBatchContext(cachedContext.handle);
            cachedContext = null;
        }
    }

    /** A contingency to analyze on the device: exactly one of — 1–2 branch outages (N-1 / N-2, disabled in
     *  the per-scenario element mask), a single generator loss ({@code lostGenerator}, applied as a per-scenario
     *  target override), or one+ fixed-shunt disconnections ({@code outagedShunts}, disabled in the element
     *  mask like branches). */
    /** Separator inside a (contingency, strategy) pair's BatchContingency id — pairs are solved in a
     *  SECOND phase, WARM-CONTINUED from their parent contingency's state and reactive-limits pinning. */
    public static final char STRATEGY_KEY_SEP = '\u0001';

    /** A lost load's per-bus injection deltas (pu): removing it RAISES the bus P/Q targets. */
    public record LoadLoss(LfBus bus, double p, double q) {
    }

    /** A per-scenario shunt admittance change (pu deltas ADDED to the packed g/b): section shifts
     *  in either direction; full disconnection = minus the section admittance; reconnection = plus. */
    public record ShuntShift(LfShunt shunt, double dg, double db) {
    }

    public record BatchContingency(String contingencyId, List<LfBranch> outagedBranches, List<LfGenerator> lostGenerators,
                                   List<LoadLoss> lostLoads, List<ShuntShift> shuntShifts, List<LfShunt> outagedShunts) {
        public BatchContingency {
            Objects.requireNonNull(contingencyId);
            Objects.requireNonNull(outagedBranches);
            Objects.requireNonNull(lostGenerators);
            Objects.requireNonNull(lostLoads);
            Objects.requireNonNull(shuntShifts);
            Objects.requireNonNull(outagedShunts);
            boolean isInjection = !lostGenerators.isEmpty() || !lostLoads.isEmpty() || !shuntShifts.isEmpty();
            boolean isShunt = !outagedShunts.isEmpty();
            boolean isBranch = !outagedBranches.isEmpty();
            // Batchable shapes: 1-2 branches, N generators, N loads, and branch+injection COMBOS
            // (independent mechanisms: element mask vs target/row-mode overrides); fixed shunts
            // stay exclusive (their DISTR_Q interplay is validated separately).
            if ((!isInjection && !isShunt && !isBranch) || outagedBranches.size() > 2
                    || (isShunt && (isInjection || isBranch))) {
                throw new IllegalArgumentException(
                        "a batched contingency is 1-2 branches and/or injection loss(es), or fixed shunt(s) alone");
            }
            outagedBranches = List.copyOf(outagedBranches);
            lostGenerators = List.copyOf(lostGenerators);
            lostLoads = List.copyOf(lostLoads);
            shuntShifts = List.copyOf(shuntShifts);
            outagedShunts = List.copyOf(outagedShunts);
        }

        /** A branch outage (N-1 or N-2). */
        public BatchContingency(String contingencyId, List<LfBranch> outagedBranches) {
            this(contingencyId, outagedBranches, List.of(), List.of(), List.of(), List.of());
        }

        /** Convenience for the common single-branch (N-1) case. */
        public BatchContingency(String contingencyId, LfBranch outagedBranch) {
            this(contingencyId, List.of(Objects.requireNonNull(outagedBranch)), List.of(), List.of(), List.of(), List.of());
        }

        /** A combined branch + generator/load contingency. */
        public static BatchContingency branchesAndInjections(String contingencyId, List<LfBranch> branches,
                                                             List<LfGenerator> gens, List<LoadLoss> loads,
                                                             List<ShuntShift> shifts) {
            return new BatchContingency(contingencyId, branches, gens, loads, shifts, List.of());
        }

        /** A generator loss. */
        public static BatchContingency generator(String contingencyId, LfGenerator lostGenerator) {
            return new BatchContingency(contingencyId, List.of(), List.of(Objects.requireNonNull(lostGenerator)), List.of(), List.of(), List.of());
        }

        /** A fixed-shunt disconnection (the shunt element is masked off, like a branch outage). */
        public static BatchContingency shunt(String contingencyId, LfShunt outagedShunt) {
            return new BatchContingency(contingencyId, List.of(), List.of(), List.of(), List.of(), List.of(Objects.requireNonNull(outagedShunt)));
        }
    }

    /** The per-scenario target/row-mode changes a generator loss applies on the device: remove the lost
     *  generator's P from its bus's P target ({@code pRow}→{@code pTarget}); if the loss un-controls the bus
     *  ({@code toPq}) flip its BUS_V row to the reactive-power equation and set {@code vRow}→{@code qTarget}.
     *  Mirrors {@code LfContingency.processLostGenerators}.
     *
     *  <p>When the lost generator sits on a distributed-slack participant bus, {@code dsBus} is that bus's
     *  DS-pack index (else -1) and the {@code ds*Delta} fields carry the reweight the device applies: the
     *  bus's participation factor, base target P, headroom and footroom each shrink by the lost gen's share,
     *  and the per-scenario factor sum drops by {@code dsFactorDelta} so the other buses absorb that share.
     *  This is only valid when the slack itself does not participate (otherwise the loss is rejected). */
    public record GenOutage(int pRow, double pTarget, int vRow, double qTarget, boolean toPq, boolean qOverride,
                            int dsBus, double dsFactorDelta, double dsBaseDelta, double dsMaxDelta, double dsMinDelta) {
    }

    /** Convergence status: the scenario converged within the tolerance. */
    public static final int STATUS_CONVERGED = 0;
    /** Convergence status: the scenario hit the iteration cap without converging. */
    public static final int STATUS_MAX_ITERATION = 1;
    /** Convergence status: the scenario diverged (NaN/Inf mismatch). */
    public static final int STATUS_DIVERGED = -1;
    /**
     * Convergence status: the contingency was not batched on the GPU (it fragments the
     * network or its branch is absent from the GPU model) and must be solved by the
     * sequential CPU security-analysis path. The state is the base case (unchanged).
     */
    public static final int STATUS_REJECTED = -2;

    /**
     * The per-scenario outcome: the converged state, iteration count, final mismatch and
     * the convergence status (see the {@code STATUS_*} constants).
     */
    public record BatchScenarioResult(String contingencyId, double[] state, int iterations,
                                      double finalMismatch, int convergenceStatus) {
        public BatchScenarioResult {
            Objects.requireNonNull(contingencyId);
            Objects.requireNonNull(state);
        }

        public boolean hasConverged() {
            return convergenceStatus == STATUS_CONVERGED;
        }

        /** True if this contingency must fall back to the sequential CPU path. */
        public boolean needsCpuFallback() {
            return convergenceStatus == STATUS_REJECTED;
        }
    }

    /**
     * Solver for batched contingencies: given a base-case state and a list of
     * single-branch contingencies, extract one batched GPU context and run all
     * scenarios in a single Newton-Raphson loop.
     *
     * @param network         the AC network (base case)
     * @param baseState       the converged base-case state vector
     * @param contingencies   list of single-branch outages to analyze
     * @param equationSystem  the base-case equation system
     * @param targetVector    the base-case targets
     * @param parameters      Newton-Raphson stopping criteria
     * @param reportNode      for logging and diagnostics
     * @return per-scenario results (state, iterations, convergence status)
     */
    public static List<BatchScenarioResult> solveBatchN1(LfNetwork network, double[] baseState,
                                                         List<BatchContingency> contingencies,
                                                         EquationSystem<AcVariableType, AcEquationType> equationSystem,
                                                         TargetVector<AcVariableType, AcEquationType> targetVector,
                                                         NewtonRaphsonParameters parameters,
                                                         ReportNode reportNode) {
        return solveBatchN1(network, baseState, contingencies, equationSystem, targetVector, parameters,
                GpuNewtonRaphson.GPU_TOLERANCE, true, true, reportNode);
    }

    /**
     * As {@link #solveBatchN1(LfNetwork, double[], List, EquationSystem, TargetVector, NewtonRaphsonParameters, ReportNode)}
     * but with the run's actual outer loops declared. {@code distributedSlack} / {@code reactiveLimits} must
     * match the outer loops OLF applied to the base case: the device runs the distributed-slack pass only when
     * {@code distributedSlack} and the PV→PQ reactive-limits pass only when {@code reactiveLimits}, so a
     * disabled outer loop is never run on the device. Passing the wrong flags diverges from the CPU under a
     * non-trivial slack mismatch or a reactive-limit violation.
     */
    public static List<BatchScenarioResult> solveBatchN1(LfNetwork network, double[] baseState,
                                                         List<BatchContingency> contingencies,
                                                         EquationSystem<AcVariableType, AcEquationType> equationSystem,
                                                         TargetVector<AcVariableType, AcEquationType> targetVector,
                                                         NewtonRaphsonParameters parameters,
                                                         boolean distributedSlack, boolean reactiveLimits,
                                                         ReportNode reportNode) {
        return solveBatchN1(network, baseState, contingencies, equationSystem, targetVector, parameters,
                GpuNewtonRaphson.GPU_TOLERANCE, distributedSlack, reactiveLimits, reportNode);
    }

    /**
     * As the {@code (distributedSlack, reactiveLimits)} overload but with the device outer-loop tolerances
     * declared explicitly (per-unit) so the device aligns to the CPU loadflow's outer-loop criteria. Used by
     * the production SA path, which reads them from the run's loadflow parameters.
     */
    public static List<BatchScenarioResult> solveBatchN1(LfNetwork network, double[] baseState,
                                                         List<BatchContingency> contingencies,
                                                         EquationSystem<AcVariableType, AcEquationType> equationSystem,
                                                         TargetVector<AcVariableType, AcEquationType> targetVector,
                                                         NewtonRaphsonParameters parameters,
                                                         boolean distributedSlack, boolean reactiveLimits,
                                                         double reactiveLimitTol, double slackDistTol,
                                                         ReportNode reportNode) {
        return solveBatchN1(network, baseState, contingencies, equationSystem, targetVector, parameters,
                GpuNewtonRaphson.GPU_TOLERANCE, distributedSlack, reactiveLimits, reactiveLimitTol, slackDistTol,
                reportNode);
    }

    /**
     * As the {@code (reactiveLimitTol, slackDistTol)} overload but with an explicit {@code convEpsPerEq} for the
     * device inner Newton (device tol defaults to {@link GpuNewtonRaphson#GPU_TOLERANCE} as the fixed-iter
     * sentinel). {@code convEpsPerEq > 0} makes the device stop on OLF's exact {@code ‖F‖₂ < sqrt(convEpsPerEq²·n)}
     * so its reactive-limit switch decisions are taken at the SAME convergence depth as the CPU — used by the
     * production SA path so GPU and CPU agree beyond the outer-loop tolerance band.
     */
    public static List<BatchScenarioResult> solveBatchN1(LfNetwork network, double[] baseState,
                                                         List<BatchContingency> contingencies,
                                                         EquationSystem<AcVariableType, AcEquationType> equationSystem,
                                                         TargetVector<AcVariableType, AcEquationType> targetVector,
                                                         NewtonRaphsonParameters parameters,
                                                         boolean distributedSlack, boolean reactiveLimits,
                                                         double reactiveLimitTol, double slackDistTol,
                                                         double convEpsPerEq, ReportNode reportNode) {
        return solveBatchN1(network, baseState, contingencies, equationSystem, targetVector, parameters,
                GpuNewtonRaphson.GPU_TOLERANCE, distributedSlack, reactiveLimits, reactiveLimitTol, slackDistTol,
                convEpsPerEq, reportNode);
    }

    /**
     * As {@link #solveBatchN1(LfNetwork, double[], List, EquationSystem, TargetVector, NewtonRaphsonParameters, ReportNode)}
     * but with an explicit mismatch tolerance. Passing {@code tol <= 0} selects the native
     * DEVICE-SIDE fixed-iteration mode: every scenario runs exactly {@code parameters.getMaxIterations()}
     * Newton steps entirely on the device, with NO per-iteration mismatch copy back to the host and
     * NO convergence retirement (the mismatch is read back once, at the end, only to report it).
     * This is a fixed-iteration NR loop — it drops the per-iteration PCIe sync, so it is
     * the fastest way to benchmark raw batched throughput. Tradeoff: a diverging scenario is not
     * isolated and can poison a uniform-batched factorization, so use {@code tol > 0} when scenarios
     * may diverge.
     */
    public static List<BatchScenarioResult> solveBatchN1(LfNetwork network, double[] baseState,
                                                         List<BatchContingency> contingencies,
                                                         EquationSystem<AcVariableType, AcEquationType> equationSystem,
                                                         TargetVector<AcVariableType, AcEquationType> targetVector,
                                                         NewtonRaphsonParameters parameters,
                                                         double tol,
                                                         ReportNode reportNode) {
        return solveBatchN1(network, baseState, contingencies, equationSystem, targetVector, parameters,
                tol, true, true, reportNode);
    }

    /** OLF default outer-loop convergence tolerances (per-unit on the 100 MVA base): slackBusPMaxMismatch =
     *  1 MW → 1e-2 pu, maxReactivePowerMismatch = 1 MVAr → 1e-2/… → 1e-4 pu. The device uses these so its
     *  outer-loop fixed point matches the CPU loadflow's; a test wanting machine-precision agreement passes
     *  tight values on BOTH sides (a loose tolerance leaves a redistribution band ~its size). */
    public static final double DEFAULT_REACTIVE_LIMIT_TOL = 1e-4;
    public static final double DEFAULT_SLACK_DIST_TOL = 1e-2;

    /**
     * As {@link #solveBatchN1(LfNetwork, double[], List, EquationSystem, TargetVector, NewtonRaphsonParameters, double, boolean, boolean, ReportNode)}
     * with the device outer-loop tolerances defaulting to OLF's ({@link #DEFAULT_REACTIVE_LIMIT_TOL} /
     * {@link #DEFAULT_SLACK_DIST_TOL}).
     */
    public static List<BatchScenarioResult> solveBatchN1(LfNetwork network, double[] baseState,
                                                         List<BatchContingency> contingencies,
                                                         EquationSystem<AcVariableType, AcEquationType> equationSystem,
                                                         TargetVector<AcVariableType, AcEquationType> targetVector,
                                                         NewtonRaphsonParameters parameters,
                                                         double tol,
                                                         boolean distributedSlack, boolean reactiveLimits,
                                                         ReportNode reportNode) {
        return solveBatchN1(network, baseState, contingencies, equationSystem, targetVector, parameters, tol,
                distributedSlack, reactiveLimits, DEFAULT_REACTIVE_LIMIT_TOL, DEFAULT_SLACK_DIST_TOL, reportNode);
    }

    /**
     * Lowest-level entry point. {@code reactiveLimitTol} / {@code slackDistTol} are the device outer-loop
     * convergence tolerances (pu) — set them to the run's loadflow criteria (maxReactivePowerMismatch /
     * slackBusPMaxMismatch, per-unitized) so the device runs the DS/RL outer loops to the SAME tolerance as
     * the CPU. With matching parameters and the same equations the GPU and CPU outer-loop fixed points agree
     * to the inner-Newton precision.
     */
    public static List<BatchScenarioResult> solveBatchN1(LfNetwork network, double[] baseState,
                                                         List<BatchContingency> contingencies,
                                                         EquationSystem<AcVariableType, AcEquationType> equationSystem,
                                                         TargetVector<AcVariableType, AcEquationType> targetVector,
                                                         NewtonRaphsonParameters parameters,
                                                         double tol,
                                                         boolean distributedSlack, boolean reactiveLimits,
                                                         double reactiveLimitTol, double slackDistTol,
                                                         ReportNode reportNode) {
        return solveBatchN1(network, baseState, contingencies, equationSystem, targetVector, parameters, tol,
                distributedSlack, reactiveLimits, reactiveLimitTol, slackDistTol, 0.0, reportNode);
    }

    /**
     * As the lowest {@code (reactiveLimitTol, slackDistTol)} overload but with an explicit {@code convEpsPerEq}:
     * when {@code > 0} the device inner-Newton stops on OLF's EXACT criterion
     * ({@code DefaultNewtonRaphsonStoppingCriteria}: {@code ‖F‖₂ < sqrt(convEpsPerEq² · n)}) instead of the
     * {@code tol} infinity-norm floor — so a batched SA can be run with the SAME stopping decision as the CPU
     * loadflow for an apples-to-apples GPU-vs-CPU timing comparison. {@code convEpsPerEq <= 0} keeps the
     * infinity-norm floor (the default the correctness/parity tests rely on).
     */
    public static List<BatchScenarioResult> solveBatchN1(LfNetwork network, double[] baseState,
                                                         List<BatchContingency> contingencies,
                                                         EquationSystem<AcVariableType, AcEquationType> equationSystem,
                                                         TargetVector<AcVariableType, AcEquationType> targetVector,
                                                         NewtonRaphsonParameters parameters,
                                                         double tol,
                                                         boolean distributedSlack, boolean reactiveLimits,
                                                         double reactiveLimitTol, double slackDistTol,
                                                         double convEpsPerEq,
                                                         ReportNode reportNode) {
        Objects.requireNonNull(network);
        Objects.requireNonNull(baseState);
        Objects.requireNonNull(contingencies);
        Objects.requireNonNull(equationSystem);
        Objects.requireNonNull(targetVector);
        Objects.requireNonNull(parameters);

        if (contingencies.isEmpty()) {
            LOGGER.debug("Batched N-1 with empty contingency list");
            return List.of();
        }

        // Extract base-case GPU data once (shared structure, all scenarios use same pattern). The batch path
        // now carries all three family passes: DISTR_Q (remote/distributed generator VC), DISTR_RHO
        // (transformer ratio distribution) and zero-impedance branches (bus couplers). A contingency that
        // outages a DISTR_RHO controller or a zero-impedance branch is routed to the CPU (below / via the
        // element mapper) because the fixed batch pattern cannot represent the resulting structural change.
        GpuAcData gpuData = GpuAcDataCache.extract(network, equationSystem, distributedSlack, reactiveLimits);
        GpuAcActivation baseActivation = GpuAcActivation.fromEquationSystem(network, equationSystem);
        GpuBatchElementMapper elementMapper = new GpuBatchElementMapper(network);

        int n = baseState.length;
        int nElements = gpuData.getTotalElementCount();
        double[] baseTargets = baseActivation.applyTargets(targetVector.getArray());

        // Qualify each contingency. Outages that keep the main component intact batch as a value-only
        // mask over the fixed pattern. A FRAGMENTING outage batches too in the full-loadflow mode
        // (tol > 0): its islanded buses are frozen to identity (held at base) in that scenario's row
        // mode, so the islanded sub-block stays non-singular and the main component solves correctly.
        // In the fixed-iter benchmark mode (tol <= 0) islanding routes to the CPU path.
        boolean handleIslanding = tol > 0;
        List<GpuBatchContingencyValidator.Qualification> qualifications =
                GpuBatchContingencyValidator.qualify(network, contingencies, elementMapper, handleIslanding);
        // Outaging a DISTR_RHO controller transformer reforms the ratio-distribution 1/count coefficients,
        // which the fixed batch pattern cannot represent — route those to the CPU (the DISTR_Q pass IS
        // batched, so DISTR_RHO is the only family that needs a contributor-outage rejection).
        Set<LfBranch> distrRhoControllers = gpuData.getDistrRhoContributionCount() > 0
                ? GpuAcDataExtractor.distrRhoControllerBranches(network, equationSystem) : Set.of();
        // Distributed-slack factor denominator (Σ maxP/droop over ALL participating gens, incl. the slack's)
        // and whether the slack itself participates: a generator at a DS-participant bus is batched with a
        // per-scenario reweight (computeGenOutage), but only when the slack does not participate.
        double dsSum = 0;
        if (distributedSlack) {                              // no slack distribution ⇒ no per-scenario DS reweight
            for (LfBus b : network.getBuses()) {
                for (LfGenerator g : b.getGenerators()) {
                    if (g.isParticipating() && g.getDroop() != 0) {
                        dsSum += g.getMaxP() / g.getDroop();
                    }
                }
            }
        }
        boolean slackParticipates = java.util.Arrays.stream(gpuData.dsFactor()).sum() < 1 - 1e-9;
        List<Integer> batchedIndices = new ArrayList<>();
        int[][] frozenRowsByIndex = new int[contingencies.size()][];     // islanded buses' variable rows, per contingency
        double[][] frozenTargetsByIndex = new double[contingencies.size()][];   // the clean value each frozen row holds (V=1, phi=0)
        GenOutage[][] genOutagesByIndex = new GenOutage[contingencies.size()][];   // per-bus merged outages, else null
        int islandingCount = 0;
        for (int i = 0; i < qualifications.size(); i++) {
            GpuBatchContingencyValidator.Qualification q = qualifications.get(i);
            if (!q.qualified()) {
                continue;
            }
            BatchContingency c = contingencies.get(i);
            if (!c.lostGenerators().isEmpty() || !c.lostLoads().isEmpty() || !c.shuntShifts().isEmpty()) {   // injection/shift (or branch+) contingency
                if (!c.shuntShifts().isEmpty()
                        && !canBatchShuntOutage(c.shuntShifts().stream().map(ShuntShift::shunt).toList(), elementMapper, gpuData)) {
                    continue;                                // unpacked / DISTR_Q-contributing shunt — CPU
                }
                GenOutage[] gos = computeInjectionOutages(equationSystem, gpuData, dsSum, slackParticipates,
                        c.lostGenerators(), c.lostLoads());
                if (gos == null) {                           // slack / >1 DS bus / remote gen — route to CPU
                    continue;
                }
                if (!c.outagedBranches().isEmpty()) {        // COMBO: the branch part must also qualify
                    if (c.outagedBranches().stream().anyMatch(distrRhoControllers::contains)) {
                        continue;
                    }
                    // conservative: a lost generator on an ISLANDED bus is out of combo scope
                    FrozenIsland fi = islandedFrozen(equationSystem, q.islandedBuses());
                    boolean genOnIsland = false;
                    for (GenOutage go : gos) {
                        for (int fr : fi.rows()) {
                            if (fr == go.pRow() || fr == go.vRow()) {
                                genOnIsland = true;
                            }
                        }
                    }
                    if (genOnIsland) {
                        continue;
                    }
                    frozenRowsByIndex[i] = fi.rows();
                    frozenTargetsByIndex[i] = fi.targets();
                    if (fi.rows().length > 0) {
                        islandingCount++;
                    }
                } else {
                    frozenRowsByIndex[i] = NO_FROZEN_ROWS;
                    frozenTargetsByIndex[i] = NO_FROZEN_TARGETS;
                }
                genOutagesByIndex[i] = gos;
                batchedIndices.add(i);
            } else if (!c.outagedShunts().isEmpty()) {       // fixed-shunt disconnection: mask the shunt element
                if (canBatchShuntOutage(c.outagedShunts(), elementMapper, gpuData)) {
                    batchedIndices.add(i);
                    frozenRowsByIndex[i] = NO_FROZEN_ROWS;
                    frozenTargetsByIndex[i] = NO_FROZEN_TARGETS;
                }                                            // else: not a packed fixed shunt / DISTR_Q contributor → CPU
            } else if (c.outagedBranches().stream().noneMatch(distrRhoControllers::contains)) {
                batchedIndices.add(i);
                FrozenIsland fi = islandedFrozen(equationSystem, q.islandedBuses());
                frozenRowsByIndex[i] = fi.rows();
                frozenTargetsByIndex[i] = fi.targets();
                if (frozenRowsByIndex[i].length > 0) {
                    islandingCount++;
                }
            }
        }
        int rejected = contingencies.size() - batchedIndices.size();
        if (rejected > 0 || islandingCount > 0) {
            LOGGER.info("Batched N-1: {} of {} contingencies routed to the CPU path (absent from the GPU "
                    + "model or islanding the slack); {} fragmenting outages solved on-GPU with frozen islands",
                    rejected, contingencies.size(), islandingCount);
        }

        // Solve the qualified scenarios on the GPU, in chunks bounded by the device-memory budget (the
        // uniform batched factorization holds an LU per scenario, so the footprint scales with the chunk).
        double[][] resultsByIndex = new double[contingencies.size()][];
        if (!batchedIndices.isEmpty()) {
            solveQualifiedInChunks(batchedIndices, contingencies, elementMapper, gpuData, baseState,
                    baseTargets, baseActivation.rowMode(), frozenRowsByIndex, frozenTargetsByIndex, genOutagesByIndex,
                    n, nElements, parameters, tol, reactiveLimitTol, slackDistTol, convEpsPerEq, resultsByIndex,
                    batchLsMode == 2 ? GpuAcDataExtractor.extractVarType(equationSystem, n) : null,
                    batchTransformerControl
                            ? buildTransformerControllers(network, equationSystem, targetVector, baseActivation,
                                    baseTargets, elementMapper)
                            : null);
        }

        // Assemble results in the original contingency order: batched scenarios carry their
        // GPU outcome; rejected ones carry the base state with STATUS_REJECTED.
        List<BatchScenarioResult> results = new ArrayList<>(contingencies.size());
        for (int i = 0; i < contingencies.size(); i++) {
            String id = contingencies.get(i).contingencyId();
            double[] result = resultsByIndex[i];
            if (result != null) {
                double[] state = new double[n];
                System.arraycopy(result, 0, state, 0, n);
                results.add(new BatchScenarioResult(id, state, (int) result[n], result[n + 1], (int) result[n + 2]));
            } else {
                results.add(new BatchScenarioResult(id, baseState.clone(), 0, Double.NaN, STATUS_REJECTED));
            }
        }
        return results;
    }

    private static final int[] NO_FROZEN_ROWS = new int[0];
    private static final double[] NO_FROZEN_TARGETS = new double[0];

    /** The islanded buses' variable rows to freeze + the clean flat value each holds (phi=0, V=1 pu). */
    private record FrozenIsland(int[] rows, double[] targets) { }

    /**
     * The variable rows (BUS_PHI + BUS_V) of every islanded bus — frozen to identity in a fragmenting
     * scenario's row mode — paired with the clean flat value each is HELD at: {@code phi=0}, {@code V=1} pu.
     * With the outaged branch (the only island↔main link) already masked, freezing these rows makes the
     * islanded sub-block a benign identity block that cannot make the batched factorization singular, while
     * the main component solves correctly. Holding the island at a flat 1∠0 (rather than the base equation
     * target, a P/Q value that produces a nonsensical islanded voltage/angle) gives the disconnected buses a
     * well-defined reported state — no garbage voltage, no spurious limit violations — without touching the main.
     */
    private static FrozenIsland islandedFrozen(EquationSystem<AcVariableType, AcEquationType> es, List<LfBus> islandedBuses) {
        if (islandedBuses.isEmpty()) {
            return new FrozenIsland(NO_FROZEN_ROWS, NO_FROZEN_TARGETS);
        }
        List<Integer> rows = new ArrayList<>(islandedBuses.size() * 2);
        List<Double> targets = new ArrayList<>(islandedBuses.size() * 2);
        for (LfBus bus : islandedBuses) {
            addFrozenVarRow(es, bus.getNum(), AcVariableType.BUS_PHI, 0.0, rows, targets);   // angle 0
            addFrozenVarRow(es, bus.getNum(), AcVariableType.BUS_V, 1.0, rows, targets);     // voltage 1 pu
        }
        int[] outRows = new int[rows.size()];
        double[] outTargets = new double[targets.size()];
        for (int i = 0; i < outRows.length; i++) {
            outRows[i] = rows.get(i);
            outTargets[i] = targets.get(i);
        }
        return new FrozenIsland(outRows, outTargets);
    }

    private static void addFrozenVarRow(EquationSystem<AcVariableType, AcEquationType> es, int busNum,
                                        AcVariableType type, double target, List<Integer> rows, List<Double> targets) {
        Variable<AcVariableType> v = es.getVariableSet().getVariable(busNum, type);
        if (v != null && v.getRow() >= 0) {
            rows.add(v.getRow());
            targets.add(target);
        }
    }

    /**
     * The per-scenario device changes a single generator loss applies (mirrors {@code LfContingency
     * .processLostGenerators}: remove the generator's P, and PV→PQ-flip the bus if it loses its last voltage
     * controller), or {@code null} when the generator is not batchable on the device and must fall through to
     * the CPU — the slack/reference bus, a remote / distributed (DISTR_Q) voltage controller, or a
     * distributed-slack participant bus when the slack ITSELF participates (the on-device retain math only
     * holds for a non-participating slack).
     *
     * <p>A generator at a DS-participant bus (non-participating slack) IS batched: {@code dsBus} carries the
     * bus's DS-pack index and the reweight deltas, applied by {@code distributeSlackBatch}.
     *
     * @param dsSum             Σ(maxP/droop) over ALL participating generators (incl. the slack's) — the
     *                          un-normalized factor denominator, to normalize the lost gen's factor delta
     * @param slackParticipates whether the slack bus itself has participating generators (then DS gens reject)
     */
    /** Merged per-BUS outages for a MULTI-generator contingency: gens grouped by bus, each bus's
     *  targets/DS deltas summed. Stage-1 constraint: at most ONE distributed-slack-participant bus
     *  per contingency (the native DS override carries one entry per scenario) — more, or any
     *  per-bus disqualification, returns null and the contingency routes to the CPU. */
    private static GenOutage[] computeInjectionOutages(EquationSystem<AcVariableType, AcEquationType> es, GpuAcData gpuData,
                                                       double dsSum, boolean slackParticipates,
                                                       List<LfGenerator> gens, List<LoadLoss> loads) {
        java.util.Map<LfBus, List<LfGenerator>> byBus = new java.util.LinkedHashMap<>();
        java.util.Map<LfBus, List<LoadLoss>> loadsByBus = new java.util.LinkedHashMap<>();
        for (LfGenerator g : gens) {
            if (g.getBus() == null) {
                return null;
            }
            byBus.computeIfAbsent(g.getBus(), b -> new ArrayList<>()).add(g);
        }
        for (LoadLoss l : loads) {
            loadsByBus.computeIfAbsent(l.bus(), b -> new ArrayList<>()).add(l);
            byBus.computeIfAbsent(l.bus(), b -> new ArrayList<>());   // load-only bus: empty gen list
        }
        GenOutage[] out = new GenOutage[byBus.size()];
        int k = 0;
        int dsBuses = 0;
        for (java.util.Map.Entry<LfBus, List<LfGenerator>> e : byBus.entrySet()) {
            GenOutage go = computeBusGenOutage(es, gpuData, dsSum, slackParticipates, e.getKey(), e.getValue(),
                    loadsByBus.getOrDefault(e.getKey(), List.of()));
            if (go == null) {
                return null;
            }
            if (go.dsBus() >= 0 && ++dsBuses > 1) {
                LOGGER.debug("Multi-injection contingency touches several DS-participant buses — CPU fallback");
                return null;
            }
            out[k++] = go;
        }
        return out;
    }

    private static GenOutage computeBusGenOutage(EquationSystem<AcVariableType, AcEquationType> es, GpuAcData gpuData,
                                                 double dsSum, boolean slackParticipates, LfBus bus,
                                                 List<LfGenerator> lost, List<LoadLoss> lostLoads) {
        int busNum = bus.getNum();
        if (es.getEquation(busNum, AcEquationType.BUS_TARGET_PHI).isPresent()        // slack/reference: P is free
                || es.getEquation(busNum, AcEquationType.DISTR_Q).isPresent()) {     // remote/distributed VC controller
            return null;
        }
        java.util.Set<LfGenerator> lostSet = new java.util.HashSet<>(lost);
        int pRow = es.getVariableSet().getVariable(busNum, AcVariableType.BUS_PHI).getRow();
        int vRow = es.getVariableSet().getVariable(busNum, AcVariableType.BUS_V).getRow();
        if (pRow < 0 || vRow < 0) {
            return null;
        }
        // Is this bus a distributed-slack participant? Only relevant when GENERATORS are lost here
        // (their participation leaves the DS pack); pure injection deltas (load losses, redispatch
        // actions) change targets but NOT participation, so they carry no DS reweight/constraint.
        int dsBus = -1;
        if (!lost.isEmpty()) {
            int[] dsPhiRow = gpuData.dsPhiRow();
            for (int i = 0; i < dsPhiRow.length; i++) {
                if (dsPhiRow[i] == pRow) {
                    dsBus = i;
                    break;
                }
            }
        }
        if (dsBus >= 0 && slackParticipates) {
            return null;                            // DS reweight + participating slack: retain math doesn't hold
        }
        if (bus.getGeneratorVoltageControl().map(vc -> vc.getControlledBus() != null
                && vc.getControlledBus().getNum() != busNum).orElse(false)) {
            return null;                                                            // remote voltage control
        }
        double lostP = lost.stream().mapToDouble(LfGenerator::getTargetP).sum();
        double lostLoadP = lostLoads.stream().mapToDouble(LoadLoss::p).sum();
        double lostLoadQ = lostLoads.stream().mapToDouble(LoadLoss::q).sum();
        double pTarget = bus.getTargetP() - lostP + lostLoadP;   // shedding load RAISES the P target
        boolean busPv = bus.isGeneratorVoltageControlEnabled();
        if (busPv && lostLoads.stream().anyMatch(l -> l.q() != 0)) {
            return null;                                     // Q-carrying load delta at a voltage-controlling bus: the
        }                                                    // RL pack's static loadQ would go stale — CPU fallback
                                                             // (P-only deltas, e.g. redispatch actions, are safe: PV Q is free)
        boolean lostControls = lost.stream()
                .anyMatch(g -> g.getGeneratorControlType() == LfGenerator.GeneratorControlType.VOLTAGE);
        boolean otherController = bus.getGenerators().stream()
                .anyMatch(g -> !lostSet.contains(g) && g.getGeneratorControlType() == LfGenerator.GeneratorControlType.VOLTAGE);
        boolean toPq = busPv && lostControls && !otherController;
        // The bus Q TARGET must shed the lost gens' targetQ whenever the V row solves the reactive
        // equation in the scenario: on the PV->PQ flip AND on an ALREADY-PQ bus losing fixed-Q
        // generation (task #21 — the flip-only version silently kept the lost Q injected; measured
        // 0.27 kV voltage divergence on ieee14 for a 15-MVAr fixed-Q gen loss).
        boolean qOverride = toPq || !busPv;
        double qTarget = 0;
        if (qOverride) {
            double remainingGenQ = bus.getGenerators().stream().filter(g -> !lostSet.contains(g))
                    .mapToDouble(LfGenerator::getTargetQ).sum();
            qTarget = remainingGenQ - (bus.getLoadTargetQ() - lostLoadQ);
        }
        // Distributed-slack reweight for the affected bus. The bus's base target P always drops by the lost
        // gen's P. The participation factor / headroom / footroom drop only if the LOST gen itself participated
        // (mirrors the extractor, which sums these over participating gens with a non-zero droop).
        double dsFactorDelta = 0;
        double dsBaseDelta = 0;
        double dsMaxDelta = 0;
        double dsMinDelta = 0;
        if (dsBus >= 0) {
            dsBaseDelta = lostP;
            for (LfGenerator gen : lost) {                   // per-gen deltas compose additively
                if (gen.isParticipating() && gen.getDroop() != 0) {
                    dsFactorDelta += (gen.getMaxP() / gen.getDroop()) / dsSum;
                    double minTp = gen.getMinTargetP();
                    double maxTp = gen.getMaxTargetP();
                    double tp = gen.getTargetP();
                    if (tp < 0) {
                        maxTp = Math.min(maxTp, 0);
                    } else {
                        minTp = Math.max(minTp, 0);
                    }
                    dsMaxDelta += maxTp - tp;
                    dsMinDelta += minTp - tp;
                }
            }
        }
        return new GenOutage(pRow, pTarget, vRow, qTarget, toPq, qOverride, dsBus, dsFactorDelta, dsBaseDelta, dsMaxDelta, dsMinDelta);
    }

    /**
     * Whether a fixed-shunt disconnection can be batched as a plain element mask. Requires every outaged shunt
     * to be a packed fixed shunt ({@code getElementIndex >= 0}) and NOT a reactive-distribution (DISTR_Q)
     * contributor — a shunt whose reactive feeds a controlled bus's DISTR_Q sum is read unconditionally by the
     * batched DISTR_Q fill (it ignores the disabled mask), so masking it would leave the sum stale. Such a shunt
     * outage is routed to the CPU instead.
     */
    private static boolean canBatchShuntOutage(List<LfShunt> shunts, GpuBatchElementMapper elementMapper,
                                               GpuAcData gpuData) {
        int shOffset = elementMapper.getClosedBranchCount() + elementMapper.getOpenBranchCount();
        int[] dqElem = gpuData.dqElem();
        int[] dqKind = gpuData.dqKind();
        for (LfShunt shunt : shunts) {
            int elementIndex = elementMapper.getElementIndex(shunt);
            if (elementIndex < 0) {
                return false;                                // not a packed fixed shunt (controller/SVC shunt)
            }
            int packIndex = elementIndex - shOffset;
            for (int i = 0; i < dqKind.length; i++) {
                if (dqKind[i] == 1 && dqElem[i] == packIndex) {
                    return false;                            // shunt feeds a DISTR_Q sum — mask wouldn't update it
                }
            }
        }
        return true;
    }

    /**
     * Solve the qualified scenarios on the GPU in chunks sized by {@link #resolveChunkSize}
     * (auto from device memory, or a manual cap). ONE native context, sized to the chunk, is created and reused across all
     * chunks: cuDSS's symbolic ANALYSIS (on the shared pattern, identical for every chunk) is
     * done once on the first chunk and reused, so chunking costs nothing in amortization — it
     * only bounds the resident per-scenario LU factors. The last (partial) chunk is padded
     * with base-case scenarios (no element disabled): they converge at iteration 0 and their
     * results are discarded.
     */
    /**
     * Per-scenario port of TransformerVoltageControlOuterLoop (AFTER_GENERATOR mode), run as a
     * Java orchestrator around repeated batched solves on THIS chunk, mirroring the CPU steps:
     * INITIAL (deadband check on the solved states -> enable out-of-band controls via the
     * activation-diffed row/target deltas), CONTROL (clamp continuous r1 outside [minR1,maxR1]
     * to the extreme and disable, iterate), then round to the closest discrete tap, final solve.
     * Scenarios re-solve WARM from their previous states (x0PerScenario). Returns the merged
     * chunk results (unaffected slots keep the incoming ones). Stage-1 scope: no generator
     * voltage controls under the transformer-controlled level (checked by the provider guard).
     */
    private static double[][] orchestrateTransformerControl(long handle, int chunkSize, int realCount, int n,
            double[][] chunkResults, List<TransformerController> tcs,
            int[][] rowModePerScenario, int[] baseRowMode, boolean[][] disabledMask,
            int[] targetOverrideIndex, double[] targetOverrideValue,
            int[] dsOvBus, double[] dsOvFactorDelta, double[] dsOvBaseDelta, double[] dsOvMaxDelta, double[] dsOvMinDelta,
            double[] baseState, double[] baseTargets, NewtonRaphsonParameters parameters,
            double tol, double reactiveLimitTol, double slackDistTol, double convEpsPerEq, int[] varType) {
        // INITIAL: which (slot, controller) pairs are out of deadband on the converged first solve?
        List<int[]> enabled = new ArrayList<>();             // {slot, controllerIdx}
        for (int slot = 0; slot < realCount; slot++) {
            double[] r = chunkResults[slot];
            if (r == null || (int) r[n + 2] != STATUS_CONVERGED) {
                continue;
            }
            for (int c = 0; c < tcs.size(); c++) {
                TransformerController tc = tcs.get(c);
                if (disabledMask[slot][tc.elemIndex()]) {
                    continue;                                // the controller transformer itself is outaged
                }
                if (Math.abs(tc.targetV() - r[tc.vCol()]) > tc.halfDeadband()) {
                    enabled.add(new int[] {slot, c});
                }
            }
        }
        if (enabled.isEmpty()) {
            return chunkResults;
        }
        java.util.Set<Integer> affected = new java.util.TreeSet<>();
        boolean[][] on = new boolean[chunkSize][tcs.size()];
        for (int[] e : enabled) {
            affected.add(e[0]);
            on[e[0]][e[1]] = true;
        }
        LOGGER.info("Transformer control: {} out-of-deadband controls across {} scenario(s) of the chunk", enabled.size(), affected.size());

        // Base per-scenario overrides (islanding/gen-outage) are kept and extended per call.
        List<Integer> ovIdx = new ArrayList<>();
        List<Double> ovVal = new ArrayList<>();
        double[] rounded = new double[chunkSize * tcs.size()];   // fixed r1 once clamped/rounded (NaN = still continuous)
        java.util.Arrays.fill(rounded, Double.NaN);

        int[][] rowMode = new int[chunkSize][];
        for (int step = 0; step < 8; step++) {               // INITIAL solve, clamp iterations, final rounded solve
            ovIdx.clear();
            ovVal.clear();
            for (int k = 0; k < targetOverrideIndex.length; k++) {
                ovIdx.add(targetOverrideIndex[k]);
                ovVal.add(targetOverrideValue[k]);
            }
            double[] x0Full = new double[chunkSize * n];
            for (int slot = 0; slot < chunkSize; slot++) {
                double[] r = slot < realCount ? chunkResults[slot] : null;
                for (int i = 0; i < n; i++) {
                    x0Full[slot * n + i] = r != null ? r[i] : baseState[i];
                }
                rowMode[slot] = rowModePerScenario[slot];
                if (!affected.contains(slot)) {
                    continue;
                }
                int[] row = rowModePerScenario[slot].clone();
                for (int c = 0; c < tcs.size(); c++) {
                    TransformerController tc = tcs.get(c);
                    double fixedR1 = rounded[slot * tcs.size() + c];
                    if (on[slot][c] && Double.isNaN(fixedR1)) {          // control ACTIVE: apply the ON deltas
                        for (int k = 0; k < tc.deltaRows().length; k++) {
                            row[tc.deltaRows()[k]] = tc.deltaModesOn()[k];
                            ovIdx.add(slot * n + tc.deltaRows()[k]);
                            ovVal.add(tc.deltaTargetsOn()[k]);
                        }
                    } else if (!Double.isNaN(fixedR1)) {                 // clamped/rounded: identity at the fixed r1
                        ovIdx.add(slot * n + tc.rhoRow());
                        ovVal.add(fixedR1);
                    }
                }
                rowMode[slot] = row;
            }
            double[][] r2 = GpuAcNewtonSolver.solveBatchContext(handle, baseState, baseTargets,
                    rowMode, baseRowMode, disabledMask,
                    ovIdx.stream().mapToInt(Integer::intValue).toArray(),
                    ovVal.stream().mapToDouble(Double::doubleValue).toArray(),
                    dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta,
                    parameters.getMaxIterations(), tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                    batchLsMode, batchMaxDv, batchMaxDphi,
                    batchLsMode == 2 ? varType : null, x0Full, null, null, null, null);
            for (int slot : affected) {
                chunkResults[slot] = r2[slot];
            }
            // CONTROL: clamp out-of-range continuous ratios; when none, ROUND the remaining ones.
            boolean anyClamped = false;
            boolean anyContinuous = false;
            for (int slot : affected) {
                double[] r = chunkResults[slot];
                if (r == null || (int) r[n + 2] != STATUS_CONVERGED) {
                    continue;
                }
                for (int c = 0; c < tcs.size(); c++) {
                    TransformerController tc = tcs.get(c);
                    if (!on[slot][c] || !Double.isNaN(rounded[slot * tcs.size() + c])) {
                        continue;
                    }
                    double r1 = r[tc.rhoRow()];
                    if (r1 < tc.minR1() || r1 > tc.maxR1()) {
                        rounded[slot * tcs.size() + c] = Math.max(tc.minR1(), Math.min(tc.maxR1(), r1));
                        anyClamped = true;
                    } else {
                        anyContinuous = true;
                    }
                }
            }
            if (anyClamped) {
                continue;                                    // re-solve with the clamped ones fixed
            }
            if (anyContinuous) {                             // round ALL remaining continuous ratios, one final solve
                for (int slot : affected) {
                    double[] r = chunkResults[slot];
                    if (r == null) {
                        continue;
                    }
                    for (int c = 0; c < tcs.size(); c++) {
                        TransformerController tc = tcs.get(c);
                        if (on[slot][c] && Double.isNaN(rounded[slot * tcs.size() + c])) {
                            rounded[slot * tcs.size() + c] = tc.closestTap(r[tc.rhoRow()]);
                        }
                    }
                }
                continue;                                    // the final solve runs with every control fixed
            }
            break;                                           // everything fixed and re-solved: done
        }
        return chunkResults;
    }

    private static void solveQualifiedInChunks(List<Integer> batchedIndices, List<BatchContingency> contingencies,
                                               GpuBatchElementMapper elementMapper, GpuAcData gpuData,
                                               double[] baseState, double[] baseTargets, int[] baseRowMode,
                                               int[][] frozenRowsByIndex, double[][] frozenTargetsByIndex,
                                               GenOutage[][] genOutagesByIndex,
                                               int n, int nElements, NewtonRaphsonParameters parameters,
                                               double tol, double reactiveLimitTol, double slackDistTol,
                                               double convEpsPerEq, double[][] resultsByIndex, int[] varType,
                                               List<TransformerController> transformerControllers) {
        int nb = batchedIndices.size();
        int chunkSize = resolveChunkSize(nb, n, gpuData);
        int nChunks = (nb + chunkSize - 1) / chunkSize;

        // Reuse a cached analyzed context when the pattern/admittances/chunk/mode match (across
        // SA runs on the same network) — the context is NOT destroyed here, it stays cached.
        long handle = acquireBatchContext(gpuData, chunkSize, resolveBatchMode(), n);

        // The base state / targets are IDENTICAL for every scenario (a single-branch N-1 starts from the
        // same converged base operating point): passed as single length-n rows and tiled on-device. The
        // disabled mask is per-scenario (each chunk clears the one flag each slot set, sets the new one).
        // The ROW MODE is per-scenario too: a non-fragmenting outage shares the base row (same PV/slack
        // layout); a fragmenting one gets a cloned base row with its islanded buses frozen to identity.
        boolean[][] disabledMask = new boolean[chunkSize][nElements];
        int[][] disabledIdx = new int[chunkSize][];      // element indices each slot currently disables (1-2), null = none
        int[][] rowModePerScenario = new int[chunkSize][];   // per-scenario row mode (re-assigned every chunk)
        // Per-scenario target overrides (generator contingencies): flat (dTargetIndex = slot*n + row, value)
        // lists rebuilt per chunk. Empty when no generator outage is in the chunk.
        List<Integer> ovIndexList = new ArrayList<>();
        List<Double> ovValueList = new ArrayList<>();
        List<Integer> shOvSlotElemList = new ArrayList<>();  // per-scenario shunt admittance overrides
        List<Double> shOvGbList = new ArrayList<>();
        int shPackOffset = elementMapper.getClosedBranchCount() + elementMapper.getOpenBranchCount();
        // Per-scenario distributed-slack reweight (a generator loss at a DS-participant bus): one affected DS
        // bus per slot (dsOvBus[s] = its DS-pack index, -1 = none) whose factor/base/headroom/footroom and the
        // factor sum shrink by the lost gen's share. Only allocated when the network has distributed slack.
        boolean hasDs = gpuData.dsPhiRow().length > 0;
        int[] dsOvBus = hasDs ? new int[chunkSize] : new int[0];
        double[] dsOvFactorDelta = hasDs ? new double[chunkSize] : new double[0];
        double[] dsOvBaseDelta = hasDs ? new double[chunkSize] : new double[0];
        double[] dsOvMaxDelta = hasDs ? new double[chunkSize] : new double[0];
        double[] dsOvMinDelta = hasDs ? new double[chunkSize] : new double[0];
        // TWO-PHASE ordering (operator strategies, warm continuation): plain contingencies first,
        // then (ctg, strategy) PAIRS — each pair chunk is seeded with its PARENT contingency's
        // solved state (x0PerScenario) and reactive-limits pinning (rlStateIn), so the strategy
        // solve continues the contingency's outer-loop trajectory exactly like the CPU's
        // per-contingency action solve. Pair whose parent did not converge is dropped (CPU path).
        List<Integer> ctgIdxList = new ArrayList<>();
        List<Integer> pairIdxList = new ArrayList<>();
        for (int idx : batchedIndices) {
            (contingencies.get(idx).contingencyId().indexOf(STRATEGY_KEY_SEP) >= 0 ? pairIdxList : ctgIdxList).add(idx);
        }
        java.util.Map<String, Integer> indexById = new java.util.HashMap<>();
        for (int i = 0; i < contingencies.size(); i++) {
            indexById.put(contingencies.get(i).contingencyId(), i);
        }
        int nrl = gpuData.rlVrow().length;
        java.util.Map<Integer, int[]> rlByIndex = new java.util.HashMap<>();    // ctg index -> its nrl RL-state slice
        // Host-side base RL seed (mirrors initRlLimitTypeBatch) for PADDING slots of pair chunks.
        int[] baseRlSeed = new int[nrl];
        for (int i = 0; i < nrl; i++) {
            int vrow = gpuData.rlVrow()[i];
            baseRlSeed[i] = baseRowMode[vrow] == 1 ? (baseState[vrow] < gpuData.rlVtarget()[i] ? 2 : 1) : 0;
        }
        try {
            long startTime = System.currentTimeMillis();
            for (List<Integer> group : List.of(ctgIdxList, pairIdxList)) {
            boolean pairPhase = group == pairIdxList;
            int nbg = group.size();
            for (int chunkStart = 0; chunkStart < nbg; chunkStart += chunkSize) {
                int realCount = Math.min(chunkSize, nbg - chunkStart);
                ovIndexList.clear();
                ovValueList.clear();
                shOvSlotElemList.clear();
                shOvGbList.clear();
                if (hasDs) {
                    java.util.Arrays.fill(dsOvBus, -1);
                    java.util.Arrays.fill(dsOvFactorDelta, 0);
                    java.util.Arrays.fill(dsOvBaseDelta, 0);
                    java.util.Arrays.fill(dsOvMaxDelta, 0);
                    java.util.Arrays.fill(dsOvMinDelta, 0);
                }
                for (int s = 0; s < chunkSize; s++) {
                    if (disabledIdx[s] != null) {        // clear this slot's outage(s) from the previous chunk
                        for (int ei : disabledIdx[s]) {
                            disabledMask[s][ei] = false;
                        }
                        disabledIdx[s] = null;
                    }
                    rowModePerScenario[s] = baseRowMode;     // default: base layout (padding + non-fragmenting)
                    if (s < realCount) {                 // real scenario: disable its outaged branch(es)
                        int ci = group.get(chunkStart + s);
                        List<LfBranch> branches = contingencies.get(ci).outagedBranches();
                        List<LfShunt> shunts = contingencies.get(ci).outagedShunts();
                        int[] eis = new int[branches.size() + shunts.size()];
                        int ei = 0;
                        for (LfBranch branch : branches) {
                            int idx = elementMapper.getElementIndex(branch);          // qualification guarantees >= 0
                            disabledMask[s][idx] = true;
                            eis[ei++] = idx;
                        }
                        for (LfShunt shunt : shunts) {                                // fixed-shunt disconnection
                            int idx = elementMapper.getElementIndex(shunt);          // qualification guarantees >= 0
                            disabledMask[s][idx] = true;
                            eis[ei++] = idx;
                        }
                        disabledIdx[s] = eis;
                        int[] frozen = frozenRowsByIndex[ci];
                        if (frozen.length > 0) {         // fragmenting outage: freeze the islanded buses' rows
                            int[] row = baseRowMode.clone();
                            double[] frozenTargets = frozenTargetsByIndex[ci];
                            for (int k = 0; k < frozen.length; k++) {
                                row[frozen[k]] = 0;      // identity row — bus decoupled from the main component
                                // ...held at a CLEAN flat state (V=1, phi=0) via the target override, NOT the base
                                // equation target (a P/Q value that gives a nonsensical islanded voltage/angle).
                                ovIndexList.add(s * n + frozen[k]);
                                ovValueList.add(frozenTargets[k]);
                            }
                            rowModePerScenario[s] = row;
                        }
                        for (ShuntShift sh : contingencies.get(ci).shuntShifts()) {
                            shOvSlotElemList.add(s);
                            shOvSlotElemList.add(elementMapper.getElementIndex(sh.shunt()) - shPackOffset);
                            shOvGbList.add(sh.dg());
                            shOvGbList.add(sh.db());
                        }
                        GenOutage[] gos = genOutagesByIndex[ci];
                        if (gos != null) {               // generator loss(es): per-BUS target overrides (merged by bus;
                            for (GenOutage go : gos) {   // at most one DS-participant bus per contingency by construction)
                                ovIndexList.add(s * n + go.pRow());
                                ovValueList.add(go.pTarget());
                                if (go.qOverride()) {    // Q target sheds the lost gens' Q: on the PV->PQ flip AND
                                    ovIndexList.add(s * n + go.vRow());   // on an ALREADY-PQ bus losing fixed-Q gens (#21)
                                    ovValueList.add(go.qTarget());
                                }
                                if (go.toPq()) {
                                    int[] row = rowModePerScenario[s] == baseRowMode ? baseRowMode.clone() : rowModePerScenario[s];
                                    row[go.vRow()] = 1;  // BUS_V row becomes the reactive-power equation (PV→PQ)
                                    rowModePerScenario[s] = row;
                                }
                                if (hasDs && go.dsBus() >= 0) {   // lost gen(s) on a DS-participant bus: per-scenario reweight
                                    dsOvBus[s] = go.dsBus();
                                    dsOvFactorDelta[s] = go.dsFactorDelta();
                                    dsOvBaseDelta[s] = go.dsBaseDelta();
                                    dsOvMaxDelta[s] = go.dsMaxDelta();
                                    dsOvMinDelta[s] = go.dsMinDelta();
                                }
                            }
                        }
                    }
                    // else: padding scenario = base case (no element disabled), discarded below
                }
                int[] targetOverrideIndex = ovIndexList.stream().mapToInt(Integer::intValue).toArray();
                double[] targetOverrideValue = ovValueList.stream().mapToDouble(Double::doubleValue).toArray();
                // Pair phase: warm continuation from each parent contingency's state + RL pinning.
                double[] x0Full = null;
                int[] rlStateIn = null;
                if (pairPhase) {
                    x0Full = new double[chunkSize * n];
                    rlStateIn = nrl > 0 ? new int[chunkSize * nrl] : null;
                    for (int s = 0; s < chunkSize; s++) {
                        double[] parent = null;
                        int[] parentRl = null;
                        if (s < realCount) {
                            String pairId = contingencies.get(group.get(chunkStart + s)).contingencyId();
                            Integer pIdx = indexById.get(pairId.substring(0, pairId.indexOf(STRATEGY_KEY_SEP)));
                            if (pIdx != null && resultsByIndex[pIdx] != null
                                    && (int) resultsByIndex[pIdx][n + 2] == STATUS_CONVERGED) {
                                parent = resultsByIndex[pIdx];
                                parentRl = rlByIndex.get(pIdx);
                            }
                        }
                        for (int i = 0; i < n; i++) {
                            x0Full[s * n + i] = parent != null ? parent[i] : baseState[i];
                        }
                        if (rlStateIn != null) {
                            System.arraycopy(parentRl != null ? parentRl : baseRlSeed, 0, rlStateIn, s * nrl, nrl);
                        }
                    }
                }
                int[] rlStateOut = !pairPhase && !pairIdxList.isEmpty() && nrl > 0 ? new int[chunkSize * nrl] : null;
                double[][] chunkResults = GpuAcNewtonSolver.solveBatchContext(handle, baseState, baseTargets,
                        rowModePerScenario, baseRowMode, disabledMask, targetOverrideIndex, targetOverrideValue,
                        dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta,
                        parameters.getMaxIterations(), tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                        batchLsMode, batchMaxDv, batchMaxDphi,
                        batchLsMode == 2 ? varType : null, x0Full,
                        shOvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                        shOvGbList.stream().mapToDouble(Double::doubleValue).toArray(),
                        rlStateIn, rlStateOut);
                if (rlStateOut != null) {                    // capture each ctg's final RL pinning for its pairs
                    for (int s = 0; s < realCount; s++) {
                        int[] slice = new int[nrl];
                        System.arraycopy(rlStateOut, s * nrl, slice, 0, nrl);
                        rlByIndex.put(group.get(chunkStart + s), slice);
                    }
                }
                if (transformerControllers != null && !transformerControllers.isEmpty()) {
                    chunkResults = orchestrateTransformerControl(handle, chunkSize, realCount, n,
                            chunkResults, transformerControllers, rowModePerScenario, baseRowMode, disabledMask,
                            targetOverrideIndex, targetOverrideValue,
                            dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta,
                            baseState, baseTargets, parameters, tol, reactiveLimitTol, slackDistTol, convEpsPerEq, varType);
                }
                for (int s = 0; s < realCount; s++) {
                    resultsByIndex[group.get(chunkStart + s)] = chunkResults[s];
                }
            }
            }
            long elapsed = System.currentTimeMillis() - startTime;
            LOGGER.info("Batched N-1: {} GPU scenarios in {} chunk(s) of <= {} solved in {} ms ({} ms/scenario)",
                    nb, nChunks, chunkSize, elapsed, (double) elapsed / nb);
        } catch (RuntimeException e) {
            clearContextCache();             // a failed solve may leave the context inconsistent — rebuild next time
            throw e;
        }
    }

    /** Absolute upper bound on an auto-sized chunk, so a huge free-memory reading never yields a pathological chunk. */
    static final int AUTO_MAX_CHUNK = 4096;

    /**
     * Resolve the GPU batch chunk size, bounding the resident device memory (the uniform
     * batched factorization holds an LU per scenario, so the footprint scales with the chunk).
     * <ul>
     *   <li>{@value #MAX_BATCH_PROPERTY} set to a positive value — an explicit manual cap.</li>
     *   <li>otherwise — AUTO-SIZED from free device memory via
     *       {@link GpuAcNewtonSolver#recommendBatchSize} (real Jacobian nnz + heuristic LU
     *       estimate + safety margin).</li>
     *   <li>if auto-sizing is unavailable (no GPU / query error) — {@value #DEFAULT_MAX_BATCH}.</li>
     * </ul>
     * The result is always clamped to {@code [1, nb]}.
     */
    static int resolveChunkSize(int nb, int n, GpuAcData gpuData) {
        Integer explicit = Integer.getInteger(MAX_BATCH_PROPERTY);
        if (explicit != null && explicit > 0) {
            return Math.min(nb, explicit);
        }
        try {
            // sized on the shared structure (its exact nnz + a cuDSS LU probe cached there, so
            // only the first query per extraction pays the probe analysis)
            int rec = GpuAcNewtonSolver.recommendBatchSize(GpuAcStructures.handle(n, gpuData),
                    Math.min(nb, AUTO_MAX_CHUNK));
            return Math.min(nb, Math.max(1, rec));
        } catch (Throwable t) {
            LOGGER.debug("GPU batch auto-size unavailable ({}); using default chunk {}", t.toString(), DEFAULT_MAX_BATCH);
            return Math.min(nb, DEFAULT_MAX_BATCH);
        }
    }

    private GpuBatchedSecurityAnalysisSolver() {
    }
}
