/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

import com.powsybl.commons.PowsyblException;
import com.powsybl.commons.report.ReportNode;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivation;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivationExtractor;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataCache;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcStructures;
import com.powsybl.openloadflow.ac.solver.AbstractAcSolver;
import com.powsybl.openloadflow.ac.solver.AcSolverResult;
import com.powsybl.openloadflow.ac.solver.AcSolverStatus;
import com.powsybl.openloadflow.ac.solver.AcSolverUtil;
import com.powsybl.openloadflow.ac.solver.DefaultNewtonRaphsonStoppingCriteria;
import com.powsybl.openloadflow.ac.solver.NewtonRaphson;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonStoppingCriteria;
import com.powsybl.openloadflow.ac.solver.PerEquationTypeStoppingCriteria;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchContextCache;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.EquationSystemIndex;
import com.powsybl.openloadflow.equations.EquationSystemIndexListener;
import com.powsybl.openloadflow.equations.EquationVector;
import com.powsybl.openloadflow.equations.JacobianMatrix;
import com.powsybl.openloadflow.equations.TargetVector;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfHvdc;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfNetworkListener;
import com.powsybl.openloadflow.network.LfSynchronousNetwork;
import com.powsybl.openloadflow.network.util.VoltageInitializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.ref.Cleaner;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

/**
 * Newton-Raphson run ENTIRELY on the GPU behind one JNI call ({@link GpuAcNewtonSolver}):
 * the fixed superset-pattern CSR Jacobian and the mismatch are filled on the device by
 * the generated fused kernels (closed-branch / open-branch / shunt / HVDC AC emulation),
 * cuDSS analyzes once and refactorizes/solves per iteration, the state update is a
 * kernel.
 *
 * <p>Invalidation is EVENT-DRIVEN, mirroring {@link JacobianMatrix}'s pattern (the same
 * {@link EquationSystemIndexListener} / {@link LfNetworkListener} interfaces, with this
 * consumer's own granularity): structural events (topology, branch parameters) force a
 * full re-extract + device context rebuild; activation events (PV→PQ switching, slack
 * change) only refresh the row-mode mask and target mapping; with no events, a rerun
 * goes straight to the device solve — targets and state are re-read every solve. Both
 * interfaces are implemented DIRECTLY (not via the no-op adapter) so any event added to
 * OLF later forces a compile-time classification here, keeping invalidation conservative.
 *
 * <p>Scope: plain AC power flow. {@link GpuAcDataExtractor} validates the equation
 * system on every structural rebuild and throws on anything not covered; an unsupported
 * equation type ACTIVATED between rebuilds fails loudly in the activation walk. The
 * device loop iterates to an infinity-norm mismatch of {@value GPU_TOLERANCE} (stricter
 * than the usual stopping criteria); the final CONVERGED/MAX_ITERATION_REACHED status is
 * then decided by the CONFIGURED stopping criteria, evaluated on the CPU from the
 * written-back state, exactly as for the CPU Newton-Raphson. State-vector scaling
 * options are not applied (no line search on the device).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public class GpuNewtonRaphson extends AbstractAcSolver {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuNewtonRaphson.class);

    private static final Cleaner CLEANER = Cleaner.create();

    public static final double GPU_TOLERANCE = 1e-8;

    // Diagnostics for the invalidation taxonomy: how the cached native context is reconciled across
    // solves — STRUCTURE rebuilds (pattern + cuDSS re-analysis) vs cheap VALUES re-uploads (a transformer
    // tap / shunt-susceptance move keeps the pattern, only the element packs change). Process-wide totals.
    private static final java.util.concurrent.atomic.AtomicLong CONTEXT_REBUILDS = new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong VALUE_REFRESHES = new java.util.concurrent.atomic.AtomicLong();

    public static long contextRebuildCount() {
        return CONTEXT_REBUILDS.get();
    }

    public static long valueRefreshCount() {
        return VALUE_REFRESHES.get();
    }

    protected final NewtonRaphsonParameters parameters;

    /**
     * The run's outer-loop settings, so the batch-of-one can drive the SAME device loops the batched
     * SA drives. A batch of one is still a batch: nothing about nS == 1 justifies restricting it.
     *
     * <p>{@code driverOuterLoops} off keeps the historical behaviour — the solve neutralises the
     * driver ({@code slackDistTol} 1e300, {@code skipRlPhase} 1) and OLF's Java loops run around it
     * on the CPU.
     */
    private final boolean driverOuterLoops;
    private final double driverSlackDistTol;
    private final double driverReactiveLimitTol;
    private final int driverMaxOuterIterations;
    private final int driverMaxPqPvSwitch;

    /** Read the bands out of the run's OWN configured loops, so the device decides at OLF's threshold. */
    static double[] driverSettings(com.powsybl.openloadflow.ac.AcLoadFlowParameters p) {
        double slackTol = Double.MAX_VALUE;
        double rlTol = 0;
        int maxSwitch = 0;
        for (var ol : p.getOuterLoops()) {
            if (ol instanceof com.powsybl.openloadflow.ac.outerloop.DistributedSlackOuterLoop ds) {
                slackTol = ds.getSlackBusPMaxMismatch() / com.powsybl.openloadflow.util.PerUnit.SB;
            } else if (ol instanceof com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop rl) {
                rlTol = rl.getMaxReactivePowerMismatch() / com.powsybl.openloadflow.util.PerUnit.SB;
                maxSwitch = rl.getMaxPqPvSwitch();
            }
        }
        return new double[] {slackTol, rlTol, p.getMaxOuterLoopIterations(), maxSwitch};
    }

    private static double deviceConvEpsPerEq(NewtonRaphsonParameters parameters) {
        if (parameters.getStoppingCriteria() instanceof DefaultNewtonRaphsonStoppingCriteria criteria) {
            return criteria.getConvEpsPerEq();
        }
        return 0.0;
    }

    /** System property: fall back to the CPU Newton-Raphson when the GPU path rejects the
     *  network or fails at runtime (default true). Set to false to fail loudly instead. */
    public static final String CPU_FALLBACK_PROPERTY = "olf.gpu.cpuFallback";

    /**
     * Number of solves that bailed out of the GPU path since JVM start. Tests assert on it because a
     * silent fallback is INVISIBLE otherwise: results stay CPU-identical (that is the whole point of the
     * contract), the warning is logged through a logger the test config does not attach an appender to,
     * and {@link #CPU_FALLBACK_PROPERTY}=false only covers throws that reach the catch below. A GPU test
     * that does not check this counter can pass while never touching the GPU at all — that is exactly how
     * the BRANCH_ALPHA1 packing looked "green but uncovered" until the counter existed.
     */
    public static final java.util.concurrent.atomic.AtomicLong CPU_FALLBACK_COUNT =
            new java.util.concurrent.atomic.AtomicLong();

    private final GpuNativeContext context = new GpuNativeContext();

    private final GpuSolverInvalidation invalidation = new GpuSolverInvalidation();

    private GpuAcData cachedData;

    private GpuAcActivation activation;

    // per-row variable kind (0=BUS_PHI, 1=BUS_V, 2=other) for the MAX_VOLTAGE_CHANGE device step
    // clamp; structural, recomputed whenever the activation/structure is rebuilt
    private int[] varType;

    // PER_EQUATION_TYPE row thresholds need (re)installing on the context: after a context rebuild
    // and after any activation change (the thresholds follow the variables at each row)
    private boolean criteriaDirty = true;

    /** Install (or clear) the PER_EQUATION_TYPE per-row stopping thresholds on the batch context,
     *  from the run's actual criteria object — the device then stops on the SAME decision as OLF's
     *  PerEquationTypeStoppingCriteria (every row's |F| under its type threshold). */
    private void installRowCriteria(int n) {
        if (parameters.getStoppingCriteria() instanceof PerEquationTypeStoppingCriteria c) {
            double[][] thr = GpuAcActivationExtractor.extractRowCriteria(equationSystem, n,
                    c.getMaxActivePowerMismatch(), c.getMaxReactivePowerMismatch(), c.getMaxVoltageMismatch(),
                    c.getMaxDefaultAngleMismatch(), c.getMaxDefaultRatioMismatch(),
                    c.getMaxDefaultSusceptanceMismatch(), c.getConvEpsPerEq());
            GpuAcNewtonSolver.setBatchRowCriteria(context.handle.get(), thr[0], thr[1]);
        } else {
            GpuAcNewtonSolver.setBatchRowCriteria(context.handle.get(), null, null);
        }
    }

    public GpuNewtonRaphson(LfNetwork network, NewtonRaphsonParameters parameters,
                            EquationSystem<AcVariableType, AcEquationType> equationSystem,
                            JacobianMatrix<AcVariableType, AcEquationType> j,
                            TargetVector<AcVariableType, AcEquationType> targetVector,
                            EquationVector<AcVariableType, AcEquationType> equationVector,
                            boolean detailedReport) {
        this(network, parameters, equationSystem, j, targetVector, equationVector, detailedReport,
                false, Double.MAX_VALUE, 0, 0, 0);
    }

    @SuppressWarnings("java:S107")
    public GpuNewtonRaphson(LfNetwork network, NewtonRaphsonParameters parameters,
                            EquationSystem<AcVariableType, AcEquationType> equationSystem,
                            JacobianMatrix<AcVariableType, AcEquationType> j,
                            TargetVector<AcVariableType, AcEquationType> targetVector,
                            EquationVector<AcVariableType, AcEquationType> equationVector,
                            boolean detailedReport, boolean driverOuterLoops,
                            double driverSlackDistTol, double driverReactiveLimitTol,
                            int driverMaxOuterIterations, int driverMaxPqPvSwitch) {
        super(network, equationSystem, j, targetVector, equationVector, detailedReport);
        this.parameters = Objects.requireNonNull(parameters);
        this.driverOuterLoops = driverOuterLoops;
        this.driverSlackDistTol = driverSlackDistTol;
        this.driverReactiveLimitTol = driverReactiveLimitTol;
        this.driverMaxOuterIterations = driverMaxOuterIterations;
        this.driverMaxPqPvSwitch = driverMaxPqPvSwitch;
        EquationSystemIndex<AcVariableType, AcEquationType> index = equationSystem.getIndex();
        // Purge listeners of ALREADY-COLLECTED solvers synchronously (we are on the solve thread,
        // no dispatch in flight) — this caps accumulation on a long-lived LfNetwork at ~1 while
        // keeping the cleaner thread away from the (non-thread-safe) listener lists.
        for (LfNetworkListener l : List.copyOf(network.getListeners())) {
            if (l instanceof GpuSolverInvalidation inv && inv.dead) {
                network.removeListener(l);
            }
        }
        index.addListener(invalidation);
        network.addListener(invalidation);
        // the cleaning action must not reference the solver itself (it would never become
        // phantom-reachable); the Invalidation listener object is referenced instead. It only
        // dead-flags: the index-side listener stays registered (no-op) for the es lifetime.
        GpuSolverInvalidation listener = invalidation;
        context.removeListeners = () -> listener.dead = true;
        CLEANER.register(this, context);
    }

    @Override
    public String getName() {
        return "GPU Newton-Raphson";
    }

    @Override
    public AcSolverResult run(VoltageInitializer voltageInitializer, ReportNode reportNode) {
        if (!GpuAcNewtonSolver.isAvailable()) {
            throw new PowsyblException("GPU Newton-Raphson requested but the native library is not available "
                    + "(build with native/build-gpu.sh and provide cuDSS — see GpuAcNewtonSolver)");
        }
        AcSolverUtil.initStateVector(network, equationSystem, voltageInitializer);

        AcSolverStatus status;
        int iterations = 0;
        int n = equationSystem.getIndex().getSortedVariablesToFind().size();
        try {
            long t0 = System.nanoTime();
            boolean rebuilt = false;
            if (invalidation.structureDirty) {
                // validate + build, through the shared cache (the batched SA reuses this extraction)
                GpuAcData data = GpuAcDataCache.extract(network, equationSystem, true, true);
                // Three-level reconciliation, decided by the DATA not the event: a changed CSR pattern
                // (index packs) is STRUCTURE -> rebuild + cuDSS re-analysis; an unchanged pattern with
                // only changed element PARAMETERS (tap r1/a1, shunt b) is VALUES -> a cheap in-place
                // pack re-upload reusing the analysis; an unchanged data is a no-op (context reused).
                if (!data.sameStructureAs(cachedData)) {
                    GpuAcNewtonSolver.destroyBatchContext(context.handle.getAndSet(0));
                    // BATCH-OF-1 (2026-09-08): the sequential solve runs on the BATCH context in
                    // pure-NR mode (skipRlPhase=1, slackDistTol=inf neutralize the in-kernel outer
                    // phases; OLF's own outer loops stay on the CPU around this solver). One kernel
                    // set instead of two: per-row PER_EQUATION_TYPE criteria, slope-SVC (mode 2),
                    // continuous phase (mode 3) and the vdl/ZIP fills come from the batch pipeline
                    // instead of being ported twice. The shared structure comes from the
                    // per-extraction cache — the batched SA building its context from the SAME
                    // extraction reuses it device-side.
                    context.handle.set(GpuAcNewtonSolver.createBatchContext(GpuAcStructures.handle(n, data), 1,
                            GpuBatchContextCache.resolveBatchMode(),
                            data.rlVrow(), data.rlMinQ(), data.rlMaxQ(), data.rlLoadQ(), data.rlVtarget(),
                            data.rlMonitored(), data.rowSlope(),
                            data.dsPhiRow(), data.dsFactor(), data.dsSlackPhiRow(), data.dsSlackTargetP(),
                            data.dsBaseTargetP(), data.dsMaxDelta(), data.dsMinDelta(),
                            data.dsMode(), data.dsMarginUp(), data.dsMarginDown(),
                            data.dsSlackMarginUp(), data.dsSlackMarginDown()));
                    cachedData = data;
                    rebuilt = true;
                    criteriaDirty = true;
                    CONTEXT_REBUILDS.incrementAndGet();
                } else if (!data.sameAs(cachedData)) {
                    GpuAcNewtonSolver.refreshStructureValues(GpuAcStructures.handle(n, cachedData),
                            data.cbIn(), data.obIn(), data.shIn(), data.hvIn(), data.ziIn());
                    // the SHARED structure now holds the new values — re-key it to the new
                    // extraction so content matching never aliases it under the old one
                    GpuAcStructures.rebind(cachedData, data);
                    cachedData = data;
                    VALUE_REFRESHES.incrementAndGet();
                }
                invalidation.structureDirty = false;
                invalidation.activationDirty = true;
            }
            if (invalidation.activationDirty) {
                activation = GpuAcActivationExtractor.extractActivation(network, equationSystem, n);
                varType = GpuAcActivationExtractor.extractVarType(equationSystem, n);
                invalidation.activationDirty = false;
                criteriaDirty = true;                        // per-row thresholds follow the variables
            }
            if (criteriaDirty) {
                installRowCriteria(n);
                criteriaDirty = false;
            }
            // Since the batch-of-1 route (2026-09-08) modes 2 (slope-SVC) and 3 (continuous phase)
            // solve on-device — the batch kernels carry them. Mode 4 (a PINNED remote gen-VC member,
            // a base reactive-limit switch during this engine run) stays DELIBERATELY CPU-routed:
            // the pack's DISTR_Q weights are STATIC extraction-time qPct, and OLF's pin renormalizes
            // them (updateReactiveKeys) with NO equation-system event — solving on-device would use
            // stale weights. Every post-pin solve of the run runs on the CPU (correct, just
            // un-accelerated).
            for (int mode : activation.rowMode()) {
                if (mode >= 4) {
                    throw new UnsupportedOperationException(
                            "row mode " + mode + " (pinned remote gen-VC member) uses stale static qPct on-device");
                }
            }
            double[] target = activation.applyTargets(targetVector.getArray());
            double[] x0 = equationSystem.getStateVector().get().clone();
            // Device-side state-vector scaling: NONE = undamped Newton; LINE_SEARCH backtracks on the
            // device (mirrors LineSearchStateVectorScaling); MAX_VOLTAGE_CHANGE clamps each step so the
            // largest |Δv|/|Δφ| stays within maxDv/maxDphi (mirrors MaxVoltageChangeStateVectorScaling).
            int lsMode = switch (parameters.getStateVectorScalingMode()) {
                case NONE -> 0;
                case LINE_SEARCH -> 1;
                case MAX_VOLTAGE_CHANGE -> 2;
            };
            long t1 = System.nanoTime();
            double convEpsPerEq = deviceConvEpsPerEq(parameters);
            if (lsMode == 1) {                               // batch loop LINE_SEARCH (ported 2026-09-08)
                GpuAcNewtonSolver.setBatchLineSearch(context.handle.get(),
                        parameters.getLineSearchStateVectorScalingMaxIteration(),
                        parameters.getLineSearchStateVectorScalingStepFold());
            }
            // Pure-NR batch-of-1: skipRlPhase=1 disables the in-kernel reactive-limit phase and
            // the huge slackDistTol makes the distributed-slack phase a no-op (fabs(rem) <= tol
            // -> return) — OLF's outer loops run on the CPU around this solver, exactly as before.
            // AC-emulation HVDC statuses: OLF's freezing/limits loops DEACTIVATE the hvdc coupling
            // terms for any non-LINEAR status (the injections ride the target vector as VSC station
            // targetP, re-extracted every solve) — mirror it with the hv element's disabled bit,
            // which fillHvdcBatch honors. Without this the fill emitted the LINEAR coupling on top
            // of the frozen/saturated injections (the former frozen-HVDC cpu-base-only condition).
            boolean[][] elementMask = new boolean[1][cachedData.getTotalElementCount()];
            List<LfHvdc> hvPacked = network.getHvdcs().stream()
                    .filter(h -> h.getBus1() != null && h.getBus2() != null && h.isAcEmulation())
                    .toList();                               // the packer's exact enumeration order
            int hvOffset = cachedData.getTotalElementCount() - hvPacked.size();
            for (int h = 0; h < hvPacked.size(); h++) {
                if (hvPacked.get(h).getAcEmulationControl().getAcEmulationStatus()
                        != LfHvdc.AcEmulationControl.AcEmulationStatus.LINEAR_MODE) {
                    elementMask[0][hvOffset + h] = true;
                }
            }
            double[] out = GpuAcNewtonSolver.solveBatchContext(context.handle.get(),
                    x0, target, new int[][] {activation.rowMode()}, activation.rowMode(),
                    elementMask,
                    new int[0], new double[0],
                    new int[0], new double[0], new double[0], new double[0], new double[0], null, null,
                    null, null,
                    parameters.getMaxIterations(), GPU_TOLERANCE,
                    driverOuterLoops ? driverReactiveLimitTol : 0.0,
                    driverOuterLoops ? driverSlackDistTol : 1e300, convEpsPerEq,
                    lsMode,
                    parameters.getMaxVoltageChangeStateVectorScalingMaxDv(),
                    parameters.getMaxVoltageChangeStateVectorScalingMaxDphi(),
                    varType, null,
                    null, null, null, null, null, null,
                    driverOuterLoops ? driverMaxOuterIterations : 0,
                    driverOuterLoops ? driverMaxPqPvSwitch : 0, 0,
                    driverOuterLoops ? 0 : 1,                // skipRlPhase: the driver owns RL now
                    0, null, null)[0];
            long t2 = System.nanoTime();
            LOGGER.debug("GPU Newton-Raphson timing: prepare {} ms (context {}), solve {} ms",
                    (t1 - t0) / 1_000_000, rebuilt ? "rebuilt" : "reused", (t2 - t1) / 1_000_000);
            iterations = (int) out[n];
            // write the device solution back; the equation vector re-evaluates from it
            equationSystem.getStateVector().set(Arrays.copyOf(out, n));
            equationVector.minus(targetVector);
            NewtonRaphsonStoppingCriteria.TestResult result =
                    parameters.getStoppingCriteria().test(equationVector.getArray(), equationSystem);
            // OLF_GPU_RESIDUAL_TRACE: the device solved to ITS residual, this is OLF's own residual at
            // the state it returned — a large value here means the device's equations disagree with
            // OLF's (a packing/row-mapping bug), not that the solver failed to converge. The
            // by-equation-type breakdown names the family to look at. (Test logback is broken —
            // stderr lever.)
            if (System.getenv("OLF_GPU_RESIDUAL_TRACE") != null) {
                double[] fx = equationVector.getArray();
                double n2 = 0;
                int worstRow = -1;
                double worst = 0;
                for (int i2 = 0; i2 < fx.length; i2++) {
                    n2 += fx[i2] * fx[i2];
                    if (Math.abs(fx[i2]) > worst) {
                        worst = Math.abs(fx[i2]);
                        worstRow = i2;
                    }
                }
                System.err.println("RESID_TRACE rows=" + fx.length + " cpuNorm2=" + Math.sqrt(n2)
                        + " cpuInfNorm=" + worst + " worstRow=" + worstRow
                        + " deviceIters=" + iterations + " stop=" + result.isStop());
                var worstEq = equationSystem.getIndex().getEquationAtColumn(worstRow);
                System.err.println("RESID_TRACE worst equation type=" + worstEq.getType()
                        + " elementNum=" + worstEq.getElementNum());
                java.util.Map<String, Integer> byType = new java.util.TreeMap<>();
                java.util.Map<String, Double> maxByType = new java.util.TreeMap<>();
                for (int i2 = 0; i2 < fx.length; i2++) {
                    if (Math.abs(fx[i2]) > 1e-4) {
                        String t = equationSystem.getIndex().getEquationAtColumn(i2).getType().toString();
                        byType.merge(t, 1, Integer::sum);
                        maxByType.merge(t, Math.abs(fx[i2]), Math::max);
                    }
                }
                System.err.println("RESID_TRACE rows over 1e-4 by type=" + byType + " max=" + maxByType);
            }
            status = result.isStop() ? AcSolverStatus.CONVERGED : AcSolverStatus.MAX_ITERATION_REACHED;
            // DEVICE-BROKEN result safety net: a non-finite or absurd residual is a device-side
            // numerical failure (e.g. the open zi-with-loaded-far-bus-behind-a-remote-member shape,
            // where the sequential solve returns phi ~1e22 while the SAME system solves on the CPU
            // and in the batch context — see GpuSeqBaseFallbackTest). Returning it would poison
            // OLF's outer loops and fail the whole run (an SA base would come back FAILED where the
            // stock CPU provider succeeds); throw into the CPU-fallback path instead, so the solve
            // reruns on the CPU Newton-Raphson: never wrong, only un-accelerated. Ordinary
            // non-convergence (a plausible norm) is NOT retried — the CPU would reproduce it.
            if (status == AcSolverStatus.MAX_ITERATION_REACHED
                    && (!Double.isFinite(result.getNorm()) || result.getNorm() > 1e2)) {
                throw new PowsyblException("GPU Newton-Raphson returned a device-broken state (|f(x)|="
                        + result.getNorm() + ")");
            }
            LOGGER.debug("GPU Newton-Raphson: {} iterations, |f(x)|={}, {}", iterations, result.getNorm(), status);
            if (System.getenv("OLF_GPU_SEQ_TRACE") != null) {   // test logback is broken — stderr trace
                System.err.println("SEQ_TRACE solve iters=" + iterations + " norm=" + result.getNorm()
                        + " devNorm=" + out[n + 1] + " status=" + status);
                double[] fv = equationVector.getArray();
                int[] tm = activation.targetMap();
                var vars = equationSystem.getIndex().getSortedVariablesToFind();
                for (int c = 0; c < fv.length; c++) {
                    if (Math.abs(fv[c]) > 1e-4) {
                        int row = -1;
                        for (int r = 0; r < tm.length; r++) {
                            if (tm[r] == c) {
                                row = r;
                            }
                        }
                        final int rr = row;
                        var vv = vars.stream().filter(x -> x.getRow() == rr).findFirst().orElse(null);
                        System.err.println("SEQ_TRACE   |f|=" + Math.abs(fv[c]) + " col=" + c + " row=" + row
                                + " mode=" + (row >= 0 ? activation.rowMode()[row] : -9)
                                + " var=" + (vv != null ? vv.getType() + "#" + vv.getElementNum() : "?"));
                    }
                }
            }
        } catch (RuntimeException e) {
            CPU_FALLBACK_COUNT.incrementAndGet();
            // the native context may be mid-iteration inconsistent — rebuild next run
            GpuAcNewtonSolver.destroyBatchContext(context.handle.getAndSet(0));
            cachedData = null;
            invalidation.structureDirty = true;
            if (Boolean.parseBoolean(System.getProperty(CPU_FALLBACK_PROPERTY, "true"))) {
                // Graceful degradation: anything the GPU path cannot handle — an extractor
                // validation rejection (unsupported variable/equation/term, remote control,
                // zero-impedance shapes) or a native/device error — is re-run on the standard
                // CPU Newton-Raphson with the SAME equation system and parameters. This makes
                // the GPU solver safe as a DEFAULT (e.g. the SA base load flow): never wrong,
                // only un-accelerated.
                LOGGER.warn("GPU Newton-Raphson unavailable for this run ({}); falling back to the CPU Newton-Raphson",
                        e.toString());
                return new NewtonRaphson(network, parameters, equationSystem, j, targetVector, equationVector,
                        detailedReport).run(voltageInitializer, reportNode);
            }
            LOGGER.error("GPU Newton-Raphson failed: {}", e.toString());
            status = AcSolverStatus.SOLVER_FAILED;
        }

        if (status == AcSolverStatus.CONVERGED || parameters.isAlwaysUpdateNetwork()) {
            AcSolverUtil.updateNetwork(network, equationSystem);
        }

        Map<Integer, Double> slackBusActivePowerMismatch = new TreeMap<>();
        for (LfSynchronousNetwork lfScNetwork : network.getSynchronousNetworks()) {
            slackBusActivePowerMismatch.put(lfScNetwork.getNumSC(), lfScNetwork.getSlackBuses().stream().mapToDouble(LfBus::getMismatchP).sum());
        }
        return new AcSolverResult(status, iterations, slackBusActivePowerMismatch);
    }
}
