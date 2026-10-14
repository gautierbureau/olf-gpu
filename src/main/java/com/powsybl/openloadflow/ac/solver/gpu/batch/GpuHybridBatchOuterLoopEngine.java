/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.commons.PowsyblException;
import com.powsybl.openloadflow.ac.AcLoadFlowContext;
import com.powsybl.openloadflow.ac.AcOuterLoopContext;
import com.powsybl.openloadflow.ac.AcTargetVector;
import com.powsybl.openloadflow.ac.AcloadFlowEngine;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivation;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.outerloop.AcOuterLoop;
import com.powsybl.openloadflow.ac.solver.AcSolverResult;
import com.powsybl.openloadflow.ac.solver.AcSolverStatus;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfHvdc;
import com.powsybl.openloadflow.network.LfLoad;
import com.powsybl.openloadflow.network.LfNetwork;
import org.apache.commons.lang3.tuple.Pair;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The Java-outer-loop hybrid batch security-analysis driver: batch the inner Newton solves on the device
 * (cuDSS via {@link GpuAcNewtonSolver#solveBatchContext}), and make the outer-loop DECISIONS with OLF's own
 * validated Java outer loops (via {@link GpuHybridOuterLoopBridge}) — so the decisions match the CPU exactly by
 * construction, with no second outer-loop implementation to keep in sync.
 *
 * <p>One round solves every still-active scenario's inner Newton (batched), then for each scenario loads that
 * state, runs OLF's outer loops (which mutate the activation / distributed targets), re-extracts the device row
 * mode, and warm-restarts. Rounds repeat until every scenario is STABLE or routed. Four production optimizations:
 *
 * <ul>
 *   <li><b>Cross-round compaction</b> — re-solve only still-active scenarios each round (recreate the device
 *       batch context, cheap with empty DS/RL packs, when the active count shrinks);</li>
 *   <li><b>Compact per-scenario snapshot</b> ({@link HybridScenarioState}) — isolate scenarios' controller-bus /
 *       HVDC state + generator {@code targetP}, not a full {@code NetworkState};</li>
 *   <li><b>Snapshot save-skip</b> — snapshot a scenario only when it continues to the next round;</li>
 *   <li><b>Straggler early-routing</b> — route a scenario to CPU after {@code rejectRouteThreshold} consecutive
 *       device rejects (status {@code -2}) or on exceeding the outer-iteration budget.</li>
 * </ul>
 *
 * <p>See {@code docs/hybrid-outer-loop-production-design.md}. The correctness invariants (equation system built
 * AFTER the base LF's DC init; {@code initialTargetP} anchored at the original generation; device state-vector
 * scaling mirroring the CPU solver; per-scenario state isolation; faithful nested {@code runOuterLoop} in the
 * bridge) are the caller's / the bridge's responsibility.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuHybridBatchOuterLoopEngine {

    private static final int[] EMPTY_INT = new int[0];

    private static final org.slf4j.Logger LOGGER =
            org.slf4j.LoggerFactory.getLogger(GpuHybridBatchOuterLoopEngine.class);

    private static final int MAX_ROUNDS = 40;

    private GpuHybridBatchOuterLoopEngine() {
    }

    /** LIVE distribution coefficients for one scenario (weights for DISTR_Q, coefs for DISTR_RHO),
     *  re-read off the mutated network exactly as OLF's live suppliers would evaluate them. Returns
     *  null when every value equals the static pack — the common no-pin/no-outage case. */
    private static double[] liveDqWeights(GpuAcData d, LfBus[] memberBus) {
        int ndq = d.dqKMember().length;
        double[] w = null;
        for (int c = 0; c < ndq; c++) {
            int kM = d.dqKMember()[c];
            if (kM < 0) {
                continue;
            }
            double live = memberBus[kM].getRemoteControlReactivePercent()
                    - (d.dqJMember()[c] == kM ? 1.0 : 0.0);
            if (live != d.dqWeight()[c]) {
                if (w == null) {
                    w = d.dqWeight().clone();
                }
                w[c] = live;
            }
        }
        return w;
    }

    private static double[] liveDrCoefs(GpuAcData d, LfBranch[] kBranch, LfBranch[] jBranch) {
        int ndr = d.drRow().length;
        double[] cfs = null;
        for (int c = 0; c < ndr; c++) {
            LfBranch k = kBranch[c];
            LfBranch j = jBranch[c];
            if (k == null || j == null) {
                continue;
            }
            double live;
            if (j.isDisabled() || k.isDisabled()) {
                live = 0.0;                                  // terms touching a disabled element deactivate
            } else {
                long count = k.getVoltageControl().map(vc -> vc.getMergedControllerElements().stream()
                        .filter(b -> !b.isDisabled()).count()).orElse(0L);
                live = count == 0 ? 0.0 : 1.0 / count - (j == k ? 1.0 : 0.0);
            }
            if (live != d.drCoef()[c]) {
                if (cfs == null) {
                    cfs = d.drCoef().clone();
                }
                cfs[c] = live;
            }
        }
        return cfs;
    }

    /** Key of one equation-system variable: (type, element) — stable across activation AND layout
     *  changes, unlike the row. */
    private static long varKey(int typeOrdinal, int elementNum) {
        return ((long) typeOrdinal << 32) ^ (elementNum & 0xFFFFFFFFL);
    }

    /** The live-row -> base-row variable map for the CURRENT es layout, or {@code null} when the
     *  live layout equals the base one (the common case: OLF's index assigns rows in sorted
     *  variable order, so an unchanged variable SET means unchanged rows). A shared-TVC member
     *  OUTAGE removes the member's BRANCH_RHO1 variable from the live index (every referencing
     *  equation deactivates with the element), shrinking and SHIFTING the live rows off the
     *  device's frozen base superset — this map is how the hybrid's pushes/reads cross that. */
    private static int[] liveToBaseMap(EquationSystem<AcVariableType, AcEquationType> es,
                                       java.util.Map<Long, Integer> baseRowByVar, int n) {
        var vars = es.getIndex().getSortedVariablesToFind();
        if (vars.size() == n) {
            return null;
        }
        int[] map = new int[vars.size()];
        for (var v : vars) {
            Integer b = baseRowByVar.get(varKey(v.getType().ordinal(), v.getElementNum()));
            if (b == null) {
                throw new com.powsybl.commons.PowsyblException(
                        "GPU hybrid: live variable absent from the base layout: " + v);
            }
            map[v.getRow()] = b;
        }
        return map;
    }

    /** Base-indexed device state -> live-row state for the bridge/es (identity when map is null). */
    private static double[] compactState(double[] base, int[] liveToBase) {
        if (liveToBase == null) {
            return base;
        }
        double[] out = new double[liveToBase.length];
        for (int i = 0; i < liveToBase.length; i++) {
            out[i] = base[liveToBase[i]];
        }
        return out;
    }

    /** Live-row modes -> base-indexed device row modes; a base row with NO live variable becomes an
     *  IDENTITY pin (mode 0) — the device analogue of the removed variable, held at the base state
     *  by {@link #expandTargets}/{@link #expandStateInto} (exactly the DEVICE path's outaged-member
     *  rho encoding, plan A5). */
    private static int[] expandModes(int[] liveModes, int[] liveToBase, int n) {
        if (liveToBase == null) {
            return liveModes;
        }
        int[] out = new int[n];
        for (int i = 0; i < liveToBase.length; i++) {
            out[liveToBase[i]] = liveModes[i];
        }
        return out;
    }

    private static double[] expandTargets(double[] liveTargets, int[] liveToBase, double[] baseState, int n) {
        if (liveToBase == null) {
            return liveTargets;
        }
        double[] out = new double[n];
        System.arraycopy(baseState, 0, out, 0, n);           // unmapped rows: identity target = base value
        for (int i = 0; i < liveToBase.length; i++) {
            out[liveToBase[i]] = liveTargets[i];
        }
        return out;
    }

    private static void expandStateInto(double[] liveState, int[] liveToBase, double[] baseState, double[] dst, int n) {
        if (liveToBase == null) {
            System.arraycopy(liveState, 0, dst, 0, n);
            return;
        }
        System.arraycopy(baseState, 0, dst, 0, n);           // unmapped rows sit AT their pin from x0 on
        for (int i = 0; i < liveToBase.length; i++) {
            dst[liveToBase[i]] = liveState[i];
        }
    }

    /** The branch's current static device row [y g1 g2 b1 b2 ksi r1 a1] — the cbIn packing. */
    private static double[] cbRowOf(LfBranch branch) {
        var pi = branch.getPiModel();
        return new double[] {pi.getY(), pi.getG1(), pi.getG2(), pi.getB1(), pi.getB2(), pi.getKsi(), pi.getR1(), pi.getA1()};
    }

    /**
     * Inputs to {@link #solve}. The caller owns the device {@code structure} (created once, destroyed after);
     * the engine owns the batch-context {@code handle} lifecycle (created / recreated on compaction / destroyed).
     *
     * @param structure device structure handle (from {@code GpuAcNewtonSolver.createStructure})
     * @param mode      batch mode (3 = UBATCH)
     * @param gpuData   extracted device data (DS/RL packs — EMPTY for the hybrid; used to (re)create the context)
     * @param nElements total device element count (for the per-scenario disabled masks)
     * @param es        the equation system built AFTER the base LF's DC init (see the design)
     * @param context   the base AC loadflow context — its outer loops are replayed per scenario
     * @param baseState / baseTargets / baseRowMode the base-converged device state / targets / row mode
     * @param ctgs      the qualified single-branch contingencies (one scenario each)
     * @param mapper    element mapper (contingency branch -> device element index)
     * @param gens / baseGenP / originalGenP the generators, their base-distributed targetP, and their ORIGINAL
     *                  setpoints (the distributed-slack accounting anchor)
     * @param ctrlBuses / hvdcs / transformerBranches the compact-snapshot element sets (transformer controller
     *                  branches empty unless the run has transformer voltage control)
     * @param lsMode / lsMaxDv / lsMaxDphi / varType device state-vector scaling mirroring the CPU solver
     * @param maxIterations device inner-Newton max iterations
     * @param maxOuter      per-scenario outer-iteration budget before routing to CPU
     * @param rejectRouteThreshold consecutive device rejects before routing a straggler to CPU
     */
    public record Inputs(long structure, int mode, GpuAcData gpuData, int nElements,
                         LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                         AcLoadFlowContext context, int n,
                         double[] baseState, double[] baseTargets, int[] baseRowMode,
                         List<BatchContingency> ctgs, GpuBatchElementMapper mapper,
                         List<LfGenerator> gens, double[] baseGenP, double[] originalGenP,
                         List<LfBus> ctrlBuses, List<LfHvdc> hvdcs, List<LfBranch> transformerBranches,
                         int lsMode, double lsMaxDv, double lsMaxDphi, int[] varType,
                         int maxIterations, int maxOuter, int rejectRouteThreshold,
                         double tol, double reactiveLimitTol, double slackDistTol, double convEpsPerEq,
                         double[][] rowCriteria) {

        /** The same inputs over a SLICE of the scenarios — {@code ctgs} is the only per-scenario field. */
        Inputs withCtgs(List<BatchContingency> chunkCtgs) {
            return new Inputs(structure, mode, gpuData, nElements, network, es, context, n,
                    baseState, baseTargets, baseRowMode, chunkCtgs, mapper, gens, baseGenP, originalGenP,
                    ctrlBuses, hvdcs, transformerBranches, lsMode, lsMaxDv, lsMaxDphi, varType,
                    maxIterations, maxOuter, rejectRouteThreshold, tol, reactiveLimitTol, slackDistTol,
                    convEpsPerEq, rowCriteria);
        }
    }

    /**
     * Per-scenario outcome + diagnostics. {@code gpuConverged[s]} false means the scenario was routed to the CPU
     * fallback. The arrays are indexed by the scenario order in {@link Inputs#ctgs}.
     *
     * @param states          [nS][n+3] final device state per scenario: x(n), iterations, ||F||inf, status
     * @param round0States    [nS][] the round-0 device state (base targets, pure inner) — NULL when the
     *                        solve was CHUNKED (diagnostic only, and holding it for every scenario of a
     *                        full run costs hundreds of MB); likewise rowMode and scenTargets
     * @param rowMode         [nS][] final per-scenario device row mode
     * @param scenTargets     [nS][] final per-scenario target vector
     * @param retired         [nS] scenario reached a terminal state (converged OR routed) within the round budget
     * @param routedByBudget  [nS] scenario was routed to the CPU fallback (straggler / budget / device reject)
     * @param bridges         [nS] the per-scenario outer-loop bridges (loop names / unstable counts,
     *                        diagnostics) — NULL when the solve was CHUNKED, for the same reason
     */
    public record Result(double[][] states, double[][] round0States, int[][] rowMode, double[][] scenTargets,
                         boolean[] retired, boolean[] routedByBudget, GpuHybridOuterLoopBridge[] bridges,
                         int rounds, long solveNanos, long javaNanos, long snapshotNanos, long javaCalls,
                         double[] distributedActivePower) {

        /** true = converged on the GPU (reached a terminal state and was NOT routed to CPU). */
        public boolean gpuConverged(int scenario) {
            return retired[scenario] && !routedByBudget[scenario];
        }

        public int gpuConvergedCount() {
            int c = 0;
            for (int s = 0; s < retired.length; s++) {
                if (gpuConverged(s)) {
                    c++;
                }
            }
            return c;
        }
    }

    private static long createHandle(long structure, int count, int mode, GpuAcData d, double[][] rowCriteria) {
        long h = createHandleRaw(structure, count, mode, d);
        // PER_EQUATION_TYPE stopping criterion: install the per-row thresholds so the hybrid's
        // inner solves stop exactly where OLF's PerEquationTypeStoppingCriteria would.
        GpuAcNewtonSolver.installRowCriteria(h, rowCriteria);
        // plan-B4 (JAVA): when a remote group has a pilot-Q relocation table, OLF's RL loop may pin
        // the FREE member — the pushed mode-4 row re-picks the carrier in the fills, and these
        // tables gate the new carrier's DISTR_Q row off (its row hosts the relocated pilot balance,
        // whose target arrives through the per-round activation/target push).
        boolean anyQr = false;
        for (int g : d.dqMemberQrGroup()) {
            anyQr |= g >= 0;
        }
        GpuAcNewtonSolver.setBatchDqQrSuppression(h,
                anyQr ? d.dqJMember() : null,
                anyQr ? d.dqMemberQrGroup() : null);
        return h;
    }

    private static long createHandleRaw(long structure, int count, int mode, GpuAcData d) {
        long h = GpuAcNewtonSolver.createBatchContext(structure, count, mode,
                d.rlVrow(), d.rlMinQ(), d.rlMaxQ(), d.rlLoadQ(), d.rlVtarget(), d.rlMonitored(), d.rowSlope(),
                d.dsPhiRow(), d.dsFactor(), d.dsSlackPhiRow(), d.dsSlackTargetP(),
                d.dsBaseTargetP(), d.dsMaxDelta(), d.dsMinDelta(),
                d.dsMode(), d.dsMarginUp(), d.dsMarginDown(), d.dsSlackMarginUp(), d.dsSlackMarginDown());
        GpuBatchOuterLoopTables.installRlElection(h, d);
        return h;
    }

    /**
     * Solve every scenario, in CHUNKS sized like the DEVICE path's.
     *
     * <p>The uniform batched factorization holds an LU per scenario, so one context for the whole
     * contingency set is priced against the whole set: on rte6515 with 9042 branch contingencies
     * {@code createBatchContext} returned {@code CUDA error: out of memory}, the precomputer's
     * catch-all swallowed it, and the ENTIRE run silently fell back to per-contingency CPU solves —
     * a 0% batched share that reads like a scope rejection rather than an allocation failure. The
     * DEVICE path never had this because it has always chunked.
     *
     * <p>Chunking is a pure partition here: {@code ctgs} is the only per-scenario field of
     * {@link Inputs}, every result array is indexed by scenario, and the per-scenario bridges restore
     * the base network state, so a chunk boundary changes nothing but which scenarios share a device
     * context. A run whose recommended chunk already covers everything takes the single-context path
     * unchanged.
     *
     * <p>An allocation failure that still happens (the auto-size probe is an estimate) halves the
     * chunk and retries the SAME scenarios, remembering the ceiling for later runs — the chunk
     * solver's rule, so both paths converge on the same size on a small card.
     */
    public static Result solve(Inputs in) {
        long t0 = System.nanoTime();
        Result r = solveAll(in);
        if (GpuBatchPhaseStats.ENABLED) {
            // The hybrid's cost is three distinct things and only one of them is the device: the batched
            // inner solves, OLF's Java outer loops replayed PER SCENARIO, and the per-scenario state
            // snapshots that let a scenario be resumed in the next round. Printing the split is the
            // difference between optimising the hybrid and guessing at it.
            long wall = System.nanoTime() - t0;
            System.err.printf("HYBRID_PHASE_TIME scenarios=%d rounds=%d wall=%d ms | deviceSolve=%d ms "
                    + "javaOuterLoops=%d ms snapshots=%d ms | javaCalls=%d other=%d ms%n",
                    in.ctgs().size(), r.rounds(), wall / 1_000_000, r.solveNanos() / 1_000_000,
                    r.javaNanos() / 1_000_000, r.snapshotNanos() / 1_000_000, r.javaCalls(),
                    (wall - r.solveNanos() - r.javaNanos() - r.snapshotNanos()) / 1_000_000);
        }
        return r;
    }

    private static Result solveAll(Inputs in) {
        int nS = in.ctgs().size();
        int chunk = GpuBatchContextCache.resolveChunkSize(nS, in.n(), in.gpuData());
        if (chunk >= nS) {
            return solveChunk(in);
        }
        // Only `states` (plus the per-scenario flags) is consumed in production; rowMode / scenTargets /
        // round0States are diagnostics a validation test reads, and that test runs cases small enough to
        // take the single-chunk path above. Retaining all four at full size is ~2.4 GB of heap at 6474
        // scenarios on a 13040-row system (675 + 675 + 675 + 338 MB) and OOMs the JVM two thirds of the
        // way through a full run, so the chunked path keeps only what is used.
        double[][] states = new double[nS][];
        boolean[] retired = new boolean[nS];
        boolean[] routedByBudget = new boolean[nS];
        double[] distributed = new double[nS];
        int rounds = 0;
        long tSolve = 0;
        long tJava = 0;
        long tSnap = 0;
        long nJavaCalls = 0;
        int start = 0;
        while (start < nS) {
            int size = Math.min(chunk, nS - start);
            Result r;
            try {
                r = solveChunk(in.withCtgs(in.ctgs().subList(start, start + size)));
            } catch (RuntimeException e) {
                if (chunk <= 1 || !isDeviceAllocationFailure(e)) {
                    throw e;
                }
                GpuBatchContextCache.noteChunkAllocFailure(chunk);
                int retryChunk = Math.max(1, chunk / 2);
                LOGGER.warn("GPU hybrid SA: device allocation failed at chunk {} ({}); retrying at chunk {}",
                        chunk, e.getMessage(), retryChunk);
                // the test harness's logback appenders are broken — a silent shrink would look like an
                // unexplained slowdown, so say it on stderr too (the chunk solver's convention)
                System.err.println("GPU_HYBRID_CHUNK_SHRINK device allocation failed at chunk " + chunk
                        + "; retrying at " + retryChunk);
                chunk = retryChunk;
                continue;
            }
            System.arraycopy(r.states(), 0, states, start, size);
            System.arraycopy(r.retired(), 0, retired, start, size);
            System.arraycopy(r.distributedActivePower(), 0, distributed, start, size);
            System.arraycopy(r.routedByBudget(), 0, routedByBudget, start, size);
            // NOT the bridges: each one retains a full per-scenario set of OLF outer-loop objects and
            // their contexts (reactive-limits / distributed-slack context data included). Holding 6474 of
            // them across the whole run is what still exhausted the heap after the arrays above were
            // trimmed; dropping them lets each chunk's bridges be collected when the chunk ends.

            rounds = Math.max(rounds, r.rounds());   // rounds is a DEPTH, not a total
            tSolve += r.solveNanos();
            tJava += r.javaNanos();
            tSnap += r.snapshotNanos();
            nJavaCalls += r.javaCalls();
            start += size;
        }
        return new Result(states, null, null, null,
                retired, routedByBudget, null, rounds, tSolve, tJava, tSnap, nJavaCalls, distributed);
    }

    /** Every per-scenario array the native solve dereferences must have exactly {@code nA} non-null
     *  rows; a null row faults inside JNI with no Java frame identifying it. */
    private static void checkScenarioRows(int[][] rowMode, boolean[][] disabled, int nA) {
        require("rowMode", rowMode == null ? -1 : rowMode.length, nA);
        require("disabled", disabled == null ? -1 : disabled.length, nA);
        for (int i = 0; i < nA; i++) {
            if (rowMode[i] == null || disabled[i] == null) {
                throw new PowsyblException("GPU hybrid: null per-scenario row at active index " + i + " of " + nA
                        + " (rowMode=" + (rowMode[i] != null) + " disabled=" + (disabled[i] != null) + ")");
            }
        }
    }

    private static void require(String what, int actual, int expected) {
        if (actual != expected) {
            throw new PowsyblException("GPU hybrid: " + what + " has " + actual
                    + " rows, expected " + expected + " (the context's active scenario count)");
        }
    }

    /** {@code OLF_GPU_DIRTY_PROBE=1}: how many DISTINCT elements a scenario's outer-loop step actually
     *  mutates, versus the 1006 controller buses the snapshot saves and restores wholesale. Decides
     *  whether dirty-tracking the snapshot is worth building at all. */
    private static final DirtyProbe DIRTY_PROBE =
            System.getenv("OLF_GPU_DIRTY_PROBE") != null ? new DirtyProbe() : null;

    private static final class DirtyProbe extends com.powsybl.openloadflow.network.AbstractLfNetworkListener {

        private final java.util.Set<Object> touched = new java.util.HashSet<>();
        private long scenarios;
        private long touchedTotal;
        private long ofTotal;
        private boolean attached;

        void startScenario() {
            touched.clear();
        }

        void endScenario(int ctrlBusCount) {
            scenarios++;
            touchedTotal += touched.size();
            ofTotal += ctrlBusCount;
            if (scenarios % 200 == 0) {
                System.err.printf("DIRTY_PROBE scenarios=%d avgTouchedElements=%.1f of %d controller buses | new events: %s%n",
                        scenarios, touchedTotal / (double) scenarios, ctrlBusCount, newEventCounts());
            }
        }

        void attachTo(LfNetwork network) {
            if (!attached) {
                network.addListener(this);
                attached = true;
            }
        }

        @Override
        public void onGeneratorVoltageControlChange(LfBus controllerBus, boolean e) {
            touched.add(controllerBus);
        }

        @Override
        public void onGeneratorReactivePowerControlChange(LfBus controllerBus, boolean e) {
            touched.add(controllerBus);
        }

        @Override
        public void onGenerationReactivePowerTargetChange(LfBus bus, double o, double nw) {
            touched.add(bus);
        }

        @Override
        public void onGenerationActivePowerTargetChange(LfGenerator g, double o, double nw) {
            touched.add(g);
        }

        @Override
        public void onLoadActivePowerTargetChange(com.powsybl.openloadflow.network.LfLoad l, double o, double nw) {
            touched.add(l);
        }

        @Override
        public void onLoadReactivePowerTargetChange(com.powsybl.openloadflow.network.LfLoad l, double o, double nw) {
            touched.add(l);
        }

        @Override
        public void onShuntVoltageControlChange(com.powsybl.openloadflow.network.LfShunt sh, boolean e) {
            touched.add(sh);
        }

        @Override
        public void onShuntSusceptanceChange(com.powsybl.openloadflow.network.LfShunt sh, double b) {
            touched.add(sh);
        }

        @Override
        public void onDisableChange(com.powsybl.openloadflow.network.LfElement el, boolean d) {
            touched.add(el);
        }

        @Override
        public void onTapPositionChange(LfBranch br, int o, int nw) {
            touched.add(br);
        }

        // The callbacks added to OLF for exactly this purpose. Counted separately: if any of them never
        // fires, dirty-tracking built on it would silently fail to restore that field.
        private long qLimitTypeEvents;
        private long frozenEvents;
        private long controlTypeEvents;
        private long absVariableEvents;
        private long disablingStatusEvents;

        @Override
        public void onQLimitTypeChange(LfBus bus, LfBus.QLimitType t) {
            qLimitTypeEvents++;
            touched.add(bus);
        }

        @Override
        public void onGenerationTargetQFrozenChange(LfBus bus, boolean frozen) {
            frozenEvents++;
            touched.add(bus);
        }

        @Override
        public void onGeneratorControlTypeChange(LfGenerator g, LfGenerator.GeneratorControlType t) {
            controlTypeEvents++;
            touched.add(g);
        }

        @Override
        public void onLoadAbsVariableTargetPChange(com.powsybl.openloadflow.network.LfLoad l, double o, double nw) {
            absVariableEvents++;
            touched.add(l);
        }

        @Override
        public void onLoadOriginalDisablingStatusChange(com.powsybl.openloadflow.network.LfLoad l) {
            disablingStatusEvents++;
            touched.add(l);
        }

        String newEventCounts() {
            return "qLimitType=" + qLimitTypeEvents + " targetQFrozen=" + frozenEvents
                    + " controlType=" + controlTypeEvents + " absVariableTargetP=" + absVariableEvents
                    + " disablingStatus=" + disablingStatusEvents;
        }
    }

    /**
     * Which elements a scenario's outer-loop step actually mutated.
     *
     * <p>The engine used to save and restore ~7100 elements per scenario per round — every controller
     * bus, every generator, every load, every generator voltage control — when a step touches ~80 of
     * them (measured, {@code OLF_GPU_DIRTY_PROBE}). This records the touched ones so the swap between
     * two scenarios costs the size of their dirty sets instead of the size of the network.
     *
     * <p>Bus GRANULARITY for anything a {@link HybridScenarioState} covers: {@code BusState} saves a bus
     * together with its generators and loads, so a touched generator marks its bus dirty and the bus is
     * saved and restored whole. Generators, loads and voltage controls are ALSO tracked in their own
     * right, because the engine keeps network-wide arrays for them (a generator on a non-controller bus
     * is moved by distributed slack and no BusState covers it).
     *
     * <p>MUTED during restore: several of the listener events (notably the load disabling-status one)
     * fire while state is being put back, and recording those would mark everything dirty.
     */
    private static final class DirtyTracker extends com.powsybl.openloadflow.network.AbstractLfNetworkListener {

        private final java.util.Map<LfBus, Integer> busIndex = new java.util.IdentityHashMap<>();
        private final java.util.Map<LfGenerator, Integer> genIndex = new java.util.IdentityHashMap<>();
        private final java.util.Map<com.powsybl.openloadflow.network.LfLoad, Integer> loadIndex = new java.util.IdentityHashMap<>();
        private final java.util.Map<com.powsybl.openloadflow.network.GeneratorVoltageControl, Integer> vcIndex = new java.util.IdentityHashMap<>();
        private final java.util.Map<com.powsybl.openloadflow.network.LfShunt, LfBus> shuntBus = new java.util.IdentityHashMap<>();
        private final java.util.Map<LfGenerator, LfBus> genBus = new java.util.IdentityHashMap<>();

        private final java.util.BitSet buses = new java.util.BitSet();
        private final java.util.BitSet gens = new java.util.BitSet();
        private final java.util.BitSet loads = new java.util.BitSet();
        private final java.util.BitSet vcs = new java.util.BitSet();
        private boolean recording;

        DirtyTracker(List<LfBus> ctrlBuses, List<LfGenerator> allGens,
                     List<com.powsybl.openloadflow.network.LfLoad> allLoads,
                     List<com.powsybl.openloadflow.network.GeneratorVoltageControl> allVcs) {
            for (int i = 0; i < ctrlBuses.size(); i++) {
                LfBus bus = ctrlBuses.get(i);
                busIndex.put(bus, i);
                bus.getGenerators().forEach(g -> genBus.put(g, bus));
                bus.getShunt().ifPresent(sh -> shuntBus.put(sh, bus));
                bus.getControllerShunt().ifPresent(sh -> shuntBus.put(sh, bus));
                bus.getSvcShunt().ifPresent(sh -> shuntBus.put(sh, bus));
            }
            for (int i = 0; i < allGens.size(); i++) {
                genIndex.put(allGens.get(i), i);
            }
            for (int i = 0; i < allLoads.size(); i++) {
                loadIndex.put(allLoads.get(i), i);
            }
            for (int i = 0; i < allVcs.size(); i++) {
                vcIndex.put(allVcs.get(i), i);
            }
        }

        void start() {
            buses.clear();
            gens.clear();
            loads.clear();
            vcs.clear();
            recording = true;
        }

        void stop() {
            recording = false;
        }

        java.util.BitSet accumulateBuses(java.util.BitSet into) {
            return accumulate(into, buses);
        }

        java.util.BitSet accumulateGens(java.util.BitSet into) {
            return accumulate(into, gens);
        }

        java.util.BitSet accumulateLoads(java.util.BitSet into) {
            return accumulate(into, loads);
        }

        java.util.BitSet accumulateVcs(java.util.BitSet into) {
            return accumulate(into, vcs);
        }

        private static java.util.BitSet accumulate(java.util.BitSet into, java.util.BitSet round) {
            java.util.BitSet out = into == null ? new java.util.BitSet() : into;
            out.or(round);
            return out;
        }

        private void markBus(LfBus bus) {
            if (recording) {
                Integer i = busIndex.get(bus);
                if (i != null) {
                    buses.set(i);
                }
            }
        }

        private void markGenerator(LfGenerator g) {
            if (recording) {
                Integer i = genIndex.get(g);
                if (i != null) {
                    gens.set(i);
                }
                LfBus bus = genBus.get(g);
                if (bus != null) {
                    markBus(bus);
                }
            }
        }

        private void markLoad(com.powsybl.openloadflow.network.LfLoad load) {
            if (recording) {
                Integer i = loadIndex.get(load);
                if (i != null) {
                    loads.set(i);
                }
                markBus(load.getBus());
            }
        }

        @Override
        public void onGeneratorVoltageControlChange(LfBus b, boolean e) {
            markBus(b);
        }

        @Override
        public void onGeneratorReactivePowerControlChange(LfBus b, boolean e) {
            markBus(b);
        }

        @Override
        public void onGenerationReactivePowerTargetChange(LfBus b, double o, double nw) {
            markBus(b);
        }

        @Override
        public void onQLimitTypeChange(LfBus b, LfBus.QLimitType t) {
            markBus(b);
        }

        @Override
        public void onGenerationTargetQFrozenChange(LfBus b, boolean f) {
            markBus(b);
        }

        @Override
        public void onGenerationActivePowerTargetChange(LfGenerator g, double o, double nw) {
            markGenerator(g);
        }

        @Override
        public void onGeneratorControlTypeChange(LfGenerator g, LfGenerator.GeneratorControlType t) {
            markGenerator(g);
        }

        @Override
        public void onLoadActivePowerTargetChange(com.powsybl.openloadflow.network.LfLoad l, double o, double nw) {
            markLoad(l);
        }

        @Override
        public void onLoadReactivePowerTargetChange(com.powsybl.openloadflow.network.LfLoad l, double o, double nw) {
            markLoad(l);
        }

        @Override
        public void onLoadAbsVariableTargetPChange(com.powsybl.openloadflow.network.LfLoad l, double o, double nw) {
            markLoad(l);
        }

        @Override
        public void onLoadOriginalDisablingStatusChange(com.powsybl.openloadflow.network.LfLoad l) {
            markLoad(l);
        }

        @Override
        public void onShuntVoltageControlChange(com.powsybl.openloadflow.network.LfShunt sh, boolean e) {
            markShunt(sh);
        }

        @Override
        public void onShuntSusceptanceChange(com.powsybl.openloadflow.network.LfShunt sh, double b) {
            markShunt(sh);
        }

        private void markShunt(com.powsybl.openloadflow.network.LfShunt sh) {
            if (recording) {
                LfBus bus = shuntBus.get(sh);
                if (bus != null) {
                    markBus(bus);
                }
            }
        }

        @Override
        public void onGeneratorVoltageControlTargetChange(com.powsybl.openloadflow.network.GeneratorVoltageControl c, double t) {
            if (recording) {
                Integer i = vcIndex.get(c);
                if (i != null) {
                    vcs.set(i);
                }
            }
        }

        @Override
        public void onDisableChange(com.powsybl.openloadflow.network.LfElement el, boolean d) {
            if (recording) {
                if (el instanceof LfBus b) {
                    markBus(b);
                } else if (el instanceof LfGenerator g) {
                    markGenerator(g);
                } else if (el instanceof com.powsybl.openloadflow.network.LfLoad l) {
                    markLoad(l);
                }
            }
        }
    }

    private static <T> List<T> subList(List<T> all, int[] indices) {
        List<T> out = new ArrayList<>(indices.length);
        for (int i : indices) {
            out.add(all.get(i));
        }
        return out;
    }

    private static double[] valuesOf(int[] indices, java.util.function.IntToDoubleFunction get) {
        double[] out = new double[indices.length];
        for (int k = 0; k < indices.length; k++) {
            out[k] = get.applyAsDouble(indices[k]);
        }
        return out;
    }

    /** A device-side allocation failure, as the native layer reports it. */
    private static boolean isDeviceAllocationFailure(RuntimeException e) {
        String m = e.getMessage();
        return m != null && (m.contains("out of memory") || m.contains("OUT_OF_MEMORY")
                || m.contains("batched context creation failed"));
    }

    private static Result solveChunk(Inputs in) {
        int nS = in.ctgs().size();
        int n = in.n();

        // OLF's outer loops, one context PER (scenario, loop) via the core factory — no reflection, and
        // filtered/initialized exactly as AcloadFlowEngine.run() does, so per-scenario contexts match core.
        GpuHybridOuterLoopBridge[] bridges = new GpuHybridOuterLoopBridge[nS];
        for (int s = 0; s < nS; s++) {
            // initialize=false: the loops are initialized PER SCENARIO by the bridge (after restoring the base),
            // exactly as OLF re-runs the outer loops per contingency. Initializing here would enable
            // transformer/shunt voltage control on the shared network at setup, corrupting the base snapshot.
            List<Pair<AcOuterLoop, AcOuterLoopContext>> pairs = AcloadFlowEngine.createOuterLoopsAndContexts(in.context(), false);
            List<AcOuterLoop> loops = pairs.stream().map(Pair::getLeft).toList();
            List<AcOuterLoopContext> ctxs = pairs.stream().map(Pair::getRight).toList();
            for (AcOuterLoopContext ctx : ctxs) {
                ctx.setLastSolverResult(new AcSolverResult(AcSolverStatus.CONVERGED, 1, Map.of()));
            }
            bridges[s] = new GpuHybridOuterLoopBridge(in.network(), in.es(), n, loops, ctxs);
        }

        // Device pi-model tracking for the transformer controller branches. cbIn bakes the 8 static doubles
        // [y g1 g2 b1 b2 ksi r1 a1] at extraction; OLF's PiModelArray returns R/X/Y/ksi from the CURRENT TAP,
        // so a tap change in the Java transformer-VC loop changes the branch admittance on the network while
        // the device would keep the extraction-time values (only the ratio follows, read from the state
        // vector). Capture the base row per controller branch, and after each scenario's outer-loop step
        // capture that scenario's row; any row that DIFFERS is pushed to the device as an r1Ov override (the
        // same per-(scenario, element) full-row channel the DEVICE orchestrator uses).
        List<LfBranch> tapTracked = in.transformerBranches().stream()
                .filter(b -> in.mapper().getElementIndex(b) >= 0)
                .toList();
        double[][] baseCbRow = new double[tapTracked.size()][];
        for (int b = 0; b < tapTracked.size(); b++) {
            baseCbRow[b] = cbRowOf(tapTracked.get(b));
        }
        double[][][] scenCbRow = new double[nS][][];         // [scenario][controller branch][8], null = base row

        // Shunt analogue of the pi-model tracking: a discrete section move (IncrementalShuntVoltageControl)
        // or dispatch (ShuntVoltageControl rounding) changes the controller shunt's (g, b). b follows via the
        // pinned SHUNT_B variable (capability-based, target re-read per round), but g is a PACKED static —
        // push per-scenario (dg, db) deltas through the shOv channel (db is harmless when SHUNT_B is a
        // variable: the fill overwrites b from the state after applying the override).
        List<com.powsybl.openloadflow.network.LfShunt> shuntTracked =
                in.network().<com.powsybl.openloadflow.network.LfShunt>getControllerElements(
                        com.powsybl.openloadflow.network.VoltageControl.Type.SHUNT).stream()
                .filter(sh -> in.mapper().getElementIndex(sh) >= 0)
                .toList();
        int shPackOffset = in.mapper().getClosedBranchCount() + in.mapper().getOpenBranchCount();
        double[][] baseShGb = new double[shuntTracked.size()][];
        for (int c = 0; c < shuntTracked.size(); c++) {
            baseShGb[c] = new double[] {shuntTracked.get(c).getG(), shuntTracked.get(c).getB()};
        }
        double[][][] scenShGb = new double[nS][][];

        // LIVE distribution-coefficient tracking (JAVA-hybrid group dynamics, 2026-09-07).
        // The device DISTR_Q / DISTR_RHO fills read STATIC pack-time coefficients, but OLF's loops
        // renormalize them per scenario: a reactive-limit pin zeroes the pinned member's reactive key
        // (updateReactiveKeys) and re-normalizes qPct over the survivors. After each scenario's
        // outer-loop step the engine re-reads the LIVE coefficients off the network and pushes any
        // that differ from the pack through the existing per-scenario dqOv / drOv channels. (The
        // DISTR_RHO side is a dormant safety net today: member OUTAGES — the one thing that reforms
        // the live 1/enabled-count coupling — are dropped to the CPU at qualification because the
        // element disable shrinks the es variable layout, and TVC clamps disable only the CONTROL,
        // which the live coefficient does not depend on.)
        int ndqContrib = in.gpuData().dqKMember().length;
        LfBus[] dqMemberBus = new LfBus[in.gpuData().dqMemberKey().length];
        if (ndqContrib > 0 && dqMemberBus.length > 0) {
            // rebuild the member table in the PACK's enumeration order (pilots in bus order,
            // merged controllers in order) — GpuAcOuterLoopPacks.distrQPack's exact loop
            int mi = 0;
            for (LfBus pilot : in.network().getBuses()) {
                var gvc = pilot.getGeneratorVoltageControl()
                        .filter(c -> c.getMergeStatus() == com.powsybl.openloadflow.network.VoltageControl.MergeStatus.MAIN)
                        .filter(c -> pilot.isGeneratorVoltageControlled() && !c.isLocalControl())
                        .orElse(null);
                if (gvc == null) {
                    continue;
                }
                for (LfBus m : gvc.getMergedControllerElements()) {
                    dqMemberBus[mi++] = m;
                }
            }
        }
        int ndrContrib = in.gpuData().drRow().length;
        LfBranch[] drKBranch = new LfBranch[ndrContrib];
        LfBranch[] drJBranch = new LfBranch[ndrContrib];
        if (ndrContrib > 0) {
            java.util.Map<Integer, LfBranch> branchByRhoCol = new java.util.HashMap<>();
            for (LfBranch br : in.network().getBranches()) {
                var rhoVar = in.es().getVariableSet().getVariable(br.getNum(), AcVariableType.BRANCH_RHO1);
                if (rhoVar != null && rhoVar.getRow() >= 0) {
                    branchByRhoCol.put(rhoVar.getRow(), br);
                }
            }
            for (int c = 0; c < ndrContrib; c++) {
                drKBranch[c] = branchByRhoCol.get(in.gpuData().drRow()[c]);
                drJBranch[c] = branchByRhoCol.get(in.gpuData().drCol()[c]);
            }
        }
        double[][] scenDqW = new double[nS][];               // per-scenario LIVE weights, null = pack values
        double[][] scenDrC = new double[nS][];

        // Packed AC-emulation HVDCs (the device hv family), in the packer's enumeration order. OLF's HVDC
        // outer loops (FreezingHvdcACEmulation / AcHvdcAcEmulationLimits) flip the emulation status per
        // scenario: any non-LINEAR status DEACTIVATES the hvdc P coupling terms on the CPU (the injections
        // then enter through the bus targets, which are re-read every round) — mirror it on the device by
        // setting the hv element's per-scenario DISABLED bit, which fillHvdcBatch honors.
        List<LfHvdc> hvPacked = in.network().getHvdcs().stream()
                .filter(h -> h.getBus1() != null && h.getBus2() != null && h.isAcEmulation())
                .toList();
        int hvOffset = in.nElements() - hvPacked.size();

        // per-scenario evolving state (isolated across scenarios via the compact snapshot + explicit targetP)
        HybridScenarioState baseCompact = new HybridScenarioState(in.ctrlBuses(), in.hvdcs(), in.transformerBranches());
        List<LfBranch> networkDisabled = new ArrayList<>();
        HybridScenarioState[] snaps = new HybridScenarioState[nS];
        // Load-based distributed slack (balanceType PROPORTIONAL_TO_[CONFORM_]LOAD) mutates LfLoad targetP — and
        // targetQ under loadPowerFactorConstant — across ARBITRARY buses, which the compact bus/branch snapshot
        // does not cover. Isolate them per scenario with an explicit flat array over ALL loads (parallel to the
        // generator array). Restoring to base each round is a harmless no-op for generation-only slack.
        List<LfLoad> loads = in.network().getBuses().stream().flatMap(b -> b.getLoads().stream()).toList();
        double[] baseLoadP = loads.stream().mapToDouble(LfLoad::getTargetP).toArray();
        double[] baseLoadQ = loads.stream().mapToDouble(LfLoad::getTargetQ).toArray();
        // Secondary voltage control mutates GENERATOR-VC TARGET VALUES (vc.setTargetValue) — not captured by
        // BusState — so isolate every MAIN generator voltage control's target per scenario, like gen/load targets.
        List<com.powsybl.openloadflow.network.GeneratorVoltageControl> genVcs = in.network().getBuses().stream()
                .flatMap(b -> b.getGeneratorVoltageControl().stream())
                .filter(vc -> vc.getMergeStatus() == com.powsybl.openloadflow.network.VoltageControl.MergeStatus.MAIN)
                .distinct().toList();
        double[] baseVcTarget = genVcs.stream().mapToDouble(com.powsybl.openloadflow.network.GeneratorVoltageControl::getTargetValue).toArray();
        // Dirty-tracking: a scenario's outer-loop step touches ~80 of the ~7100 elements the engine used
        // to save and restore wholesale, so each scenario records only what it moved and the swap between
        // two scenarios costs their dirty sets, not the network.
        // ATTACHED ONLY around the outer-loop step, never during restore: restoring fires the same
        // events (BusState.restore alone triggers the q-limit, disabling-status and abs-variable ones per
        // bus), and a muted listener still costs a dispatch per event. Detaching means nothing dispatches.
        DirtyTracker tracker = new DirtyTracker(in.ctrlBuses(), in.gens(), loads, genVcs);
        if (GpuBatchPhaseStats.ENABLED) {
            System.err.println("HYBRID_LISTENERS " + in.network().getListeners().stream()
                    .map(l -> l.getClass().getSimpleName()).toList());
        }
        // CUMULATIVE per scenario, not per round: a scenario that moves bus B in one round and bus C in
        // the next still has B at its own value on the network, so its record must keep B too — otherwise
        // the next restore reverts B to base and the scenario silently loses it.
        java.util.BitSet[] busBits = new java.util.BitSet[nS];
        java.util.BitSet[] genBits = new java.util.BitSet[nS];
        java.util.BitSet[] loadBits = new java.util.BitSet[nS];
        java.util.BitSet[] vcBits = new java.util.BitSet[nS];
        int[][] dirtyBuses = new int[nS][];
        int[][] dirtyGens = new int[nS][];
        double[][] dirtyGenP = new double[nS][];
        int[][] dirtyLoads = new int[nS][];
        double[][] dirtyLoadP = new double[nS][];
        double[][] dirtyLoadQ = new double[nS][];
        int[][] dirtyVcs = new int[nS][];
        double[][] dirtyVcTarget = new double[nS][];
        int[] curBuses = EMPTY_INT;
        int[] curGens = EMPTY_INT;
        int[] curLoads = EMPTY_INT;
        int[] curVcs = EMPTY_INT;
        int[][] rowMode = new int[nS][];
        // live-row -> base-row map per scenario (null = identity): a member outage that removes a
        // VARIABLE from the live es shrinks the layout; the outage set is fixed per scenario, and
        // outer loops only ever change ACTIVATION (never the element-disable set), so the map is
        // computed once per scenario after its outage is applied.
        java.util.Map<Long, Integer> baseRowByVar = new java.util.HashMap<>();
        for (var v : in.es().getIndex().getSortedVariablesToFind()) {
            baseRowByVar.put(varKey(v.getType().ordinal(), v.getElementNum()), v.getRow());
        }
        int[][] liveToBase = new int[nS][];
        boolean[][] disabled = new boolean[nS][in.nElements()];
        double[][] scenTargets = new double[nS][];
        boolean[] retired = new boolean[nS];
        boolean[] routedByBudget = new boolean[nS];
        int[] scenPasses = new int[nS];
        int[] consecReject = new int[nS];
        for (int s = 0; s < nS; s++) {
            rowMode[s] = in.baseRowMode().clone();
            scenTargets[s] = in.baseTargets().clone();
            for (LfBranch b : in.ctgs().get(s).outagedBranches()) {
                disabled[s][in.mapper().getElementIndex(b)] = true;
            }
        }

        double[][] states = new double[nS][];
        double[][] round0States = null;
        if (DIRTY_PROBE != null) {
            DIRTY_PROBE.attachTo(in.network());
        }
        int ctxSize = nS;
        long handle = createHandle(in.structure(), nS, in.mode(), in.gpuData(), in.rowCriteria());

        int rounds = 0;
        long tSolve = 0;
        long tJava = 0;
        long tSnap = 0;
        long revertedBuses = 0;
        long appliedBuses = 0;
        long visits = 0;
        long tRevert = 0;
        long tApply = 0;
        long tSnapElem = 0;                                  // HybridScenarioState (OLF BusState/HvdcState/BranchState objects)
        long tSnapArr = 0;                                   // the four target arrays (built with streams)
        long nJavaCalls = 0;

        // ---- OLF-faithful pre-round-0 initialization: OLF initialize()s every outer loop BEFORE the
        // initial solver run, and a loop's initialize can CHANGE the activation (SimpleTransformerVoltageControl
        // / ShuntVoltageControl enable continuous control there). Round 0 must solve under the POST-initialize
        // activation and the PreviousValues x0 read from the post-initialize network — an extra pre-initialize
        // solve is a round OLF never performs and shifts the trajectory (~1e-7 measured on Simple TVC). ----
        for (int s = 0; s < nS; s++) {
            long tj0 = System.nanoTime();
            for (LfBranch b : networkDisabled) {
                b.setDisabled(false);
            }
            // revert only what the previous scenario moved (tracker not attached: nothing dispatches)
            for (int i : curBuses) {
                baseCompact.restoreBus(i);
            }
            baseCompact.restoreHvdcsAndBranches();
            for (int i : curGens) {
                in.gens().get(i).setTargetP(in.baseGenP()[i]);
            }
            for (int i : curLoads) {
                loads.get(i).setTargetP(baseLoadP[i]);
                loads.get(i).setTargetQ(baseLoadQ[i]);
            }
            for (int i : curVcs) {
                genVcs.get(i).setTargetValue(baseVcTarget[i]);
            }
            curBuses = EMPTY_INT;
            curGens = EMPTY_INT;
            curLoads = EMPTY_INT;
            curVcs = EMPTY_INT;
            // initialTargetP is the distributed-slack anchor and is NOT tracked (no listener, and
            // BusDcState restores it for a dirty bus) — re-anchor it for every generator, as before.
            for (int g = 0; g < in.gens().size(); g++) {
                in.gens().get(g).setInitialTargetP(in.originalGenP()[g]);
            }
            for (LfBranch b : in.ctgs().get(s).outagedBranches()) {
                b.setDisabled(true);
            }
            networkDisabled = in.ctgs().get(s).outagedBranches();
            liveToBase[s] = liveToBaseMap(in.es(), baseRowByVar, n);
            tracker.start();
            in.network().addListener(tracker);
            GpuAcActivation act = bridges[s].initializeScenario(compactState(in.baseState(), liveToBase[s]));
            in.network().removeListener(tracker);
            tracker.stop();
            rowMode[s] = expandModes(act.rowMode(), liveToBase[s], n);
            scenTargets[s] = expandTargets(act.applyTargets(new AcTargetVector(in.network(), in.es()).getArray()),
                    liveToBase[s], in.baseState(), n);
            com.powsybl.openloadflow.ac.solver.AcSolverUtil.initStateVector(in.network(), in.es(),
                    new com.powsybl.openloadflow.network.util.PreviousValueVoltageInitializer());
            double[] x0 = in.es().getStateVector().get();
            states[s] = new double[n + 3];
            expandStateInto(x0, liveToBase[s], in.baseState(), states[s], n);
            if (!tapTracked.isEmpty()) {                     // initialize may not move taps, but stay uniform
                double[][] rr = new double[tapTracked.size()][];
                for (int b = 0; b < tapTracked.size(); b++) {
                    rr[b] = cbRowOf(tapTracked.get(b));
                }
                scenCbRow[s] = rr;
            }
            if (!shuntTracked.isEmpty()) {
                double[][] gb = new double[shuntTracked.size()][];
                for (int c = 0; c < shuntTracked.size(); c++) {
                    gb[c] = new double[] {shuntTracked.get(c).getG(), shuntTracked.get(c).getB()};
                }
                scenShGb[s] = gb;
            }
            for (int h = 0; h < hvPacked.size(); h++) {
                disabled[s][hvOffset + h] = hvPacked.get(h).getAcEmulationControl().getAcEmulationStatus()
                        != LfHvdc.AcEmulationControl.AcEmulationStatus.LINEAR_MODE;
            }
            scenDqW[s] = liveDqWeights(in.gpuData(), dqMemberBus);
            scenDrC[s] = liveDrCoefs(in.gpuData(), drKBranch, drJBranch);
            busBits[s] = tracker.accumulateBuses(busBits[s]);
            genBits[s] = tracker.accumulateGens(genBits[s]);
            loadBits[s] = tracker.accumulateLoads(loadBits[s]);
            vcBits[s] = tracker.accumulateVcs(vcBits[s]);
            dirtyBuses[s] = busBits[s].stream().toArray();
            snaps[s] = new HybridScenarioState(subList(in.ctrlBuses(), dirtyBuses[s]),
                    in.hvdcs(), in.transformerBranches());
            dirtyGens[s] = genBits[s].stream().toArray();
            dirtyGenP[s] = valuesOf(dirtyGens[s], i -> in.gens().get(i).getTargetP());
            dirtyLoads[s] = loadBits[s].stream().toArray();
            dirtyLoadP[s] = valuesOf(dirtyLoads[s], i -> loads.get(i).getTargetP());
            dirtyLoadQ[s] = valuesOf(dirtyLoads[s], i -> loads.get(i).getTargetQ());
            dirtyVcs[s] = vcBits[s].stream().toArray();
            dirtyVcTarget[s] = valuesOf(dirtyVcs[s], i -> genVcs.get(i).getTargetValue());
            curBuses = dirtyBuses[s];
            curGens = dirtyGens[s];
            curLoads = dirtyLoads[s];
            curVcs = dirtyVcs[s];
            tJava += System.nanoTime() - tj0;
            nJavaCalls++;
        }

        for (int round = 0; round < MAX_ROUNDS; round++) {
            rounds = round + 1;
            // cross-round compaction: solve only still-active scenarios; recreate the (cheap, empty-pack) context
            // when the active count shrinks.
            List<Integer> active = new ArrayList<>();
            for (int s = 0; s < nS; s++) {
                if (!retired[s]) {
                    active.add(s);
                }
            }
            if (active.isEmpty()) {
                break;
            }
            int nA = active.size();
            if (nA != ctxSize) {
                GpuAcNewtonSolver.destroyBatchContext(handle);
                handle = createHandle(in.structure(), nA, in.mode(), in.gpuData(), in.rowCriteria());
                ctxSize = nA;
            }
            // compacted per-scenario inputs + sparse target overrides, indexed by the COMPACTED scenario index i.
            int[][] rowModeC = new int[nA][];
            boolean[][] disabledC = new boolean[nA][];
            List<Integer> ovIdx = new ArrayList<>();
            List<Double> ovVal = new ArrayList<>();
            List<Integer> r1OvSlotElem = new ArrayList<>();  // per-scenario refreshed pi-model rows (tap changes)
            List<Double> r1OvVal = new ArrayList<>();
            List<Integer> shOvSlotElem = new ArrayList<>();  // per-scenario controller-shunt (dg, db) refresh
            List<Double> shOvGb = new ArrayList<>();
            double[] x0C = new double[nA * n];               // round 0 included: the pre-pass filled states[s]
            for (int i = 0; i < nA; i++) {
                int s = active.get(i);
                rowModeC[i] = rowMode[s];
                disabledC[i] = disabled[s];
                for (int r = 0; r < n; r++) {
                    if (scenTargets[s][r] != in.baseTargets()[r]) {
                        ovIdx.add(i * n + r);
                        ovVal.add(scenTargets[s][r]);
                    }
                }
                // refresh any controller-branch pi-model row this scenario's outer loops moved off the base
                if (scenCbRow[s] != null) {
                    for (int b = 0; b < tapTracked.size(); b++) {
                        if (scenCbRow[s][b] != null && !java.util.Arrays.equals(scenCbRow[s][b], baseCbRow[b])) {
                            r1OvSlotElem.add(i);
                            r1OvSlotElem.add(in.mapper().getElementIndex(tapTracked.get(b)));
                            for (double v : scenCbRow[s][b]) {
                                r1OvVal.add(v);
                            }
                        }
                    }
                }
                // refresh any controller-shunt (g, b) this scenario's outer loops moved off the base
                if (scenShGb[s] != null) {
                    for (int c = 0; c < shuntTracked.size(); c++) {
                        if (scenShGb[s][c] != null && !java.util.Arrays.equals(scenShGb[s][c], baseShGb[c])) {
                            shOvSlotElem.add(i);
                            shOvSlotElem.add(in.mapper().getElementIndex(shuntTracked.get(c)) - shPackOffset);
                            shOvGb.add(scenShGb[s][c][0] - baseShGb[c][0]);
                            shOvGb.add(scenShGb[s][c][1] - baseShGb[c][1]);
                        }
                    }
                }
                System.arraycopy(states[s], 0, x0C, i * n, n);
            }
            // LIVE distribution coefficients that moved off the pack this round (pins / member outages):
            // push them through the per-scenario dqOv / drOv channels, or CLEAR a previous round's set.
            List<Integer> dqOvIdx = new ArrayList<>();
            List<Double> dqOvW = new ArrayList<>();
            List<Integer> drOvIdx = new ArrayList<>();
            List<Double> drOvC = new ArrayList<>();
            for (int i = 0; i < nA; i++) {
                int s = active.get(i);
                if (scenDqW[s] != null) {
                    for (int c = 0; c < scenDqW[s].length; c++) {
                        if (scenDqW[s][c] != in.gpuData().dqWeight()[c]) {
                            dqOvIdx.add(i);
                            dqOvIdx.add(c);
                            dqOvW.add(scenDqW[s][c]);
                        }
                    }
                }
                if (scenDrC[s] != null) {
                    for (int c = 0; c < scenDrC[s].length; c++) {
                        if (scenDrC[s][c] != in.gpuData().drCoef()[c]) {
                            drOvIdx.add(i);
                            drOvIdx.add(c);
                            drOvC.add(scenDrC[s][c]);
                        }
                    }
                }
            }
            // per-scenario ACTIVE row count: the CPU's L2 stopping criterion is sqrt(eps^2 * nLive),
            // and a shrunk live layout has fewer rows than the device's frozen superset (A5 lesson:
            // a one-row threshold difference is a measurable trajectory shift at 1e-12).
            int[] nActiveC = null;
            for (int i = 0; i < nA; i++) {
                if (liveToBase[active.get(i)] != null) {
                    if (nActiveC == null) {
                        nActiveC = new int[nA];
                        java.util.Arrays.fill(nActiveC, n);
                    }
                    nActiveC[i] = liveToBase[active.get(i)].length;
                }
            }
            GpuAcNewtonSolver.setBatchActiveRowCounts(handle, nActiveC);
            GpuAcNewtonSolver.setBatchDqWeightOverrides(handle,
                    dqOvIdx.isEmpty() ? null : dqOvIdx.stream().mapToInt(Integer::intValue).toArray(),
                    dqOvW.isEmpty() ? null : dqOvW.stream().mapToDouble(Double::doubleValue).toArray());
            GpuAcNewtonSolver.setBatchDrCoefOverrides(handle,
                    drOvIdx.isEmpty() ? null : drOvIdx.stream().mapToInt(Integer::intValue).toArray(),
                    drOvC.isEmpty() ? null : drOvC.stream().mapToDouble(Double::doubleValue).toArray());
            // The native side resolves every per-scenario row as a JNI handle without a null check, so a
            // null row is a SIGSEGV inside solveBatchContext with no Java frame to blame (observed twice,
            // deterministically, ~chunk 7 of a chunked 9042 run). Name it here instead.
            checkScenarioRows(rowModeC, disabledC, nA);
            long ts0 = System.nanoTime();
            double[][] statesC = GpuAcNewtonSolver.solveBatchContext(handle, in.baseState(), in.baseTargets(),
                    rowModeC, in.baseRowMode(), disabledC,
                    ovIdx.stream().mapToInt(Integer::intValue).toArray(), ovVal.stream().mapToDouble(Double::doubleValue).toArray(),
                    new int[0], new double[0], new double[0], new double[0], new double[0], null, null, null, null,
                    null, null,
                    null, null,
                    // the RUN's inner-solve stopping criteria — convEpsPerEq drives the norm-2 stop, matching the
                    // CPU (RL/DS tols are moot here: the packs are empty and OLF's Java loops do DS/RL).
                    in.maxIterations(), in.tol(), in.reactiveLimitTol(), in.slackDistTol(), in.convEpsPerEq(),
                    in.lsMode(), in.lsMaxDv(), in.lsMaxDphi(), in.varType(), x0C,
                    shOvSlotElem.stream().mapToInt(Integer::intValue).toArray(),
                    shOvGb.stream().mapToDouble(Double::doubleValue).toArray(),
                    null, null,                                          // rlStateIn/Out
                    r1OvSlotElem.stream().mapToInt(Integer::intValue).toArray(),   // per-scenario pi-model refresh
                    r1OvVal.stream().mapToDouble(Double::doubleValue).toArray(),
                    30, 3, 0, 0, 0, null, null);                         // maxOuterIterations (moot: DS pack empty), maxPqPvSwitch, ...
            tSolve += System.nanoTime() - ts0;
            for (int i = 0; i < nA; i++) {
                states[active.get(i)] = statesC[i];
            }
            if (round == 0) {
                round0States = states.clone();
            }

            boolean allStable = true;
            for (int s = 0; s < nS; s++) {
                if (retired[s]) {
                    continue;
                }
                long tr0 = System.nanoTime();
                // incremental branch management: re-enable the previous scenario's outage, restore this scenario's
                // compact controller-bus / HVDC state (+ its accumulated distributed targetP).
                for (LfBranch b : networkDisabled) {
                    b.setDisabled(false);
                }
                // Revert exactly what the PREVIOUS scenario moved, then apply this one's dirty set.
                // The tracker is not attached here, so none of the restore's own events dispatch.
                long trev0 = System.nanoTime();
                // Revert only the DIFFERENCE. An element dirty in BOTH the previous scenario and this one
                // is about to be overwritten by the apply below, so resetting it to base first is pure
                // waste — and the sets overlap heavily, because both are "what this chunk has touched".
                java.util.BitSet keepBuses = busBits[s];
                java.util.BitSet keepGens = genBits[s];
                java.util.BitSet keepLoads = loadBits[s];
                java.util.BitSet keepVcs = vcBits[s];
                visits++;
                appliedBuses += dirtyBuses[s] == null ? 0 : dirtyBuses[s].length;
                for (int i : curBuses) {
                    if (keepBuses == null || !keepBuses.get(i)) {
                        revertedBuses++;
                        baseCompact.restoreBus(i);
                    }
                }
                for (int i : curGens) {
                    if (keepGens == null || !keepGens.get(i)) {
                        in.gens().get(i).setTargetP(in.baseGenP()[i]);
                    }
                }
                for (int i : curLoads) {
                    if (keepLoads == null || !keepLoads.get(i)) {
                        loads.get(i).setTargetP(baseLoadP[i]);
                        loads.get(i).setTargetQ(baseLoadQ[i]);
                    }
                }
                for (int i : curVcs) {
                    if (keepVcs == null || !keepVcs.get(i)) {
                        genVcs.get(i).setTargetValue(baseVcTarget[i]);
                    }
                }
                long tapp0 = System.nanoTime();
                tRevert += tapp0 - trev0;
                if (snaps[s] != null) {
                    snaps[s].restore();                      // this scenario's dirty buses + HVDCs + branches
                    for (int k = 0; k < dirtyGens[s].length; k++) {
                        in.gens().get(dirtyGens[s][k]).setTargetP(dirtyGenP[s][k]);
                    }
                    for (int k = 0; k < dirtyLoads[s].length; k++) {
                        loads.get(dirtyLoads[s][k]).setTargetP(dirtyLoadP[s][k]);
                        loads.get(dirtyLoads[s][k]).setTargetQ(dirtyLoadQ[s][k]);
                    }
                    for (int k = 0; k < dirtyVcs[s].length; k++) {
                        genVcs.get(dirtyVcs[s][k]).setTargetValue(dirtyVcTarget[s][k]);
                    }
                    curBuses = dirtyBuses[s];
                    curGens = dirtyGens[s];
                    curLoads = dirtyLoads[s];
                    curVcs = dirtyVcs[s];
                    tApply += System.nanoTime() - tapp0;
                } else {
                    baseCompact.restoreHvdcsAndBranches();   // buses/gens/loads are already back at base
                    curBuses = EMPTY_INT;
                    curGens = EMPTY_INT;
                    curLoads = EMPTY_INT;
                    curVcs = EMPTY_INT;
                    tApply += System.nanoTime() - tapp0;
                }
                // anchor initialTargetP at the ORIGINAL generation for the distributed-slack cumulative accounting.
                for (int g = 0; g < in.gens().size(); g++) {
                    in.gens().get(g).setInitialTargetP(in.originalGenP()[g]);
                }
                for (LfBranch b : in.ctgs().get(s).outagedBranches()) {
                    b.setDisabled(true);
                }
                networkDisabled = in.ctgs().get(s).outagedBranches();
                tSnap += System.nanoTime() - tr0;

                double[] st = new double[n];
                System.arraycopy(states[s], 0, st, 0, n);
                long tj0 = System.nanoTime();
                if (DIRTY_PROBE != null) {
                    DIRTY_PROBE.startScenario();
                }
                // load state -> OLF outer loops -> re-extract (state compacted to the live layout)
                tracker.start();
                in.network().addListener(tracker);
                GpuAcActivation act = bridges[s].applyOuterLoopStep(compactState(st, liveToBase[s]));
                in.network().removeListener(tracker);
                tracker.stop();
                if (DIRTY_PROBE != null) {
                    DIRTY_PROBE.endScenario(in.ctrlBuses().size());
                }
                // capture this scenario's controller-branch pi-model rows: the outer loops may have moved a tap,
                // which changes R/X/Y/ksi on the network (PiModelArray reads them from the CURRENT tap) — the
                // device row must follow, or the next round solves with a stale admittance.
                if (!tapTracked.isEmpty()) {
                    double[][] rows = new double[tapTracked.size()][];
                    for (int b = 0; b < tapTracked.size(); b++) {
                        rows[b] = cbRowOf(tapTracked.get(b));
                    }
                    scenCbRow[s] = rows;
                }
                if (!shuntTracked.isEmpty()) {
                    double[][] gb = new double[shuntTracked.size()][];
                    for (int c = 0; c < shuntTracked.size(); c++) {
                        gb[c] = new double[] {shuntTracked.get(c).getG(), shuntTracked.get(c).getB()};
                    }
                    scenShGb[s] = gb;
                }
                for (int h = 0; h < hvPacked.size(); h++) {
                    disabled[s][hvOffset + h] = hvPacked.get(h).getAcEmulationControl().getAcEmulationStatus()
                            != LfHvdc.AcEmulationControl.AcEmulationStatus.LINEAR_MODE;
                }
                scenDqW[s] = liveDqWeights(in.gpuData(), dqMemberBus);
                scenDrC[s] = liveDrCoefs(in.gpuData(), drKBranch, drJBranch);
                rowMode[s] = expandModes(act.rowMode(), liveToBase[s], n);
                scenTargets[s] = expandTargets(act.applyTargets(new AcTargetVector(in.network(), in.es()).getArray()),
                        liveToBase[s], in.baseState(), n);
                // OLF-faithful re-solve initialization — the EXACT call OLF's runOuterLoop makes before every
                // re-solve: solver.run(new PreviousValueVoltageInitializer()) -> AcSolverUtil.initStateVector
                // re-reads EVERY variable from the network (v/phi from the bus, rho/a1 from the pi-model — so a
                // just-rounded ratio starts the next Newton EXACTLY at its pinned value — b from the shunt, and
                // DUMMY_P/DUMMY_Q reset to 0). Warm-starting from the raw previous device state instead leaves
                // the continuous->rounded jump to be absorbed inside the next Newton iteration, whose
                // second-order remainder passes the stopping criterion ~1e-2 off the CPU; and any partial
                // approximation of this call (e.g. only snapping identity rows to their targets) leaves a
                // ~1e-6 residual vs the CPU where plain solves agree at ~1e-13. The bridge left the network
                // holding this scenario's state (updateNetwork) + the outer loops' mutations, so the state
                // vector after this call IS the CPU's x0, on the stable variable rows.
                com.powsybl.openloadflow.ac.solver.AcSolverUtil.initStateVector(in.network(), in.es(),
                        new com.powsybl.openloadflow.network.util.PreviousValueVoltageInitializer());
                double[] cpuX0 = in.es().getStateVector().get();
                expandStateInto(cpuX0, liveToBase[s], in.baseState(), states[s], n);
                tJava += System.nanoTime() - tj0;
                nJavaCalls++;

                // straggler early-routing on persistent device rejection.
                consecReject[s] = ((int) states[s][n + 2]) == -2 ? consecReject[s] + 1 : 0;
                boolean willRetire;
                if (bridges[s].isStable()) {
                    retired[s] = true;
                    willRetire = true;
                } else if (consecReject[s] >= in.rejectRouteThreshold() || ++scenPasses[s] >= in.maxOuter()) {
                    retired[s] = true;
                    routedByBudget[s] = true;
                    willRetire = true;
                } else {
                    allStable = false;
                    willRetire = false;
                }
                // snapshot save-skip: only persist a scenario that continues to the next round.
                if (!willRetire) {
                    long tsv0 = System.nanoTime();
                    busBits[s] = tracker.accumulateBuses(busBits[s]);
                    genBits[s] = tracker.accumulateGens(genBits[s]);
                    loadBits[s] = tracker.accumulateLoads(loadBits[s]);
                    vcBits[s] = tracker.accumulateVcs(vcBits[s]);
                    dirtyBuses[s] = busBits[s].stream().toArray();
                    snaps[s] = new HybridScenarioState(subList(in.ctrlBuses(), dirtyBuses[s]),
                            in.hvdcs(), in.transformerBranches());
                    long tsv1 = System.nanoTime();
                    tSnapElem += tsv1 - tsv0;
                    dirtyGens[s] = genBits[s].stream().toArray();
                    dirtyGenP[s] = valuesOf(dirtyGens[s], i -> in.gens().get(i).getTargetP());
                    dirtyLoads[s] = loadBits[s].stream().toArray();
                    dirtyLoadP[s] = valuesOf(dirtyLoads[s], i -> loads.get(i).getTargetP());
                    dirtyLoadQ[s] = valuesOf(dirtyLoads[s], i -> loads.get(i).getTargetQ());
                    dirtyVcs[s] = vcBits[s].stream().toArray();
                    dirtyVcTarget[s] = valuesOf(dirtyVcs[s], i -> genVcs.get(i).getTargetValue());
                    curBuses = dirtyBuses[s];
                    curGens = dirtyGens[s];
                    curLoads = dirtyLoads[s];
                    curVcs = dirtyVcs[s];
                    tSnapArr += System.nanoTime() - tsv1;
                    tSnap += System.nanoTime() - tsv0;
                }
            }
            if (allStable) {
                break;
            }
        }
        GpuAcNewtonSolver.destroyBatchContext(handle);
        if (GpuBatchPhaseStats.ENABLED) {
            long cumBuses = 0;
            long cumOther = 0;
            int counted = 0;
            for (int s2 = 0; s2 < nS; s2++) {
                if (busBits[s2] != null) {
                    counted++;
                    cumBuses += busBits[s2].cardinality();
                    cumOther += genBits[s2].cardinality() + loadBits[s2].cardinality() + vcBits[s2].cardinality();
                }
            }
            System.err.printf("HYBRID_DIRTY_CUM scenarios=%d avgDirtyBuses=%.1f of %d avgDirtyOther=%.1f%n",
                    counted, counted == 0 ? 0 : cumBuses / (double) counted, in.ctrlBuses().size(),
                    counted == 0 ? 0 : cumOther / (double) counted);
            if (System.getenv("OLF_BUSSTATE_PROFILE") != null) {
                System.err.println("BUSSTATE_PROFILE " + com.powsybl.openloadflow.network.BusState.profile());
            }
            System.err.printf("HYBRID_RESTORE_SPLIT scenarios=%d revert=%d ms apply=%d ms | perVisit: "
                    + "reverted=%.1f applied=%.1f buses of %d%n",
                    nS, tRevert / 1_000_000, tApply / 1_000_000,
                    visits == 0 ? 0 : revertedBuses / (double) visits,
                    visits == 0 ? 0 : appliedBuses / (double) visits, in.ctrlBuses().size());
            System.err.printf("HYBRID_SNAP_SPLIT scenarios=%d elementState=%d ms targetArrays=%d ms "
                    + "| perSnapshot: ctrlBuses=%d hvdcs=%d transformers=%d gens=%d loads=%d genVcs=%d%n",
                    nS, tSnapElem / 1_000_000, tSnapArr / 1_000_000,
                    in.ctrlBuses().size(), in.hvdcs().size(), in.transformerBranches().size(),
                    in.gens().size(), loads.size(), genVcs.size());
        }

        // Hand the network back the way we found it. Every scenario iteration re-enables the PREVIOUS
        // scenario's outage at its START, so on exit the LAST scenario's branches are still disabled and
        // its outer-loop mutations are still applied. That was invisible while a run was one chunk; with
        // chunking the next chunk captures ITS baseCompact from this corrupted network and never
        // re-enables those branches, and a bus that loses its angle variable while keeping an active
        // BUS_TARGET_P then blows up in extractActivation (index -1) or, worse, reaches the device as a
        // bad row and segfaults inside the JNI call.
        for (LfBranch b : networkDisabled) {
            b.setDisabled(false);
        }
        baseCompact.restore();
        for (int g = 0; g < in.gens().size(); g++) {
            in.gens().get(g).setTargetP(in.baseGenP()[g]);
            in.gens().get(g).setInitialTargetP(in.originalGenP()[g]);
        }
        for (int l = 0; l < loads.size(); l++) {
            loads.get(l).setTargetP(baseLoadP[l]);
            loads.get(l).setTargetQ(baseLoadQ[l]);
        }
        for (int v = 0; v < genVcs.size(); v++) {
            genVcs.get(v).setTargetValue(baseVcTarget[v]);
        }

        // what each scenario's active-power distribution loop placed, the load flow's own part of what OLF reports
        double[] distributed = new double[nS];
        for (int s = 0; s < nS; s++) {
            distributed[s] = bridges[s].distributedActivePower();
        }
        return new Result(states, round0States, rowMode, scenTargets, retired, routedByBudget, bridges,
                rounds, tSolve, tJava, tSnap, nJavaCalls, distributed);
    }
}
