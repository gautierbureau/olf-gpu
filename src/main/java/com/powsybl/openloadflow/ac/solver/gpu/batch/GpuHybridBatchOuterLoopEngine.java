/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

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

    private static final int MAX_ROUNDS = 40;

    private GpuHybridBatchOuterLoopEngine() {
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
                         int maxIterations, int maxOuter, int rejectRouteThreshold) {
    }

    /**
     * Per-scenario outcome + diagnostics. {@code gpuConverged[s]} false means the scenario was routed to the CPU
     * fallback. The arrays are indexed by the scenario order in {@link Inputs#ctgs}.
     *
     * @param states          [nS][n+3] final device state per scenario: x(n), iterations, ||F||inf, status
     * @param round0States    [nS][] the round-0 device state (base targets, pure inner)
     * @param rowMode         [nS][] final per-scenario device row mode
     * @param scenTargets     [nS][] final per-scenario target vector
     * @param retired         [nS] scenario reached a terminal state (converged OR routed) within the round budget
     * @param routedByBudget  [nS] scenario was routed to the CPU fallback (straggler / budget / device reject)
     * @param bridges         [nS] the per-scenario outer-loop bridges (loop names / unstable counts, diagnostics)
     */
    public record Result(double[][] states, double[][] round0States, int[][] rowMode, double[][] scenTargets,
                         boolean[] retired, boolean[] routedByBudget, GpuHybridOuterLoopBridge[] bridges,
                         int rounds, long solveNanos, long javaNanos, long snapshotNanos, long javaCalls) {

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

    private static long createHandle(long structure, int count, int mode, GpuAcData d) {
        return GpuAcNewtonSolver.createBatchContext(structure, count, mode,
                d.rlVrow(), d.rlMinQ(), d.rlMaxQ(), d.rlLoadQ(), d.rlVtarget(),
                d.dsPhiRow(), d.dsFactor(), d.dsSlackPhiRow(), d.dsSlackTargetP(),
                d.dsBaseTargetP(), d.dsMaxDelta(), d.dsMinDelta());
    }

    public static Result solve(Inputs in) {
        int nS = in.ctgs().size();
        int n = in.n();

        // OLF's outer loops, one context PER (scenario, loop) via the core factory — no reflection, and
        // filtered/initialized exactly as AcloadFlowEngine.run() does, so per-scenario contexts match core.
        GpuHybridOuterLoopBridge[] bridges = new GpuHybridOuterLoopBridge[nS];
        for (int s = 0; s < nS; s++) {
            List<Pair<AcOuterLoop, AcOuterLoopContext>> pairs = AcloadFlowEngine.createOuterLoopsAndContexts(in.context());
            List<AcOuterLoop> loops = pairs.stream().map(Pair::getLeft).toList();
            List<AcOuterLoopContext> ctxs = pairs.stream().map(Pair::getRight).toList();
            for (AcOuterLoopContext ctx : ctxs) {
                ctx.setLastSolverResult(new AcSolverResult(AcSolverStatus.CONVERGED, 1, Map.of()));
            }
            bridges[s] = new GpuHybridOuterLoopBridge(in.network(), in.es(), n, loops, ctxs);
        }

        // per-scenario evolving state (isolated across scenarios via the compact snapshot + explicit targetP)
        HybridScenarioState baseCompact = new HybridScenarioState(in.ctrlBuses(), in.hvdcs(), in.transformerBranches());
        List<LfBranch> networkDisabled = new ArrayList<>();
        HybridScenarioState[] snaps = new HybridScenarioState[nS];
        double[][] genSnap = new double[nS][];
        // Load-based distributed slack (balanceType PROPORTIONAL_TO_[CONFORM_]LOAD) mutates LfLoad targetP — and
        // targetQ under loadPowerFactorConstant — across ARBITRARY buses, which the compact bus/branch snapshot
        // does not cover. Isolate them per scenario with an explicit flat array over ALL loads (parallel to the
        // generator array). Restoring to base each round is a harmless no-op for generation-only slack.
        List<LfLoad> loads = in.network().getBuses().stream().flatMap(b -> b.getLoads().stream()).toList();
        double[] baseLoadP = loads.stream().mapToDouble(LfLoad::getTargetP).toArray();
        double[] baseLoadQ = loads.stream().mapToDouble(LfLoad::getTargetQ).toArray();
        double[][] loadSnapP = new double[nS][];
        double[][] loadSnapQ = new double[nS][];
        int[][] rowMode = new int[nS][];
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
        int ctxSize = nS;
        long handle = createHandle(in.structure(), nS, in.mode(), in.gpuData());

        int rounds = 0;
        long tSolve = 0;
        long tJava = 0;
        long tSnap = 0;
        long nJavaCalls = 0;
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
                handle = createHandle(in.structure(), nA, in.mode(), in.gpuData());
                ctxSize = nA;
            }
            // compacted per-scenario inputs + sparse target overrides, indexed by the COMPACTED scenario index i.
            int[][] rowModeC = new int[nA][];
            boolean[][] disabledC = new boolean[nA][];
            List<Integer> ovIdx = new ArrayList<>();
            List<Double> ovVal = new ArrayList<>();
            double[] x0C = round == 0 ? null : new double[nA * n];
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
                if (round > 0) {
                    System.arraycopy(states[s], 0, x0C, i * n, n);
                }
            }
            long ts0 = System.nanoTime();
            double[][] statesC = GpuAcNewtonSolver.solveBatchContext(handle, in.baseState(), in.baseTargets(),
                    rowModeC, in.baseRowMode(), disabledC,
                    ovIdx.stream().mapToInt(Integer::intValue).toArray(), ovVal.stream().mapToDouble(Double::doubleValue).toArray(),
                    new int[0], new double[0], new double[0], new double[0], new double[0],
                    in.maxIterations(), 1e-6, 1e-4, 1e-2, 1e-4,          // tol, reactiveLimitTol(unused), slackDistTol, convEpsPerEq
                    in.lsMode(), in.lsMaxDv(), in.lsMaxDphi(), in.varType(), x0C,
                    null, null, null, null, null, null,                  // shuntOv, rlStateIn/Out, r1Ov
                    30, 3, 0, 0, null, null);                            // maxOuterIterations (moot: DS pack empty), maxPqPvSwitch, ...
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
                if (snaps[s] != null) {
                    snaps[s].restore();
                    for (int g = 0; g < in.gens().size(); g++) {
                        in.gens().get(g).setTargetP(genSnap[s][g]);
                    }
                    for (int l = 0; l < loads.size(); l++) {
                        loads.get(l).setTargetP(loadSnapP[s][l]);
                        loads.get(l).setTargetQ(loadSnapQ[s][l]);
                    }
                } else {
                    baseCompact.restore();
                    for (int g = 0; g < in.gens().size(); g++) {
                        in.gens().get(g).setTargetP(in.baseGenP()[g]);
                    }
                    for (int l = 0; l < loads.size(); l++) {
                        loads.get(l).setTargetP(baseLoadP[l]);
                        loads.get(l).setTargetQ(baseLoadQ[l]);
                    }
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
                GpuAcActivation act = bridges[s].applyOuterLoopStep(st);  // load state -> OLF outer loops -> re-extract
                rowMode[s] = act.rowMode();
                scenTargets[s] = act.applyTargets(new AcTargetVector(in.network(), in.es()).getArray());
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
                    snaps[s] = new HybridScenarioState(in.ctrlBuses(), in.hvdcs(), in.transformerBranches());
                    genSnap[s] = in.gens().stream().mapToDouble(LfGenerator::getTargetP).toArray();
                    loadSnapP[s] = loads.stream().mapToDouble(LfLoad::getTargetP).toArray();
                    loadSnapQ[s] = loads.stream().mapToDouble(LfLoad::getTargetQ).toArray();
                    tSnap += System.nanoTime() - tsv0;
                }
            }
            if (allStable) {
                break;
            }
        }
        GpuAcNewtonSolver.destroyBatchContext(handle);

        return new Result(states, round0States, rowMode, scenTargets, retired, routedByBudget, bridges,
                rounds, tSolve, tJava, tSnap, nJavaCalls);
    }
}
