/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.commons.report.ReportNode;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.json.JsonLoadFlowParameters;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcLoadFlowContext;
import com.powsybl.openloadflow.ac.AcLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcTargetVector;
import com.powsybl.openloadflow.ac.AcloadFlowEngine;
import com.powsybl.openloadflow.ac.equations.AcEquationSystemCreator;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuTestPaths;
import com.powsybl.openloadflow.ac.solver.AcSolverResult;
import com.powsybl.openloadflow.ac.solver.AcSolverUtil;
import com.powsybl.openloadflow.ac.solver.NewtonRaphson;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.EquationVector;
import com.powsybl.openloadflow.equations.JacobianMatrix;
import com.powsybl.openloadflow.equations.StateVector;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.lf.outerloop.OuterLoopStatus;
import com.powsybl.openloadflow.network.util.VoltageInitializer;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfSynchronousNetwork;
import com.powsybl.openloadflow.network.impl.LfNetworkLoaderImpl;
import com.powsybl.openloadflow.network.util.PreviousValueVoltageInitializer;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Decides the LINE-26-4231 crux: does CPU's PURE inner Newton (NO outer loops at all), started from the
 * distributed base exactly like the hybrid prototype, land on the SAME operating point as the device
 * pure-inner solve (slack mismatch ~6.44 pu) or on CPU's normal one (~1.17 pu)?
 *
 * <p>If CPU pure-inner == 6.44, the device is correct and the 6.44 is the genuine pre-distribution
 * imbalance (the question becomes DS participation/limits). If CPU pure-inner == 1.17, the device
 * warm-started onto a DIFFERENT valid solution branch than CPU.
 */
class GpuCpuPureInnerProbe {

    private static final Path CASES = GpuTestPaths.caseDir();
    private static final String CASE = "case9241pegase.xiidm.gz";
    private static final int MAX_ITER = 40;
    private static final double TOL = 1e-12;

    static {
        GpuTestPaths.init();
    }

    /**
     * CONTROLLED scalar-vs-vectorized test with NO confounds: build BOTH equation systems freshly at the same
     * point (after the base LF, contingency applied), then run OLF's NewtonRaphson on each from the identical base
     * state + baseGenP. Compares the converged slack AND the active-equation/variable counts (to rule out a PV/PQ
     * activation difference). If they land on the same root -> my earlier "vectorized vs scalar differ" was a
     * confound; if they genuinely differ -> it is a real ill-conditioned knife-edge.
     */
    @Test
    void scalarVsVectorizedBranch() throws Exception {
        assumeTrue(Files.exists(CASES.resolve(CASE)), CASE + " not found");
        assumeTrue(Files.exists(Path.of("rte-loadflow-default-parameters.json")), "rte params not found");
        String branchId = System.getenv().getOrDefault("OLF_GPU_DISC_CTG", "LINE-26-4231");
        Network iidm = Network.read(CASES.resolve(CASE));
        LoadFlowParameters lfp = JsonLoadFlowParameters.read(Path.of("rte-loadflow-default-parameters.json"));
        OpenLoadFlowParameters olfp = OpenLoadFlowParameters.get(lfp);
        AcLoadFlowParameters acp = OpenLoadFlowParameters.createAcParameters(iidm, lfp, olfp,
                new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>(), false, false);
        acp.getNetworkParameters().setMaxSlackBusCount(1);
        LfNetwork network = LfNetwork.load(iidm, new LfNetworkLoaderImpl(), acp.getNetworkParameters(), ReportNode.NO_OP).get(0);

        // es created BEFORE the base LF — exactly like the hybrid/device (line ~101 there, base run at ~113)
        EquationSystem<AcVariableType, AcEquationType> esBefore =
                new AcEquationSystemCreator(network, acp.getEquationSystemCreationParameters()).create();
        AcSolverUtil.initStateVector(network, esBefore, acp.getVoltageInitializer());

        // base LF via its own context (matches production)
        AcLoadFlowContext baseCtx = new AcLoadFlowContext(network, acp);
        new AcloadFlowEngine(baseCtx).run();
        List<LfGenerator> gens = network.getBuses().stream().flatMap(b -> b.getGenerators().stream()).toList();
        double[] baseGenP = gens.stream().mapToDouble(LfGenerator::getTargetP).toArray();
        network.getBranchById(branchId).setDisabled(true);

        System.out.printf("%n=== scalar-vs-vectorized branch %s ===%n", branchId);
        // Solve with: esBefore (created before base run, like the hybrid), then FRESH es created here (after base
        // run). If esBefore -> high-loss and fresh -> low-loss, the confound is es-creation TIMING, not vectorization.
        solveWith(network, esBefore, gens, baseGenP, acp, "esBEFORE-scalar");
        solveWith(network, new AcEquationSystemCreator(network, acp.getEquationSystemCreationParameters()).create(),
                gens, baseGenP, acp, "esAFTER-scalar ");
        solveWith(network, new com.powsybl.openloadflow.ac.equations.vector.AcVectorizedEquationSystemCreator(
                network, acp.getEquationSystemCreationParameters()).create(), gens, baseGenP, acp, "esAFTER-vector ");
    }

    private static void solveWith(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> e,
                                  List<LfGenerator> gens, double[] baseGenP, AcLoadFlowParameters acp, String label) {
        for (int g = 0; g < gens.size(); g++) {
            gens.get(g).setTargetP(baseGenP[g]);
        }
        AcSolverUtil.initStateVector(network, e, new PreviousValueVoltageInitializer());   // warm from current network (base)
        AcTargetVector tgt = new AcTargetVector(network, e);
        EquationVector<AcVariableType, AcEquationType> ev = new EquationVector<>(e);
        try (JacobianMatrix<AcVariableType, AcEquationType> jm = new JacobianMatrix<>(e, new SparseMatrixFactory())) {
            NewtonRaphson nr = new NewtonRaphson(network, (NewtonRaphsonParameters) acp.getAcSolverParameters(), e, jm, tgt, ev, false);
            var r = nr.run(new PreviousValueVoltageInitializer(), ReportNode.NO_OP);
            double slack = r.getSlackBusActivePowerMismatch().values().stream().reduce(0., Double::sum);
            System.out.printf("  [%s] status=%s iters=%d slack=%.4f pu | vars=%d cols=%d%n",
                    label, r.getStatus(), r.getIterations(), slack,
                    e.getIndex().getSortedVariablesToFind().size(), e.getIndex().getColumnCount());
        }
    }

    /**
     * CLEAN, isolated test (no DC prepare or other solves to pollute network state first): does OLF's FULL
     * engine, forced to warm-start from the base state (exactly like the hybrid device solve), converge
     * LINE-26-4231 to the low-loss branch? If yes, the hybrid's failure is in its own DS driving / re-solve,
     * not the physics.
     */
    @Test
    void fullEngineWarmFromBaseClean() throws Exception {
        assumeTrue(Files.exists(CASES.resolve(CASE)), CASE + " not found");
        assumeTrue(Files.exists(Path.of("rte-loadflow-default-parameters.json")), "rte params not found");
        String branchId = System.getenv().getOrDefault("OLF_GPU_DISC_CTG", "LINE-26-4231");
        Network iidm = Network.read(CASES.resolve(CASE));
        LoadFlowParameters lfp = JsonLoadFlowParameters.read(Path.of("rte-loadflow-default-parameters.json"));
        OpenLoadFlowParameters olfp = OpenLoadFlowParameters.get(lfp);
        AcLoadFlowParameters acp = OpenLoadFlowParameters.createAcParameters(iidm, lfp, olfp,
                new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>(), false, false);
        acp.getNetworkParameters().setMaxSlackBusCount(1);
        LfNetwork network = LfNetwork.load(iidm, new LfNetworkLoaderImpl(), acp.getNetworkParameters(), ReportNode.NO_OP).get(0);
        EquationSystem<AcVariableType, AcEquationType> es =
                new AcEquationSystemCreator(network, acp.getEquationSystemCreationParameters()).create();
        AcSolverUtil.initStateVector(network, es, acp.getVoltageInitializer());
        AcLoadFlowContext baseCtx = new AcLoadFlowContext(network, acp);
        new AcloadFlowEngine(baseCtx).run();
        AcSolverUtil.initStateVector(network, es, new PreviousValueVoltageInitializer());
        double[] baseState = es.getStateVector().get().clone();
        List<LfGenerator> gens = network.getBuses().stream().flatMap(b -> b.getGenerators().stream()).toList();
        double[] baseGenP = gens.stream().mapToDouble(LfGenerator::getTargetP).toArray();

        System.out.printf("%n=== CLEAN full-engine WARM-from-base %s ===%n", branchId);
        System.out.printf("  [OUTER-LOOP ORDER] %s%n", acp.getOuterLoops().stream()
                .map(o -> o.getClass().getSimpleName()).toList());
        for (String ln : new String[]{"com.powsybl.openloadflow.network.util.ActivePowerDistribution",
                "com.powsybl.openloadflow.ac.outerloop.DistributedSlackOuterLoop",
                "com.powsybl.openloadflow.ac.solver.NewtonRaphson"}) {
            ((ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory.getLogger(ln))
                    .setLevel(ch.qos.logback.classic.Level.DEBUG);
        }
        network.getBranchById(branchId).setDisabled(true);

        // 1st: a CLEAN standalone NewtonRaphson warm-from-base (no outer loops) — is it 6.44 (high-loss) here?
        runNewton(network, es, gens, baseGenP, baseState, acp, es.getStateVector(),
                new PreviousValueVoltageInitializer(), "CLEAN-standalone-warm-from-base");

        // 1b: MANUAL runOuterLoop replication (NR re-solve + DS.check) with per-step mismatch — shows WHEN/HOW
        //     the engine escapes the 6.44 high-loss branch, which the hybrid's device re-solve fails to do.
        {
            for (int g = 0; g < gens.size(); g++) {
                gens.get(g).setTargetP(baseGenP[g]);
            }
            es.getStateVector().set(baseState.clone());
            AcSolverUtil.updateNetwork(network, es);
            AcTargetVector tgt = new AcTargetVector(network, es);
            EquationVector<AcVariableType, AcEquationType> ev = new EquationVector<>(es);
            try (JacobianMatrix<AcVariableType, AcEquationType> jm = new JacobianMatrix<>(es, new SparseMatrixFactory())) {
                NewtonRaphson nr = new NewtonRaphson(network, (NewtonRaphsonParameters) acp.getAcSolverParameters(), es, jm, tgt, ev, false);
                com.powsybl.openloadflow.ac.outerloop.AcOuterLoop dsLoop = acp.getOuterLoops().stream()
                        .filter(o -> o instanceof com.powsybl.openloadflow.ac.outerloop.DistributedSlackOuterLoop).findFirst().orElseThrow();
                java.lang.reflect.Constructor<com.powsybl.openloadflow.ac.AcOuterLoopContext> ctor =
                        com.powsybl.openloadflow.ac.AcOuterLoopContext.class.getDeclaredConstructor(LfNetwork.class);
                ctor.setAccessible(true);
                com.powsybl.openloadflow.ac.AcOuterLoopContext dsCtx = ctor.newInstance(network);
                dsCtx.setLoadFlowContext(new AcLoadFlowContext(network, acp));
                dsLoop.initialize(dsCtx);
                // EXACT runOuterLoop structure: DS.check reads the PREVIOUS solver result, distributes, and only
                // THEN do we re-solve (updating lastResult). NO extra solve before check, no fresh-mismatch recompute
                // — that is what keeps GenerationActivePowerDistributionStep.resetToInitialState's cumulative
                // (relative-to-initial) accounting consistent.
                AcSolverResult lastResult = nr.run(new PreviousValueVoltageInitializer(), ReportNode.NO_OP);
                System.out.printf("  [MANUAL DS-LOOP] initial NR: iters=%d slack=%.4f pu%n", lastResult.getIterations(),
                        lastResult.getSlackBusActivePowerMismatch().values().stream().reduce(0., Double::sum));
                for (int it = 0; it < 8; it++) {
                    dsCtx.setLastSolverResult(lastResult);
                    double gB = gens.stream().mapToDouble(LfGenerator::getTargetP).sum();
                    var st = dsLoop.check(dsCtx, ReportNode.NO_OP).status();
                    double gA = gens.stream().mapToDouble(LfGenerator::getTargetP).sum();
                    if (st == OuterLoopStatus.STABLE) {
                        System.out.printf("  [MANUAL DS-LOOP] it=%d DS=STABLE distributed=%.4f pu -> CONVERGED%n", it, gA - gB);
                        break;
                    }
                    lastResult = nr.run(new PreviousValueVoltageInitializer(), ReportNode.NO_OP);
                    double maxV = network.getBuses().stream().filter(b -> !b.isDisabled()).mapToDouble(LfBus::getV).max().orElse(0);
                    double minV = network.getBuses().stream().filter(b -> !b.isDisabled() && b.getV() > 0).mapToDouble(LfBus::getV).min().orElse(0);
                    System.out.printf("  [MANUAL DS-LOOP] it=%d DS=%s distributed=%.4f pu -> NR status=%s iters=%d slack=%.4f pu maxV=%.3f minV=%.3f%n",
                            it, st, gA - gB, lastResult.getStatus(), lastResult.getIterations(),
                            lastResult.getSlackBusActivePowerMismatch().values().stream().reduce(0., Double::sum), maxV, minV);
                }
            } catch (ReflectiveOperationException e) {
                System.out.println("  [MANUAL DS-LOOP] reflection failed: " + e);
            }
        }

        // 1c: engine warm-from-base with distributed slack OFF — reveals the branch the engine's INNER solve
        //     reaches (vs the standalone NR's -6.44 high-loss). If ~small, the engine's inner solve is low-loss.
        acp.setVoltageInitializer(new PreviousValueVoltageInitializer());
        {
            java.util.List<com.powsybl.openloadflow.ac.outerloop.AcOuterLoop> savedLoops = acp.getOuterLoops();
            acp.setOuterLoops(savedLoops.stream()
                    .filter(o -> !(o instanceof com.powsybl.openloadflow.ac.outerloop.DistributedSlackOuterLoop)).toList());
            for (int g = 0; g < gens.size(); g++) {
                gens.get(g).setTargetP(baseGenP[g]);
            }
            es.getStateVector().set(baseState.clone());
            AcSolverUtil.updateNetwork(network, es);
            var noDsRes = new AcloadFlowEngine(new AcLoadFlowContext(network, acp)).run();
            double s = 0;
            for (LfSynchronousNetwork sc : network.getSynchronousNetworks()) {
                for (LfBus sb : sc.getSlackBuses()) {
                    s += sb.getMismatchP();
                }
            }
            System.out.printf("  [ENGINE DS-loop removed, warm-from-base] solverStatus=%s solverIters=%d slackMismatch=%.4f pu%n",
                    noDsRes.getSolverStatus(), noDsRes.getSolverIterations(), s);
            // isolate the escaper: run each non-DS outer loop ALONE, warm-from-base, report the slack it reaches
            for (var loop : savedLoops) {
                if (loop instanceof com.powsybl.openloadflow.ac.outerloop.DistributedSlackOuterLoop) {
                    continue;
                }
                acp.setOuterLoops(java.util.List.of(loop));
                for (int g = 0; g < gens.size(); g++) {
                    gens.get(g).setTargetP(baseGenP[g]);
                }
                es.getStateVector().set(baseState.clone());
                AcSolverUtil.updateNetwork(network, es);
                var r = new AcloadFlowEngine(new AcLoadFlowContext(network, acp)).run();
                double s2 = 0;
                for (LfSynchronousNetwork sc : network.getSynchronousNetworks()) {
                    for (LfBus sb : sc.getSlackBuses()) {
                        s2 += sb.getMismatchP();
                    }
                }
                System.out.printf("  [ENGINE ONLY %s] status=%s iters=%d slack=%.4f pu%n",
                        loop.getClass().getSimpleName(), r.getSolverStatus(), r.getSolverIterations(), s2);
            }
            acp.setOuterLoops(savedLoops);
        }

        // 2nd: OLF's FULL engine, forced warm-from-base — run with maxOuter capped at 1,2,3 to see the trajectory
        acp.setVoltageInitializer(new PreviousValueVoltageInitializer());   // FORCE warm-start-from-base
        for (int cap : new int[]{1, 2, 3}) {
            for (int g = 0; g < gens.size(); g++) {
                gens.get(g).setTargetP(baseGenP[g]);
            }
            es.getStateVector().set(baseState.clone());
            AcSolverUtil.updateNetwork(network, es);
            acp.setMaxOuterLoopIterations(cap);
            var capRes = new AcloadFlowEngine(new AcLoadFlowContext(network, acp)).run();
            double capSlack = 0;
            for (LfSynchronousNetwork sc : network.getSynchronousNetworks()) {
                for (LfBus sb : sc.getSlackBuses()) {
                    capSlack += sb.getMismatchP();
                }
            }
            System.out.printf("  [ENGINE maxOuter=%d] solverStatus=%s outerLoopIters=%d dist=%.4f pu slackMismatch=%.4f pu genSum=%.4f%n",
                    cap, capRes.getSolverStatus(), capRes.getOuterLoopIterations(), capRes.getDistributedActivePower(),
                    capSlack, gens.stream().mapToDouble(LfGenerator::getTargetP).sum());
        }
        acp.setMaxOuterLoopIterations(30);

        // 1d: run the solver EXACTLY as AcloadFlowEngine does — via the context's solverFactory + the context's
        //     own Jacobian/Target/Equation vectors — with NO outer loops. If this reaches +0.05 (low-loss) while
        //     the standalone runNewton reached -6.44, the difference is the solver-construction path (context
        //     vectors), pinpointing what the hybrid device solve must replicate.
        {
            for (int g = 0; g < gens.size(); g++) {
                gens.get(g).setTargetP(baseGenP[g]);
            }
            es.getStateVector().set(baseState.clone());
            AcSolverUtil.updateNetwork(network, es);
            AcLoadFlowContext ctx = new AcLoadFlowContext(network, acp);
            var solver = acp.getSolverFactory().create(network, acp, ctx.getEquationSystem(), ctx.getJacobianMatrix(),
                    ctx.getTargetVector(), ctx.getEquationVector());
            var r = solver.run(new PreviousValueVoltageInitializer(), ReportNode.NO_OP);
            double s = r.getSlackBusActivePowerMismatch().values().stream().reduce(0., Double::sum);
            System.out.printf("  [CONTEXT-SOLVER inner only] status=%s iters=%d slack=%.4f pu | isVectorized=%b | es(shared) vars=%d ctx.es vars=%d%n",
                    r.getStatus(), r.getIterations(), s, acp.isVectorized(),
                    es.getIndex().getSortedVariablesToFind().size(), ctx.getEquationSystem().getIndex().getSortedVariablesToFind().size());
            ctx.close();
        }

        for (int g = 0; g < gens.size(); g++) {
            gens.get(g).setTargetP(baseGenP[g]);
        }
        es.getStateVector().set(baseState.clone());
        AcSolverUtil.updateNetwork(network, es);
        AcLoadFlowContext ctgCtx = new AcLoadFlowContext(network, acp);
        var res = new AcloadFlowEngine(ctgCtx).run();
        System.out.printf("  full engine WARM-from-base: solverStatus=%s solverIters=%d outerLoopIters=%d outerLoopStatus=%s distributedActivePower=%.4f pu (%.1f MW)%n",
                res.getSolverStatus(), res.getSolverIterations(), res.getOuterLoopIterations(),
                res.getOuterLoopResult().status(), res.getDistributedActivePower(), res.getDistributedActivePower() * 100);
        reportSlack(network, branchId + " out, CLEAN full engine WARM-from-base");
    }

    @Test
    void cpuPureInnerSlackMismatch() throws Exception {
        assumeTrue(Files.exists(CASES.resolve(CASE)), CASE + " not found");
        assumeTrue(Files.exists(Path.of("rte-loadflow-default-parameters.json")), "rte params not found");

        String branchId = System.getenv().getOrDefault("OLF_GPU_DISC_CTG", "LINE-26-4231");
        Network iidm = Network.read(CASES.resolve(CASE));
        LoadFlowParameters lfp = JsonLoadFlowParameters.read(Path.of("rte-loadflow-default-parameters.json"));
        OpenLoadFlowParameters olfp = OpenLoadFlowParameters.get(lfp);
        AcLoadFlowParameters acp = OpenLoadFlowParameters.createAcParameters(iidm, lfp, olfp,
                new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>(), false, false);
        acp.getNetworkParameters().setMaxSlackBusCount(1);

        LfNetwork network = LfNetwork.load(iidm, new LfNetworkLoaderImpl(), acp.getNetworkParameters(), ReportNode.NO_OP).get(0);
        EquationSystem<AcVariableType, AcEquationType> es =
                new AcEquationSystemCreator(network, acp.getEquationSystemCreationParameters()).create();
        AcSolverUtil.initStateVector(network, es, acp.getVoltageInitializer());

        // 1. base case exactly like the hybrid (full outer loops incl. distributed slack)
        AcLoadFlowContext context = new AcLoadFlowContext(network, acp);
        new AcloadFlowEngine(context).run();
        AcSolverUtil.initStateVector(network, es, new PreviousValueVoltageInitializer());
        double[] baseState = es.getStateVector().get().clone();
        List<LfGenerator> gens = network.getBuses().stream().flatMap(b -> b.getGenerators().stream()).toList();
        double[] baseGenP = gens.stream().mapToDouble(LfGenerator::getTargetP).toArray();

        System.out.printf("%n=== CPU PURE-INNER (no outer loops) %s ===%n", branchId);
        for (LfSynchronousNetwork sc : network.getSynchronousNetworks()) {
            LfGenerator refGen = sc.getReferenceGenerator();
            System.out.printf("  [REF-GEN] sc%d referenceGenerator=%s%s%n", sc.getNumSC(),
                    refGen == null ? "NULL" : refGen.getId(),
                    refGen == null ? "" : String.format(" (targetP=%.4f initialTargetP=%.4f)", refGen.getTargetP(), refGen.getInitialTargetP()));
        }
        reportSlack(network, "base intact (post-baseRun)");

        // 2. apply the outage on the network (device does this via a mask; CPU via disable)
        LfBranch br = network.getBranchById(branchId);
        br.setDisabled(true);

        StateVector sv = es.getStateVector();

        // 3. OLF's ACTUAL production NewtonRaphson (NO outer loops), warm-started from the base state with
        //    baseGenP fixed — the exact analog of the hybrid's round-0 pure-inner device solve.
        System.out.printf("  [INITIALIZER] acp.getVoltageInitializer()=%s solverParams=%s%n",
                acp.getVoltageInitializer().getClass().getSimpleName(), acp.getAcSolverParameters().getClass().getSimpleName());
        runNewton(network, es, gens, baseGenP, baseState, acp, sv, new PreviousValueVoltageInitializer(),
                "WARM-from-base");

        // 3b. OLF NewtonRaphson but COLD-started with the engine's configured voltage initializer (DC) — what
        //     CPU SA / LF actually uses per contingency.
        VoltageInitializer dcInit = acp.getVoltageInitializer();
        dcInit.prepare(network, ReportNode.NO_OP);
        runNewton(network, es, gens, baseGenP, baseState, acp, sv, dcInit, "COLD-DC-init");

        // 3c. Replicate the HYBRID BRIDGE's single DS step: NR warm-from-base -> 6.44 state, then build a
        //     DistributedSlackOuterLoop context by reflection (exactly like GpuHybridOuterLoopBridge) and call
        //     check() ONCE with the per-SC slack mismatch as the solver result. Report how much it distributes.
        //     If it distributes ~644 MW, the DS.check itself over-distributes (state/limits); if ~5 MW, the
        //     bridge context differs from AcloadFlowEngine's.
        runNewton(network, es, gens, baseGenP, baseState, acp, sv, new PreviousValueVoltageInitializer(), "WARM-for-DScheck");
        try {
            com.powsybl.openloadflow.ac.outerloop.AcOuterLoop dsLoop = acp.getOuterLoops().stream()
                    .filter(o -> o instanceof com.powsybl.openloadflow.ac.outerloop.DistributedSlackOuterLoop)
                    .findFirst().orElseThrow();
            java.lang.reflect.Constructor<com.powsybl.openloadflow.ac.AcOuterLoopContext> ctor =
                    com.powsybl.openloadflow.ac.AcOuterLoopContext.class.getDeclaredConstructor(LfNetwork.class);
            ctor.setAccessible(true);
            com.powsybl.openloadflow.ac.AcOuterLoopContext dsCtx = ctor.newInstance(network);
            AcLoadFlowContext lfCtx = new AcLoadFlowContext(network, acp);
            dsCtx.setLoadFlowContext(lfCtx);
            dsLoop.initialize(dsCtx);
            java.util.Map<Integer, Double> mism = new java.util.TreeMap<>();
            for (LfSynchronousNetwork sc : network.getSynchronousNetworks()) {
                mism.put(sc.getNumSC(), sc.getSlackBuses().stream().mapToDouble(LfBus::getMismatchP).sum());
            }
            dsCtx.setLastSolverResult(new AcSolverResult(com.powsybl.openloadflow.ac.solver.AcSolverStatus.CONVERGED, 1, mism));
            double gBefore = gens.stream().mapToDouble(LfGenerator::getTargetP).sum();
            var dsRes = dsLoop.check(dsCtx, ReportNode.NO_OP);
            double gAfter = gens.stream().mapToDouble(LfGenerator::getTargetP).sum();
            System.out.printf("  [BRIDGE-STYLE DS.check] status=%s fed mismatch=%.4f pu, distributed genSumDelta=%.4f pu (%.1f MW)%n",
                    dsRes.status(), mism.values().stream().reduce(0., Double::sum), gAfter - gBefore, (gAfter - gBefore) * 100);
        } catch (ReflectiveOperationException e) {
            System.out.println("  [BRIDGE-STYLE DS.check] reflection failed: " + e);
        }

        // 4. THE decisive CPU-SA-equivalent: from the SAME distributed base, with the branch out, run OLF's
        //    FULL outer-loop engine (distributed slack + reactive limits + ...). Does CPU's own DS resolve the
        //    6.44 pu quickly (=> the hybrid bridge is failing to replicate it), or does CPU also crawl (=> the
        //    pre-distributed base is the wrong starting point and contingencies should start from original P)?
        for (int g = 0; g < gens.size(); g++) {
            gens.get(g).setTargetP(baseGenP[g]);
        }
        sv.set(baseState.clone());
        AcSolverUtil.updateNetwork(network, es);
        AcLoadFlowContext ctgContext = new AcLoadFlowContext(network, acp);
        var result = new AcloadFlowEngine(ctgContext).run();
        System.out.printf("  FULL outer-loop run (CPU-SA equiv, distributed base + %s out): solverStatus=%s solverIters=%d outerLoopIters=%d outerLoopStatus=%s distributedActivePower=%.4f pu (%.1f MW)%n",
                branchId, result.getSolverStatus(), result.getSolverIterations(), result.getOuterLoopIterations(),
                result.getOuterLoopResult().status(), result.getDistributedActivePower(), result.getDistributedActivePower() * 100);
        reportSlack(network, branchId + " out, CPU FULL outer loops (from distributed base)");

        // 4b. DECISIVE: run the FULL engine but FORCED to warm-start from the base state (PreviousValue init) —
        //     exactly what the hybrid device solve does. If OLF's OWN outer loops (DS) rescue the base-warm-start
        //     6.44 pu and converge to low-loss, then the hybrid's DS-DRIVING is the bug. If the engine also gets
        //     stuck / fails from base warm start, then INITIALIZATION (DC vs base) is the fix.
        for (int g = 0; g < gens.size(); g++) {
            gens.get(g).setTargetP(baseGenP[g]);
        }
        sv.set(baseState.clone());
        AcSolverUtil.updateNetwork(network, es);
        acp.setVoltageInitializer(new PreviousValueVoltageInitializer());   // FORCE warm-start-from-base
        AcLoadFlowContext warmContext = new AcLoadFlowContext(network, acp);
        var warmResult = new AcloadFlowEngine(warmContext).run();
        System.out.printf("  FULL outer-loop run WARM-from-base (PreviousValue init, %s out): solverStatus=%s solverIters=%d outerLoopIters=%d outerLoopStatus=%s distributedActivePower=%.4f pu (%.1f MW)%n",
                branchId, warmResult.getSolverStatus(), warmResult.getSolverIterations(), warmResult.getOuterLoopIterations(),
                warmResult.getOuterLoopResult().status(), warmResult.getDistributedActivePower(), warmResult.getDistributedActivePower() * 100);
        reportSlack(network, branchId + " out, CPU FULL outer loops WARM-from-base");

        // 5. BRANCH TEST: the network now sits on the low-loss solution the engine found. Re-fix generation at
        //    baseGenP (undo the 5.3 MW distribution) and run a PURE inner Newton (no DS) WARM-started from this
        //    low-loss state. If it STAYS low-loss, LINE-26-4231 has two AC solutions and the 6.44 pu is just the
        //    branch the base-warm-started pure-inner solve happens to land on (branch-selection problem). If it
        //    jumps back to 6.44, then 6.44 is the unique solution and DS was holding the state off it.
        double[] lowLossState = es.getStateVector().get().clone();
        for (int g = 0; g < gens.size(); g++) {
            gens.get(g).setTargetP(baseGenP[g]);
        }
        sv.set(lowLossState);
        AcTargetVector target3 = new AcTargetVector(network, es);
        EquationVector<AcVariableType, AcEquationType> eq3 = new EquationVector<>(es);
        int iters3 = 0;
        try (JacobianMatrix<AcVariableType, AcEquationType> j = new JacobianMatrix<>(es, new SparseMatrixFactory())) {
            eq3.minus(target3);
            for (int it = 0; it < MAX_ITER && maxAbs(eq3.getArray()) >= TOL; it++) {
                double[] dx = eq3.getArray().clone();
                j.solveTransposed(dx);
                sv.minus(dx);
                eq3.minus(target3);
                iters3++;
            }
        }
        AcSolverUtil.updateNetwork(network, es);
        System.out.printf("  pure-inner Newton (WARM from LOW-LOSS state, baseGenP fixed) converged in %d iters, final ||F||inf=%.3e%n", iters3, maxAbs(eq3.getArray()));
        reportSlack(network, branchId + " out, CPU pure-inner WARM-from-LOW-LOSS (baseGenP fixed)");
    }

    private static void runNewton(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                  List<LfGenerator> gens, double[] baseGenP, double[] baseState,
                                  AcLoadFlowParameters acp, StateVector sv, VoltageInitializer init, String label) {
        for (int g = 0; g < gens.size(); g++) {
            gens.get(g).setTargetP(baseGenP[g]);
        }
        sv.set(baseState.clone());
        AcSolverUtil.updateNetwork(network, es);   // so a PreviousValue initializer reads the base state
        AcTargetVector target = new AcTargetVector(network, es);
        EquationVector<AcVariableType, AcEquationType> eq = new EquationVector<>(es);
        try (JacobianMatrix<AcVariableType, AcEquationType> j = new JacobianMatrix<>(es, new SparseMatrixFactory())) {
            NewtonRaphson nr = new NewtonRaphson(network, (NewtonRaphsonParameters) acp.getAcSolverParameters(),
                    es, j, target, eq, false);
            AcSolverResult res = nr.run(init, ReportNode.NO_OP);
            double mism = res.getSlackBusActivePowerMismatch().values().stream().reduce(0., Double::sum);
            System.out.printf("  NewtonRaphson (%s): status=%s iters=%d slackMismatchSum=%.4f pu (%.1f MW)%n",
                    label, res.getStatus(), res.getIterations(), mism, mism * 100);
        }
        reportSlack(network, "NR " + label);
    }

    private static void reportSlack(LfNetwork network, String label) {
        for (LfSynchronousNetwork sc : network.getSynchronousNetworks()) {
            for (LfBus sb : sc.getSlackBuses()) {
                System.out.printf("  [%s] sc%d slack %s tP=%.4f mismP=%.4f pu (%.1f MW)%n",
                        label, sc.getNumSC(), sb.getId(), sb.getTargetP(), sb.getMismatchP(), sb.getMismatchP() * 100);
            }
        }
    }

    private static double maxAbs(double[] a) {
        double m = 0;
        for (double v : a) {
            m = Math.max(m, Math.abs(v));
        }
        return m;
    }
}
