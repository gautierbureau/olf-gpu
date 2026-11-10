/*
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.contingency.BranchContingency;
import com.powsybl.contingency.Contingency;
import com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory;
import com.powsybl.iidm.network.Network;
import com.powsybl.openloadflow.CommonTestConfig;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcStructures;
import com.powsybl.openloadflow.ac.solver.gpu.BatchedLuNative;
import com.powsybl.openloadflow.ac.solver.gpu.GpuBatchContextCache;
import com.powsybl.openloadflow.ac.solver.gpu.GpuBatchedLuOrdering;
import com.powsybl.openloadflow.ac.solver.gpu.GpuLinearSolver;
import com.powsybl.openloadflow.ac.solver.gpu.GpuLoadFlowParameters;
import com.powsybl.openloadflow.ac.solver.gpu.GpuMatchingAlgorithm;
import com.powsybl.openloadflow.ac.solver.gpu.GpuPivotType;
import com.powsybl.openloadflow.ac.solver.gpu.GpuReorderingAlgorithm;
import com.powsybl.openloadflow.ac.solver.gpu.NewtonSettingsNative;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchedSecurityAnalysisSolver;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.GpuBatchScopeCensus;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuLinearSolveMode;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.sa.AcSecurityAnalysis;
import com.powsybl.openloadflow.sa.OpenSecurityAnalysisProvider;
import com.powsybl.security.SecurityAnalysisResult;
import com.powsybl.security.results.PostContingencyResult;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static com.powsybl.openloadflow.sa.gpu.GpuSaTestNetworks.*;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GPU-batched security analysis against the CPU: the batched solver itself: line search, stopping criteria, chunking, robust mode and the CPU fallback.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuSaSolverTest extends AbstractGpuSaTest {

    GpuSaSolverTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    /** MULTI-CHUNK parity: the same contingencies with the chunk capped at 2, so the batch spans
     *  several chunks — every other test fits in one, so nothing else here exercises a chunk
     *  boundary at all. Mixed generator + branch outages (generator loss pulls in the
     *  loss-pre-distribution override composition). Same results as the CPU, at 1e-12.
     *
     *  <p>HONEST SCOPE: this covers chunk-boundary PARITY. It does NOT reproduce the 2026-09-08
     *  stale-per-chunk-index regression (an index over the per-chunk override list that outlived
     *  its list): on a case this small the composition path either does not run or a stale index
     *  still lands in range, so the test passes with and without the fix. That regression was
     *  caught on rte6515 (4 chunks of 643) and is only reachable at that scale. */
    @Test
    void deviceMultiChunkMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        // ORDER MATTERS. Generator outages carry a scheduled active-power loss, so they go through
        // the loss pre-distribution, which COMPOSES per-scenario DS-row overrides onto the chunk's
        // override list — the per-chunk scratch this test exists to guard. With the chunk capped at 2:
        //   chunk 1 = (gen, gen)   -> a LONG override list, DS rows sitting at high positions
        //   chunk 2 = (branch, gen) -> slot 0 contributes nothing, so the list is SHORT when slot 1
        //                              composes its DS rows under the SAME slot-keyed keys
        // A per-chunk index that outlives its list therefore resolves chunk-1 positions against a
        // shorter chunk-2 list: out of range (throws, batch lost) or, worse, in range and pointing
        // at another row (silent wrong override). Identical chunk shapes hide it, which is why the
        // kinds are interleaved rather than grouped.
        List<Contingency> ctgs = List.of(
                new Contingency("B2-G", new com.powsybl.contingency.GeneratorContingency("B2-G")),
                new Contingency("B3-G", new com.powsybl.contingency.GeneratorContingency("B3-G")),
                new Contingency("L1-2-1", new BranchContingency("L1-2-1")),
                new Contingency("B6-G", new com.powsybl.contingency.GeneratorContingency("B6-G")),
                new Contingency("L2-3-1", new BranchContingency("L2-3-1")),
                new Contingency("B8-G", new com.powsybl.contingency.GeneratorContingency("B8-G")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        GpuBatchContextCache.forceChunkSizeForTests(2);   // -> 4 chunks
        try {
            assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
            // > one chunk's worth proves the multi-chunk path ran (one generator outage is
            // CPU-routed by qualification on this network, which is by design)
            assertTrue(GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get() > 2,
                    "the batch must span several chunks");
        } finally {
            GpuBatchContextCache.forceChunkSizeForTests(null);
        }
    }

    /** PER_EQUATION_TYPE stopping criteria: the batch installs per-row thresholds
     *  (setBatchRowCriteria — every row's |F| under its type threshold, OLF's
     *  PerEquationTypeStoppingCriteria) and the base LF stays on the CPU; shallow explicit
     *  thresholds make the stopping DECISION the thing under test. DS+RL on, ieee14. */
    @Test
    void devicePerEquationTypeCriteriaMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        List<Contingency> ctgs = List.of(new Contingency("L1-2-1", new BranchContingency("L1-2-1")),
                new Contingency("L1-5-1", new BranchContingency("L1-5-1")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setNewtonRaphsonStoppingCriteriaType(
                        com.powsybl.openloadflow.ac.solver.NewtonRaphsonStoppingCriteriaType.PER_EQUATION_TYPE_CRITERIA)
                .setMaxActivePowerMismatch(1e-2).setMaxReactivePowerMismatch(1e-2)
                .setMaxVoltageMismatch(1e-4).setMaxAngleMismatch(1e-4)
                .setMaxRatioMismatch(1e-4).setMaxSusceptanceMismatch(1e-4);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }

    /** PER_EQUATION_TYPE criteria on relocated rows. The coupler-tree relocation puts a child section's balances on
     *  its parent coupler's dummy rows and the coupler's ZERO equations on the child's rows; the device took each row's
     *  threshold from the VARIABLE at its column, so a child's P balance was judged at the angle threshold, its
     *  coupler's ZERO_V at the reactive one, and the reverse once a section loss re-homed the child. Thresholds far
     *  apart (1 MW / 1 MVar against 1e-6 pu for voltages and angles) make the stopping decision depend on the type.
     *  The three-section chain, each section lost in turn. 1e-12 on device. */
    @Test
    void devicePerEquationTypeCriteriaOnRelocatedRowsMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = liveSectionChainNetwork();
        List<Contingency> ctgs = new java.util.ArrayList<>();
        for (int k = 0; k < 3; k++) {
            ctgs.add(new Contingency("bus-" + k, new com.powsybl.contingency.BusbarSectionContingency("bbs" + k)));
        }
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl0"))
                .setNewtonRaphsonStoppingCriteriaType(
                        com.powsybl.openloadflow.ac.solver.NewtonRaphsonStoppingCriteriaType.PER_EQUATION_TYPE_CRITERIA)
                .setMaxActivePowerMismatch(1).setMaxReactivePowerMismatch(1)
                .setMaxVoltageMismatch(1e-6).setMaxAngleMismatch(1e-6)
                .setMaxRatioMismatch(1e-6).setMaxSusceptanceMismatch(1e-6);
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp, true);
        assertEquals(3, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** A scenario's answer must not depend on which scenarios share its chunk. The adaptive iterative refinement of
     *  coupler systems used to refine EVERY lane while the worst one was above tolerance, so a lane took a different
     *  number of steps in a batch than alone (network.xiidm busbar JZDQ: 0.125 MW off the CPU in a 700-wide chunk,
     *  1.7e-8 MW in another). Every tie of a coupler chain solved as one chunk and one by one must give the same bus
     *  voltages to 1e-11 kV - the worst-lane rule put t17 1.8e-9 kV apart; what is left is cuDSS's rounding, which
     *  changes from one run to the next (<= 2.8e-13 kV over 8 runs under AMD, 0 in cuDSS's deterministic mode).
     *  By exception 1e-8 kV under nested dissection, where that rounding reaches 1.2e-9 kV (8 runs: the chunk against
     *  each scenario alone 3e-13 to 1.2e-9, the same chunk twice up to 6.2e-10): there the test only catches a
     *  dependence above that noise. */
    @Test
    void deviceScenarioDoesNotDependOnItsChunkMates() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = couplerChainSaNetwork(40);
        List<String> ties = network.getLineStream().map(l -> l.getId()).filter(id -> id.startsWith("t")).sorted().toList();
        List<Contingency> ctgs = new java.util.ArrayList<>(ties.stream()
                .map(id -> new Contingency(id, new com.powsybl.contingency.BranchContingency(id))).toList());
        for (int k = 0; k + 1 < ties.size(); k++) {       // two adjacent ties out: a longer segment hangs on couplers alone
            ctgs.add(new Contingency(ties.get(k) + "+" + ties.get(k + 1), new com.powsybl.contingency.BranchContingency(ties.get(k)),
                    new com.powsybl.contingency.BranchContingency(ties.get(k + 1))));
        }
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters();
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        double tolerance = nestedDissectionOrdering(lfp) ? 1e-8 : 1e-11;
        java.util.Set<String> allVls = network.getVoltageLevelStream().map(vl -> vl.getId())
                .collect(java.util.stream.Collectors.toSet());
        List<com.powsybl.security.monitor.StateMonitor> monitors = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(), allVls, java.util.Set.of()));
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            SecurityAnalysisResult together = runSecurityAnalysis(network, ctgs, monitors, sap);
            long engagedTogether = GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get();
            com.powsybl.openloadflow.ac.solver.gpu.GpuBatchContextCache.forceChunkSizeForTests(1);
            SecurityAnalysisResult alone = runSecurityAnalysis(network, ctgs, monitors, sap);
            assertEquals(ctgs.size(), engagedTogether, "every tie outage on the device");
            assertEquals(engagedTogether, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "the same scenarios on the device");
            for (int i = 0; i < ctgs.size(); i++) {
                var t = together.getPostContingencyResults().get(i);
                var a = alone.getPostContingencyResults().get(i);
                assertEquals(a.getStatus(), t.getStatus());
                var tb = t.getNetworkResult().getBusResults();
                var ab = a.getNetworkResult().getBusResults();
                assertEquals(ab.size(), tb.size());
                for (int b = 0; b < ab.size(); b++) {
                    assertEquals(ab.get(b).getV(), tb.get(b).getV(), tolerance,
                            "bus " + ab.get(b).getBusId() + " for " + t.getContingency().getId() + ": chunk of " + ctgs.size() + " vs alone");
                }
            }
        } finally {
            com.powsybl.openloadflow.ac.solver.gpu.GpuBatchContextCache.forceChunkSizeForTests(null);
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    /** ... nor at the Newton budget: the pass that runs it out ends with a mismatch-only fill, whose relocated reactive
     *  outputs took the carrier rows of the chunk's FIRST scenario (its pilot gates and DISTR_Q subgroups) instead of
     *  their own scenario's. Three merged sections whose main's generator the base switches to PQ, so the pilot's own
     *  reactive balance rides a carrier row picked per scenario: in one chunk with the other section losses, the loss
     *  of bbs2 ended its budget on a mismatch of 0.81 where it ends on 2.9e-6 alone. The budget is the smallest the CPU converges
     *  every section loss under, one short. */
    @Test
    void deviceMismatchAtTheBudgetDoesNotDependOnItsChunkMates() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = splitGroupNetwork(3, java.util.Set.of(0));
        network.getLineStream().forEach(l -> l.setX(30));  // longer lines: a section loss needs a step more than the base
        List<Contingency> ctgs = new java.util.ArrayList<>();
        for (int k = 0; k < 3; k++) {                       // every section listed: OLF retains every coupler
            ctgs.add(new Contingency("bus-" + k, new com.powsybl.contingency.BusbarSectionContingency("bbs" + k)));
        }
        java.util.function.Supplier<com.powsybl.loadflow.LoadFlowParameters> parameters = () -> {
            com.powsybl.loadflow.LoadFlowParameters p = new com.powsybl.loadflow.LoadFlowParameters()
                    .setDistributedSlack(false).setUseReactiveLimits(true);
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(p)
                    .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                    .setSlackBusesIds(List.of("vl0_b"));
            return p;
        };
        // the solved base state on the network, so the base load flow needs as few steps as it can
        new com.powsybl.loadflow.LoadFlow.Runner(new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory()))
                .run(network, parameters.get());
        com.powsybl.loadflow.LoadFlowParameters lfp = parameters.get()
                .setVoltageInitMode(com.powsybl.loadflow.LoadFlowParameters.VoltageInitMode.PREVIOUS_VALUES);
        com.powsybl.openloadflow.OpenLoadFlowParameters ext = com.powsybl.openloadflow.OpenLoadFlowParameters.get(lfp);
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        int budget = -1;                                     // the smallest budget the CPU converges every section loss under
        int baseBudget = -1;                                 // ... and the base load flow
        for (int m = 2; m <= 30 && budget < 0; m++) {
            ext.setMaxNewtonRaphsonIterations(m);
            SecurityAnalysisResult cpu = runSecurityAnalysis(network, ctgs, java.util.Collections.emptyList(), sap);
            if (cpu.getPreContingencyResult().getStatus() != com.powsybl.loadflow.LoadFlowResult.ComponentResult.Status.CONVERGED) {
                continue;                                    // a failed base has no post-contingency results
            }
            baseBudget = baseBudget < 0 ? m : baseBudget;
            if (cpu.getPostContingencyResults().stream()
                    .allMatch(r -> r.getStatus() == com.powsybl.security.PostContingencyComputationStatus.CONVERGED)) {
                budget = m;
            }
        }
        assertTrue(budget > baseBudget, "a section loss must need more steps than the base, smallest converging budgets "
                + budget + " and " + baseBudget);
        ext.setMaxNewtonRaphsonIterations(budget - 1);
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            GpuBatchScopeCensus.reset();
            runSecurityAnalysis(network, ctgs, java.util.Collections.emptyList(), sap);
            java.util.Map<String, Double> together = new java.util.HashMap<>(GpuBatchedAcSecurityAnalysis.LAST_DEVICE_FINAL_MISMATCH);
            assertTrue(GpuBatchScopeCensus.snapshot().getOrDefault("DEVICE:MAX_ITERATION_REACHED", 0L) >= 1,
                    "a section loss runs its budget out on the device; census=" + GpuBatchScopeCensus.snapshot());
            com.powsybl.openloadflow.ac.solver.gpu.GpuBatchContextCache.forceChunkSizeForTests(1);
            runSecurityAnalysis(network, ctgs, java.util.Collections.emptyList(), sap);
            java.util.Map<String, Double> alone = new java.util.HashMap<>(GpuBatchedAcSecurityAnalysis.LAST_DEVICE_FINAL_MISMATCH);
            assertEquals(ctgs.size(), together.size(), "every section loss on the device: " + together.keySet());
            assertEquals(together.keySet(), alone.keySet());
            for (String id : together.keySet()) {
                assertEquals(alone.get(id), together.get(id), 1e-8, id + ": chunk of " + ctgs.size() + " vs alone");
            }
        } finally {
            com.powsybl.openloadflow.ac.solver.gpu.GpuBatchContextCache.forceChunkSizeForTests(null);
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    /** Each scenario runs OLF's outer-loop sequence on its own (AcloadFlowEngine.runOuterLoops): checked after every
     *  solve at the loop that made it re-solve, on along the next loops while they are STABLE, and done when a sweep
     *  re-solved nothing or a new sweep reaches the loop it was last UNSTABLE at. With ReactiveLimits the only loop, OLF
     *  checks once per solve: l0p switches nothing (one solve, one check), l0d switches g1 to PQ (two and two). The batch
     *  used to move every scenario from loop to loop together and restart every sweep when one re-solved: l0p was
     *  checked a second time for l0d's sake, and l0d a third time, past its last unstable loop. The counts must not
     *  depend on the chunk either. */
    @Test
    void deviceChecksEachScenarioLikeOlfWhateverItsChunkMates() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = pilotSelfSwitchingToPqNetwork(80);
        List<Contingency> ctgs = List.of(new Contingency("l0p", new BranchContingency("l0p")),
                new Contingency("l0d", new BranchContingency("l0d")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(false).setUseReactiveLimits(true);
        Map<String, int[]> together = new HashMap<>();
        Map<String, int[]> alone = new HashMap<>();
        try {
            GpuBatchedSecurityAnalysisSolver.recordOuterLoopStatsForTests(together);
            assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
            assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "both outages on the device");
            GpuBatchContextCache.forceChunkSizeForTests(1);
            GpuBatchedSecurityAnalysisSolver.recordOuterLoopStatsForTests(alone);
            assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        } finally {
            GpuBatchedSecurityAnalysisSolver.recordOuterLoopStatsForTests(null);
            GpuBatchContextCache.forceChunkSizeForTests(null);
        }
        // {solve passes, first solve's Newton iterations, total Newton iterations, checks}
        for (Map<String, int[]> stats : List.of(together, alone)) {
            assertEquals(1, stats.get("l0p")[0], "l0p switches nothing: one solve");
            assertEquals(1, stats.get("l0p")[3], "l0p: one check");
            assertEquals(2, stats.get("l0d")[0], "l0d switches g1 to PQ: two solves");
            assertEquals(2, stats.get("l0d")[3], "l0d: a check after each solve");
        }
    }

    /** The scenarios' outer loops run inside the batched Newton loop: a scenario whose solve converges is checked at
     *  once and, when a check is UNSTABLE, restarts its Newton in the same batched iteration, while its chunk mates go
     *  on with theirs. A chunk then takes one batched iteration more than the most Newton iterations one of its
     *  scenarios took over all its solves, whatever the others do. The batch used to solve in passes, each as long as
     *  its slowest scenario's solve plus a test iteration of its own: l0d's re-solve waited for the first pass to end. */
    @Test
    void deviceRestartsAScenarioWithoutWaitingForItsChunkMates() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = pilotSelfSwitchingToPqNetwork(80);
        List<Contingency> ctgs = List.of(new Contingency("l0p", new BranchContingency("l0p")),
                new Contingency("l0d", new BranchContingency("l0d")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(false).setUseReactiveLimits(true);
        Map<String, int[]> stats = new HashMap<>();
        try {
            GpuBatchedSecurityAnalysisSolver.recordOuterLoopStatsForTests(stats);
            assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
            assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "both outages on the device");
        } finally {
            GpuBatchedSecurityAnalysisSolver.recordOuterLoopStatsForTests(null);
        }
        // {solve passes, first solve's Newton iterations, total Newton iterations, checks, the chunk's batched iterations}
        assertEquals(2, stats.get("l0d")[0], "l0d switches g1 to PQ: two solves");
        assertEquals(stats.get("l0p")[4], stats.get("l0d")[4], "one chunk");
        int mostNewtonIterations = Math.max(stats.get("l0p")[2], stats.get("l0d")[2]);
        assertEquals(mostNewtonIterations + 1, stats.get("l0d")[4], "the chunk's batched Newton iterations");
    }

    /** The robust solve mode (6: one matrix object repointed per scenario, cuDSS pivoting on the raw matrix - the
     *  retry of a rejected scenario, or a base load flow that fell into it) runs the in-kernel transformer, shunt
     *  and secondary voltage control loops like the UBATCH modes do, solving their sensitivities scenario by
     *  scenario on its own factors. They used to report themselves STABLE there, so the same fixtures under ROBUST
     *  ended at the pre-loop state: BUS_2 0.6 kV, b2 0.3 kV and B2 0.2 kV off OLF. UBATCH's results are unchanged. */
    @Test
    void deviceRobustModeRunsTheInKernelLoops() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        GpuBatchContextCache.forceModeForTests(GpuLinearSolveMode.ROBUST);
        try {
            assertIncrementalTvcTapMovesInKernel();
            assertIncrementalSharedTvcMatchesCpu();
            assertIncrementalShuntVoltageControlMatchesCpu();
            assertSecondaryVoltageControlMatchesCpu();
            assertSecondaryVoltageControlRemoteMatchesCpu();
        } finally {
            GpuBatchContextCache.forceModeForTests(null);
        }
    }

    /** Our batched LU (BATCHED_LU mode) runs the in-kernel transformer, shunt and secondary voltage control loops too, their
     *  sensitivities solved on its own factors of every slot. */
    @Test
    void deviceBatchedLuRunsTheInKernelLoops() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        GpuBatchContextCache.forceModeForTests(GpuLinearSolveMode.BATCHED_LU);
        try {
            assertIncrementalTvcTapMovesInKernel();
            assertIncrementalSharedTvcMatchesCpu();
            assertIncrementalShuntVoltageControlMatchesCpu();
            assertSecondaryVoltageControlMatchesCpu();
            assertSecondaryVoltageControlRemoteMatchesCpu();
        } finally {
            GpuBatchContextCache.forceModeForTests(null);
        }
    }

    /** linearSolver = BATCHED_LU solves the batched Newton steps on our batched LU, which factorizes only the
     *  contingencies still iterating: outages of different difficulty retire at different iterations, and a retired
     *  slot must keep its state while the others go on. */
    @Test
    void batchedLuParameterSolvesOnOurLuAndMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create118();
        List<Contingency> ctgs = network.getLineStream().limit(60)
                .map(l -> new Contingency(l.getId(), new BranchContingency(l.getId()))).toList();
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters();
        GpuLoadFlowParameters.create(lfp).setLinearSolver(GpuLinearSolver.BATCHED_LU);
        long factorizations = BatchedLuNative.factorizationCounts()[0];
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertTrue(BatchedLuNative.factorizationCounts()[0] > factorizations, "the run factorizes on our batched LU");
    }

    /** LINE_SEARCH state-vector scaling in the batched SA. The scope used to map it to an undamped Newton (only
     *  MAX_VOLTAGE_CHANGE reached the device), so a contingency whose first steps overshoot took another trajectory
     *  than OLF's backtracking one. Two-bus network loaded near the nose of its PV curve with a parallel line: losing
     *  that line leaves the single-line case undamped Newton cannot solve. 1e-12 on device. */
    @Test
    void deviceLineSearchScalingMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.TwoBusNetworkFactory.create();
        network.getLoad("l1").setP0(3.902);
        com.powsybl.iidm.network.Line l0 = network.getLine("l12");
        network.newLine().setId("l12_BIS")
                .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l0.getR() * 0.2).setX(l0.getX() * 0.2).setG1(0).setB1(0).setG2(0).setB2(0).add();
        List<Contingency> ctgs = List.of(new Contingency("l12_BIS", new BranchContingency("l12_BIS")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(false).setUseReactiveLimits(false);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setStateVectorScalingMode(com.powsybl.openloadflow.ac.solver.StateVectorScalingMode.LINE_SEARCH)
                .setMaxNewtonRaphsonIterations(30);
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** The robust single-matrix mode (6) takes OLF's MAX_VOLTAGE_CHANGE step like every other mode. Its per-scenario
     *  loop applied the full Newton step, so a scenario retried in that mode followed another trajectory and stopped
     *  at another point inside the tolerance. The limits are tight enough to clamp the first steps of every outage;
     *  the whole run, base load flow included, is solved in ROBUST. */
    @Test
    void robustModeClampsTheStepLikeOlf() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        List<Contingency> ctgs = network.getLineStream().limit(6)
                .map(l -> new Contingency(l.getId(), new BranchContingency(l.getId()))).toList();
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters();
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setStateVectorScalingMode(com.powsybl.openloadflow.ac.solver.StateVectorScalingMode.MAX_VOLTAGE_CHANGE)
                .setMaxVoltageChangeStateVectorScalingMaxDv(0.01)
                .setMaxVoltageChangeStateVectorScalingMaxDphi(0.01)
                .setMaxNewtonRaphsonIterations(100);              // the clamped base needs ~30 steps from flat angles
        GpuBatchContextCache.forceModeForTests(GpuLinearSolveMode.ROBUST);
        try {
            assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
            assertEquals(ctgs.size(), GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get());
        } finally {
            GpuBatchContextCache.forceModeForTests(null);
        }
    }

    /** OLF's NewtonRaphson.run counts a step BEFORE testing it and reports MAX_ITERATION_REACHED whenever
     *  iterations >= maxIterations: a post-contingency solve whose stopping test passes after the LAST
     *  allowed step is MAX_ITERATION_REACHED on the CPU, with no network result. The device tested
     *  convergence before it tested the budget and answered CONVERGED, with a state and violations, for
     *  that same solve. The budget is the exact number of steps the contingency needs, found on the CPU;
     *  the base load flow starts from its own solved state (PREVIOUS_VALUES), so it needs one step
     *  under any budget. */
    @Test
    void deviceReportsMaxIterationWhenConvergingOnTheLastAllowedStep() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertDeviceDecidesTheIterationBudget(IeeeCdfNetworkFactory.create14(),
                List.of(new Contingency("L1-2-1", new BranchContingency("L1-2-1"))));
    }

    /** The pass that runs the Newton budget out ends with a mismatch-only fill, after which the device scores the
     *  pass's last linear solve on the values that solve factorized: a sound solve is OLF's MAX_ITERATION_REACHED, a
     *  junk one a reject the CPU replays. That fill added the couplers' zero-impedance terms and the merged controls'
     *  DISTR_Q terms into those values a second time (a coupler's +-1 became +-2), so on a coupler chain the sound
     *  last solve of a contingency one step short scored as junk, and the device handed the contingency to the CPU
     *  instead of deciding it. */
    @Test
    void deviceDecidesTheIterationBudgetOnACouplerChain() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertDeviceDecidesTheIterationBudget(couplerChainSaNetwork(8), List.of(new Contingency("t5", new BranchContingency("t5"))));
    }

    /** One contingency at the exact number of steps it needs on the CPU, and one short: the device itself reports
     *  MAX_ITERATION_REACHED one short, CONVERGED at the budget. */
    private void assertDeviceDecidesTheIterationBudget(Network network, List<Contingency> contingencies) {
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(false).setUseReactiveLimits(false)
                .setVoltageInitMode(com.powsybl.loadflow.LoadFlowParameters.VoltageInitMode.PREVIOUS_VALUES);
        com.powsybl.openloadflow.OpenLoadFlowParameters ext = com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp);
        // the solved base state on the network, so both arms' base load flow needs a single step
        new com.powsybl.loadflow.LoadFlow.Runner(new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory()))
                .run(network, new com.powsybl.loadflow.LoadFlowParameters().setDistributedSlack(false).setUseReactiveLimits(false));
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        int budget = -1;                                     // the smallest budget the CPU converges under
        for (int m = 2; m <= 30 && budget < 0; m++) {
            ext.setMaxNewtonRaphsonIterations(m);
            SecurityAnalysisResult cpu = runSecurityAnalysis(network, contingencies, java.util.Collections.emptyList(), sap);
            assertEquals(com.powsybl.loadflow.LoadFlowResult.ComponentResult.Status.CONVERGED,
                    cpu.getPreContingencyResult().getStatus(), "the base must converge under budget " + m);
            if (cpu.getPostContingencyResults().get(0).getStatus() == com.powsybl.security.PostContingencyComputationStatus.CONVERGED) {
                budget = m;
            }
        }
        assertTrue(budget >= 3, "the contingency must need at least two steps, smallest converging budget " + budget);
        // one step short: OLF's test passes ON the last allowed step and it reports MAX_ITERATION_REACHED
        ext.setMaxNewtonRaphsonIterations(budget - 1);
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpuNoEngage(network, contingencies, 1e-12, lfp);
        assertEquals(1L, GpuBatchScopeCensus.snapshot().getOrDefault("DEVICE:MAX_ITERATION_REACHED", 0L),
                "the device itself must have decided it; census=" + GpuBatchScopeCensus.snapshot());
        // one more step allowed: CONVERGED on both arms, solved on the device
        ext.setMaxNewtonRaphsonIterations(budget);
        assertGpuMatchesCpu(network, contingencies, 1e-12, lfp);
    }

    /** At its Newton budget the device scores the solve's last linear solve, J.dx - B against the mismatch, to tell OLF's
     *  MAX_ITERATION_REACHED (the budget spent on sound steps) from a factorization that went wrong. It scored it after
     *  the budget's F-only fill, which keeps the Jacobian of that solve but whose reactive power distribution, zero
     *  impedance and slack distribution fills add into it all the same: the score saw those terms twice, a sound solve
     *  looked like a wrong factorization, and the scenario was REJECTED to a robust retry that recovered nothing, then
     *  to a CPU replay that gave OLF's own MAX_ITERATION_REACHED (rte6515 busbar sections: 57 contingencies; on
     *  network.xiidm every MAX_ITERATION_REACHED of the branch and busbar scopes). Same budget search as
     *  {@link #deviceReportsMaxIterationWhenConvergingOnTheLastAllowedStep}, on three generators sharing a remote
     *  voltage control, whose reactive power distribution equations hold such terms. */
    @Test
    void deviceReportsMaxIterationWithSharedRemoteVoltageControl() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createWithGeneratorRemoteControl();
        network.newLine().setId("L14")
                .setVoltageLevel1("vl1").setBus1("b1").setVoltageLevel2("vl4").setBus2("b4")
                .setR(1.5).setX(40).setG1(0).setB1(0).setG2(0).setB2(0).add();
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(false).setUseReactiveLimits(false)
                .setVoltageInitMode(com.powsybl.loadflow.LoadFlowParameters.VoltageInitMode.PREVIOUS_VALUES);
        com.powsybl.openloadflow.OpenLoadFlowParameters ext = com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp);
        // the solved base state on the network, so both arms' base load flow needs a single step
        new com.powsybl.loadflow.LoadFlow.Runner(new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory()))
                .run(network, new com.powsybl.loadflow.LoadFlowParameters().setDistributedSlack(false).setUseReactiveLimits(false));
        List<Contingency> contingencies = List.of(new Contingency("L14", new BranchContingency("L14")));
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        int budget = -1;                                     // the smallest budget the CPU converges under
        for (int m = 2; m <= 30 && budget < 0; m++) {
            ext.setMaxNewtonRaphsonIterations(m);
            SecurityAnalysisResult cpu = runSecurityAnalysis(network, contingencies, java.util.Collections.emptyList(), sap);
            assertEquals(com.powsybl.loadflow.LoadFlowResult.ComponentResult.Status.CONVERGED,
                    cpu.getPreContingencyResult().getStatus(), "the base must converge under budget " + m);
            if (cpu.getPostContingencyResults().get(0).getStatus() == com.powsybl.security.PostContingencyComputationStatus.CONVERGED) {
                budget = m;
            }
        }
        assertTrue(budget >= 3, "the contingency must need at least two steps, smallest converging budget " + budget);
        ext.setMaxNewtonRaphsonIterations(budget - 1);
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpuNoEngage(network, contingencies, 1e-12, lfp);
        assertEquals(1L, GpuBatchScopeCensus.snapshot().getOrDefault("DEVICE:MAX_ITERATION_REACHED", 0L),
                "the device itself must have decided it; census=" + GpuBatchScopeCensus.snapshot());
    }

    @Test
    void failedBaseLoadFlowYieldsNoPostContingencyResults() {
        // Contract check for the FAILED-base question (2026-07-17): the parent only calls
        // afterPreContingencySimulation (hence precomputeBatch) and the post-contingency loop when
        // the base LF SUCCEEDED — so a failed base must yield zero post results on the GPU
        // provider, exactly like AcSecurityAnalysis.
        Network network = IeeeCdfNetworkFactory.create14();
        List<Contingency> contingencies = network.getLineStream().limit(3)
                .map(l -> new Contingency(l.getId(), new BranchContingency(l.getId()))).toList();
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters();
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp).setMaxNewtonRaphsonIterations(1);
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            SecurityAnalysisResult r = runSecurityAnalysis(network, contingencies, java.util.Collections.emptyList(), sap);
            assertTrue(r.getPostContingencyResults().isEmpty(), "failed base must yield NO post-contingency results");
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    @Test
    void gpuBatchedFallsBackToStandardWhenNoGpu() {
        Network network = IeeeCdfNetworkFactory.create14();
        List<Contingency> contingencies = network.getLineStream()
                .limit(5)
                .map(line -> new Contingency(line.getId(), new BranchContingency(line.getId())))
                .toList();

        SecurityAnalysisResult standard = runSecurityAnalysis(network, contingencies);

        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        SecurityAnalysisResult batched;
        try {
            batched = runSecurityAnalysis(network, contingencies);
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }

        // Pre-contingency must match.
        assertEquals(standard.getPreContingencyResult().getStatus(), batched.getPreContingencyResult().getStatus());

        // Same set of post-contingency results, with identical status and violation counts
        // (no GPU -> GpuBatchedAcSecurityAnalysis delegates to super for every contingency).
        List<PostContingencyResult> standardPost = standard.getPostContingencyResults();
        List<PostContingencyResult> batchedPost = batched.getPostContingencyResults();
        assertEquals(standardPost.size(), batchedPost.size(), "same number of post-contingency results");
        for (int i = 0; i < standardPost.size(); i++) {
            PostContingencyResult s = standardPost.get(i);
            PostContingencyResult b = batchedPost.get(i);
            assertEquals(s.getContingency().getId(), b.getContingency().getId());
            assertEquals(s.getStatus(), b.getStatus(), "status must match for " + s.getContingency().getId());
            assertEquals(s.getLimitViolationsResult().getLimitViolations().size(),
                    b.getLimitViolationsResult().getLimitViolations().size(),
                    "violation count must match for " + s.getContingency().getId());
        }
    }

    /**
     * On a device-equipped machine the SAME run must ENGAGE the GPU — the security-analysis contingency
     * branches are flagged disconnection-allowed, which the extractor now accepts (it used to reject them,
     * forcing a silent CPU fallback) — and still produce results identical to the CPU {@link AcSecurityAnalysis}.
     * Proves the production entry point ({@link GpuSecurityAnalysisProvider}) reaches the device on a real N-1 set, not
     * just the unit-tested batched solver. Skipped unless the native lib is available.
     */
    @Test
    void gpuBatchedEngagesAndMatchesCpuOnDevice() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU native lib not available (build it with native/build-gpu.sh + cuDSS)");

        Network network = IeeeCdfNetworkFactory.create14();
        List<Contingency> contingencies = network.getLineStream()
                .limit(8)
                .map(line -> new Contingency(line.getId(), new BranchContingency(line.getId())))
                .toList();

        SecurityAnalysisResult standard = runSecurityAnalysis(network, contingencies);

        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        // start from a cold batch context so this run exercises the base->batch structure hand-off
        GpuBatchContextCache.clearContextCache();
        long structureHits0 = GpuAcStructures.hitCount();
        SecurityAnalysisResult batched;
        try {
            batched = runSecurityAnalysis(network, contingencies);
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }

        // The GPU actually ran: the batch precomputed at least one contingency on the device (not a silent
        // fallback). Before the disconnection-allowed guard was relaxed this was always a fallback (count -1).
        int precomputed = GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get();
        assertTrue(precomputed > 0, "expected the GPU batch to precompute at least one contingency on the device "
                + "(count = " + precomputed + " — negative/zero means it fell back to the CPU)");

        // The base->batch DEVICE hand-off engaged: the base load flow (base-on-GPU swap) built the shared
        // native structure from the cached extraction, and the batch (its chunk sizing and context build)
        // then REUSED it instead of rebuilding the pattern/families on device (GpuAcStructures cache hit).
        assertTrue(GpuAcStructures.hitCount() > structureHits0,
                "expected the batched SA to reuse the base load flow's shared GPU structure");

        // ... and the GPU-engaged results are identical to the CPU security analysis (the safety contract).
        assertEquals(standard.getPreContingencyResult().getStatus(), batched.getPreContingencyResult().getStatus());
        List<PostContingencyResult> standardPost = standard.getPostContingencyResults();
        List<PostContingencyResult> batchedPost = batched.getPostContingencyResults();
        assertEquals(standardPost.size(), batchedPost.size(), "same number of post-contingency results");
        for (int i = 0; i < standardPost.size(); i++) {
            PostContingencyResult s = standardPost.get(i);
            PostContingencyResult b = batchedPost.get(i);
            assertEquals(s.getContingency().getId(), b.getContingency().getId());
            assertEquals(s.getStatus(), b.getStatus(), "status must match for " + s.getContingency().getId());
            assertEquals(s.getLimitViolationsResult().getLimitViolations().size(),
                    b.getLimitViolationsResult().getLimitViolations().size(),
                    "violation count must match for " + s.getContingency().getId());
        }
    }

    /** A security analysis prices its chunk against device memory WITHOUT the batched context the JVM's previous
     *  analysis left behind, which it destroys anyway. Priced while that context still held its LU factors, the
     *  second analysis of a JVM on rte6515's 1000 branch contingencies took 3 chunks of 334 where the first took 2 of
     *  500. The pricing stand-in records what the device would see: whether a batched context is allocated. */
    @Test
    void chunkIsPricedWithoutThePreviousAnalysisContext() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        List<Contingency> contingencies = network.getLineStream()
                .limit(8)
                .map(line -> new Contingency(line.getId(), new BranchContingency(line.getId())))
                .toList();
        List<Boolean> contextAllocatedAtPricing = new java.util.ArrayList<>();
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        GpuBatchContextCache.clearContextCache();
        GpuBatchContextCache.priceChunksForTests(() -> {
            contextAllocatedAtPricing.add(GpuBatchContextCache.holdsAContextForTests());
            return 3;
        });
        try {
            runSecurityAnalysis(network, contingencies);
            assertTrue(GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get() > 0, "the first analysis ran on the device");
            // another extraction for the second analysis, so its chunk is priced again
            var line = network.getLine("L1-2-1");
            line.setR(line.getR() * 1.01);
            runSecurityAnalysis(network, contingencies);
            assertTrue(GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get() > 0, "the second analysis ran on the device");
        } finally {
            GpuBatchContextCache.priceChunksForTests(null);
            GpuBatchContextCache.clearContextCache();
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
        assertEquals(List.of(false, false), contextAllocatedAtPricing,
                "each analysis prices its chunk once, with no batched context allocated");
    }

    /** The batched contingency solves run with the GPU solver parameters of the security analysis's own load flow
     *  parameters, not the configuration module's nor a previous run's. */
    @Test
    void batchedSolvesUseTheRunsGpuParameters() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters();
        GpuLoadFlowParameters.create(lfp)
                .setReorderingAlgorithm(GpuReorderingAlgorithm.NESTED_DISSECTION)
                .setBatchedLuOrdering(GpuBatchedLuOrdering.MINIMUM_DEGREE)
                .setBatchedLuSplitLongRows(false)
                .setMatchingAlgorithm(GpuMatchingAlgorithm.MAX_DIAG_SUM)
                .setPivotType(GpuPivotType.GLOBAL_ROW)
                .setPivotThreshold(0.25)
                .setIterativeRefinementBaseSteps(2)
                .setIterativeRefinementMaxSteps(5)
                .setIterativeRefinementTolerance(1e-11);
        List<Contingency> ctgs = List.of(new Contingency("L1-2-1", new BranchContingency("L1-2-1")));
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            runSecurityAnalysis(IeeeCdfNetworkFactory.create14(), ctgs, List.of(), lfp);
            assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "solved on the device");
            assertArrayEquals(new double[] {4, 4, 3, 0.25, 2, 5, 1e-11, 1, 0}, NewtonSettingsNative.readLinearSolverSettings());
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
            new GpuLoadFlowParameters().install();
        }
    }
}
