/*
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa;

import com.powsybl.contingency.BranchContingency;
import com.powsybl.contingency.Contingency;
import com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory;
import com.powsybl.iidm.network.Network;
import com.powsybl.openloadflow.CommonTestConfig;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcStructures;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuTestPaths;
import com.powsybl.openloadflow.ac.solver.GpuBatchedSecurityAnalysisSolver;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.security.SecurityAnalysisResult;
import com.powsybl.security.results.PostContingencyResult;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The GPU-batched AC security analysis ({@link GpuBatchedAcSecurityAnalysis}), selected via
 * the {@link GpuSecurityAnalysisProvider}, must produce IDENTICAL results to the
 * standard {@link AcSecurityAnalysis} when no GPU is present (every contingency falls back
 * to the CPU path). This is the integration's safety contract — enabling the flag never
 * changes results, it only accelerates what the device can solve. CPU-only, runs in the
 * normal suite; the actual GPU acceleration is exercised on a device-equipped machine.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuBatchedAcSecurityAnalysisTest extends AbstractOpenSecurityAnalysisTest {

    static {
        GpuTestPaths.init();
    }

    GpuBatchedAcSecurityAnalysisTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    @Test
    void sequentialFallThroughOnGpuSolverWithReactiveLimits() {
        // Regression guard around the historical CME (#15, root-caused since: cleaner-thread
        // listener removal, fixed). NB the 2-branch contingency here is BATCHED (multi-branch IS
        // in batch scope) — the test's value is the GPU-provider-vs-CPU status parity under
        // reactive limits, not sequential-path coverage.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        List<String> lines = network.getLineStream().limit(2).map(l -> l.getId()).toList();
        List<Contingency> contingencies = List.of(
                new Contingency("multi", new BranchContingency(lines.get(0)), new BranchContingency(lines.get(1))),
                new Contingency(lines.get(0), new BranchContingency(lines.get(0))));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setUseReactiveLimits(true).setDistributedSlack(true);
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, contingencies, java.util.Collections.emptyList(), sap);
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            SecurityAnalysisResult gpu = runSecurityAnalysis(network, contingencies, java.util.Collections.emptyList(), sap);
            assertEquals(cpu.getPostContingencyResults().size(), gpu.getPostContingencyResults().size());
            for (int i = 0; i < cpu.getPostContingencyResults().size(); i++) {
                assertEquals(cpu.getPostContingencyResults().get(i).getStatus(),
                        gpu.getPostContingencyResults().get(i).getStatus(),
                        "status mismatch for " + cpu.getPostContingencyResults().get(i).getContingency().getId());
            }
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    @Test
    void transformerVoltageControlBatchMatchesCpu() {
        // #16 acceptance: transformer voltage control (AFTER_GENERATOR mode) now BATCHES via the
        // per-scenario orchestrator — the guard must ENGAGE the batch (not fall back) and the
        // results must match the stock CPU SA exactly.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
        com.powsybl.iidm.network.TwoWindingsTransformer t2wt = network.getTwoWindingsTransformer("T2wT");
        t2wt.getRatioTapChanger()
                .setTargetDeadband(0)
                .setRegulating(true)
                .setTapPosition(0)
                .setRegulationTerminal(t2wt.getTerminal2())
                .setTargetV(34.0);
        // The factory network is RADIAL — any outage fragments it and is rightly rejected by the
        // batch validator. Add a parallel circuit to the first line and outage the CLONE: the grid
        // stays connected, flows shift, and the transformer control has real work per scenario.
        com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
        network.newLine()
                .setId(l0.getId() + "_BIS")
                .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId())
                .setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId())
                .setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l0.getR() * 2).setX(l0.getX() * 2)
                .setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2())
                .add();
        List<Contingency> contingencies = List.of(new Contingency(l0.getId() + "_BIS", new BranchContingency(l0.getId() + "_BIS")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setTransformerVoltageControlOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL);
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, contingencies, java.util.Collections.emptyList(), sap);
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            SecurityAnalysisResult gpu = runSecurityAnalysis(network, contingencies, java.util.Collections.emptyList(), sap);
            assertTrue(GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get() > 0,
                    "the transformer-control run must ENGAGE the batch (guard fell back -> the port is not exercised)");
            assertEquals(cpu.getPostContingencyResults().size(), gpu.getPostContingencyResults().size());
            for (int i = 0; i < cpu.getPostContingencyResults().size(); i++) {
                assertEquals(cpu.getPostContingencyResults().get(i).getStatus(),
                        gpu.getPostContingencyResults().get(i).getStatus());
                assertEquals(cpu.getPostContingencyResults().get(i).getLimitViolationsResult().getLimitViolations().size(),
                        gpu.getPostContingencyResults().get(i).getLimitViolationsResult().getLimitViolations().size(),
                        "violation count must match for " + cpu.getPostContingencyResults().get(i).getContingency().getId());
            }
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    private static Network distributedSlackNetworkWithExtras() {
        Network network = com.powsybl.openloadflow.network.DistributedSlackNetworkFactory.create();
        // Second generator on b2 (same-bus multi-gen loss) — participating like g2.
        com.powsybl.iidm.network.Generator g2b = network.getVoltageLevel(network.getGenerator("g2").getTerminal().getVoltageLevel().getId())
                .newGenerator().setId("g2b").setBus("b2").setConnectableBus("b2")
                .setMinP(0).setMaxP(100).setTargetP(30).setTargetQ(50).setVoltageRegulatorOn(false).add();
        g2b.newExtension(com.powsybl.iidm.network.extensions.ActivePowerControlAdder.class)
                .withParticipate(true).withDroop(2).add();
        com.powsybl.iidm.network.Generator g2c = network.getVoltageLevel(network.getGenerator("g2").getTerminal().getVoltageLevel().getId())
                .newGenerator().setId("g2c").setBus("b2").setConnectableBus("b2")
                .setMinP(0).setMaxP(50).setTargetP(20).setTargetQ(30).setVoltageRegulatorOn(false).add();
        g2c.newExtension(com.powsybl.iidm.network.extensions.ActivePowerControlAdder.class)
                .withParticipate(true).withDroop(2).add();
        // Parallel circuit to l24 so a branch outage keeps the star grid connected (batch validator).
        com.powsybl.iidm.network.Line l24 = network.getLine("l24");
        network.newLine().setId("l24_bis")
                .setVoltageLevel1(l24.getTerminal1().getVoltageLevel().getId()).setBus1(l24.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l24.getTerminal2().getVoltageLevel().getId()).setBus2(l24.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l24.getR() * 2).setX(l24.getX() * 2).setG1(0).setB1(0).setG2(0).setB2(0).add();
        return network;
    }

    private void assertGpuMatchesCpu(Network network, List<Contingency> contingencies) {
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, contingencies, java.util.Collections.emptyList(), sap);
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            SecurityAnalysisResult gpu = runSecurityAnalysis(network, contingencies, java.util.Collections.emptyList(), sap);
            assertTrue(GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get() > 0,
                    "the batch must ENGAGE (fallen-back run proves nothing)");
            assertEquals(cpu.getPostContingencyResults().size(), gpu.getPostContingencyResults().size());
            for (int i = 0; i < cpu.getPostContingencyResults().size(); i++) {
                assertEquals(cpu.getPostContingencyResults().get(i).getStatus(),
                        gpu.getPostContingencyResults().get(i).getStatus());
                assertEquals(cpu.getPostContingencyResults().get(i).getLimitViolationsResult().getLimitViolations().size(),
                        gpu.getPostContingencyResults().get(i).getLimitViolationsResult().getLimitViolations().size(),
                        "violation count for " + cpu.getPostContingencyResults().get(i).getContingency().getId());
            }
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    @Test
    void multiGeneratorOutageBatchMatchesCpu() {
        // #20 stage 1: one contingency losing TWO generators on the same bus (one DS-participant
        // bus -> in batch scope), plus a plain single-gen one as control.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = distributedSlackNetworkWithExtras();
        assertGpuMatchesCpu(network, List.of(
                new Contingency("g2b+g2c", new com.powsybl.contingency.GeneratorContingency("g2b"), new com.powsybl.contingency.GeneratorContingency("g2c")),
                new Contingency("g3", new com.powsybl.contingency.GeneratorContingency("g3"))));
    }

    @Test
    void branchPlusGeneratorComboBatchMatchesCpu() {
        // #20 stage 2: one contingency losing a BRANCH and a GENERATOR together.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = distributedSlackNetworkWithExtras();
        assertGpuMatchesCpu(network, List.of(
                new Contingency("l24bis+g3", new BranchContingency("l24_bis"), new com.powsybl.contingency.GeneratorContingency("g3"))));
    }

    @Test
    void failedBaseLoadFlowYieldsNoPostContingencyResults() {
        // Contract check for the FAILED-base question (2026-07-17): the parent only calls
        // afterPreContingencySimulation (hence precomputeBatch) and the post-contingency loop when
        // the base LF SUCCEEDED — so a failed base must yield zero post results on the GPU
        // provider, exactly like AcSecurityAnalysis. CPU base forced (property) so the failure is
        // deterministic on device machines too.
        Network network = IeeeCdfNetworkFactory.create14();
        List<Contingency> contingencies = network.getLineStream().limit(3)
                .map(l -> new Contingency(l.getId(), new BranchContingency(l.getId()))).toList();
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters();
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp).setMaxNewtonRaphsonIterations(1);
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        String old = System.getProperty(GpuBatchedAcSecurityAnalysis.GPU_BASE_LOADFLOW_PROPERTY);
        System.setProperty(GpuBatchedAcSecurityAnalysis.GPU_BASE_LOADFLOW_PROPERTY, "false");
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            SecurityAnalysisResult r = runSecurityAnalysis(network, contingencies, java.util.Collections.emptyList(), sap);
            assertTrue(r.getPostContingencyResults().isEmpty(), "failed base must yield NO post-contingency results");
        } finally {
            if (old == null) {
                System.clearProperty(GpuBatchedAcSecurityAnalysis.GPU_BASE_LOADFLOW_PROPERTY);
            } else {
                System.setProperty(GpuBatchedAcSecurityAnalysis.GPU_BASE_LOADFLOW_PROPERTY, old);
            }
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
        GpuBatchedSecurityAnalysisSolver.clearContextCache();
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
}
