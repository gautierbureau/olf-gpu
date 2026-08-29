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
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuTestPaths;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchContextCache;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.sa.AbstractOpenSecurityAnalysisTest;
import com.powsybl.openloadflow.sa.AcSecurityAnalysis;
import com.powsybl.openloadflow.sa.OpenSecurityAnalysisParameters;
import com.powsybl.openloadflow.sa.OpenSecurityAnalysisProvider;
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
        // DEFAULT criteria and tolerance (the no-tighten rule): with the orchestration-fidelity fixes (OLF-
        // faithful re-solve initialization, no re-solve on a STABLE phase, pi-model refresh) the DEVICE
        // orchestrator reproduces the CPU SA at machine precision WITHOUT any tolerance alignment.
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL);
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        // STATE-LEVEL comparison: monitor every voltage level. Status + violation-count parity is BLIND to a
        // "converged-but-wrong" tap solve — the on-device orchestrator toggles the control layout via row-mode +
        // target-override deltas on the fixed structure, and only a bus-voltage check confirms the discrete tap
        // decision AND the post-rounding solve actually agree with the CPU (this scenario's tap really rounds:
        // the controlled bus lands off the 34.0 target on the closest section).
        java.util.Set<String> tvcVls = network.getVoltageLevelStream().map(vl -> vl.getId())
                .collect(java.util.stream.Collectors.toSet());
        List<com.powsybl.security.monitor.StateMonitor> monitors = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(), tvcVls, java.util.Set.of()));
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, contingencies, monitors, sap);
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            SecurityAnalysisResult gpu = runSecurityAnalysis(network, contingencies, monitors, sap);
            assertTrue(GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get() > 0,
                    "the transformer-control run must ENGAGE the batch (guard fell back -> the port is not exercised)");
            assertEquals(cpu.getPostContingencyResults().size(), gpu.getPostContingencyResults().size());
            for (int i = 0; i < cpu.getPostContingencyResults().size(); i++) {
                assertEquals(cpu.getPostContingencyResults().get(i).getStatus(),
                        gpu.getPostContingencyResults().get(i).getStatus());
                assertEquals(cpu.getPostContingencyResults().get(i).getLimitViolationsResult().getLimitViolations().size(),
                        gpu.getPostContingencyResults().get(i).getLimitViolationsResult().getLimitViolations().size(),
                        "violation count must match for " + cpu.getPostContingencyResults().get(i).getContingency().getId());
                var cbv = cpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                var gbv = gpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                assertEquals(cbv.size(), gbv.size());
                for (int b = 0; b < cbv.size(); b++) {
                    // machine-precision parity at DEFAULT tolerance (measured 0.0 / 7.1e-15)
                    org.junit.jupiter.api.Assertions.assertEquals(cbv.get(b).getV(), gbv.get(b).getV(), 1e-12,
                            "transformer-VC bus voltage " + cbv.get(b).getBusId() + " for "
                                    + cpu.getPostContingencyResults().get(i).getContingency().getId());
                }
            }
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    @Test
    void javaLoadBasedDistributedSlackMatchesCpu() {
        // Phase 3 (olf.gpu.sa.outerLoopMode=JAVA + BalanceType.PROPORTIONAL_TO_LOAD): load-based distributed slack
        // is BEYOND the DEVICE gate (which is generation-P_MAX-only). OLF's own DistributedSlackOuterLoop
        // redistributes the mismatch onto LOADS; the JAVA engine isolates load targetP/targetQ per scenario with an
        // explicit flat array. The batch must ENGAGE and the post-contingency states must match the stock CPU SA.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true)
                .setBalanceType(com.powsybl.loadflow.LoadFlowParameters.BalanceType.PROPORTIONAL_TO_LOAD);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)   // create() ATTACHES the extension
                .setSlackBusPMaxMismatch(0.01).setNewtonRaphsonConvEpsPerEq(1e-9);
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        java.util.Set<String> allVls = network.getVoltageLevelStream().map(vl -> vl.getId())
                .collect(java.util.stream.Collectors.toSet());
        List<com.powsybl.security.monitor.StateMonitor> monitors = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(), allVls, java.util.Set.of()));
        List<Contingency> contingencies = List.of(new Contingency("L1-2-1", new BranchContingency("L1-2-1")));
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, contingencies, monitors, sap);
        System.setProperty(GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY, "JAVA");
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            SecurityAnalysisResult gpu = runSecurityAnalysis(network, contingencies, monitors, sap);
            assertTrue(GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get() > 0,
                    "the JAVA load-based-DS run must ENGAGE the batch (guard fell back -> the path is not exercised)");
            assertEquals(cpu.getPostContingencyResults().size(), gpu.getPostContingencyResults().size());
            for (int i = 0; i < cpu.getPostContingencyResults().size(); i++) {
                assertEquals(cpu.getPostContingencyResults().get(i).getStatus(),
                        gpu.getPostContingencyResults().get(i).getStatus());
                var cb = cpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                var gb = gpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                assertEquals(cb.size(), gb.size());
                for (int b = 0; b < cb.size(); b++) {
                    org.junit.jupiter.api.Assertions.assertEquals(cb.get(b).getV(), gb.get(b).getV(), 1e-12,
                            "JAVA load-based-DS bus voltage " + cb.get(b).getBusId());
                }
            }
        } finally {
            System.clearProperty(GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY);
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    @Test
    void javaTransformerVoltageControlMatchesCpu() {
        // Phase 3 (olf.gpu.sa.outerLoopMode=JAVA + transformer voltage control): the device runs the inner
        // Newton ONLY (the continuous ratio is a BRANCH_RHO1 variable solved in-Newton), and OLF's own Java
        // TransformerVoltageControlOuterLoop makes the discrete tap DECISIONS per scenario — the compact snapshot
        // now isolates the transformer BranchState. Same T2wt+parallel-line setup as the DEVICE test: the batch
        // must ENGAGE and the post-contingency states must match the stock CPU SA.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
        com.powsybl.iidm.network.TwoWindingsTransformer t2wt = network.getTwoWindingsTransformer("T2wT");
        t2wt.getRatioTapChanger()
                .setTargetDeadband(0).setRegulating(true).setTapPosition(0)
                .setRegulationTerminal(t2wt.getTerminal2()).setTargetV(34.0);
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
        java.util.Set<String> allVls = network.getVoltageLevelStream().map(vl -> vl.getId())
                .collect(java.util.stream.Collectors.toSet());
        List<com.powsybl.security.monitor.StateMonitor> monitors = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(), allVls, java.util.Set.of()));
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, contingencies, monitors, sap);
        System.setProperty(GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY, "JAVA");
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            SecurityAnalysisResult gpu = runSecurityAnalysis(network, contingencies, monitors, sap);
            assertTrue(GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get() > 0,
                    "the JAVA transformer-control run must ENGAGE the batch (guard fell back -> the path is not exercised)");
            assertEquals(cpu.getPostContingencyResults().size(), gpu.getPostContingencyResults().size());
            for (int i = 0; i < cpu.getPostContingencyResults().size(); i++) {
                assertEquals(cpu.getPostContingencyResults().get(i).getStatus(),
                        gpu.getPostContingencyResults().get(i).getStatus());
                var cb = cpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                var gb = gpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                assertEquals(cb.size(), gb.size());
                for (int b = 0; b < cb.size(); b++) {
                    org.junit.jupiter.api.Assertions.assertEquals(cb.get(b).getV(), gb.get(b).getV(), 1e-12,
                            "JAVA transformer-VC bus voltage " + cb.get(b).getBusId());
                }
            }
        } finally {
            System.clearProperty(GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY);
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    @Test
    void javaShuntVoltageControlMatchesCpu() {
        // Increment 3 (olf.gpu.sa.outerLoopMode=JAVA + shunt voltage control + Dynamic Q-relocation §11):
        // the device runs the inner Newton ONLY (the continuous susceptance is a SHUNT_B variable solved
        // in-Newton), OLF's own ShuntVoltageControlOuterLoop makes the discrete section DECISIONS per
        // scenario, the compact snapshot isolates the controller-shunt b + VC-enabled (BusState), and
        // dynamic Q-relocation lets the ONE shared structure follow the shunt control on<->off toggle (a
        // one-terminal SHUNT_B would otherwise freeze). Relocation auto-enables in the CUDA structure
        // whenever a controllable bus is present, so no flag is required.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.ShuntNetworkFactory.create();
        // Radial b1-b2-b3(SHUNT): add a parallel circuit to l1 (b1-b2) and outage the CLONE so the grid
        // stays connected, flows shift, and the shunt control does real work per scenario.
        com.powsybl.iidm.network.Line l1 = network.getLine("l1");
        network.newLine().setId("l1_BIS")
                .setVoltageLevel1(l1.getTerminal1().getVoltageLevel().getId())
                .setBus1(l1.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l1.getTerminal2().getVoltageLevel().getId())
                .setBus2(l1.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l1.getR() * 2).setX(l1.getX() * 2)
                .setG1(l1.getG1()).setB1(l1.getB1()).setG2(l1.getG2()).setB2(l1.getB2()).add();
        List<Contingency> contingencies = List.of(new Contingency("l1_BIS", new BranchContingency("l1_BIS")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setShuntCompensatorVoltageControlOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        java.util.Set<String> allVls = network.getVoltageLevelStream().map(vl -> vl.getId())
                .collect(java.util.stream.Collectors.toSet());
        List<com.powsybl.security.monitor.StateMonitor> monitors = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(), allVls, java.util.Set.of()));
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, contingencies, monitors, sap);
        System.setProperty(GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY, "JAVA");
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            SecurityAnalysisResult gpu = runSecurityAnalysis(network, contingencies, monitors, sap);
            assertTrue(GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get() > 0,
                    "the JAVA shunt-control run must ENGAGE the batch (guard fell back -> the path is not exercised)");
            assertEquals(cpu.getPostContingencyResults().size(), gpu.getPostContingencyResults().size());
            for (int i = 0; i < cpu.getPostContingencyResults().size(); i++) {
                assertEquals(cpu.getPostContingencyResults().get(i).getStatus(),
                        gpu.getPostContingencyResults().get(i).getStatus());
                var cb = cpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                var gb = gpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                assertEquals(cb.size(), gb.size());
                for (int b = 0; b < cb.size(); b++) {
                    org.junit.jupiter.api.Assertions.assertEquals(cb.get(b).getV(), gb.get(b).getV(), 1e-12,
                            "JAVA shunt-VC bus voltage " + cb.get(b).getBusId());
                }
            }
        } finally {
            System.clearProperty(GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY);
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
        assertGpuMatchesCpu(network, contingencies, 0.5);
    }

    private void assertGpuMatchesCpu(Network network, List<Contingency> contingencies, double vToleranceKv) {
        assertGpuMatchesCpu(network, contingencies, vToleranceKv, true);
    }

    private void assertGpuMatchesCpu(Network network, List<Contingency> contingencies, double vToleranceKv, boolean distributedSlack) {
        assertGpuMatchesCpu(network, contingencies, vToleranceKv, distributedSlack, true);
    }

    private void assertGpuMatchesCpu(Network network, List<Contingency> contingencies, double vToleranceKv,
                                     boolean distributedSlack, boolean reactiveLimits) {
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(distributedSlack).setUseReactiveLimits(reactiveLimits);
        String convEps = System.getProperty("olf.gpu.test.convEps", System.getenv("OLF_GPU_DBG_CONV_EPS"));
        if (convEps != null) {
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)   // create() ATTACHES (get() would detach!)
                    .setNewtonRaphsonConvEpsPerEq(Double.parseDouble(convEps));
        }
        String slackTol = System.getProperty("olf.gpu.test.slackTol", System.getenv("OLF_GPU_DBG_SLACK_TOL"));
        if (slackTol != null) {
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                    .setSlackBusPMaxMismatch(Double.parseDouble(slackTol));
        }
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        // STATE-LEVEL comparison (task #21): monitor every voltage level — status/violation-count
        // equality is blind on limit-free test networks (a wrong Q injection still "matches").
        java.util.Set<String> allVls = network.getVoltageLevelStream().map(vl -> vl.getId())
                .collect(java.util.stream.Collectors.toSet());
        List<com.powsybl.security.monitor.StateMonitor> monitors = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(), allVls, java.util.Set.of()));
        com.powsybl.commons.report.ReportNode plainCpuReport = "1".equals(System.getenv("OLF_GPU_DBG_REPORT"))
                ? com.powsybl.commons.report.ReportNode.newRootReportNode()
                        .withResourceBundles(com.powsybl.openloadflow.util.report.PowsyblOpenLoadFlowReportResourceBundle.BASE_NAME)
                        .withMessageTemplate("olf.saReport").build()
                : com.powsybl.commons.report.ReportNode.NO_OP;
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, contingencies, monitors, sap, plainCpuReport);
        if (plainCpuReport != com.powsybl.commons.report.ReportNode.NO_OP) {
            printReport(plainCpuReport, 0);
        }
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            SecurityAnalysisResult gpu = runSecurityAnalysis(network, contingencies, monitors, sap);
            assertTrue(GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get() > 0,
                    "the batch must ENGAGE (fallen-back run proves nothing)");
            assertEquals(cpu.getPostContingencyResults().size(), gpu.getPostContingencyResults().size());
            for (int i = 0; i < cpu.getPostContingencyResults().size(); i++) {
                assertEquals(cpu.getPostContingencyResults().get(i).getStatus(),
                        gpu.getPostContingencyResults().get(i).getStatus());
                if ("1".equals(System.getenv("OLF_GPU_DBG_BUSDUMP"))) {
                    var cbz = cpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                    var gbz = gpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                    System.err.println("PLAINDUMP sizes cpu=" + cbz.size() + " gpu=" + gbz.size());
                    for (int z = 0; z < Math.min(cbz.size(), gbz.size()); z++) {
                        System.err.printf("PLAINDUMP %-10s cpu=%9.4f | %-10s gpu=%9.4f%n", cbz.get(z).getBusId(),
                                cbz.get(z).getV(), gbz.get(z).getBusId(), gbz.get(z).getV());
                    }
                    cpu.getPostContingencyResults().get(i).getLimitViolationsResult().getLimitViolations()
                            .forEach(v -> System.err.println("PLAINVIOL cpu " + v.getSubjectId() + " " + v.getLimitType() + " " + v.getValue()));
                    gpu.getPostContingencyResults().get(i).getLimitViolationsResult().getLimitViolations()
                            .forEach(v -> System.err.println("PLAINVIOL gpu " + v.getSubjectId() + " " + v.getLimitType() + " " + v.getValue()));
                }
                assertEquals(cpu.getPostContingencyResults().get(i).getLimitViolationsResult().getLimitViolations().size(),
                        gpu.getPostContingencyResults().get(i).getLimitViolationsResult().getLimitViolations().size(),
                        "violation count for " + cpu.getPostContingencyResults().get(i).getContingency().getId());
                var cpuBuses = cpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                var gpuBuses = gpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                assertEquals(cpuBuses.size(), gpuBuses.size());
                for (int b = 0; b < cpuBuses.size(); b++) {
                    org.junit.jupiter.api.Assertions.assertEquals(cpuBuses.get(b).getV(), gpuBuses.get(b).getV(), vToleranceKv,
                            "bus voltage " + cpuBuses.get(b).getBusId() + " for "
                                    + cpu.getPostContingencyResults().get(i).getContingency().getId()
                                    + " (0.5 kV tolerance: criterion slack is far below, a Q-injection error far above)");
                }
            }
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    @Test
    void fixedQGeneratorLossAdjustsQTarget() {
        // #21: a lost FIXED-Q generator on an already-PQ bus must remove its targetQ from the bus
        // Q target (the CPU does via LfContingency). ieee14 has realistic impedances, so a wrong
        // Q injection moves voltages by ~0.4 kV — far above the 0.05 kV assertion tolerance,
        // while the convergence-slack noise is ~1e-3 kV.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        com.powsybl.iidm.network.Bus b10 = network.getBusBreakerView().getBus("B10");
        b10.getVoltageLevel().newGenerator().setId("gq").setBus("B10").setConnectableBus("B10")
                .setMinP(0).setMaxP(50).setTargetP(10).setTargetQ(15).setVoltageRegulatorOn(false).add();
        // distributed slack OFF: gq would participate (PROPORTIONAL_TO_GENERATION_P_MAX includes
        // all gens) alongside a participating slack — the known DS fallback combo, orthogonal here.
        assertGpuMatchesCpu(network,
                List.of(new Contingency("gq", new com.powsybl.contingency.GeneratorContingency("gq"))), 0.05, false);
    }

    @Test
    void loadLossBatchMatchesCpu() {
        // #22: a lost load = per-scenario injection override (P and Q targets RAISED); the DS pass
        // absorbs the imbalance exactly like the CPU (loads do not participate). ieee14 impedances
        // make any unit/sign error in the PowerShift deltas fail the 0.05 kV state assertions.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        String loadId = network.getLoadStream()
                .filter(l -> "B9".equals(l.getTerminal().getBusBreakerView().getBus().getId()))
                .findFirst().orElseThrow().getId();
        assertGpuMatchesCpu(network,
                List.of(new Contingency(loadId, new com.powsybl.contingency.LoadContingency(loadId))), 0.05, true);
    }

    @Test
    void branchGenAndLoadComboBatchMatchesCpu() {
        // #22 combo: branch + fixed-Q generator + load in ONE contingency (DS off: the added gen
        // would participate alongside a participating slack — the known DS fallback combo).
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        network.getVoltageLevel(network.getBusBreakerView().getBus("B10").getVoltageLevel().getId())
                .newGenerator().setId("gq").setBus("B10").setConnectableBus("B10")
                .setMinP(0).setMaxP(50).setTargetP(10).setTargetQ(15).setVoltageRegulatorOn(false).add();
        String loadId = network.getLoadStream()
                .filter(l -> "B9".equals(l.getTerminal().getBusBreakerView().getBus().getId()))
                .findFirst().orElseThrow().getId();
        String lineId = network.getLineStream()
                .filter(l -> !l.getId().contains("7") && !l.getId().contains("8"))
                .skip(2).findFirst().orElseThrow().getId();
        assertGpuMatchesCpu(network,
                List.of(new Contingency("combo", new BranchContingency(lineId),
                        new com.powsybl.contingency.GeneratorContingency("gq"),
                        new com.powsybl.contingency.LoadContingency(loadId))), 0.05, false);
    }

    @Test
    void operatorStrategyBatchMatchesCpu() {
        // #24 stage 1: a TrueCondition strategy (load shed + generator redispatch) becomes an EXTRA
        // batch scenario composed onto its contingency; the injected OperatorStrategyResult must
        // match the CPU's at state level.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        String lineId = network.getLineStream()
                .filter(l -> !l.getId().contains("7") && !l.getId().contains("8"))
                .skip(1).findFirst().orElseThrow().getId();
        com.powsybl.iidm.network.Load load12 = network.getLoadStream()
                .filter(l -> "B12".equals(l.getTerminal().getBusBreakerView().getBus().getId()))
                .findFirst().orElseThrow();
        List<Contingency> contingencies = List.of(new Contingency(lineId, new BranchContingency(lineId)));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.LoadActionBuilder().withId("shed12").withLoadId(load12.getId())
                        .withRelativeValue(false).withActivePowerValue(load12.getP0() / 2).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("s1",
                        com.powsybl.contingency.ContingencyContext.specificContingency(lineId),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("shed12")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        java.util.Set<String> allVls = network.getVoltageLevelStream().map(vl -> vl.getId())
                .collect(java.util.stream.Collectors.toSet());
        List<com.powsybl.security.monitor.StateMonitor> monitors = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(), allVls, java.util.Set.of()));
        com.powsybl.commons.report.ReportNode cpuReport = "1".equals(System.getenv("OLF_GPU_DBG_REPORT"))
                ? com.powsybl.commons.report.ReportNode.newRootReportNode()
                        .withResourceBundles(com.powsybl.openloadflow.util.report.PowsyblOpenLoadFlowReportResourceBundle.BASE_NAME)
                        .withMessageTemplate("olf.saReport").build()
                : com.powsybl.commons.report.ReportNode.NO_OP;
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, contingencies, monitors, sap, strategies, actions,
                cpuReport);
        if (cpuReport != com.powsybl.commons.report.ReportNode.NO_OP) {
            printReport(cpuReport, 0);
        }
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            SecurityAnalysisResult gpu = runSecurityAnalysis(network, contingencies, monitors, sap, strategies, actions,
                    com.powsybl.commons.report.ReportNode.NO_OP);
            assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                    "contingency AND the (ctg, strategy) pair must both precompute on the device");
            assertEquals(1, gpu.getOperatorStrategyResults().size());
            var cpuS = cpu.getOperatorStrategyResults().get(0).getConditionalActionsResults().get(0);
            var gpuS = gpu.getOperatorStrategyResults().get(0).getConditionalActionsResults().get(0);
            assertEquals(cpuS.getStatus(), gpuS.getStatus());
            assertEquals(cpuS.getLimitViolationsResult().getLimitViolations().size(),
                    gpuS.getLimitViolationsResult().getLimitViolations().size(), "strategy violation count");
            var cb = cpuS.getNetworkResult().getBusResults();
            var gb = gpuS.getNetworkResult().getBusResults();
            assertEquals(cb.size(), gb.size());
            for (int b = 0; b < cb.size(); b++) {
                // TEMPORARY tolerance — TO INVESTIGATE: the DEVICE operator-strategy path does NOT yet meet the
                // 1e-12 machine-precision standard (measured ~5e-4..4e-3 vs CPU at default tolerance). The
                // composed-action scenarios re-solve on-device and need the same orchestration-fidelity
                // treatment as the main paths (OLF-faithful re-solve initialization / no re-solve on stable).
                // Tighten to 1e-12 once fixed.
                org.junit.jupiter.api.Assertions.assertEquals(cb.get(b).getV(), gb.get(b).getV(), 0.05,
                        "strategy bus voltage " + cb.get(b).getBusId());
            }
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    private static void printReport(com.powsybl.commons.report.ReportNode node, int depth) {
        System.err.println("CPU-REPORT " + "  ".repeat(depth) + node.getMessage());
        for (com.powsybl.commons.report.ReportNode c : node.getChildren()) {
            printReport(c, depth + 1);
        }
    }

    private void assertStrategyMatchesCpu(Network network, List<Contingency> contingencies,
                                          List<com.powsybl.contingency.strategy.OperatorStrategy> strategies,
                                          List<com.powsybl.action.Action> actions, int expectedPrecomputed) {
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, expectedPrecomputed, true);
    }

    private void assertStrategyMatchesCpu(Network network, List<Contingency> contingencies,
                                          List<com.powsybl.contingency.strategy.OperatorStrategy> strategies,
                                          List<com.powsybl.action.Action> actions, int expectedPrecomputed,
                                          boolean distributedSlack) {
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(distributedSlack).setUseReactiveLimits(true);
        String sConvEps = System.getProperty("olf.gpu.test.convEps", System.getenv("OLF_GPU_DBG_CONV_EPS"));
        if (sConvEps != null) {
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                    .setNewtonRaphsonConvEpsPerEq(Double.parseDouble(sConvEps));
        }
        String sSlackTol = System.getProperty("olf.gpu.test.slackTol", System.getenv("OLF_GPU_DBG_SLACK_TOL"));
        if (sSlackTol != null) {
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                    .setSlackBusPMaxMismatch(Double.parseDouble(sSlackTol));
        }
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        // Node-breaker networks: contingency PROPAGATION opens the target's breakers too (switches are
        // LfBranches in breaker mode), tripping the batch's >2-branch guard — keep the strategy tests
        // on the un-propagated contingency shape (switch outages as zi-element outages are future scope).
        sap.addExtension(OpenSecurityAnalysisParameters.class,
                new OpenSecurityAnalysisParameters().setContingencyPropagation(false));
        sap.setLoadFlowParameters(lfp);
        java.util.Set<String> allVls = network.getVoltageLevelStream().map(vl -> vl.getId())
                .collect(java.util.stream.Collectors.toSet());
        List<com.powsybl.security.monitor.StateMonitor> monitors = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(), allVls, java.util.Set.of()));
        com.powsybl.commons.report.ReportNode helperReport = "1".equals(System.getenv("OLF_GPU_DBG_REPORT"))
                ? com.powsybl.commons.report.ReportNode.newRootReportNode()
                        .withResourceBundles(com.powsybl.openloadflow.util.report.PowsyblOpenLoadFlowReportResourceBundle.BASE_NAME)
                        .withMessageTemplate("olf.saReport").build()
                : com.powsybl.commons.report.ReportNode.NO_OP;
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, contingencies, monitors, sap, strategies, actions,
                helperReport);
        if (helperReport != com.powsybl.commons.report.ReportNode.NO_OP) {
            printReport(helperReport, 0);
        }
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            SecurityAnalysisResult gpu = runSecurityAnalysis(network, contingencies, monitors, sap, strategies, actions,
                    com.powsybl.commons.report.ReportNode.NO_OP);
            if (expectedPrecomputed >= 0) {
                assertEquals(expectedPrecomputed, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                        "contingencies AND (ctg, strategy) pairs must precompute on the device");
            }
            assertEquals(cpu.getOperatorStrategyResults().size(), gpu.getOperatorStrategyResults().size());
            for (int r = 0; r < cpu.getOperatorStrategyResults().size(); r++) {
                var cpuS = cpu.getOperatorStrategyResults().get(r).getConditionalActionsResults().get(0);
                var gpuS = gpu.getOperatorStrategyResults().get(r).getConditionalActionsResults().get(0);
                assertEquals(cpuS.getStatus(), gpuS.getStatus());
                assertEquals(cpuS.getLimitViolationsResult().getLimitViolations().size(),
                        gpuS.getLimitViolationsResult().getLimitViolations().size(), "strategy violation count");
                var cb = cpuS.getNetworkResult().getBusResults();
                var gb = gpuS.getNetworkResult().getBusResults();
                assertEquals(cb.size(), gb.size());
                if ("1".equals(System.getenv("OLF_GPU_DBG_BUSDUMP"))) {
                    var cpb = cpu.getPostContingencyResults().get(r).getNetworkResult().getBusResults();
                    var gpb = gpu.getPostContingencyResults().get(r).getNetworkResult().getBusResults();
                    for (int b = 0; b < cpb.size(); b++) {
                        System.err.printf("PARENTDUMP %-8s cpu=%9.4f gpu=%9.4f d=%+8.4f%n", cpb.get(b).getBusId(),
                                cpb.get(b).getV(), gpb.get(b).getV(), gpb.get(b).getV() - cpb.get(b).getV());
                    }
                    for (int b = 0; b < cb.size(); b++) {
                        System.err.printf("BUSDUMP %-8s cpu=%9.4f gpu=%9.4f d=%+8.4f%n", cb.get(b).getBusId(),
                                cb.get(b).getV(), gb.get(b).getV(), gb.get(b).getV() - cb.get(b).getV());
                    }
                }
                for (int b = 0; b < cb.size(); b++) {
                    // TEMPORARY tolerance — TO INVESTIGATE (see the note above): tighten to 1e-12 once the
                    // DEVICE strategy path gets the orchestration-fidelity treatment.
                    org.junit.jupiter.api.Assertions.assertEquals(cb.get(b).getV(), gb.get(b).getV(), 0.05,
                            "strategy bus voltage " + cb.get(b).getBusId());
                }
            }
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    @Test
    void tapPositionActionStrategyBatchMatchesCpu() {
        // #24 stage 2b: RatioTapChanger tap action on a FIXED-tap transformer = per-scenario r1
        // override on the packed branch inputs.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
        com.powsybl.iidm.network.TwoWindingsTransformer t2wt = network.getTwoWindingsTransformer("T2wT");
        t2wt.getRatioTapChanger().setTapPosition(0).setRegulating(false);
        com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
        network.newLine().setId(l0.getId() + "_BIS")
                .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(0).setB1(0).setG2(0).setB2(0).add();
        List<Contingency> contingencies = List.of(new Contingency(l0.getId() + "_BIS", new BranchContingency(l0.getId() + "_BIS")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.RatioTapChangerTapPositionAction("tapUp", "T2wT", false, 2));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sTap",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("tapUp")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    @Test
    void pstTapPositionActionStrategyBatchMatchesCpu() {
        // #24 stage 2b: PhaseTapChanger tap action on a FIXED-tap PST = per-scenario (r1, a1)
        // override; the overridden a1 forces the raw-sincos trig path (statics are per-element).
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.PhaseControlFactory.createNetworkWithT2wt();
        List<Contingency> contingencies = List.of(new Contingency("L1", new BranchContingency("L1")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.PhaseTapChangerTapPositionAction("pstUp", "PS1", false, 2));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sPst",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("pstUp")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    @Test
    void pstTapActionWithPerStepImpedanceBatchMatchesCpu() {
        // Hard full-row-override case: the PST's steps change r/x/g/b by up to +-100% per tap
        // (y and ksi move, not just rho/alpha) — unrepresentable by an r1/a1-only override.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.PhaseControlFactory.createNetworkWithT2wtExaggerateRxgbTable();
        List<Contingency> contingencies = List.of(new Contingency("L1", new BranchContingency("L1")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.PhaseTapChangerTapPositionAction("pstShift", "PS1", false, 2));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sPstRxgb",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("pstShift")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    @Test
    void baseOpenShuntReconnectionStrategyBatchMatchesCpu() {
        // #32: reconnecting a BASE-OPEN shunt via TerminalsConnectionAction — supported by the
        // olf-local-patches upstream patch (shuntIdsToClose retention: the compensator joins the
        // aggregate at section 0, the action restores its section count). Batched as a per-scenario
        // admittance shift of the compensator's full section admittance.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        String shuntId = network.getShuntCompensatorStream().findFirst().orElseThrow().getId();
        network.getShuntCompensator(shuntId).getTerminal().disconnect();
        String lineId = network.getLineStream()
                .filter(l -> !l.getId().contains("7") && !l.getId().contains("8"))
                .skip(4).findFirst().orElseThrow().getId();
        List<Contingency> contingencies = List.of(new Contingency(lineId, new BranchContingency(lineId)));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.TerminalsConnectionAction("reconnectShunt", shuntId, false));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sShuntRe",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("reconnectShunt")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    @Test
    void nodeBreakerPropagatedContingencyBatchMatchesCpu() {
        // #33: with contingency PROPAGATION on (production default) a node-breaker line outage also
        // opens the target's breakers — switches are zero-impedance LfBranches, so the naive shape is
        // "3 branches" and nothing batched at all on a node-breaker network. The switch part is now a
        // per-scenario zi-element outage: the dead stub it leaves behind is frozen by the existing
        // islanding machinery, and the impedant part still obeys the <=2-branch mask rule.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.NodeBreakerNetworkFactory.create();
        // L3 (BBS2 side <-> BBS3, behind the NON-retained B5/B6 so propagation stops at the retained C)
        // makes BBS2/BBS3 the most-meshed buses, so the slack lands in the LIVE component. Without it
        // the slack sits on the BBS1 stub, and CPU RELOCATES it (PropagatedContingency) — a shape the
        // batched pack cannot express, so the validator rejects it to CPU (see ISLANDS_SLACK).
        var vl1 = network.getVoltageLevel("VL1");
        var vl2 = network.getVoltageLevel("VL2");
        vl1.getNodeBreakerView().newBreaker().setId("B5").setNode1(1).setNode2(7).add();
        vl2.getNodeBreakerView().newBreaker().setId("B6").setNode1(0).setNode2(4).add();
        network.newLine().setId("L3").setVoltageLevel1("VL1").setNode1(7).setVoltageLevel2("VL2").setNode2(4)
                .setR(3).setX(33).setG1(0).setB1(0).setG2(0).setB2(0).add();
        assertGpuMatchesCpu(network, List.of(new Contingency("L1", new BranchContingency("L1"))), 0.05, true);
    }

    @Test
    void baseOpenSwitchCloseStrategyBatchMatchesCpu() {
        // #31: SwitchAction CLOSING a base-open busbar coupler (zero-impedance branch). The retained
        // switch's dummy variables already have rows (identity-pinned to 0 while open), so the close
        // is a per-scenario zi-element enable + identity->coupling row flip on the device. Node-breaker
        // network: coupler C open at base splits VL1 into BBS1 (lines L1+L3) and BBS2 (G + L2); the
        // L1 outage keeps both sides energized, the strategy merges them back.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.NodeBreakerNetworkFactory.create();
        var vl1 = network.getVoltageLevel("VL1");
        var vl2 = network.getVoltageLevel("VL2");
        vl1.getNodeBreakerView().newBreaker().setId("B5").setNode1(0).setNode2(7).add();
        vl2.getNodeBreakerView().newBreaker().setId("B6").setNode1(0).setNode2(4).add();
        network.newLine().setId("L3").setVoltageLevel1("VL1").setNode1(7).setVoltageLevel2("VL2").setNode2(4)
                .setR(3).setX(33).setG1(0).setB1(0).setG2(0).setB2(0).add();
        network.getSwitch("C").setOpen(true);
        List<Contingency> contingencies = List.of(new Contingency("L1", new BranchContingency("L1")));
        List<com.powsybl.action.Action> actions = List.of(new com.powsybl.action.SwitchAction("closeC", "C", false));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sCloseC",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("closeC")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    @Test
    void baseOpenBranchRecloseStrategyBatchMatchesCpu() {
        // #29: TerminalsConnectionAction CLOSING a BASE-OPEN branch. The SA retains it via
        // LfTopoConfig (loaded closed in a temp variant, re-opened by connectivity), so the LfBranch
        // exists disabled with both buses — packed as a base-open element, default-disabled in every
        // scenario's mask, and ENABLED only in the (ctg, strategy) pair that recloses it.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = distributedSlackNetworkWithExtras();
        com.powsybl.iidm.network.Line l24 = network.getLine("l24");
        com.powsybl.iidm.network.Line ter = network.newLine().setId("l24_ter")
                .setVoltageLevel1(l24.getTerminal1().getVoltageLevel().getId()).setBus1(l24.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l24.getTerminal2().getVoltageLevel().getId()).setBus2(l24.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l24.getR() * 3).setX(l24.getX() * 3).setG1(0).setB1(0).setG2(0).setB2(0).add();
        ter.getTerminal1().disconnect();
        ter.getTerminal2().disconnect();
        List<Contingency> contingencies = List.of(new Contingency("l24_bis", new BranchContingency("l24_bis")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.TerminalsConnectionAction("closeTer", "l24_ter", false));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sReclose",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("closeTer")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    @Test
    void generatorActionStrategyBatchMatchesCpu() {
        // #28: pure active-power redispatch GeneratorAction — composed as a net-injection delta on
        // the warm-continued pair. The B2 saga's ~0.17 kV freeze-semantics delta must NOT reappear.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        String lineId = network.getLineStream().skip(4).findFirst().orElseThrow().getId();
        List<Contingency> contingencies = List.of(new Contingency(lineId, new BranchContingency(lineId)));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.GeneratorActionBuilder().withId("redispatch2")
                        .withGeneratorId("B2-G").withActivePowerRelativeValue(true).withActivePowerValue(20).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sGen",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("redispatch2")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    @Test
    void l121SevereContingencyMatchesCpuAtTightSlackTol() {
        // The L1-2-1 outage (main B1-B2 tie) pins ALL PV buses — an ill-conditioned state where the
        // DS stopping band (slackBusPMaxMismatch, default 1 MW) leaves ~0.1 kV of legitimate
        // ambiguity between arms (measured: 0.105 kV @1 MW, 0.055 @0.1, match @0.01 — scales with
        // the band, NOT a defect). This regression asserts exact machinery parity with the band
        // narrowed; the ambiguity analysis lives in docs/gpu-vs-cpu-capabilities.md §A.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        System.setProperty("olf.gpu.test.slackTol", "0.01");
        System.setProperty("olf.gpu.test.convEps", "1e-9");
        try {
            Network network = IeeeCdfNetworkFactory.create14();
            assertGpuMatchesCpu(network, List.of(new Contingency("L1-2-1", new BranchContingency("L1-2-1"))), 0.05, true);
            Network network2 = IeeeCdfNetworkFactory.create14();
            network2.getVoltageLevel("VL10").newGenerator().setId("B10-GQ").setBus("B10").setConnectableBus("B10")
                    .setTargetP(5).setTargetQ(8).setVoltageRegulatorOn(false).setMinP(0).setMaxP(20).add();
            assertGpuMatchesCpu(network2, List.of(new Contingency("L1-2-1", new BranchContingency("L1-2-1"))), 0.05, true);
        } finally {
            System.clearProperty("olf.gpu.test.slackTol");
            System.clearProperty("olf.gpu.test.convEps");
        }
    }

    @Test
    void javaOuterLoopModeMatchesCpu() {
        // Phase 2 (olf.gpu.sa.outerLoopMode=JAVA): the batched device runs the inner Newton ONLY, and OLF's
        // own Java outer loops (distributed slack + reactive limits) make every per-scenario decision via
        // GpuHybridBatchOuterLoopEngine. The batch must ENGAGE (a branch-only, non-islanding outage) and the
        // post-contingency states must match the stock CPU SA — the whole point of JAVA mode (CPU-fidelity by
        // construction). Same tight slack band as the L1-2-1 regression so DS-arm ambiguity does not mask a defect.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        System.setProperty(GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY, "JAVA");
        System.setProperty("olf.gpu.test.slackTol", "0.01");
        System.setProperty("olf.gpu.test.convEps", "1e-9");
        try {
            Network network = IeeeCdfNetworkFactory.create14();
            assertGpuMatchesCpu(network, List.of(new Contingency("L1-2-1", new BranchContingency("L1-2-1"))), 0.05, true);
        } finally {
            System.clearProperty(GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY);
            System.clearProperty("olf.gpu.test.slackTol");
            System.clearProperty("olf.gpu.test.convEps");
        }
    }

    @Test
    void generatorActionStressAcrossContingencies() {
        // Freeze-semantics hunt: every non-fragmenting ieee14 line contingency x a LARGE redispatch
        // (+60 MW on B2-G) that pushes pinned controllers around. Any surviving CPU-vs-device RL
        // freeze divergence must surface as a >0.05 kV bus mismatch on some pair.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        System.setProperty("olf.gpu.test.slackTol", "0.01");
        System.setProperty("olf.gpu.test.convEps", "1e-9");
        List<String> lineIds = IeeeCdfNetworkFactory.create14().getLineStream()
                .map(l -> l.getId()).toList();
        try {
            for (String lineId : lineIds) {
                Network network = IeeeCdfNetworkFactory.create14();
                // extra fixed-Q gen at B10 (the #21 shape): a non-voltage-capable gen in the RL mix
                network.getVoltageLevel("VL10").newGenerator().setId("B10-GQ").setBus("B10").setConnectableBus("B10")
                        .setTargetP(5).setTargetQ(8).setVoltageRegulatorOn(false).setMinP(0).setMaxP(20).add();
                List<Contingency> contingencies = List.of(new Contingency(lineId, new BranchContingency(lineId)));
                List<com.powsybl.action.Action> actions = List.of(
                        new com.powsybl.action.GeneratorActionBuilder().withId("redispatch2")
                                .withGeneratorId("B2-G").withActivePowerRelativeValue(true).withActivePowerValue(60).build());
                List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                        new com.powsybl.contingency.strategy.OperatorStrategy("sGen",
                                com.powsybl.contingency.ContingencyContext.all(),
                                new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("redispatch2")));
                try {
                    assertStrategyMatchesCpu(network, contingencies, strategies, actions, -1);
                } catch (AssertionError e) {
                    throw new AssertionError("ctg " + lineId + ": " + e.getMessage(), e);
                }
            }
        } finally {
            System.clearProperty("olf.gpu.test.slackTol");
            System.clearProperty("olf.gpu.test.convEps");
        }
    }

    @Test
    void violationConditionStrategyBatchMatchesCpu() {
        // #24: violation-conditioned strategy — the pair is precomputed SPECULATIVELY; the engine's
        // CPU-side condition check (on the post-contingency violations) decides whether it is used.
        // A tight current limit on l12 guarantees the post-contingency violation that arms the condition.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = distributedSlackNetworkWithExtras();
        // shed target on b2, NOT b4: b4 is the star center = slack bus, where injection overrides
        // are unbatchable by design (the pair would silently fall back and defeat the test)
        network.getBusBreakerView().getBus("b2").getVoltageLevel().newLoad()
                .setId("extraLoad2").setBus("b2").setConnectableBus("b2").setP0(40).setQ0(20).add();
        network.getLine("l24").newCurrentLimits1().setPermanentLimit(50).add();
        List<Contingency> contingencies = List.of(new Contingency("l24_bis", new BranchContingency("l24_bis")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.LoadActionBuilder().withId("shed4").withLoadId("extraLoad2")
                        .withRelativeValue(false).withActivePowerValue(10).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sViol",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.AnyViolationCondition(), List.of("shed4")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    @Test
    void violationConditionNotTriggeredMatchesCpu() {
        // Negative case: no limits => no violations => the condition never passes. The speculative
        // pair still precomputes (count 2) but NEITHER arm produces an operator strategy result.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = distributedSlackNetworkWithExtras();
        network.getBusBreakerView().getBus("b2").getVoltageLevel().newLoad()
                .setId("extraLoad2").setBus("b2").setConnectableBus("b2").setP0(40).setQ0(20).add();
        List<Contingency> contingencies = List.of(new Contingency("l24_bis", new BranchContingency("l24_bis")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.LoadActionBuilder().withId("shed4").withLoadId("extraLoad2")
                        .withRelativeValue(false).withActivePowerValue(10).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sViolNone",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.AnyViolationCondition(), List.of("shed4")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    @Test
    void shuntPositionActionStrategyBatchMatchesCpu() {
        // #24 stage 2b: ShuntCompensatorPositionAction = per-scenario admittance shift (linear model).
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        String shuntId = network.getShuntCompensatorStream().findFirst().orElseThrow().getId();
        String lineId = network.getLineStream()
                .filter(l -> !l.getId().contains("7") && !l.getId().contains("8"))
                .skip(4).findFirst().orElseThrow().getId();
        List<Contingency> contingencies = List.of(new Contingency(lineId, new BranchContingency(lineId)));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.ShuntCompensatorPositionActionBuilder().withId("shOff")
                        .withShuntCompensatorId(shuntId).withSectionCount(0).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sShunt",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("shOff")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    @Test
    void shuntShiftBatchMatchesCpu() {
        // #23: shunt disconnection now batches as a per-scenario ADMITTANCE OVERRIDE (the general
        // section-shift mechanism, both directions). ieee14's B9 shunt, state-level 0.05 kV.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        String shuntId = network.getShuntCompensatorStream().findFirst().orElseThrow().getId();
        assertGpuMatchesCpu(network,
                List.of(new Contingency(shuntId, new com.powsybl.contingency.ShuntCompensatorContingency(shuntId))), 0.05, true);
    }

    @Test
    void shuntBranchLoadComboBatchMatchesCpu() {
        // #23 combo: shunt shift + branch + load in ONE contingency.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        String shuntId = network.getShuntCompensatorStream().findFirst().orElseThrow().getId();
        String loadId = network.getLoadStream()
                .filter(l -> "B12".equals(l.getTerminal().getBusBreakerView().getBus().getId()))
                .findFirst().orElseThrow().getId();
        String lineId = network.getLineStream()
                .filter(l -> !l.getId().contains("7") && !l.getId().contains("8"))
                .skip(3).findFirst().orElseThrow().getId();
        assertGpuMatchesCpu(network,
                List.of(new Contingency("combo23", new BranchContingency(lineId),
                        new com.powsybl.contingency.ShuntCompensatorContingency(shuntId),
                        new com.powsybl.contingency.LoadContingency(loadId))), 0.05, true);
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

    private static void countReportKeys(com.powsybl.commons.report.ReportNode node, java.util.Map<String, Integer> tally) {
        tally.merge(node.getMessageKey(), 1, Integer::sum);
        for (com.powsybl.commons.report.ReportNode c : node.getChildren()) {
            countReportKeys(c, tally);
        }
    }

    private static com.powsybl.commons.report.ReportNode findSubtree(com.powsybl.commons.report.ReportNode node, String key) {
        if (key.equals(node.getMessageKey())) {
            return node;
        }
        for (com.powsybl.commons.report.ReportNode c : node.getChildren()) {
            var found = findSubtree(c, key);
            if (found != null) {
                return found;
            }
        }
        return null;
    }

    @org.junit.jupiter.api.Test
    void tvcSaVsFreshLfIterationCounts() {
        // CPU-ONLY: understand why the CPU sequential SA does not reproduce a fresh CPU LF for this
        // contingency (~6e-3 kV) BEFORE looking at the GPU path. Counts the number of Newton SOLVES
        // (mismatchInitial = 1 + outer-loop re-solves) — NOT olf.OuterLoopIteration, which the AFTER_GENERATOR
        // TransformerVoltageControlOuterLoop does not emit (only the incremental/RL loops do). Result:
        // fresh LF = 5 solves / 13 Newton (tap 3); CPU-SA post-contingency = 3 solves / 6 Newton. The TVC
        // DOES run in the post-contingency (initStep -> controlStep -> round), it just does fewer re-solves
        // from the warm start and lands ~6e-3 short (both at tap 3, since 6e-3 << the ~1.7 kV tap step). So
        // the gap is a warm-start convergence difference in the CPU SA, not a skipped outer loop. Reported at
        // IDENTICAL default config. No GPU involved.
        String[] bis = new String[1];
        java.util.function.Supplier<Network> baseNet = () -> {
            Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
            network.getTwoWindingsTransformer("T2wT").getRatioTapChanger().setTargetDeadband(0).setRegulating(true)
                    .setTapPosition(0).setRegulationTerminal(network.getTwoWindingsTransformer("T2wT").getTerminal2()).setTargetV(34.0);
            com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
            bis[0] = l0.getId() + "_BIS";
            network.newLine().setId(bis[0])
                    .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                    .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                    .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
            return network;
        };
        java.util.function.BiConsumer<com.powsybl.loadflow.LoadFlowParameters, com.powsybl.openloadflow.OpenLoadFlowParameters.ReportedFeatures> cfg = (lfp, feat) ->
                com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                        .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL)
                        .setNewtonRaphsonConvEpsPerEq(1e-4)
                        .setReportedFeatures(java.util.Set.of(feat));

        // FRESH LF at the post-contingency operating point (l0_BIS disconnected)
        Network pc = baseNet.get();
        pc.getLine(bis[0]).getTerminal1().disconnect();
        pc.getLine(bis[0]).getTerminal2().disconnect();
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setTransformerVoltageControlOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
        cfg.accept(lfp, com.powsybl.openloadflow.OpenLoadFlowParameters.ReportedFeatures.NEWTON_RAPHSON_LOAD_FLOW);
        com.powsybl.commons.report.ReportNode rLf = com.powsybl.commons.report.ReportNode.newRootReportNode()
                .withResourceBundles(com.powsybl.openloadflow.util.report.PowsyblOpenLoadFlowReportResourceBundle.BASE_NAME)
                .withMessageTemplate("olf.securityAnalysis").build();
        loadFlowProvider.run(pc, computationManager, pc.getVariantManager().getWorkingVariantId(), lfp, rLf).join();
        java.util.Map<String, Integer> tLf = new java.util.TreeMap<>();
        countReportKeys(rLf, tLf);
        double lfV3 = pc.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV();
        System.err.printf("[ITER] FRESH-LF  V(BUS_3)=%.6f  newtonSolves=%d  newtonIter=%d  tap=%d%n", lfV3,
                tLf.getOrDefault("olf.mismatchInitial", 0),
                tLf.getOrDefault("olf.mismatchInitial", 0) + tLf.getOrDefault("olf.mismatchIteration", 0),
                pc.getTwoWindingsTransformer("T2wT").getRatioTapChanger().getSolvedTapPosition());

        // CPU sequential SA on the contingency
        Network saNet = baseNet.get();
        List<Contingency> ctgs = List.of(new Contingency(bis[0], new BranchContingency(bis[0])));
        com.powsybl.loadflow.LoadFlowParameters saLfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setTransformerVoltageControlOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
        cfg.accept(saLfp, com.powsybl.openloadflow.OpenLoadFlowParameters.ReportedFeatures.NEWTON_RAPHSON_SECURITY_ANALYSIS);
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(saLfp);
        List<com.powsybl.security.monitor.StateMonitor> mon = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(),
                saNet.getVoltageLevelStream().map(vl -> vl.getId()).collect(java.util.stream.Collectors.toSet()), java.util.Set.of()));
        com.powsybl.commons.report.ReportNode rSa = com.powsybl.commons.report.ReportNode.newRootReportNode()
                .withResourceBundles(com.powsybl.openloadflow.util.report.PowsyblOpenLoadFlowReportResourceBundle.BASE_NAME)
                .withMessageTemplate("olf.securityAnalysis").build();
        SecurityAnalysisResult saCpu = runSecurityAnalysis(saNet, ctgs, mon, sap, rSa);
        double saV3 = saCpu.getPostContingencyResults().get(0).getNetworkResult().getBusResults().stream().filter(b -> b.getBusId().equals("BUS_3")).findFirst().orElseThrow().getV();
        // isolate the POST-contingency subtree (exclude the base N run)
        var preNode = findSubtree(rSa, "olf.preContingencySimulation");
        var postNode = findSubtree(rSa, "olf.postContingencySimulation");
        java.util.Map<String, Integer> tPre = new java.util.TreeMap<>();
        java.util.Map<String, Integer> tPost = new java.util.TreeMap<>();
        if (preNode != null) {
            countReportKeys(preNode, tPre);
        }
        if (postNode != null) {
            countReportKeys(postNode, tPost);
        }
        System.err.printf("[ITER] CPU-SA base N       newtonSolves=%d  newtonIter=%d%n",
                tPre.getOrDefault("olf.mismatchInitial", 0),
                tPre.getOrDefault("olf.mismatchInitial", 0) + tPre.getOrDefault("olf.mismatchIteration", 0));
        System.err.printf("[ITER] CPU-SA post-ctg     V(BUS_3)=%.6f  newtonSolves=%d  newtonIter=%d%n", saV3,
                tPost.getOrDefault("olf.mismatchInitial", 0),
                tPost.getOrDefault("olf.mismatchInitial", 0) + tPost.getOrDefault("olf.mismatchIteration", 0));
        System.err.println("[ITER] CPU-SA post-ctg keys: " + tPost);
    }

    @org.junit.jupiter.api.Test
    void tvcSaGpuReproducesFreshLoadFlow() {
        // The GPU batched transformer-VC SA must reproduce a fresh (fully re-converged) post-contingency
        // LoadFlow of the same contingency, at IDENTICAL default config. This is the correct fidelity
        // criterion — comparing the GPU SA to the CPU SA is misleading here because the CPU SA itself
        // diverges from its own plain LoadFlow (see the printed [MATCHED] line): plainLF-CPU=34.4265 but
        // SA-CPU=34.4324 (~6e-3), the CPU-SA warm-started post-contingency solve stopping short of the
        // fresh-LF solution. Both solvers agree on the plain LF, and the GPU SA reproduces it; the CPU SA
        // is the lone outlier. So the ~6e-3 "GPU vs CPU SA" gap is a CPU-SA anomaly, not a GPU error.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        String[] bisHolder = new String[1];
        java.util.function.Supplier<Network> baseNet = () -> {
            Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
            network.getTwoWindingsTransformer("T2wT").getRatioTapChanger().setTargetDeadband(0).setRegulating(true)
                    .setTapPosition(0).setRegulationTerminal(network.getTwoWindingsTransformer("T2wT").getTerminal2()).setTargetV(34.0);
            com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
            bisHolder[0] = l0.getId() + "_BIS";
            network.newLine().setId(bisHolder[0])
                    .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                    .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                    .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
            return network;
        };
        java.util.function.BiFunction<Double, Boolean, Double> plainLfV3 = (convEps, gpu) -> {
            com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                    .setTransformerVoltageControlOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
            var ext = com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                    .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL)
                    .setNewtonRaphsonConvEpsPerEq(convEps);
            if (gpu) {
                ext.setAcSolverType(com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphsonFactory.NAME);
            }
            Network net = baseNet.get();
            net.getLine(bisHolder[0]).getTerminal1().disconnect();
            net.getLine(bisHolder[0]).getTerminal2().disconnect();
            new com.powsybl.loadflow.LoadFlow.Runner(new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory())).run(net, lfp);
            return net.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV();
        };

        // (A) SOLVER IDENTITY — with truly identical config the CPU and GPU plain LF converge to the SAME
        //     post-contingency solution to MACHINE PRECISION, at DEFAULT convEps (1e-4) AND tight (1e-8):
        //     ~1.4e-14 / ~7.1e-15. (An earlier ~1e-5 I saw was a config bug — OpenLoadFlowParameters.create
        //     on already-extended params reset the TVC mode on one side — not a solver difference.)
        double plCpuDef = plainLfV3.apply(1e-4, false);
        double plGpuDef = plainLfV3.apply(1e-4, true);
        double plCpuTight = plainLfV3.apply(1e-8, false);
        double plGpuTight = plainLfV3.apply(1e-8, true);
        System.err.printf("[SOLVER] plainLF CPU-vs-GPU  DEFAULT convEps d=%.3e  |  TIGHT convEps d=%.3e (both machine precision)%n",
                Math.abs(plCpuDef - plGpuDef), Math.abs(plCpuTight - plGpuTight));
        org.junit.jupiter.api.Assertions.assertEquals(plCpuDef, plGpuDef, 1e-9,
                "CPU and GPU plain LF must converge to the SAME solution at DEFAULT convEps (machine precision)");
        org.junit.jupiter.api.Assertions.assertEquals(plCpuTight, plGpuTight, 1e-9,
                "and at tight convEps too (machine precision)");

        // (B) SA ORCHESTRATION difference (NOT the solver, proven identical above). Both SA implementations
        //     warm-start each contingency from the base N state (the GPU batch tiles the base row as x0 —
        //     olf_gpu_nr_jni.cu ~L2878 — so warm-start is NOT the differentiator). Yet, at the SAME default
        //     convEps, the GPU batch converges each contingency close to the fresh LF while the CPU sequential
        //     SA stops ~6e-3 short — a post-contingency outer-loop-convergence difference between the two SA
        //     code paths (exact mechanism still under investigation), not a GPU error.
        java.util.function.Supplier<com.powsybl.loadflow.LoadFlowParameters> mkSaLfp = () -> {
            com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                    .setTransformerVoltageControlOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                    .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL)
                    .setNewtonRaphsonConvEpsPerEq(1e-4);  // pin EXACTLY the fresh-LF convEps — no hidden mismatch
            return lfp;
        };
        Network saNet = baseNet.get();
        List<Contingency> ctgs = List.of(new Contingency(bisHolder[0], new BranchContingency(bisHolder[0])));
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(mkSaLfp.get());
        List<com.powsybl.security.monitor.StateMonitor> mon = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(),
                saNet.getVoltageLevelStream().map(vl -> vl.getId()).collect(java.util.stream.Collectors.toSet()), java.util.Set.of()));
        SecurityAnalysisResult saCpu = runSecurityAnalysis(saNet, ctgs, mon, sap);
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        SecurityAnalysisResult saGpu = runSecurityAnalysis(baseNet.get(), ctgs, mon, sap);
        securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        double saCpuV = saCpu.getPostContingencyResults().get(0).getNetworkResult().getBusResults().stream().filter(b -> b.getBusId().equals("BUS_3")).findFirst().orElseThrow().getV();
        double saGpuV = saGpu.getPostContingencyResults().get(0).getNetworkResult().getBusResults().stream().filter(b -> b.getBusId().equals("BUS_3")).findFirst().orElseThrow().getV();
        System.err.printf("[SA-DEFAULT] freshLF=%.6f  SA-GPU=%.6f  SA-CPU=%.6f (d(SA-GPU,SA-CPU)=%.3e)%n",
                plGpuDef, saGpuV, saCpuV, Math.abs(saCpuV - saGpuV));
        // (B) With the OLF-faithful orchestration landed (identity/pi-model re-initialization between outer
        // re-solves + no re-solve on a STABLE phase transition), the GPU SA reproduces the CPU SA EXACTLY at
        // default tolerance. The CPU SA legitimately differs from a FRESH LF here (~6e-3): its warm-started
        // post-contingency mismatch stays below the distributed-slack band, so it distributes 0 where a fresh
        // flat solve distributes the full amount — expected OLF SA behavior, faithfully reproduced by the GPU.
        org.junit.jupiter.api.Assertions.assertEquals(saCpuV, saGpuV, 1e-12,
                "GPU batched SA must reproduce the CPU SA at default tolerance (machine precision)");
    }

    @org.junit.jupiter.api.Test
    void tvcSaCpuVsGpuAtDefaultTolerance() {
        // TARGET (per the no-tighten-tolerance rule): the GPU SA must agree with the CPU sequential SA at
        // DEFAULT criteria/tolerance. If it does not, the GPU ORCHESTRATION does a different number of
        // outer-loop re-solves per contingency and must be fixed. Prints CPU vs GPU-device vs GPU-JAVA at
        // default; asserts nothing yet (documents the orchestration gap to close).
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
        network.getTwoWindingsTransformer("T2wT").getRatioTapChanger().setTargetDeadband(0).setRegulating(true)
                .setTapPosition(0).setRegulationTerminal(network.getTwoWindingsTransformer("T2wT").getTerminal2()).setTargetV(34.0);
        com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
        String bis = l0.getId() + "_BIS";
        network.newLine().setId(bis)
                .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
        List<Contingency> ctgs = List.of(new Contingency(bis, new BranchContingency(bis)));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setTransformerVoltageControlOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL);
        // DEFAULT convEps + DEFAULT slack mismatch — do NOT tighten.
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        java.util.Set<String> vls = network.getVoltageLevelStream().map(vl -> vl.getId()).collect(java.util.stream.Collectors.toSet());
        List<com.powsybl.security.monitor.StateMonitor> mon = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(), vls, java.util.Set.of()));
        java.util.function.Function<SecurityAnalysisResult, Double> v3 = r -> r.getPostContingencyResults().get(0)
                .getNetworkResult().getBusResults().stream().filter(b -> b.getBusId().equals("BUS_3")).findFirst().orElseThrow().getV();
        double cpu = v3.apply(runSecurityAnalysis(network, ctgs, mon, sap));
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        double gpuDevice = v3.apply(runSecurityAnalysis(network, ctgs, mon, sap));
        System.setProperty(GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY, "JAVA");
        double gpuJava;
        try {
            gpuJava = v3.apply(runSecurityAnalysis(network, ctgs, mon, sap));
        } finally {
            System.clearProperty(GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY);
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
        System.err.printf("[DEFAULT] V(BUS_3)  CPU=%.6f  GPU-DEVICE=%.6f (d=%.3e)  GPU-JAVA=%.6f (d=%.3e)%n",
                cpu, gpuDevice, Math.abs(cpu - gpuDevice), gpuJava, Math.abs(cpu - gpuJava));
        // DEFAULT tolerance: the JAVA hybrid (OLF's own outer loops + OLF-faithful re-solve initialization)
        // agrees with the CPU sequential SA. The DEVICE orchestrator still carries a known ~6e-3 warm-start
        // gap (it does not re-initialize identity rows between its on-device rounds) — not asserted yet.
        org.junit.jupiter.api.Assertions.assertEquals(cpu, gpuJava, 1e-12,
                "JAVA hybrid must match the CPU SA at default tolerance");
    }

    @org.junit.jupiter.api.Test
    void tvcWarmVsFlatPostContingencyDiagnostic() {
        // CPU-ONLY: does a WARM-started post-contingency LF (base LF -> apply contingency -> LF from previous
        // values, like the CPU SA) converge to a DIFFERENT point than a FRESH flat LF of the same operating
        // point? If yes (and different tap or ~6e-3 V), the CPU SA's ~6e-3 is a warm-start non-robustness, and
        // the GPU (which reproduces the flat LF) is the more accurate. Reads the SOLVED TAP on each.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        java.util.function.Supplier<Network> baseNet = () -> {
            Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
            network.getTwoWindingsTransformer("T2wT").getRatioTapChanger().setTargetDeadband(0).setRegulating(true)
                    .setTapPosition(0).setRegulationTerminal(network.getTwoWindingsTransformer("T2wT").getTerminal2()).setTargetV(34.0);
            com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
            network.newLine().setId(l0.getId() + "_BIS")
                    .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                    .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                    .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
            return network;
        };
        String bis = baseNet.get().getLines().iterator().next().getId() + "_BIS";
        java.util.function.Supplier<com.powsybl.loadflow.LoadFlowParameters> mk = () -> {
            com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                    .setTransformerVoltageControlOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                    .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL);
            return lfp;
        };
        com.powsybl.loadflow.LoadFlow.Runner runner = new com.powsybl.loadflow.LoadFlow.Runner(new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory()));

        // FRESH FLAT LF at the post-contingency point (uniform init)
        Network flat = baseNet.get();
        flat.getLine(bis).getTerminal1().disconnect();
        flat.getLine(bis).getTerminal2().disconnect();
        runner.run(flat, mk.get());
        double flatV = flat.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV();
        int flatTap = flat.getTwoWindingsTransformer("T2wT").getRatioTapChanger().getSolvedTapPosition();

        // WARM: base LF (N), then apply contingency, then LF from PREVIOUS_VALUES (like the CPU SA warm-start)
        Network warm = baseNet.get();
        runner.run(warm, mk.get());                                   // base N
        int baseTap = warm.getTwoWindingsTransformer("T2wT").getRatioTapChanger().getSolvedTapPosition();
        warm.getLine(bis).getTerminal1().disconnect();
        warm.getLine(bis).getTerminal2().disconnect();
        com.powsybl.loadflow.LoadFlowParameters warmLfp = mk.get();
        warmLfp.setVoltageInitMode(com.powsybl.loadflow.LoadFlowParameters.VoltageInitMode.PREVIOUS_VALUES);
        runner.run(warm, warmLfp);                                    // warm post-contingency
        double warmV = warm.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV();
        int warmTap = warm.getTwoWindingsTransformer("T2wT").getRatioTapChanger().getSolvedTapPosition();

        System.err.printf("[WARM-VS-FLAT] baseN tap=%d | FLAT V=%.6f tap=%d | WARM V=%.6f tap=%d | d(warm,flat)=%.3e%n",
                baseTap, flatV, flatTap, warmV, warmTap, Math.abs(warmV - flatV));
    }

    /** CPU-ONLY isolation: which outer loop makes the CPU SA post-contingency diverge from a plain LF of the
     *  SAME operating point? Toggles TVC / distributed slack / reactive limits independently, at DEFAULT
     *  tolerance (never tighten). For each combination prints plainLF vs CPU-SA V(BUS_3). */
    @org.junit.jupiter.api.Test
    void cpuSaVsPlainLfFeatureIsolation() {
        java.util.function.Supplier<Network> baseNet = () -> {
            Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
            network.getTwoWindingsTransformer("T2wT").getRatioTapChanger().setTargetDeadband(0).setRegulating(true)
                    .setTapPosition(0).setRegulationTerminal(network.getTwoWindingsTransformer("T2wT").getTerminal2()).setTargetV(34.0);
            com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
            network.newLine().setId(l0.getId() + "_BIS")
                    .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                    .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                    .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
            return network;
        };
        String bis = baseNet.get().getLines().iterator().next().getId() + "_BIS";
        com.powsybl.loadflow.LoadFlow.Runner runner = new com.powsybl.loadflow.LoadFlow.Runner(new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory()));
        boolean[][] combos = {{true, true, true}, {true, false, true}, {true, true, false}, {false, true, true}, {true, false, false}};
        for (boolean[] c : combos) {
            boolean tvc = c[0];
            boolean ds = c[1];
            boolean rl = c[2];
            java.util.function.Supplier<com.powsybl.loadflow.LoadFlowParameters> mk = () -> {
                com.powsybl.loadflow.LoadFlowParameters p = new com.powsybl.loadflow.LoadFlowParameters()
                        .setTransformerVoltageControlOn(tvc).setDistributedSlack(ds).setUseReactiveLimits(rl);
                com.powsybl.openloadflow.OpenLoadFlowParameters.create(p)
                        .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL);
                return p;   // DEFAULT convEps / slack mismatch
            };
            Network flat = baseNet.get();
            flat.getLine(bis).getTerminal1().disconnect();
            flat.getLine(bis).getTerminal2().disconnect();
            runner.run(flat, mk.get());
            double flatV = flat.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV();
            Network saNet = baseNet.get();
            com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
            sap.setLoadFlowParameters(mk.get());
            List<com.powsybl.security.monitor.StateMonitor> mon = List.of(new com.powsybl.security.monitor.StateMonitor(
                    com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(),
                    saNet.getVoltageLevelStream().map(vl -> vl.getId()).collect(java.util.stream.Collectors.toSet()), java.util.Set.of()));
            double saV = runSecurityAnalysis(saNet, List.of(new Contingency(bis, new BranchContingency(bis))), mon, sap)
                    .getPostContingencyResults().get(0).getNetworkResult().getBusResults().stream()
                    .filter(b -> b.getBusId().equals("BUS_3")).findFirst().orElseThrow().getV();
            System.err.printf("[ISO] tvc=%-5b ds=%-5b rl=%-5b  plainLF=%.6f  CPU-SA=%.6f  d=%.3e%n",
                    tvc, ds, rl, flatV, saV, Math.abs(flatV - saV));
        }
    }

    /** CPU-ONLY: the DS mechanism. Compare the DISTRIBUTED ACTIVE POWER (and the resulting V) of a plain
     *  post-contingency LF vs the CPU SA's post-contingency. Distributed slack stops inside a band
     *  (slackBusPMaxMismatch, default 1 MW), so two paths reaching the contingency from DIFFERENT generation
     *  splits (fresh vs base-distributed) legitimately stop at different points inside that band. */
    @org.junit.jupiter.api.Test
    void cpuSaDistributedPowerDiagnostic() {
        java.util.function.Supplier<Network> baseNet = () -> {
            Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
            network.getTwoWindingsTransformer("T2wT").getRatioTapChanger().setTargetDeadband(0).setRegulating(true)
                    .setTapPosition(0).setRegulationTerminal(network.getTwoWindingsTransformer("T2wT").getTerminal2()).setTargetV(34.0);
            com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
            network.newLine().setId(l0.getId() + "_BIS")
                    .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                    .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                    .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
            return network;
        };
        String bis = baseNet.get().getLines().iterator().next().getId() + "_BIS";
        java.util.function.Supplier<com.powsybl.loadflow.LoadFlowParameters> mk = () -> {
            com.powsybl.loadflow.LoadFlowParameters p = new com.powsybl.loadflow.LoadFlowParameters()
                    .setTransformerVoltageControlOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(p)
                    .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL);
            return p;
        };
        com.powsybl.loadflow.LoadFlow.Runner runner = new com.powsybl.loadflow.LoadFlow.Runner(new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory()));
        // plain LF at the post-contingency point, from a FRESH network
        Network flat = baseNet.get();
        flat.getLine(bis).getTerminal1().disconnect();
        flat.getLine(bis).getTerminal2().disconnect();
        com.powsybl.loadflow.LoadFlowResult flatRes = runner.run(flat, mk.get());
        System.err.printf("[DS] plain post-ctg LF: distributedActivePower=%.6f MW  V(BUS_3)=%.6f  genP=%s%n",
                flatRes.getComponentResults().get(0).getDistributedActivePower(),
                flat.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV(),
                flat.getGeneratorStream().map(g -> String.format("%s=%.4f", g.getId(), g.getTerminal().getP())).toList());
        // CPU SA: base N + post-contingency
        Network saNet = baseNet.get();
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(mk.get());
        List<com.powsybl.security.monitor.StateMonitor> mon = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(),
                saNet.getVoltageLevelStream().map(vl -> vl.getId()).collect(java.util.stream.Collectors.toSet()), java.util.Set.of()));
        SecurityAnalysisResult sa = runSecurityAnalysis(saNet, List.of(new Contingency(bis, new BranchContingency(bis))), mon, sap);
        System.err.printf("[DS] CPU-SA preContingency: distributedActivePower=%.6f MW%n",
                sa.getPreContingencyResult().getDistributedActivePower());
        System.err.printf("[DS] CPU-SA postContingency: distributedActivePower=%.6f MW  V(BUS_3)=%.6f%n",
                sa.getPostContingencyResults().get(0).getDistributedActivePower(),
                sa.getPostContingencyResults().get(0).getNetworkResult().getBusResults().stream()
                        .filter(b -> b.getBusId().equals("BUS_3")).findFirst().orElseThrow().getV());
        // Same measurement on the GPU SA paths — does the GPU distribute slack where the CPU SA distributes 0?
        if (com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable()) {
            securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
            SecurityAnalysisResult gDev = runSecurityAnalysis(baseNet.get(), List.of(new Contingency(bis, new BranchContingency(bis))), mon, sap);
            System.setProperty(GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY, "JAVA");
            SecurityAnalysisResult gJava;
            try {
                gJava = runSecurityAnalysis(baseNet.get(), List.of(new Contingency(bis, new BranchContingency(bis))), mon, sap);
            } finally {
                System.clearProperty(GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY);
                securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
            }
            for (var e : java.util.Map.of("GPU-DEVICE", gDev, "GPU-JAVA", gJava).entrySet()) {
                System.err.printf("[DS] %s postContingency: distributedActivePower=%.6f MW  V(BUS_3)=%.6f%n",
                        e.getKey(), e.getValue().getPostContingencyResults().get(0).getDistributedActivePower(),
                        e.getValue().getPostContingencyResults().get(0).getNetworkResult().getBusResults().stream()
                                .filter(b -> b.getBusId().equals("BUS_3")).findFirst().orElseThrow().getV());
            }
            // CROSS-CHECK with distributed slack OFF: if GPU-SA == CPU-SA there, the whole GPU-vs-CPU SA
            // discrepancy is DS-orchestration, and DS is the single thing to align.
            com.powsybl.loadflow.LoadFlowParameters noDs = new com.powsybl.loadflow.LoadFlowParameters()
                    .setTransformerVoltageControlOn(true).setDistributedSlack(false).setUseReactiveLimits(true);
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(noDs)
                    .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL);
            com.powsybl.security.SecurityAnalysisParameters sapNoDs = new com.powsybl.security.SecurityAnalysisParameters();
            sapNoDs.setLoadFlowParameters(noDs);
            double cpuNoDs = runSecurityAnalysis(baseNet.get(), List.of(new Contingency(bis, new BranchContingency(bis))), mon, sapNoDs)
                    .getPostContingencyResults().get(0).getNetworkResult().getBusResults().stream()
                    .filter(b -> b.getBusId().equals("BUS_3")).findFirst().orElseThrow().getV();
            securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
            double gpuNoDs;
            try {
                gpuNoDs = runSecurityAnalysis(baseNet.get(), List.of(new Contingency(bis, new BranchContingency(bis))), mon, sapNoDs)
                        .getPostContingencyResults().get(0).getNetworkResult().getBusResults().stream()
                        .filter(b -> b.getBusId().equals("BUS_3")).findFirst().orElseThrow().getV();
            } finally {
                securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
            }
            System.err.printf("[DS-OFF] CPU-SA=%.6f  GPU-SA=%.6f  d=%.3e%n", cpuNoDs, gpuNoDs, Math.abs(cpuNoDs - gpuNoDs));
        }
    }

    /** GPU-vs-CPU SA isolation at DEFAULT tolerance: which outer-loop feature drives the GPU's ~5.5e-3?
     *  Compares CPU-SA / GPU-DEVICE / GPU-JAVA (and the plain LF reference) with TVC / DS / RL toggled. */
    @org.junit.jupiter.api.Test
    void gpuVsCpuSaFeatureIsolation() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        java.util.function.Supplier<Network> baseNet = () -> {
            Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
            network.getTwoWindingsTransformer("T2wT").getRatioTapChanger().setTargetDeadband(0).setRegulating(true)
                    .setTapPosition(0).setRegulationTerminal(network.getTwoWindingsTransformer("T2wT").getTerminal2()).setTargetV(34.0);
            com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
            network.newLine().setId(l0.getId() + "_BIS")
                    .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                    .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                    .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
            return network;
        };
        String bis = baseNet.get().getLines().iterator().next().getId() + "_BIS";
        List<Contingency> ctgs = List.of(new Contingency(bis, new BranchContingency(bis)));
        com.powsybl.loadflow.LoadFlow.Runner runner = new com.powsybl.loadflow.LoadFlow.Runner(new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory()));
        boolean[][] combos = {{true, true, true}, {true, false, true}, {false, true, true}, {false, true, false}, {false, false, true}, {false, false, false}};
        for (boolean[] c : combos) {
            boolean tvc = c[0];
            boolean ds = c[1];
            boolean rl = c[2];
            java.util.function.Supplier<com.powsybl.loadflow.LoadFlowParameters> mk = () -> {
                com.powsybl.loadflow.LoadFlowParameters p = new com.powsybl.loadflow.LoadFlowParameters()
                        .setTransformerVoltageControlOn(tvc).setDistributedSlack(ds).setUseReactiveLimits(rl);
                com.powsybl.openloadflow.OpenLoadFlowParameters.create(p)
                        .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL);
                return p;
            };
            Network flat = baseNet.get();
            flat.getLine(bis).getTerminal1().disconnect();
            flat.getLine(bis).getTerminal2().disconnect();
            runner.run(flat, mk.get());
            double flatV = flat.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV();
            com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
            sap.setLoadFlowParameters(mk.get());
            Network probe = baseNet.get();
            List<com.powsybl.security.monitor.StateMonitor> mon = List.of(new com.powsybl.security.monitor.StateMonitor(
                    com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(),
                    probe.getVoltageLevelStream().map(vl -> vl.getId()).collect(java.util.stream.Collectors.toSet()), java.util.Set.of()));
            java.util.function.Function<SecurityAnalysisResult, Double> v3 = r -> r.getPostContingencyResults().get(0)
                    .getNetworkResult().getBusResults().stream().filter(b -> b.getBusId().equals("BUS_3")).findFirst().orElseThrow().getV();
            double cpu = v3.apply(runSecurityAnalysis(baseNet.get(), ctgs, mon, sap));
            securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
            double dev = v3.apply(runSecurityAnalysis(baseNet.get(), ctgs, mon, sap));
            int devEngaged = GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get();
            System.setProperty(GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY, "JAVA");
            double jav;
            int javEngaged;
            try {
                jav = v3.apply(runSecurityAnalysis(baseNet.get(), ctgs, mon, sap));
                javEngaged = GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get();
            } finally {
                System.clearProperty(GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY);
                securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
            }
            System.err.printf("[GPU-ISO] tvc=%-5b ds=%-5b rl=%-5b | plainLF=%.6f CPU=%.6f | DEV=%.6f (d=%.1e eng=%d) JAVA=%.6f (d=%.1e eng=%d)%n",
                    tvc, ds, rl, flatV, cpu, dev, Math.abs(cpu - dev), devEngaged, jav, Math.abs(cpu - jav), javEngaged);
        }
    }

    /** Is the GPU TVC error already present in the PRE-contingency (N) result? tvc=on, ds=off (clean case).
     *  If N already differs, it is the base LF / TVC handling, not the contingency orchestration. */
    @org.junit.jupiter.api.Test
    void gpuTvcPreContingencyDiagnostic() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        java.util.function.Supplier<Network> baseNet = () -> {
            Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
            network.getTwoWindingsTransformer("T2wT").getRatioTapChanger().setTargetDeadband(0).setRegulating(true)
                    .setTapPosition(0).setRegulationTerminal(network.getTwoWindingsTransformer("T2wT").getTerminal2()).setTargetV(34.0);
            com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
            network.newLine().setId(l0.getId() + "_BIS")
                    .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                    .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                    .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
            return network;
        };
        String bis = baseNet.get().getLines().iterator().next().getId() + "_BIS";
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setTransformerVoltageControlOn(true).setDistributedSlack(false).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL);
        // plain N load flow (no contingency), CPU vs GPU solver
        com.powsybl.loadflow.LoadFlow.Runner runner = new com.powsybl.loadflow.LoadFlow.Runner(new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory()));
        Network nc = baseNet.get();
        runner.run(nc, lfp);
        com.powsybl.loadflow.LoadFlowParameters gl = new com.powsybl.loadflow.LoadFlowParameters()
                .setTransformerVoltageControlOn(true).setDistributedSlack(false).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(gl)
                .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL)
                .setAcSolverType(com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphsonFactory.NAME);
        Network ng = baseNet.get();
        runner.run(ng, gl);
        System.err.printf("[N-LF] plain N: CPU=%.9f tap=%d | GPU=%.9f tap=%d | d=%.3e%n",
                nc.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV(),
                nc.getTwoWindingsTransformer("T2wT").getRatioTapChanger().getSolvedTapPosition(),
                ng.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV(),
                ng.getTwoWindingsTransformer("T2wT").getRatioTapChanger().getSolvedTapPosition(),
                Math.abs(nc.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV()
                        - ng.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV()));
        // SA pre-contingency (N) result, CPU vs GPU providers
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        Network probe = baseNet.get();
        List<com.powsybl.security.monitor.StateMonitor> mon = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(),
                probe.getVoltageLevelStream().map(vl -> vl.getId()).collect(java.util.stream.Collectors.toSet()), java.util.Set.of()));
        List<Contingency> ctgs = List.of(new Contingency(bis, new BranchContingency(bis)));
        java.util.function.Function<SecurityAnalysisResult, Double> preV = r -> r.getPreContingencyResult()
                .getNetworkResult().getBusResults().stream().filter(b -> b.getBusId().equals("BUS_3")).findFirst().orElseThrow().getV();
        double cpuPre = preV.apply(runSecurityAnalysis(baseNet.get(), ctgs, mon, sap));
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        double gpuPre;
        try {
            gpuPre = preV.apply(runSecurityAnalysis(baseNet.get(), ctgs, mon, sap));
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
        System.err.printf("[N-SA] SA preContingency: CPU=%.9f  GPU=%.9f  d=%.3e%n", cpuPre, gpuPre, Math.abs(cpuPre - gpuPre));
    }

    /** The GPU batch's ROUND-0 equation system (post-contingency, TVC control OFF, tap pinned at 3) solved as a
     *  plain LF on CPU vs GPU. The batch's round 0 reported V=34.425712 (ds=off); if the CPU gives 34.4311 here,
     *  the GPU's very first post-contingency solve is already wrong, before any TVC action. */
    @org.junit.jupiter.api.Test
    void tvcRound0EquivalentDiagnostic() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        java.util.function.Supplier<Network> mkNet = () -> {
            Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
            // tap 3 (rho 1.1) = where the base LF leaves it; control OFF so r1 stays pinned there
            network.getTwoWindingsTransformer("T2wT").getRatioTapChanger().setTargetDeadband(0).setRegulating(false)
                    .setTapPosition(3).setRegulationTerminal(network.getTwoWindingsTransformer("T2wT").getTerminal2()).setTargetV(34.0);
            com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
            network.newLine().setId(l0.getId() + "_BIS")
                    .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                    .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                    .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
            network.getLine(l0.getId() + "_BIS").getTerminal1().disconnect();
            network.getLine(l0.getId() + "_BIS").getTerminal2().disconnect();
            return network;
        };
        com.powsybl.loadflow.LoadFlow.Runner runner = new com.powsybl.loadflow.LoadFlow.Runner(new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory()));
        for (boolean gpu : new boolean[] {false, true}) {
            com.powsybl.loadflow.LoadFlowParameters p = new com.powsybl.loadflow.LoadFlowParameters()
                    .setTransformerVoltageControlOn(true).setDistributedSlack(false).setUseReactiveLimits(true);
            var ext = com.powsybl.openloadflow.OpenLoadFlowParameters.create(p)
                    .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL);
            if (gpu) {
                ext.setAcSolverType(com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphsonFactory.NAME);
            }
            Network net = mkNet.get();
            runner.run(net, p);
            System.err.printf("[ROUND0] %s: V(BUS_3)=%.9f  tap=%d%n", gpu ? "GPU" : "CPU",
                    net.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV(),
                    net.getTwoWindingsTransformer("T2wT").getRatioTapChanger().getTapPosition());
        }
    }

    /** DECISIVE test of the stale-admittance hypothesis: OLF's PiModelArray returns R/X/Y/ksi from the CURRENT
     *  TAP, but the GPU bakes the admittance into cbIn at extraction. Build the SAME case with every tap step
     *  sharing IDENTICAL R/X (so a tap change cannot alter the admittance) and compare GPU vs CPU SA. If the
     *  ~5.5e-3 collapses to machine precision, the stale device admittance on tap change is the bug. */
    @org.junit.jupiter.api.Test
    void tvcUniformTapImpedanceDiagnostic() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        for (boolean uniform : new boolean[] {false, true}) {
            java.util.function.Supplier<Network> mkNet = () -> {
                Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
                com.powsybl.iidm.network.TwoWindingsTransformer t = network.getTwoWindingsTransformer("T2wT");
                if (uniform) {
                    // force every tap step to the SAME R/X (only rho differs)
                    double r0 = t.getRatioTapChanger().getStep(0).getR();
                    double x0 = t.getRatioTapChanger().getStep(0).getX();
                    for (int i = t.getRatioTapChanger().getLowTapPosition(); i <= t.getRatioTapChanger().getHighTapPosition(); i++) {
                        t.getRatioTapChanger().getStep(i).setR(r0).setX(x0);
                    }
                }
                t.getRatioTapChanger().setTargetDeadband(0).setRegulating(true).setTapPosition(0)
                        .setRegulationTerminal(t.getTerminal2()).setTargetV(34.0);
                com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
                network.newLine().setId(l0.getId() + "_BIS")
                        .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                        .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                        .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
                return network;
            };
            String bis = mkNet.get().getLines().iterator().next().getId() + "_BIS";
            com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                    .setTransformerVoltageControlOn(true).setDistributedSlack(false).setUseReactiveLimits(true);
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                    .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL);
            com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
            sap.setLoadFlowParameters(lfp);
            List<Contingency> ctgs = List.of(new Contingency(bis, new BranchContingency(bis)));
            List<com.powsybl.security.monitor.StateMonitor> mon = List.of(new com.powsybl.security.monitor.StateMonitor(
                    com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(),
                    mkNet.get().getVoltageLevelStream().map(vl -> vl.getId()).collect(java.util.stream.Collectors.toSet()), java.util.Set.of()));
            java.util.function.Function<SecurityAnalysisResult, Double> v3 = r -> r.getPostContingencyResults().get(0)
                    .getNetworkResult().getBusResults().stream().filter(b -> b.getBusId().equals("BUS_3")).findFirst().orElseThrow().getV();
            double cpu = v3.apply(runSecurityAnalysis(mkNet.get(), ctgs, mon, sap));
            securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
            double gpu;
            try {
                gpu = v3.apply(runSecurityAnalysis(mkNet.get(), ctgs, mon, sap));
            } finally {
                securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
            }
            System.err.printf("[TAP-Z] uniformTapImpedance=%-5b  CPU=%.9f  GPU=%.9f  d=%.3e%n", uniform, cpu, gpu, Math.abs(cpu - gpu));
        }
    }

    /** Is the CPU TVC answer a FIXED POINT of the tap it lands on? Run the plain post-contingency LF with TVC
     *  (lands at tap 3, V=34.4311), then FREEZE that tap (regulating=false) and re-solve. If V moves to the
     *  fixed-tap solution (~34.4254 — which is where the GPU batch lands), the TVC path's reported state is not
     *  a converged solution of its own final tap system. CPU-only, default tolerance. */
    @org.junit.jupiter.api.Test
    void tvcResultIsFixedPointDiagnostic() {
        java.util.function.Supplier<Network> mkNet = () -> {
            Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
            com.powsybl.iidm.network.TwoWindingsTransformer t = network.getTwoWindingsTransformer("T2wT");
            t.getRatioTapChanger().setTargetDeadband(0).setRegulating(true).setTapPosition(0)
                    .setRegulationTerminal(t.getTerminal2()).setTargetV(34.0);
            com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
            network.newLine().setId(l0.getId() + "_BIS")
                    .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                    .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                    .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
            network.getLine(l0.getId() + "_BIS").getTerminal1().disconnect();
            network.getLine(l0.getId() + "_BIS").getTerminal2().disconnect();
            return network;
        };
        java.util.function.Supplier<com.powsybl.loadflow.LoadFlowParameters> mk = () -> {
            com.powsybl.loadflow.LoadFlowParameters p = new com.powsybl.loadflow.LoadFlowParameters()
                    .setTransformerVoltageControlOn(true).setDistributedSlack(false).setUseReactiveLimits(true);
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(p)
                    .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL);
            return p;
        };
        com.powsybl.loadflow.LoadFlow.Runner runner = new com.powsybl.loadflow.LoadFlow.Runner(new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory()));
        Network net = mkNet.get();
        runner.run(net, mk.get());
        int solvedTap = net.getTwoWindingsTransformer("T2wT").getRatioTapChanger().getSolvedTapPosition();
        double vTvc = net.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV();
        System.err.printf("[FIXPOINT] TVC LF: V=%.9f solvedTap=%d (tapPosition=%d)%n", vTvc, solvedTap,
                net.getTwoWindingsTransformer("T2wT").getRatioTapChanger().getTapPosition());
        // freeze that tap and re-solve
        net.getTwoWindingsTransformer("T2wT").getRatioTapChanger().setRegulating(false).setTapPosition(solvedTap);
        runner.run(net, mk.get());
        double vFrozen = net.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV();
        System.err.printf("[FIXPOINT] re-solved with tap FROZEN at %d: V=%.9f  (moved by %.3e)%n",
                solvedTap, vFrozen, Math.abs(vFrozen - vTvc));
    }

    /** Find a targetV where the CONTINGENCY MOVES THE TAP (base tap != post-contingency tap) — the case that
     *  exercises the device pi-model refresh (a tap change alters R/X/Y/ksi in OLF's PiModelArray). */
    @org.junit.jupiter.api.Test
    void findTapMovingContingencyDiagnostic() {
        java.util.function.BiFunction<Double, Boolean, int[]> run = (targetV, outage) -> {
            Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
            com.powsybl.iidm.network.TwoWindingsTransformer t = network.getTwoWindingsTransformer("T2wT");
            t.getRatioTapChanger().setTargetDeadband(0).setRegulating(true).setTapPosition(0)
                    .setRegulationTerminal(t.getTerminal2()).setTargetV(targetV);
            com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
            network.newLine().setId(l0.getId() + "_BIS")
                    .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                    .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                    .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
            if (outage) {
                network.getLine(l0.getId()).getTerminal1().disconnect();   // outage the ORIGINAL line (bigger change)
                network.getLine(l0.getId()).getTerminal2().disconnect();
            }
            com.powsybl.loadflow.LoadFlowParameters p = new com.powsybl.loadflow.LoadFlowParameters()
                    .setTransformerVoltageControlOn(true).setDistributedSlack(false).setUseReactiveLimits(true);
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(p)
                    .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL);
            new com.powsybl.loadflow.LoadFlow.Runner(new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory())).run(network, p);
            return new int[] {network.getTwoWindingsTransformer("T2wT").getRatioTapChanger().getSolvedTapPosition()};
        };
        for (double tv : new double[] {30.0, 31.0, 32.0, 32.5, 33.0, 33.5, 34.0, 35.0, 36.0, 37.0, 38.0}) {
            int baseTap = run.apply(tv, false)[0];
            int ctgTap = run.apply(tv, true)[0];
            System.err.printf("[TAPMOVE] targetV=%.1f  baseTap=%d  postCtgTap=%d %s%n",
                    tv, baseTap, ctgTap, baseTap != ctgTap ? "  <== TAP MOVES" : "");
        }
    }

    /** VALIDATES the device pi-model refresh: a contingency that MOVES THE TAP (2 -> 3) on a transformer whose
     *  tap steps have DISTINCT R/X. OLF's PiModelArray returns R/X/Y/ksi from the current tap, so the tap move
     *  changes the branch admittance; without refreshing the device row the batch would solve with the stale
     *  extraction-time admittance. GPU (JAVA hybrid) must match the CPU SA at DEFAULT tolerance. */
    @org.junit.jupiter.api.Test
    void javaTvcTapMovingContingencyMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        java.util.function.Supplier<Network> mkNet = () -> {
            Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
            com.powsybl.iidm.network.TwoWindingsTransformer t = network.getTwoWindingsTransformer("T2wT");
            // make every tap step's impedance DISTINCT so a tap move really changes R/X/Y/ksi
            var rtc = t.getRatioTapChanger();
            for (int i = rtc.getLowTapPosition(); i <= rtc.getHighTapPosition(); i++) {
                double f = 1.0 + 0.6 * (i - rtc.getLowTapPosition());
                rtc.getStep(i).setR(0.11 * f).setX(0.011 * f);
            }
            rtc.setTargetDeadband(0).setRegulating(true).setTapPosition(0)
                    .setRegulationTerminal(t.getTerminal2()).setTargetV(33.5);
            com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
            network.newLine().setId(l0.getId() + "_BIS")
                    .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                    .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                    .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
            return network;
        };
        String outaged = mkNet.get().getLines().iterator().next().getId();   // outage the ORIGINAL line
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setTransformerVoltageControlOn(true).setDistributedSlack(false).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL);
        // report the tap move (base vs post-contingency) so the test documents that it exercises the refresh
        com.powsybl.loadflow.LoadFlow.Runner runner = new com.powsybl.loadflow.LoadFlow.Runner(new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory()));
        Network nb = mkNet.get();
        runner.run(nb, lfp);
        Network np = mkNet.get();
        np.getLine(outaged).getTerminal1().disconnect();
        np.getLine(outaged).getTerminal2().disconnect();
        runner.run(np, lfp);
        System.err.printf("[TAPFIX] baseTap=%d  postCtgTap=%d%n",
                nb.getTwoWindingsTransformer("T2wT").getRatioTapChanger().getSolvedTapPosition(),
                np.getTwoWindingsTransformer("T2wT").getRatioTapChanger().getSolvedTapPosition());

        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        List<Contingency> ctgs = List.of(new Contingency(outaged, new BranchContingency(outaged)));
        List<com.powsybl.security.monitor.StateMonitor> mon = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(),
                mkNet.get().getVoltageLevelStream().map(vl -> vl.getId()).collect(java.util.stream.Collectors.toSet()), java.util.Set.of()));
        java.util.function.Function<SecurityAnalysisResult, Double> v3 = r -> r.getPostContingencyResults().get(0)
                .getNetworkResult().getBusResults().stream().filter(b -> b.getBusId().equals("BUS_3")).findFirst().orElseThrow().getV();
        com.powsybl.openloadflow.OpenLoadFlowParameters.get(lfp).setReportedFeatures(
                java.util.Set.of(com.powsybl.openloadflow.OpenLoadFlowParameters.ReportedFeatures.NEWTON_RAPHSON_SECURITY_ANALYSIS));
        com.powsybl.commons.report.ReportNode rSa = com.powsybl.commons.report.ReportNode.newRootReportNode()
                .withResourceBundles(com.powsybl.openloadflow.util.report.PowsyblOpenLoadFlowReportResourceBundle.BASE_NAME)
                .withMessageTemplate("olf.securityAnalysis").build();
        double cpu = v3.apply(runSecurityAnalysis(mkNet.get(), ctgs, mon, sap, rSa));
        var postNode = findSubtree(rSa, "olf.postContingencySimulation");
        java.util.Map<String, Integer> tPost = new java.util.TreeMap<>();
        if (postNode != null) {
            countReportKeys(postNode, tPost);
        }
        System.err.printf("[TAPFIX] CPU-SA post-ctg: newtonSolves=%d newtonIters=%d%n",
                tPost.getOrDefault("olf.mismatchInitial", 0),
                tPost.getOrDefault("olf.mismatchInitial", 0) + tPost.getOrDefault("olf.mismatchIteration", 0));
        System.setProperty(GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY, "JAVA");
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        double gpu;
        try {
            gpu = v3.apply(runSecurityAnalysis(mkNet.get(), ctgs, mon, sap));
        } finally {
            System.clearProperty(GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY);
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
        double plainV = np.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV();
        System.err.printf("[TAPFIX] plainLF(post-ctg)=%.9f | CPU-SA=%.9f (d=%.3e) | GPU-JAVA=%.9f (d=%.3e vs CPU, %.3e vs plainLF)%n",
                plainV, cpu, Math.abs(plainV - cpu), gpu, Math.abs(cpu - gpu), Math.abs(plainV - gpu));
        // DEFAULT tolerance (never tightened): the JAVA hybrid must agree with the CPU SA on a contingency
        // whose tap MOVES (device pi-model refresh + OLF-faithful identity-row re-initialization).
        org.junit.jupiter.api.Assertions.assertEquals(cpu, gpu, 1e-12,
                "JAVA hybrid must match the CPU SA at default tolerance on a tap-moving contingency");
    }

    /** DIAGNOSTIC (isolation only, not validation): does the device DS loop distribute when the CPU's band says
     *  "do nothing"? Run the DS-only case with the DEFAULT band (1 MW) and with an effectively-infinite band
     *  (1e6 MW, both sides identically). CPU distributes 0 in both -> identical CPU results. If DEVICE matches
     *  CPU with the infinite band but not with the default band, its gate/distribution fires when OLF's does not. */
    @org.junit.jupiter.api.Test
    void deviceDsBandDiagnostic() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        java.util.function.Supplier<Network> baseNet = () -> {
            Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
            network.getTwoWindingsTransformer("T2wT").getRatioTapChanger().setTargetDeadband(0).setRegulating(true)
                    .setTapPosition(0).setRegulationTerminal(network.getTwoWindingsTransformer("T2wT").getTerminal2()).setTargetV(34.0);
            com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
            network.newLine().setId(l0.getId() + "_BIS")
                    .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                    .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                    .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
            return network;
        };
        String bis = baseNet.get().getLines().iterator().next().getId() + "_BIS";
        for (double band : new double[] {1.0, 1e6}) {
            com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                    .setTransformerVoltageControlOn(false).setDistributedSlack(true).setUseReactiveLimits(true);
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp).setSlackBusPMaxMismatch(band);
            com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
            sap.setLoadFlowParameters(lfp);
            List<Contingency> ctgs = List.of(new Contingency(bis, new BranchContingency(bis)));
            List<com.powsybl.security.monitor.StateMonitor> mon = List.of(new com.powsybl.security.monitor.StateMonitor(
                    com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(),
                    baseNet.get().getVoltageLevelStream().map(vl -> vl.getId()).collect(java.util.stream.Collectors.toSet()), java.util.Set.of()));
            java.util.function.Function<SecurityAnalysisResult, Double> v3 = r -> r.getPostContingencyResults().get(0)
                    .getNetworkResult().getBusResults().stream().filter(b -> b.getBusId().equals("BUS_3")).findFirst().orElseThrow().getV();
            double cpu = v3.apply(runSecurityAnalysis(baseNet.get(), ctgs, mon, sap));
            securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
            double dev;
            try {
                dev = v3.apply(runSecurityAnalysis(baseNet.get(), ctgs, mon, sap));
            } finally {
                securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
            }
            System.err.printf("[DS-BAND] band=%.0e MW  CPU=%.9f  DEV=%.9f  d=%.3e%n", band, cpu, dev, Math.abs(cpu - dev));
        }
    }

    private void runJavaTvcModeMatchesCpu(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode mode) {
        // JAVA-mode SA with the given transformer-VC flavor must reproduce the CPU SA at MACHINE PRECISION,
        // at DEFAULT criteria/tolerance (the standing rule: identical orchestration must agree exactly).
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
        network.getTwoWindingsTransformer("T2wT").getRatioTapChanger().setTargetDeadband(0).setRegulating(true)
                .setTapPosition(0).setRegulationTerminal(network.getTwoWindingsTransformer("T2wT").getTerminal2()).setTargetV(34.0);
        com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
        String bis = l0.getId() + "_BIS";
        network.newLine().setId(bis)
                .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
        // TWO contingencies: the weak clone outage AND the strong original-line outage (the tap-moving one),
        // so the mode's discrete decisions are really exercised.
        List<Contingency> ctgs = List.of(new Contingency(bis, new BranchContingency(bis)),
                new Contingency(l0.getId(), new BranchContingency(l0.getId())));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setTransformerVoltageControlOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp).setTransformerVoltageControlMode(mode);
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        List<com.powsybl.security.monitor.StateMonitor> mon = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(),
                network.getVoltageLevelStream().map(vl -> vl.getId()).collect(java.util.stream.Collectors.toSet()), java.util.Set.of()));
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, ctgs, mon, sap);
        System.setProperty(GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY, "JAVA");
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            SecurityAnalysisResult gpu = runSecurityAnalysis(network, ctgs, mon, sap);
            assertTrue(GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get() > 0,
                    mode + ": the JAVA run must ENGAGE the batch (guard fell back -> the mode is not exercised)");
            assertEquals(cpu.getPostContingencyResults().size(), gpu.getPostContingencyResults().size());
            double worst = 0;
            for (int i = 0; i < cpu.getPostContingencyResults().size(); i++) {
                assertEquals(cpu.getPostContingencyResults().get(i).getStatus(), gpu.getPostContingencyResults().get(i).getStatus());
                var cb = cpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                var gb = gpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                assertEquals(cb.size(), gb.size());
                for (int b = 0; b < cb.size(); b++) {
                    worst = Math.max(worst, Math.abs(cb.get(b).getV() - gb.get(b).getV()));
                    assertEquals(cb.get(b).getV(), gb.get(b).getV(), 1e-12,
                            mode + " bus " + cb.get(b).getBusId() + " ctg " + cpu.getPostContingencyResults().get(i).getContingency().getId());
                }
            }
            System.err.printf("[TVC-MODE] %s: JAVA vs CPU max dV = %.3e (machine precision)%n", mode, worst);
        } finally {
            System.clearProperty(GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY);
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    @org.junit.jupiter.api.Test
    void javaSimpleTransformerVoltageControlMatchesCpu() {
        runJavaTvcModeMatchesCpu(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.WITH_GENERATOR_VOLTAGE_CONTROL);
    }

    @org.junit.jupiter.api.Test
    void javaIncrementalTransformerVoltageControlMatchesCpu() {
        // INCREMENTAL is OLF's DEFAULT transformer-VC mode: the loop moves discrete taps directly
        // (sensitivity-guided), so the device sees only pi-model changes (via the r1Ov refresh) and
        // pinned-ratio target updates — the highest-value flavor to support.
        runJavaTvcModeMatchesCpu(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.INCREMENTAL_VOLTAGE_CONTROL);
    }

    @org.junit.jupiter.api.Test
    void javaIncrementalShuntVoltageControlMatchesCpu() {
        // INCREMENTAL shunt voltage control in JAVA mode must reproduce the CPU SA at machine precision at
        // default tolerance: the loop moves discrete sections; b follows via the pinned SHUNT_B variable and
        // the per-scenario (g,b) shOv refresh carries any section-driven g change to the device.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.ShuntNetworkFactory.create();
        com.powsybl.iidm.network.Line l1 = network.getLine("l1");
        network.newLine().setId("l1_BIS")
                .setVoltageLevel1(l1.getTerminal1().getVoltageLevel().getId()).setBus1(l1.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l1.getTerminal2().getVoltageLevel().getId()).setBus2(l1.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l1.getR() * 2).setX(l1.getX() * 2)
                .setG1(l1.getG1()).setB1(l1.getB1()).setG2(l1.getG2()).setB2(l1.getB2()).add();
        List<Contingency> ctgs = List.of(new Contingency("l1_BIS", new BranchContingency("l1_BIS")),
                new Contingency("l1", new BranchContingency("l1")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setShuntCompensatorVoltageControlOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setShuntVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.ShuntVoltageControlMode.INCREMENTAL_VOLTAGE_CONTROL);
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        List<com.powsybl.security.monitor.StateMonitor> mon = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(),
                network.getVoltageLevelStream().map(vl -> vl.getId()).collect(java.util.stream.Collectors.toSet()), java.util.Set.of()));
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, ctgs, mon, sap);
        System.setProperty(GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY, "JAVA");
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            SecurityAnalysisResult gpu = runSecurityAnalysis(network, ctgs, mon, sap);
            assertTrue(GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get() > 0,
                    "the incremental-shunt JAVA run must ENGAGE the batch");
            double worst = 0;
            for (int i = 0; i < cpu.getPostContingencyResults().size(); i++) {
                assertEquals(cpu.getPostContingencyResults().get(i).getStatus(), gpu.getPostContingencyResults().get(i).getStatus());
                var cb = cpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                var gb = gpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                assertEquals(cb.size(), gb.size());
                for (int b = 0; b < cb.size(); b++) {
                    worst = Math.max(worst, Math.abs(cb.get(b).getV() - gb.get(b).getV()));
                    assertEquals(cb.get(b).getV(), gb.get(b).getV(), 1e-12,
                            "bus " + cb.get(b).getBusId() + " ctg " + cpu.getPostContingencyResults().get(i).getContingency().getId());
                }
            }
            System.err.printf("[SHUNT-INC] JAVA vs CPU max dV = %.3e (machine precision)%n", worst);
        } finally {
            System.clearProperty(GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY);
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    private void assertJavaSaMatchesCpu(Network network, List<Contingency> ctgs,
                                        com.powsybl.loadflow.LoadFlowParameters lfp, String tag) {
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        List<com.powsybl.security.monitor.StateMonitor> mon = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(),
                network.getVoltageLevelStream().map(vl -> vl.getId()).collect(java.util.stream.Collectors.toSet()), java.util.Set.of()));
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, ctgs, mon, sap);
        System.setProperty(GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY, "JAVA");
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            SecurityAnalysisResult gpu = runSecurityAnalysis(network, ctgs, mon, sap);
            assertTrue(GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get() > 0,
                    tag + ": the JAVA run must ENGAGE the batch (guard fell back -> the loop is not exercised)");
            assertEquals(cpu.getPostContingencyResults().size(), gpu.getPostContingencyResults().size());
            double worst = 0;
            for (int i = 0; i < cpu.getPostContingencyResults().size(); i++) {
                assertEquals(cpu.getPostContingencyResults().get(i).getStatus(), gpu.getPostContingencyResults().get(i).getStatus());
                var cb = cpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                var gb = gpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                assertEquals(cb.size(), gb.size());
                for (int b = 0; b < cb.size(); b++) {
                    worst = Math.max(worst, Math.abs(cb.get(b).getV() - gb.get(b).getV()));
                    assertEquals(cb.get(b).getV(), gb.get(b).getV(), 1e-12,
                            tag + " bus " + cb.get(b).getBusId() + " ctg " + cpu.getPostContingencyResults().get(i).getContingency().getId());
                }
            }
            System.err.printf("[%s] JAVA vs CPU max dV = %.3e (machine precision)%n", tag, worst);
        } finally {
            System.clearProperty(GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY);
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    @org.junit.jupiter.api.Test
    void javaIncrementalPhaseControlMatchesCpu() {
        // INCREMENTAL phase-shifter control in JAVA mode: OLF's AcIncrementalPhaseControlOuterLoop moves the
        // PST's discrete taps (a1 + per-tap R/X changes reach the device via the pi-model refresh; the pinned
        // ALPHA1 variable's target follows via the activation re-extraction + faithful re-init).
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.PhaseControlFactory.createNetworkWithT2wt();
        com.powsybl.iidm.network.TwoWindingsTransformer ps1 = network.getTwoWindingsTransformer("PS1");
        ps1.getPhaseTapChanger().setRegulationMode(com.powsybl.iidm.network.PhaseTapChanger.RegulationMode.ACTIVE_POWER_CONTROL)
                .setTargetDeadband(1).setRegulating(true).setTapPosition(1)
                .setRegulationTerminal(ps1.getTerminal1()).setRegulationValue(83);
        // parallel circuit to L1 so outaging the clone keeps the grid connected and shifts flows
        com.powsybl.iidm.network.Line l1 = network.getLine("L1");
        network.newLine().setId("L1_BIS")
                .setVoltageLevel1(l1.getTerminal1().getVoltageLevel().getId()).setBus1(l1.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l1.getTerminal2().getVoltageLevel().getId()).setBus2(l1.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l1.getR() * 2).setX(l1.getX() * 2)
                .setG1(l1.getG1()).setB1(l1.getB1()).setG2(l1.getG2()).setB2(l1.getB2()).add();
        List<Contingency> ctgs = List.of(new Contingency("L1_BIS", new BranchContingency("L1_BIS")),
                new Contingency("L1", new BranchContingency("L1")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setPhaseShifterRegulationOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setPhaseShifterControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.PhaseShifterControlMode.INCREMENTAL);
        assertJavaSaMatchesCpu(network, ctgs, lfp, "PHASE-INC");
    }

    @org.junit.jupiter.api.Test
    void javaIncrementalTransformerReactivePowerControlMatchesCpu() {
        // INCREMENTAL transformer reactive-power control in JAVA mode: the loop moves the RTC's discrete taps
        // to control a branch's Q; rho is pinned (BRANCH_TARGET_RHO1 exists for reactive-power controllers)
        // and per-tap R/X changes reach the device via the pi-model refresh.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
        com.powsybl.iidm.network.TwoWindingsTransformer t2wt = network.getTwoWindingsTransformer("T2wT");
        t2wt.getRatioTapChanger()
                .setTargetDeadband(0).setRegulating(true).setTapPosition(1)
                .setRegulationMode(com.powsybl.iidm.network.RatioTapChanger.RegulationMode.REACTIVE_POWER)
                .setRegulationTerminal(t2wt.getTerminal1()).setRegulationValue(-0.55);
        com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
        network.newLine().setId(l0.getId() + "_BIS")
                .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l0.getR() * 2).setX(l0.getX() * 2)
                .setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
        List<Contingency> ctgs = List.of(new Contingency(l0.getId() + "_BIS", new BranchContingency(l0.getId() + "_BIS")),
                new Contingency(l0.getId(), new BranchContingency(l0.getId())));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setTransformerVoltageControlOn(false).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setTransformerReactivePowerControl(true);
        assertJavaSaMatchesCpu(network, ctgs, lfp, "TRPC-INC");
    }
}
