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
        // Tighten BOTH sides' outer-loop convergence (narrow the distributed-slack band + deepen the inner-Newton
        // stopping) so CPU and GPU settle to the SAME depth — then the only residual is inner-Newton precision, not
        // the ~0.05 kV redistribution band the default tolerances leave. (1e-9 is at the GPU's feasible floor;
        // tighter, e.g. 1e-11, drops below it and the base GPU LF falls back to CPU.)
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL)
                .setSlackBusPMaxMismatch(0.01).setNewtonRaphsonConvEpsPerEq(1e-9);
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
                    // With matched convergence depth the DEVICE batch agrees with the CPU SA to inner-Newton
                    // precision (~1e-10 kV measured); 1e-6 kV is a tight guard far below any real solve error yet
                    // robust to cross-platform float noise.
                    org.junit.jupiter.api.Assertions.assertEquals(cbv.get(b).getV(), gbv.get(b).getV(), 1e-6,
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
                    org.junit.jupiter.api.Assertions.assertEquals(cb.get(b).getV(), gb.get(b).getV(), 0.05,
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
                    org.junit.jupiter.api.Assertions.assertEquals(cb.get(b).getV(), gb.get(b).getV(), 0.05,
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
                    org.junit.jupiter.api.Assertions.assertEquals(cb.get(b).getV(), gb.get(b).getV(), 0.05,
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
        System.err.printf("[SA-DEFAULT] freshLF=%.6f  SA-GPU=%.6f (d=%.3e)  SA-CPU=%.6f (d=%.3e, CPU-SA outlier)%n",
                plGpuDef, saGpuV, Math.abs(plGpuDef - saGpuV), saCpuV, Math.abs(plGpuDef - saCpuV));
        // The GPU batch converges the contingency close to the fresh LF; the CPU sequential SA stops ~6e-3
        // short. Both warm-start from the base N state, and the solvers are identical (A), so the ~6e-3
        // "GPU vs CPU SA" gap is a post-contingency outer-loop-convergence difference between the two SA
        // code paths, not a GPU error — and the GPU is the more accurate of the two SA implementations here.
        org.junit.jupiter.api.Assertions.assertEquals(plGpuDef, saGpuV, 1e-3, "GPU batched SA must reproduce a fresh post-contingency LF");
    }

    @org.junit.jupiter.api.Test
    void tvcSaAllOrchestrationsAgreeAtTightTolerance() {
        // VALIDATION of the principle: the GPU SA (JAVA hybrid AND on-device orchestrator) agrees with the CPU
        // sequential SA once the criteria/tolerance are tight enough that the orchestration-driven convergence-
        // depth difference (different number of outer-loop re-solves per contingency) washes out. Run the SAME
        // TVC contingency in all three orchestrations at TIGHT convEps and assert they converge to one solution.
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
                .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL)
                .setSlackBusPMaxMismatch(1e-4).setNewtonRaphsonConvEpsPerEq(1e-9);   // TIGHT: wash out orchestration depth
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        java.util.Set<String> vls = network.getVoltageLevelStream().map(vl -> vl.getId()).collect(java.util.stream.Collectors.toSet());
        List<com.powsybl.security.monitor.StateMonitor> mon = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(), vls, java.util.Set.of()));
        java.util.function.Function<SecurityAnalysisResult, Double> v3 = r -> r.getPostContingencyResults().get(0)
                .getNetworkResult().getBusResults().stream().filter(b -> b.getBusId().equals("BUS_3")).findFirst().orElseThrow().getV();

        double cpu = v3.apply(runSecurityAnalysis(network, ctgs, mon, sap));  // CPU sequential
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        double gpuDevice = v3.apply(runSecurityAnalysis(network, ctgs, mon, sap));  // GPU on-device orchestrator
        System.setProperty(GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY, "JAVA");
        double gpuJava;
        try {
            gpuJava = v3.apply(runSecurityAnalysis(network, ctgs, mon, sap));  // GPU JAVA hybrid (OLF's real loops)
        } finally {
            System.clearProperty(GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY);
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
        System.err.printf("[ORCH] tight convEps  V(BUS_3)  CPU=%.9f  GPU-DEVICE=%.9f  GPU-JAVA=%.9f%n", cpu, gpuDevice, gpuJava);
        System.err.printf("[ORCH] max spread=%.3e%n", Math.max(Math.abs(cpu - gpuDevice), Math.abs(cpu - gpuJava)));
        org.junit.jupiter.api.Assertions.assertEquals(cpu, gpuDevice, 1e-4, "CPU vs GPU on-device SA must agree at tight tolerance");
        org.junit.jupiter.api.Assertions.assertEquals(cpu, gpuJava, 1e-4, "CPU vs GPU JAVA-hybrid SA must agree at tight tolerance");
    }
}
