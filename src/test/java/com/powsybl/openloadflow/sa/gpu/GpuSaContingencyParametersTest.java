/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.contingency.BranchContingency;
import com.powsybl.contingency.Contingency;
import com.powsybl.contingency.GeneratorContingency;
import com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.openloadflow.CommonTestConfig;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.sa.extensions.ContingencyLoadFlowParameters;
import org.junit.jupiter.api.Test;

import java.util.List;

import static com.powsybl.openloadflow.sa.gpu.GpuSaTestNetworks.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * A contingency carrying its own load flow parameters (ContingencyLoadFlowParameters) runs with the outer loops OLF
 * builds for it (AcSecurityAnalysis.applySpecificContingencyParameters), on the device, in a solve of its own beside the
 * run's: tasks #89. Each test sets one override and compares with OLF at 1e-12, the contingency answered on the device.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuSaContingencyParametersTest extends AbstractGpuSaTest {

    GpuSaContingencyParametersTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    private static Contingency withOwnParameters(Contingency contingency, ContingencyLoadFlowParameters parameters) {
        contingency.addExtension(ContingencyLoadFlowParameters.class, parameters);
        return contingency;
    }

    private static void assertOnDevice(String... contingencyIds) {
        for (String id : contingencyIds) {
            assertTrue(GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_IDS.contains(id), id + " on the device");
        }
    }

    /** l1_BIS without the shunt control loop: OLF holds SHUNT at its base susceptance, which the run's loops would
     *  have moved (b3 390.929 kV where OLF has 395.710 kV, when the device ran it with the run's loops). */
    @Test
    void ownOuterLoops() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.ShuntNetworkFactory.create();
        com.powsybl.iidm.network.Line l1 = network.getLine("l1");
        network.newLine().setId("l1_BIS")
                .setVoltageLevel1(l1.getTerminal1().getVoltageLevel().getId()).setBus1(l1.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l1.getTerminal2().getVoltageLevel().getId()).setBus2(l1.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l1.getR() * 2).setX(l1.getX() * 2).add();
        List<Contingency> ctgs = List.of(new Contingency("l1", new BranchContingency("l1")),
                withOwnParameters(new Contingency("l1_BIS", new BranchContingency("l1_BIS")),
                        new ContingencyLoadFlowParameters().setOuterLoopNames(List.of("DistributedSlack", "ReactiveLimits"))));
        LoadFlowParameters lfp = new LoadFlowParameters()
                .setShuntCompensatorVoltageControlOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
        OpenLoadFlowParameters.create(lfp)
                .setShuntVoltageControlMode(OpenLoadFlowParameters.ShuntVoltageControlMode.WITH_GENERATOR_VOLTAGE_CONTROL);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertOnDevice("l1", "l1_BIS");
    }

    /** B2-G's loss (40 MW) without the distributed slack: the slack bus takes it. */
    @Test
    void ownDistributedSlack() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        List<Contingency> ctgs = List.of(new Contingency("L2-3-1", new BranchContingency("L2-3-1")),
                withOwnParameters(new Contingency("B2-G", new GeneratorContingency("B2-G")),
                        new ContingencyLoadFlowParameters().setDistributedSlack(false)));
        assertGpuMatchesCpu(network, ctgs, 1e-12, new LoadFlowParameters().setDistributedSlack(true).setUseReactiveLimits(true));
        assertOnDevice("L2-3-1", "B2-G");
    }

    /** B2-G's loss (40 MW) distributed over the loads, the run's over the generators. */
    @Test
    void ownBalanceType() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        List<Contingency> ctgs = List.of(new Contingency("L2-3-1", new BranchContingency("L2-3-1")),
                withOwnParameters(new Contingency("B2-G", new GeneratorContingency("B2-G")),
                        new ContingencyLoadFlowParameters().setBalanceType(LoadFlowParameters.BalanceType.PROPORTIONAL_TO_LOAD)));
        assertGpuMatchesCpu(network, ctgs, 1e-12, new LoadFlowParameters().setDistributedSlack(true).setUseReactiveLimits(true)
                .setBalanceType(LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX));
        assertOnDevice("L2-3-1", "B2-G");
    }

    /** The parameters hold for the contingency's strategies too (the default scope): B2-G's 20 MW more after L2-4-1,
     *  without the distributed slack, on the device in the contingency's solve. */
    @Test
    void ownParametersOfAContingencyAndItsStrategy() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        List<Contingency> ctgs = List.of(new Contingency("L2-3-1", new BranchContingency("L2-3-1")),
                withOwnParameters(new Contingency("L2-4-1", new BranchContingency("L2-4-1")),
                        new ContingencyLoadFlowParameters().setDistributedSlack(false)));
        List<com.powsybl.action.Action> actions = List.of(genAction("up2", "B2-G", true, 20));
        assertStrategyMatchesCpu(network, ctgs, strategiesOf("L2-4-1", "up2"), actions, 3);
        assertOnDevice("L2-3-1", "L2-4-1");
    }

    /** With scope CONTINGENCY_ONLY the strategy runs with the run's loops from the contingency's state, which another
     *  solve holds: the contingency and its strategy run on the CPU (the contingency's state is solved and unused). */
    @Test
    void ownParametersOfAContingencyOnly() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        List<Contingency> ctgs = List.of(new Contingency("L2-3-1", new BranchContingency("L2-3-1")),
                withOwnParameters(new Contingency("L2-4-1", new BranchContingency("L2-4-1")),
                        new ContingencyLoadFlowParameters().setDistributedSlack(false)
                                .setScope(ContingencyLoadFlowParameters.Scope.CONTINGENCY_ONLY)));
        List<com.powsybl.action.Action> actions = List.of(genAction("up2", "B2-G", true, 20));
        assertStrategyMatchesCpu(network, ctgs, strategiesOf("L2-4-1", "up2"), actions, 2);
        assertOnDevice("L2-3-1");
    }

    /** An explicit loop list without FreezingHvdcACEmulation: OLF does not freeze that contingency's AC emulation, while
     *  the run's contingencies start frozen (the security analysis default). */
    @Test
    void ownOuterLoopsWithoutTheHvdcFreeze() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.HvdcNetworkFactory.createHvdcInAcEmulationInSymetricNetwork();
        com.powsybl.iidm.network.Line l0 = network.getLine("l12");
        network.newLine().setId("l12_BIS")
                .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(0).setB1(0).setG2(0).setB2(0).add();
        List<Contingency> ctgs = List.of(new Contingency("l12", new BranchContingency("l12")),
                withOwnParameters(new Contingency("l12_BIS", new BranchContingency("l12_BIS")),
                        new ContingencyLoadFlowParameters().setOuterLoopNames(List.of("DistributedSlack", "ReactiveLimits"))));
        LoadFlowParameters lfp = new LoadFlowParameters().setHvdcAcEmulation(true).setDistributedSlack(true).setUseReactiveLimits(true);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertOnDevice("l12", "l12_BIS");
        // both buses hold their voltage: the link's flow shows in the branches' (the device left the link out of the
        // slack check when no loop froze or limited it, and distributed 2.6 MW where OLF distributes none)
        assertBranchFlowsMatchCpu(network, ctgs, lfp);
    }

    /** Every branch's active power on both sides, after every contingency, GPU against CPU at 1e-9 MW. */
    private void assertBranchFlowsMatchCpu(Network network, List<Contingency> ctgs, LoadFlowParameters lfp) {
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        java.util.Set<String> branches = network.getBranchStream().map(b -> b.getId()).collect(java.util.stream.Collectors.toSet());
        List<com.powsybl.security.monitor.StateMonitor> monitors = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), branches, java.util.Set.of(), java.util.Set.of()));
        com.powsybl.security.SecurityAnalysisResult cpu = runSecurityAnalysis(network, ctgs, monitors, sap);
        com.powsybl.security.SecurityAnalysisResult gpu;
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(),
                new com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            gpu = runSecurityAnalysis(network, ctgs, monitors, sap);
        } finally {
            securityAnalysisProvider = new com.powsybl.openloadflow.sa.OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(),
                    new com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
        for (int i = 0; i < cpu.getPostContingencyResults().size(); i++) {
            String ctg = cpu.getPostContingencyResults().get(i).getContingency().getId();
            var gpuBranches = gpu.getPostContingencyResults().get(i).getNetworkResult();
            for (var c : cpu.getPostContingencyResults().get(i).getNetworkResult().getBranchResults()) {
                var g = gpuBranches.getBranchResult(c.getBranchId());
                assertEquals(c.getP1(), g.getP1(), 1e-9, "p1 " + c.getBranchId() + " after " + ctg);
                assertEquals(c.getP2(), g.getP2(), 1e-9, "p2 " + c.getBranchId() + " after " + ctg);
            }
        }
    }
}
