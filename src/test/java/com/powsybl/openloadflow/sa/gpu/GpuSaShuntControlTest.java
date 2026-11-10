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
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.sa.OpenSecurityAnalysisProvider;
import com.powsybl.security.SecurityAnalysisResult;
import org.junit.jupiter.api.Test;

import java.util.List;

import static com.powsybl.openloadflow.sa.gpu.GpuSaTestNetworks.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * GPU-batched security analysis against the CPU: shunt voltage control, incremental and discrete, and shunt section shifts.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuSaShuntControlTest extends AbstractGpuSaTest {

    GpuSaShuntControlTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    @Test
    void shuntVoltageControlMatchesCpu() {
        // Shunt voltage control + dynamic Q-relocation (§11): the continuous susceptance is a SHUNT_B variable
        // solved in-Newton, the discrete section DECISIONS are made per scenario, and dynamic Q-relocation
        // lets the ONE shared structure follow the shunt control on<->off toggle (a
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
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            SecurityAnalysisResult gpu = runSecurityAnalysis(network, contingencies, monitors, sap);
            assertTrue(GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get() > 0,
                    "the shunt-control run must ENGAGE the batch (guard fell back -> the path is not exercised)");
            assertEquals(cpu.getPostContingencyResults().size(), gpu.getPostContingencyResults().size());
            for (int i = 0; i < cpu.getPostContingencyResults().size(); i++) {
                assertEquals(cpu.getPostContingencyResults().get(i).getStatus(),
                        gpu.getPostContingencyResults().get(i).getStatus());
                var cpuBuses = cpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                var gpuBuses = gpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                assertEquals(cpuBuses.size(), gpuBuses.size());
                for (int b = 0; b < cpuBuses.size(); b++) {
                    org.junit.jupiter.api.Assertions.assertEquals(cpuBuses.get(b).getV(), gpuBuses.get(b).getV(), 1e-12,
                            "shunt-VC bus voltage " + cpuBuses.get(b).getBusId());
                }
            }
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    /** DEVICE-mode IncrementalShuntVoltageControl: in-kernel dV/dB sensitivities + OLF's section
     *  decisions per scenario (IncrementalShuntVoltageControl); GPU SA == CPU SA at 1e-12. */
    @Test
    void deviceIncrementalShuntVoltageControlMatchesCpu() {
        assertIncrementalShuntVoltageControlMatchesCpu();
    }

    /** DEVICE-mode CONTINUOUS shunt VC (WITH_GENERATOR — OLF's DEFAULT shuntVoltageControlMode):
     *  the SimpleTvc two-phase flow for shunts (GpuShuntVoltageControlOuterLoop) — per scenario, the VC-ON
     *  toggle-diff activation solves the continuous b (the controlled bus V held at its target, the bus Q
     *  relocated onto the SHUNT_B row), then OLF's dispatchB rounds b to the closest sections and
     *  the VC-OFF re-solve runs with the identity held there + the section (dG,dB) applied.
     *  The l1 outage reroutes the flow over the doubled-impedance clone (V sag → a different
     *  continuous b → a different dispatched section than the base). GPU SA == CPU SA at 1e-12. */
    @Test
    void deviceSimpleShuntVoltageControlMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = shuntNetworkWithParallelL1();
        List<Contingency> ctgs = List.of(new Contingency("l1_BIS", new BranchContingency("l1_BIS")),
                new Contingency("l1", new BranchContingency("l1")));
        assertGpuMatchesCpu(network, ctgs, 1e-12, continuousShuntParameters());
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "continuous shunt VC must ENGAGE the batch (was a whole-run CPU gate)");
    }

    /** OLF restores the whole network after every contingency; after one the device answered, the GPU arm restores only
     *  the elements that contingency touched. The continuous control's susceptance is a state variable, which the device
     *  state writes into the controller shunt (AcSolverUtil.updateNetwork). l1's outage dispatches SHUNT on another section
     *  than the base (1 instead of 2); l1_BIS then runs on the CPU, its strategy having more conditional stages than the
     *  device composes, and without the shunt control loop (its own outer loops): its Newton holds the susceptance at the
     *  shunt's: the base one in OLF, l1's where the restore left it (1.65 MW of slack distributed where OLF has none). */
    @Test
    void cpuContingencyAfterADeviceOneHoldsTheBaseSusceptance() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Contingency l1Bis = new Contingency("l1_BIS", new BranchContingency("l1_BIS"));
        l1Bis.addExtension(com.powsybl.openloadflow.sa.extensions.ContingencyLoadFlowParameters.class,
                new com.powsybl.openloadflow.sa.extensions.ContingencyLoadFlowParameters()
                        .setOuterLoopNames(List.of("DistributedSlack", "ReactiveLimits")));
        List<com.powsybl.action.Action> actions = List.of(new com.powsybl.action.LoadActionBuilder().withId("ld1Up")
                .withLoadId("ld1").withRelativeValue(true).withActivePowerValue(1).build());
        assertStrategyMatchesCpu(shuntNetworkWithParallelL1(), List.of(new Contingency("l1", new BranchContingency("l1")), l1Bis),
                List.of(fourConditionalStageStrategy("sCpu", "l1_BIS", "ld1Up", "ld1Up")), actions, -1, continuousShuntParameters());
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_TARGETED_RESTORE_COUNT.get(), "l1 answered on the device, its restore targeted");
    }

    /** The incremental control moves SHUNT's sections from where they are, so OLF's restore has to put the sections back
     *  with the susceptance (fixed in OLF, tasks #90): l1's outage takes SHUNT from section 2 to section 1, ld1's loss
     *  alone keeps it on 2, and ld1 after l1 then started from section 1 in OLF (1.9e-5 kV on one bus). The device answers
     *  every contingency from the base sections. */
    @Test
    void deviceIncrementalShuntSectionsAfterAContingencyThatSwitchedOne() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = shuntNetworkWithParallelL1(10);
        network.getShuntCompensator("SHUNT").setTargetV(395).setTargetDeadband(0.1);
        List<Contingency> ctgs = List.of(new Contingency("l1", new BranchContingency("l1")),
                new Contingency("ld1", new com.powsybl.contingency.LoadContingency("ld1")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setShuntCompensatorVoltageControlOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setShuntVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.ShuntVoltageControlMode.INCREMENTAL_VOLTAGE_CONTROL);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "both contingencies on the device");
    }

    /** SHUNT3 and SHUNT3B make one shunt of b3, whose susceptance is the sum of their sections. OLF's restore after
     *  strategy1 put the sum back but left SHUNT3 on strategy1's section, which strategy2's action summed in (fixed in
     *  OLF, tasks #90: b4 at 392.407 kV instead of 399.617 kV). The device composes every pair from the
     *  post-contingency state. */
    @Test
    void deviceShuntActionAfterAnotherOnTheSameBus() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createWithShuntSharedRemoteControl();
        network.getShuntCompensator("SHUNT2").setSectionCount(25);
        network.getShuntCompensator("SHUNT3").setSectionCount(25);
        network.getVoltageLevel("vl3").newShuntCompensator().setId("SHUNT3B").setBus("b3").setConnectableBus("b3")
                .setSectionCount(25).newLinearModel().setMaximumSectionCount(50).setBPerSection(-1E-2).setGPerSection(0.0).add().add();
        List<Contingency> ctgs = List.of(new Contingency("tr2", new com.powsybl.contingency.TwoWindingsTransformerContingency("tr2")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.ShuntCompensatorPositionActionBuilder().withId("shunt3To50").withShuntCompensatorId("SHUNT3")
                        .withSectionCount(50).build(),
                new com.powsybl.action.ShuntCompensatorPositionActionBuilder().withId("shunt3bTo0").withShuntCompensatorId("SHUNT3B")
                        .withSectionCount(0).build());
        SecurityAnalysisResult cpu = assertStrategyMatchesCpu(network, ctgs, strategiesOf("tr2", "shunt3To50", "shunt3bTo0"), actions, 3);
        assertEquals(399.617, cpu.getOperatorStrategyResults().get(1).getConditionalActionsResults().get(0).getNetworkResult()
                .getBusResult("b4").getV(), 1e-3, "strategy2 on its own sections");
    }

    /** ShuntNetworkFactory with l1 doubled by l1_BIS (twice its impedance), so that either outage keeps the grid
     *  connected and moves the voltage SHUNT controls. */
    private static Network shuntNetworkWithParallelL1() {
        return shuntNetworkWithParallelL1(2);
    }

    private static Network shuntNetworkWithParallelL1(double impedanceFactor) {
        Network network = com.powsybl.openloadflow.network.ShuntNetworkFactory.create();
        com.powsybl.iidm.network.Line l1 = network.getLine("l1");
        network.newLine().setId("l1_BIS")
                .setVoltageLevel1(l1.getTerminal1().getVoltageLevel().getId()).setBus1(l1.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l1.getTerminal2().getVoltageLevel().getId()).setBus2(l1.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l1.getR() * impedanceFactor).setX(l1.getX() * impedanceFactor)
                .setG1(l1.getG1()).setB1(l1.getB1()).setG2(l1.getG2()).setB2(l1.getB2()).add();
        return network;
    }

    /** Continuous shunt voltage control (WITH_GENERATOR_VOLTAGE_CONTROL, OLF's default mode): SHUNT_B is a variable. */
    private static com.powsybl.loadflow.LoadFlowParameters continuousShuntParameters() {
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setShuntCompensatorVoltageControlOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setShuntVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.ShuntVoltageControlMode.WITH_GENERATOR_VOLTAGE_CONTROL);
        return lfp;
    }

    /** WITH_GENERATOR shunt voltage control CO-PRESENT with a transformer voltage control (AFTER_GENERATOR,
     *  then INCREMENTAL): the shunt at the 132 kV bus and the transformer at the 33 kV bus both act in
     *  the same run. The former co-presence gate (the host orchestrator chain could not interleave
     *  simultaneously unstable loops) is obsolete now that both are driver phases swept in OLF's
     *  order — lifted 2026-09-15; batched at 1e-12 with the count asserted. */
    @Test
    void deviceSimpleShuntWithTransformerControlMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        for (var mode : new com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode[] {
            com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL,
            com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.INCREMENTAL_VOLTAGE_CONTROL}) {
            Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
            var t2wt = network.getTwoWindingsTransformer("T2wT");
            t2wt.getRatioTapChanger().setTargetDeadband(0).setRegulating(true).setTapPosition(0)
                    .setRegulationTerminal(t2wt.getTerminal2()).setTargetV(34.0);
            network.getVoltageLevel("VL_2").newShuntCompensator().setId("SHUNT_2").setBus("BUS_2").setConnectableBus("BUS_2")
                    .setSectionCount(0).setVoltageRegulatorOn(true).setTargetV(133.0).setTargetDeadband(0.5)
                    .newNonLinearModel()
                    .beginSection().setB(1e-3).setG(0.0).endSection()
                    .beginSection().setB(3e-3).setG(0.0).endSection()
                    .add().add();
            com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
            String bis = l0.getId() + "_BIS";
            network.newLine().setId(bis)
                    .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                    .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                    .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
            List<Contingency> ctgs = List.of(new Contingency(bis, new BranchContingency(bis)),
                    new Contingency(l0.getId(), new BranchContingency(l0.getId())));
            com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                    .setTransformerVoltageControlOn(true).setShuntCompensatorVoltageControlOn(true)
                    .setDistributedSlack(true).setUseReactiveLimits(true);
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp).setTransformerVoltageControlMode(mode)
                    .setShuntVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.ShuntVoltageControlMode.WITH_GENERATOR_VOLTAGE_CONTROL);
            assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
            assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                    "shunt VC co-present with " + mode + " must batch");
        }
    }

    /** A shunt outage must land on ITS pack slot. The pack holds, per bus, the fixed shunt, then the
     *  voltage-controlling one, then the SVC stand-by one; an element map that counted fixed shunts
     *  alone pointed every shunt after a controller shunt one slot too early, so the outage of the
     *  fixed shunt on b4 masked b3's CONTROLLER shunt instead (measured on network.xiidm / LITK:
     *  a 14 MVar reactor lost on an islanded section came off a live 80 MVar one, 1.2 kV).
     *  GPU SA == CPU SA at 1e-12. */
    @Test
    void shuntOutageAfterControllerShuntKeepsItsPackSlotMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.ShuntNetworkFactory.create();
        // b4 hangs off b2 and carries a FIXED 80 MVar capacitor; b3's SHUNT (created before it)
        // is the voltage-controlling one
        var vl4 = network.newSubstation().setId("S4").add().newVoltageLevel().setId("vl4").setNominalV(400)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
        vl4.getBusBreakerView().newBus().setId("b4").add();
        vl4.newLoad().setId("ld4").setConnectableBus("b4").setBus("b4").setP0(40).setQ0(20).add();
        vl4.newShuntCompensator().setId("SHFIX").setConnectableBus("b4").setBus("b4").setSectionCount(1)
                .setVoltageRegulatorOn(false).newLinearModel().setBPerSection(5e-4).setMaximumSectionCount(1).add().add();
        network.newLine().setId("l3").setVoltageLevel1("vl2").setBus1("b2").setVoltageLevel2("vl4").setBus2("b4")
                .setR(1).setX(3).setG1(0).setB1(0).setG2(0).setB2(0).add();
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setShuntCompensatorVoltageControlOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
        assertGpuMatchesCpu(network,
                List.of(new Contingency("SHFIX", new com.powsybl.contingency.ShuntCompensatorContingency("SHFIX"))), 1e-12, lfp);
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
                List.of(new Contingency(shuntId, new com.powsybl.contingency.ShuntCompensatorContingency(shuntId))), 1e-12, true);
    }
}
