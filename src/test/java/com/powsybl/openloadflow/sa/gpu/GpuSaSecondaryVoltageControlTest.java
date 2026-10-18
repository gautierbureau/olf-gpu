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
import org.junit.jupiter.api.Test;

import java.util.List;

import static com.powsybl.openloadflow.sa.gpu.GpuSaTestNetworks.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GPU-batched security analysis against the CPU: secondary voltage control.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuSaSecondaryVoltageControlTest extends AbstractGpuSaTest {

    GpuSaSecondaryVoltageControlTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    /** SecondaryVoltageControl CO-PRESENT with an AFTER_GENERATOR transformer voltage control: the zone
     *  moves GEN_1's target to bring the 132 kV pilot BUS_2 to its target while T2wT drives the 33 kV
     *  BUS_3. SVC is OLF order 4, the transformer loop order 8, and SVC is the last HOST orchestrator on
     *  the chunk path — it runs AFTER the whole device sweep, so each decides on a state the other has
     *  already moved. Measured: 1.7e-3 kV with the control loops in-kernel, 2.4e-1 kV with the
     *  pre-2026-09-15 host-orchestrator chain (silently wrong, its re-solves not composing). Fixed by
     *  running SVC as a DRIVER PAUSE at its own slot: the same decision code, in OLF's order. */
    @Test
    void deviceSecondaryVcWithAfterGeneratorTvcMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
        var t2wt = network.getTwoWindingsTransformer("T2wT");
        t2wt.getRatioTapChanger().setTargetDeadband(0).setRegulating(true).setTapPosition(0)
                .setRegulationTerminal(t2wt.getTerminal2()).setTargetV(34.0);
        network.newExtension(com.powsybl.iidm.network.extensions.SecondaryVoltageControlAdder.class)
                .newControlZone().withName("z1")
                .newPilotPoint().withTargetV(133.5).withBusbarSectionsOrBusesIds(List.of("BUS_2")).add()
                .newControlUnit().withId("GEN_1").add()
                .add().add();
        com.powsybl.iidm.network.Line l0 = network.getLine("LINE_12");
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
                .setSecondaryVoltageControl(true).setMaxPlausibleTargetVoltage(1.6);
        assertGpuMatchesCpu(network, ctgs, 1e-11, lfp);
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "SVC co-present with a control loop must batch (the driver pause keeps OLF's order)");
    }

    /** TWO secondary voltage control zones in one system. OLF assembles one system over every enabled controller of
     *  every zone and replaces each zone's last row by the pilot row built from EVERY controlled bus's sensitivity at
     *  THAT zone's pilot (createJvpp over allControlledBuses): a controller of one zone moves the other zone's pilot
     *  through the network. The device stored each controlled bus's pilot sensitivity for its own zone only, so a
     *  pilot row read the other zone's columns at the other zone's pilot. ieee14: z1 (pilot B10: B6-G, B8-G) and
     *  z2 (pilot B4: B2-G, B3-G). 1e-11 like the single-zone fixtures. */
    @Test
    void deviceTwoZoneSecondaryVoltageControlMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory.create14();
        network.newExtension(com.powsybl.iidm.network.extensions.SecondaryVoltageControlAdder.class)
                .newControlZone().withName("z1")
                .newPilotPoint().withTargetV(13).withBusbarSectionsOrBusesIds(List.of("B10")).add()
                .newControlUnit().withId("B6-G").add()
                .newControlUnit().withId("B8-G").add()
                .add()
                .newControlZone().withName("z2")
                .newPilotPoint().withTargetV(140).withBusbarSectionsOrBusesIds(List.of("B4")).add()
                .newControlUnit().withId("B2-G").add()
                .newControlUnit().withId("B3-G").add()
                .add().add();
        List<String> ctgIds = List.of("L1-2-1", "L2-3-1");
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSecondaryVoltageControl(true).setMaxPlausibleTargetVoltage(1.6);
        // ONE contingency per analysis: OLF's security analysis does not restore a voltage-control target between
        // contingencies (BusState saves no control target), and secondary voltage control moves those targets, so
        // the CPU's second contingency starts from the first one's targets (measured: L1-2-1 after L2-3-1 ends B2 at
        // 128.6831 kV, alone at 128.6782; the device gives 128.6782 in either order).
        for (String id : ctgIds) {
            GpuBatchScopeCensus.reset();
            assertGpuMatchesCpu(network, List.of(new Contingency(id, new BranchContingency(id))), 1e-11, lfp);
            assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "census=" + GpuBatchScopeCensus.snapshot());
        }
    }

    /** DEVICE-mode SecondaryVoltageControl (SecondaryVoltageControlLoop): device-batched dV/dVtarget
     *  sensitivities + OLF's zone system (A/K0/Jk/Jvpp + tiny LU) replayed per scenario on the host;
     *  GPU SA == CPU SA (1e-11: the SVC math amplifies inner-solve ULP noise to ~1e-12). */
    @Test
    void deviceSecondaryVoltageControlMatchesCpu() {
        assertSecondaryVoltageControlMatchesCpu();
    }

    /** DEVICE-mode SecondaryVoltageControl with a REMOTE controller (was rejected by the LOCAL-only
     *  stage-1 scope): B6-G regulates B5 through a line terminal — the zone system's sensitivity
     *  unit row and the moved zone target now live on the CONTROLLED bus B5's pinned V row while
     *  the controller's RL entry is the rlRemote member entry (pilot-referenced switch-back via
     *  rlVtOv). B6-G keeps its stock (pinnable) limits: the singleton remote group batches since
     *  the table-less-free-member gate retired (backlog item 4, folded in here — the singleton's
     *  relocation table is [pilot fallback, member], its only member's pin IS the all-pinned
     *  pilot-vEq corner). B8-G stays local in the same zone (controllers <-> controlled buses
     *  stay a bijection: {B5, B8}). 1e-11 (the established SecondaryVc knife-edge tolerance). */
    @Test
    void deviceSecondaryVoltageControlRemoteMatchesCpu() {
        assertSecondaryVoltageControlRemoteMatchesCpu();
    }
}
