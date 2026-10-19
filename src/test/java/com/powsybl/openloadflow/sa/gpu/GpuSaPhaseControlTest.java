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
import com.powsybl.iidm.network.Network;
import com.powsybl.openloadflow.CommonTestConfig;
import org.junit.jupiter.api.Test;

import java.util.List;

import static com.powsybl.openloadflow.sa.gpu.GpuSaTestNetworks.*;

/**
 * GPU-batched security analysis against the CPU: phase control, incremental and continuous, and the transformer reactive power control.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuSaPhaseControlTest extends AbstractGpuSaTest {

    GpuSaPhaseControlTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    /** DEVICE-mode IncrementalPhaseControl (ACTIVE_POWER_CONTROL): a1 taps drive the controlled branch's
     *  P via in-kernel dP/da1 sensitivities + OLF's own flow terms (IncrementalFlowControl);
     *  the tap-moving strong-line outage included, GPU SA == CPU SA at 1e-12. */
    @Test
    void deviceIncrementalPhaseControlMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.PhaseControlFactory.createNetworkWithT2wt();
        com.powsybl.iidm.network.TwoWindingsTransformer ps1 = network.getTwoWindingsTransformer("PS1");
        ps1.getPhaseTapChanger().setRegulationMode(com.powsybl.iidm.network.PhaseTapChanger.RegulationMode.ACTIVE_POWER_CONTROL)
                .setTargetDeadband(1).setRegulating(true).setTapPosition(1)
                .setRegulationTerminal(ps1.getTerminal1()).setRegulationValue(83);
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
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }

    /** DEVICE-mode IncrementalTransformerReactivePowerControl: rho taps drive the controlled branch's Q
     *  via in-kernel dQ/drho sensitivities + OLF's own flow terms; GPU SA == CPU SA at 1e-12. */
    @Test
    void deviceIncrementalTransformerReactivePowerControlMatchesCpu() {
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
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }

    /** DEVICE-mode IncrementalPhaseControl in CURRENT_LIMITER mode: one-sided i > limit checks and the
     *  exceed-A1 tap walk (FirstTapPositionAboveFinder port); GPU SA == CPU SA at 1e-12. */
    @Test
    void deviceIncrementalPhaseControlCurrentLimiterMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.PhaseControlFactory.createNetworkWithT2wt();
        com.powsybl.iidm.network.TwoWindingsTransformer ps1 = network.getTwoWindingsTransformer("PS1");
        ps1.getPhaseTapChanger().setRegulationMode(com.powsybl.iidm.network.PhaseTapChanger.RegulationMode.CURRENT_LIMITER)
                .setTargetDeadband(1).setRegulating(true).setTapPosition(1)
                .setRegulationTerminal(ps1.getTerminal1()).setRegulationValue(100);
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
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }

    /** DEVICE-mode CONTINUOUS phase control (CONTINUOUS_WITH_DISCRETISATION, CONTROLLER): the first
     *  solve runs with the a1 row flipped to the controlled P-flow equation (row mode 3), then the
     *  loop rounds a1 to the closest tap and re-solves with the tap's full pi row pinned
     *  (GpuPhaseControlOuterLoop); GPU SA == CPU SA at 1e-12. */
    @Test
    void deviceContinuousPhaseControlMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.PhaseControlFactory.createNetworkWithT2wt();
        com.powsybl.iidm.network.TwoWindingsTransformer ps1 = network.getTwoWindingsTransformer("PS1");
        ps1.getPhaseTapChanger().setRegulationMode(com.powsybl.iidm.network.PhaseTapChanger.RegulationMode.ACTIVE_POWER_CONTROL)
                .setTargetDeadband(1).setRegulating(true).setTapPosition(1)
                .setRegulationTerminal(ps1.getTerminal1()).setRegulationValue(83);
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
                .setPhaseShifterControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.PhaseShifterControlMode.CONTINUOUS_WITH_DISCRETISATION);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }

    /** DEVICE-mode CONTINUOUS phase control in CURRENT_LIMITER mode: one tap shift per round while
     *  I exceeds the limit, direction from the sign of dI/da1 on OLF's own current term
     *  (GpuPhaseControlOuterLoop LIMITER branch); GPU SA == CPU SA at 1e-12. */
    @Test
    void deviceContinuousPhaseControlCurrentLimiterMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.PhaseControlFactory.createNetworkWithT2wt();
        com.powsybl.iidm.network.TwoWindingsTransformer ps1 = network.getTwoWindingsTransformer("PS1");
        ps1.getPhaseTapChanger().setRegulationMode(com.powsybl.iidm.network.PhaseTapChanger.RegulationMode.CURRENT_LIMITER)
                .setTargetDeadband(1).setRegulating(true).setTapPosition(1)
                .setRegulationTerminal(ps1.getTerminal1()).setRegulationValue(100);
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
                .setPhaseShifterControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.PhaseShifterControlMode.CONTINUOUS_WITH_DISCRETISATION);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }

    /** REMOTE continuous phase control (CONTROLLER mode): PS1's a1 drives the P flow of the PARALLEL
     *  line L2 (controlled branch != controller) — the controlled element re-emits its P block at the
     *  CONTROLLER's a1 row (cbPcSide encodes the controller's a1 column); GPU SA == CPU SA at 1e-12. */
    @Test
    void deviceRemoteContinuousPhaseControlMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.PhaseControlFactory.createNetworkWithT2wt();
        com.powsybl.iidm.network.TwoWindingsTransformer ps1 = network.getTwoWindingsTransformer("PS1");
        ps1.getPhaseTapChanger().setRegulationMode(com.powsybl.iidm.network.PhaseTapChanger.RegulationMode.ACTIVE_POWER_CONTROL)
                .setTargetDeadband(1).setRegulating(true).setTapPosition(1)
                .setRegulationTerminal(network.getLine("L2").getTerminal1()).setRegulationValue(40);
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
                .setPhaseShifterControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.PhaseShifterControlMode.CONTINUOUS_WITH_DISCRETISATION);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }

    /** REMOTE continuous phase control in CURRENT_LIMITER mode: PS1's regulation terminal sits on the
     *  PARALLEL line L2, so the phase control's controlled branch is not its controller. OLF's
     *  PhaseControlOuterLoop is local BY CONSTRUCTION there and not by restriction — changeTapPositions
     *  caps {@code controllerBranch.getI1()} against the target and differentiates that same term for
     *  the shift direction, so the controlled branch never enters the decision. The device builds its I
     *  term from the controller too, which is why this batches at 1e-12 rather than routing to the CPU. */
    @Test
    void deviceRemoteContinuousPhaseControlCurrentLimiterMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertGpuMatchesCpu(remoteCurrentLimiterNetwork(), remoteCurrentLimiterContingencies(), 1e-12,
                remoteCurrentLimiterParameters());
    }
}
