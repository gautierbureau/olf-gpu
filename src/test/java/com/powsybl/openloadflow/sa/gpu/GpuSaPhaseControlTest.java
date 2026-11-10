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
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.List;

import static com.powsybl.openloadflow.sa.gpu.GpuSaTestNetworks.*;
import static org.junit.jupiter.api.Assertions.assertEquals;

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
     *  P via in-kernel dP/da1 sensitivities + OLF's flow terms (GpuIncrementalPhaseControlOuterLoop);
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

    /** A PHASE tap the device's loops move after an outage, seen in the reported FLOWS, not only in the voltages
     *  (assertFlowsMatchCpu): PS1's fixture tap changer has three taps, too coarse for any post-contingency move, so it
     *  gets 21 steps of one degree, each with its own impedance — a move changes the angle and the branch model. The
     *  incremental loop (a flow controller) moves PS1 from tap index 17 to 15 after L1. The continuous loop is run from
     *  three initial taps: the one its base load flow rounds to (7) and two it rounds away from (0, 6). OLF's phase
     *  control target P is a flow term of its own, which kept the admittance of the INITIAL tap after the base rounding
     *  until olf-local-patches refreshed it on every tap change: the device's continuous a1 after L1_BIS then differed
     *  from OLF's by up to two taps (tasks.md #86). */
    @ParameterizedTest
    @CsvSource({"INCREMENTAL, 0", "CONTINUOUS_WITH_DISCRETISATION, 0", "CONTINUOUS_WITH_DISCRETISATION, 6",
        "CONTINUOUS_WITH_DISCRETISATION, 7"})
    void devicePhaseTapMoveIsInTheReportedBranchFlows(OpenLoadFlowParameters.PhaseShifterControlMode mode, int initialTap) {
        Network network = com.powsybl.openloadflow.network.PhaseControlFactory.createNetworkWithT2wt();
        com.powsybl.iidm.network.TwoWindingsTransformer ps1 = network.getTwoWindingsTransformer("PS1");
        com.powsybl.iidm.network.PhaseTapChangerAdder ptc = ps1.newPhaseTapChanger().setLowTapPosition(-10).setTapPosition(initialTap)
                .setRegulationMode(com.powsybl.iidm.network.PhaseTapChanger.RegulationMode.ACTIVE_POWER_CONTROL)
                .setTargetDeadband(1).setRegulating(true).setRegulationTerminal(ps1.getTerminal1()).setRegulationValue(83);
        for (int t = -10; t <= 10; t++) {
            ptc.beginStep().setAlpha(t).setRho(1.0).setR(0.3 * (t + 11)).setX(3.0 * (t + 11)).setG(0).setB(0).endStep();
        }
        ptc.add();
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
        OpenLoadFlowParameters.create(lfp).setPhaseShifterControlMode(mode);
        assertFlowsMatchCpu(network, ctgs, lfp, 2);
    }

    /** The same for the reactive-power control's RHO taps (a flow controller on T2wT): 21 steps from 0.7 to 1.3, each
     *  with its own impedance: the base load flow lands on tap 14 and both outages move it to 15. */
    @Test
    void deviceReactivePowerControlTapMoveIsInTheReportedBranchFlows() {
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
        com.powsybl.iidm.network.TwoWindingsTransformer t2wt = network.getTwoWindingsTransformer("T2wT");
        com.powsybl.iidm.network.RatioTapChangerAdder rtc = t2wt.newRatioTapChanger().setLowTapPosition(-10).setTapPosition(0)
                .setLoadTapChangingCapabilities(true).setTargetDeadband(0).setRegulating(true)
                .setRegulationMode(com.powsybl.iidm.network.RatioTapChanger.RegulationMode.REACTIVE_POWER)
                .setRegulationTerminal(t2wt.getTerminal1()).setRegulationValue(-0.55);
        for (int t = -10; t <= 10; t++) {
            rtc.beginStep().setRho(1.0 + 0.03 * t).setR(0.3 * (t + 11)).setX(3.0 * (t + 11)).setG(0).setB(0).endStep();
        }
        rtc.add();
        com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
        String bis = l0.getId() + "_BIS";
        network.newLine().setId(bis)
                .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l0.getR() * 2).setX(l0.getX() * 2)
                .setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
        List<Contingency> ctgs = List.of(new Contingency(bis, new BranchContingency(bis)),
                new Contingency(l0.getId(), new BranchContingency(l0.getId())));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setTransformerVoltageControlOn(false).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp).setTransformerReactivePowerControl(true);
        assertFlowsMatchCpu(network, ctgs, lfp, 2);
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
     *  solve runs with the a1 row flipped to the controlled P-flow equation (row mode BRANCH_TARGET_P), then the
     *  loop rounds a1 to the closest tap and re-solves with the tap's full pi row fixed
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
     *  CONTROLLER's a1 row (closedBranchPhaseControlSide encodes the controller's a1 column); GPU SA == CPU SA at 1e-12. */
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
     *  term from the controller too, which is why this batches at 1e-12 rather than routing to the CPU.
     *  The base load flow has the PhaseControl loop and no controller the device tables hold: the device owns the
     *  loop with nothing to do, and the base solve stays on the device (it used to pause the driver for OLF's loop). */
    @Test
    void deviceRemoteContinuousPhaseControlCurrentLimiterMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        long fallbacksBefore = com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson.CPU_FALLBACK_COUNT.get();
        assertGpuMatchesCpu(remoteCurrentLimiterNetwork(), remoteCurrentLimiterContingencies(), 1e-12,
                remoteCurrentLimiterParameters());
        assertEquals(fallbacksBefore, com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson.CPU_FALLBACK_COUNT.get(),
                "the base load flow must solve on the device");
    }
}
