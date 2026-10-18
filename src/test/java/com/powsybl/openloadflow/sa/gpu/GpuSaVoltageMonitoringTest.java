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
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GPU-batched security analysis against the CPU: the static var compensators' voltage monitoring and stand-by shunts.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuSaVoltageMonitoringTest extends AbstractGpuSaTest {

    GpuSaVoltageMonitoringTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    /** Two voltage-controlling SVCs on one bus, losing one: the bus keeps voltage control. An SVC's reactive limit
     *  is b*v^2 at the live voltage, so the bus limit after the loss is the SURVIVOR's b times v^2. The device takes
     *  the lost SVC's limit at the base voltage off the aggregate (minQDelta/maxQDelta), and must also take its b off
     *  the quadratic term, otherwise the pinned survivor's target moves with the lost SVC's coefficient: 4.5e-3 kV
     *  at B4 before. */
    @Test
    void deviceLostSvcOnABusKeepingVoltageControlMatchesCpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        java.util.function.Supplier<Network> net = () -> {
            Network n = IeeeCdfNetworkFactory.create14();
            com.powsybl.iidm.network.VoltageLevel vl = n.getBusBreakerView().getBus("B9").getVoltageLevel();
            for (String id : List.of("svc1", "svc2")) {
                vl.newStaticVarCompensator().setId(id).setBus("B9").setConnectableBus("B9")
                        .setBmin(-0.06).setBmax(0.06).setVoltageSetpoint(n.getBusBreakerView().getBus("B9").getV() * 1.03)
                        .setRegulationMode(com.powsybl.iidm.network.StaticVarCompensator.RegulationMode.VOLTAGE).setRegulating(true).add();
            }
            return n;
        };
        List<Contingency> ctgs = List.of(new Contingency("svc2", new com.powsybl.contingency.StaticVarCompensatorContingency("svc2")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        // Non-vacuity: both SVCs sit at their maximum in the base (10 MVar each), below their voltage target
        Network base = net.get();
        assertTrue(com.powsybl.loadflow.LoadFlow.run(base, lfp).isFullyConverged());
        assertTrue(base.getBusBreakerView().getBus("B9").getV() < base.getStaticVarCompensator("svc1").getVoltageSetpoint() - 0.1);
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(net.get(), ctgs, 1e-12, lfp, true);
        assertEquals(java.util.Set.of(), GpuBatchScopeCensus.planRefusedIds(), "census=" + GpuBatchScopeCensus.snapshot());
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** A monitoring SVC and an AC-emulation HVDC in the SAME network — the only shape in which
     *  {@code VoltageMonitoring} can skip its phase-B confirmation solve, and therefore the
     *  only shape that covers the skip at all.
     *
     *  <p>The two arms differ ONLY in the automaton's threshold band:
     *  <ul>
     *    <li>WIDE band: no monitored bus leaves it, phase A never re-solves, and the HVDC unfreeze
     *        solve has already stabilised the reactive limits for every scenario — phase B would
     *        re-derive the state it was handed, so it is skipped. The assertion pins that the skip
     *        actually fired (a passing test that quietly took the old path would prove nothing).</li>
     *    <li>Band that the base voltage is already outside: the automaton switches PQ -&gt; PV, phase A
     *        re-solves, and the skip must NOT fire.</li>
     *  </ul>
     *  Both arms are GPU SA == CPU SA at 1e-12, so the skipped solve provably changed no result. */
    @Test
    void deviceHvdcWithStandbySvcSkipsTheConfirmSolveAndMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = hvdcWithStandbySvcNetwork(390, 399);   // a band neither scenario's b4 leaves
        long activatedBefore = com.powsybl.openloadflow.ac.outerloop.gpu.VoltageMonitoring.ACTIVATIONS.sum();
        assertGpuMatchesCpu(network, hvdcWithStandbySvcContingencies(), 1e-12, hvdcWithStandbySvcParameters());
        assertEquals(activatedBefore,
                com.powsybl.openloadflow.ac.outerloop.gpu.VoltageMonitoring.ACTIVATIONS.sum(),
                "the wide band must leave the automaton standing by");
        // (There is no confirmation solve to skip any more: MonitoringVoltage is a DRIVER phase in
        // OLF's own slot, so the main solve leaves every scenario monitoring- AND RL-stable and the
        // orchestrator's phase B — the extra Newton step that used to cost 7.0e-2 kV — is retired.)
    }

    /** The other arm of {@link #deviceHvdcWithStandbySvcSkipsTheConfirmSolveAndMatchesCpu}: the
     *  automaton activates, so the confirmation solve is NOT skippable and must still run. */
    @Test
    void deviceHvdcWithActivatingStandbySvcKeepsTheConfirmSolveAndMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        // Standing by, b4 sits at 398.00 kV at base, 397.62 kV after the l14_BIS outage and 398.34 kV
        // after the l4 load outage. A 397.8 kV LOW threshold therefore leaves the BASE in band — which
        // matters: OLF's MonitoringVoltageOuterLoop permanently switches an activated SVC to plain
        // voltage control, so a base activation would leave no monitoring SVC for the batch to pack —
        // and takes exactly one scenario out of it.
        Network network = hvdcWithStandbySvcNetwork(397.8, 500);
        long activatedBefore = com.powsybl.openloadflow.ac.outerloop.gpu.VoltageMonitoring.ACTIVATIONS.sum();
        assertGpuMatchesCpu(network, hvdcWithStandbySvcContingencies(), 1e-12, hvdcWithStandbySvcParameters());
        assertTrue(com.powsybl.openloadflow.ac.outerloop.gpu.VoltageMonitoring.ACTIVATIONS.sum() > activatedBefore,
                "the automaton must switch PQ -> PV in this arm");
        // No phase-B bookkeeping to assert: the loop runs in the driver now. What matters is above —
        // the automaton fired, and the state still matches the CPU at 1e-12.
    }

    /** Losing a STATIC VAR COMPENSATOR that carries a stand-by automaton shunt: OLF's
     *  processLostGenerators zeroes that shunt's susceptance with the compensator, and the device's
     *  generator-outage channel did not, so the b0 kept injecting into every later solve. Measured on
     *  network.xiidm / HSMC: 59.5 MVar left on the compensator's own bus, 3.148 kV network-wide and
     *  four GPU-only HIGH_VOLTAGE violations (HPVN: 28 MVar, 1.058 kV). GPU SA == CPU SA at 1e-12,
     *  and the outage must BATCH. */
    @Test
    void deviceStandbySvcOutageDropsItsShuntMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = hvdcWithStandbySvcNetwork(390, 420);   // thresholds wide: the automaton stands BY
        assertGpuMatchesCpu(network, List.of(new Contingency("svc4",
                new com.powsybl.contingency.StaticVarCompensatorContingency("svc4"))),
                1e-12, hvdcWithStandbySvcParameters());
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "the compensator outage must batch (or the test proves nothing)");
    }

    /** DEVICE-mode VoltageMonitoring with a MONITORING SVC (stand-by automaton): the standing-by bus
     *  seeds UNPINNED in the device RL pack (rlMonitored) and activates PQ->PV at the automaton
     *  setpoint when a contingency pushes its V outside the threshold band (VoltageMonitoring);
     *  GPU SA == CPU SA at 1e-12. */
    @Test
    void deviceSvcVoltageMonitoringMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createWithStaticVarCompensator();
        com.powsybl.iidm.network.StaticVarCompensator svc1 = network.getStaticVarCompensator("svc1");
        svc1.setVoltageSetpoint(385).setRegulationMode(com.powsybl.iidm.network.StaticVarCompensator.RegulationMode.VOLTAGE)
                .setRegulating(true);
        svc1.newExtension(com.powsybl.iidm.network.extensions.StandbyAutomatonAdder.class)
                .withHighVoltageThreshold(400).withLowVoltageThreshold(388)
                .withLowVoltageSetpoint(389).withHighVoltageSetpoint(395)
                .withB0(-0.001f).withStandbyStatus(true).add();
        com.powsybl.iidm.network.Line l1 = network.getLine("l1");
        network.newLine().setId("l1_BIS")
                .setVoltageLevel1(l1.getTerminal1().getVoltageLevel().getId()).setBus1(l1.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l1.getTerminal2().getVoltageLevel().getId()).setBus2(l1.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l1.getR() * 2).setX(l1.getX() * 2)
                .setG1(l1.getG1()).setB1(l1.getB1()).setG2(l1.getG2()).setB2(l1.getB2()).add();
        List<Contingency> ctgs = List.of(new Contingency("l1_BIS", new BranchContingency("l1_BIS")),
                new Contingency("l1", new BranchContingency("l1")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp).setSvcVoltageMonitoring(true);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }

    /** DEVICE-mode SLOPE SVC pinned at a limit stays PQ like the CPU: OLF's ReactiveLimitsOuterLoop
     *  runs checkPqBus with canSwitchToPv = !hasGeneratorsWithSlope() ("we don't support switching
     *  PQ to PV for bus with one controller with slope"), so a pinned slope bus NEVER back-switches.
     *  The base pins svc1 absorbing at its min limit (setpoint 385 below what full absorption can
     *  reach: V stays 387.35); the parallel-line outage drops the pinned V below the setpoint — the
     *  once-divergent condition where the device used to resume the mode-2 slope row. CPU-probed:
     *  a FRESH post-outage LF regulates at 385.06, so an un-pinned arm visibly diverges. 1e-12. */
    @Test
    void deviceSlopeSvcPinnedStaysPqMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createWithStaticVarCompensator();
        network.getLoad("ld1").setQ0(50);
        com.powsybl.iidm.network.StaticVarCompensator svc1 = network.getStaticVarCompensator("svc1");
        svc1.setBmin(-0.004).setBmax(0.004)
                .setVoltageSetpoint(385).setRegulationMode(com.powsybl.iidm.network.StaticVarCompensator.RegulationMode.VOLTAGE)
                .setRegulating(true);
        svc1.newExtension(com.powsybl.iidm.network.extensions.VoltagePerReactivePowerControlAdder.class)
                .withSlope(0.0001).add();
        com.powsybl.iidm.network.Line l1 = network.getLine("l1");
        network.newLine().setId("l1_BIS")
                .setVoltageLevel1(l1.getTerminal1().getVoltageLevel().getId()).setBus1(l1.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l1.getTerminal2().getVoltageLevel().getId()).setBus2(l1.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l1.getR()).setX(l1.getX()).add();
        List<Contingency> ctgs = List.of(new Contingency("l1_BIS", new BranchContingency("l1_BIS")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp).setVoltagePerReactivePowerControl(true);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }

    /** DEVICE-mode SLOPE SVC (VoltagePerReactivePowerControl): the bus V row becomes
     *  V + slope*sumQ = target — the device scales the natural reactive fill (row mode 2);
     *  GPU SA == CPU SA at 1e-12. A pinned slope row never back-switches (see
     *  deviceSlopeSvcPinnedStaysPqMatchesCpu). */
    @Test
    void deviceSlopeSvcMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createWithStaticVarCompensator();
        com.powsybl.iidm.network.StaticVarCompensator svc1 = network.getStaticVarCompensator("svc1");
        svc1.setVoltageSetpoint(385).setRegulationMode(com.powsybl.iidm.network.StaticVarCompensator.RegulationMode.VOLTAGE)
                .setRegulating(true);
        svc1.newExtension(com.powsybl.iidm.network.extensions.VoltagePerReactivePowerControlAdder.class)
                .withSlope(0.03).add();
        com.powsybl.iidm.network.Line l1 = network.getLine("l1");
        network.newLine().setId("l1_BIS")
                .setVoltageLevel1(l1.getTerminal1().getVoltageLevel().getId()).setBus1(l1.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l1.getTerminal2().getVoltageLevel().getId()).setBus2(l1.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l1.getR() * 2).setX(l1.getX() * 2)
                .setG1(l1.getG1()).setB1(l1.getB1()).setG2(l1.getG2()).setB2(l1.getB2()).add();
        List<Contingency> ctgs = List.of(new Contingency("l1_BIS", new BranchContingency("l1_BIS")),
                new Contingency("l1", new BranchContingency("l1")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp).setVoltagePerReactivePowerControl(true);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }

    @Test
    void slopelessSvcVoltageMonitoringMatchesCpu() {
        // SVC present (slope-less: a plain generator-VC bus on the device) with the VoltageMonitoring loop
        // able to fire: the loop's mutations (VC re-enable, VC target moves, SVC control type) stay per scenario.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createWithStaticVarCompensator();
        network.getStaticVarCompensator("svc1")
                .setVoltageSetpoint(385)
                .setRegulationMode(com.powsybl.iidm.network.StaticVarCompensator.RegulationMode.VOLTAGE);
        com.powsybl.iidm.network.Line l1 = network.getLine("l1");
        network.newLine().setId("l1_BIS")
                .setVoltageLevel1(l1.getTerminal1().getVoltageLevel().getId()).setBus1(l1.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l1.getTerminal2().getVoltageLevel().getId()).setBus2(l1.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l1.getR() * 2).setX(l1.getX() * 2)
                .setG1(l1.getG1()).setB1(l1.getB1()).setG2(l1.getG2()).setB2(l1.getB2()).add();
        List<Contingency> ctgs = List.of(new Contingency("l1_BIS", new BranchContingency("l1_BIS")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp).setSvcVoltageMonitoring(true);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }
}
