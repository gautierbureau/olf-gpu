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
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import org.junit.jupiter.api.Test;

import java.util.List;

import static com.powsybl.openloadflow.sa.gpu.GpuSaTestNetworks.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GPU-batched security analysis against the CPU: HVDC lines: AC emulation (freeze and limits), set points and HVDC actions.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuSaHvdcTest extends AbstractGpuSaTest {

    GpuSaHvdcTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    /** An AC-emulation HVDC with a converter ON THE SLACK BUS, carrying a flow above the slack-distribution gate.
     *  OLF's slack mismatch (getMismatchP) carries the emulation term. The device's bus-balance buffer has no HVDC
     *  term, and this still matches at 1e-12; adding the term to that buffer was tried and counts the flow twice
     *  (measured: the slack then never settles, outer-loop budget exhausted). Kept as coverage of the shape. The symmetric two-bus network with the emulation's P0 at 2.5 MW (above the 1 MW gate)
     *  and a parallel line to lose. 1e-12 on device. */
    @Test
    void deviceHvdcEmulationOnTheSlackBusEntersTheSlackMismatchMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.HvdcNetworkFactory.createHvdcInAcEmulationInSymetricNetwork();
        network.getHvdcLine("hvdc12").getExtension(com.powsybl.iidm.network.extensions.HvdcAngleDroopActivePowerControl.class).setP0(2.5f);
        com.powsybl.iidm.network.Line l0 = network.getLine("l12");
        network.newLine().setId("l12_BIS")
                .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(0).setB1(0).setG2(0).setB2(0).add();
        List<Contingency> ctgs = List.of(new Contingency("l12_BIS", new BranchContingency("l12_BIS")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setHvdcAcEmulation(true).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("b1"));                     // the converter cs1's bus
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** DEVICE-mode HVDC AC emulation WITH HVDCs present (GpuBatchHvdcAcEmulation): the freeze /
     *  unfreeze phases and the linear<->saturation limit transitions run per scenario around batched
     *  solves — coupling via the hv disabled bit, VSC injections via converter-bus target overrides;
     *  GPU SA == CPU SA at 1e-12. */
    @Test
    void deviceHvdcAcEmulationMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.HvdcNetworkFactory.createHvdcInAcEmulationInSymetricNetwork();
        com.powsybl.iidm.network.Line l12 = network.getLine("l12");
        network.newLine().setId("l12_BIS")
                .setVoltageLevel1(l12.getTerminal1().getVoltageLevel().getId()).setBus1(l12.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l12.getTerminal2().getVoltageLevel().getId()).setBus2(l12.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l12.getR() * 2).setX(l12.getX() * 2)
                .setG1(l12.getG1()).setB1(l12.getB1()).setG2(l12.getG2()).setB2(l12.getB2()).add();
        List<Contingency> ctgs = List.of(new Contingency("l12_BIS", new BranchContingency("l12_BIS")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setHvdcAcEmulation(true).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }

    /** PROBE: contingency ON the AC-emulation HVDC line itself — the coupling must be disabled AND
     *  the station injections removed per scenario; expect CPU routing or exact match, never drift. */
    @Test
    void deviceHvdcContingencyMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.HvdcNetworkFactory.createHvdcInAcEmulationInSymetricNetwork();
        com.powsybl.iidm.network.Line l12 = network.getLine("l12");
        network.newLine().setId("l12_BIS")
                .setVoltageLevel1(l12.getTerminal1().getVoltageLevel().getId()).setBus1(l12.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l12.getTerminal2().getVoltageLevel().getId()).setBus2(l12.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l12.getR() * 2).setX(l12.getX() * 2)
                .setG1(l12.getG1()).setB1(l12.getB1()).setG2(l12.getG2()).setB2(l12.getB2()).add();
        List<Contingency> ctgs = List.of(new Contingency("l12_BIS", new BranchContingency("l12_BIS")),
                new Contingency("hvdc12", new com.powsybl.contingency.HvdcLineContingency("hvdc12")),
                new Contingency("combo-l12bis-hvdc12", new BranchContingency("l12_BIS"),
                        new com.powsybl.contingency.HvdcLineContingency("hvdc12")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setHvdcAcEmulation(true).setDistributedSlack(true).setUseReactiveLimits(true);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }

    /** HVDC contingency BATCHED: stations off the slack bus (4-bus network), the hv element's
     *  per-scenario disabled bit removes the AC-emulation coupling and the stations' losses ride
     *  the generator-outage channel; solo and combined with a (non-islanding) branch outage. */
    @Test
    void deviceHvdcContingencyBatchedMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.HvdcNetworkFactory.createHvdcLinkedByTwoLinesAndSwitch(
                com.powsybl.iidm.network.HvdcConverterStation.HvdcType.VSC);
        // the factory's maxP == P0 == 200 sits exactly ON the saturation boundary — a knife-edge
        network.getHvdcLine("hvdc23").setMaxP(500);
        network.getLineStream().forEach(l -> {
            if (!"l14_BIS".equals(l.getId())) {
                l.setR(1).setX(10);           // the factory's 0.1-OHM lines are nearly superconducting — load them
            }
        });

        com.powsybl.iidm.network.Line l14 = network.getLine("l14");
        network.newLine().setId("l14_BIS")
                .setVoltageLevel1(l14.getTerminal1().getVoltageLevel().getId()).setBus1(l14.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l14.getTerminal2().getVoltageLevel().getId()).setBus2(l14.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(3).setX(25)
                .setG1(l14.getG1()).setB1(l14.getB1()).setG2(l14.getG2()).setB2(l14.getB2()).add();
        // l14_BIS (hvdc coupling LIVE post-contingency) once drifted ~1e-4 kV: the CPU SA freezes AC
        // emulation for EVERY contingency solve by DEFAULT (SA-level startWithFrozenACEmulation=true,
        // absent from the BASE outer-loop list) and the scope keyed on the base list only, skipping the
        // batch's freeze phase. Fixed in GpuSaRunScope (2026-09-05); this scenario now guards it.
        List<Contingency> ctgs = List.of(
                new Contingency("l14_BIS", new BranchContingency("l14_BIS")),
                new Contingency("hvdc23", new com.powsybl.contingency.HvdcLineContingency("hvdc23")),
                new Contingency("combo-l14bis-hvdc23", new BranchContingency("l14_BIS"),
                        new com.powsybl.contingency.HvdcLineContingency("hvdc23")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setHvdcAcEmulation(true).setDistributedSlack(true).setUseReactiveLimits(true);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }

    /** The same HVDC outage with AC emulation OFF — SETPOINT mode. Such a link has NO element in the
     *  equation system at all: AcEquationSystemCreator#createHvdcAcEmulationEquations builds its terms
     *  only for an AC-emulation link and does "nothing to do" otherwise. So there is nothing to mask,
     *  and the whole outage is the two converter stations' injection loss — which the propagation
     *  already hands over in generatorIdsToLose and the generator-outage channel already carries.
     *
     *  <p>The mapper rejected it anyway (MAPPER:hvdc-outage-unsupported), sending every setpoint-mode
     *  HVDC outage to the CPU with both its stations ALREADY resolved: measured on network.xiidm,
     *  LVKR and LVKS composed as gens=2 hvdcs=null and were then thrown away. Letting the mapper skip
     *  a non-AC-emulation link instead of rejecting took that scope from 269/273 to 271/273 with the
     *  answers unchanged (violations 652/652 diff 0).
     *
     *  <p>VERIFIED by restoring the reject and watching the engagement assert below fail. That assert
     *  is the whole point of this test: the ANSWER is right either way, because the CPU fallback
     *  computes it correctly — only the batching changes, so a state-only comparison proves nothing. */
    @Test
    void deviceSetpointHvdcContingencyBatchesThroughTheGeneratorChannel() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.HvdcNetworkFactory.createHvdcLinkedByTwoLinesAndSwitch(
                com.powsybl.iidm.network.HvdcConverterStation.HvdcType.VSC);
        network.getHvdcLine("hvdc23").setMaxP(500);
        network.getLineStream().forEach(l -> l.setR(1).setX(10));
        List<Contingency> ctgs = List.of(
                new Contingency("hvdc23", new com.powsybl.contingency.HvdcLineContingency("hvdc23")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setHvdcAcEmulation(false).setDistributedSlack(true).setUseReactiveLimits(true);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "the setpoint-mode HVDC outage must BATCH — it is the only scenario here, so a CPU "
                        + "fallback would still produce the right answer and the comparison above "
                        + "would pass while proving nothing");
    }

    /** DEVICE-mode HVDC AC emulation with the FREEZE phase active (startWithFrozenACEmulation) AND a
     *  tight pMax forcing LINEAR -> SATURATION transitions; GPU SA == CPU SA at 1e-12. */
    @Test
    void deviceHvdcAcEmulationFreezeAndSaturationMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.HvdcNetworkFactory.createHvdcInAcEmulationInSymetricNetwork();
        // tighten the emulated-power range so the post-outage droop flow SATURATES
        network.getHvdcLine("hvdc12").newExtension(com.powsybl.iidm.network.extensions.HvdcOperatorActivePowerRangeAdder.class)
                .withOprFromCS1toCS2(1.4f).withOprFromCS2toCS1(1.4f).add();
        com.powsybl.iidm.network.Line l12 = network.getLine("l12");
        network.newLine().setId("l12_BIS")
                .setVoltageLevel1(l12.getTerminal1().getVoltageLevel().getId()).setBus1(l12.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l12.getTerminal2().getVoltageLevel().getId()).setBus2(l12.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l12.getR() * 2).setX(l12.getX() * 2)
                .setG1(l12.getG1()).setB1(l12.getB1()).setG2(l12.getG2()).setB2(l12.getB2()).add();
        List<Contingency> ctgs = List.of(new Contingency("l12_BIS", new BranchContingency("l12_BIS")),
                new Contingency("l12", new BranchContingency("l12")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setHvdcAcEmulation(true).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp).setStartWithFrozenACEmulation(true)
                .setSlackBusPMaxMismatch(1e-4);   // MW-scale network: micro-threshold so DS genuinely distributes (both arms)
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }

    /** TWO AC-emulation links between the SAME bus pair (a doubled corridor) whose converter buses
     *  host NO slack participant (like the real doubled 500 MW link: two VSC stations and nothing
     *  else on each end). The two links' converter-row target updates run as two device threads of
     *  one scenario: the freeze/unfreeze increments must compose ATOMICALLY, or one link's frozen
     *  flow stays counted twice after the unfreeze (measured: 0.126 kV on every batched branch
     *  contingency, the base at 1e-10). A converter bus WITH a participant hides the race (the
     *  distributed-slack pass re-derives that row from its anchor) and so does a voltage-controlled
     *  one (only the angles move). GPU SA == CPU SA at 1e-12. */
    @Test
    void deviceDoubledHvdcCorridorUnfreezeMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = doubledHvdcCorridorNetwork();
        List<Contingency> ctgs = List.of(new Contingency("l12_BIS", new BranchContingency("l12_BIS")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setHvdcAcEmulation(true).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }

    @Test
    void deviceConverterOnKilledCouplerChildMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        List<Contingency> ctgs = List.of(new Contingency("bus-1", new com.powsybl.contingency.BusbarSectionContingency("bbs1")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setHvdcAcEmulation(true).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl_a"));
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(converterOnChildSectionNetwork(), ctgs, 1e-12, lfp, true);
        assertEquals(java.util.Set.of(), GpuBatchScopeCensus.planRefusedIds(), "census=" + GpuBatchScopeCensus.snapshot());
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** A setpoint-mode HVDC link whose rectifier bus a section loss islands on its own: OLF zeroes both stations
     *  (processHvdcsWithoutPower), so the inverter's injection leaves the live side - the device must take it off the
     *  survivor's bus. Its twin whose islanded bus also holds a load is NOT isolated in OLF's sense: the link keeps
     *  its power and nothing moves on the live side. Both on device, both at machine precision. */
    @Test
    void deviceSetpointHvdcLosingItsIslandedEndpointMatchesCpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = setpointHvdcSectionsNetwork();
        List<Contingency> ctgs = List.of(
                new Contingency("bus-1", new com.powsybl.contingency.BusbarSectionContingency("bbs1")),
                new Contingency("bus-2", new com.powsybl.contingency.BusbarSectionContingency("bbs2")));
        var pcs = com.powsybl.openloadflow.network.impl.PropagatedContingency.createList(network, ctgs,
                new com.powsybl.openloadflow.network.LfTopoConfig(),
                new com.powsybl.openloadflow.network.impl.PropagatedContingencyCreationParameters());
        assertEquals(java.util.Set.of("c01", "c12", "bs1"), pcs.get(0).getBranchIdsToOpen().keySet(), "the station's breaker opens");
        assertEquals(java.util.Set.of("c12", "bs2"), pcs.get(1).getBranchIdsToOpen().keySet());
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl_a"));
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp, true);
        assertEquals(java.util.Set.of(), GpuBatchScopeCensus.planRefusedIds(), "census=" + GpuBatchScopeCensus.snapshot());
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** The same two links in AC emulation, with a droop pushing their flow past maxP: the base ends SATURATED, and
     *  a saturated link carries its power as converter targetP. OLF's AC-emulation station reports targetP 0 as soon
     *  as a bus of its link is disabled (LfVscConverterStationImpl.getTargetP), so the SURVIVING station loses its
     *  injection under both section losses - the isolated station of link a (bbs1) and the one sharing its bus with
     *  a load on link b (bbs2), where a setpoint link would keep its power. Before, the device kept it in both:
     *  2.6e-2 kV at h. */
    @Test
    void deviceSaturatedAcEmulationHvdcLosingItsIslandedEndpointMatchesCpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        java.util.function.Supplier<Network> saturated = () -> {
            Network n = setpointHvdcSectionsNetwork();
            for (String link : List.of("a", "b")) {
                com.powsybl.iidm.network.HvdcLine h = n.getHvdcLine("hvdc" + link);
                h.setMaxP(45);
                h.newExtension(com.powsybl.iidm.network.extensions.HvdcAngleDroopActivePowerControlAdder.class)
                        .withP0(40).withDroop(180).withEnabled(true).add();
            }
            return n;
        };
        List<Contingency> ctgs = List.of(
                new Contingency("bus-1", new com.powsybl.contingency.BusbarSectionContingency("bbs1")),
                new Contingency("bus-2", new com.powsybl.contingency.BusbarSectionContingency("bbs2")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true).setHvdcAcEmulation(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl_a"));
        // Non-vacuity: the base saturates both links (rectifier at maxP, off the droop line)
        Network base = saturated.get();
        assertTrue(com.powsybl.loadflow.LoadFlow.run(base, lfp).isFullyConverged());
        for (String link : List.of("a", "b")) {
            assertEquals(45, base.getVscConverterStation("cs1" + link).getTerminal().getP(), 1e-9, "link " + link + " saturated");
        }
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(saturated.get(), ctgs, 1e-12, lfp, true);
        assertEquals(java.util.Set.of(), GpuBatchScopeCensus.planRefusedIds(), "census=" + GpuBatchScopeCensus.snapshot());
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** An AC-emulation VSC pinned at its curve limit: OLF reads the curve at the live flow of the link
     *  (LfVscConverterStationImpl.getReactiveLimitsActivePower), so the limit moves with the flow inside the solve.
     *  l13 carries nothing (the limit must stay where the base left it), l12a moves the flow (and the limit). */
    @Test
    void deviceAcEmulationVscCurveLimitFollowsTheFlowMatchesCpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        List<Contingency> ctgs = List.of(new Contingency("l13", new BranchContingency("l13")),
                new Contingency("l12a", new BranchContingency("l12a")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true).setHvdcAcEmulation(true);
        // Non-vacuity: cs2 sits at its curve maximum at the power it delivers
        Network base = acEmulationCurveLimitedVscNetwork();
        assertTrue(com.powsybl.loadflow.LoadFlow.run(base, lfp).isFullyConverged());
        com.powsybl.iidm.network.Terminal t2 = base.getVscConverterStation("cs2").getTerminal();
        assertEquals(5 + 55 * -t2.getP() / 100, -t2.getQ(), 1e-6);
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(acEmulationCurveLimitedVscNetwork(), ctgs, 1e-12, lfp, true);
        assertEquals(java.util.Set.of(), GpuBatchScopeCensus.planRefusedIds(), "census=" + GpuBatchScopeCensus.snapshot());
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** An AC-emulation link whose rectifier sits on the SLACK bus. OLF's slack mismatch is slack.getP() -
     *  slack.getTargetP(): the link's emulation term while it is linear, its station's target P while it is frozen
     *  (the post-contingency solve starts frozen). The device read the branch flows against the base target only:
     *  84 MW of "mismatch" on the frozen first check, a slack pass OLF never ran, and an extra Newton step that
     *  left b2 1.6e-6 pu off. */
    @Test
    void deviceAcEmulationLinkOnTheSlackBusMatchesCpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        List<Contingency> ctgs = List.of(new Contingency("l13", new BranchContingency("l13")),
                new Contingency("l12a", new BranchContingency("l12a")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true).setHvdcAcEmulation(true);
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(acEmulationVscNetwork(false), ctgs, 1e-12, lfp, true);
        assertEquals(java.util.Set.of(), GpuBatchScopeCensus.planRefusedIds(), "census=" + GpuBatchScopeCensus.snapshot());
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** DEVICE-mode HVDC AC emulation with a SATURATED BASE (was a whole-run CPU gate — the pack
     *  assumed every AC-emu HVDC starts LINEAR): maxP 100 MW sits under the ~135 MW unconstrained
     *  droop flow, so the BASE limits loop pins hvdc23 at SAT_1_TO_2 (CPU-probed: base cs2.P = 100).
     *  The orchestrator's base frame composes every status override as (existing override |
     *  baseTargets) - baseD + d(status) and seeds the base statuses/hv bits (padding slots
     *  included). The l4 load outage (NOT a converter bus, so the strip rows are clean) RELIEVES
     *  the corridor to LINEAR (CPU-probed: relieved flow 68.4 < 100) — the once-divergent
     *  base-injection strip; the l14_BIS outage pushes more through the corridor and keeps the
     *  saturation (an l14 outage would ISLAND b3/b4 — the hvdc is not an AC connectivity branch —
     *  the documented islanded-hv-endpoint CPU route). GPU SA == CPU SA at 1e-12. */
    @Test
    void deviceHvdcSaturatedBaseMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.HvdcNetworkFactory.createHvdcLinkedByTwoLinesAndSwitch(
                com.powsybl.iidm.network.HvdcConverterStation.HvdcType.VSC);
        network.getHvdcLine("hvdc23").setMaxP(100);
        network.getLineStream().forEach(l -> l.setR(1).setX(10));
        com.powsybl.iidm.network.Line l14 = network.getLine("l14");
        network.newLine().setId("l14_BIS")
                .setVoltageLevel1(l14.getTerminal1().getVoltageLevel().getId()).setBus1(l14.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l14.getTerminal2().getVoltageLevel().getId()).setBus2(l14.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(3).setX(25)
                .setG1(l14.getG1()).setB1(l14.getB1()).setG2(l14.getG2()).setB2(l14.getB2()).add();
        List<Contingency> ctgs = List.of(new Contingency("l14_BIS", new BranchContingency("l14_BIS")),
                new Contingency("l4", new com.powsybl.contingency.LoadContingency("l4")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setHvdcAcEmulation(true).setDistributedSlack(true).setUseReactiveLimits(true);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "the saturated-base hvdc run must ENGAGE the batch (was a whole-run CPU gate)");
    }

    /** SATURATED base + a LOST LOAD ON THE CONVERTER BUS itself (the symmetric 2-bus network:
     *  the l2 load shares b2 with cs2): the status overrides must COMPOSE onto the load-loss
     *  override already present at that row — a row must never carry two target overrides (the
     *  parallel scatter has no ordering; the first cut duplicated the row and the result depended
     *  on a scatter race). The other converter bus is the SLACK (its P row is the phi identity).
     *  opr 1.0 MW under the ~2.4 MW droop flow pins the base; the l2 outage relieves. 1e-12. */
    @Test
    void deviceHvdcSaturatedBaseConverterBusLoadLossMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.HvdcNetworkFactory.createHvdcInAcEmulationInSymetricNetwork();
        network.getHvdcLine("hvdc12").newExtension(com.powsybl.iidm.network.extensions.HvdcOperatorActivePowerRangeAdder.class)
                .withOprFromCS1toCS2(1.0f).withOprFromCS2toCS1(1.0f).add();
        com.powsybl.iidm.network.Line l12 = network.getLine("l12");
        network.newLine().setId("l12_BIS")
                .setVoltageLevel1(l12.getTerminal1().getVoltageLevel().getId()).setBus1(l12.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l12.getTerminal2().getVoltageLevel().getId()).setBus2(l12.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l12.getR() * 2).setX(l12.getX() * 2)
                .setG1(l12.getG1()).setB1(l12.getB1()).setG2(l12.getG2()).setB2(l12.getB2()).add();
        List<Contingency> ctgs = List.of(new Contingency("l12_BIS", new BranchContingency("l12_BIS")),
                new Contingency("l2", new com.powsybl.contingency.LoadContingency("l2")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setHvdcAcEmulation(true).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp).setSlackBusPMaxMismatch(1e-4);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "the saturated-base hvdc run must ENGAGE the batch (was a whole-run CPU gate)");
    }

    /** An HvdcAction switching AC emulation off (OLF's LfHvdcAction): the link becomes two fixed converter set points
     *  at the flows the strategy's starting state carries (the contingency's solution), not clamped at the link's
     *  limits, and keeps them for the whole strategy load flow. An action that does not say "off" is a no-op in OLF. */
    @Test
    void hvdcActionSwitchingAcEmulationOffMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.HvdcNetworkFactory.createHvdcInAcEmulationInSymetricNetwork();
        com.powsybl.iidm.network.Line l12 = network.getLine("l12");
        network.newLine().setId("l12_BIS")
                .setVoltageLevel1(l12.getTerminal1().getVoltageLevel().getId()).setBus1(l12.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l12.getTerminal2().getVoltageLevel().getId()).setBus2(l12.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l12.getR() * 2).setX(l12.getX() * 2)
                .setG1(l12.getG1()).setB1(l12.getB1()).setG2(l12.getG2()).setB2(l12.getB2()).add();
        List<Contingency> contingencies = List.of(new Contingency("l12_BIS", new BranchContingency("l12_BIS")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.HvdcActionBuilder().withId("acEmuOff").withHvdcId("hvdc12")
                        .withAcEmulationEnabled(false).build(),
                new com.powsybl.action.HvdcActionBuilder().withId("setpointOnly").withHvdcId("hvdc12")
                        .withActivePowerSetpoint(50.0).build());
        assertStrategyMatchesCpu(network, contingencies, strategiesOf("l12_BIS", "acEmuOff", "setpointOnly"), actions, 3);
        // ...and a link whose stations are both off the slack bus (their set points on their own buses' rows)
        Network network2 = com.powsybl.openloadflow.network.HvdcNetworkFactory.createHvdcLinkedByTwoLinesAndSwitch(
                com.powsybl.iidm.network.HvdcConverterStation.HvdcType.VSC);
        network2.getHvdcLine("hvdc23").setMaxP(500);
        network2.getLineStream().forEach(l -> l.setR(1).setX(10));
        com.powsybl.iidm.network.Line l14 = network2.getLine("l14");
        network2.newLine().setId("l14_BIS")
                .setVoltageLevel1(l14.getTerminal1().getVoltageLevel().getId()).setBus1(l14.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l14.getTerminal2().getVoltageLevel().getId()).setBus2(l14.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(3).setX(25)
                .setG1(l14.getG1()).setB1(l14.getB1()).setG2(l14.getG2()).setB2(l14.getB2()).add();
        List<com.powsybl.action.Action> actions2 = List.of(new com.powsybl.action.HvdcActionBuilder().withId("acEmuOff23")
                .withHvdcId("hvdc23").withAcEmulationEnabled(false).build());
        assertStrategyMatchesCpu(network2, List.of(new Contingency("l14_BIS", new BranchContingency("l14_BIS"))),
                strategiesOf("l14_BIS", "acEmuOff23"), actions2, 2);
    }

    /** An HvdcAction on a link that carries no power after its contingency - one of its converter buses isolated, or
     *  the link itself tripped - changes nothing in OLF (its stations are already at zero). The device evaluated the
     *  flow at the pair's start, where an isolated bus sits at its frozen 1 pu / 0 rad, and routed a tripped link to
     *  the CPU. And on a link that does carry power, after a line contingency, the action holds it at its flows. */
    @Test
    void hvdcActionOnALinkWithoutPowerMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.HvdcNetworkFactory.createWithHvdcInAcEmulation();
        network.getHvdcLine("hvdc34").newExtension(com.powsybl.iidm.network.extensions.HvdcAngleDroopActivePowerControlAdder.class)
                .withDroop(180).withP0(0.f).withEnabled(true).add();
        List<Contingency> contingencies = List.of(
                new Contingency("l45+l46", new BranchContingency("l45"), new BranchContingency("l46")),   // b4 holds cs4 alone
                new Contingency("hvdc34", new com.powsybl.contingency.HvdcLineContingency("hvdc34")),
                new Contingency("l12", new BranchContingency("l12")));
        List<com.powsybl.action.Action> actions = List.of(new com.powsybl.action.HvdcActionBuilder().withId("acEmuOff")
                .withHvdcId("hvdc34").withAcEmulationEnabled(false).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = new java.util.ArrayList<>();
        for (Contingency c : contingencies) {
            strategies.add(new com.powsybl.contingency.strategy.OperatorStrategy("s-" + c.getId(),
                    com.powsybl.contingency.ContingencyContext.specificContingency(c.getId()),
                    new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("acEmuOff")));
        }
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 6);
    }
}
