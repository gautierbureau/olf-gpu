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
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.GpuBatchScopeCensus;
import org.junit.jupiter.api.Test;

import java.util.List;

import static com.powsybl.openloadflow.sa.gpu.GpuSaTestNetworks.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * GPU-batched security analysis against the CPU: remote and merged generator voltage controls: member PV -> PQ switches, pilots, split and re-merged groups.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuSaRemoteVoltageControlTest extends AbstractGpuSaTest {

    GpuSaRemoteVoltageControlTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    /** Losing the only generator of one section of a zero-impedance-MERGED voltage control. b1 and b2 are joined by
     *  a zero-impedance line, so OLF merges g1's and g2's controls: one stays MAIN, the other becomes DEPENDENT and
     *  the group shares its reactive power through DISTR_Q. Losing either generator leaves the other controlling the
     *  merged group. The plan used to ask the section's OWN control, which for the DEPENDENT one is a one-member
     *  group with no survivor, and refused the outage as DISTR_Q_LAST_CONTROLLER - 285 of rte6515's full-list
     *  generator outages (busbar retained switches put most generators alone on a coupled section). */
    @Test
    void deviceMergedControlLosingOneSectionsGeneratorMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = Network.create("merged-gen-vc-sa", "code");
        com.powsybl.iidm.network.Bus[] b = new com.powsybl.iidm.network.Bus[4];
        for (int i = 0; i < 4; i++) {
            com.powsybl.iidm.network.VoltageLevel vl = network.newSubstation().setId("s" + i).setCountry(com.powsybl.iidm.network.Country.FR).add()
                    .newVoltageLevel().setId("b" + i + "_vl").setNominalV(225).setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
            b[i] = vl.getBusBreakerView().newBus().setId("b" + i).add();
        }
        java.util.function.BiConsumer<Integer, double[]> gen = (i, pv) -> b[i].getVoltageLevel().newGenerator()
                .setId("g" + i).setBus("b" + i).setConnectableBus("b" + i)
                .setMinP(0).setMaxP(2 * pv[0]).setTargetP(pv[0]).setTargetV(pv[1]).setVoltageRegulatorOn(true).add();
        gen.accept(0, new double[] {60, 230});
        gen.accept(1, new double[] {30, 228});
        gen.accept(2, new double[] {30, 228});
        b[3].getVoltageLevel().newLoad().setId("ld3").setBus("b3").setConnectableBus("b3").setP0(110).setQ0(40).add();
        java.util.function.BiConsumer<String, double[]> line = (id, ends) -> network.newLine().setId(id)
                .setBus1("b" + (int) ends[0]).setConnectableBus1("b" + (int) ends[0])
                .setBus2("b" + (int) ends[1]).setConnectableBus2("b" + (int) ends[1])
                .setR(ends[2]).setX(ends[3]).add();
        line.accept("l03", new double[] {0, 3, 1, 10});
        line.accept("l13", new double[] {1, 3, 1, 12});
        line.accept("l23", new double[] {2, 3, 1, 14});
        line.accept("l12", new double[] {1, 2, 0, 0});     // zero impedance: merges the controls of g1 and g2
        List<Contingency> ctgs = List.of(
                new Contingency("g1", new com.powsybl.contingency.GeneratorContingency("g1")),
                new Contingency("g2", new com.powsybl.contingency.GeneratorContingency("g2")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(false).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("b0_vl"));
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "both generator outages must run on the DEVICE - one of them loses a DEPENDENT control's section");
    }

    /** The merged-control fixture with TIGHT reactive limits: after losing one section's generator, the SURVIVING
     *  controller cannot hold the group's voltage and switches to PQ at its max. Every controller of the group is then PQ,
     *  so the group's voltage control is gone and the lost section's own reactive balance must be enforced on its own
     *  row (rte6515 full list: GEN-5997#0, GEN-6003#0/#1, GEN-6112#0, GEN-6157#0, GEN-6194#0/#1, GEN-6229#0,
     *  GEN-6231#0 - rejected at runtime once the group had no enabled controller left, replayed on the CPU). */
    @Test
    void deviceMergedControlSurvivorSwitchesToPqAfterLosingOneSectionMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = Network.create("merged-gen-vc-sa", "code");
        com.powsybl.iidm.network.Bus[] b = new com.powsybl.iidm.network.Bus[4];
        for (int i = 0; i < 4; i++) {
            com.powsybl.iidm.network.VoltageLevel vl = network.newSubstation().setId("s" + i).setCountry(com.powsybl.iidm.network.Country.FR).add()
                    .newVoltageLevel().setId("b" + i + "_vl").setNominalV(225).setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
            b[i] = vl.getBusBreakerView().newBus().setId("b" + i).add();
        }
        java.util.function.BiConsumer<Integer, double[]> gen = (i, pv) -> b[i].getVoltageLevel().newGenerator()
                .setId("g" + i).setBus("b" + i).setConnectableBus("b" + i)
                .setMinP(0).setMaxP(2 * pv[0]).setTargetP(pv[0]).setTargetV(pv[1]).setVoltageRegulatorOn(true).add();
        gen.accept(0, new double[] {60, 230});
        gen.accept(1, new double[] {30, 236});           // above g0's 230: the group must export Q
        gen.accept(2, new double[] {30, 236});
        network.getGenerator("g0").newMinMaxReactiveLimits().setMinQ(-50).setMaxQ(50).add();   // takes the rest
        for (int i = 1; i <= 2; i++) {                     // both group units switch to PQ already in the BASE
            network.getGenerator("g" + i).newMinMaxReactiveLimits().setMinQ(-5).setMaxQ(5).add();
        }
        b[3].getVoltageLevel().newLoad().setId("ld3").setBus("b3").setConnectableBus("b3").setP0(110).setQ0(45).add();
        java.util.function.BiConsumer<String, double[]> line = (id, ends) -> network.newLine().setId(id)
                .setBus1("b" + (int) ends[0]).setConnectableBus1("b" + (int) ends[0])
                .setBus2("b" + (int) ends[1]).setConnectableBus2("b" + (int) ends[1])
                .setR(ends[2]).setX(ends[3]).add();
        line.accept("l03", new double[] {0, 3, 1, 10});
        line.accept("l13", new double[] {1, 3, 1, 12});
        line.accept("l23", new double[] {2, 3, 1, 14});
        line.accept("l12", new double[] {1, 2, 0, 0});     // zero impedance: merges the controls of g1 and g2
        List<Contingency> ctgs = List.of(
                new Contingency("g1", new com.powsybl.contingency.GeneratorContingency("g1")),
                new Contingency("g2", new com.powsybl.contingency.GeneratorContingency("g2")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(false).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("b0_vl"));
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "both outages must run on the DEVICE, survivor switched to PQ included");
    }

    /** The same merged control with only ONE section switched to PQ in the base: g2's +-5 MVAr leave it at MAX_Q while g1
     *  (+-150 MVAr) keeps controlling. Losing g2 leaves the group's control LIVE on b1, and b2 - already PQ, its
     *  balance on its coupler's DUMMY_Q row - gets OLF's recomputed target (no generator left: -loadQ = 0) on that
     *  row; losing g1 kills the control (the survivor b2 is PQ) and b1's own balance takes over. */
    @Test
    void deviceMergedControlSwitchedToPqInBaseMemberLosesItsGeneratorWhileTheSurvivorControlsMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = Network.create("merged-gen-vc-sa", "code");
        com.powsybl.iidm.network.Bus[] b = new com.powsybl.iidm.network.Bus[4];
        for (int i = 0; i < 4; i++) {
            com.powsybl.iidm.network.VoltageLevel vl = network.newSubstation().setId("s" + i).setCountry(com.powsybl.iidm.network.Country.FR).add()
                    .newVoltageLevel().setId("b" + i + "_vl").setNominalV(225).setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
            b[i] = vl.getBusBreakerView().newBus().setId("b" + i).add();
        }
        java.util.function.BiConsumer<Integer, double[]> gen = (i, pv) -> b[i].getVoltageLevel().newGenerator()
                .setId("g" + i).setBus("b" + i).setConnectableBus("b" + i)
                .setMinP(0).setMaxP(2 * pv[0]).setTargetP(pv[0]).setTargetV(pv[1]).setVoltageRegulatorOn(true).add();
        gen.accept(0, new double[] {60, 230});
        gen.accept(1, new double[] {30, 236});           // above g0's 230: the group must export Q
        gen.accept(2, new double[] {30, 236});
        network.getGenerator("g0").newMinMaxReactiveLimits().setMinQ(-150).setMaxQ(150).add();   // the slack unit stays PV
        network.getGenerator("g1").newMinMaxReactiveLimits().setMinQ(-150).setMaxQ(150).add();   // keeps controlling (base q 1.05 pu)
        network.getGenerator("g2").newMinMaxReactiveLimits().setMinQ(-5).setMaxQ(5).add();     // switched to PQ in the BASE
        for (int i = 1; i <= 2; i++) {                     // EQUAL keys (not range-proportional): b2's half of the
            network.getGenerator("g" + i).newExtension(com.powsybl.iidm.network.extensions.CoordinatedReactiveControlAdder.class)
                    .withQPercent(50).add();               // group's ~1.1 pu is far over its 0.05 limit, b1's is not
        }
        b[3].getVoltageLevel().newLoad().setId("ld3").setBus("b3").setConnectableBus("b3").setP0(110).setQ0(45).add();
        java.util.function.BiConsumer<String, double[]> line = (id, ends) -> network.newLine().setId(id)
                .setBus1("b" + (int) ends[0]).setConnectableBus1("b" + (int) ends[0])
                .setBus2("b" + (int) ends[1]).setConnectableBus2("b" + (int) ends[1])
                .setR(ends[2]).setX(ends[3]).add();
        line.accept("l03", new double[] {0, 3, 1, 10});
        line.accept("l13", new double[] {1, 3, 1, 12});
        line.accept("l23", new double[] {2, 3, 1, 14});
        line.accept("l12", new double[] {1, 2, 0, 0});     // zero impedance: merges the controls of g1 and g2
        List<Contingency> ctgs = List.of(
                new Contingency("g1", new com.powsybl.contingency.GeneratorContingency("g1")),
                new Contingency("g2", new com.powsybl.contingency.GeneratorContingency("g2")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(false).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("b0_vl"));
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "both outages must run on the DEVICE");
    }

    /** The same merged control with NOTHING switched to PQ in the base: equal keys give each section half the group's
     *  reactive power (~2 MVAr each), within their +-3 MVAr limits. Losing either section leaves the whole
     *  share (~4 MVAr) to the survivor, which switches to PQ at MAX_Q AT RUNTIME - the shape of rte6515's GEN-6231#0
     *  (VL-6231_1 PV in the base, rejected at runtime once its group had no enabled controller left). */
    @Test
    void deviceMergedControlSurvivorSwitchesToPqAtRuntimeAfterLosingOneSectionMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = Network.create("merged-gen-vc-sa", "code");
        com.powsybl.iidm.network.Bus[] b = new com.powsybl.iidm.network.Bus[4];
        for (int i = 0; i < 4; i++) {
            com.powsybl.iidm.network.VoltageLevel vl = network.newSubstation().setId("s" + i).setCountry(com.powsybl.iidm.network.Country.FR).add()
                    .newVoltageLevel().setId("b" + i + "_vl").setNominalV(225).setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
            b[i] = vl.getBusBreakerView().newBus().setId("b" + i).add();
        }
        java.util.function.BiConsumer<Integer, double[]> gen = (i, pv) -> b[i].getVoltageLevel().newGenerator()
                .setId("g" + i).setBus("b" + i).setConnectableBus("b" + i)
                .setMinP(0).setMaxP(2 * pv[0]).setTargetP(pv[0]).setTargetV(pv[1]).setVoltageRegulatorOn(true).add();
        gen.accept(0, new double[] {60, 230});
        gen.accept(1, new double[] {30, 228});
        gen.accept(2, new double[] {30, 228});
        network.getGenerator("g0").newMinMaxReactiveLimits().setMinQ(-150).setMaxQ(150).add();   // the slack unit stays PV
        for (int i = 1; i <= 2; i++) {                     // above a half share (~2 MVAr), below the whole (~4)
            network.getGenerator("g" + i).newMinMaxReactiveLimits().setMinQ(-3).setMaxQ(3).add();
        }
        for (int i = 1; i <= 2; i++) {                     // EQUAL keys (not range-proportional): b2's half of the
            network.getGenerator("g" + i).newExtension(com.powsybl.iidm.network.extensions.CoordinatedReactiveControlAdder.class)
                    .withQPercent(50).add();               // group's ~1.1 pu is far over its 0.05 limit, b1's is not
        }
        b[3].getVoltageLevel().newLoad().setId("ld3").setBus("b3").setConnectableBus("b3").setP0(110).setQ0(45).add();
        java.util.function.BiConsumer<String, double[]> line = (id, ends) -> network.newLine().setId(id)
                .setBus1("b" + (int) ends[0]).setConnectableBus1("b" + (int) ends[0])
                .setBus2("b" + (int) ends[1]).setConnectableBus2("b" + (int) ends[1])
                .setR(ends[2]).setX(ends[3]).add();
        line.accept("l03", new double[] {0, 3, 1, 10});
        line.accept("l13", new double[] {1, 3, 1, 12});
        line.accept("l23", new double[] {2, 3, 1, 14});
        line.accept("l12", new double[] {1, 2, 0, 0});     // zero impedance: merges the controls of g1 and g2
        List<Contingency> ctgs = List.of(
                new Contingency("g1", new com.powsybl.contingency.GeneratorContingency("g1")),
                new Contingency("g2", new com.powsybl.contingency.GeneratorContingency("g2")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(false).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("b0_vl"));
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "both outages must run on the DEVICE");
    }

    /** PROBE: remote generator voltage control (DISTR_Q — three generators regulating one pilot bus)
     *  under batched SA: parallel-line outage keeps all controllers live. DS+RL on. */
    /** The group's FREE member loses its last voltage-controlling generator while TWO others survive,
     *  on a group whose pilot is NOT one of its controllers. OLF moves the free role — it deactivates
     *  the new first-enabled controller's DISTR_Q and re-activates the dead member's own reactive
     *  balance — and the device follows because the pilot HAS a relocation table here: the in-kernel
     *  mode scan re-picks the carrier. The pilot and the dead member both carry reactive load, so a
     *  mis-assembled row would show. This is the shape rte6515 GEN-5958 has (3 members, 2 survivors,
     *  3.8e-11), and the reason the gate's exclusion is narrowed to pilot-IS-controller groups: an
     *  earlier, wider rule refused this one too and cost a contingency for nothing. */
    @Test
    void deviceRemoteGroupLosingFreeMemberMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory
                .createWithGeneratorRemoteControl();
        network.newLine().setId("L14")
                .setVoltageLevel1("vl1").setBus1("b1").setVoltageLevel2("vl4").setBus2("b4")
                .setR(1.5).setX(40).setG1(0).setB1(0).setG2(0).setB2(0).add();
        for (String id : List.of("g1", "g2", "g3")) {
            network.getGenerator(id).setMaxP(500);
        }
        network.getVoltageLevel("vl1").newLoad()
                .setId("l1q").setBus("b1").setConnectableBus("b1").setP0(10).setQ0(60).add();
        List<Contingency> ctgs = List.of(
                new Contingency("g1", new com.powsybl.contingency.GeneratorContingency("g1")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        assertGpuMatchesCpu(network, ctgs, 1e-11, lfp);
    }

    /** rte6515 GEN-1263 / GEN-1264, reduced: a remote group of exactly TWO controllers whose PILOT is
     *  one of them, and the pilot carries reactive load. Losing either controller deactivates it, and
     *  the survivor's DISTR_Q row renormalizes to qPct = 1 — which IS the deactivated bus's reactive
     *  balance, so the device solves OLF's equation. Its TARGET has to renormalize with it, and the
     *  kernel that maintains DISTR_Q targets runs only inside the reactive-limits phase: with no PV -> PQ switch in
     *  the scenario the row was left solving against the BASE target. The pilot's 60 MVAr load is what
     *  makes the two targets differ (a group whose buses carry no reactive load has base == renormalized,
     *  which is why the earlier gate's "pilot load Q is zero" proxy looked like a rule).
     *
     *  <p>VERIFIED by disabling the seeded targets (GenOutage.seedsDistrQTargets) and watching both
     *  contingencies fail: on rte6515 the same defect is 0.247 kV (GEN-1263) and 0.134 kV (GEN-1264),
     *  and OLF's own residual at the device's state is 0.076 pu on exactly one equation — the
     *  deactivated bus's reactive balance. */
    @Test
    void deviceTwoMemberPilotControllerLosingAControllerMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = twoMemberPilotIsControllerNetwork();
        List<Contingency> ctgs = List.of(
                new Contingency("g1", new com.powsybl.contingency.GeneratorContingency("g1")),
                new Contingency("g2", new com.powsybl.contingency.GeneratorContingency("g2")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        assertGpuMatchesCpu(network, ctgs, 1e-11, lfp);
    }

    /** THE pilot-is-also-a-controller shape, reduced from rte6515's GEN-1263/GEN-1264/GEN-646/GEN-4714
     *  (all four leave the batch through DISTR_Q_LAST_CONTROLLER). Every generator regulates b1, so the
     *  group's PILOT is b1 and b1 is ALSO one of its three controllers — the shape ExtractionColumns
     *  deliberately keeps out of the Q-relocation table, because while it controls it has no reactive
     *  balance of its own to relocate. Losing g1 makes b1 stop controlling while b2 and b3 still hold
     *  its voltage, so b1 ACQUIRES a Q balance mid-scenario, which rides the new free member's row. b1 also
     *  carries reactive load, which is what makes that balance (and its share in b3's target) non-trivial. */
    @Test
    void devicePilotControllerLosingItsOwnGeneratorMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = pilotIsControllerNetwork();
        List<Contingency> ctgs = List.of(
                new Contingency("g1", new com.powsybl.contingency.GeneratorContingency("g1")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        // TWO survivors: the free role moves to b2, which hosts the pilot's reactive balance (its own DISTR_Q
        // suppressed) while b3 keeps a DISTR_Q renormalized over {b2, b3} whose target still carries the pilot's
        // targetQ (-0.6: OLF sums every merged controller's target, keys alone renormalize). The device solved
        // b2's row at 0.5 x -0.6 then at 0 - 0 (OLF's BUS_TARGET_Q of the pilot 0.3 then 0.6 pu off at the
        // device state, OLF_GPU_SA_RESID): the seed-time carrier stamp was gated off by outerLoopCheckActive, and the
        // dynamic limit arrays it reads were first filled in the reactive-limit pass, after the first solve.
        // Exact at 1e-11 since, and the outage must BATCH.
        assertGpuMatchesCpu(network, ctgs, 1e-11, lfp);
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "the outage must batch (or the test proves nothing)");
    }

    /** The SINGLE-survivor variant of the shape above, with the survivor driven past its reactive limit:
     *  g1 and g2 both regulate b1 (the pilot is one of its own controllers), g1 is lost, and g2 — capped at
     *  30 MVar against b1's 60 MVar load — SWITCHES TO PQ. OLF then drops b1's BUS_TARGET_V and solves its BUS_TARGET_Q
     *  (the dead bus's own balance). The device arms the outaged-pilot row flip for exactly this
     *  (GenOutage.flipsPilotRowWithoutEnabledController: mode 4, the own-Q balance the member alternate dest routes
     *  the pilot's q to), but the pilot-self kernel's "control lost" layout ran first and put the row in
     *  mode 1 — an EMPTY row for a pilot-that-is-its-own-controller — so the voltage stayed exactly at the
     *  identity's value. Measured on rte6515 GEN-1522: VL-1522_0 at 1.122 (its target) against OLF's 1.0811,
     *  3.2 kV at VL-1525 and 1.98e-2 MW of distributed power, with identical switch sets on both arms.
     *  GPU SA == CPU SA at 1e-12, and the outage must BATCH. */
    @Test
    void devicePilotLosingItsOwnGeneratorWithASurvivorSwitchingToPqMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = pilotIsControllerSingleSurvivorNetwork();
        List<Contingency> ctgs = List.of(
                new Contingency("g1", new com.powsybl.contingency.GeneratorContingency("g1")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl0_0"));
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "the outage must batch (or the test proves nothing)");
    }

    /** Four sections, the main at the end: losing bbs1 leaves the main alone on one side and a second cluster
     *  {bbs2, bbs3} that OLF re-merges into a NEW group on the other — two groups on one candidate table (tasks
     *  #71, rte6515 BUS-6002_BBS_2 and ten more). Every section's loss is listed (couplers retained), so the same
     *  run also covers the pilot loss (#70) and the leaf loss. */
    @Test
    void deviceSplitGroupBehindSurvivingMainMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertSplitGroupMatchesCpu(4);
    }

    /** The main's pilot section islanded with its generator, ONE remote survivor re-merged onto its own section
     *  (network.xiidm GOOD: GOMV_1 -> GOPN_1): on the device, at 1e-9 kV. The surviving section's identity goes on
     *  its own row (the killed coupler's child), its reactive balance on the free controller's row through its
     *  dependent-pilot pool, and the controller's limit check reads the new pilot. With the "control dies" flip
     *  the device used to hold vg1 at 404.07 kV against OLF's 405.87 (bbs1's balance enforced nowhere). */
    @Test
    void deviceMergedRemoteControlSurvivorReMergesOntoItsOwnSectionMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertRemoteMergedSectionsMatchCpu(2, java.util.Set.of());
    }

    /** Three sections: losing bbs0 re-merges TWO remote survivors onto one control (network.xiidm CXUJ: CXSS_2/_4 ->
     *  CXYW_1) - on the device, the new pilot's balance on the first free controller, the other's row a sharing row
     *  renormalized over the two; losing the middle bbs1 cuts bbs2 off a SURVIVING main and re-merges vg2 onto it -
     *  a second group on the one table, still the CPU's (VC_REMERGE_REMOTE_SUBGROUP); losing bbs2 only renormalizes
     *  the main's group and stays on the device. */
    @Test
    void deviceMergedRemoteControlTwoSurvivorsReMergeMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertRemoteMergedSectionsMatchCpu(3, java.util.Set.of("bus-1"));
    }

    /** The re-merged controller SWITCHES TO PQ: the same two sections, g0 given a wide reactive range so the merged group's
     *  keys (OLF: proportional to the ranges) leave g1 about a tenth of the base duty, and g1 capped at 80 MVAr -
     *  above that base share, below the ~121 MVAr holding its section alone at 404 kV takes once bbs0 and g0 are
     *  gone. OLF re-merges g1 onto bbs1, switches it to PQ at MAX_Q at the first reactive-limit
     *  pass and drops BUS_TARGET_V on bbs1 (no enabled controller): bbs1 solves its own reactive balance and g1
     *  holds its limit (network.xiidm GOOD, GOMV_1 switched to PQ at MAX_Q). On the device the re-pointed limit check
     *  reads bbs1's voltage, the switch to PQ lands on g1's own row, and the dependent-pilot group flips bbs1's identity to
     *  its balance on its own row. GPU SA == CPU SA at 1e-9 kV, batched. */
    @Test
    void deviceMergedRemoteControlReMergedControllerSwitchesToPqMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = remoteMergedSectionsNetwork(2);
        network.getGenerator("g0").newMinMaxReactiveLimits().setMinQ(-3000).setMaxQ(3000).add();
        network.getGenerator("g1").newMinMaxReactiveLimits().setMinQ(-300).setMaxQ(80).add();
        // the BASE must leave g1 free, or the PV -> PQ switch under test is inherited rather than decided in the scenario
        Network base = remoteMergedSectionsNetwork(2);
        base.getGenerator("g0").newMinMaxReactiveLimits().setMinQ(-3000).setMaxQ(3000).add();
        base.getGenerator("g1").newMinMaxReactiveLimits().setMinQ(-300).setMaxQ(80).add();
        com.powsybl.loadflow.LoadFlowParameters baseLfp = new com.powsybl.loadflow.LoadFlowParameters();
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(baseLfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl0"));
        assertTrue(new com.powsybl.loadflow.LoadFlow.Runner(new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory()))
                .run(base, baseLfp).isFullyConverged());
        double baseQ1 = -base.getGenerator("g1").getTerminal().getQ();
        assertTrue(baseQ1 > 0 && baseQ1 < 79.0, "base: g1 must be free (q = " + baseQ1 + " MVAr against an 80 MVAr cap)");
        List<Contingency> ctgs = List.of(new Contingency("bus-0", new com.powsybl.contingency.BusbarSectionContingency("bbs0")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters();
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl0"));
        // the PV -> PQ switch must happen on the CPU, or the test proves nothing about that path
        double vBbs1 = cpuPostContingencyV(network, ctgs, lfp, "vl12");
        assertTrue(Math.abs(vBbs1 - 404.0) > 0.05, "CPU: bbs1 must leave its 404 kV target once g1 switches to PQ, got " + vBbs1);
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(network, ctgs, 1e-9, lfp, true);
        assertEquals(java.util.Set.of(), GpuBatchScopeCensus.planRefusedIds(), "census=" + GpuBatchScopeCensus.snapshot());
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** Two sections with the SURVIVOR's generator switched to PQ in the base: losing the main re-merges a LOCAL control to
     *  a bus OLF has PQ at its limit — the device starts the row as the reactive balance at the frozen target Q and
     *  checks the entry as a local PV bus at the re-merged target, so a switch back to PV restores the identity (rte6515
     *  BUS-6157_BBS_1 switches back to PV then to PQ again at the other limit, BUS-6194_BBS_2 switches back to PV,
     *  BUS-6229_BBS_1 stays). */
    @Test
    void deviceLostPilotLocalSurvivorSwitchedToPqMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertSplitGroupMatchesCpu(2, java.util.Set.of(1));
    }

    /** Three sections with BOTH survivors switched to PQ in the base: the re-merged cluster has no enabled controller, so
     *  OLF keeps BUS_TARGET_V off — plain PQ at the limits, and the device installs no identity while every controller
     *  stays PQ (rte6515 BUS-6194_BBS_1, BUS-6003_BBS_1: OLF never switches them back to PV). */
    @Test
    void deviceLostPilotClusterWithoutEnabledControllerMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertSplitGroupMatchesCpu(3, java.util.Set.of(1, 2));
    }

    /** Three sections, the elected main's generator (bbs1) switched to PQ in the base and bbs2's free: losing bbs0 re-merges
     *  them onto vl12_1, the opened coupler's child. OLF keeps the main PQ at its limit and its BUS_TARGET_V through
     *  bbs2's generator; the device refused the contingency (a re-merged main switched to PQ in the base). The main
     *  starts as its own reactive balance at the frozen target Q and bbs2's sharing row holds the voltage target. */
    @Test
    void deviceLostPilotMainSwitchedToPqWithAnEnabledControllerMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertSplitGroupMatchesCpu(3, java.util.Set.of(1));
    }

    /** Three sections, the middle one WITHOUT voltage control: losing the main re-merges a LOCAL control to bbs2,
     *  whose own V row hosts the LIVE coupler c12's ZERO_V — the identity goes on c12's DUMMY_Q row as a cross
     *  identity on bbs2's V column (rte6515 BUS-6014_BBS_1/2, BUS-6029_BBS_4: VC_REMERGE_RELOC_VROW). */
    @Test
    void deviceLostPilotReMergedUnderLiveCouplerMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertSplitGroupMatchesCpu(3, java.util.Set.of(), java.util.Set.of(1));
    }

    /** The same re-merge with bbs2 joined by couplers only (no line): its reactive balance is all coupler flows, so no
     *  dq/dV term of its own reserves the cross identity's slot on c12's DUMMY_Q row (network.xiidm AJSL and six more:
     *  the row came out empty, a coupler flow was left free, linear-solve failure at the first step). */
    @Test
    void deviceLostPilotReMergedOnCouplerOnlySectionMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertSplitGroupMatchesCpu(3, java.util.Set.of(), java.util.Set.of(1), java.util.Set.of(2));
    }

    /** Four sections with the second uncontrolled: losing the main leaves a REMOTE cluster {bbs2, bbs3} whose elected
     *  main bbs2 sits under the live coupler c12 (rte6515 BUS-6029_BBS_1/2). */
    @Test
    void deviceLostPilotRemoteClusterUnderLiveCouplerMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertSplitGroupMatchesCpu(4, java.util.Set.of(), java.util.Set.of(1));
    }

    /** The same under-live-coupler shape with the re-merged bus SWITCHED TO PQ in the base: the cross row starts as the
     *  balance at the frozen target Q and returns to the cross identity when the bus switches back to PV. */
    @Test
    void deviceLostPilotSwitchedToPqUnderLiveCouplerMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertSplitGroupMatchesCpu(3, java.util.Set.of(2), java.util.Set.of(1));
    }

    /** Five sections: losing bbs2 keeps TWO controllers with the main (bbs0, bbs1 — its DISTR_Q row stays a
     *  sharing row of subgroup 0) and splits off {bbs3, bbs4}. */
    @Test
    void deviceSplitGroupWithTwoControllersOnMainSideMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertSplitGroupMatchesCpu(5);
    }

    @Test
    void deviceLostPilotSectionReMergesSurvivingClusterMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        // bbs2's loss is in the list ONLY so that c12 exists on the LF network: OLF retains exactly the
        // node-breaker switches some listed contingency opens (Networks.retainAndCloseNecessarySwitches
        // resets every other retained flag first), so with bbs0 alone c12 collapses, vl12_1 and vl12_2
        // are ONE bus and the scenario degenerates to the single-bus LOCAL re-merge. Measured: the device
        // extraction then had two vl12 buses and one coupler, and this test passed through the old path.
        assertMergedPilotLossMatchesCpu(lostPilotSectionNetwork(),
                List.of(new Contingency("bus-pilot", new com.powsybl.contingency.BusbarSectionContingency("bbs0")),
                        new Contingency("bus-leaf", new com.powsybl.contingency.BusbarSectionContingency("bbs2"))), 2);
    }

    /** The same loss with the elected main the coupler tree's ROOT (bbs0 has no line, so it cannot root a
     *  tree): vl12_1 keeps its own row and its voltage identity is the explicit ROW_V_IDENTITY, on the very row
     *  the islanded-pilot path also seeds with vl12_1's renormalized DISTR_Q target. rte6515 BUS-2640_BBS_1:
     *  the two overrides raced on the device, the 0.0 seed won in the 6667-contingency batch (V driven to 0,
     *  193.85 MW of distributed active power and +197 violations against OLF) and the 1.034 identity won in a
     *  3-contingency batch. A mode-0 row no longer takes a DISTR_Q seed and GpuBatchOverrides never hands the
     *  device one row twice. */
    @Test
    void deviceLostPilotSectionRootMainExplicitIdentityMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        GpuBatchScopeCensus.reset();
        assertMergedPilotLossMatchesCpu(lostPilotSectionNetwork(false),
                List.of(new Contingency("bus-pilot", new com.powsybl.contingency.BusbarSectionContingency("bbs0")),
                        new Contingency("bus-leaf", new com.powsybl.contingency.BusbarSectionContingency("bbs2"))), 2);
        assertEquals(0L, GpuBatchScopeCensus.snapshot().getOrDefault("HOST:OVERRIDE_CONFLICT", 0L).longValue(),
                "a sound plan writes every row's target once");
    }

    /** The lost pilot was SWITCHED TO PQ at its reactive ceiling in the BASE (g0 capped at 5 MVar). OLF: the base keeps
     *  BUS_TARGET_V on vl12_0 with g1/g2 controlling and g0 PQ at its limit; the contingency then removes the bus
     *  and re-merges the survivors exactly as when the pilot was PV. On the device the islanded pilot's RL
     *  entry kept its base limit type, which the pilot-self gate read as "switched to PQ: its balance rides the carrier" —
     *  it suppressed the carrier member's own DISTR_Q for a balance OLF had removed with the bus. rte6515
     *  BUS-6194_BBS_1 (FAILED where OLF converges, 60 violations lost) and BUS-6003_BBS_1 (0.473 MW). */
    @Test
    void deviceLostSwitchedToPqInBasePilotSectionMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        GpuBatchScopeCensus.reset();
        assertMergedPilotLossMatchesCpu(lostPilotSectionNetwork(true, true),
                List.of(new Contingency("bus-pilot", new com.powsybl.contingency.BusbarSectionContingency("bbs0")),
                        new Contingency("bus-leaf", new com.powsybl.contingency.BusbarSectionContingency("bbs2"))), 2);
        assertEquals(0L, GpuBatchScopeCensus.snapshot().getOrDefault("HOST:OVERRIDE_CONFLICT", 0L).longValue(),
                "a sound plan writes every row's target once");
    }

    /** g1/g2 regulate the pilot section bbs0 REMOTELY; losing bbs0 kills their control (OLF: BUS_TARGET_Q at both,
     *  the survivors stay a coupler chain vl12_1 — c12 — vl12_2). vl12_1's own reactive balance, back on its own
     *  row once c01 is dead, must still carry c12's +dummy_q: the device routed that term through a two-way
     *  alternate selected on the DEAD coupler's dummy row and sent it into that identity (wiped). rte6515
     *  BUS-1359_BBS_1 / BUS-1869_BBS_1: 0.465 / 0.338 kV on the survivors, +-2 violations, in every run. */
    @Test
    void deviceLostPilotWithRemoteSurvivorsOnCouplerChainMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertMergedPilotLossMatchesCpu(lostPilotSectionNetwork(true, false, true),
                List.of(new Contingency("bus-pilot", new com.powsybl.contingency.BusbarSectionContingency("bbs0")),
                        new Contingency("bus-leaf", new com.powsybl.contingency.BusbarSectionContingency("bbs2"))), 2);
    }

    @Test
    void deviceRemoteGroupWithoutEnabledControllerInBaseMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = noEnabledControllerRemoteGroupNetwork();
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl0_0"));
        assertGpuMatchesCpu(network, List.of(new Contingency("l0c1", new com.powsybl.contingency.LineContingency("l0c1"))),
                1e-12, lfp);
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "the batch must precompute with the whole group switched to PQ in the base (or the test proves nothing)");
    }

    /** A remote voltage-control group that LOST CONTROL IN THE BASE and gets it back in a contingency.
     *  The pilot vlp is a pure transit bus (a load, no generator); its two controllers gc1/gc2 hold it
     *  at 412 kV from vlc1/vlc2. In the base a strong 420 kV source next door keeps vlp above target
     *  even with both controllers absorbing at their minQ, so both are switched to PQ at MIN_Q, the group controls
     *  nothing and the pilot's BUS_TARGET_V is inactive. Losing the direct line l0p drops vlp well
     *  below 412: both controllers want back (PQ at MIN_Q, v below target), and OLF restores the
     *  pilot's V equation IN THE SAME PASS it switches them back to PV (updateRemoteVoltageControlEquations).
     *
     *  <p>The device had no such restore for a group that came in lost: the pilot's row rode its own
     *  local reactive-limit entry — an empty [0, 0] range with a limit type GUESSED from where its target
     *  sat — and that entry's own voltage test decides on the pilot's V against the group target with
     *  the guessed type, which is the wrong trigger: for a MAX guess (the pilot carries no load Q, so
     *  the guess lands there, as on network.xiidm/IZUM_0) it needs v ABOVE target and never fires
     *  while the members switch back to PV on v BELOW it. The device then solved a pass with two sharing rows and
     *  no voltage identity. On network.xiidm/LNER that one pass put the distributed slack 0.151 MW off.
     *  Now updatePilotTargetVEquationBatch restores the row from the members' state; the local entry is inert.
     *  With that restore disabled this fails at 407.08 kV against the CPU's 412.00 - the group never
     *  gets its voltage back. Three earlier shapes of this fixture PASSED without the fix and proved
     *  nothing: the members never switched back to PV, because active power through a near-lossless line moves
     *  angles, not magnitudes, and the pilot stayed above target. GPU SA == CPU SA at 1e-12, and the
     *  contingency must batch. */
    @Test
    void deviceRemoteGroupLostInBaseRegainsControlInSamePassMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = Network.create("lost-in-base-remote-group", "code");
        com.powsybl.iidm.network.Substation s = network.newSubstation().setId("s").add();
        for (String vl : List.of("vl0", "vlp", "vlc1", "vlc2")) {
            com.powsybl.iidm.network.VoltageLevel v = s.newVoltageLevel().setId(vl).setNominalV(400)
                    .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
            v.getBusBreakerView().newBus().setId(vl + "_b").add();
        }
        // the strong source: 425 kV one SHORT line from the pilot, so the base holds vlp ABOVE the
        // group's 412 whatever the two controllers absorb
        network.getVoltageLevel("vl0").newGenerator().setId("g0").setBus("vl0_b").setConnectableBus("vl0_b")
                .setMinP(0).setMaxP(2000).setTargetP(800).setTargetV(425).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl0").newLoad().setId("ld0").setBus("vl0_b").setConnectableBus("vl0_b").setP0(100).setQ0(0).add();
        // the pilot: a heavy load with NO reactive part (so the guessed limit type of its empty range
        // is MAX, as on IZUM_0); once the short line is gone it hangs off two LONG paths and sags
        network.getVoltageLevel("vlp").newLoad().setId("ldp").setBus("vlp_b").setConnectableBus("vlp_b").setP0(300).setQ0(0).add();
        for (String c : List.of("c1", "c2")) {
            network.getVoltageLevel("vl" + c).newGenerator().setId("g" + c).setBus("vl" + c + "_b").setConnectableBus("vl" + c + "_b")
                    .setMinP(0).setMaxP(500).setTargetP(120).setTargetQ(0).setTargetV(412).setVoltageRegulatorOn(true)
                    .setRegulatingTerminal(network.getLoad("ldp").getTerminal()).add();
            // enough MAX_Q to hold 412 once the short line is gone: the group must END in control
            network.getGenerator("g" + c).newMinMaxReactiveLimits().setMinQ(-20).setMaxQ(200).add();
        }
        // R matters here: active power through a lossless line moves angles, not magnitudes, and
        // the pilot has to SAG once the short line is gone - the long paths are resistive.
        java.util.function.BiConsumer<String[], double[]> line = (ends, rx) -> network.newLine().setId(ends[0])
                .setVoltageLevel1(ends[1]).setBus1(ends[1] + "_b").setConnectableBus1(ends[1] + "_b")
                .setVoltageLevel2(ends[2]).setBus2(ends[2] + "_b").setConnectableBus2(ends[2] + "_b")
                .setR(rx[0]).setX(rx[1]).setG1(0).setB1(0).setG2(0).setB2(0).add();
        line.accept(new String[] {"l0p", "vl0", "vlp"}, new double[] {0.4, 5});     // the short path that props vlp up
        line.accept(new String[] {"lpc1", "vlp", "vlc1"}, new double[] {20, 50});   // the long ones it falls back on
        line.accept(new String[] {"lpc2", "vlp", "vlc2"}, new double[] {20, 50});
        line.accept(new String[] {"l0c1", "vl0", "vlc1"}, new double[] {20, 50});
        line.accept(new String[] {"l0c2", "vl0", "vlc2"}, new double[] {20, 50});
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl0_0"));
        assertGpuMatchesCpu(network, List.of(new Contingency("l0p", new com.powsybl.contingency.LineContingency("l0p"))),
                1e-12, lfp);
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "the contingency must batch with the group lost in the base (or the test proves nothing)");
    }

    /** A remote group member SWITCHED TO PQ in the base survives the outage that islands another member. Same
     *  network as the base-lost fixture above (both controllers switched to PQ at MIN_Q in the base under a strong
     *  source), but vlc2 hangs off the pilot by ONE line, so losing lpc2 islands gc2 - a member outage
     *  by connectivity, 120 MW of generation gone, the group still lost (gc1 stays PQ).
     *
     *  <p>The generator-outage path re-seeds the group's DISTR_Q targets over the survivors, as OLF's
     *  updateReactiveKeys + AcTargetVector do for the members that SHARE - and it wrote that formula on
     *  every member with a row, one switched to PQ in the base included. That member's row is not a share: it
     *  holds the frozen limit, targetQ_frozen - loadQ, and OLF never touches it. With no live key the
     *  seeded value is -targetQ_m: the frozen Q with its SIGN FLIPPED. Measured on network.xiidm/IZUZ
     *  (IZSV_0 islanded, IZTH_0 switched to PQ at MIN_Q in the base): the device's first solve ran IZTH_0 against
     *  +0.276956953 pu where OLF had -0.276956953, entry residual 0.5539 = twice the frozen Q, the first
     *  Newton step left 0.1126 on that row so the solve took 2 iterations where OLF took 1, and the
     *  distributed-slack pass that followed read -0.034956058 against OLF's -0.035800839 - the
     *  0.0845 MW the arms reported apart, with 12781 of 13502 buses off (worst 1.648e-03 kV). The
     *  reactive-limit loop only repaired the target later, through a curve-drift re-freeze. Here there
     *  is no curve, so without the fix the flipped target stands to the end. GPU SA == CPU SA at 1e-12,
     *  and the contingency must batch. */
    @Test
    void deviceSwitchedToPqInBaseMemberKeepsItsQWhenASiblingIsIslandedMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = Network.create("switched-to-pq-member-sibling-islanded", "code");
        com.powsybl.iidm.network.Substation s = network.newSubstation().setId("s").add();
        for (String vl : List.of("vl0", "vlp", "vlc1", "vlc2")) {
            com.powsybl.iidm.network.VoltageLevel v = s.newVoltageLevel().setId(vl).setNominalV(400)
                    .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
            v.getBusBreakerView().newBus().setId(vl + "_b").add();
        }
        network.getVoltageLevel("vl0").newGenerator().setId("g0").setBus("vl0_b").setConnectableBus("vl0_b")
                .setMinP(0).setMaxP(2000).setTargetP(800).setTargetV(425).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl0").newLoad().setId("ld0").setBus("vl0_b").setConnectableBus("vl0_b").setP0(100).setQ0(0).add();
        network.getVoltageLevel("vlp").newLoad().setId("ldp").setBus("vlp_b").setConnectableBus("vlp_b").setP0(300).setQ0(0).add();
        for (String c : List.of("c1", "c2")) {
            network.getVoltageLevel("vl" + c).newGenerator().setId("g" + c).setBus("vl" + c + "_b").setConnectableBus("vl" + c + "_b")
                    .setMinP(0).setMaxP(500).setTargetP(120).setTargetQ(0).setTargetV(412).setVoltageRegulatorOn(true)
                    .setRegulatingTerminal(network.getLoad("ldp").getTerminal()).add();
            network.getGenerator("g" + c).newMinMaxReactiveLimits().setMinQ(-20).setMaxQ(200).add();
        }
        // a reactive load on the bus of the survivor switched to PQ, so its row target is (limit - loadQ) and a
        // sign flip is not a symmetric no-op
        network.getVoltageLevel("vlc1").newLoad().setId("ldc1").setBus("vlc1_b").setConnectableBus("vlc1_b").setP0(0).setQ0(15).add();
        java.util.function.BiConsumer<String[], double[]> line = (ends, rx) -> network.newLine().setId(ends[0])
                .setVoltageLevel1(ends[1]).setBus1(ends[1] + "_b").setConnectableBus1(ends[1] + "_b")
                .setVoltageLevel2(ends[2]).setBus2(ends[2] + "_b").setConnectableBus2(ends[2] + "_b")
                .setR(rx[0]).setX(rx[1]).setG1(0).setB1(0).setG2(0).setB2(0).add();
        line.accept(new String[] {"l0p", "vl0", "vlp"}, new double[] {0.4, 5});
        line.accept(new String[] {"lpc1", "vlp", "vlc1"}, new double[] {20, 50});
        line.accept(new String[] {"lpc2", "vlp", "vlc2"}, new double[] {20, 50});   // vlc2's ONLY line
        line.accept(new String[] {"l0c1", "vl0", "vlc1"}, new double[] {20, 50});
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl0_0"));
        assertGpuMatchesCpu(network, List.of(new Contingency("lpc2", new com.powsybl.contingency.LineContingency("lpc2"))),
                1e-12, lfp);
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "the islanding outage must batch (or the test proves nothing)");
    }

    /** A remote voltage-control group whose members are ALL switched to PQ at MAX_Q in the base (a radial
     *  200 MVar load at vlq) and all SWITCHED BACK TO PV by the contingency (losing that radial line). OLF's
     *  switch back to PV recomputes each member's targetQ from its generators (0 here) and the group then shares its
     *  DEVIATION from those targets; the device carried each member's targetQ FROZEN at the limit, so it
     *  shared the deviation from +20 / +40 MVar instead and the two members settled at different
     *  reactive injections than the CPU's (network.xiidm / LJYK: 6e-3 kV at convEps 1e-12 on both
     *  arms). The limits are asymmetric so the keys (Q ranges 90 : 60) are not proportional to the
     *  frozen limits (30 : 50 MVar) - otherwise the two sharings coincide and the test proves nothing.
     *  GPU SA == CPU SA at 1e-12. */
    @Test
    void deviceRemoteGroupSwitchedBackToPvByContingencyRecomputesTargetQMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = remoteGroupSwitchedBackToPvNetwork();
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl0_0"));
        assertGpuMatchesCpu(network, List.of(new Contingency("lpq", new com.powsybl.contingency.LineContingency("lpq"))),
                1e-12, lfp);
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "the contingency must batch (or the test proves nothing)");
    }

    @Test
    void deviceRemoteGeneratorControlMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createWithGeneratorRemoteControl();
        network.newLine().setId("L14")
                .setVoltageLevel1("vl1").setBus1("b1").setVoltageLevel2("vl4").setBus2("b4")
                .setR(1.5).setX(40).setG1(0).setB1(0).setG2(0).setB2(0).add();
        List<Contingency> ctgs = List.of(new Contingency("L14", new BranchContingency("L14")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }

    /** Remote control + TIGHT reactive limits on a NON-free member (the CPU switches b2 to PQ in the BASE):
     *  batched on-device since plan-B3 (mode-4 BUS_TARGET_Q rows, dynamic reactive-key renormalization and
     *  frozen-member target compensation on the survivors' DISTR_Q rows). Since plan-B4 a limited
     *  FREE member batches too; only a group whose enabled members are ALL limited still routes
     *  the whole run to the CPU when unrepresentable; the corner with no enabled controller batches too
     *  ({@link #deviceRemoteAllMembersSwitchedToPqMatchesCpu}). */
    @Test
    void deviceRemoteGeneratorControlReactiveLimitsMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createWithGeneratorRemoteControl();
        network.getGenerator("g2").newMinMaxReactiveLimits().setMinQ(-30).setMaxQ(30).add();
        network.newLine().setId("L14")
                .setVoltageLevel1("vl1").setBus1("b1").setVoltageLevel2("vl4").setBus2("b4")
                .setR(1.5).setX(40).setG1(0).setB1(0).setG2(0).setB2(0).add();
        List<Contingency> ctgs = List.of(new Contingency("L14", new BranchContingency("L14")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }

    /** OLF's GLOBAL "all PV buses should switch — strongest stays PV" rule (this network's ONLY
     *  PV buses are the three limited members; the L14 outage drives every one over its limit and
     *  the CPU FORCES g1 to stay PV at q=1742 > maxQ=1000). The device runs the same election
     *  IN-KERNEL after its PV -> PQ pass (reactiveLimitsForcedPvBatch: highest controlled-bus nominal V, then
     *  highest live target P, then lowest id — the elected bus's switch is undone), so the scenario SOLVES
     *  on the device at 1e-12 instead of being rejected to the CPU (pre-election it solved 1.9 kV
     *  wrong; the detector then rejected it — count 0). */
    @Test
    void deviceRemoteForcedPvElectionMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = forcedPvNetwork();
        List<Contingency> ctgs = List.of(new Contingency("L14", new BranchContingency("L14")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "the forced-PV scenario must SOLVE on the device (in-kernel election), not be rejected");
    }

    /** OLF's "strongest stays PV" rule (switchPvPq at remainingPvBusCount 0) counts the buses its check leaves PV, so g6,
     *  whose reactive range is the single value 5 MVar at its 10 MW and which the base switched to PQ, changes nothing:
     *  losing c4 drives g1, g2 and g3 over their limits and OLF keeps one of them PV. The device decided once per
     *  network whether the rule could fire, and read a single-value range as a bus that never switches to PQ. */
    @Test
    void deviceForcedPvElectionBesideASingleValueRangeControllerMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = capacitorLossForcedPvNetworkWithSingleValueRangeController();
        List<Contingency> ctgs = List.of(new Contingency("c4", new com.powsybl.contingency.ShuntCompensatorContingency("c4")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        assertTrue(cpuForcedPvElections(network, ctgs, lfp) > 0, "OLF must keep a bus PV after c4, or the test proves nothing");
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "c4 must solve on the device");
    }

    /** g6 has no reactive limits, so while it is connected OLF's "strongest stays PV" rule cannot fire: losing c4 alone
     *  switches g1, g2 and g3 to PQ and leaves g6 PV. Losing g6 with c4 leaves the three as the only PV buses, all over
     *  their limits, and OLF keeps one of them PV. The device ruled the election out for the whole network because one
     *  of its PV buses had no limits. */
    @Test
    void deviceForcedPvElectionAfterLosingTheUnlimitedControllerMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = capacitorLossForcedPvNetworkWithUnlimitedController();
        List<Contingency> ctgs = List.of(
                new Contingency("c4", new com.powsybl.contingency.ShuntCompensatorContingency("c4")),
                new Contingency("g6+c4", new com.powsybl.contingency.GeneratorContingency("g6"),
                        new com.powsybl.contingency.ShuntCompensatorContingency("c4")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        assertEquals(0, cpuForcedPvElections(network, ctgs.subList(0, 1), lfp), "OLF must not keep a bus PV after c4 alone");
        assertTrue(cpuForcedPvElections(network, ctgs, lfp) > 0, "OLF must keep a bus PV after g6+c4, or the test proves nothing");
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "both contingencies must solve on the device");
    }

    /** OLF counts the buses its check leaves PV before it switches anything. Losing c4 and c6 drives g1, g2 and g3 over
     *  their limits in the very check where g6, switched to PQ at MIN_Q in the base, switches back to PV: g6 is not one
     *  of the buses left PV, so OLF keeps g1 PV, and keeps it PV in the next check, where g6 switches to PQ at MAX_Q.
     *  The device counted the buses PV after its check, g6 among them, and switched all three to PQ. */
    @Test
    void deviceForcedPvElectionWhileAControllerSwitchesBackToPvMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = capacitorLossForcedPvNetworkWithControllerSwitchingBack();
        List<Contingency> ctgs = List.of(new Contingency("c4+c6", new com.powsybl.contingency.ShuntCompensatorContingency("c4"),
                new com.powsybl.contingency.ShuntCompensatorContingency("c6")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        assertTrue(cpuForcedPvElections(network, ctgs, lfp) > 0, "OLF must keep a bus PV after c4+c6, or the test proves nothing");
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "c4+c6 must solve on the device");
    }

    /** OLF keeps g1 PV when losing c2 drives g0 and g1 over their limits (both control a 400 kV bus, g1 has the higher
     *  target P). The coupler-tree relocation puts g1's reactive balance, and its switch to PQ, on the coupler's DUMMY_Q
     *  row, while its own V row carries the coupler's voltage equation. Opening c01, the second contingency, is what
     *  keeps the coupler in the network. */
    @Test
    void deviceForcedPvElectionOfARelocatedControllerMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = coupledSectionsForcedPvNetwork();
        List<Contingency> ctgs = List.of(
                new Contingency("c2", new com.powsybl.contingency.ShuntCompensatorContingency("c2")),
                new Contingency("c01", new com.powsybl.contingency.SwitchContingency("c01")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        assertTrue(cpuForcedPvElections(network, ctgs, lfp) > 0, "OLF must keep a bus PV after c2, or the test proves nothing");
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "both contingencies must solve on the device");
    }

    /** {@link #deviceForcedPvElectionOfARelocatedControllerMatchesCpu} with a load on bbs1, so that the reactive power
     *  distribution equation g1's switch to PQ replaces on the coupler's DUMMY_Q row has a target that is not 0. Keeping
     *  g1 PV must give that row its equation and target back, and leave the coupler's voltage equation on g1's own V row
     *  alone. */
    @Test
    void deviceForcedPvElectionOfARelocatedControllerWithALoadMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = coupledSectionsForcedPvNetworkWithLoadOnBbs1();
        List<Contingency> ctgs = List.of(
                new Contingency("c2", new com.powsybl.contingency.ShuntCompensatorContingency("c2")),
                new Contingency("c01", new com.powsybl.contingency.SwitchContingency("c01")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        assertTrue(cpuForcedPvElections(network, ctgs, lfp) > 0, "OLF must keep a bus PV after c2, or the test proves nothing");
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "both contingencies must solve on the device");
    }

    /** How many times OLF's reactive-limit loop kept the strongest bus PV in a CPU run, read from its report. */
    private int cpuForcedPvElections(Network network, List<Contingency> ctgs, com.powsybl.loadflow.LoadFlowParameters lfp) {
        com.powsybl.commons.report.ReportNode report = com.powsybl.commons.report.ReportNode.newRootReportNode()
                .withResourceBundles(com.powsybl.openloadflow.util.report.PowsyblOpenLoadFlowReportResourceBundle.BASE_NAME)
                .withMessageTemplate("olf.saReport").build();
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        runSecurityAnalysis(network, ctgs, List.of(), sap, report);
        return countReports(report, "olf.busForcedToBePv");
    }

    private static int countReports(com.powsybl.commons.report.ReportNode node, String messageKey) {
        int count = messageKey.equals(node.getMessageKey()) ? 1 : 0;
        for (com.powsybl.commons.report.ReportNode child : node.getChildren()) {
            count += countReports(child, messageKey);
        }
        return count;
    }

    /** Plan-B3/B5: a POST-CONTINGENCY PV -> PQ switch of a NON-free remote group member runs
     *  on-device (mode-4 transition + own-q accumulation + pilot-referenced switch conditions) at
     *  1e-12. g2 (non-free member: free = g1, unlimited) has maxQ between its base Q (~183 MVAr)
     *  and its post-outage Q (~189 MVAr), so losing one parallel b4-b5 line switches it to PQ — verified on
     *  the CPU side by the differing per-member Q split in the reference results. */
    @Test
    void deviceRemoteMemberPostContingencySwitchToPqMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = remoteReactiveLimitNetwork("g2", 186.0);
        List<Contingency> ctgs = List.of(
                new Contingency("l45b", new BranchContingency("l45b")));
        assertGpuMatchesCpu(network, ctgs, 1e-12, remoteReactiveLimitParameters());
    }

    /** Plan-B3/B5 continuation of a base switch: g2 is switched to PQ in the BASE case (maxQ 180 < base need
     *  ~183) and a strongly relieving load contingency (-300 MVAr) hits the group. OLF's remote
     *  switch-back tests the PILOT's V against the GROUP target — the surviving members keep the
     *  pilot AT target, so the member STAYS PQ on the CPU, and the device must reproduce that
     *  exactly: mode-4 seed, renormalized weights, and the frozen-member TARGET compensation on
     *  the survivors' DISTR_Q rows (qPct x frozen targetQ). A decisive remote switch BACK TO PV requires the
     *  group to lose control margin, which the B4 pre-gate keeps on the CPU (free member limited). */
    @Test
    void deviceRemoteMemberSwitchedToPqInBaseReliefMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = remoteReactiveLimitNetwork("g2", 180.0);
        List<Contingency> ctgs = List.of(
                new Contingency("l5", new com.powsybl.contingency.LoadContingency("l5")));
        assertGpuMatchesCpu(network, ctgs, 1e-12, remoteReactiveLimitParameters());
    }

    /** A remote controller whose reactive range is a single value at its active power (g2: [100, 100] MVar) is still
     *  one of OLF's controllers: its share of the group's reactive need is not 100 MVar, so ReactiveLimitsOuterLoop
     *  switches it PV -> PQ at that value in the base and in every contingency. The device gave a remote controller a
     *  reactive-limit entry only for minQ < maxQ, so it never checked g2, which kept sharing like an unlimited
     *  controller. 1e-12 on device. */
    @Test
    void deviceRemoteControllerWithASingleValueRangeSwitchesToPqMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = singleValueRangeRemoteMemberNetwork(100.0);
        List<Contingency> ctgs = List.of(
                new Contingency("l45b", new BranchContingency("l45b")),
                new Contingency("l5", new com.powsybl.contingency.LoadContingency("l5")),
                new Contingency("g1", new com.powsybl.contingency.GeneratorContingency("g1")));
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(network, ctgs, 1e-12, remoteReactiveLimitParameters());
        assertEquals(3, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** A merged DEPENDENT control keeps its own target (OLF's ReactiveLimitsOuterLoop compares the bus's own control:
     *  getBusV / getBusTargetV read bus.getGeneratorVoltageControl(), the dependent's), while the cluster sits at the
     *  MAIN's target. Two sections: g0 (main, 401 kV, +-300 MVar) and g1 (dependent, 400.5 kV, 5 MVar ceiling), g1
     *  switched to PQ at MAX_Q in the base by the keys. Losing the 40 MVar load on section 0 cuts the group's need:
     *  OLF switches g1 back to PV (the cluster's 401 kV is above g1's OWN 400.5 kV target) and it shares again;
     *  compared against the MAIN's 401 kV the device saw no reason to switch it back and kept g1 at its ceiling. 1e-12
     *  on device. */
    @Test
    void deviceMergedDependentSwitchesBackAgainstItsOwnTargetMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = splitGroupNetwork(2, java.util.Set.of(1));
        // a lightly loaded group: the external loads carry no reactive power and the targets sit close to the
        // slack's 400 kV, so the group's reactive need is about the 40 MVar of ld0 - enough for the keys (g0 +-300,
        // g1 [-300, 5]) to switch g1 to PQ in the base, gone with ld0
        for (String ld : List.of("lds", "ld3", "ld4")) {
            network.getLoad(ld).setQ0(0);
        }
        network.getGenerator("g0").setTargetV(401);
        network.getGenerator("g1").setTargetV(400.5);
        // the section loss keeps the coupler RETAINED (OLF retains only the switches a listed contingency opens);
        // without it the two sections are one bus with one control and nothing is merged
        List<Contingency> ctgs = List.of(
                new Contingency("bus-0", new com.powsybl.contingency.BusbarSectionContingency("bbs0")),
                new Contingency("ld0", new com.powsybl.contingency.LoadContingency("ld0")));
        GpuBatchScopeCensus.reset();
        assertMergedPilotLossMatchesCpu(network, ctgs, 2);
    }

    /** A generator loss that leaves its bus a controller, composed with an in-flight PV -> PQ switch in the same group. OLF's
     *  reactive keys after the loss (updateReactiveKeys: the bus keeps a smaller key) are renormalized again when g1
     *  switches to PQ. The device carried the loss as per-contribution WEIGHTS fixed at plan time, which the fill
     *  applied over its recompute that follows the PV -> PQ switches - the rows kept the outage-only split while the
     *  targets followed the switch. Now the plan hands the kernels the post-loss member KEYS and both renormalize them
     *  over the members a PV -> PQ switch leaves.
     *  remoteReactiveLimitNetwork with a second generator g2b on g2's bus: losing g2b and l45b together. 1e-12 on device. */
    @Test
    void deviceGeneratorLossWithInFlightSwitchToPqRenormalizesKeysMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = remoteReactiveLimitNetwork("g1", 186.0);
        com.powsybl.iidm.network.Generator g2 = network.getGenerator("g2");
        g2.getTerminal().getVoltageLevel().newGenerator().setId("g2b").setBus("b2").setConnectableBus("b2")
                .setMinP(0).setMaxP(200).setTargetP(40).setTargetQ(0).setVoltageRegulatorOn(true)
                .setRegulatingTerminal(g2.getRegulatingTerminal()).setTargetV(g2.getTargetV()).add()
                .newMinMaxReactiveLimits().setMinQ(-150).setMaxQ(150).add();
        List<Contingency> ctgs = List.of(new Contingency("g2b+l45b",
                new com.powsybl.contingency.GeneratorContingency("g2b"), new BranchContingency("l45b")));
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(network, ctgs, 1e-12, remoteReactiveLimitParameters());
        assertEquals(java.util.Set.of(), GpuBatchScopeCensus.planRefusedIds(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** Two remote voltage controls merged through a zero-impedance line (g1, g1b, g2 -> b4 main; g3, g3b -> b5
     *  dependent), every generator with a 1000 MVar range, so the keys follow the RANGES (b1 40 %, b2 20 %, b3 40 %).
     *  OLF's key rules skip a lost (disabled) generator and renormalize on the MAIN control: losing g1b gives
     *  25/25/50 %, losing g3b 50/25/25 %. The plan's key replay must disable the generator as OLF does, otherwise
     *  its range stays in its bus's key. */
    @Test
    void deviceGeneratorLossInRangeKeyedMergedControlsMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createWithGeneratorRemoteControl();
        network.getSubstation("s").newVoltageLevel().setId("vl5").setNominalV(400)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
        network.getVoltageLevel("vl5").getBusBreakerView().newBus().setId("b5").add();
        com.powsybl.iidm.network.Line zl = network.newLine().setId("zl45").setBus1("b4").setConnectableBus1("b4")
                .setBus2("b5").setConnectableBus2("b5").setR(0).setX(0).add();
        com.powsybl.iidm.network.Generator g1 = network.getGenerator("g1");
        com.powsybl.iidm.network.Generator g3 = network.getGenerator("g3");
        g3.setRegulatingTerminal(zl.getTerminal2());
        network.getVoltageLevel("vl1").newGenerator().setId("g1b").setBus("b1").setConnectableBus("b1")
                .setMinP(0).setMaxP(200).setTargetP(50).setVoltageRegulatorOn(true)
                .setRegulatingTerminal(g1.getRegulatingTerminal()).setTargetV(g1.getTargetV()).add();
        network.getVoltageLevel("vl3").newGenerator().setId("g3b").setBus("b3").setConnectableBus("b3")
                .setMinP(0).setMaxP(200).setTargetP(50).setVoltageRegulatorOn(true)
                .setRegulatingTerminal(zl.getTerminal2()).setTargetV(g3.getTargetV()).add();
        network.getGeneratorStream().forEach(g -> g.newMinMaxReactiveLimits().setMinQ(-500).setMaxQ(500).add());
        List<Contingency> ctgs = List.of(
                new Contingency("g1b", new com.powsybl.contingency.GeneratorContingency("g1b")),
                new Contingency("g3b", new com.powsybl.contingency.GeneratorContingency("g3b")));
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(network, ctgs, 1e-12);
        assertEquals(java.util.Set.of(), GpuBatchScopeCensus.planRefusedIds(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** A shared transformer group whose outage leaves ONE survivor. OLF picks the adjustment path on the controllers
     *  left after the outage (adjustWithOneController) and reads the controlled bus's deviation; the device picked it
     *  on the static group size and, on the shared path, read the deviation off the group's FIRST member - the
     *  outaged one, whose deadband check writes nothing: the survivor never moved (0 on a fresh context) or moved
     *  on a stale value. Two parallel transformers regulating BUS_3 with no deadband: losing either, the other
     *  must bring BUS_3 back alone. 1e-12, both on device. */
    @Test
    void deviceSharedGroupSurvivorAdjustsAloneMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createWithTransformerSharedRemoteControl();
        for (String id : new String[] {"T2wT", "T2wT2"}) {
            com.powsybl.iidm.network.TwoWindingsTransformer t = network.getTwoWindingsTransformer(id);
            t.getRatioTapChanger().setTargetDeadband(0).setTapPosition(0)
                    .setRegulationMode(com.powsybl.iidm.network.RatioTapChanger.RegulationMode.VOLTAGE)
                    .setRegulationTerminal(t.getTerminal2()).setTargetV(33.0).setRegulating(true);
        }
        List<Contingency> ctgs = List.of(
                new Contingency("T2wT", new BranchContingency("T2wT")),
                new Contingency("T2wT2", new BranchContingency("T2wT2")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setTransformerVoltageControlOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.INCREMENTAL_VOLTAGE_CONTROL);
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** Plan-B4: a POST-CONTINGENCY PV -> PQ switch of the FREE member itself. g1 (free = first
     *  enabled, the base pilot-Q carrier) gets the maxQ knob; losing one parallel b4-b5 line pushes
     *  its share over the limit, OLF switches it to PQ and re-picks the free role to g2 — on-device via
     *  the pilot-Q relocation candidates (mode-scan carrier select), the carrier-row DISTR_Q
     *  suppression and the captured pilot-Q target handoff. g2/g3 stay unlimited so the
     *  gate on groups whose members are all limited lets it through. 1e-12 with engagement. */
    @Test
    void deviceRemoteFreeMemberPostContingencySwitchToPqMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = remoteReactiveLimitNetwork("g1", 186.0);
        List<Contingency> ctgs = List.of(
                new Contingency("l45b", new BranchContingency("l45b")));
        assertGpuMatchesCpu(network, ctgs, 1e-12, remoteReactiveLimitParameters());
    }

    /** Plan-B4 FREE member switched to PQ in the base: g1 is switched to PQ in the BASE case, so the base free role (and
     *  the pilot-Q carrier row in the extracted structure) is already g2; the l45b outage loads the
     *  group further and g1 STAYS PQ. Exercises the base-mode-keyed carrier select and the
     *  frozen-free-member target compensation on the survivors. 1e-12. */
    @Test
    void deviceRemoteFreeMemberSwitchedToPqInBaseMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = remoteReactiveLimitNetwork("g1", 180.0);
        List<Contingency> ctgs = List.of(
                new Contingency("l45b", new BranchContingency("l45b")));
        assertGpuMatchesCpu(network, ctgs, 1e-12, remoteReactiveLimitParameters());
    }

    /** Plan-B4 chained PV -> PQ switches: BOTH g1 (free) and g2 are limited, g3 stays unlimited (the gate
     *  passes). The outage pushes the group over both limits — the CPU freezes both members and
     *  the pilot-Q carrier lands on g3; the device must follow the same switch sequence, with TWO
     *  frozen members' target compensation on g3's rows. 1e-12. */
    @Test
    void deviceRemoteFreeAndSiblingSwitchToPqMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = remoteReactiveLimitNetwork("g1", 186.0);
        network.getGenerator("g2").newMinMaxReactiveLimits().setMinQ(-1000).setMaxQ(186.0).add();
        List<Contingency> ctgs = List.of(
                new Contingency("l45b", new BranchContingency("l45b")));
        assertGpuMatchesCpu(network, ctgs, 1e-12, remoteReactiveLimitParameters());
    }

    /** The formerly-guarded B3-class gap, now LIFTED by the ob alternate-dest channel: a LIMITED
     *  member with a HALF-OPEN branch at its bus switches to PQ ON-DEVICE — the open line's charging q
     *  follows the mode-4 row through the ob two-way select (it was 3.3e-4 kV silently MISSING
     *  from the member's PQ row before the channel, then whole-run CPU-gated). 1e-12 with engagement. */
    @Test
    void deviceRemoteOpenBranchMemberSwitchToPqMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = remoteReactiveLimitNetwork("g2", 186.0);
        network.newLine().setId("lob")
                .setVoltageLevel1("vl2").setBus1("b2").setConnectableBus1("b2")
                .setVoltageLevel2("vl5").setBus2("b5").setConnectableBus2("b5")
                .setR(1.5).setX(40.0).setG1(0).setB1(1e-4).setG2(0).setB2(1e-4).add();
        network.getLine("lob").getTerminal2().disconnect();
        List<Contingency> ctgs = List.of(
                new Contingency("l45b", new BranchContingency("l45b")));
        assertGpuMatchesCpu(network, ctgs, 1e-12, remoteReactiveLimitParameters());
    }

    /** Zero-impedance twin: a LIMITED member with a ZERO-IMPEDANCE spanning-tree branch at its
     *  bus — the dummy-Q coupling follows the mode-4 row through the zero-impedance dynamic slots. 1e-12. */
    @Test
    void deviceRemoteZeroImpMemberSwitchToPqMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = remoteReactiveLimitNetwork("g2", 186.0);
        var vl8 = network.getSubstation("s").newVoltageLevel().setId("vl8").setNominalV(400)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
        vl8.getBusBreakerView().newBus().setId("b8").add();
        vl8.newLoad().setId("l8").setBus("b8").setConnectableBus("b8").setP0(5).setQ0(10).add();
        network.newLine().setId("lzi")                       // R = X = 0: OLF zero-impedance branch,
                .setVoltageLevel1("vl2").setBus1("b2").setConnectableBus1("b2")   // spanning-tree edge
                .setVoltageLevel2("vl8").setBus2("b8").setConnectableBus2("b8")   // (b8's only feed)
                .setR(0.0).setX(0.0).setG1(0).setB1(0).setG2(0).setB2(0).add();
        List<Contingency> ctgs = List.of(
                new Contingency("l45b", new BranchContingency("l45b")));
        assertGpuMatchesCpu(network, ctgs, 1e-12, remoteReactiveLimitParameters());   // GPU base self-heals via
    }                                                        // the device-broken-solve safety net

    /** Half-open branch at the PILOT + a FREE-member PV -> PQ switch: the pilot's ob charging q rides the
     *  relocation GROUP pool (the qr table now exists for unclean pilots) and follows the
     *  re-picked carrier. 1e-12 with engagement. */
    @Test
    void deviceRemoteOpenBranchPilotFreeMemberSwitchToPqMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = remoteReactiveLimitNetwork("g1", 186.0);
        network.newLine().setId("lob4")
                .setVoltageLevel1("vl4").setBus1("b4").setConnectableBus1("b4")
                .setVoltageLevel2("vl5").setBus2("b5").setConnectableBus2("b5")
                .setR(1.5).setX(40.0).setG1(0).setB1(1e-4).setG2(0).setB2(1e-4).add();
        network.getLine("lob4").getTerminal2().disconnect();
        List<Contingency> ctgs = List.of(
                new Contingency("l45b", new BranchContingency("l45b")));
        assertGpuMatchesCpu(network, ctgs, 1e-12, remoteReactiveLimitParameters());
    }

    /** The former gate's member-targeting twin, still valid post-lift: the half-open branch at an
     *  UNLIMITED member's bus (g3) batches with the limited clean member (g2)
     *  switching to PQ on-device. 1e-12 with engagement. */
    @Test
    void deviceRemoteUncleanUnlimitedMemberStillBatches() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = remoteReactiveLimitNetwork("g2", 186.0);
        network.newLine().setId("lob")
                .setVoltageLevel1("vl3").setBus1("b3").setConnectableBus1("b3")
                .setVoltageLevel2("vl5").setBus2("b5").setConnectableBus2("b5")
                .setR(1.5).setX(40.0).setG1(0).setB1(1e-4).setG2(0).setB2(1e-4).add();
        network.getLine("lob").getTerminal2().disconnect();
        List<Contingency> ctgs = List.of(
                new Contingency("l45b", new BranchContingency("l45b")));
        assertGpuMatchesCpu(network, ctgs, 1e-12, remoteReactiveLimitParameters());
    }

    /** Continuation of a base switch, branch-outage flavor: g2 switched to PQ in the base, the l45b outage loads
     *  the group further and the member STAYS PQ on both sides. 1e-12. */
    @Test
    void remoteMemberSwitchedToPqInBaseMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = remoteReactiveLimitNetwork("g2", 180.0);
        List<Contingency> ctgs = List.of(
                new Contingency("l45b", new BranchContingency("l45b")));
        assertGpuMatchesCpu(network, ctgs, 1e-12, remoteReactiveLimitParameters());
    }

    /** DEVICE corner with no enabled controller: when the LAST live member switches to PQ, updatePilotTargetVEquationBatch flips the
     *  pilot's V identity to a power row carrying the captured pilot-Q target (the relocation
     *  mode-scan already lands the pilot's balance there via the fallback candidate), and
     *  restores it on the first switch back to PV. 1e-12 with engagement. */
    @Test
    void deviceRemoteAllMembersSwitchedToPqMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = noEnabledControllerCornerNetwork();
        List<Contingency> ctgs = List.of(
                new Contingency("l45b", new BranchContingency("l45b")));
        assertNoEnabledControllerCornerSemantics(network, ctgs);
        assertGpuMatchesCpu(network, ctgs, 1e-12, remoteReactiveLimitParameters());
    }

    /** CPU-side semantics check (ground rule: prove the scenario exercises what it claims): the
     *  BASE holds the pilot AT the group target, the post-outage state converges measurably OFF
     *  it — control genuinely lost in flight. */
    /** SINGLETON remote group (one controller, g1 -> b4; g2/g3 turned LOCAL) whose only member
     *  SWITCHES TO PQ under the l45b outage — the shape the retired table-less-free-member whole-run gate
     *  used to CPU-route (2026-09-08): the singleton now carries a relocation table
     *  [pilot fallback, member], and its member's switch to PQ IS the corner with no enabled controller — the in-kernel
     *  mode-scan finds no live candidate and updatePilotTargetVEquationBatch flips the pilot's V identity to
     *  a power row with the captured pilot Q. Base keeps control, the outage loses it (asserted
     *  on the CPU oracle); GPU SA == CPU SA at 1e-12 with the batch ENGAGED. */
    @Test
    void deviceRemoteSingletonMemberSwitchToPqMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = singletonRemoteReactiveLimitNetwork();
        List<Contingency> ctgs = List.of(new Contingency("l45b", new BranchContingency("l45b")));
        assertNoEnabledControllerCornerSemantics(network, ctgs);
        assertGpuMatchesCpu(network, ctgs, 1e-12, remoteReactiveLimitParameters());
    }

    /** Generator outage AT a DISTR_Q controller bus that KEEPS another controller: the group's qPct
     *  weights are baked into the equation system on both arms, so the outage is a plain P-target
     *  drop + DS reweight — the former blanket DISTR_Q rejection was over-conservative. */
    @Test
    void deviceGenOutageAtDistrQBusMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createWithGeneratorRemoteControl();
        com.powsybl.iidm.network.Generator g1 = network.getGenerator("g1");
        network.getVoltageLevel("vl1").newGenerator().setId("g1b").setBus("b1").setConnectableBus("b1")
                .setMinP(0).setMaxP(100).setTargetP(30).setTargetV(g1.getTargetV()).setVoltageRegulatorOn(true)
                .setRegulatingTerminal(g1.getRegulatingTerminal()).add();
        List<Contingency> ctgs = List.of(
                new Contingency("g1b", new com.powsybl.contingency.GeneratorContingency("g1b")),
                // BOTH of b1's controllers: the group restructures (free-controller reassignment) —
                // the weight rediff detects the structural drift and routes to the CPU, exact either way
                new Contingency("g1+g1b", new com.powsybl.contingency.GeneratorContingency("g1"),
                        new com.powsybl.contingency.GeneratorContingency("g1b")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }

    @Test
    void deviceRemoteGeneratorControlControllerOutageMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createWithGeneratorRemoteControl();
        network.newLine().setId("L14")
                .setVoltageLevel1("vl1").setBus1("b1").setVoltageLevel2("vl4").setBus2("b4")
                .setR(1.5).setX(40).setG1(0).setB1(0).setG2(0).setB2(0).add();
        List<Contingency> ctgs = List.of(new Contingency("L14", new BranchContingency("L14")),
                new Contingency("tr1", new BranchContingency("tr1")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }

    /** OLF's security analysis with the GPU Newton-Raphson as its AC solver, a contingency islanding one controller of
     *  a remote voltage control. The post-contingency extraction sees that controller's bus disabled: no V column, a
     *  DISTR_Q equation disabled with it, a key OLF set to 0. The DISTR_Q pack still packed it and the device refused
     *  the structure ("bad distrQRow ... -1"); then the reactive-capability channel's replay of the reactive-limit pack
     *  kept it where the pack skips it ("RL-pack enumeration drift"). Every post-contingency solve fell back to the
     *  CPU: right, but not on the device. Once on the device, the surviving controller that carries the pilot's
     *  reactive balance was FROZEN (the device-only base code 6, "PV/PQ state not on its own row"): OLF applies its
     *  reactive limits like any other's, and the device ran out of iterations where OLF converges, with limits
     *  reached (maxQ 60) or not (1000). */
    @Test
    void sequentialSolveWithAnIslandedRemoteControllerRunsOnTheDevice() {
        assertSequentialGpuSaMatchesCpu(limitedRemoteControlGroupNetwork(1000));
        assertSequentialGpuSaMatchesCpu(limitedRemoteControlGroupNetwork(60));
    }

    private void assertSequentialGpuSaMatchesCpu(Network network) {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        List<Contingency> ctgs = List.of(new Contingency("tr3", new BranchContingency("tr3")),
                new Contingency("tr2", new BranchContingency("tr2")));
        List<com.powsybl.security.monitor.StateMonitor> monitors = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(),
                java.util.Set.of("vl1", "vl2", "vl3", "vl4"), java.util.Set.of()));
        com.powsybl.security.SecurityAnalysisResult cpu = runSecurityAnalysis(network, ctgs, monitors, sequentialSaParameters(false));
        long fallbacksBefore = com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson.CPU_FALLBACK_COUNT.get();
        com.powsybl.security.SecurityAnalysisResult gpu = runSecurityAnalysis(network, ctgs, monitors, sequentialSaParameters(true));
        assertEquals(0, com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson.CPU_FALLBACK_COUNT.get() - fallbacksBefore,
                "every solve on the device");
        assertEquals(cpu.getPreContingencyResult().getStatus(), gpu.getPreContingencyResult().getStatus());
        for (int i = 0; i < ctgs.size(); i++) {
            var c = cpu.getPostContingencyResults().get(i);
            var g = gpu.getPostContingencyResults().get(i);
            assertEquals(c.getStatus(), g.getStatus(), c.getContingency().getId());
            var cpuBuses = c.getNetworkResult().getBusResults();
            var gpuBuses = g.getNetworkResult().getBusResults();
            assertEquals(cpuBuses.size(), gpuBuses.size());
            for (int b = 0; b < cpuBuses.size(); b++) {
                assertEquals(cpuBuses.get(b).getV(), gpuBuses.get(b).getV(), 1e-12,
                        "bus voltage " + cpuBuses.get(b).getBusId() + " for " + c.getContingency().getId());
            }
        }
    }

    private static com.powsybl.security.SecurityAnalysisParameters sequentialSaParameters(boolean gpu) {
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters ext = com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setVoltageRemoteControl(true);
        if (gpu) {
            ext.setAcSolverType(com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphsonFactory.NAME);
        }
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        return sap;
    }
}
