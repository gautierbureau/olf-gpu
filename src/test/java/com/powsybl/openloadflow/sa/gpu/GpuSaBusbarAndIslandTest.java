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
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.GpuBatchScopeCensus;
import org.junit.jupiter.api.Test;

import java.util.List;

import static com.powsybl.openloadflow.sa.gpu.GpuSaTestNetworks.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GPU-batched security analysis against the CPU: busbar sections, couplers and islanding contingencies.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuSaBusbarAndIslandTest extends AbstractGpuSaTest {

    GpuSaBusbarAndIslandTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    /** SINGLE-SIDE branch opening (2026-09-08): a BOUNDARY-line contingency on a tie line opens the
     *  tie-line branch on ONE side only (DisabledBranchStatus SIDE_x) — the batch masks the closed
     *  element and enables a pre-packed side-open OB view for the surviving terminal (same slots,
     *  unchanged CSR pattern, per-scenario element mask). Both boundary sides + a full-line control
     *  at 1e-12, count asserted. */
    @Test
    void deviceBusLossMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        // BUS LOSS = every incident branch opened on the LOST side (DisabledBranchStatus SIDE_x per
        // OLF's PropagatedContingency): the batch masks each closed element and enables its
        // pre-packed side-open OB view at the SURVIVING terminal (dangling-half charging included);
        // the lost bus islands (frozen rows) and its injections ride the scheduled-loss accounting.
        Network network = com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory.create14();
        network.getGenerators().forEach(g -> g.newMinMaxReactiveLimits().setMinQ(-300).setMaxQ(300).add());
        List<Contingency> ctgs = List.of(
                new Contingency("N-1", new BranchContingency("L2-3-1")),
                new Contingency("bus-B5", new com.powsybl.contingency.BusContingency("B5")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "the N-1 AND the bus-loss scenario must batch on-device");
    }

    /** An ISLANDING outage that strands a WHOLE remote gen-VC group — pilot AND every controller.
     *  OLF removes those buses' equations outright; the device freezes their rows to a flat identity
     *  instead, and updatePilotTargetVEquationBatch then read the frozen member modes as "no carrier left" and
     *  flipped the frozen PILOT row from its voltage identity into a power row. The frozen target
     *  override re-stamps the voltage value over that power row on the next pass, so the row carries
     *  a residual of exactly 1.0 that no Newton step can clear — and the whole batch then spends
     *  iterations on it that OLF never runs, moving the SURVIVING component off the CPU answer.
     *  Reduced from rte6515 LINE-3938-646, where it cost 8.777e-08 kV across 22985 buses with both
     *  arms running the identical three post-contingency solves. */
    @Test
    void deviceIslandedRemoteControlGroupMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = islandableRemoteControlGroupNetwork();
        List<Contingency> ctgs = List.of(new Contingency("LTIE", new BranchContingency("LTIE")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        // The surviving component must hold the slack, or the "island" is the part OLF keeps and the
        // group is never frozen at all.
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vlm1_0"));   // OLF names BUS-VIEW buses <vlId>_<n>
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }

    /** BUS loss that ISLANDS a distributed-slack participant (rte6515 BUS-1238_BBS in 10 buses): the
     *  lost bus vlm5_b takes the stub bus vlm6 and its participating generator gm6 with it. The bus-loss
     *  (injection) qualification path froze the island's rows but never set the islanded-DS/RL masks
     *  the branch-outage tail sets, so the water-fill read the frozen P rows' identity targets (angles)
     *  as absorbed power, reset them to the base P and gave the island shares of the mismatch that
     *  never reached the grid — a second DS round OLF does not run, 1.18e-03 kV on rte6515. */
    @Test
    void deviceBusLossIslandingDistributedSlackParticipantMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = islandableRemoteControlGroupNetwork();
        com.powsybl.iidm.network.VoltageLevel v6 = network.getSubstation("sm").newVoltageLevel().setId("vlm6")
                .setNominalV(400).setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
        v6.getBusBreakerView().newBus().setId("vlm6_b").add();
        // Large enough that re-dispatching its loss moves the network losses past the 1 MW slack gate,
        // so the post-contingency DistributedSlack round actually runs (at 150 MW it did not, and the
        // masks' absence was invisible).
        v6.newGenerator().setId("gm6").setBus("vlm6_b").setConnectableBus("vlm6_b")
                .setMinP(0).setMaxP(2000).setTargetP(900).setTargetV(400).setVoltageRegulatorOn(true).add();
        v6.newLoad().setId("lm6").setBus("vlm6_b").setConnectableBus("vlm6_b").setP0(40).setQ0(10).add();
        // Two hops behind the lost bus (rte6515: VL-1238 -> VL-5242 -> VL-4263 -> VL-6494): the island
        // is found by connectivity, not by the contingency's own element list.
        com.powsybl.iidm.network.VoltageLevel v7 = network.getSubstation("sm").newVoltageLevel().setId("vlm7")
                .setNominalV(400).setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
        v7.getBusBreakerView().newBus().setId("vlm7_b").add();
        v7.newLoad().setId("lm7").setBus("vlm7_b").setConnectableBus("vlm7_b").setP0(60).setQ0(20).add();
        v7.newGenerator().setId("gm7").setBus("vlm7_b").setConnectableBus("vlm7_b")
                .setMinP(0).setMaxP(1000).setTargetP(300).setTargetV(400).setVoltageRegulatorOn(true).add();
        // A generator ON the lost bus: a lost generator is what routes the plan through the bus-loss
        // (injection) qualification branch, as rte6515's busbars are — a lost load alone rides the
        // side-open cover and takes the branch-outage tail, where the masks were always set.
        network.getVoltageLevel("vlm5").newGenerator().setId("gm5").setBus("vlm5_b").setConnectableBus("vlm5_b")
                .setMinP(0).setMaxP(300).setTargetP(100).setTargetV(400).setVoltageRegulatorOn(true).add();
        mainLine(network, "LM57", "vlm5", "vlm7");
        mainLine(network, "LM76", "vlm7", "vlm6");
        // The bus loss PLUS a lost generator on the surviving side: a lost injection is what routes the
        // plan through the injection qualification branch (rte6515's busbar sections carry them), and
        // the islanding then takes its island-injection path. A bare bus loss rides the side-open
        // cover through the branch-outage tail instead, where the masks were always set.
        List<Contingency> ctgs = List.of(new Contingency("bus-vlm5",
                new com.powsybl.contingency.BusContingency("vlm5_b"),
                new com.powsybl.contingency.GeneratorContingency("gm3")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vlm1_0"))
                // The pre-distribution leaves ~0.5 MW at the slack after the post-contingency solve;
                // the DistributedSlack round that exposes the unmasked participant must actually run.
                .setSlackBusPMaxMismatch(0.01);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "the islanding bus loss must batch on-device (or the test proves nothing)");
        // The plan must have taken gm6 AND gm7 out of the distribution. On this fixture the state
        // comparison alone does not catch a missing mask (the host pre-distribution derives its own
        // islanded set from the frozen rows and the residual mismatch is small); on rte6515
        // BUS-1238_BBS the missing mask was 1.18e-03 kV.
        assertTrue(GpuBatchScopeCensus.LAST_ISLANDED_DS_MASKED.get() >= 2,
                "the island's two participating generators must be masked out of the slack distribution, got "
                        + GpuBatchScopeCensus.LAST_ISLANDED_DS_MASKED.get());
    }

    /** A bus loss that islands ONE MEMBER of a remote voltage-control group (rte6515 BUS-1113_BBS_2 and 21
     *  more of the 500 busbar sections): b1 and its generator g1 leave, b2/b3 keep regulating b4 and share
     *  the reactive power over the two of them. The plan's island-injection path used to refuse the
     *  RENORMALIZE verdict ("would have to compose with the generator-outage rediff") and the composer
     *  rejected the island's own generator as a DISTR_Q member losing its last controller
     *  (DISTR_Q_LAST_CONTROLLER, 17 sections). Both are the freeze's business: the island's rows are
     *  frozen whole; only the survivors' reactive keys need the same rediff the branch-outage tail does.
     *  A surviving-side generator outage rides along so the plan takes the injection path, as a busbar
     *  section carrying generators does. */
    @Test
    void deviceBusLossIslandingGroupMemberRenormalizesMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = islandableRemoteControlGroupNetwork();
        List<Contingency> ctgs = List.of(new Contingency("bus-b1",
                new com.powsybl.contingency.BusContingency("b1"),
                new com.powsybl.contingency.GeneratorContingency("gm3")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vlm1_0"));
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "the member-islanding bus loss must batch on-device (or the test proves nothing)");
    }

    /** A bus loss that islands the group's BASE CARRIER member (b1 is the first controller, whose row
     *  hosts the pilot b4's reactive balance): the mode scan moves the balance to b2's row, and the
     *  pilot-Q target it carries was captured from b1's row AFTER the scenario froze it to a flat
     *  voltage identity - so the pilot's balance was solved against 1.0 instead of its -2.0 pu load
     *  (OLF's BUS_TARGET_Q at b4 off by exactly the load, b2 0.45 kV wrong; rte6515: five busbar
     *  sections up to 1.3 kV). The capture now reads the BASE targets. Plain bus loss: the plan takes
     *  the branch-outage tail (RENORMALIZE), the same shape through the injection path is
     *  {@link #deviceBusLossIslandingGroupMemberRenormalizesMatchesCpu}. */
    @Test
    void deviceBusLossIslandingBaseCarrierMemberMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = islandableRemoteControlGroupNetwork();
        List<Contingency> ctgs = List.of(new Contingency("bus-b1", new com.powsybl.contingency.BusContingency("b1")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vlm1_0"));
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "the carrier-islanding bus loss must batch on-device (or the test proves nothing)");
    }

    /** Five sections, only the middle one (bbs2) regulating: its V row holds the tree's voltage identity, so the
     *  coupler tree is rooted there and every coupler is relocated (CouplerTreeRelocation#spanningTreeRelocation). Rooted at
     *  bbs0 by bus order instead, c12 kept the stock placement and the numeric repair borrowed a live balance row
     *  for its two zero diagonals. Every section's loss on the device, against OLF. */
    @Test
    void deviceCouplerTreeRootedAtItsRegulatedSectionMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertSplitGroupMatchesCpu(5, java.util.Set.of(), java.util.Set.of(0, 1, 3, 4));
    }

    /** A lost section takes two of a merged control's four remote controllers with it: the contingency opens the
     *  breakers of every feeder at bbs1, and the lines of g1a/g1b are left open at that end with their live end in
     *  the island. OLF disables those buses and their branches; the device kept each line DANGLING (its charging moved
     *  to the live end) at the island's frozen 1 pu, and the controllers' reactive injections reach every member's
     *  sharing row: g0b's row, keys renormalized over g0a/g0b, took 0.25 x -0.64 pu per lost line and the survivors
     *  split their reactive power apart (vg0a_b 406.77 kV against OLF's 406.30). Network.xiidm CXUL: CXSS_2/_4's
     *  step-up transformers behind section CXUL, 4.3e-3 kV at CXSS_1/_3. */
    @Test
    void deviceLostSectionDropsTheFeedersItsIslandKeepsMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = Network.create("merged-pairs", "code");
        com.powsybl.iidm.network.Substation s = network.newSubstation().setId("s").add();
        java.util.function.Function<String, com.powsybl.iidm.network.VoltageLevel> bbVl = id -> {
            com.powsybl.iidm.network.VoltageLevel v = s.newVoltageLevel().setId(id).setNominalV(400)
                    .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
            v.getBusBreakerView().newBus().setId(id + "_b").add();
            return v;
        };
        bbVl.apply("vl0");
        bbVl.apply("vl3");
        com.powsybl.iidm.network.VoltageLevel vl12 = s.newVoltageLevel().setId("vl12").setNominalV(400)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.NODE_BREAKER).add();
        vl12.getNodeBreakerView().newBusbarSection().setId("bbs0").setNode(0).add();
        vl12.getNodeBreakerView().newBusbarSection().setId("bbs1").setNode(1).add();
        vl12.getNodeBreakerView().newBreaker().setId("c01").setNode1(0).setNode2(1).setRetained(true).setOpen(false).add();
        int node = 100;
        String[][] gens = {{"g0a", "0", "30"}, {"g0b", "0", "10"}, {"g1a", "1", "40"}, {"g1b", "1", "20"}};
        for (String[] g : gens) {
            int k = Integer.parseInt(g[1]);
            com.powsybl.iidm.network.VoltageLevel vg = bbVl.apply("v" + g[0]);
            vg.newGenerator().setId(g[0]).setBus("v" + g[0] + "_b").setConnectableBus("v" + g[0] + "_b")
                    .setMinP(0).setMaxP(500).setTargetP(60).setTargetQ(0).setTargetV(405)
                    .setVoltageRegulatorOn(true)
                    .setRegulatingTerminal(vl12.getNodeBreakerView().getBusbarSection("bbs" + k).getTerminal()).add();
            network.getGenerator(g[0]).newMinMaxReactiveLimits().setMinQ(-300).setMaxQ(300).add();
            network.getGenerator(g[0]).newExtension(com.powsybl.iidm.network.extensions.CoordinatedReactiveControlAdder.class)
                    .withQPercent(Double.parseDouble(g[2])).add();
            int n = node++;
            vl12.getNodeBreakerView().newDisconnector().setId("dl" + g[0]).setNode1(k).setNode2(n + 50).setOpen(false).add();
            vl12.getNodeBreakerView().newBreaker().setId("bl" + g[0]).setNode1(n + 50).setNode2(n).setOpen(false).add();
            network.newLine().setId("l" + g[0]).setVoltageLevel1("vl12").setNode1(n)   // charged: 0.32 pu a side
                    .setVoltageLevel2("v" + g[0]).setBus2("v" + g[0] + "_b").setConnectableBus2("v" + g[0] + "_b")
                    .setR(0.4).setX(6).setG1(0).setB1(2e-4).setG2(0).setB2(2e-4).add();
        }
        for (int k = 0; k < 2; k++) {
            int n = node++;
            String far = k == 0 ? "vl0" : "vl3";
            vl12.getNodeBreakerView().newDisconnector().setId("dl" + k).setNode1(k).setNode2(n + 50).setOpen(false).add();
            vl12.getNodeBreakerView().newBreaker().setId("bl" + k).setNode1(n + 50).setNode2(n).setOpen(false).add();
            network.newLine().setId("l" + k).setVoltageLevel1("vl12").setNode1(n)
                    .setVoltageLevel2(far).setBus2(far + "_b").setConnectableBus2(far + "_b")
                    .setR(0.4).setX(10).setG1(0).setB1(0).setG2(0).setB2(0).add();
        }
        network.getVoltageLevel("vl0").newGenerator().setId("gs").setBus("vl0_b").setConnectableBus("vl0_b")
                .setMinP(0).setMaxP(2000).setTargetP(500).setTargetV(400).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl0").newLoad().setId("lds").setBus("vl0_b").setConnectableBus("vl0_b").setP0(500).setQ0(100).add();
        network.getVoltageLevel("vl3").newLoad().setId("ld3").setBus("vl3_b").setConnectableBus("vl3_b").setP0(240).setQ0(160).add();
        network.newLine().setId("l03").setVoltageLevel1("vl0").setBus1("vl0_b").setConnectableBus1("vl0_b")
                .setVoltageLevel2("vl3").setBus2("vl3_b").setConnectableBus2("vl3_b").setR(0.4).setX(10).setG1(0).setB1(0).setG2(0).setB2(0).add();
        List<Contingency> ctgs = List.of(new Contingency("bus-1", new com.powsybl.contingency.BusbarSectionContingency("bbs1")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters();
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl0"));
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp, true);
    }

    /** A line outage islands one of a remote control's three controllers together with the bus it hangs from, so
     *  the charged line between them is INSIDE the island. OLF disables both buses and the line; the device froze the
     *  buses at 1 pu and kept the line, whose charging is part of gc's reactive injection and reached gb's sharing
     *  row: 0.25 x -0.32 pu, the survivors' split 384.2 / 117.4 MVAr against OLF's 376.1 / 125.4 (their 30:10 keys). */
    @Test
    void deviceIslandedControllersChargedLineMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = Network.create("islanded-controller-line", "code");
        com.powsybl.iidm.network.Substation s = network.newSubstation().setId("s").add();
        java.util.function.Function<String, com.powsybl.iidm.network.VoltageLevel> bbVl = id -> {
            com.powsybl.iidm.network.VoltageLevel v = s.newVoltageLevel().setId(id).setNominalV(400)
                    .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
            v.getBusBreakerView().newBus().setId(id + "_b").add();
            return v;
        };
        for (String vl : List.of("vl0", "vl3", "vp", "vx")) {
            bbVl.apply(vl);
        }
        java.util.function.BiConsumer<String, String[]> line = (id, ends) -> network.newLine().setId(id)
                .setVoltageLevel1(ends[0]).setBus1(ends[0] + "_b").setConnectableBus1(ends[0] + "_b")
                .setVoltageLevel2(ends[1]).setBus2(ends[1] + "_b").setConnectableBus2(ends[1] + "_b")
                .setR(0.4).setX(Double.parseDouble(ends[2])).setG1(0).setB1(Double.parseDouble(ends[3]))
                .setG2(0).setB2(Double.parseDouble(ends[3])).add();
        String[][] gens = {{"ga", "30"}, {"gb", "10"}, {"gc", "40"}};
        for (String[] g : gens) {
            com.powsybl.iidm.network.VoltageLevel vg = bbVl.apply("v" + g[0]);
            vg.newGenerator().setId(g[0]).setBus("v" + g[0] + "_b").setConnectableBus("v" + g[0] + "_b")
                    .setMinP(0).setMaxP(500).setTargetP(60).setTargetQ(0).setTargetV(405).setVoltageRegulatorOn(true).add();
            network.getGenerator(g[0]).newMinMaxReactiveLimits().setMinQ(-3000).setMaxQ(3000).add();
            network.getGenerator(g[0]).newExtension(com.powsybl.iidm.network.extensions.CoordinatedReactiveControlAdder.class)
                    .withQPercent(Double.parseDouble(g[1])).add();
        }
        network.getVoltageLevel("vp").newLoad().setId("ldp").setBus("vp_b").setConnectableBus("vp_b").setP0(100).setQ0(150).add();
        for (String[] g : gens) {                                          // all three regulate vp remotely
            network.getGenerator(g[0]).setRegulatingTerminal(network.getLoad("ldp").getTerminal());
        }
        line.accept("lga", new String[] {"vga", "vp", "6", "0"});
        line.accept("lgb", new String[] {"vgb", "vp", "6", "0"});
        line.accept("lgc", new String[] {"vgc", "vx", "6", "2e-4"});    // inside the island once lx goes
        line.accept("lx", new String[] {"vx", "vp", "6", "0"});
        line.accept("lp0", new String[] {"vp", "vl0", "10", "0"});
        line.accept("lp3", new String[] {"vp", "vl3", "10", "0"});
        line.accept("l03", new String[] {"vl0", "vl3", "10", "0"});
        network.getVoltageLevel("vl0").newGenerator().setId("gs").setBus("vl0_b").setConnectableBus("vl0_b")
                .setMinP(0).setMaxP(2000).setTargetP(400).setTargetV(400).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl0").newLoad().setId("lds").setBus("vl0_b").setConnectableBus("vl0_b").setP0(400).setQ0(100).add();
        network.getVoltageLevel("vl3").newLoad().setId("ld3").setBus("vl3_b").setConnectableBus("vl3_b").setP0(180).setQ0(80).add();
        List<Contingency> ctgs = List.of(new Contingency("lx", new BranchContingency("lx")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters();
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl0"));
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp, true);
    }

    /** Losing either hub re-elects OLF's coupler tree through the other star. The device packed the merged controls'
     *  reactive coupler terms for the BASE tree only and put a voltage identity on every child of an opened coupler:
     *  the re-coupled star's reactive flow was in no equation (network.xiidm busbar HFU: 1.25e12 pu circulating, the
     *  answer moving with run-to-run noise; HFUS/HFUT: linear solve failed in every run). Every section on the device,
     *  bus voltages to 1e-9 kV.
     *  By exception 1e-8 kV under nested dissection: a generator sits on its reactive limit with a margin of exactly
     *  zero, which OLF's KLU and AMD solve exactly while nested dissection lands one ulp above it; the generator switches
     *  back to PV for one outer pass and to PQ again at the same fixed point, 4.97e-9 kV from OLF's answer in each of 8 runs
     *  (5.7e-14 under AMD; it passes at convEps 1e-12). */
    @Test
    void deviceDoubleStarSectionsReTreeMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        List<Contingency> ctgs = new java.util.ArrayList<>();
        for (int k = 0; k < 5; k++) {
            ctgs.add(new Contingency("bus-" + k, new com.powsybl.contingency.BusbarSectionContingency("bbs" + k)));
        }
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters();
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl0"));
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(doubleStarSectionsNetwork(), ctgs, nestedDissectionOrdering(lfp) ? 1e-8 : 1e-9, lfp, true);
        assertEquals(java.util.Set.of(), GpuBatchScopeCensus.planRefusedIds(), "census=" + GpuBatchScopeCensus.snapshot());
        assertEquals(5, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** Losing bbs2 re-merges the other two sections' controls onto vl12_3. OLF keeps BUS_TARGET_V on that main while
     *  ANY merged controller is enabled; the device tied the voltage identity to the main's own controller, so when g3
     *  switched to PQ (100 MVAr, a 10% key) the cluster was left with no voltage equation while g4 still had 300 MVAr of
     *  headroom: the solve diverged (status -2) and the scenario went back to the CPU. g4 takes the target, every
     *  contingency on device, bus voltages to 1e-9 kV. */
    @Test
    void deviceReMergedMainSwitchesToPqAndTheNextControllerHoldsTheVoltageMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = doubleStarSectionsNetwork();
        String[][] generators = {{"g2", "3000", "80"}, {"g3", "100", "10"}, {"g4", "300", "10"}};   // id, max MVAr, key
        for (String[] g : generators) {
            network.getGenerator(g[0]).newMinMaxReactiveLimits().setMinQ(-300).setMaxQ(Double.parseDouble(g[1])).add();
            network.getGenerator(g[0]).newExtension(com.powsybl.iidm.network.extensions.CoordinatedReactiveControlAdder.class)
                    .withQPercent(Double.parseDouble(g[2])).add();
        }
        List<Contingency> ctgs = new java.util.ArrayList<>();
        for (int k = 0; k < 5; k++) {
            ctgs.add(new Contingency("bus-" + k, new com.powsybl.contingency.BusbarSectionContingency("bbs" + k)));
        }
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters();
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl0"));
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(network, ctgs, 1e-9, lfp, true);
        assertEquals(java.util.Set.of(), GpuBatchScopeCensus.planRefusedIds(), "census=" + GpuBatchScopeCensus.snapshot());
        assertEquals(5, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** The same re-merge with the main SWITCHED TO PQ in the base (g3 at its 30 MVAr ceiling) while g4 has 400 MVAr of headroom.
     *  OLF keeps vl12_3 PQ at its limit and its BUS_TARGET_V through g4; the device refused a contingency whose re-merged
     *  main was PQ in the base. vl12_3 sits under the live coupler c03: its own reactive balance starts on c03's DUMMY_Q row and
     *  g4's sharing row holds the target. Every contingency on device, bus voltages to 1e-9 kV. */
    @Test
    void deviceReMergedMainSwitchedToPqInTheBaseUnderALiveCouplerMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = doubleStarSectionsNetwork();
        String[][] generators = {{"g2", "3000", "80"}, {"g3", "30", "10"}, {"g4", "400", "10"}};   // id, max MVAr, key
        for (String[] g : generators) {
            network.getGenerator(g[0]).newMinMaxReactiveLimits().setMinQ(-300).setMaxQ(Double.parseDouble(g[1])).add();
            network.getGenerator(g[0]).newExtension(com.powsybl.iidm.network.extensions.CoordinatedReactiveControlAdder.class)
                    .withQPercent(Double.parseDouble(g[2])).add();
        }
        List<Contingency> ctgs = new java.util.ArrayList<>();
        for (int k = 0; k < 5; k++) {
            ctgs.add(new Contingency("bus-" + k, new com.powsybl.contingency.BusbarSectionContingency("bbs" + k)));
        }
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters();
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl0"));
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(network, ctgs, 1e-9, lfp, true);
        assertEquals(java.util.Set.of(), GpuBatchScopeCensus.planRefusedIds(), "census=" + GpuBatchScopeCensus.snapshot());
        assertEquals(5, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** Losing the chain's root section bbs0 opens c01 and re-merges the other two sections' controls onto vl12_1, the
     *  opened coupler's child, whose voltage identity the opening path installs. Its own controller was never tested
     *  against its limits there: g1 kept 1.97 pu against its 1.6 pu ceiling and the sections held 405 kV, where OLF
     *  switches g1 to PQ, hands the target to g2, switches g2 (2.2 pu) to PQ too and lets the sections fall to 404.79 kV. Every
     *  contingency on device, bus voltages to 1e-9 kV. */
    @Test
    void deviceReMergedMainBehindItsOpenedCouplerSwitchesToPqMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        List<Contingency> ctgs = new java.util.ArrayList<>();
        for (int k = 0; k < 3; k++) {
            ctgs.add(new Contingency("bus-" + k, new com.powsybl.contingency.BusbarSectionContingency("bbs" + k)));
        }
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters();
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl0"));
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(selfControlledSectionChainNetwork(), ctgs, 1e-9, lfp, true);
        assertEquals(java.util.Set.of(), GpuBatchScopeCensus.planRefusedIds(), "census=" + GpuBatchScopeCensus.snapshot());
        assertEquals(3, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** Every generator of the chain switched to PQ at its ceiling in the base (a 1200 MVAr load on bbs0 holds the sections at
     *  396.5 kV): losing bbs0 re-merges the other two onto vl12_1 with no enabled controller, OLF's BUS_TARGET_V off. Their
     *  6 pu then push the sections to 408.3 kV, above the target, and OLF switches both back to PV in one pass: the main takes the
     *  voltage back, g2 shares. The device forbade that switch back to PV and sent the contingency back to the CPU. */
    @Test
    void deviceReMergedControllersAllSwitchBackToPvMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        List<Contingency> ctgs = new java.util.ArrayList<>();
        for (int k = 0; k < 3; k++) {
            ctgs.add(new Contingency("bus-" + k, new com.powsybl.contingency.BusbarSectionContingency("bbs" + k)));
        }
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters();
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl0"));
        Network network = selfControlledSectionChainNetwork(new double[][] {{-300, 300, 60}, {-300, 300, 20}, {-300, 300, 20}},
                new double[] {1200, 0, 0});
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(network, ctgs, 1e-9, lfp, true);
        assertEquals(java.util.Set.of(), GpuBatchScopeCensus.planRefusedIds(), "census=" + GpuBatchScopeCensus.snapshot());
        assertEquals(3, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    @Test
    void deviceMiddleSectionLossRenormalizesDistrQTargetMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertMergedPilotLossMatchesCpu(middleSectionLossNetwork(false),
                List.of(new Contingency("bus-mid", new com.powsybl.contingency.BusbarSectionContingency("bbs1"))));
    }

    /** The same loss with a fixed SHUNT on the islanded section (rte6515 BUS-1251_BBS_1): the shunt
     *  feeds the group's DISTR_Q sums, which used to reject the scenario (a masked shunt would not
     *  update the sums). On an islanded bus the shunt's row is frozen and the member's weights are
     *  renormalized to zero, so the shift is simply not a shift for the live network. */
    @Test
    void deviceMiddleSectionLossWithShuntOnIslandMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertMergedPilotLossMatchesCpu(middleSectionLossNetwork(true),
                List.of(new Contingency("bus-mid", new com.powsybl.contingency.BusbarSectionContingency("bbs1"))));
    }

    /** A BRANCH outage combined with a GENERATOR outage on the bus that branch islands, where the lost
     *  generator is a member of a remote voltage-control group (the group survives and renormalizes its
     *  reactive keys over the two members left). The combo path used to demand a fully CONTAINED island
     *  and sent every other shape to the CPU replay; on network.xiidm 65 of 11249 branch contingencies
     *  were exactly this (a line whose loss islands one generator bus). GPU SA == CPU SA at 1e-12, and
     *  the contingency must BATCH. */
    @Test
    void deviceBranchOutageIslandingAGroupMemberBatchesAndMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = Network.create("island-group-member", "code");
        com.powsybl.iidm.network.Substation s = network.newSubstation().setId("s").add();
        for (String vl : List.of("vl0", "vlp", "vlc1", "vlc2", "vlr")) {
            com.powsybl.iidm.network.VoltageLevel v = s.newVoltageLevel().setId(vl).setNominalV(400)
                    .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
            v.getBusBreakerView().newBus().setId(vl + "_b").add();
        }
        network.getVoltageLevel("vl0").newGenerator().setId("g0").setBus("vl0_b").setConnectableBus("vl0_b")
                .setMinP(0).setMaxP(2000).setTargetP(400).setTargetV(405).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl0").newLoad().setId("ld0").setBus("vl0_b").setConnectableBus("vl0_b").setP0(200).setQ0(40).add();
        network.getVoltageLevel("vlp").newLoad().setId("ldp").setBus("vlp_b").setConnectableBus("vlp_b").setP0(400).setQ0(120).add();
        network.getVoltageLevel("vlr").newLoad().setId("ldr").setBus("vlr_b").setConnectableBus("vlr_b").setP0(60).setQ0(20).add();
        // three remote members of one group (pilot = the ldp bus); gr sits on the radial bus vlr
        for (String[] c : new String[][] {{"c1", "vlc1"}, {"c2", "vlc2"}, {"r", "vlr"}}) {
            network.getVoltageLevel(c[1]).newGenerator().setId("g" + c[0]).setBus(c[1] + "_b")
                    .setConnectableBus(c[1] + "_b").setMinP(0).setMaxP(500).setTargetP(120).setTargetQ(0)
                    .setTargetV(404).setVoltageRegulatorOn(true).setRegulatingTerminal(network.getLoad("ldp").getTerminal()).add();
            network.getGenerator("g" + c[0]).newMinMaxReactiveLimits().setMinQ(-150).setMaxQ(150).add();
        }
        java.util.function.BiConsumer<String, String[]> line = (id, ends) -> network.newLine().setId(id)
                .setVoltageLevel1(ends[0]).setBus1(ends[0] + "_b").setConnectableBus1(ends[0] + "_b")
                .setVoltageLevel2(ends[1]).setBus2(ends[1] + "_b").setConnectableBus2(ends[1] + "_b")
                .setR(0.4).setX(10).setG1(0).setB1(0).setG2(0).setB2(0).add();
        line.accept("l0p", new String[] {"vl0", "vlp"});
        line.accept("lpc1", new String[] {"vlp", "vlc1"});
        line.accept("lpc2", new String[] {"vlp", "vlc2"});
        line.accept("l0c1", new String[] {"vl0", "vlc1"});
        line.accept("l0c2", new String[] {"vl0", "vlc2"});
        line.accept("lpr", new String[] {"vlp", "vlr"});     // the RADIAL feeder of the third member
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl0_0"));
        // the feeder AND the generator behind it: the island takes one member of the group
        assertGpuMatchesCpu(network, List.of(new Contingency("lpr+gr",
                new com.powsybl.contingency.LineContingency("lpr"),
                new com.powsybl.contingency.GeneratorContingency("gr"))), 1e-12, lfp);
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "the branch+generator combo islanding a group member must BATCH (or the test proves nothing)");
    }

    /** A busbar-section loss that islands the PILOT (controlled section) of a voltage control whose
     *  other controller sits on the surviving section (rte6515 BUS-1055_BBS_1, BUS-11_BBS_1,
     *  BUS-1263_BBS, BUS-1282_BBS_1, BUS-1320_BBS_1 - up to 1.3 kV). OLF: the control dies, g2's
     *  section goes PQ at g2's targetQ. On the device the survivor is a packed DISTR_Q member, so its
     *  row must host its OWN reactive balance (BUS_TARGET_Q), not a plain reactive row (EQUATION) the DISTR_Q
     *  family keeps writing into. Bare section loss: the island-injection path. */
    @Test
    void deviceBusLossIslandingMergedPilotSurvivorMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertMergedPilotLossMatchesCpu(mergedPilotLossNetwork(),
                List.of(new Contingency("bus-vl1", new com.powsybl.contingency.BusbarSectionContingency("bbs1"))));
    }

    /** A RETAINED SWITCH carrying real power, across MORE THAN ONE solve — the smallest network
     *  that exhibits the carried-dummy divergence. OLF's {@code AcSolverUtil.initStateVector}
     *  re-zeroes every DUMMY_P/DUMMY_Q at the start of each solve (v and phi carry over); the
     *  in-kernel driver never returns to the host between passes and carries them instead, so the
     *  two arms ENTER every solve after the first from different states. On the L2 outage here,
     *  measured: the CPU enters solve 2 at {@code ||F||2 = 6.51796067833562400e+00} — exactly
     *  VL1_1's whole target, its injection reading zero because the dummy was cleared — while the
     *  device enters at {@code 4.56467216750267163e-02}. On rte6515 the same shape appears at the
     *  second solve of nearly every busbar outage, at exactly 2.7e-02 on the two ends of
     *  BUS-1014_COUPLER_1.
     *
     *  <p>BE HONEST ABOUT WHAT THIS ASSERTS: the final states agree here to 1e-12 with or without
     *  the device-side reset, because this system converges to round-off (1e-15) and squeezes the
     *  trajectory difference out. rte6515 stops at 1.7e-05 and does not. So this test is COVERAGE
     *  for the shape, not a regression test for the reset — the divergence is only visible in the
     *  per-solve entry residuals (OLF_NR_ITER_TRACE against OLF_GPU_BATCH_TRACE). */
    @Test
    void deviceRetainedSwitchCarryingPowerMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.NodeBreakerNetworkFactory.create();
        // B3 (VL2 node 0-1) is declared RETAINED, but a switch is only actually retained when the
        // SA needs the breaker view - with branch contingencies alone OLF merges it away and there
        // is no zero-impedance branch to carry anything. The busbar-section outage in the list is
        // what forces retention; L2's outage is then the one that pushes every MW of LD through B3.
        List<Contingency> ctgs = List.of(
                Contingency.busbarSection("BBS1"),
                new Contingency("L2", new BranchContingency("L2")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }

    /** A voltage-dependent (ZIP) load on a relocated coupler-tree child whose coupler opens. The relocation puts the
     *  child's balances on its parent coupler's dummy rows, and the ZIP family wrote its terms there in every lane:
     *  once the coupler opens those rows are identities, and the load's voltage-dependent part vanished from the
     *  child's balance. The three-section chain with a ZIP load on every section, each coupler opened in turn (both
     *  sides stay energized). 1e-12 on device. */
    @Test
    void deviceZipLoadOnAReMergedSectionMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = liveSectionChainNetwork();
        com.powsybl.iidm.network.VoltageLevel vl12 = network.getVoltageLevel("vl12");
        for (int k = 0; k < 3; k++) {
            vl12.getNodeBreakerView().newBreaker().setId("bz" + k).setNode1(k).setNode2(50 + k).setOpen(false).add();
            vl12.newLoad().setId("zl" + k).setNode(50 + k).setP0(20 + 5 * k).setQ0(8)
                    .newZipModel().setC0p(0.2).setC0q(0.3).setC1p(0.3).setC1q(0.3).setC2p(0.5).setC2q(0.4).add()
                    .add();
        }
        List<Contingency> ctgs = List.of(new Contingency("c01", new com.powsybl.contingency.SwitchContingency("c01")),
                new Contingency("c12", new com.powsybl.contingency.SwitchContingency("c12")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp).setUseLoadModel(true)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl0"));
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp, true);
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** A section loss that leaves BOTH sides of the opened coupler energized: the section's bus survives through its
     *  line, its load is lost, the coupler(s) open. The device solves it (no refusal) and matches OLF, on the root
     *  section and on the relocated children alike - a child's lost load must be taken off the child's OWN rows,
     *  where the opening put its balances back, not off the coupler's rows it had in the base. */
    @Test
    void deviceSectionLossLeavingBothCouplerSidesEnergizedMatchesCpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = liveSectionChainNetwork();
        List<Contingency> ctgs = new java.util.ArrayList<>();
        for (int k = 0; k < 3; k++) {
            ctgs.add(new Contingency("bus-" + k, new com.powsybl.contingency.BusbarSectionContingency("bbs" + k)));
        }
        // the fixture's shape, from OLF's own propagation: each section loss opens its coupler(s) and loses its load,
        // and nothing else - no line, no bus
        var pcs = com.powsybl.openloadflow.network.impl.PropagatedContingency.createList(network, ctgs,
                new com.powsybl.openloadflow.network.LfTopoConfig(),
                new com.powsybl.openloadflow.network.impl.PropagatedContingencyCreationParameters());
        assertEquals(java.util.Set.of("c01"), pcs.get(0).getBranchIdsToOpen().keySet());
        assertEquals(java.util.Set.of("c01", "c12"), pcs.get(1).getBranchIdsToOpen().keySet());
        assertEquals(java.util.Set.of("c12", "bf"), pcs.get(2).getBranchIdsToOpen().keySet());
        for (int k = 0; k < 3; k++) {
            assertEquals(java.util.Set.of("ld" + k), pcs.get(k).getLoadIdsToLose().keySet(), "bus-" + k);
        }
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl0"));
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp, true);
        assertEquals(java.util.Set.of(), GpuBatchScopeCensus.planRefusedIds(), "census=" + GpuBatchScopeCensus.snapshot());
        assertEquals(3, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** A line outage whose propagation also trips a load OUTSIDE the solved component: the load's level hangs off a
     *  line that is open on the main side, so it is its own component, and the branch tripping of that line still
     *  disconnects the load behind its disconnector. OLF resolves loadIdsToLose with getLoadById and skips the null
     *  ("could be in another component") - the mapper refused the whole contingency on it (network.xiidm's fixed list
     *  CONTINGENCY_13703, lines LOSL + LOSM, load EAAK), while the main-side line outage it carries is a plain batch. */
    @Test
    void deviceSkipsLostLoadOutsideTheSolvedComponentLikeOlf() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        com.powsybl.iidm.network.VoltageLevel vx = network.newVoltageLevel().setId("vx").setNominalV(135)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.NODE_BREAKER).add();
        vx.getNodeBreakerView().newBusbarSection().setId("bbsx").setNode(0).add();
        vx.getNodeBreakerView().newDisconnector().setId("dx").setNode1(0).setNode2(1).setOpen(false).add();
        vx.getNodeBreakerView().newBreaker().setId("bx").setNode1(1).setNode2(2).setOpen(false).add();
        vx.newLoad().setId("ldx").setNode(2).setP0(12).setQ0(4).add();
        vx.getNodeBreakerView().newDisconnector().setId("dlx").setNode1(0).setNode2(3).setOpen(false).add();
        network.newLine().setId("lx").setVoltageLevel1("VL4").setConnectableBus1("B4")   // open at the main side
                .setVoltageLevel2("vx").setNode2(3).setR(0.5).setX(5).setG1(0).setB1(0).setG2(0).setB2(0).add();
        List<Contingency> ctgs = List.of(new Contingency("l23+lx", new BranchContingency("L2-3-1"), new BranchContingency("lx")));
        var propagatedContingency = com.powsybl.openloadflow.network.impl.PropagatedContingency.createList(network, ctgs,
                new com.powsybl.openloadflow.network.LfTopoConfig(),
                new com.powsybl.openloadflow.network.impl.PropagatedContingencyCreationParameters()).get(0);
        assertEquals(java.util.Set.of("ldx"), propagatedContingency.getLoadIdsToLose().keySet(), "the propagation trips the far load");
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(network, ctgs, 1e-12);
        assertEquals(java.util.Set.of(), GpuBatchScopeCensus.planRefusedIds(), "census=" + GpuBatchScopeCensus.snapshot());
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** An outage that isolates the slack: OLF re-picks one in the main component with the run's own selector, its
     *  angle becomes the reference and the distributed slack reads its balance. Batched beside a plain outage. */
    @Test
    void deviceIsolatedSlackRelocatesLikeOlf() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = isolatedSlackHubNetwork();
        String plain = network.getLineStream().map(l -> l.getId()).filter(id -> !id.startsWith("l")).findFirst().orElseThrow();
        List<Contingency> ctgs = List.of(
                new Contingency("lH", new BranchContingency("lH")),
                new Contingency(plain, new BranchContingency(plain)));
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(network, ctgs, 1e-12, new com.powsybl.loadflow.LoadFlowParameters(), true);
        assertEquals(java.util.Set.of(), GpuBatchScopeCensus.planRefusedIds(), "census=" + GpuBatchScopeCensus.snapshot());
        assertTrue(GpuBatchScopeCensus.snapshot().containsKey("PLAN:SLACK_RELOCATED"),
                "the outage must take the relocation path: census=" + GpuBatchScopeCensus.snapshot());
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** A coupler opened by a SwitchContingency that islands nothing still SPLITS the merged voltage control of its two
     *  sections: OLF re-merges per zero-impedance cluster, so g1 regulates bbs1 on its own, at its own target. The
     *  device planned that re-merging only for outages that island a bus; here the entries stayed the base group's, and a
     *  member the base switched to PQ went back, once released, to the group's sharing row while the group's voltage identity
     *  came back on bbs0, whose generator is still PQ: a system OLF does not have. Both sections switched to PQ at MAX_Q in the
     *  base: the device diverged (MAX_ITERATION_REACHED) where OLF converges after releasing g1; both MIN_Q: the same;
     *  g1 alone switched to PQ at MIN_Q: converged 5.6 kV off at vl3. */
    @Test
    void deviceCouplerOpeningSplitsASwitchedToPqInBaseMergedControlMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        for (double[] shape : new double[][] {{60, 200}, {60, 40}, {300, 200}}) {   // {g0/g1 max Q, ld0 Q}
            Network network = splitGroupNetwork(2);
            for (String g : List.of("g0", "g1")) {
                network.getGenerator(g).setTargetV(398);
                network.getGenerator(g).newMinMaxReactiveLimits().setMinQ(-5).setMaxQ(shape[0]).add();
            }
            network.getLoad("ld0").setQ0(shape[1]);
            List<Contingency> ctgs = List.of(new Contingency("c01", new com.powsybl.contingency.SwitchContingency("c01")));
            assertStrategyMatchesCpu(network, ctgs, List.of(), List.of(), 1, true,
                    p -> p.setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                            .setSlackBusesIds(List.of("vl0_b")));
        }
    }

    /** The same split in BREAKER mode, every member free: a busbar-section contingency in the list makes OLF model every
     *  switch, so opening c01 separates the generator nodes of the coupled curve sections into two clusters, each with
     *  its own control. Without the re-merging the device kept the merged control's layout and held bbs0 at 400.9745 kV
     *  where OLF holds its 401.0 target (the section alone, in bus-breaker topology, happened to come out right). */
    @Test
    void deviceCouplerOpeningInBreakerModeMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = pilotSelfCurveNetwork(true, 100);
        List<Contingency> ctgs = List.of(new Contingency("bus-0", new com.powsybl.contingency.BusbarSectionContingency("bbs0")),
                new Contingency("c01", new com.powsybl.contingency.SwitchContingency("c01")));
        assertStrategyMatchesCpu(network, ctgs, List.of(), List.of(), 2, true,
                p -> p.setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                        .setSlackBusesIds(List.of("vl0_b")));
    }

    /** A lost load's bus must come from OLF's OWN load map, never from the bus view. On a node-breaker
     *  level the bus-view buses and the breaker-mode LfBuses share the "<VL>_<n>" id pattern but
     *  partition the level differently: the bus view merges across the closed RETAINED breaker B3,
     *  so LD1 (behind B3, next to L1's terminal) reports bus-view bus "VL2_0" while its LfBus is
     *  "VL2_1". The propagated L1 outage opens B3 and loses LD1: resolved through the bus view the
     *  device removed LD1's 30 MW / 10 MVar from the LIVE busbar VL2_0 instead of the frozen stub
     *  (measured on network.xiidm / LJBS: 3.1 kV). GPU SA == CPU SA at 1e-12. */
    @Test
    void deviceLostLoadBehindRetainedBreakerResolvesItsOwnBusMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.NodeBreakerNetworkFactory.create();
        var vl1 = network.getVoltageLevel("VL1");
        var vl2 = network.getVoltageLevel("VL2");
        vl1.getNodeBreakerView().newBreaker().setId("B5").setNode1(1).setNode2(7).add();
        vl2.getNodeBreakerView().newBreaker().setId("B6").setNode1(0).setNode2(4).add();
        network.newLine().setId("L3").setVoltageLevel1("VL1").setNode1(7).setVoltageLevel2("VL2").setNode2(4)
                .setR(3).setX(33).setG1(0).setB1(0).setG2(0).setB2(0).add();
        // LD1 on L1's side of the retained breaker B3 (VL2 node 1), on its own node
        vl2.getNodeBreakerView().newInternalConnection().setNode1(1).setNode2(5).add();
        vl2.newLoad().setId("LD1").setNode(5).setP0(30).setQ0(10).add();
        assertGpuMatchesCpu(network, List.of(new Contingency("L1", new BranchContingency("L1"))), 1e-12, true);
    }
}
