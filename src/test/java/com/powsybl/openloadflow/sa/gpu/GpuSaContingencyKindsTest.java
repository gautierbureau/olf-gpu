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
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.GpuBatchScopeCensus;
import org.junit.jupiter.api.Test;

import java.util.List;

import static com.powsybl.openloadflow.sa.gpu.GpuSaTestNetworks.*;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * GPU-batched security analysis against the CPU: contingency kinds: load and generator losses, combinations, side-opened and boundary lines.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuSaContingencyKindsTest extends AbstractGpuSaTest {

    GpuSaContingencyKindsTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    /** N-k branch outages (the former <=2 cap was validated-scope only, lifted 2026-09-08): the
     *  per-element mask carries any outage count and the validator rejects fragmenting sets — an
     *  N-3 and an N-4 on ieee14 batch alongside a plain N-1 at 1e-12, count asserted. */
    @Test
    void deviceMultiBranchOutageMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory.create14();
        // ieee14's tight Q limits collapse under N-3-class redistributions (the CPU MAX_ITERATIONs
        // too) — widen them so the scenarios are physically feasible on BOTH arms
        network.getGenerators().forEach(g -> g.newMinMaxReactiveLimits().setMinQ(-300).setMaxQ(300).add());
        List<Contingency> ctgs = List.of(
                new Contingency("N-1", new BranchContingency("L2-3-1")),
                new Contingency("N-3", new BranchContingency("L2-4-1"), new BranchContingency("L3-4-1"),
                        new BranchContingency("L4-5-1")),
                new Contingency("N-4", new BranchContingency("L2-4-1"), new BranchContingency("L3-4-1"),
                        new BranchContingency("L4-5-1"), new BranchContingency("L2-5-1")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(3, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "N-1, N-3 and N-4 must all batch on-device");
    }

    /** A FIXED-Q generator lost from a bus its sibling keeps in voltage control. A bus's reactive
     *  aggregate is NOT the sum of its generators' getMinQ()/getMaxQ(): OLF's AbstractLfBus.getLimitQ
     *  takes a generator's LIMIT only when it controls voltage (or remote reactive power) and its
     *  TARGET Q otherwise. So B6-Q, regulating nothing, contributes its 5 MVAr target to the aggregate
     *  and not its +-100 MVAr capability, and that 5 MVAr is all the outage may take off.
     *
     *  <p>B6-G survives and holds B6 in voltage control with tight limits, so the bus switches to PQ and the
     *  aggregate reaches the answer; charging the lost generator's limit instead moves the frozen
     *  target by 100 MVAr rather than 5. Measured on rte6515 GEN-2899 and GEN-820#0 as 0.32 kV and
     *  0.19 kV, five differing violations between them — invisible until the generator set was widened
     *  below 220 kV, where the fixed-Q machines live. VERIFIED by reverting to the plain sum and
     *  watching this test fail. */
    @Test
    void deviceLosingFixedQGeneratorFromControllingBusMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        com.powsybl.iidm.network.Generator g1 = network.getGenerator("B1-G");
        g1.setTargetP(g1.getTargetP() - 5);                  // keep the case's generation/load balance
        com.powsybl.iidm.network.Generator bq = network.getVoltageLevel("VL6").newGenerator()
                .setId("B6-Q").setBus("B6").setConnectableBus("B6")
                .setMinP(0).setMaxP(50).setTargetP(5).setTargetQ(5).setVoltageRegulatorOn(false)
                .add();
        bq.newMinMaxReactiveLimits().setMinQ(-100).setMaxQ(100).add();   // wide, and NOT what counts
        // Out of the slack distribution: losing a PARTICIPATING generator while the slack bus also
        // participates is a CPU route by design (the retain math does not hold), and would take the
        // scenario off the device before the aggregate is ever read. Its 5 MW still shifts the row
        // target, which is a pure base shift and keeps the factors exact.
        bq.newExtension(com.powsybl.iidm.network.extensions.ActivePowerControlAdder.class)
                .withParticipate(false).withParticipationFactor(0).add();
        network.getGenerator("B6-G").newMinMaxReactiveLimits().setMinQ(-5).setMaxQ(5).add();
        for (String wide : List.of("B1-G", "B2-G", "B3-G", "B8-G")) {    // only B6 switches to PQ
            network.getGenerator(wide).newMinMaxReactiveLimits().setMinQ(-990).setMaxQ(990).add();
        }
        List<Contingency> ctgs = List.of(
                new Contingency("B6-Q", new com.powsybl.contingency.GeneratorContingency("B6-Q")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        assertGpuMatchesCpu(network, ctgs, 1e-11, lfp);
    }

    /** A busbar-section loss that islands the PILOT of a merged voltage control while its merged
     *  dependent section survives, connected on its own (rte6515 BUS-1055_BBS_1, BUS-11_BBS_1,
     *  BUS-1263_BBS, BUS-1282_BBS_1, BUS-1320_BBS_1 - up to 1.3 kV): b1 (main, the pilot) and b2
     *  (dependent) are merged by the coupler c12; b2 keeps its own line to b0. OLF: the control dies,
     *  b2 goes PQ at its own generation Q. On the device the survivor is a member of the pilot's
     *  relocation group, so its row must host its OWN reactive balance (mode 4), not a plain reactive
     *  row (mode 1) the mode scan takes for a live carrier and lands the frozen pilot's balance on.
     *  <pre>
     *    ld0
     *     |
     *    b0 --- l01 --- b1 === c12 === b2 --- l20 --- b0
     *           g0      g1            g2
     *  </pre> */
    /** A BOUNDARY LINE outage. BoundaryLine extends Injection, not Branch, so this population sits
     *  outside every scope the harness had: it is not reached by the branch scope, and it is neither a
     *  generator nor a load. OLF models one as an LfBoundaryLineGenerator plus the boundary's own P0/Q0
     *  at the same bus, so losing it takes BOTH away at once — a shape no other kind produces.
     *
     *  <p>Neither production case carries one (network.xiidm and rte6515 both have zero boundary
     *  lines), so the full-scope runs cannot cover this and a fixture is the only coverage there is.
     *  bl1 carries P0 = 91 MW against a 101.3664 MW generator, so the outage is a large injection step
     *  and distributed slack has to move real power to absorb it. */
    @Test
    void deviceBoundaryLineOutageMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.BoundaryFactory.createWithLoad();
        List<Contingency> ctgs = List.of(
                new Contingency("bl1", new com.powsybl.contingency.BoundaryLineContingency("bl1")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        assertGpuMatchesCpu(network, ctgs, 1e-11, lfp);
    }

    /** The same section loss, but the line on that section is ALREADY open at its far end in the
     *  base. The propagation still puts l1 in branchIdsToOpen with a one-side status, and opening its
     *  remaining end leaves it connected at NEITHER end — so there is no dangling half left to inject
     *  anywhere and the outage is simply the element's disabled bit.
     *
     *  <p>closedBranchOutages already returns exactly this branch for exactly this reason (it is how
     *  2049 of network.xiidm's 11249 branch outages batch), but sideOpens() rejected the same entry,
     *  so the two halves of the same map disagreed and the contingency went to the CPU with its mask
     *  already built. Every side-open reject on network.xiidm was this one shape and nothing else:
     *  JHKH, LOJD, LQDZ on the branch scope and ACLA, BCR, FXSW, HRZA, VXD, YFS on the load scope.
     *
     *  <p>VERIFIED by restoring the reject and watching the engagement assert fail. As with the
     *  setpoint-HVDC case the ANSWER is right either way — the CPU fallback computes it — so only the
     *  engagement assert can tell the two apart. */
    @Test
    void deviceSideOpenOfAnAlreadyOpenEndedBranchBatches() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = middleSectionLossNetwork(false);
        network.getLine("l1").getTerminal2().disconnect();    // open-ended at vl3 in the BASE
        assertMergedPilotLossMatchesCpu(network,
                List.of(new Contingency("bus-mid", new com.powsybl.contingency.BusbarSectionContingency("bbs1"))));
    }

    /** The outage of a branch already OPEN AT ONE END in the base (network.xiidm: 2049 of 11249 branch
     *  contingencies): the device packs such a branch as an open-branch element (the connected end's
     *  view: its charging), so the outage is that element's disabled bit. The mapper used to call it
     *  unresolvable and every one of them replayed on the CPU. l12 carries charging (B1 = B2 = 1e-4 S)
     *  and is disconnected at vl2, so losing it moves vl1's reactive balance by a measurable amount. */
    @Test
    void deviceHalfOpenBranchOutageMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = Network.create("half-open-branch", "code");
        com.powsybl.iidm.network.Substation s = network.newSubstation().setId("s").add();
        for (String vl : List.of("vl0", "vl1", "vl2", "vl3")) {
            com.powsybl.iidm.network.VoltageLevel v = s.newVoltageLevel().setId(vl).setNominalV(400)
                    .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
            v.getBusBreakerView().newBus().setId(vl + "_b").add();
        }
        network.getVoltageLevel("vl0").newGenerator().setId("g0").setBus("vl0_b").setConnectableBus("vl0_b")
                .setMinP(0).setMaxP(2000).setTargetP(500).setTargetV(400).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl1").newLoad().setId("ld1").setBus("vl1_b").setConnectableBus("vl1_b").setP0(200).setQ0(60).add();
        network.getVoltageLevel("vl3").newLoad().setId("ld3").setBus("vl3_b").setConnectableBus("vl3_b").setP0(250).setQ0(90).add();
        network.getVoltageLevel("vl3").newGenerator().setId("g3").setBus("vl3_b").setConnectableBus("vl3_b")
                .setMinP(0).setMaxP(500).setTargetP(100).setTargetV(401).setVoltageRegulatorOn(true).add();
        java.util.function.BiConsumer<String, String[]> line = (id, ends) -> network.newLine().setId(id)
                .setVoltageLevel1(ends[0]).setBus1(ends[0] + "_b").setConnectableBus1(ends[0] + "_b")
                .setVoltageLevel2(ends[1]).setBus2(ends[1] + "_b").setConnectableBus2(ends[1] + "_b")
                .setR(0.4).setX(10).setG1(0).setB1(1e-4).setG2(0).setB2(1e-4).add();
        line.accept("l01", new String[] {"vl0", "vl1"});
        line.accept("l12", new String[] {"vl1", "vl2"});
        line.accept("l13", new String[] {"vl1", "vl3"});
        line.accept("l03", new String[] {"vl0", "vl3"});
        line.accept("l23", new String[] {"vl2", "vl3"});
        network.getLine("l12").getTerminal2().disconnect();   // open at the vl2 end in the base
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl0_0"));
        assertGpuMatchesCpu(network, List.of(new Contingency("l12", new com.powsybl.contingency.LineContingency("l12"))),
                1e-12, lfp);
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "a half-open branch outage must batch (or the test proves nothing)");
    }

    /** SIDE-OPEN reproducer, small and fast — the rte6515 busbar failure in 9 buses.
     *
     *  <p>A busbar-section contingency that disconnects ONE terminal of a line (not both) leaves OLF
     *  with a DANGLING branch whose connected-side charging still injects: OLF's own
     *  testLineOpenOneSideContingency measures q2 = -55.873 MVAr on exactly this shape. The device
     *  applies a SideOpen as a FULL element disable, so that injection vanishes.
     *
     *  <p>Measured on rte6515 BUS-1003_BBS the error is EXACTLY -2 x b_side x v^2 at every bus
     *  bordering the opened side (six buses, ratios -1.994 to -2.004): b1 in parallel with
     *  (series (+) b2) is the equivalent shunt OLF keeps and the device deletes. Iterating that on
     *  rte6515 costs minutes a run; this costs seconds. */
    @Test
    void deviceLineOpenOneSideMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.NodeBreakerNetworkFactory.create();
        // ...the same three lines OLF's own one-side-open test uses: break the internal connection
        // and re-make it through a breaker, so the busbar outage opens L1 at side 1 ONLY.
        com.powsybl.iidm.network.VoltageLevel vl1 = network.getVoltageLevel("VL1");
        vl1.getNodeBreakerView().removeInternalConnections(0, 5);
        vl1.getNodeBreakerView().newBreaker().setId("NBR").setNode1(0).setNode2(5).setOpen(false).add();
        List<Contingency> ctgs = List.of(Contingency.busbarSection("BBS1"));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("VL1_1"));
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }

    /** BOUNDARY LINES — a coverage hole, not a suspected bug: rte6515 contains ZERO BoundaryLine
     *  and ZERO TieLine (OLF_GPU_SA_INVENTORY), and the GPU main path has no reference to either, so
     *  nothing in this project has ever solved one. OLF models a boundary line as a boundary BUS plus
     *  a branch in the LfNetwork, which the packers iterate like any other bus and branch — so the
     *  expectation is that it simply works, and this test is what makes that a fact rather than an
     *  assumption. Both the outage of an ordinary branch WITH a boundary line present (the boundary
     *  bus must solve alongside it) and the outage of the boundary line ITSELF. */
    @Test
    void deviceBoundaryLineMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        // Injection ZERO. The boundary BUS and its branch exist either way, so the shape IS exercised; the
        // outage of a boundary line WITH its injection is deviceBoundaryLineOutageMatchesCpu. DL-9 could
        // not carry one here anyway: 12 ohm on the 12 kV level is 8.3 pu, which delivers about 4 MW at most
        // at the power factor of the 5/2 MW once tried on it, so OLF's own base load flow stops at its
        // iteration limit, and the device's with it (measured 2026-10-23; 8/3 MW on DL-4 alone converges
        // on both).
        addBoundaryLine(network, "VL4", "DL-4", 0, 0);
        addBoundaryLine(network, "VL9", "DL-9", 0, 0);
        List<Contingency> ctgs = List.of(
                new Contingency("L1-2-1", new BranchContingency("L1-2-1")),
                new Contingency("DL-4", new com.powsybl.contingency.BoundaryLineContingency("DL-4")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true)
                .setUseReactiveLimits(true);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }

    /** Voltage-dependent (ZIP) load model under batched SA: the load model family evaluates
     *  c_n*target*V^n on the bus P/Q rows per scenario; DS + RL on; parallel-line outage. */
    @Test
    void deviceZipLoadModelMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.EurostagFactory.fix(
                com.powsybl.iidm.network.test.EurostagTutorialExample1Factory.create());
        network.getVoltageLevel("VLLOAD").newLoad()
                .setId("ZIPLOAD").setBus("NLOAD").setP0(50).setQ0(30)
                .newZipModel()
                    .setC0p(0.5).setC0q(0.55)
                    .setC1p(0.3).setC1q(0.35)
                    .setC2p(0.2).setC2q(0.1)
                    .add()
                .add();
        List<Contingency> ctgs = List.of(new Contingency("NHV1_NHV2_1", new BranchContingency("NHV1_NHV2_1")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp).setUseLoadModel(true);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }

    /** The pilot of a remote voltage control keeps its own reactive balance, which rides the free member's row with a
     *  target the device captured from the BASE: a reactive load or a fixed-Q generator lost at the pilot stayed in
     *  that balance (the lost 20 MVar load: every controller bus 3.7e-2 kV off OLF). */
    @Test
    void deviceInjectionLossAtRemotePilotMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        List<Contingency> ctgs = List.of(
                new Contingency("l4b", new com.powsybl.contingency.LoadContingency("l4b")),
                new Contingency("g4q", new com.powsybl.contingency.GeneratorContingency("g4q")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters();
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl1"));
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(remotePilotInjectionsNetwork(), ctgs, 1e-12, lfp, true);
        assertEquals(java.util.Set.of(), GpuBatchScopeCensus.planRefusedIds(), "census=" + GpuBatchScopeCensus.snapshot());
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** A load lost at the SLACK bus, which has no P balance row: its scheduled P moves only what the distributed-slack
     *  check reads. Here the slack b4 is also the remote control's pilot, so its reactive balance moves too. */
    @Test
    void deviceLoadLossAtSlackPilotMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        List<Contingency> ctgs = List.of(new Contingency("l4b", new com.powsybl.contingency.LoadContingency("l4b")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters();
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl4"));
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(remotePilotInjectionsNetwork(), ctgs, 1e-12, lfp, true);
        assertEquals(java.util.Set.of(), GpuBatchScopeCensus.planRefusedIds(), "census=" + GpuBatchScopeCensus.snapshot());
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** A load lost at a generator slack bus whose generator takes part in the distribution (IEEE 14, a load added at
     *  B1): the slack keeps its retained share of a mismatch measured against its moved target. */
    @Test
    void deviceLoadLossAtParticipatingSlackMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        network.getVoltageLevel("VL1").newLoad()
                .setId("L1x")
                .setBus("B1")
                .setConnectableBus("B1")
                .setP0(20)
                .setQ0(10)
                .add();
        List<Contingency> ctgs = List.of(new Contingency("L1x", new com.powsybl.contingency.LoadContingency("L1x")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters();
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("VL1"));
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp, true);
        assertEquals(java.util.Set.of(), GpuBatchScopeCensus.planRefusedIds(), "census=" + GpuBatchScopeCensus.snapshot());
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** A load lost at a TRANSFORMER-controlled bus (T2wT regulates BUS_3, AFTER_GENERATOR): while the control is on,
     *  the bus's reactive balance rides the transformer's ratio row, whose target the layout took from the base. */
    @Test
    void deviceLoadLossAtTransformerControlledBusMatchesCpu() {
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
        network.getLoad("LOAD_3").getTerminal().getVoltageLevel().newLoad()
                .setId("LOAD_3b")
                .setBus("BUS_3")
                .setConnectableBus("BUS_3")
                .setP0(3)
                .setQ0(5)
                .add();
        List<Contingency> ctgs = List.of(new Contingency("LOAD_3b", new com.powsybl.contingency.LoadContingency("LOAD_3b")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setTransformerVoltageControlOn(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL);
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp, true);
        assertEquals(java.util.Set.of(), GpuBatchScopeCensus.planRefusedIds(), "census=" + GpuBatchScopeCensus.snapshot());
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "census=" + GpuBatchScopeCensus.snapshot());
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
                List.of(new Contingency(loadId, new com.powsybl.contingency.LoadContingency(loadId))), 1e-12, true);
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
                        new com.powsybl.contingency.LoadContingency(loadId))), 1e-12, false);
    }

    @Test
    void nodeBreakerPropagatedContingencyBatchMatchesCpu() {
        // #33: with contingency PROPAGATION on (production default) a node-breaker line outage also
        // opens the target's breakers — switches are zero-impedance LfBranches, so the naive shape is
        // "3 branches" and nothing batched at all on a node-breaker network. The switch part is now a
        // per-scenario zero-impedance-element outage: the dead stub it leaves behind is frozen by the existing
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
        assertGpuMatchesCpu(network, List.of(new Contingency("L1", new BranchContingency("L1"))), 1e-12, true);
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
                        new com.powsybl.contingency.LoadContingency(loadId))), 1e-12, true);
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
}
