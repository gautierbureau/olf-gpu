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

import java.util.List;

/**
 * The networks, contingency lists and load flow parameters the GPU-batched security analysis tests build.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuSaTestNetworks {

    private GpuSaTestNetworks() {
    }

    static Network distributedSlackNetworkWithExtras() {
        Network network = com.powsybl.openloadflow.network.DistributedSlackNetworkFactory.create();
        // Second generator on b2 (same-bus multi-gen loss) — participating like g2.
        com.powsybl.iidm.network.Generator g2b = network.getVoltageLevel(network.getGenerator("g2").getTerminal().getVoltageLevel().getId())
                .newGenerator().setId("g2b").setBus("b2").setConnectableBus("b2")
                .setMinP(0).setMaxP(100).setTargetP(30).setTargetQ(50).setVoltageRegulatorOn(false).add();
        g2b.newExtension(com.powsybl.iidm.network.extensions.ActivePowerControlAdder.class)
                .withParticipate(true).withDroop(2).add();
        com.powsybl.iidm.network.Generator g2c = network.getVoltageLevel(network.getGenerator("g2").getTerminal().getVoltageLevel().getId())
                .newGenerator().setId("g2c").setBus("b2").setConnectableBus("b2")
                .setMinP(0).setMaxP(50).setTargetP(20).setTargetQ(30).setVoltageRegulatorOn(false).add();
        g2c.newExtension(com.powsybl.iidm.network.extensions.ActivePowerControlAdder.class)
                .withParticipate(true).withDroop(2).add();
        // Parallel circuit to l24 so a branch outage keeps the star grid connected (batch validator).
        com.powsybl.iidm.network.Line l24 = network.getLine("l24");
        network.newLine().setId("l24_bis")
                .setVoltageLevel1(l24.getTerminal1().getVoltageLevel().getId()).setBus1(l24.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l24.getTerminal2().getVoltageLevel().getId()).setBus2(l24.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l24.getR() * 2).setX(l24.getX() * 2).setG1(0).setB1(0).setG2(0).setB2(0).add();
        return network;
    }

    static Network remoteCurrentLimiterNetwork() {
        Network network = com.powsybl.openloadflow.network.PhaseControlFactory.createNetworkWithT2wt();
        com.powsybl.iidm.network.TwoWindingsTransformer ps1 = network.getTwoWindingsTransformer("PS1");
        ps1.getPhaseTapChanger().setRegulationMode(com.powsybl.iidm.network.PhaseTapChanger.RegulationMode.CURRENT_LIMITER)
                .setTargetDeadband(1).setRegulating(true).setTapPosition(1)
                .setRegulationTerminal(network.getLine("L2").getTerminal1()).setRegulationValue(100);
        com.powsybl.iidm.network.Line l1 = network.getLine("L1");
        network.newLine().setId("L1_BIS")
                .setVoltageLevel1(l1.getTerminal1().getVoltageLevel().getId()).setBus1(l1.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l1.getTerminal2().getVoltageLevel().getId()).setBus2(l1.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l1.getR() * 2).setX(l1.getX() * 2)
                .setG1(l1.getG1()).setB1(l1.getB1()).setG2(l1.getG2()).setB2(l1.getB2()).add();
        return network;
    }

    static List<Contingency> remoteCurrentLimiterContingencies() {
        return List.of(new Contingency("L1_BIS", new BranchContingency("L1_BIS")),
                new Contingency("L1", new BranchContingency("L1")));
    }

    static com.powsybl.loadflow.LoadFlowParameters remoteCurrentLimiterParameters() {
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setPhaseShifterRegulationOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setPhaseShifterControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.PhaseShifterControlMode.CONTINUOUS_WITH_DISCRETISATION);
        return lfp;
    }

    static Network twoMemberPilotIsControllerNetwork() {
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory
                .createWithGeneratorRemoteControl();
        network.newLine().setId("L14")
                .setVoltageLevel1("vl1").setBus1("b1").setVoltageLevel2("vl4").setBus2("b4")
                .setR(1.5).setX(40).setG1(0).setB1(0).setG2(0).setB2(0).add();
        // g1 and g2 regulate b1: the pilot is one of its own TWO controllers. g3 holds its own bus, so
        // it is a separate (local) control and leaves the remote group at exactly two members.
        for (String id : List.of("g1", "g2")) {
            network.getGenerator(id)
                    .setRegulatingTerminal(network.getGenerator("g1").getTerminal())
                    .setTargetV(21.0).setMaxP(500);
        }
        network.getGenerator("g3")
                .setRegulatingTerminal(network.getGenerator("g3").getTerminal())
                .setTargetV(20.5).setMaxP(500);
        // Reactive load on BOTH member buses, so the base and renormalized DISTR_Q targets differ
        // whichever of the two is lost.
        network.getVoltageLevel("vl1").newLoad()
                .setId("l1q").setBus("b1").setConnectableBus("b1").setP0(10).setQ0(60).add();
        network.getVoltageLevel("vl2").newLoad()
                .setId("l2q").setBus("b2").setConnectableBus("b2").setP0(10).setQ0(40).add();
        return network;
    }

    static Network switchedToPqInBaseDeadBusNetwork() {
        // A 400 kV mesh: gs at vl0 is the slack; g1 at vlb (250 MW, Q range [0, 20] MVar) regulates vlb locally at
        // 399.646 kV, under the 399.797 kV the bus reaches at Q = 0, so the base pins it at its MIN_Q; with g1 lost
        // vlb falls to 399.496 kV, under the target, which is what makes OLF switch the dead bus back to PV.
        Network network = Network.create("base-pinned-dead-bus", "code");
        com.powsybl.iidm.network.Substation s = network.newSubstation().setId("s").add();
        for (String vl : List.of("vl0", "vlb", "vl2", "vld")) {
            com.powsybl.iidm.network.VoltageLevel v = s.newVoltageLevel().setId(vl).setNominalV(400)
                    .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
            v.getBusBreakerView().newBus().setId(vl + "_b").add();
        }
        network.getVoltageLevel("vl0").newGenerator().setId("gs").setBus("vl0_b").setConnectableBus("vl0_b")
                .setMinP(0).setMaxP(2000).setTargetP(300).setTargetV(400).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl0").newLoad().setId("ld0").setBus("vl0_b").setConnectableBus("vl0_b").setP0(200).setQ0(40).add();
        network.getVoltageLevel("vlb").newGenerator().setId("g1").setBus("vlb_b").setConnectableBus("vlb_b")
                .setMinP(0).setMaxP(500).setTargetP(150).setTargetV(399.646).setVoltageRegulatorOn(true).add();
        network.getGenerator("g1").newMinMaxReactiveLimits().setMinQ(0).setMaxQ(20).add();
        network.getVoltageLevel("vlb").newLoad().setId("ldb").setBus("vlb_b").setConnectableBus("vlb_b").setP0(50).setQ0(10).add();
        network.getVoltageLevel("vl2").newGenerator().setId("g2").setBus("vl2_b").setConnectableBus("vl2_b")
                .setMinP(0).setMaxP(500).setTargetP(100).setTargetV(401).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vld").newLoad().setId("ldd").setBus("vld_b").setConnectableBus("vld_b").setP0(300).setQ0(60).add();
        java.util.function.BiConsumer<String, String[]> line = (id, ends) -> network.newLine().setId(id)
                .setVoltageLevel1(ends[0]).setBus1(ends[0] + "_b").setConnectableBus1(ends[0] + "_b")
                .setVoltageLevel2(ends[1]).setBus2(ends[1] + "_b").setConnectableBus2(ends[1] + "_b")
                .setR(2).setX(12).setG1(0).setB1(0).setG2(0).setB2(0).add();
        line.accept("l0b", new String[] {"vl0", "vlb"});
        line.accept("lb2", new String[] {"vlb", "vl2"});
        line.accept("l02", new String[] {"vl0", "vl2"});
        line.accept("lbd", new String[] {"vlb", "vld"});
        line.accept("l0d", new String[] {"vl0", "vld"});
        return network;
    }

    static Network pilotIsControllerSingleSurvivorNetwork() {
        // A 400 kV mesh: the pilot bus vlp carries its own generator g1 (regulating vlp) and reactive load; g2 at
        // vlc regulates vlp REMOTELY, capped at 15 MVar; gs at vl0 is the slack. The reactive keys follow the
        // ranges, so g1 (+-300) carries the base need and g2 stays inside its ceiling; with g1 lost, g2 alone
        // cannot hold vlp and pins.
        Network network = Network.create("pilot-self-single-survivor", "code");
        com.powsybl.iidm.network.Substation s = network.newSubstation().setId("s").add();
        for (String vl : List.of("vl0", "vlp", "vlc", "vld")) {
            com.powsybl.iidm.network.VoltageLevel v = s.newVoltageLevel().setId(vl).setNominalV(400)
                    .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
            v.getBusBreakerView().newBus().setId(vl + "_b").add();
        }
        network.getVoltageLevel("vl0").newGenerator().setId("gs").setBus("vl0_b").setConnectableBus("vl0_b")
                .setMinP(0).setMaxP(2000).setTargetP(400).setTargetV(400).setVoltageRegulatorOn(true).add();
        // a lost PARTICIPATING generator with a participating slack is a CPU shape (PARTICIPATING_SLACK):
        // keep the slack out of the distribution so the loss pre-distributes over g2 on the device
        network.getGenerator("gs").newExtension(com.powsybl.iidm.network.extensions.ActivePowerControlAdder.class)
                .withParticipate(false).add();
        network.getVoltageLevel("vl0").newLoad().setId("ld0").setBus("vl0_b").setConnectableBus("vl0_b").setP0(300).setQ0(60).add();
        network.getVoltageLevel("vlp").newGenerator().setId("g1").setBus("vlp_b").setConnectableBus("vlp_b")
                .setMinP(0).setMaxP(500).setTargetP(100).setTargetV(401).setVoltageRegulatorOn(true).add();
        network.getGenerator("g1").newMinMaxReactiveLimits().setMinQ(-300).setMaxQ(300).add();
        network.getVoltageLevel("vlp").newLoad().setId("ldp").setBus("vlp_b").setConnectableBus("vlp_b").setP0(80).setQ0(40).add();
        network.getVoltageLevel("vlc").newGenerator().setId("g2").setBus("vlc_b").setConnectableBus("vlc_b")
                .setMinP(0).setMaxP(500).setTargetP(100).setTargetV(401).setVoltageRegulatorOn(true)
                .setRegulatingTerminal(network.getGenerator("g1").getTerminal()).add();
        network.getGenerator("g2").newMinMaxReactiveLimits().setMinQ(-15).setMaxQ(15).add();
        network.getVoltageLevel("vld").newLoad().setId("ldd").setBus("vld_b").setConnectableBus("vld_b").setP0(200).setQ0(50).add();
        java.util.function.BiConsumer<String, String[]> line = (id, ends) -> network.newLine().setId(id)
                .setVoltageLevel1(ends[0]).setBus1(ends[0] + "_b").setConnectableBus1(ends[0] + "_b")
                .setVoltageLevel2(ends[1]).setBus2(ends[1] + "_b").setConnectableBus2(ends[1] + "_b")
                .setR(0.5).setX(12).setG1(0).setB1(0).setG2(0).setB2(0).add();
        line.accept("l0p", new String[] {"vl0", "vlp"});
        line.accept("lpc", new String[] {"vlp", "vlc"});
        line.accept("l0c", new String[] {"vl0", "vlc"});
        line.accept("lpd", new String[] {"vlp", "vld"});
        line.accept("l0d", new String[] {"vl0", "vld"});
        return network;
    }

    static Network pilotIsControllerNetwork() {
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory
                .createWithGeneratorRemoteControl();
        network.newLine().setId("L14")
                .setVoltageLevel1("vl1").setBus1("b1").setVoltageLevel2("vl4").setBus2("b4")
                .setR(1.5).setX(40).setG1(0).setB1(0).setG2(0).setB2(0).add();
        // Everyone regulates b1: the pilot becomes one of its own controllers.
        for (String id : List.of("g1", "g2", "g3")) {
            network.getGenerator(id)
                    .setRegulatingTerminal(network.getGenerator("g1").getTerminal())
                    .setTargetV(21.0)
                    .setMaxP(500);                           // headroom: a saturated slack distribution
        }                                                    // is a CPU route by design and would prove nothing
        network.getVoltageLevel("vl1").newLoad()
                .setId("l1q").setBus("b1").setConnectableBus("b1")
                .setP0(10).setQ0(60).add();                  // the pilot carries reactive load
        return network;
    }

    /** vl12 = two busbar sections joined by a retained coupler breaker. bbs1 carries g1 and a load and
     *  hangs off the coupler only; g1 (local) and g2 on bbs2 (remote, through the coupler) BOTH
     *  regulate bbs1, so losing bbs1 kills the whole control and leaves g2 alone at its fixed
     *  targetQ - rte6515's pilot-section loss shape: the coupler is a zero-impedance outage, the
     *  generator and the load are injection losses, no full branch outage (a line on the lost section
     *  would become one and route the plan to the branch-combo path instead). Equipment switches are
     *  disconnectors so the trip propagates through them (a breaker would stop it). */
    /**
     * The dominant NON-BATCHED shape of rte6515's full busbar scope (tasks #69, 31 of 46 refused re-merges):
     * a merged voltage-control group whose PILOT SECTION is lost while TWO coupler-joined sections survive,
     * each with its own LOCAL generator. OLF re-runs the merge on the surviving zero-impedance network,
     * elects a new main (highest target, then controlled-bus id), rebuilds DISTR_Q over the survivors with
     * keys renormalized over them, and puts BUS_TARGET_V on the new main. The device used to refuse the
     * whole contingency as PLAN:VC_REMERGE_MULTI ("a cluster of several controls is a NEW remote group the
     * tables cannot host"). It CAN host it: the lost pilot is an inactive member (its key leaves the
     * renormalization), the killed coupler's child gets the voltage identity the stock placement already
     * installs, and the surviving sharing rows are plain DISTR_Q rows.
     */
    static Network lostPilotSectionNetwork() {
        return lostPilotSectionNetwork(true);
    }

    /** A chain of {@code sections} busbar sections, every section with its OWN local control (g0 at the end
     *  section bbs0 carries the highest target, so it is the base MAIN), each section tied to the meshed external
     *  network by a line. Listing the loss of EVERY section keeps every coupler retained (OLF retains only the
     *  switches a listed contingency opens), so a middle-section loss leaves the main's cluster on one side and a
     *  second cluster on the other — tasks #71's shape. */
    static Network splitGroupNetwork(int sections) {
        return splitGroupNetwork(sections, java.util.Set.of());
    }

    /** @param pinnedGens sections whose generator gets a 5 MVar reactive ceiling, so the BASE load flow pins it. */
    static Network splitGroupNetwork(int sections, java.util.Set<Integer> pinnedGens) {
        return splitGroupNetwork(sections, pinnedGens, java.util.Set.of());
    }

    /** @param noControl sections whose generator does NOT regulate voltage (a plain PQ section in the chain). */
    static Network splitGroupNetwork(int sections, java.util.Set<Integer> pinnedGens, java.util.Set<Integer> noControl) {
        return splitGroupNetwork(sections, pinnedGens, noControl, java.util.Set.of());
    }

    /** @param noLine sections without their line to the external network: joined to the rest by couplers only. */
    static Network splitGroupNetwork(int sections, java.util.Set<Integer> pinnedGens, java.util.Set<Integer> noControl,
                                             java.util.Set<Integer> noLine) {
        Network network = Network.create("split-group-" + sections, "code");
        com.powsybl.iidm.network.Substation s = network.newSubstation().setId("s").add();
        for (String vl : List.of("vl0", "vl3", "vl4")) {
            com.powsybl.iidm.network.VoltageLevel v = s.newVoltageLevel().setId(vl).setNominalV(400)
                    .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
            v.getBusBreakerView().newBus().setId(vl + "_b").add();
        }
        com.powsybl.iidm.network.VoltageLevel vl12 = s.newVoltageLevel().setId("vl12").setNominalV(400)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.NODE_BREAKER).add();
        for (int k = 0; k < sections; k++) {
            vl12.getNodeBreakerView().newBusbarSection().setId("bbs" + k).setNode(k).add();
            if (k > 0) {
                vl12.getNodeBreakerView().newBreaker().setId("c" + (k - 1) + k).setNode1(k - 1).setNode2(k)
                        .setRetained(true).setOpen(false).add();
            }
        }
        int node = 100;
        String[] externals = {"vl0", "vl3", "vl4"};
        for (int k = 0; k < sections; k++) {
            int gNode = node++;
            int lNode = node++;
            for (int[] pair : noLine.contains(k) ? new int[][] {{gNode, 0}} : new int[][] {{gNode, 0}, {lNode, 1}}) {
                vl12.getNodeBreakerView().newDisconnector().setId("d" + k + "_" + pair[1]).setNode1(k).setNode2(pair[0] + 50).setOpen(false).add();
                vl12.getNodeBreakerView().newBreaker().setId("b" + k + "_" + pair[1]).setNode1(pair[0] + 50).setNode2(pair[0]).setOpen(false).add();
            }
            vl12.newGenerator().setId("g" + k).setNode(gNode)
                    .setMinP(0).setMaxP(500).setTargetP(80).setTargetQ(0).setTargetV(k == 0 ? 406 : 405)
                    .setVoltageRegulatorOn(!noControl.contains(k)).add();
            network.getGenerator("g" + k).newMinMaxReactiveLimits().setMinQ(-300).setMaxQ(pinnedGens.contains(k) ? 5 : 300).add();
            if (noLine.contains(k)) {
                continue;
            }
            String ext = externals[k % 3];
            network.newLine().setId("l" + k).setVoltageLevel1("vl12").setNode1(lNode)
                    .setVoltageLevel2(ext).setBus2(ext + "_b").setConnectableBus2(ext + "_b")
                    .setR(0.4).setX(10).setG1(0).setB1(0).setG2(0).setB2(0).add();
        }
        vl12.newLoad().setId("ld0").setNode(node).setP0(30).setQ0(40).add();
        vl12.getNodeBreakerView().newDisconnector().setId("dld").setNode1(0).setNode2(node + 50).setOpen(false).add();
        vl12.getNodeBreakerView().newBreaker().setId("bld").setNode1(node + 50).setNode2(node).setOpen(false).add();
        network.getVoltageLevel("vl0").newGenerator().setId("gs").setBus("vl0_b").setConnectableBus("vl0_b")
                .setMinP(0).setMaxP(2000).setTargetP(500).setTargetV(400).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl0").newLoad().setId("lds").setBus("vl0_b").setConnectableBus("vl0_b").setP0(500).setQ0(100).add();
        network.getVoltageLevel("vl3").newLoad().setId("ld3").setBus("vl3_b").setConnectableBus("vl3_b").setP0(150).setQ0(50).add();
        network.getVoltageLevel("vl4").newLoad().setId("ld4").setBus("vl4_b").setConnectableBus("vl4_b").setP0(150).setQ0(50).add();
        network.newLine().setId("l34").setVoltageLevel1("vl3").setBus1("vl3_b").setConnectableBus1("vl3_b")
                .setVoltageLevel2("vl4").setBus2("vl4_b").setConnectableBus2("vl4_b").setR(0.4).setX(10).setG1(0).setB1(0).setG2(0).setB2(0).add();
        network.newLine().setId("l04").setVoltageLevel1("vl0").setBus1("vl0_b").setConnectableBus1("vl0_b")
                .setVoltageLevel2("vl4").setBus2("vl4_b").setConnectableBus2("vl4_b").setR(0.4).setX(10).setG1(0).setB1(0).setG2(0).setB2(0).add();
        return network;
    }

    /** A chain of {@code couplers} zero-impedance couplers with a voltage-controlling generator on every bus (OLF
     *  merges their controls), the shape whose unpivoted factorization needs iterative refinement; every third chain
     *  bus is also tied to the load bus b0 by an impedant line, so each tie's loss is a different scenario of the
     *  same structure. */
    static Network couplerChainSaNetwork(int couplers) {
        Network network = Network.create("zi-chain-sa-" + couplers, "code");
        com.powsybl.iidm.network.Substation s = network.newSubstation().setId("s").add();
        java.util.function.Function<String, com.powsybl.iidm.network.Bus> bus = id -> {
            com.powsybl.iidm.network.VoltageLevel vl = s.newVoltageLevel().setId("vl_" + id).setNominalV(400)
                    .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
            return vl.getBusBreakerView().newBus().setId(id).add();
        };
        java.util.function.BiFunction<com.powsybl.iidm.network.Bus, String, com.powsybl.iidm.network.Generator> gen = (b, id) ->
                b.getVoltageLevel().newGenerator().setId(id).setBus(b.getId()).setConnectableBus(b.getId())
                        .setMinP(0).setMaxP(500).setTargetP(20).setTargetV(404).setVoltageRegulatorOn(true).add();
        com.powsybl.iidm.network.Bus b0 = bus.apply("b0");
        b0.getVoltageLevel().newLoad().setId("ld0").setBus("b0").setConnectableBus("b0").setP0(20.0 * couplers).setQ0(10.0 * couplers).add();
        com.powsybl.iidm.network.Bus prev = bus.apply("b1");
        gen.apply(prev, "g1");
        java.util.function.BiConsumer<String[], double[]> line = (ids, rx) -> network.newLine().setId(ids[0])
                .setVoltageLevel1("vl_" + ids[1]).setBus1(ids[1]).setConnectableBus1(ids[1])
                .setVoltageLevel2("vl_" + ids[2]).setBus2(ids[2]).setConnectableBus2(ids[2])
                .setR(rx[0]).setX(rx[1]).setG1(0).setB1(0).setG2(0).setB2(0).add();
        line.accept(new String[] {"t1", "b0", "b1"}, new double[] {0.4, 10});
        for (int i = 2; i <= couplers + 1; i++) {
            com.powsybl.iidm.network.Bus b = bus.apply("b" + i);
            gen.apply(b, "g" + i);
            line.accept(new String[] {"c" + i, prev.getId(), b.getId()}, new double[] {0, 0});
            if (i % 4 == 1) {
                line.accept(new String[] {"t" + i, "b0", b.getId()}, new double[] {0.4, 10 + i});
            }
            prev = b;
        }
        return network;
    }

    /** {@code sections} busbar sections joined by retained couplers, each REMOTELY regulated by its own generator
     *  behind a line in its own voltage level (OLF merges the controls across the couplers; bbs0 has the highest
     *  target, so its control is the MAIN). Losing bbs0 islands the main's pilot together with its generator, and
     *  OLF re-merges the survivors: they regulate their OWN sections again (network.xiidm busbar GOOD, CKTZ,
     *  CXUJ, FUDD - generators behind transformers regulating the sections). With the device's "control dies"
     *  reading the lone survivor vg1 came out 408.48 kV against OLF's 405.87 (lg1 -296.6 vs -120.9 MVAr). */
    static Network remoteMergedSectionsNetwork(int sections) {
        Network network = Network.create("remote-merged-" + sections, "code");
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
        for (int k = 0; k < sections; k++) {
            vl12.getNodeBreakerView().newBusbarSection().setId("bbs" + k).setNode(k).add();
            if (k > 0) {
                vl12.getNodeBreakerView().newBreaker().setId("c" + (k - 1) + k).setNode1(k - 1).setNode2(k)
                        .setRetained(true).setOpen(false).add();
            }
        }
        int node = 100;
        for (int k = 0; k < sections; k++) {
            com.powsybl.iidm.network.VoltageLevel vg = bbVl.apply("vg" + k);
            vg.newGenerator().setId("g" + k).setBus("vg" + k + "_b").setConnectableBus("vg" + k + "_b")
                    .setMinP(0).setMaxP(500).setTargetP(60).setTargetQ(0).setTargetV(k == 0 ? 406 : 404)
                    .setVoltageRegulatorOn(true)
                    .setRegulatingTerminal(vl12.getNodeBreakerView().getBusbarSection("bbs" + k).getTerminal()).add();
            network.getGenerator("g" + k).newMinMaxReactiveLimits().setMinQ(-300).setMaxQ(300).add();
            String[] feeders = {"lg" + k, "l" + k};
            for (int f = 0; f < feeders.length; f++) {
                int n = node++;
                vl12.getNodeBreakerView().newDisconnector().setId("d" + feeders[f]).setNode1(k).setNode2(n + 50).setOpen(false).add();
                vl12.getNodeBreakerView().newBreaker().setId("b" + feeders[f]).setNode1(n + 50).setNode2(n).setOpen(false).add();
                String far = f == 0 ? "vg" + k : (k % 2 == 0 ? "vl0" : "vl3");
                network.newLine().setId(feeders[f]).setVoltageLevel1("vl12").setNode1(n)
                        .setVoltageLevel2(far).setBus2(far + "_b").setConnectableBus2(far + "_b")
                        .setR(0.4).setX(f == 0 ? 6 : 10).setG1(0).setB1(0).setG2(0).setB2(0).add();
            }
        }
        network.getVoltageLevel("vl0").newGenerator().setId("gs").setBus("vl0_b").setConnectableBus("vl0_b")
                .setMinP(0).setMaxP(2000).setTargetP(500).setTargetV(400).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl0").newLoad().setId("lds").setBus("vl0_b").setConnectableBus("vl0_b").setP0(500).setQ0(100).add();
        network.getVoltageLevel("vl3").newLoad().setId("ld3").setBus("vl3_b").setConnectableBus("vl3_b").setP0(60.0 * sections).setQ0(80).add();
        network.newLine().setId("l03").setVoltageLevel1("vl0").setBus1("vl0_b").setConnectableBus1("vl0_b")
                .setVoltageLevel2("vl3").setBus2("vl3_b").setConnectableBus2("vl3_b").setR(0.4).setX(10).setG1(0).setB1(0).setG2(0).setB2(0).add();
        return network;
    }

    /** Two coupler STARS over the same three controlled sections (network.xiidm VL HFT): hubs bbs0 and bbs1 carry no
     *  injection, each leaf bbs2..bbs4 has its own generator regulating it (equal targets, so OLF merges the three
     *  controls) and a line to the meshed network. OLF's spanning tree takes one star (plus the edge that reaches the
     *  other hub); losing that hub's section makes OLF re-elect the tree through the OTHER star - couplers out of the
     *  BASE tree start carrying the leaves' flows. */
    static Network doubleStarSectionsNetwork() {
        Network network = Network.create("double-star", "code");
        com.powsybl.iidm.network.Substation s = network.newSubstation().setId("s").add();
        for (String vl : List.of("vl0", "vl3")) {
            s.newVoltageLevel().setId(vl).setNominalV(400).setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER)
                    .add().getBusBreakerView().newBus().setId(vl + "_b").add();
        }
        com.powsybl.iidm.network.VoltageLevel vl12 = s.newVoltageLevel().setId("vl12").setNominalV(400)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.NODE_BREAKER).add();
        for (int k = 0; k < 5; k++) {
            vl12.getNodeBreakerView().newBusbarSection().setId("bbs" + k).setNode(k).add();
        }
        for (int hub = 0; hub < 2; hub++) {
            for (int leaf = 2; leaf < 5; leaf++) {
                vl12.getNodeBreakerView().newBreaker().setId("c" + hub + leaf).setNode1(hub).setNode2(leaf)
                        .setRetained(true).setOpen(false).add();
            }
        }
        int node = 100;
        for (int hub = 0; hub < 2; hub++) {                  // a small load on each hub (network.xiidm HFT_0: a shunt)
            int hNode = node++;
            vl12.getNodeBreakerView().newDisconnector().setId("dh" + hub).setNode1(hub).setNode2(hNode + 50).setOpen(false).add();
            vl12.getNodeBreakerView().newBreaker().setId("bh" + hub).setNode1(hNode + 50).setNode2(hNode).setOpen(false).add();
            vl12.newLoad().setId("lh" + hub).setNode(hNode).setP0(5).setQ0(2).add();
        }
        for (int k = 2; k < 5; k++) {
            int gNode = node++;
            int lNode = node++;
            for (int[] pair : new int[][] {{gNode, 0}, {lNode, 1}}) {
                vl12.getNodeBreakerView().newDisconnector().setId("d" + k + "_" + pair[1]).setNode1(k).setNode2(pair[0] + 50).setOpen(false).add();
                vl12.getNodeBreakerView().newBreaker().setId("b" + k + "_" + pair[1]).setNode1(pair[0] + 50).setNode2(pair[0]).setOpen(false).add();
            }
            vl12.newGenerator().setId("g" + k).setNode(gNode)
                    .setMinP(0).setMaxP(500).setTargetP(60 + 10 * k).setTargetQ(0).setTargetV(405).setVoltageRegulatorOn(true).add();
            // g4's ceiling binds under OLF's equal reactive shares: which generator pins depends on the SHARING,
            // the one thing a free circulating coupler flow leaves undetermined
            network.getGenerator("g" + k).newMinMaxReactiveLimits().setMinQ(-300).setMaxQ(k == 4 ? 25 : 300).add();
            String ext = k % 2 == 0 ? "vl0" : "vl3";
            network.newLine().setId("l" + k).setVoltageLevel1("vl12").setNode1(lNode)
                    .setVoltageLevel2(ext).setBus2(ext + "_b").setConnectableBus2(ext + "_b")
                    .setR(0.4).setX(8 + k).setG1(0).setB1(0).setG2(0).setB2(0).add();
        }
        network.getVoltageLevel("vl0").newGenerator().setId("gs").setBus("vl0_b").setConnectableBus("vl0_b")
                .setMinP(0).setMaxP(2000).setTargetP(400).setTargetV(400).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl0").newLoad().setId("lds").setBus("vl0_b").setConnectableBus("vl0_b").setP0(400).setQ0(100).add();
        network.getVoltageLevel("vl3").newLoad().setId("ld3").setBus("vl3_b").setConnectableBus("vl3_b").setP0(210).setQ0(160).add();
        network.newLine().setId("l03").setVoltageLevel1("vl0").setBus1("vl0_b").setConnectableBus1("vl0_b")
                .setVoltageLevel2("vl3").setBus2("vl3_b").setConnectableBus2("vl3_b").setR(0.4).setX(10).setG1(0).setB1(0).setG2(0).setB2(0).add();
        return network;
    }

    /** Three sections in a chain (bbs0-bbs1-bbs2, couplers c01 and c12), each with a generator regulating it (equal
     *  targets, so OLF merges the three controls under vl12_0) and a line to the meshed network. The coupler tree is
     *  rooted at vl12_0: losing bbs0 opens c01 and OLF re-elects vl12_1, the opened coupler's CHILD, as the main of the
     *  two sections left. The reactive keys (60/20/20) leave g1 free in the base (1.29 pu of its 1.6) and pinned once
     *  g0 is gone (1.97 pu), and g2's 2.2 pu ceiling binds too once it holds the sections alone. */
    static Network selfControlledSectionChainNetwork() {
        Network network = Network.create("self-controlled-section-chain", "code");
        com.powsybl.iidm.network.Substation s = network.newSubstation().setId("s").add();
        for (String vl : List.of("vl0", "vl3")) {
            s.newVoltageLevel().setId(vl).setNominalV(400).setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER)
                    .add().getBusBreakerView().newBus().setId(vl + "_b").add();
        }
        com.powsybl.iidm.network.VoltageLevel vl12 = s.newVoltageLevel().setId("vl12").setNominalV(400)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.NODE_BREAKER).add();
        for (int k = 0; k < 3; k++) {
            vl12.getNodeBreakerView().newBusbarSection().setId("bbs" + k).setNode(k).add();
            if (k > 0) {
                vl12.getNodeBreakerView().newBreaker().setId("c" + (k - 1) + k).setNode1(k - 1).setNode2(k)
                        .setRetained(true).setOpen(false).add();
            }
        }
        String[][] generators = {{"g0", "3000", "60"}, {"g1", "160", "20"}, {"g2", "220", "20"}};   // id, max MVAr, key
        int node = 100;
        for (int k = 0; k < 3; k++) {
            int gNode = node++;
            int lNode = node++;
            for (int[] pair : new int[][] {{gNode, 0}, {lNode, 1}}) {
                vl12.getNodeBreakerView().newDisconnector().setId("d" + k + "_" + pair[1]).setNode1(k).setNode2(pair[0] + 50).setOpen(false).add();
                vl12.getNodeBreakerView().newBreaker().setId("b" + k + "_" + pair[1]).setNode1(pair[0] + 50).setNode2(pair[0]).setOpen(false).add();
            }
            vl12.newGenerator().setId(generators[k][0]).setNode(gNode)
                    .setMinP(0).setMaxP(500).setTargetP(60).setTargetQ(0).setTargetV(405).setVoltageRegulatorOn(true).add();
            network.getGenerator(generators[k][0]).newMinMaxReactiveLimits().setMinQ(-300).setMaxQ(Double.parseDouble(generators[k][1])).add();
            network.getGenerator(generators[k][0]).newExtension(com.powsybl.iidm.network.extensions.CoordinatedReactiveControlAdder.class)
                    .withQPercent(Double.parseDouble(generators[k][2])).add();
            String ext = k % 2 == 0 ? "vl0" : "vl3";
            network.newLine().setId("l" + k).setVoltageLevel1("vl12").setNode1(lNode)
                    .setVoltageLevel2(ext).setBus2(ext + "_b").setConnectableBus2(ext + "_b")
                    .setR(0.4).setX(8 + k).setG1(0).setB1(0).setG2(0).setB2(0).add();
        }
        network.getVoltageLevel("vl0").newGenerator().setId("gs").setBus("vl0_b").setConnectableBus("vl0_b")
                .setMinP(0).setMaxP(2000).setTargetP(400).setTargetV(400).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl0").newLoad().setId("lds").setBus("vl0_b").setConnectableBus("vl0_b").setP0(400).setQ0(100).add();
        network.getVoltageLevel("vl3").newLoad().setId("ld3").setBus("vl3_b").setConnectableBus("vl3_b").setP0(210).setQ0(160).add();
        network.newLine().setId("l03").setVoltageLevel1("vl0").setBus1("vl0_b").setConnectableBus1("vl0_b")
                .setVoltageLevel2("vl3").setBus2("vl3_b").setConnectableBus2("vl3_b").setR(0.4).setX(10).setG1(0).setB1(0).setG2(0).setB2(0).add();
        return network;
    }

    /** @param pilotLine whether the pilot section bbs0 carries a line. WITH it bbs0 is the coupler tree's ROOT
     *  (CouplerTreeRelocation roots a tree at the first bus with an impedant branch) and the elected main vl12_1 is
     *  the killed coupler's CHILD — its identity comes from the stock placement. WITHOUT it bbs0 cannot root
     *  a tree, vl12_1 becomes the root and keeps its OWN row, so its identity is the explicit ROW_V_IDENTITY
     *  of remoteMainIdentityEntries — rte6515 BUS-2640_BBS_1's shape, where that row also carries the
     *  member's DISTR_Q seed. */
    static Network lostPilotSectionNetwork(boolean pilotLine) {
        return lostPilotSectionNetwork(pilotLine, false);
    }

    /** @param pilotPinned cap g0's reactive ceiling so the BASE load flow pins the pilot at MAX_Q before it is
     *  lost (rte6515 BUS-6194_BBS_1 / BUS-6003_BBS_1: the islanded pilot's limit type still read its base pin,
     *  which the pilot-self gate took as "its balance rides the carrier"). */
    static Network lostPilotSectionNetwork(boolean pilotLine, boolean pilotPinned) {
        return lostPilotSectionNetwork(pilotLine, pilotPinned, false);
    }

    /** @param remoteSurvivors g1/g2 regulate bbs0 remotely (one REMOTE group) instead of their own sections. */
    static Network lostPilotSectionNetwork(boolean pilotLine, boolean pilotPinned, boolean remoteSurvivors) {
        Network network = Network.create("lost-pilot-section", "code");
        com.powsybl.iidm.network.Substation s = network.newSubstation().setId("s").add();
        for (String vl : List.of("vl0", "vl3", "vl4")) {
            com.powsybl.iidm.network.VoltageLevel v = s.newVoltageLevel().setId(vl).setNominalV(400)
                    .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
            v.getBusBreakerView().newBus().setId(vl + "_b").add();
        }
        com.powsybl.iidm.network.VoltageLevel vl12 = s.newVoltageLevel().setId("vl12").setNominalV(400)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.NODE_BREAKER).add();
        vl12.getNodeBreakerView().newBusbarSection().setId("bbs0").setNode(0).add();
        vl12.getNodeBreakerView().newBusbarSection().setId("bbs1").setNode(1).add();
        vl12.getNodeBreakerView().newBusbarSection().setId("bbs2").setNode(2).add();
        // setRetained here is INERT for the security analysis: OLF re-derives the retained set from the
        // contingency list (only a switch some listed contingency opens stays a bus-breaker separator and
        // becomes an LfSwitch). The couplers exist on the LF network because the test lists a loss on each
        // side of them, see deviceLostPilotSectionReMergesSurvivingClusterMatchesCpu.
        vl12.getNodeBreakerView().newBreaker().setId("c01").setNode1(0).setNode2(1).setRetained(true).setOpen(false).add();
        vl12.getNodeBreakerView().newBreaker().setId("c12").setNode1(1).setNode2(2).setRetained(true).setOpen(false).add();
        int node = 10;
        for (String[] eq : new String[][] {{"g0", "0"}, {"ld0", "0"}, {"l0", "0"}, {"g1", "1"}, {"l1", "1"}, {"g2", "2"}, {"l2", "2"}}) {
            vl12.getNodeBreakerView().newDisconnector().setId("d_" + eq[0]).setNode1(Integer.parseInt(eq[1])).setNode2(node + 10).setOpen(false).add();
            vl12.getNodeBreakerView().newBreaker().setId("b_" + eq[0]).setNode1(node + 10).setNode2(node).setOpen(false).add();
            node++;
        }
        // every section regulates ITS OWN bus: three LOCAL controls that the retained couplers MERGE into one
        // group. bbs0 carries the highest target so it is the base MAIN; bbs1 outranks bbs2 by id at equal
        // targets, so OLF elects bbs1 once bbs0 is lost.
        vl12.newGenerator().setId("g0").setNode(10)
                .setMinP(0).setMaxP(500).setTargetP(100).setTargetQ(0).setTargetV(406).setVoltageRegulatorOn(true).add();
        vl12.newLoad().setId("ld0").setNode(11).setP0(30).setQ0(40).add();
        for (String[] g : new String[][] {{"g1", "13"}, {"g2", "15"}}) {
            // remoteSurvivors: g1/g2 regulate bbs0 REMOTELY (one remote group, pilot bbs0), so losing bbs0 kills
            // their control instead of re-merging it — two dead-control survivors on a coupler chain.
            var adder = vl12.newGenerator().setId(g[0]).setNode(Integer.parseInt(g[1]))
                    .setMinP(0).setMaxP(500).setTargetP(100).setTargetQ(0).setVoltageRegulatorOn(true)
                    .setTargetV(remoteSurvivors ? 406 : 405);
            if (remoteSurvivors) {
                adder.setRegulatingTerminal(vl12.getNodeBreakerView().getBusbarSection("bbs0").getTerminal());
            }
            adder.add();
        }
        for (String g : List.of("g0", "g1", "g2")) {
            // pilotPinned: g0 cannot cover its share of the base reactive need, so the BASE load flow pins it at
            // MAX_Q while g1/g2 keep holding 406 kV (OLF: BUS_TARGET_V stays on the main, g0 PQ at its limit).
            network.getGenerator(g).newMinMaxReactiveLimits().setMinQ(-300)
                    .setMaxQ(pilotPinned && "g0".equals(g) ? 5 : 300).add();
        }
        network.getVoltageLevel("vl0").newGenerator().setId("gs").setBus("vl0_b").setConnectableBus("vl0_b")
                .setMinP(0).setMaxP(2000).setTargetP(500).setTargetV(400).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl0").newLoad().setId("lds").setBus("vl0_b").setConnectableBus("vl0_b").setP0(500).setQ0(100).add();
        network.getVoltageLevel("vl3").newLoad().setId("ld3").setBus("vl3_b").setConnectableBus("vl3_b").setP0(150).setQ0(50).add();
        network.getVoltageLevel("vl4").newLoad().setId("ld4").setBus("vl4_b").setConnectableBus("vl4_b").setP0(150).setQ0(50).add();
        java.util.function.BiConsumer<String[], Integer> line = (ends, n) -> network.newLine().setId(ends[0])
                .setVoltageLevel1("vl12").setNode1(n)
                .setVoltageLevel2(ends[1]).setBus2(ends[1] + "_b").setConnectableBus2(ends[1] + "_b")
                .setR(0.4).setX(10).setG1(0).setB1(0).setG2(0).setB2(0).add();
        if (pilotLine) {
            line.accept(new String[] {"l0", "vl0"}, 12);
        }
        line.accept(new String[] {"l1", "vl3"}, 14);
        line.accept(new String[] {"l2", "vl4"}, 16);
        network.newLine().setId("l34").setVoltageLevel1("vl3").setBus1("vl3_b").setConnectableBus1("vl3_b")
                .setVoltageLevel2("vl4").setBus2("vl4_b").setConnectableBus2("vl4_b").setR(0.4).setX(10).setG1(0).setB1(0).setG2(0).setB2(0).add();
        network.newLine().setId("l04").setVoltageLevel1("vl0").setBus1("vl0_b").setConnectableBus1("vl0_b")
                .setVoltageLevel2("vl4").setBus2("vl4_b").setConnectableBus2("vl4_b").setR(0.4).setX(10).setG1(0).setB1(0).setG2(0).setB2(0).add();
        return network;
    }

    static Network mergedPilotLossNetwork() {
        Network network = Network.create("merged-pilot-loss", "code");
        com.powsybl.iidm.network.Substation s = network.newSubstation().setId("s").add();
        for (String vl : List.of("vl0", "vl3")) {
            com.powsybl.iidm.network.VoltageLevel v = s.newVoltageLevel().setId(vl).setNominalV(400)
                    .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
            v.getBusBreakerView().newBus().setId(vl + "_b").add();
        }
        com.powsybl.iidm.network.VoltageLevel vl12 = s.newVoltageLevel().setId("vl12").setNominalV(400)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.NODE_BREAKER).add();
        vl12.getNodeBreakerView().newBusbarSection().setId("bbs1").setNode(0).add();
        vl12.getNodeBreakerView().newBusbarSection().setId("bbs2").setNode(1).add();
        vl12.getNodeBreakerView().newBreaker().setId("c12").setNode1(0).setNode2(1).setRetained(true).setOpen(false).add();
        vl12.getNodeBreakerView().newDisconnector().setId("bg1").setNode1(0).setNode2(2).setOpen(false).add();
        vl12.getNodeBreakerView().newDisconnector().setId("bld1").setNode1(0).setNode2(7).setOpen(false).add();
        vl12.getNodeBreakerView().newDisconnector().setId("bg2").setNode1(1).setNode2(3).setOpen(false).add();
        vl12.getNodeBreakerView().newDisconnector().setId("bl20").setNode1(1).setNode2(5).setOpen(false).add();
        vl12.getNodeBreakerView().newDisconnector().setId("bl23").setNode1(1).setNode2(6).setOpen(false).add();
        vl12.newGenerator().setId("g1").setNode(2)
                .setMinP(0).setMaxP(500).setTargetP(150).setTargetQ(50).setTargetV(405).setVoltageRegulatorOn(true).add();
        vl12.newLoad().setId("ld1").setNode(7).setP0(40).setQ0(10).add();
        vl12.newGenerator().setId("g2").setNode(3)
                .setMinP(0).setMaxP(500).setTargetP(150).setTargetQ(50).setTargetV(405).setVoltageRegulatorOn(true)
                .setRegulatingTerminal(network.getBusbarSection("bbs1").getTerminal()).add();
        network.getGenerator("g1").newMinMaxReactiveLimits().setMinQ(-300).setMaxQ(300).add();
        network.getGenerator("g2").newMinMaxReactiveLimits().setMinQ(-300).setMaxQ(300).add();
        network.getVoltageLevel("vl0").newGenerator().setId("g0").setBus("vl0_b").setConnectableBus("vl0_b")
                .setMinP(0).setMaxP(2000).setTargetP(500).setTargetV(400).setVoltageRegulatorOn(true).add();
        // g0's reactive output goes from -44 MVar in the base to +150 MVar once bbs1 is gone (OLF's
        // RL probe): the ceiling below pins it post-contingency only, so the reactive-limits loop
        // runs a pass and RE-SOLVES after it. That pass is where the DISTR_Q target maintenance used
        // to put the base target back on the survivor's row; without a re-solve the stamp changes
        // nothing and the test proves nothing.
        network.getGenerator("g0").newMinMaxReactiveLimits().setMinQ(-100).setMaxQ(100).add();
        network.getVoltageLevel("vl0").newLoad().setId("ld0").setBus("vl0_b").setConnectableBus("vl0_b").setP0(600).setQ0(150).add();
        network.getVoltageLevel("vl3").newLoad().setId("ld3").setBus("vl3_b").setConnectableBus("vl3_b").setP0(200).setQ0(80).add();
        // a second PV bus that outlives the contingency: OLF never switches its LAST PV bus to PQ, and
        // without g3 the pin on g0 above would be refused (RL loop STABLE, no re-solve).
        network.getVoltageLevel("vl3").newGenerator().setId("g3").setBus("vl3_b").setConnectableBus("vl3_b")
                .setMinP(0).setMaxP(500).setTargetP(100).setTargetV(400).setVoltageRegulatorOn(true).add();
        network.getGenerator("g3").newMinMaxReactiveLimits().setMinQ(-1000).setMaxQ(1000).add();
        network.newLine().setId("l20").setVoltageLevel1("vl12").setNode1(5)
                .setVoltageLevel2("vl0").setBus2("vl0_b").setConnectableBus2("vl0_b").setR(0.4).setX(10).setG1(0).setB1(0).setG2(0).setB2(0).add();
        network.newLine().setId("l23").setVoltageLevel1("vl12").setNode1(6)
                .setVoltageLevel2("vl3").setBus2("vl3_b").setConnectableBus2("vl3_b").setR(0.4).setX(10).setG1(0).setB1(0).setG2(0).setB2(0).add();
        return network;
    }

    /** Three busbar sections in a chain (bbs0 - c01 - bbs1 - c12 - bbs2), each with its own line out;
     *  g0 on bbs0 regulates bbs0 locally, g1 and g2 regulate bbs0 REMOTELY, and bbs0 carries a
     *  reactive load. Losing the MIDDLE section leaves g0 and g2 sharing the pilot's Q through
     *  DISTR_Q while no longer zero-impedance-connected (rte6515 BUS-1359_BBS_2, 8.5e-2 kV): the
     *  survivors' keys renormalize 1/3 -> 1/2 AND the DISTR_Q target moves with them, because the
     *  pilot's load makes the target key-dependent (a group without load has target 0 either way and
     *  cannot see this). */
    static Network middleSectionLossNetwork(boolean shuntOnMiddle) {
        Network network = Network.create("middle-section-loss", "code");
        com.powsybl.iidm.network.Substation s = network.newSubstation().setId("s").add();
        for (String vl : List.of("vl0", "vl3", "vl4")) {
            com.powsybl.iidm.network.VoltageLevel v = s.newVoltageLevel().setId(vl).setNominalV(400)
                    .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
            v.getBusBreakerView().newBus().setId(vl + "_b").add();
        }
        com.powsybl.iidm.network.VoltageLevel vl12 = s.newVoltageLevel().setId("vl12").setNominalV(400)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.NODE_BREAKER).add();
        vl12.getNodeBreakerView().newBusbarSection().setId("bbs0").setNode(0).add();
        vl12.getNodeBreakerView().newBusbarSection().setId("bbs1").setNode(1).add();
        vl12.getNodeBreakerView().newBusbarSection().setId("bbs2").setNode(2).add();
        vl12.getNodeBreakerView().newBreaker().setId("c01").setNode1(0).setNode2(1).setRetained(true).setOpen(false).add();
        vl12.getNodeBreakerView().newBreaker().setId("c12").setNode1(1).setNode2(2).setRetained(true).setOpen(false).add();
        // Feeders as in a real substation: a disconnector to the section, then a BREAKER before the
        // equipment. OLF's trip stops at the breaker and disconnects the terminal behind it, so a
        // LINE on the lost section is opened on THAT side only (rte6515's sideOpens=1); a feeder with
        // the disconnector alone lets the trip cross the line and open both ends, which composes as a
        // full branch outage and routes the plan to the branch-combo path instead of this one.
        int node = 10;
        for (String[] eq : new String[][] {{"g0", "0"}, {"ldp", "0"}, {"l0", "0"}, {"g1", "1"}, {"l1", "1"}, {"g2", "2"}, {"l2", "2"}, {"sh1", "1"}}) {
            vl12.getNodeBreakerView().newDisconnector().setId("d_" + eq[0]).setNode1(Integer.parseInt(eq[1])).setNode2(node + 10).setOpen(false).add();
            vl12.getNodeBreakerView().newBreaker().setId("b_" + eq[0]).setNode1(node + 10).setNode2(node).setOpen(false).add();
            node++;
        }
        vl12.newGenerator().setId("g0").setNode(10)
                .setMinP(0).setMaxP(500).setTargetP(100).setTargetQ(0).setTargetV(405).setVoltageRegulatorOn(true).add();
        vl12.newLoad().setId("ldp").setNode(11).setP0(30).setQ0(40).add();          // the pilot's reactive load
        vl12.newGenerator().setId("g1").setNode(13)
                .setMinP(0).setMaxP(500).setTargetP(100).setTargetQ(0).setTargetV(405).setVoltageRegulatorOn(true)
                .setRegulatingTerminal(network.getBusbarSection("bbs0").getTerminal()).add();
        vl12.newGenerator().setId("g2").setNode(15)
                .setMinP(0).setMaxP(500).setTargetP(100).setTargetQ(0).setTargetV(405).setVoltageRegulatorOn(true)
                .setRegulatingTerminal(network.getBusbarSection("bbs0").getTerminal()).add();
        for (String g : List.of("g0", "g1", "g2")) {
            network.getGenerator(g).newMinMaxReactiveLimits().setMinQ(-300).setMaxQ(300).add();
        }
        if (shuntOnMiddle) {                                 // a fixed shunt on the section the contingency islands
            vl12.newShuntCompensator().setId("sh1").setNode(17).setSectionCount(1)
                    .newLinearModel().setBPerSection(30e-6).setMaximumSectionCount(1).add().add();
        }
        network.getVoltageLevel("vl0").newGenerator().setId("gs").setBus("vl0_b").setConnectableBus("vl0_b")
                .setMinP(0).setMaxP(2000).setTargetP(500).setTargetV(400).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl0").newLoad().setId("ld0").setBus("vl0_b").setConnectableBus("vl0_b").setP0(500).setQ0(100).add();
        network.getVoltageLevel("vl3").newLoad().setId("ld3").setBus("vl3_b").setConnectableBus("vl3_b").setP0(150).setQ0(50).add();
        network.getVoltageLevel("vl4").newLoad().setId("ld4").setBus("vl4_b").setConnectableBus("vl4_b").setP0(150).setQ0(50).add();
        java.util.function.BiConsumer<String[], Integer> line = (ends, n) -> network.newLine().setId(ends[0])
                .setVoltageLevel1("vl12").setNode1(n)
                .setVoltageLevel2(ends[1]).setBus2(ends[1] + "_b").setConnectableBus2(ends[1] + "_b")
                .setR(0.4).setX(10).setG1(0).setB1(0).setG2(0).setB2(0).add();
        line.accept(new String[] {"l0", "vl0"}, 12);
        line.accept(new String[] {"l1", "vl3"}, 14);
        line.accept(new String[] {"l2", "vl4"}, 16);
        network.newLine().setId("l34").setVoltageLevel1("vl3").setBus1("vl3_b").setConnectableBus1("vl3_b")
                .setVoltageLevel2("vl4").setBus2("vl4_b").setConnectableBus2("vl4_b").setR(0.4).setX(10).setG1(0).setB1(0).setG2(0).setB2(0).add();
        network.newLine().setId("l04").setVoltageLevel1("vl0").setBus1("vl0_b").setConnectableBus1("vl0_b")
                .setVoltageLevel2("vl4").setBus2("vl4_b").setConnectableBus2("vl4_b").setR(0.4).setX(10).setG1(0).setB1(0).setG2(0).setB2(0).add();
        return network;
    }

    /** A remote gen-VC group whose controllers are ALL pinned at their reactive ceiling in the BASE
     *  state (rte6515 VL-6003_0 on the full busbar scope): gp on the pilot itself, c1 and c2 remotely
     *  regulate p at 412 kV with 20 MVar each against a 260 MVar load, so the base reactive-limits
     *  loop pins all three and the control is off before any contingency. The extractor used to ask for "the first enabled
     *  controller" to host the pilot's DISTR_Q, found none, and the whole precompute fell to the CPU
     *  (GPU_PRECOMPUTE_FAILED); the choice is structural now and the row modes say what is live. */
    static Network allPinnedRemoteGroupNetwork() {
        Network network = Network.create("all-pinned-remote-group", "code");
        com.powsybl.iidm.network.Substation s = network.newSubstation().setId("s").add();
        for (String vl : List.of("vl0", "vlp", "vlc1", "vlc2")) {
            com.powsybl.iidm.network.VoltageLevel v = s.newVoltageLevel().setId(vl).setNominalV(400)
                    .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
            v.getBusBreakerView().newBus().setId(vl + "_b").add();
        }
        network.getVoltageLevel("vl0").newGenerator().setId("g0").setBus("vl0_b").setConnectableBus("vl0_b")
                .setMinP(0).setMaxP(2000).setTargetP(600).setTargetV(400).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl0").newLoad().setId("ld0").setBus("vl0_b").setConnectableBus("vl0_b").setP0(300).setQ0(60).add();
        network.getVoltageLevel("vlp").newLoad().setId("ldp").setBus("vlp_b").setConnectableBus("vlp_b").setP0(400).setQ0(260).add();
        // the pilot is one of its own controllers (rte6515 VL-6003_0's shape: the throwing site)
        network.getVoltageLevel("vlp").newGenerator().setId("gp").setBus("vlp_b").setConnectableBus("vlp_b")
                .setMinP(0).setMaxP(500).setTargetP(100).setTargetQ(0).setTargetV(412).setVoltageRegulatorOn(true).add();
        network.getGenerator("gp").newMinMaxReactiveLimits().setMinQ(-20).setMaxQ(20).add();
        for (String c : List.of("c1", "c2")) {
            network.getVoltageLevel("vl" + c).newGenerator().setId("g" + c).setBus("vl" + c + "_b").setConnectableBus("vl" + c + "_b")
                    .setMinP(0).setMaxP(500).setTargetP(120).setTargetQ(0).setTargetV(412).setVoltageRegulatorOn(true)
                    .setRegulatingTerminal(network.getLoad("ldp").getTerminal()).add();
            network.getGenerator("g" + c).newMinMaxReactiveLimits().setMinQ(-20).setMaxQ(20).add();
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
        return network;
    }

    /** A remote voltage-control group whose members are ALL pinned at MAX_Q in the base (a radial 200 MVar
     *  load at vlq) and all UNPINNED by the contingency (losing that radial line) - see
     *  {@link #deviceRemoteGroupUnpinnedByContingencyRecomputesTargetQMatchesCpu()}. */
    static Network unpinnedRemoteGroupNetwork() {
        Network network = Network.create("unpinned-remote-group", "code");
        com.powsybl.iidm.network.Substation s = network.newSubstation().setId("s").add();
        for (String vl : List.of("vl0", "vlp", "vlq", "vlc1", "vlc2")) {
            com.powsybl.iidm.network.VoltageLevel v = s.newVoltageLevel().setId(vl).setNominalV(400)
                    .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
            v.getBusBreakerView().newBus().setId(vl + "_b").add();
        }
        network.getVoltageLevel("vl0").newGenerator().setId("g0").setBus("vl0_b").setConnectableBus("vl0_b")
                .setMinP(0).setMaxP(2000).setTargetP(300).setTargetV(413).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl0").newLoad().setId("ld0").setBus("vl0_b").setConnectableBus("vl0_b").setP0(100).setQ0(0).add();
        network.getVoltageLevel("vlp").newLoad().setId("ldp").setBus("vlp_b").setConnectableBus("vlp_b").setP0(400).setQ0(60).add();
        network.getVoltageLevel("vlq").newLoad().setId("ldq").setBus("vlq_b").setConnectableBus("vlq_b").setP0(0).setQ0(200).add();
        // ASYMMETRIC limits: the reactive keys come from the Q RANGES (90 : 60) while the pins sit at
        // maxQ (30 : 50) - with symmetric limits the two ratios coincide and so do both sharings
        for (String[] c : new String[][] {{"c1", "-60", "30"}, {"c2", "-10", "50"}}) {
            network.getVoltageLevel("vl" + c[0]).newGenerator().setId("g" + c[0]).setBus("vl" + c[0] + "_b")
                    .setConnectableBus("vl" + c[0] + "_b").setMinP(0).setMaxP(500).setTargetP(50).setTargetQ(0)
                    .setTargetV(412).setVoltageRegulatorOn(true).setRegulatingTerminal(network.getLoad("ldp").getTerminal()).add();
            network.getGenerator("g" + c[0]).newMinMaxReactiveLimits()
                    .setMinQ(Double.parseDouble(c[1])).setMaxQ(Double.parseDouble(c[2])).add();
        }
        java.util.function.BiConsumer<String, String[]> line = (id, ends) -> network.newLine().setId(id)
                .setVoltageLevel1(ends[0]).setBus1(ends[0] + "_b").setConnectableBus1(ends[0] + "_b")
                .setVoltageLevel2(ends[1]).setBus2(ends[1] + "_b").setConnectableBus2(ends[1] + "_b")
                .setR(0.4).setX(10).setG1(0).setB1(0).setG2(0).setB2(0).add();
        line.accept("l0p", new String[] {"vl0", "vlp"});
        line.accept("lpq", new String[] {"vlp", "vlq"});
        line.accept("lpc1", new String[] {"vlp", "vlc1"});
        line.accept("lpc2", new String[] {"vlp", "vlc2"});
        line.accept("l0c1", new String[] {"vl0", "vlc1"});
        line.accept("l0c2", new String[] {"vl0", "vlc2"});
        return network;
    }

    /** The remote-control group (b1/b2/b3 regulating pilot b4) hung off a meshed main component by the
     *  single tie LTIE, so outaging LTIE islands the entire group — pilot included. */
    static Network islandableRemoteControlGroupNetwork() {
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory
                .createWithGeneratorRemoteControl();
        com.powsybl.iidm.network.Substation sm = network.newSubstation().setId("sm").add();
        // FIVE main buses against the group's four: the validator requires the surviving side to be
        // the LARGEST connected component, not merely the one holding the slack.
        for (String vl : List.of("vlm1", "vlm2", "vlm3", "vlm4", "vlm5")) {
            com.powsybl.iidm.network.VoltageLevel v = sm.newVoltageLevel().setId(vl).setNominalV(400)
                    .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
            v.getBusBreakerView().newBus().setId(vl + "_b").add();
        }
        network.getVoltageLevel("vlm1").newGenerator().setId("gm").setBus("vlm1_b").setConnectableBus("vlm1_b")
                .setMinP(0).setMaxP(2000).setTargetP(700).setTargetV(400).setVoltageRegulatorOn(true).add();
        // A SURVIVING-side controller the outage drives over its reactive limit: the post-contingency
        // ReactiveLimits pass then owes a genuine re-solve, which is the solve the frozen pilot row's
        // permanent residual lengthens by an iteration. Without it the main component is already at
        // 1e-14 when the spurious pass runs and the extra step cannot move any bus.
        network.getVoltageLevel("vlm3").newGenerator().setId("gm3").setBus("vlm3_b").setConnectableBus("vlm3_b")
                .setMinP(0).setMaxP(500).setTargetP(200).setTargetV(400).setVoltageRegulatorOn(true).add();
        network.getGenerator("gm3").newMinMaxReactiveLimits().setMinQ(-15).setMaxQ(15).add();
        network.getVoltageLevel("vlm2").newLoad().setId("lm2").setBus("vlm2_b").setConnectableBus("vlm2_b")
                .setP0(300).setQ0(120).add();
        network.getVoltageLevel("vlm3").newLoad().setId("lm3").setBus("vlm3_b").setConnectableBus("vlm3_b")
                .setP0(250).setQ0(90).add();
        network.getVoltageLevel("vlm4").newLoad().setId("lm4").setBus("vlm4_b").setConnectableBus("vlm4_b")
                .setP0(180).setQ0(70).add();
        network.getVoltageLevel("vlm5").newLoad().setId("lm5").setBus("vlm5_b").setConnectableBus("vlm5_b")
                .setP0(140).setQ0(50).add();
        mainLine(network, "LM12", "vlm1", "vlm2");
        mainLine(network, "LM23", "vlm2", "vlm3");
        mainLine(network, "LM13", "vlm1", "vlm3");
        mainLine(network, "LM34", "vlm3", "vlm4");
        mainLine(network, "LM45", "vlm4", "vlm5");
        mainLine(network, "LM15", "vlm1", "vlm5");
        network.newLine().setId("LTIE")
                .setVoltageLevel1("vlm1").setBus1("vlm1_b").setConnectableBus1("vlm1_b")
                .setVoltageLevel2("vl4").setBus2("b4").setConnectableBus2("b4")
                .setR(0.5).setX(12).setG1(0).setB1(0).setG2(0).setB2(0).add();
        return network;
    }

    static void mainLine(Network n, String id, String vlA, String vlB) {
        n.newLine().setId(id)
                .setVoltageLevel1(vlA).setBus1(vlA + "_b").setConnectableBus1(vlA + "_b")
                .setVoltageLevel2(vlB).setBus2(vlB + "_b").setConnectableBus2(vlB + "_b")
                .setR(0.4).setX(10).setG1(0).setB1(0).setG2(0).setB2(0).add();
    }

    static void addPlainLoad(Network network, String vlId, String id, double p0, double q0) {
        com.powsybl.iidm.network.VoltageLevel vl = network.getVoltageLevel(vlId);
        String bus = vl.getBusBreakerView().getBusStream().findFirst().orElseThrow().getId();
        vl.newLoad().setId(id).setBus(bus).setConnectableBus(bus).setP0(p0).setQ0(q0).add();
    }

    static void addBoundaryLine(Network network, String vlId, String id, double p0, double q0) {
        com.powsybl.iidm.network.VoltageLevel vl = network.getVoltageLevel(vlId);
        String bus = vl.getBusBreakerView().getBusStream().findFirst().orElseThrow().getId();
        vl.newBoundaryLine().setId(id).setBus(bus).setConnectableBus(bus)
                .setP0(p0).setQ0(q0).setR(1.5).setX(12.0).setG(0).setB(0).add();
    }

    /** OLF's three-generator remote voltage control (g1/g2/g3 behind tr1/tr2/tr3 regulating b4), every generator
     *  limited to [-maxQ, maxQ]. Losing tr2 or tr3 islands one controller: its bus is disabled after the contingency. */
    static Network limitedRemoteControlGroupNetwork(double maxQ) {
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createWithGeneratorRemoteControl();
        for (String g : new String[] {"g1", "g2", "g3"}) {
            network.getGenerator(g).newMinMaxReactiveLimits().setMinQ(-maxQ).setMaxQ(maxQ).add();
        }
        return network;
    }

    /** Every PV-capable bus limited (the three remote members are the network's ONLY PV buses),
     *  plus an L14 outage strong enough to drive all of them over their limits. */
    static Network forcedPvNetwork() {
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createWithGeneratorRemoteControl();
        for (String g : new String[] {"g1", "g2", "g3"}) {
            network.getGenerator(g).newMinMaxReactiveLimits().setMinQ(-1000).setMaxQ(1000).add();
        }
        network.newLine().setId("L14")
                .setVoltageLevel1("vl1").setBus1("b1").setVoltageLevel2("vl4").setBus2("b4")
                .setR(1.5).setX(40).setG1(0).setB1(0).setG2(0).setB2(0).add();
        return network;
    }

    static List<Contingency> remoteFreezeContingencies() {
        return List.of(new Contingency("LINE_3R_BIS", new BranchContingency("LINE_3R_BIS")));
    }

    static com.powsybl.loadflow.LoadFlowParameters remoteFreezeParameters(double minNominalV) {
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setTransformerVoltageControlOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL)
                .setVoltageRemoteControl(true)
                .setGeneratorVoltageControlMinNominalVoltage(minNominalV);
        return lfp;
    }

    /** GEN_R sits on its own 33 kV bus and regulates BUS_S, a THIRD 33 kV bus down the feeder — so it is
     *  a REMOTE controller, and its controlled bus is under the transformer-controlled level (33 kV)
     *  without being the bus the transformers control, which would have taken their control away.
     *  Distinct Q limits give it a reactive-limit entry, which is what carries the frozen limit type. */
    static Network remoteFreezeNetwork() {
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWith2T2wt();
        var t1 = network.getTwoWindingsTransformer("T2wT1");
        var t2 = network.getTwoWindingsTransformer("T2wT2");
        t1.getRatioTapChanger().setTargetDeadband(0).setRegulating(true).setTapPosition(3)
                .setRegulationTerminal(t1.getTerminal2()).setTargetV(34.0);
        t2.getRatioTapChanger().setTargetDeadband(0).setRegulating(true).setTapPosition(0)
                .setRegulationTerminal(t2.getTerminal2()).setTargetV(34.0);
        var sr = network.newSubstation().setId("SUBSTATION_R").add();
        var vlr = sr.newVoltageLevel().setId("VL_R").setNominalV(33.0)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
        vlr.getBusBreakerView().newBus().setId("BUS_R").add();
        var vls = sr.newVoltageLevel().setId("VL_S").setNominalV(33.0)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
        vls.getBusBreakerView().newBus().setId("BUS_S").add();
        vls.newLoad().setId("LOAD_S").setBus("BUS_S").setConnectableBus("BUS_S").setP0(3.0).setQ0(1.0).add();
        var genR = vlr.newGenerator().setId("GEN_R").setBus("BUS_R").setConnectableBus("BUS_R")
                .setMinP(0).setMaxP(20).setTargetP(5).setTargetV(33.5).setVoltageRegulatorOn(true)
                .setRegulatingTerminal(network.getLoad("LOAD_S").getTerminal()).add();
        genR.newMinMaxReactiveLimits().setMinQ(-20).setMaxQ(20).add();
        network.newLine().setId("LINE_3R")
                .setVoltageLevel1("VL_3").setBus1("BUS_3").setConnectableBus1("BUS_3")
                .setVoltageLevel2("VL_R").setBus2("BUS_R").setConnectableBus2("BUS_R")
                .setR(0.5).setX(1.5).add();
        network.newLine().setId("LINE_RS")
                .setVoltageLevel1("VL_R").setBus1("BUS_R").setConnectableBus1("BUS_R")
                .setVoltageLevel2("VL_S").setBus2("BUS_S").setConnectableBus2("BUS_S")
                .setR(0.5).setX(1.5).add();
        network.newLine().setId("LINE_3R_BIS")               // the contingency, clear of the transformer group
                .setVoltageLevel1("VL_3").setBus1("BUS_3").setConnectableBus1("BUS_3")
                .setVoltageLevel2("VL_R").setBus2("BUS_R").setConnectableBus2("BUS_R")
                .setR(1.0).setX(3.0).add();
        return network;
    }

    static Network sharedThreeMemberTvcNetwork() {
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createWithTransformerSharedRemoteControl();
        com.powsybl.iidm.network.TwoWindingsTransformer t2 = network.getTwoWindingsTransformer("T2wT2");
        com.powsybl.iidm.network.TwoWindingsTransformer t3 = network.getSubstation("SUBSTATION").newTwoWindingsTransformer()
                .setId("T2wT3").setRatedU1(t2.getRatedU1()).setRatedU2(t2.getRatedU2())
                .setR(t2.getR() * 1.5).setX(t2.getX() * 1.5).setG(t2.getG()).setB(t2.getB())
                .setBus1(t2.getTerminal1().getBusBreakerView().getBus().getId())
                .setBus2(t2.getTerminal2().getBusBreakerView().getBus().getId()).add();
        var rtc2 = t2.getRatioTapChanger();
        var adder = t3.newRatioTapChanger();
        for (int p = rtc2.getLowTapPosition(); p <= rtc2.getHighTapPosition(); p++) {
            var step = rtc2.getStep(p);
            adder.beginStep().setRho(step.getRho()).setR(step.getR()).setX(step.getX())
                    .setG(step.getG()).setB(step.getB()).endStep();
        }
        adder.setLowTapPosition(rtc2.getLowTapPosition()).setTapPosition(0).setLoadTapChangingCapabilities(true).add();
        for (String id : new String[] {"T2wT", "T2wT2", "T2wT3"}) {
            com.powsybl.iidm.network.TwoWindingsTransformer t = network.getTwoWindingsTransformer(id);
            t.getRatioTapChanger().setTargetDeadband(0).setTapPosition(0)
                    .setRegulationMode(com.powsybl.iidm.network.RatioTapChanger.RegulationMode.VOLTAGE)
                    .setRegulationTerminal(t.getTerminal2()).setTargetV(33.0).setRegulating(true);
        }
        com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
        network.newLine().setId(l0.getId() + "_BIS")
                .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l0.getR() * 2).setX(l0.getX() * 2)
                .setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
        return network;
    }

    static com.powsybl.loadflow.LoadFlowParameters robustRealisticVParameters(boolean robustMode) {
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setVoltageRemoteControl(true)
                .setMinRealisticVoltage(0.8)
                .setMaxRealisticVoltage(1.2)
                .setVoltageRemoteControlRobustMode(robustMode);
        return lfp;
    }

    /** GeneratorRemoteControlPQSwitchTest's collapsing remote-control network, with the b2-b3 line
     *  doubled so a branch outage leaves the network connected (and reproduces the single-line
     *  impedance the CPU test collapses at). */
    static Network robustRealisticVNetwork() {
        Network network = com.powsybl.iidm.network.NetworkFactory.findDefault().createNetwork("robustRealisticV", "java");
        com.powsybl.iidm.network.Substation s12 = network.newSubstation().setId("s12").add();
        com.powsybl.iidm.network.VoltageLevel vl1 = s12.newVoltageLevel().setId("vl1").setNominalV(20)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
        com.powsybl.iidm.network.VoltageLevel vl2 = s12.newVoltageLevel().setId("vl2").setNominalV(400)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
        com.powsybl.iidm.network.VoltageLevel vl3 = network.newVoltageLevel().setId("vl3").setNominalV(400)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
        vl1.getBusBreakerView().newBus().setId("b1").add();
        vl2.getBusBreakerView().newBus().setId("b2").add();
        vl3.getBusBreakerView().newBus().setId("b3").add();
        for (String id : List.of("l23a", "l23b")) {
            network.newLine().setId(id).setR(0.2).setX(0.2)
                    .setBus1("b2").setConnectableBus1("b2")
                    .setBus2("b3").setConnectableBus2("b3").add();
        }
        com.powsybl.iidm.network.TwoWindingsTransformer t12a = s12.newTwoWindingsTransformer().setId("t12a")
                .setR(0.1).setX(15)
                .setBus1("b1").setConnectableBus1("b1")
                .setBus2("b2").setConnectableBus2("b2").add();
        com.powsybl.iidm.network.Generator g1 = vl1.newGenerator().setId("g1")
                .setMaxP(500).setMinP(0).setTargetP(200)
                .setBus("b1").setConnectableBus("b1")
                .setVoltageRegulatorOn(true).setRegulatingTerminal(t12a.getTerminal2())
                .setTargetV(401).setTargetQ(10).add();
        g1.newMinMaxReactiveLimits().setMinQ(-3000).setMaxQ(6000).add();
        vl3.newGenerator().setId("g3").setMaxP(10).setMinP(0).setTargetP(0)
                .setBus("b3").setConnectableBus("b3")
                .setVoltageRegulatorOn(true).setTargetV(402).add();
        vl3.newLoad().setId("l3").setBus("b3").setConnectableBus("b3").setP0(200).setQ0(0).add();
        return network;
    }

    /** robustRealisticVNetwork with two more REMOTE controllers of b2 on healthy 400 kV buses (g4 at b4, g5 at b5,
     *  reactive ranges +-3000 MVar against g1's [-3000, 6000], so the keys leave g1 43 % of the duty and its 20 kV
     *  bus still collapses on the l23b outage while g4 and g5 stay inside their ranges). OLF then freezes g1 at its INITIAL generation target Q (10 MVar,
     *  MIN_REALISTIC_V) while g4 and g5 stay a two-member sharing group whose DISTR_Q targets carry
     *  qPct x targetQ(g1) - the frozen 10 MVar, not g1's 6000 MVar ceiling. */
    static Network realisticFreezePeersNetwork() {
        Network network = robustRealisticVNetwork();
        // twice the transformer reactance: g1 now takes 43 % of the group's absorption instead of all of it, and its
        // 20 kV bus must still leave the realistic band on the outage
        network.getTwoWindingsTransformer("t12a").setX(30);
        com.powsybl.iidm.network.Terminal pilot = network.getTwoWindingsTransformer("t12a").getTerminal2();
        for (String k : List.of("4", "5")) {
            com.powsybl.iidm.network.VoltageLevel vl = network.newVoltageLevel().setId("vl" + k).setNominalV(400)
                    .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
            vl.getBusBreakerView().newBus().setId("b" + k).add();
            network.newLine().setId("l2" + k).setR(0.5).setX(5)
                    .setBus1("b2").setConnectableBus1("b2").setBus2("b" + k).setConnectableBus2("b" + k).add();
            vl.newGenerator().setId("g" + k).setMaxP(300).setMinP(0).setTargetP(50)
                    .setBus("b" + k).setConnectableBus("b" + k)
                    .setVoltageRegulatorOn(true).setRegulatingTerminal(pilot).setTargetV(401).setTargetQ(0).add()
                    .newMinMaxReactiveLimits().setMinQ(-3000).setMaxQ(3000).add();
        }
        return network;
    }

    static com.powsybl.loadflow.LoadFlowParameters remotePinParameters() {
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp).setVoltageRemoteControl(true);
        return lfp;
    }

    /** Remote gen-VC group (g1/g2/g3 behind tr1/tr2/tr3 regulating b4) + a Q-heavy load bus b5 on
     *  two parallel lines; the named generator gets the given maxQ (the pin knob), the others stay
     *  unlimited — the plan-B4 all-members-pinnable gate always passes and the batch engages. */
    static Network remotePinNetwork(String genId, double maxQ) {
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createWithGeneratorRemoteControl();
        var s = network.getSubstation("s");
        var vl5 = s.newVoltageLevel().setId("vl5").setNominalV(400)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
        vl5.getBusBreakerView().newBus().setId("b5").add();
        vl5.newLoad().setId("l5").setBus("b5").setConnectableBus("b5").setP0(150).setQ0(300).add();
        for (String id : new String[] {"l45", "l45b"}) {
            network.newLine().setId(id)
                    .setVoltageLevel1("vl4").setBus1("b4").setConnectableBus1("b4")
                    .setVoltageLevel2("vl5").setBus2("b5").setConnectableBus2("b5")
                    .setR(1.5).setX(40.0).setG1(0).setB1(0).setG2(0).setB2(0).add();
        }
        network.getGenerator(genId).newMinMaxReactiveLimits().setMinQ(-1000).setMaxQ(maxQ).add();
        return network;
    }

    /** ALL THREE members pin under the l45b outage and the group LOSES control (in-flight pilot
     *  vEq deactivation). g5, a DISTANT unlimited PV behind a 400-ohm line, keeps OLF's global
     *  "keep one PV bus" rule out of the way AND keeps the all-pinned system regular (with no
     *  voltage reference at all the CPU itself hits MAX_ITERATION — a nearer/stronger keeper
     *  instead holds the group's job, which is why the caps sit in the narrow 185-vs-183
     *  window). */
    static Network allPinnedCornerNetwork() {
        Network network = remotePinNetwork("g1", 185.0);
        network.getGenerator("g2").newMinMaxReactiveLimits().setMinQ(-1000).setMaxQ(185.0).add();
        network.getGenerator("g3").newMinMaxReactiveLimits().setMinQ(-1000).setMaxQ(185.0).add();
        var vl7 = network.getSubstation("s").newVoltageLevel().setId("vl7").setNominalV(400)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
        vl7.getBusBreakerView().newBus().setId("b7").add();
        vl7.newGenerator().setId("g5").setBus("b7").setConnectableBus("b7")
                .setMinP(0).setMaxP(10).setTargetP(1).setVoltageRegulatorOn(true).setTargetV(408).add();
        network.newLine().setId("l47")                       // DISTANT but real voltage reference:
                .setVoltageLevel1("vl4").setBus1("b4").setConnectableBus1("b4")   // keeps the PV
                .setVoltageLevel2("vl7").setBus2("b7").setConnectableBus2("b7")   // count > 0 and the
                .setR(10.0).setX(400.0).setG1(0).setB1(0).setG2(0).setB2(0).add();   // system regular
        network.getLine("l45").setX(20.0);                   // keep decent post-outage coupling: the
        // corner is CONTROL loss, not voltage collapse
        return network;
    }

    /** g2/g3 turned LOCAL (own-bus 1.05 pu) so g1 alone carries the b4 group; maxQ 800 is the
     *  swept knob (750: the BASE already loses control at 411.3 kV; 850: the outage no longer
     *  pins) — the singleton holds in the base and PINS under l45b. */
    static Network singletonRemotePinNetwork() {
        Network network = remotePinNetwork("g1", 800);
        for (String id : new String[] {"g2", "g3"}) {
            var g = network.getGenerator(id);
            g.setRegulatingTerminal(g.getTerminal())
                    .setTargetV(1.05 * g.getTerminal().getVoltageLevel().getNominalV());
        }
        return network;
    }

    /** Three busbar sections in a chain, bbs0 - c01 - bbs1 - c12 - bbs2, the AC-emulation converter cs1 on bbs2 and
     *  each end section fed by its own line. bbs0 roots the coupler tree, so bbs2 is c12's child and its P balance
     *  lives on c12's DUMMY_P row; losing bbs1 kills c12 and brings that balance home. The link's terms stayed on
     *  the dead coupler's identity row: once the freeze loop released the link, the converter bus carried no HVDC
     *  flow at all (network.xiidm busbar ADNO: 5.05 pu gone from one converter of two parallel links, 696 MW put
     *  on the distributed slack). */
    static Network converterOnChildSectionNetwork() {
        Network network = Network.create("converter-child-section", "code");
        com.powsybl.iidm.network.Substation s = network.newSubstation().setId("s").add();
        for (String b : List.of("a", "b", "h")) {
            s.newVoltageLevel().setId("vl_" + b).setNominalV(400).setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER)
                    .add().getBusBreakerView().newBus().setId(b).add();
        }
        com.powsybl.iidm.network.VoltageLevel vlc = s.newVoltageLevel().setId("vlc").setNominalV(400)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.NODE_BREAKER).add();
        for (int k = 0; k < 3; k++) {
            vlc.getNodeBreakerView().newBusbarSection().setId("bbs" + k).setNode(k).add();
        }
        vlc.getNodeBreakerView().newBreaker().setId("c01").setNode1(0).setNode2(1).setRetained(true).setOpen(false).add();
        vlc.getNodeBreakerView().newBreaker().setId("c12").setNode1(1).setNode2(2).setRetained(true).setOpen(false).add();
        int[][] feeders = {{10, 0}, {11, 1}, {12, 2}, {13, 2}};   // la on bbs0, ld1 on bbs1, lb and cs1 on bbs2
        for (int[] f : feeders) {
            vlc.getNodeBreakerView().newDisconnector().setId("d" + f[0]).setNode1(f[1]).setNode2(f[0] + 50).setOpen(false).add();
            vlc.getNodeBreakerView().newBreaker().setId("b" + f[0]).setNode1(f[0] + 50).setNode2(f[0]).setOpen(false).add();
        }
        network.newLine().setId("la").setVoltageLevel1("vlc").setNode1(10).setVoltageLevel2("vl_a").setBus2("a").setConnectableBus2("a")
                .setR(0.4).setX(10).setG1(0).setB1(0).setG2(0).setB2(0).add();
        vlc.newLoad().setId("ld1").setNode(11).setP0(20).setQ0(5).add();
        network.newLine().setId("lb").setVoltageLevel1("vlc").setNode1(12).setVoltageLevel2("vl_b").setBus2("b").setConnectableBus2("b")
                .setR(0.4).setX(12).setG1(0).setB1(0).setG2(0).setB2(0).add();
        java.util.function.BiConsumer<String[], double[]> line = (ids, rx) -> network.newLine().setId(ids[0])
                .setVoltageLevel1("vl_" + ids[1]).setBus1(ids[1]).setConnectableBus1(ids[1])
                .setVoltageLevel2("vl_" + ids[2]).setBus2(ids[2]).setConnectableBus2(ids[2])
                .setR(rx[0]).setX(rx[1]).setG1(0).setB1(0).setG2(0).setB2(0).add();
        line.accept(new String[] {"lab", "a", "b"}, new double[] {0.4, 10});
        line.accept(new String[] {"lah", "a", "h"}, new double[] {0.4, 15});
        line.accept(new String[] {"lbh", "b", "h"}, new double[] {0.4, 15});
        network.getVoltageLevel("vl_a").newGenerator().setId("ga").setBus("a").setConnectableBus("a")
                .setMinP(0).setMaxP(1000).setTargetP(200).setTargetV(400).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl_b").newGenerator().setId("gb").setBus("b").setConnectableBus("b")
                .setMinP(0).setMaxP(1000).setTargetP(100).setTargetV(402).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl_b").newLoad().setId("ldb").setBus("b").setConnectableBus("b").setP0(180).setQ0(40).add();
        network.getVoltageLevel("vl_h").newLoad().setId("ldh").setBus("h").setConnectableBus("h").setP0(100).setQ0(30).add();
        // PQ converters: the lost flow must show in the voltage magnitudes the comparison reads
        vlc.newVscConverterStation().setId("cs1").setNode(13).setVoltageRegulatorOn(false).setVoltageSetpoint(400)
                .setReactivePowerSetpoint(0).setLossFactor(1.1f).add();
        network.getVoltageLevel("vl_h").newVscConverterStation().setId("cs2").setBus("h").setConnectableBus("h")
                .setVoltageRegulatorOn(false).setVoltageSetpoint(400).setReactivePowerSetpoint(0).setLossFactor(1.1f).add();
        network.newHvdcLine().setId("hvdc").setConverterStationId1("cs1").setConverterStationId2("cs2")
                .setNominalV(400).setR(1).setActivePowerSetpoint(60)
                .setConvertersMode(com.powsybl.iidm.network.HvdcLine.ConvertersMode.SIDE_1_RECTIFIER_SIDE_2_INVERTER)
                .setMaxP(500).add()
                .newExtension(com.powsybl.iidm.network.extensions.HvdcAngleDroopActivePowerControlAdder.class)
                .withDroop(20).withP0(60).withEnabled(true).add();
        return network;
    }

    /** Three coupled sections (bbs0-bbs1-bbs2, coupler breakers c01 and c12) of one node-breaker level, every section
     *  PQ: a load behind its own breaker at the section node, and a line to the meshed external network behind a
     *  DISCONNECTOR to a hub node whose only other switch is the line's breaker. OLF's tripping of a section
     *  (NodeBreakerTraverser) then stops at that disconnector WITHOUT opening it (every other edge at the section node
     *  is a breaker), disconnects the load's terminal (an end node) and retains-and-opens the coupler breakers: the
     *  section's bus SURVIVES its own loss through its line, with the coupler(s) open and the load gone.
     *  network.xiidm's DBIF/DBIG (level DBIE, coupler DBIS, loads DBJC/DBJA behind the breakers DBIY/DBIM, lines
     *  behind the disconnectors DBII/DBIK) are exactly this. bbs2 also carries a feeder of the GPAX kind: a breaker
     *  to a hub node, then a disconnector to the line, so the breaker is retained and opened while the hub bus stays
     *  energized from the line's far end. */
    static Network liveSectionChainNetwork() {
        Network network = Network.create("live-section-chain", "code");
        com.powsybl.iidm.network.Substation s = network.newSubstation().setId("s").add();
        for (String vl : List.of("vl0", "vl3", "vl4")) {
            com.powsybl.iidm.network.VoltageLevel v = s.newVoltageLevel().setId(vl).setNominalV(400)
                    .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
            v.getBusBreakerView().newBus().setId(vl + "_b").add();
        }
        com.powsybl.iidm.network.VoltageLevel vl12 = s.newVoltageLevel().setId("vl12").setNominalV(400)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.NODE_BREAKER).add();
        String[] externals = {"vl0", "vl3", "vl4"};
        for (int k = 0; k < 3; k++) {
            vl12.getNodeBreakerView().newBusbarSection().setId("bbs" + k).setNode(k).add();
            if (k > 0) {
                vl12.getNodeBreakerView().newBreaker().setId("c" + (k - 1) + k).setNode1(k - 1).setNode2(k).setOpen(false).add();
            }
            int hub = 10 + k;
            int lineNode = 20 + k;
            int loadNode = 30 + k;
            vl12.getNodeBreakerView().newDisconnector().setId("d" + k).setNode1(k).setNode2(hub).setOpen(false).add();
            vl12.getNodeBreakerView().newBreaker().setId("bh" + k).setNode1(hub).setNode2(lineNode).setOpen(false).add();
            network.newLine().setId("l" + k).setVoltageLevel1("vl12").setNode1(lineNode)
                    .setVoltageLevel2(externals[k]).setBus2(externals[k] + "_b").setConnectableBus2(externals[k] + "_b")
                    .setR(0.4).setX(10).setG1(0).setB1(0).setG2(0).setB2(0).add();
            vl12.getNodeBreakerView().newBreaker().setId("bl" + k).setNode1(k).setNode2(loadNode).setOpen(false).add();
            vl12.newLoad().setId("ld" + k).setNode(loadNode).setP0(40 + 10 * k).setQ0(10 + 5 * k).add();
        }
        // the GPAX kind on bbs2: breaker, hub, disconnector, line
        vl12.getNodeBreakerView().newBreaker().setId("bf").setNode1(2).setNode2(40).setOpen(false).add();
        vl12.getNodeBreakerView().newDisconnector().setId("df").setNode1(40).setNode2(41).setOpen(false).add();
        network.newLine().setId("lf").setVoltageLevel1("vl12").setNode1(41)
                .setVoltageLevel2("vl3").setBus2("vl3_b").setConnectableBus2("vl3_b")
                .setR(0.4).setX(12).setG1(0).setB1(0).setG2(0).setB2(0).add();
        network.getVoltageLevel("vl0").newGenerator().setId("gs").setBus("vl0_b").setConnectableBus("vl0_b")
                .setMinP(0).setMaxP(2000).setTargetP(500).setTargetV(400).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl0").newLoad().setId("lds").setBus("vl0_b").setConnectableBus("vl0_b").setP0(300).setQ0(100).add();
        network.getVoltageLevel("vl3").newGenerator().setId("g3").setBus("vl3_b").setConnectableBus("vl3_b")
                .setMinP(0).setMaxP(1000).setTargetP(150).setTargetV(402).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl3").newLoad().setId("ld3").setBus("vl3_b").setConnectableBus("vl3_b").setP0(150).setQ0(50).add();
        network.getVoltageLevel("vl4").newLoad().setId("ld4").setBus("vl4_b").setConnectableBus("vl4_b").setP0(150).setQ0(50).add();
        network.newLine().setId("l34").setVoltageLevel1("vl3").setBus1("vl3_b").setConnectableBus1("vl3_b")
                .setVoltageLevel2("vl4").setBus2("vl4_b").setConnectableBus2("vl4_b").setR(0.4).setX(10).setG1(0).setB1(0).setG2(0).setB2(0).add();
        network.newLine().setId("l04").setVoltageLevel1("vl0").setBus1("vl0_b").setConnectableBus1("vl0_b")
                .setVoltageLevel2("vl4").setBus2("vl4_b").setConnectableBus2("vl4_b").setR(0.4).setX(10).setG1(0).setB1(0).setG2(0).setB2(0).add();
        return network;
    }

    /** Two SETPOINT-mode HVDC links (no AC emulation extension) from a node-breaker level to the external bus h, each
     *  rectifier station behind its own breaker at a section node. A converter station is not an end node for OLF's
     *  tripping, so the section's loss retains and opens that breaker and the station's bus is islanded on its own.
     *  Link a (section bbs1): that bus is ISOLATED in OLF's sense (Networks.isIsolatedBusForHvdc - a one-bus component,
     *  no load, no generator but the station) and LfContingency.processHvdcsWithoutPower zeroes BOTH stations: h loses
     *  cs2a's injection. Link b (section bbs2): the station shares its bus with a load through an internal connection,
     *  so OLF keeps the link's power and cs2b goes on injecting at h with its counterpart gone. The sections keep their
     *  own bus alive through a line behind a disconnector-hub-breaker feeder, and lose a load behind a breaker.
     *  network.xiidm EWVZ/EWWB/ZJI/ZJJ are link a's shape (EWVX_5 alone behind EWZM). */
    static Network setpointHvdcSectionsNetwork() {
        Network network = Network.create("setpoint-hvdc-sections", "code");
        com.powsybl.iidm.network.Substation s = network.newSubstation().setId("s").add();
        for (String b : List.of("a", "b", "h")) {
            s.newVoltageLevel().setId("vl_" + b).setNominalV(400).setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER)
                    .add().getBusBreakerView().newBus().setId(b).add();
        }
        com.powsybl.iidm.network.VoltageLevel vlc = s.newVoltageLevel().setId("vlc").setNominalV(400)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.NODE_BREAKER).add();
        String[] externals = {"a", "b", "a"};
        for (int k = 0; k < 3; k++) {
            vlc.getNodeBreakerView().newBusbarSection().setId("bbs" + k).setNode(k).add();
            if (k > 0) {
                vlc.getNodeBreakerView().newBreaker().setId("c" + (k - 1) + k).setNode1(k - 1).setNode2(k).setOpen(false).add();
            }
            vlc.getNodeBreakerView().newDisconnector().setId("d" + k).setNode1(k).setNode2(10 + k).setOpen(false).add();
            vlc.getNodeBreakerView().newBreaker().setId("bh" + k).setNode1(10 + k).setNode2(20 + k).setOpen(false).add();
            network.newLine().setId("l" + k).setVoltageLevel1("vlc").setNode1(20 + k)
                    .setVoltageLevel2("vl_" + externals[k]).setBus2(externals[k]).setConnectableBus2(externals[k])
                    .setR(0.4).setX(10 + 2 * k).setG1(0).setB1(0).setG2(0).setB2(0).add();
            if (k < 2) {
                vlc.getNodeBreakerView().newBreaker().setId("bl" + k).setNode1(k).setNode2(40 + k).setOpen(false).add();
                vlc.newLoad().setId("ld" + k).setNode(40 + k).setP0(30 + 10 * k).setQ0(10).add();
            }
        }
        // link a: the station alone behind its breaker on bbs1
        vlc.getNodeBreakerView().newBreaker().setId("bs1").setNode1(1).setNode2(31).setOpen(false).add();
        vlc.newVscConverterStation().setId("cs1a").setNode(31).setVoltageRegulatorOn(false).setVoltageSetpoint(400)
                .setReactivePowerSetpoint(0).setLossFactor(1.1f).add();
        // link b: the station shares its bus with a load (internal connection 32-33) on bbs2
        vlc.getNodeBreakerView().newBreaker().setId("bs2").setNode1(2).setNode2(32).setOpen(false).add();
        vlc.getNodeBreakerView().newInternalConnection().setNode1(32).setNode2(33).add();
        vlc.newVscConverterStation().setId("cs1b").setNode(32).setVoltageRegulatorOn(false).setVoltageSetpoint(400)
                .setReactivePowerSetpoint(0).setLossFactor(1.1f).add();
        vlc.newLoad().setId("ldb").setNode(33).setP0(15).setQ0(5).add();
        for (String link : List.of("a", "b")) {
            network.getVoltageLevel("vl_h").newVscConverterStation().setId("cs2" + link).setBus("h").setConnectableBus("h")
                    .setVoltageRegulatorOn(false).setVoltageSetpoint(400).setReactivePowerSetpoint(0).setLossFactor(1.1f).add();
            network.newHvdcLine().setId("hvdc" + link).setConverterStationId1("cs1" + link).setConverterStationId2("cs2" + link)
                    .setNominalV(400).setR(1).setActivePowerSetpoint("a".equals(link) ? 60 : 40)
                    .setConvertersMode(com.powsybl.iidm.network.HvdcLine.ConvertersMode.SIDE_1_RECTIFIER_SIDE_2_INVERTER)
                    .setMaxP(500).add();
        }
        java.util.function.BiConsumer<String[], double[]> line = (ids, rx) -> network.newLine().setId(ids[0])
                .setVoltageLevel1("vl_" + ids[1]).setBus1(ids[1]).setConnectableBus1(ids[1])
                .setVoltageLevel2("vl_" + ids[2]).setBus2(ids[2]).setConnectableBus2(ids[2])
                .setR(rx[0]).setX(rx[1]).setG1(0).setB1(0).setG2(0).setB2(0).add();
        line.accept(new String[] {"lab", "a", "b"}, new double[] {0.4, 10});
        line.accept(new String[] {"lah", "a", "h"}, new double[] {0.4, 15});
        line.accept(new String[] {"lbh", "b", "h"}, new double[] {0.4, 15});
        network.getVoltageLevel("vl_a").newGenerator().setId("ga").setBus("a").setConnectableBus("a")
                .setMinP(0).setMaxP(1000).setTargetP(250).setTargetV(400).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl_b").newGenerator().setId("gb").setBus("b").setConnectableBus("b")
                .setMinP(0).setMaxP(1000).setTargetP(100).setTargetV(402).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl_b").newLoad().setId("ldbb").setBus("b").setConnectableBus("b").setP0(120).setQ0(40).add();
        network.getVoltageLevel("vl_h").newLoad().setId("ldh").setBus("h").setConnectableBus("h").setP0(180).setQ0(30).add();
        return network;
    }

    /** b1 (slack g1) -- l12a, l12b -- b2 (load), an AC-emulation link cs1 (b1) -> cs2 (b2) whose inverter regulates b2
     *  with a curve opening with P (+-5 MVar at 0 MW, +-60 at +-100 MW); b3 hangs off b1 through l13 and carries
     *  nothing. OLF's own fixture for the curve read at the link flow (testAcEmulationVscReactiveLimitsFollowTheFlow). */
    static Network acEmulationCurveLimitedVscNetwork() {
        return acEmulationVscNetwork(true);
    }

    /** The same link, its inverter with a curve or with plain +-60 MVar limits. */
    static Network acEmulationVscNetwork(boolean curve) {
        Network network = Network.create("acEmulationCurve", "code");
        com.powsybl.iidm.network.Substation s = network.newSubstation().setId("s").add();
        for (String b : List.of("b1", "b2", "b3")) {
            s.newVoltageLevel().setId("vl" + b.substring(1)).setNominalV(400)
                    .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER)
                    .add().getBusBreakerView().newBus().setId(b).add();
        }
        network.getVoltageLevel("vl1").newGenerator().setId("g1").setBus("b1").setConnectableBus("b1")
                .setMinP(0).setMaxP(1000).setTargetP(200).setTargetV(400).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl2").newLoad().setId("ld2").setBus("b2").setConnectableBus("b2").setP0(150).setQ0(60).add();
        for (String id : List.of("l12a", "l12b")) {
            network.newLine().setId(id).setBus1("b1").setConnectableBus1("b1").setBus2("b2").setConnectableBus2("b2")
                    .setR(1).setX(20).setG1(0).setB1(0).setG2(0).setB2(0).add();
        }
        network.newLine().setId("l13").setBus1("b1").setConnectableBus1("b1").setBus2("b3").setConnectableBus2("b3")
                .setR(1).setX(20).setG1(0).setB1(0).setG2(0).setB2(0).add();
        network.getVoltageLevel("vl1").newVscConverterStation().setId("cs1").setBus("b1").setConnectableBus("b1")
                .setLossFactor(1.1f).setReactivePowerSetpoint(0).setVoltageRegulatorOn(false).add();
        com.powsybl.iidm.network.VscConverterStation cs2 = network.getVoltageLevel("vl2").newVscConverterStation()
                .setId("cs2").setBus("b2").setConnectableBus("b2").setLossFactor(1.1f).setVoltageSetpoint(410)
                .setVoltageRegulatorOn(true).add();
        if (curve) {
            cs2.newReactiveCapabilityCurve()
                    .beginPoint().setP(-100).setMinQ(-60).setMaxQ(60).endPoint()
                    .beginPoint().setP(0).setMinQ(-5).setMaxQ(5).endPoint()
                    .beginPoint().setP(100).setMinQ(-60).setMaxQ(60).endPoint()
                    .add();
        } else {
            cs2.newMinMaxReactiveLimits().setMinQ(-60).setMaxQ(60).add();
        }
        network.newHvdcLine().setId("hvdc12").setConverterStationId1("cs1").setConverterStationId2("cs2")
                .setNominalV(400).setR(1).setMaxP(200).setActivePowerSetpoint(60)
                .setConvertersMode(com.powsybl.iidm.network.HvdcLine.ConvertersMode.SIDE_1_RECTIFIER_SIDE_2_INVERTER)
                .add()
                .newExtension(com.powsybl.iidm.network.extensions.HvdcAngleDroopActivePowerControlAdder.class)
                .withP0(40).withDroop(180).withEnabled(true).add();
        return network;
    }

    /** OLF's g1/g2/g3 -> b4 remote voltage control with what a contingency can take from the PILOT b4: a second load
     *  (l4b, 30 MW / 20 MVar) and a fixed-Q generator (g4q, 10 MW / 15 MVar, outside the slack distribution - a lost
     *  participant beside a participating slack is another shape). The slack is b1, so b4 is the pilot only. */
    static Network remotePilotInjectionsNetwork() {
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createWithGeneratorRemoteControl();
        network.getVoltageLevel("vl4").newLoad()
                .setId("l4b")
                .setBus("b4")
                .setConnectableBus("b4")
                .setP0(30)
                .setQ0(20)
                .add();
        network.getVoltageLevel("vl4").newGenerator()
                .setId("g4q")
                .setBus("b4")
                .setConnectableBus("b4")
                .setMinP(0)
                .setMaxP(50)
                .setTargetP(10)
                .setTargetQ(15)
                .setVoltageRegulatorOn(false)
                .add();
        network.getGenerator("g4q").newExtension(com.powsybl.iidm.network.extensions.ActivePowerControlAdder.class)
                .withParticipate(false)
                .add();
        return network;
    }

    /** IEEE 14 with a 135 kV HUB H: one line to B1 and six to dead-end buses (two carrying 5 MW), so H is the most
     *  meshed bus and OLF's slack (the importer's slack terminal removed). Losing the line to B1 isolates the hub's
     *  star. */
    static Network isolatedSlackHubNetwork() {
        Network network = IeeeCdfNetworkFactory.create14();
        network.getVoltageLevels().forEach(vl -> vl.removeExtension(com.powsybl.iidm.network.extensions.SlackTerminal.class));
        com.powsybl.iidm.network.VoltageLevel vl1 = network.getVoltageLevel("VL1");
        com.powsybl.iidm.network.VoltageLevel vlh = vl1.getSubstation().orElseThrow().newVoltageLevel()
                .setId("VLH")
                .setNominalV(vl1.getNominalV())
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER)
                .add();
        vlh.getBusBreakerView().newBus().setId("H").add();
        network.newLine()
                .setId("lH")
                .setVoltageLevel1("VL1").setBus1("B1").setConnectableBus1("B1")
                .setVoltageLevel2("VLH").setBus2("H").setConnectableBus2("H")
                .setR(0.5).setX(5)
                .add();
        for (int k = 1; k <= 6; k++) {
            vlh.getBusBreakerView().newBus().setId("L" + k).add();
            network.newLine()
                    .setId("lL" + k)
                    .setVoltageLevel1("VLH").setBus1("H").setConnectableBus1("H")
                    .setVoltageLevel2("VLH").setBus2("L" + k).setConnectableBus2("L" + k)
                    .setR(0.5).setX(5)
                    .add();
        }
        for (int k = 1; k <= 2; k++) {
            vlh.newLoad().setId("dL" + k).setBus("L" + k).setConnectableBus("L" + k).setP0(5).setQ0(2).add();
        }
        return network;
    }

    /** g1 - b1 == l12 / l12_BIS == b2 - g2, with converter buses b1h / b2h hung off b1 / b2 by short
     *  lines; hvdc12 (droop 1, P0 0) and hvdc12_BIS (droop 1, P0 1 MW) both run b1h -> b2h. */
    static Network doubledHvdcCorridorNetwork() {
        Network network = Network.create("doubled-corridor", "code");
        String[] buses = {"b1", "b2", "b1h", "b2h"};
        for (String b : buses) {
            var vl = network.newSubstation().setId("s_" + b).add()
                    .newVoltageLevel().setId("vl_" + b).setNominalV(1).setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
            vl.getBusBreakerView().newBus().setId(b).add();
        }
        network.getVoltageLevel("vl_b1").newGenerator().setId("g1").setBus("b1").setConnectableBus("b1")
                .setEnergySource(com.powsybl.iidm.network.EnergySource.OTHER).setMinP(0).setMaxP(10)
                .setTargetP(5).setTargetV(1).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl_b2").newGenerator().setId("g2").setBus("b2").setConnectableBus("b2")
                .setEnergySource(com.powsybl.iidm.network.EnergySource.OTHER).setMinP(0).setMaxP(10)
                .setTargetP(0).setTargetV(1).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl_b1").newLoad().setId("l1").setBus("b1").setConnectableBus("b1").setP0(3).setQ0(0).add();
        network.getVoltageLevel("vl_b2").newLoad().setId("l2").setBus("b2").setConnectableBus("b2").setP0(3).setQ0(0).add();
        network.getVoltageLevel("vl_b1h").newLoad().setId("l1h").setBus("b1h").setConnectableBus("b1h").setP0(0.5).setQ0(0.1).add();
        network.getVoltageLevel("vl_b2h").newLoad().setId("l2h").setBus("b2h").setConnectableBus("b2h").setP0(0.5).setQ0(0.1).add();
        java.util.function.BiConsumer<String[], double[]> line = (ids, rx) -> network.newLine().setId(ids[0])
                .setVoltageLevel1("vl_" + ids[1]).setBus1(ids[1]).setConnectableBus1(ids[1])
                .setVoltageLevel2("vl_" + ids[2]).setBus2(ids[2]).setConnectableBus2(ids[2])
                .setR(rx[0]).setX(rx[1]).setG1(0).setB1(0).setG2(0).setB2(0).add();
        line.accept(new String[] {"l12", "b1", "b2"}, new double[] {0, 0.1});
        line.accept(new String[] {"l12_BIS", "b1", "b2"}, new double[] {0, 0.2});
        line.accept(new String[] {"ln1h", "b1", "b1h"}, new double[] {0.002, 0.02});
        line.accept(new String[] {"ln2h", "b2", "b2h"}, new double[] {0.002, 0.02});
        // PQ converter buses: a voltage-controlled converter bus pins |V| and the lost increment
        // then moves only the ANGLES, which the bus-voltage comparison does not see
        for (String[] station : new String[][] {{"cs1", "b1h", "false"}, {"cs2", "b2h", "false"}, {"cs1b", "b1h", "false"}, {"cs2b", "b2h", "false"}}) {
            network.getVoltageLevel("vl_" + station[1]).newVscConverterStation().setId(station[0]).setBus(station[1]).setConnectableBus(station[1])
                    .setVoltageRegulatorOn(Boolean.parseBoolean(station[2])).setVoltageSetpoint(1).setReactivePowerSetpoint(0)
                    .setLossFactor(1.1f).add();
        }
        for (String[] hvdcLine : new String[][] {{"hvdc12", "cs1", "cs2", "0"}, {"hvdc12_BIS", "cs1b", "cs2b", "1"}}) {
            network.newHvdcLine().setId(hvdcLine[0]).setConverterStationId1(hvdcLine[1]).setConverterStationId2(hvdcLine[2])
                    .setNominalV(400).setR(0.1).setActivePowerSetpoint(2)
                    .setConvertersMode(com.powsybl.iidm.network.HvdcLine.ConvertersMode.SIDE_1_INVERTER_SIDE_2_RECTIFIER)
                    .setMaxP(10).add()
                    .newExtension(com.powsybl.iidm.network.extensions.HvdcAngleDroopActivePowerControlAdder.class)
                    .withDroop(1).withP0(Float.parseFloat(hvdcLine[3])).withEnabled(true).add();
        }
        return network;
    }

    static Network switchedToPqInBaseRemoteMemberNetwork() {
        // g1, g2 and g3 regulate b4 (413.4 kV) remotely; g5 holds b5, behind a 60 Ohm line, on its own
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createWithGeneratorRemoteControl();
        network.getLoad("l4").setQ0(-80);
        network.getVoltageLevel("vl4").newLoad().setId("lq").setBus("b4").setConnectableBus("b4").setP0(0).setQ0(250).add();
        double[] qPercent = {30, 40, 30};
        for (int i = 1; i <= 3; i++) {
            com.powsybl.iidm.network.Generator g = network.getGenerator("g" + i);
            g.newExtension(com.powsybl.iidm.network.extensions.CoordinatedReactiveControlAdder.class)
                    .withQPercent(qPercent[i - 1]).add();
            if (i == 2) {
                g.newMinMaxReactiveLimits().setMinQ(-30).setMaxQ(30).add();   // base share 95 MVar: pinned
            } else {
                g.newMinMaxReactiveLimits().setMinQ(-5).setMaxQ(300).add();   // base 105 MVar, -22 once lq is lost
            }
        }
        com.powsybl.iidm.network.VoltageLevel vl5 = network.getSubstation("s").newVoltageLevel().setId("vl5").setNominalV(400)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
        vl5.getBusBreakerView().newBus().setId("b5").add();
        vl5.newGenerator().setId("g5").setBus("b5").setConnectableBus("b5").setMinP(0).setMaxP(300).setTargetP(100)
                .setTargetV(405).setVoltageRegulatorOn(true).add();
        network.getGenerator("g5").newMinMaxReactiveLimits().setMinQ(-500).setMaxQ(500).add();
        vl5.newLoad().setId("l5").setBus("b5").setConnectableBus("b5").setP0(100).setQ0(10).add();
        network.newLine().setId("l45").setVoltageLevel1("vl4").setBus1("b4").setVoltageLevel2("vl5").setBus2("b5")
                .setR(1).setX(60).setG1(0).setB1(0).setG2(0).setB2(0).add();
        return network;
    }

    /** splitGroupNetwork(2) with a reactive-capability curve on g0 (MAX_Q 60 -> 62 MVAr over 0..500 MW, so a
     *  slack move of a few hundred kW drifts the limit by ~1e-5 pu) and a 60 MVAr ceiling on g1. */
    static Network pilotSelfCurveNetwork() {
        return pilotSelfCurveNetwork(false, 0);
    }

    /** @param curveOnChild the same curve on g1 (the relocated child section's generator) as well
     *  @param g3TargetP active power on the vl3 support generator, pre-distributed by the SA when it is lost */
    static Network pilotSelfCurveNetwork(boolean curveOnChild, double g3TargetP) {
        Network network = splitGroupNetwork(2);
        network.getGenerator("g0").setTargetV(PSC_TARGET_V);
        network.getGenerator("g1").setTargetV(PSC_TARGET_V);
        network.getGenerator("g0").newReactiveCapabilityCurve()
                .beginPoint().setP(0).setMinQ(-300).setMaxQ(PSC_MAX_Q).endPoint()
                .beginPoint().setP(500).setMinQ(-300).setMaxQ(PSC_MAX_Q + 2).endPoint()
                .add();
        if (curveOnChild) {
            network.getGenerator("g1").newReactiveCapabilityCurve()
                    .beginPoint().setP(0).setMinQ(-300).setMaxQ(PSC_MAX_Q).endPoint()
                    .beginPoint().setP(500).setMinQ(-300).setMaxQ(PSC_MAX_Q + 2).endPoint()
                    .add();
        } else {
            network.getGenerator("g1").newMinMaxReactiveLimits().setMinQ(-300).setMaxQ(PSC_MAX_Q).add();
        }
        network.getLoad("ld0").setQ0(PSC_LOAD_Q);
        // heavy reactive loads behind vl3: g3 carries them in the base, the merged sections once g3 is lost
        network.getLoad("ld3").setQ0(PSC_REMOTE_Q);
        network.getLoad("ld4").setQ0(PSC_REMOTE_Q);
        // reactive support at vl3 with no active power: its loss moves (almost) no P, only the reactive need
        network.getVoltageLevel("vl3").newGenerator().setId("g3").setBus("vl3_b").setConnectableBus("vl3_b")
                .setMinP(0).setMaxP(Math.max(100, 2 * g3TargetP)).setTargetP(g3TargetP).setTargetV(400).setVoltageRegulatorOn(true).add();
        network.getGenerator("g3").newMinMaxReactiveLimits().setMinQ(-300).setMaxQ(300).add();
        if (g3TargetP > 0) {
            // a lost PARTICIPATING generator with a participating slack is a shape the device routes to the
            // CPU (PARTICIPATING_SLACK): keep the slack out of the distribution so the loss pre-distributes
            // over g0/g1 on the device
            network.getGenerator("gs").newExtension(com.powsybl.iidm.network.extensions.ActivePowerControlAdder.class)
                    .withParticipate(false).add();
        }
        return network;
    }

    static final double PSC_TARGET_V = 401;

    static final double PSC_MAX_Q = 60;

    static final double PSC_LOAD_Q = 0;

    static final double PSC_REMOTE_Q = 150;

    /** {@code deviceHvdcSaturatedBaseMatchesCpu}'s network plus a stand-by SVC on b4 (400 kV). */
    static Network hvdcWithStandbySvcNetwork(double lowThreshold, double highThreshold) {
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
        network.getVoltageLevel("b4_vl").newStaticVarCompensator()
                .setId("svc4").setConnectableBus("b4").setBus("b4")
                .setBmin(-0.008).setBmax(0.008)
                .setVoltageSetpoint(400)
                .setRegulationMode(com.powsybl.iidm.network.StaticVarCompensator.RegulationMode.VOLTAGE)
                .setRegulating(true)
                .add();
        network.getStaticVarCompensator("svc4")
                .newExtension(com.powsybl.iidm.network.extensions.StandbyAutomatonAdder.class)
                .withLowVoltageThreshold(lowThreshold).withHighVoltageThreshold(highThreshold)
                .withLowVoltageSetpoint(398.5).withHighVoltageSetpoint(399)
                .withB0(-0.001f).withStandbyStatus(true).add();
        return network;
    }

    static List<Contingency> hvdcWithStandbySvcContingencies() {
        return List.of(new Contingency("l14_BIS", new BranchContingency("l14_BIS")),
                new Contingency("l4", new com.powsybl.contingency.LoadContingency("l4")));
    }

    static com.powsybl.loadflow.LoadFlowParameters hvdcWithStandbySvcParameters() {
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setHvdcAcEmulation(true).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp).setSvcVoltageMonitoring(true);
        return lfp;
    }

    static Network secondaryVcRemoteNetwork() {
        Network network = com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory.create14();
        com.powsybl.iidm.network.Generator b6g = network.getGenerator("B6-G");
        b6g.setRegulatingTerminal(network.getLine("L2-5-1").getTerminal2())   // B5 end
                .setTargetV(1.02 * network.getBusBreakerView().getBus("B5").getVoltageLevel().getNominalV())
                .setVoltageRegulatorOn(true);
        network.newExtension(com.powsybl.iidm.network.extensions.SecondaryVoltageControlAdder.class)
                .newControlZone().withName("z1")
                .newPilotPoint().withTargetV(13).withBusbarSectionsOrBusesIds(List.of("B10")).add()
                .newControlUnit().withId("B6-G").add()
                .newControlUnit().withId("B8-G").add()
                .add().add();
        return network;
    }

    /** ieee14 with a radial bus B15 behind line L14-15-1, equipped by {@code equip}. */
    static Network ieee14WithRadialBus(java.util.function.Consumer<com.powsybl.iidm.network.VoltageLevel> equip) {
        Network network = IeeeCdfNetworkFactory.create14();
        com.powsybl.iidm.network.VoltageLevel vl14 = network.getVoltageLevel("VL14");
        com.powsybl.iidm.network.VoltageLevel vl15 = network.newSubstation().setId("S15").add().newVoltageLevel().setId("VL15")
                .setNominalV(vl14.getNominalV()).setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
        vl15.getBusBreakerView().newBus().setId("B15").add();
        equip.accept(vl15);
        network.newLine().setId("L14-15-1").setVoltageLevel1("VL14").setBus1("B14").setVoltageLevel2("VL15").setBus2("B15")
                .setR(0.05).setX(0.2).setG1(0).setB1(0).setG2(0).setB2(0).add();
        return network;
    }

    /** A GeneratorAction re-checks its generator's participation (OLF's reApplyActivePowerControlChecks): a target
     *  outside [minP, maxP] takes it out of the distribution, a target back inside brings it in. B4's load loss
     *  lowers generation; every generator has room, so only the actions decide who moves. */
    static Network participationRecheckNetwork() {
        Network network = IeeeCdfNetworkFactory.create14();
        network.getGenerator("B1-G").setMinP(0).setMaxP(400);
        network.getGenerator("B2-G").setMinP(30).setMaxP(60);
        network.getVoltageLevel("VL6").newGenerator().setId("B6-GP").setBus("B6").setConnectableBus("B6")
                .setTargetP(20).setTargetQ(0).setVoltageRegulatorOn(false).setMinP(0).setMaxP(80).add();
        // starts outside its limits: not a participant until an action puts it inside
        network.getVoltageLevel("VL8").newGenerator().setId("B8-GP").setBus("B8").setConnectableBus("B8")
                .setTargetP(5).setTargetQ(0).setVoltageRegulatorOn(false).setMinP(10).setMaxP(50).add();
        return network;
    }

    /** A condition that always holds without being a TrueCondition: every one of no element is violated. */
    static com.powsybl.contingency.strategy.condition.Condition alwaysHolds() {
        return new com.powsybl.contingency.strategy.condition.AllViolationCondition(List.of());
    }

    static com.powsybl.contingency.strategy.ConditionalActions stage(String id,
                                                                           com.powsybl.contingency.strategy.condition.Condition condition,
                                                                           String... actionIds) {
        return new com.powsybl.contingency.strategy.ConditionalActions(id, condition, List.of(actionIds));
    }

    /** More conditional stages than the device speculates on (all holding): the strategy stays on the CPU. */
    static com.powsybl.contingency.strategy.OperatorStrategy fourConditionalStageStrategy(String id, String ctgId,
                                                                                                String action1, String action2) {
        return new com.powsybl.contingency.strategy.OperatorStrategy(id,
                com.powsybl.contingency.ContingencyContext.specificContingency(ctgId), List.of(
                        stage("c1", alwaysHolds(), action1), stage("c2", alwaysHolds(), action2),
                        stage("c3", alwaysHolds(), action1), stage("c4", alwaysHolds(), action1)));
    }

    static com.powsybl.action.Action genAction(String id, String gen, boolean relative, double value) {
        return new com.powsybl.action.GeneratorActionBuilder().withId(id).withGeneratorId(gen)
                .withActivePowerRelativeValue(relative).withActivePowerValue(value).build();
    }

    static com.powsybl.iidm.network.Load loadAtBus(Network network, String busId) {
        return network.getLoadStream()
                .filter(l -> busId.equals(l.getTerminal().getBusBreakerView().getBus().getId()))
                .findFirst().orElseThrow();
    }
}
