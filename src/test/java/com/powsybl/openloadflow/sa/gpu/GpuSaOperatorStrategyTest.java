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
import com.powsybl.openloadflow.network.gpu.ZeroImpedanceTreeReplayTest;
import com.powsybl.openloadflow.sa.OpenSecurityAnalysisProvider;
import com.powsybl.security.SecurityAnalysisResult;
import org.junit.jupiter.api.Test;

import java.util.List;

import static com.powsybl.openloadflow.sa.gpu.GpuSaTestNetworks.*;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * GPU-batched security analysis against the CPU: operator strategies and their actions.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuSaOperatorStrategyTest extends AbstractGpuSaTest {

    GpuSaOperatorStrategyTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    /** Strategy pairs on the coupler-joined curve sections. Opening the coupler (c01) brings the child's balance home
     *  to its own angle row: the pair re-anchors its distribution on what the parent absorbed THERE, not on the
     *  coupler's dummy row, a fixed stub the parent's capture read as absorption (rte6515 BUS-1359_BBS_2: 19.27 MW).
     *  Losing g3 (the slack out of the distribution) moves the slack's share of the parent: none. Losing g3 switches both sections to PQ, so the group loses control and the pilot's
     *  own row hosts its reactive balance: a pair starts with that flip at the parent's frozen target, which the
     *  seed took from the base limit instead (rte6515 BUS-6066_BBS_1, VL-6231_0: 3.8e-5 kV even for a no-op). */
    @Test
    void strategiesOnCoupledCurveSectionsMatchCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = pilotSelfCurveNetwork(true, 100);
        List<Contingency> ctgs = List.of(
                new Contingency("c01", new com.powsybl.contingency.SwitchContingency("c01")),
                new Contingency("g3", new com.powsybl.contingency.GeneratorContingency("g3")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.LoadActionBuilder().withId("more4").withLoadId("ld4").withRelativeValue(true)
                        .withActivePowerValue(20).build(),
                new com.powsybl.action.LoadActionBuilder().withId("same4").withLoadId("ld4").withRelativeValue(true)
                        .withActivePowerValue(0).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = new java.util.ArrayList<>();
        for (Contingency c : ctgs) {
            for (String a : List.of("more4", "same4")) {
                strategies.add(new com.powsybl.contingency.strategy.OperatorStrategy("s-" + a + "-" + c.getId(),
                        com.powsybl.contingency.ContingencyContext.specificContingency(c.getId()),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of(a)));
            }
        }
        assertStrategyMatchesCpu(network, ctgs, strategies, actions, 6, true,
                p -> p.setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                        .setSlackBusesIds(List.of("vl0_b")));
    }

    /** A merged control whose group is lost in the BASE (both sections switched to PQ at MAX_Q under a heavy local reactive
     *  load) and switched to PQ at the other limit by the contingency (the load lost: both absorb down to MIN_Q). A pair starts
     *  with the pilot's row as the reactive balance the base activation laid out, at the BASE frozen target Q; OLF continues the
     *  contingency's (rte6515 BUS-6230_BBS, VL-6229_0: MAX_Q 0.0 in the base, MIN_Q -0.06 after it, the pair at 0.0). */
    @Test
    void strategiesOnABaseLostMergedControlMatchCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = splitGroupNetwork(2);
        for (String g : List.of("g0", "g1")) {
            network.getGenerator(g).setTargetV(398);
            network.getGenerator(g).newMinMaxReactiveLimits().setMinQ(-5).setMaxQ(60).add();
        }
        network.getLoad("ld0").setQ0(200);
        // a listed section loss keeps the coupler retained: two sections under one merged control, the pilot one of
        // its own controllers
        List<Contingency> ctgs = List.of(new Contingency("bus-1", new com.powsybl.contingency.BusbarSectionContingency("bbs1")),
                new Contingency("ld0", new com.powsybl.contingency.LoadContingency("ld0")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.LoadActionBuilder().withId("more4").withLoadId("ld4").withRelativeValue(true)
                        .withActivePowerValue(20).build(),
                new com.powsybl.action.LoadActionBuilder().withId("same4").withLoadId("ld4").withRelativeValue(true)
                        .withActivePowerValue(0).build());
        assertStrategyMatchesCpu(network, ctgs, strategiesOf("ld0", "more4", "same4"), actions, 4, true,
                p -> p.setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                        .setSlackBusesIds(List.of("vl0_b")));
    }

    /** A remote member switched to PQ in the BASE that the contingency's reactive-limit pass RELEASES: g2's 40% key is far over
     *  its 30 MVAr ceiling under the base need, so the base switches it to PQ at MAX_Q; losing the reactive load lq leaves the group
     *  absorbing, switches g1 and g3 to PQ at their -5 MVAr floor (g5 keeps the network a PV bus), the voltage climbs over its
     *  target and g2 goes back on control. A pair starts from that state, g2 a DISTR_Q member again; the seed released
     *  only LOCAL buses switched to PQ in the base, so the pair solved g2 at its base frozen target Q first (network.xiidm LLJP: a member at -0.2954 the
     *  parent had released, 5.46 kV). */
    @Test
    void strategiesAfterTheReleaseOfASwitchedToPqInBaseRemoteMemberMatchCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = switchedToPqInBaseRemoteMemberNetwork();
        List<Contingency> ctgs = List.of(new Contingency("lq", new com.powsybl.contingency.LoadContingency("lq")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.LoadActionBuilder().withId("more4").withLoadId("l4").withRelativeValue(true)
                        .withActivePowerValue(20).build(),
                new com.powsybl.action.LoadActionBuilder().withId("same4").withLoadId("l4").withRelativeValue(true)
                        .withActivePowerValue(0).build());
        assertStrategyMatchesCpu(network, ctgs, strategiesOf("lq", "more4", "same4"), actions, 3, true);
    }

    /** OLF runs a strategy as a new load flow, and its reactive-limit loop counts PV -> PQ switches afresh in every
     *  load flow. Losing the capacitor switches g1 to PQ at its 100 MVAr ceiling, its one switch allowed; the strategy takes the
     *  reactive load off its bus, the voltage climbs over the target and OLF puts g1 back on control. The pair took
     *  its parent's count along and kept g1 blocked at the ceiling (network.xiidm LLFP: IZSV_0 and IZTH_0/2 at three
     *  switches each after the contingency's oscillation, 8.5e-4 kV at IZTK). */
    @Test
    void strategiesCountReactiveLimitSwitchesAfreshLikeCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = Network.create("pv-switch-count", "code");
        com.powsybl.iidm.network.Substation s = network.newSubstation().setId("s").add();
        for (String vl : List.of("vl0", "vlg")) {
            s.newVoltageLevel().setId(vl).setNominalV(400).setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER)
                    .add().getBusBreakerView().newBus().setId(vl + "_b").add();
        }
        network.getVoltageLevel("vl0").newGenerator().setId("gs").setBus("vl0_b").setConnectableBus("vl0_b")
                .setMinP(0).setMaxP(2000).setTargetP(300).setTargetV(400).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl0").newLoad().setId("ld0").setBus("vl0_b").setConnectableBus("vl0_b").setP0(100).setQ0(20).add();
        network.getVoltageLevel("vlg").newGenerator().setId("g1").setBus("vlg_b").setConnectableBus("vlg_b")
                .setMinP(0).setMaxP(500).setTargetP(100).setTargetV(400.5).setVoltageRegulatorOn(true).add();
        network.getGenerator("g1").newMinMaxReactiveLimits().setMinQ(-50).setMaxQ(100).add();
        network.getVoltageLevel("vlg").newLoad().setId("ldg").setBus("vlg_b").setConnectableBus("vlg_b").setP0(300).setQ0(60).add();
        network.getVoltageLevel("vlg").newLoad().setId("cap").setBus("vlg_b").setConnectableBus("vlg_b").setP0(0).setQ0(-80).add();
        for (String id : List.of("la", "lb")) {
            network.newLine().setId(id).setVoltageLevel1("vl0").setBus1("vl0_b").setConnectableBus1("vl0_b")
                    .setVoltageLevel2("vlg").setBus2("vlg_b").setConnectableBus2("vlg_b")
                    .setR(2).setX(12).setG1(0).setB1(0).setG2(0).setB2(0).add();
        }
        List<Contingency> ctgs = List.of(new Contingency("cap", new com.powsybl.contingency.LoadContingency("cap")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.LoadActionBuilder().withId("lessQ").withLoadId("ldg").withRelativeValue(true)
                        .withReactivePowerValue(-60).build(),
                new com.powsybl.action.LoadActionBuilder().withId("sameQ").withLoadId("ldg").withRelativeValue(true)
                        .withReactivePowerValue(0).build());
        assertStrategyMatchesCpu(network, ctgs, strategiesOf("cap", "lessQ", "sameQ"), actions, 3, true,
                p -> p.setReactiveLimitsMaxPqPvSwitch(1).setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                        .setSlackBusesIds(List.of("vl0_b")));
    }

    /** OLF operates a branch's terminals only when its model has both ends (LfTerminalsConnectionAction: "has one
     *  missing bus" otherwise, and the action changes nothing): a line connected at one end in the network stays as the
     *  contingency left it, whether a strategy closes or opens it. The device composed the close as a reclose, putting
     *  the line's dangling charging back (network.xiidm LJQY, LJRT, LNBF: open at one end, their reclose 2.03, 0.65 and
     *  0.06 kV off, OLF's strategy equal to its contingency), and the open as an outage. */
    @Test
    void strategiesOperatingALineOpenAtOneEndMatchCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = Network.create("half-open-line", "code");
        com.powsybl.iidm.network.Substation s = network.newSubstation().setId("s").add();
        for (String vl : List.of("vl0", "vl1", "vl2")) {
            s.newVoltageLevel().setId(vl).setNominalV(400).setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER)
                    .add().getBusBreakerView().newBus().setId(vl + "_b").add();
        }
        network.getVoltageLevel("vl0").newGenerator().setId("gs").setBus("vl0_b").setConnectableBus("vl0_b")
                .setMinP(0).setMaxP(2000).setTargetP(300).setTargetV(400).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl1").newLoad().setId("ld1").setBus("vl1_b").setConnectableBus("vl1_b").setP0(250).setQ0(60).add();
        network.getVoltageLevel("vl2").newLoad().setId("ld2").setBus("vl2_b").setConnectableBus("vl2_b").setP0(40).setQ0(10).add();
        for (String id : List.of("l01a", "l01b")) {
            network.newLine().setId(id).setVoltageLevel1("vl0").setBus1("vl0_b").setConnectableBus1("vl0_b")
                    .setVoltageLevel2("vl1").setBus2("vl1_b").setConnectableBus2("vl1_b")
                    .setR(2).setX(20).setG1(0).setB1(0).setG2(0).setB2(0).add();
        }
        network.newLine().setId("l02").setVoltageLevel1("vl0").setBus1("vl0_b").setConnectableBus1("vl0_b")
                .setVoltageLevel2("vl2").setBus2("vl2_b").setConnectableBus2("vl2_b")
                .setR(2).setX(20).setG1(0).setB1(0).setG2(0).setB2(0).add();
        // lh: connected at vl1 only, charged - its dangling half is a reactive source at vl1
        network.newLine().setId("lh").setVoltageLevel1("vl1").setBus1("vl1_b").setConnectableBus1("vl1_b")
                .setVoltageLevel2("vl2").setConnectableBus2("vl2_b")
                .setR(4).setX(40).setG1(0).setB1(5e-4).setG2(0).setB2(5e-4).add();
        List<Contingency> ctgs = List.of(new Contingency("lh", new BranchContingency("lh")),
                new Contingency("l01b", new BranchContingency("l01b")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.TerminalsConnectionActionBuilder().withId("close-lh").withNetworkElementId("lh").withOpen(false).build(),
                new com.powsybl.action.TerminalsConnectionActionBuilder().withId("open-lh").withNetworkElementId("lh").withOpen(true).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = new java.util.ArrayList<>(strategiesOf("lh", "close-lh"));
        strategies.addAll(strategiesOf("l01b", "open-lh"));
        assertStrategyMatchesCpu(network, ctgs, strategies, actions, 4, true);
    }

    /** A reclose that brings an islanded bus back: OLF re-enables every branch at a bus it adds back to the main component
     *  that is not connected at both ends (AbstractLfBranchAction.updateBusesAndBranchStatus), even one the contingency
     *  itself outaged. The contingency takes la and the dangling lh off vx; closing la brings vx back with lh's charging,
     *  which the device left out (network.xiidm LUSW, propagated to LSZO and the dangling LVAV: LAHV 99.08 kV against
     *  OLF's 99.82). */
    @Test
    void reclosingIntoAnIslandBringsBackItsDanglingBranchesLikeCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = Network.create("reclose-dangling", "code");
        com.powsybl.iidm.network.Substation s = network.newSubstation().setId("s").add();
        for (String vl : List.of("vl0", "vl1", "vx", "vy")) {
            s.newVoltageLevel().setId(vl).setNominalV(400).setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER)
                    .add().getBusBreakerView().newBus().setId(vl + "_b").add();
        }
        network.getVoltageLevel("vl0").newGenerator().setId("gs").setBus("vl0_b").setConnectableBus("vl0_b")
                .setMinP(0).setMaxP(2000).setTargetP(300).setTargetV(400).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl1").newLoad().setId("ld1").setBus("vl1_b").setConnectableBus("vl1_b").setP0(250).setQ0(60).add();
        network.getVoltageLevel("vx").newLoad().setId("ldx").setBus("vx_b").setConnectableBus("vx_b").setP0(10).setQ0(2).add();
        network.getVoltageLevel("vy").newLoad().setId("ldy").setBus("vy_b").setConnectableBus("vy_b").setP0(5).setQ0(1).add();
        java.util.function.BiConsumer<String, String[]> line = (id, e) -> network.newLine().setId(id)
                .setVoltageLevel1(e[0]).setBus1(e[0] + "_b").setConnectableBus1(e[0] + "_b")
                .setVoltageLevel2(e[1]).setBus2(e[1] + "_b").setConnectableBus2(e[1] + "_b")
                .setR(2).setX(20).setG1(0).setB1(0).setG2(0).setB2(0).add();
        line.accept("l01a", new String[] {"vl0", "vl1"});
        line.accept("l01b", new String[] {"vl0", "vl1"});
        line.accept("l0y", new String[] {"vl0", "vy"});
        line.accept("la", new String[] {"vl1", "vx"});
        // lh: connected at vx only, charged
        network.newLine().setId("lh").setVoltageLevel1("vx").setBus1("vx_b").setConnectableBus1("vx_b")
                .setVoltageLevel2("vy").setConnectableBus2("vy_b")
                .setR(4).setX(40).setG1(0).setB1(5e-4).setG2(0).setB2(5e-4).add();
        List<Contingency> ctgs = List.of(new Contingency("la-lh", new BranchContingency("la"), new BranchContingency("lh")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.TerminalsConnectionActionBuilder().withId("close-la").withNetworkElementId("la").withOpen(false).build());
        assertStrategyMatchesCpu(network, ctgs, strategiesOf("la-lh", "close-la"), actions, 2, true);
    }

    @Test
    void operatorStrategyBatchMatchesCpu() {
        // #24 stage 1: a TrueCondition strategy (load shed + generator redispatch) becomes an EXTRA
        // batch scenario composed onto its contingency; the injected OperatorStrategyResult must
        // match the CPU's at state level.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        String lineId = network.getLineStream()
                .filter(l -> !l.getId().contains("7") && !l.getId().contains("8"))
                .skip(1).findFirst().orElseThrow().getId();
        com.powsybl.iidm.network.Load load12 = network.getLoadStream()
                .filter(l -> "B12".equals(l.getTerminal().getBusBreakerView().getBus().getId()))
                .findFirst().orElseThrow();
        List<Contingency> contingencies = List.of(new Contingency(lineId, new BranchContingency(lineId)));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.LoadActionBuilder().withId("shed12").withLoadId(load12.getId())
                        .withRelativeValue(false).withActivePowerValue(load12.getP0() / 2).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("s1",
                        com.powsybl.contingency.ContingencyContext.specificContingency(lineId),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("shed12")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        java.util.Set<String> allVls = network.getVoltageLevelStream().map(vl -> vl.getId())
                .collect(java.util.stream.Collectors.toSet());
        List<com.powsybl.security.monitor.StateMonitor> monitors = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(), allVls, java.util.Set.of()));
        com.powsybl.commons.report.ReportNode cpuReport = "1".equals(System.getenv("OLF_GPU_DBG_REPORT"))
                ? com.powsybl.commons.report.ReportNode.newRootReportNode()
                        .withResourceBundles(com.powsybl.openloadflow.util.report.PowsyblOpenLoadFlowReportResourceBundle.BASE_NAME)
                        .withMessageTemplate("olf.saReport").build()
                : com.powsybl.commons.report.ReportNode.NO_OP;
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, contingencies, monitors, sap, strategies, actions,
                cpuReport);
        if (cpuReport != com.powsybl.commons.report.ReportNode.NO_OP) {
            printReport(cpuReport, 0);
        }
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            SecurityAnalysisResult gpu = runSecurityAnalysis(network, contingencies, monitors, sap, strategies, actions,
                    com.powsybl.commons.report.ReportNode.NO_OP);
            assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                    "contingency AND the (ctg, strategy) pair must both precompute on the device");
            assertEquals(1, gpu.getOperatorStrategyResults().size());
            var cpuS = cpu.getOperatorStrategyResults().get(0).getConditionalActionsResults().get(0);
            var gpuS = gpu.getOperatorStrategyResults().get(0).getConditionalActionsResults().get(0);
            assertEquals(cpuS.getStatus(), gpuS.getStatus());
            assertEquals(cpuS.getLimitViolationsResult().getLimitViolations().size(),
                    gpuS.getLimitViolationsResult().getLimitViolations().size(), "strategy violation count");
            assertEquals(cpuS.getDistributedActivePower(), gpuS.getDistributedActivePower(), 1e-9,
                    "strategy distributed active power");
            var cpuBuses = cpuS.getNetworkResult().getBusResults();
            var gpuBuses = gpuS.getNetworkResult().getBusResults();
            assertEquals(cpuBuses.size(), gpuBuses.size());
            for (int b = 0; b < cpuBuses.size(); b++) {
                org.junit.jupiter.api.Assertions.assertEquals(cpuBuses.get(b).getV(), gpuBuses.get(b).getV(), 1e-12,
                        "strategy bus voltage " + cpuBuses.get(b).getBusId());
            }
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    @Test
    void tapPositionActionStrategyBatchMatchesCpu() {
        // #24 stage 2b: RatioTapChanger tap action on a FIXED-tap transformer = per-scenario r1
        // override on the packed branch inputs.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
        com.powsybl.iidm.network.TwoWindingsTransformer t2wt = network.getTwoWindingsTransformer("T2wT");
        t2wt.getRatioTapChanger().setTapPosition(0).setRegulating(false);
        com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
        network.newLine().setId(l0.getId() + "_BIS")
                .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(0).setB1(0).setG2(0).setB2(0).add();
        List<Contingency> contingencies = List.of(new Contingency(l0.getId() + "_BIS", new BranchContingency(l0.getId() + "_BIS")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.RatioTapChangerTapPositionAction("tapUp", "T2wT", false, 2));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sTap",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("tapUp")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    @Test
    void pstTapPositionActionStrategyBatchMatchesCpu() {
        // #24 stage 2b: PhaseTapChanger tap action on a FIXED-tap PST = per-scenario (r1, a1)
        // override; the overridden a1 forces the raw-sincos trig path (statics are per-element).
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.PhaseControlFactory.createNetworkWithT2wt();
        List<Contingency> contingencies = List.of(new Contingency("L1", new BranchContingency("L1")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.PhaseTapChangerTapPositionAction("pstUp", "PS1", false, 2));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sPst",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("pstUp")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    @Test
    void pstTapActionWithPerStepImpedanceBatchMatchesCpu() {
        // Hard full-row-override case: the PST's steps change r/x/g/b by up to +-100% per tap
        // (y and ksi move, not just rho/alpha) — unrepresentable by an r1/a1-only override.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.PhaseControlFactory.createNetworkWithT2wtExaggerateRxgbTable();
        List<Contingency> contingencies = List.of(new Contingency("L1", new BranchContingency("L1")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.PhaseTapChangerTapPositionAction("pstShift", "PS1", false, 2));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sPstRxgb",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("pstShift")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    @Test
    void pstTapActionOnDisconnectedTransformerBatchMatchesCpu() {
        // A tap action on a transformer the LfNetwork does not hold (open at both ends: no LfBranch) is invalid
        // in OLF (AbstractLfTapChangerAction: branch null, apply does nothing) and the strategy re-solves its
        // contingency's state unchanged. The pair is that re-solve on the device, not a CPU strategy (network.xiidm:
        // 3 of the 47 phase shifters of a synthetic PST strategy case, 10 % of its strategies).
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.PhaseControlFactory.createNetworkWithT2wt();
        com.powsybl.iidm.network.TwoWindingsTransformer ps2 = network.getSubstation("S1").newTwoWindingsTransformer()
                .setId("PS2").setVoltageLevel1("VL1").setConnectableBus1("B1")
                .setVoltageLevel2("VL3").setConnectableBus2("B3")      // no bus on either side: open at both ends
                .setRatedU1(380.0).setRatedU2(380.0).setR(2.0).setX(100.0).setG(0.0).setB(0.0).add();
        com.powsybl.iidm.network.PhaseTapChangerAdder adder = ps2.newPhaseTapChanger().setTapPosition(1)
                .setRegulationMode(com.powsybl.iidm.network.PhaseTapChanger.RegulationMode.CURRENT_LIMITER).setRegulationValue(200)
                .setRegulating(false).setRegulationTerminal(ps2.getTerminal1());
        for (double alpha : new double[] {-20.0, 0.0, 20.0}) {
            adder.beginStep().setAlpha(alpha).setRho(1.0).setR(0.0).setX(0.0).setG(0.0).setB(0.0).endStep();
        }
        adder.add();
        List<Contingency> contingencies = List.of(new Contingency("L1", new BranchContingency("L1")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.PhaseTapChangerTapPositionAction("ps2Up", "PS2", false, 2));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sPs2",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("ps2Up")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    @Test
    void pstTapActionViolationOnTheRetappedTransformerMatchesCpu() {
        // The strategy's violations come from the device screen's candidates, and the screen reads every branch with
        // its BASE parameters: a retapped transformer's own flow is not what the screen evaluates. The tap moves PS1's
        // current over a limit set between its post-contingency current and its post-action one; that violation is
        // only found if the retapped branch is scanned in full on the host.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.PhaseControlFactory.createNetworkWithT2wt();
        com.powsybl.iidm.network.Line l1 = network.getLine("L1");
        network.newLine().setId("L1_BIS")
                .setVoltageLevel1(l1.getTerminal1().getVoltageLevel().getId()).setBus1(l1.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l1.getTerminal2().getVoltageLevel().getId()).setBus2(l1.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l1.getR()).setX(l1.getX()).setG1(0).setB1(0).setG2(0).setB2(0).add();
        com.powsybl.iidm.network.TwoWindingsTransformer ps1 = network.getTwoWindingsTransformer("PS1");
        // PS1's current after the contingency, at its own tap and at each other tap (OLF load flows on the network itself)
        int baseTap = ps1.getPhaseTapChanger().getTapPosition();
        network.getLine("L1_BIS").disconnect();
        java.util.Map<Integer, Double> currentByTap = new java.util.TreeMap<>();
        for (int tap = ps1.getPhaseTapChanger().getLowTapPosition(); tap <= ps1.getPhaseTapChanger().getHighTapPosition(); tap++) {
            ps1.getPhaseTapChanger().setTapPosition(tap);
            new com.powsybl.loadflow.LoadFlow.Runner(new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory())).run(network);
            currentByTap.put(tap, ps1.getTerminal1().getI());
        }
        ps1.getPhaseTapChanger().setTapPosition(baseTap);
        network.getLine("L1_BIS").connect();
        int actionTap = currentByTap.entrySet().stream().max(java.util.Map.Entry.comparingByValue()).orElseThrow().getKey();
        double before = currentByTap.get(baseTap);
        double after = currentByTap.get(actionTap);
        org.junit.jupiter.api.Assertions.assertTrue(after > 1.2 * before, "the fixture needs a tap that raises PS1's current: " + currentByTap);
        ps1.newCurrentLimits1().setPermanentLimit((before + after) / 2).add();
        List<Contingency> contingencies = List.of(new Contingency("L1_BIS", new BranchContingency("L1_BIS")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.PhaseTapChangerTapPositionAction("ps1Over", "PS1", false, actionTap));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sPs1Over",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("ps1Over")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    @Test
    void activePowerViolationAfterAStrategyReadsTheStrategyFlows() {
        // A device-answered strategy refreshes only the flows its detection reads: the branches the screen flagged for
        // the pair among them. An active power limit reads the branch's cached flow (unlike a current, evaluated from the
        // state): L1 carries more after the tap moves flow off PS1's path, over a limit set between its post-contingency
        // flow and its post-action one, and that violation is found only if L1's flow is refreshed for the pair.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.PhaseControlFactory.createNetworkWithT2wt();
        com.powsybl.iidm.network.Line l1 = network.getLine("L1");
        network.newLine().setId("L1_BIS")
                .setVoltageLevel1(l1.getTerminal1().getVoltageLevel().getId()).setBus1(l1.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l1.getTerminal2().getVoltageLevel().getId()).setBus2(l1.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l1.getR()).setX(l1.getX()).setG1(0).setB1(0).setG2(0).setB2(0).add();
        com.powsybl.iidm.network.TwoWindingsTransformer ps1 = network.getTwoWindingsTransformer("PS1");
        // L1's active power after the contingency, at PS1's own tap and at each other tap
        int baseTap = ps1.getPhaseTapChanger().getTapPosition();
        network.getLine("L1_BIS").disconnect();
        java.util.Map<Integer, Double> flowByTap = new java.util.TreeMap<>();
        for (int tap = ps1.getPhaseTapChanger().getLowTapPosition(); tap <= ps1.getPhaseTapChanger().getHighTapPosition(); tap++) {
            ps1.getPhaseTapChanger().setTapPosition(tap);
            new com.powsybl.loadflow.LoadFlow.Runner(new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory())).run(network);
            flowByTap.put(tap, Math.abs(l1.getTerminal1().getP()));
        }
        ps1.getPhaseTapChanger().setTapPosition(baseTap);
        network.getLine("L1_BIS").connect();
        int actionTap = flowByTap.entrySet().stream().max(java.util.Map.Entry.comparingByValue()).orElseThrow().getKey();
        double before = flowByTap.get(baseTap);
        double after = flowByTap.get(actionTap);
        org.junit.jupiter.api.Assertions.assertTrue(after > 1.2 * before, "the fixture needs a tap that raises L1's flow: " + flowByTap);
        l1.newActivePowerLimits1().setPermanentLimit((before + after) / 2).add();
        List<Contingency> contingencies = List.of(new Contingency("L1_BIS", new BranchContingency("L1_BIS")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.PhaseTapChangerTapPositionAction("ps1Shift", "PS1", false, actionTap));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sPs1Shift",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("ps1Shift")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    @Test
    void thresholdConditionAfterADeviceStrategyReadsThatStrategysFlows() {
        // OLF evaluates a strategy's condition on the network the previous strategy of the contingency left: the restore
        // between them puts the network's elements back, not the state its load flow solved, so a branch threshold reads
        // the previous strategy's flows. A device-answered strategy refreshes only the flows its own detection reads, and
        // L1 is not among them here (a limit far above its flow): the condition read L1's post-contingency flow. The
        // first strategy moves PS1's tap, which raises L1's flow over the second one's threshold.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.PhaseControlFactory.createNetworkWithT2wt();
        com.powsybl.iidm.network.Line l1 = network.getLine("L1");
        network.newLine().setId("L1_BIS")
                .setVoltageLevel1(l1.getTerminal1().getVoltageLevel().getId()).setBus1(l1.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l1.getTerminal2().getVoltageLevel().getId()).setBus2(l1.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l1.getR()).setX(l1.getX()).setG1(0).setB1(0).setG2(0).setB2(0).add();
        com.powsybl.iidm.network.PhaseTapChanger ptc = network.getTwoWindingsTransformer("PS1").getPhaseTapChanger();
        // L1's active power after the contingency, at PS1's own tap and at each other tap
        int baseTap = ptc.getTapPosition();
        network.getLine("L1_BIS").disconnect();
        java.util.Map<Integer, Double> flowByTap = new java.util.TreeMap<>();
        for (int tap = ptc.getLowTapPosition(); tap <= ptc.getHighTapPosition(); tap++) {
            ptc.setTapPosition(tap);
            new com.powsybl.loadflow.LoadFlow.Runner(new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory())).run(network);
            flowByTap.put(tap, l1.getTerminal1().getP());
        }
        ptc.setTapPosition(baseTap);
        network.getLine("L1_BIS").connect();
        int actionTap = flowByTap.entrySet().stream().max(java.util.Map.Entry.comparingByValue()).orElseThrow().getKey();
        double before = flowByTap.get(baseTap);
        double after = flowByTap.get(actionTap);
        org.junit.jupiter.api.Assertions.assertTrue(after > before + 10, "the fixture needs a tap that raises L1's flow: " + flowByTap);
        int otherTap = flowByTap.keySet().stream().filter(t -> t != baseTap && t != actionTap).findFirst().orElseThrow();
        l1.newActivePowerLimits1().setPermanentLimit(10 * Math.max(Math.abs(before), Math.abs(after))).add();
        List<Contingency> contingencies = List.of(new Contingency("L1_BIS", new BranchContingency("L1_BIS")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.PhaseTapChangerTapPositionAction("ps1Raise", "PS1", false, actionTap),
                new com.powsybl.action.PhaseTapChangerTapPositionAction("ps1Other", "PS1", false, otherTap));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sRaise",
                        com.powsybl.contingency.ContingencyContext.specificContingency("L1_BIS"),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("ps1Raise")),
                new com.powsybl.contingency.strategy.OperatorStrategy("sOnL1Flow",
                        com.powsybl.contingency.ContingencyContext.specificContingency("L1_BIS"),
                        new com.powsybl.contingency.strategy.condition.BranchThresholdCondition("L1",
                                com.powsybl.contingency.strategy.condition.AbstractThresholdCondition.Variable.ACTIVE_POWER,
                                com.powsybl.contingency.strategy.condition.AbstractThresholdCondition.ComparisonType.GREATER_THAN,
                                (before + after) / 2, com.powsybl.iidm.network.TwoSides.ONE),
                        List.of("ps1Other")));
        // the contingency and both pairs on the device (the conditional one speculatively)
        SecurityAnalysisResult cpu = assertStrategyMatchesCpu(network, contingencies, strategies, actions, 3);
        assertEquals(2, cpu.getOperatorStrategyResults().size(), "OLF runs the conditional strategy on the first one's flows");
    }

    @Test
    void strategyAfterADeviceTapStrategyStartsFromThePostContingencyState() {
        // Between two strategies of a contingency OLF restores the post-contingency state; after a device-answered
        // strategy with tap actions only, the GPU security analysis restores the retapped branches (and lazily the
        // voltages) instead of the whole network. The second strategy here runs OLF's own load flow (its tap action is
        // on a transformer open at one end: a half-open branch, kept on the CPU), which starts from the network as it
        // stands: it is right only if the first strategy's tap was put back.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.PhaseControlFactory.createNetworkWithT2wt();
        com.powsybl.iidm.network.Line l1 = network.getLine("L1");
        network.newLine().setId("L1_BIS")
                .setVoltageLevel1(l1.getTerminal1().getVoltageLevel().getId()).setBus1(l1.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l1.getTerminal2().getVoltageLevel().getId()).setBus2(l1.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l1.getR()).setX(l1.getX()).setG1(0).setB1(0).setG2(0).setB2(0).add();
        com.powsybl.iidm.network.TwoWindingsTransformer ps2 = network.getSubstation("S1").newTwoWindingsTransformer()
                .setId("PS2").setVoltageLevel1("VL1").setConnectableBus1("B1").setBus1("B1")
                .setVoltageLevel2("VL3").setConnectableBus2("B3")      // open at side 2: a half-open branch
                .setRatedU1(380.0).setRatedU2(380.0).setR(2.0).setX(100.0).setG(0.0).setB(0.0).add();
        com.powsybl.iidm.network.PhaseTapChangerAdder adder = ps2.newPhaseTapChanger().setTapPosition(1)
                .setRegulationMode(com.powsybl.iidm.network.PhaseTapChanger.RegulationMode.CURRENT_LIMITER).setRegulationValue(200)
                .setRegulating(false).setRegulationTerminal(ps2.getTerminal1());
        for (double alpha : new double[] {-20.0, 0.0, 20.0}) {
            adder.beginStep().setAlpha(alpha).setRho(1.0).setR(0.0).setX(0.0).setG(0.0).setB(0.0).endStep();
        }
        adder.add();
        int ps1Tap = network.getTwoWindingsTransformer("PS1").getPhaseTapChanger().getTapPosition();
        int ps1Other = ps1Tap == network.getTwoWindingsTransformer("PS1").getPhaseTapChanger().getHighTapPosition()
                ? ps1Tap - 1 : ps1Tap + 1;
        List<Contingency> contingencies = List.of(new Contingency("L1_BIS", new BranchContingency("L1_BIS")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.PhaseTapChangerTapPositionAction("ps1Move", "PS1", false, ps1Other),
                new com.powsybl.action.PhaseTapChangerTapPositionAction("ps2Move", "PS2", false, 2));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sPs1Device",
                        com.powsybl.contingency.ContingencyContext.specificContingency("L1_BIS"),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("ps1Move")),
                new com.powsybl.contingency.strategy.OperatorStrategy("sPs2Cpu",
                        com.powsybl.contingency.ContingencyContext.specificContingency("L1_BIS"),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("ps2Move")));
        // the contingency and the PS1 pair on the device; the PS2 strategy on the CPU
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_TARGETED_STRATEGY_RESTORE_COUNT.get(),
                "the first strategy's restore must be the targeted one");
    }

    @Test
    void baseOpenShuntReconnectionStrategyBatchMatchesCpu() {
        // #32: reconnecting a BASE-OPEN shunt via TerminalsConnectionAction — supported by the
        // olf-local-patches upstream patch (shuntIdsToClose retention: the compensator joins the
        // aggregate at section 0, the action restores its section count). Batched as a per-scenario
        // admittance shift of the compensator's full section admittance.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        String shuntId = network.getShuntCompensatorStream().findFirst().orElseThrow().getId();
        network.getShuntCompensator(shuntId).getTerminal().disconnect();
        String lineId = network.getLineStream()
                .filter(l -> !l.getId().contains("7") && !l.getId().contains("8"))
                .skip(4).findFirst().orElseThrow().getId();
        List<Contingency> contingencies = List.of(new Contingency(lineId, new BranchContingency(lineId)));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.TerminalsConnectionAction("reconnectShunt", shuntId, false));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sShuntRe",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("reconnectShunt")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    @Test
    void baseOpenSwitchCloseStrategyBatchMatchesCpu() {
        // #31: SwitchAction CLOSING a base-open busbar coupler (zero-impedance branch). The retained
        // switch's dummy variables already have rows (identities held at 0 while open), so the close
        // is a per-scenario zero-impedance-element enable + identity->coupling row flip on the device. Node-breaker
        // network: coupler C open at base splits VL1 into BBS1 (lines L1+L3) and BBS2 (G + L2); the
        // L1 outage keeps both sides energized, the strategy merges them back.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = coupledSectionsWithThirdLine();
        network.getSwitch("C").setOpen(true);
        List<Contingency> contingencies = List.of(new Contingency("L1", new BranchContingency("L1")));
        List<com.powsybl.action.Action> actions = List.of(new com.powsybl.action.SwitchAction("closeC", "C", false));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sCloseC",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("closeC")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    @Test
    void closedCouplerOpenStrategyBatchMatchesCpu() {
        // SwitchAction OPENING the closed coupler C after the L1 outage: BBS1 keeps L3, BBS2 keeps G and L2, both stay
        // energized through VL2. The pair solves the contingency's network with the coupler out of its zero-impedance
        // tree; composing nothing for the open solved the contingency alone.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = coupledSectionsWithThirdLine();
        List<Contingency> contingencies = List.of(new Contingency("L1", new BranchContingency("L1")));
        List<com.powsybl.action.Action> actions = List.of(new com.powsybl.action.SwitchAction("openC", "C", true));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sOpenC",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("openC")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    @Test
    void lineBreakerOpenStrategyBatchMatchesCpu() {
        // SwitchAction OPENING the breaker B5 between BBS1 and L3 after the L1 outage: L3's VL1 end is left on a bus of
        // its own, which leaves the main component with the line, as a busbar outage's dead stub does.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = coupledSectionsWithThirdLine();
        List<Contingency> contingencies = List.of(new Contingency("L1", new BranchContingency("L1")));
        List<com.powsybl.action.Action> actions = List.of(new com.powsybl.action.SwitchAction("openB5", "B5", true));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sOpenB5",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("openB5")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    /** OLF's node-breaker network with a third line L3 from BBS1 (breaker B5) to BBS3 (breaker B6): with the coupler C
     *  open, VL1 splits into BBS1 (L1 and L3) and BBS2 (G and L2). */
    private static Network coupledSectionsWithThirdLine() {
        Network network = com.powsybl.openloadflow.network.NodeBreakerNetworkFactory.create();
        var vl1 = network.getVoltageLevel("VL1");
        var vl2 = network.getVoltageLevel("VL2");
        vl1.getNodeBreakerView().newBreaker().setId("B5").setNode1(0).setNode2(7).add();
        vl2.getNodeBreakerView().newBreaker().setId("B6").setNode1(0).setNode2(4).add();
        network.newLine().setId("L3").setVoltageLevel1("VL1").setNode1(7).setVoltageLevel2("VL2").setNode2(4)
                .setR(3).setX(33).setG1(0).setB1(0).setG2(0).setB2(0).add();
        return network;
    }

    @Test
    void baseOpenBranchRecloseStrategyBatchMatchesCpu() {
        // #29: TerminalsConnectionAction CLOSING a BASE-OPEN branch. The SA retains it via
        // LfTopoConfig (loaded closed in a temp variant, re-opened by connectivity), so the LfBranch
        // exists disabled with both buses — packed as a base-open element, default-disabled in every
        // scenario's mask, and ENABLED only in the (ctg, strategy) pair that recloses it.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = distributedSlackNetworkWithExtras();
        com.powsybl.iidm.network.Line l24 = network.getLine("l24");
        com.powsybl.iidm.network.Line ter = network.newLine().setId("l24_ter")
                .setVoltageLevel1(l24.getTerminal1().getVoltageLevel().getId()).setBus1(l24.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l24.getTerminal2().getVoltageLevel().getId()).setBus2(l24.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l24.getR() * 3).setX(l24.getX() * 3).setG1(0).setB1(0).setG2(0).setB2(0).add();
        ter.getTerminal1().disconnect();
        ter.getTerminal2().disconnect();
        List<Contingency> contingencies = List.of(new Contingency("l24_bis", new BranchContingency("l24_bis")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.TerminalsConnectionAction("closeTer", "l24_ter", false));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sReclose",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("closeTer")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    @Test
    void generatorActionStrategyBatchMatchesCpu() {
        // #28: pure active-power redispatch GeneratorAction — composed as a net-injection delta on
        // the warm-continued pair. The B2 saga's ~0.17 kV freeze-semantics delta must NOT reappear.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        String lineId = network.getLineStream().skip(4).findFirst().orElseThrow().getId();
        List<Contingency> contingencies = List.of(new Contingency(lineId, new BranchContingency(lineId)));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.GeneratorActionBuilder().withId("redispatch2")
                        .withGeneratorId("B2-G").withActivePowerRelativeValue(true).withActivePowerValue(20).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sGen",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("redispatch2")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    @Test
    void loadActionReactivePowerStrategyBatchMatchesCpu() {
        // OLF's LfLoadAction shifts the load's P AND Q. The composer used to keep P only, and sent a
        // reactive-only action to the CPU: the pair's bus voltages followed a load that kept its Q.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        String lineId = network.getLineStream().skip(4).findFirst().orElseThrow().getId();
        com.powsybl.iidm.network.Load load12 = loadAtBus(network, "B12");
        com.powsybl.iidm.network.Load load14 = loadAtBus(network, "B14");
        List<Contingency> contingencies = List.of(new Contingency(lineId, new BranchContingency(lineId)));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.LoadActionBuilder().withId("halve12").withLoadId(load12.getId())
                        .withRelativeValue(false).withActivePowerValue(load12.getP0() / 2)
                        .withReactivePowerValue(load12.getQ0() / 2).build(),
                new com.powsybl.action.LoadActionBuilder().withId("q14").withLoadId(load14.getId())
                        .withRelativeValue(true).withReactivePowerValue(8).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sPQ",
                        com.powsybl.contingency.ContingencyContext.specificContingency(lineId),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("halve12")),
                new com.powsybl.contingency.strategy.OperatorStrategy("sQ",
                        com.powsybl.contingency.ContingencyContext.specificContingency(lineId),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("q14")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 3);
    }

    /** A strategy whose state leaves the realistic band FAILS like OLF's. 20 MVar more at B14's load after the L2-5-1
     *  outage take B14 to 0.974 pu, while the base and the contingency stay at 1.01 pu or above: with a minimum
     *  realistic voltage of 1.0 pu, OLF's load flow ends UNREALISTIC_STATE, which the security analysis reports FAILED,
     *  without violations or network results. The host builds a device-answered pair's result as CONVERGED, so the pair
     *  must never come back answered: the device checks the band where OLF does (at the end of the last loop able to
     *  fix the state, then after every later solve) and rejects it, and the CPU replays it. Only the contingency is
     *  answered on the device. */
    @Test
    void strategyWithAnUnrealisticStateFailsLikeCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        String lineId = network.getLineStream().skip(4).findFirst().orElseThrow().getId();
        com.powsybl.iidm.network.Load load14 = loadAtBus(network, "B14");
        List<Contingency> contingencies = List.of(new Contingency(lineId, new BranchContingency(lineId)));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.LoadActionBuilder().withId("q14").withLoadId(load14.getId())
                        .withRelativeValue(true).withReactivePowerValue(20).build());
        SecurityAnalysisResult cpu = assertStrategyMatchesCpu(network, contingencies, strategiesOf(lineId, "q14"), actions,
                1, true, p -> p.setMinNominalVoltageRealisticVoltageCheck(0).setMinRealisticVoltage(1.0));
        assertEquals(com.powsybl.security.PostContingencyComputationStatus.CONVERGED,
                cpu.getPostContingencyResults().get(0).getStatus(), "the contingency alone stays realistic");
        assertEquals(com.powsybl.security.PostContingencyComputationStatus.FAILED,
                cpu.getOperatorStrategyResults().get(0).getConditionalActionsResults().get(0).getStatus(),
                "the strategy's state is unrealistic");
    }

    @Test
    void loadActionOnTheLostLoadStrategyBatchMatchesCpu() {
        // A load action on the load its contingency loses does nothing in OLF (LfLoadAction.apply checks
        // isOriginalLoadDisabled): composing its shift on top of the outage would move the bus again.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        com.powsybl.iidm.network.Load load12 = loadAtBus(network, "B12");
        List<Contingency> contingencies = List.of(new Contingency(load12.getId(),
                new com.powsybl.contingency.LoadContingency(load12.getId())));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.LoadActionBuilder().withId("halve12").withLoadId(load12.getId())
                        .withRelativeValue(false).withActivePowerValue(load12.getP0() / 2)
                        .withReactivePowerValue(load12.getQ0() / 2).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sLost",
                        com.powsybl.contingency.ContingencyContext.specificContingency(load12.getId()),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("halve12")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    @Test
    void absoluteGeneratorActionAfterADistributingContingencyStrategyBatchMatchesCpu() {
        // OLF's LfGeneratorAction SETS targetP to an absolute value, starting from the parent contingency's
        // distributed targets: the generator ends at the value, whatever share of the lost load it took.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        String loadId = loadAtBus(network, "B9").getId();
        List<Contingency> contingencies = List.of(new Contingency(loadId,
                new com.powsybl.contingency.LoadContingency(loadId)));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.GeneratorActionBuilder().withId("set2")
                        .withGeneratorId("B2-G").withActivePowerRelativeValue(false).withActivePowerValue(60).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sSet",
                        com.powsybl.contingency.ContingencyContext.specificContingency(loadId),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("set2")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);

        // A second participant on B2: how the parent's share of the bus splits between the two is not known to the
        // pair leg, so the pair goes to the CPU (the contingency alone on the device).
        Network shared = IeeeCdfNetworkFactory.create14();
        shared.getVoltageLevel("VL2").newGenerator().setId("B2-G2").setBus("B2").setConnectableBus("B2")
                .setTargetP(10).setTargetQ(0).setVoltageRegulatorOn(false).setMinP(0).setMaxP(50).add();
        assertStrategyMatchesCpu(shared, contingencies, strategies, actions, 1);
    }

    /** Strategy pairs on a network with capability curves. OLF starts a strategy leg from each generator's
     *  post-contingency target (setGeneratorsInitialTargetPToTargetP), and a GeneratorAction sets its generator's
     *  target and initial target; the curve limits are read at those targets. B3 holds two participants: B3-G
     *  controls B3's voltage with a curve that puts B3 at its max Q, and B3-G2 hits its min P during the parent's
     *  distribution, so B3's move does not split between them by their factors. The pair's curve generator sits at
     *  its OWN parent target (plus the action), which a reconstruction from B3's row target cannot give. */
    @Test
    void strategyPairsOnCurveGeneratorsMatchCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        com.powsybl.iidm.network.Generator g3 = network.getGenerator("B3-G");
        g3.setTargetP(40).setMinP(0).setMaxP(200);
        g3.newReactiveCapabilityCurve()
                .beginPoint().setP(0).setMinQ(-40).setMaxQ(5).endPoint()
                .beginPoint().setP(200).setMinQ(-40).setMaxQ(-45).endPoint()
                .add();
        // regulator off: a constant targetQ in B3's box, low enough that B3 sits at its max Q
        network.getVoltageLevel("VL3").newGenerator().setId("B3-G2").setBus("B3").setConnectableBus("B3")
                .setTargetP(60).setMinP(59.9).setMaxP(80).setTargetQ(-10).setVoltageRegulatorOn(false).add();
        com.powsybl.iidm.network.Generator g1 = network.getGenerator("B1-G");
        g1.setTargetP(g1.getTargetP() - 100);
        g1.newMinMaxReactiveLimits().setMinQ(-990).setMaxQ(990).add();
        network.getGenerator("B2-G").setMinP(0).setMaxP(200);
        network.getGenerator("B6-G").setMinP(0).setMaxP(100);
        network.getGenerator("B8-G").setMinP(0).setMaxP(100);
        // the slack's generator sits out, as on rte6515 and network.xiidm (a participating slack whose peers
        // saturate: participatingSlackWithASaturatingPeerMatchesCpu)
        g1.newExtension(com.powsybl.iidm.network.extensions.ActivePowerControlAdder.class)
                .withParticipate(false).add();
        String loadId = loadAtBus(network, "B4").getId();
        List<Contingency> contingencies = List.of(new Contingency(loadId,
                new com.powsybl.contingency.LoadContingency(loadId)));
        assertGpuMatchesCpu(network, contingencies);        // the parents first: a pair starts where its parent ends
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.GeneratorActionBuilder().withId("up3")
                        .withGeneratorId("B3-G").withActivePowerRelativeValue(true).withActivePowerValue(20).build(),
                new com.powsybl.action.TerminalsConnectionActionBuilder().withId("open1011")
                        .withNetworkElementId("L10-11-1").withOpen(true).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sUp3",
                        com.powsybl.contingency.ContingencyContext.specificContingency(loadId),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("up3")),
                new com.powsybl.contingency.strategy.OperatorStrategy("sOpen",
                        com.powsybl.contingency.ContingencyContext.specificContingency(loadId),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("open1011")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 3);
    }

    /** A GeneratorAction on a voltage-controlling curve generator that does not take part in the slack distribution:
     *  the device knows the P of the generators in its slack pack only, so this one's limits would stay at its base
     *  P. B6-G (P 0) sits at its curve's max Q and the action moves it 10 MW, which lowers the max by 2.5 MVAr: the
     *  pair goes to the CPU, the contingency stays on the device. */
    @Test
    void generatorActionOnAnUntrackedCurveGeneratorRunsOnCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        com.powsybl.iidm.network.Generator g6 = network.getGenerator("B6-G");
        g6.setMinP(0).setMaxP(100);
        g6.newReactiveCapabilityCurve()
                .beginPoint().setP(0).setMinQ(-6).setMaxQ(5).endPoint()
                .beginPoint().setP(100).setMinQ(-6).setMaxQ(-20).endPoint()
                .add();
        g6.newExtension(com.powsybl.iidm.network.extensions.ActivePowerControlAdder.class).withParticipate(false).add();
        String loadId = loadAtBus(network, "B9").getId();
        List<Contingency> contingencies = List.of(new Contingency(loadId,
                new com.powsybl.contingency.LoadContingency(loadId)));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.GeneratorActionBuilder().withId("up6")
                        .withGeneratorId("B6-G").withActivePowerRelativeValue(true).withActivePowerValue(10).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sUp6",
                        com.powsybl.contingency.ContingencyContext.specificContingency(loadId),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("up6")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 1);
    }

    @Test
    void generatorActionOnTheLostGeneratorStrategyBatchMatchesCpu() {
        // A generator action on the generator its contingency loses does nothing in OLF
        // (LfGeneratorAction.apply checks isDisabled).
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        List<Contingency> contingencies = List.of(new Contingency("B3-G",
                new com.powsybl.contingency.GeneratorContingency("B3-G")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.GeneratorActionBuilder().withId("up3")
                        .withGeneratorId("B3-G").withActivePowerRelativeValue(true).withActivePowerValue(20).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sLostGen",
                        com.powsybl.contingency.ContingencyContext.specificContingency("B3-G"),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("up3")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    /** OLF runs a contingency's strategies even when the contingency's own load flow fails, from what that run left:
     *  the generators' targets (here the loss pre-distribution of B4's load, which become the strategy's initial
     *  targets), the bus types its reactive-limit checks switched, and the voltages of its last converged solve (the
     *  base ones when none converged). L1-2-1 with B4's load lost runs out of Newton iterations on its first solve,
     *  on both arms; reclosing L1-2-1 converges. */
    @Test
    void strategyOfAContingencyThatRanOutOfIterationsMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        for (String g : List.of("B2-G", "B3-G", "B6-G", "B8-G")) {
            network.getGenerator(g).newMinMaxReactiveLimits().setMinQ(-6).setMaxQ(12).add();
        }
        String loadId = loadAtBus(network, "B4").getId();
        String ctgId = "L1-2-1+" + loadId;
        List<Contingency> contingencies = List.of(new Contingency(ctgId,
                List.of(new BranchContingency("L1-2-1"), new com.powsybl.contingency.LoadContingency(loadId))));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.TerminalsConnectionActionBuilder().withId("close")
                        .withNetworkElementId("L1-2-1").withOpen(false).build());
        // the pair on the device, the contingency reported from the device
        assertStrategyMatchesCpu(network, contingencies, strategiesOf(ctgId, "close"), actions, 1, true,
                p -> p.setMaxNewtonRaphsonIterations(8));
        // ...and with a second strategy the device does not compose (four conditional stages), which OLF starts from
        // its own failed run of the contingency: that run has to happen
        List<com.powsybl.action.Action> actions2 = List.of(actions.get(0),
                new com.powsybl.action.GeneratorActionBuilder().withId("closeAndUp2").withGeneratorId("B2-G")
                        .withActivePowerRelativeValue(true).withActivePowerValue(10).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies2 = new java.util.ArrayList<>(strategiesOf(ctgId, "close"));
        strategies2.add(fourConditionalStageStrategy("s-closeAndUp2", ctgId, "close", "closeAndUp2"));
        assertStrategyMatchesCpu(network, contingencies, strategies2, actions2, 1, true,
                p -> p.setMaxNewtonRaphsonIterations(8));
        // A contingency that CONVERGES four times (reactive-limit switches in between), then runs out of iterations:
        // its strategies start from the fourth solve's voltages, with the bus types and targets its outer loops left
        Network network2 = IeeeCdfNetworkFactory.create14();
        for (String g : List.of("B2-G", "B3-G", "B6-G", "B8-G")) {
            network2.getGenerator(g).newMinMaxReactiveLimits().setMinQ(-20).setMaxQ(20).add();
        }
        List<Contingency> contingencies2 = List.of(new Contingency("L1-2-1", new BranchContingency("L1-2-1")));
        assertStrategyMatchesCpu(network2, contingencies2, strategiesOf("L1-2-1", "close"), actions, 1, true,
                p -> p.setMaxNewtonRaphsonIterations(8));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies3 = new java.util.ArrayList<>(strategiesOf("L1-2-1", "close"));
        strategies3.add(fourConditionalStageStrategy("s-closeAndUp2", "L1-2-1", "close", "closeAndUp2"));
        assertStrategyMatchesCpu(network2, contingencies2, strategies3, actions2, 1, true,
                p -> p.setMaxNewtonRaphsonIterations(8));
    }

    /** A contingency the device hands back to the CPU: its strategy starts from what OLF's run of it leaves, which
     *  only that run has, so the strategy goes back too. B4's load loss saturates the only participant (B2-G, 2 MW
     *  above its minP) and the device does not keep a saturated distribution; the strategy adds the lost load back
     *  at B9, which a distribution from the base would not see the same way. */
    @Test
    void strategyOfAContingencyTheDeviceHandsBackRunsOnCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        network.getGenerator("B1-G").newExtension(com.powsybl.iidm.network.extensions.ActivePowerControlAdder.class)
                .withParticipate(false).add();
        network.getGenerator("B2-G").setMinP(38).setMaxP(200);
        String loadId = loadAtBus(network, "B4").getId();
        String load9 = loadAtBus(network, "B9").getId();
        List<Contingency> contingencies = List.of(new Contingency(loadId, new com.powsybl.contingency.LoadContingency(loadId)));
        List<com.powsybl.action.Action> actions = List.of(new com.powsybl.action.LoadActionBuilder().withId("more9")
                .withLoadId(load9).withRelativeValue(true).withActivePowerValue(47.8).build());
        assertStrategyMatchesCpu(network, contingencies, strategiesOf(loadId, "more9"), actions, 0);
    }

    /** A strategy in several stages: OLF evaluates every stage's condition on the post-contingency violations and
     *  applies the actions of the stages that hold, in stage order (none holding: no result). The device precomputes
     *  one pair per combination of the conditional stages and answers with the one OLF selects. ieee14 has no
     *  violation after B4's load loss, so AnyViolationCondition fails and an empty AllViolationCondition holds. */
    @Test
    void multiStageStrategiesMatchCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        String loadId = loadAtBus(network, "B4").getId();
        List<Contingency> contingencies = List.of(new Contingency(loadId, new com.powsybl.contingency.LoadContingency(loadId)));
        List<com.powsybl.action.Action> actions = List.of(
                genAction("up2", "B2-G", true, 10),
                new com.powsybl.action.TerminalsConnectionActionBuilder().withId("open1011")
                        .withNetworkElementId("L10-11-1").withOpen(true).build());
        var ctx = com.powsybl.contingency.ContingencyContext.specificContingency(loadId);
        var always = new com.powsybl.contingency.strategy.condition.TrueCondition();
        var anyViolation = new com.powsybl.contingency.strategy.condition.AnyViolationCondition();
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                // both stages always: up2 then open1011
                new com.powsybl.contingency.strategy.OperatorStrategy("sBoth", ctx,
                        List.of(stage("a", always, "up2"), stage("b", always, "open1011"))),
                // the second stage's condition fails: up2 alone
                new com.powsybl.contingency.strategy.OperatorStrategy("sFirstOnly", ctx,
                        List.of(stage("a", always, "up2"), stage("b", anyViolation, "open1011"))),
                // a conditional stage that holds, one that does not, and the same action twice
                new com.powsybl.contingency.strategy.OperatorStrategy("sMixed", ctx,
                        List.of(stage("a", alwaysHolds(), "open1011", "up2"), stage("b", anyViolation, "up2"),
                                stage("c", always, "up2"))),
                // nothing holds: no result on either side
                new com.powsybl.contingency.strategy.OperatorStrategy("sNone", ctx,
                        List.of(stage("a", anyViolation, "up2"), stage("b", anyViolation, "open1011"))));
        // the contingency, sBoth (1), sFirstOnly (2 combinations), sMixed (4), sNone (3)
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 11);
    }

    /** A strategy pair re-anchors its distribution on what its parent's moved, read off the parent's final targets.
     *  A participant the contingency islanded absorbed nothing: its row is a FROZEN identity (an angle), and read as a
     *  P it handed the pairs minus its whole target, which a reclose then composed onto the bus's live row (and a pair
     *  keeping the island onto its frozen one: rte6515 LINE-4468-670, 9 Newton iterations where OLF takes one, 0.007
     *  MW). ieee14 with a radial B15 generator behind line L14-15-1 taking most of any distribution: its island kept,
     *  reclosed, and created by a strategy after the loss of B2-G. */
    @Test
    void strategiesAroundAnIslandedDistributionParticipantMatchCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = ieee14WithRadialBus(vl -> vl.newGenerator().setId("B15-G").setBus("B15").setConnectableBus("B15")
                .setMinP(0).setMaxP(4000).setTargetP(40).setVoltageRegulatorOn(false).setTargetQ(0).add());
        // B15-G takes most of any distribution: its share of B2-G's loss is past the angle step clamp
        network.getGenerator("B1-G").setMaxP(400);
        for (String condenser : List.of("B3-G", "B6-G", "B8-G")) {
            network.getGenerator(condenser).newExtension(com.powsybl.iidm.network.extensions.ActivePowerControlAdder.class)
                    .withParticipate(false).withDroop(4).add();
        }
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.GeneratorActionBuilder().withId("up2").withGeneratorId("B2-G")
                        .withActivePowerRelativeValue(true).withActivePowerValue(10).build(),
                new com.powsybl.action.TerminalsConnectionActionBuilder().withId("close15").withNetworkElementId("L14-15-1")
                        .withOpen(false).build(),
                new com.powsybl.action.TerminalsConnectionActionBuilder().withId("open15").withNetworkElementId("L14-15-1")
                        .withOpen(true).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = new java.util.ArrayList<>(
                strategiesOf("L14-15-1", "up2", "close15"));
        strategies.addAll(strategiesOf("B2-G", "open15"));
        assertStrategyMatchesCpu(network, List.of(new Contingency("L14-15-1", new BranchContingency("L14-15-1")),
                        new Contingency("B2-G", new com.powsybl.contingency.GeneratorContingency("B2-G"))),
                strategies, actions, 5, true,
                p -> p.setStateVectorScalingMode(com.powsybl.openloadflow.ac.solver.StateVectorScalingMode.MAX_VOLTAGE_CHANGE)
                        .setMaxVoltageChangeStateVectorScalingMaxDphi(0.05));
    }

    /** Strategy pairs of a lost distribution participant with the slack out of the distribution: the pair starts from
     *  the parent's distribution with the slack keeping none of it. ieee14 with B1-G (on the slack bus) out of the
     *  distribution and B3-G, B6-G producing, losing B2-G; the pair then loads B9. */
    @Test
    void strategyAfterLosingADistributionParticipantMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        network.getGenerator("B1-G").newExtension(com.powsybl.iidm.network.extensions.ActivePowerControlAdder.class)
                .withParticipate(false).withDroop(4).add();
        network.getGenerator("B3-G").setMaxP(100).setTargetP(30);   // two participants that stay
        network.getGenerator("B6-G").setMaxP(150).setTargetP(30);
        List<com.powsybl.action.Action> actions = List.of(new com.powsybl.action.LoadActionBuilder().withId("more9")
                .withLoadId("B9-L").withRelativeValue(true).withActivePowerValue(10).build());
        assertStrategyMatchesCpu(network, List.of(new Contingency("B2-G", new com.powsybl.contingency.GeneratorContingency("B2-G"))),
                strategiesOf("B2-G", "more9"), actions, 2, true, p -> p.setSlackBusSelectionMode(
                        com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME).setSlackBusesIds(List.of("VL1_0")));
    }

    /** Reclosing the line whose loss islanded loads. OLF re-adds the reclosed branch to the post-contingency
     *  connectivity, from which the contingency also removed the island's own branches (both ends lost): the branch's
     *  end comes back, the rest of the island stays out. ieee14 with a radial chain B14 - B15 - B16, a load on each:
     *  losing L15-16-1 islands B16 alone, which its reclose brings back whole, on the device as in OLF (and a B16
     *  load the contingency names stays lost); losing L14-15-1 islands B15 and B16, and its reclose brings back B15
     *  only, where the device would reconnect both (rte6515 LINE-3273-1812: 154.7 MW, 3 kV): CPU. */
    @Test
    void recloseIntoAnIslandMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = ieee14WithRadialBus(vl -> {
            vl.newLoad().setId("B15-L").setBus("B15").setConnectableBus("B15").setP0(20).setQ0(5).add();
            vl.getBusBreakerView().newBus().setId("B16").add();
            vl.newLoad().setId("B16-L").setBus("B16").setConnectableBus("B16").setP0(15).setQ0(4).add();
        });
        network.newLine().setId("L15-16-1").setVoltageLevel1("VL15").setBus1("B15").setVoltageLevel2("VL15").setBus2("B16")
                .setR(0.05).setX(0.2).setG1(0).setB1(0).setG2(0).setB2(0).add();
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.TerminalsConnectionActionBuilder().withId("close15").withNetworkElementId("L14-15-1")
                        .withOpen(false).build(),
                new com.powsybl.action.TerminalsConnectionActionBuilder().withId("close16").withNetworkElementId("L15-16-1")
                        .withOpen(false).build());
        List<Contingency> contingencies = List.of(new Contingency("L15-16-1", new BranchContingency("L15-16-1")),
                new Contingency("L15-16-1+B16-L", List.of(new BranchContingency("L15-16-1"),
                        new com.powsybl.contingency.LoadContingency("B16-L"))),
                new Contingency("L14-15-1", new BranchContingency("L14-15-1")));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = new java.util.ArrayList<>(strategiesOf("L15-16-1", "close16"));
        strategies.add(new com.powsybl.contingency.strategy.OperatorStrategy("s-close16-named",
                com.powsybl.contingency.ContingencyContext.specificContingency("L15-16-1+B16-L"),
                new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("close16")));
        strategies.add(new com.powsybl.contingency.strategy.OperatorStrategy("s-close15",
                com.powsybl.contingency.ContingencyContext.specificContingency("L14-15-1"),
                new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("close15")));
        // the three contingencies and the two single-bus reclosures; the two-bus one is OLF's
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 5);
    }

    /** A synchronous condenser switched to PQ at its reactive limit in the base, islanded by the contingency. A pair keeping
     *  the island must leave its frozen rows alone: stamping the continued switch to PQ turned the islanded bus's V row into a
     *  reactive balance (rte6515 LINE-3477-2893, 0.0075 MW). A pair reclosing it re-energizes the bus where OLF does:
     *  from its stored voltage (not the frozen 1 pu at angle 0), PQ at its limit (not at the frozen row's value,
     *  which the export of the frozen target Q read as a reactive power). */
    @Test
    void strategiesAroundAnIslandedCondenserSwitchedToPqMatchCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = ieee14WithRadialBus(vl -> vl.newGenerator().setId("B15-G").setBus("B15").setConnectableBus("B15")
                .setMinP(0).setMaxP(50).setTargetP(0).setVoltageRegulatorOn(true).setTargetV(vl.getNominalV() * 1.08)
                .add().newMinMaxReactiveLimits().setMinQ(-1).setMaxQ(1).add());
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.GeneratorActionBuilder().withId("up2").withGeneratorId("B2-G")
                        .withActivePowerRelativeValue(true).withActivePowerValue(10).build(),
                new com.powsybl.action.TerminalsConnectionActionBuilder().withId("close").withNetworkElementId("L14-15-1")
                        .withOpen(false).build());
        assertStrategyMatchesCpu(network, List.of(new Contingency("L14-15-1", new BranchContingency("L14-15-1"))),
                strategiesOf("L14-15-1", "up2", "close"), actions, 3, true,
                p -> p.setStateVectorScalingMode(com.powsybl.openloadflow.ac.solver.StateVectorScalingMode.MAX_VOLTAGE_CHANGE));
    }

    /** A contingency with a device state can still be solved on OLF's own path: when one of its strategies is not
     *  answered on the device. It then starts from every bus's voltage, so the voltages the previous contingency's
     *  injection left must be back at the base first (rte6515 LINE-2349-694 after LINE-2348-693: 1.44 MW). */
    @Test
    void contingencyWithACpuStrategyStartsFromTheBaseMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = participationRecheckNetwork();
        String loadId = loadAtBus(network, "B4").getId();
        List<Contingency> contingencies = List.of(new Contingency("L1-2-1", new BranchContingency("L1-2-1")),
                new Contingency(loadId, new com.powsybl.contingency.LoadContingency(loadId)));
        List<com.powsybl.action.Action> actions = List.of(genAction("start8", "B8-GP", false, 20));
        assertStrategyMatchesCpu(network, contingencies, strategiesOf(loadId, "start8"), actions, 2);
    }

    /** A generator switched to PQ at its reactive limit in the base and released by the contingency: its strategy pairs
     *  start with it back on voltage control, as OLF's strategies do. The base activation still held it PQ, and a
     *  pair solved the generator at its base limit (rte6515 LINE-3273-1812: 0.016 kV for a strategy that changes
     *  nothing). B6-G is switched to PQ at 5 MVar in the base and released once B6's load is lost. */
    @Test
    void strategyAfterAContingencySwitchingBackToPvMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        network.getGenerator("B6-G").newMinMaxReactiveLimits().setMinQ(-6).setMaxQ(5).add();
        String loadId = loadAtBus(network, "B6").getId();
        List<com.powsybl.action.Action> actions = List.of(new com.powsybl.action.LoadActionBuilder().withId("more9")
                .withLoadId(loadAtBus(network, "B9").getId()).withRelativeValue(true).withActivePowerValue(5).build());
        assertStrategyMatchesCpu(network, List.of(new Contingency(loadId, new com.powsybl.contingency.LoadContingency(loadId))),
                strategiesOf(loadId, "more9"), actions, 2);
    }

    /** A bus switched to PQ keeps the reactive power it was frozen at until the limit moves by more than the reactive
     *  power mismatch: OLF's strategy starts from that frozen value, which the parent's distribution may leave a little
     *  off the curve. The pair stamped its frozen target Q at the curve instead (rte6515 VL-6095: frozen 0.7518298, the curve at
     *  0.7518918). B2-G sits on a sloped curve at its maximum, takes a small share of the contingency's distribution,
     *  and the strategy switches the B9 shunt off. */
    @Test
    void strategyKeepsTheParentsFrozenReactivePowerMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        network.getGenerator("B1-G").setMaxP(1000);
        com.powsybl.iidm.network.Generator b2 = network.getGenerator("B2-G");
        b2.setMaxP(60);
        b2.newReactiveCapabilityCurve()
                .beginPoint().setP(0).setMinQ(-40).setMaxQ(30).endPoint()
                .beginPoint().setP(100).setMinQ(-40).setMaxQ(25).endPoint()
                .add();
        String shuntId = network.getShuntCompensatorStream().findFirst().orElseThrow().getId();
        List<com.powsybl.action.Action> actions = List.of(new com.powsybl.action.ShuntCompensatorPositionActionBuilder()
                .withId("shuntOff").withShuntCompensatorId(shuntId).withSectionCount(0).build());
        assertStrategyMatchesCpu(network, List.of(new Contingency("L2-4-1", new BranchContingency("L2-4-1"))),
                strategiesOf("L2-4-1", "shuntOff"), actions, 2);
    }

    /** A strategy pair is solved on the zero-impedance tree OLF reaches it with: its contingency's. Opening coupler
     *  z12 puts z23 back in the tree; a pair left on the base tree would hold z23 out and leave b2 hanging. */
    @Test
    void strategyPairKeepsItsContingencysCouplerTreeMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = ZeroImpedanceTreeReplayTest.couplerLoop();
        network.getBusBreakerView().getBus("b2").getVoltageLevel().newLoad().setId("l2").setBus("b2").setConnectableBus("b2")
                .setP0(0.5).setQ0(0.2).add();
        List<Contingency> contingencies = List.of(new Contingency("z12", new BranchContingency("z12")));
        List<com.powsybl.action.Action> actions = List.of(new com.powsybl.action.LoadActionBuilder().withId("more4")
                .withLoadId("l4").withRelativeValue(true).withActivePowerValue(0.3).build());
        assertStrategyMatchesCpu(network, contingencies, strategiesOf("z12", "more4"), actions, 2, false);
    }

    /** A TerminalsConnectionAction on a three-winding transformer opens or closes its three legs (OLF's
     *  getBranchesByOriginalId). ieee14 with a three-winding transformer between B4, B7 and B9, in parallel with the
     *  two-winding ones there: opening it after B4's load loss; after losing it, reclosing it or opening a line. A
     *  pair keeps its contingency's one-side openings (the legs at the star bus). */
    @Test
    void threeWindingsTransformerConnectionActionMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        var vl4 = network.getVoltageLevel("VL4");
        vl4.getSubstation().orElseThrow().newThreeWindingsTransformer().setId("T3W")
                .newLeg1().setVoltageLevel("VL4").setBus("B4").setConnectableBus("B4")
                        .setRatedU(vl4.getNominalV()).setR(0.5).setX(20).setG(0).setB(0).add()
                .newLeg2().setVoltageLevel("VL7").setBus("B7").setConnectableBus("B7")
                        .setRatedU(network.getVoltageLevel("VL7").getNominalV()).setR(0.5).setX(20).setG(0).setB(0).add()
                .newLeg3().setVoltageLevel("VL9").setBus("B9").setConnectableBus("B9")
                        .setRatedU(network.getVoltageLevel("VL9").getNominalV()).setR(0.5).setX(20).setG(0).setB(0).add()
                .add();
        String loadId = loadAtBus(network, "B4").getId();
        List<Contingency> contingencies = List.of(
                new Contingency(loadId, new com.powsybl.contingency.LoadContingency(loadId)),
                new Contingency("T3W", new com.powsybl.contingency.ThreeWindingsTransformerContingency("T3W")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.TerminalsConnectionActionBuilder().withId("open3w")
                        .withNetworkElementId("T3W").withOpen(true).build(),
                new com.powsybl.action.TerminalsConnectionActionBuilder().withId("close3w")
                        .withNetworkElementId("T3W").withOpen(false).build(),
                new com.powsybl.action.TerminalsConnectionActionBuilder().withId("open1011")
                        .withNetworkElementId("L10-11-1").withOpen(true).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = new java.util.ArrayList<>(strategiesOf(loadId, "open3w"));
        // reclosing or opening the outaged transformer stays on the CPU: the contingency opened its legs on ONE side
        strategies.addAll(strategiesOf("T3W", "close3w", "open1011", "open3w"));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 4);
    }

    /** OLF's LfGeneratorAction applies the active power only: a target Q, a target V or a regulator flag carried by
     *  the same action changes nothing in OLF's equations, and these pairs run on the device. */
    @Test
    void generatorActionWithVoltageFieldsMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        String loadId = loadAtBus(network, "B4").getId();
        List<Contingency> contingencies = List.of(new Contingency(loadId, new com.powsybl.contingency.LoadContingency(loadId)));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.GeneratorActionBuilder().withId("upQ").withGeneratorId("B2-G")
                        .withActivePowerRelativeValue(true).withActivePowerValue(10).withTargetQ(5).build(),
                new com.powsybl.action.GeneratorActionBuilder().withId("upV").withGeneratorId("B2-G")
                        .withActivePowerRelativeValue(true).withActivePowerValue(10).withTargetV(140).build(),
                new com.powsybl.action.GeneratorActionBuilder().withId("upOff").withGeneratorId("B2-G")
                        .withActivePowerRelativeValue(false).withActivePowerValue(50).withVoltageRegulatorOn(false).build());
        assertStrategyMatchesCpu(network, contingencies, strategiesOf(loadId, "upQ", "upV", "upOff"), actions, 4);
    }

    @Test
    void generatorActionTakingAParticipantOutMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = participationRecheckNetwork();
        String loadId = loadAtBus(network, "B4").getId();
        List<Contingency> contingencies = List.of(new Contingency(loadId, new com.powsybl.contingency.LoadContingency(loadId)));
        List<com.powsybl.action.Action> actions = List.of(
                genAction("over2", "B2-G", false, 70),          // above its maxP 60
                genAction("under2", "B2-G", true, -15),         // below its minP 30
                genAction("over6", "B6-GP", true, 70),          // above its maxP 80
                genAction("inside2", "B2-G", false, 50),        // stays a participant
                // below its minP only from where the contingency left it (B2-G ends the parent near 31.6 MW)
                genAction("nudge2", "B2-G", true, -4));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = new java.util.ArrayList<>(
                strategiesOf(loadId, "over2", "under2", "over6", "inside2", "nudge2"));
        // two actions on one generator: OLF re-checks after each, the last one decides
        strategies.add(new com.powsybl.contingency.strategy.OperatorStrategy("s-over-then-inside",
                com.powsybl.contingency.ContingencyContext.specificContingency(loadId),
                new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("over2", "inside2")));
        strategies.add(new com.powsybl.contingency.strategy.OperatorStrategy("s-inside-then-over",
                com.powsybl.contingency.ContingencyContext.specificContingency(loadId),
                new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("inside2", "over2")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 8);
    }

    /** Only the slack bus's generator participates. An absolute action above its maxP leaves OLF with no participant
     *  at all, which the device's slack-only check cannot express. Any action on the slack bus of a slack-only
     *  distribution stays on the CPU (SLACK_BUS); both pairs must still match. */
    @Test
    void generatorActionOnASlackOnlyParticipantMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        network.getGenerator("B1-G").setMinP(0).setMaxP(400);
        for (String id : List.of("B2-G", "B3-G", "B6-G", "B8-G")) {
            network.getGenerator(id).newExtension(com.powsybl.iidm.network.extensions.ActivePowerControlAdder.class)
                    .withParticipate(false).add();
        }
        String loadId = loadAtBus(network, "B4").getId();
        List<Contingency> contingencies = List.of(new Contingency(loadId, new com.powsybl.contingency.LoadContingency(loadId)));
        List<com.powsybl.action.Action> actions = List.of(
                genAction("up1", "B1-G", true, 10),
                genAction("over1", "B1-G", false, 450));
        assertStrategyMatchesCpu(network, contingencies, strategiesOf(loadId, "up1", "over1"), actions, 1);
    }

    @Test
    void generatorActionBringingAGeneratorInRunsOnCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = participationRecheckNetwork();
        String loadId = loadAtBus(network, "B4").getId();
        List<Contingency> contingencies = List.of(new Contingency(loadId, new com.powsybl.contingency.LoadContingency(loadId)));
        List<com.powsybl.action.Action> actions = List.of(
                genAction("start8", "B8-GP", false, 20),        // inside [10, 50]: it now participates
                genAction("nudge8", "B8-GP", true, 1));         // still below its minP: stays out
        assertStrategyMatchesCpu(network, contingencies, strategiesOf(loadId, "start8", "nudge8"), actions, 2);
    }

    @Test
    void generatorActionStressAcrossContingencies() {
        // Freeze-semantics hunt: every non-fragmenting ieee14 line contingency x a LARGE redispatch
        // (+60 MW on B2-G) that pushes controllers switched to PQ around. Any surviving CPU-vs-device RL
        // freeze divergence must surface as a >0.05 kV bus mismatch on some pair.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        System.setProperty("olf.gpu.test.slackTol", "0.01");
        System.setProperty("olf.gpu.test.convEps", "1e-9");
        List<String> lineIds = IeeeCdfNetworkFactory.create14().getLineStream()
                .map(l -> l.getId()).toList();
        try {
            for (String lineId : lineIds) {
                Network network = IeeeCdfNetworkFactory.create14();
                // extra fixed-Q gen at B10 (the #21 shape): a non-voltage-capable gen in the RL mix
                network.getVoltageLevel("VL10").newGenerator().setId("B10-GQ").setBus("B10").setConnectableBus("B10")
                        .setTargetP(5).setTargetQ(8).setVoltageRegulatorOn(false).setMinP(0).setMaxP(20).add();
                List<Contingency> contingencies = List.of(new Contingency(lineId, new BranchContingency(lineId)));
                List<com.powsybl.action.Action> actions = List.of(
                        new com.powsybl.action.GeneratorActionBuilder().withId("redispatch2")
                                .withGeneratorId("B2-G").withActivePowerRelativeValue(true).withActivePowerValue(60).build());
                List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                        new com.powsybl.contingency.strategy.OperatorStrategy("sGen",
                                com.powsybl.contingency.ContingencyContext.all(),
                                new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("redispatch2")));
                try {
                    assertStrategyMatchesCpu(network, contingencies, strategies, actions, -1);
                } catch (AssertionError e) {
                    throw new AssertionError("ctg " + lineId + ": " + e.getMessage(), e);
                }
            }
        } finally {
            System.clearProperty("olf.gpu.test.slackTol");
            System.clearProperty("olf.gpu.test.convEps");
        }
    }

    @Test
    void violationConditionStrategyBatchMatchesCpu() {
        // #24: violation-conditioned strategy — the pair is precomputed SPECULATIVELY; the engine's
        // CPU-side condition check (on the post-contingency violations) decides whether it is used.
        // A tight current limit on l12 guarantees the post-contingency violation that arms the condition.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = distributedSlackNetworkWithExtras();
        // shed target on b2, NOT b4: b4 is the star center = slack bus, where injection overrides
        // are unbatchable by design (the pair would silently fall back and defeat the test)
        network.getBusBreakerView().getBus("b2").getVoltageLevel().newLoad()
                .setId("extraLoad2").setBus("b2").setConnectableBus("b2").setP0(40).setQ0(20).add();
        network.getLine("l24").newCurrentLimits1().setPermanentLimit(50).add();
        List<Contingency> contingencies = List.of(new Contingency("l24_bis", new BranchContingency("l24_bis")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.LoadActionBuilder().withId("shed4").withLoadId("extraLoad2")
                        .withRelativeValue(false).withActivePowerValue(10).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sViol",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.AnyViolationCondition(), List.of("shed4")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    @Test
    void violationConditionNotTriggeredMatchesCpu() {
        // Negative case: no limits => no violations => the condition never passes. The speculative
        // pair still precomputes (count 2) but NEITHER arm produces an operator strategy result.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = distributedSlackNetworkWithExtras();
        network.getBusBreakerView().getBus("b2").getVoltageLevel().newLoad()
                .setId("extraLoad2").setBus("b2").setConnectableBus("b2").setP0(40).setQ0(20).add();
        List<Contingency> contingencies = List.of(new Contingency("l24_bis", new BranchContingency("l24_bis")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.LoadActionBuilder().withId("shed4").withLoadId("extraLoad2")
                        .withRelativeValue(false).withActivePowerValue(10).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sViolNone",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.AnyViolationCondition(), List.of("shed4")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    @Test
    void shuntPositionActionStrategyBatchMatchesCpu() {
        // #24 stage 2b: ShuntCompensatorPositionAction = per-scenario admittance shift (linear model).
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        String shuntId = network.getShuntCompensatorStream().findFirst().orElseThrow().getId();
        String lineId = network.getLineStream()
                .filter(l -> !l.getId().contains("7") && !l.getId().contains("8"))
                .skip(4).findFirst().orElseThrow().getId();
        List<Contingency> contingencies = List.of(new Contingency(lineId, new BranchContingency(lineId)));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.ShuntCompensatorPositionActionBuilder().withId("shOff")
                        .withShuntCompensatorId(shuntId).withSectionCount(0).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sShunt",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("shOff")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }
}
