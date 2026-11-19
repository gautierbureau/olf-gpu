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
     *  that is not connected at both ends (AbstractLfBranchAction.getNetworkActivations), even one the contingency
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
                        .withResourceBundles(com.powsybl.openloadflow.lf.PowsyblOpenLoadFlowReportResourceBundle.BASE_NAME)
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

    /** An action on a load or a generator OLF's network does not hold (here an island outside the main component): OLF's
     *  LfLoadAction and LfGeneratorAction are invalid there, report the failure and change nothing, and the strategy's other
     *  actions apply. The device composes the pair without that action, alone or beside a tap action. */
    @Test
    void actionsOnElementsOutsideTheNetworkChangeNothingLikeOlf() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
        network.getTwoWindingsTransformer("T2wT").getRatioTapChanger().setTapPosition(0).setRegulating(false);
        com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
        network.newLine().setId(l0.getId() + "_BIS")
                .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(0).setB1(0).setG2(0).setB2(0).add();
        com.powsybl.iidm.network.VoltageLevel island = network.newSubstation().setId("SI").add().newVoltageLevel().setId("VLI")
                .setNominalV(l0.getTerminal1().getVoltageLevel().getNominalV())
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
        island.getBusBreakerView().newBus().setId("BI").add();
        island.newLoad().setId("LI").setBus("BI").setConnectableBus("BI").setP0(10).setQ0(5).add();
        island.newGenerator().setId("GI").setBus("BI").setConnectableBus("BI").setMinP(0).setMaxP(50).setTargetP(10)
                .setVoltageRegulatorOn(true).setTargetV(island.getNominalV()).add();
        List<Contingency> contingencies = List.of(new Contingency(l0.getId() + "_BIS", new BranchContingency(l0.getId() + "_BIS")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.RatioTapChangerTapPositionAction("tapUp", "T2wT", false, 2),
                new com.powsybl.action.LoadActionBuilder().withId("moreIsland").withLoadId("LI").withRelativeValue(true)
                        .withActivePowerValue(5).build(),
                new com.powsybl.action.GeneratorActionBuilder().withId("genIsland").withGeneratorId("GI")
                        .withActivePowerRelativeValue(false).withActivePowerValue(20).build());
        com.powsybl.contingency.ContingencyContext all = com.powsybl.contingency.ContingencyContext.all();
        com.powsybl.contingency.strategy.condition.TrueCondition always = new com.powsybl.contingency.strategy.condition.TrueCondition();
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sLoad", all, always, List.of("moreIsland")),
                new com.powsybl.contingency.strategy.OperatorStrategy("sGenTap", all, always, List.of("genIsland", "tapUp")),
                new com.powsybl.contingency.strategy.OperatorStrategy("sLoadGen", all, always, List.of("moreIsland", "genIsland")));
        // the contingency and its three pairs, all on the device
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 4);
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
        // The strategy's violations come from the device screen's candidates. The tap moves PS1's current over a limit
        // set between its post-contingency current and its post-action one, and the host no longer scans the retapped
        // transformer in full: the screen must read it with the action's tap row (GpuScreenScenario.ROW), as the device
        // solved it, to find that violation.
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

    /** A strategy condition on a generator's active power reads the target OLF's post-contingency load flow leaves it
     *  (ThresholdConditionEvaluator): the loss pre-distribution and the distributed-slack passes. A line outage loses no
     *  injection, so B6-G's target moves by the passes alone, which the device ran; the host network held the base
     *  target. The threshold sits half-way between B6-G's base target and the one the outage leaves it (both read from
     *  plain CPU load flows: a non-slack generator's P is its target). B3, B6 and B8 get 20 MW and B1-G is out of the
     *  distribution, as in GpuSaDistributedSlackTest's loss replay test. */
    @Test
    void injectionConditionReadsTheDevicesSlackDistribution() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        java.util.function.Supplier<Network> fixture = () -> {
            Network n = IeeeCdfNetworkFactory.create14();
            for (String g : List.of("B3-G", "B6-G", "B8-G")) {
                n.getGenerator(g).setTargetP(20);
            }
            n.getGenerator("B1-G").newExtension(com.powsybl.iidm.network.extensions.ActivePowerControlAdder.class)
                    .withParticipate(false).add();
            return n;
        };
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.loadflow.LoadFlow.Runner runner = new com.powsybl.loadflow.LoadFlow.Runner(
                new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory()));
        Network base = fixture.get();
        org.junit.jupiter.api.Assertions.assertTrue(runner.run(base, lfp).isFullyConverged());
        double baseTarget = -base.getGenerator("B6-G").getTerminal().getP();
        Network outage = fixture.get();
        outage.getLine("L1-2-1").disconnect();
        org.junit.jupiter.api.Assertions.assertTrue(runner.run(outage, lfp).isFullyConverged());
        double outageTarget = -outage.getGenerator("B6-G").getTerminal().getP();
        org.junit.jupiter.api.Assertions.assertTrue(Math.abs(outageTarget - baseTarget) > 0.2,
                "the outage must move B6-G's target: " + baseTarget + " -> " + outageTarget);
        Network network = fixture.get();
        List<Contingency> contingencies = List.of(new Contingency("L1-2-1", new BranchContingency("L1-2-1")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.GeneratorActionBuilder().withId("redispatch3")
                        .withGeneratorId("B3-G").withActivePowerRelativeValue(true).withActivePowerValue(10).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sB6Moved",
                        com.powsybl.contingency.ContingencyContext.specificContingency("L1-2-1"),
                        new com.powsybl.contingency.strategy.condition.InjectionThresholdCondition("B6-G",
                                com.powsybl.contingency.strategy.condition.AbstractThresholdCondition.Variable.ACTIVE_POWER,
                                outageTarget > baseTarget
                                        ? com.powsybl.contingency.strategy.condition.AbstractThresholdCondition.ComparisonType.GREATER_THAN
                                        : com.powsybl.contingency.strategy.condition.AbstractThresholdCondition.ComparisonType.LESS_THAN,
                                (baseTarget + outageTarget) / 2),
                        List.of("redispatch3")));
        // the contingency and its pair on the device (the conditional pair speculatively)
        SecurityAnalysisResult cpu = assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2, lfp);
        assertEquals(1, cpu.getOperatorStrategyResults().size(), "OLF's condition reads the target its slack distribution left");
    }

    /** OLF still evaluates the strategies of a contingency whose load flow ran out of Newton iterations
     *  (MAX_ITERATION_REACHED), and a branch threshold reads the flows of that run's last iterate (NewtonRaphson leaves
     *  the state vector there). The device answered such a contingency without injecting anything, so the condition read
     *  the flows the network held before it: the base's here. The budget is one step short of what L1-2-1 needs (as in
     *  GpuSaSolverTest's budget tests), so the last iterate is close to the outage's solution and L1-5-1 carries far more
     *  than in the base; the strategy recloses L1-2-1, a pair the device answers. */
    @Test
    void thresholdConditionAfterAnExhaustedContingencyReadsOlfsLastIterate() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        com.powsybl.loadflow.LoadFlowParameters plain = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(false).setUseReactiveLimits(false);
        com.powsybl.loadflow.LoadFlow.Runner runner = new com.powsybl.loadflow.LoadFlow.Runner(
                new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory()));
        Network outage = IeeeCdfNetworkFactory.create14();
        outage.getLine("L1-2-1").disconnect();
        org.junit.jupiter.api.Assertions.assertTrue(runner.run(outage, plain).isFullyConverged());
        double after = outage.getLine("L1-5-1").getTerminal1().getP();
        // the solved base on the network, so both arms' base load flow needs a single step
        Network network = IeeeCdfNetworkFactory.create14();
        org.junit.jupiter.api.Assertions.assertTrue(runner.run(network, plain).isFullyConverged());
        double before = network.getLine("L1-5-1").getTerminal1().getP();
        org.junit.jupiter.api.Assertions.assertTrue(after > before + 20, "L1-2-1's outage must load L1-5-1: " + before + " -> " + after);
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(false).setUseReactiveLimits(false)
                .setVoltageInitMode(com.powsybl.loadflow.LoadFlowParameters.VoltageInitMode.PREVIOUS_VALUES);
        com.powsybl.openloadflow.OpenLoadFlowParameters ext = com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp);
        List<Contingency> contingencies = List.of(new Contingency("L1-2-1", new BranchContingency("L1-2-1")));
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        int budget = -1;                                     // the smallest budget the CPU converges L1-2-1 under
        for (int m = 2; m <= 30 && budget < 0; m++) {
            ext.setMaxNewtonRaphsonIterations(m);
            if (runSecurityAnalysis(network, contingencies, java.util.Collections.emptyList(), sap).getPostContingencyResults().get(0)
                    .getStatus() == com.powsybl.security.PostContingencyComputationStatus.CONVERGED) {
                budget = m;
            }
        }
        org.junit.jupiter.api.Assertions.assertTrue(budget >= 3, "L1-2-1 must need at least two steps, smallest converging budget " + budget);
        ext.setMaxNewtonRaphsonIterations(budget - 1);
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.TerminalsConnectionAction("reclose", "L1-2-1", false));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sOnL15Flow",
                        com.powsybl.contingency.ContingencyContext.specificContingency("L1-2-1"),
                        new com.powsybl.contingency.strategy.condition.BranchThresholdCondition("L1-5-1",
                                com.powsybl.contingency.strategy.condition.AbstractThresholdCondition.Variable.ACTIVE_POWER,
                                com.powsybl.contingency.strategy.condition.AbstractThresholdCondition.ComparisonType.GREATER_THAN,
                                (before + after) / 2, com.powsybl.iidm.network.TwoSides.ONE),
                        List.of("reclose")));
        // the pair on the device (the contingency ran out of iterations there too)
        SecurityAnalysisResult cpu = assertStrategyMatchesCpu(network, contingencies, strategies, actions, 1, lfp);
        assertEquals(com.powsybl.security.PostContingencyComputationStatus.MAX_ITERATION_REACHED,
                cpu.getPostContingencyResults().get(0).getStatus());
        assertEquals(1, cpu.getOperatorStrategyResults().size(), "OLF's condition reads the flows of its failed run's last iterate");
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
    void baseOpenCouplerClosedOntoWhatTheContingencyCutsOffMatchesCpu() {
        // VL1: BBS1 - C12 - BBS2 and BBS3 - C34 - BBS4, C23 open at base between them. Losing L1 cuts BBS1 and BBS2 (G1, LD2)
        // off; closing C23 brings BBS2 back (OLF adds C23 to the post-contingency connectivity, from which the contingency
        // also removed C12: BBS1 stays out). The islanding analysis never added the strategy's closes back and reused the
        // contingency's verdict, the cut being the same: the pair passed for one bringing nothing back, and the device
        // solved it with BBS2 still out. Seen now, the pair goes to the CPU (PAIR:RECLOSE_INTO_ISLAND)
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        var cpu = assertTwoCouplerTreesStrategyMatchesCpu("L1", List.of("closeC23"), 1);
        assertEquals(3, cpu.getOperatorStrategyResults().get(0).getNetworkResult().getBusResults().stream()
                .filter(b -> b.getVoltageLevelId().equals("VL1")).count(), "the strategy brings BBS2 back, not BBS1");
    }

    @Test
    void baseOpenCouplerClosedKeepingWhatAnOpenCutsOffMatchesCpu() {
        // the same network after losing L4: the strategy opens C12 and closes C23, so BBS2 (LD2) stays in the network through
        // C23 alone. The islanding analysis left C23 out and took BBS2 for an island: the device dropped LD2. And BBS2 is a
        // child of the C12 tree (BBS1, with G1, roots it), its P balance relocated onto C12's DUMMY_P row: C23, packed open
        // at base, put its flow into BBS2's PHI row, C12's ZERO_PHI - the device did not converge on the pair
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        var cpu = assertTwoCouplerTreesStrategyMatchesCpu("L4", List.of("closeC23", "openC12"), 2);
        assertEquals(4, cpu.getOperatorStrategyResults().get(0).getNetworkResult().getBusResults().stream()
                .filter(b -> b.getVoltageLevelId().equals("VL1")).count(), "BBS2 stays in the network");
    }

    @Test
    void couplerClosedParallelToOneTheStrategyOpensMatchesCpu() {
        // CP12, open at base, runs beside C12 between BBS1 and BBS2: closing it alone closes a loop, nothing to compose. The
        // strategy also opens C12, so CP12 is what keeps BBS2 (LD2) on BBS1: the close joins two parts again. Taken for a
        // loop at base, it was left out and the device solved BBS2 cut off
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = twoCouplerTrees();
        network.getVoltageLevel("VL1").getNodeBreakerView().newBreaker().setId("CP12").setNode1(0).setNode2(1).setOpen(true).add();
        List<Contingency> contingencies = List.of(new Contingency("L4", new BranchContingency("L4")));
        List<com.powsybl.action.Action> actions = List.of(new com.powsybl.action.SwitchAction("closeCP12", "CP12", false),
                new com.powsybl.action.SwitchAction("openC12", "C12", true), new com.powsybl.action.SwitchAction("openC34", "C34", true));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("s", com.powsybl.contingency.ContingencyContext.specificContingency("L4"),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("openC12", "closeCP12")));
        var cpu = assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
        assertEquals(1, cpu.getOperatorStrategyResults().size(), "the strategy has a result");
        assertEquals(4, cpu.getOperatorStrategyResults().get(0).getNetworkResult().getBusResults().stream()
                .filter(b -> b.getVoltageLevelId().equals("VL1")).count(), "BBS2 stays in the network through CP12");
    }

    @Test
    void switchClosedThenOpenedByOneStrategyMatchesCpu() {
        // OLF applies a strategy's actions in their order: closing C23 then opening it leaves it open, as at base. The
        // composer took the close and skipped the open (C23 is open at base): the device solved the pair with C23 closed. G4
        // holds no voltage, so BBS3 and BBS4 tied to BBS2 would show it in their voltage
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = twoCouplerTrees();
        network.getGenerator("G4").setTargetQ(0).setVoltageRegulatorOn(false);
        List<Contingency> contingencies = List.of(new Contingency("L4", new BranchContingency("L4")));
        List<com.powsybl.action.Action> actions = List.of(new com.powsybl.action.SwitchAction("closeC23", "C23", false),
                new com.powsybl.action.SwitchAction("openC23", "C23", true), new com.powsybl.action.SwitchAction("openC34", "C34", true));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("s", com.powsybl.contingency.ContingencyContext.specificContingency("L4"),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("closeC23", "openC23")));
        var cpu = assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
        assertEquals(1, cpu.getOperatorStrategyResults().size(), "the strategy has a result");
    }

    private SecurityAnalysisResult assertTwoCouplerTreesStrategyMatchesCpu(String contingencyId, List<String> actionIds, int expectedPrecomputed) {
        Network network = twoCouplerTrees();
        List<Contingency> contingencies = List.of(new Contingency(contingencyId, new BranchContingency(contingencyId)));
        // the actions opening C12 and C34 also make OLF retain them: BBS1 to BBS4 are four buses
        List<com.powsybl.action.Action> actions = List.of(new com.powsybl.action.SwitchAction("closeC23", "C23", false),
                new com.powsybl.action.SwitchAction("openC12", "C12", true), new com.powsybl.action.SwitchAction("openC34", "C34", true));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("s", com.powsybl.contingency.ContingencyContext.specificContingency(contingencyId),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), actionIds));
        var cpu = assertStrategyMatchesCpu(network, contingencies, strategies, actions, expectedPrecomputed);
        assertEquals(1, cpu.getOperatorStrategyResults().size(), "the strategy has a result");
        return cpu;
    }

    /** VL1 (400 kV, node-breaker): BBS1..BBS4 on nodes 0..3, couplers C12 and C34 closed, C23 open; G1 on BBS1, LD2 on BBS2,
     *  LD3 on BBS3, G4 on BBS4; lines L1 (BBS1), L3 (BBS3) and L4 (BBS4) to VL2, which has G2 and LDV2. */
    private static Network twoCouplerTrees() {
        Network network = Network.create("twoCouplerTrees", "test");
        var vl1 = network.newSubstation().setId("S1").add().newVoltageLevel().setId("VL1").setNominalV(400)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.NODE_BREAKER).add();
        var v1 = vl1.getNodeBreakerView();
        for (int i = 0; i < 4; i++) {
            v1.newBusbarSection().setId("BBS" + (i + 1)).setNode(i).add();
        }
        v1.newBreaker().setId("C12").setNode1(0).setNode2(1).setOpen(false).add();
        v1.newBreaker().setId("C23").setNode1(1).setNode2(2).setOpen(true).add();
        v1.newBreaker().setId("C34").setNode1(2).setNode2(3).setOpen(false).add();
        int[][] feeders = {{0, 10}, {1, 11}, {2, 12}, {3, 13}, {0, 20}, {2, 21}, {3, 22}};
        for (int[] f : feeders) {
            v1.newBreaker().setId("B" + f[1]).setNode1(f[0]).setNode2(f[1]).setOpen(false).add();
        }
        vl1.newGenerator().setId("G1").setNode(10).setMinP(0).setMaxP(300).setTargetP(100).setTargetV(400)
                .setVoltageRegulatorOn(true).add();
        vl1.newLoad().setId("LD2").setNode(11).setP0(80).setQ0(20).add();
        vl1.newLoad().setId("LD3").setNode(12).setP0(60).setQ0(15).add();
        vl1.newGenerator().setId("G4").setNode(13).setMinP(0).setMaxP(200).setTargetP(50).setTargetV(400)
                .setVoltageRegulatorOn(true).add();
        var vl2 = network.newSubstation().setId("S2").add().newVoltageLevel().setId("VL2").setNominalV(400)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.NODE_BREAKER).add();
        var v2 = vl2.getNodeBreakerView();
        v2.newBusbarSection().setId("BBSV2").setNode(0).add();
        for (int node = 1; node <= 5; node++) {
            v2.newBreaker().setId("BV" + node).setNode1(0).setNode2(node).setOpen(false).add();
        }
        vl2.newGenerator().setId("G2").setNode(1).setMinP(0).setMaxP(600).setTargetP(200).setTargetV(400)
                .setVoltageRegulatorOn(true).add();
        vl2.newLoad().setId("LDV2").setNode(2).setP0(250).setQ0(50).add();
        String[][] lines = {{"L1", "20", "3"}, {"L3", "21", "4"}, {"L4", "22", "5"}};
        for (String[] l : lines) {
            network.newLine().setId(l[0]).setVoltageLevel1("VL1").setNode1(Integer.parseInt(l[1]))
                    .setVoltageLevel2("VL2").setNode2(Integer.parseInt(l[2])).setR(1).setX(10).setG1(0).setB1(0).setG2(0).setB2(0).add();
        }
        return network;
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

    @Test
    void closedCouplerCloseStrategyBatchMatchesCpu() {
        // SwitchAction CLOSING the coupler C, closed already: OLF retains no switch for it (the action changes no state)
        // and its LfSwitchAction changes nothing, so the strategy re-solves the contingency's network. The pair was left
        // to the CPU, and its contingency with it.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = coupledSectionsWithThirdLine();
        List<Contingency> contingencies = List.of(new Contingency("L1", new BranchContingency("L1")));
        List<com.powsybl.action.Action> actions = List.of(new com.powsybl.action.SwitchAction("closeC", "C", false));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sCloseC",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("closeC")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    @Test
    void openCouplerOpenStrategyBatchMatchesCpu() {
        // SwitchAction OPENING the coupler C, open already at base: no retained switch, nothing changes in OLF
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = coupledSectionsWithThirdLine();
        network.getSwitch("C").setOpen(true);
        List<Contingency> contingencies = List.of(new Contingency("L1", new BranchContingency("L1")));
        List<com.powsybl.action.Action> actions = List.of(new com.powsybl.action.SwitchAction("openC", "C", true));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sOpenC",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("openC")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
    }

    @Test
    void retainedClosedCouplerCloseStrategyBatchMatchesCpu() {
        // Two strategies on the closed coupler C: opening it makes OLF retain C, so the one closing it finds a closed
        // LfBranch, which OLF enables again: nothing changes. Both pairs on the device.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = coupledSectionsWithThirdLine();
        List<Contingency> contingencies = List.of(new Contingency("L1", new BranchContingency("L1")));
        List<com.powsybl.action.Action> actions = List.of(new com.powsybl.action.SwitchAction("openC", "C", true),
                new com.powsybl.action.SwitchAction("closeC", "C", false));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sOpenC",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("openC")),
                new com.powsybl.contingency.strategy.OperatorStrategy("sCloseC",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("closeC")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 3);
    }

    @Test
    void switchClosableOntoADeadBusKeepsTheRunOnTheDevice() {
        // BD is open at base and leads from BBS3 to BBS4, whose load nothing else feeds: a strategy closing it makes OLF
        // retain the switch, with its DUMMY_P / DUMMY_Q held at 0 next to a dead bus. No family packs such a switch, and its
        // two rows left the device structure unbuildable: every contingency and every strategy of the run went to the CPU.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = withDeadSection(true, false);
        List<Contingency> contingencies = List.of(new Contingency("L1", new BranchContingency("L1")));
        List<com.powsybl.action.Action> actions = List.of(new com.powsybl.action.SwitchAction("closeBD", "BD", false),
                new com.powsybl.action.SwitchAction("openC", "C", true));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sCloseBD",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("closeBD")),
                new com.powsybl.contingency.strategy.OperatorStrategy("sOpenC",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("openC")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 3);
    }

    @Test
    void deadLoadedSectionCloseStrategyBatchMatchesCpu() {
        // Closing BD re-energizes BBS4 and its load LD4 through a switch alone: BBS4 is BBS3's electrical node, so the pair
        // is the contingency with LD4 at BBS3, and BBS4 takes BBS3's voltage (compared with the CPU's, BBS4 included)
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertDeadSectionCloseMatchesCpu(withDeadSection(true, false), 2);
    }

    /** The dead buses a strategy's switch may bring back have device rows past OLF's, one BUS_PHI and one BUS_V row each, when
     *  their island holds an impedant branch (BBS4 and BBS7, joined by LX); a section of switches alone (BBS4 without LX) is
     *  merged into the switch's bus and has none. */
    @Test
    void onlyIslandsWithABranchGetRowsPastOlfs() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        GpuBatchPrecomputer.LAST_DEAD_BUS_ROW_COUNT.set(-1);
        assertDeadSectionCloseMatchesCpu(withDeadSection(true, false), 2);
        assertEquals(0, GpuBatchPrecomputer.LAST_DEAD_BUS_ROW_COUNT.get(), "BBS4 is merged into BBS3: no rows");
        GpuBatchPrecomputer.LAST_DEAD_BUS_ROW_COUNT.set(-1);
        assertDeadSectionCloseMatchesCpu(withDeadSectionAndLine(420.), 2);
        assertEquals(4, GpuBatchPrecomputer.LAST_DEAD_BUS_ROW_COUNT.get(), "BBS4's and BBS7's BUS_PHI and BUS_V rows");
    }

    @Test
    void deadEmptySectionCloseStrategyBatchMatchesCpu() {
        // the same with nothing at BBS4: the pair is the contingency's network, BBS4 at BBS3's voltage
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertDeadSectionCloseMatchesCpu(withDeadSection(false, false), 2);
    }

    @Test
    void deadSectionWithAGeneratorCloseStrategyStaysOnCpu() {
        // a generator at BBS4 joins the network's controls when BD closes: not composed, the pair and its contingency run on
        // the CPU (the device still answers nothing else wrong)
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertDeadSectionCloseMatchesCpu(withDeadSection(true, true), 1);
    }

    @Test
    void shuntOnADeadSectionStaysOutOfTheDeviceFill() {
        // a shunt SH4 at BBS4, out of the network at base: it sits on BBS4's rows past OLF's (its island holds a shunt, whose
        // term depends on BBS4's own voltage), and every element mask starts with it disabled (OLF evaluates nothing
        // there). The fill used to read the voltage at column -1, out of bounds in the first scenario of a chunk: a CUDA
        // illegal memory access on network.xiidm (run this test under compute-sanitizer memcheck to see such a read).
        // Closing BD brings SH4 in: the pair enables it, on the device
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = withDeadSection(false, false);
        var vl2 = network.getVoltageLevel("VL2");
        vl2.getNodeBreakerView().newInternalConnection().setNode1(20).setNode2(23).add();
        vl2.newShuntCompensator().setId("SH4").setNode(23).setSectionCount(1)
                .newLinearModel().setBPerSection(1e-3).setGPerSection(1e-4).setMaximumSectionCount(1).add().add();
        assertDeadSectionCloseMatchesCpu(network, 2);
    }

    /** A line LY from BBS7 to the bus VL4_BUS of VL4, connected at BBS7 only: OLF enables it with BBS7 when BD closes, its charging
     *  at BBS7 (an open branch on BBS7's rows past OLF's). */
    @Test
    void branchOpenAtOneSideBeyondALineCloseStrategyBatchMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = withDeadSectionAndLine(420.);
        network.getVoltageLevel("VL3").getNodeBreakerView().newInternalConnection().setNode1(0).setNode2(3).add();
        var vl4 = network.getSubstation("S3").newVoltageLevel().setId("VL4").setNominalV(400)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
        vl4.getBusBreakerView().newBus().setId("VL4_BUS").add();
        network.newLine().setId("LY").setVoltageLevel1("VL3").setNode1(3).setVoltageLevel2("VL4").setConnectableBus2("VL4_BUS")
                .setR(1).setX(15).setG1(0).setB1(5e-4).setG2(0).setB2(5e-4).add();
        assertDeadSectionCloseMatchesCpu(network, 2);
    }

    /** BBS4 reached from BBS3 by BD or by BE, both open at base, one strategy closing each: BBS4's rows are packed as BD's
     *  child (its balances on BD's dummy rows), and the pair closing BE alone has BE couple BBS3 and BBS4 on BE's own dummy
     *  rows, BBS4's balances back on its own rows. */
    @Test
    void deadSectionReachedThroughAnotherSwitchStrategyBatchMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = withDeadSectionAndLine(420.);
        network.getVoltageLevel("VL2").getNodeBreakerView().newBreaker().setId("BE").setNode1(0).setNode2(20).setOpen(true).add();
        List<Contingency> contingencies = List.of(new Contingency("L1", new BranchContingency("L1")));
        List<com.powsybl.action.Action> actions = List.of(new com.powsybl.action.SwitchAction("closeBD", "BD", false),
                new com.powsybl.action.SwitchAction("closeBE", "BE", false));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sCloseBD", com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("closeBD")),
                new com.powsybl.contingency.strategy.OperatorStrategy("sCloseBE", com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("closeBE")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 3);
    }

    /** BBS8, out of the network at base, reached from BBS4 by the breaker BF, open: one strategy closing BD and
     *  BF brings back an island of two sections. BBS8 is BF's child, BF packed between BBS4's rows and BBS8's. */
    @Test
    void deadIslandOfTwoSectionsClosedByOneStrategyBatchMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = withDeadSectionAndLine(420.);
        var vl2 = network.getVoltageLevel("VL2");
        vl2.getNodeBreakerView().newBusbarSection().setId("BBS8").setNode(25).add();
        vl2.getNodeBreakerView().newBreaker().setId("BF").setNode1(20).setNode2(25).setOpen(true).add();
        // C open: BBS1 and BBS2 apart, the energized network larger than the dead one (its component the largest)
        network.getSwitch("C").setOpen(true);
        assertSwitchClosesMatchCpu(network, List.of("BD", "BF"));
    }

    /** A shunt SH7 at BBS7, beyond LX: closing BD brings it back with BBS7, its term on BBS7's rows past OLF's. */
    @Test
    void shuntBeyondALineCloseStrategyBatchMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = withDeadSectionAndLine(420.);
        var vl3 = network.getVoltageLevel("VL3");
        vl3.getNodeBreakerView().newInternalConnection().setNode1(0).setNode2(2).add();
        vl3.newShuntCompensator().setId("SH7").setNode(2).setSectionCount(1)
                .newLinearModel().setBPerSection(-1e-3).setGPerSection(1e-5).setMaximumSectionCount(1).add().add();
        assertDeadSectionCloseMatchesCpu(network, 2);
    }

    /** A section out of the network at base (BBS4, which a strategy on L1 may bring back) has no voltage row: the
     *  contingency without a strategy (L3) is still answered from its state on the device, the comparison deciding each
     *  answer from the written-back network too. OLF disables the branches of such a bus with it, so nothing reads its
     *  rows; a state view that wanted rows for every bus answered no contingency of the run from its state. */
    @Test
    void sectionOutOfTheNetworkKeepsTheAnswersFromTheState() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = withDeadSection(true, false);
        List<Contingency> contingencies = List.of(new Contingency("L1", new BranchContingency("L1")),
                new Contingency("L3", new BranchContingency("L3")));
        List<com.powsybl.action.Action> actions = List.of(new com.powsybl.action.SwitchAction("closeBD", "BD", false));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sCloseBD",
                        com.powsybl.contingency.ContingencyContext.specificContingency("L1"),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("closeBD")));
        long mismatches = GpuStateViewCheck.DECISION_MISMATCHES.sum();
        GpuStateViewCheck.compareInTests(true);
        try {
            assertStrategyMatchesCpu(network, contingencies, strategies, actions, 3);
        } finally {
            GpuStateViewCheck.compareInTests(false);
        }
        org.junit.jupiter.api.Assertions.assertTrue(GpuBatchedAcSecurityAnalysis.LAST_ANSWERED_FROM_STATE_COUNT.get() > 0,
                "the contingency without a strategy must be answered from its state");
        assertEquals(mismatches, GpuStateViewCheck.DECISION_MISMATCHES.sum());
    }

    private void assertDeadSectionCloseMatchesCpu(Network network, int expectedPrecomputed) {
        List<Contingency> contingencies = List.of(new Contingency("L1", new BranchContingency("L1")));
        List<com.powsybl.action.Action> actions = List.of(new com.powsybl.action.SwitchAction("closeBD", "BD", false));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sCloseBD",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("closeBD")));
        var cpu = assertStrategyMatchesCpu(network, contingencies, strategies, actions, expectedPrecomputed);
        assertEquals(1, cpu.getOperatorStrategyResults().size(), "the strategy has a result");
    }

    @Test
    void deadSectionCouplingTwoBusesCloseStrategyBatchMatchesCpu() {
        // With C open, BBS1 and BBS2 are two buses. BBS5 and BBS6, out of the network at base and joined by the breaker BM, reach
        // BBS1 by BA and BBS2 by BB, both open: a strategy closing both couples BBS1 and BBS2 through them, as closing C would.
        // The device holds it as one coupler between BBS1 and BBS2 on BA's dummy rows, LD5 at the coupled node
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertDeadCouplingSectionCloseMatchesCpu(withDeadCouplingSection(true), 2);
    }

    @Test
    void emptyDeadSectionCouplingTwoBusesCloseStrategyBatchMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertDeadCouplingSectionCloseMatchesCpu(withDeadCouplingSection(false), 2);
    }

    @Test
    void couplerOpenedByTheContingencyClosedByTheStrategyBatchMatchesCpu() {
        // the contingency opens the coupler C, splitting VL1 into BBS1 and BBS2; the strategy closes it back: the pair is the
        // contingency without C's opening
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertCloseOfContingencySwitchMatchesCpu(coupledSectionsWithThirdLine(), "C");
    }

    @Test
    void lineBreakerOpenedByTheContingencyClosedByTheStrategyBatchMatchesCpu() {
        // the contingency opens the breaker B1 between BBS2 and L2, which loses L2's end at VL1; closing B1 brings it back
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertCloseOfContingencySwitchMatchesCpu(coupledSectionsWithThirdLine(), "B1");
    }

    private void assertCloseOfContingencySwitchMatchesCpu(Network network, String switchId) {
        List<Contingency> contingencies = List.of(new Contingency(switchId, new com.powsybl.contingency.SwitchContingency(switchId)));
        List<com.powsybl.action.Action> actions = List.of(new com.powsybl.action.SwitchAction("close", switchId, false));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sClose",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("close")));
        var cpu = assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
        assertEquals(1, cpu.getOperatorStrategyResults().size(), "the strategy has a result");
    }

    @Test
    void twoDeadSectionsClosedByOneStrategyBatchMatchesCpu() {
        // one strategy closes BD onto BBS4 (from BBS3, load LD4) and BE onto BBS5 (from BBS2, C open, load LD5): two sections,
        // each hanging on its own bus, each load at the bus its section joins
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = withDeadSection(true, false);
        network.getSwitch("C").setOpen(true);                // BBS2 its own bus, off the slack
        var vl1 = network.getVoltageLevel("VL1");
        vl1.getNodeBreakerView().newBusbarSection().setId("BBS5").setNode(30).add();
        vl1.getNodeBreakerView().newBreaker().setId("BE").setNode1(1).setNode2(30).setOpen(true).add();
        vl1.getNodeBreakerView().newInternalConnection().setNode1(30).setNode2(31).add();
        vl1.newLoad().setId("LD5").setNode(31).setP0(4).setQ0(2).add();
        List<Contingency> contingencies = List.of(new Contingency("L1", new BranchContingency("L1")));
        List<com.powsybl.action.Action> actions = List.of(new com.powsybl.action.SwitchAction("closeBD", "BD", false),
                new com.powsybl.action.SwitchAction("closeBE", "BE", false));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sCloseBoth",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("closeBD", "closeBE")));
        var cpu = assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
        assertEquals(1, cpu.getOperatorStrategyResults().size(), "the strategy has a result");
    }

    private void assertDeadCouplingSectionCloseMatchesCpu(Network network, int expectedPrecomputed) {
        List<Contingency> contingencies = List.of(new Contingency("L1", new BranchContingency("L1")));
        List<com.powsybl.action.Action> actions = List.of(new com.powsybl.action.SwitchAction("closeBA", "BA", false),
                new com.powsybl.action.SwitchAction("closeBB", "BB", false),
                new com.powsybl.action.SwitchAction("openBM", "BM", true));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sCouple",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("closeBA", "closeBB")));
        var cpu = assertStrategyMatchesCpu(network, contingencies, strategies, actions, expectedPrecomputed);
        assertEquals(1, cpu.getOperatorStrategyResults().size(), "the strategy has a result");
    }

    /** {@link #coupledSectionsWithThirdLine} with C open, and BBS5 - BM - BBS6 in VL1 out of the network at base, reached from
     *  BBS1 by BA and from BBS2 by BB, both open; a load LD5 at BBS6 at will. The action list opening BM keeps it retained, so
     *  the dead section has two buses. */
    private static Network withDeadCouplingSection(boolean load) {
        Network network = coupledSectionsWithThirdLine();
        network.getSwitch("C").setOpen(true);
        var vl1 = network.getVoltageLevel("VL1");
        vl1.getNodeBreakerView().newBusbarSection().setId("BBS5").setNode(30).add();
        vl1.getNodeBreakerView().newBusbarSection().setId("BBS6").setNode(31).add();
        vl1.getNodeBreakerView().newBreaker().setId("BA").setNode1(0).setNode2(30).setOpen(true).add();
        vl1.getNodeBreakerView().newBreaker().setId("BM").setNode1(30).setNode2(31).setOpen(false).add();
        vl1.getNodeBreakerView().newBreaker().setId("BB").setNode1(31).setNode2(1).setOpen(true).add();
        if (load) {
            vl1.getNodeBreakerView().newInternalConnection().setNode1(31).setNode2(32).add();
            vl1.newLoad().setId("LD5").setNode(32).setP0(5).setQ0(1).add();
        }
        return network;
    }

    /** {@link #coupledSectionsWithThirdLine} with BBS4 in VL2, reached from BBS3 by the breaker BD only, open: out of the network
     *  at base, with a load LD4 and a generator G4 at will. */
    private static Network withDeadSection(boolean load, boolean generator) {
        Network network = coupledSectionsWithThirdLine();
        var vl2 = network.getVoltageLevel("VL2");
        vl2.getNodeBreakerView().newBusbarSection().setId("BBS4").setNode(20).add();
        vl2.getNodeBreakerView().newBreaker().setId("BD").setNode1(0).setNode2(20).setOpen(true).add();
        if (load) {
            vl2.getNodeBreakerView().newInternalConnection().setNode1(20).setNode2(21).add();
            vl2.newLoad().setId("LD4").setNode(21).setP0(5).setQ0(1).add();
        }
        if (generator) {
            vl2.getNodeBreakerView().newInternalConnection().setNode1(20).setNode2(22).add();
            vl2.newGenerator().setId("G4").setNode(22).setMinP(0).setMaxP(50).setTargetP(3).setTargetV(400)
                    .setVoltageRegulatorOn(true).add();
        }
        return network;
    }

    /** {@link #withDeadSection} without a load or a generator, BBS4 leading by the closed breaker BX and the line LX (with its
     *  charging) to BBS7, alone in the voltage level VL3 of a substation S3, reached by the closed breaker BY: BBS4 and BBS7
     *  are out of the network at base, and closing BD brings both back, BBS7 across an impedant branch. */
    private static Network withDeadSectionAndLine(double vl3HighVoltageLimit) {
        Network network = withDeadSection(false, false);
        network.getVoltageLevel("VL2").getNodeBreakerView().newBreaker().setId("BX").setNode1(20).setNode2(24).setOpen(false).add();
        var vl3 = network.newSubstation().setId("S3").add().newVoltageLevel().setId("VL3").setNominalV(400)
                .setLowVoltageLimit(370.).setHighVoltageLimit(vl3HighVoltageLimit)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.NODE_BREAKER).add();
        vl3.getNodeBreakerView().newBusbarSection().setId("BBS7").setNode(0).add();
        vl3.getNodeBreakerView().newBreaker().setId("BY").setNode1(0).setNode2(1).setOpen(false).add();
        network.newLine().setId("LX").setVoltageLevel1("VL2").setNode1(24).setVoltageLevel2("VL3").setNode2(1)
                .setR(2).setX(20).setG1(0).setB1(2e-4).setG2(0).setB2(2e-4).add();
        return network;
    }

    /** Closing BD brings back BBS4 and BBS7 beyond the line LX: both get their own rows, BD couples BBS3 and BBS4 (BBS4's balances
     *  on BD's dummy rows, BD's ZERO_PHI / ZERO_V on BBS4's), LX joins BBS4 to BBS7, and the pair matches OLF, BBS7's voltage
     *  (the line's charging raises it) included. Merged into BBS3 instead, BBS4 started from BBS3's voltage where OLF starts it
     *  from its own, and the pair converged 5e-4 kV away from OLF's. */
    @Test
    void deadBusBeyondALineCloseStrategyBatchMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertDeadSectionCloseMatchesCpu(withDeadSectionAndLine(420.), 2);
    }

    /** One strategy closing the coupler C (open at base: BBS1 and BBS2 apart) and BD, which brings BBS4 and BBS7 back across
     *  LX: two switch closes in one pair, both edges of OLF's tree. */
    @Test
    void couplerAndDeadSectionClosedByOneStrategyBatchMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = withDeadSectionAndLine(420.);
        network.getSwitch("C").setOpen(true);
        assertSwitchClosesMatchCpu(network, List.of("BD", "C"));
    }

    /** One strategy closing the couplers C and C2, both open at base between BBS1 and BBS2: OLF's tree takes one of them
     *  and leaves the other out, whose dummies its DUMMY_TARGET identities hold at 0, so the device keeps that one out of the
     *  coupling too. Coupled both, the two ZERO_PHI / ZERO_V pairs are redundant: the buses come out the same (the device
     *  copes with the singular split), but it is not OLF's system, and OLF reports no switch flow to tell them apart. */
    @Test
    void parallelCouplersClosedByOneStrategyBatchMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = coupledSectionsWithThirdLine();
        network.getSwitch("C").setOpen(true);
        network.getVoltageLevel("VL1").getNodeBreakerView().newBreaker().setId("C2").setNode1(0).setNode2(1).setOpen(true).add();
        var leftOut = com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchedSecurityAnalysisSolver.LAST_SWITCH_CLOSES_OUT_OF_TREE;
        leftOut.set(-1);
        assertSwitchClosesMatchCpu(network, List.of("C", "C2"));
        assertEquals(1, leftOut.get(), "one of C and C2 out of OLF's tree for the pair");
        leftOut.set(-1);
        Network withDeadSection = withDeadSectionAndLine(420.);
        withDeadSection.getSwitch("C").setOpen(true);
        assertSwitchClosesMatchCpu(withDeadSection, List.of("BD", "C"));
        assertEquals(0, leftOut.get(), "C and BD both in OLF's tree for the pair");
    }

    /** The contingency L1 with one strategy closing {@code switchIds}: the contingency and the pair on the device. */
    private void assertSwitchClosesMatchCpu(Network network, List<String> switchIds) {
        List<Contingency> contingencies = List.of(new Contingency("L1", new BranchContingency("L1")));
        List<com.powsybl.action.Action> actions = switchIds.stream()
                .map(id -> (com.powsybl.action.Action) new com.powsybl.action.SwitchAction("close" + id, id, false)).toList();
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sClose",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(),
                        actions.stream().map(com.powsybl.action.Action::getId).toList()));
        var cpu = assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
        assertEquals(1, cpu.getOperatorStrategyResults().size(), "the strategy has a result");
    }

    /** OLF's equation count, behind its stopping threshold sqrt(eps^2 * count), takes the active and reactive power balances
     *  of every bus a pair brings back (each switch keeps two equations, ZERO_PHI / ZERO_V closed and DUMMY_TARGET_P /
     *  DUMMY_TARGET_Q open): BBS4 merged into BBS3, BBS4 and BBS7 on their rows past OLF's, BBS5 and BBS6 coupling BBS1 and
     *  BBS2. The merged buses used to add nothing. */
    @Test
    void pairsCountTheEquationsOfTheBusesTheyBringBack() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        var added = com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchedSecurityAnalysisSolver.LAST_BROUGHT_BACK_EQUATIONS;
        added.set(-1);
        assertDeadSectionCloseMatchesCpu(withDeadSection(true, false), 2);
        assertEquals(2, added.get(), "BBS4, merged into BBS3: its two balances");
        added.set(-1);
        assertDeadSectionCloseMatchesCpu(withDeadSectionAndLine(420.), 2);
        assertEquals(4, added.get(), "BBS4 and BBS7, on their rows past OLF's");
        added.set(-1);
        assertDeadCouplingSectionCloseMatchesCpu(withDeadCouplingSection(true), 2);
        assertEquals(4, added.get(), "BBS5 and BBS6, merged into the coupled node");
    }

    /** BBS7's voltage above its limit after the close: the violation OLF reports at a bus the actions bring back, whose
     *  voltage the device screen does not read (it has no base row), is found on the host. */
    @Test
    void deadBusBeyondALineReportsItsVoltageViolation() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        List<Contingency> contingencies = List.of(new Contingency("L1", new BranchContingency("L1")));
        List<com.powsybl.action.Action> actions = List.of(new com.powsybl.action.SwitchAction("closeBD", "BD", false));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sCloseBD",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("closeBD")));
        var cpu = assertStrategyMatchesCpu(withDeadSectionAndLine(380.), contingencies, strategies, actions, 2);
        org.junit.jupiter.api.Assertions.assertTrue(cpu.getOperatorStrategyResults().get(0).getConditionalActionsResults().get(0)
                .getLimitViolationsResult()
                .getLimitViolations().stream().anyMatch(v -> "VL3".equals(v.getSubjectId())
                        && v.getLimitType() == com.powsybl.contingency.violations.LimitViolationType.HIGH_VOLTAGE),
                "OLF reports BBS7's high voltage: the comparison is not vacuous");
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
    void baseOpenBranchReclosedByAStrategyViolatesItsLimitLikeCpu() {
        // The screen skips a base-open branch except where a scenario closes it (GpuScreenScenario.IN), and the host no
        // longer scans the branches the pair's actions close: the reclosed line's violation is found by the screen alone.
        // It is open in the base and after the contingency, so its limit is crossed after the action only.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = distributedSlackNetworkWithExtras();
        com.powsybl.iidm.network.Line l24 = network.getLine("l24");
        com.powsybl.iidm.network.Line ter = network.newLine().setId("l24_ter")
                .setVoltageLevel1(l24.getTerminal1().getVoltageLevel().getId()).setBus1(l24.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l24.getTerminal2().getVoltageLevel().getId()).setBus2(l24.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l24.getR() * 3).setX(l24.getX() * 3).setG1(0).setB1(0).setG2(0).setB2(0).add();
        ter.getOrCreateSelectedOperationalLimitsGroup1().newCurrentLimits().setPermanentLimit(1).add();
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
                .beginPoint().setP(200).setMinQ(-50).setMaxQ(-45).endPoint()
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

    /** A GeneratorAction on a voltage-controlling curve generator that does not take part in the slack distribution: the
     *  device follows the P of the generators in its slack pack only, and this one's P is the action's for the whole solve,
     *  so its bus's reactive limits move by the curve's change between its base P and that P (OLF's own evaluation).
     *  The action moves B6-G 40 MW, which lowers its max Q from 30 to -10 MVAr: below what B6's voltage asks, so the bus
     *  switches to PQ there, where with its base limits it would stay PV. */
    @Test
    void generatorActionOnACurveGeneratorOutOfTheSlackPackStrategyBatchMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = withCurveOnB6();
        network.getGenerator("B6-G").newExtension(com.powsybl.iidm.network.extensions.ActivePowerControlAdder.class)
                .withParticipate(false).add();
        assertCurveGeneratorActionMatchesCpu(network, new com.powsybl.action.GeneratorActionBuilder().withId("up6")
                .withGeneratorId("B6-G").withActivePowerRelativeValue(true).withActivePowerValue(40).build(), true);
    }

    /** The same without a slack distribution, the action absolute: nothing moves B6-G but the action, set to 40 MW. */
    @Test
    void absoluteGeneratorActionOnACurveGeneratorWithoutDistributionStrategyBatchMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = withCurveOnB6();
        network.getGenerator("B6-G").newExtension(com.powsybl.iidm.network.extensions.ActivePowerControlAdder.class)
                .withParticipate(false).add();
        assertCurveGeneratorActionMatchesCpu(network, new com.powsybl.action.GeneratorActionBuilder().withId("up6")
                .withGeneratorId("B6-G").withActivePowerRelativeValue(false).withActivePowerValue(40).build(), false);
    }

    /** Without a slack distribution a generator's participation changes nothing: B6-G, out of OLF's participation at P 0
     *  and in it at 40 MW, does not keep the pair off the device. */
    @Test
    void generatorActionWithoutDistributionStrategyBatchMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertCurveGeneratorActionMatchesCpu(withCurveOnB6(), new com.powsybl.action.GeneratorActionBuilder().withId("up6")
                .withGeneratorId("B6-G").withActivePowerRelativeValue(false).withActivePowerValue(40).build(), false);
    }

    /** With a distribution, B6-G (out of OLF's participation at P 0, so out of the device's slack pack) joins it at 40 MW:
     *  the device cannot add a generator to its pack, and the pair stays on the CPU. */
    @Test
    void generatorActionStartingParticipationRunsOnCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = withCurveOnB6();
        String loadId = loadAtBus(network, "B9").getId();
        List<Contingency> contingencies = List.of(new Contingency(loadId, new com.powsybl.contingency.LoadContingency(loadId)));
        com.powsybl.action.Action action = new com.powsybl.action.GeneratorActionBuilder().withId("up6")
                .withGeneratorId("B6-G").withActivePowerRelativeValue(false).withActivePowerValue(40).build();
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sUp6",
                        com.powsybl.contingency.ContingencyContext.specificContingency(loadId),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("up6")));
        assertStrategyMatchesCpu(network, contingencies, strategies, List.of(action), 1, true);
    }

    /** IEEE 14 with a reactive capability curve on B6-G (P 0 at base): its max Q falls 1 MVAr per MW from 30 MVAr, room
     *  to hold B6's voltage at base, not after a 40 MW action. */
    private static Network withCurveOnB6() {
        Network network = IeeeCdfNetworkFactory.create14();
        com.powsybl.iidm.network.Generator g6 = network.getGenerator("B6-G");
        g6.setMinP(0).setMaxP(100);
        g6.newReactiveCapabilityCurve()
                .beginPoint().setP(0).setMinQ(-30).setMaxQ(30).endPoint()
                .beginPoint().setP(100).setMinQ(-70).setMaxQ(-70).endPoint()
                .add();
        return network;
    }

    /** B9's load lost, one strategy applying {@code action}: the contingency and the pair on the device. */
    private void assertCurveGeneratorActionMatchesCpu(Network network, com.powsybl.action.Action action, boolean distributedSlack) {
        String loadId = loadAtBus(network, "B9").getId();
        List<Contingency> contingencies = List.of(new Contingency(loadId,
                new com.powsybl.contingency.LoadContingency(loadId)));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sUp6",
                        com.powsybl.contingency.ContingencyContext.specificContingency(loadId),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of(action.getId())));
        assertStrategyMatchesCpu(network, contingencies, strategies, List.of(action), 2, distributedSlack);
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
     *  violation after B4's load loss, so AnyViolationCondition fails and an empty AllViolationCondition holds: once the
     *  contingency is handed back without a violation, only the combination its conditions select is solved. */
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
        // the contingency, then of the combinations (sBoth 1, sFirstOnly 2, sMixed 4, sNone 3) the one each strategy's
        // conditions select without a violation: sBoth's, sFirstOnly's first stage alone, sMixed's stages a and c, none
        // of sNone's
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 4);
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
        // freeze divergence must surface as a bus voltage mismatch on some pair, at OLF's default criteria.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        List<String> lineIds = IeeeCdfNetworkFactory.create14().getLineStream()
                .map(l -> l.getId()).toList();
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
        // Negative case: no limits => no violations => the condition never passes, NEITHER arm produces an operator
        // strategy result. The contingency is handed back without a violation, so its pair is not solved (count 1).
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
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 1);
    }

    /** A violation of the base case the contingency leaves as it was is not one OLF's condition sees: its filtering
     *  against the base case drops it (same limit, value within the flow proportional threshold), so the contingency
     *  selects no violation-conditioned action and the device does not solve the pair. ieee14 after B4's load loss, a
     *  current limit under the base current of a line whose current moves by less than 2 %. */
    @Test
    void unchangedBaseViolationDoesNotSelectTheStrategy() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        String loadId = loadAtBus(network, "B4").getId();
        limitLineUnderItsBaseCurrent(network, loadId, ratio -> Math.abs(ratio - 1) < 0.02);
        var cpu = assertStrategyMatchesCpu(network, List.of(new Contingency(loadId, new com.powsybl.contingency.LoadContingency(loadId))),
                strategiesOnViolation(loadId), List.of(genAction("up2", "B2-G", true, 10)), 1);
        assertEquals(0, cpu.getOperatorStrategyResults().size(), "the fixture needs OLF to filter the base violation out");
    }

    /** The same base violation made worse by the contingency, past the flow proportional threshold: OLF reports it, the
     *  condition holds and the device solves the pair. A line whose current rises by more than 20 %. */
    @Test
    void worsenedBaseViolationSelectsTheStrategy() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        String loadId = loadAtBus(network, "B4").getId();
        limitLineUnderItsBaseCurrent(network, loadId, ratio -> ratio > 1.2);
        var cpu = assertStrategyMatchesCpu(network, List.of(new Contingency(loadId, new com.powsybl.contingency.LoadContingency(loadId))),
                strategiesOnViolation(loadId), List.of(genAction("up2", "B2-G", true, 10)), 2);
        assertEquals(1, cpu.getOperatorStrategyResults().size(), "the fixture needs OLF to report the worsened violation");
    }

    private static List<com.powsybl.contingency.strategy.OperatorStrategy> strategiesOnViolation(String contingencyId) {
        return List.of(new com.powsybl.contingency.strategy.OperatorStrategy("sOnViolation",
                com.powsybl.contingency.ContingencyContext.specificContingency(contingencyId),
                new com.powsybl.contingency.strategy.condition.AnyViolationCondition(), List.of("up2")));
    }

    /** Gives the first line whose side-1 current, from OLF's load flows with and without load {@code loadId}, moves by a
     *  ratio {@code accepted} takes a permanent limit at 90 % of its base current: a violation of the base case. */
    private void limitLineUnderItsBaseCurrent(Network network, String loadId, java.util.function.DoublePredicate accepted) {
        var runner = new com.powsybl.loadflow.LoadFlow.Runner(
                new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory()));
        var lfp = new com.powsybl.loadflow.LoadFlowParameters().setDistributedSlack(true).setUseReactiveLimits(true);
        runner.run(network, lfp);
        java.util.Map<String, Double> base = new java.util.HashMap<>();
        network.getLineStream().forEach(line -> base.put(line.getId(), line.getTerminal1().getI()));
        var load = network.getLoad(loadId);
        load.getTerminal().disconnect();
        runner.run(network, lfp);
        var line = network.getLineStream()
                .filter(l -> base.get(l.getId()) > 1 && accepted.test(l.getTerminal1().getI() / base.get(l.getId())))
                .findFirst().orElseThrow();
        load.getTerminal().connect();
        line.newCurrentLimits1().setPermanentLimit(0.9 * base.get(line.getId())).add();
    }

    /** A violation the device screen does not see keeps the pair: the screen evaluates the branches' and buses' limits,
     *  the host checks the voltage-angle limits on every contingency. OLF's voltage-angle fixture: with NHV1_NHV2_2 out,
     *  the angle across NHV1_NHV2_1 reaches -7.5 degrees, below the limit's -7, and nothing else is violated. */
    @Test
    void voltageAngleViolationKeepsItsStrategyPair() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.EurostagFactory.fix(
                com.powsybl.iidm.network.test.EurostagTutorialExample1Factory.create());
        var line = network.getLine("NHV1_NHV2_1");
        network.newVoltageAngleLimit().setId("val").from(line.getTerminal1()).to(line.getTerminal2())
                .setHighLimit(7.0).setLowLimit(-7.0).add();
        List<Contingency> contingencies = List.of(new Contingency("NHV1_NHV2_2", new BranchContingency("NHV1_NHV2_2")));
        List<com.powsybl.action.Action> actions = List.of(new com.powsybl.action.LoadActionBuilder().withId("shed")
                .withLoadId("LOAD").withRelativeValue(false).withActivePowerValue(400).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sAngle",
                        com.powsybl.contingency.ContingencyContext.all(),
                        new com.powsybl.contingency.strategy.condition.AnyViolationCondition(), List.of("shed")));
        var cpu = assertStrategyMatchesCpu(network, contingencies, strategies, actions, 2);
        assertEquals(1, cpu.getOperatorStrategyResults().size(), "the fixture needs the angle violation to select the strategy");
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
