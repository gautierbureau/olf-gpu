/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.contingency.BranchContingency;
import com.powsybl.contingency.Contingency;
import com.powsybl.contingency.ContingencyContext;
import com.powsybl.contingency.LoadContingency;
import com.powsybl.iidm.network.Network;
import com.powsybl.iidm.network.Substation;
import com.powsybl.iidm.network.TopologyKind;
import com.powsybl.iidm.network.VoltageLevel;
import com.powsybl.iidm.network.extensions.ActivePowerControlAdder;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.openloadflow.CommonTestConfig;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.network.SlackBusSelectionMode;
import com.powsybl.openloadflow.sa.AbstractOpenSecurityAnalysisTest;
import com.powsybl.openloadflow.sa.OpenSecurityAnalysisProvider;
import com.powsybl.security.PostContingencyComputationStatus;
import com.powsybl.security.SecurityAnalysisParameters;
import com.powsybl.security.SecurityAnalysisResult;
import com.powsybl.security.monitor.StateMonitor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;

import static com.powsybl.loadflow.LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX;
import static com.powsybl.loadflow.LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_REMAINING_MARGIN;
import static com.powsybl.loadflow.LoadFlowResult.ComponentResult.Status.CONVERGED;
import static com.powsybl.loadflow.LoadFlowResult.ComponentResult.Status.FAILED;
import static com.powsybl.openloadflow.OpenLoadFlowParameters.SlackDistributionFailureBehavior.FAIL;
import static com.powsybl.openloadflow.OpenLoadFlowParameters.SlackDistributionFailureBehavior.LEAVE_ON_SLACK_BUS;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * A distribution that cannot place the slack mismatch is decided by {@code slackDistributionFailureBehavior}:
 * under FAIL OLF's {@code DistributedSlackOuterLoop} reports the load flow FAILED, under the default
 * LEAVE_ON_SLACK_BUS it converges with the mismatch left at the slack. The security analysis's base load
 * flow runs on the device driver, which passed a literal 0 for that behaviour where the batched
 * post-contingency solves pass the run's flag, so every FAIL configuration converged there. Three ways the
 * distribution has nothing to place: the participants saturate in the water-fill, nobody has any margin at
 * entry (REMAINING_MARGIN with everybody at its limit: OLF's participating set is empty, and the kernel's
 * gate read that as "already balanced" on every path), and nobody participates at all (the device's
 * slack-only path, which has no limits to decide a failure with).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuDistributionFailureBehaviorTest extends AbstractOpenSecurityAnalysisTest {

    GpuDistributionFailureBehaviorTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    @BeforeEach
    void checkGpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(),
                "GPU full-NR native lib not available (build it with native/build-gpu.sh + cuDSS)");
    }

    /** Three buses in a triangle, a generator on vl1 (the slack) and vl2, a load on vl3. */
    private static Network triangle(double minP, double maxP, double targetP1, double targetP2, double loadP) {
        Network network = Network.create("ds-failure-behavior", "code");
        Substation s = network.newSubstation().setId("s").add();
        for (String vl : List.of("vl1", "vl2", "vl3")) {
            VoltageLevel v = s.newVoltageLevel().setId(vl).setNominalV(400).setTopologyKind(TopologyKind.BUS_BREAKER).add();
            v.getBusBreakerView().newBus().setId(vl + "_b").add();
        }
        network.getVoltageLevel("vl1").newGenerator().setId("gvl1").setBus("vl1_b").setConnectableBus("vl1_b")
                .setMinP(minP).setMaxP(maxP).setTargetP(targetP1).setTargetV(400).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl2").newGenerator().setId("gvl2").setBus("vl2_b").setConnectableBus("vl2_b")
                .setMinP(minP).setMaxP(maxP).setTargetP(targetP2).setTargetV(400).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vl3").newLoad().setId("ld").setBus("vl3_b").setConnectableBus("vl3_b")
                .setP0(loadP).setQ0(50).add();
        for (String[] ends : new String[][] {{"vl1", "vl2"}, {"vl1", "vl3"}, {"vl2", "vl3"}}) {
            network.newLine().setId("l" + ends[0] + ends[1])
                    .setVoltageLevel1(ends[0]).setBus1(ends[0] + "_b").setConnectableBus1(ends[0] + "_b")
                    .setVoltageLevel2(ends[1]).setBus2(ends[1] + "_b").setConnectableBus2(ends[1] + "_b")
                    .setR(0.5).setX(10).setG1(0).setB1(0).setG2(0).setB2(0).add();
        }
        return network;
    }

    /** Both generators at their maximum and a load above their sum: the slack must ADD power that nobody can take. */
    private static Network everyoneAtMaxP() {
        return triangle(0, 100, 100, 100, 250);
    }

    /** Both generators at their minimum, balanced in the base: losing the load asks everybody to go DOWN. */
    private static Network everyoneAtMinP() {
        return triangle(50, 200, 50, 50, 100);
    }

    private static LoadFlowParameters params(LoadFlowParameters.BalanceType balance,
                                             OpenLoadFlowParameters.SlackDistributionFailureBehavior behavior) {
        LoadFlowParameters p = new LoadFlowParameters()
                .setDistributedSlack(true)
                .setUseReactiveLimits(false)
                .setBalanceType(balance);
        OpenLoadFlowParameters.create(p)
                .setSlackBusSelectionMode(SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl1_0"))
                .setSlackDistributionFailureBehavior(behavior);
        return p;
    }

    private SecurityAnalysisResult run(boolean gpu, Network network, List<Contingency> contingencies, LoadFlowParameters lfp) {
        SecurityAnalysisParameters sap = new SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        Set<String> allVls = network.getVoltageLevelStream().map(VoltageLevel::getId).collect(java.util.stream.Collectors.toSet());
        List<StateMonitor> monitors = List.of(new StateMonitor(ContingencyContext.all(), Set.of(), allVls, Set.of()));
        securityAnalysisProvider = gpu
                ? new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>())
                : new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            return runSecurityAnalysis(network, contingencies, monitors, sap);
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    /** The BASE load flow: LEAVE_ON_SLACK_BUS converges on both arms with the device answering, FAIL fails on both. */
    private void assertBaseFailureBehaviorFollowsOlf(Network network, LoadFlowParameters.BalanceType balance) {
        List<Contingency> contingencies = List.of(new Contingency("lvl1vl2", new BranchContingency("lvl1vl2")));
        long fallbacks = GpuNewtonRaphson.CPU_FALLBACK_COUNT.get();
        assertEquals(CONVERGED, run(false, network, contingencies, params(balance, LEAVE_ON_SLACK_BUS)).getPreContingencyResult().getStatus(),
                "OLF leaves the mismatch at the slack");
        assertEquals(CONVERGED, run(true, network, contingencies, params(balance, LEAVE_ON_SLACK_BUS)).getPreContingencyResult().getStatus(),
                "device: same");
        assertEquals(fallbacks, GpuNewtonRaphson.CPU_FALLBACK_COUNT.get(),
                "the LEAVE_ON_SLACK_BUS base must be the device driver's own answer");
        assertEquals(FAILED, run(false, network, contingencies, params(balance, FAIL)).getPreContingencyResult().getStatus(),
                "OLF: the failure behaviour FAIL fails the base load flow");
        assertEquals(FAILED, run(true, network, contingencies, params(balance, FAIL)).getPreContingencyResult().getStatus(),
                "device: same");
    }

    @Test
    void baseSaturatedParticipantsFailLikeOlf() {
        assertBaseFailureBehaviorFollowsOlf(everyoneAtMaxP(), PROPORTIONAL_TO_GENERATION_P_MAX);
    }

    @Test
    void baseNoMarginAtEntryFailsLikeOlf() {
        assertBaseFailureBehaviorFollowsOlf(everyoneAtMaxP(), PROPORTIONAL_TO_GENERATION_REMAINING_MARGIN);
    }

    @Test
    void baseNoParticipantFailsLikeOlf() {
        Network network = everyoneAtMaxP();
        for (String g : List.of("gvl1", "gvl2")) {
            network.getGenerator(g).newExtension(ActivePowerControlAdder.class).withParticipate(false).add();
        }
        assertBaseFailureBehaviorFollowsOlf(network, PROPORTIONAL_TO_GENERATION_P_MAX);
    }

    /** A POST-CONTINGENCY solve with nobody able to move: losing the whole load asks every generator to
     *  go down from its minimum. REMAINING_MARGIN: OLF's participating set is empty and the behaviour
     *  decides; the device's gate read a zero participant share as "already balanced" and converged with
     *  the mismatch at the slack whatever the behaviour. Under FAIL the scenario must leave the device
     *  (the CPU replays it and fails it); under LEAVE_ON_SLACK_BUS the device's answer is OLF's. */
    @Test
    void contingencyNoMarginAtEntryFailsLikeOlf() {
        Network network = everyoneAtMinP();
        List<Contingency> contingencies = List.of(new Contingency("ld", new LoadContingency("ld")));
        SecurityAnalysisResult cpu = run(false, network, contingencies, params(PROPORTIONAL_TO_GENERATION_REMAINING_MARGIN, LEAVE_ON_SLACK_BUS));
        assertEquals(CONVERGED, cpu.getPreContingencyResult().getStatus());
        assertEquals(PostContingencyComputationStatus.CONVERGED, cpu.getPostContingencyResults().get(0).getStatus(),
                "OLF leaves the mismatch at the slack");
        GpuBatchScopeCensus.reset();
        SecurityAnalysisResult gpu = run(true, network, contingencies, params(PROPORTIONAL_TO_GENERATION_REMAINING_MARGIN, LEAVE_ON_SLACK_BUS));
        assertEquals(PostContingencyComputationStatus.CONVERGED, gpu.getPostContingencyResults().get(0).getStatus());
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "the load outage must batch (or the test proves nothing); census=" + GpuBatchScopeCensus.snapshot());
        var cpuBuses = cpu.getPostContingencyResults().get(0).getNetworkResult().getBusResults();
        var gpuBuses = gpu.getPostContingencyResults().get(0).getNetworkResult().getBusResults();
        assertEquals(cpuBuses.size(), gpuBuses.size());
        for (int b = 0; b < cpuBuses.size(); b++) {
            assertEquals(cpuBuses.get(b).getV(), gpuBuses.get(b).getV(), 1e-12, "bus voltage " + cpuBuses.get(b).getBusId());
        }

        cpu = run(false, network, contingencies, params(PROPORTIONAL_TO_GENERATION_REMAINING_MARGIN, FAIL));
        assertEquals(CONVERGED, cpu.getPreContingencyResult().getStatus());
        assertEquals(PostContingencyComputationStatus.FAILED, cpu.getPostContingencyResults().get(0).getStatus(),
                "OLF: the failure behaviour FAIL fails the post-contingency load flow");
        GpuBatchScopeCensus.reset();
        gpu = run(true, network, contingencies, params(PROPORTIONAL_TO_GENERATION_REMAINING_MARGIN, FAIL));
        assertEquals(PostContingencyComputationStatus.FAILED, gpu.getPostContingencyResults().get(0).getStatus(), "device: same");
        assertEquals(1L, GpuBatchScopeCensus.snapshot().getOrDefault("RUNTIME:status=-2", 0L),
                "the device must have rejected the scenario itself; census=" + GpuBatchScopeCensus.snapshot());
    }

    /** A POST-CONTINGENCY slack-only distribution beyond the slack's rooms: the slack generator is the only
     *  participant, at its minimum, and losing the load asks it to go down. OLF's water-fill places what the
     *  slack generator's rooms allow and the behaviour decides on the rest. The device's slack-only check used
     *  to place everything, unbounded: under FAIL the scenario converged on the device where OLF fails it, and
     *  under LEAVE_ON_SLACK_BUS it reported more distributed power than OLF. */
    @Test
    void contingencySlackOnlyBeyondItsRoomsFailsLikeOlf() {
        Network network = everyoneAtMinP();
        network.getGenerator("gvl2").newExtension(ActivePowerControlAdder.class).withParticipate(false).add();
        List<Contingency> contingencies = List.of(new Contingency("ld", new LoadContingency("ld")));
        SecurityAnalysisResult cpu = run(false, network, contingencies, params(PROPORTIONAL_TO_GENERATION_P_MAX, LEAVE_ON_SLACK_BUS));
        assertEquals(PostContingencyComputationStatus.CONVERGED, cpu.getPostContingencyResults().get(0).getStatus(),
                "OLF leaves the mismatch at the slack");
        GpuBatchScopeCensus.reset();
        SecurityAnalysisResult gpu = run(true, network, contingencies, params(PROPORTIONAL_TO_GENERATION_P_MAX, LEAVE_ON_SLACK_BUS));
        assertEquals(PostContingencyComputationStatus.CONVERGED, gpu.getPostContingencyResults().get(0).getStatus());
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "the load outage must batch (or the test proves nothing); census=" + GpuBatchScopeCensus.snapshot());
        assertEquals(cpu.getPostContingencyResults().get(0).getDistributedActivePower(),
                gpu.getPostContingencyResults().get(0).getDistributedActivePower(), 1e-9,
                "what the slack generator's rooms let OLF place");
        var cpuBuses = cpu.getPostContingencyResults().get(0).getNetworkResult().getBusResults();
        var gpuBuses = gpu.getPostContingencyResults().get(0).getNetworkResult().getBusResults();
        for (int b = 0; b < cpuBuses.size(); b++) {
            assertEquals(cpuBuses.get(b).getV(), gpuBuses.get(b).getV(), 1e-12, "bus voltage " + cpuBuses.get(b).getBusId());
        }

        cpu = run(false, network, contingencies, params(PROPORTIONAL_TO_GENERATION_P_MAX, FAIL));
        assertEquals(CONVERGED, cpu.getPreContingencyResult().getStatus());
        assertEquals(PostContingencyComputationStatus.FAILED, cpu.getPostContingencyResults().get(0).getStatus(),
                "OLF: the failure behaviour FAIL fails the post-contingency load flow");
        GpuBatchScopeCensus.reset();
        gpu = run(true, network, contingencies, params(PROPORTIONAL_TO_GENERATION_P_MAX, FAIL));
        assertEquals(CONVERGED, gpu.getPreContingencyResult().getStatus());
        assertEquals(PostContingencyComputationStatus.FAILED, gpu.getPostContingencyResults().get(0).getStatus(), "device: same");
        assertEquals(1L, GpuBatchScopeCensus.snapshot().getOrDefault("RUNTIME:status=-2", 0L),
                "the device must have rejected the scenario itself; census=" + GpuBatchScopeCensus.snapshot());
    }
}
