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
import com.powsybl.contingency.LoadContingency;
import com.powsybl.iidm.network.Network;
import com.powsybl.iidm.network.ThreeSides;
import com.powsybl.iidm.network.TwoSides;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.openloadflow.CommonTestConfig;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuAutomationSystemOuterLoop;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.GpuBatchScopeCensus;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.network.AutomationSystemNetworkFactory;
import com.powsybl.openloadflow.sa.OpenSecurityAnalysisProvider;
import com.powsybl.security.SecurityAnalysisParameters;
import com.powsybl.security.SecurityAnalysisResult;
import com.powsybl.security.monitor.StateMonitor;
import com.powsybl.security.results.BranchResult;
import com.powsybl.security.results.BusResult;
import com.powsybl.security.results.PostContingencyResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * AutomationSystem in the batched security analysis: the device's AUTOMATION_SYSTEM phase opens or closes a contingency's
 * operated branch by its bit in the scenario's mask, and the analysis gives that switch to the network before it builds
 * the contingency's results, then takes it back. A contingency that touches a system's branches, or that the systems'
 * openings could island, goes to the CPU. Every result must match plain OLF: statuses, violations, the branch results (none
 * for a branch a system opened) and the bus voltages.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuSaAutomationSystemTest extends AbstractGpuSaTest {

    private static final double TOLERANCE = 1e-9;

    GpuSaAutomationSystemTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    @BeforeEach
    void checkGpu() {
        assumeTrue(com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
    }

    private static List<Contingency> branchContingencies(String... ids) {
        return java.util.Arrays.stream(ids).map(id -> new Contingency(id, new BranchContingency(id))).toList();
    }

    /** Run plain OLF and the GPU security analysis on two copies and compare every result; returns the GPU's. */
    private SecurityAnalysisResult assertAutomationSystemMatchesCpu(Supplier<Network> networkSupplier, List<Contingency> contingencies) {
        LoadFlowParameters loadFlowParameters = new LoadFlowParameters();
        OpenLoadFlowParameters.create(loadFlowParameters).setSimulateAutomationSystems(true);
        SecurityAnalysisParameters parameters = new SecurityAnalysisParameters().setLoadFlowParameters(loadFlowParameters);
        Network cpuNetwork = networkSupplier.get();
        List<StateMonitor> monitors = createNetworkMonitors(cpuNetwork);
        SecurityAnalysisResult cpu = runSecurityAnalysis(cpuNetwork, contingencies, monitors, parameters);
        GpuBatchScopeCensus.reset();
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        SecurityAnalysisResult gpu;
        try {
            gpu = runSecurityAnalysis(networkSupplier.get(), contingencies, monitors, parameters);
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
        assertEquals(cpu.getPreContingencyResult().getStatus(), gpu.getPreContingencyResult().getStatus(), "pre-contingency status");
        assertEquals(cpu.getPostContingencyResults().size(), gpu.getPostContingencyResults().size());
        for (PostContingencyResult cpuResult : cpu.getPostContingencyResults()) {
            String id = cpuResult.getContingency().getId();
            PostContingencyResult gpuResult = getPostContingencyResult(gpu, id);
            assertEquals(cpuResult.getStatus(), gpuResult.getStatus(), id + ": status");
            assertEquals(cpuResult.getLimitViolationsResult().getLimitViolations().stream().map(v -> v.getSubjectId() + " " + v.getLimitType()).toList(),
                    gpuResult.getLimitViolationsResult().getLimitViolations().stream().map(v -> v.getSubjectId() + " " + v.getLimitType()).toList(),
                    id + ": violations");
            assertEquals(cpuResult.getNetworkResult().getBranchResults().stream().map(BranchResult::getBranchId).toList(),
                    gpuResult.getNetworkResult().getBranchResults().stream().map(BranchResult::getBranchId).toList(),
                    id + ": the branches with a result");
            double worstI = 0;
            double worstFlow = 0;
            for (BranchResult branchResult : cpuResult.getNetworkResult().getBranchResults()) {
                BranchResult gpuBranchResult = gpuResult.getNetworkResult().getBranchResult(branchResult.getBranchId());
                worstI = Math.max(worstI, Math.max(Math.abs(branchResult.getI1() - gpuBranchResult.getI1()),
                        Math.abs(branchResult.getI2() - gpuBranchResult.getI2())));
                // OLF's vectorized system evaluates P and Q on its branch vector, which a state update leaves stale on a
                // branch disabled at that time
                worstFlow = Math.max(worstFlow, Math.max(Math.max(Math.abs(branchResult.getP1() - gpuBranchResult.getP1()),
                                Math.abs(branchResult.getQ1() - gpuBranchResult.getQ1())),
                        Math.max(Math.abs(branchResult.getP2() - gpuBranchResult.getP2()), Math.abs(branchResult.getQ2() - gpuBranchResult.getQ2()))));
            }
            double worstV = 0;
            for (BusResult busResult : cpuResult.getNetworkResult().getBusResults()) {
                worstV = Math.max(worstV, Math.abs(busResult.getV() - gpuResult.getNetworkResult().getBusResult(busResult.getBusId()).getV()));
            }
            System.out.printf("AutomationSystem SA GPU vs CPU [%s]: %s, max dI %.2e A, dP/dQ %.2e MW/MVar, dV %.2e kV%n", id,
                    gpuResult.getStatus(), worstI, worstFlow, worstV);
            assertEquals(0, worstI, TOLERANCE, id + ": branch currents");
            assertEquals(0, worstFlow, TOLERANCE, id + ": branch flows");
            assertEquals(0, worstV, TOLERANCE, id + ": bus voltages");
        }
        return gpu;
    }

    /**
     * The outages of tr2, l34b and l3b4 overload l34 (296, 174 and 154 A), and its system opens l33p above 150 A; those of
     * l12, l12b and tr1 do not. Every one solves on the device, the three switches with it; the outages of l34, which the
     * system monitors, and of l33p, which it operates, go to the CPU.
     */
    @Test
    void branchOutagesOpeningABranchOnTheDevice() {
        long switchedBefore = GpuAutomationSystemOuterLoop.DEVICE_SWITCHED_BRANCHES.sum();
        SecurityAnalysisResult gpu = assertAutomationSystemMatchesCpu(AutomationSystemNetworkFactory::createMeshedWithBranchTripping,
                branchContingencies("tr2", "l34b", "l3b4", "l12", "l12b", "tr1", "l34", "l33p"));
        assertEquals(6, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "six outages on the device");
        assertEquals(Set.of("l34", "l33p"), GpuBatchScopeCensus.planRefusedIds(), "the system's own branches on the CPU");
        assertEquals("PLAN:AUTOMATION_SYSTEM_BRANCH_OUTAGE", GpuBatchScopeCensus.planRefusalTag("l34"));
        assertEquals("PLAN:AUTOMATION_SYSTEM_BRANCH_OUTAGE", GpuBatchScopeCensus.planRefusalTag("l33p"));
        assertEquals(3, GpuAutomationSystemOuterLoop.DEVICE_SWITCHED_BRANCHES.sum() - switchedBefore, "l33p opened on the device three times");
        for (String id : List.of("tr2", "l34b", "l3b4")) {
            assertNull(getPostContingencyResult(gpu, id).getNetworkResult().getBranchResult("l33p"), id + ": l33p opened");
        }
        assertNotNull(getPostContingencyResult(gpu, "l12").getNetworkResult().getBranchResult("l33p"), "l12: l33p closed");
    }

    /** createMeshedWithBranchTripping with l33p open and a system that closes it above 140 A at l12's side 1. */
    private static Network meshedWithBranchToClose() {
        Network network = AutomationSystemNetworkFactory.createMeshedWithBranchTripping();
        network.getOverloadManagementSystem("l34_opens_l33p").remove();
        network.getLine("l33p").getTerminal1().disconnect();
        network.getLine("l33p").getTerminal2().disconnect();
        network.getSubstation("s1").newOverloadManagementSystem()
                .setId("l12_closes_l33p")
                .setEnabled(true)
                .setMonitoredElementId("l12")
                .setMonitoredElementSide(ThreeSides.ONE)
                .newBranchTripping()
                .setKey("l33p key")
                .setBranchToOperateId("l33p")
                .setSideToOperate(TwoSides.TWO)
                .setOpenAction(false)
                .setCurrentLimit(140.)
                .add()
                .add();
        return network;
    }

    /**
     * l33p is open in the base, which packs it as a base-open branch, and l12's system closes it above 140 A: the outages of
     * l12b (208 A), tr1 and l3b4 (149 A) close it on the device, those of tr2, l34 and l34b leave it open, and l12's goes
     * to the CPU. The load outage beside them changes nothing. A closed branch is enabled before the state is injected, so
     * that its flows are computed with the others.
     */
    @Test
    void branchOutagesClosingABranchOnTheDevice() {
        long switchedBefore = GpuAutomationSystemOuterLoop.DEVICE_SWITCHED_BRANCHES.sum();
        List<Contingency> contingencies = new java.util.ArrayList<>(branchContingencies("l12b", "tr1", "l3b4", "tr2", "l34", "l34b", "l12"));
        contingencies.add(new Contingency("ld3", new LoadContingency("ld3")));
        SecurityAnalysisResult gpu = assertAutomationSystemMatchesCpu(GpuSaAutomationSystemTest::meshedWithBranchToClose, contingencies);
        assertEquals(7, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "seven outages on the device");
        assertEquals(Set.of("l12"), GpuBatchScopeCensus.planRefusedIds(), "the monitored branch's outage on the CPU");
        assertEquals(3, GpuAutomationSystemOuterLoop.DEVICE_SWITCHED_BRANCHES.sum() - switchedBefore, "l33p closed on the device three times");
        for (String id : List.of("l12b", "tr1", "l3b4")) {
            assertNotNull(getPostContingencyResult(gpu, id).getNetworkResult().getBranchResult("l33p"), id + ": l33p closed");
        }
        assertNull(getPostContingencyResult(gpu, "tr2").getNetworkResult().getBranchResult("l33p"), "tr2: l33p open");
    }

    /**
     * As above, l33p with a 1 A limit: open in the base, it violates only where the device closes it. The screen skips a
     * base-open branch except in the scenarios that close it (GpuScreenScenario.IN), and the host no longer scans the
     * switched branches in full: the violation is found by the screen alone.
     */
    @Test
    void branchClosedOnTheDeviceViolatesItsLimit() {
        List<Contingency> contingencies = branchContingencies("l12b", "tr1", "l3b4", "tr2");
        SecurityAnalysisResult gpu = assertAutomationSystemMatchesCpu(() -> {
            Network network = meshedWithBranchToClose();
            network.getLine("l33p").getOrCreateSelectedOperationalLimitsGroup1().newCurrentLimits().setPermanentLimit(1).add();
            return network;
        }, contingencies);
        for (String id : List.of("l12b", "tr1", "l3b4")) {
            assertTrue(getPostContingencyResult(gpu, id).getLimitViolationsResult().getLimitViolations().stream()
                    .anyMatch(v -> v.getSubjectId().equals("l33p")), id + ": l33p closed over its limit");
        }
    }

    /**
     * With l3b4 and tr1 out, b3 hangs on l33p alone: the outage leaves no island, but the system opening l33p would, which
     * the device does not represent. That contingency goes to the CPU, l3b4's alone stays on the device.
     */
    @Test
    void outageThatAnOpeningWouldIslandGoesToTheCpu() {
        List<Contingency> contingencies = List.of(
                new Contingency("l3b4", new BranchContingency("l3b4")),
                new Contingency("l3b4+tr1", new BranchContingency("l3b4"), new BranchContingency("tr1")));
        assertAutomationSystemMatchesCpu(AutomationSystemNetworkFactory::createMeshedWithBranchTripping, contingencies);
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "l3b4's outage on the device");
        assertEquals(Set.of("l3b4+tr1"), GpuBatchScopeCensus.planRefusedIds());
        assertEquals("PLAN:AUTOMATION_SYSTEM_ISLAND_IF_OPENED", GpuBatchScopeCensus.planRefusalTag("l3b4+tr1"));
    }

    /**
     * At 100 A, l34's system opens l33p in the pre-contingency load flow already (106 A): the device switches it there, and
     * every contingency starts from l33p open, packed as a base-open branch whose system has nothing left to open.
     */
    @Test
    void branchOpenedByThePreContingencyLoadFlow() {
        long switchedBefore = GpuAutomationSystemOuterLoop.DEVICE_SWITCHED_BRANCHES.sum();
        SecurityAnalysisResult gpu = assertAutomationSystemMatchesCpu(() -> {
            Network network = AutomationSystemNetworkFactory.createMeshedWithBranchTripping();
            network.getOverloadManagementSystem("l34_opens_l33p").getTrippings().getFirst().setCurrentLimit(100.);
            return network;
        }, branchContingencies("tr2", "l34b", "l3b4", "l12", "l12b", "tr1"));
        assertEquals(6, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "every outage on the device");
        assertEquals(1, GpuAutomationSystemOuterLoop.DEVICE_SWITCHED_BRANCHES.sum() - switchedBefore, "l33p opened once, in the base");
        assertNull(gpu.getPreContingencyResult().getNetworkResult().getBranchResult("l33p"), "l33p opened in the base");
    }
}
