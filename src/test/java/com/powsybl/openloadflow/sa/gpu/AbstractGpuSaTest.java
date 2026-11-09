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
import com.powsybl.openloadflow.ac.solver.gpu.GpuLoadFlowParameters;
import com.powsybl.openloadflow.ac.solver.gpu.GpuReorderingAlgorithm;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.GpuBatchScopeCensus;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.sa.AbstractOpenSecurityAnalysisTest;
import com.powsybl.openloadflow.sa.OpenSecurityAnalysisParameters;
import com.powsybl.openloadflow.sa.OpenSecurityAnalysisProvider;
import com.powsybl.openloadflow.util.gpu.GpuTrace;
import com.powsybl.security.SecurityAnalysisResult;

import java.util.List;

import static com.powsybl.openloadflow.sa.gpu.GpuSaTestNetworks.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The GPU-batched AC security analysis ({@link GpuBatchedAcSecurityAnalysis}), selected via the
 * {@link GpuSecurityAnalysisProvider}, must produce the same results as the standard
 * {@link com.powsybl.openloadflow.sa.AcSecurityAnalysis}: enabling it never changes results, it only accelerates what
 * the device can solve. The comparisons every area's tests share.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
abstract class AbstractGpuSaTest extends AbstractOpenSecurityAnalysisTest {

    AbstractGpuSaTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    /** The scope census is the process's: a test reading it must find its own runs only. The area interchange control
     *  tests read their plan refusals without a reset, and run after the automation system tests they found that class's
     *  refusals there too. */
    @org.junit.jupiter.api.BeforeEach
    void resetScopeCensus() {
        GpuBatchScopeCensus.reset();
    }

    /** {@code check} with its scenarios in a bank of 2 slots, each refilled as soon as its scenario ends: a scenario installed
     *  while another still solves must find its slot afresh and leave the other as it was. The check must refill at least
     *  once. */
    void refilled(Runnable check) {
        com.powsybl.openloadflow.ac.solver.gpu.GpuBatchContextCache.forceChunkSizeForTests(2);
        com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchedSecurityAnalysisSolver.refillAtForTests(1);
        int refills;
        try {
            check.run();
            refills = com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchedSecurityAnalysisSolver.refillsForTests();
        } finally {
            com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchedSecurityAnalysisSolver.refillAtForTests(null);
            com.powsybl.openloadflow.ac.solver.gpu.GpuBatchContextCache.forceChunkSizeForTests(null);
        }
        org.junit.jupiter.api.Assertions.assertTrue(refills > 0, "a scenario must be installed while another still solves");
    }

    void assertSplitGroupMatchesCpu(int sections) {
        assertSplitGroupMatchesCpu(sections, java.util.Set.of());
    }

    void assertSplitGroupMatchesCpu(int sections, java.util.Set<Integer> switchedToPqGenerators) {
        assertSplitGroupMatchesCpu(sections, switchedToPqGenerators, java.util.Set.of());
    }

    void assertSplitGroupMatchesCpu(int sections, java.util.Set<Integer> switchedToPqGenerators, java.util.Set<Integer> noControl) {
        assertSplitGroupMatchesCpu(sections, switchedToPqGenerators, noControl, java.util.Set.of());
    }

    void assertSplitGroupMatchesCpu(int sections, java.util.Set<Integer> switchedToPqGenerators, java.util.Set<Integer> noControl,
                                            java.util.Set<Integer> noLine) {
        List<Contingency> ctgs = new java.util.ArrayList<>();
        for (int k = 0; k < sections; k++) {
            ctgs.add(new Contingency("bus-" + k, new com.powsybl.contingency.BusbarSectionContingency("bbs" + k)));
        }
        GpuBatchScopeCensus.reset();
        assertMergedPilotLossMatchesCpu(splitGroupNetwork(sections, switchedToPqGenerators, noControl, noLine), ctgs, sections);
        assertEquals(0L, GpuBatchScopeCensus.snapshot().getOrDefault("HOST:OVERRIDE_CONFLICT", 0L).longValue(),
                "a sound plan writes every row's target once");
    }

    void assertRemoteMergedSectionsMatchCpu(int sections, java.util.Set<String> refused) {
        assertRemoteMergedSectionsMatchCpu(sections, refused, "PLAN:VC_REMERGE_REMOTE_SUBGROUP");
    }

    void assertRemoteMergedSectionsMatchCpu(int sections, java.util.Set<String> refused, String refusalTag) {
        List<Contingency> ctgs = new java.util.ArrayList<>();
        for (int k = 0; k < sections; k++) {
            ctgs.add(new Contingency("bus-" + k, new com.powsybl.contingency.BusbarSectionContingency("bbs" + k)));
        }
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters();
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)       // the slack away from the lost sections
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl0"));
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(remoteMergedSectionsNetwork(sections), ctgs, 1e-9, lfp, true);
        // A section whose loss re-merges the survivors onto a REMOTELY controlled cluster is hosted on the device
        // once the base pilot is lost (the new pilot's balance rides the first free controller through its
        // dependent-pilot pool); a cluster splitting off BEHIND a surviving main is still the CPU's, tagged.
        assertEquals(refused, GpuBatchScopeCensus.planRefusedIds());
        refused.forEach(id -> assertEquals(refusalTag, GpuBatchScopeCensus.planRefusalTag(id)));
        assertEquals(sections - refused.size(), GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** The post-contingency voltage of a monitored voltage level's bus, CPU security analysis. */
    double cpuPostContingencyV(Network network, List<Contingency> contingencies,
                                       com.powsybl.loadflow.LoadFlowParameters lfp, String voltageLevelId) {
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        var monitors = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(), java.util.Set.of(voltageLevelId), java.util.Set.of()));
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, contingencies, monitors, sap);
        assertEquals(com.powsybl.security.PostContingencyComputationStatus.CONVERGED, cpu.getPostContingencyResults().get(0).getStatus());
        return cpu.getPostContingencyResults().get(0).getNetworkResult().getBusResults().get(0).getV();
    }

    void assertMergedPilotLossMatchesCpu(Network network, List<Contingency> ctgs) {
        assertMergedPilotLossMatchesCpu(network, ctgs, 1);
    }

    void assertMergedPilotLossMatchesCpu(Network network, List<Contingency> ctgs, int expectedOnDevice) {
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl0_b"));           // breakers mode: bus/breaker ids
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(expectedOnDevice, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "every listed bus loss must batch on-device (or the test proves nothing)");
    }

    void runTvcGeneratorUnderLevel(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode mode) {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.FourBusNetworkFactory.createWithSeveralTransformerVoltageControls();
        com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
        String bis = l0.getId() + "_BIS";
        network.newLine().setId(bis)
                .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
        List<Contingency> ctgs = List.of(new Contingency(bis, new BranchContingency(bis)),
                new Contingency(l0.getId(), new BranchContingency(l0.getId())));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setTransformerVoltageControlOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp).setTransformerVoltageControlMode(mode);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "generators under the controlled level must not gate " + mode);
    }

    void assertNoEnabledControllerCornerSemantics(Network network, List<Contingency> ctgs) {
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(remoteReactiveLimitParameters());
        List<com.powsybl.security.monitor.StateMonitor> monitors = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(), java.util.Set.of("vl4"), java.util.Set.of()));
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, ctgs, monitors, sap);
        double targetV = network.getGenerator("g1").getTargetV();
        double basePilotV = cpu.getPreContingencyResult().getNetworkResult().getBusResults().stream()
                .filter(b -> "b4".equals(b.getBusId())).findFirst().orElseThrow().getV();
        assertTrue(Math.abs(basePilotV - targetV) < 0.05,
                "the BASE must keep group control (pilot " + basePilotV + " vs target " + targetV
                        + ") — the corner under test is the IN-FLIGHT vEq deactivation");
        double pilotV = cpu.getPostContingencyResults().get(0).getNetworkResult().getBusResults().stream()
                .filter(b -> "b4".equals(b.getBusId())).findFirst().orElseThrow().getV();
        assertTrue(Math.abs(pilotV - targetV) > 0.1,
                "the scenario must LOSE group control (pilot " + pilotV + " vs target " + targetV
                        + ") — otherwise it does not exercise the no-enabled-controller corner");
    }

    /** The whole-run-fallback contract: the batch must NOT engage (the scope routed the run to the
     *  stock CPU path) and the results must equal the plain CPU SA bit-for-bit. */
    void assertGpuFallsBackWholeRunAndMatchesCpu(Network network, List<Contingency> ctgs,
                                                         com.powsybl.loadflow.LoadFlowParameters lfp) {
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        java.util.Set<String> allVls = network.getVoltageLevelStream().map(vl -> vl.getId())
                .collect(java.util.stream.Collectors.toSet());
        List<com.powsybl.security.monitor.StateMonitor> monitors = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(), allVls, java.util.Set.of()));
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, ctgs, monitors, sap);
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            SecurityAnalysisResult gpu = runSecurityAnalysis(network, ctgs, monitors, sap);
            assertTrue(GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get() <= 0,
                    "the scope gate must route this run to the CPU (batching it was 3.7e-3 kV wrong)");
            assertEquals(cpu.getPostContingencyResults().size(), gpu.getPostContingencyResults().size());
            for (int i = 0; i < cpu.getPostContingencyResults().size(); i++) {
                var cpuBuses = cpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                var gpuBuses = gpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                assertEquals(cpuBuses.size(), gpuBuses.size());
                for (int b = 0; b < cpuBuses.size(); b++) {
                    org.junit.jupiter.api.Assertions.assertEquals(cpuBuses.get(b).getV(), gpuBuses.get(b).getV(), 0.0,
                            "bus " + cpuBuses.get(b).getBusId());
                }
            }
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    /** Whether the run orders the Jacobian by cuDSS's nested dissection (LIBRARY_DEFAULT orders these matrices the
     *  same way). Under it cuDSS rounds differently from one factorization to the next, and two tests take a looser
     *  tolerance there by exception; every other test holds its tolerance under both orderings. */
    static boolean nestedDissectionOrdering(com.powsybl.loadflow.LoadFlowParameters lfp) {
        GpuReorderingAlgorithm ordering = GpuLoadFlowParameters.get(lfp).getReorderingAlgorithm();
        return ordering == GpuReorderingAlgorithm.NESTED_DISSECTION || ordering == GpuReorderingAlgorithm.LIBRARY_DEFAULT;
    }

    void assertGpuMatchesCpu(Network network, List<Contingency> contingencies) {
        assertGpuMatchesCpu(network, contingencies, 1e-12);
    }

    void assertGpuMatchesCpu(Network network, List<Contingency> contingencies, double vToleranceKv) {
        assertGpuMatchesCpu(network, contingencies, vToleranceKv, true);
    }

    void assertGpuMatchesCpu(Network network, List<Contingency> contingencies, double vToleranceKv, boolean distributedSlack) {
        assertGpuMatchesCpu(network, contingencies, vToleranceKv, distributedSlack, true);
    }

    void assertGpuMatchesCpu(Network network, List<Contingency> contingencies, double vToleranceKv,
                                     boolean distributedSlack, boolean reactiveLimits) {
        assertGpuMatchesCpu(network, contingencies, vToleranceKv, new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(distributedSlack).setUseReactiveLimits(reactiveLimits));
    }

    void assertGpuMatchesCpu(Network network, List<Contingency> contingencies, double vToleranceKv,
                                     com.powsybl.loadflow.LoadFlowParameters lfp) {
        assertGpuMatchesCpu(network, contingencies, vToleranceKv, lfp, true);
    }

    /** Same state-level comparison but WITHOUT the engagement assert — for contracts where the
     *  GPU provider is EXPECTED to route the scenarios back to the CPU (results must still match). */
    void assertGpuMatchesCpuNoEngage(Network network, List<Contingency> contingencies, double vToleranceKv,
                                             com.powsybl.loadflow.LoadFlowParameters lfp) {
        assertGpuMatchesCpu(network, contingencies, vToleranceKv, lfp, false);
    }

    void assertGpuMatchesCpu(Network network, List<Contingency> contingencies, double vToleranceKv,
                                     com.powsybl.loadflow.LoadFlowParameters lfp, boolean requireEngage) {
        String convEps = System.getProperty("olf.gpu.test.convEps", System.getenv("OLF_GPU_DBG_CONV_EPS"));
        if (convEps != null) {
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)   // create() ATTACHES (get() would detach!)
                    .setNewtonRaphsonConvEpsPerEq(Double.parseDouble(convEps));
        }
        String slackTol = System.getProperty("olf.gpu.test.slackTol", System.getenv("OLF_GPU_DBG_SLACK_TOL"));
        if (slackTol != null) {
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                    .setSlackBusPMaxMismatch(Double.parseDouble(slackTol));
        }
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        // STATE-LEVEL comparison (task #21): monitor every voltage level — status/violation-count
        // equality is blind on limit-free test networks (a wrong Q injection still "matches").
        java.util.Set<String> allVls = network.getVoltageLevelStream().map(vl -> vl.getId())
                .collect(java.util.stream.Collectors.toSet());
        java.util.Set<String> monBranches = "1".equals(GpuTrace.DBG_BUSDUMP.value())
                ? network.getBranchStream().map(b -> b.getId()).collect(java.util.stream.Collectors.toSet())
                : java.util.Set.of();
        List<com.powsybl.security.monitor.StateMonitor> monitors = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), monBranches, allVls, java.util.Set.of()));
        com.powsybl.commons.report.ReportNode plainCpuReport = "1".equals(System.getenv("OLF_GPU_DBG_REPORT"))
                ? com.powsybl.commons.report.ReportNode.newRootReportNode()
                        .withResourceBundles(com.powsybl.openloadflow.util.report.PowsyblOpenLoadFlowReportResourceBundle.BASE_NAME)
                        .withMessageTemplate("olf.saReport").build()
                : com.powsybl.commons.report.ReportNode.NO_OP;
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, contingencies, monitors, sap, plainCpuReport);
        if (plainCpuReport != com.powsybl.commons.report.ReportNode.NO_OP) {
            printReport(plainCpuReport, 0);
        }
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            SecurityAnalysisResult gpu = runSecurityAnalysis(network, contingencies, monitors, sap);
            if (System.getenv("OLF_GPU_DBG_CENSUS") != null) {
                System.err.println("CENSUS " + GpuBatchScopeCensus.snapshot());
            }
            assertEquals(cpu.getPreContingencyResult().getStatus(), gpu.getPreContingencyResult().getStatus(),
                    "pre-contingency status");
            if (requireEngage) {
                assertTrue(GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get() > 0,
                        "the batch must ENGAGE (fallen-back run proves nothing); census="
                                + GpuBatchScopeCensus.snapshot());
            }
            assertEquals(cpu.getPostContingencyResults().size(), gpu.getPostContingencyResults().size());
            for (int i = 0; i < cpu.getPostContingencyResults().size(); i++) {
                assertEquals(cpu.getPostContingencyResults().get(i).getStatus(),
                        gpu.getPostContingencyResults().get(i).getStatus());
                if ("1".equals(GpuTrace.DBG_BUSDUMP.value())) {
                    var cbr = cpu.getPostContingencyResults().get(i).getNetworkResult().getBranchResults();
                    var gbr = gpu.getPostContingencyResults().get(i).getNetworkResult().getBranchResults();
                    for (int z = 0; z < Math.min(cbr.size(), gbr.size()); z++) {
                        System.err.printf("BRDUMP %-8s cpuQ1=%9.4f | %-8s gpuQ1=%9.4f%n", cbr.get(z).getBranchId(),
                                cbr.get(z).getQ1(), gbr.get(z).getBranchId(), gbr.get(z).getQ1());
                    }
                    var cbz = cpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                    var gbz = gpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                    System.err.println("PLAINDUMP sizes cpu=" + cbz.size() + " gpu=" + gbz.size());
                    for (int z = 0; z < Math.min(cbz.size(), gbz.size()); z++) {
                        System.err.printf("PLAINDUMP %-10s cpu=%9.4f | %-10s gpu=%9.4f%n", cbz.get(z).getBusId(),
                                cbz.get(z).getV(), gbz.get(z).getBusId(), gbz.get(z).getV());
                    }
                    cpu.getPostContingencyResults().get(i).getLimitViolationsResult().getLimitViolations()
                            .forEach(v -> System.err.println("PLAINVIOL cpu " + v.getSubjectId() + " " + v.getLimitType() + " " + v.getValue()));
                    gpu.getPostContingencyResults().get(i).getLimitViolationsResult().getLimitViolations()
                            .forEach(v -> System.err.println("PLAINVIOL gpu " + v.getSubjectId() + " " + v.getLimitType() + " " + v.getValue()));
                }
                assertEquals(cpu.getPostContingencyResults().get(i).getLimitViolationsResult().getLimitViolations().size(),
                        gpu.getPostContingencyResults().get(i).getLimitViolationsResult().getLimitViolations().size(),
                        "violation count for " + cpu.getPostContingencyResults().get(i).getContingency().getId());
                assertEquals(cpu.getPostContingencyResults().get(i).getDistributedActivePower(),
                        gpu.getPostContingencyResults().get(i).getDistributedActivePower(), 1e-9,
                        "distributed active power for " + cpu.getPostContingencyResults().get(i).getContingency().getId());
                var cpuBuses = cpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                var gpuBuses = gpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                assertEquals(cpuBuses.size(), gpuBuses.size());
                for (int b = 0; b < cpuBuses.size(); b++) {
                    org.junit.jupiter.api.Assertions.assertEquals(cpuBuses.get(b).getV(), gpuBuses.get(b).getV(), vToleranceKv,
                            "bus voltage " + cpuBuses.get(b).getBusId() + " for "
                                    + cpu.getPostContingencyResults().get(i).getContingency().getId()
                                    + " (" + vToleranceKv + " kV tolerance)");
                }
            }
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    static void printReport(com.powsybl.commons.report.ReportNode node, int depth) {
        System.err.println("CPU-REPORT " + "  ".repeat(depth) + node.getMessage());
        for (com.powsybl.commons.report.ReportNode c : node.getChildren()) {
            printReport(c, depth + 1);
        }
    }

    void assertStrategyMatchesCpu(Network network, List<Contingency> contingencies,
                                          List<com.powsybl.contingency.strategy.OperatorStrategy> strategies,
                                          List<com.powsybl.action.Action> actions, int expectedPrecomputed) {
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, expectedPrecomputed, true);
    }

    void assertStrategyMatchesCpu(Network network, List<Contingency> contingencies,
                                          List<com.powsybl.contingency.strategy.OperatorStrategy> strategies,
                                          List<com.powsybl.action.Action> actions, int expectedPrecomputed,
                                          boolean distributedSlack) {
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, expectedPrecomputed, distributedSlack, p -> { });
    }

    void assertStrategyMatchesCpu(Network network, List<Contingency> contingencies,
                                          List<com.powsybl.contingency.strategy.OperatorStrategy> strategies,
                                          List<com.powsybl.action.Action> actions, int expectedPrecomputed,
                                          boolean distributedSlack,
                                          java.util.function.Consumer<com.powsybl.openloadflow.OpenLoadFlowParameters> customize) {
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(distributedSlack).setUseReactiveLimits(true);
        customize.accept(com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp));
        String sConvEps = System.getProperty("olf.gpu.test.convEps", System.getenv("OLF_GPU_DBG_CONV_EPS"));
        if (sConvEps != null) {
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                    .setNewtonRaphsonConvEpsPerEq(Double.parseDouble(sConvEps));
        }
        String sSlackTol = System.getProperty("olf.gpu.test.slackTol", System.getenv("OLF_GPU_DBG_SLACK_TOL"));
        if (sSlackTol != null) {
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                    .setSlackBusPMaxMismatch(Double.parseDouble(sSlackTol));
        }
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        // Node-breaker networks: contingency PROPAGATION opens the target's breakers too (switches are
        // LfBranches in breaker mode), tripping the batch's >2-branch guard — keep the strategy tests
        // on the un-propagated contingency shape (switch outages as zero-impedance-element outages are future scope).
        sap.addExtension(OpenSecurityAnalysisParameters.class,
                new OpenSecurityAnalysisParameters().setContingencyPropagation(false));
        sap.setLoadFlowParameters(lfp);
        java.util.Set<String> allVls = network.getVoltageLevelStream().map(vl -> vl.getId())
                .collect(java.util.stream.Collectors.toSet());
        List<com.powsybl.security.monitor.StateMonitor> monitors = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(), allVls, java.util.Set.of()));
        com.powsybl.commons.report.ReportNode helperReport = "1".equals(System.getenv("OLF_GPU_DBG_REPORT"))
                ? com.powsybl.commons.report.ReportNode.newRootReportNode()
                        .withResourceBundles(com.powsybl.openloadflow.util.report.PowsyblOpenLoadFlowReportResourceBundle.BASE_NAME)
                        .withMessageTemplate("olf.saReport").build()
                : com.powsybl.commons.report.ReportNode.NO_OP;
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, contingencies, monitors, sap, strategies, actions,
                helperReport);
        if (helperReport != com.powsybl.commons.report.ReportNode.NO_OP) {
            printReport(helperReport, 0);
        }
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            SecurityAnalysisResult gpu = runSecurityAnalysis(network, contingencies, monitors, sap, strategies, actions,
                    com.powsybl.commons.report.ReportNode.NO_OP);
            if (expectedPrecomputed >= 0) {
                assertEquals(expectedPrecomputed, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                        "contingencies AND (ctg, strategy) pairs must precompute on the device (pre-contingency cpu="
                                + cpu.getPreContingencyResult().getStatus() + " gpu=" + gpu.getPreContingencyResult().getStatus()
                                + ", census " + GpuBatchScopeCensus.snapshot() + ")");
            }
            assertEquals(cpu.getPostContingencyResults().size(), gpu.getPostContingencyResults().size());
            for (int r = 0; r < cpu.getPostContingencyResults().size(); r++) {
                var cpuC = cpu.getPostContingencyResults().get(r);
                var gpuC = gpu.getPostContingencyResults().get(r);
                String ctgId = cpuC.getContingency().getId();
                assertEquals(cpuC.getStatus(), gpuC.getStatus(), "contingency status of " + ctgId);
                assertEquals(cpuC.getDistributedActivePower(), gpuC.getDistributedActivePower(), 1e-9,
                        "contingency distributed active power of " + ctgId);
                var cpuBuses = cpuC.getNetworkResult().getBusResults();
                var gpuBuses = gpuC.getNetworkResult().getBusResults();
                assertEquals(cpuBuses.size(), gpuBuses.size());
                for (int b = 0; b < cpuBuses.size(); b++) {
                    org.junit.jupiter.api.Assertions.assertEquals(cpuBuses.get(b).getV(), gpuBuses.get(b).getV(), 1e-12,
                            "contingency " + ctgId + " bus voltage " + cpuBuses.get(b).getBusId());
                }
            }
            assertEquals(cpu.getOperatorStrategyResults().size(), gpu.getOperatorStrategyResults().size());
            for (int r = 0; r < cpu.getOperatorStrategyResults().size(); r++) {
                var cpuS = cpu.getOperatorStrategyResults().get(r).getConditionalActionsResults().get(0);
                var gpuS = gpu.getOperatorStrategyResults().get(r).getConditionalActionsResults().get(0);
                String strategyId = cpu.getOperatorStrategyResults().get(r).getOperatorStrategy().getId();
                assertEquals(cpuS.getStatus(), gpuS.getStatus(), "strategy status of " + strategyId);
                assertEquals(cpuS.getLimitViolationsResult().getLimitViolations().size(),
                        gpuS.getLimitViolationsResult().getLimitViolations().size(), "strategy violation count of " + strategyId);
                assertEquals(cpuS.getDistributedActivePower(), gpuS.getDistributedActivePower(), 1e-9,
                        "strategy distributed active power of " + strategyId);
                var cpuBuses = cpuS.getNetworkResult().getBusResults();
                var gpuBuses = gpuS.getNetworkResult().getBusResults();
                assertEquals(cpuBuses.size(), gpuBuses.size());
                if ("1".equals(GpuTrace.DBG_BUSDUMP.value())) {
                    var cpb = cpu.getPostContingencyResults().get(r).getNetworkResult().getBusResults();
                    var gpb = gpu.getPostContingencyResults().get(r).getNetworkResult().getBusResults();
                    for (int b = 0; b < cpb.size(); b++) {
                        System.err.printf("PARENTDUMP %-8s cpu=%9.4f gpu=%9.4f d=%+8.4f%n", cpb.get(b).getBusId(),
                                cpb.get(b).getV(), gpb.get(b).getV(), gpb.get(b).getV() - cpb.get(b).getV());
                    }
                    for (int b = 0; b < cpuBuses.size(); b++) {
                        System.err.printf("BUSDUMP %-8s cpu=%9.4f gpu=%9.4f d=%+8.4f%n", cpuBuses.get(b).getBusId(),
                                cpuBuses.get(b).getV(), gpuBuses.get(b).getV(), gpuBuses.get(b).getV() - cpuBuses.get(b).getV());
                    }
                }
                for (int b = 0; b < cpuBuses.size(); b++) {
                    org.junit.jupiter.api.Assertions.assertEquals(cpuBuses.get(b).getV(), gpuBuses.get(b).getV(), 1e-12,
                            "strategy bus voltage " + cpuBuses.get(b).getBusId());
                }
            }
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    static List<com.powsybl.contingency.strategy.OperatorStrategy> strategiesOf(String ctgId, String... actionIds) {
        List<com.powsybl.contingency.strategy.OperatorStrategy> out = new java.util.ArrayList<>();
        for (String a : actionIds) {
            out.add(new com.powsybl.contingency.strategy.OperatorStrategy("s-" + a,
                    com.powsybl.contingency.ContingencyContext.specificContingency(ctgId),
                    new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of(a)));
        }
        return out;
    }

    static void countReportKeys(com.powsybl.commons.report.ReportNode node, java.util.Map<String, Integer> tally) {
        tally.merge(node.getMessageKey(), 1, Integer::sum);
        for (com.powsybl.commons.report.ReportNode c : node.getChildren()) {
            countReportKeys(c, tally);
        }
    }

    static com.powsybl.commons.report.ReportNode findSubtree(com.powsybl.commons.report.ReportNode node, String key) {
        if (key.equals(node.getMessageKey())) {
            return node;
        }
        for (com.powsybl.commons.report.ReportNode c : node.getChildren()) {
            var found = findSubtree(c, key);
            if (found != null) {
                return found;
            }
        }
        return null;
    }

    /** The body of {@code deviceIncrementalTvcTapMovesInKernel} in {@link GpuSaTransformerControlTest}, run again under the robust solve mode by
     *  {@code deviceRobustModeRunsTheInKernelLoops}. */
    void assertIncrementalTvcTapMovesInKernel() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = incrementalTvcTapMoveNetwork();
        assertGpuMatchesCpu(network, incrementalTvcTapMoveContingencies(network), 1e-12, incrementalTvcParameters());
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "both scenarios must batch on-device");
    }

    /**
     * GPU against CPU per contingency on what the result REPORTS about the solved state: every bus voltage (1e-12 kV),
     * then every branch's P, Q and I on both sides (1e-9 MW, MVar, A). The voltages say the device solved the scenario
     * OLF solved; the flows say the host evaluated them with the same branch models — a device tap move missing from the
     * host's transformer model leaves the voltages right and the transformer's flow a few percent off. At least one tap
     * move must have reached the network, so a fixture where no tap moves cannot pass for one where the move is right.
     */
    void assertFlowsMatchCpu(Network network, List<Contingency> contingencies, com.powsybl.loadflow.LoadFlowParameters lfp,
                             int expectedOnDevice) {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        java.util.Set<String> branches = network.getBranchStream().map(b -> b.getId())
                .collect(java.util.stream.Collectors.toSet());
        java.util.Set<String> voltageLevels = network.getVoltageLevelStream().map(vl -> vl.getId())
                .collect(java.util.stream.Collectors.toSet());
        List<com.powsybl.security.monitor.StateMonitor> monitors = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), branches, voltageLevels, java.util.Set.of()));
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, contingencies, monitors, sap);
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            GpuDeviceTaps.CHANGES.set(0);
            SecurityAnalysisResult gpu = runSecurityAnalysis(network, contingencies, monitors, sap);
            assertEquals(expectedOnDevice, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                    "scenarios batched on-device; census=" + GpuBatchScopeCensus.snapshot());
            assertTrue(GpuDeviceTaps.CHANGES.get() > 0, "a device tap move must reach the network: the fixture moves no tap");
            assertEquals(cpu.getPostContingencyResults().size(), gpu.getPostContingencyResults().size());
            for (int i = 0; i < cpu.getPostContingencyResults().size(); i++) {
                String ctg = cpu.getPostContingencyResults().get(i).getContingency().getId();
                assertEquals(cpu.getPostContingencyResults().get(i).getStatus(), gpu.getPostContingencyResults().get(i).getStatus());
                var cbv = cpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                var gbv = gpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                assertEquals(cbv.size(), gbv.size(), "bus results for " + ctg);
                for (int b = 0; b < cbv.size(); b++) {
                    assertEquals(cbv.get(b).getV(), gbv.get(b).getV(), 1e-12, "v " + cbv.get(b).getBusId() + " after " + ctg);
                }
                var cbr = cpu.getPostContingencyResults().get(i).getNetworkResult().getBranchResults();
                var gbr = gpu.getPostContingencyResults().get(i).getNetworkResult().getBranchResults();
                assertEquals(cbr.size(), gbr.size(), "branch results for " + ctg);
                for (var c : cbr) {
                    var g = gbr.stream().filter(r -> r.getBranchId().equals(c.getBranchId())).findFirst().orElseThrow();
                    String what = c.getBranchId() + " after " + ctg;
                    assertEquals(c.getP1(), g.getP1(), 1e-9, "p1 " + what);
                    assertEquals(c.getQ1(), g.getQ1(), 1e-9, "q1 " + what);
                    assertEquals(c.getI1(), g.getI1(), 1e-9, "i1 " + what);
                    assertEquals(c.getP2(), g.getP2(), 1e-9, "p2 " + what);
                    assertEquals(c.getQ2(), g.getQ2(), 1e-9, "q2 " + what);
                    assertEquals(c.getI2(), g.getI2(), 1e-9, "i2 " + what);
                }
            }
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    /** T2wT regulating at 33.5 kV with a zero deadband: the base load flow lands on tap 2 (of 0..3); the strong-line
     *  outage moves it to 3 while the clone outage leaves it (both verified against OLF's own OL_CHECK sequence). */
    static Network incrementalTvcTapMoveNetwork() {
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
        network.getTwoWindingsTransformer("T2wT").getRatioTapChanger().setTargetDeadband(0).setRegulating(true)
                .setTapPosition(0).setRegulationTerminal(network.getTwoWindingsTransformer("T2wT").getTerminal2()).setTargetV(33.5);
        com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
        String bis = l0.getId() + "_BIS";
        network.newLine().setId(bis)
                .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
        return network;
    }

    /** The clone outage (tap stays) and the strong-line outage (tap moves 2 -> 3) of {@link #incrementalTvcTapMoveNetwork}. */
    static List<Contingency> incrementalTvcTapMoveContingencies(Network network) {
        com.powsybl.iidm.network.Line l0 = network.getLineStream()
                .filter(l -> network.getLine(l.getId() + "_BIS") != null).findFirst().orElseThrow();
        String bis = l0.getId() + "_BIS";
        return List.of(new Contingency(bis, new BranchContingency(bis)),
                new Contingency(l0.getId(), new BranchContingency(l0.getId())));
    }

    static com.powsybl.loadflow.LoadFlowParameters incrementalTvcParameters() {
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setTransformerVoltageControlOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp).setTransformerVoltageControlMode(
                com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.INCREMENTAL_VOLTAGE_CONTROL);
        return lfp;
    }

    /** The body of {@code deviceIncrementalSharedTvcMatchesCpu} in {@link GpuSaTransformerControlTest}, run again under the robust solve mode by
     *  {@code deviceRobustModeRunsTheInKernelLoops}. */
    void assertIncrementalSharedTvcMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = sharedThreeMemberTvcNetwork();
        for (String id : new String[] {"T2wT", "T2wT2", "T2wT3"}) {
            network.getTwoWindingsTransformer(id).getRatioTapChanger().setTargetV(34.5);   // several taps must move
        }
        com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
        List<Contingency> ctgs = List.of(
                new Contingency(l0.getId() + "_BIS", new BranchContingency(l0.getId() + "_BIS")),
                new Contingency(l0.getId(), new BranchContingency(l0.getId())),
                new Contingency("T2wT2", new BranchContingency("T2wT2")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setTransformerVoltageControlOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp).setTransformerVoltageControlMode(
                com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.INCREMENTAL_VOLTAGE_CONTROL);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(3, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "all three scenarios (incl. the member outage) must batch on-device");
    }

    /** The body of {@code deviceIncrementalShuntVoltageControlMatchesCpu} in {@link GpuSaShuntControlTest}, run again under the robust solve mode by
     *  {@code deviceRobustModeRunsTheInKernelLoops}. */
    void assertIncrementalShuntVoltageControlMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.ShuntNetworkFactory.create();
        com.powsybl.iidm.network.Line l1 = network.getLine("l1");
        network.newLine().setId("l1_BIS")
                .setVoltageLevel1(l1.getTerminal1().getVoltageLevel().getId()).setBus1(l1.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l1.getTerminal2().getVoltageLevel().getId()).setBus2(l1.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l1.getR() * 2).setX(l1.getX() * 2)
                .setG1(l1.getG1()).setB1(l1.getB1()).setG2(l1.getG2()).setB2(l1.getB2()).add();
        List<Contingency> ctgs = List.of(new Contingency("l1_BIS", new BranchContingency("l1_BIS")),
                new Contingency("l1", new BranchContingency("l1")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setShuntCompensatorVoltageControlOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setShuntVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.ShuntVoltageControlMode.INCREMENTAL_VOLTAGE_CONTROL);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }

    /** The body of {@code deviceSecondaryVoltageControlMatchesCpu} in {@link GpuSaSecondaryVoltageControlTest}, run again under the robust solve mode by
     *  {@code deviceRobustModeRunsTheInKernelLoops}. */
    void assertSecondaryVoltageControlMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory.create14();
        network.newExtension(com.powsybl.iidm.network.extensions.SecondaryVoltageControlAdder.class)
                .newControlZone().withName("z1")
                .newPilotPoint().withTargetV(13).withBusbarSectionsOrBusesIds(List.of("B10")).add()
                .newControlUnit().withId("B6-G").add()
                .newControlUnit().withId("B8-G").add()
                .add().add();
        List<Contingency> ctgs = List.of(new Contingency("L1-2-1", new BranchContingency("L1-2-1")),
                new Contingency("L2-3-1", new BranchContingency("L2-3-1")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSecondaryVoltageControl(true).setMaxPlausibleTargetVoltage(1.6);
        assertGpuMatchesCpu(network, ctgs, 1e-11, lfp);
    }

    /** The body of {@code deviceSecondaryVoltageControlRemoteMatchesCpu} in {@link GpuSaSecondaryVoltageControlTest}, run again under the robust solve mode by
     *  {@code deviceRobustModeRunsTheInKernelLoops}. */
    void assertSecondaryVoltageControlRemoteMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = secondaryVcRemoteNetwork();
        List<Contingency> ctgs = List.of(new Contingency("L1-2-1", new BranchContingency("L1-2-1")),
                new Contingency("L2-3-1", new BranchContingency("L2-3-1")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSecondaryVoltageControl(true).setMaxPlausibleTargetVoltage(1.6);
        assertGpuMatchesCpu(network, ctgs, 1e-11, lfp);
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "the remote-SVC run must ENGAGE the batch (was a whole-run CPU gate)");
    }
}
