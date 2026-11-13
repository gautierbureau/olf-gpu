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
import com.powsybl.contingency.violations.LimitViolation;
import com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory;
import com.powsybl.iidm.network.Branch;
import com.powsybl.iidm.network.Network;
import com.powsybl.iidm.network.TwoSides;
import com.powsybl.iidm.network.test.EurostagTutorialExample1Factory;
import com.powsybl.loadflow.LoadFlow;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.CommonTestConfig;
import com.powsybl.openloadflow.OpenLoadFlowProvider;
import com.powsybl.openloadflow.dc.gpu.GpuDcSolver;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.sa.AbstractOpenSecurityAnalysisTest;
import com.powsybl.openloadflow.sa.OpenSecurityAnalysisParameters;
import com.powsybl.openloadflow.sa.OpenSecurityAnalysisProvider;
import com.powsybl.security.SecurityAnalysisParameters;
import com.powsybl.security.SecurityAnalysisResult;
import com.powsybl.security.monitor.StateMonitor;
import com.powsybl.security.results.BranchResult;
import com.powsybl.security.results.PostContingencyResult;
import org.junit.jupiter.api.Test;

import java.util.*;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The GPU Woodbury DC security analysis against the stock one: same violations (key and value),
 * same monitored flows, same connectivity results, same statuses — and the device actually ran.
 *
 * <p>Bound: 1e-10 pu (1e-8 MW). Two different direct solvers on the same matrix differ by
 * cond·eps, measured 1.2e-11 pu on rte6515; there is no iteration here to loosen.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuWoodburyDcSecurityAnalysisTest extends AbstractOpenSecurityAnalysisTest {

    private static final double TOL_MW = 1e-8;

    GpuWoodburyDcSecurityAnalysisTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    private static SecurityAnalysisParameters dcFastParameters() {
        SecurityAnalysisParameters sap = new SecurityAnalysisParameters();
        sap.getLoadFlowParameters().setDc(true);
        sap.addExtension(OpenSecurityAnalysisParameters.class, new OpenSecurityAnalysisParameters().setDcFastMode(true));
        return sap;
    }

    private static List<Contingency> allBranches(Network network) {
        return network.getBranchStream().map(b -> new Contingency(b.getId(), new BranchContingency(b.getId()))).toList();
    }

    private static List<StateMonitor> monitorAll(Network network) {
        Set<String> branchIds = network.getBranchStream().map(Branch::getId).collect(Collectors.toCollection(LinkedHashSet::new));
        Set<String> vlIds = network.getVoltageLevelStream().map(vl -> vl.getId()).collect(Collectors.toCollection(LinkedHashSet::new));
        return List.of(new StateMonitor(ContingencyContext.all(), branchIds, vlIds, Set.of()));
    }

    /** Active-power limits on every branch at a fraction of the base DC flow: lots of post-contingency violations. */
    private static void addActivePowerLimits(Network network, double fraction) {
        LoadFlowParameters lfp = new LoadFlowParameters().setDc(true);
        new LoadFlow.Runner(new OpenLoadFlowProvider(new SparseMatrixFactory())).run(network, lfp);
        for (Branch<?> branch : network.getBranches()) {
            double p = Math.max(Math.abs(branch.getTerminal1().getP()), 1.0);
            for (TwoSides side : TwoSides.values()) {
                var group = side == TwoSides.ONE ? branch.getOrCreateSelectedOperationalLimitsGroup1() : branch.getOrCreateSelectedOperationalLimitsGroup2();
                group.newActivePowerLimits()
                        .setPermanentLimit(p * fraction)
                        .beginTemporaryLimit().setName("20'").setAcceptableDuration(1200).setValue(p * fraction * 1.1).endTemporaryLimit()
                        .beginTemporaryLimit().setName("5'").setAcceptableDuration(300).setValue(p * fraction * 1.3).endTemporaryLimit()
                        .add();
            }
        }
    }

    private SecurityAnalysisResult runStock(Network network, List<Contingency> contingencies, List<StateMonitor> monitors) {
        return runStock(network, contingencies, monitors, dcFastParameters());
    }

    private SecurityAnalysisResult runStock(Network network, List<Contingency> contingencies, List<StateMonitor> monitors, SecurityAnalysisParameters sap) {
        securityAnalysisProvider = new OpenSecurityAnalysisProvider(new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        return runSecurityAnalysis(network, contingencies, monitors, sap);
    }

    private SecurityAnalysisResult runGpu(Network network, List<Contingency> contingencies, List<StateMonitor> monitors) {
        return runGpu(network, contingencies, monitors, dcFastParameters());
    }

    private SecurityAnalysisResult runGpu(Network network, List<Contingency> contingencies, List<StateMonitor> monitors, SecurityAnalysisParameters sap) {
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        GpuWoodburyDcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.set(-1);
        try {
            return runSecurityAnalysis(network, contingencies, monitors, sap);
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    /** All branches, generators and loads as single-element contingencies. */
    private static List<Contingency> allElements(Network network) {
        List<Contingency> list = new ArrayList<>(allBranches(network));
        network.getGeneratorStream().forEach(g -> list.add(new Contingency(g.getId(), new com.powsybl.contingency.GeneratorContingency(g.getId()))));
        network.getLoadStream().forEach(l -> list.add(new Contingency(l.getId(), new com.powsybl.contingency.LoadContingency(l.getId()))));
        return list;
    }

    /** The device count accumulates over the connected components, and a contingency impacting two
     *  components (an HVDC line between them) is solved in each: at least as many as reported. */
    private static void assertAllOnDevice(SecurityAnalysisResult cpu) {
        assertTrue(GpuWoodburyDcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get() >= cpu.getPostContingencyResults().size(),
                "every reported contingency on the device: " + GpuWoodburyDcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get()
                        + " on the device for " + cpu.getPostContingencyResults().size() + " results");
    }

    private static String key(LimitViolation v) {
        return v.getSubjectId() + "|" + v.getLimitType() + "|" + v.getSide() + "|" + v.getLimitName() + "|" + v.getAcceptableDuration();
    }

    static void assertSameResults(SecurityAnalysisResult cpu, SecurityAnalysisResult gpu) {
        // pre-contingency
        assertEquals(cpu.getPreContingencyResult().getStatus(), gpu.getPreContingencyResult().getStatus());
        assertSameViolations("pre-contingency", cpu.getPreContingencyResult().getLimitViolationsResult().getLimitViolations(),
                gpu.getPreContingencyResult().getLimitViolationsResult().getLimitViolations());
        assertSameBranchResults("pre-contingency", cpu.getPreContingencyResult().getNetworkResult().getBranchResults(),
                gpu.getPreContingencyResult().getNetworkResult().getBranchResults());

        // post-contingency: same contingencies, same order
        List<String> cpuIds = cpu.getPostContingencyResults().stream().map(r -> r.getContingency().getId()).toList();
        List<String> gpuIds = gpu.getPostContingencyResults().stream().map(r -> r.getContingency().getId()).toList();
        assertEquals(cpuIds, gpuIds, "post-contingency result list");
        Map<String, PostContingencyResult> gpuById = gpu.getPostContingencyResults().stream()
                .collect(Collectors.toMap(r -> r.getContingency().getId(), r -> r));
        for (PostContingencyResult c : cpu.getPostContingencyResults()) {
            String id = c.getContingency().getId();
            PostContingencyResult g = gpuById.get(id);
            assertEquals(c.getStatus(), g.getStatus(), id + ": status");
            assertSameViolations(id, c.getLimitViolationsResult().getLimitViolations(), g.getLimitViolationsResult().getLimitViolations());
            assertSameBranchResults(id, c.getNetworkResult().getBranchResults(), g.getNetworkResult().getBranchResults());
            assertEquals(c.getNetworkResult().getBusResults().size(), g.getNetworkResult().getBusResults().size(), id + ": bus results");
            assertEquals(c.getConnectivityResult().getCreatedSynchronousComponentCount(), g.getConnectivityResult().getCreatedSynchronousComponentCount(), id + ": components");
            assertEquals(c.getConnectivityResult().getDisconnectedElements(), g.getConnectivityResult().getDisconnectedElements(), id + ": disconnected elements");
            assertEquals(c.getConnectivityResult().getDisconnectedLoadActivePower(), g.getConnectivityResult().getDisconnectedLoadActivePower(), TOL_MW, id + ": lost load");
            assertEquals(c.getConnectivityResult().getDisconnectedGenerationActivePower(), g.getConnectivityResult().getDisconnectedGenerationActivePower(), TOL_MW, id + ": lost generation");
        }

        // operator strategies: same list, same order, same conditional-action results
        List<String> cpuOs = cpu.getOperatorStrategyResults().stream().map(r -> r.getOperatorStrategy().getId()).toList();
        List<String> gpuOs = gpu.getOperatorStrategyResults().stream().map(r -> r.getOperatorStrategy().getId()).toList();
        assertEquals(cpuOs, gpuOs, "operator strategy result list");
        for (int i = 0; i < cpuOs.size(); i++) {
            var c = cpu.getOperatorStrategyResults().get(i);
            var g = gpu.getOperatorStrategyResults().get(i);
            String id = "strategy " + cpuOs.get(i);
            assertEquals(c.getConditionalActionsResults().size(), g.getConditionalActionsResults().size(), id + ": conditional results");
            for (int k = 0; k < c.getConditionalActionsResults().size(); k++) {
                var cc = c.getConditionalActionsResults().get(k);
                var gg = g.getConditionalActionsResults().get(k);
                assertEquals(cc.getStatus(), gg.getStatus(), id + ": status");
                assertSameViolations(id, cc.getLimitViolationsResult().getLimitViolations(), gg.getLimitViolationsResult().getLimitViolations());
                assertSameBranchResults(id, cc.getNetworkResult().getBranchResults(), gg.getNetworkResult().getBranchResults());
            }
        }
    }

    private static void assertSameViolations(String where, List<LimitViolation> cpu, List<LimitViolation> gpu) {
        Map<String, LimitViolation> a = cpu.stream().collect(Collectors.toMap(GpuWoodburyDcSecurityAnalysisTest::key, v -> v, (x, y) -> x, TreeMap::new));
        Map<String, LimitViolation> b = gpu.stream().collect(Collectors.toMap(GpuWoodburyDcSecurityAnalysisTest::key, v -> v, (x, y) -> x, TreeMap::new));
        assertEquals(a.keySet(), b.keySet(), where + ": violation set");
        for (String k : a.keySet()) {
            LimitViolation va = a.get(k);
            LimitViolation vb = b.get(k);
            assertEquals(va.getLimit(), vb.getLimit(), 0.0, where + " " + k + ": limit");
            assertEquals(va.getValue(), vb.getValue(), TOL_MW * Math.max(1.0, Math.abs(va.getValue())), where + " " + k + ": value");
            assertEquals(va.getLimitReduction(), vb.getLimitReduction(), 0.0, where + " " + k + ": reduction");
        }
    }

    private static void assertSameBranchResults(String where, List<BranchResult> cpu, List<BranchResult> gpu) {
        Map<String, BranchResult> b = gpu.stream().collect(Collectors.toMap(BranchResult::getBranchId, r -> r, (x, y) -> x));
        assertEquals(cpu.stream().map(BranchResult::getBranchId).collect(Collectors.toSet()), b.keySet(), where + ": monitored branches");
        for (BranchResult r : cpu) {
            BranchResult g = b.get(r.getBranchId());
            assertEquals(r.getP1(), g.getP1(), TOL_MW, where + " " + r.getBranchId() + ": p1");
            assertEquals(r.getP2(), g.getP2(), TOL_MW, where + " " + r.getBranchId() + ": p2");
            assertEquals(r.getI1(), g.getI1(), TOL_MW * 10, where + " " + r.getBranchId() + ": i1");
            assertEquals(r.getI2(), g.getI2(), TOL_MW * 10, where + " " + r.getBranchId() + ": i2");
            if (Double.isNaN(r.getFlowTransfer())) {
                assertTrue(Double.isNaN(g.getFlowTransfer()), where + " " + r.getBranchId() + ": flowTransfer NaN");
            } else {
                assertEquals(r.getFlowTransfer(), g.getFlowTransfer(), 1e-9, where + " " + r.getBranchId() + ": flowTransfer");
            }
        }
    }

    @Test
    void eurostagCurrentLimitsMatchStock() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        Network network = EurostagTutorialExample1Factory.createWithFixedCurrentLimits();
        List<Contingency> contingencies = allBranches(network);
        SecurityAnalysisResult cpu = runStock(network, contingencies, monitorAll(network));
        SecurityAnalysisResult gpu = runGpu(network, contingencies, monitorAll(network));
        assertSameResults(cpu, gpu);
        // the two parallel lines, and the two transformers that island a bus (modified target) — all on the device
        assertEquals(4, GpuWoodburyDcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "contingencies solved on the device");
        assertTrue(cpu.getPostContingencyResults().stream().anyMatch(r -> !r.getLimitViolationsResult().getLimitViolations().isEmpty()),
                "the case must produce post-contingency violations for the comparison to mean anything");
    }

    @Test
    void ieee14ActivePowerLimitsMatchStock() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        Network network = IeeeCdfNetworkFactory.create14();
        addActivePowerLimits(network, 0.9);
        List<Contingency> contingencies = allBranches(network);
        SecurityAnalysisResult cpu = runStock(network, contingencies, monitorAll(network));
        SecurityAnalysisResult gpu = runGpu(network, contingencies, monitorAll(network));
        assertSameResults(cpu, gpu);
        int onDevice = GpuWoodburyDcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get();
        assertEquals(20, onDevice, "all 20 branch outages on the device (the radial L7-8-1 islands bus 8: modified target)");
        long violations = cpu.getPostContingencyResults().stream().mapToLong(r -> r.getLimitViolationsResult().getLimitViolations().size()).sum();
        assertTrue(violations > 20, "the case must produce post-contingency violations, got " + violations);
    }

    /** A phase-shifting transformer outage: its alpha row is zeroed in the modified target (the stock's
     *  lostPhaseControllers re-solve) — the PST network of OLF's phase-control tests, with limits. */
    @Test
    void phaseShifterOutageMatchesStock() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        Network network = com.powsybl.openloadflow.network.PhaseControlFactory.createWithOneT2wtTwoLines();
        addActivePowerLimits(network, 0.7);
        List<Contingency> contingencies = allBranches(network);
        SecurityAnalysisResult cpu = runStock(network, contingencies, monitorAll(network));
        SecurityAnalysisResult gpu = runGpu(network, contingencies, monitorAll(network));
        assertSameResults(cpu, gpu);
        assertEquals(contingencies.size(), GpuWoodburyDcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "all outages on the device, the PST's included");
        assertTrue(network.getTwoWindingsTransformerStream().anyMatch(t -> t.hasPhaseTapChanger()), "the case must have a phase shifter");
    }

    /** Lost generation and load from the contingency itself: the stock's lost-power branch (bus DC states
     *  saved, processLostPowerChanges, redistribution), replayed on the host and solved on the device. */
    @Test
    void generatorAndLoadContingenciesMatchStock() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        Network network = IeeeCdfNetworkFactory.create14();
        addActivePowerLimits(network, 0.9);
        List<Contingency> contingencies = allElements(network);
        SecurityAnalysisResult cpu = runStock(network, contingencies, monitorAll(network));
        SecurityAnalysisResult gpu = runGpu(network, contingencies, monitorAll(network));
        assertSameResults(cpu, gpu);
        assertAllOnDevice(cpu);
        assertTrue(cpu.getPostContingencyResults().size() > 20, "generators and loads must have produced results");
    }

    /** Node-breaker busbar-section outages: OLF's cleanup turns them into the incident branch outages plus
     *  the islanded bus; the connectivity result carries the lost injections. */
    @Test
    void busbarSectionContingenciesMatchStock() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        Network network = com.powsybl.openloadflow.network.NodeBreakerNetworkFactory.create();
        addActivePowerLimits(network, 0.8);
        List<Contingency> contingencies = new ArrayList<>(allBranches(network));
        network.getBusbarSectionStream().forEach(b -> contingencies.add(new Contingency(b.getId(), new com.powsybl.contingency.BusbarSectionContingency(b.getId()))));
        SecurityAnalysisResult cpu = runStock(network, contingencies, monitorAll(network));
        SecurityAnalysisResult gpu = runGpu(network, contingencies, monitorAll(network));
        assertSameResults(cpu, gpu);
        assertAllOnDevice(cpu);
    }

    /** VSC HVDC: stations are generators in DC; an outage islanding a station leaves the HVDC without power. */
    @Test
    void hvdcNetworkMatchesStock() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        Network network = com.powsybl.openloadflow.network.HvdcNetworkFactory.createTwoCcLinkedByAHvdcVscWithGenerators();
        addActivePowerLimits(network, 0.8);
        List<Contingency> contingencies = new ArrayList<>(allElements(network));
        network.getHvdcLineStream().forEach(h -> contingencies.add(new Contingency(h.getId(), new com.powsybl.contingency.HvdcLineContingency(h.getId()))));
        SecurityAnalysisResult cpu = runStock(network, contingencies, monitorAll(network));
        SecurityAnalysisResult gpu = runGpu(network, contingencies, monitorAll(network));
        assertSameResults(cpu, gpu);
        assertAllOnDevice(cpu);
    }

    /** A load-based balance type: no array model, the exact host replay builds the column. */
    @Test
    void loadBalanceTypeMatchesStock() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        Network network = IeeeCdfNetworkFactory.create14();
        addActivePowerLimits(network, 0.9);
        List<Contingency> contingencies = allElements(network);
        SecurityAnalysisParameters sap = dcFastParameters();
        sap.getLoadFlowParameters().setBalanceType(LoadFlowParameters.BalanceType.PROPORTIONAL_TO_LOAD);
        SecurityAnalysisResult cpu = runStock(network, contingencies, monitorAll(network), sap);
        SecurityAnalysisResult gpu = runGpu(network, contingencies, monitorAll(network), sap);
        assertSameResults(cpu, gpu);
        assertAllOnDevice(cpu);
    }

    /** Two slack buses: the target vector carries the shared mismatch term — the replay reads it as OLF builds it. */
    @Test
    void multiSlackMatchesStock() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        Network network = IeeeCdfNetworkFactory.create14();
        addActivePowerLimits(network, 0.9);
        List<Contingency> contingencies = allElements(network);
        SecurityAnalysisParameters sap = dcFastParameters();
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(sap.getLoadFlowParameters()).setMaxSlackBusCount(2);
        SecurityAnalysisResult cpu = runStock(network, contingencies, monitorAll(network), sap);
        SecurityAnalysisResult gpu = runGpu(network, contingencies, monitorAll(network), sap);
        assertSameResults(cpu, gpu);
        assertAllOnDevice(cpu);
    }

    /**
     * The modified target columns built on the device are the host's bit for bit, on every network and
     * parameter set that builds them: lost generators and loads, islanding outages, busbar sections, VSC
     * stations left without power, two slack buses, every generation balance type, the slack distributed
     * or not, and IEEE 118's 150-odd generator and load outages (columns past the first thread block). The
     * host reference is the exact replay of the stock column. Then the results match the stock as everywhere else.
     */
    @Test
    void deviceTargetsAreTheHostOnesBitForBit() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        GpuWoodburyDcSecurityAnalysis.CHECK_DEVICE_TARGETS.set(true);
        GpuWoodburyDcSecurityAnalysis.DEVICE_TARGET_COUNT.set(0);
        GpuWoodburyDcSecurityAnalysis.TARGET_MISMATCH_COUNT.set(0);
        try {
            List<java.util.function.Supplier<Network>> networks = List.of(
                    IeeeCdfNetworkFactory::create14,
                    IeeeCdfNetworkFactory::create118,               // more than 64 columns in a chunk: every thread block
                    com.powsybl.openloadflow.network.NodeBreakerNetworkFactory::create,
                    com.powsybl.openloadflow.network.HvdcNetworkFactory::createTwoCcLinkedByAHvdcVscWithGenerators,
                    com.powsybl.openloadflow.network.PhaseControlFactory::createWithOneT2wtTwoLines);
            for (java.util.function.Supplier<Network> factory : networks) {
                for (LoadFlowParameters.BalanceType balanceType : List.of(LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX,
                        LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_P,
                        LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_REMAINING_MARGIN)) {
                    for (int variant = 0; variant < 3; variant++) {
                        Network network = factory.get();
                        addActivePowerLimits(network, 0.9);
                        List<Contingency> contingencies = new ArrayList<>(allElements(network));
                        network.getBusbarSectionStream().forEach(b -> contingencies.add(new Contingency(b.getId(),
                                new com.powsybl.contingency.BusbarSectionContingency(b.getId()))));
                        network.getHvdcLineStream().forEach(h -> contingencies.add(new Contingency(h.getId(),
                                new com.powsybl.contingency.HvdcLineContingency(h.getId()))));
                        SecurityAnalysisParameters sap = dcFastParameters();
                        sap.getLoadFlowParameters().setBalanceType(balanceType).setDistributedSlack(variant != 1);
                        if (variant == 2) {
                            com.powsybl.openloadflow.OpenLoadFlowParameters.create(sap.getLoadFlowParameters()).setMaxSlackBusCount(2);
                        }
                        SecurityAnalysisResult cpu = runStock(network, contingencies, monitorAll(network), sap);
                        SecurityAnalysisResult gpu = runGpu(network, contingencies, monitorAll(network), sap);
                        assertSameResults(cpu, gpu);
                    }
                }
            }
        } finally {
            GpuWoodburyDcSecurityAnalysis.CHECK_DEVICE_TARGETS.set(false);
        }
        assertTrue(GpuWoodburyDcSecurityAnalysis.DEVICE_TARGET_COUNT.get() > 100,
                "the device must have built the columns: " + GpuWoodburyDcSecurityAnalysis.DEVICE_TARGET_COUNT.get());
        assertEquals(0, GpuWoodburyDcSecurityAnalysis.TARGET_MISMATCH_COUNT.get(), "device target values differing from the host's");
    }

    /**
     * The monitored branches' results built from the branches found once per run are OLF's own update's, bit for bit
     * and in the same order, on every contingency: branch-only monitors (the case the fast path covers), the
     * modification threshold at 0 and at 1 MW, the result extension off and on, networks with islanding, busbar
     * sections, VSC HVDC and a phase shifter. Then the results match the stock arm as everywhere else.
     */
    @Test
    void monitoredBranchResultsAreOlfsBitForBit() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        GpuWoodburyDcSecurityAnalysis.CHECK_NETWORK_RESULTS.set(true);
        GpuWoodburyDcSecurityAnalysis.NETWORK_RESULT_MISMATCH_COUNT.set(0);
        GpuWoodburyDcSecurityAnalysis.FAST_NETWORK_RESULT_COUNT.set(0);
        GpuWoodburyDcSecurityAnalysis.STATE_FREE_CONTINGENCY_COUNT.set(0);
        try {
            List<java.util.function.Supplier<Network>> networks = List.of(
                    IeeeCdfNetworkFactory::create14,
                    IeeeCdfNetworkFactory::create118,
                    com.powsybl.openloadflow.network.NodeBreakerNetworkFactory::create,
                    com.powsybl.openloadflow.network.HvdcNetworkFactory::createTwoCcLinkedByAHvdcVscWithGenerators,
                    com.powsybl.openloadflow.network.PhaseControlFactory::createWithOneT2wtTwoLines);
            for (java.util.function.Supplier<Network> factory : networks) {
                for (int variant = 0; variant < 3; variant++) {
                    Network network = factory.get();
                    addActivePowerLimits(network, 0.9);
                    List<Contingency> contingencies = new ArrayList<>(allElements(network));
                    network.getBusbarSectionStream().forEach(b -> contingencies.add(new Contingency(b.getId(),
                            new com.powsybl.contingency.BusbarSectionContingency(b.getId()))));
                    Set<String> branchIds = network.getBranchStream().map(Branch::getId).collect(Collectors.toCollection(LinkedHashSet::new));
                    List<StateMonitor> monitors = List.of(new StateMonitor(ContingencyContext.all(), branchIds, Set.of(), Set.of()));
                    SecurityAnalysisParameters sap = dcFastParameters();
                    if (variant == 1) {
                        sap.getModifiedMonitoredElementsParameters().setPowerModificationThreshold(1);
                    }
                    if (variant == 2) {
                        sap.getExtension(OpenSecurityAnalysisParameters.class).setCreateResultExtension(true);
                    }
                    SecurityAnalysisResult cpu = runStock(network, contingencies, monitors, sap);
                    SecurityAnalysisResult gpu = runGpu(network, contingencies, monitors, sap);
                    assertSameResults(cpu, gpu);
                }
            }
        } finally {
            GpuWoodburyDcSecurityAnalysis.CHECK_NETWORK_RESULTS.set(false);
        }
        // a run that answers its contingencies from their state builds them there (contingenciesAnsweredFromTheirState
        // checks those against this path); the others on the monitored branches found once, checked here against OLF's
        assertTrue(GpuWoodburyDcSecurityAnalysis.FAST_NETWORK_RESULT_COUNT.get() > 100,
                "the fast path must have built results: " + GpuWoodburyDcSecurityAnalysis.FAST_NETWORK_RESULT_COUNT.get());
        assertTrue(GpuWoodburyDcSecurityAnalysis.FAST_NETWORK_RESULT_COUNT.get() + GpuWoodburyDcSecurityAnalysis.STATE_FREE_CONTINGENCY_COUNT.get() > 500,
                "the results must have been built off OLF's update");
        assertEquals(0, GpuWoodburyDcSecurityAnalysis.NETWORK_RESULT_MISMATCH_COUNT.get(), "network results differing from OLF's update");
    }

    private SecurityAnalysisResult runStockWithStrategies(Network network, List<Contingency> contingencies, List<StateMonitor> monitors,
                                                          List<com.powsybl.contingency.strategy.OperatorStrategy> strategies, List<com.powsybl.action.Action> actions) {
        securityAnalysisProvider = new OpenSecurityAnalysisProvider(new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        return runSecurityAnalysis(network, contingencies, monitors, dcFastParameters(), strategies, actions, com.powsybl.commons.report.ReportNode.NO_OP);
    }

    private SecurityAnalysisResult runGpuWithStrategies(Network network, List<Contingency> contingencies, List<StateMonitor> monitors,
                                                        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies, List<com.powsybl.action.Action> actions) {
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        GpuWoodburyDcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.set(-1);
        try {
            return runSecurityAnalysis(network, contingencies, monitors, dcFastParameters(), strategies, actions, com.powsybl.commons.report.ReportNode.NO_OP);
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    /** OLF's own Woodbury-with-actions scenario: absolute and relative phase-tap actions after the line outage. */
    @Test
    void phaseTapActionsMatchStock() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        Network network = com.powsybl.openloadflow.network.PhaseControlFactory.createWithOneT2wtTwoLines();
        addActivePowerLimits(network, 0.7);
        List<Contingency> contingencies = allBranches(network);
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.PhaseTapChangerTapPositionAction("pstAbsChange", "PS1", false, 0),
                new com.powsybl.action.PhaseTapChangerTapPositionAction("pstRelChange", "PS1", true, -1));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("strategyTapAbsChange", ContingencyContext.specificContingency("L1"),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("pstAbsChange")),
                new com.powsybl.contingency.strategy.OperatorStrategy("strategyTapRelChange", ContingencyContext.specificContingency("L1"),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("pstRelChange")),
                new com.powsybl.contingency.strategy.OperatorStrategy("strategyOnViolation", ContingencyContext.specificContingency("L2"),
                        new com.powsybl.contingency.strategy.condition.AnyViolationCondition(), List.of("pstAbsChange")));
        SecurityAnalysisResult cpu = runStockWithStrategies(network, contingencies, strategyMonitors(network), strategies, actions);
        SecurityAnalysisResult gpu = runGpuWithStrategies(network, contingencies, strategyMonitors(network), strategies, actions);
        assertSameResults(cpu, gpu);
        assertAllOnDevice(cpu);
        assertFalse(cpu.getOperatorStrategyResults().isEmpty(), "the strategies must have produced results");
    }

    /** Two phase shifters re-tapped together (two action elements in one Woodbury system). */
    @Test
    void twoPhaseTapActionsMatchStock() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        Network network = com.powsybl.openloadflow.network.PhaseControlFactory.createWithTwoT2wtTwoLines();
        addActivePowerLimits(network, 0.7);
        List<Contingency> contingencies = allBranches(network);
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.PhaseTapChangerTapPositionAction("pst1Change", "PS1", false, 0),
                new com.powsybl.action.PhaseTapChangerTapPositionAction("pst2Change", "PS2", false, 2));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("strategyTapChange", ContingencyContext.specificContingency("L1"),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("pst1Change", "pst2Change")));
        SecurityAnalysisResult cpu = runStockWithStrategies(network, contingencies, strategyMonitors(network), strategies, actions);
        SecurityAnalysisResult gpu = runGpuWithStrategies(network, contingencies, strategyMonitors(network), strategies, actions);
        assertSameResults(cpu, gpu);
        assertFalse(cpu.getOperatorStrategyResults().isEmpty());
    }

    /** Switch actions on the node-breaker network: a topological action, the connectivity re-check, the switch element. */
    @Test
    void switchActionsMatchStock() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        Network network = com.powsybl.openloadflow.network.NodeBreakerNetworkFactory.create();
        addActivePowerLimits(network, 0.8);
        List<Contingency> contingencies = allBranches(network);
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.SwitchAction("openC", "C", true),
                new com.powsybl.action.SwitchAction("closeC", "C", false));
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = new ArrayList<>();
        for (Contingency c : contingencies) {
            strategies.add(new com.powsybl.contingency.strategy.OperatorStrategy("open-" + c.getId(), ContingencyContext.specificContingency(c.getId()),
                    new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("openC")));
            strategies.add(new com.powsybl.contingency.strategy.OperatorStrategy("close-" + c.getId(), ContingencyContext.specificContingency(c.getId()),
                    new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("closeC")));
        }
        SecurityAnalysisResult cpu = runStockWithStrategies(network, contingencies, strategyMonitors(network), strategies, actions);
        SecurityAnalysisResult gpu = runGpuWithStrategies(network, contingencies, strategyMonitors(network), strategies, actions);
        assertSameResults(cpu, gpu);
        assertFalse(cpu.getOperatorStrategyResults().isEmpty());
    }

    /** Generator and load set-point actions: right-hand-side shifts, no action element. */
    @Test
    void generatorAndLoadActionsMatchStock() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        Network network = IeeeCdfNetworkFactory.create14();
        addActivePowerLimits(network, 0.9);
        List<Contingency> contingencies = allBranches(network);
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.GeneratorActionBuilder().withId("gen2Down").withGeneratorId("B2-G").withActivePowerRelativeValue(false).withActivePowerValue(20.0).build(),
                new com.powsybl.action.LoadActionBuilder().withId("load9Down").withLoadId("B9-L").withRelativeValue(true).withActivePowerValue(-15.0).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("genAndLoad", ContingencyContext.specificContingency("L1-2-1"),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("gen2Down", "load9Down")),
                new com.powsybl.contingency.strategy.OperatorStrategy("loadOnly", ContingencyContext.specificContingency("L2-3-1"),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("load9Down")));
        SecurityAnalysisResult cpu = runStockWithStrategies(network, contingencies, strategyMonitors(network), strategies, actions);
        SecurityAnalysisResult gpu = runGpuWithStrategies(network, contingencies, strategyMonitors(network), strategies, actions);
        assertSameResults(cpu, gpu);
        assertFalse(cpu.getOperatorStrategyResults().isEmpty());
    }

    /**
     * Generator and load actions as a DC load flow takes them (OLF's fast DC, 2026-11): one on a generator the contingency
     * disconnects does not apply (B8-G, behind L7-8-1), and with a distributed slack their power enters the balance the
     * distribution restores (an unbalanced redispatch, a load change). The device builds those columns bit for bit.
     */
    @Test
    void injectionActionsEnterTheSlackDistribution() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        Network network = IeeeCdfNetworkFactory.create14();
        addActivePowerLimits(network, 0.9);
        List<Contingency> contingencies = allBranches(network);
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.GeneratorActionBuilder().withId("g8Up").withGeneratorId("B8-G").withActivePowerRelativeValue(true).withActivePowerValue(20.0).build(),
                new com.powsybl.action.GeneratorActionBuilder().withId("g2Down").withGeneratorId("B2-G").withActivePowerRelativeValue(true).withActivePowerValue(-20.0).build(),
                new com.powsybl.action.GeneratorActionBuilder().withId("g3Set").withGeneratorId("B3-G").withActivePowerRelativeValue(false).withActivePowerValue(10.0).build(),
                new com.powsybl.action.LoadActionBuilder().withId("load9Up").withLoadId("B9-L").withRelativeValue(true).withActivePowerValue(15.0).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = new ArrayList<>();
        for (Contingency c : contingencies) {
            strategies.add(new com.powsybl.contingency.strategy.OperatorStrategy("redispatch-" + c.getId(), ContingencyContext.specificContingency(c.getId()),
                    new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("g8Up", "g2Down")));
            strategies.add(new com.powsybl.contingency.strategy.OperatorStrategy("unbalanced-" + c.getId(), ContingencyContext.specificContingency(c.getId()),
                    new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("g3Set", "load9Up")));
        }
        GpuWoodburyDcSecurityAnalysis.CHECK_DEVICE_TARGETS.set(true);
        GpuWoodburyDcSecurityAnalysis.DEVICE_TARGET_COUNT.set(0);
        GpuWoodburyDcSecurityAnalysis.TARGET_MISMATCH_COUNT.set(0);
        try {
            SecurityAnalysisResult cpu = runStockWithStrategies(network, contingencies, strategyMonitors(network), strategies, actions);
            SecurityAnalysisResult gpu = runGpuWithStrategies(network, contingencies, strategyMonitors(network), strategies, actions);
            assertSameResults(cpu, gpu);
            assertEquals(2 * contingencies.size(), cpu.getOperatorStrategyResults().size());
        } finally {
            GpuWoodburyDcSecurityAnalysis.CHECK_DEVICE_TARGETS.set(false);
        }
        assertTrue(GpuWoodburyDcSecurityAnalysis.DEVICE_TARGET_COUNT.get() > 0, "the device must have built the columns");
        assertEquals(0, GpuWoodburyDcSecurityAnalysis.TARGET_MISMATCH_COUNT.get(), "device columns differing from the host's");
    }

    /** The security analysis on cuDSS too (GpuLoadFlowParameters.linearSolver = CUDSS): the default is our batched LU. */
    @Test
    void cudssLinearSolverMatchesStock() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        Network network = IeeeCdfNetworkFactory.create118();
        addActivePowerLimits(network, 0.9);
        List<Contingency> contingencies = allElements(network);
        SecurityAnalysisParameters sap = dcFastParameters();
        com.powsybl.openloadflow.ac.solver.gpu.GpuLoadFlowParameters.create(sap.getLoadFlowParameters())
                .setLinearSolver(com.powsybl.openloadflow.ac.solver.gpu.GpuLinearSolver.CUDSS);
        SecurityAnalysisResult cpu = runStock(network, contingencies, monitorAll(network), sap);
        long before = com.powsybl.openloadflow.ac.solver.gpu.BatchedLuNative.factorizationCounts()[0];
        SecurityAnalysisResult gpu = runGpu(network, contingencies, monitorAll(network), sap);
        assertEquals(0, com.powsybl.openloadflow.ac.solver.gpu.BatchedLuNative.factorizationCounts()[0] - before,
                "cuDSS solved it: our batched LU did not factorize");
        assertSameResults(cpu, gpu);
        assertAllOnDevice(cpu);
    }

    /**
     * Each contingency's report is a child of the network's, as the AC security analysis builds it: siblings, every one
     * at the same depth, not each nested in the previous contingency's (which also made each report re-check the
     * values of every report before it).
     */
    @Test
    void postContingencyReportsAreSiblings() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        Network network = IeeeCdfNetworkFactory.create14();
        List<Contingency> contingencies = allBranches(network);
        com.powsybl.commons.report.ReportNode root = com.powsybl.commons.report.ReportNode.newRootReportNode()
                .withMessageTemplate("test").build();
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            runSecurityAnalysis(network, contingencies, monitorAll(network), dcFastParameters(), root);
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
        java.util.Map<Integer, Integer> postContingencyNodesByDepth = new java.util.TreeMap<>();
        java.util.Deque<java.util.Map.Entry<com.powsybl.commons.report.ReportNode, Integer>> stack = new java.util.ArrayDeque<>();
        stack.push(java.util.Map.entry(root, 0));
        while (!stack.isEmpty()) {
            var e = stack.pop();
            if ("olf.postContingencySimulation".equals(e.getKey().getMessageKey())) {
                postContingencyNodesByDepth.merge(e.getValue(), 1, Integer::sum);
            }
            e.getKey().getChildren().forEach(child -> stack.push(java.util.Map.entry(child, e.getValue() + 1)));
        }
        assertEquals(1, postContingencyNodesByDepth.size(), "post-contingency reports at several depths: " + postContingencyNodesByDepth);
        assertTrue(postContingencyNodesByDepth.values().iterator().next() >= 10, "the contingencies have their reports");
    }

    /**
     * Each contingency answered from its state, without loading it into OLF's model (the AC path's design): every branch
     * value the state reader gives (P, I, S of both sides) is OLF's own evaluation with the state loaded, bit for bit, the
     * monitored results are OLF's, and the results match the stock arm. Networks with islanding, busbar sections, a
     * VSC HVDC link and a phase shifter; the modification threshold at 0 and at 1 MW.
     */
    @Test
    void contingenciesAnsweredFromTheirState() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        GpuWoodburyDcSecurityAnalysis.CHECK_STATE_READER.set(true);
        GpuWoodburyDcSecurityAnalysis.STATE_READER_MISMATCH_COUNT.set(0);
        GpuWoodburyDcSecurityAnalysis.STATE_FREE_CONTINGENCY_COUNT.set(0);
        try {
            List<java.util.function.Supplier<Network>> networks = List.of(
                    IeeeCdfNetworkFactory::create14,
                    IeeeCdfNetworkFactory::create118,
                    com.powsybl.openloadflow.network.NodeBreakerNetworkFactory::create,
                    com.powsybl.openloadflow.network.HvdcNetworkFactory::createTwoCcLinkedByAHvdcVscWithGenerators,
                    com.powsybl.openloadflow.network.PhaseControlFactory::createWithOneT2wtTwoLines);
            for (java.util.function.Supplier<Network> factory : networks) {
                for (int variant = 0; variant < 2; variant++) {
                    Network network = factory.get();
                    addActivePowerLimits(network, 0.9);
                    List<Contingency> contingencies = new ArrayList<>(allElements(network));
                    network.getBusbarSectionStream().forEach(b -> contingencies.add(new Contingency(b.getId(),
                            new com.powsybl.contingency.BusbarSectionContingency(b.getId()))));
                    Set<String> branchIds = network.getBranchStream().map(Branch::getId).collect(Collectors.toCollection(LinkedHashSet::new));
                    List<StateMonitor> monitors = List.of(new StateMonitor(ContingencyContext.all(), branchIds, Set.of(), Set.of()));
                    SecurityAnalysisParameters sap = dcFastParameters();
                    if (variant == 1) {
                        sap.getModifiedMonitoredElementsParameters().setPowerModificationThreshold(1);
                    }
                    SecurityAnalysisResult cpu = runStock(network, contingencies, monitors, sap);
                    SecurityAnalysisResult gpu = runGpu(network, contingencies, monitors, sap);
                    assertSameResults(cpu, gpu);
                }
            }
        } finally {
            GpuWoodburyDcSecurityAnalysis.CHECK_STATE_READER.set(false);
        }
        assertTrue(GpuWoodburyDcSecurityAnalysis.STATE_FREE_CONTINGENCY_COUNT.get() > 500,
                "contingencies answered from their state: " + GpuWoodburyDcSecurityAnalysis.STATE_FREE_CONTINGENCY_COUNT.get());
        assertEquals(0, GpuWoodburyDcSecurityAnalysis.STATE_READER_MISMATCH_COUNT.get(), "values differing between the state reader and OLF's model");
    }

    /** The operator strategy tests' monitors: every branch and voltage level, or (strategiesAnsweredFromTheirState)
     *  every branch only, so the strategies are answered from their device state. */
    private boolean branchOnlyStrategyMonitors;

    private List<StateMonitor> strategyMonitors(Network network) {
        if (!branchOnlyStrategyMonitors) {
            return monitorAll(network);
        }
        Set<String> branchIds = network.getBranchStream().map(Branch::getId).collect(Collectors.toCollection(LinkedHashSet::new));
        return List.of(new StateMonitor(ContingencyContext.all(), branchIds, Set.of(), Set.of()));
    }

    /**
     * Operator strategies answered from their device state, as the AC path starts a strategy from it: the actions applied
     * on the network (topology, taps), every value read from the state (the action branches on their current model),
     * bit for bit OLF's with the state loaded; the strategies' target columns built on the device, bit for bit the host
     * replay's; and voltage-angle limits checked on the state's angles. The strategy tests
     * rerun with branch monitors only, then a network with violated voltage-angle limits.
     */
    @Test
    void strategiesAndAngleLimitsAnsweredFromTheirState() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        GpuWoodburyDcSecurityAnalysis.CHECK_STATE_READER.set(true);
        GpuWoodburyDcSecurityAnalysis.STATE_READER_MISMATCH_COUNT.set(0);
        GpuWoodburyDcSecurityAnalysis.STATE_FREE_CONTINGENCY_COUNT.set(0);
        GpuWoodburyDcSecurityAnalysis.CHECK_DEVICE_TARGETS.set(true);
        GpuWoodburyDcSecurityAnalysis.DEVICE_TARGET_COUNT.set(0);
        GpuWoodburyDcSecurityAnalysis.TARGET_MISMATCH_COUNT.set(0);
        branchOnlyStrategyMonitors = true;
        try {
            phaseTapActionsMatchStock();
            twoPhaseTapActionsMatchStock();
            switchActionsMatchStock();
            generatorAndLoadActionsMatchStock();

            Network network = IeeeCdfNetworkFactory.create14();
            addActivePowerLimits(network, 0.9);
            network.newVoltageAngleLimit().setId("VAL-1-14").from(network.getGenerator("B1-G").getTerminal())
                    .to(network.getLoad("B14-L").getTerminal()).setHighLimit(0.05).setLowLimit(-0.05).add();
            network.newVoltageAngleLimit().setId("VAL-2-9").from(network.getGenerator("B2-G").getTerminal())
                    .to(network.getLoad("B9-L").getTerminal()).setHighLimit(0.02).add();
            List<Contingency> contingencies = allBranches(network);
            List<StateMonitor> monitors = strategyMonitors(network);
            SecurityAnalysisResult cpu = runStock(network, contingencies, monitors, dcFastParameters());
            SecurityAnalysisResult gpu = runGpu(network, contingencies, monitors, dcFastParameters());
            assertSameResults(cpu, gpu);
            assertTrue(cpu.getPostContingencyResults().stream().anyMatch(r -> r.getLimitViolationsResult().getLimitViolations().stream()
                    .anyMatch(v -> v.getLimitType() == com.powsybl.contingency.violations.LimitViolationType.HIGH_VOLTAGE_ANGLE
                            || v.getLimitType() == com.powsybl.contingency.violations.LimitViolationType.LOW_VOLTAGE_ANGLE)),
                    "the voltage-angle limits must be violated somewhere");
        } finally {
            branchOnlyStrategyMonitors = false;
            GpuWoodburyDcSecurityAnalysis.CHECK_STATE_READER.set(false);
            GpuWoodburyDcSecurityAnalysis.CHECK_DEVICE_TARGETS.set(false);
        }
        assertEquals(0, GpuWoodburyDcSecurityAnalysis.STATE_READER_MISMATCH_COUNT.get(), "values differing between the state reader and OLF's model");
        // the strategies' target columns built on the device (contingency changes + the actions' row edits) are the replay's
        assertEquals(0, GpuWoodburyDcSecurityAnalysis.TARGET_MISMATCH_COUNT.get(), "device target values differing from the replay's");
        assertTrue(GpuWoodburyDcSecurityAnalysis.DEVICE_TARGET_COUNT.get() > 0, "the device built target columns");
        assertTrue(GpuWoodburyDcSecurityAnalysis.STATE_FREE_CONTINGENCY_COUNT.get() > 40,
                "answered from their state: " + GpuWoodburyDcSecurityAnalysis.STATE_FREE_CONTINGENCY_COUNT.get());
    }

    @Test
    void ieee14WithoutMonitorsMatchStock() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        Network network = IeeeCdfNetworkFactory.create14();
        addActivePowerLimits(network, 0.8);
        List<Contingency> contingencies = allBranches(network);
        SecurityAnalysisResult cpu = runStock(network, contingencies, List.of());
        SecurityAnalysisResult gpu = runGpu(network, contingencies, List.of());
        assertSameResults(cpu, gpu);
        assertTrue(GpuWoodburyDcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get() > 0);
    }
}
