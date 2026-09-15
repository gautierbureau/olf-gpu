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
import com.powsybl.iidm.network.LimitType;
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
        securityAnalysisProvider = new OpenSecurityAnalysisProvider(new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        return runSecurityAnalysis(network, contingencies, monitors, dcFastParameters());
    }

    private SecurityAnalysisResult runGpu(Network network, List<Contingency> contingencies, List<StateMonitor> monitors) {
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        GpuWoodburyDcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.set(-1);
        try {
            return runSecurityAnalysis(network, contingencies, monitors, dcFastParameters());
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
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
        // the two parallel lines batch; the transformers island a bus and go to the stock path
        assertEquals(2, GpuWoodburyDcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "contingencies solved on the device");
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
        assertTrue(onDevice >= 15, "expected most of the 20 branch outages on the device, got " + onDevice);
        long violations = cpu.getPostContingencyResults().stream().mapToLong(r -> r.getLimitViolationsResult().getLimitViolations().size()).sum();
        assertTrue(violations > 20, "the case must produce post-contingency violations, got " + violations);
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
