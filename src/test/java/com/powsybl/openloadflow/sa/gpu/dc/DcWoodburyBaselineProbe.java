/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu.dc;

import com.powsybl.contingency.BranchContingency;
import com.powsybl.contingency.Contingency;
import com.powsybl.contingency.violations.LimitViolation;
import com.powsybl.iidm.network.Branch;
import com.powsybl.iidm.network.Bus;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.json.JsonLoadFlowParameters;
import com.powsybl.math.matrix.CuDssMatrixFactory;
import com.powsybl.math.matrix.MatrixFactory;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.sa.OpenSecurityAnalysisParameters;
import com.powsybl.openloadflow.sa.OpenSecurityAnalysisProvider;
import com.powsybl.security.SecurityAnalysisParameters;
import com.powsybl.security.SecurityAnalysisResult;
import com.powsybl.security.SecurityAnalysisRunParameters;
import com.powsybl.security.monitor.StateMonitor;
import com.powsybl.security.results.BranchResult;
import com.powsybl.security.results.PostContingencyResult;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * DC port, phase 0 — the ZERO-code baseline (docs/dc-port-plan.md §4).
 *
 * <p>Stock {@code WoodburyDcSecurityAnalysis} (dcFastMode) on rte6515_full, ALL-branch N-1 of the main
 * synchronous component, RTE production load-flow parameters switched to DC, run twice: on KLU
 * ({@link SparseMatrixFactory}) and on cuDSS ({@link CuDssMatrixFactory}, the core fork's GPU LU behind
 * OLF's unchanged Java). Reports the wall time of each arm and checks the two agree: per-contingency
 * statuses, the violation sets (keyed subject|limitType|side|limitName, values compared) and the
 * post-contingency P1 of a monitored branch sample. The disagreement between the arms is
 * LU-ordering roundoff only (cond * eps ~ 4e-12 relative; measured 1.2e-11 pu) — anything above 1e-10 pu
 * is a bug, never a tolerance to widen.
 *
 * <p>Knobs: {@code OLF_DC_MAX_CTG} caps the contingency list (default: all), {@code OLF_DC_THREADS}
 * sets the CPU arms' threadCount (default 8 — the GPU arm is single-threaded by construction), {@code OLF_DC_MONITORED} the number of monitored branches
 * (default 100), {@code OLF_DC_SKIP_CUDSS=1} runs the KLU arm only (CPU profiling runs).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class DcWoodburyBaselineProbe {

    /** {@code OLF_DC_CASE_FILE}: the network (default rte6515_full.xiidm.gz, e.g. network.xiidm). */
    private static final Path CASE_FILE = Path.of(System.getenv().getOrDefault("OLF_DC_CASE_FILE", "rte6515_full.xiidm.gz"));
    /** {@code OLF_DC_PARAMS_FILE}: the load flow parameters (default the RTE ones at the repo root), switched to DC. */
    private static final Path PARAMS_FILE = Path.of(System.getenv().getOrDefault("OLF_DC_PARAMS_FILE", "rte-loadflow-default-parameters.json"));

    private record Arm(String name, SecurityAnalysisResult result, long wallMs) {
    }

    /** Plain-JVM entry point (profiling runs: {@code java -XX:StartFlightRecording ... DcWoodburyBaselineProbe}). */
    public static void main(String[] args) {
        new DcWoodburyBaselineProbe().woodburyKluVersusCuDss();
    }

    @Test
    void woodburyKluVersusCuDss() {
        assumeTrue(Files.exists(CASE_FILE), CASE_FILE + " not found in module dir");
        assumeTrue(Files.exists(PARAMS_FILE), PARAMS_FILE + " not found in module dir");

        Network network = Network.read(CASE_FILE);
        // OLF_DC_CTG_FILE: a contingency list (JSON array of {id, elements}, or an object with "contingencies")
        // instead of every main-synchronous-component branch
        String ctgFile = System.getenv("OLF_DC_CTG_FILE");
        List<Contingency> contingencies = ctgFile != null && !ctgFile.isEmpty()
                ? readContingencies(Path.of(ctgFile))
                : mainComponentBranchContingencies(network);
        int maxCtg = Integer.parseInt(System.getenv().getOrDefault("OLF_DC_MAX_CTG", "0"));
        if (maxCtg > 0 && contingencies.size() > maxCtg) {
            contingencies = contingencies.subList(0, maxCtg);
        }
        int monitored = Integer.parseInt(System.getenv().getOrDefault("OLF_DC_MONITORED", "100"));
        List<String> monitoredBranches = mainComponentBranchContingencies(network).stream().limit(monitored).map(Contingency::getId).toList();
        // the performance comparison: GPU single-thread vs the CPU with 8 threads (user rule, 2026-09-15)
        int threads = Integer.parseInt(System.getenv().getOrDefault("OLF_DC_THREADS", "8"));

        System.out.printf("DC-BASELINE %s | %d contingencies (%s) | %d monitored branches | CPU threadCount %d, GPU 1%n",
                CASE_FILE.getFileName(), contingencies.size(), ctgFile != null && !ctgFile.isEmpty() ? ctgFile : "main-component branch N-1", monitoredBranches.size(), threads);

        if ("1".equals(System.getenv("OLF_DC_SKIP_KLU")) && "1".equals(System.getenv("OLF_DC_GPU"))) {
            // profiling run: the GPU arm alone
            assumeTrue(com.powsybl.openloadflow.dc.gpu.GpuDcSolver.isAvailable(), "GPU native library not available");
            Arm gpu = run("GPU", new SparseMatrixFactory(), network, contingencies, monitoredBranches, 1, true);
            System.out.printf("DC-BASELINE %-5s wall %6d ms | %.3f ms/ctg | statuses %s | violations %d (GPU-only run)%n",
                    gpu.name(), gpu.wallMs(), gpu.wallMs() / (double) contingencies.size(), statusCounts(gpu.result()), violationCount(gpu.result()));
            return;
        }
        if ("1".equals(System.getenv("OLF_DC_SKIP_KLU"))) {
            // isolation run: the cuDSS arm alone, in a JVM that never loaded KLU
            assumeTrue(CuDssMatrixFactory.isAvailable(), "cuDSS matrix factory not available");
            Arm only = run("cuDSS", new CuDssMatrixFactory(), network, contingencies, monitoredBranches, threads);
            System.out.printf("DC-BASELINE %-5s wall %6d ms | %.3f ms/ctg | statuses %s | violations %d (cuDSS-only run)%n",
                    only.name(), only.wallMs(), only.wallMs() / (double) contingencies.size(), statusCounts(only.result()), violationCount(only.result()));
            return;
        }
        Arm klu = run("KLU", new SparseMatrixFactory(), network, contingencies, monitoredBranches, threads);
        System.out.printf("DC-BASELINE %-5s wall %6d ms | %.3f ms/ctg | statuses %s | violations %d%n",
                klu.name(), klu.wallMs(), klu.wallMs() / (double) contingencies.size(), statusCounts(klu.result()), violationCount(klu.result()));

        if ("1".equals(System.getenv("OLF_DC_GPU"))) {
            // phase 1: the GPU Woodbury path (device S, in-kernel Woodbury, flows, screen) vs the stock KLU arm
            assumeTrue(com.powsybl.openloadflow.dc.gpu.GpuDcSolver.isAvailable(), "GPU native library not available");
            com.powsybl.openloadflow.sa.gpu.GpuWoodburyDcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.set(-1);
            Arm gpu = run("GPU", new SparseMatrixFactory(), network, contingencies, monitoredBranches, 1, true);
            int onDevice = com.powsybl.openloadflow.sa.gpu.GpuWoodburyDcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get();
            System.out.printf("DC-BASELINE %-5s wall %6d ms | %.3f ms/ctg | statuses %s | violations %d | x%.2f vs KLU | %d/%d on the device%n",
                    gpu.name(), gpu.wallMs(), gpu.wallMs() / (double) contingencies.size(), statusCounts(gpu.result()),
                    violationCount(gpu.result()), klu.wallMs() / (double) gpu.wallMs(), onDevice, contingencies.size());
            compare(klu, gpu, contingencies.size());
            assertTrue(onDevice > 0, "the device must have solved contingencies");
            return;
        }
        if ("1".equals(System.getenv("OLF_DC_SKIP_CUDSS"))) {
            return;
        }
        assumeTrue(CuDssMatrixFactory.isAvailable(), "cuDSS matrix factory not available");
        Arm cudss = run("cuDSS", new CuDssMatrixFactory(), network, contingencies, monitoredBranches, threads);
        System.out.printf("DC-BASELINE %-5s wall %6d ms | %.3f ms/ctg | statuses %s | violations %d | x%.2f vs KLU%n",
                cudss.name(), cudss.wallMs(), cudss.wallMs() / (double) contingencies.size(), statusCounts(cudss.result()),
                violationCount(cudss.result()), klu.wallMs() / (double) cudss.wallMs());

        compare(klu, cudss, contingencies.size());
    }

    private static void compare(Arm klu, Arm cudss, int contingencyCount) {
        // --- parity: the two arms run the SAME Java, only the LU differs
        assertEquals(statusCounts(klu.result()), statusCounts(cudss.result()), "post-contingency statuses differ");
        assertEquals(klu.result().getPreContingencyResult().getStatus(), cudss.result().getPreContingencyResult().getStatus());

        Map<String, Map<String, LimitViolation>> vKlu = violationsByCtg(klu.result());
        Map<String, Map<String, LimitViolation>> vCudss = violationsByCtg(cudss.result());
        int differingLines = 0;
        double maxValueDiff = 0;
        for (String ctg : new TreeSet<>(union(vKlu.keySet(), vCudss.keySet()))) {
            Map<String, LimitViolation> a = vKlu.getOrDefault(ctg, Map.of());
            Map<String, LimitViolation> b = vCudss.getOrDefault(ctg, Map.of());
            for (String key : union(a.keySet(), b.keySet())) {
                LimitViolation va = a.get(key);
                LimitViolation vb = b.get(key);
                if (va == null || vb == null) {
                    if (differingLines < 20) {
                        System.out.printf("DC-BASELINE DIFF ctg=%s %s only in %s (value %.6f limit %.6f)%n", ctg, key,
                                va == null ? "cuDSS" : "KLU", (va == null ? vb : va).getValue(), (va == null ? vb : va).getLimit());
                    }
                    differingLines++;
                } else {
                    maxValueDiff = Math.max(maxValueDiff, Math.abs(va.getValue() - vb.getValue()));
                }
            }
        }
        System.out.printf("DC-BASELINE violation parity: %d differing lines, max common-violation value diff %.3e%n", differingLines, maxValueDiff);

        // --- monitored branch P1 sample
        Map<String, Map<String, Double>> pKlu = monitoredP1(klu.result());
        Map<String, Map<String, Double>> pCudss = monitoredP1(cudss.result());
        double maxP1Diff = 0;
        long compared = 0;
        Map<String, Double> worstByCtg = new HashMap<>();
        String worst = "";
        for (var e : pKlu.entrySet()) {
            Map<String, Double> other = pCudss.get(e.getKey());
            if (other == null) {
                continue;
            }
            for (var b : e.getValue().entrySet()) {
                Double o = other.get(b.getKey());
                if (o != null && Double.isFinite(o) && Double.isFinite(b.getValue())) {
                    double d = Math.abs(o - b.getValue());
                    if (d > maxP1Diff) {
                        worst = String.format("ctg=%s branch=%s KLU=%.9f other=%.9f", e.getKey(), b.getKey(), b.getValue(), o);
                    }
                    maxP1Diff = Math.max(maxP1Diff, d);
                    worstByCtg.merge(e.getKey(), d, Math::max);
                    compared++;
                }
            }
        }
        System.out.printf("DC-BASELINE monitored P1: %d values compared, max |diff| %.3e MW (%s)%n", compared, maxP1Diff, worst);
        long above1e8 = worstByCtg.values().stream().filter(d -> d > 1e-8).count();
        long above1e6 = worstByCtg.values().stream().filter(d -> d > 1e-6).count();
        System.out.printf("DC-BASELINE monitored P1: contingencies with a diff > 1e-8 MW: %d, > 1e-6 MW: %d, of %d%n",
                above1e8, above1e6, worstByCtg.size());
        worstByCtg.entrySet().stream().sorted(Map.Entry.<String, Double>comparingByValue().reversed()).limit(10)
                .forEach(e -> System.out.printf("DC-BASELINE   worst ctg %s %.3e MW%n", e.getKey(), e.getValue()));

        // Two different LU codes on the same ~4e4-conditioned DC matrix (max|v|/min|v|) differ by
        // cond * eps ~ 4e-12 relative: measured 1.2e-9 MW = 1.2e-11 pu on 9042 contingencies. The
        // bound is 1e-10 pu (1e-8 MW), the user's "anything above 1e-10 is wrong" line — NOT a
        // convergence tolerance, there is no iteration here to loosen.
        assertEquals(0, differingLines, "violation sets differ between KLU and cuDSS");
        assertTrue(maxValueDiff < 1e-8, "common-violation values differ by " + maxValueDiff + " MW (> 1e-10 pu)");
        assertTrue(maxP1Diff < 1e-8, "monitored P1 differ by " + maxP1Diff + " MW (> 1e-10 pu)");
    }

    private static Arm run(String name, MatrixFactory matrixFactory, Network network, List<Contingency> contingencies,
                           List<String> monitoredBranches, int threads) {
        return run(name, matrixFactory, network, contingencies, monitoredBranches, threads, false);
    }

    private static Arm run(String name, MatrixFactory matrixFactory, Network network, List<Contingency> contingencies,
                           List<String> monitoredBranches, int threads, boolean gpuProvider) {
        LoadFlowParameters lfp = JsonLoadFlowParameters.read(PARAMS_FILE);
        lfp.setDc(true);
        SecurityAnalysisParameters sap = new SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        sap.addExtension(OpenSecurityAnalysisParameters.class,
                new OpenSecurityAnalysisParameters().setDcFastMode(true).setThreadCount(threads));
        List<StateMonitor> monitors = List.of(new StateMonitor(com.powsybl.contingency.ContingencyContext.all(),
                new LinkedHashSet<>(monitoredBranches), Set.of(), Set.of()));
        OpenSecurityAnalysisProvider provider = gpuProvider
                ? new com.powsybl.openloadflow.sa.gpu.GpuSecurityAnalysisProvider(matrixFactory, new EvenShiloachGraphDecrementalConnectivityFactory<>())
                : new OpenSecurityAnalysisProvider(matrixFactory, new EvenShiloachGraphDecrementalConnectivityFactory<>());
        SecurityAnalysisRunParameters runParameters = new SecurityAnalysisRunParameters()
                .setSecurityAnalysisParameters(sap)
                .setMonitors(monitors);
        long t0 = System.nanoTime();
        SecurityAnalysisResult result = provider.run(network, network.getVariantManager().getWorkingVariantId(),
                        n -> contingencies, runParameters)
                .join().getResult();
        long wallMs = (System.nanoTime() - t0) / 1_000_000;
        return new Arm(name, result, wallMs);
    }

    private static List<Contingency> readContingencies(Path file) {
        try {
            com.fasterxml.jackson.databind.ObjectMapper om = com.powsybl.commons.json.JsonUtil.createObjectMapper()
                    .registerModule(new com.powsybl.contingency.json.ContingencyJsonModule());
            com.fasterxml.jackson.databind.JsonNode root = om.readTree(file.toFile());
            com.fasterxml.jackson.databind.JsonNode array = root.isArray() ? root : root.get("contingencies");
            return new ArrayList<>(om.readerFor(new com.fasterxml.jackson.core.type.TypeReference<List<Contingency>>() { })
                    .readValue(array));
        } catch (java.io.IOException e) {
            throw new java.io.UncheckedIOException(e);
        }
    }

    /** All branches connected on both sides inside the main synchronous component, in network order. */
    static List<Contingency> mainComponentBranchContingencies(Network network) {
        List<Contingency> list = new ArrayList<>();
        for (Branch<?> branch : network.getBranches()) {
            Bus b1 = branch.getTerminal1().getBusView().getBus();
            Bus b2 = branch.getTerminal2().getBusView().getBus();
            if (b1 != null && b2 != null && b1.getSynchronousComponent() != null && b2.getSynchronousComponent() != null
                    && b1.getSynchronousComponent().getNum() == 0 && b2.getSynchronousComponent().getNum() == 0) {
                list.add(new Contingency(branch.getId(), new BranchContingency(branch.getId())));
            }
        }
        return list;
    }

    private static Map<String, Long> statusCounts(SecurityAnalysisResult r) {
        Map<String, Long> counts = new TreeMap<>();
        for (PostContingencyResult p : r.getPostContingencyResults()) {
            counts.merge(p.getStatus().name(), 1L, Long::sum);
        }
        return counts;
    }

    private static long violationCount(SecurityAnalysisResult r) {
        return r.getPreContingencyResult().getLimitViolationsResult().getLimitViolations().size()
                + r.getPostContingencyResults().stream()
                .mapToLong(p -> p.getLimitViolationsResult().getLimitViolations().size())
                .sum();
    }

    private static String key(LimitViolation v) {
        return v.getSubjectId() + "|" + v.getLimitType() + "|" + v.getSide() + "|" + v.getLimitName();
    }

    private static Map<String, Map<String, LimitViolation>> violationsByCtg(SecurityAnalysisResult r) {
        Map<String, Map<String, LimitViolation>> m = new HashMap<>();
        m.put("", r.getPreContingencyResult().getLimitViolationsResult().getLimitViolations().stream()
                .collect(Collectors.toMap(DcWoodburyBaselineProbe::key, v -> v, (a, b) -> a)));
        for (PostContingencyResult p : r.getPostContingencyResults()) {
            m.put(p.getContingency().getId(), p.getLimitViolationsResult().getLimitViolations().stream()
                    .collect(Collectors.toMap(DcWoodburyBaselineProbe::key, v -> v, (a, b) -> a)));
        }
        return m;
    }

    private static Map<String, Map<String, Double>> monitoredP1(SecurityAnalysisResult r) {
        Map<String, Map<String, Double>> m = new HashMap<>();
        for (PostContingencyResult p : r.getPostContingencyResults()) {
            Map<String, Double> byBranch = new HashMap<>();
            for (BranchResult br : p.getNetworkResult().getBranchResults()) {
                byBranch.put(br.getBranchId(), br.getP1());
            }
            m.put(p.getContingency().getId(), byBranch);
        }
        Map<String, Double> base = new HashMap<>();
        for (BranchResult br : r.getPreContingencyResult().getNetworkResult().getBranchResults()) {
            base.put(br.getBranchId(), br.getP1());
        }
        m.put("<pre-contingency>", base);
        return m;
    }

    private static <T> Set<T> union(Set<T> a, Set<T> b) {
        Set<T> u = new HashSet<>(a);
        u.addAll(b);
        return u;
    }
}
