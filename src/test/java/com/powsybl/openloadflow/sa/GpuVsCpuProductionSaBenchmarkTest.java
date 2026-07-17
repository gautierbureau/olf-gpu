/*
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa;

import com.powsybl.contingency.BranchContingency;
import com.powsybl.contingency.Contingency;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.json.JsonLoadFlowParameters;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.CommonTestConfig;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuTestPaths;
import com.powsybl.openloadflow.ac.solver.GpuBatchedSecurityAnalysisSolver;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.security.SecurityAnalysisParameters;
import com.powsybl.security.SecurityAnalysisResult;
import com.powsybl.security.results.PostContingencyResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * PRODUCTION-parameters security-analysis benchmark: the real {@link AcSecurityAnalysis} product on
 * both sides — CPU ({@link OpenSecurityAnalysisProvider}) vs GPU-batched
 * ({@link GpuSecurityAnalysisProvider}) — on PEGASE-9241 with the SAME contingencies (the matched
 * 500 from {@code OLF_GPU_CONT_FILE}) and the SAME parameters: THE RTE production parameter file
 * ({@code rte-loadflow-default-parameters.json} at the repo root, loaded via
 * {@link JsonLoadFlowParameters}), full outer loops on both sides. Unlike {@link
 * com.powsybl.openloadflow.ac.equations.vector.gpu.GpuBatchedN1FixedIterTest} (bare fixed-iteration
 * throughput, tol == 0, no violations) this measures what a production run actually costs:
 * convergence-based iteration, outer-loop passes, retirement, violation detection.
 *
 * <p>Reported COLD-FIRST: the GPU's first run includes CUDA init + extraction + cuDSS analysis; a
 * second GPU run shows the warm-service cost. The CPU runs once (its long run amortizes JIT).
 * Result parity (post-contingency status counts) is asserted, not just printed.
 *
 * <p>Machine-local: skipped unless the GPU lib is available AND the case + contingency file exist.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuVsCpuProductionSaBenchmarkTest extends AbstractOpenSecurityAnalysisTest {

    private static final Path CASES = GpuTestPaths.caseDir();
    private static final String CASE = "case9241pegase.xiidm.gz";

    static {
        GpuTestPaths.init();
    }

    GpuVsCpuProductionSaBenchmarkTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    @BeforeEach
    void checkGpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU native lib not available");
        assumeTrue(Files.exists(CASES.resolve(CASE)), CASE + " not found");
        assumeTrue(System.getenv("OLF_GPU_CONT_FILE") != null, "OLF_GPU_CONT_FILE not set");
        // Same env-to-property plumbing as the fixed-iter bench (env is inherited by the surefire fork).
        if (System.getProperty(GpuBatchedSecurityAnalysisSolver.MAX_BATCH_PROPERTY) == null) {
            System.setProperty(GpuBatchedSecurityAnalysisSolver.MAX_BATCH_PROPERTY,
                    System.getenv().getOrDefault("OLF_GPU_MAXBATCH", "32"));
        }
        if (System.getProperty(GpuBatchedSecurityAnalysisSolver.BATCH_MODE_PROPERTY) == null) {
            System.setProperty(GpuBatchedSecurityAnalysisSolver.BATCH_MODE_PROPERTY,
                    System.getenv().getOrDefault("OLF_GPU_BATCHMODE", "0"));
        }
        // SPARSE on both arms — the harness default (commonTestConfig.matrixFactory()) can be DENSE,
        // which at 9241 buses makes the base LF thrash for ~200 s and FAIL (measured), while the same
        // case + parameters converge in ~2 s / 10 iterations with a sparse factory.
        securityAnalysisProvider = new OpenSecurityAnalysisProvider(new SparseMatrixFactory(),
                new EvenShiloachGraphDecrementalConnectivityFactory<>());
    }

    private static List<Contingency> loadContingencies(Network network) throws Exception {
        Path file = Path.of(System.getenv("OLF_GPU_CONT_FILE"));
        return Files.readAllLines(file).stream()
                .map(String::trim)
                .filter(id -> !id.isEmpty())
                .filter(id -> network.getBranch(id) != null)
                .map(id -> new Contingency(id, new BranchContingency(id)))
                .toList();
    }

    private static Map<String, Long> statusCounts(SecurityAnalysisResult r) {
        Map<String, Long> counts = new TreeMap<>();
        for (PostContingencyResult p : r.getPostContingencyResults()) {
            counts.merge(p.getStatus().name(), 1L, Long::sum);
        }
        return counts;
    }

    private static long violationCount(SecurityAnalysisResult r) {
        return r.getPostContingencyResults().stream()
                .mapToLong(p -> p.getLimitViolationsResult().getLimitViolations().size())
                .sum();
    }

    @Test
    void productionSaGpuVsCpu() throws Exception {
        Network network = Network.read(CASES.resolve(CASE));
        List<Contingency> contingencies = loadContingencies(network);
        // THE RTE production parameters, loaded from the JSON checked in at the repo root
        // (rte-loadflow-default-parameters.json): DC init, reactive limits, twt split shunt
        // admittance, HVDC AC emulation, transformer voltage control AFTER_GENERATOR_VOLTAGE_CONTROL,
        // MAX_VOLTAGE_CHANGE state-vector scaling, maxNR/maxOuterLoop 30, realistic-voltage checks.
        Path paramsFile = Path.of("rte-loadflow-default-parameters.json");
        assumeTrue(Files.exists(paramsFile), "rte-loadflow-default-parameters.json not found in module dir");
        LoadFlowParameters lfp = JsonLoadFlowParameters.read(paramsFile);

        // The FAIR production CPU arm is multithreaded — OLF SA parallelizes contingencies across
        // threads in production. The single-thread arm is an optional reference (OLF_GPU_SA_CPU_SINGLE=1).
        // Default 4 threads (this machine's realistic budget); override with OLF_GPU_SA_CPU_THREADS.
        int threads = Integer.parseInt(System.getenv().getOrDefault("OLF_GPU_SA_CPU_THREADS", "4"));
        SecurityAnalysisParameters sap = new SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        sap.addExtension(OpenSecurityAnalysisParameters.class, new OpenSecurityAnalysisParameters().setThreadCount(threads));

        long cpu1Ms = -1;
        if ("1".equals(System.getenv("OLF_GPU_SA_CPU_SINGLE"))) {
            long c1 = System.nanoTime();
            runSecurityAnalysis(network, contingencies, lfp);
            cpu1Ms = (System.nanoTime() - c1) / 1_000_000;
        }
        long cpuT0 = System.nanoTime();
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, contingencies, Collections.emptyList(), sap);
        long cpuMs = (System.nanoTime() - cpuT0) / 1_000_000;

        securityAnalysisProvider = new GpuSecurityAnalysisProvider(new SparseMatrixFactory(),
                new EvenShiloachGraphDecrementalConnectivityFactory<>());
        SecurityAnalysisResult gpuCold;
        SecurityAnalysisResult gpuWarm;
        long gpuColdMs;
        long gpuWarmMs;
        try {
            // GPU arm runs host-side SINGLE-threaded: threadCount > 1 makes the SA harness call the
            // batched native path from several ForkJoinPool workers concurrently, which SIGSEGVs in
            // cuLaunchKernel (measured — hs_err in precomputeBatch). The GPU's parallelism IS the
            // batch; provider-side serialization is the product fix to make threadCount safe.
            SecurityAnalysisParameters sapGpu = new SecurityAnalysisParameters();
            sapGpu.setLoadFlowParameters(lfp);
            long g0 = System.nanoTime();
            gpuCold = runSecurityAnalysis(network, contingencies, Collections.emptyList(), sapGpu);
            gpuColdMs = (System.nanoTime() - g0) / 1_000_000;
            long g1 = System.nanoTime();
            gpuWarm = runSecurityAnalysis(network, contingencies, Collections.emptyList(), sapGpu);
            gpuWarmMs = (System.nanoTime() - g1) / 1_000_000;
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(new SparseMatrixFactory(),
                    new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }

        int n = contingencies.size();
        System.out.printf("PROD-SA case9241pegase | %d ctg | RTE production parameters (rte-loadflow-default-parameters.json), full outer loops%n", n);
        if (cpu1Ms >= 0) {
            System.out.printf("  CPU  AcSecurityAnalysis (1 thread)   : %6d ms | %.2f ms/ctg%n", cpu1Ms, cpu1Ms / (double) n);
        }
        System.out.printf("  CPU  AcSecurityAnalysis (%2d threads) : %6d ms | %.2f ms/ctg%n", threads, cpuMs, cpuMs / (double) n);
        System.out.printf("  GPU  batched SA (COLD, run 1)        : %6d ms | %.2f ms/ctg | vs CPU-MT x%.2f%n", gpuColdMs, gpuColdMs / (double) n, cpuMs / (double) gpuColdMs);
        System.out.printf("  GPU  batched SA (warm, run 2)        : %6d ms | %.2f ms/ctg | vs CPU-MT x%.2f%n", gpuWarmMs, gpuWarmMs / (double) n, cpuMs / (double) gpuWarmMs);
        System.out.printf("  pre-contingency CPU=%s GPU=%s%n",
                cpu.getPreContingencyResult().getStatus(), gpuCold.getPreContingencyResult().getStatus());
        System.out.printf("  statuses CPU=%s GPU(cold)=%s | violations CPU=%d GPU=%d%n",
                statusCounts(cpu), statusCounts(gpuCold), violationCount(cpu), violationCount(gpuCold));
        List<String> mismatched = cpu.getPostContingencyResults().stream()
                .filter(p -> {
                    PostContingencyResult g = gpuCold.getPostContingencyResults().stream()
                            .filter(q -> q.getContingency().getId().equals(p.getContingency().getId()))
                            .findFirst().orElse(null);
                    return g == null || g.getStatus() != p.getStatus();
                })
                .map(p -> p.getContingency().getId())
                .toList();
        if (!mismatched.isEmpty()) {
            System.out.printf("  STATUS MISMATCHES (%d): %s%n", mismatched.size(),
                    mismatched.subList(0, Math.min(10, mismatched.size())));
        }
        diffViolations(cpu, gpuCold);
        Objects.requireNonNull(gpuWarm);
        assertEquals(statusCounts(cpu), statusCounts(gpuCold), "post-contingency status counts must match CPU");
    }

    /**
     * Margin-annotated per-contingency violation diff. With the device stopping on OLF's exact
     * criterion, states agree within the criterion's slack — so a violation present on one side
     * only is EXPECTED only when its value sits within epsilon of the limit (border flip). Any
     * one-sided violation with a LARGE relative margin is a genuine functional divergence.
     */
    private static void diffViolations(SecurityAnalysisResult cpu, SecurityAnalysisResult gpu) {
        Map<String, Map<String, com.powsybl.contingency.violations.LimitViolation>> cpuV = violationsByCtg(cpu);
        Map<String, Map<String, com.powsybl.contingency.violations.LimitViolation>> gpuV = violationsByCtg(gpu);
        int onlyCpu = 0;
        int onlyGpu = 0;
        int border = 0;                                      // relative margin < 1e-3: epsilon flip
        Map<String, Integer> ctgHisto = new TreeMap<>();
        List<String> worst = new java.util.ArrayList<>();
        for (String ctg : cpuV.keySet()) {
            Map<String, com.powsybl.contingency.violations.LimitViolation> c = cpuV.get(ctg);
            Map<String, com.powsybl.contingency.violations.LimitViolation> g = gpuV.getOrDefault(ctg, Map.of());
            for (Map.Entry<String, com.powsybl.contingency.violations.LimitViolation> e : c.entrySet()) {
                if (!g.containsKey(e.getKey())) {
                    onlyCpu++;
                    if (tally(e.getValue(), ctg, ctgHisto, worst, "CPU-only")) {
                        border++;
                    }
                }
            }
        }
        for (String ctg : gpuV.keySet()) {
            Map<String, com.powsybl.contingency.violations.LimitViolation> g = gpuV.get(ctg);
            Map<String, com.powsybl.contingency.violations.LimitViolation> c = cpuV.getOrDefault(ctg, Map.of());
            for (Map.Entry<String, com.powsybl.contingency.violations.LimitViolation> e : g.entrySet()) {
                if (!c.containsKey(e.getKey())) {
                    onlyGpu++;
                    if (tally(e.getValue(), ctg, ctgHisto, worst, "GPU-only")) {
                        border++;
                    }
                }
            }
        }
        worst.sort(java.util.Comparator.reverseOrder());
        System.out.printf("  VIOLATION DIFF: onlyCPU=%d onlyGPU=%d | border (rel margin < 1e-3) = %d of %d | contingencies affected=%d%n",
                onlyCpu, onlyGpu, border, onlyCpu + onlyGpu, ctgHisto.size());
        System.out.printf("  worst margins:%n");
        worst.stream().limit(10).forEach(w -> System.out.println("    " + w));
    }

    /** @return true when the violation is a border flip (relative margin under 1e-3). */
    private static boolean tally(com.powsybl.contingency.violations.LimitViolation v, String ctg,
                                 Map<String, Integer> ctgHisto, List<String> worst, String sideTag) {
        double margin = Math.abs(v.getValue() - v.getLimit() * v.getLimitReduction()) / Math.max(1e-9, Math.abs(v.getLimit()));
        ctgHisto.merge(ctg, 1, Integer::sum);
        worst.add(String.format("%.6f rel | %s | %s %s %s | value=%.4f limit=%.4f | ctg=%s",
                margin, sideTag, v.getSubjectId(), v.getLimitType(), v.getSide(), v.getValue(), v.getLimit(), ctg));
        return margin < 1e-3;
    }

    private static Map<String, Map<String, com.powsybl.contingency.violations.LimitViolation>> violationsByCtg(SecurityAnalysisResult r) {
        Map<String, Map<String, com.powsybl.contingency.violations.LimitViolation>> out = new TreeMap<>();
        for (PostContingencyResult p : r.getPostContingencyResults()) {
            Map<String, com.powsybl.contingency.violations.LimitViolation> m = new TreeMap<>();
            for (com.powsybl.contingency.violations.LimitViolation v : p.getLimitViolationsResult().getLimitViolations()) {
                m.put(v.getSubjectId() + "|" + v.getLimitType() + "|" + v.getSide(), v);
            }
            out.put(p.getContingency().getId(), m);
        }
        return out;
    }
}
