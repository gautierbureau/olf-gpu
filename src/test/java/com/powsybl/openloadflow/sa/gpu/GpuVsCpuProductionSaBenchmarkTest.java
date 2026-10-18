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
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.json.JsonLoadFlowParameters;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.CommonTestConfig;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuTestPaths;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchContextCache;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.sa.AbstractOpenSecurityAnalysisTest;
import com.powsybl.openloadflow.sa.OpenSecurityAnalysisParameters;
import com.powsybl.openloadflow.sa.OpenSecurityAnalysisProvider;
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

    GpuVsCpuProductionSaBenchmarkTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    @BeforeEach
    void checkGpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU native lib not available");
        assumeTrue(Files.exists(CASES.resolve(CASE)), CASE + " not found");
        assumeTrue(System.getenv("OLF_GPU_CONT_FILE") != null, "OLF_GPU_CONT_FILE not set");
        // Env-to-property plumbing (env is inherited by the surefire fork). Only override when the
        // env var IS set: a bare run must measure the PRODUCTION defaults (mode 3 UBATCH, memory
        // auto-sized chunk) — the old hardcoded 0/32 fallbacks made it read ~2x slower than what
        // any real provider user gets.
        if (System.getenv("OLF_GPU_MAXBATCH") != null) {
            GpuBatchContextCache.forceChunkSizeForTests(Integer.valueOf(System.getenv("OLF_GPU_MAXBATCH")));
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
    void threadCountAboveOneRunsSafelyOnGpuProvider() throws Exception {
        // The GPU provider IGNORES threadCount > 1 and runs single-partition, so that the base load
        // flow also runs on the device: OLF gives each partition its own base load flow, and the GPU
        // one is unsafe concurrently on a single device, which would leave a CPU-solved reference
        // filtering device-solved contingencies — measured to change the reported violations. So the
        // run must COMPLETE with the stock CPU run's statuses AND the same violation count, and it
        // must leave the caller's threadCount as it found it.
        Network network = Network.read(CASES.resolve(CASE));
        List<Contingency> contingencies = loadContingencies(network).subList(0, 20);
        Path paramsFile = Path.of("rte-loadflow-default-parameters.json");
        assumeTrue(Files.exists(paramsFile));
        SecurityAnalysisParameters sap = new SecurityAnalysisParameters();
        sap.setLoadFlowParameters(JsonLoadFlowParameters.read(paramsFile));
        sap.addExtension(OpenSecurityAnalysisParameters.class, new OpenSecurityAnalysisParameters().setThreadCount(2));
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, contingencies, Collections.emptyList(), sap);
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(new SparseMatrixFactory(),
                new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            SecurityAnalysisResult gpu = runSecurityAnalysis(network, contingencies, Collections.emptyList(), sap);
            assertEquals(statusCounts(cpu), statusCounts(gpu), "threadCount=2: statuses must match stock CPU");
            assertEquals(violationCount(cpu), violationCount(gpu),
                    "threadCount=2: the GPU provider must report the same violations as the stock CPU run");
            assertEquals(2, sap.getExtension(OpenSecurityAnalysisParameters.class).getThreadCount(),
                    "the caller's threadCount must be restored — they may reuse these parameters for a CPU run");
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(new SparseMatrixFactory(),
                    new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    @Test
    void productionSaGpuVsCpu() throws Exception {
        Network network = Network.read(CASES.resolve(CASE));
        List<Contingency> contingencies = loadContingencies(network);
        // OLF_GPU_SA_DEBUG_CTG=<id>: single-contingency drill-down — run ONLY that contingency on both
        // arms and print every violation at full precision (see the debug print below). Minutes, not a
        // full 500-contingency run, for one-problem-at-a-time debugging.
        String debugCtg = System.getenv("OLF_GPU_SA_DEBUG_CTG");
        if (debugCtg != null && !debugCtg.isBlank()) {
            contingencies = contingencies.stream().filter(c -> c.getId().equals(debugCtg)).toList();
            assumeTrue(!contingencies.isEmpty(), "contingency " + debugCtg + " not in the file");
        }
        // OLF_GPU_DBG_RL: TRACE the CPU arm's ReactiveLimitsOuterLoop so every PV<->PQ switch (bus id, q,
        // qLimit) is logged, to diff against the device batch's RLSWITCH trace and the RLMAP (vrow->bus).
        if (System.getenv("OLF_GPU_DBG_RL") != null) {
            ch.qos.logback.classic.Logger rlLog = (ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory
                    .getLogger("com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop");
            rlLog.setLevel(ch.qos.logback.classic.Level.TRACE);
            ch.qos.logback.classic.Logger root = (ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory
                    .getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME);
            if (!root.iteratorForAppenders().hasNext()) {       // no appender configured — attach a console one
                ch.qos.logback.classic.encoder.PatternLayoutEncoder enc = new ch.qos.logback.classic.encoder.PatternLayoutEncoder();
                enc.setContext(root.getLoggerContext());
                enc.setPattern("%msg%n");
                enc.start();
                ch.qos.logback.core.ConsoleAppender<ch.qos.logback.classic.spi.ILoggingEvent> ca = new ch.qos.logback.core.ConsoleAppender<>();
                ca.setContext(root.getLoggerContext());
                ca.setEncoder(enc);
                ca.start();
                rlLog.addAppender(ca);
            }
        }
        // THE RTE production parameters, loaded from the JSON checked in at the repo root
        // (rte-loadflow-default-parameters.json): DC init, reactive limits, twt split shunt
        // admittance, HVDC AC emulation, transformer voltage control AFTER_GENERATOR_VOLTAGE_CONTROL,
        // MAX_VOLTAGE_CHANGE state-vector scaling, maxNR/maxOuterLoop 30, realistic-voltage checks.
        Path paramsFile = Path.of("rte-loadflow-default-parameters.json");
        assumeTrue(Files.exists(paramsFile), "rte-loadflow-default-parameters.json not found in module dir");
        LoadFlowParameters lfp = JsonLoadFlowParameters.read(paramsFile);
        // OLF_GPU_SA_CONV_EPS: override newtonRaphsonConvEpsPerEq on BOTH arms (root-cause experiments).
        String convEpsEnv = System.getenv("OLF_GPU_SA_CONV_EPS");
        if (convEpsEnv != null) {
            com.powsybl.openloadflow.OpenLoadFlowParameters.get(lfp)
                    .setNewtonRaphsonConvEpsPerEq(Double.parseDouble(convEpsEnv));
        }
        // OLF_GPU_SA_SLACK_TOL / OLF_GPU_SA_RL_TOL: override the outer-loop tolerances on BOTH arms (pu →
        // MW/MVar, SB=100) with the inner eps left at the file value — isolates outer-loop trigger from
        // inner-Newton stop as the source of the batch's production-tolerance gap.
        String slackTolEnv = System.getenv("OLF_GPU_SA_SLACK_TOL");
        if (slackTolEnv != null) {
            lfp.setDistributedSlack(true);
            com.powsybl.openloadflow.OpenLoadFlowParameters.get(lfp)
                    .setSlackBusPMaxMismatch(Double.parseDouble(slackTolEnv) * 100.0);
        }
        String rlTolEnv = System.getenv("OLF_GPU_SA_RL_TOL");
        if (rlTolEnv != null) {
            com.powsybl.openloadflow.OpenLoadFlowParameters.get(lfp)
                    .setMaxReactivePowerMismatch(Double.parseDouble(rlTolEnv) * 100.0);
        }
        // OLF_GPU_SA_MAX_OUTER: cap the outer-loop iterations on BOTH arms (trajectory tracing — stop
        // after N outer passes and compare CPU vs GPU state to see where they first diverge). The GPU
        // batch driver caps RL_MAX_OUTER by the same maxOuterLoopIterations.
        String maxOuterEnv = System.getenv("OLF_GPU_SA_MAX_OUTER");
        if (maxOuterEnv != null) {
            com.powsybl.openloadflow.OpenLoadFlowParameters.get(lfp)
                    .setMaxOuterLoopIterations(Integer.parseInt(maxOuterEnv));
        }
        // OLF_GPU_SA_NO_RL / OLF_GPU_SA_NO_DS: disable an outer loop on BOTH arms (stage isolation —
        // inner-Newton-only vs +DS vs +DS+RL — to see which stage first introduces the CPU/GPU divergence).
        if ("1".equals(System.getenv("OLF_GPU_SA_NO_RL"))) {
            lfp.setUseReactiveLimits(false);
        }
        if ("1".equals(System.getenv("OLF_GPU_SA_NO_DS"))) {
            lfp.setDistributedSlack(false);
        }
        // OLF_GPU_SA_GPU_CONV_EPS: override the GPU arm's eps ONLY, leaving the CPU arm at its value.
        // This is the honest production framing: CPU runs the file's production criterion (1e-4) while
        // the GPU runs its fast operating point (1e-7, tighter → an at-least-as-converged answer). Same
        // run, same warmup, so the comparison has no cross-run machine-state drift. copy() is a full
        // JSON round-trip → an independent LoadFlowParameters preserving every file setting (30-iter
        // outer budget, MAX_VOLTAGE_CHANGE scaling, …), so only the eps differs between the arms.
        // GPU-arm-only outer-loop tolerance knobs (root-cause: inner-Newton stop vs outer-loop trigger).
        // OLF_GPU_SA_GPU_SLACK_TOL: distributed-slack tolerance in pu (→ slackBusPMaxMismatch MW, SB=100).
        // OLF_GPU_SA_GPU_RL_TOL: reactive-limits tolerance in pu (→ maxReactivePowerMismatch MVar, SB=100).
        LoadFlowParameters lfpGpu = lfp;
        String gpuConvEpsEnv = System.getenv("OLF_GPU_SA_GPU_CONV_EPS");
        String gpuSlackTolEnv = System.getenv("OLF_GPU_SA_GPU_SLACK_TOL");
        String gpuRlTolEnv = System.getenv("OLF_GPU_SA_GPU_RL_TOL");
        if (gpuConvEpsEnv != null || gpuSlackTolEnv != null || gpuRlTolEnv != null) {
            lfpGpu = lfp.copy();
            com.powsybl.openloadflow.OpenLoadFlowParameters gpuOlf =
                    com.powsybl.openloadflow.OpenLoadFlowParameters.get(lfpGpu);
            if (gpuConvEpsEnv != null) {
                gpuOlf.setNewtonRaphsonConvEpsPerEq(Double.parseDouble(gpuConvEpsEnv));
            }
            if (gpuSlackTolEnv != null) {
                lfpGpu.setDistributedSlack(true);
                gpuOlf.setSlackBusPMaxMismatch(Double.parseDouble(gpuSlackTolEnv) * 100.0);
            }
            if (gpuRlTolEnv != null) {
                gpuOlf.setMaxReactivePowerMismatch(Double.parseDouble(gpuRlTolEnv) * 100.0);
            }
        }

        // The FAIR production CPU arm is multithreaded — OLF SA parallelizes contingencies across
        // threads in production. The single-thread arm is an optional reference (OLF_GPU_SA_CPU_SINGLE=1).
        // Default 4 threads (this machine's realistic budget); override with OLF_GPU_SA_CPU_THREADS.
        int threads = Integer.parseInt(System.getenv().getOrDefault("OLF_GPU_SA_CPU_THREADS", "4"));
        SecurityAnalysisParameters sap = new SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        // Non-zero increased-violations thresholds (0.5 kV / 0.1 %): with powsybl's ZERO defaults, a
        // pre-existing violation is re-reported whenever the post value exceeds the run's own pre value
        // by a single ulp — solver-tolerance flicker (~1e-6 pu), not physics. Measured 2026-07-17: the
        // 6,137 one-sided violation diff at identical monitored states was exactly this artifact.
        sap.setIncreasedViolationsParameters(new SecurityAnalysisParameters.IncreasedViolationsParameters(
                0.5, 0.001, 0.5, 0.001, 0.001));
        sap.addExtension(OpenSecurityAnalysisParameters.class, new OpenSecurityAnalysisParameters().setThreadCount(threads));

        long cpu1Ms = -1;
        if ("1".equals(System.getenv("OLF_GPU_SA_CPU_SINGLE"))) {
            long c1 = System.nanoTime();
            runSecurityAnalysis(network, contingencies, lfp);
            cpu1Ms = (System.nanoTime() - c1) / 1_000_000;
        }
        long cpuT0 = System.nanoTime();
        // Diagnostic monitor on the dominant violation-diff bus (VL-3632): report its per-contingency
        // voltage on both sides so control-state divergences are visible directly, not only via violations.
        java.util.Set<String> monitoredVls = java.util.Set.of(
                System.getenv().getOrDefault("OLF_GPU_SA_MONITOR_VL", "VL-3632").split(","));
        List<com.powsybl.security.monitor.StateMonitor> monitors = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(), monitoredVls, java.util.Set.of()));
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, contingencies, monitors, sap);
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
            sapGpu.setLoadFlowParameters(lfpGpu);
            sapGpu.setIncreasedViolationsParameters(sap.getIncreasedViolationsParameters());
            long g0 = System.nanoTime();
            gpuCold = runSecurityAnalysis(network, contingencies, monitors, sapGpu);
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
        System.out.printf("  conv eps: CPU=%.1e  GPU=%.1e%s%n",
                com.powsybl.openloadflow.OpenLoadFlowParameters.get(lfp).getNewtonRaphsonConvEpsPerEq(),
                com.powsybl.openloadflow.OpenLoadFlowParameters.get(lfpGpu).getNewtonRaphsonConvEpsPerEq(),
                lfpGpu == lfp ? " (shared)" : " (per-arm split: CPU production criterion, GPU fast point)");
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
        printMonitoredBus(cpu, "CPU");
        printMonitoredBus(gpuCold, "GPU");
        printMonitorDiff(cpu, gpuCold);
        if (debugCtg != null && !debugCtg.isBlank()) {
            dumpAllViolations(cpu, "CPU", debugCtg);
            dumpAllViolations(gpuCold, "GPU", debugCtg);
            System.out.printf("  DEBUG pre-ctg violation counts: CPU=%d GPU=%d%n",
                    cpu.getPreContingencyResult().getLimitViolationsResult().getLimitViolations().size(),
                    gpuCold.getPreContingencyResult().getLimitViolationsResult().getLimitViolations().size());
            cpu.getPreContingencyResult().getLimitViolationsResult().getLimitViolations().stream().limit(5)
                    .forEach(v -> System.out.printf("  DEBUG CPU-pre | %s %s | value=%.9f%n", v.getSubjectId(), v.getLimitType(), v.getValue()));
            gpuCold.getPreContingencyResult().getLimitViolationsResult().getLimitViolations().stream().limit(5)
                    .forEach(v -> System.out.printf("  DEBUG GPU-pre | %s %s | value=%.9f%n", v.getSubjectId(), v.getLimitType(), v.getValue()));
        }
        Objects.requireNonNull(gpuWarm);
        assertEquals(statusCounts(cpu), statusCounts(gpuCold), "post-contingency status counts must match CPU");
    }

    /** Full-precision violation dump for the single-contingency debug mode. */
    private static void dumpAllViolations(SecurityAnalysisResult r, String tag, String ctgId) {
        r.getPostContingencyResults().stream()
                .filter(p -> p.getContingency().getId().equals(ctgId))
                .flatMap(p -> p.getLimitViolationsResult().getLimitViolations().stream())
                .forEach(v -> System.out.printf("  DEBUG %s | %s %s %s | value=%.9f limit=%.9f%n",
                        tag, v.getSubjectId(), v.getLimitType(), v.getSide(), v.getValue(), v.getLimit()));
    }

    /** High-precision per-stage CPU-vs-GPU monitored-bus state diff (trajectory tracing). */
    private static void printMonitorDiff(SecurityAnalysisResult cpu, SecurityAnalysisResult gpu) {
        Map<String, Double> g = new java.util.HashMap<>();
        gpu.getPostContingencyResults().forEach(p -> p.getNetworkResult().getBusResults()
                .forEach(b -> g.put(p.getContingency().getId() + "|" + b.getBusId(), b.getV())));
        double max = 0;
        String worst = null;
        for (PostContingencyResult p : cpu.getPostContingencyResults()) {
            for (com.powsybl.security.results.BusResult b : p.getNetworkResult().getBusResults()) {
                Double gv = g.get(p.getContingency().getId() + "|" + b.getBusId());
                if (gv != null) {
                    double d = Math.abs(b.getV() - gv);
                    System.out.printf("  MONDIFF %s %s |dV|=%.3e cpu=%.9f gpu=%.9f%n",
                            p.getContingency().getId(), b.getBusId(), d, b.getV(), gv);
                    if (d > max) {
                        max = d;
                        worst = b.getBusId();
                    }
                }
            }
        }
        System.out.printf("  MONITOR-DIFF max|dV|=%.3e at %s%n", max, worst);
    }

    private static void printMonitoredBus(SecurityAnalysisResult r, String tag) {
        r.getPreContingencyResult().getNetworkResult().getBusResults()
                .forEach(b -> System.out.printf("  MONITOR %s pre-ctg %s v=%.5f%n", tag, b.getBusId(), b.getV()));
        r.getPostContingencyResults().stream().limit(3)
                .forEach(p -> p.getNetworkResult().getBusResults()
                        .forEach(b -> System.out.printf("  MONITOR %s post %s %s v=%.5f%n", tag, p.getContingency().getId(), b.getBusId(), b.getV())));
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
        worst.stream().limit(25).forEach(w -> System.out.println("    " + w));
        ctgHisto.forEach((c, k) -> System.out.printf("  DIFF-CTG %s: %d%n", c, k));
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
