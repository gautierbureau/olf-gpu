/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.commons.report.ReportNode;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcLoadFlowContext;
import com.powsybl.openloadflow.ac.AcLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcloadFlowEngine;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcStructures;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuTestPaths;
import com.powsybl.openloadflow.ac.solver.AcSolverUtil;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphsonFactory;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.TargetVector;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.impl.LfNetworkLoaderImpl;
import com.powsybl.openloadflow.network.util.PreviousValueVoltageInitializer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * BARE fixed-iteration batched N-1 throughput on PEGASE-9241 — the standard fixed-iteration
 * batched-solver benchmark methodology, so the numbers are directly comparable to other GPU N-1 tools.
 *
 * <p>A bare fixed-iteration batched solver runs a <em>fixed</em> number of Newton-Raphson iterations
 * per chunk (typically 4, sometimes 10) over the <em>non-islanding</em>
 * contingencies only — exactly the scope our {@link GpuBatchContingencyValidator}
 * already enforces (a contingency whose branch islands buses from the main component is
 * rejected and routed to the CPU path; a connectivity check marks
 * the same outages disconnected and skips them). Such a solve does <em>no</em>
 * convergence retirement: every live scenario runs all K steps in lockstep.
 *
 * <p>This test reproduces that: it qualifies the non-islanding subset and runs a sweep of
 * fixed iteration counts with {@code tol == 0} — which disables convergence-based retirement,
 * so the run is a clean measure of K lockstep Newton iterations over the contingency set. For
 * each K it reports wall time, throughput, and the post-K accuracy (how many scenarios reached
 * a 1e-6 mismatch), which should climb with K. Unlike {@link GpuBatchedN1BenchmarkTest} there
 * is no CPU comparison — this is a pure GPU-throughput measurement, not a speedup.
 *
 * <p>Machine-local: skipped unless the GPU lib is available AND the cases directory exists.
 * Select the native solve mode at runtime with {@code OLF_GPU_BATCHMODE=1} (uniform batched
 * cuDSS) or {@code 0} (per-scenario fallback, default);
 * both are compiled in (cuDSS >= 0.8 required).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuBatchedN1FixedIterTest {

    private static final Path CASES = GpuTestPaths.caseDir();
    private static final String CASE = "case9241pegase.xiidm.gz";
    // Typical fixed-iteration counts: 4 (default), 10. With the AC hot-start (converge the base case first,
    // start every contingency from it — as OLF CPU SA does) we converge to
    // 1e-12 at K=4. The sweep still climbs K to show margin.
    // The K-sweep. Default is just K=4 — the regression harness for the fixed-iteration throughput path
    // (run after every batched-path change). Override with OLF_GPU_FIXED_ITERS="4,6,10" to see the margin.
    private static final int[] FIXED_ITERS = parseFixedIters(System.getenv("OLF_GPU_FIXED_ITERS"), new int[] {4});
    private static final double ACCURACY_TOL = 1e-6;            // the reference mismatch tolerance
    // OLF_GPU_TOL > 0 switches every solve in this bench to the PRODUCTION path: per-iteration
    // convergence check + retirement (host sync + full-mismatch D->H copy each iteration) instead
    // of device fixed-iteration mode. Use with a generous OLF_GPU_FIXED_ITERS (e.g. 15) so the
    // iteration cap never bites; K then only bounds maxIterations. Default 0 = bench as before.
    private static final double TOL = Double.parseDouble(System.getenv().getOrDefault("OLF_GPU_TOL", "0"));

    private static int[] parseFixedIters(String env, int[] fallback) {
        if (env == null || env.isBlank()) {
            return fallback;
        }
        String[] parts = env.split(",");
        int[] ks = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            ks[i] = Integer.parseInt(parts[i].trim());
        }
        return ks;
    }

    private static final int MAX_CONTINGENCIES = 1000;

    static {
        GpuTestPaths.init();
    }

    @BeforeEach
    void check() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU native lib not available");
        assumeTrue(Files.isDirectory(CASES), "benchmark cases not found at " + CASES);
        assumeTrue(Files.exists(CASES.resolve(CASE)), CASE + " not found");
        // Env-to-property plumbing (env is inherited by the surefire fork, unlike -D). Only override
        // when the env var IS set: a bare run measures the PRODUCTION defaults (mode 3 UBATCH,
        // memory auto-sized chunk). Caveat kept from the old hardcoded cap: selecting mode 1
        // (OLF_GPU_BATCHMODE=1) without an explicit OLF_GPU_MAXBATCH cap can over-commit device
        // memory at PEGASE-9241 on a small card — mode 1's memory model is not auto-size-aware.
        if (System.getProperty(GpuBatchContextCache.MAX_BATCH_PROPERTY) == null
                && System.getenv("OLF_GPU_MAXBATCH") != null) {
            System.setProperty(GpuBatchContextCache.MAX_BATCH_PROPERTY, System.getenv("OLF_GPU_MAXBATCH"));
        }
        if (System.getProperty(GpuBatchContextCache.BATCH_MODE_PROPERTY) == null
                && System.getenv("OLF_GPU_BATCHMODE") != null) {
            System.setProperty(GpuBatchContextCache.BATCH_MODE_PROPERTY, System.getenv("OLF_GPU_BATCHMODE"));
        }
    }

    private record Problem(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                           TargetVector<AcVariableType, AcEquationType> target, AcLoadFlowContext context) {
    }

    /**
     * Load an LfNetwork + AC equation system with the GPU-scope parameters (the extractor's scope).
     * ONE {@link AcLoadFlowContext} provides the equation system for BOTH the base load flow and the
     * batched contingency sweep — exactly the production SA shape ({@code GpuBatchedAcSecurityAnalysis}
     * batches on the engine's own context), so with the base on the GPU the batch reuses the base's
     * extraction ({@code GpuAcDataCache}) AND its device structure ({@code GpuAcStructures}). The caller
     * closes the context.
     */
    private static Problem loadGpuScope(Network iidm) {
        // DC_VALUES init for the network state vector; the caller then converges the base case to
        // AC before the contingency sweep (the hot start). Same scope as the GPU extractor.
        LoadFlowParameters lfp = new LoadFlowParameters()
                .setVoltageInitMode(LoadFlowParameters.VoltageInitMode.DC_VALUES);
        OpenLoadFlowParameters olfp = OpenLoadFlowParameters.create(lfp)
                .setVoltageRemoteControl(false)
                .setLowImpedanceBranchMode(OpenLoadFlowParameters.LowImpedanceBranchMode.REPLACE_BY_MIN_IMPEDANCE_LINE);
        // OLF_GPU_BASE_ON_GPU: run the base/warm-up load flow on the GPU (the full GpuNewtonRaphson) instead
        // of the CPU. This is the base-case warm-up strategy — it both runs faster (~0.4 vs ~1.5 s) and
        // ABSORBS the one-time CUDA init into the base phase (same .so), so the cold contingency pass is then
        // warm. Moves CUDA init out of the SA phase, closing the cold-E2E gap.
        if (System.getenv("OLF_GPU_BASE_ON_GPU") != null) {
            olfp.setAcSolverType(GpuNewtonRaphsonFactory.NAME);
        }
        AcLoadFlowParameters acp = OpenLoadFlowParameters.createAcParameters(iidm, lfp, olfp,
                new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>(), false, false);
        acp.getNetworkParameters().setMaxSlackBusCount(1);
        LfNetwork network = LfNetwork.load(iidm, new LfNetworkLoaderImpl(), acp.getNetworkParameters(), ReportNode.NO_OP).get(0);
        AcLoadFlowContext context = new AcLoadFlowContext(network, acp);
        EquationSystem<AcVariableType, AcEquationType> es = context.getEquationSystem();
        AcSolverUtil.initStateVector(network, es, acp.getVoltageInitializer());
        return new Problem(network, es, context.getTargetVector(), context);
    }

    @Test
    void pegase9241FixedIterations() throws Exception {
        Network iidm = Network.read(CASES.resolve(CASE));
        Problem p = loadGpuScope(iidm);
        try (AcLoadFlowContext context = p.context()) {
            runPegase9241FixedIterations(p, context);
        }
        // Warm-JIT cold-GPU end-to-end: a SECOND full pass in the same JVM on a freshly built
        // problem (new LfNetwork/context/es; the Java-side GPU caches are cleared, so extraction,
        // shared structure, LU probe, cuDSS analyses and solver contexts all rebuild — without the
        // clear, content matching would alias the first pass's device structure and un-cold the
        // measurement). JIT and the CUDA driver are warm; everything GPU-side is cold. This is the
        // long-running-service scenario: a new network snapshot arriving at a warm process,
        // repeated in-process runs. The [E2E] line from the first pass stays the single-shot cold number.
        if (System.getenv("OLF_GPU_E2E") != null && !"0".equals(System.getenv("OLF_GPU_E2E_WARMJIT"))) {
            Problem p2 = loadGpuScope(iidm);
            try (AcLoadFlowContext context = p2.context()) {
                e2ePass(p2, context, "E2E-warmJIT");
            }
        }
    }

    /**
     * One end-to-end pass on a fresh problem: the Java-side GPU caches are cleared FIRST (so with
     * the base on the GPU, the base pays the cold extraction + structure build like a real new
     * snapshot and the SA then reuses them through the production hand-off), then the timed base
     * load flow through {@code context}, then ONE timed contingency sweep at the E2E K.
     */
    private static void e2ePass(Problem p, AcLoadFlowContext context, String label) {
        LfNetwork network = p.network();
        EquationSystem<AcVariableType, AcEquationType> es = p.es();
        TargetVector<AcVariableType, AcEquationType> target = p.target();
        GpuBatchContextCache.clearContextCache();
        GpuAcStructures.clear();
        long baseLfT0 = System.nanoTime();
        new AcloadFlowEngine(context).run();
        long baseLfMs = (System.nanoTime() - baseLfT0) / 1_000_000;
        AcSolverUtil.initStateVector(network, es, new PreviousValueVoltageInitializer());
        double[] baseState = es.getStateVector().get().clone();
        List<BatchContingency> contingencies = buildContingencies(network);
        int kE2E = Integer.parseInt(System.getenv().getOrDefault("OLF_GPU_E2E_K", "4"));
        NewtonRaphsonParameters params = new NewtonRaphsonParameters();
        params.setMaxIterations(kE2E);
        long g0 = System.nanoTime();
        List<BatchScenarioResult> cold = GpuBatchedSecurityAnalysisSolver.solveBatchN1(
                network, baseState, contingencies, es, target, params, TOL, ReportNode.NO_OP);
        long gpuMs = (System.nanoTime() - g0) / 1_000_000;
        long cb = cold.stream().filter(b -> !b.needsCpuFallback()).count();
        String baseWhere = System.getenv("OLF_GPU_BASE_ON_GPU") != null ? "GPU" : "CPU";
        System.out.printf("  [%s K=%d] base %s loadflow %d ms + cold GPU SA %d ms = %d ms total | %d ctg | %.2f ms/ctg%n",
                label, kE2E, baseWhere, baseLfMs, gpuMs,
                baseLfMs + gpuMs, cb, (baseLfMs + gpuMs) / (double) Math.max(1, cb));
    }

    /**
     * The bench contingency set: {@code OLF_GPU_CONT_FILE} (one IIDM branch id per line, file
     * order — a fixed, reproducible physical set) or the first {@link #MAX_CONTINGENCIES}
     * connected branches.
     */
    private static List<BatchContingency> buildContingencies(LfNetwork network) {
        List<BatchContingency> contingencies = new ArrayList<>();
        String contFile = System.getenv("OLF_GPU_CONT_FILE");
        if (contFile != null) {
            java.util.Map<String, LfBranch> byId = new java.util.HashMap<>();
            for (LfBranch br : network.getBranches()) {
                byId.put(br.getId(), br);
            }
            try {
                for (String id : Files.readAllLines(Path.of(contFile))) {
                    String key = id.strip();
                    LfBranch br = key.isEmpty() ? null : byId.get(key);
                    if (br != null && !br.isDisabled() && br.getBus1() != null && br.getBus2() != null) {
                        contingencies.add(new BatchContingency(key, br));
                    }
                }
            } catch (java.io.IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
            System.out.printf("  (explicit set from %s: %d branches resolved)%n", contFile, contingencies.size());
        } else {
            for (LfBranch br : network.getBranches()) {
                if (!br.isDisabled() && br.getBus1() != null && br.getBus2() != null) {
                    contingencies.add(new BatchContingency(br.getId(), br));
                    if (contingencies.size() >= MAX_CONTINGENCIES) {
                        break;
                    }
                }
            }
        }
        return contingencies;
    }

    private void runPegase9241FixedIterations(Problem p, AcLoadFlowContext context) {
        LfNetwork network = p.network();
        EquationSystem<AcVariableType, AcEquationType> es = p.es();
        TargetVector<AcVariableType, AcEquationType> target = p.target();

        // Every contingency hot-starts from the CONVERGED base-case AC state — exactly what OLF's
        // CPU security analysis does (base load flow, then PreviousValue init per contingency) and
        // what the production GPU path does (GpuBatchedAcSecurityAnalysis precomputes from the
        // converged base in afterPreContingencySimulation). A single-branch N-1 is a small
        // perturbation of the base case, so from this start it converges to 1e-12 in K=4. (DC-init
        // would need K=6+ — that gap is the START POINT, not the formulation.) Run a real AC load
        // flow on the base case THROUGH THE SAME CONTEXT the batch then uses (the production SA
        // shape) — with OLF_GPU_BASE_ON_GPU the batch reuses the base's extraction and device
        // structure instead of rebuilding them.
        long baseLfT0 = System.nanoTime();
        new AcloadFlowEngine(context).run();
        long baseLfMs = (System.nanoTime() - baseLfT0) / 1_000_000;
        AcSolverUtil.initStateVector(network, es, new PreviousValueVoltageInitializer());
        double[] baseState = es.getStateVector().get().clone();
        int n = baseState.length;

        // OLF_GPU_CONT_FILE (one IIDM branch id per line) selects an EXPLICIT contingency set in
        // file order — used to run a fixed, reproducible physical set (e.g. the
        // list from codegen/bench/gen_contingencies.py). Unset → the first MAX_CONTINGENCIES branches.
        List<BatchContingency> contingencies = buildContingencies(network);

        System.out.printf("BARE-FIXED-ITER %s (n=%d): %d single-branch contingencies (non-islanding subset "
                + "qualified on-GPU; islanding outages routed to CPU)%n",
                CASE.replace(".xiidm.gz", ""), n, contingencies.size());

        // End-to-end (OLF_GPU_E2E): CPU base load flow + ONE COLD GPU SA pass at K (CUDA init +
        // extract + analysis + solve fall into this first GPU call — we have no GPU base-case phase
        // to absorb CUDA init, unlike the base-case warm-up path). The honest single-pass wall time.
        if (System.getenv("OLF_GPU_E2E") != null) {
            int kE2E = Integer.parseInt(System.getenv().getOrDefault("OLF_GPU_E2E_K", "4"));
            NewtonRaphsonParameters params = new NewtonRaphsonParameters();
            params.setMaxIterations(kE2E);
            GpuBatchContextCache.clearContextCache();
            long g0 = System.nanoTime();
            List<BatchScenarioResult> cold = GpuBatchedSecurityAnalysisSolver.solveBatchN1(
                    network, baseState, contingencies, es, target, params, TOL, ReportNode.NO_OP);
            long gpuMs = (System.nanoTime() - g0) / 1_000_000;
            long cb = cold.stream().filter(b -> !b.needsCpuFallback()).count();
            String baseWhere = System.getenv("OLF_GPU_BASE_ON_GPU") != null ? "GPU" : "CPU";
            System.out.printf("  [E2E K=%d] base %s loadflow %d ms + %s GPU SA %d ms = %d ms total | %d ctg | %.2f ms/ctg%n",
                    kE2E, baseWhere, baseLfMs, "GPU".equals(baseWhere) ? "warm" : "cold", gpuMs,
                    baseLfMs + gpuMs, cb, (baseLfMs + gpuMs) / (double) Math.max(1, cb));
        }

        long totalBatched = 0;
        for (int k : FIXED_ITERS) {
            NewtonRaphsonParameters params = new NewtonRaphsonParameters();
            params.setMaxIterations(k);

            // Warm-up (CUDA context + cuDSS analysis) so the timed run measures steady-state throughput.
            GpuBatchedSecurityAnalysisSolver.solveBatchN1(network, baseState, contingencies, es, target, params, TOL, ReportNode.NO_OP);

            long t0 = System.nanoTime();
            List<BatchScenarioResult> batch = GpuBatchedSecurityAnalysisSolver.solveBatchN1(
                    network, baseState, contingencies, es, target, params, TOL, ReportNode.NO_OP);
            long ms = (System.nanoTime() - t0) / 1_000_000;

            long batched = batch.stream().filter(b -> !b.needsCpuFallback()).count();
            double[] mismatches = batch.stream()
                    .filter(b -> !b.needsCpuFallback())
                    .mapToDouble(BatchScenarioResult::finalMismatch)
                    .sorted()
                    .toArray();
            long accurate = java.util.Arrays.stream(mismatches).filter(m -> m < ACCURACY_TOL).count();
            double minMismatch = mismatches.length > 0 ? mismatches[0] : Double.NaN;
            double medMismatch = mismatches.length > 0 ? mismatches[mismatches.length / 2] : Double.NaN;
            totalBatched = batched;
            System.out.printf("  K=%-2d : %5d ms | %d batched | %d/%d reached %.0e | min/med ||F||inf %.1e/%.1e "
                            + "| %.1f scenarios/s | %.2f ms/scenario%n",
                    k, ms, batched, accurate, batched, ACCURACY_TOL, minMismatch, medMismatch,
                    batched * 1000.0 / Math.max(1, ms), ms / (double) Math.max(1, batched));
        }
        assertTrue(totalBatched > 0, "at least one non-islanding contingency must be batched on the GPU");
    }
}
