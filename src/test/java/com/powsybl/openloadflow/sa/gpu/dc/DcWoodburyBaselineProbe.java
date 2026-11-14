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
import com.powsybl.security.results.OperatorStrategyResult;
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

    /** The non-fast DC security analysis arm (a full DC load flow per state): the ground truth of a disagreement. */
    private static final String REFERENCE = "DC-REF";

    /** The operator strategies of the run ({@code OLF_DC_STRATEGIES} per contingency), or null: N-1 only. */
    private static TopologyStudy.Study study;

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
        // OLF_DC_STRATEGIES=n: n operator strategies per contingency (TopologyStudy: coupler/breaker switching near it,
        // a share OLF_DC_REDISPATCH_SHARE (default 0.3) of OLF_DC_REDISPATCH_MW (default 50) redispatch pairs);
        // OLF_DC_RETAIN_BREAKERS=1 declares an action on every breaker, so OLF keeps them all
        int strategiesPerContingency = Integer.parseInt(System.getenv().getOrDefault("OLF_DC_STRATEGIES", "0"));
        study = strategiesPerContingency > 0
                ? TopologyStudy.generate(network, contingencies, strategiesPerContingency,
                        Double.parseDouble(System.getenv().getOrDefault("OLF_DC_REDISPATCH_SHARE", "0.3")),
                        Double.parseDouble(System.getenv().getOrDefault("OLF_DC_REDISPATCH_MW", "50")),
                        "1".equals(System.getenv("OLF_DC_RETAIN_BREAKERS")),
                        Long.parseLong(System.getenv().getOrDefault("OLF_DC_SEED", "42")),
                        // OLF_DC_KINDS: the topological strategy kinds cycled through (coupler, move, breaker)
                        System.getenv("OLF_DC_KINDS") != null ? List.of(System.getenv("OLF_DC_KINDS").split(",")) : TopologyStudy.DEFAULT_KINDS)
                : null;
        // OLF_DC_ONLY_STRATEGIES=id,id: replay these strategies of the study alone, with their contingencies (diagnosis)
        String onlyStrategies = System.getenv("OLF_DC_ONLY_STRATEGIES");
        if (study != null && onlyStrategies != null && !onlyStrategies.isEmpty()) {
            Set<String> ids = Set.of(onlyStrategies.split(","));
            List<com.powsybl.contingency.strategy.OperatorStrategy> kept = study.strategies().stream().filter(s -> ids.contains(s.getId())).toList();
            Set<String> ctgIds = kept.stream().map(s -> s.getContingencyContext().getContingencyId()).collect(Collectors.toSet());
            contingencies = contingencies.stream().filter(c -> ctgIds.contains(c.getId())).toList();
            study = new TopologyStudy.Study(study.actions(), kept, study.description() + ", replaying " + kept.size());
            kept.forEach(s -> System.out.printf("DC-BASELINE replay %s: %s%n", s.getId(), s.getConditionalActions().get(0).getActionIds()));
        }
        if (study != null) {
            System.out.printf("DC-BASELINE study: %s%n", study.description());
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
            if ("1".equals(System.getenv("OLF_DC_REFERENCE"))) {
                // OLF_DC_REFERENCE=1 with OLF_DC_SKIP_KLU=1: the GPU against the (non-fast) DC security analysis only
                Arm reference = run(REFERENCE, new SparseMatrixFactory(), network, contingencies, monitoredBranches, 1);
                printBranches(List.of(reference, gpu));
            }
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
            com.powsybl.openloadflow.sa.gpu.GpuDcBaseAccuracy.Result baseAccuracy = com.powsybl.openloadflow.sa.gpu.GpuWoodburyDcSecurityAnalysis.LAST_BASE_ACCURACY.get();
            // OLF_DC_REFERENCE=1: also the non-fast DC security analysis, and OLF_DC_PRINT_BRANCHES=id,id printed in each arm
            if ("1".equals(System.getenv("OLF_DC_REFERENCE"))) {
                Arm reference = run(REFERENCE, new SparseMatrixFactory(), network, contingencies, monitoredBranches, threads);
                printBranches(List.of(reference, klu, gpu));
            }
            compare(klu, gpu, contingencies.size(),
                    new FlowTolerance(baseAccuracy, network,
                            JsonLoadFlowParameters.read(PARAMS_FILE).getDcPowerFactor()));
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

        compare(klu, cudss, contingencies.size(), new FlowTolerance(null, network, 1));
    }

    /**
     * The flow tolerance of a comparison, per branch. Two LU codes on a well-conditioned DC matrix agree to cond * eps:
     * 1e-10 pu (1e-8 MW) everywhere, the bound this probe always held (rte6515: 2.5e-10 MW measured). A network whose
     * DC matrix has stiff branches - OLF models a switch the contingency list retains as a line with x = 1e-8 pu -
     * determines its angles only to the uncertainty the base diagnostic measures (each arm's distance to a state refined
     * with exact residuals, summed): two correct solves can differ on branch b by up to 2 * power(b) * that, on each
     * solve a post-contingency state composes (base, element state, target column), hence SOLVES. So there the
     * tolerance is that bound, and never below 1e-8 MW. A violation's value is in its limit's unit: MW for active power,
     * MW / power factor for apparent power, A for current (P * 1000 / (sqrt(3) * Vnom * power factor), as the DC result).
     */
    private record FlowTolerance(com.powsybl.openloadflow.sa.gpu.GpuDcBaseAccuracy.Result accuracy, Network network, double dcPowerFactor) {

        static final double FLOOR_MW = 1e-8;
        static final double SOLVES = 3;

        boolean derived() {
            return accuracy != null && accuracy.stiffBranches() > 0;
        }

        double megawatts(String branchId) {
            if (!derived()) {
                return FLOOR_MW;
            }
            Double power = accuracy.branchPower().get(branchId);
            return power == null ? FLOOR_MW
                    : Math.max(FLOOR_MW, SOLVES * 2 * power * accuracy.stateUncertainty() * com.powsybl.openloadflow.util.PerUnit.SB);
        }

        double inUnit(LimitViolation v) {
            double mw = megawatts(v.getSubjectId());
            return switch (v.getLimitType()) {
                case ACTIVE_POWER -> mw;
                case APPARENT_POWER -> mw / dcPowerFactor;
                case CURRENT -> {
                    Branch<?> branch = network.getBranch(v.getSubjectId());
                    double nominalV = branch == null ? Double.NaN
                            : branch.getTerminal(v.getSide().toTwoSides()).getVoltageLevel().getNominalV();
                    yield Double.isNaN(nominalV) ? mw : mw * 1000 / (Math.sqrt(3) * nominalV * dcPowerFactor);
                }
                default -> mw;
            };
        }

        String describe() {
            return derived()
                    ? String.format("derived: %d stiff branches, state uncertainty %.3e rad (device %.3e + LU %.3e), %.0f x 2 x power x that",
                            accuracy.stiffBranches(), accuracy.stateUncertainty(), accuracy.deviceError(), accuracy.luError(), SOLVES)
                    : String.format("%.0e MW (no stiff branch%s)", FLOOR_MW, accuracy == null ? ", no base measurement" : "");
        }
    }

    private static void compare(Arm klu, Arm cudss, int contingencyCount, FlowTolerance tolerance) {
        // --- parity: the two arms run the SAME Java, only the LU differs
        assertEquals(statusCounts(klu.result()), statusCounts(cudss.result()), "post-contingency statuses differ");
        assertEquals(klu.result().getPreContingencyResult().getStatus(), cudss.result().getPreContingencyResult().getStatus());

        Map<String, Map<String, LimitViolation>> vKlu = violationsByCtg(klu.result());
        Map<String, Map<String, LimitViolation>> vCudss = violationsByCtg(cudss.result());
        System.out.printf("DC-BASELINE tolerance: %s%n", tolerance.describe());
        int differingLines = 0;
        double maxValueDiff = 0;
        double worstViolationRatio = 0;
        String worstViolation = "";
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
                    double diff = Math.abs(va.getValue() - vb.getValue());
                    maxValueDiff = Math.max(maxValueDiff, diff);
                    double ratio = diff / tolerance.inUnit(va);
                    if (ratio > worstViolationRatio) {
                        worstViolationRatio = ratio;
                        worstViolation = ctg + " " + key;
                    }
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
        double worstP1Ratio = 0;
        String worstP1 = "";
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
                    double ratio = d / tolerance.megawatts(b.getKey());
                    if (ratio > worstP1Ratio) {
                        worstP1Ratio = ratio;
                        worstP1 = String.format("ctg=%s branch=%s diff %.3e MW, tolerance %.3e MW", e.getKey(), b.getKey(), d,
                                tolerance.megawatts(b.getKey()));
                    }
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

        System.out.printf("DC-BASELINE tolerance used: violations %.3f (%s), monitored P1 %.3f (%s)%n",
                worstViolationRatio, worstViolation, worstP1Ratio, worstP1);
        // The tolerance is FlowTolerance's: 1e-10 pu (1e-8 MW) on a well-conditioned matrix (rte6515: cond ~4e4,
        // measured 1.2e-11 pu), the measured conditioning bound on one with stiff branches. Never a convergence
        // tolerance: there is no iteration here to loosen.
        assertEquals(0, differingLines, "violation sets differ between the arms");
        assertTrue(worstViolationRatio <= 1, "common-violation values beyond the tolerance: " + worstViolation);
        assertTrue(worstP1Ratio <= 1, "monitored P1 beyond the tolerance: " + worstP1);
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
                new OpenSecurityAnalysisParameters().setDcFastMode(!name.equals(REFERENCE)).setThreadCount(threads));
        List<StateMonitor> monitors = List.of(new StateMonitor(com.powsybl.contingency.ContingencyContext.all(),
                new LinkedHashSet<>(monitoredBranches), Set.of(), Set.of()));
        OpenSecurityAnalysisProvider provider = gpuProvider
                ? new com.powsybl.openloadflow.sa.gpu.GpuSecurityAnalysisProvider(matrixFactory, new EvenShiloachGraphDecrementalConnectivityFactory<>())
                : new OpenSecurityAnalysisProvider(matrixFactory, new EvenShiloachGraphDecrementalConnectivityFactory<>());
        SecurityAnalysisRunParameters runParameters = new SecurityAnalysisRunParameters()
                .setSecurityAnalysisParameters(sap)
                .setMonitors(monitors);
        if (study != null) {
            runParameters.setOperatorStrategies(study.strategies()).setActions(study.actions());
        }
        // OLF_DC_CHECK_TARGETS=1: the device-built modified targets checked bit for bit against the host's
        boolean checkTargets = gpuProvider && "1".equals(System.getenv("OLF_DC_CHECK_TARGETS"));
        com.powsybl.openloadflow.sa.gpu.GpuWoodburyDcSecurityAnalysis.CHECK_DEVICE_TARGETS.set(checkTargets);
        // OLF_DC_BASE_DIAG=1: the device base state's accuracy against OLF's own system (stderr DC_BASE_ACCURACY)
        // the device base state's accuracy against OLF's own system (stderr DC_BASE_ACCURACY): sizes the flow tolerance
        com.powsybl.openloadflow.sa.gpu.GpuWoodburyDcSecurityAnalysis.BASE_ACCURACY_DIAGNOSTIC.set(gpuProvider);
        com.powsybl.openloadflow.sa.gpu.GpuWoodburyDcSecurityAnalysis.LAST_BASE_ACCURACY.set(null);
        com.powsybl.openloadflow.sa.gpu.GpuWoodburyDcSecurityAnalysis.DEVICE_TARGET_COUNT.set(0);
        com.powsybl.openloadflow.sa.gpu.GpuWoodburyDcSecurityAnalysis.TARGET_MISMATCH_COUNT.set(0);
        // OLF_DC_CHECK_RESULTS=1: the monitored branch results of the fast path checked bit for bit against OLF's update
        boolean checkResults = gpuProvider && "1".equals(System.getenv("OLF_DC_CHECK_RESULTS"));
        com.powsybl.openloadflow.sa.gpu.GpuWoodburyDcSecurityAnalysis.CHECK_NETWORK_RESULTS.set(checkResults);
        com.powsybl.openloadflow.sa.gpu.GpuWoodburyDcSecurityAnalysis.NETWORK_RESULT_MISMATCH_COUNT.set(0);
        com.powsybl.openloadflow.sa.gpu.GpuWoodburyDcSecurityAnalysis.FAST_NETWORK_RESULT_COUNT.set(0);
        // OLF_DC_CHECK_STATE=1: every branch value read from a state checked bit for bit against OLF's model
        boolean checkState = gpuProvider && "1".equals(System.getenv("OLF_DC_CHECK_STATE"));
        com.powsybl.openloadflow.sa.gpu.GpuWoodburyDcSecurityAnalysis.CHECK_STATE_READER.set(checkState);
        com.powsybl.openloadflow.sa.gpu.GpuWoodburyDcSecurityAnalysis.STATE_READER_MISMATCH_COUNT.set(0);
        com.powsybl.openloadflow.sa.gpu.GpuWoodburyDcSecurityAnalysis.STATE_FREE_CONTINGENCY_COUNT.set(0);
        long t0 = System.nanoTime();
        SecurityAnalysisResult result = provider.run(network, network.getVariantManager().getWorkingVariantId(),
                        n -> contingencies, runParameters)
                .join().getResult();
        long wallMs = (System.nanoTime() - t0) / 1_000_000;
        com.powsybl.openloadflow.sa.gpu.GpuWoodburyDcSecurityAnalysis.CHECK_DEVICE_TARGETS.set(false);
        com.powsybl.openloadflow.sa.gpu.GpuWoodburyDcSecurityAnalysis.CHECK_NETWORK_RESULTS.set(false);
        com.powsybl.openloadflow.sa.gpu.GpuWoodburyDcSecurityAnalysis.CHECK_STATE_READER.set(false);
        if (gpuProvider) {
            System.out.printf("DC-BASELINE contingencies answered from their state: %d%s%n",
                    com.powsybl.openloadflow.sa.gpu.GpuWoodburyDcSecurityAnalysis.STATE_FREE_CONTINGENCY_COUNT.get(),
                    checkState ? " | values differing from OLF's model: "
                            + com.powsybl.openloadflow.sa.gpu.GpuWoodburyDcSecurityAnalysis.STATE_READER_MISMATCH_COUNT.get() : "");
            System.out.printf("DC-BASELINE network results from the monitored branches found once: %d%s%n",
                    com.powsybl.openloadflow.sa.gpu.GpuWoodburyDcSecurityAnalysis.FAST_NETWORK_RESULT_COUNT.get(),
                    checkResults ? " | differing from OLF's update: "
                            + com.powsybl.openloadflow.sa.gpu.GpuWoodburyDcSecurityAnalysis.NETWORK_RESULT_MISMATCH_COUNT.get() : "");
            System.out.printf("DC-BASELINE device-built modified targets: %d%s%n",
                    com.powsybl.openloadflow.sa.gpu.GpuWoodburyDcSecurityAnalysis.DEVICE_TARGET_COUNT.get(),
                    checkTargets ? " | values differing from the host build: "
                            + com.powsybl.openloadflow.sa.gpu.GpuWoodburyDcSecurityAnalysis.TARGET_MISMATCH_COUNT.get() : "");
        }
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
        for (OperatorStrategyResult o : r.getOperatorStrategyResults()) {
            counts.merge("strategy " + o.getStatus().name(), 1L, Long::sum);
        }
        return counts;
    }

    private static long violationCount(SecurityAnalysisResult r) {
        return r.getPreContingencyResult().getLimitViolationsResult().getLimitViolations().size()
                + r.getPostContingencyResults().stream()
                .mapToLong(p -> p.getLimitViolationsResult().getLimitViolations().size())
                .sum()
                + r.getOperatorStrategyResults().stream()
                .mapToLong(o -> o.getLimitViolationsResult().getLimitViolations().size())
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
        for (OperatorStrategyResult o : r.getOperatorStrategyResults()) {
            m.put("strategy " + o.getOperatorStrategy().getId(), o.getLimitViolationsResult().getLimitViolations().stream()
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
        for (OperatorStrategyResult o : r.getOperatorStrategyResults()) {
            Map<String, Double> byBranch = new HashMap<>();
            for (BranchResult br : o.getNetworkResult().getBranchResults()) {
                byBranch.put(br.getBranchId(), br.getP1());
            }
            m.put("strategy " + o.getOperatorStrategy().getId(), byBranch);
        }
        Map<String, Double> base = new HashMap<>();
        for (BranchResult br : r.getPreContingencyResult().getNetworkResult().getBranchResults()) {
            base.put(br.getBranchId(), br.getP1());
        }
        m.put("<pre-contingency>", base);
        return m;
    }

    private static void printBranches(List<Arm> arms) {
        String ids = System.getenv("OLF_DC_PRINT_BRANCHES");
        if (ids == null || ids.isEmpty()) {
            return;
        }
        for (Arm arm : arms) {
            for (PostContingencyResult p : arm.result().getPostContingencyResults()) {
                System.out.printf("DC-BASELINE CONNECTIVITY %s %s: %d components, lost gen %.3f MW load %.3f MW, elements %s%n", arm.name(),
                        p.getContingency().getId(), p.getConnectivityResult().getCreatedSynchronousComponentCount(),
                        p.getConnectivityResult().getDisconnectedGenerationActivePower(), p.getConnectivityResult().getDisconnectedLoadActivePower(),
                        p.getConnectivityResult().getDisconnectedElements());
            }
        }
        List<Map<String, Map<String, Double>>> p1 = arms.stream().map(a -> monitoredP1(a.result())).toList();
        for (String state : new TreeSet<>(p1.get(0).keySet())) {
            for (String branch : ids.split(",")) {
                StringBuilder line = new StringBuilder(String.format("DC-BASELINE P1 %-22s %-8s", state, branch));
                for (int i = 0; i < arms.size(); i++) {
                    Double v = p1.get(i).getOrDefault(state, Map.of()).get(branch);
                    line.append(String.format(" %s=%s", arms.get(i).name(), v == null ? "-" : String.format("%.6f", v)));
                }
                System.out.println(line);
            }
        }
    }

    private static <T> Set<T> union(Set<T> a, Set<T> b) {
        Set<T> u = new HashSet<>(a);
        u.addAll(b);
        return u;
    }
}
