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
import com.powsybl.iidm.network.Branch;
import com.powsybl.iidm.network.Bus;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.json.JsonLoadFlowParameters;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.CommonTestConfig;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuTestPaths;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.sa.AbstractOpenSecurityAnalysisTest;
import com.powsybl.security.SecurityAnalysisParameters;
import com.powsybl.security.SecurityAnalysisResult;
import com.powsybl.security.results.PostContingencyResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * SCOPE CENSUS under the RTE production parameters: how many of a network's contingencies actually
 * solve ON THE DEVICE, and — for the ones that do not — which qualification reason routed them to
 * the CPU. This is the measurement behind "do ALL contingencies go through the GPU?" on the
 * biggest public case we have (case9241pegase), with the same parameter file production uses.
 *
 * <p>GPU arm only (no CPU comparison — correctness is the other tests' job): the point is the
 * batched SHARE and the reason histogram. Run it with {@code OLF_GPU_QUALIFY_TRACE=1} to get the
 * per-rejection reason lines on stderr; this test prints the totals.
 *
 * <p>Machine-local and OPT-IN ({@code OLF_GPU_SCOPE_CENSUS=1}) — it runs every branch of a 9241-bus
 * network as an N-1, which takes minutes.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuRteScopeCensusTest extends AbstractOpenSecurityAnalysisTest {

    private static final Path CASES = GpuTestPaths.caseDir();
    private static final String CASE = "case9241pegase.xiidm.gz";

    static {
        GpuTestPaths.init();
    }

    GpuRteScopeCensusTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    /** The case to census: {@code OLF_GPU_CENSUS_CASE} (an absolute path — the richer-modelling
     *  networks live outside the case dir) else {@code <caseDir>/case9241pegase.xiidm.gz}. */
    private static Path censusCase() {
        String env = System.getenv("OLF_GPU_CENSUS_CASE");
        return env != null ? Path.of(env) : CASES.resolve(CASE);
    }

    @BeforeEach
    void checkGpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU native lib not available");
        assumeTrue(Files.exists(censusCase()), censusCase() + " not found");
        assumeTrue("1".equals(System.getenv("OLF_GPU_SCOPE_CENSUS")), "OLF_GPU_SCOPE_CENSUS not set");
        // OLF_GPU_CENSUS_CPU=1 runs the STOCK provider — the control arm that says whether a base
        // failure is the network/parameters or the GPU path.
        securityAnalysisProvider = "1".equals(System.getenv("OLF_GPU_CENSUS_CPU"))
                ? new com.powsybl.openloadflow.sa.OpenSecurityAnalysisProvider(new SparseMatrixFactory(),
                        new EvenShiloachGraphDecrementalConnectivityFactory<>())
                : new GpuSecurityAnalysisProvider(new SparseMatrixFactory(),
                        new EvenShiloachGraphDecrementalConnectivityFactory<>());
    }

    @Test
    void censusAllBranchContingenciesUnderRteParameters() throws Exception {
        Network network = Network.read(censusCase());
        Path paramsFile = Path.of("rte-loadflow-default-parameters.json");
        assumeTrue(Files.exists(paramsFile), "rte-loadflow-default-parameters.json not found");
        LoadFlowParameters lfp = JsonLoadFlowParameters.read(paramsFile);
        // OLF_GPU_CENSUS_CONV_EPS: DIAGNOSTIC ONLY — tighten the inner-Newton stopping criterion on BOTH
        // arms to ask whether a residual GPU/CPU difference is just the slack each solver is allowed at
        // the production tolerance (1e-4 per equation by default). Never a production setting, and never
        // a way to claim agreement: the production comparison is the one at the production tolerance.
        if ("1".equals(System.getenv("OLF_GPU_CENSUS_NO_RL"))) {
            lfp.setUseReactiveLimits(false);                  // diagnostic: is a gap the reactive-limits loop?
        }
        if ("1".equals(System.getenv("OLF_GPU_CENSUS_NO_DS"))) {
            lfp.setDistributedSlack(false);                   // diagnostic: is a gap the slack redistribution?
        }
        String convEps = System.getenv("OLF_GPU_CENSUS_CONV_EPS");
        if (convEps != null) {
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                    .setNewtonRaphsonConvEpsPerEq(Double.parseDouble(convEps));
        }
        SecurityAnalysisParameters sap = new SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        // OLF_GPU_CENSUS_VIOL_TOL=<kV>: give the post-vs-base violation comparison a non-zero tolerance
        // on BOTH arms. powsybl defaults the voltage thresholds to 0.0, so an element already violating
        // at base is re-reported on any strictly-worse value, down to one ulp. Diagnostic: it separates
        // "the two arms disagree about the physics" from "the two arms disagree about a tie".
        String violTol = System.getenv("OLF_GPU_CENSUS_VIOL_TOL");
        if (violTol != null) {
            // BOTH thresholds: the rule is min(absolute, value * proportional) and the PROPORTIONAL
            // default is 0.0, so setting the absolute alone leaves the min at zero and changes nothing.
            double t = Double.parseDouble(violTol);
            sap.getIncreasedViolationsParameters()
                    .setLowVoltageAbsoluteThreshold(t).setLowVoltageProportionalThreshold(t)
                    .setHighVoltageAbsoluteThreshold(t).setHighVoltageProportionalThreshold(t);
        }
        // OLF_GPU_CENSUS_THREADS: OpenSecurityAnalysisParameters.threadCount for BOTH arms — the
        // CPU baseline is only honest at the thread count production runs (default here is OLF's 1).
        int threads = Integer.parseInt(System.getenv().getOrDefault("OLF_GPU_CENSUS_THREADS", "1"));
        // OLF_GPU_CENSUS_NOFREEZE: turn OFF the SA's startWithFrozenACEmulation for BOTH arms. The
        // freeze/unfreeze pair is a second solve per contingency on either path, and the only way to
        // know whether the device port pays MORE for it than the CPU does is to price it on both.
        boolean noFreeze = "1".equals(System.getenv("OLF_GPU_CENSUS_NOFREEZE"));
        if (threads > 1 || noFreeze) {
            var ext = new com.powsybl.openloadflow.sa.OpenSecurityAnalysisParameters();
            if (threads > 1) {
                ext.setThreadCount(threads);
            }
            if (noFreeze) {
                ext.setStartWithFrozenACEmulation(false);
            }
            sap.addExtension(com.powsybl.openloadflow.sa.OpenSecurityAnalysisParameters.class, ext);
        }

        List<Contingency> ctgs = new ArrayList<>();
        // OLF_GPU_CENSUS_ONLY=<branch,branch,...>: run just these contingencies. For chasing a handful
        // of cases where the two arms disagree, without paying the whole N-1 each iteration.
        String only = System.getenv("OLF_GPU_CENSUS_ONLY");
        java.util.Set<String> onlyIds = only == null ? null
                : new java.util.HashSet<>(java.util.Arrays.asList(only.split(",")));
        int limit = Integer.parseInt(System.getenv().getOrDefault("OLF_GPU_CENSUS_LIMIT", "0"));
        for (Branch<?> b : network.getBranches()) {
            Bus b1 = b.getTerminal1().getBusView().getBus();
            Bus b2 = b.getTerminal2().getBusView().getBus();
            if (b1 == null || b2 == null
                    || !b1.isInMainSynchronousComponent() || !b2.isInMainSynchronousComponent()) {
                continue;
            }
            if (onlyIds != null && !onlyIds.contains(b.getId())) {
                continue;
            }
            ctgs.add(new Contingency(b.getId(), new BranchContingency(b.getId())));
            if (limit > 0 && ctgs.size() >= limit) {
                break;
            }
        }

        // OLF_GPU_CENSUS_CHUNK: force the batch chunk size (the system property cannot be set from the
        // command line — the parent pom owns surefire's argLine — and chunk size is the knob a
        // throughput measurement most needs to sweep).
        // OLF_GPU_CENSUS_CPU_BASE=1: keep the batched contingency solves on the device but run the BASE
        // load flow on the CPU solver. The reference filtering compares each contingency against the base,
        // with zero tolerance, so a base that differs from the CPU's by a few ulps is enough to move
        // violations. This isolates that.
        if ("1".equals(System.getenv("OLF_GPU_CENSUS_CPU_BASE"))) {
            System.setProperty(GpuBatchedAcSecurityAnalysis.GPU_BASE_LOADFLOW_PROPERTY, "false");
        } else {
            System.clearProperty(GpuBatchedAcSecurityAnalysis.GPU_BASE_LOADFLOW_PROPERTY);
        }
        // OLF_GPU_CENSUS_OUTER_MODE=DEVICE|JAVA: which batched outer-loop engine runs. Comparing the two
        // against the CPU separates a device-kernel difference from an orchestration one.
        String outerMode = System.getenv("OLF_GPU_CENSUS_OUTER_MODE");
        if (outerMode != null) {
            System.setProperty(GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY, outerMode);
        } else {
            System.clearProperty(GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY);
        }
        String chunk = System.getenv("OLF_GPU_CENSUS_CHUNK");
        if (chunk != null) {
            System.setProperty(com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchContextCache.MAX_BATCH_PROPERTY, chunk);
        }

        // OLF_GPU_CENSUS_MONITOR_VL=<vl,vl,...>: monitor those voltage levels so BOTH arms report the
        // post-contingency bus voltages, which is the only way to ask how far apart the two arms'
        // STATES actually are at a bus whose violation they disagree about.
        List<com.powsybl.security.monitor.StateMonitor> monitors = new ArrayList<>();
        // OLF_GPU_CENSUS_MONITOR_ALL_VL=1: monitor EVERY voltage level, so a single contingency's whole
        // voltage profile can be diffed between the arms and the difference located spatially.
        String monitorVl = "1".equals(System.getenv("OLF_GPU_CENSUS_MONITOR_ALL_VL"))
                ? network.getVoltageLevelStream().map(com.powsybl.iidm.network.Identifiable::getId)
                        .collect(java.util.stream.Collectors.joining(","))
                : System.getenv("OLF_GPU_CENSUS_MONITOR_VL");
        String monitorBranch = System.getenv("OLF_GPU_CENSUS_MONITOR_BRANCH");
        if (monitorVl != null || monitorBranch != null) {
            monitors.add(new com.powsybl.security.monitor.StateMonitor(
                    com.powsybl.contingency.ContingencyContext.all(),
                    monitorBranch == null ? java.util.Set.of()
                            : new java.util.HashSet<>(java.util.Arrays.asList(monitorBranch.split(","))),
                    monitorVl == null ? java.util.Set.of()
                            : new java.util.HashSet<>(java.util.Arrays.asList(monitorVl.split(","))),
                    java.util.Set.of()));
        }

        GpuBatchScopeCensus.reset();
        long t0 = System.nanoTime();
        SecurityAnalysisResult result = runSecurityAnalysis(network, ctgs, monitors, sap);
        long ms = (System.nanoTime() - t0) / 1_000_000;

        int batched = GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get();
        Map<String, Long> statuses = new TreeMap<>();
        for (PostContingencyResult p : result.getPostContingencyResults()) {
            statuses.merge(p.getStatus().name(), 1L, Long::sum);
        }
        System.out.printf("%nSCOPE-CENSUS %s | RTE production parameters%n", censusCase().getFileName());
        System.out.printf("  buses (main SC)     : %d%n",
                network.getBusView().getBusStream().filter(Bus::isInMainSynchronousComponent).count());
        System.out.printf("  contingencies       : %d (all main-SC branches)%n", ctgs.size());
        System.out.printf("  solved ON GPU       : %d%n", Math.max(batched, 0));
        System.out.printf("  routed to CPU       : %d%n", ctgs.size() - Math.max(batched, 0));
        System.out.printf("  batched share       : %.1f%%%n",
                100.0 * Math.max(batched, 0) / Math.max(ctgs.size(), 1));
        System.out.printf("  pre-ctg status      : %s (%d violations)%n",
                result.getPreContingencyResult().getStatus(),
                result.getPreContingencyResult().getLimitViolationsResult().getLimitViolations().size());
        System.out.printf("  post-ctg statuses   : %s%n", statuses);
        // How MUCH of the run actually has something to report: the per-contingency violation
        // detection is the largest host-side block, so whether a device-side pre-screen could skip
        // it depends entirely on how many contingencies are violation-free.
        long withViolations = result.getPostContingencyResults().stream()
                .filter(p -> !p.getLimitViolationsResult().getLimitViolations().isEmpty()).count();
        long totalViolations = result.getPostContingencyResults().stream()
                .mapToLong(p -> p.getLimitViolationsResult().getLimitViolations().size()).sum();
        // The pre-contingency violation SET, not just its size: a post-contingency violation is
        // suppressed when the base already had it (LimitViolationManager's reference filtering), so
        // ONE element swapping in or out of the base set shifts every contingency's count at once.
        if (System.getenv("OLF_GPU_CENSUS_BASE_VIOLATIONS") != null) {
            result.getPreContingencyResult().getLimitViolationsResult().getLimitViolations().stream()
                    .map(v -> v.getSubjectId() + "|" + v.getSide() + "|" + v.getLimitType() + "|"
                            + String.format("%.9f", v.getValue()) + "|" + String.format("%.9f", v.getLimit()))
                    .sorted()
                    .forEach(v -> System.out.println("BASE-VIOLATION " + v));
        }
        // OLF_GPU_CENSUS_VIOLATION_DUMP=<file>: every post-contingency violation as a sorted line, so
        // two runs can be DIFFED. A count tells you something moved; only the set tells you what.
        String busDump = System.getenv("OLF_GPU_CENSUS_BUS_DUMP");
        if (busDump != null) {
            List<String> lines = new ArrayList<>();
            // The BASE state too, under the id "BASE": the reference filtering compares each arm's
            // post-contingency value against its OWN base, so a base difference is as decisive as a
            // post difference. Double.toString round-trips exactly — %f would hide the last bits,
            // which are the whole question here.
            for (var br : result.getPreContingencyResult().getNetworkResult().getBusResults()) {
                lines.add("BASE|" + br.getVoltageLevelId() + "|" + br.getBusId() + "|" + Double.toString(br.getV()));
            }
            for (var br : result.getPreContingencyResult().getNetworkResult().getBranchResults()) {
                lines.add("BRANCH|BASE|" + br.getBranchId()
                        + "|p1=" + Double.toString(br.getP1()) + "|q1=" + Double.toString(br.getQ1())
                        + "|i1=" + Double.toString(br.getI1())
                        + "|p2=" + Double.toString(br.getP2()) + "|q2=" + Double.toString(br.getQ2())
                        + "|i2=" + Double.toString(br.getI2()));
            }
            for (PostContingencyResult p : result.getPostContingencyResults()) {
                for (var br : p.getNetworkResult().getBusResults()) {
                    lines.add(p.getContingency().getId() + "|" + br.getVoltageLevelId() + "|"
                            + br.getBusId() + "|" + Double.toString(br.getV()));
                }
                for (var br : p.getNetworkResult().getBranchResults()) {
                    lines.add("BRANCH|" + p.getContingency().getId() + "|" + br.getBranchId()
                            + "|p1=" + Double.toString(br.getP1()) + "|q1=" + Double.toString(br.getQ1())
                            + "|i1=" + Double.toString(br.getI1())
                            + "|p2=" + Double.toString(br.getP2()) + "|q2=" + Double.toString(br.getQ2())
                            + "|i2=" + Double.toString(br.getI2()));
                }
            }
            java.util.Collections.sort(lines);
            Files.write(Path.of(busDump), lines);
        }
        if (System.getenv("OLF_GPU_CENSUS_CONNECTIVITY") != null) {
            for (PostContingencyResult p : result.getPostContingencyResults()) {
                var cr = p.getConnectivityResult();
                System.out.printf("CONNECTIVITY %s components=%d lostLoad=%s lostGen=%s disconnected=%d preDist=%s%n",
                        p.getContingency().getId(), cr.getCreatedSynchronousComponentCount(),
                        Double.toString(cr.getDisconnectedLoadActivePower()),
                        Double.toString(cr.getDisconnectedGenerationActivePower()),
                        cr.getDisconnectedElements().size(),
                        Double.toString(p.getConnectivityResult().getDisconnectedGenerationActivePower()
                                + p.getConnectivityResult().getDisconnectedLoadActivePower()));
            }
        }
        String dump = System.getenv("OLF_GPU_CENSUS_VIOLATION_DUMP");
        if (dump != null) {
            List<String> lines = new ArrayList<>();
            for (PostContingencyResult p : result.getPostContingencyResults()) {
                // the STATUS line first: a contingency the two arms disagree about is usually one that
                // one arm did not converge, and a violation-set diff alone shows that as "N extra
                // violations" rather than as what it is.
                lines.add(p.getContingency().getId() + "|@STATUS|" + p.getStatus() + "||");
                for (var v : p.getLimitViolationsResult().getLimitViolations()) {
                    lines.add(p.getContingency().getId() + "|" + v.getSubjectId() + "|" + v.getSide()
                            + "|" + v.getLimitType() + "|" + v.getOperationalLimitsGroupId());
                }
            }
            java.util.Collections.sort(lines);
            Files.write(Path.of(dump), lines);
        }
        System.out.printf("  ctg with violations : %d of %d (%d violations total)%n",
                withViolations, result.getPostContingencyResults().size(), totalViolations);
        System.out.printf("  CPU-routing reasons : %s%n",
                GpuBatchScopeCensus.snapshot().isEmpty() ? "(none — everything batched)"
                        : GpuBatchScopeCensus.snapshot());
        // A silent CPU fallback of the BASE load flow is invisible in the results but decisive for the
        // violations: the reference would then come from the CPU solver while the contingencies come
        // from the device, and they disagree about anything filtered against the base.
        System.out.printf("  base/seq CPU fallbacks: %d%n",
                com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson.CPU_FALLBACK_COUNT.get());
        System.out.printf("  threads             : %d%n", threads);
        System.out.printf("  wall                : %d ms%n", ms);
        if (batched < 0) {
            System.out.printf("  *** WHOLE-RUN CPU FALLBACK — an unsupported outer loop or an extraction "
                    + "reject killed the batch (run with OLF_GPU_BATCH_TRACE=1 for the scope reason)%n");
        }
    }
}
