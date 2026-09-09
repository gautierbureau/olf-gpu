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
        SecurityAnalysisParameters sap = new SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
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
        int limit = Integer.parseInt(System.getenv().getOrDefault("OLF_GPU_CENSUS_LIMIT", "0"));
        for (Branch<?> b : network.getBranches()) {
            Bus b1 = b.getTerminal1().getBusView().getBus();
            Bus b2 = b.getTerminal2().getBusView().getBus();
            if (b1 == null || b2 == null
                    || !b1.isInMainSynchronousComponent() || !b2.isInMainSynchronousComponent()) {
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
        String chunk = System.getenv("OLF_GPU_CENSUS_CHUNK");
        if (chunk != null) {
            System.setProperty(com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchContextCache.MAX_BATCH_PROPERTY, chunk);
        }

        GpuBatchScopeCensus.reset();
        long t0 = System.nanoTime();
        SecurityAnalysisResult result = runSecurityAnalysis(network, ctgs, List.of(), sap);
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
        System.out.printf("  ctg with violations : %d of %d (%d violations total)%n",
                withViolations, result.getPostContingencyResults().size(), totalViolations);
        System.out.printf("  CPU-routing reasons : %s%n",
                GpuBatchScopeCensus.snapshot().isEmpty() ? "(none — everything batched)"
                        : GpuBatchScopeCensus.snapshot());
        System.out.printf("  threads             : %d%n", threads);
        System.out.printf("  wall                : %d ms%n", ms);
        if (System.getenv("OLF_PROBE_STATE_UPDATE") != null) {
            System.out.printf("  onStateUpdate split : updateVariables=%d ms updateNetworkState=%d ms%n",
                    com.powsybl.openloadflow.ac.equations.vector.AcNetworkVector.VARS_NS.sum() / 1_000_000,
                    com.powsybl.openloadflow.ac.equations.vector.AcNetworkVector.STATE_NS.sum() / 1_000_000);
        }
        if (batched < 0) {
            System.out.printf("  *** WHOLE-RUN CPU FALLBACK — an unsupported outer loop or an extraction "
                    + "reject killed the batch (run with OLF_GPU_BATCH_TRACE=1 for the scope reason)%n");
        }
    }
}
