/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.google.auto.service.AutoService;
import com.powsybl.contingency.ContingenciesProvider;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.math.matrix.MatrixFactory;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.graph.GraphConnectivityFactory;
import com.powsybl.openloadflow.graph.NaiveGraphConnectivityFactory;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.sa.AcSecurityAnalysis;
import com.powsybl.openloadflow.sa.OpenSecurityAnalysisParameters;
import com.powsybl.openloadflow.sa.OpenSecurityAnalysisProvider;
import com.powsybl.openloadflow.util.PowsyblOpenLoadFlowVersion;
import com.powsybl.security.SecurityAnalysisParameters;
import com.powsybl.security.SecurityAnalysisProvider;
import com.powsybl.security.SecurityAnalysisReport;
import com.powsybl.security.SecurityAnalysisRunParameters;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/**
 * GPU-batched AC security analysis provider (name {@value #NAME}): the AC path runs
 * {@link GpuBatchedAcSecurityAnalysis} — single-branch N-1 contingencies solved on the device in
 * one batch — and DEGRADES to the standard {@link AcSecurityAnalysis} behavior when no GPU is
 * present (the GPU analysis falls back internally). DC parameters delegate to the standard
 * OpenLoadFlow provider path.
 *
 * <p>This lives in the powsybl-open-loadflow-gpu EXTENSION: selecting this provider (by name) is
 * the GPU opt-in, replacing the {@code olf.sa.gpuBatched} system-property hook the fork-based
 * prototype patched into {@link OpenSecurityAnalysisProvider} — stock OLF is never modified.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
@AutoService(SecurityAnalysisProvider.class)
public class GpuSecurityAnalysisProvider extends OpenSecurityAnalysisProvider {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuSecurityAnalysisProvider.class);

    public static final String NAME = "OpenLoadFlowGpu";

    private final MatrixFactory matrixFactory;

    private final GraphConnectivityFactory<LfBus, LfBranch> connectivityFactory;

    public GpuSecurityAnalysisProvider(MatrixFactory matrixFactory, GraphConnectivityFactory<LfBus, LfBranch> connectivityFactory) {
        super(matrixFactory, connectivityFactory);
        this.matrixFactory = Objects.requireNonNull(matrixFactory);
        this.connectivityFactory = Objects.requireNonNull(connectivityFactory);
    }

    public GpuSecurityAnalysisProvider() {
        this(new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
    }

    /** The last run's GPU engagement counter (see {@code GpuBatchedAcSecurityAnalysis}):
     *  -1 = the WHOLE run fell back to the sequential CPU path, 0 = the batch engaged but every
     *  contingency was individually rejected to CPU, N = contingencies actually solved on the
     *  device. External validation harnesses read this to prove a comparison was real. */
    public static int lastGpuPrecomputedCount() {
        return GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get();
    }

    /** Reset the engagement counter to the whole-run-fallback sentinel before a run. */
    public static void resetLastGpuPrecomputedCount() {
        GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.set(-1);
        GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_IDS.clear();
        GpuBatchedAcSecurityAnalysis.LAST_REPLAYED_LOSS_COUNT.set(0);
        GpuBatchedAcSecurityAnalysis.LAST_TARGETED_RESTORE_COUNT.set(0);
    }

    /** System property: allow the GPU provider to honour {@code threadCount > 1} again. Off by
     *  default — see {@link #forceSinglePartition}. */
    public static final String ALLOW_MULTI_THREAD_PROPERTY = "olf.gpu.sa.allowMultiThread";

    /**
     * Force the GPU security analysis onto ONE partition.
     *
     * <p>On this path everything is meant to run on the device, the base load flow included — that is
     * the point of the provider, and the batched contingency solves are anchored on the base state it
     * produces. OLF partitions the contingency list across threads and gives each partition its own
     * LfNetwork and its own base load flow, and the GPU base load flow is skipped when there is more
     * than one, because concurrent partition load flows race in native code on the single device.
     *
     * <p>That leaves the reference (a CPU-solved base) and the contingency states (device-solved)
     * coming from different solvers. Post-contingency violations are filtered against the base with
     * NO tolerance, so a far-field bus that should reproduce the base exactly instead lands a couple
     * of ulps away and trips the comparison: measured on rte6515 at 2000 contingencies, threadCount 1
     * reports VL-1512 zero times and threadCount 4 reports it 500 times. A threading option must not
     * change which violations a security analysis reports.
     *
     * <p>So until the two paths are validated as equivalent, the GPU provider runs single-partition
     * and keeps every solve on the device. The CPU provider is unaffected and remains the way to use
     * many threads. Lift this with {@code -Dolf.gpu.sa.allowMultiThread=true} once parity is
     * established.
     */
    private static int forceSinglePartition(SecurityAnalysisParameters securityAnalysisParameters) {
        if (Boolean.getBoolean(ALLOW_MULTI_THREAD_PROPERTY)) {
            return -1;
        }
        OpenSecurityAnalysisParameters ext =
                securityAnalysisParameters.getExtension(OpenSecurityAnalysisParameters.class);
        if (ext == null || ext.getThreadCount() <= 1) {
            return -1;
        }
        int requested = ext.getThreadCount();
        LOGGER.warn("GPU security analysis: threadCount {} ignored, running single-partition so the BASE "
                + "load flow also runs on the device (a CPU-solved base and device-solved contingencies "
                + "disagree about violations). Use the CPU provider for multi-threading, or set -D{}=true",
                requested, ALLOW_MULTI_THREAD_PROPERTY);
        ext.setThreadCount(1);
        return requested;
    }

    /** Put back the caller's threadCount once the run is over: the parameters belong to the caller,
     *  who may well reuse them for a CPU run — silently leaving them at 1 would change THAT run. */
    private static void restoreThreadCount(SecurityAnalysisParameters securityAnalysisParameters, int requested) {
        if (requested > 1) {
            OpenSecurityAnalysisParameters ext =
                    securityAnalysisParameters.getExtension(OpenSecurityAnalysisParameters.class);
            if (ext != null) {
                ext.setThreadCount(requested);
            }
        }
    }

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    public CompletableFuture<SecurityAnalysisReport> run(Network network,
                                                         String workingVariantId,
                                                         ContingenciesProvider contingenciesProvider,
                                                         SecurityAnalysisRunParameters runParameters) {
        Objects.requireNonNull(network);
        Objects.requireNonNull(workingVariantId);
        Objects.requireNonNull(contingenciesProvider);
        Objects.requireNonNull(runParameters);

        LoadFlowParameters loadFlowParameters = runParameters.getSecurityAnalysisParameters().getLoadFlowParameters();
        if (loadFlowParameters.isDc()) {
            // DC has no GPU path: the standard OpenLoadFlow security analysis handles it.
            return super.run(network, workingVariantId, contingenciesProvider, runParameters);
        }

        int requestedThreads = forceSinglePartition(runParameters.getSecurityAnalysisParameters());
        resetLastGpuPrecomputedCount();                  // the counter ACCUMULATES over the run's partitions
        LOGGER.info("Version: {} (GPU-batched AC security analysis)", new PowsyblOpenLoadFlowVersion());
        OpenLoadFlowParameters loadFlowParametersExt = OpenLoadFlowParameters.get(loadFlowParameters);
        OpenLoadFlowParameters.log(loadFlowParameters, loadFlowParametersExt);

        GraphConnectivityFactory<LfBus, LfBranch> selectedConnectivityFactory;
        if (runParameters.getOperatorStrategies().isEmpty() && !loadFlowParametersExt.isSimulateAutomationSystems()) {
            selectedConnectivityFactory = connectivityFactory;
        } else {
            LOGGER.warn("Naive (and slow!!!) connectivity algorithm has been selected because at least one operator strategy is configured");
            selectedConnectivityFactory = new NaiveGraphConnectivityFactory<>(LfBus::getNum);
        }

        // GPU-batched AC security analysis (single-branch N-1 solved on the device in one
        // batch). Degrades to the standard AC path when no GPU is present.
        GpuBatchedAcSecurityAnalysis securityAnalysis = new GpuBatchedAcSecurityAnalysis(
                network, matrixFactory, selectedConnectivityFactory, runParameters.getMonitors(), runParameters.getReportNode());
        return securityAnalysis.run(workingVariantId, runParameters.getSecurityAnalysisParameters(), contingenciesProvider,
                runParameters.getComputationManager(), runParameters.getOperatorStrategies(), runParameters.getActions(), runParameters.getLimitReductions())
                .whenComplete((r, t) -> restoreThreadCount(runParameters.getSecurityAnalysisParameters(), requestedThreads));
    }
}
