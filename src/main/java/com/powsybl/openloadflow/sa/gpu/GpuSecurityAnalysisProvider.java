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
import com.powsybl.openloadflow.sa.OpenSecurityAnalysisProvider;
import com.powsybl.openloadflow.util.PowsyblOpenLoadFlowVersion;
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
                runParameters.getComputationManager(), runParameters.getOperatorStrategies(), runParameters.getActions(), runParameters.getLimitReductions());
    }
}
