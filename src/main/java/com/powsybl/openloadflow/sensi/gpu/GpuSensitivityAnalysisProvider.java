/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sensi.gpu;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.auto.service.AutoService;
import com.powsybl.commons.PowsyblException;
import com.powsybl.commons.extensions.ExtensionJsonSerializer;
import com.powsybl.iidm.network.Network;
import com.powsybl.math.matrix.MatrixFactory;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.gpu.GpuLoadFlowProvider;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.graph.GraphConnectivityFactory;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.sensi.OpenSensitivityAnalysisParameters;
import com.powsybl.openloadflow.sensi.OpenSensitivityAnalysisProvider;
import com.powsybl.sensitivity.SensitivityAnalysisParameters;
import com.powsybl.sensitivity.SensitivityAnalysisProvider;
import com.powsybl.sensitivity.SensitivityAnalysisRunParameters;
import com.powsybl.sensitivity.SensitivityFactorReader;
import com.powsybl.sensitivity.SensitivityResultWriter;
import com.powsybl.sensitivity.json.JsonSensitivityAnalysisParameters;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;

/**
 * OLF's sensitivity analysis with its AC load flows on the GPU (name {@value #NAME}, the same as the GPU load flow and
 * security analysis providers'): the base case and every post-contingency load flow run on the GPU Newton-Raphson,
 * as {@link GpuLoadFlowProvider} runs a load flow, and the sensitivities are then computed by OLF on the Jacobian at
 * the state those load flows reach. The GPU solver hands its final state back to OLF's network as OLF's own outer
 * loops do (the state vector, the generators it switched to PQ at their reactive limits and back), so that Jacobian
 * holds the equations the run converged on.
 *
 * <p>OLF's sensitivity parameters and its load flow parameters are used as they are, on a copy: the caller's
 * parameters are left as they were, so reusing them with OLF's provider runs on the CPU. The copy selects the GPU
 * Newton-Raphson and runs single-threaded, the device solving one load flow at a time. A DC sensitivity analysis has
 * no GPU path and runs as in OLF. Without the GPU library an AC run fails, as the GPU load flow does.
 *
 * <p>The load flow provider of this provider is {@link GpuLoadFlowProvider}, so a caller building the run's load flow
 * parameters from provider parameters by name (pypowsybl's {@code provider_parameters}) sets the GPU solver's and
 * OLF's.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
@AutoService(SensitivityAnalysisProvider.class)
public class GpuSensitivityAnalysisProvider extends OpenSensitivityAnalysisProvider {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuSensitivityAnalysisProvider.class);

    public static final String NAME = GpuLoadFlowProvider.NAME;

    public GpuSensitivityAnalysisProvider() {
        this(new SparseMatrixFactory());
    }

    public GpuSensitivityAnalysisProvider(MatrixFactory matrixFactory) {
        this(matrixFactory, new EvenShiloachGraphDecrementalConnectivityFactory<>());
    }

    public GpuSensitivityAnalysisProvider(MatrixFactory matrixFactory, GraphConnectivityFactory<LfBus, LfBranch> connectivityFactory) {
        super(matrixFactory, connectivityFactory);
    }

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    public Optional<String> getLoadFlowProviderName() {
        return Optional.of(GpuLoadFlowProvider.NAME);
    }

    /**
     * None: this provider's specific parameters are OLF's ({@link OpenSensitivityAnalysisParameters}), which OLF's
     * provider already (de)serializes. powsybl collects the serializers of every sensitivity analysis provider by
     * extension name, two under one name are a duplicate key, and the JSON form of sensitivity parameters failed to
     * write or read at all while both providers were registered.
     */
    @Override
    public Optional<ExtensionJsonSerializer> getSpecificParametersSerializer() {
        return Optional.empty();
    }

    @Override
    public CompletableFuture<Void> run(Network network, String workingVariantId, SensitivityFactorReader factorReader,
                                       SensitivityResultWriter resultWriter, SensitivityAnalysisRunParameters runParameters) {
        Objects.requireNonNull(runParameters);
        SensitivityAnalysisParameters parameters = Objects.requireNonNull(runParameters.getSensitivityAnalysisParameters());
        if (parameters.getLoadFlowParameters().isDc()) {
            return super.run(network, workingVariantId, factorReader, resultWriter, runParameters);
        }
        SensitivityAnalysisRunParameters onDevice = new SensitivityAnalysisRunParameters()
                .setParameters(withGpuSolver(parameters))
                .setComputationManager(runParameters.getComputationManager())
                .setContingencies(runParameters.getContingencies())
                .setVariableSets(runParameters.getVariableSets())
                .setOperatorStrategies(runParameters.getOperatorStrategies())
                .setActions(runParameters.getActions())
                .setReportNode(runParameters.getReportNode());
        return super.run(network, workingVariantId, factorReader, resultWriter, onDevice);
    }

    /**
     * The caller's parameters for a run on the device: a copy (through the JSON form, so it keeps every extension,
     * OLF's sensitivity parameters and the load flow parameters' own), its load flow parameters those of a GPU load
     * flow ({@link GpuLoadFlowProvider#withGpuSolver}), and one thread: the device runs one load flow at a time, and
     * concurrent native solves on one device are not supported.
     */
    static SensitivityAnalysisParameters withGpuSolver(SensitivityAnalysisParameters parameters) {
        SensitivityAnalysisParameters copy = copy(parameters);
        copy.setLoadFlowParameters(GpuLoadFlowProvider.withGpuSolver(copy.getLoadFlowParameters()));
        OpenSensitivityAnalysisParameters parametersExt = copy.getExtension(OpenSensitivityAnalysisParameters.class);
        if (parametersExt != null && parametersExt.getThreadCount() > 1) {
            LOGGER.warn("GPU sensitivity analysis: threadCount {} ignored, the device runs one load flow at a time. "
                    + "Use the CPU provider for multi-threading", parametersExt.getThreadCount());
            parametersExt.setThreadCount(1);
        }
        return copy;
    }

    private static SensitivityAnalysisParameters copy(SensitivityAnalysisParameters parameters) {
        try {
            ObjectMapper mapper = JsonSensitivityAnalysisParameters.createObjectMapper();
            return JsonSensitivityAnalysisParameters.read(new ByteArrayInputStream(mapper.writeValueAsBytes(parameters)));
        } catch (IOException e) {
            throw new PowsyblException("GPU sensitivity analysis: cannot copy the sensitivity analysis parameters", e);
        }
    }
}
