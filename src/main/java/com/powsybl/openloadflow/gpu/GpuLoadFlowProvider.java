/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.gpu;

import com.google.auto.service.AutoService;
import com.powsybl.commons.config.ModuleConfig;
import com.powsybl.commons.config.PlatformConfig;
import com.powsybl.commons.extensions.Extension;
import com.powsybl.commons.extensions.ExtensionJsonSerializer;
import com.powsybl.commons.parameters.Parameter;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.LoadFlowProvider;
import com.powsybl.loadflow.LoadFlowResult;
import com.powsybl.loadflow.LoadFlowRunParameters;
import com.powsybl.math.matrix.MatrixFactory;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.OpenLoadFlowProvider;
import com.powsybl.openloadflow.ac.solver.gpu.GpuLoadFlowParameterJsonSerializer;
import com.powsybl.openloadflow.ac.solver.gpu.GpuLoadFlowParameters;
import com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphsonFactory;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.graph.GraphConnectivityFactory;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;

/**
 * OLF's load flow, with the AC Newton-Raphson on the GPU (name {@value #NAME}, the same as the GPU security
 * analysis provider's). Choosing this provider is choosing the GPU; there is no solver type to set.
 *
 * <p>OLF's own parameters are reused as they are: the {@link OpenLoadFlowParameters} extension, its JSON entry, its
 * configuration module and its defaults. This provider's specific parameters are only the GPU solver's,
 * {@link GpuLoadFlowParameters}, so one parameters file carries both extensions. A DC load flow has no GPU path and
 * runs as in OLF.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
@AutoService(LoadFlowProvider.class)
public class GpuLoadFlowProvider extends OpenLoadFlowProvider {

    public static final String NAME = "OpenLoadFlowGpu";

    public GpuLoadFlowProvider() {
        this(new SparseMatrixFactory());
    }

    public GpuLoadFlowProvider(MatrixFactory matrixFactory) {
        this(matrixFactory, new EvenShiloachGraphDecrementalConnectivityFactory<>());
    }

    public GpuLoadFlowProvider(MatrixFactory matrixFactory, GraphConnectivityFactory<LfBus, LfBranch> connectivityFactory) {
        super(matrixFactory, connectivityFactory);
    }

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    public CompletableFuture<LoadFlowResult> run(Network network, String workingVariantId, LoadFlowRunParameters loadFlowRunParameters) {
        Objects.requireNonNull(loadFlowRunParameters);
        LoadFlowParameters parameters = Objects.requireNonNull(loadFlowRunParameters.getLoadFlowParameters());
        LoadFlowRunParameters onDevice = new LoadFlowRunParameters()
                .setParameters(withGpuSolver(parameters))
                .setComputationManager(loadFlowRunParameters.getComputationManager())
                .setReportNode(loadFlowRunParameters.getReportNode());
        return super.run(network, workingVariantId, onDevice);
    }

    /**
     * The caller's parameters with the GPU Newton-Raphson as the AC solver. A copy: the caller's parameters are
     * left as they were, so reusing them with OLF's provider runs on the CPU. {@link LoadFlowParameters#copy()}
     * goes through the JSON form, so it keeps every extension, OLF's and the GPU solver's.
     */
    static LoadFlowParameters withGpuSolver(LoadFlowParameters parameters) {
        if (parameters.isDc()) {
            return parameters;
        }
        LoadFlowParameters copy = parameters.copy();
        OpenLoadFlowParameters parametersExt = copy.getExtension(OpenLoadFlowParameters.class);
        if (parametersExt == null) {
            parametersExt = OpenLoadFlowParameters.create(copy);   // OLF's defaults, as OpenLoadFlowParameters.get gives
        }
        parametersExt.setAcSolverType(GpuNewtonRaphsonFactory.NAME);
        return copy;
    }

    @Override
    public Optional<ExtensionJsonSerializer> getSpecificParametersSerializer() {
        return Optional.of(new GpuLoadFlowParameterJsonSerializer());
    }

    @Override
    public Optional<Extension<LoadFlowParameters>> loadSpecificParameters(PlatformConfig platformConfig) {
        return Optional.of(GpuLoadFlowParameters.load(platformConfig));
    }

    @Override
    public Optional<Extension<LoadFlowParameters>> loadSpecificParameters(Map<String, String> properties) {
        return Optional.of(GpuLoadFlowParameters.load(properties));
    }

    @Override
    public List<Parameter> getRawSpecificParameters() {
        return GpuLoadFlowParameters.SPECIFIC_PARAMETERS;
    }

    @Override
    public void updateSpecificParameters(Extension<LoadFlowParameters> extension, Map<String, String> properties) {
        ((GpuLoadFlowParameters) extension).update(properties);
    }

    @Override
    public void updateSpecificParameters(Extension<LoadFlowParameters> extension, PlatformConfig config) {
        ((GpuLoadFlowParameters) extension).update(config);
    }

    @Override
    public Optional<Class<? extends Extension<LoadFlowParameters>>> getSpecificParametersClass() {
        return Optional.of(GpuLoadFlowParameters.class);
    }

    @Override
    public Map<String, String> createMapFromSpecificParameters(Extension<LoadFlowParameters> extension) {
        Map<String, String> map = new HashMap<>();
        ((GpuLoadFlowParameters) extension).toMap().forEach((key, value) -> map.put(key, Objects.toString(value)));
        return map;
    }

    @Override
    public Optional<ModuleConfig> getModuleConfig(PlatformConfig platformConfig) {
        return platformConfig.getOptionalModuleConfig(GpuLoadFlowParameters.MODULE);
    }
}
