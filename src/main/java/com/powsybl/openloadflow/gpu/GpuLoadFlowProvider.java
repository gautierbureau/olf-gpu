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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Collectors;

/**
 * OLF's load flow, with the AC Newton-Raphson on the GPU (name {@value #NAME}, the same as the GPU security
 * analysis provider's). Choosing this provider is choosing the GPU; there is no solver type to set.
 *
 * <p>OLF's own parameters are reused as they are: the {@link OpenLoadFlowParameters} extension, its JSON entry, its
 * configuration module and its defaults. This provider's specific parameters are only the GPU solver's,
 * {@link GpuLoadFlowParameters}, so one parameters file carries both extensions. By name and value (pypowsybl's
 * {@code provider_parameters}) this provider also takes OLF's parameters: the provider SPI gives a provider a single
 * specific-parameters extension, so the GPU one carries them and the run applies them over its OLF parameters. A DC
 * load flow has no GPU path and runs as in OLF.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
@AutoService(LoadFlowProvider.class)
public class GpuLoadFlowProvider extends OpenLoadFlowProvider {

    public static final String NAME = "OpenLoadFlowGpu";

    private static final Set<String> OPEN_LOAD_FLOW_PARAMETER_NAMES = OpenLoadFlowParameters.SPECIFIC_PARAMETERS.stream()
            .map(Parameter::getName)
            .collect(Collectors.toUnmodifiableSet());

    /** OLF's parameters, then the GPU solver's; no name is both. */
    static final List<Parameter> SPECIFIC_PARAMETERS = specificParameters();

    private static List<Parameter> specificParameters() {
        List<Parameter> parameters = new ArrayList<>(OpenLoadFlowParameters.SPECIFIC_PARAMETERS);
        for (Parameter gpu : GpuLoadFlowParameters.SPECIFIC_PARAMETERS) {
            if (OPEN_LOAD_FLOW_PARAMETER_NAMES.contains(gpu.getName())) {
                throw new IllegalStateException("GPU solver parameter " + gpu.getName() + " is also an OLF parameter");
            }
            parameters.add(gpu);
        }
        return List.copyOf(parameters);
    }

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
     * The caller's parameters with OLF's parameters set by name through this provider applied, and the GPU
     * Newton-Raphson as the AC solver. A copy: the caller's parameters are left as they were, so reusing them with
     * OLF's provider runs on the CPU. {@link LoadFlowParameters#copy()} goes through the JSON form, so it keeps every
     * extension, OLF's and the GPU solver's. A DC load flow keeps OLF's solver, and is not copied when there is
     * nothing to apply. The GPU sensitivity analysis provider runs its load flows on these parameters too.
     */
    public static LoadFlowParameters withGpuSolver(LoadFlowParameters parameters) {
        GpuLoadFlowParameters gpu = parameters.getExtension(GpuLoadFlowParameters.class);
        Map<String, String> openLoadFlowProperties = gpu != null ? gpu.getOpenLoadFlowParameters() : Map.of();
        if (parameters.isDc() && openLoadFlowProperties.isEmpty()) {
            return parameters;
        }
        LoadFlowParameters copy = parameters.copy();
        OpenLoadFlowParameters parametersExt = copy.getExtension(OpenLoadFlowParameters.class);
        if (parametersExt == null) {
            parametersExt = OpenLoadFlowParameters.create(copy);   // OLF's defaults, as OpenLoadFlowParameters.get gives
        }
        parametersExt.update(openLoadFlowProperties);
        if (!copy.isDc()) {
            parametersExt.setAcSolverType(GpuNewtonRaphsonFactory.NAME);
        }
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

    /** By name and value: the GPU solver's parameters, and OLF's, applied over the run's OLF parameters. */
    @Override
    public Optional<Extension<LoadFlowParameters>> loadSpecificParameters(Map<String, String> properties) {
        return Optional.of(GpuLoadFlowParameters.load(properties).setOpenLoadFlowParameters(openLoadFlowProperties(properties)));
    }

    /** OLF's parameters, then the GPU solver's: every name this provider takes by name and value. */
    @Override
    public List<Parameter> getRawSpecificParameters() {
        return SPECIFIC_PARAMETERS;
    }

    @Override
    public void updateSpecificParameters(Extension<LoadFlowParameters> extension, Map<String, String> properties) {
        ((GpuLoadFlowParameters) extension).update(properties).putOpenLoadFlowParameters(openLoadFlowProperties(properties));
    }

    @Override
    public void updateSpecificParameters(Extension<LoadFlowParameters> extension, PlatformConfig config) {
        ((GpuLoadFlowParameters) extension).update(config);
    }

    @Override
    public Optional<Class<? extends Extension<LoadFlowParameters>>> getSpecificParametersClass() {
        return Optional.of(GpuLoadFlowParameters.class);
    }

    /** The GPU solver's parameters, and OLF's set by name through this provider. */
    @Override
    public Map<String, String> createMapFromSpecificParameters(Extension<LoadFlowParameters> extension) {
        GpuLoadFlowParameters gpu = (GpuLoadFlowParameters) extension;
        Map<String, String> map = new HashMap<>(gpu.getOpenLoadFlowParameters());
        gpu.toMap().forEach((key, value) -> map.put(key, Objects.toString(value)));
        return map;
    }

    /**
     * The OLF parameters among {@code properties}, validated: a value OLF rejects fails here, when it is set, not
     * at the run. The GPU solver's names are left to {@link GpuLoadFlowParameters#update(Map)}.
     */
    static Map<String, String> openLoadFlowProperties(Map<String, String> properties) {
        Map<String, String> olf = new LinkedHashMap<>();
        properties.forEach((name, value) -> {
            if (OPEN_LOAD_FLOW_PARAMETER_NAMES.contains(name)) {
                olf.put(name, value);
            }
        });
        OpenLoadFlowParameters.load(olf);
        return olf;
    }

    @Override
    public Optional<ModuleConfig> getModuleConfig(PlatformConfig platformConfig) {
        return platformConfig.getOptionalModuleConfig(GpuLoadFlowParameters.MODULE);
    }
}
