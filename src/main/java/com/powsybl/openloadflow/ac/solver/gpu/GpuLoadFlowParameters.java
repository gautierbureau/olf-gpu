/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

import com.powsybl.commons.config.PlatformConfig;
import com.powsybl.commons.extensions.AbstractExtension;
import com.powsybl.commons.parameters.Parameter;
import com.powsybl.commons.parameters.ParameterScope;
import com.powsybl.commons.parameters.ParameterType;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The GPU solver's own parameters, the way OLF carries a solver's settings (Newton-Krylov's in
 * OpenLoadFlowParameters): a {@link LoadFlowParameters} extension, read by {@link GpuNewtonRaphsonFactory} into the
 * solver parameters and by the GPU security analysis for its batched solves. They pick the batched linear solver,
 * configure cuDSS's linear solve and the iterative refinement on systems with zero-impedance couplers; the defaults
 * are what the device has always used. When a run's parameters carry none, the {@value #MODULE} configuration module provides them.
 * They are the specific parameters of the {@code OpenLoadFlowGpu} load flow provider, next to OLF's own
 * {@code OpenLoadFlowParameters}, so a parameters JSON file, a name/value map or the configuration module can
 * set them like any other provider parameter.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public class GpuLoadFlowParameters extends AbstractExtension<LoadFlowParameters> {

    public static final String NAME = "gpu-load-flow-parameters";
    public static final String MODULE = "open-loadflow-gpu-default-parameters";

    public static final String LINEAR_SOLVER_PARAM_NAME = "linearSolver";
    public static final String REORDERING_ALGORITHM_PARAM_NAME = "reorderingAlgorithm";
    public static final String MATCHING_ALGORITHM_PARAM_NAME = "matchingAlgorithm";
    public static final String PIVOT_TYPE_PARAM_NAME = "pivotType";
    public static final String PIVOT_THRESHOLD_PARAM_NAME = "pivotThreshold";
    public static final String ITERATIVE_REFINEMENT_BASE_STEPS_PARAM_NAME = "iterativeRefinementBaseSteps";
    public static final String ITERATIVE_REFINEMENT_MAX_STEPS_PARAM_NAME = "iterativeRefinementMaxSteps";
    public static final String ITERATIVE_REFINEMENT_TOLERANCE_PARAM_NAME = "iterativeRefinementTolerance";

    public static final GpuLinearSolver LINEAR_SOLVER_DEFAULT_VALUE = GpuLinearSolver.CUDSS;
    public static final GpuReorderingAlgorithm REORDERING_ALGORITHM_DEFAULT_VALUE = GpuReorderingAlgorithm.AUTO;
    public static final GpuMatchingAlgorithm MATCHING_ALGORITHM_DEFAULT_VALUE = GpuMatchingAlgorithm.AUTO;
    public static final GpuPivotType PIVOT_TYPE_DEFAULT_VALUE = GpuPivotType.BY_ORDERING;
    /** NaN: cuDSS's own threshold. */
    public static final double PIVOT_THRESHOLD_DEFAULT_VALUE = Double.NaN;
    public static final int ITERATIVE_REFINEMENT_BASE_STEPS_DEFAULT_VALUE = 1;
    public static final int ITERATIVE_REFINEMENT_MAX_STEPS_DEFAULT_VALUE = 4;
    public static final double ITERATIVE_REFINEMENT_TOLERANCE_DEFAULT_VALUE = 1e-12;

    private static final String SOLVER_CATEGORY_KEY = "Solver";

    public static final List<Parameter> SPECIFIC_PARAMETERS = List.of(
        new Parameter(LINEAR_SOLVER_PARAM_NAME, ParameterType.STRING, "Linear solver of the batched Newton-Raphson",
            LINEAR_SOLVER_DEFAULT_VALUE.name(), names(GpuLinearSolver.values()), ParameterScope.TECHNICAL, SOLVER_CATEGORY_KEY),
        new Parameter(REORDERING_ALGORITHM_PARAM_NAME, ParameterType.STRING, "cuDSS fill-reducing ordering",
            REORDERING_ALGORITHM_DEFAULT_VALUE.name(), names(GpuReorderingAlgorithm.values()), ParameterScope.TECHNICAL, SOLVER_CATEGORY_KEY),
        new Parameter(MATCHING_ALGORITHM_PARAM_NAME, ParameterType.STRING, "cuDSS matching algorithm (robust single-matrix solve)",
            MATCHING_ALGORITHM_DEFAULT_VALUE.name(), names(GpuMatchingAlgorithm.values()), ParameterScope.TECHNICAL, SOLVER_CATEGORY_KEY),
        new Parameter(PIVOT_TYPE_PARAM_NAME, ParameterType.STRING, "cuDSS pivoting (robust single-matrix solve)",
            PIVOT_TYPE_DEFAULT_VALUE.name(), names(GpuPivotType.values()), ParameterScope.TECHNICAL, SOLVER_CATEGORY_KEY),
        new Parameter(PIVOT_THRESHOLD_PARAM_NAME, ParameterType.DOUBLE, "cuDSS pivot threshold in [0, 1], NaN for cuDSS's own",
            PIVOT_THRESHOLD_DEFAULT_VALUE, ParameterScope.TECHNICAL, SOLVER_CATEGORY_KEY),
        new Parameter(ITERATIVE_REFINEMENT_BASE_STEPS_PARAM_NAME, ParameterType.INTEGER,
            "Iterative refinement steps of every linear solve (on the batched LU, of every refinement correction: its Newton solve"
                + " is refined adaptively only)",
            ITERATIVE_REFINEMENT_BASE_STEPS_DEFAULT_VALUE, ParameterScope.TECHNICAL, SOLVER_CATEGORY_KEY),
        new Parameter(ITERATIVE_REFINEMENT_MAX_STEPS_PARAM_NAME, ParameterType.INTEGER, "Iterative refinement steps at most",
            ITERATIVE_REFINEMENT_MAX_STEPS_DEFAULT_VALUE, ParameterScope.TECHNICAL, SOLVER_CATEGORY_KEY),
        new Parameter(ITERATIVE_REFINEMENT_TOLERANCE_PARAM_NAME, ParameterType.DOUBLE, "Iterative refinement relative residual target",
            ITERATIVE_REFINEMENT_TOLERANCE_DEFAULT_VALUE, ParameterScope.TECHNICAL, SOLVER_CATEGORY_KEY)
    );

    private static List<Object> names(Enum<?>[] values) {
        return Arrays.stream(values).map(Enum::name).map(Object.class::cast).toList();
    }

    private GpuLinearSolver linearSolver = LINEAR_SOLVER_DEFAULT_VALUE;
    private GpuReorderingAlgorithm reorderingAlgorithm = REORDERING_ALGORITHM_DEFAULT_VALUE;
    private GpuMatchingAlgorithm matchingAlgorithm = MATCHING_ALGORITHM_DEFAULT_VALUE;
    private GpuPivotType pivotType = PIVOT_TYPE_DEFAULT_VALUE;
    private double pivotThreshold = PIVOT_THRESHOLD_DEFAULT_VALUE;
    private int iterativeRefinementBaseSteps = ITERATIVE_REFINEMENT_BASE_STEPS_DEFAULT_VALUE;
    private int iterativeRefinementMaxSteps = ITERATIVE_REFINEMENT_MAX_STEPS_DEFAULT_VALUE;
    private double iterativeRefinementTolerance = ITERATIVE_REFINEMENT_TOLERANCE_DEFAULT_VALUE;

    /**
     * OLF's parameters set by name through the {@code OpenLoadFlowGpu} provider's specific parameters (pypowsybl's
     * {@code provider_parameters}), applied over the run's {@code OpenLoadFlowParameters}. The provider SPI gives a
     * provider ONE specific-parameters extension, this one, so it carries them; they are kept as text, the way they
     * were given, and validated by the provider when they are set.
     */
    private Map<String, String> openLoadFlowParameters = new LinkedHashMap<>();

    /** The settings last handed to the device (see {@link #install}). */
    private static GpuLoadFlowParameters installed = new GpuLoadFlowParameters();

    @Override
    public String getName() {
        return NAME;
    }

    /** The run's GPU parameters: its extension, or the {@value #MODULE} configuration module's. */
    public static GpuLoadFlowParameters get(LoadFlowParameters parameters) {
        GpuLoadFlowParameters ext = parameters.getExtension(GpuLoadFlowParameters.class);
        return ext != null ? ext : load(PlatformConfig.defaultConfig());
    }

    /** Creates the extension on {@code parameters} if it has none, and returns it. */
    public static GpuLoadFlowParameters create(LoadFlowParameters parameters) {
        GpuLoadFlowParameters ext = parameters.getExtension(GpuLoadFlowParameters.class);
        if (ext == null) {
            ext = load(PlatformConfig.defaultConfig());
            parameters.addExtension(GpuLoadFlowParameters.class, ext);
        }
        return ext;
    }

    public static GpuLoadFlowParameters load(PlatformConfig platformConfig) {
        return new GpuLoadFlowParameters().update(platformConfig);
    }

    public static GpuLoadFlowParameters load(Map<String, String> properties) {
        return new GpuLoadFlowParameters().update(properties);
    }

    /** Sets the parameters the {@value #MODULE} configuration module holds; the others are unchanged. */
    public GpuLoadFlowParameters update(PlatformConfig platformConfig) {
        platformConfig.getOptionalModuleConfig(MODULE).ifPresent(config -> {
            config.getOptionalEnumProperty(LINEAR_SOLVER_PARAM_NAME, GpuLinearSolver.class).ifPresent(this::setLinearSolver);
            config.getOptionalEnumProperty(REORDERING_ALGORITHM_PARAM_NAME, GpuReorderingAlgorithm.class)
                    .ifPresent(this::setReorderingAlgorithm);
            config.getOptionalEnumProperty(MATCHING_ALGORITHM_PARAM_NAME, GpuMatchingAlgorithm.class)
                    .ifPresent(this::setMatchingAlgorithm);
            config.getOptionalEnumProperty(PIVOT_TYPE_PARAM_NAME, GpuPivotType.class).ifPresent(this::setPivotType);
            config.getOptionalDoubleProperty(PIVOT_THRESHOLD_PARAM_NAME).ifPresent(this::setPivotThreshold);
            config.getOptionalIntProperty(ITERATIVE_REFINEMENT_BASE_STEPS_PARAM_NAME)
                    .ifPresent(this::setIterativeRefinementBaseSteps);
            config.getOptionalIntProperty(ITERATIVE_REFINEMENT_MAX_STEPS_PARAM_NAME)
                    .ifPresent(this::setIterativeRefinementMaxSteps);
            config.getOptionalDoubleProperty(ITERATIVE_REFINEMENT_TOLERANCE_PARAM_NAME)
                    .ifPresent(this::setIterativeRefinementTolerance);
        });
        return this;
    }

    /** Sets the parameters named in {@code properties}; the others are unchanged. */
    public GpuLoadFlowParameters update(Map<String, String> properties) {
        Objects.requireNonNull(properties);
        String value = properties.get(LINEAR_SOLVER_PARAM_NAME);
        if (value != null) {
            setLinearSolver(GpuLinearSolver.valueOf(value));
        }
        value = properties.get(REORDERING_ALGORITHM_PARAM_NAME);
        if (value != null) {
            setReorderingAlgorithm(GpuReorderingAlgorithm.valueOf(value));
        }
        value = properties.get(MATCHING_ALGORITHM_PARAM_NAME);
        if (value != null) {
            setMatchingAlgorithm(GpuMatchingAlgorithm.valueOf(value));
        }
        value = properties.get(PIVOT_TYPE_PARAM_NAME);
        if (value != null) {
            setPivotType(GpuPivotType.valueOf(value));
        }
        value = properties.get(PIVOT_THRESHOLD_PARAM_NAME);
        if (value != null) {
            setPivotThreshold(Double.parseDouble(value));
        }
        value = properties.get(ITERATIVE_REFINEMENT_BASE_STEPS_PARAM_NAME);
        if (value != null) {
            setIterativeRefinementBaseSteps(Integer.parseInt(value));
        }
        value = properties.get(ITERATIVE_REFINEMENT_MAX_STEPS_PARAM_NAME);
        if (value != null) {
            setIterativeRefinementMaxSteps(Integer.parseInt(value));
        }
        value = properties.get(ITERATIVE_REFINEMENT_TOLERANCE_PARAM_NAME);
        if (value != null) {
            setIterativeRefinementTolerance(Double.parseDouble(value));
        }
        return this;
    }

    /** Every parameter by name, in {@link #SPECIFIC_PARAMETERS} order. */
    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put(LINEAR_SOLVER_PARAM_NAME, linearSolver);
        map.put(REORDERING_ALGORITHM_PARAM_NAME, reorderingAlgorithm);
        map.put(MATCHING_ALGORITHM_PARAM_NAME, matchingAlgorithm);
        map.put(PIVOT_TYPE_PARAM_NAME, pivotType);
        map.put(PIVOT_THRESHOLD_PARAM_NAME, pivotThreshold);
        map.put(ITERATIVE_REFINEMENT_BASE_STEPS_PARAM_NAME, iterativeRefinementBaseSteps);
        map.put(ITERATIVE_REFINEMENT_MAX_STEPS_PARAM_NAME, iterativeRefinementMaxSteps);
        map.put(ITERATIVE_REFINEMENT_TOLERANCE_PARAM_NAME, iterativeRefinementTolerance);
        return map;
    }

    /**
     * Hands these settings to the device, for the contexts built from now on and for every solve's refinement.
     * cuDSS takes its configuration when a context is built, so a change evicts the cached batch context. The linear
     * solver is the mode a context is built in ({@link GpuBatchContextCache#resolveBatchMode(GpuLoadFlowParameters)}).
     */
    public void install() {
        if (!GpuAcNewtonSolver.isAvailable()) {
            return;
        }
        synchronized (GpuLoadFlowParameters.class) {
            if (sameSettings(installed)) {
                return;
            }
            NewtonSettingsNative.setLinearSolverSettings(reorderingAlgorithm.nativeCode(), matchingAlgorithm.nativeCode(),
                    pivotType.nativeCode(), pivotThreshold, iterativeRefinementBaseSteps, iterativeRefinementMaxSteps,
                    iterativeRefinementTolerance);
            GpuBatchContextCache.clearContextCache();
            installed = copy();
        }
    }

    private GpuLoadFlowParameters copy() {
        return new GpuLoadFlowParameters()
                .setLinearSolver(linearSolver)
                .setReorderingAlgorithm(reorderingAlgorithm)
                .setMatchingAlgorithm(matchingAlgorithm)
                .setPivotType(pivotType)
                .setPivotThreshold(pivotThreshold)
                .setIterativeRefinementBaseSteps(iterativeRefinementBaseSteps)
                .setIterativeRefinementMaxSteps(iterativeRefinementMaxSteps)
                .setIterativeRefinementTolerance(iterativeRefinementTolerance);
    }

    private boolean sameSettings(GpuLoadFlowParameters other) {
        return linearSolver == other.linearSolver
                && reorderingAlgorithm == other.reorderingAlgorithm
                && matchingAlgorithm == other.matchingAlgorithm
                && pivotType == other.pivotType
                && Double.compare(pivotThreshold, other.pivotThreshold) == 0
                && iterativeRefinementBaseSteps == other.iterativeRefinementBaseSteps
                && iterativeRefinementMaxSteps == other.iterativeRefinementMaxSteps
                && Double.compare(iterativeRefinementTolerance, other.iterativeRefinementTolerance) == 0;
    }

    public GpuLinearSolver getLinearSolver() {
        return linearSolver;
    }

    public GpuLoadFlowParameters setLinearSolver(GpuLinearSolver linearSolver) {
        this.linearSolver = Objects.requireNonNull(linearSolver);
        return this;
    }

    public GpuReorderingAlgorithm getReorderingAlgorithm() {
        return reorderingAlgorithm;
    }

    public GpuLoadFlowParameters setReorderingAlgorithm(GpuReorderingAlgorithm reorderingAlgorithm) {
        this.reorderingAlgorithm = Objects.requireNonNull(reorderingAlgorithm);
        return this;
    }

    public GpuMatchingAlgorithm getMatchingAlgorithm() {
        return matchingAlgorithm;
    }

    public GpuLoadFlowParameters setMatchingAlgorithm(GpuMatchingAlgorithm matchingAlgorithm) {
        this.matchingAlgorithm = Objects.requireNonNull(matchingAlgorithm);
        return this;
    }

    public GpuPivotType getPivotType() {
        return pivotType;
    }

    public GpuLoadFlowParameters setPivotType(GpuPivotType pivotType) {
        this.pivotType = Objects.requireNonNull(pivotType);
        return this;
    }

    public double getPivotThreshold() {
        return pivotThreshold;
    }

    /** NaN for cuDSS's own threshold, otherwise in [0, 1]. */
    public GpuLoadFlowParameters setPivotThreshold(double pivotThreshold) {
        if (!Double.isNaN(pivotThreshold) && (pivotThreshold < 0 || pivotThreshold > 1)) {
            throw new IllegalArgumentException("Invalid value for parameter " + PIVOT_THRESHOLD_PARAM_NAME + ": " + pivotThreshold);
        }
        this.pivotThreshold = pivotThreshold;
        return this;
    }

    public int getIterativeRefinementBaseSteps() {
        return iterativeRefinementBaseSteps;
    }

    public GpuLoadFlowParameters setIterativeRefinementBaseSteps(int iterativeRefinementBaseSteps) {
        if (iterativeRefinementBaseSteps < 0) {
            throw new IllegalArgumentException("Invalid value for parameter " + ITERATIVE_REFINEMENT_BASE_STEPS_PARAM_NAME
                    + ": " + iterativeRefinementBaseSteps);
        }
        this.iterativeRefinementBaseSteps = iterativeRefinementBaseSteps;
        return this;
    }

    public int getIterativeRefinementMaxSteps() {
        return iterativeRefinementMaxSteps;
    }

    public GpuLoadFlowParameters setIterativeRefinementMaxSteps(int iterativeRefinementMaxSteps) {
        if (iterativeRefinementMaxSteps < 0) {
            throw new IllegalArgumentException("Invalid value for parameter " + ITERATIVE_REFINEMENT_MAX_STEPS_PARAM_NAME
                    + ": " + iterativeRefinementMaxSteps);
        }
        this.iterativeRefinementMaxSteps = iterativeRefinementMaxSteps;
        return this;
    }

    /** OLF's parameters set by name through the provider, by OLF parameter name (an unmodifiable view). */
    public Map<String, String> getOpenLoadFlowParameters() {
        return Collections.unmodifiableMap(openLoadFlowParameters);
    }

    /** Replaces OLF's parameters set by name through the provider. */
    public GpuLoadFlowParameters setOpenLoadFlowParameters(Map<String, String> openLoadFlowParameters) {
        this.openLoadFlowParameters = new LinkedHashMap<>(Objects.requireNonNull(openLoadFlowParameters));
        return this;
    }

    /** Adds OLF's parameters set by name through the provider; a name already set takes the new value. */
    public GpuLoadFlowParameters putOpenLoadFlowParameters(Map<String, String> openLoadFlowParameters) {
        this.openLoadFlowParameters.putAll(Objects.requireNonNull(openLoadFlowParameters));
        return this;
    }

    public double getIterativeRefinementTolerance() {
        return iterativeRefinementTolerance;
    }

    public GpuLoadFlowParameters setIterativeRefinementTolerance(double iterativeRefinementTolerance) {
        if (!(iterativeRefinementTolerance > 0)) {
            throw new IllegalArgumentException("Invalid value for parameter " + ITERATIVE_REFINEMENT_TOLERANCE_PARAM_NAME
                    + ": " + iterativeRefinementTolerance);
        }
        this.iterativeRefinementTolerance = iterativeRefinementTolerance;
        return this;
    }

    @Override
    public String toString() {
        return "GpuLoadFlowParameters(" + LINEAR_SOLVER_PARAM_NAME + "=" + linearSolver
                + ", " + REORDERING_ALGORITHM_PARAM_NAME + "=" + reorderingAlgorithm
                + ", " + MATCHING_ALGORITHM_PARAM_NAME + "=" + matchingAlgorithm
                + ", " + PIVOT_TYPE_PARAM_NAME + "=" + pivotType
                + ", " + PIVOT_THRESHOLD_PARAM_NAME + "=" + pivotThreshold
                + ", " + ITERATIVE_REFINEMENT_BASE_STEPS_PARAM_NAME + "=" + iterativeRefinementBaseSteps
                + ", " + ITERATIVE_REFINEMENT_MAX_STEPS_PARAM_NAME + "=" + iterativeRefinementMaxSteps
                + ", " + ITERATIVE_REFINEMENT_TOLERANCE_PARAM_NAME + "=" + iterativeRefinementTolerance
                + (openLoadFlowParameters.isEmpty() ? "" : ", openLoadFlowParameters=" + openLoadFlowParameters) + ")";
    }
}
