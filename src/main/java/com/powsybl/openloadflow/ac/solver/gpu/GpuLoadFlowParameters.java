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
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchContextCache;

import java.util.Objects;

/**
 * The GPU solver's own parameters, the way OLF carries a solver's settings (Newton-Krylov's in
 * OpenLoadFlowParameters): a {@link LoadFlowParameters} extension, read by {@link GpuNewtonRaphsonFactory} into the
 * solver parameters and by the GPU security analysis for its batched solves. They configure cuDSS's linear solve
 * and the iterative refinement on systems with zero-impedance couplers; the defaults are what the device has
 * always used. When a run's parameters carry none, the {@value #MODULE} configuration module provides them.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public class GpuLoadFlowParameters extends AbstractExtension<LoadFlowParameters> {

    public static final String NAME = "gpu-load-flow-parameters";
    public static final String MODULE = "open-loadflow-gpu-default-parameters";

    public static final String REORDERING_ALGORITHM_PARAM_NAME = "reorderingAlgorithm";
    public static final String MATCHING_ALGORITHM_PARAM_NAME = "matchingAlgorithm";
    public static final String PIVOT_TYPE_PARAM_NAME = "pivotType";
    public static final String PIVOT_THRESHOLD_PARAM_NAME = "pivotThreshold";
    public static final String ITERATIVE_REFINEMENT_BASE_STEPS_PARAM_NAME = "iterativeRefinementBaseSteps";
    public static final String ITERATIVE_REFINEMENT_MAX_STEPS_PARAM_NAME = "iterativeRefinementMaxSteps";
    public static final String ITERATIVE_REFINEMENT_TOLERANCE_PARAM_NAME = "iterativeRefinementTolerance";

    public static final GpuReorderingAlgorithm REORDERING_ALGORITHM_DEFAULT_VALUE = GpuReorderingAlgorithm.AUTO;
    public static final GpuMatchingAlgorithm MATCHING_ALGORITHM_DEFAULT_VALUE = GpuMatchingAlgorithm.AUTO;
    public static final GpuPivotType PIVOT_TYPE_DEFAULT_VALUE = GpuPivotType.BY_ORDERING;
    /** NaN: cuDSS's own threshold. */
    public static final double PIVOT_THRESHOLD_DEFAULT_VALUE = Double.NaN;
    public static final int ITERATIVE_REFINEMENT_BASE_STEPS_DEFAULT_VALUE = 1;
    public static final int ITERATIVE_REFINEMENT_MAX_STEPS_DEFAULT_VALUE = 4;
    public static final double ITERATIVE_REFINEMENT_TOLERANCE_DEFAULT_VALUE = 1e-12;

    private GpuReorderingAlgorithm reorderingAlgorithm = REORDERING_ALGORITHM_DEFAULT_VALUE;
    private GpuMatchingAlgorithm matchingAlgorithm = MATCHING_ALGORITHM_DEFAULT_VALUE;
    private GpuPivotType pivotType = PIVOT_TYPE_DEFAULT_VALUE;
    private double pivotThreshold = PIVOT_THRESHOLD_DEFAULT_VALUE;
    private int iterativeRefinementBaseSteps = ITERATIVE_REFINEMENT_BASE_STEPS_DEFAULT_VALUE;
    private int iterativeRefinementMaxSteps = ITERATIVE_REFINEMENT_MAX_STEPS_DEFAULT_VALUE;
    private double iterativeRefinementTolerance = ITERATIVE_REFINEMENT_TOLERANCE_DEFAULT_VALUE;

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
        GpuLoadFlowParameters parameters = new GpuLoadFlowParameters();
        platformConfig.getOptionalModuleConfig(MODULE).ifPresent(config -> {
            config.getOptionalEnumProperty(REORDERING_ALGORITHM_PARAM_NAME, GpuReorderingAlgorithm.class)
                    .ifPresent(parameters::setReorderingAlgorithm);
            config.getOptionalEnumProperty(MATCHING_ALGORITHM_PARAM_NAME, GpuMatchingAlgorithm.class)
                    .ifPresent(parameters::setMatchingAlgorithm);
            config.getOptionalEnumProperty(PIVOT_TYPE_PARAM_NAME, GpuPivotType.class).ifPresent(parameters::setPivotType);
            config.getOptionalDoubleProperty(PIVOT_THRESHOLD_PARAM_NAME).ifPresent(parameters::setPivotThreshold);
            config.getOptionalIntProperty(ITERATIVE_REFINEMENT_BASE_STEPS_PARAM_NAME)
                    .ifPresent(parameters::setIterativeRefinementBaseSteps);
            config.getOptionalIntProperty(ITERATIVE_REFINEMENT_MAX_STEPS_PARAM_NAME)
                    .ifPresent(parameters::setIterativeRefinementMaxSteps);
            config.getOptionalDoubleProperty(ITERATIVE_REFINEMENT_TOLERANCE_PARAM_NAME)
                    .ifPresent(parameters::setIterativeRefinementTolerance);
        });
        return parameters;
    }

    /**
     * Hands these settings to the device, for the contexts built from now on and for every solve's refinement.
     * cuDSS takes its configuration when a context is built, so a change evicts the cached batch context.
     */
    public void install() {
        if (!GpuAcNewtonSolver.isAvailable()) {
            return;
        }
        synchronized (GpuLoadFlowParameters.class) {
            if (sameSettings(installed)) {
                return;
            }
            GpuAcNewtonSolver.setLinearSolverSettings(reorderingAlgorithm.nativeCode(), matchingAlgorithm.nativeCode(),
                    pivotType.nativeCode(), pivotThreshold, iterativeRefinementBaseSteps, iterativeRefinementMaxSteps,
                    iterativeRefinementTolerance);
            GpuBatchContextCache.clearContextCache();
            installed = copy();
        }
    }

    private GpuLoadFlowParameters copy() {
        return new GpuLoadFlowParameters()
                .setReorderingAlgorithm(reorderingAlgorithm)
                .setMatchingAlgorithm(matchingAlgorithm)
                .setPivotType(pivotType)
                .setPivotThreshold(pivotThreshold)
                .setIterativeRefinementBaseSteps(iterativeRefinementBaseSteps)
                .setIterativeRefinementMaxSteps(iterativeRefinementMaxSteps)
                .setIterativeRefinementTolerance(iterativeRefinementTolerance);
    }

    private boolean sameSettings(GpuLoadFlowParameters other) {
        return reorderingAlgorithm == other.reorderingAlgorithm
                && matchingAlgorithm == other.matchingAlgorithm
                && pivotType == other.pivotType
                && Double.compare(pivotThreshold, other.pivotThreshold) == 0
                && iterativeRefinementBaseSteps == other.iterativeRefinementBaseSteps
                && iterativeRefinementMaxSteps == other.iterativeRefinementMaxSteps
                && Double.compare(iterativeRefinementTolerance, other.iterativeRefinementTolerance) == 0;
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
        return "GpuLoadFlowParameters(" + REORDERING_ALGORITHM_PARAM_NAME + "=" + reorderingAlgorithm
                + ", " + MATCHING_ALGORITHM_PARAM_NAME + "=" + matchingAlgorithm
                + ", " + PIVOT_TYPE_PARAM_NAME + "=" + pivotType
                + ", " + PIVOT_THRESHOLD_PARAM_NAME + "=" + pivotThreshold
                + ", " + ITERATIVE_REFINEMENT_BASE_STEPS_PARAM_NAME + "=" + iterativeRefinementBaseSteps
                + ", " + ITERATIVE_REFINEMENT_MAX_STEPS_PARAM_NAME + "=" + iterativeRefinementMaxSteps
                + ", " + ITERATIVE_REFINEMENT_TOLERANCE_PARAM_NAME + "=" + iterativeRefinementTolerance + ")";
    }
}
