/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.gpu;

import com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory;
import com.powsybl.iidm.network.Bus;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlow;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.LoadFlowProvider;
import com.powsybl.loadflow.json.JsonLoadFlowParameters;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonFactory;
import com.powsybl.openloadflow.ac.solver.gpu.GpuLoadFlowParameters;
import com.powsybl.openloadflow.ac.solver.gpu.GpuMatchingAlgorithm;
import com.powsybl.openloadflow.ac.solver.gpu.GpuPivotType;
import com.powsybl.openloadflow.ac.solver.gpu.GpuReorderingAlgorithm;
import com.powsybl.openloadflow.ac.solver.gpu.NewtonSettingsNative;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuLoadFlowProviderTest {

    private static LoadFlowParameters jsonRoundTrip(LoadFlowParameters parameters) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        JsonLoadFlowParameters.write(parameters, out);
        return JsonLoadFlowParameters.read(new ByteArrayInputStream(out.toByteArray()));
    }

    @Test
    void isFoundByName() {
        assertEquals(GpuLoadFlowProvider.NAME, LoadFlow.find(GpuLoadFlowProvider.NAME).getName());
    }

    /** One parameters file carries OLF's parameters and the GPU solver's, each under its own name. */
    @Test
    void aParametersFileCarriesBothExtensions() {
        LoadFlowParameters parameters = new LoadFlowParameters();
        OpenLoadFlowParameters.create(parameters).setMaxNewtonRaphsonIterations(17);
        GpuLoadFlowParameters.create(parameters)
                .setReorderingAlgorithm(GpuReorderingAlgorithm.NESTED_DISSECTION)
                .setPivotThreshold(0.25)
                .setIterativeRefinementMaxSteps(6);

        LoadFlowParameters read = jsonRoundTrip(parameters);

        assertEquals(17, read.getExtension(OpenLoadFlowParameters.class).getMaxNewtonRaphsonIterations());
        GpuLoadFlowParameters gpu = read.getExtension(GpuLoadFlowParameters.class);
        assertEquals(GpuReorderingAlgorithm.NESTED_DISSECTION, gpu.getReorderingAlgorithm());
        assertEquals(0.25, gpu.getPivotThreshold());
        assertEquals(6, gpu.getIterativeRefinementMaxSteps());
        assertEquals(GpuMatchingAlgorithm.AUTO, gpu.getMatchingAlgorithm());
    }

    /** The default pivot threshold, NaN (cuDSS's own), survives the file. */
    @Test
    void theDefaultPivotThresholdSurvivesTheFile() {
        LoadFlowParameters parameters = new LoadFlowParameters();
        GpuLoadFlowParameters.create(parameters);
        assertTrue(Double.isNaN(jsonRoundTrip(parameters).getExtension(GpuLoadFlowParameters.class).getPivotThreshold()));
    }

    /** By name and value, as pypowsybl's provider parameters set them. */
    @Test
    void nameValueParameters() {
        LoadFlowProvider provider = new GpuLoadFlowProvider();
        assertTrue(provider.getRawSpecificParameters().stream()
                .anyMatch(p -> p.getName().equals(GpuLoadFlowParameters.REORDERING_ALGORITHM_PARAM_NAME)));
        GpuLoadFlowParameters gpu = (GpuLoadFlowParameters) provider.loadSpecificParameters(Map.of(
                GpuLoadFlowParameters.REORDERING_ALGORITHM_PARAM_NAME, "AMD",
                GpuLoadFlowParameters.PIVOT_TYPE_PARAM_NAME, "GLOBAL_ROW",
                GpuLoadFlowParameters.ITERATIVE_REFINEMENT_TOLERANCE_PARAM_NAME, "1e-10")).orElseThrow();
        assertEquals(GpuReorderingAlgorithm.AMD, gpu.getReorderingAlgorithm());
        assertEquals(GpuPivotType.GLOBAL_ROW, gpu.getPivotType());
        assertEquals(1e-10, gpu.getIterativeRefinementTolerance());

        provider.updateSpecificParameters(gpu, Map.of(GpuLoadFlowParameters.ITERATIVE_REFINEMENT_BASE_STEPS_PARAM_NAME, "3"));
        Map<String, String> map = provider.createMapFromSpecificParameters(gpu);
        assertEquals("AMD", map.get(GpuLoadFlowParameters.REORDERING_ALGORITHM_PARAM_NAME));
        assertEquals("3", map.get(GpuLoadFlowParameters.ITERATIVE_REFINEMENT_BASE_STEPS_PARAM_NAME));
    }

    /**
     * Choosing the provider is choosing the GPU: with no solver type set, the run solves on the device with the
     * run's GPU parameters, gives OLF's CPU answer, and leaves the caller's parameters as they were.
     */
    @Test
    void runsOnTheDeviceWithTheRunsParameters() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        LoadFlowParameters parameters = new LoadFlowParameters();
        GpuLoadFlowParameters.create(parameters)
                .setReorderingAlgorithm(GpuReorderingAlgorithm.NESTED_DISSECTION)
                .setMatchingAlgorithm(GpuMatchingAlgorithm.MAX_DIAG_SUM)
                .setPivotType(GpuPivotType.GLOBAL_ROW)
                .setPivotThreshold(0.25)
                .setIterativeRefinementBaseSteps(2)
                .setIterativeRefinementMaxSteps(5)
                .setIterativeRefinementTolerance(1e-11);
        Network gpuNetwork = IeeeCdfNetworkFactory.create14();
        Network cpuNetwork = IeeeCdfNetworkFactory.create14();
        try {
            assertTrue(LoadFlow.find(GpuLoadFlowProvider.NAME).run(gpuNetwork, parameters).isFullyConverged());
            assertArrayEquals(new double[] {4, 4, 3, 0.25, 2, 5, 1e-11}, NewtonSettingsNative.readLinearSolverSettings());
        } finally {
            new GpuLoadFlowParameters().install();               // the next tests run with the defaults
        }
        assertNull(parameters.getExtension(OpenLoadFlowParameters.class), "the caller's parameters are not modified");

        assertTrue(LoadFlow.find("OpenLoadFlow").run(cpuNetwork, new LoadFlowParameters()).isFullyConverged());
        for (Bus cpuBus : cpuNetwork.getBusView().getBuses()) {
            Bus gpuBus = gpuNetwork.getBusView().getBus(cpuBus.getId());
            assertEquals(cpuBus.getV(), gpuBus.getV(), 1e-9, cpuBus.getId());
            assertEquals(cpuBus.getAngle(), gpuBus.getAngle(), 1e-9, cpuBus.getId());
        }
    }

    /** The copy the provider runs on selects the GPU solver; the caller's keeps OLF's. */
    @Test
    void theRunUsesACopyWithTheGpuSolver() {
        LoadFlowParameters parameters = new LoadFlowParameters();
        OpenLoadFlowParameters.create(parameters).setMaxNewtonRaphsonIterations(17);
        LoadFlowParameters onDevice = GpuLoadFlowProvider.withGpuSolver(parameters);
        assertEquals("GPU_NEWTON_RAPHSON", OpenLoadFlowParameters.get(onDevice).getAcSolverType());
        assertEquals(17, OpenLoadFlowParameters.get(onDevice).getMaxNewtonRaphsonIterations());
        assertEquals(NewtonRaphsonFactory.NAME, OpenLoadFlowParameters.get(parameters).getAcSolverType());
    }

    /** A DC load flow has no GPU path: it runs as in OLF. */
    @Test
    void dcRunsAsInOlf() {
        LoadFlowParameters parameters = new LoadFlowParameters().setDc(true);
        assertTrue(LoadFlow.find(GpuLoadFlowProvider.NAME).run(IeeeCdfNetworkFactory.create14(), parameters).isFullyConverged());
        assertNull(parameters.getExtension(OpenLoadFlowParameters.class));
    }
}
