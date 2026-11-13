/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sensi.gpu;

import com.powsybl.contingency.BranchContingency;
import com.powsybl.contingency.Contingency;
import com.powsybl.contingency.ContingencyContext;
import com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory;
import com.powsybl.iidm.network.Line;
import com.powsybl.iidm.network.Network;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonFactory;
import com.powsybl.openloadflow.ac.solver.gpu.GpuLoadFlowParameters;
import com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphsonFactory;
import com.powsybl.openloadflow.ac.solver.gpu.GpuReorderingAlgorithm;
import com.powsybl.openloadflow.gpu.GpuLoadFlowProvider;
import com.powsybl.openloadflow.sensi.OpenSensitivityAnalysisParameters;
import com.powsybl.sensitivity.SensitivityAnalysis;
import com.powsybl.sensitivity.SensitivityAnalysisParameters;
import com.powsybl.sensitivity.SensitivityAnalysisResult;
import com.powsybl.sensitivity.SensitivityAnalysisRunParameters;
import com.powsybl.sensitivity.SensitivityFactor;
import com.powsybl.sensitivity.SensitivityFunctionType;
import com.powsybl.sensitivity.SensitivityValue;
import com.powsybl.sensitivity.SensitivityVariableType;
import com.powsybl.sensitivity.json.JsonSensitivityAnalysisParameters;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeFalse;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuSensitivityAnalysisProviderTest {

    private static final String NAME = GpuSensitivityAnalysisProvider.NAME;

    /** IEEE 14 with the synchronous condenser of bus 8 limited to +-6 MVar: it holds its voltage with about 17 MVar,
     *  so the load flow switches it to PQ at its limit, and a sensitivity to its target voltage is 0. */
    private static Network ieee14WithACondenserAtItsLimit() {
        Network network = IeeeCdfNetworkFactory.create14();
        network.getGenerator("B8-G").newMinMaxReactiveLimits().setMinQ(-6).setMaxQ(6).add();
        return network;
    }

    /** The flow of every line per injection of two generators, and the voltage of four buses per the target voltage
     *  of the condenser of bus 6 and of the limited one of bus 8, in every state. */
    private static List<SensitivityFactor> factors(Network network) {
        List<SensitivityFactor> factors = new ArrayList<>();
        for (Line line : network.getLines()) {
            for (String generator : List.of("B2-G", "B3-G")) {
                factors.add(new SensitivityFactor(SensitivityFunctionType.BRANCH_ACTIVE_POWER_1, line.getId(),
                        SensitivityVariableType.INJECTION_ACTIVE_POWER, generator, false, ContingencyContext.all()));
            }
        }
        for (String bus : List.of("B4", "B8", "B9", "B13")) {
            for (String generator : List.of("B6-G", "B8-G")) {
                factors.add(new SensitivityFactor(SensitivityFunctionType.BUS_VOLTAGE, bus,
                        SensitivityVariableType.BUS_TARGET_VOLTAGE, generator, false, ContingencyContext.all()));
            }
        }
        return factors;
    }

    private static List<Contingency> contingencies() {
        return List.of(new Contingency("L1-2-1", new BranchContingency("L1-2-1")),
                new Contingency("L2-3-1", new BranchContingency("L2-3-1")));
    }

    /** The flow of every line per injection of two generators: what a DC sensitivity analysis computes. */
    private static List<SensitivityFactor> flowFactors(Network network) {
        return factors(network).stream()
                .filter(f -> f.getFunctionType() == SensitivityFunctionType.BRANCH_ACTIVE_POWER_1)
                .toList();
    }

    private static SensitivityAnalysisResult run(String provider, Network network, List<SensitivityFactor> factors,
                                                 SensitivityAnalysisParameters parameters) {
        return SensitivityAnalysis.find(provider).run(network, network.getVariantManager().getWorkingVariantId(),
                factors, new SensitivityAnalysisRunParameters().setParameters(parameters)
                        .setContingencies(contingencies()));
    }

    /** The values of a result by (factor, contingency), the pre-contingency state as contingency -1. */
    private static Map<List<Integer>, SensitivityValue> byFactorAndContingency(SensitivityAnalysisResult result) {
        Map<List<Integer>, SensitivityValue> values = new HashMap<>();
        for (SensitivityValue value : result.getValues()) {
            values.put(List.of(value.getFactorIndex(), value.getContingencyIndex()), value);
        }
        return values;
    }

    @Test
    void isFoundByName() {
        assertEquals(NAME, SensitivityAnalysis.find(NAME).getName());
        assertEquals(Optional.of(GpuLoadFlowProvider.NAME), new GpuSensitivityAnalysisProvider().getLoadFlowProviderName());
    }

    /**
     * The run's parameters are a copy: the GPU Newton-Raphson as the AC solver, one thread, every other setting of the
     * sensitivity parameters and of the load flow parameters kept; the caller's are left as they were.
     */
    @Test
    void theRunUsesACopyWithTheGpuSolverAndOneThread() {
        SensitivityAnalysisParameters parameters = new SensitivityAnalysisParameters().setFlowFlowSensitivityValueThreshold(1e-3);
        OpenLoadFlowParameters.create(parameters.getLoadFlowParameters()).setMaxNewtonRaphsonIterations(17);
        OpenSensitivityAnalysisParameters sensitivityParametersExt = new OpenSensitivityAnalysisParameters()
                .setThreadCount(4)
                .setStartWithFrozenACEmulation(false);
        parameters.addExtension(OpenSensitivityAnalysisParameters.class, sensitivityParametersExt);

        SensitivityAnalysisParameters onDevice = GpuSensitivityAnalysisProvider.withGpuSolver(parameters);

        OpenLoadFlowParameters loadFlowParametersExt = OpenLoadFlowParameters.get(onDevice.getLoadFlowParameters());
        assertEquals(GpuNewtonRaphsonFactory.NAME, loadFlowParametersExt.getAcSolverType());
        assertEquals(17, loadFlowParametersExt.getMaxNewtonRaphsonIterations());
        assertEquals(1e-3, onDevice.getFlowFlowSensitivityValueThreshold());
        OpenSensitivityAnalysisParameters onDeviceExt = onDevice.getExtension(OpenSensitivityAnalysisParameters.class);
        assertEquals(1, onDeviceExt.getThreadCount());
        assertFalse(onDeviceExt.isStartWithFrozenACEmulation());

        assertNotSame(parameters, onDevice);
        assertEquals(4, sensitivityParametersExt.getThreadCount(), "the caller's parameters are not modified");
        assertEquals(NewtonRaphsonFactory.NAME, OpenLoadFlowParameters.get(parameters.getLoadFlowParameters()).getAcSolverType());
    }

    /** The GPU solver's parameters, and OLF's set by name through the GPU load flow provider, reach the copy. */
    @Test
    void theGpuSolverParametersAndOlfOnesByNameReachTheCopy() {
        SensitivityAnalysisParameters parameters = new SensitivityAnalysisParameters();
        GpuLoadFlowParameters gpu = (GpuLoadFlowParameters) new GpuLoadFlowProvider().loadSpecificParameters(Map.of(
                GpuLoadFlowParameters.REORDERING_ALGORITHM_PARAM_NAME, "AMD",
                OpenLoadFlowParameters.MAX_NEWTON_RAPHSON_ITERATIONS_PARAM_NAME, "7")).orElseThrow();
        parameters.getLoadFlowParameters().addExtension(GpuLoadFlowParameters.class, gpu);

        SensitivityAnalysisParameters onDevice = GpuSensitivityAnalysisProvider.withGpuSolver(parameters);

        assertEquals(GpuReorderingAlgorithm.AMD,
                onDevice.getLoadFlowParameters().getExtension(GpuLoadFlowParameters.class).getReorderingAlgorithm());
        assertEquals(7, OpenLoadFlowParameters.get(onDevice.getLoadFlowParameters()).getMaxNewtonRaphsonIterations());
    }

    /**
     * The JSON form of sensitivity parameters writes and reads with this provider registered beside OLF's: it adds no
     * serializer of its own under the name of OLF's extension (a duplicate key for powsybl, which failed every JSON
     * write and read).
     */
    @Test
    void sensitivityParametersStillWriteAndReadAsJson() throws IOException {
        SensitivityAnalysisParameters parameters = new SensitivityAnalysisParameters();
        parameters.addExtension(OpenSensitivityAnalysisParameters.class, new OpenSensitivityAnalysisParameters().setThreadCount(3));
        byte[] json = JsonSensitivityAnalysisParameters.createObjectMapper().writeValueAsBytes(parameters);
        SensitivityAnalysisParameters read = JsonSensitivityAnalysisParameters.read(new ByteArrayInputStream(json));
        assertEquals(3, read.getExtension(OpenSensitivityAnalysisParameters.class).getThreadCount());
        assertTrue(new GpuSensitivityAnalysisProvider().getSpecificParametersSerializer().isEmpty());
    }

    /** A DC sensitivity analysis has no GPU path: it runs as in OLF, to the last bit. */
    @Test
    void dcRunsAsInOlf() {
        Supplier<SensitivityAnalysisParameters> dc = () -> {
            SensitivityAnalysisParameters parameters = new SensitivityAnalysisParameters();
            parameters.getLoadFlowParameters().setDc(true);
            return parameters;
        };
        Network network = IeeeCdfNetworkFactory.create14();
        Map<List<Integer>, SensitivityValue> gpu = byFactorAndContingency(run(NAME, network, flowFactors(network), dc.get()));
        Map<List<Integer>, SensitivityValue> cpu = byFactorAndContingency(run("OpenLoadFlow", network, flowFactors(network),
                dc.get()));
        assertFalse(cpu.isEmpty());
        assertEquals(cpu.keySet(), gpu.keySet());
        for (Map.Entry<List<Integer>, SensitivityValue> entry : cpu.entrySet()) {
            assertEquals(entry.getValue().getValue(), gpu.get(entry.getKey()).getValue(), 0.0, "sensitivity " + entry.getKey());
            assertEquals(entry.getValue().getFunctionReference(), gpu.get(entry.getKey()).getFunctionReference(), 0.0,
                    "function reference " + entry.getKey());
        }
    }

    /** Without the GPU library an AC sensitivity analysis fails, as a GPU load flow does: it does not silently run on the
     *  CPU. */
    @Test
    void anAcRunWithoutTheGpuLibraryFails() {
        assumeFalse(GpuAcNewtonSolver.isAvailable(), "the GPU library is available");
        Network network = IeeeCdfNetworkFactory.create14();
        RuntimeException e = assertThrows(RuntimeException.class,
                () -> run(NAME, network, factors(network), new SensitivityAnalysisParameters()));
        boolean said = false;
        for (Throwable t = e; t != null && !said; t = t.getCause()) {
            said = t.getMessage() != null && t.getMessage().contains("native library is not available");
        }
        assertTrue(said, "the failure names the missing GPU library: " + e);
    }

    /**
     * On the device, every load flow of the run - the base case and both contingencies - runs on the GPU, ends with OLF's
     * status (L1-2-1's fails, as on the CPU) and the sensitivities OLF computes on the Jacobian the converged ones leave
     * are OLF's own on the CPU, function references included.
     * The condenser of bus 8 ends PQ at its limit in every state, so its target voltage moves nothing (0 on both): the
     * GPU solver handed the switch back to OLF's equation system, which the Jacobian is built from.
     */
    @Test
    void acSensitivitiesAreOlfsOnTheCpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Supplier<SensitivityAnalysisParameters> tight = () -> {
            SensitivityAnalysisParameters parameters = new SensitivityAnalysisParameters();
            OpenLoadFlowParameters.create(parameters.getLoadFlowParameters()).setNewtonRaphsonConvEpsPerEq(1e-10);
            return parameters;
        };
        Network gpuNetwork = ieee14WithACondenserAtItsLimit();
        Network cpuNetwork = ieee14WithACondenserAtItsLimit();
        SensitivityAnalysisResult gpuResult = run(NAME, gpuNetwork, factors(gpuNetwork), tight.get());
        SensitivityAnalysisResult cpuResult = run("OpenLoadFlow", cpuNetwork, factors(cpuNetwork), tight.get());
        // L1-2-1's load flow fails in OLF (FAILURE, no values), L2-3-1's converges: the device must say the same
        List<SensitivityAnalysisResult.Status> statuses = cpuResult.getStateStatuses().stream()
                .map(SensitivityAnalysisResult.SensitivityStateStatus::getStatus).toList();
        assertEquals(statuses, gpuResult.getStateStatuses().stream().map(SensitivityAnalysisResult.SensitivityStateStatus::getStatus).toList());
        long solvedContingencies = statuses.stream().filter(status -> status == SensitivityAnalysisResult.Status.SUCCESS).count();
        assertTrue(solvedContingencies > 0, "a contingency's sensitivities are compared");
        Map<List<Integer>, SensitivityValue> gpu = byFactorAndContingency(gpuResult);
        Map<List<Integer>, SensitivityValue> cpu = byFactorAndContingency(cpuResult);
        assertEquals(cpu.keySet(), gpu.keySet());
        assertEquals(factors(cpuNetwork).size() * (1 + solvedContingencies), cpu.size(), "every factor in every state OLF solved");
        List<SensitivityFactor> factors = factors(cpuNetwork);
        for (Map.Entry<List<Integer>, SensitivityValue> entry : cpu.entrySet()) {
            SensitivityValue c = entry.getValue();
            SensitivityValue g = gpu.get(entry.getKey());
            assertClose(c.getValue(), g.getValue(), 1e-6, "sensitivity " + entry.getKey());
            assertClose(c.getFunctionReference(), g.getFunctionReference(), 1e-4, "function reference " + entry.getKey());
            if (factors.get(c.getFactorIndex()).getVariableId().equals("B8-G")) {
                assertEquals(0.0, c.getValue(), 0.0, "the condenser of bus 8 is PQ: " + entry.getKey());
                assertEquals(0.0, g.getValue(), 0.0, "the condenser of bus 8 is PQ on the device too: " + entry.getKey());
            }
        }
    }

    /** {@code actual} within {@code tolerance} of {@code expected}, relative beyond 1; NaN where OLF gives NaN. */
    private static void assertClose(double expected, double actual, double tolerance, String what) {
        if (Double.isNaN(expected)) {
            assertTrue(Double.isNaN(actual), what + ": NaN expected, was " + actual);
        } else {
            assertEquals(expected, actual, tolerance * (1 + Math.abs(expected)), what);
        }
    }
}
