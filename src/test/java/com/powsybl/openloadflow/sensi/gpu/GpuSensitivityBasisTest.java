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
import com.powsybl.contingency.GeneratorContingency;
import com.powsybl.contingency.LoadContingency;
import com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory;
import com.powsybl.iidm.network.Line;
import com.powsybl.iidm.network.Network;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.sensi.OpenSensitivityAnalysisProvider;
import com.powsybl.sensitivity.SensitivityAnalysisParameters;
import com.powsybl.sensitivity.SensitivityAnalysisProvider;
import com.powsybl.sensitivity.SensitivityAnalysisResult;
import com.powsybl.sensitivity.SensitivityAnalysisRunParameters;
import com.powsybl.sensitivity.SensitivityFactor;
import com.powsybl.sensitivity.SensitivityFactorModelReader;
import com.powsybl.sensitivity.SensitivityFunctionType;
import com.powsybl.sensitivity.SensitivityResultModelWriter;
import com.powsybl.sensitivity.SensitivityValue;
import com.powsybl.sensitivity.SensitivityVariableSet;
import com.powsybl.sensitivity.SensitivityVariableType;
import com.powsybl.sensitivity.WeightedSensitivityVariable;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The sensitivities of contingencies brought from the batch, solved from the device's sensitivity basis
 * ({@link GpuSensitivityBasis}): OLF's own right-hand sides decomposed exactly on the basis solved at the converged
 * state, the solved states rebuilt from it, OLF's sensitivities to 1e-9.
 *
 * <p>Without a device the basis is solved by OLF on the CPU, as the device lays it out ({@link CpuPostContingencyBatches}):
 * that tests the basis, the participation, the decomposition and the rebuilt states against OLF's own run; with a
 * device the GPU provider solves it in the batch.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuSensitivityBasisTest {

    private static final String GLSK = "LOADS";

    /** IEEE 14 with the condenser of bus 8 capped at 17 MVar (PQ in the base, back to PV after T4-7-1). */
    private static Network network() {
        Network network = IeeeCdfNetworkFactory.create14();
        network.getGenerator("B8-G").newMinMaxReactiveLimits().setMinQ(-6).setMaxQ(17).add();
        return network;
    }

    /** Line outages switching generators to PQ (L1-2-1, L2-3-1), one switching the condenser back (T4-7-1), one islanding
     *  bus 8 (L7-8-1), one switching nothing (L3-4-1), and a generator outage, which renormalizes the participation. */
    private static List<Contingency> contingencies(Contingency... more) {
        List<Contingency> contingencies = new ArrayList<>(Stream.of("L1-2-1", "L2-3-1", "T4-7-1", "L7-8-1", "L3-4-1")
                .map(id -> new Contingency(id, new BranchContingency(id)))
                .toList());
        contingencies.add(new Contingency("B3-G", new GeneratorContingency("B3-G")));
        contingencies.addAll(List.of(more));
        return contingencies;
    }

    private static List<SensitivityVariableSet> variableSets() {
        return List.of(new SensitivityVariableSet(GLSK, List.of(new WeightedSensitivityVariable("B4-L", 2),
                new WeightedSensitivityVariable("B5-L", 1), new WeightedSensitivityVariable("B9-L", 3))));
    }

    /**
     * Active flows per two generators' and a load's injections and per the GLSK, currents per an injection, reactive flows
     * per a target voltage, and bus voltages per three target voltages: unit, weighted and participation columns, and
     * right-hand sides on active-power, reactive and voltage-target rows.
     */
    private static List<SensitivityFactor> factors(Network network) {
        List<SensitivityFactor> factors = new ArrayList<>();
        for (Line line : network.getLines()) {
            for (String injection : List.of("B2-G", "B6-G", "B9-L")) {
                factors.add(new SensitivityFactor(SensitivityFunctionType.BRANCH_ACTIVE_POWER_1, line.getId(),
                        SensitivityVariableType.INJECTION_ACTIVE_POWER, injection, false, ContingencyContext.all()));
            }
            factors.add(new SensitivityFactor(SensitivityFunctionType.BRANCH_ACTIVE_POWER_1, line.getId(),
                    SensitivityVariableType.INJECTION_ACTIVE_POWER, GLSK, true, ContingencyContext.all()));
            factors.add(new SensitivityFactor(SensitivityFunctionType.BRANCH_REACTIVE_POWER_2, line.getId(),
                    SensitivityVariableType.BUS_TARGET_VOLTAGE, "B3-G", false, ContingencyContext.all()));
        }
        for (String line : List.of("L2-4-1", "L6-13-1", "L9-14-1")) {
            factors.add(new SensitivityFactor(SensitivityFunctionType.BRANCH_CURRENT_1, line,
                    SensitivityVariableType.INJECTION_ACTIVE_POWER, "B2-G", false, ContingencyContext.all()));
        }
        for (String bus : List.of("B4", "B9", "B13")) {
            for (String generator : List.of("B2-G", "B6-G", "B8-G")) {
                factors.add(new SensitivityFactor(SensitivityFunctionType.BUS_VOLTAGE, bus,
                        SensitivityVariableType.BUS_TARGET_VOLTAGE, generator, false, ContingencyContext.all()));
            }
        }
        return factors;
    }

    private static SensitivityAnalysisParameters parameters(boolean distributedSlack) {
        SensitivityAnalysisParameters parameters = new SensitivityAnalysisParameters();
        parameters.getLoadFlowParameters().setDistributedSlack(distributedSlack);
        OpenLoadFlowParameters.create(parameters.getLoadFlowParameters()).setNewtonRaphsonConvEpsPerEq(1e-10);
        return parameters;
    }

    /** A run's values by (factor, contingency), and the statuses of its contingencies. */
    private record Run(Map<List<Integer>, SensitivityValue> values, List<SensitivityAnalysisResult.Status> statuses) {

        /** The contingencies whose load flow converged, the ones OLF computes sensitivities for. */
        long solved() {
            return statuses.stream().filter(status -> status == SensitivityAnalysisResult.Status.SUCCESS).count();
        }
    }

    private static Run run(SensitivityAnalysisProvider provider, Network network, List<Contingency> contingencies,
                           SensitivityAnalysisParameters parameters) {
        SensitivityResultModelWriter writer = new SensitivityResultModelWriter(contingencies, List.of());
        provider.run(network, network.getVariantManager().getWorkingVariantId(), new SensitivityFactorModelReader(factors(network), network),
                writer, new SensitivityAnalysisRunParameters().setParameters(parameters).setContingencies(contingencies)
                        .setVariableSets(variableSets())).join();
        Map<List<Integer>, SensitivityValue> values = new HashMap<>();
        for (SensitivityValue value : writer.getValues()) {
            values.put(List.of(value.getFactorIndex(), value.getContingencyIndex()), value);
        }
        return new Run(values, writer.getStateStatuses().stream().map(SensitivityAnalysisResult.SensitivityStateStatus::getStatus).toList());
    }

    private static void assertSame(Run expected, Run actual, double tolerance) {
        assertEquals(expected.statuses(), actual.statuses());
        assertEquals(expected.values().keySet(), actual.values().keySet());
        for (Map.Entry<List<Integer>, SensitivityValue> entry : expected.values().entrySet()) {
            assertClose(entry.getValue().getValue(), actual.values().get(entry.getKey()).getValue(), tolerance, "sensitivity " + entry.getKey());
            assertClose(entry.getValue().getFunctionReference(), actual.values().get(entry.getKey()).getFunctionReference(), tolerance,
                    "function reference " + entry.getKey());
        }
    }

    private static void assertClose(double expected, double actual, double tolerance, String what) {
        if (Double.isNaN(expected)) {
            assertTrue(Double.isNaN(actual), what + ": NaN expected, was " + actual);
        } else {
            assertEquals(expected, actual, tolerance * (1 + Math.abs(expected)), what);
        }
    }

    /** Every contingency brought from the batch, and its sensitivities solved from the basis. */
    @Test
    void contingencySensitivitiesSolvedFromTheBasisAreOlfs() {
        List<Contingency> contingencies = contingencies();
        Run olf = run(new OpenSensitivityAnalysisProvider(), network(), contingencies, parameters(true));
        long brought = GpuBatchedAcSensitivityAnalysis.BATCHED_CONTINGENCIES.sum();
        long solved = GpuBatchedAcSensitivityAnalysis.BASIS_SOLVED_CONTINGENCIES.sum();
        Run batched = run(new CpuBatchedProvider(Set.of(), false, true), network(), contingencies, parameters(true));
        assertEquals(contingencies.size(), olf.solved(), "OLF solves every contingency");
        assertEquals(contingencies.size(), GpuBatchedAcSensitivityAnalysis.BATCHED_CONTINGENCIES.sum() - brought);
        assertEquals(olf.solved(), GpuBatchedAcSensitivityAnalysis.BASIS_SOLVED_CONTINGENCIES.sum() - solved,
                "every contingency's sensitivities solved from the basis");
        assertSame(olf, batched, 1e-9);
    }

    /** Without distributed slack the basis has no participation column, and the injections' right-hand sides are units. */
    @Test
    void withoutDistributedSlackTheBasisIsUnitsAndWeights() {
        List<Contingency> contingencies = contingencies();
        Run olf = run(new OpenSensitivityAnalysisProvider(), network(), contingencies, parameters(false));
        long solved = GpuBatchedAcSensitivityAnalysis.BASIS_SOLVED_CONTINGENCIES.sum();
        Run batched = run(new CpuBatchedProvider(Set.of(), false, true), network(), contingencies, parameters(false));
        assertEquals(olf.solved(), GpuBatchedAcSensitivityAnalysis.BASIS_SOLVED_CONTINGENCIES.sum() - solved,
                "every contingency OLF solves (L1-2-1's load flow fails without the slack distributed) from the basis");
        assertSame(olf, batched, 1e-9);
    }

    /** A contingency that loses an element of the GLSK has OLF rescale it: that right-hand side leaves the basis and the
     *  contingency's sensitivities are solved on the CPU, the others' from the basis. */
    @Test
    void aGlskLosingAnElementIsSolvedOnTheCpu() {
        List<Contingency> contingencies = contingencies(new Contingency("B9-L", new LoadContingency("B9-L")));
        Run olf = run(new OpenSensitivityAnalysisProvider(), network(), contingencies, parameters(true));
        long solved = GpuBatchedAcSensitivityAnalysis.BASIS_SOLVED_CONTINGENCIES.sum();
        Run batched = run(new CpuBatchedProvider(Set.of(), false, true), network(), contingencies, parameters(true));
        assertEquals(olf.solved() - 1, GpuBatchedAcSensitivityAnalysis.BASIS_SOLVED_CONTINGENCIES.sum() - solved,
                "all but the load outage solved from the basis");
        assertSame(olf, batched, 1e-9);
    }

    /** The device: the batch solves the basis at every converged state, and every contingency's sensitivities come from it. */
    @Test
    void deviceBasisGivesOlfsSensitivities() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        List<Contingency> contingencies = contingencies();
        Run olf = run(new OpenSensitivityAnalysisProvider(), network(), contingencies, parameters(true));
        long solved = GpuBatchedAcSensitivityAnalysis.BASIS_SOLVED_CONTINGENCIES.sum();
        Run gpu = run(new GpuSensitivityAnalysisProvider(), network(), contingencies, parameters(true));
        assertEquals(olf.solved(), GpuBatchedAcSensitivityAnalysis.BASIS_SOLVED_CONTINGENCIES.sum() - solved,
                "every contingency's sensitivities solved from the device's basis");
        assertSame(olf, gpu, 1e-6);
    }
}
