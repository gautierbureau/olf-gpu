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
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.math.matrix.MatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcLoadFlowContext;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.graph.GraphConnectivityFactory;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.impl.PropagatedContingency;
import com.powsybl.openloadflow.sa.gpu.CpuPostContingencyBatches;
import com.powsybl.openloadflow.sa.gpu.GpuPostContingencyBatch;
import com.powsybl.openloadflow.sensi.AcSensitivityAnalysis;
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
import com.powsybl.sensitivity.SensitivityVariableType;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * {@link GpuBatchedAcSensitivityAnalysis}: a contingency brought from the batch (its converged state set whole, its
 * reactive-limit switches handed to the network) gives exactly the sensitivities OLF computes after its own load flow.
 *
 * <p>Without a device the batch is made by OLF on the CPU ({@link CpuPostContingencyBatches}: the same states and
 * decisions, in the device's layout and encoding), which tests everything but the device's numerics: the injection,
 * the handback, the hooks in OLF's analysis and that skipping the load flow loses nothing the sensitivities read. With
 * a device the GPU provider runs the real batch.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuBatchedAcSensitivityAnalysisTest {

    /**
     * IEEE 14 with the condenser of bus 8 capped at 17 MVar: it needs 17.6 MVar to hold its voltage, so the base load
     * flow switches it to PQ at that limit, and an outage that relieves it switches it back to PV (T4-7-1: 13.6 MVar,
     * L7-9-1: 14.2). Other outages switch generators to PQ at their stock limits (L1-2-1: B2-G, B3-G and B6-G; L2-3-1:
     * B3-G; T5-6-1: B6-G), L7-8-1 islands bus 8 and L3-4-1 switches nothing at all.
     */
    private static Network network() {
        Network network = IeeeCdfNetworkFactory.create14();
        network.getGenerator("B8-G").newMinMaxReactiveLimits().setMinQ(-6).setMaxQ(17).add();
        return network;
    }

    private static List<Contingency> contingencies() {
        return List.of("L1-2-1", "L2-3-1", "T5-6-1", "T4-7-1", "L7-9-1", "L7-8-1", "L3-4-1").stream()
                .map(id -> new Contingency(id, new BranchContingency(id)))
                .toList();
    }

    /** Active flows per generator injections, reactive flows and bus voltages per generator target voltages: the
     *  reactive-limit states shape the last two (a PQ generator's target voltage moves nothing). */
    private static List<SensitivityFactor> factors(Network network) {
        List<SensitivityFactor> factors = new ArrayList<>();
        for (Line line : network.getLines()) {
            for (String generator : List.of("B2-G", "B3-G")) {
                factors.add(new SensitivityFactor(SensitivityFunctionType.BRANCH_ACTIVE_POWER_1, line.getId(),
                        SensitivityVariableType.INJECTION_ACTIVE_POWER, generator, false, ContingencyContext.all()));
            }
            factors.add(new SensitivityFactor(SensitivityFunctionType.BRANCH_REACTIVE_POWER_1, line.getId(),
                    SensitivityVariableType.BUS_TARGET_VOLTAGE, "B6-G", false, ContingencyContext.all()));
        }
        for (String bus : List.of("B4", "B9", "B13")) {
            for (String generator : List.of("B2-G", "B3-G", "B6-G", "B8-G")) {
                factors.add(new SensitivityFactor(SensitivityFunctionType.BUS_VOLTAGE, bus,
                        SensitivityVariableType.BUS_TARGET_VOLTAGE, generator, false, ContingencyContext.all()));
            }
        }
        return factors;
    }

    private static SensitivityAnalysisParameters parameters() {
        SensitivityAnalysisParameters parameters = new SensitivityAnalysisParameters();
        OpenLoadFlowParameters.create(parameters.getLoadFlowParameters()).setNewtonRaphsonConvEpsPerEq(1e-10);
        return parameters;
    }

    /** One run's values by (factor, contingency), and its contingency statuses. */
    private record Run(Map<List<Integer>, SensitivityValue> values, List<SensitivityAnalysisResult.SensitivityStateStatus> statuses) {
    }

    private static Run run(SensitivityAnalysisProvider provider, Network network, SensitivityAnalysisParameters parameters) {
        List<Contingency> contingencies = contingencies();
        SensitivityResultModelWriter writer = new SensitivityResultModelWriter(contingencies, List.of());
        provider.run(network, network.getVariantManager().getWorkingVariantId(), new SensitivityFactorModelReader(factors(network), network),
                writer, new SensitivityAnalysisRunParameters().setParameters(parameters).setContingencies(contingencies)).join();
        Map<List<Integer>, SensitivityValue> values = new HashMap<>();
        for (SensitivityValue value : writer.getValues()) {
            values.put(List.of(value.getFactorIndex(), value.getContingencyIndex()), value);
        }
        return new Run(values, writer.getStateStatuses());
    }

    /** OLF's provider whose AC analysis brings the contingencies from a batch OLF solves on the CPU. */
    private static final class CpuBatchedProvider extends OpenSensitivityAnalysisProvider {

        private final Set<String> leftOut;
        private final boolean baseDecisions;
        private CpuPostContingencyBatches.Made made;

        CpuBatchedProvider(Set<String> leftOut, boolean baseDecisions) {
            this.leftOut = leftOut;
            this.baseDecisions = baseDecisions;
        }

        @Override
        protected AcSensitivityAnalysis createAcSensitivityAnalysis(MatrixFactory matrixFactory,
                                                                    GraphConnectivityFactory<LfBus, LfBranch> connectivityFactory,
                                                                    SensitivityAnalysisParameters sensitivityAnalysisParameters) {
            return new GpuBatchedAcSensitivityAnalysis(matrixFactory, connectivityFactory, sensitivityAnalysisParameters) {
                @Override
                protected GpuPostContingencyBatch solveBatch(Network network, LfNetwork lfNetwork, AcLoadFlowContext context,
                                                             List<PropagatedContingency> contingencies, LoadFlowParameters lfParameters,
                                                             OpenLoadFlowParameters contingencyLfParametersExt) {
                    made = CpuPostContingencyBatches.solve(lfNetwork, context, contingencies, lfParameters, contingencyLfParametersExt,
                            leftOut, baseDecisions);
                    return made.batch();
                }
            };
        }
    }

    private static void assertSameSensitivities(Run expected, Run actual, double tolerance) {
        assertEquals(expected.values().keySet(), actual.values().keySet());
        for (Map.Entry<List<Integer>, SensitivityValue> entry : expected.values().entrySet()) {
            SensitivityValue e = entry.getValue();
            SensitivityValue a = actual.values().get(entry.getKey());
            assertClose(e.getValue(), a.getValue(), tolerance, "sensitivity " + entry.getKey());
            assertClose(e.getFunctionReference(), a.getFunctionReference(), tolerance, "function reference " + entry.getKey());
        }
        assertEquals(expected.statuses().size(), actual.statuses().size());
        for (int i = 0; i < expected.statuses().size(); i++) {
            assertEquals(expected.statuses().get(i).getStatus(), actual.statuses().get(i).getStatus(),
                    "status of " + expected.statuses().get(i).getState());
        }
    }

    /** Within {@code tolerance} relative, or both NaN: OLF's value for a factor the contingency islands (L7-8-1's bus 8). */
    private static void assertClose(double expected, double actual, double tolerance, String what) {
        if (Double.isNaN(expected)) {
            assertTrue(Double.isNaN(actual), what + ": NaN expected, was " + actual);
        } else {
            assertEquals(expected, actual, tolerance * (1 + Math.abs(expected)), what);
        }
    }

    @Test
    void contingenciesBroughtFromTheBatchHaveOlfsSensitivities() {
        Run olf = run(new OpenSensitivityAnalysisProvider(), network(), parameters());
        CpuBatchedProvider provider = new CpuBatchedProvider(Set.of(), false);
        long before = GpuBatchedAcSensitivityAnalysis.BATCHED_CONTINGENCIES.sum();
        Run batched = run(provider, network(), parameters());
        assertEquals(contingencies().size(), GpuBatchedAcSensitivityAnalysis.BATCHED_CONTINGENCIES.sum() - before,
                "every contingency brought from the batch");
        assertTrue(provider.made.switchedToPq() > 0, "a contingency switches a generator to PQ");
        assertTrue(provider.made.switchedBackToPv() > 0, "a contingency switches the condenser back to PV");
        assertSameSensitivities(olf, batched, 1e-9);
    }

    @Test
    void aContingencyLeftOutOfTheBatchRunsItsOwnLoadFlow() {
        Run olf = run(new OpenSensitivityAnalysisProvider(), network(), parameters());
        long before = GpuBatchedAcSensitivityAnalysis.BATCHED_CONTINGENCIES.sum();
        Run batched = run(new CpuBatchedProvider(Set.of("L1-2-1"), false), network(), parameters());
        assertEquals(contingencies().size() - 1L, GpuBatchedAcSensitivityAnalysis.BATCHED_CONTINGENCIES.sum() - before);
        assertSameSensitivities(olf, batched, 1e-9);
    }

    /** The control: the same states with the base's PV/PQ states instead of their own give other sensitivities, so the
     *  agreement above holds because the switches are handed back. */
    @Test
    void withoutItsSwitchesAContingencyStateGivesOtherSensitivities() {
        Run olf = run(new OpenSensitivityAnalysisProvider(), network(), parameters());
        Run batched = run(new CpuBatchedProvider(Set.of(), true), network(), parameters());
        int l12 = contingencies().stream().map(Contingency::getId).toList().indexOf("L1-2-1");
        double largest = 0;
        for (Map.Entry<List<Integer>, SensitivityValue> entry : olf.values().entrySet()) {
            if (entry.getKey().get(1) == l12 && !Double.isNaN(entry.getValue().getValue())) {
                largest = Math.max(largest, Math.abs(entry.getValue().getValue() - batched.values().get(entry.getKey()).getValue()));
            }
        }
        assertTrue(largest > 1e-3, "L1-2-1's sensitivities without its switches to PQ: largest difference " + largest);
    }

    /** The device batch: every contingency converged on the device and brought from there, the sensitivities OLF's on
     *  the CPU within the load flows' tolerance. */
    @Test
    void deviceBatchGivesOlfsSensitivities() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Run olf = run(new OpenSensitivityAnalysisProvider(), network(), parameters());
        long before = GpuBatchedAcSensitivityAnalysis.BATCHED_CONTINGENCIES.sum();
        Run gpu = run(new GpuSensitivityAnalysisProvider(), network(), parameters());
        assertEquals(contingencies().size(), GpuBatchedAcSensitivityAnalysis.BATCHED_CONTINGENCIES.sum() - before,
                "every contingency brought from the device batch");
        assertSameSensitivities(olf, gpu, 1e-6);
    }
}
