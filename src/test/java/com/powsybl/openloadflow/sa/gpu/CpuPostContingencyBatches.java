/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcLoadFlowContext;
import com.powsybl.openloadflow.ac.AcLoadFlowResult;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor;
import com.powsybl.openloadflow.ac.solver.gpu.batch.BatchScenarioResult;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuQLimitType;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfContingency;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.NetworkState;
import com.powsybl.openloadflow.network.impl.PropagatedContingency;
import com.powsybl.openloadflow.network.util.ActivePowerDistribution;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * Test support: a {@link GpuPostContingencyBatch} holding what the device's batch holds — every contingency's converged
 * state in the base variable-row layout, its islanded buses and its reactive-limit decisions as the device encodes them —
 * made by OLF's own load flows on the CPU. It runs each contingency's load flow as the AC sensitivity analysis runs it
 * (the contingency applied, its loss distributed when the slack is, the context's load flow from the restored base),
 * reads the result back in the device's form, and restores the network. A sensitivity analysis bringing these states
 * must then compute exactly what OLF computes, which tests the injection and the handback without a device.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class CpuPostContingencyBatches {

    /** The batch, and how many reactive-limit entries the contingencies' load flows switched from the base: to PQ, and
     *  back to PV. */
    public record Made(GpuPostContingencyBatch batch, int switchedToPq, int switchedBackToPv) {
    }

    private CpuPostContingencyBatches() {
    }

    /**
     * Solves the contingencies (but {@code leftOut}, which the batch does not bring) from the base state the context
     * holds. With {@code baseDecisions} every contingency's decisions are the base's instead of its own, as if the
     * device's switches were lost: a control showing that the handback matters.
     */
    public static Made solve(LfNetwork lfNetwork, AcLoadFlowContext context, List<PropagatedContingency> contingencies,
                             LoadFlowParameters lfParameters, OpenLoadFlowParameters lfParametersExt, Set<String> leftOut,
                             boolean baseDecisions) {
        EquationSystem<AcVariableType, AcEquationType> es = context.getEquationSystem();
        Map<Variable<AcVariableType>, Integer> baseRows = new HashMap<>();
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            baseRows.put(v, v.getRow());
        }
        double[] baseState = es.getStateVector().get().clone();
        GpuAcData data = GpuAcDataExtractor.extract(lfNetwork, es);
        int[] busNum = data.reactiveLimits().busNum();
        BatchScenarioResult.ReactiveLimitDecisions base = decisions(lfNetwork, data);
        NetworkState networkState = NetworkState.save(lfNetwork);
        Map<String, double[]> states = new HashMap<>();
        Map<String, int[]> islanded = new HashMap<>();
        Map<String, BatchScenarioResult.ReactiveLimitDecisions> decisions = new HashMap<>();
        int switchedToPq = 0;
        int switchedBackToPv = 0;
        for (PropagatedContingency contingency : contingencies) {
            Optional<LfContingency> optionalLfContingency = contingency.toLfContingency(lfNetwork);
            if (optionalLfContingency.isEmpty() || leftOut.contains(contingency.getContingency().getId())) {
                continue;
            }
            LfContingency lfContingency = optionalLfContingency.get();
            lfContingency.apply(lfParameters.getBalanceType());
            if (lfParameters.isDistributedSlack() && Math.abs(lfContingency.getActivePowerLoss()) > 0) {
                ActivePowerDistribution.create(lfParameters.getBalanceType(), lfParametersExt.isLoadPowerFactorConstant(),
                        lfParametersExt.isUseActiveLimits()).run(lfNetwork.getSynchronousNetworks().getFirst(), lfContingency.getActivePowerLoss());
            }
            AcLoadFlowResult result = context.getParameters().getSolverFactory().createLoadFlowEngine(context).run();
            if (result.isSuccess()) {
                double[] state = new double[baseState.length];
                baseRows.forEach((v, row) -> state[row] = v.getType() == AcVariableType.BUS_V ? 1 : 0);   // the device's island
                double[] x = es.getStateVector().get();
                for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
                    state[baseRows.get(v)] = x[v.getRow()];
                }
                states.put(lfContingency.getId(), state);
                int[] disabled = lfContingency.getDisabledNetwork().getBuses().stream().mapToInt(LfBus::getNum).toArray();
                if (disabled.length > 0) {
                    islanded.put(lfContingency.getId(), disabled);
                }
                BatchScenarioResult.ReactiveLimitDecisions own = decisions(lfNetwork, data);
                for (int i = 0; i < busNum.length; i++) {
                    if (!lfNetwork.getBus(busNum[i]).isDisabled()) {
                        switchedToPq += base.finalState()[i] == GpuQLimitType.NONE && own.finalState()[i] != GpuQLimitType.NONE ? 1 : 0;
                        switchedBackToPv += base.finalState()[i] != GpuQLimitType.NONE && own.finalState()[i] == GpuQLimitType.NONE ? 1 : 0;
                    }
                }
                decisions.put(lfContingency.getId(), baseDecisions ? base : own);
            }
            networkState.restore();
        }
        es.getStateVector().set(baseState);                     // the context at the base state, as the device leaves it
        return new Made(GpuPostContingencyBatch.of(new PrecomputedBatch(states, islanded, Set.of(), baseRows, es, data, null,
                Map.of(), Map.of(), Map.of(), Map.of(), Map.of(), Map.of(), Map.of(), Map.of(), null, null, decisions)),
                switchedToPq, switchedBackToPv);
    }

    /** The network's PV/PQ states in the device's encoding: per entry its limit type (none for a bus holding its voltage,
     *  or PQ without a limit) and the target its switch froze on its row. */
    private static BatchScenarioResult.ReactiveLimitDecisions decisions(LfNetwork lfNetwork, GpuAcData data) {
        int[] busNum = data.reactiveLimits().busNum();
        int[] finalState = new int[busNum.length];
        double[] frozenRowTargetQ = new double[busNum.length];
        for (int i = 0; i < busNum.length; i++) {
            LfBus bus = lfNetwork.getBus(busNum[i]);
            finalState[i] = bus.isDisabled() || bus.isGeneratorVoltageControlEnabled() ? GpuQLimitType.NONE : GpuQLimitType.of(bus);
            frozenRowTargetQ[i] = finalState[i] != GpuQLimitType.NONE
                    ? bus.getGenerationTargetQ() - data.reactiveLimits().loadQ()[i] : Double.NaN;
        }
        return new BatchScenarioResult.ReactiveLimitDecisions(finalState, frozenRowTargetQ);
    }
}
