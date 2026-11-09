/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.math.matrix.DenseMatrix;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcLoadFlowContext;
import com.powsybl.openloadflow.ac.AcLoadFlowResult;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor;
import com.powsybl.openloadflow.ac.solver.gpu.batch.BatchScenarioResult;
import com.powsybl.openloadflow.ac.solver.gpu.batch.SensitivityBasisRequest;
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
     * device's switches were lost: a control showing that the handback matters. With a {@code sensitivityBasis}, each
     * converged contingency also has that basis solved on OLF's Jacobian at its state, laid out as the device lays it
     * out ({@link SensitivityBasisRequest#resultLength()} values, column-major).
     */
    @SuppressWarnings("java:S107")
    public static Made solve(LfNetwork lfNetwork, AcLoadFlowContext context, List<PropagatedContingency> contingencies,
                             LoadFlowParameters lfParameters, OpenLoadFlowParameters lfParametersExt, Set<String> leftOut,
                             boolean baseDecisions, SensitivityBasisRequest sensitivityBasis) {
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
        Map<String, double[]> sensitivityBases = new HashMap<>();
        int busCount = lfNetwork.getBuses().size();
        int[] busPhiRow = new int[busCount];
        int[] busVRow = new int[busCount];
        java.util.Arrays.fill(busPhiRow, -1);
        java.util.Arrays.fill(busVRow, -1);
        baseRows.forEach((v, row) -> {
            if (v.getType() == AcVariableType.BUS_PHI && v.getElementNum() < busCount) {
                busPhiRow[v.getElementNum()] = row;
            } else if (v.getType() == AcVariableType.BUS_V && v.getElementNum() < busCount) {
                busVRow[v.getElementNum()] = row;
            }
        });
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
                if (sensitivityBasis != null) {
                    sensitivityBases.put(lfContingency.getId(), solveBasis(context, lfNetwork, baseRows, busPhiRow, busVRow,
                            sensitivityBasis, sensitivityBasis.participation(lfContingency.getId())));
                }
            }
            networkState.restore();
        }
        es.getStateVector().set(baseState);                     // the context at the base state, as the device leaves it
        return new Made(GpuPostContingencyBatch.of(new PrecomputedBatch(states, islanded, Set.of(), baseRows, es, data, null,
                Map.of(), Map.of(), Map.of(), Map.of(), Map.of(), decisions, sensitivityBases)), switchedToPq, switchedBackToPv);
    }

    /**
     * The device's sensitivity basis step, on the CPU: each basis column (stable rows) placed on the equation the device
     * holds on that row (a bus's P row holds its active-power balance, or its angle reference on the slack; its BUS_V row
     * its voltage target while it holds one, its reactive balance otherwise), solved on OLF's Jacobian at the converged
     * state, read at the gather rows.
     */
    private static double[] solveBasis(AcLoadFlowContext context, LfNetwork lfNetwork, Map<Variable<AcVariableType>, Integer> baseRows,
                                       int[] busPhiRow, int[] busVRow, SensitivityBasisRequest basis,
                                       SensitivityBasisRequest.SparseColumn participation) {
        EquationSystem<AcVariableType, AcEquationType> es = context.getEquationSystem();
        int columnCount = es.getIndex().getColumnCount();
        int[] equationOfStableRow = new int[baseRows.size()];
        java.util.Arrays.fill(equationOfStableRow, -1);
        for (int num = 0; num < busPhiRow.length; num++) {
            if (busPhiRow[num] >= 0) {
                equationOfStableRow[busPhiRow[num]] = firstActiveColumn(es, num, AcEquationType.BUS_TARGET_P, AcEquationType.BUS_TARGET_PHI);
            }
            if (busVRow[num] >= 0) {
                equationOfStableRow[busVRow[num]] = firstActiveColumn(es, num, AcEquationType.BUS_TARGET_V, AcEquationType.BUS_TARGET_Q);
            }
        }
        DenseMatrix rhs = new DenseMatrix(columnCount, basis.columnCount());
        int[] unitRows = basis.unitRows();
        for (int j = 0; j < unitRows.length; j++) {
            place(rhs, equationOfStableRow, unitRows[j], j, 1);
        }
        for (int w = 0; w < basis.weightedColumnCount(); w++) {
            for (int k = basis.weightedStart()[w]; k < basis.weightedStart()[w + 1]; k++) {
                place(rhs, equationOfStableRow, basis.weightedRows()[k], unitRows.length + w, basis.weightedValues()[k]);
            }
        }
        if (basis.hasParticipation()) {
            for (int k = 0; k < participation.rows().length; k++) {
                place(rhs, equationOfStableRow, participation.rows()[k], basis.participationColumn(), participation.values()[k]);
            }
        }
        context.getJacobianMatrix().solveTransposed(rhs);
        int[] currentRowOfBase = new int[baseRows.size()];
        java.util.Arrays.fill(currentRowOfBase, -1);
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            Integer baseRow = baseRows.get(v);
            if (baseRow != null) {
                currentRowOfBase[baseRow] = v.getRow();
            }
        }
        int[] gatherRows = basis.gatherRows();
        double[] u = new double[basis.resultLength()];
        for (int j = 0; j < basis.columnCount(); j++) {
            for (int f = 0; f < gatherRows.length; f++) {
                int row = currentRowOfBase[gatherRows[f]];
                u[j * gatherRows.length + f] = row >= 0 ? rhs.get(row, j) : 0;
            }
        }
        return u;
    }

    private static int firstActiveColumn(EquationSystem<AcVariableType, AcEquationType> es, int num, AcEquationType... types) {
        for (AcEquationType type : types) {
            int column = es.getEquation(num, type).map(e -> e.getColumn()).orElse(-1);
            if (column >= 0) {
                return column;
            }
        }
        return -1;
    }

    private static void place(DenseMatrix rhs, int[] equationOfStableRow, int stableRow, int column, double value) {
        int equation = stableRow < equationOfStableRow.length ? equationOfStableRow[stableRow] : -1;
        if (equation >= 0) {
            rhs.add(equation, column, value);
        }
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
