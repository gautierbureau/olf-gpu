/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sensi.gpu;

import com.powsybl.math.matrix.DenseMatrix;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcElementMapper;
import com.powsybl.openloadflow.ac.solver.gpu.batch.SensitivityBasisRequest;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuSensitivityBasisKind;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuSensitivityFunction;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.impl.LfLegBranch;
import com.powsybl.openloadflow.sensi.GpuSensitivityFactorGroups;
import com.powsybl.openloadflow.util.Derivable;
import com.powsybl.sensitivity.SensitivityFunctionType;
import com.powsybl.sensitivity.SensitivityVariableType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * The right-hand sides a batched contingency's sensitivities are solved on by the device, and OLF's own right-hand side
 * of that contingency rebuilt from their solutions.
 *
 * <p>OLF writes the right-hand side of a factor group itself, per contingency, from that contingency's equation system
 * and slack participation ({@code fillRhs}): which equations are active, the participation renormalized over what the
 * contingency leaves, a GLSK rescaled when it disconnects an element. The device cannot know that before the batch, so
 * it solves a BASIS instead, at the scenario's converged state: a unit column per equation a group writes with a
 * coefficient of its own, a weighted column per multi-variable group, and the contingency's participation column, which
 * OLF computes the same way here ({@code getParticipatingElements} on the contingency's slack component). For a
 * contingency, OLF's right-hand side is then decomposed EXACTLY on that basis, group by group (the weighted and
 * participation coefficients by least squares on the rows the group writes no coefficient of its own, the own
 * coefficients by difference), and checked: a residual above a relative 1e-10 anywhere means the right-hand side left
 * the basis (a rescaled GLSK, an equation the basis has no row for), and the contingency is solved on the CPU instead.
 * The solved states are the basis solutions combined with those coefficients, at the rows the factors' functions read,
 * which is all OLF reads of them ({@code calculateSensi}).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuSensitivityBasis {

    private static final double RELATIVE_TOLERANCE = 1e-10;

    /** Per group (by its right-hand side column): its own rows, their kinds (GpuSensitivityBasisKind), their unit
     *  columns, its weighted column, whether participation enters it. */
    private record GroupColumns(int[] unitRows, int[] unitKinds, int[] unitColumns, int weightedColumn, Map<Integer, Double> weights,
                                boolean participation) {
    }

    private final GroupColumns[] groupColumns;
    private final SensitivityBasisRequest request;
    private final int[] busPhiRow;
    private final int[] busVRow;
    private final Map<String, Map<Integer, Double>> participationRows;

    private GpuSensitivityBasis(GroupColumns[] groupColumns, SensitivityBasisRequest request, int[] busPhiRow, int[] busVRow,
                                Map<String, Map<Integer, Double>> participationRows) {
        this.groupColumns = groupColumns;
        this.request = request;
        this.busPhiRow = busPhiRow;
        this.busVRow = busVRow;
        this.participationRows = participationRows;
    }

    /** The basis, or why there is none. */
    record Built(GpuSensitivityBasis basis, String notBuiltReason) {
    }

    /**
     * The basis of a run's factor groups, from the base state the equation system holds: {@code participationByContingency}
     * OLF's participation of each contingency (bus to minus its factor, as {@code computeLfContingency} computes it), null
     * when the slack is not distributed; {@code functions} the factors' functions the device contracts the basis into, null
     * when the host contracts the basis solutions itself.
     */
    @SuppressWarnings("java:S107")
    static Built build(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                       List<GpuSensitivityFactorGroups.Group> groups, Set<GpuSensitivityFactorGroups.VariableKey> functionVariables,
                       List<? extends Derivable<AcVariableType>> functionTerms, List<GpuSensitivityFactorGroups.FactorFunction> functions,
                       Map<String, Map<LfBus, Double>> participationByContingency) {
        if (functionVariables == null) {
            return new Built(null, "a factor's function is not a branch flow or current, a bus voltage or a bus reactive injection");
        }
        for (GpuSensitivityFactorGroups.Group group : groups) {
            if (!group.supported()) {
                return new Built(null, "factors of variable type " + group.variableType() + " are not solved on the device");
            }
        }
        // base variable rows by (type, element)
        List<Variable<AcVariableType>> variables = es.getIndex().getSortedVariablesToFind();
        int n = variables.size();
        int busCount = network.getBuses().size();
        int[] busPhiRow = new int[busCount];
        int[] busVRow = new int[busCount];
        Arrays.fill(busPhiRow, -1);
        Arrays.fill(busVRow, -1);
        Map<GpuSensitivityFactorGroups.VariableKey, Integer> rowOfVariable = new HashMap<>();
        for (Variable<AcVariableType> v : variables) {
            rowOfVariable.put(new GpuSensitivityFactorGroups.VariableKey(v.getType(), v.getElementNum()), v.getRow());
            if (v.getType() == AcVariableType.BUS_PHI && v.getElementNum() < busCount) {
                busPhiRow[v.getElementNum()] = v.getRow();
            } else if (v.getType() == AcVariableType.BUS_V && v.getElementNum() < busCount) {
                busVRow[v.getElementNum()] = v.getRow();
            }
        }
        // the rows the factors' functions read, checked against what they really read: a probe NaN everywhere else
        int[] gatherRows = functionVariables.stream().map(rowOfVariable::get).filter(Objects::nonNull).mapToInt(Integer::intValue).toArray();
        DenseMatrix probe = new DenseMatrix(n, 1);
        for (int r = 0; r < n; r++) {
            probe.set(r, 0, Double.NaN);
        }
        for (int r : gatherRows) {
            probe.set(r, 0, 0);
        }
        for (Derivable<AcVariableType> term : functionTerms) {
            if (term.isActive() && !Double.isFinite(term.calculateSensi(probe, 0))) {
                return new Built(null, "a factor's function reads states outside the rows derived from its type");
            }
        }
        // the unit columns, the weighted columns, then the participation column
        Map<Integer, Integer> unitColumnOfRow = new LinkedHashMap<>();
        GroupColumns[] groupColumns = new GroupColumns[groups.size()];
        List<Map<Integer, Double>> weightedColumns = new ArrayList<>();
        LfBus slackBus = network.getBuses().stream().filter(LfBus::isSlack).findFirst().orElse(null);
        for (GpuSensitivityFactorGroups.Group group : groups) {
            List<Integer> rows = new ArrayList<>();
            List<Integer> kinds = new ArrayList<>();
            for (GpuSensitivityFactorGroups.EquationKey key : group.units()) {
                int row = stableRow(key, busPhiRow, busVRow);
                if (row == UNKNOWN) {
                    return new Built(null, "factors writing " + key.type() + " right-hand sides are not solved on the device");
                }
                if (row >= 0 && !rows.contains(row)) {
                    rows.add(row);
                    kinds.add(kindOf(key.type()));
                    unitColumnOfRow.putIfAbsent(row, unitColumnOfRow.size());
                }
            }
            Map<Integer, Double> weights = new LinkedHashMap<>();
            for (Map.Entry<GpuSensitivityFactorGroups.EquationKey, Double> e : group.weights().entrySet()) {
                int row = stableRow(e.getKey(), busPhiRow, busVRow);
                if (row == UNKNOWN) {
                    return new Built(null, "factors writing " + e.getKey().type() + " right-hand sides are not solved on the device");
                }
                if (row >= 0 && (slackBus == null || e.getKey().elementNum() != slackBus.getNum())) {
                    weights.merge(row, e.getValue(), Double::sum);   // addBusInjection writes nothing on the slack bus
                }
            }
            int weightedColumn = -1;
            if (!weights.isEmpty()) {
                weightedColumn = weightedColumns.size();     // offset by the unit count below
                weightedColumns.add(weights);
            }
            groupColumns[group.index()] = new GroupColumns(rows.stream().mapToInt(Integer::intValue).toArray(),
                    kinds.stream().mapToInt(Integer::intValue).toArray(), null,
                    weightedColumn, weights, group.participation() && participationByContingency != null);
        }
        int nUnits = unitColumnOfRow.size();
        for (int g = 0; g < groupColumns.length; g++) {
            GroupColumns c = groupColumns[g];
            int[] unitColumns = Arrays.stream(c.unitRows()).map(unitColumnOfRow::get).toArray();
            groupColumns[g] = new GroupColumns(c.unitRows(), c.unitKinds(), unitColumns,
                    c.weightedColumn() >= 0 ? nUnits + c.weightedColumn() : -1, c.weights(), c.participation());
        }
        int[] weightedStart = new int[weightedColumns.size() + 1];
        List<Integer> weightedRowList = new ArrayList<>();
        List<Double> weightedValueList = new ArrayList<>();
        for (int w = 0; w < weightedColumns.size(); w++) {
            weightedColumns.get(w).forEach((row, value) -> {
                weightedRowList.add(row);
                weightedValueList.add(value);
            });
            weightedStart[w + 1] = weightedRowList.size();
        }
        Map<String, SensitivityBasisRequest.SparseColumn> participationColumns = null;
        Map<String, Map<Integer, Double>> participationRows = Map.of();
        if (participationByContingency != null) {
            participationColumns = new HashMap<>();
            participationRows = new HashMap<>();
            for (Map.Entry<String, Map<LfBus, Double>> e : participationByContingency.entrySet()) {
                Map<Integer, Double> column = participationColumn(e.getValue(), busPhiRow);
                participationRows.put(e.getKey(), column);
                participationColumns.put(e.getKey(), new SensitivityBasisRequest.SparseColumn(
                        column.keySet().stream().mapToInt(Integer::intValue).toArray(),
                        column.values().stream().mapToDouble(Double::doubleValue).toArray()));
            }
        }
        int[] unitRows = unitColumnOfRow.keySet().stream().mapToInt(Integer::intValue).toArray();
        int participationColumn = participationColumns != null ? nUnits + weightedColumns.size() : -1;
        SensitivityBasisRequest.Contraction contraction = contraction(network, groups, groupColumns, functions, unitRows, gatherRows,
                busPhiRow, busVRow, slackBus, participationColumn);
        SensitivityBasisRequest request = new SensitivityBasisRequest(unitRows,
                weightedStart, weightedRowList.stream().mapToInt(Integer::intValue).toArray(),
                weightedValueList.stream().mapToDouble(Double::doubleValue).toArray(), gatherRows, participationColumns, contraction);
        return new Built(new GpuSensitivityBasis(groupColumns, request, busPhiRow, busVRow, participationRows), null);
    }

    private static int kindOf(AcEquationType type) {
        return switch (type) {
            case BUS_TARGET_P -> GpuSensitivityBasisKind.P;
            case BUS_TARGET_Q -> GpuSensitivityBasisKind.Q;
            default -> GpuSensitivityBasisKind.V;
        };
    }

    /**
     * The contraction of the basis into the factors' sensitivities on the device, or null when a group or a factor is
     * not one the device contracts: a group of an injection (active or reactive, single or GLSK) or a target voltage,
     * whose coefficients are 0 or 1 by the scenario's row modes; a factor of a closed branch's flow or current on side 1
     * or 2, or of a bus voltage. The run's states are then solved from the basis on the host instead.
     */
    @SuppressWarnings("java:S107")
    private static SensitivityBasisRequest.Contraction contraction(LfNetwork network, List<GpuSensitivityFactorGroups.Group> groups,
                                                                    GroupColumns[] groupColumns, List<GpuSensitivityFactorGroups.FactorFunction> functions,
                                                                    int[] unitRows, int[] gatherRows, int[] busPhiRow, int[] busVRow,
                                                                    LfBus slackBus, int participationColumn) {
        if (functions == null || slackBus == null) {
            return null;
        }
        for (GpuSensitivityFactorGroups.Group group : groups) {
            if (group.variableType() != SensitivityVariableType.INJECTION_ACTIVE_POWER
                    && group.variableType() != SensitivityVariableType.INJECTION_REACTIVE_POWER
                    && group.variableType() != SensitivityVariableType.BUS_TARGET_VOLTAGE) {
                return null;
            }
        }
        // per unit column, the active power row of its bus
        int[] busOfRow = new int[gatherRowsBound(busPhiRow, busVRow)];
        Arrays.fill(busOfRow, -1);
        for (int num = 0; num < busPhiRow.length; num++) {
            if (busPhiRow[num] >= 0) {
                busOfRow[busPhiRow[num]] = num;
            }
            if (busVRow[num] >= 0) {
                busOfRow[busVRow[num]] = num;
            }
        }
        int[] unitBusPRow = new int[unitRows.length];
        for (int u = 0; u < unitRows.length; u++) {
            int bus = busOfRow[unitRows[u]];
            unitBusPRow[u] = bus >= 0 ? busPhiRow[bus] : -1;
        }
        // per group, the columns it combines and how
        int[] groupColumnStart = new int[groupColumns.length + 1];
        List<Integer> columns = new ArrayList<>();
        List<Integer> kinds = new ArrayList<>();
        for (int g = 0; g < groupColumns.length; g++) {
            GroupColumns c = groupColumns[g];
            for (int k = 0; k < c.unitColumns().length; k++) {
                columns.add(c.unitColumns()[k]);
                kinds.add(c.unitKinds()[k]);
            }
            if (c.weightedColumn() >= 0) {
                columns.add(c.weightedColumn());
                kinds.add(GpuSensitivityBasisKind.ONE);
            }
            if (c.participation() && participationColumn >= 0) {
                columns.add(participationColumn);
                kinds.add(GpuSensitivityBasisKind.ONE);
            }
            groupColumnStart[g + 1] = columns.size();
        }
        // per factor, its function, its closed branch and the gather positions of what its gradient reads
        Map<Integer, Integer> gatherPosition = new HashMap<>();
        for (int f = 0; f < gatherRows.length; f++) {
            gatherPosition.put(gatherRows[f], f);
        }
        GpuAcElementMapper elementMapper = new GpuAcElementMapper(network);
        int nFactors = functions.size();
        int[] factorGroup = new int[nFactors];
        int[] factorFunction = new int[nFactors];
        int[] factorElement = new int[nFactors];
        int[] factorGather = new int[4 * nFactors];
        Arrays.fill(factorGather, -1);
        for (int i = 0; i < nFactors; i++) {
            GpuSensitivityFactorGroups.FactorFunction function = functions.get(i);
            factorGroup[i] = function.groupIndex();
            int code = functionCode(function.type());
            if (code < 0) {
                return null;
            }
            if (function.element() instanceof LfLegBranch) {
                code = sideOneOf(code);                      // OLF reads a leg's side-2 function on its side 1
            }
            factorFunction[i] = code;
            if (code == GpuSensitivityFunction.BUS_VOLTAGE) {
                factorElement[i] = -1;
                Integer v = gatherPosition.get(busVRow[function.element().getNum()]);
                if (v == null) {
                    return null;
                }
                factorGather[4 * i] = v;
            } else {
                LfBranch branch = (LfBranch) function.element();
                int element = elementMapper.getElementIndex(branch);
                if (branch.getBus1() == null || branch.getBus2() == null || element < 0 || element >= elementMapper.getClosedBranchCount()) {
                    return null;                             // only a closed branch's gradient is evaluated on the device
                }
                factorElement[i] = element;
                Integer[] positions = {gatherPosition.get(busPhiRow[branch.getBus1().getNum()]), gatherPosition.get(busVRow[branch.getBus1().getNum()]),
                    gatherPosition.get(busPhiRow[branch.getBus2().getNum()]), gatherPosition.get(busVRow[branch.getBus2().getNum()])};
                for (int a = 0; a < 4; a++) {
                    if (positions[a] == null) {
                        return null;
                    }
                    factorGather[4 * i + a] = positions[a];
                }
            }
        }
        return new SensitivityBasisRequest.Contraction(unitBusPRow, busPhiRow[slackBus.getNum()], groupColumnStart,
                columns.stream().mapToInt(Integer::intValue).toArray(), kinds.stream().mapToInt(Integer::intValue).toArray(),
                factorGroup, factorFunction, factorElement, factorGather);
    }

    private static int gatherRowsBound(int[] busPhiRow, int[] busVRow) {
        int max = 0;
        for (int r : busPhiRow) {
            max = Math.max(max, r + 1);
        }
        for (int r : busVRow) {
            max = Math.max(max, r + 1);
        }
        return max;
    }

    /** The device's function code of a factor's function type, -1 for one it does not contract: OLF's
     *  {@code getFunctionEquationTerm}, which reads a side-3 function on its leg's side 1. */
    private static int functionCode(SensitivityFunctionType type) {
        return switch (type) {
            case BRANCH_ACTIVE_POWER_1, BRANCH_ACTIVE_POWER_3 -> GpuSensitivityFunction.ACTIVE_POWER_1;
            case BRANCH_REACTIVE_POWER_1, BRANCH_REACTIVE_POWER_3 -> GpuSensitivityFunction.REACTIVE_POWER_1;
            case BRANCH_ACTIVE_POWER_2 -> GpuSensitivityFunction.ACTIVE_POWER_2;
            case BRANCH_REACTIVE_POWER_2 -> GpuSensitivityFunction.REACTIVE_POWER_2;
            case BRANCH_CURRENT_1, BRANCH_CURRENT_3 -> GpuSensitivityFunction.CURRENT_1;
            case BRANCH_CURRENT_2 -> GpuSensitivityFunction.CURRENT_2;
            case BUS_VOLTAGE -> GpuSensitivityFunction.BUS_VOLTAGE;
            default -> -1;
        };
    }

    private static int sideOneOf(int code) {
        return switch (code) {
            case GpuSensitivityFunction.ACTIVE_POWER_2 -> GpuSensitivityFunction.ACTIVE_POWER_1;
            case GpuSensitivityFunction.REACTIVE_POWER_2 -> GpuSensitivityFunction.REACTIVE_POWER_1;
            case GpuSensitivityFunction.CURRENT_2 -> GpuSensitivityFunction.CURRENT_1;
            default -> code;
        };
    }

    /** True when the device contracts the basis into the factors' sensitivities ({@link #verifiedValues}). */
    boolean contracted() {
        return request.contraction() != null;
    }

    /**
     * The device's sensitivities of a contingency ({@code result}: the factors' values then the unit columns' row modes,
     * {@link SensitivityBasisRequest#resultLength()}), when OLF's own right-hand sides {@code b} of that contingency are the
     * combinations the device contracted: for each group, its unit columns with the coefficients the device read from the
     * row modes, plus its weighted and participation columns, equal to OLF's column on every row (a relative 1e-10). The
     * combination is compared, not its coefficients: they are not unique when the participation lives on the group's own
     * rows. Null otherwise (a rescaled GLSK, an equation the device saw otherwise): the contingency is then solved on the
     * CPU.
     */
    double[] verifiedValues(DenseMatrix b, EquationSystem<AcVariableType, AcEquationType> es, String contingencyId, double[] result) {
        SensitivityBasisRequest.Contraction contraction = request.contraction();
        if (contraction == null || result == null || result.length != request.resultLength() || b.getColumnCount() != groupColumns.length) {
            return null;
        }
        int nFactors = contraction.factorCount();
        int[] stableOfColumn = stableRowsOfEquationColumns(es, b.getRowCount());
        Map<Integer, Double> participation = request.hasParticipation()
                ? participationRows.getOrDefault(contingencyId, Map.of()) : Map.of();
        for (int g = 0; g < groupColumns.length; g++) {
            Map<Integer, Double> rhs = rightHandSide(b, g, stableOfColumn);
            if (rhs == null) {
                return null;
            }
            GroupColumns c = groupColumns[g];
            Map<Integer, Double> combined = new HashMap<>();
            for (int k = 0; k < c.unitColumns().length; k++) {
                int u = c.unitColumns()[k];
                double coefficient = SensitivityBasisRequest.Contraction.coefficient(c.unitKinds()[k], (int) result[nFactors + 2 * u],
                        (int) result[nFactors + 2 * u + 1], contraction.unitBusPRow()[u] == contraction.slackPRow());
                combined.merge(c.unitRows()[k], coefficient, Double::sum);
            }
            if (c.weightedColumn() >= 0) {
                c.weights().forEach((row, weight) -> combined.merge(row, weight, Double::sum));
            }
            if (c.participation()) {
                participation.forEach((row, value) -> combined.merge(row, value, Double::sum));
            }
            double scale = 1;
            for (double v : rhs.values()) {
                scale = Math.max(scale, Math.abs(v));
            }
            Set<Integer> rows = new HashSet<>(rhs.keySet());
            rows.addAll(combined.keySet());
            for (int row : rows) {
                if (Math.abs(rhs.getOrDefault(row, 0.0) - combined.getOrDefault(row, 0.0)) > RELATIVE_TOLERANCE * scale) {
                    return null;
                }
            }
        }
        double[] values = Arrays.copyOf(result, nFactors);
        for (double value : values) {
            if (Double.isNaN(value)) {
                return null;
            }
        }
        return values;
    }

    /** Column {@code g} of OLF's right-hand sides on stable rows, null when it writes an equation the basis has no row for. */
    private static Map<Integer, Double> rightHandSide(DenseMatrix b, int g, int[] stableOfColumn) {
        Map<Integer, Double> rhs = new HashMap<>();
        for (int row = 0; row < b.getRowCount(); row++) {
            double value = b.get(row, g);
            if (value != 0) {
                int stable = stableOfColumn[row];
                if (stable < 0) {
                    return null;
                }
                rhs.merge(stable, value, Double::sum);
            }
        }
        return rhs;
    }

    private static final int UNKNOWN = Integer.MIN_VALUE;

    /** A bus equation's stable row (its P row is its BUS_PHI row, its Q and voltage-target rows its BUS_V row), -1 when
     *  the bus has no such variable, UNKNOWN for an equation the basis has no row for. */
    private static int stableRow(GpuSensitivityFactorGroups.EquationKey key, int[] busPhiRow, int[] busVRow) {
        int num = key.elementNum();
        return switch (key.type()) {
            case BUS_TARGET_P -> num < busPhiRow.length ? busPhiRow[num] : -1;
            case BUS_TARGET_Q, BUS_TARGET_V -> num < busVRow.length ? busVRow[num] : -1;
            default -> UNKNOWN;
        };
    }

    /** OLF's participation of a contingency on stable rows, as {@code addBusInjection} writes it: not on the slack bus. */
    private static Map<Integer, Double> participationColumn(Map<LfBus, Double> participationByBus, int[] busPhiRow) {
        Map<Integer, Double> column = new LinkedHashMap<>();
        participationByBus.forEach((bus, value) -> {
            if (!bus.isSlack() && bus.getNum() < busPhiRow.length && busPhiRow[bus.getNum()] >= 0) {
                column.merge(busPhiRow[bus.getNum()], value, Double::sum);
            }
        });
        return column;
    }

    SensitivityBasisRequest request() {
        return request;
    }

    int[] gatherRows() {
        return request.gatherRows();
    }

    /**
     * Rebuilds the solved states of OLF's right-hand sides {@code b} (in place, as {@code solveFactorsStates} leaves them)
     * from the contingency's basis solutions {@code u} (column-major, {@link SensitivityBasisRequest#resultLength()}),
     * on the current equation system; {@code currentRowOfGather} the current variable row of each gather row (-1 when
     * the variable is gone). False, with {@code b} untouched, when a right-hand side is not exactly in the basis.
     */
    boolean solve(DenseMatrix b, EquationSystem<AcVariableType, AcEquationType> es, String contingencyId, double[] u,
                  int[] currentRowOfGather) {
        int nGather = request.gatherRows().length;
        if (u == null || u.length != request.resultLength() || currentRowOfGather.length != nGather
                || b.getColumnCount() != groupColumns.length) {
            return false;
        }
        int[] stableOfColumn = stableRowsOfEquationColumns(es, b.getRowCount());
        Map<Integer, Double> participation = request.hasParticipation()
                ? participationRows.getOrDefault(contingencyId, Map.of()) : Map.of();
        int participationColumn = request.participationColumn();
        double[][] solved = new double[groupColumns.length][];
        for (int g = 0; g < groupColumns.length; g++) {
            Map<Integer, Double> rhs = rightHandSide(b, g, stableOfColumn);
            if (rhs == null) {
                return false;                                // an equation the basis has no row for
            }
            double[] coefficients = decompose(groupColumns[g], rhs, participation);
            if (coefficients == null) {
                return false;
            }
            GroupColumns c = groupColumns[g];
            double[] x = new double[nGather];
            for (int k = 0; k < c.unitColumns().length; k++) {
                addColumn(x, u, c.unitColumns()[k], coefficients[k], nGather);
            }
            int last = c.unitColumns().length;
            if (c.weightedColumn() >= 0) {
                addColumn(x, u, c.weightedColumn(), coefficients[last], nGather);
            }
            if (c.participation() && participationColumn >= 0) {
                addColumn(x, u, participationColumn, coefficients[last + 1], nGather);
            }
            for (double value : x) {
                if (!Double.isFinite(value)) {
                    return false;                            // a column the device could not solve
                }
            }
            solved[g] = x;
        }
        b.reset();
        for (int g = 0; g < groupColumns.length; g++) {
            for (int f = 0; f < nGather; f++) {
                if (currentRowOfGather[f] >= 0) {
                    b.set(currentRowOfGather[f], g, solved[g][f]);
                }
            }
        }
        return true;
    }

    private static void addColumn(double[] x, double[] u, int column, double coefficient, int nGather) {
        if (coefficient == 0) {
            return;
        }
        int offset = column * nGather;
        for (int f = 0; f < nGather; f++) {
            x[f] += coefficient * u[offset + f];
        }
    }

    /**
     * The coefficients of a group's right-hand side {@code rhs} (stable row to value) on its basis columns: its own unit
     * columns, then its weighted column, then the participation column; null when {@code rhs} is not their combination.
     * The weighted and participation coefficients come from the rows the group writes no coefficient of its own (least
     * squares over the two, or either alone), the own coefficients are what is left on their rows.
     */
    static double[] decompose(GroupColumns c, Map<Integer, Double> rhs, Map<Integer, Double> participation) {
        Map<Integer, Double> weights = c.weightedColumn() >= 0 ? c.weights() : Map.of();
        Map<Integer, Double> pi = c.participation() ? participation : Map.of();
        Set<Integer> own = new HashSet<>();
        for (int r : c.unitRows()) {
            own.add(r);
        }
        Set<Integer> rows = new HashSet<>(rhs.keySet());
        rows.addAll(weights.keySet());
        rows.addAll(pi.keySet());
        rows.removeAll(own);
        double sww = 0;
        double spp = 0;
        double swp = 0;
        double swb = 0;
        double spb = 0;
        double scale = 1;
        for (double v : rhs.values()) {
            scale = Math.max(scale, Math.abs(v));
        }
        for (int r : rows) {
            double w = weights.getOrDefault(r, 0.0);
            double p = pi.getOrDefault(r, 0.0);
            double v = rhs.getOrDefault(r, 0.0);
            sww += w * w;
            spp += p * p;
            swp += w * p;
            swb += w * v;
            spb += p * v;
        }
        double gamma = 0;
        double alpha = 0;
        double det = sww * spp - swp * swp;
        if (sww > 0 && spp > 0 && Math.abs(det) > 1e-12 * sww * spp) {
            gamma = (swb * spp - spb * swp) / det;
            alpha = (spb * sww - swb * swp) / det;
        } else if (spp > 0) {
            alpha = spb / spp;
        } else if (sww > 0) {
            gamma = swb / sww;
        }
        for (int r : rows) {
            double residual = rhs.getOrDefault(r, 0.0) - gamma * weights.getOrDefault(r, 0.0) - alpha * pi.getOrDefault(r, 0.0);
            if (Math.abs(residual) > RELATIVE_TOLERANCE * scale) {
                return null;
            }
        }
        double[] coefficients = new double[c.unitRows().length + 2];
        for (int k = 0; k < c.unitRows().length; k++) {
            int r = c.unitRows()[k];
            coefficients[k] = rhs.getOrDefault(r, 0.0) - gamma * weights.getOrDefault(r, 0.0) - alpha * pi.getOrDefault(r, 0.0);
        }
        coefficients[c.unitRows().length] = gamma;
        coefficients[c.unitRows().length + 1] = alpha;
        return coefficients;
    }

    /** The stable row of each of OLF's equation columns of the current system: a bus's P equation on its BUS_PHI row,
     *  its Q or voltage-target equation on its BUS_V row; -1 for any other equation. */
    private int[] stableRowsOfEquationColumns(EquationSystem<AcVariableType, AcEquationType> es, int columnCount) {
        int[] stable = new int[columnCount];
        Arrays.fill(stable, -1);
        for (int num = 0; num < busPhiRow.length; num++) {
            mark(stable, es, num, AcEquationType.BUS_TARGET_P, busPhiRow[num]);
            mark(stable, es, num, AcEquationType.BUS_TARGET_Q, busVRow[num]);
            mark(stable, es, num, AcEquationType.BUS_TARGET_V, busVRow[num]);
        }
        return stable;
    }

    private static void mark(int[] stable, EquationSystem<AcVariableType, AcEquationType> es, int num, AcEquationType type, int row) {
        if (row < 0) {
            return;
        }
        es.getEquation(num, type).ifPresent(equation -> {
            int column = equation.getColumn();
            if (column >= 0 && column < stable.length) {
                stable[column] = row;
            }
        });
    }
}
