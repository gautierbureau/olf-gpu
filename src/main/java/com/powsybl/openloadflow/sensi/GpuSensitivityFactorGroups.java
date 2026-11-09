/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sensi;

import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.equations.Equation;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfElement;
import com.powsybl.openloadflow.util.Derivable;
import com.powsybl.sensitivity.SensitivityFunctionType;
import com.powsybl.sensitivity.SensitivityVariableType;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * OLF's sensitivity factor groups and factors as the GPU batch reads them, from inside OLF's package: a group's variable
 * elements are protected there.
 *
 * <p>For each group, the equations its right-hand side can write ({@code AbstractSensitivityAnalysis}'s {@code fillRhs}):
 * with a coefficient of its own (the variable's equation, or a branch parameter's four flow partials), with the weights
 * of a multi-variable group, and whether the slack participation enters it. For each factor, the variables its function
 * reads: what {@code calculateSensi} reads of the solved states.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuSensitivityFactorGroups {

    /** An equation, by type and element. */
    public record EquationKey(AcEquationType type, int elementNum) {
    }

    /** A variable, by type and element. */
    public record VariableKey(AcVariableType type, int elementNum) {
    }

    /**
     * A factor group's right-hand side: its column ({@code index}), the equations it writes with coefficients of its own
     * ({@code units}), the normalized weights of a multi-variable group (empty otherwise), whether the slack participation
     * enters it, and whether the batch represents it at all (a phase shifter's angle, an SVC pilot point: not).
     */
    public record Group(int index, SensitivityVariableType variableType, List<EquationKey> units, Map<EquationKey, Double> weights,
                        boolean participation, boolean supported) {
    }

    private GpuSensitivityFactorGroups() {
    }

    public static List<Group> of(AbstractSensitivityAnalysis.SensitivityFactorGroupList<AcVariableType, AcEquationType> factorGroups) {
        List<Group> groups = new ArrayList<>(factorGroups.getList().size());
        for (AbstractSensitivityAnalysis.SensitivityFactorGroup<AcVariableType, AcEquationType> group : factorGroups.getList()) {
            groups.add(describe(group));
        }
        return groups;
    }

    private static Group describe(AbstractSensitivityAnalysis.SensitivityFactorGroup<AcVariableType, AcEquationType> group) {
        int index = group.getIndex();
        if (group instanceof AbstractSensitivityAnalysis.SingleVariableFactorGroup<AcVariableType, AcEquationType> single) {
            SensitivityVariableType type = single.variableType;
            var factor = (AbstractSensitivityAnalysis.SingleVariableLfSensitivityFactor<AcVariableType, AcEquationType>) single.getFactors().getFirst();
            LfElement element = factor.getVariableElement();
            List<EquationKey> units = new ArrayList<>();
            boolean supported = true;
            boolean participation = false;
            switch (type) {
                case INJECTION_ACTIVE_POWER -> {
                    units.add(new EquationKey(AcEquationType.BUS_TARGET_P, element.getNum()));
                    participation = true;
                }
                case INJECTION_REACTIVE_POWER, SHUNT_COMPENSATOR_SUSCEPTANCE ->
                    units.add(new EquationKey(AcEquationType.BUS_TARGET_Q, element.getNum()));
                case BUS_TARGET_VOLTAGE -> {
                    Equation<AcVariableType, AcEquationType> equation = factor.getVariableEquation();
                    if (equation != null) {
                        units.add(new EquationKey(equation.getType(), equation.getElementNum()));
                    }
                }
                case BRANCH_RESISTANCE, BRANCH_REACTANCE, BRANCH_ADMITTANCE -> {
                    LfBranch branch = (LfBranch) element;
                    for (LfBus bus : new LfBus[] {branch.getBus1(), branch.getBus2()}) {
                        if (bus != null) {
                            units.add(new EquationKey(AcEquationType.BUS_TARGET_P, bus.getNum()));
                            units.add(new EquationKey(AcEquationType.BUS_TARGET_Q, bus.getNum()));
                        }
                    }
                }
                default -> supported = false;           // a phase shifter's angle, an SVC pilot point
            }
            return new Group(index, type, units, Map.of(), participation, supported);
        }
        if (group instanceof AbstractSensitivityAnalysis.MultiVariablesFactorGroup<AcVariableType, AcEquationType> multi) {
            SensitivityVariableType type = multi.variableType;
            Map<LfElement, Double> elements = multi.getVariableElements();
            Map<EquationKey, Double> weights = new LinkedHashMap<>();
            switch (type) {
                case INJECTION_ACTIVE_POWER -> {
                    // fillRhs: weight / sum of |weights| over the elements in the main component (all of them here: a
                    // contingency that disconnects one rescales, and its right-hand side then leaves the basis)
                    double weightSum = elements.values().stream().mapToDouble(Math::abs).sum();
                    elements.forEach((e, w) -> weights.merge(new EquationKey(AcEquationType.BUS_TARGET_P, e.getNum()), w / weightSum, Double::sum));
                }
                case HVDC_LINE_ACTIVE_POWER ->
                    elements.forEach((e, w) -> weights.merge(new EquationKey(AcEquationType.BUS_TARGET_P, e.getNum()), w, Double::sum));
                default -> {
                    return new Group(index, type, List.of(), Map.of(), false, false);
                }
            }
            return new Group(index, type, List.of(), weights, true, true);
        }
        return new Group(index, null, List.of(), Map.of(), false, false);
    }

    /**
     * The variables the function of each factor reads, by function type: a branch flow or current reads the voltage and
     * angle of its two buses, a bus voltage its own voltage, a bus reactive injection those of the bus and of every bus
     * its branches reach. Null when a factor's function is of another kind: the caller then has no row set it can trust.
     * What a function really reads is checked against this by the caller (a probe through {@code calculateSensi}).
     */
    public static Set<VariableKey> functionVariables(List<AbstractSensitivityAnalysis.LfSensitivityFactor<AcVariableType, AcEquationType>> factors) {
        Set<VariableKey> variables = new LinkedHashSet<>();
        for (var factor : factors) {
            if (factor.getStatus() != AbstractSensitivityAnalysis.LfSensitivityFactor.Status.VALID) {
                continue;                                    // its sensitivity is not read from the states
            }
            SensitivityFunctionType functionType = factor.getFunctionType();
            LfElement element = factor.getFunctionElement();
            switch (functionType) {
                case BRANCH_ACTIVE_POWER_1, BRANCH_ACTIVE_POWER_2, BRANCH_ACTIVE_POWER_3,
                     BRANCH_REACTIVE_POWER_1, BRANCH_REACTIVE_POWER_2, BRANCH_REACTIVE_POWER_3,
                     BRANCH_CURRENT_1, BRANCH_CURRENT_2, BRANCH_CURRENT_3 -> addBranchBuses(variables, (LfBranch) element);
                case BUS_VOLTAGE -> variables.add(new VariableKey(AcVariableType.BUS_V, element.getNum()));
                case BUS_REACTIVE_POWER -> {
                    LfBus bus = (LfBus) element;
                    addBus(variables, bus);
                    for (LfBranch branch : bus.getBranches()) {
                        addBranchBuses(variables, branch);
                    }
                }
                default -> {
                    return null;
                }
            }
        }
        return variables;
    }

    /** The function terms of the factors whose sensitivity is read from the solved states (status VALID). */
    public static List<Derivable<AcVariableType>> functionTerms(List<AbstractSensitivityAnalysis.LfSensitivityFactor<AcVariableType, AcEquationType>> factors) {
        List<Derivable<AcVariableType>> terms = new ArrayList<>();
        for (var factor : factors) {
            if (factor.getStatus() == AbstractSensitivityAnalysis.LfSensitivityFactor.Status.VALID) {
                terms.add(factor.getFunctionEquationTerm());
            }
        }
        return terms;
    }

    private static void addBranchBuses(Set<VariableKey> variables, LfBranch branch) {
        addBus(variables, branch.getBus1());
        addBus(variables, branch.getBus2());
    }

    private static void addBus(Set<VariableKey> variables, LfBus bus) {
        if (bus != null) {
            variables.add(new VariableKey(AcVariableType.BUS_V, bus.getNum()));
            variables.add(new VariableKey(AcVariableType.BUS_PHI, bus.getNum()));
        }
    }
}
