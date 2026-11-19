/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.contingency.violations.LimitViolation;
import com.powsybl.contingency.violations.LimitViolationType;
import com.powsybl.iidm.network.TwoSides;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuTwoSides;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.sa.LimitViolationManager;
import com.powsybl.security.SecurityAnalysisParameters;
import com.powsybl.security.limitscaling.LimitScaling;
import org.apache.commons.lang3.tuple.Pair;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The base case's violations as OLF's reference filtering sees them, per entry of the limit table, for the device screen
 * to tell which violations of a post-contingency state OLF would report (the types it may report, per scenario).
 *
 * <p>OLF keeps a post-contingency violation unless the base case has one under the same key (branch side and limits
 * group; bus) that it is weaker than or equivalent to ({@code LimitViolationManager.violationWeakenedOrEquivalent}): same
 * type, and a smaller limit, or the same limit and a value within the increased-violations thresholds. So, per limit entry
 * (the violation reported when that limit is the group's first exceeded) and per bus side, one of: {@link #NEW} (no base
 * violation of that type under the key, or a larger limit), {@link #FILTERED} (a smaller limit), {@link #VALUE} (the same
 * limit: kept when the value passes {@code value}, in the table's per-unit, negated for a low voltage).
 *
 * <p>The base violations come from the base state through the same detection and the same keys as OLF's pre-contingency
 * manager: a manager that records every violation it is fed, under OLF's key, last fed kept (OLF's map).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuBaseViolations {

    static final int NEW = 0;
    static final int FILTERED = 1;
    static final int VALUE = 2;

    /** The possible-types bits the screen leaves per scenario: a branch group's quantity q is bit q (GpuLimitTable). */
    static final int CURRENT = 1;
    static final int ACTIVE_POWER = 1 << 1;
    static final int APPARENT_POWER = 1 << 2;
    static final int HIGH_VOLTAGE = 1 << 3;
    static final int LOW_VOLTAGE = 1 << 4;
    /** Any other type (a voltage angle): never left by the screen, set where the screen does not decide. */
    static final int OTHER = 1 << 5;
    /** Every type: the screen does not decide. */
    static final int ALL = (1 << 6) - 1;

    /** OLF's LimitViolationManager.EQUALITY_EPSILON: the relative margin of its equal-value comparisons. */
    private static final double EQUALITY_EPSILON = 1e-12;

    final int[] limitKind;
    final double[] limitValue;
    final int[] busHighKind;
    final double[] busHighValue;
    final int[] busLowKind;
    final double[] busLowValue;

    private GpuBaseViolations(int[] limitKind, double[] limitValue, int[] busHighKind, double[] busHighValue,
                              int[] busLowKind, double[] busLowValue) {
        this.limitKind = limitKind;
        this.limitValue = limitValue;
        this.busHighKind = busHighKind;
        this.busHighValue = busHighValue;
        this.busLowKind = busLowKind;
        this.busLowValue = busLowValue;
    }

    /** The bits of the violations of the types {@code types} (OTHER for a type the screen does not leave). */
    static int bits(Set<LimitViolationType> types) {
        if (types.isEmpty()) {
            return ALL;
        }
        int bits = 0;
        for (LimitViolationType type : types) {
            bits |= switch (type) {
                case CURRENT -> CURRENT;
                case ACTIVE_POWER -> ACTIVE_POWER;
                case APPARENT_POWER -> APPARENT_POWER;
                case HIGH_VOLTAGE -> HIGH_VOLTAGE;
                case LOW_VOLTAGE -> LOW_VOLTAGE;
                default -> OTHER;
            };
        }
        return bits;
    }

    /** From the base state the network holds (its pre-contingency load flow). */
    static GpuBaseViolations of(LfNetwork network, GpuLimitTable table, List<LimitScaling> limitScalings,
                                SecurityAnalysisParameters.IncreasedViolationsParameters parameters) {
        Map<Object, LimitViolation> base = baseViolations(network, table, limitScalings);
        int[] limitKind = new int[table.limitCount()];
        double[] limitValue = new double[table.limitCount()];
        LfBranch[] branches = table.branches();
        for (int g = 0; g < table.groupCount(); g++) {
            LfBranch branch = branches[table.groupBranch(g)];
            LimitViolationType type = GpuLimitTable.violationType(table.quantity(g));
            TwoSides side = table.side(g) == GpuTwoSides.ONE ? TwoSides.ONE : TwoSides.TWO;
            for (int l = table.limitStart(g); l < table.limitEnd(g); l++) {
                LimitViolation reported = LimitViolationManager.createLimitViolation(branch, table.groupId(g), table.limit(l),
                        type, table.scale(g), 0.0, side);
                LimitViolation reference = base.get(branchKey(reported));
                limitKind[l] = kind(reference, reported);
                if (limitKind[l] == VALUE) {
                    limitValue[l] = Math.abs(reference.getValue())
                            * (1 + parameters.getFlowProportionalThreshold() + EQUALITY_EPSILON) / table.scale(g);
                }
            }
        }
        LfBus[] buses = table.buses();
        int[] busHighKind = new int[buses.length];
        double[] busHighValue = new double[buses.length];
        int[] busLowKind = new int[buses.length];
        double[] busLowValue = new double[buses.length];
        for (int i = 0; i < buses.length; i++) {
            LfBus bus = buses[i];
            double scale = table.busScale(i);
            LimitViolation high = LimitViolationManager.createBusLimitViolation(bus, LimitViolationType.HIGH_VOLTAGE,
                    table.busHighLimit(i) * scale, 0.0);
            LimitViolation reference = base.get(busKey(bus, high));
            busHighKind[i] = kind(reference, high);
            if (busHighKind[i] == VALUE) {
                double v1 = reference.getValue();
                busHighValue[i] = (v1 + Math.min(parameters.getHighVoltageAbsoluteThreshold(),
                        v1 * parameters.getHighVoltageProportionalThreshold()) + Math.abs(v1) * EQUALITY_EPSILON) / scale;
            }
            LimitViolation low = LimitViolationManager.createBusLimitViolation(bus, LimitViolationType.LOW_VOLTAGE,
                    table.busLowLimit(i) * scale, 0.0);
            reference = base.get(busKey(bus, low));
            busLowKind[i] = kind(reference, low);
            if (busLowKind[i] == VALUE) {
                double v1 = reference.getValue();
                busLowValue[i] = (v1 - Math.min(parameters.getLowVoltageAbsoluteThreshold(),
                        v1 * parameters.getLowVoltageProportionalThreshold()) - Math.abs(v1) * EQUALITY_EPSILON) / scale;
            }
        }
        return new GpuBaseViolations(limitKind, limitValue, busHighKind, busHighValue, busLowKind, busLowValue);
    }

    /** violationWeakenedOrEquivalent(reference, reported) as a kind: the value is all it leaves to the scenario. */
    private static int kind(LimitViolation reference, LimitViolation reported) {
        if (reference == null || reference.getLimitType() != reported.getLimitType()) {
            return NEW;
        }
        if (reported.getLimit() < reference.getLimit()) {
            return FILTERED;
        }
        return reported.getLimit() == reference.getLimit() ? VALUE : NEW;
    }

    /** OLF's key of a branch violation (LimitViolationManager.addBranchLimitViolation). */
    private static Object branchKey(LimitViolation violation) {
        return Pair.of(Pair.of(violation.getSubjectId(), violation.getSide()), violation.getOperationalLimitsGroupId());
    }

    /** OLF's key of a bus violation (LimitViolationManager.addBusLimitViolation). */
    private static Object busKey(LfBus bus, LimitViolation violation) {
        return Pair.of(bus.getId(), violation.getOperationalLimitsGroupId());
    }

    /** The base violations under OLF's keys, the last one fed under a key kept, as OLF's manager keeps it. */
    private static Map<Object, LimitViolation> baseViolations(LfNetwork network, GpuLimitTable table,
                                                              List<LimitScaling> limitScalings) {
        Map<Object, LimitViolation> violations = new HashMap<>();
        LimitViolationManager capture = new LimitViolationManager(limitScalings) {
            @Override
            public void addBranchLimitViolation(LimitViolation violation) {
                violations.put(branchKey(violation), violation);
            }

            @Override
            public void addBusLimitViolation(LimitViolation violation, LfBus bus) {
                violations.put(busKey(bus, violation), violation);
            }

            @Override
            public void addVoltageAngleLimitViolation(LimitViolation violation, LfNetwork.LfVoltageAngleLimit limit) {
                // a network with voltage-angle limits leaves every type possible (GpuStateHandback)
            }
        };
        new GpuLimitScanner(table).detectViolations(network, capture);
        return violations;
    }
}
