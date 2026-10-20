/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.contingency.violations.LimitViolationType;
import com.powsybl.iidm.network.LimitType;
import com.powsybl.iidm.network.TwoSides;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.sa.LimitReductionManager;
import com.powsybl.openloadflow.util.PerUnit;

import java.util.ArrayList;
import java.util.List;

/**
 * The run's limit thresholds, flattened once into primitive arrays.
 *
 * <p>{@code LimitViolationManager.detectViolations} walks, for every branch and every contingency,
 * two sides x three limit types x their operational limits groups, through a
 * {@code Supplier<Map<...>>} built per call and a {@code List<LfLimit>} per group. The comparison
 * itself is one {@code >}; everything around it is the cost. Measured on rte6515 at 1000
 * contingencies: 5473 ms of scan against 86 ms of actually building the violations, so 98.4% of
 * detection is the walk, not the reporting.
 *
 * <p>Two properties make the flattening exact rather than approximate, both read out of OLF's own
 * source and not assumed:
 * <ol>
 *   <li><b>The thresholds are static for the run.</b> {@code AbstractLfBranch} builds each side's
 *       {@code List<LfLimitsGroup>} lazily and CACHES it keyed only by limit type, with the
 *       reductions already applied — OLF's comment in {@code createLimits} gives the reason it may
 *       do so: <i>"the only supported ContingencyContext for LimitReduction is ALL"</i>. So a
 *       contingency cannot change a threshold, and building this table also warms exactly the cache
 *       the stock path would have built.</li>
 *   <li><b>Only the VALUE varies.</b> {@code createLimitViolation} takes every other field from the
 *       branch and the {@code LfLimit} — subject, group id, type, limit name, acceptable duration,
 *       {@code limit * scale}, reduction, side. So an entry can carry the metadata and the detector
 *       needs to carry only {@code (entry, value)}.</li>
 * </ol>
 *
 * <p>What stays DYNAMIC and must therefore be tested per contingency, not baked in: whether a branch
 * is disabled, and whether a bus is disabled. A branch's {@code getBus1()}/{@code getBus2()} are
 * structural and are resolved here.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuLimitTable {

    /** Quantity a group compares against: side current, side active power (compared in absolute
     *  value), side apparent power (skipped when NaN — fictitious branches). */
    static final byte QTY_I = 0;
    static final byte QTY_P = 1;
    static final byte QTY_S = 2;

    // --- branches, in LfNetwork order; a branch's groups are contiguous ---
    private final LfBranch[] branches;
    private final int[] branchGroupStart;
    private final int[] branchGroupEnd;

    // --- one entry per (branch, side, limit type, operational limits group) ---
    private final byte[] groupSide;                          // a GpuTwoSides
    private final byte[] groupQuantity;
    private final double[] groupScale;                       // per-unit -> reported unit
    private final String[] groupId;
    private final int[] groupLimitStart;
    private final int[] groupLimitEnd;

    // --- the sorted limits of each group, most severe first (OLF's order, kept verbatim) ---
    private final double[] limitThreshold;                   // LfLimit.getReducedValue(), per unit
    private final LfBranch.LfLimit[] limits;
    private final int[] limitGroup;                          // back-pointer: a screen emits limit indices
    private final int[] groupBranch;                         // the branch a group belongs to

    // --- buses ---
    private final LfBus[] buses;
    private final double[] busHighLimit;                     // NaN when absent — see the scan
    private final double[] busLowLimit;
    private final double[] busScale;

    private GpuLimitTable(LfBranch[] branches, int[] branchGroupStart, int[] branchGroupEnd,
                          byte[] groupSide, byte[] groupQuantity, double[] groupScale, String[] groupId,
                          int[] groupLimitStart, int[] groupLimitEnd,
                          double[] limitThreshold, LfBranch.LfLimit[] limits,
                          LfBus[] buses, double[] busHighLimit, double[] busLowLimit, double[] busScale) {
        this.limitGroup = new int[limitThreshold.length];
        this.groupBranch = new int[groupSide.length];
        for (int b = 0; b < branches.length; b++) {
            for (int g = branchGroupStart[b]; g < branchGroupEnd[b]; g++) {
                this.groupBranch[g] = b;
                for (int l = groupLimitStart[g]; l < groupLimitEnd[g]; l++) {
                    this.limitGroup[l] = g;
                }
            }
        }
        this.branches = branches;
        this.branchGroupStart = branchGroupStart;
        this.branchGroupEnd = branchGroupEnd;
        this.groupSide = groupSide;
        this.groupQuantity = groupQuantity;
        this.groupScale = groupScale;
        this.groupId = groupId;
        this.groupLimitStart = groupLimitStart;
        this.groupLimitEnd = groupLimitEnd;
        this.limitThreshold = limitThreshold;
        this.limits = limits;
        this.buses = buses;
        this.busHighLimit = busHighLimit;
        this.busLowLimit = busLowLimit;
        this.busScale = busScale;
    }

    LfBranch[] branches() {
        return branches;
    }

    int groupStart(int branchIndex) {
        return branchGroupStart[branchIndex];
    }

    int groupEnd(int branchIndex) {
        return branchGroupEnd[branchIndex];
    }

    byte side(int group) {
        return groupSide[group];
    }

    byte quantity(int group) {
        return groupQuantity[group];
    }

    double scale(int group) {
        return groupScale[group];
    }

    String groupId(int group) {
        return groupId[group];
    }

    int limitStart(int group) {
        return groupLimitStart[group];
    }

    int limitEnd(int group) {
        return groupLimitEnd[group];
    }

    double threshold(int limit) {
        return limitThreshold[limit];
    }

    LfBranch.LfLimit limit(int limit) {
        return limits[limit];
    }

    LfBus[] buses() {
        return buses;
    }

    double busHighLimit(int busIndex) {
        return busHighLimit[busIndex];
    }

    double busLowLimit(int busIndex) {
        return busLowLimit[busIndex];
    }

    double busScale(int busIndex) {
        return busScale[busIndex];
    }

    int groupCount() {
        return groupSide.length;
    }

    /** The group a limit belongs to — a screen emits limit indices, and the "first exceeded of the
     *  group wins" rule is applied by the consumer, which needs to know where a group ends. */
    int limitGroup(int limit) {
        return limitGroup[limit];
    }

    /** The branch index (into {@link #branches()}) a group belongs to. */
    int groupBranch(int group) {
        return groupBranch[group];
    }

    int limitCount() {
        return limitThreshold.length;
    }

    /**
     * Flatten the network's limits. Call ONCE per run, on the base network — the lazily built
     * {@code LfLimitsGroup} lists it forces are the same ones the stock detection would have built,
     * and they outlive every contingency.
     */
    static GpuLimitTable build(LfNetwork network, LimitReductionManager limitReductionManager) {
        List<LfBranch> networkBranches = network.getBranches();
        LfBranch[] branches = networkBranches.toArray(new LfBranch[0]);
        int[] groupStart = new int[branches.length];
        int[] groupEnd = new int[branches.length];

        List<Byte> sides = new ArrayList<>();
        List<Byte> quantities = new ArrayList<>();
        List<Double> scales = new ArrayList<>();
        List<String> ids = new ArrayList<>();
        List<Integer> limitStarts = new ArrayList<>();
        List<Integer> limitEnds = new ArrayList<>();
        List<Double> thresholds = new ArrayList<>();
        List<LfBranch.LfLimit> flatLimits = new ArrayList<>();

        for (int b = 0; b < branches.length; b++) {
            LfBranch branch = branches[b];
            groupStart[b] = sides.size();
            // The stock detection tests getBus1()/getBus2() for null and skips the side entirely.
            if (branch.getBus1() != null) {
                addSide(branch, branch.getBus1(), TwoSides.ONE, limitReductionManager,
                        sides, quantities, scales, ids, limitStarts, limitEnds, thresholds, flatLimits);
            }
            if (branch.getBus2() != null) {
                addSide(branch, branch.getBus2(), TwoSides.TWO, limitReductionManager,
                        sides, quantities, scales, ids, limitStarts, limitEnds, thresholds, flatLimits);
            }
            groupEnd[b] = sides.size();
        }

        List<LfBus> networkBuses = network.getBuses();
        LfBus[] buses = networkBuses.toArray(new LfBus[0]);
        double[] busHigh = new double[buses.length];
        double[] busLow = new double[buses.length];
        double[] busScale = new double[buses.length];
        for (int i = 0; i < buses.length; i++) {
            busHigh[i] = buses[i].getHighVoltageLimit();
            busLow[i] = buses[i].getLowVoltageLimit();
            busScale[i] = buses[i].getNominalV();
        }

        GpuLimitTable table = new GpuLimitTable(branches, groupStart, groupEnd,
                toByteArray(sides), toByteArray(quantities), toDoubleArray(scales), ids.toArray(new String[0]),
                toIntArray(limitStarts), toIntArray(limitEnds),
                toDoubleArray(thresholds), flatLimits.toArray(new LfBranch.LfLimit[0]),
                buses, busHigh, busLow, busScale);
        if (System.getenv("OLF_GPU_PHASE_TIME") != null) {
            // What one contingency's scan actually costs is this table's size, so print it: it is the
            // denominator of every per-entry number, and the work a device kernel would take over.
            long nCurrent = quantities.stream().filter(q -> q == QTY_I).count();
            long nActivePower = quantities.stream().filter(q -> q == QTY_P).count();
            long nApparentPower = quantities.stream().filter(q -> q == QTY_S).count();
            System.err.printf("LIMIT_TABLE branches=%d withLimits=%d groups=%d (I=%d P=%d S=%d) limits=%d buses=%d%n",
                    branches.length, countBranchesWithLimits(groupStart, groupEnd), sides.size(), nCurrent, nActivePower, nApparentPower,
                    thresholds.size(), buses.length);
        }
        return table;
    }

    private static void addSide(LfBranch branch, LfBus bus, TwoSides side, LimitReductionManager limitReductionManager,
                                List<Byte> sides, List<Byte> quantities, List<Double> scales, List<String> ids,
                                List<Integer> limitStarts, List<Integer> limitEnds,
                                List<Double> thresholds, List<LfBranch.LfLimit> flatLimits) {
        // Same order as detectBranchSideViolations: CURRENT, then ACTIVE_POWER, then APPARENT_POWER.
        // The order decides nothing about WHICH violations are found (each group keys its own entry
        // in the manager), but keeping it makes a table-vs-stock diff line up group by group.
        addGroups(branch, side, LimitType.CURRENT, QTY_I, PerUnit.ib(bus.getNominalV()),
                limitReductionManager, sides, quantities, scales, ids, limitStarts, limitEnds, thresholds, flatLimits);
        addGroups(branch, side, LimitType.ACTIVE_POWER, QTY_P, PerUnit.SB,
                limitReductionManager, sides, quantities, scales, ids, limitStarts, limitEnds, thresholds, flatLimits);
        addGroups(branch, side, LimitType.APPARENT_POWER, QTY_S, PerUnit.SB,
                limitReductionManager, sides, quantities, scales, ids, limitStarts, limitEnds, thresholds, flatLimits);
    }

    private static void addGroups(LfBranch branch, TwoSides side, LimitType type, byte quantity, double scale,
                                  LimitReductionManager limitReductionManager,
                                  List<Byte> sides, List<Byte> quantities, List<Double> scales, List<String> ids,
                                  List<Integer> limitStarts, List<Integer> limitEnds,
                                  List<Double> thresholds, List<LfBranch.LfLimit> flatLimits) {
        List<LfBranch.LfLimitsGroup> groups = side == TwoSides.ONE
                ? branch.getLimits1(type, limitReductionManager)
                : branch.getLimits2(type, limitReductionManager);
        for (LfBranch.LfLimitsGroup group : groups) {
            List<LfBranch.LfLimit> sortedLimits = group.getSortedLimits();
            if (sortedLimits.isEmpty()) {
                continue;                                    // nothing to compare — the stock loop would find nothing
            }
            sides.add((byte) side.ordinal());
            quantities.add(quantity);
            scales.add(scale);
            ids.add(group.getOperationalLimitsGroupId());
            limitStarts.add(thresholds.size());
            for (LfBranch.LfLimit limit : sortedLimits) {
                thresholds.add(limit.getReducedValue());
                flatLimits.add(limit);
            }
            limitEnds.add(thresholds.size());
        }
    }

    /** The violation type a quantity reports. */
    static LimitViolationType violationType(byte quantity) {
        return switch (quantity) {
            case QTY_I -> LimitViolationType.CURRENT;
            case QTY_P -> LimitViolationType.ACTIVE_POWER;
            case QTY_S -> LimitViolationType.APPARENT_POWER;
            default -> throw new IllegalArgumentException("unknown limit quantity " + quantity);
        };
    }

    private static int countBranchesWithLimits(int[] groupStart, int[] groupEnd) {
        int n = 0;
        for (int b = 0; b < groupStart.length; b++) {
            if (groupStart[b] != groupEnd[b]) {
                n++;
            }
        }
        return n;
    }

    private static byte[] toByteArray(List<Byte> list) {
        byte[] out = new byte[list.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = list.get(i);
        }
        return out;
    }

    private static int[] toIntArray(List<Integer> list) {
        int[] out = new int[list.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = list.get(i);
        }
        return out;
    }

    private static double[] toDoubleArray(List<Double> list) {
        double[] out = new double[list.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = list.get(i);
        }
        return out;
    }
}
