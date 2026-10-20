/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.iidm.network.TwoSides;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuTwoSides;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.sa.LimitViolationManager;

/**
 * Violation detection over a flattened {@link GpuLimitTable}, decision for decision the same as
 * {@code LimitViolationManager.detectViolations}.
 *
 * <p>This is the host half of on-device violation detection, and it is useful twice. It replaces the
 * per-contingency walk through suppliers and lists with a walk over primitive arrays — the scan that
 * measured 98.4% of detection. And it is the REFERENCE the device kernel is validated against: the
 * kernel will emit {@code (scenario, group, limit, value)} records and this same code will rebuild
 * the violations from them, so a kernel that disagrees disagrees about the comparison alone.
 *
 * <p>Everything it decides, it decides the way OLF does:
 * <ul>
 *   <li>within a group, the FIRST exceeded limit wins and the walk stops — OLF's {@code break},
 *       whose comment explains it as <i>"Only detect the most serious one ... limit violations are
 *       ordered by severity"</i>;</li>
 *   <li>active power compares {@code |p|} but REPORTS the signed value;</li>
 *   <li>apparent power skips a NaN (fictitious branches);</li>
 *   <li>a bus limit of NaN never fires — here by letting the NaN comparison be false, which is what
 *       OLF's explicit {@code !isNaN(limit) &&} guard amounts to.</li>
 * </ul>
 *
 * <p>The values themselves come from the branch's own {@code Evaluable}s, the same array reads the
 * stock detection does, so a violation found here carries bit-identical numbers.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuLimitScanner {

    private final GpuLimitTable table;

    /** The six per-side quantities of the branch being scanned, each computed at most once: the
     *  stock path re-evaluates per limits group, which is free on a network with one group per side
     *  and not free on one with several. */
    private final double[] quantityCache = new double[6];
    private final boolean[] quantityCached = new boolean[6];

    GpuLimitScanner(GpuLimitTable table) {
        this.table = table;
    }

    /**
     * Detect and report every branch and bus violation, then hand the voltage-angle limits back to
     * the manager. Mirrors {@code detectViolations(network)}: a disabled branch and a disabled bus
     * are skipped.
     */
    void detectViolations(LfNetwork network, LimitViolationManager manager) {
        LfBranch[] branches = table.branches();
        for (int b = 0; b < branches.length; b++) {
            int groupStart = table.groupStart(b);
            int groupEnd = table.groupEnd(b);
            if (groupStart == groupEnd) {
                continue;                                    // no limits anywhere on this branch
            }
            LfBranch branch = branches[b];
            if (branch.isDisabled()) {
                continue;
            }
            java.util.Arrays.fill(quantityCached, false);
            for (int g = groupStart; g < groupEnd; g++) {
                scanGroup(branch, g, manager);
            }
        }

        LfBus[] buses = table.buses();
        for (int i = 0; i < buses.length; i++) {
            LfBus bus = buses[i];
            if (bus.isDisabled()) {
                continue;
            }
            double v = bus.getV();
            // NaN limits make both comparisons false, exactly like OLF's !isNaN(limit) guard.
            if (v > table.busHighLimit(i)) {
                manager.addBusLimitViolation(LimitViolationManager.createBusLimitViolation(bus,
                        com.powsybl.contingency.violations.LimitViolationType.HIGH_VOLTAGE,
                        table.busHighLimit(i) * table.busScale(i), v * table.busScale(i)), bus);
            }
            if (v < table.busLowLimit(i)) {
                manager.addBusLimitViolation(LimitViolationManager.createBusLimitViolation(bus,
                        com.powsybl.contingency.violations.LimitViolationType.LOW_VOLTAGE,
                        table.busLowLimit(i) * table.busScale(i), v * table.busScale(i)), bus);
            }
        }

        manager.detectVoltageAngleViolations(network);
    }

    /**
     * Report the violations among a SCREEN's candidates, plus a full scan of whatever the screen
     * does not cover.
     *
     * <p>The screen may only over-emit (see {@link GpuViolationScreen}), so every candidate is
     * re-tested here with OLF's exact {@code >} against OLF's OWN value — the same
     * {@code branch.getI1().eval()} the stock detection reads. That is what makes the reported set
     * independent of the arithmetic that produced the candidates, on the host or on a device.
     *
     * <p>Candidates arrive in table order, so a group's entries are consecutive and most-severe
     * first: the first that confirms wins and the rest of that group is skipped, which is OLF's
     * {@code break}.
     *
     * <p>{@code sideOpened} (nullable, table indices) are branches the contingency opened at one side: the screen
     * evaluated them closed, so their candidates are dropped and they are scanned in full here instead — in place, at
     * their table position, so the violations come out in the stock detection's order.
     */
    void detectFromCandidates(LfNetwork network, LimitViolationManager manager, int[] candidates,
                              GpuViolationScreen screen, int[] uncoveredBranches, java.util.BitSet sideOpened) {
        LfBranch[] branches = table.branches();
        int reportedGroup = -1;
        int currentBranch = -1;
        int nextSideOpened = sideOpened != null ? sideOpened.nextSetBit(0) : -1;
        int i = 0;
        int busBase = table.limitCount();
        for (; i < candidates.length && candidates[i] < busBase; i++) {
            int limit = candidates[i];
            int group = table.limitGroup(limit);
            if (group == reportedGroup) {
                continue;                                    // this group already reported its most serious limit
            }
            int b = table.groupBranch(group);
            while (nextSideOpened >= 0 && nextSideOpened <= b) {
                scanBranch(nextSideOpened, manager);
                currentBranch = -1;
                nextSideOpened = sideOpened.nextSetBit(nextSideOpened + 1);
            }
            LfBranch branch = branches[b];
            if (branch.isDisabled() || sideOpened != null && sideOpened.get(b)) {
                continue;                                    // taken out by the contingency, or scanned in full above
            }
            if (b != currentBranch) {
                java.util.Arrays.fill(quantityCached, false);
                currentBranch = b;
            }
            if (confirm(branch, group, limit, manager)) {
                reportedGroup = group;
            }
        }
        for (; nextSideOpened >= 0; nextSideOpened = sideOpened.nextSetBit(nextSideOpened + 1)) {
            scanBranch(nextSideOpened, manager);
        }
        for (; i < candidates.length; i++) {
            int record = candidates[i];
            int busIndex = GpuViolationScreen.busOf(table, record);
            LfBus bus = table.buses()[busIndex];
            if (bus.isDisabled()) {
                continue;
            }
            double v = bus.getV();
            if (GpuViolationScreen.isHighVoltage(table, record)) {
                if (v > table.busHighLimit(busIndex)) {
                    manager.addBusLimitViolation(LimitViolationManager.createBusLimitViolation(bus,
                            com.powsybl.contingency.violations.LimitViolationType.HIGH_VOLTAGE,
                            table.busHighLimit(busIndex) * table.busScale(busIndex), v * table.busScale(busIndex)), bus);
                }
            } else if (v < table.busLowLimit(busIndex)) {
                manager.addBusLimitViolation(LimitViolationManager.createBusLimitViolation(bus,
                        com.powsybl.contingency.violations.LimitViolationType.LOW_VOLTAGE,
                        table.busLowLimit(busIndex) * table.busScale(busIndex), v * table.busScale(busIndex)), bus);
            }
        }
        for (int b : uncoveredBranches) {
            if (sideOpened == null || !sideOpened.get(b)) {
                scanBranch(b, manager);
            }
        }
        manager.detectVoltageAngleViolations(network);
    }

    /** Re-test one candidate exactly and report it if it holds. */
    private boolean confirm(LfBranch branch, int group, int limit, LimitViolationManager manager) {
        byte side = table.side(group);
        byte quantity = table.quantity(group);
        double value = quantity(branch, side, quantity);
        if (quantity == GpuLimitTable.QTY_S && Double.isNaN(value)) {
            return false;
        }
        double compared = quantity == GpuLimitTable.QTY_P ? Math.abs(value) : value;
        if (compared <= table.threshold(limit)) {
            return false;                                    // inside the band, but not actually violating
        }
        manager.addBranchLimitViolation(LimitViolationManager.createLimitViolation(branch,
                table.groupId(group), table.limit(limit), GpuLimitTable.violationType(quantity),
                table.scale(group), value, side == GpuTwoSides.ONE ? TwoSides.ONE : TwoSides.TWO));
        return true;
    }

    /** The full host scan of one branch's groups, as the stock detection walks them. */
    private void scanBranch(int b, LimitViolationManager manager) {
        LfBranch branch = table.branches()[b];
        if (branch.isDisabled()) {
            return;
        }
        java.util.Arrays.fill(quantityCached, false);
        for (int g = table.groupStart(b); g < table.groupEnd(b); g++) {
            scanGroup(branch, g, manager);
        }
    }

    private void scanGroup(LfBranch branch, int group, LimitViolationManager manager) {
        byte side = table.side(group);
        byte quantity = table.quantity(group);
        double value = quantity(branch, side, quantity);
        if (quantity == GpuLimitTable.QTY_S && Double.isNaN(value)) {
            return;                                          // apparent power is not relevant for fictitious branches
        }
        double compared = quantity == GpuLimitTable.QTY_P ? Math.abs(value) : value;
        for (int l = table.limitStart(group); l < table.limitEnd(group); l++) {
            if (compared > table.threshold(l)) {
                manager.addBranchLimitViolation(LimitViolationManager.createLimitViolation(branch,
                        table.groupId(group), table.limit(l), GpuLimitTable.violationType(quantity),
                        table.scale(group), value, side == GpuTwoSides.ONE ? TwoSides.ONE : TwoSides.TWO));
                return;                                      // most serious one only
            }
        }
    }

    private double quantity(LfBranch branch, byte side, byte quantity) {
        int slot = side * 3 + quantity;
        if (quantityCached[slot]) {
            return quantityCache[slot];
        }
        double value;
        if (side == GpuTwoSides.ONE) {
            value = switch (quantity) {
                case GpuLimitTable.QTY_I -> branch.getI1().eval();
                case GpuLimitTable.QTY_P -> branch.getP1().eval();
                default -> branch.computeApparentPower1();
            };
        } else {
            value = switch (quantity) {
                case GpuLimitTable.QTY_I -> branch.getI2().eval();
                case GpuLimitTable.QTY_P -> branch.getP2().eval();
                default -> branch.computeApparentPower2();
            };
        }
        quantityCache[slot] = value;
        quantityCached[slot] = true;
        return value;
    }
}
