/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.iidm.network.TwoSides;
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
                        table.scale(group), value, side == 1 ? TwoSides.ONE : TwoSides.TWO));
                return;                                      // most serious one only
            }
        }
    }

    private double quantity(LfBranch branch, byte side, byte quantity) {
        int slot = (side - 1) * 3 + quantity;
        if (quantityCached[slot]) {
            return quantityCache[slot];
        }
        double value;
        if (side == 1) {
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
