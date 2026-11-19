/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch.plan;

import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * The device rows of the buses out of OLF's network at base that a strategy's switch close can bring back. OLF's
 * equation system has no row for such a bus (its equations are inactive, so its variables are out of the index), and
 * that system is shared with the CPU paths, so it is never touched: the device appends two rows per bus after OLF's
 * {@code n}, the bus's BUS_PHI row (its active power balance) and its BUS_V row (its reactive power balance), the
 * pairing the stable rows use. Every lane holds them as fixed variables at the value OLF's post-action Newton-Raphson
 * would start the bus from ({@code PreviousValueVoltageInitializer(true)}: the bus's own voltage, or 1 pu and 0 rad
 * when it has none); OLF's equation count {@code nActive} leaves them out.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class DeadBusRows {

    private static final DeadBusRows NONE = new DeadBusRows(0, List.of(), new int[0]);

    private final int olfRowCount;
    private final List<LfBus> buses;
    private final int[] indexOfBusNum;                     // -1: not a dead bus with rows

    private DeadBusRows(int olfRowCount, List<LfBus> buses, int[] indexOfBusNum) {
        this.olfRowCount = olfRowCount;
        this.buses = buses;
        this.indexOfBusNum = indexOfBusNum;
    }

    public static DeadBusRows none() {
        return NONE;
    }

    /**
     * The dead buses the given branches reach: a branch with an end out of the network leads to that end, and from there
     * every branch with both ends present leads to the other end while it is out of the network too (the island OLF's
     * connectivity enables when the branch closes).
     */
    public static DeadBusRows of(LfNetwork network, int olfRowCount, Collection<LfBranch> closableBranches) {
        Set<LfBus> reached = new LinkedHashSet<>();
        Deque<LfBus> queue = new ArrayDeque<>();
        for (LfBranch branch : closableBranches) {
            for (LfBus end : Arrays.asList(branch.getBus1(), branch.getBus2())) {
                if (end != null && end.isDisabled() && reached.add(end)) {
                    queue.add(end);
                }
            }
        }
        while (!queue.isEmpty()) {
            LfBus bus = queue.poll();
            for (LfBranch branch : bus.getBranches()) {
                LfBus other = branch.getBus1() == bus ? branch.getBus2() : branch.getBus1();
                if (other != null && other.isDisabled() && reached.add(other)) {
                    queue.add(other);
                }
            }
        }
        if (reached.isEmpty()) {
            return NONE;
        }
        List<LfBus> ordered = new ArrayList<>(reached);
        ordered.sort((a, b) -> Integer.compare(a.getNum(), b.getNum()));    // rows in bus order, whatever the walk
        int[] indexOfBusNum = new int[network.getBuses().size()];
        Arrays.fill(indexOfBusNum, -1);
        for (int k = 0; k < ordered.size(); k++) {
            indexOfBusNum[ordered.get(k).getNum()] = k;
        }
        return new DeadBusRows(olfRowCount, Collections.unmodifiableList(ordered), indexOfBusNum);
    }

    public boolean isEmpty() {
        return buses.isEmpty();
    }

    /** OLF's row count: the rows of its equation system's index. */
    public int olfRowCount() {
        return olfRowCount;
    }

    /** The device's row count: OLF's rows and two per dead bus. */
    public int rowCount() {
        return olfRowCount + 2 * buses.size();
    }

    public List<LfBus> buses() {
        return buses;
    }

    /** The bus's BUS_PHI row (its active power balance), -1 when it has none here. */
    public int phiRow(LfBus bus) {
        int k = bus.getNum() < indexOfBusNum.length ? indexOfBusNum[bus.getNum()] : -1;
        return k < 0 ? -1 : olfRowCount + 2 * k;
    }

    /** The bus's BUS_V row (its reactive power balance), -1 when it has none here. */
    public int vRow(LfBus bus) {
        int k = bus.getNum() < indexOfBusNum.length ? indexOfBusNum[bus.getNum()] : -1;
        return k < 0 ? -1 : olfRowCount + 2 * k + 1;
    }

    /** Every row this adds, in row order. */
    public int[] rows() {
        int[] rows = new int[2 * buses.size()];
        for (int r = 0; r < rows.length; r++) {
            rows[r] = olfRowCount + r;
        }
        return rows;
    }

    /** OLF's post-action start value of the bus's voltage magnitude (pu): its own, or 1 when it has none. */
    public static double startV(LfBus bus) {
        double v = bus.getV();
        return Double.isNaN(v) ? 1 : v;
    }

    /** OLF's post-action start value of the bus's voltage angle (rad): its own, or 0 when it has none. */
    public static double startPhi(LfBus bus) {
        double phi = bus.getAngle();
        return Double.isNaN(phi) ? 0 : phi;
    }

    /** The array lengthened to {@link #rowCount()}: OLF's rows as they are, the dead buses' BUS_PHI and BUS_V rows at their
     *  start values (a state vector, a target vector). */
    public double[] withStartValues(double[] olfRows) {
        if (isEmpty()) {
            return olfRows;
        }
        double[] out = Arrays.copyOf(olfRows, rowCount());
        for (LfBus bus : buses) {
            out[phiRow(bus)] = startPhi(bus);
            out[vRow(bus)] = startV(bus);
        }
        return out;
    }

    /** The array lengthened to {@link #rowCount()}, the added rows at {@code fill}. */
    public int[] lengthened(int[] olfRows, int fill) {
        if (isEmpty()) {
            return olfRows;
        }
        int[] out = Arrays.copyOf(olfRows, rowCount());
        Arrays.fill(out, olfRows.length, out.length, fill);
        return out;
    }

    /** The array lengthened to {@link #rowCount()}, the added rows at {@code fill}. */
    public double[] lengthened(double[] olfRows, double fill) {
        if (isEmpty()) {
            return olfRows;
        }
        double[] out = Arrays.copyOf(olfRows, rowCount());
        Arrays.fill(out, olfRows.length, out.length, fill);
        return out;
    }
}
