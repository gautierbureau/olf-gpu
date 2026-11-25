/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.openloadflow.graph.GraphConnectivity;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LoadFlowModel;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The buses out of OLF's network at base that a strategy's switch close can bring back across an impedant branch, and
 * their device rows. OLF's equation system has no row for such a bus (its equations are inactive, so its variables are
 * out of the index), and that system is shared with the CPU paths, so it is never touched.
 *
 * <p>The island a close brings back is the one OLF's connectivity adds to the main component
 * (AbstractLfBranchAction.getNetworkActivations). One linked by switches alone is the electrical node of the switch's
 * bus, and the pairs merge it there (GpuStrategyComposer): it gets nothing here. One holding an impedant branch gets, for
 * each of its buses, two device rows after OLF's {@code n}: its BUS_PHI row and its BUS_V row, the pairing the stable
 * rows use, as OLF's own equations need them. Every lane holds them as fixed variables at the value OLF's post-action
 * Newton-Raphson would start the bus from ({@code PreviousValueVoltageInitializer(true)}: the bus's own voltage, or 1 pu
 * and 0 rad when it has none), out of OLF's equation count {@code nActive}; a pair bringing the island back turns them
 * into its equations.
 *
 * <p>The bus at the dead end of a closing switch is that switch's CHILD, in the coupler-tree relocation's layout
 * (CouplerTreeRelocation): its active and reactive power balances sit on the switch's DUMMY_P and DUMMY_Q rows, where the
 * switch's dummy flow is their diagonal, and the switch's ZERO_PHI and ZERO_V on its own rows. A switch open, its dummy
 * rows hold OLF's DUMMY_TARGET identities and the bus's rows their fixed values; closed, all four are equations. The
 * first close reaching the bus is its switch; another one is not packed.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class DeadBusRows {

    private static final DeadBusRows NONE = new DeadBusRows(0, List.of(), new int[0], Map.of());

    private final int olfRowCount;
    private final List<LfBus> buses;                       // the buses with rows, in row order
    private final int[] indexOfBusNum;                     // -1: not a bus with rows
    private final Map<LfBus, LfBranch> switchOf;           // child bus -> the closing switch whose dummy rows hold its balances

    private DeadBusRows(int olfRowCount, List<LfBus> buses, int[] indexOfBusNum, Map<LfBus, LfBranch> switchOf) {
        this.olfRowCount = olfRowCount;
        this.buses = buses;
        this.indexOfBusNum = indexOfBusNum;
        this.switchOf = switchOf;
    }

    public static DeadBusRows none() {
        return NONE;
    }

    /**
     * The rows of the buses each strategy's switch closes bring back, its closes taken together on OLF's connectivity at base
     * (a close with no end out of the network brings nothing back). A switch's child is the bus out of the network it
     * reaches first from the network's side, in the strategy's order; the first switch to reach a bus keeps it.
     */
    public static DeadBusRows of(LfNetwork network, int olfRowCount, Collection<? extends Collection<LfBranch>> closeGroups) {
        Set<LfBus> withRows = new LinkedHashSet<>();
        Map<LfBus, LfBranch> switchOf = new HashMap<>();
        GraphConnectivity<LfBus, LfBranch> connectivity = network.getConnectivity();
        LfBus slack = network.getSynchronousNetworks().getFirst().getSlackBuses().getFirst();
        for (Collection<LfBranch> group : closeGroups) {
            List<LfBranch> closes = group.stream().filter(b -> b.getBus1() != null && b.getBus2() != null
                    && (b.getBus1().isDisabled() || b.getBus2().isDisabled())).distinct().toList();
            if (closes.isEmpty()) {
                continue;
            }
            Set<LfBus> island;
            Set<LfBranch> edges;
            connectivity.startTemporaryChanges();
            try {
                connectivity.setMainComponentVertex(slack);
                closes.forEach(b -> connectivity.addEdge(b.getBus1(), b.getBus2(), b));
                island = new HashSet<>(connectivity.getVerticesAddedToMainComponent());
                edges = new HashSet<>(connectivity.getEdgesAddedToMainComponent());
            } finally {
                connectivity.undoTemporaryChanges();
                connectivity.setMainComponentVertex(slack);
            }
            if (island.isEmpty() || !needsRows(island, edges)) {
                continue;                                    // switches alone: merged by the pairs, no rows
            }
            withRows.addAll(island);
            assignChildren(closes, island, edges, switchOf);
        }
        if (withRows.isEmpty()) {
            return NONE;
        }
        List<LfBus> ordered = new ArrayList<>(withRows);
        ordered.sort((a, b) -> Integer.compare(a.getNum(), b.getNum()));    // rows in bus order, whatever the walk
        int[] indexOfBusNum = new int[network.getBuses().size()];
        Arrays.fill(indexOfBusNum, -1);
        for (int k = 0; k < ordered.size(); k++) {
            indexOfBusNum[ordered.get(k).getNum()] = k;
        }
        return new DeadBusRows(olfRowCount, Collections.unmodifiableList(ordered), indexOfBusNum, switchOf);
    }

    /** From the network's side outward: a close from a reached bus (one in the network, or of the island already reached) to
     *  one not reached yet makes that bus its child, and what the island's other branches join to it is reached with it. */
    private static void assignChildren(List<LfBranch> closes, Set<LfBus> island, Set<LfBranch> edges, Map<LfBus, LfBranch> switchOf) {
        Set<LfBus> reached = new HashSet<>();
        boolean progress = true;
        while (progress) {
            progress = false;
            for (LfBranch close : closes) {
                LfBus child = childOf(close, island, reached);
                if (child == null) {
                    continue;
                }
                switchOf.putIfAbsent(child, close);
                java.util.Deque<LfBus> queue = new java.util.ArrayDeque<>(List.of(child));
                reached.add(child);
                while (!queue.isEmpty()) {
                    LfBus bus = queue.poll();
                    for (LfBranch branch : bus.getBranches()) {
                        if (!edges.contains(branch) || closes.contains(branch) || branch.getBus1() == null || branch.getBus2() == null) {
                            continue;
                        }
                        LfBus other = branch.getBus1() == bus ? branch.getBus2() : branch.getBus1();
                        if (island.contains(other) && reached.add(other)) {
                            queue.add(other);
                        }
                    }
                }
                progress = true;
            }
        }
    }

    /** The island bus a close reaches from a reached end (a bus of the network, or one already reached), null if none. */
    private static LfBus childOf(LfBranch close, Set<LfBus> island, Set<LfBus> reached) {
        LfBus bus1 = close.getBus1();
        LfBus bus2 = close.getBus2();
        boolean from1 = !island.contains(bus1) || reached.contains(bus1);
        boolean from2 = !island.contains(bus2) || reached.contains(bus2);
        if (from1 && island.contains(bus2) && !reached.contains(bus2)) {
            return bus2;
        }
        if (from2 && island.contains(bus1) && !reached.contains(bus1)) {
            return bus1;
        }
        return null;
    }

    /**
     * Whether an island needs OLF's own equations on rows: it holds an impedant branch, a shunt or a branch open at one side
     * (OLF enables it with its bus: its charging there), whose terms depend on
     * their buses' own voltages, which OLF starts from each bus's own value (merged into the switch's bus, the first
     * Newton step would be evaluated elsewhere). Switches alone carry nothing that depends on a voltage.
     */
    private static boolean needsRows(Set<LfBus> island, Set<LfBranch> edges) {
        return edges.stream().anyMatch(b -> !b.isZeroImpedance(LoadFlowModel.AC))
                || island.stream().anyMatch(bus -> bus.getShunt().isPresent() || bus.getControllerShunt().isPresent()
                        || bus.getSvcShunt().isPresent() || bus.getBranches().stream().anyMatch(b -> !b.isConnectedAtBothSides()));
    }

    public boolean isEmpty() {
        return buses.isEmpty();
    }

    /** OLF's row count: the rows of its equation system's index. */
    public int olfRowCount() {
        return olfRowCount;
    }

    /** The device's row count: OLF's rows and two per bus with rows. */
    public int rowCount() {
        return olfRowCount + 2 * buses.size();
    }

    /** The buses with rows, in row order. */
    public List<LfBus> buses() {
        return buses;
    }

    /** The closing switch whose DUMMY_P / DUMMY_Q rows hold the bus's balances (its child), null for any other bus. */
    public LfBranch switchOf(LfBus bus) {
        return switchOf.get(bus);
    }

    /** Whether a branch end is on the device: a bus of the network, or a bus with rows here. */
    public boolean onDevice(LfBus bus) {
        return bus != null && (!bus.isDisabled() || phiRow(bus) >= 0);
    }

    /** The bus of a branch open at one side whose open-branch element sits on rows past OLF's: its connected end, when that
     *  bus has rows here; null otherwise. */
    public LfBus oneSideBusWithRows(LfBranch branch) {
        LfBus bus1 = branch.getBus1();
        LfBus bus2 = branch.getBus2();
        if (!branch.isDisabled() || branch.isZeroImpedance(LoadFlowModel.AC) || (bus1 == null) == (bus2 == null)) {
            return null;
        }
        LfBus bus = bus1 != null ? bus1 : bus2;
        return bus.isDisabled() && phiRow(bus) >= 0 ? bus : null;
    }

    /** Whether a branch with both ends is packed: both ends on the device ({@link #onDevice}). */
    public boolean packable(LfBranch branch) {
        return onDevice(branch.getBus1()) && onDevice(branch.getBus2());
    }

    /** The bus's BUS_PHI row (its active power balance, or its switch's ZERO_PHI for a child), -1 when it has none here. */
    public int phiRow(LfBus bus) {
        int k = bus.getNum() < indexOfBusNum.length ? indexOfBusNum[bus.getNum()] : -1;
        return k < 0 ? -1 : olfRowCount + 2 * k;
    }

    /** The bus's BUS_V row (its reactive power balance, or its switch's ZERO_V for a child), -1 when it has none here. */
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

    /** The array lengthened to {@link #rowCount()}: OLF's rows as they are, the added BUS_PHI and BUS_V rows at their
     *  start values (a state vector, a target vector). */
    public double[] withStartValues(double[] olfRows) {
        if (buses.isEmpty()) {
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
        if (buses.isEmpty()) {
            return olfRows;
        }
        int[] out = Arrays.copyOf(olfRows, rowCount());
        Arrays.fill(out, olfRows.length, out.length, fill);
        return out;
    }

    /** The array lengthened to {@link #rowCount()}, the added rows at {@code fill}. */
    public double[] lengthened(double[] olfRows, double fill) {
        if (buses.isEmpty()) {
            return olfRows;
        }
        double[] out = Arrays.copyOf(olfRows, rowCount());
        Arrays.fill(out, olfRows.length, out.length, fill);
        return out;
    }
}
