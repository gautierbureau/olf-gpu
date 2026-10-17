/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Buses an islanding contingency cuts off: their rows frozen to a flat identity state, and the distributed-slack
 * participants and reactive-limit controllers they take out of the solve.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class IslandedBuses {

    static final int[] NO_FROZEN_ROWS = new int[0];
    static final double[] NO_FROZEN_TARGETS = new double[0];
    private static final int[] NO_ISLANDED_DS = new int[0];

    private IslandedBuses() {
    }

    /** The islanded buses' variable rows to freeze + the clean flat value each holds (phi=0, V=1 pu). */
    record FrozenIsland(int[] rows, double[] targets) { }

    /**
     * The variable rows (BUS_PHI + BUS_V) of every islanded bus — frozen to identity in a fragmenting
     * scenario's row mode — paired with the clean flat value each is HELD at: {@code phi=0}, {@code V=1} pu.
     * With the outaged branch (the only island↔main link) already masked, freezing these rows makes the
     * islanded sub-block a benign identity block that cannot make the batched factorization singular, while
     * the main component solves correctly. Holding the island at a flat 1∠0 (rather than the base equation
     * target, a P/Q value that produces a nonsensical islanded voltage/angle) gives the disconnected buses a
     * well-defined reported state — no garbage voltage, no spurious limit violations — without touching the main.
     */
    static FrozenIsland islandedFrozen(EquationSystem<AcVariableType, AcEquationType> es, List<LfBus> islandedBuses,
                                       Set<LfBranch> relocatedCouplers) {
        if (islandedBuses.isEmpty()) {
            return new FrozenIsland(NO_FROZEN_ROWS, NO_FROZEN_TARGETS);
        }
        List<Integer> rows = new ArrayList<>(islandedBuses.size() * 2);
        List<Double> targets = new ArrayList<>(islandedBuses.size() * 2);
        for (LfBus bus : islandedBuses) {
            addFrozenVarRow(es, bus.getNum(), AcVariableType.BUS_PHI, 0.0, rows, targets);   // angle 0
            addFrozenVarRow(es, bus.getNum(), AcVariableType.BUS_V, 1.0, rows, targets);     // voltage 1 pu
        }
        freezeRelocatedCouplerRows(es, islandedBuses, relocatedCouplers, rows, targets);
        int[] outRows = new int[rows.size()];
        double[] outTargets = new double[targets.size()];
        for (int i = 0; i < outRows.length; i++) {
            outRows[i] = rows.get(i);
            outTargets[i] = targets.get(i);
        }
        return new FrozenIsland(outRows, outTargets);
    }

    /**
     * The extra identity rows an island needs under the COUPLER-TREE RELOCATION. Relocated, a coupler inside the
     * island hosts the child bus's P/Q balance on its own {@code DUMMY_P} / {@code DUMMY_Q} rows, while the
     * coupler's {@code ZERO_V} / {@code ZERO_PHI} moved onto that bus's V / PHI rows — which the freeze above turns
     * into identities, erasing them. Freezing the coupler's dummy rows as well makes EVERY row of the island an
     * identity again, which is what an island wants (it has no equation left to satisfy): without it the child
     * bus's power balance is stranded on a dummy row nobody pins, and the island cannot converge.
     *
     * <p>Only couplers with BOTH endpoints islanded are frozen here; the boundary coupler the contingency opened
     * already has its dummy rows pinned by the chunk solver's zi-outage path. The set is empty when the relocation
     * is off, so the default path keeps its exact row list (and with it {@code nActive}).
     */
    static void freezeRelocatedCouplerRows(EquationSystem<AcVariableType, AcEquationType> es,
                                                   List<LfBus> islandedBuses, Set<LfBranch> relocatedCouplers,
                                                   List<Integer> rows, List<Double> targets) {
        if (relocatedCouplers.isEmpty()) {
            return;
        }
        Set<Integer> islandedNums = islandedBuses.stream().map(LfBus::getNum)
                .collect(java.util.stream.Collectors.toSet());
        Set<LfBranch> done = new java.util.HashSet<>();
        for (LfBus bus : islandedBuses) {
            for (LfBranch br : bus.getBranches()) {
                if (relocatedCouplers.contains(br) && br.getBus1() != null && br.getBus2() != null
                        && islandedNums.contains(br.getBus1().getNum())
                        && islandedNums.contains(br.getBus2().getNum())
                        && done.add(br)) {
                    addFrozenVarRow(es, br.getNum(), AcVariableType.DUMMY_P, 0.0, rows, targets);
                    addFrozenVarRow(es, br.getNum(), AcVariableType.DUMMY_Q, 0.0, rows, targets);
                }
            }
        }
    }

    static void addFrozenVarRow(EquationSystem<AcVariableType, AcEquationType> es, int busNum,
                                        AcVariableType type, double target, List<Integer> rows, List<Double> targets) {
        Variable<AcVariableType> v = es.getVariableSet().getVariable(busNum, type);
        if (v != null && v.getRow() >= 0) {
            rows.add(v.getRow());
            targets.add(target);
        }
    }

    /** The distributed-slack participants whose P-target row this contingency froze — i.e. those it
     *  put in another synchronous component. OLF distributes over
     *  {@code getSynchronousNetworks().getFirst()}, so these are simply not participants for this
     *  scenario, and the device renormalizes the remaining factors over the survivors. */
    static int[] islandedDsParticipants(int[] frozenRows, GpuAcData gpuData, List<LfBus> islandedBuses) {
        int[] dsPhiRow = gpuData.dsPhiRow();
        if (dsPhiRow == null || dsPhiRow.length == 0 || frozenRows.length == 0) {
            return NO_ISLANDED_DS;
        }
        // Match on the BUS as well as on the row. The row test alone misses a participant whose P row was
        // RELOCATED by the coupler-tree relocation: dsPhiRow carries pRow, which for a relocated bus is its
        // parent coupler's DUMMY_P, while the frozen rows are the bus's own BUS_PHI / BUS_V. The device then
        // kept distributing slack to a generator on a DEAD bus — measured on BUS-6014_BBS_2 as 7.386 MW of
        // distributed active power against the CPU, with every bus in the network shifted 2-4e-2 kV in one
        // direction, while the scheduled loss itself agreed exactly (both 0.0 pu).
        java.util.Set<Integer> islandedNums = new java.util.HashSet<>();
        for (LfBus b : islandedBuses) {
            islandedNums.add(b.getNum());
        }
        int[] dsBusNum = gpuData.dsBusNum();
        int[] found = new int[dsPhiRow.length];
        int count = 0;
        for (int i = 0; i < dsPhiRow.length; i++) {
            boolean islanded = dsBusNum != null && i < dsBusNum.length && islandedNums.contains(dsBusNum[i]);
            for (int frozen : frozenRows) {
                if (frozen == dsPhiRow[i]) {
                    islanded = true;
                    break;
                }
            }
            if (islanded) {
                found[count++] = i;
            }
            if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                System.err.printf("QUALIFY_TRACE dsPart i=%d row=%d busNum=%s islanded=%b%n", i, dsPhiRow[i],
                        dsBusNum != null && i < dsBusNum.length ? Integer.toString(dsBusNum[i]) : "-", islanded);
            }
        }
        return count == 0 ? NO_ISLANDED_DS : java.util.Arrays.copyOf(found, count);
    }

    /** The reactive-limits controllers this contingency puts in another synchronous component: their
     *  BUS_V row is one of the frozen island rows. OLF disables those buses, so its ReactiveLimits
     *  loop never sees them; the device must not either, or it pins a PV bus OLF leaves alone. */
    static int[] islandedRlControllers(int[] frozenRows, GpuAcData gpuData) {
        int[] rlVrow = gpuData.rlVrow();
        if (rlVrow == null || rlVrow.length == 0 || frozenRows.length == 0) {
            return NO_ISLANDED_DS;
        }
        int[] found = new int[rlVrow.length];
        int count = 0;
        for (int i = 0; i < rlVrow.length; i++) {
            for (int frozen : frozenRows) {
                if (frozen == rlVrow[i]) {
                    found[count++] = i;
                    break;
                }
            }
        }
        return count == 0 ? NO_ISLANDED_DS : java.util.Arrays.copyOf(found, count);
    }
}
