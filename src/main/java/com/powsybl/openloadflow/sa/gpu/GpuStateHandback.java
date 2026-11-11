/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.openloadflow.ac.solver.gpu.batch.StateHandback;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.BatchContingency;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfNetwork;
import gnu.trove.list.array.TIntArrayList;

import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Predicate;

/**
 * Screens each scenario a solve call hands back on the device, where its state still is, and keeps what the host will
 * answer it with (P4 of {@code docs/device-final-results-plan.md}): its candidates, and, for a scenario the host answers
 * from its state (P1, P2), the state rows that answer reads (a {@link GpuGatheredState}), instead of its full state.
 *
 * <p>Those rows: the branches of its branch candidates and the buses of its bus candidates, the buses outside the
 * realistic-voltage band (the screen's realistic records: the host's check needs no other bus), the voltage-angle limits'
 * buses and the branches the screen does not cover (both for every scenario), and the branches the scenario opens at one
 * side, retaps or switches (the host scans those in full where the screen does not cover them). Every other row of the
 * host's scratch state is NaN.
 *
 * <p>A full state is read back for a scenario the host does not answer from its state ({@code sparse} false), one whose
 * candidates overflowed the buffer (the host screens the full state), and every converged scenario while
 * {@code compare} (the comparison answers each one from both).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuStateHandback implements StateHandback {

    private final GpuViolationScreen screen;
    private final GpuLimitTable table;
    private final GpuStateView view;
    private final LfNetwork network;
    private final Map<String, BatchContingency> scenarios;
    private final Predicate<String> sparse;
    private final boolean compare;
    private final boolean realistic;
    private final double minRealisticV;
    private final double maxRealisticV;
    private final BitSet everyScenarioRows = new BitSet();

    /** Every screened scenario's candidates (branch and bus records, sorted); absent for an overflow. */
    final Map<String, int[]> candidates = new HashMap<>();
    /** The rows of every scenario answered from them alone. */
    final Map<String, GpuGatheredState> gathered = new HashMap<>();
    /** What the handback read: rows gathered, full states read back. */
    private long rowsGathered;
    private long fullStatesRead;

    GpuStateHandback(GpuViolationScreen screen, GpuLimitTable table, GpuStateView view, LfNetwork network,
                     Map<String, BatchContingency> scenarios, Predicate<String> sparse, boolean compare, boolean realistic,
                     double minRealisticV, double maxRealisticV) {
        this.screen = screen;
        this.table = table;
        this.view = view;
        this.network = network;
        this.scenarios = scenarios;
        this.sparse = sparse;
        this.compare = compare;
        this.realistic = realistic;
        this.minRealisticV = minRealisticV;
        this.maxRealisticV = maxRealisticV;
        for (LfNetwork.LfVoltageAngleLimit limit : network.getVoltageAngleLimits()) {
            view.rowsOf(limit.getFrom(), everyScenarioRows::set);
            view.rowsOf(limit.getTo(), everyScenarioRows::set);
        }
        LfBranch[] branches = table.branches();
        for (int b = 0; b < branches.length; b++) {
            if (!screen.covers(b) && table.groupStart(b) != table.groupEnd(b)) {
                view.rowsOf(branches[b], everyScenarioRows::set);
            }
        }
    }

    @Override
    public boolean[] afterSolve(long batchHandle, String[] ids, boolean[] converged, int[][] automationSwitched, double[][] tapMoves) {
        int count = ids.length;
        boolean[] full = new boolean[count];
        GpuScreenScenario[] screened = new GpuScreenScenario[count];
        for (int s = 0; s < count; s++) {
            screened[s] = converged[s] ? screen.scenario(network, scenarios.get(ids[s]), automationSwitched[s], tapMoves[s])
                    : GpuScreenScenario.BASE;
        }
        int[][] records = screen.screenHandedBack(batchHandle, screened, realistic, minRealisticV, maxRealisticV);
        int realisticBase = screen.realisticRecordBase();
        int busBase = table.limitCount();
        TIntArrayList gatherScenario = new TIntArrayList();
        TIntArrayList gatherRow = new TIntArrayList();
        int[] start = new int[count];
        int[] end = new int[count];
        boolean[] answeredFromRows = new boolean[count];
        for (int s = 0; s < count; s++) {
            start[s] = gatherRow.size();
            end[s] = start[s];
            if (!converged[s]) {
                continue;                                    // nothing the host reads: exhausted (no result) or the CPU's
            }
            int[] scenarioRecords = records != null ? records[s] : null;
            if (scenarioRecords == null) {
                full[s] = true;                              // an overflow, or the device refused: the host screens it
                continue;
            }
            int split = 0;
            while (split < scenarioRecords.length && scenarioRecords[split] < realisticBase) {
                split++;
            }
            candidates.put(ids[s], Arrays.copyOf(scenarioRecords, split));
            boolean sparseScenario = sparse.test(ids[s]);
            full[s] = !sparseScenario || compare;
            if (!sparseScenario) {
                continue;
            }
            answeredFromRows[s] = true;
            BitSet rows = (BitSet) everyScenarioRows.clone();
            for (int k = 0; k < scenarioRecords.length; k++) {
                int record = scenarioRecords[k];
                if (record < busBase) {
                    view.rowsOf(table.branches()[table.groupBranch(table.limitGroup(record))], rows::set);
                } else if (record < realisticBase) {
                    view.rowsOf(table.buses()[(record - busBase) / 2], rows::set);
                } else {
                    view.rowsOf(table.buses()[record - realisticBase], rows::set);
                }
            }
            changedBranchRows(scenarios.get(ids[s]), automationSwitched[s], tapMoves[s], rows);
            for (int row = rows.nextSetBit(0); row >= 0; row = rows.nextSetBit(row + 1)) {
                gatherScenario.add(s);
                gatherRow.add(row);
            }
            end[s] = gatherRow.size();
        }
        double[] values = gatherRow.isEmpty() ? new double[0]
                : screen.gatherHandedBack(batchHandle, gatherScenario.toArray(), gatherRow.toArray());
        rowsGathered += gatherRow.size();
        int[] allRows = gatherRow.toArray();
        for (int s = 0; s < count; s++) {
            if (!answeredFromRows[s]) {
                continue;
            }
            if (values == null) {
                full[s] = true;                              // the device refused: the host reads the full state
                continue;
            }
            // a scenario with nothing to gather (no candidate, nothing out of the realistic band) has no row to read
            gathered.put(ids[s], new GpuGatheredState(Arrays.copyOfRange(allRows, start[s], end[s]),
                    Arrays.copyOfRange(values, start[s], end[s])));
        }
        return full;
    }

    /** The rows of the branches a scenario opens at one side, retaps or switches: the host scans those in full where the
     *  screen does not cover them. */
    private void changedBranchRows(BatchContingency contingency, int[] automationSwitched, double[] tapMoves, BitSet rows) {
        if (contingency != null) {
            for (BatchContingency.SideOpen sideOpen : contingency.sideOpenBranches()) {
                view.rowsOf(sideOpen.branch(), rows::set);
            }
            for (var tap : contingency.tapOverrides()) {
                view.rowsOf(tap.branch(), rows::set);
            }
            for (LfBranch branch : contingency.reclosedBranches()) {
                view.rowsOf(branch, rows::set);
            }
        }
        if (automationSwitched != null) {
            for (int num : automationSwitched) {
                view.rowsOf(network.getBranch(num), rows::set);
            }
        }
        if (tapMoves != null) {
            for (int k = 0; k + 2 < tapMoves.length; k += 3) {
                view.rowsOf(network.getBranch((int) tapMoves[k]), rows::set);
            }
        }
    }

    @Override
    public double[][] fullStates(long batchHandle, int[] which, int n) {
        double[][] states = screen.readHandedBackStates(batchHandle, which, n);
        if (states == null) {
            throw new IllegalStateException("The handed-back states could not be read back from the device");
        }
        fullStatesRead += which.length;
        return states;
    }

    /** Frees the screen's device table. */
    void close() {
        screen.closeDevice();
    }

    /** One line, numbers only: what was screened on the device and what came back. */
    String summary() {
        return "STATE_HANDBACK screened on the device=" + screen.onDevice() + " overflowed=" + screen.overflows()
                + " candidates=" + candidates.values().stream().mapToLong(c -> c.length).sum()
                + " answered from rows=" + gathered.size() + " rows gathered=" + rowsGathered
                + " full states read=" + fullStatesRead;
    }
}
