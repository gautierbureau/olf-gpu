/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.outerloop.gpu.ScenarioRowModes;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.hash.TIntIntHashMap;

import java.util.Arrays;

/**
 * The row modes of the scenarios one solve call installs, kept as the rows each slot sets over the base row.
 *
 * <p>A scenario sets a few rows: a lost generator's bus turned PQ, a coupler's identity rows, its islanded buses. Each
 * slot used to get a copy of the whole base row as soon as it set one, 25,126 copies of 35,374 ints per run of
 * network.xiidm's full list (3.5 GB), which the solve call then read back to find the rows that differ. Here a slot holds
 * only the rows it set, and {@link #entries} hands the device the ones that differ from the base.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuBatchRowModes implements ScenarioRowModes {

    /** The no-entry value of the per-slot maps: no row mode is negative. */
    private static final int UNSET = Integer.MIN_VALUE;

    private final int[] baseRowMode;
    /** Per slot, row -> the mode it set; null while the slot is on the base row. */
    private final TIntIntHashMap[] setRows;

    GpuBatchRowModes(int slots, int[] baseRowMode) {
        this.baseRowMode = baseRowMode;
        this.setRows = new TIntIntHashMap[slots];
    }

    @Override
    public int get(int scenario, int row) {
        TIntIntHashMap rows = setRows[scenario];
        if (rows != null) {
            int mode = rows.get(row);
            if (mode != UNSET) {
                return mode;
            }
        }
        return baseRowMode[row];
    }

    @Override
    public void set(int scenario, int row, int mode) {
        TIntIntHashMap rows = setRows[scenario];
        if (rows == null) {
            rows = new TIntIntHashMap(8, 0.5f, -1, UNSET);
            setRows[scenario] = rows;
        }
        rows.put(row, mode);
    }

    /** Put slot {@code scenario} back on the base row. */
    void reset(int scenario) {
        setRows[scenario] = null;
    }

    /**
     * The (slot, row, mode) entries of slots {@code [0, slots)} whose mode differs from the base row, slot then row
     * ascending: what the device writes over the base row it copies into every installed slot. A slot has at most one
     * entry per row.
     */
    int[] entries(int slots) {
        TIntArrayList entries = new TIntArrayList();
        for (int s = 0; s < slots; s++) {
            TIntIntHashMap rows = setRows[s];
            if (rows == null) {
                continue;
            }
            int[] setRowsOfSlot = rows.keys();
            Arrays.sort(setRowsOfSlot);
            for (int row : setRowsOfSlot) {
                int mode = rows.get(row);
                if (mode != baseRowMode[row]) {
                    entries.add(s);
                    entries.add(row);
                    entries.add(mode);
                }
            }
        }
        return entries.toArray();
    }
}
