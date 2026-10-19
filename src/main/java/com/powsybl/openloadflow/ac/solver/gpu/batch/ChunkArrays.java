/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.BatchContingency;

import java.util.List;

/**
 * Value helpers of the chunk solver: per-slot arrays flattened and split, list and array concatenation, a pair
 * slot's parent, a generator's distributed-slack slot, and the status and full state of a result row.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class ChunkArrays {

    private ChunkArrays() {
    }

    /** The DS generator slots one scenario's outage removes, out of the flat (scenario, slot) pairs.
     *  The host seed needs the same identity the kernel has, or the two model the outaged bus
     *  differently and the seed stops being the function it is meant to replay. */
    static int[] lostSlotsOf(java.util.List<Integer> pairs, int scenario) {
        java.util.List<Integer> out = new java.util.ArrayList<>();
        for (int k = 0; k + 1 < pairs.size(); k += 2) {
            if (pairs.get(k) == scenario) {
                out.add(pairs.get(k + 1));
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    static boolean contains(int[] values, int v) {
        for (int x : values) {
            if (x == v) {
                return true;
            }
        }
        return false;
    }

    static int countTrue(boolean[] flags) {
        if (flags == null) {
            return 0;
        }
        int n = 0;
        for (boolean f : flags) {
            if (f) {
                n++;
            }
        }
        return n;
    }

    /** The per-scenario reactive-limit V-target overrides of the chunk plus a decision's own, as one array. */
    static int[] concatInts(List<Integer> ours, int[] theirs) {
        if (ours.isEmpty()) {
            return theirs;
        }
        int[] a = ours.stream().mapToInt(Integer::intValue).toArray();
        if (theirs == null || theirs.length == 0) {
            return a;
        }
        int[] out = java.util.Arrays.copyOf(a, a.length + theirs.length);
        System.arraycopy(theirs, 0, out, a.length, theirs.length);
        return out;
    }

    static double[] concatDoubles(List<Double> ours, double[] theirs) {
        if (ours.isEmpty()) {
            return theirs;
        }
        double[] a = ours.stream().mapToDouble(Double::doubleValue).toArray();
        if (theirs == null || theirs.length == 0) {
            return a;
        }
        double[] out = java.util.Arrays.copyOf(a, a.length + theirs.length);
        System.arraycopy(theirs, 0, out, a.length, theirs.length);
        return out;
    }

    /** One row per slot as a flat {@code s * width + k} array (a null row is zeros); null when there are no rows. */
    static double[] flat(double[][] rows, int slots, int width) {
        if (rows == null) {
            return null;
        }
        double[] out = new double[slots * width];
        for (int s = 0; s < slots && s < rows.length; s++) {
            if (rows[s] != null) {
                System.arraycopy(rows[s], 0, out, s * width, width);
            }
        }
        return out;
    }

    /** A flat {@code s * width + k} array as one row per slot. */
    static double[][] perSlot(double[] flat, int slots, int width) {
        double[][] rows = new double[slots][];
        for (int s = 0; s < slots; s++) {
            rows[s] = java.util.Arrays.copyOfRange(flat, s * width, (s + 1) * width);
        }
        return rows;
    }

    /** The index of a pair slot's parent contingency, or -1 (padding slot, or no parent). */
    static int pIdxOf(List<BatchContingency> contingencies, List<Integer> group, int chunkStart, int s,
                              java.util.Map<String, Integer> indexById, int realCount) {
        if (s >= realCount) {
            return -1;
        }
        String pairId = contingencies.get(group.get(chunkStart + s)).contingencyId();
        int sep = pairId.indexOf(BatchContingency.STRATEGY_KEY_SEP);
        Integer p = sep < 0 ? null : indexById.get(pairId.substring(0, sep));
        return p == null ? -1 : p;
    }

    /** The generator's slot in the distributed-slack generator pack, or -1 when it does not participate. */
    static int dgSlotOf(GpuAcData gpuData, String genId) {
        String[] ids = gpuData.dgGenId();
        for (int g = 0; ids != null && g < ids.length; g++) {
            if (ids[g].equals(genId)) {
                return g;
            }
        }
        return -1;
    }

    /** The status of a result row: the full row's, or the tail's when the state went to the sink. */
    static int rowStatus(double[] row, int n) {
        return row.length == 3 ? (int) row[2] : (int) row[n + 2];
    }

    /** The full result row (state + tail), read back from the sink when the state was sunk. */
    static double[] fullRow(double[] row, int n, java.util.Map<String, double[]> sink, String contingencyId) {
        if (row.length != 3) {
            return row;
        }
        double[] state = sink.get(contingencyId);
        double[] full = java.util.Arrays.copyOf(state, n + 3);
        full[n] = row[0];
        full[n + 1] = row[1];
        full[n + 2] = row[2];
        return full;
    }

    /** The list as an array, or null when it is empty (a native setter clears its channel on null). */
    static int[] intsOrNull(List<Integer> values) {
        return values.isEmpty() ? null : values.stream().mapToInt(Integer::intValue).toArray();
    }

    /** The list as an array, or null when it is empty. */
    static double[] doublesOrNull(List<Double> values) {
        return values.isEmpty() ? null : values.stream().mapToDouble(Double::doubleValue).toArray();
    }
}
