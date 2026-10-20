/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch.plan;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.network.LfBus;

/**
 * Rows that move under relocation, read per scenario: the pilot-Q relocation candidates, the coupler-tree children's
 * rows, the reactive-limit entry of a bus and whether it is pinned in the base.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class RelocatedRows {

    private RelocatedRows() {
    }

    /** The kernels map candidate c of qr group g to dq member qrMemberStart[g] + c - 1: true when every packed
     *  member with a row sits at that candidate (merged controller order on both sides). */
    static boolean qrMemberOrderConsistent(GpuAcData gpuData) {
        int[] start = qrMemberStart(gpuData);
        int[] qrPtr = gpuData.reactiveRelocation().groupStart();
        int[] qrRows = gpuData.reactiveRelocation().candidateRows();
        int[] mRow = gpuData.distrQMembers().row();
        byte[] mHasRow = gpuData.distrQMembers().hasRow();
        int[] mQrGroup = gpuData.distrQMembers().relocationGroup();
        if (start == null || qrPtr == null || mRow == null || mQrGroup == null) {
            return false;
        }
        for (int m = 0; m < mQrGroup.length; m++) {
            int g = mQrGroup[m];
            if (g < 0 || mHasRow[m] == 0 || mRow[m] < 0) {
                continue;
            }
            int c = m - start[g] + 1;
            if (c < 1 || qrPtr[g] + c >= qrPtr[g + 1] || qrRows[qrPtr[g] + c] != mRow[m]) {
                return false;
            }
        }
        return true;
    }

    /** qr group -> its first dq member, -1 for a group without packed members. */
    public static int[] qrMemberStart(GpuAcData gpuData) {
        int[] mQrGroup = gpuData.distrQMembers().relocationGroup();
        int[] qrPtr = gpuData.reactiveRelocation().groupStart();
        if (mQrGroup == null || qrPtr == null || qrPtr.length < 2) {
            return null;
        }
        int[] start = new int[qrPtr.length - 1];
        java.util.Arrays.fill(start, -1);
        for (int m = 0; m < mQrGroup.length; m++) {
            int g = mQrGroup[m];
            if (g >= 0 && g < start.length && start[g] < 0) {
                start[g] = m;
            }
        }
        return start;
    }

    /** The bus's reactive-limit entry, or -1: a re-homed bus is checked as a LOCAL controller by the device
     *  (OLF checks it as the PV bus it now is; BUS-6113_BBS_1 pinned VL-6113_1 at RL#1, the device never
     *  looked — 7.1 kV). */
    public static int rlEntryOf(GpuAcData gpuData, LfBus bus) {
        int[] rlBusNum = gpuData.reactiveLimits().busNum();
        if (rlBusNum != null) {
            for (int e = 0; e < rlBusNum.length; e++) {
                if (rlBusNum[e] == bus.getNum()) {
                    return e;
                }
            }
        }
        return -1;
    }

    /** The row mode a surviving member of a DEAD voltage control gets on its BUS_V row. A DISTR_Q
     *  member's row must be mode 4 ("hosts its own reactive balance"): fillDistrQBatch skips every
     *  contribution whose F row is mode 4 and the member's own q rides its alternate dest, which is
     *  keyed on mode 4. Mode 1 left the DISTR_Q family writing into the row on top of nothing that
     *  balanced the bus: rte6515 BUS-1055_BBS_1 (pilot VL-1055_0 islanded, merged section VL-1055_1
     *  surviving) - OLF's BUS_TARGET_Q at the survivor off by 0.039 pu, 1.3 kV. A survivor that is
     *  not a packed DISTR_Q member keeps the plain reactive row. */
    static int deadControlRowMode(GpuAcData gpuData, LfBus survivor) {
        int slot = GpuBatchInjectionOutages.dqMemberSlot(gpuData, survivor);
        if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
            System.err.println("QUALIFY_TRACE deadControlRowMode survivor=" + survivor.getId() + " dqMemberSlot=" + slot
                    + " -> mode " + (slot >= 0 ? 4 : 1));
        }
        return slot >= 0 ? 4 : 1;
    }

    /**
     * The relocated coupler children, keyed by the child's own {@code BUS_V} variable row:
     * {@code {coupler branch num, the coupler's DUMMY_Q variable row}}. The coupler-tree relocation moved that
     * bus's reactive balance (and its {@code BUS_TARGET_Q}) onto the coupler's dummy row and put the coupler's
     * {@code ZERO_V} on the bus's own row, so a per-scenario flip of "solve your own reactive balance at this
     * target" has to land on the dummy row — the native reactive-limit pass makes exactly the same diversion
     * (see {@code rlQrow}) — and a flip to a VOLTAGE identity has nowhere to land at all.
     */
    static java.util.Map<Integer, int[]> relocatedChildRows(GpuAcData gpuData) {
        java.util.Map<Integer, int[]> out = new java.util.HashMap<>();
        int[] zi = gpuData.ziIdx();
        int[] nums = gpuData.ziBranchNums();
        for (int e = 0; e < zi.length / 11 && e < nums.length; e++) {
            int o = e * 11;
            int childSide = (zi[o + 10] >> 1) & 3;
            int ownV = childSide == 1 ? zi[o + 6] : (childSide == 2 ? zi[o + 7] : -1);
            if (ownV >= 0 && zi[o + 5] >= 0) {
                out.put(ownV, new int[] {nums[e], zi[o + 5]});
            }
        }
        return out;
    }

    /** Where a per-scenario reactive flip lands: the bus's own {@code BUS_V} row, or its parent coupler's
     *  {@code DUMMY_Q} row while that coupler stays closed. An OUTAGED coupler puts the equations back on the
     *  bus's own rows (the stock placement the chunk solver switches to), so the own row is right there. */
    static int relocQRow(int vRow, java.util.Map<Integer, int[]> relocRows, java.util.Set<Integer> outagedZiNums) {
        int[] reloc = relocRows.get(vRow);
        return reloc == null || outagedZiNums.contains(reloc[0]) ? vRow : reloc[1];
    }

    /** The INVERSE of {@link #relocatedChildRows}, keyed by the coupler's {@code DUMMY_Q} row:
     *  {@code {branchNum, the child bus's own BUS_V row}}. A consumer holding a STRUCTURAL member row
     *  (which the coupler-tree relocation already moved onto the dummy) needs this to map back, because
     *  a scenario that OUTAGES the coupler puts the child's reactive equation on the bus's own row. */
    static java.util.Map<Integer, int[]> relocatedDummyQRows(GpuAcData gpuData) {
        java.util.Map<Integer, int[]> out = new java.util.HashMap<>();
        int[] zi = gpuData.ziIdx();
        int[] nums = gpuData.ziBranchNums();
        for (int e = 0; e < zi.length / 11 && e < nums.length; e++) {
            int o = e * 11;
            int childSide = (zi[o + 10] >> 1) & 3;
            int ownV = childSide == 1 ? zi[o + 6] : (childSide == 2 ? zi[o + 7] : -1);
            if (ownV >= 0 && zi[o + 5] >= 0) {
                out.put(zi[o + 5], new int[] {nums[e], ownV});
            }
        }
        return out;
    }

    /** The row a consumer holding the structural (already relocated) row must write in THIS scenario:
     *  the child bus's own row once the coupler carrying its equations is outaged, else unchanged. */
    static int unrelocQRow(int row, java.util.Map<Integer, int[]> dummyQRows, java.util.Set<Integer> outagedZiNums) {
        int[] reloc = dummyQRows.get(row);
        return reloc != null && outagedZiNums.contains(reloc[0]) ? reloc[1] : row;
    }
}
