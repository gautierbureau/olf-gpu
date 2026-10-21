/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix;

import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuTwoSides;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;

/**
 * docs/gpu-vs-olf.md section 2.3 (zero-impedance branches: re-homing): where a relocated coupler child's equations
 * sit, read off the zero-impedance pack ({@code zeroImpedanceBranch().indices()}, {@link #STRIDE} slots per
 * branch), and where they sit in THIS scenario: on the parent coupler's {@code DUMMY_P} / {@code DUMMY_Q} rows while
 * the coupler stays closed, back home on the child's own {@code PHI} / {@code V} rows once it opens.
 *
 * <p>The pack's slot layout (written by {@code GpuAcFamilyPacker}): 0-3 the endpoint P / Q rows, {@link #DUMMY_P}
 * / {@link #DUMMY_Q} the coupler's dummy columns, {@link #V1} / {@link #V2} and {@link #PHI1} / {@link #PHI2} the
 * endpoints' own rows, {@link #FLAGS} = bit 0 spanning-tree edge, bits 1-2 the child side ({@link #encodeChildSide}). A
 * child side is only ever set by the coupler-tree walk, so a side other than NONE is what marks a coupler whose child
 * needs the per-scenario stock-placement fallback.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class CouplerTreeReHoming {

    /** Slots per zero-impedance branch in the pack. */
    public static final int STRIDE = 11;
    /** Slot of the coupler's {@code DUMMY_P} column. */
    public static final int DUMMY_P = 4;
    /** Slot of the coupler's {@code DUMMY_Q} column. */
    public static final int DUMMY_Q = 5;
    /** Slots of the side-1 / side-2 bus's own {@code BUS_V} row. */
    public static final int V1 = 6;
    public static final int V2 = 7;
    /** Slots of the side-1 / side-2 bus's own {@code BUS_PHI} row. */
    public static final int PHI1 = 8;
    public static final int PHI2 = 9;
    /** Slot of the flags. */
    public static final int FLAGS = 10;
    /** Flag bit 0: the branch is a zero-impedance spanning-tree edge. */
    public static final int FLAG_SPANNING_TREE_EDGE = 1;
    /** Flags bits 1-2: the child side, shifted by this (see {@link #encodeChildSide}). */
    public static final int CHILD_SIDE_SHIFT = 1;

    private CouplerTreeReHoming() {
    }

    /** A {@link GpuTwoSides} child side (NONE, ONE or TWO) as flags bits 1-2: side + 1, so 0 is none. The native
     *  zeroImpedanceDeadEndSideOf (shared_structure.cuh) decodes it as {@link #childSide} does. */
    public static int encodeChildSide(int childSide) {
        return (childSide + 1) << CHILD_SIDE_SHIFT;
    }

    /** The child side encoded in a branch's flags: a {@link GpuTwoSides}, NONE when there is none. */
    public static int childSide(int flags) {
        return ((flags >> CHILD_SIDE_SHIFT) & 3) - 1;
    }

    /** The relocated child's own {@code BUS_V} row of zero-impedance element {@code e}, -1 for none. */
    public static int childOwnVRow(int[] zeroImpedanceBranchIndices, int e) {
        int o = e * STRIDE;
        int childSide = childSide(zeroImpedanceBranchIndices[o + FLAGS]);
        return childSide == GpuTwoSides.ONE ? zeroImpedanceBranchIndices[o + V1] : (childSide == GpuTwoSides.TWO ? zeroImpedanceBranchIndices[o + V2] : -1);
    }

    /** The relocated child's own {@code BUS_PHI} row of zero-impedance element {@code e}, -1 for none. */
    public static int childOwnPhiRow(int[] zeroImpedanceBranchIndices, int e) {
        int o = e * STRIDE;
        int childSide = childSide(zeroImpedanceBranchIndices[o + FLAGS]);
        return childSide == GpuTwoSides.ONE ? zeroImpedanceBranchIndices[o + PHI1] : (childSide == GpuTwoSides.TWO ? zeroImpedanceBranchIndices[o + PHI2] : -1);
    }

    /**
     * The relocated {@code DUMMY_P} row that hosts the ACTIVE-power balance of the bus whose own
     * {@code BUS_PHI} column is {@code phiCol}, or {@code phiCol} itself when that bus is not a relocated
     * coupler child.
     *
     * <p>Every consumer of "where does this bus's P equation live" must ask this, exactly as the reactive
     * side must ask for the relocated Q row. Under the coupler-tree relocation a child's P balance moves to
     * its parent coupler's {@code DUMMY_P} row while its own {@code BUS_PHI} row hosts the coupler's
     * {@code ZERO_PHI} — so a row index that is correct as an ANGLE COLUMN is the WRONG place to write a P
     * target. Writing there puts the target on the angle identity: measured on rte6515 (tasks #68), the HVDC
     * AC-emulation freeze setpoints landed on all ten VSC-station breakers' {@code ZERO_PHI} rows, whose
     * target must be 0, while the {@code DUMMY_P} rows that actually carry the HVDC p1/p2 terms had target 0.
     *
     * <p>Derived from the same zero-impedance pack as the reactive twin, so the two cannot drift.
     *
     * <p>Used by {@code GpuHvdcAcEmulationLimitsOuterLoop} for the converter buses' P-target rows since 2026-10-05. The
     * first attempt (2026-10-02) was measured "catastrophic" (max-iter-bad-solve 108 -> 13194) and blamed the
     * premise; the premise was right — a relocated child's P balance DOES live on the parent's {@code DUMMY_P}
     * row, the child's own PHI row hosting the coupler's {@code ZERO_PHI} — and the failure was the kernel
     * reading the bus ANGLE at the same index it wrote the target to (a {@code DUMMY_P} variable read as an
     * angle). The angle column and the target row are separate arrays now, and a scenario that opens the parent
     * coupler diverts the write to the own row (relocatedChildOwnRow). Measured on rte6515 BUS-1219_BBS: the saturation
     * delta of SYN_HVDC_4 landing on a ZERO_PHI row reversed the emulated flow and the limits loop toggled
     * between the two saturation sides to its budget — the one device failure on an OLF-converging busbar
     * contingency (tasks #68 / #69).
     */
    public static int relocatedPRow(int[] zeroImpedanceBranchIndices, int phiCol) {
        if (zeroImpedanceBranchIndices == null || phiCol < 0) {
            return phiCol;
        }
        for (int e = 0; e < zeroImpedanceBranchIndices.length / STRIDE; e++) {
            int o = e * STRIDE;
            int ownPhi = childOwnPhiRow(zeroImpedanceBranchIndices, e);
            if (ownPhi == phiCol && zeroImpedanceBranchIndices[o + DUMMY_P] >= 0) {
                return zeroImpedanceBranchIndices[o + DUMMY_P];
            }
        }
        return phiCol;
    }

    /** The relocated {@code DUMMY_Q} row that hosts the reactive balance of the bus whose own {@code BUS_V} row
     *  is {@code vRow}, or {@code vRow} itself when that bus is not a relocated coupler child. The Java twin of
     *  the native {@code reactiveLimitReactivePowerRow} map, derived from the same zero-impedance pack. */
    public static int relocatedPinRow(int[] zeroImpedanceBranchIndices, int vRow) {
        if (zeroImpedanceBranchIndices == null || vRow < 0) {
            return vRow;
        }
        for (int e = 0; e < zeroImpedanceBranchIndices.length / STRIDE; e++) {
            int o = e * STRIDE;
            int ownV = childOwnVRow(zeroImpedanceBranchIndices, e);
            if (ownV == vRow && zeroImpedanceBranchIndices[o + DUMMY_Q] >= 0) {
                return zeroImpedanceBranchIndices[o + DUMMY_Q];
            }
        }
        return vRow;
    }

    /** The bus whose P balance the relocation moved onto the coupler {@code DUMMY_P} row {@code pRow}. */
    public static LfBus relocatedChildOfPRow(LfNetwork network, int[] zeroImpedanceBranchIndices, int[] nums, int pRow) {
        if (zeroImpedanceBranchIndices == null || nums == null) {
            return null;
        }
        for (int e = 0; e < zeroImpedanceBranchIndices.length / STRIDE && e < nums.length; e++) {
            int o = e * STRIDE;
            int childSide = childSide(zeroImpedanceBranchIndices[o + FLAGS]);
            if (childSide == GpuTwoSides.NONE || zeroImpedanceBranchIndices[o + DUMMY_P] != pRow || nums[e] < 0) {
                continue;
            }
            LfBranch br = network.getBranch(nums[e]);
            return childSide == GpuTwoSides.ONE ? br.getBus1() : br.getBus2();
        }
        return null;
    }

    /**
     * The relocated coupler children, keyed by the child's own {@code BUS_V} variable row:
     * {@code {coupler branch num, the coupler's DUMMY_Q variable row}}. The coupler-tree relocation moved that
     * bus's reactive balance (and its {@code BUS_TARGET_Q}) onto the coupler's dummy row and put the coupler's
     * {@code ZERO_V} on the bus's own row, so a per-scenario flip of "solve your own reactive balance at this
     * target" has to land on the dummy row — the native reactive-limit pass makes exactly the same diversion
     * (see {@code reactiveLimitReactivePowerRow}) — and a flip to a VOLTAGE identity has nowhere to land at all.
     */
    public static java.util.Map<Integer, int[]> relocatedChildRows(int[] zeroImpedanceBranchIndices, int[] nums) {
        java.util.Map<Integer, int[]> out = new java.util.HashMap<>();
        for (int e = 0; e < zeroImpedanceBranchIndices.length / STRIDE && e < nums.length; e++) {
            int o = e * STRIDE;
            int ownV = childOwnVRow(zeroImpedanceBranchIndices, e);
            if (ownV >= 0 && zeroImpedanceBranchIndices[o + DUMMY_Q] >= 0) {
                out.put(ownV, new int[] {nums[e], zeroImpedanceBranchIndices[o + DUMMY_Q]});
            }
        }
        return out;
    }

    /** Where a per-scenario reactive flip lands: the bus's own {@code BUS_V} row, or its parent coupler's
     *  {@code DUMMY_Q} row while that coupler stays closed. An OUTAGED coupler puts the equations back on the
     *  bus's own rows (the stock placement the chunk solver switches to), so the own row is right there. */
    public static int scenarioReactiveRowOfVRow(int vRow, java.util.Map<Integer, int[]> relocatedChildRows, java.util.Set<Integer> outagedZeroImpedanceBranchNums) {
        int[] reloc = relocatedChildRows.get(vRow);
        return reloc == null || outagedZeroImpedanceBranchNums.contains(reloc[0]) ? vRow : reloc[1];
    }

    /** The INVERSE of {@link #relocatedChildRows}, keyed by the coupler's {@code DUMMY_Q} row:
     *  {@code {branchNum, the child bus's own BUS_V row}}. A consumer holding a STRUCTURAL member row
     *  (which the coupler-tree relocation already moved onto the dummy) needs this to map back, because
     *  a scenario that OUTAGES the coupler puts the child's reactive equation on the bus's own row. */
    public static java.util.Map<Integer, int[]> relocatedDummyQRows(int[] zeroImpedanceBranchIndices, int[] nums) {
        java.util.Map<Integer, int[]> out = new java.util.HashMap<>();
        for (int e = 0; e < zeroImpedanceBranchIndices.length / STRIDE && e < nums.length; e++) {
            int o = e * STRIDE;
            int ownV = childOwnVRow(zeroImpedanceBranchIndices, e);
            if (ownV >= 0 && zeroImpedanceBranchIndices[o + DUMMY_Q] >= 0) {
                out.put(zeroImpedanceBranchIndices[o + DUMMY_Q], new int[] {nums[e], ownV});
            }
        }
        return out;
    }

    /** The row a consumer holding the structural (already relocated) row must write in THIS scenario:
     *  the child bus's own row once the coupler carrying its equations is outaged, else unchanged. */
    public static int scenarioReactiveRowOfStructuralRow(int row, java.util.Map<Integer, int[]> dummyQRows, java.util.Set<Integer> outagedZeroImpedanceBranchNums) {
        int[] reloc = dummyQRows.get(row);
        return reloc != null && outagedZeroImpedanceBranchNums.contains(reloc[0]) ? reloc[1] : row;
    }
}
