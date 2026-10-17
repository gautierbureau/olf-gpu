/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;

import java.util.ArrayList;
import java.util.List;

/**
 * The chunk solver's run-scope buffers, one per slot or per run and reused by every chunk: the per-slot disabled
 * mask and row modes, the target overrides, the zero-impedance, shunt and fixed-ratio override lists, the
 * distributed-slack reweight, and the lookups built once from the base extraction. Their lifetime is the run;
 * {@link #resetChunk()} clears what a chunk rebuilds and {@link #resetSlot(int)} clears a whole slot.
 *
 * <p>The base state / targets are IDENTICAL for every scenario (a single-branch N-1 starts from the
 * same converged base operating point): passed as single length-n rows and tiled on-device. The
 * disabled mask is per-scenario (each chunk clears the one flag each slot set, sets the new one).
 * The ROW MODE is per-scenario too: a non-fragmenting outage shares the base row (same PV/slack
 * layout); a fragmenting one gets a cloned base row with its islanded buses frozen to identity.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class RunScratch {

    final boolean[][] disabledMask;
    final int[][] disabledIdx;      // element indices each slot currently disables (1-2), null = none
    // bus V column -> the closed (P1 Q1 P2 Q2 ph1 v1 ph2 v2 ..., stride 12) and open (P1 Q1 P2 Q2 v1 v2, stride 6, after
    // the closed ones in the element order) branch elements with an end there
    final java.util.Map<Integer, List<Integer>> elementsAtVCol;
    final int ncbElems;
    final int[][] rowModePerScenario;   // per-scenario row mode (re-assigned every chunk)
    // Per-scenario target overrides (generator contingencies): flat (dTargetIndex = slot*n + row, value)
    // lists rebuilt per chunk. Empty when no generator outage is in the chunk.
    // The per-chunk override list AND its row index in one object: a row may carry only one
    // override (the device scatter has no ordering), the lookup that enforces that must not be
    // a linear scan, and the index must never outlive the entries — see GpuBatchOverrides.
    final GpuBatchOverrides overrides;
    final java.util.Map<Integer, Integer> ziElemByBranchNumChunk;
    final int nziBase;
    final java.util.Map<Integer, Integer> ziElemByNumAll;
    final List<Integer> ziOffSlotElemList; // per-scenario zi-element outages (switch opens)
    final List<Integer> ziOnSlotElemList;  // per-scenario zi-element enables (switch closes)
    final List<Integer> ziTreeSlotElemValList;   // {slot, ziElement, isTree} where OLF's tree moved
    final List<Integer> shOvSlotElemList;  // per-scenario shunt admittance overrides
    final List<Double> shOvGbList;
    final List<Integer> r1OvSlotElemList;  // per-scenario fixed-ratio tap overrides
    final List<Double> r1OvValList;
    final int shPackOffset;
    // Per-scenario distributed-slack reweight (a generator loss at a DS-participant bus): one affected DS
    // bus per slot (dsOvBus[s] = its DS-pack index, -1 = none) whose factor/base/headroom/footroom and the
    // factor sum shrink by the lost gen's share. Only allocated when the network has distributed slack.
    final boolean hasDs;
    final int[] dsOvBus;
    final double[] dsOvFactorDelta;
    final double[] dsOvBaseDelta;
    final double[] dsOvMaxDelta;
    final double[] dsOvMinDelta;
    private final int[] baseRowMode;
    private final int[] baseOpenCbElements;

    RunScratch(int chunkSize, int nElements, GpuAcData gpuData, GpuBatchElementMapper elementMapper, int[] baseRowMode) {
        this.disabledMask = new boolean[chunkSize][nElements];
        this.disabledIdx = new int[chunkSize][];
        this.elementsAtVCol = new java.util.HashMap<>();
        this.ncbElems = gpuData.cbIdx().length / 12;
        for (int e = 0; e < ncbElems; e++) {
            for (int col : new int[] {gpuData.cbIdx()[e * 12 + 5], gpuData.cbIdx()[e * 12 + 7]}) {
                if (col >= 0) {
                    elementsAtVCol.computeIfAbsent(col, k -> new ArrayList<>()).add(e);
                }
            }
        }
        for (int e = 0; e < gpuData.obIdx().length / 6; e++) {
            for (int col : new int[] {gpuData.obIdx()[e * 6 + 4], gpuData.obIdx()[e * 6 + 5]}) {
                if (col >= 0) {
                    elementsAtVCol.computeIfAbsent(col, k -> new ArrayList<>()).add(ncbElems + e);
                }
            }
        }
        this.rowModePerScenario = new int[chunkSize][];
        this.overrides = new GpuBatchOverrides();
        this.ziElemByBranchNumChunk = new java.util.HashMap<>();
        for (int z = 0; z < gpuData.baseOpenZiElements().length; z++) {
            ziElemByBranchNumChunk.put(gpuData.baseOpenZiBranchNums()[z], gpuData.baseOpenZiElements()[z]);
        }
        this.nziBase = gpuData.ziIdx().length / 11 - gpuData.baseOpenZiElements().length;
        this.ziElemByNumAll = new java.util.HashMap<>();
        for (int z = 0; z < gpuData.ziIdx().length / 11; z++) {
            ziElemByNumAll.put(gpuData.ziBranchNums()[z], z);
        }
        this.ziOffSlotElemList = new ArrayList<>();
        this.ziOnSlotElemList = new ArrayList<>();
        this.ziTreeSlotElemValList = new ArrayList<>();
        this.shOvSlotElemList = new ArrayList<>();
        this.shOvGbList = new ArrayList<>();
        this.r1OvSlotElemList = new ArrayList<>();
        this.r1OvValList = new ArrayList<>();
        this.shPackOffset = elementMapper.getClosedBranchCount() + elementMapper.getOpenBranchCount();
        this.hasDs = gpuData.dsPhiRow().length > 0;
        this.dsOvBus = hasDs ? new int[chunkSize] : new int[0];
        this.dsOvFactorDelta = hasDs ? new double[chunkSize] : new double[0];
        this.dsOvBaseDelta = hasDs ? new double[chunkSize] : new double[0];
        this.dsOvMaxDelta = hasDs ? new double[chunkSize] : new double[0];
        this.dsOvMinDelta = hasDs ? new double[chunkSize] : new double[0];
        this.baseRowMode = baseRowMode;
        this.baseOpenCbElements = gpuData.baseOpenCbElements();
    }

    /** Clear the run-scope lists a chunk rebuilds, and the distributed-slack reweight. */
    void resetChunk() {
        ziOffSlotElemList.clear();
        ziTreeSlotElemValList.clear();
        ziOnSlotElemList.clear();
        shOvSlotElemList.clear();
        shOvGbList.clear();
        r1OvSlotElemList.clear();
        r1OvValList.clear();
        if (hasDs) {
            java.util.Arrays.fill(dsOvBus, -1);
            java.util.Arrays.fill(dsOvFactorDelta, 0);
            java.util.Arrays.fill(dsOvBaseDelta, 0);
            java.util.Arrays.fill(dsOvMaxDelta, 0);
            java.util.Arrays.fill(dsOvMinDelta, 0);
        }
    }

    /**
     * Clear the WHOLE slot, not just the outage this slot carried. The chunk
     * loop reuses one disabledMask per slot, and it used to be reset by replaying
     * disabledIdx — which records only the contingency's own outaged elements. But
     * the HVDC machinery also sets bits here (seedBaseStatuses marks a non-LINEAR
     * entry's hv element, and the freeze/status paths follow it), and those were
     * never recorded. They therefore SURVIVED into the next chunk, leaving an hv
     * element disabled for whatever unrelated scenario landed in that slot.
     *
     * <p>The result was a scenario-dependent-on-its-predecessors batch: correct when
     * run alone, wrong after any preceding chunk. Measured on rte6515 at 9042
     * branch contingencies: 18 differing violation lines against the CPU, and the
     * affected set changed with the chunk size (64 -> 140 lines, 1024 -> 10)
     * because the chunking decides which scenario inherits which stale bit. With
     * the full clear: ZERO. A slot is nElements booleans; the fill costs nothing
     * measurable (9042 contingencies: 82.6 s either way).
     */
    void resetSlot(int s) {
        java.util.Arrays.fill(disabledMask[s], false);
        disabledIdx[s] = null;
        for (int e : baseOpenCbElements) {
            disabledMask[s][e] = true;           // base-open elements stay OUT unless this scenario recloses them
        }
        rowModePerScenario[s] = baseRowMode;     // default: base layout (padding + non-fragmenting)
    }
}
