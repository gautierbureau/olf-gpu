/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.network.LfBranch;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * The scenario plan being built by {@link GpuBatchQualification#planScenarios}: the batched contingency indices, the
 * per-contingency entries every planner writes (frozen rows, generator outages, scheduled losses, voltage-control
 * overrides, islanded participants, slack relocations), the lazily resolved bus rows, and the lookups the planners
 * share. {@link #build()} turns it into the {@link GpuBatchQualification.ScenarioPlan}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class ScenarioPlanBuilder {

    final List<Integer> batchedIndices;
    final java.util.Map<Integer, int[]> relocRows;
    final java.util.Map<Integer, int[]> relocDummyQRows;
    final int[][] frozenRowsByIndex;     // islanded buses' variable rows, per contingency
    final int[][] islandedNumsByIndex;   // islanded LfBus nums (result-building excludes them)
    final double[][] frozenTargetsByIndex;   // the clean value each frozen row holds (V=1, phi=0)
    final GenOutage[][] genOutagesByIndex;   // per-bus merged outages, else null
    // Per-contingency SCHEDULED active power loss (pu, CPU sign: lost generation - lost load,
    // LfContingency.getActivePowerLoss incl. islanded buses) — drives the host replay of OLF's
    // DefaultContingencyActivePowerLossDistribution (GpuBatchLossPreDistribution).
    final double[] activePowerLossByIndex;
    final List<com.powsybl.openloadflow.network.LfHvdc>[] islandedHvdcsByIndex;
    // Per-contingency DISTR_Q weight overrides: a gen outage at a remote-VC controller bus makes
    // the CPU renormalize the group's qPct keys (LfContingency.processLostGenerators ->
    // updateReactiveKeys); replay the toggle host-side and diff the extracted weights.
    final int[][] dqOvContribByIndex;
    final double[][] dqOvWByIndex;
    final int[][] dqKeyOvMemberByIndex;
    final double[][] dqKeyOvByIndex;
    // Per-contingency DISTR_RHO coefficient overrides (plan A5): outaging a NON-FREE member of a
    // shared-TVC group renormalizes the survivors' 1/count coefficients (the CPU's coefficients
    // are live suppliers over !isDisabled) and zeroes every contribution touching the outaged
    // member's rho; the FREE member's outage shifts the pilot-Q role — CPU (A6).
    final int[][] drOvContribByIndex;
    final double[][] drOvCByIndex;
    final int[][] islandedDsByIndex;
    final int[][] islandedRlByIndex;
    // {pRow, qRow} per LfBus.getNum(), resolved lazily on the first injection outage: the P/Q
    // balance rows the extraction writes targets at, which the coupler-tree relocation moves off
    // the bus's own BUS_PHI/BUS_V columns. One O(buses+branches) pass for the whole plan.
    int[][] busRows;
    final int[][] dqSubByIndex;
    final int[] relocatedSlackRow;
    final double[] relocatedSlackShift;
    int islandingCount;
    // per contingency id: the buses it islands, and the branches it opens itself
    final java.util.Map<String, Set<Integer>> islandedByContingency;
    final java.util.Map<String, List<LfBranch>> openedByContingency;

    @SuppressWarnings("unchecked")
    ScenarioPlanBuilder(int nContingencies, GpuAcData gpuData) {
        this.batchedIndices = new ArrayList<>();
        this.relocRows = RelocatedRows.relocatedChildRows(gpuData);
        this.relocDummyQRows = RelocatedRows.relocatedDummyQRows(gpuData);
        this.frozenRowsByIndex = new int[nContingencies][];
        this.islandedNumsByIndex = new int[nContingencies][];
        this.frozenTargetsByIndex = new double[nContingencies][];
        this.genOutagesByIndex = new GenOutage[nContingencies][];
        this.activePowerLossByIndex = new double[nContingencies];
        this.islandedHvdcsByIndex = new List[nContingencies];
        this.dqOvContribByIndex = new int[nContingencies][];
        this.dqOvWByIndex = new double[nContingencies][];
        this.dqKeyOvMemberByIndex = new int[nContingencies][];
        this.dqKeyOvByIndex = new double[nContingencies][];
        this.drOvContribByIndex = new int[nContingencies][];
        this.drOvCByIndex = new double[nContingencies][];
        this.islandedDsByIndex = new int[nContingencies][];
        this.islandedRlByIndex = new int[nContingencies][];
        this.dqSubByIndex = new int[nContingencies][];
        this.relocatedSlackRow = new int[nContingencies];
        java.util.Arrays.fill(relocatedSlackRow, -1);
        this.relocatedSlackShift = new double[nContingencies];
        this.islandedByContingency = new java.util.HashMap<>();
        this.openedByContingency = new java.util.HashMap<>();
    }

    /** The plan, once every contingency went through its planner. */
    GpuBatchQualification.ScenarioPlan build() {
        int maskedDs = 0;
        for (int[] m : islandedDsByIndex) {
            maskedDs += m == null ? 0 : m.length;
        }
        com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.LAST_ISLANDED_DS_MASKED.set(maskedDs);
        return new GpuBatchQualification.ScenarioPlan(batchedIndices, frozenRowsByIndex, islandedNumsByIndex,
                frozenTargetsByIndex, genOutagesByIndex, islandingCount, activePowerLossByIndex,
                dqOvContribByIndex, dqOvWByIndex, dqKeyOvMemberByIndex, dqKeyOvByIndex, drOvContribByIndex, drOvCByIndex, islandedDsByIndex,
                islandedRlByIndex, islandedHvdcsByIndex, dqSubByIndex,
                new GpuBatchQualification.SlackRelocations(relocatedSlackRow, relocatedSlackShift));
    }
}
