/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcElementMapper;
import com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix.ReactiveRelocationTables;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.BatchContingency;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuRowMode;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuVoltageControlRole;
import com.powsybl.openloadflow.network.LfBranch;

import java.util.List;

/**
 * The seeding of the buses one slot's contingency islands: the branches left on the island masked, the equations
 * that leave OLF's index counted out, the islanded rows frozen to identities at a flat state, and the pilot balances
 * a frozen carrier hands to another member of its group.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class IslandSeed {

    private IslandSeed() {
    }

    /** Seed slot {@code s} (contingency index {@code ci}, outaged {@code branches}, side-opened {@code sideOpens});
     *  adds to {@code pilotBalanceRows} the rows that host a pilot's relocated reactive balance this scenario. */
    static void seed(SlotSeeding ctx, int s, int ci, List<LfBranch> branches, List<BatchContingency.SideOpen> sideOpens,
                     gnu.trove.set.hash.TIntHashSet pilotBalanceRows) {
        final int[] baseRowMode = ctx.baseRowMode();
        final double[] baseState = ctx.baseState();
        final double[] baseTargets = ctx.baseTargets();
        final ChunkLists cl = ctx.cl();
        final List<BatchContingency> contingencies = ctx.contingencies();
        final GpuAcElementMapper elementMapper = ctx.elementMapper();
        final int[][] frozenRowsByIndex = ctx.plan().frozenRowsByIndex();
        final double[][] frozenTargetsByIndex = ctx.plan().frozenTargetsByIndex();
        final GpuAcData gpuData = ctx.gpuData();
        final int n = ctx.n();
        final RunScratch rs = ctx.rs();
        // OLF disables every bus the contingency islands and every branch left on it. The island's
        // rows are frozen identities here, but such a branch stayed live at the frozen 1 pu, and a remote
        // voltage-control member's reactive injection reads it: the charging of a line inside the
        // island, or of a feeder left dangling INTO it, entered the group's sharing rows (network.xiidm
        // CXUL: CXSS_2/_4's step-up transformers, 4.3e-3 kV at CXSS_1/_3). A side-opened branch is
        // judged by its live end alone - its open end is the lost bus, frozen by design.
        java.util.Set<Integer> frozenCols = new java.util.HashSet<>();
        for (int r : frozenRowsByIndex[ci]) {
            frozenCols.add(r);
        }
        java.util.Set<Integer> sideOpenElems = new java.util.HashSet<>();
        for (BatchContingency.SideOpen so : sideOpens) {
            int e = elementMapper.getElementIndex(so.branch());
            sideOpenElems.add(e);
            if (frozenCols.contains(gpuData.closedBranch().indices()[e * 12 + (so.side() == 1 ? 7 : 5)])) {
                rs.disabledMask[s][e] = true;
            }
        }
        for (int r : frozenRowsByIndex[ci]) {
            for (int e : rs.elementsAtVCol.getOrDefault(r, List.of())) {
                if (!sideOpenElems.contains(e)) {
                    rs.disabledMask[s][e] = true;
                }
            }
        }
        cl.nActive[s] -= frozenRowsByIndex[ci].length;  // islanded buses' equations leave the CPU index
        if (System.getenv("OLF_GPU_SA_RESID") != null) {
            System.err.printf("DEV_NACTIVE ctg=%s n=%d frozen=%d nActive=%d%n",
                    contingencies.get(ci).contingencyId(), n,
                    frozenRowsByIndex[ci].length, cl.nActive[s]);
        }
        for (LfBranch branch : branches) {           // var-carrying outaged branches: the CPU drops
            int eIdx = elementMapper.getElementIndex(branch);   // the rho/a1 identity equations too
            if (eIdx >= 0 && eIdx < elementMapper.getClosedBranchCount()) {
                if (gpuData.closedBranch().indices()[eIdx * 12 + 9] >= 0) {
                    cl.nActive[s]--;        // BRANCH_RHO1 variable + its identity leave the index
                }
                if (gpuData.closedBranch().indices()[eIdx * 12 + 8] >= 0) {
                    cl.nActive[s]--;        // BRANCH_ALPHA1 variable + its identity leave the index
                }
            }
        }
        int[] frozen = frozenRowsByIndex[ci];
        if (frozen.length > 0) {         // fragmenting outage: freeze the islanded buses' rows
            int[] row = baseRowMode.clone();
            double[] frozenTargets = frozenTargetsByIndex[ci];
            for (int k = 0; k < frozen.length; k++) {
                row[frozen[k]] = GpuRowMode.FIXED_VARIABLE;      // identity row — bus decoupled from the main component
                // ...held at a CLEAN flat state (V=1, phi=0) via the target override, NOT the base
                // equation target (a P/Q value that gives a nonsensical islanded voltage/angle).
                rs.overrides.add(s * n + frozen[k], frozenTargets[k]);
                // ...and START that row AT its target. It is an identity row, so its
                // solution IS the target and seeding it costs nothing — but leaving it
                // at the base state hands the convergence test a residual OLF never
                // sees, because OLF REMOVES an islanded bus's equations rather than
                // freezing them. Measured on rte6515 LINE-4413-5: the islanded bus's
                // angle row sits at -1.050832654 rad against a frozen target of 0, and
                // that 1.050832654 IS the device's whole initial |F|inf for the solve,
                // against OLF's 0.1296557068. The device then needs an extra Newton
                // iteration, lands at 1.68e-06 where OLF stops at 8.04e-04, and the
                // distributed-slack loop reads a slack mismatch 7.43e-04 different —
                // which is the entire branch-family error (8.328e-05 kV).
                cl.frozenSeedSlot.add(s);
                cl.frozenSeedRow.add(frozen[k]);
                cl.frozenSeedValue.add(frozenTargets[k]);
                if (System.getenv("OLF_GPU_SA_RESID") != null) {
                    System.err.printf("FROZEN ctg=%s row=%d target=%.9f base=%.9f |F0|=%.9f%n",
                            contingencies.get(ci).contingencyId(), frozen[k], frozenTargets[k],
                            baseState[frozen[k]], Math.abs(baseState[frozen[k]] - frozenTargets[k]));
                }
            }
            // A frozen row that was a remote gen-VC group's BASE CARRIER (the free
            // member's row hosting the pilot's reactive balance): the device's mode scan
            // moves that balance to the next live member's row from the first Newton
            // solve on, but that row's TARGET is its own DISTR_Q one (0) - the kernel
            // that stamps the pilot's target on the carrier runs only inside the
            // outer-loop passes, and the host re-applies its overrides at every pass.
            // Stamp it here, by the kernel's own rule (chooseCarrierRow: first candidate
            // whose mode is neither identity nor pinned). Measured on the remote-control
            // fixture, b1 islanded: OLF's BUS_TARGET_Q at the pilot off by exactly its
            // load, b2 0.45 kV; rte6515: five busbar sections up to 1.3 kV.
            int[] reactiveRelocationGroupStart = gpuData.reactiveRelocation().groupStart();
            int[] reactiveRelocationCandidateRows = gpuData.reactiveRelocation().candidateRows();
            for (int g = 0; reactiveRelocationGroupStart != null && g + 1 < reactiveRelocationGroupStart.length; g++) {
                int baseCarrier = ReactiveRelocationTables.chooseCarrierRow(gpuData.reactiveRelocation(), g, baseRowMode);
                int newCarrier = ReactiveRelocationTables.chooseCarrierRow(gpuData.reactiveRelocation(), g, row);
                // A PILOT-SELF group (rotation: the fallback row is itself a candidate) whose pilot is
                // UNPINNED in the base carries no pilot balance on its carrier: that row hosts its OWN
                // DISTR_Q, and the base carrier's target is that member's DISTR_Q target, not a pilot
                // Q — stamping it on the next member overwrote the renormalized target the outage
                // seeding had just put there (middle-section fixture: −0.1333 over −0.2, bus-mid
                // 0.32 kV off). Only a PINNED pilot's balance travels with the carrier.
                boolean unpinnedPilotSelf = false;
                int fallbackRow = reactiveRelocationCandidateRows[reactiveRelocationGroupStart[g]];
                for (int c = reactiveRelocationGroupStart[g] + 1; c < reactiveRelocationGroupStart[g + 1]; c++) {
                    if (reactiveRelocationCandidateRows[c] == fallbackRow) {
                        unpinnedPilotSelf = true;
                        break;
                    }
                }
                if (unpinnedPilotSelf && gpuData.reactiveLimits().voltageRow() != null && gpuData.reactiveLimits().voltageControlRole() != null) {
                    for (int e = 0; e < gpuData.reactiveLimits().voltageRow().length; e++) {
                        if (gpuData.reactiveLimits().voltageRow()[e] == fallbackRow && gpuData.reactiveLimits().voltageControlRole()[e] == GpuVoltageControlRole.PILOT_SELF) {
                            unpinnedPilotSelf = !gpuData.switchedPvToPqInBase(e);
                            break;
                        }
                    }
                }
                if (baseCarrier >= 0 && newCarrier >= 0 && newCarrier != baseCarrier
                        && row[baseCarrier] == GpuRowMode.FIXED_VARIABLE && baseRowMode[baseCarrier] != GpuRowMode.FIXED_VARIABLE && !unpinnedPilotSelf) {
                    rs.overrides.add(s * n + newCarrier, baseTargets[baseCarrier]);
                    pilotBalanceRows.add(newCarrier);
                    if (System.getenv("OLF_GPU_SA_RESID") != null) {
                        System.err.printf("CARRIER_MOVED ctg=%s group=%d from=%d to=%d pilotQ=%.9f%n",
                                contingencies.get(ci).contingencyId(), g, baseCarrier, newCarrier,
                                baseTargets[baseCarrier]);
                    }
                }
            }
            rs.rowModePerScenario[s] = row;
        }
    }
}
