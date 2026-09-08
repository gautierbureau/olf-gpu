/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.VoltageControl;

import java.util.ArrayList;
import java.util.List;

/**
 * The per-extraction column arrays every family and outer-loop packer reads: each bus's
 * BUS_PHI / BUS_V variable rows ({@code phiCol} / {@code vCol}, indexed by bus number), the
 * per-branch transformer-control columns ({@code branchRhoCol} / {@code branchAlphaCol}) and
 * zero-impedance dummy columns ({@code dummyPCol} / {@code dummyQCol}, indexed by branch
 * number), and the stable Q-balance row of each bus ({@code qRow}). Computed once at the top of
 * {@link GpuAcDataExtractor#extract} — a bus appears in several branches, so caching these
 * lookups avoids resolving the same variable ~2x its degree. {@code -1} marks an absent column.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
record ExtractionColumns(int[] phiCol, int[] vCol, int[] branchRhoCol, int[] branchAlphaCol,
                         int[] dummyPCol, int[] dummyQCol, int[] shuntBCol, int[] qRow, int[] qRowAlt,
                         int[] qrPtr, int[] qrRows) {

    static ExtractionColumns of(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es) {
        // Cache each bus's BUS_PHI/BUS_V column once: a bus appears in several branches (degree ~3),
        // so resolving these per branch-endpoint repeats the same variable lookups ~2x its degree.
        // Indexed by LfBus.getNum() (contiguous 0..nBuses-1); -1 means absent.
        int nBuses = network.getBuses().size();
        int[] phiCol = new int[nBuses];
        int[] vCol = new int[nBuses];
        for (LfBus bus : network.getBuses()) {
            int bn = bus.getNum();
            phiCol[bn] = GpuAcRowIndexing.col(es, bn, AcVariableType.BUS_PHI);
            vCol[bn] = GpuAcRowIndexing.col(es, bn, AcVariableType.BUS_V);
        }

        // Per-branch transformer control columns (BRANCH_RHO1 ratio, BRANCH_ALPHA1 phase), -1 when
        // the branch does not derive that control. A controlling transformer's rho column also
        // becomes the controlled bus's Q-balance row (qRow below) — the strong-diagonal pairing.
        int nBranches = network.getBranches().size();
        int[] branchRhoCol = GpuAcRowIndexing.branchVarCols(es, nBranches, AcVariableType.BRANCH_RHO1);
        int[] branchAlphaCol = GpuAcRowIndexing.branchVarCols(es, nBranches, AcVariableType.BRANCH_ALPHA1);
        int[] dummyPCol = GpuAcRowIndexing.branchVarCols(es, nBranches, AcVariableType.DUMMY_P);
        int[] dummyQCol = GpuAcRowIndexing.branchVarCols(es, nBranches, AcVariableType.DUMMY_Q);
        // Per-shunt SHUNT_B variable column (indexed by LfShunt.getNum()), -1 when the shunt derives no voltage
        // control. A controlling shunt's SHUNT_B column also becomes the controlled bus's Q-balance row (qRow),
        // the exact analogue of the transformer rho pairing.
        int nShunts = network.getShunts().size();
        int[] shuntBCol = GpuAcRowIndexing.branchVarCols(es, nShunts, AcVariableType.SHUNT_B);
        int[] qRow = GpuAcRowIndexing.computeQRow(network, es, vCol, branchRhoCol, shuntBCol);
        // Dynamic Q-relocation: the OTHER candidate Q row of each transformer/shunt voltage-controlled
        // bus, so the fill can follow a per-scenario control toggle on the shared structure (§11).
        int[] qRowAlt = GpuAcRowIndexing.computeQRowAlt(network, es, vCol, branchRhoCol, shuntBCol, qRow);
        // GROUP pilot-Q relocation candidates (plan-A6): a shared transformer-VC pilot's reactive
        // balance rides on the FREE member's rho column — "first enabled", which an outage or clamp
        // of the free member RE-PICKS per scenario. Encode such pilots' alternate as -2-groupId and
        // ship the candidate table [pilot v row (fallback), member rho cols in merged order]; the
        // fill kernels select the FIRST member row whose per-scenario mode is not identity/pinned
        // (exactly OLF's free-role rule), falling back to the pilot's own row when control is off.
        List<Integer> qrPtr = new ArrayList<>();
        List<Integer> qrRows = new ArrayList<>();
        for (LfBus bus : network.getBuses()) {
            var tvc = bus.getTransformerVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .orElse(null);
            if (tvc == null || tvc.getMergedControllerElements().size() < 2 || vCol[bus.getNum()] < 0) {
                continue;
            }
            List<Integer> memberRows = new ArrayList<>();
            boolean allResolvable = true;
            for (var m : tvc.getMergedControllerElements()) {
                int rc = branchRhoCol[m.getNum()];
                if (rc < 0) {
                    allResolvable = false;                   // a member without a rho column: keep the
                    break;                                   // plain two-way alternate (validation guards)
                }
                memberRows.add(rc);
            }
            if (!allResolvable) {
                continue;
            }
            int g = qrPtr.size();
            qrPtr.add(qrRows.size());
            qrRows.add(vCol[bus.getNum()]);                  // candidate 0 = the FALLBACK (control off)
            qrRows.addAll(memberRows);
            qRowAlt[bus.getNum()] = -2 - g;
        }
        // Remote gen-VC pilot-Q relocation (plan-B4): the pilot's reactive balance rides on the
        // FREE member's BUS_V row — "first controller-enabled", which an RL pin of the free member
        // RE-PICKS per scenario. Same candidate table + in-kernel mode-scan select as the TVC
        // groups above: [pilot v row (fallback = group control lost), member v cols in merged
        // order]. Since the ob/zi alternate-dest channels landed, half-open and zero-impedance q
        // terms at the pilot ride the relocation too — no branch-shape guard remains. SINGLETON
        // groups (one remote controller) get a table too since 2026-09-08: their candidate list
        // [pilot fallback, the member] is structurally a 2-member table, and the only member's pin
        // IS the all-pinned corner (updatePilotVeqBatch flips the pilot's V identity) — this
        // retired the last remote-pin-dynamics whole-run gate (table-less pinnable free member).
        java.util.Set<Integer> qrPilotBuses = new java.util.HashSet<>();
        for (LfBus bus : network.getBuses()) {
            var gvc = bus.getGeneratorVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .filter(c -> bus.isGeneratorVoltageControlled() && !c.isLocalControl())
                    .orElse(null);
            // A pilot that is ALSO one of its controllers has NO Q balance of its own to relocate —
            // its q feeds the group's DISTR_Q rows instead (GpuAcRowIndexing.distrQRow), and leaving
            // it in this table double-counted that q on top of the DISTR_Q terms.
            if (gvc == null || gvc.getMergedControllerElements().isEmpty() || vCol[bus.getNum()] < 0
                    || gvc.getMergedControllerElements().contains(bus)) {
                continue;
            }
            List<Integer> memberRows = new ArrayList<>();
            boolean allResolvable = true;
            for (LfBus m : gvc.getMergedControllerElements()) {
                if (vCol[m.getNum()] < 0) {
                    allResolvable = false;
                    break;
                }
                memberRows.add(vCol[m.getNum()]);
            }
            if (!allResolvable) {
                continue;
            }
            int g = qrPtr.size();
            qrPtr.add(qrRows.size());
            qrRows.add(vCol[bus.getNum()]);                  // candidate 0 = the FALLBACK (control lost)
            qrRows.addAll(memberRows);
            qRowAlt[bus.getNum()] = -2 - g;
            qrPilotBuses.add(bus.getNum());
        }
        qrPtr.add(qrRows.size());

        // Remote gen-VC group members (plan-B2), CANONICAL encoding independent of the base pin
        // state: the frozen q dest is SUPPRESSED and the member's OWN row is the ALTERNATE,
        // selected per scenario by row mode 4 ("member row hosts its own Q balance" — RL pinned).
        // The FREE member joins the canonical encoding when its pilot carries a relocation group
        // (plan-B4: the pilot's Q follows the mode-scan carrier, so the free member's row is free
        // to host its own Q balance when pinned); without a group it keeps the structure-frozen
        // pilot-carrier encoding and free pins stay whole-run CPU.
        // Every member is canonical since the ob/zi alternate-dest channels: a pinned (mode 4)
        // member's half-open charging and dummy-Q terms follow the same per-scenario routing as
        // its cb/sh terms (the pre-channel gap measured 3.3e-4 kV on a pinned member with one
        // open line, then guarded — the guard is gone).
        for (LfBus bus : network.getBuses()) {
            int bn = bus.getNum();
            if (vCol[bn] >= 0 && GpuAcRowIndexing.isRemoteGenVcMember(bus)
                    && (!GpuAcRowIndexing.isFreeRemoteGenVcMember(bus)
                        || qrPilotBuses.contains(bus.getGeneratorVoltageControl().orElseThrow()
                                .getControlledBus().getNum()))) {
                qRow[bn] = -1;
                qRowAlt[bn] = vCol[bn];
            }
        }
        return new ExtractionColumns(phiCol, vCol, branchRhoCol, branchAlphaCol, dummyPCol, dummyQCol, shuntBCol, qRow, qRowAlt,
                GpuAcArrays.toIntArray(qrPtr), GpuAcArrays.toIntArray(qrRows));
    }

}
