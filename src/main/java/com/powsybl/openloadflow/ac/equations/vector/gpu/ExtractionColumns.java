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
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LoadFlowModel;
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
                         int[] qrPtr, int[] qrRows, int[] qrGroupOfPilot, int[] ziDeadEnd,
                         /** Each bus's P-balance ROW: its own BUS_PHI column, or its parent coupler's DUMMY_P under
                          *  the coupler-tree relocation (GpuAcRowIndexing#ziTreeRelocation). */
                         int[] pRow,
                         /** Per bus, the coupler whose dummy columns host its power equations (-1 = root/none), and
                          *  per branch, the bus whose PHI/V host its zero equations (-1 = not relocated). */
                         int[] ziParentEdge, int[] ziChildBus) {

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
            // ...but only WHILE IT CONTROLS. Once a reactive limit PINS such a pilot, OLF activates its
            // BUS_TARGET_Q, which needs a row, and its own row still carries BUS_TARGET_V because a sibling
            // controller keeps the voltage — so the group DOES need the table, with the pilot's raw-q
            // emission GATED on it being pinned (psPilotPinned in olf_gpu_nr_jni.cu). Property-gated while
            // this lands; without it such a group keeps a relocation frozen at extraction, which is what
            // leaves rte6515's busbar base rank-deficient.
            boolean pilotSelf = gvc != null && gvc.getMergedControllerElements().contains(bus);
            boolean pilotSelfRotates = pilotSelf && GpuAcRowIndexing.pilotSelfRotationEnabled()
                    && gvc.getMergedControllerElements().size() >= 2;
            if (gvc == null || gvc.getMergedControllerElements().isEmpty() || vCol[bus.getNum()] < 0
                    || (pilotSelf && !pilotSelfRotates)) {
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
            if (pilotSelf && System.getenv("OLF_GPU_RL_TRACE") != null) {
                System.err.println("PILOTSELF_ENROL group=" + g + " pilot=" + bus.getId()
                        + " members=" + gvc.getMergedControllerElements().size()
                        + " fallbackRow=" + vCol[bus.getNum()]);
            }
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
        // A pilot that is ALSO one of its own controllers. computeQRow already suppressed its per-bus
        // Q scatter (qRow = -1 — its reactive injection feeds the group's DISTR_Q rows), and its own
        // comment names the move that is missing: "if the controller is switched to PQ by reactive
        // limits, BUS_TARGET_Q becomes active again and the bus must recover its own BUS_V-backed Q
        // row". That recovery is the member ALTERNATE dest + row mode 4, which this bus never got
        // because isRemoteGenVcMember is defined as controlledBus != bus and so excludes a pilot by
        // construction. Give it the alternate; without one, a row put in mode 4 assembles EMPTY (no q
        // contributions were ever routed to it), which is measurably a no-op: on rte6515 GEN-1522 the
        // solve was unchanged to 10 digits and the answer stayed 3.2 kV out.
        // INERT IN THE BASE: the alternate is read only when the row is mode 4, and while the pilot
        // controls, its row is a mode-0 BUS_TARGET_V identity.
        for (LfBus bus : network.getBuses()) {
            int bn = bus.getNum();
            if (vCol[bn] >= 0 && qRow[bn] < 0 && qRowAlt[bn] < 0
                    && GpuAcRowIndexing.isRemoteGenVcControllerElement(bus)
                    && bus.getGeneratorVoltageControl()
                            .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                            .map(c -> c.getControlledBus() == bus).orElse(false)) {
                qRowAlt[bn] = vCol[bn];
            }
        }
        // Which relocation group a pilot OWNS, as its own channel. qRowAlt = -2-g has always meant two
        // separate things at once: "route this pilot's Q through the relocation carrier" AND "this
        // pilot has a candidate table". Consumers that want only the second (the member -> group link
        // that gates the gen-VC kernels) had to decode the first, which makes it impossible to give a
        // group a table WITHOUT also re-routing its Q — exactly what a pilot that is one of its own
        // controllers needs, since its q already feeds the group's DISTR_Q rows. Derived here from the
        // final qRowAlt, so this commit is bit-identical to the decode it replaces.
        int[] qrGroupOfPilot = new int[nBuses];
        java.util.Arrays.fill(qrGroupOfPilot, -1);
        for (int b = 0; b < nBuses; b++) {
            if (qRowAlt[b] <= -2) {
                qrGroupOfPilot[b] = -2 - qRowAlt[b];
            }
        }
        // Per zero-impedance branch, which side is a DEAD-END STUB safe to relocate (0 = none):
        // GpuAcRowIndexing#ziDeadEndSide. GpuAcActivationExtractor derives the same answer from the
        // same helper, so the scatter destinations and the target map name the same rows.
        int[] ziDeadEnd = new int[nBranches];
        for (LfBranch br : network.getBranches()) {
            if (br.isZeroImpedance(LoadFlowModel.AC)) {
                ziDeadEnd[br.getNum()] = GpuAcRowIndexing.ziDeadEndSide(es, br, phiCol, vCol, qRow);
            }
        }
        // The coupler-tree relocation: a child bus's power equations move onto its parent coupler's dummy columns,
        // so every row of a coupler chain keeps a structurally non-zero diagonal.
        int[][] ziTree = GpuAcRowIndexing.ziTreeRelocation(network, es, phiCol, vCol, qRow, dummyPCol, dummyQCol);
        int[] ziParentEdge = ziTree[0];
        int[] ziChildBus = ziTree[1];
        int[] pRow = phiCol.clone();
        for (LfBus bus : network.getBuses()) {
            int bn = bus.getNum();
            int parent = ziParentEdge[bn];
            if (parent < 0) {
                continue;
            }
            pRow[bn] = dummyPCol[parent];
            if (qRow[bn] == vCol[bn]) {                       // a DISTR_Q child moves through distrQRow instead
                qRow[bn] = dummyQCol[parent];
            }
        }
        for (int br = 0; br < nBranches; br++) {
            int child = ziChildBus[br];
            if (child >= 0 && ziParentEdge[child] != br) {
                ziChildBus[br] = -1;                          // its child's relocation did not survive the check
            }
        }
        return new ExtractionColumns(phiCol, vCol, branchRhoCol, branchAlphaCol, dummyPCol, dummyQCol, shuntBCol, qRow, qRowAlt,
                GpuAcArrays.toIntArray(qrPtr), GpuAcArrays.toIntArray(qrRows), qrGroupOfPilot, ziDeadEnd,
                pRow, ziParentEdge, ziChildBus);
    }

}
