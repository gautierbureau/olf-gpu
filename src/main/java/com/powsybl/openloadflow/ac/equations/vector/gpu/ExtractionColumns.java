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
                         /** Per candidate group: the BASE group a DEPENDENT pilot's group belongs to, -1 for a base
                          *  group. A merged remote group's dependent controlled bus becomes the pilot of OLF's re-merged
                          *  group when a contingency cuts it from the main: its reactive balance then rides a member row
                          *  of the same table, which this group's pool reserves (see the dependent-pilot rule in the
                          *  native pick functions). */
                         int[] qrDepBase,
                         /** Per candidate group: the row holding the pilot's reactive target in the BASE structure for a
                          *  dependent pilot (its relocated reactive row), -1 for a base group (the base carrier's). */
                         int[] qrBaseTargetRow,
                         /** Per DEVICE row: the dependent-pilot group of the bus whose own V row this is, -1 elsewhere. */
                         int[] depGroupOfRow,
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
        // The coupler-tree relocation, computed HERE rather than after the candidate tables: a child bus's power
        // equations move onto its parent coupler's dummy columns, so every row of a coupler chain keeps a
        // structurally non-zero diagonal — and the tables below must name the row that ends up hosting each
        // member's reactive balance, which for a relocated member is that dummy column and no longer its own V row.
        int[][] ziTreeEdges = GpuAcRowIndexing.ziTreeRelocation(network, es, phiCol, vCol, qRow, dummyPCol, dummyQCol);
        // GROUP pilot-Q relocation candidates (plan-A6): a shared transformer-VC pilot's reactive
        // balance rides on the FREE member's rho column — "first enabled", which an outage or clamp
        // of the free member RE-PICKS per scenario. Encode such pilots' alternate as -2-groupId and
        // ship the candidate table [pilot v row (fallback), member rho cols in merged order]; the
        // fill kernels select the FIRST member row whose per-scenario mode is not identity/pinned
        // (exactly OLF's free-role rule), falling back to the pilot's own row when control is off.
        List<Integer> qrPtr = new ArrayList<>();
        List<Integer> qrRows = new ArrayList<>();
        List<Integer> qrDepBase = new ArrayList<>();
        List<Integer> qrBaseTargetRow = new ArrayList<>();
        int[] depGroupOfRow = new int[es.getIndex().getSortedVariablesToFind().size()];
        java.util.Arrays.fill(depGroupOfRow, -1);
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
            qrDepBase.add(-1);
            qrBaseTargetRow.add(-1);
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
            // the ONE enrolment predicate (GpuAcRowIndexing.hasCandidateTable): the same answer decides the
            // canonical, pin-state-independent encoding of the group's q rows and DISTR_Q pack
            if (!GpuAcRowIndexing.hasCandidateTable(gvc, es)) {
                continue;
            }
            // Each member's row is the one that HOSTS its reactive balance, which the coupler-tree relocation can
            // move to the parent coupler's DUMMY_Q column (exactly as GpuAcRowIndexing.distrQRow's tree overload
            // already moves its DISTR_Q). Naming its own V row instead sent a pinned pilot's raw q onto a row the
            // relocation had given to the coupler's ZERO_V, and left the pilot's own row empty: measured on
            // rte6515's [5000,6000) window with the relocation on, the failing slot's residual sat on
            // BUS_V(VL-6003_1) at 6.5e+02 and on the empty pilot row BUS_V(VL-6003_0) at 1.09e+02, every iteration.
            List<Integer> memberRows = new ArrayList<>();
            for (LfBus m : gvc.getMergedControllerElements()) {
                memberRows.add(GpuAcRowIndexing.relocatedQRow(m.getNum(), ziTreeEdges[0], dummyQCol, vCol));
            }
            int g = qrPtr.size();
            qrPtr.add(qrRows.size());
            // candidate 0 = the FALLBACK (control lost), the pilot's own reactive row — relocated like any other
            qrRows.add(GpuAcRowIndexing.relocatedQRow(bus.getNum(), ziTreeEdges[0], dummyQCol, vCol));
            qrRows.addAll(memberRows);
            qRowAlt[bus.getNum()] = -2 - g;
            qrPilotBuses.add(bus.getNum());
            qrDepBase.add(-1);
            qrBaseTargetRow.add(-1);
            // DEPENDENT PILOTS. OLF merges the controls of one zero-impedance network into this MAIN; a contingency
            // that cuts a dependent controlled bus from the main re-merges the survivors and elects a new main among
            // them (LfZeroImpedanceNetwork.updateVoltageControlMergeStatus), whose BUS_TARGET_V comes back and whose
            // reactive balance must ride a FREE controller's row - the same member rows as this table, with that
            // bus's own reactive terms in the pattern. Each such bus gets a group of its own: candidate 0 its own
            // reactive row (the fallback when the re-merged control is lost), then the base members in merged order,
            // so its carrier pick coincides with the base group's on the same modes. A dependent bus that is itself
            // one of the controllers has no balance of its own to host (pilot-self) and is left out.
            for (var dvc : gvc.getMergedDependentVoltageControls()) {
                LfBus dep = dvc.getControlledBus();
                int dn = dep.getNum();
                if (vCol[dn] < 0 || gvc.getMergedControllerElements().contains(dep) || depGroupOfRow[vCol[dn]] >= 0) {
                    continue;
                }
                int dg = qrPtr.size();
                qrPtr.add(qrRows.size());
                int depQRow = GpuAcRowIndexing.relocatedQRow(dn, ziTreeEdges[0], dummyQCol, vCol);
                qrRows.add(depQRow);
                qrRows.addAll(memberRows);
                qrDepBase.add(g);
                qrBaseTargetRow.add(depQRow);
                depGroupOfRow[vCol[dn]] = dg;
            }
            if (pilotSelf && System.getenv("OLF_GPU_RL_TRACE") != null) {
                System.err.println("PILOTSELF_ENROL group=" + g + " pilot=" + bus.getId()
                        + " members=" + gvc.getMergedControllerElements().size()
                        + " fallbackRow=" + vCol[bus.getNum()]);
            }
        }
        qrPtr.add(qrRows.size());
        if (System.getenv("OLF_GPU_QR_TRACE") != null) {   // every relocation group of THIS extraction
            for (int g = 0; g + 1 < qrPtr.size(); g++) {
                StringBuilder sb = new StringBuilder("QR_TRACE group=" + g + " rows=");
                for (int c = qrPtr.get(g); c < qrPtr.get(g + 1); c++) {
                    sb.append(qrRows.get(c)).append(c == qrPtr.get(g) ? "|" : ",");
                }
                System.err.println(sb);
            }
            for (LfBus bus : network.getBuses()) {           // every generator-VC controller: its state at THIS extraction
                if (bus.getGeneratorVoltageControl().isPresent()) {
                    System.err.println("QR_TRACE ctl bus=" + bus.getId() + " vRow=" + vCol[bus.getNum()]
                            + " enabled=" + bus.isGeneratorVoltageControlEnabled()
                            + " qLimit=" + bus.getQLimitType().map(Object::toString).orElse("-")
                            + " minQ=" + bus.getMinQ() + " maxQ=" + bus.getMaxQ()
                            + " genTargetQ=" + bus.getGenerationTargetQ() + " loadQ=" + bus.getLoadTargetQ()
                            + " q=" + (bus.getQ() != null ? bus.getQ().eval() : Double.NaN)
                            + " controlled=" + bus.getGeneratorVoltageControl().map(c -> c.getControlledBus().getId()).orElse("-")
                            + " main=" + bus.getGeneratorVoltageControl().map(c -> c.getMergeStatus().name()).orElse("-"));
                }
                if (vCol[bus.getNum()] >= 0 && qRowAlt[bus.getNum()] != -1) {
                    System.err.println("QR_TRACE bus=" + bus.getId() + " vRow=" + vCol[bus.getNum()]
                            + " qRowAlt=" + qRowAlt[bus.getNum()]
                            + " gvcMain=" + bus.getGeneratorVoltageControl().map(c -> c.getMergeStatus().name()).orElse("-")
                            + " vcOn=" + bus.isGeneratorVoltageControlled()
                            + " pinned=" + (bus.getGeneratorVoltageControl().isPresent() && !bus.isGeneratorVoltageControlEnabled()));
                }
            }
        }

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
                        // the MAIN control's pilot: on a member's own (merged, dependent) control the
                        // controlled bus is the member itself, which is never in the pilot set
                        || qrPilotBuses.contains(GpuAcRowIndexing.mainGenVc(bus).getControlledBus().getNum()))) {
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
            // `== -1` and NOT `< 0`: a group encoding is -2-groupId, which is also negative, so a `< 0`
            // test CLOBBERED the enrolment the loop above had just made — silently, and only for the
            // pilots whose q is suppressed at extraction, which is exactly the unpinned pilot-self case the
            // rotation exists for. Measured on rte6515's VL-6112 group: with the rotation property ON the
            // DISTR_Q contributions still reported qrGroup=-1, so the carrier gate in fillDistrQBatch could
            // never fire and the re-picked free member kept a DISTR_Q equation OLF had deactivated.
            if (vCol[bn] >= 0 && qRow[bn] < 0 && qRowAlt[bn] == -1
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
        int[] ziParentEdge = ziTreeEdges[0];
        int[] ziChildBus = ziTreeEdges[1];
        int[] pRow = phiCol.clone();
        if (System.getenv("OLF_GPU_ZI_RELOC_TRACE") != null) {
            long relocated = java.util.Arrays.stream(ziParentEdge).filter(x -> x >= 0).count();
            System.err.printf("ZI_RELOC relocatedBuses=%d of %d%n", relocated, nBuses);
            // =ids: WHICH buses relocate, and what each one carries.
            if ("ids".equals(System.getenv("OLF_GPU_ZI_RELOC_TRACE"))) {
                for (LfBus b : network.getBuses()) {
                    int pe = ziParentEdge[b.getNum()];
                    if (pe >= 0) {
                        System.err.printf("ZI_RELOC_BUS %s parentEdge=%d gens=%d shunt=%b loadModel=%b"
                                + " vc=%b ctrl=%b p0=%.4f q0=%.4f%n",
                                b.getId(), pe, b.getGenerators().size(),
                                b.getShunt().isPresent() || b.getControllerShunt().isPresent()
                                        || b.getSvcShunt().isPresent(),
                                b.getLoads().stream().anyMatch(l -> l.getLoadModel().isPresent()),
                                b.isGeneratorVoltageControlled(),
                                b.isGeneratorVoltageControlEnabled(), b.getLoadTargetP(), b.getLoadTargetQ());
                    }
                }
            }
        }
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
            // ...and so does the ALTERNATE row, the one every family writes the bus's OWN q to when a reactive
            // limit puts it in mode 4. For a relocated child that pin lands on this coupler's DUMMY_Q row (the
            // native pass diverts it there through rlQrow, since that is where its balance lives), so leaving the
            // alternate on the bus's own BUS_V row sent its own q onto the row now holding the coupler's ZERO_V
            // and left the DUMMY_Q column held by nothing but the DISTR_Q terms the mode-4 flip gates off —
            // a DEAD COLUMN. Measured on rte6515's full busbar scope: 9 mode-4 rows on relocated couplers of
            // VL-5997/VL-6003/VL-6194/VL-6229, deadCols=2 (first 14417 = DUMMY_Q(BUS-6003_COUPLER_2)),
            // tinyDiag=7, and 6616 of 6876 scenarios rejecting their second iteration at a ratio of 1e+10.
            // Only the plain own-row alternate moves: a candidate-table pilot's is encoded as -2-group.
            if (qRowAlt[bn] == vCol[bn] && dummyQCol[parent] >= 0) {
                qRowAlt[bn] = dummyQCol[parent];
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
                GpuAcArrays.toIntArray(qrDepBase), GpuAcArrays.toIntArray(qrBaseTargetRow), depGroupOfRow,
                pRow, ziParentEdge, ziChildBus);
    }

}
