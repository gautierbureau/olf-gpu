/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.commons.PowsyblException;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuReactiveTermKind;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.GeneratorVoltageControl;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LoadFlowModel;
import com.powsybl.openloadflow.network.VoltageControl;

import java.util.ArrayList;
import java.util.List;

/**
 * The DISTR_Q rows of remote generator voltage control: each controller's reactive share of its group, as the
 * branch and shunt reactive-flow terms that sum to it (OLF DistributionReactivePowerEquationTerm).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class DistrQPacker {

    private DistrQPacker() {
    }

    /** The DISTR_Q terms and the members they refer to. */
    record Pack(DistrQTerms terms, DistrQMembers members) {
    }

    /** Per-group member table for the IN-KERNEL reactive-key renormalization on a remote RL pin
     *  (plan-B3): OLF's updateReactiveKeys zeroes a controller-DISABLED member's key and
     *  renormalizes over the survivors — so the DISTR_Q weights are PIN-STATE-DEPENDENT and the
     *  kernel recomputes them per scenario from the RAW keys (Control.createReactiveKeys) and the
     *  per-(entry, scenario) limitType, replicating the CPU's zero-then-sum-in-member-order
     *  arithmetic bit-for-bit. distrQRowMember/distrQInjectionMember are −1 for non-group contributions. */
    private record MemberTables(java.util.Map<Integer, Integer> memberIndexByBusNum,
                                List<Double> key, List<Integer> reactiveLimitEntry,
                                List<Integer> groupStart, List<Integer> groupCount,
                                List<Integer> row, List<Byte> hasRow, List<Double> targetQ,
                                List<Integer> busNum,
                                /** Per member, the members whose targetQ OLF sums into its DISTR_Q target: the
                                 *  elements of the member's OWN voltage control, which for a control MERGED into
                                 *  another (DEPENDENT) is just its own — see the peer CSR in {@link DistrQPack}. */
                                List<List<Integer>> targetPeers) {
    }

    static Pack distrQPack(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                             ExtractionColumns columns, GpuAcFamilyPacker.Packs family, ReactiveLimitEntries reactiveLimits) {
        // member tables (plan-B3): raw keys + RL-entry mapping + group ranges, in group member order
        MemberTables memberTables = new MemberTables(new java.util.HashMap<>(), new ArrayList<>(), new ArrayList<>(),
                new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), new ArrayList<>(),
                new ArrayList<>(), new ArrayList<>());
        java.util.Map<Integer, Integer> reactiveLimitEntryByBusNum = new java.util.HashMap<>();
        for (int i = 0; i < reactiveLimits.busNum().length; i++) {
            reactiveLimitEntryByBusNum.put(reactiveLimits.busNum()[i], i);
        }
        List<Integer> relocationGroupList = new ArrayList<>();
        for (LfBus pilot : network.getBuses()) {
            GeneratorVoltageControl voltageControl = pilot.getGeneratorVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .filter(c -> pilot.isGeneratorVoltageControlled() && !c.isLocalControl())
                    .orElse(null);
            if (voltageControl == null) {
                continue;
            }
            List<LfBus> controllers = voltageControl.getMergedControllerElements();
            double[] rawKeys = com.powsybl.openloadflow.network.Control.createReactiveKeys(
                    controllers, com.powsybl.openloadflow.network.LfGenerator.GeneratorControlType.VOLTAGE);
            // A controller bus a contingency disabled takes no share (GeneratorVoltageControl.updateReactiveKeys sets
            // its key to 0) and is never the reference controller (updateRemoteVoltageControlEquations discards
            // disabled controller elements before picking the first enabled one).
            for (int i = 0; i < controllers.size(); i++) {
                if (controllers.get(i).isDisabled()) {
                    rawKeys[i] = 0;
                }
            }
            int start = memberTables.key().size();
            LfBus freeMember = controllers.stream().filter(b -> !b.isDisabled() && voltageControl.isControllerEnabled(b))
                    .findFirst().orElse(null);
            // the pilot's relocation group: the kernels re-pick the pilot-Q carrier by mode-scan, so
            // the CURRENT carrier's DISTR_Q row is suppressed and its target carries the pilot's Q
            // instead. Read from reactiveRelocationGroupOfPilot, NOT decoded from alternateQRow: table membership and Q
            // re-routing are separate facts that happened to share one encoding.
            int pilotRelocationGroup = columns.reactiveRelocationGroupOfPilot()[pilot.getNum()];
            for (int i = 0; i < controllers.size(); i++) {
                LfBus m = controllers.get(i);
                memberTables.memberIndexByBusNum().put(m.getNum(), memberTables.key().size());
                memberTables.key().add(rawKeys[i]);
                memberTables.reactiveLimitEntry().add(reactiveLimitEntryByBusNum.getOrDefault(m.getNum(), -1));
                // The member's DISTR_Q row, NOT simply its V column: under the coupler-tree relocation a
                // relocated child's DISTR_Q lives on its parent coupler's DUMMY_Q (the same move
                // buildDistrQContributions makes through distrQRow's tree overload), and its own V row
                // carries that coupler's ZERO_V instead. Pointing this bookkeeping at the V row made the
                // DISTR_Q target updater write there, which overwrote the BUS_TARGET_V target a scenario
                // opening the coupler had just set - measured as row 12566 ending at target 0 instead of
                // 1.142 on BUS-6014_BBS_2.
                int mParent = columns.spanningTreeParentBranch()[m.getNum()];
                memberTables.row().add(mParent >= 0 && columns.dummyQCol()[mParent] >= 0
                        ? columns.dummyQCol()[mParent] : columns.vCol()[m.getNum()]);
                // a member OWNS a DISTR_Q row when the equation EXISTS and it is not the FREE
                // member (whose row carries the pilot's relocated Q)
                memberTables.hasRow().add((byte) (m != freeMember && !m.isDisabled()
                        && es.getEquation(m.getNum(), AcEquationType.DISTR_Q).isPresent() ? 1 : 0));
                // the member's UNPINNED targetQ (generationTargetQ - loadTargetQ), which is what
                // AcTargetVector reads for every member the RL loop has not frozen; a frozen
                // member's targetQ is its q limit and the kernel derives that from the RL tables.
                memberTables.targetQ().add(ReactiveLimitsPacker.unpinnedTargetQ(m));
                memberTables.busNum().add(m.getNum());
                // AcTargetVector reads the member's OWN control's merged elements, so a control merged INTO this
                // group (DEPENDENT: its own list is just its controller) sums only its own targetQ. Recorded as bus
                // numbers here and resolved to member indices once every member is known.
                memberTables.targetPeers().add(m.getGeneratorVoltageControl()
                        .map(c -> c.getMergedControllerElements().stream().map(LfBus::getNum).toList())
                        .orElse(List.of(m.getNum())));
                relocationGroupList.add(pilotRelocationGroup);
            }
            for (int i = 0; i < controllers.size(); i++) {
                memberTables.groupStart().add(start);
                memberTables.groupCount().add(controllers.size());
            }
        }
        List<int[]> distrQContributions = new ArrayList<>();      // {elem, side, distrQRow, kind, kMember, jMember} per DISTR_Q contribution
        List<Double> distrQWeights = new ArrayList<>();    // weight per contribution
        buildDistrQContributions(network, es, columns.vCol(), family.closedBranchElementOf(), family.shuntElementOf(),
                family.openBranchElementOf(), columns.dummyQCol(), columns.spanningTreeParentBranch(), distrQContributions, distrQWeights, memberTables.memberIndexByBusNum());
        // distrQKind: a GpuReactiveTermKind (for DUMMY_Q, elem holds the var column)
        int[] distrQKind = new int[distrQContributions.size()];
        int[] distrQRowMember = new int[distrQContributions.size()];
        int[] distrQInjectionMember = new int[distrQContributions.size()];
        int[] distrQElement = new int[distrQContributions.size()];
        int[] distrQSide = new int[distrQContributions.size()];
        int[] distrQRow = new int[distrQContributions.size()];
        double[] distrQWeight = new double[distrQContributions.size()];
        for (int i = 0; i < distrQContributions.size(); i++) {
            distrQElement[i] = distrQContributions.get(i)[0];
            distrQSide[i] = distrQContributions.get(i)[1];
            distrQRow[i] = distrQContributions.get(i)[2];
            distrQKind[i] = distrQContributions.get(i)[3];
            distrQRowMember[i] = distrQContributions.get(i)[4];
            distrQInjectionMember[i] = distrQContributions.get(i)[5];
            distrQWeight[i] = distrQWeights.get(i);
        }
        byte[] hasRow = new byte[memberTables.hasRow().size()];
        for (int i = 0; i < hasRow.length; i++) {
            hasRow[i] = memberTables.hasRow().get(i);
        }
        int[] tgtPtr = new int[memberTables.targetPeers().size() + 1];
        List<Integer> tgtPeer = new ArrayList<>();
        for (int i = 0; i < memberTables.targetPeers().size(); i++) {
            tgtPtr[i] = tgtPeer.size();
            for (int busNum : memberTables.targetPeers().get(i)) {
                Integer peer = memberTables.memberIndexByBusNum().get(busNum);
                if (peer != null) {                          // a peer outside the packed group cannot be summed
                    tgtPeer.add(peer);
                }
            }
        }
        tgtPtr[tgtPtr.length - 1] = tgtPeer.size();
        return new Pack(new DistrQTerms(distrQElement, distrQSide, distrQWeight, distrQRow, distrQKind, distrQRowMember, distrQInjectionMember),
                new DistrQMembers(GpuAcArrays.toDoubleArray(memberTables.key()), GpuAcArrays.toIntArray(memberTables.reactiveLimitEntry()),
                GpuAcArrays.toIntArray(memberTables.groupStart()), GpuAcArrays.toIntArray(memberTables.groupCount()),
                GpuAcArrays.toIntArray(memberTables.row()), hasRow, GpuAcArrays.toIntArray(relocationGroupList),
                GpuAcArrays.toDoubleArray(memberTables.targetQ()), GpuAcArrays.toIntArray(memberTables.busNum()),
                tgtPtr, GpuAcArrays.toIntArray(tgtPeer)));
    }

    /**
     * Build the weighted closed-branch reactive contributions of every active DISTR_Q row (remote
     * generator voltage control). For a control group, the row at controller {@code Gk}'s BUS_V row is
     * {@code DISTR_Q_k = qPct_k·Σ_j q_Gj − q_Gk}, so each controller {@code Gj}'s reactive injection
     * (the q1/q2 flow of each branch incident to {@code Gj}) contributes with weight
     * {@code qPct_k − (j==k ? 1 : 0)}. A controller's reactive injection is its closed-branch q1/q2 (kind 0),
     * a fixed shunt's q (kind 1), a half-open incident branch's connected-side q (kind 2), or a zero-impedance
     * spanning-tree branch's linear DUMMY_Q variable term (kind 3) — mirroring the terms {@code createReactiveTerms}
     * sums. The pilot bus's own reactive balance is handled by qRow (routed to the free controller's BUS_V row);
     * the controllers' own per-bus Q scatter is suppressed there too.
     */
    private static void buildDistrQContributions(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                                 int[] vCol, int[] closedBranchElementOf, int[] shuntElementOf, int[] openBranchElementOf,
                                                 int[] dummyQCol, int[] spanningTreeParentBranch, List<int[]> out, List<Double> weights,
                                                 java.util.Map<Integer, Integer> memberIndexByBusNum) {
        for (LfBus pilot : network.getBuses()) {
            GeneratorVoltageControl voltageControl = pilot.getGeneratorVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .filter(c -> pilot.isGeneratorVoltageControlled() && !c.isLocalControl())
                    .orElse(null);
            if (voltageControl == null) {
                continue;
            }
            List<LfBus> controllers = voltageControl.getMergedControllerElements();
            // CANONICAL for a group with a candidate table: the free (reference) controller is the FIRST
            // member, as in the all-enabled state — the per-scenario row modes and the carrier gate decide
            // which DISTR_Q rows are live. Else (no table) the extraction-time free member, as before.
            LfBus freeCtl = GpuAcRowIndexing.hasCandidateTable(voltageControl, es)
                    ? controllers.get(0)
                    : controllers.stream().filter(b -> !b.isDisabled() && voltageControl.isControllerEnabled(b)).findFirst().orElse(null);
            for (LfBus gk : controllers) {
                // EXISTENCE-keyed (the A1 lesson): pack every non-free member whose DISTR_Q equation
                // EXISTS — a base-PINNED member's row is inactive (mode 4, own Q) but an in-flight
                // UNPIN re-activates it, and the union pattern must already hold its slots. The row-
                // mode/fill gates decide per scenario which face the row shows.
                if (gk == freeCtl || gk.isDisabled() || es.getEquation(gk.getNum(), AcEquationType.DISTR_Q).isEmpty()) {
                    continue;                                // the free controller carries the pilot's Q; a disabled
                }                                            // controller's DISTR_Q is disabled with its bus
                int distrQRow = GpuAcRowIndexing.distrQRow(es, gk, spanningTreeParentBranch, dummyQCol);
                double qpk = gk.getRemoteControlReactivePercent();
                int kIdx = memberIndexByBusNum.getOrDefault(gk.getNum(), -1);
                for (LfBus gj : controllers) {
                    if (gj.isDisabled()) {
                        continue;                            // its reactive terms are disabled with its branches
                    }
                    double weight = qpk - (gj == gk ? 1.0 : 0.0);
                    int jIdx = memberIndexByBusNum.getOrDefault(gj.getNum(), -1);
                    if (gj.getControllerShunt().isPresent()) {
                        throw new PowsyblException("GPU AC solver: DISTR_Q controller " + gj.getId()
                                + " has a voltage-controlling shunt (SHUNT_B) reactive contribution (not supported)");
                    }
                    if (gj.getShunt().isPresent()) {         // a fixed shunt's reactive joins the q sum
                        out.add(new int[] {shuntElementOf[gj.getNum()], 0, distrQRow, GpuReactiveTermKind.SHUNT_COMPENSATOR_REACTIVE_FLOW, kIdx, jIdx});
                        weights.add(weight);
                    }
                    for (LfBranch brj : gj.getBranches()) {
                        if (brj.isDisabled()) {
                            continue;
                        }
                        int side = brj.getBus1() == gj ? 0 : 1;
                        int e = closedBranchElementOf[brj.getNum()];
                        if (e >= 0) {                        // closed branch: weighted q1/q2 (kind 0)
                            out.add(new int[] {e, side, distrQRow, GpuReactiveTermKind.CLOSED_BRANCH_REACTIVE_FLOW, kIdx, jIdx});
                            weights.add(weight);
                            continue;
                        }
                        int oe = openBranchElementOf[brj.getNum()];
                        if (oe >= 0) {                       // half-open branch incident on the controller:
                            // its connected-side reactive flow (shunt + series) joins the q sum (kind 2)
                            out.add(new int[] {oe, side, distrQRow, GpuReactiveTermKind.OPEN_BRANCH_REACTIVE_FLOW, kIdx, jIdx});
                            weights.add(weight);
                            continue;
                        }
                        if (brj.isZeroImpedance(LoadFlowModel.AC)) {
                            // a zero-impedance coupler: its reactive flow is the linear DUMMY_Q variable (+1 if the
                            // controller is bus1, −1 if bus2) — the term createReactiveTerms adds for a spanning-tree
                            // edge. EVERY coupler is packed, not just the base tree's: OLF re-elects its tree per
                            // contingency, and a coupler out of the BASE tree that a contingency brings into it
                            // carries flow there. The kernel drops the term in every scenario where the coupler's
                            // dummy row is an identity (out of the tree, or opened), which is exactly the base tree's
                            // own filter. Packing only the base tree left that flow out of every member's DISTR_Q:
                            // network.xiidm busbar HFU re-trees HFT_1's couplers HHQ/HHS/HHO, and the reactive flow
                            // circulating HFT_3 -> HFT_1 -> HFT_2 was in no equation at all (1.25e12 pu after the
                            // first solve, its rounding moving the slack by +-0.058 MW from one run to the next).
                            int dummyQColumn = dummyQCol[brj.getNum()];
                            if (dummyQColumn < 0) {
                                throw new PowsyblException("GPU AC solver: DISTR_Q controller " + gj.getId()
                                        + " has a zero-impedance branch " + brj.getId() + " without a DUMMY_Q variable");
                            }
                            out.add(new int[] {dummyQColumn, side, distrQRow, GpuReactiveTermKind.DUMMY_Q, kIdx, jIdx});  // elem = the DUMMY_Q var column
                            weights.add(weight * (side == 0 ? 1.0 : -1.0));
                            continue;
                        }
                        throw new PowsyblException("GPU AC solver: DISTR_Q controller " + gj.getId()
                                + " has an unsupported branch " + brj.getId() + " reactive contribution");
                    }
                }
            }
        }
    }
}
