/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix;

import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuRowMode;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.VoltageControl;
import com.powsybl.openloadflow.network.gpu.GeneratorVoltageControlRoles;
import com.powsybl.openloadflow.util.gpu.GpuTrace;

import java.util.ArrayList;
import java.util.List;

/**
 * docs/gpu-vs-olf.md section 2.3 (remote generator voltage control: the carrier; transformer voltage control): the
 * candidate tables of the reactive relocation, built once per extraction from the base structure: per transformer
 * voltage-control group and per remote generator voltage-control group (and each of its dependent pilots), the run of
 * rows that may host the pilot's reactive balance, the canonical member encoding of {@code qRow} /
 * {@code alternateQRow}, and which group each pilot owns.
 *
 * @param candidates the candidate groups and their dependent-pilot links
 * @param groupOfPilot per bus, the relocation group the bus owns as a pilot, -1 for none
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record ReactiveRelocationTables(ReactiveRelocationCandidates candidates, int[] groupOfPilot) {

    /**
     * Builds the candidate tables. Mutates {@code qRow} and {@code alternateQRow} (the group encoding -2-groupId and
     * the canonical member encoding). {@code spanningTreeParentBranch} is the coupler-tree relocation's
     * parent-coupler-of-bus array ({@link CouplerTreeRelocation#spanningTreeRelocation}), which must be computed
     * before: the tables name the row that ends up hosting each member's reactive balance.
     */
    public static ReactiveRelocationTables build(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                                 int[] vCol, int[] branchRhoCol, int[] dummyQCol,
                                                 int[] spanningTreeParentBranch, int[] qRow, int[] alternateQRow) {
        int nBuses = network.getBuses().size();
        // GROUP pilot-Q relocation candidates (plan-A6): a shared transformer-VC pilot's reactive
        // balance rides on the FREE member's rho column — "first enabled", which an outage or clamp
        // of the free member RE-PICKS per scenario. Encode such pilots' alternate as -2-groupId and
        // ship the candidate table [pilot v row (fallback), member rho cols in merged order]; the
        // fill kernels select the FIRST member row whose per-scenario mode is neither an identity nor a BUS_TARGET_Q balance
        // (exactly OLF's free-role rule), falling back to the pilot's own row when control is off.
        List<Integer> reactiveRelocationGroupStart = new ArrayList<>();
        List<Integer> reactiveRelocationCandidateRows = new ArrayList<>();
        List<Integer> reactiveRelocationDependentBaseGroup = new ArrayList<>();
        List<Integer> reactiveRelocationBaseTargetRow = new ArrayList<>();
        int[] reactiveRelocationDependentGroupOfRow = new int[es.getIndex().getSortedVariablesToFind().size()];
        java.util.Arrays.fill(reactiveRelocationDependentGroupOfRow, -1);
        for (LfBus bus : network.getBuses()) {
            var voltageControl = bus.getTransformerVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .orElse(null);
            if (voltageControl == null || voltageControl.getMergedControllerElements().size() < 2 || vCol[bus.getNum()] < 0) {
                continue;
            }
            List<Integer> memberRows = new ArrayList<>();
            boolean allResolvable = true;
            for (var m : voltageControl.getMergedControllerElements()) {
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
            int g = reactiveRelocationGroupStart.size();
            reactiveRelocationGroupStart.add(reactiveRelocationCandidateRows.size());
            reactiveRelocationCandidateRows.add(vCol[bus.getNum()]);                  // candidate 0 = the FALLBACK (control off)
            reactiveRelocationCandidateRows.addAll(memberRows);
            alternateQRow[bus.getNum()] = -2 - g;
            reactiveRelocationDependentBaseGroup.add(-1);
            reactiveRelocationBaseTargetRow.add(-1);
        }
        // Remote gen-VC pilot-Q relocation (plan-B4): the pilot's reactive balance rides on the
        // FREE member's BUS_V row — "first controller-enabled", which a PV -> PQ switch of the free member
        // RE-PICKS per scenario. Same candidate table + in-kernel mode-scan select as the TVC
        // groups above: [pilot v row (fallback = group control lost), member v cols in merged
        // order]. Since the ob/zero-impedance alternate-dest channels landed, half-open and zero-impedance q
        // terms at the pilot ride the relocation too — no branch-shape guard remains. SINGLETON
        // groups (one remote controller) get a table too since 2026-09-08: their candidate list
        // [pilot fallback, the member] is structurally a 2-member table, and the only member's switch to PQ
        // IS the no-enabled-controller corner (updatePilotTargetVEquationBatch flips the pilot's V identity) — this
        // retired the last whole-run gate on remote PV/PQ switch dynamics (a free member with reactive limits and no table).
        java.util.Set<Integer> reactiveRelocationPilotBuses = new java.util.HashSet<>();
        addRemoteVoltageControlGroups(network, es, spanningTreeParentBranch, reactiveRelocationPilotBuses, dummyQCol, vCol, reactiveRelocationGroupStart, reactiveRelocationCandidateRows,
                                      alternateQRow, reactiveRelocationDependentBaseGroup, reactiveRelocationBaseTargetRow, reactiveRelocationDependentGroupOfRow);
        reactiveRelocationGroupStart.add(reactiveRelocationCandidateRows.size());

        // Remote gen-VC group members (plan-B2), CANONICAL encoding independent of the base PV/PQ
        // state: the frozen q dest is SUPPRESSED and the member's OWN row is the ALTERNATE,
        // selected per scenario by row mode BUS_TARGET_Q ("member row hosts its own Q balance" — switched to PQ).
        // The FREE member joins the canonical encoding when its pilot carries a relocation group
        // (plan-B4: the pilot's Q follows the mode-scan carrier, so the free member's row is free
        // to host its own Q balance when switched to PQ); without a group it keeps the structure-frozen
        // pilot-carrier encoding and a free member's switches to PQ stay whole-run CPU.
        // Every member is canonical since the ob/zero-impedance alternate-dest channels: the half-open
        // charging and dummy-Q terms of a member switched to PQ (BUS_TARGET_Q) follow the same per-scenario routing as
        // its cb/sh terms (the pre-channel gap measured 3.3e-4 kV on a member switched to PQ with one
        // open line, then guarded — the guard is gone).
        for (LfBus bus : network.getBuses()) {
            int bn = bus.getNum();
            if (vCol[bn] >= 0 && GeneratorVoltageControlRoles.isRemoteGenVcMember(bus)
                    && (!GeneratorVoltageControlRoles.isFreeRemoteGeneratorVoltageControlMember(bus)
                        // the MAIN control's pilot: on a member's own (merged, dependent) control the
                        // controlled bus is the member itself, which is never in the pilot set
                        || reactiveRelocationPilotBuses.contains(GeneratorVoltageControlRoles.mainGeneratorVoltageControl(bus).getControlledBus().getNum()))) {
                qRow[bn] = -1;
                alternateQRow[bn] = vCol[bn];
            }
        }
        // A pilot that is ALSO one of its own controllers. computeQRow already suppressed its per-bus
        // Q scatter (qRow = -1 — its reactive injection feeds the group's DISTR_Q rows), and its own
        // comment names the move that is missing: "if the controller is switched to PQ by reactive
        // limits, BUS_TARGET_Q becomes active again and the bus must recover its own BUS_V-backed Q
        // row". That recovery is the member ALTERNATE dest + row mode BUS_TARGET_Q, which this bus never got
        // because isRemoteGenVcMember is defined as controlledBus != bus and so excludes a pilot by
        // construction. Give it the alternate; without one, a row put in BUS_TARGET_Q assembles EMPTY (no q
        // contributions were ever routed to it), which is measurably a no-op: on rte6515 GEN-1522 the
        // solve was unchanged to 10 digits and the answer stayed 3.2 kV out.
        // INERT IN THE BASE: the alternate is read only when the row is BUS_TARGET_Q, and while the pilot
        // controls, its row is a FIXED_VARIABLE BUS_TARGET_V identity.
        for (LfBus bus : network.getBuses()) {
            int bn = bus.getNum();
            // `== -1` and NOT `< 0`: a group encoding is -2-groupId, which is also negative, so a `< 0`
            // test CLOBBERED the enrolment the loop above had just made — silently, and only for the
            // pilots whose q is suppressed at extraction, which is exactly the case of a pilot-self still PV that
            // the rotation exists for. Measured on rte6515's VL-6112 group: with the rotation property ON the
            // DISTR_Q contributions still reported pilotRelocationGroup=-1, so the carrier gate in fillDistrQBatch could
            // never fire and the re-picked free member kept a DISTR_Q equation OLF had deactivated.
            if (vCol[bn] >= 0 && qRow[bn] < 0 && alternateQRow[bn] == -1
                    && GeneratorVoltageControlRoles.isRemoteGeneratorVoltageControlControllerElement(bus)
                    && bus.getGeneratorVoltageControl()
                            .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                            .map(c -> c.getControlledBus() == bus).orElse(false)) {
                alternateQRow[bn] = vCol[bn];
            }
        }
        // Which relocation group a pilot OWNS, as its own channel. alternateQRow = -2-g has always meant two
        // separate things at once: "route this pilot's Q through the relocation carrier" AND "this
        // pilot has a candidate table". Consumers that want only the second (the member -> group link
        // that gates the gen-VC kernels) had to decode the first, which makes it impossible to give a
        // group a table WITHOUT also re-routing its Q — exactly what a pilot that is one of its own
        // controllers needs, since its q already feeds the group's DISTR_Q rows. Derived here from the
        // final alternateQRow, so this commit is bit-identical to the decode it replaces.
        int[] reactiveRelocationGroupOfPilot = new int[nBuses];
        java.util.Arrays.fill(reactiveRelocationGroupOfPilot, -1);
        for (int b = 0; b < nBuses; b++) {
            if (alternateQRow[b] <= -2) {
                reactiveRelocationGroupOfPilot[b] = -2 - alternateQRow[b];
            }
        }
        return new ReactiveRelocationTables(
                new ReactiveRelocationCandidates(toIntArray(reactiveRelocationGroupStart), toIntArray(reactiveRelocationCandidateRows),
                        toIntArray(reactiveRelocationDependentBaseGroup), toIntArray(reactiveRelocationBaseTargetRow), reactiveRelocationDependentGroupOfRow),
                reactiveRelocationGroupOfPilot);
    }

    /** Remote generator voltage-control pilot-Q relocation groups (plan-B4): one candidate table per remote group,
     *  [pilot V row, member V columns in merged order]; adds each group's pilot to {@code reactiveRelocationPilotBuses}. */
    private static void addRemoteVoltageControlGroups(final LfNetwork network,
                                                       final EquationSystem<AcVariableType, AcEquationType> es,
                                                       final int[] spanningTreeParentBranch,
                                                       final java.util.Set<Integer> reactiveRelocationPilotBuses,
                                                       final int[] dummyQCol, final int[] vCol,
                                                       final List<Integer> reactiveRelocationGroupStart, final List<Integer> reactiveRelocationCandidateRows,
                                                       final int[] alternateQRow, final List<Integer> reactiveRelocationDependentBaseGroup,
                                                       final List<Integer> reactiveRelocationBaseTargetRow,
                                                       final int[] reactiveRelocationDependentGroupOfRow) {
        for (LfBus bus : network.getBuses()) {
            var voltageControl = bus.getGeneratorVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .filter(c -> bus.isGeneratorVoltageControlled() && !c.isLocalControl())
                    .orElse(null);
            // A pilot that is ALSO one of its controllers has NO Q balance of its own to relocate —
            // its q feeds the group's DISTR_Q rows instead (ReactiveRelocation.distrQRow), and leaving
            // it in this table double-counted that q on top of the DISTR_Q terms.
            // ...but only WHILE IT CONTROLS. Once a reactive limit SWITCHES such a pilot TO PQ, OLF activates its
            // BUS_TARGET_Q, which needs a row, and its own row still carries BUS_TARGET_V because a sibling
            // controller keeps the voltage — so the group DOES need the table, with the pilot's raw-q
            // emission GATED on it being switched to PQ (pilotSelfGateBatch in native/cuda/outerloop/pilot_voltage.cu,
            // recomputed before every fill). Without it such a group keeps a relocation frozen at extraction,
            // which left rte6515's busbar base rank-deficient.
            boolean pilotSelf = voltageControl != null && voltageControl.getMergedControllerElements().contains(bus);
            // the ONE enrolment predicate (ReactiveRelocation.hasCandidateTable): the same answer decides the
            // canonical encoding, independent of the PV/PQ states, of the group's q rows and DISTR_Q pack
            if (!ReactiveRelocation.hasCandidateTable(voltageControl, es)) {
                continue;
            }
            // Each member's row is the one that HOSTS its reactive balance, which the coupler-tree relocation can
            // move to the parent coupler's DUMMY_Q column (exactly as ReactiveRelocation.distrQRow's tree overload
            // already moves its DISTR_Q). Naming its own V row instead sent the raw q of a pilot switched to PQ onto a row the
            // relocation had given to the coupler's ZERO_V, and left the pilot's own row empty: measured on
            // rte6515's [5000,6000) window with the relocation on, the failing slot's residual sat on
            // BUS_V(VL-6003_1) at 6.5e+02 and on the empty pilot row BUS_V(VL-6003_0) at 1.09e+02, every iteration.
            List<Integer> memberRows = new ArrayList<>();
            for (LfBus m : voltageControl.getMergedControllerElements()) {
                memberRows.add(ReactiveRelocation.relocatedQRow(m.getNum(), spanningTreeParentBranch, dummyQCol, vCol));
            }
            int g = reactiveRelocationGroupStart.size();
            reactiveRelocationGroupStart.add(reactiveRelocationCandidateRows.size());
            // candidate 0 = the FALLBACK (control lost), the pilot's own reactive row — relocated like any other
            reactiveRelocationCandidateRows.add(ReactiveRelocation.relocatedQRow(bus.getNum(), spanningTreeParentBranch, dummyQCol, vCol));
            reactiveRelocationCandidateRows.addAll(memberRows);
            alternateQRow[bus.getNum()] = -2 - g;
            reactiveRelocationPilotBuses.add(bus.getNum());
            reactiveRelocationDependentBaseGroup.add(-1);
            reactiveRelocationBaseTargetRow.add(-1);
            // DEPENDENT PILOTS. OLF merges the controls of one zero-impedance network into this MAIN; a contingency
            // that cuts a dependent controlled bus from the main re-merges the survivors and elects a new main among
            // them (LfZeroImpedanceNetwork.updateVoltageControlMergeStatus), whose BUS_TARGET_V comes back and whose
            // reactive balance must ride a FREE controller's row - the same member rows as this table, with that
            // bus's own reactive terms in the pattern. Each such bus gets a group of its own: candidate 0 its own
            // reactive row (the fallback when the re-merged control is lost), then the base members in merged order,
            // so its carrier pick coincides with the base group's on the same modes. A dependent bus that is itself
            // one of the controllers has no balance of its own to host (pilot-self) and is left out.
            for (var dependentVoltageControl : voltageControl.getMergedDependentVoltageControls()) {
                LfBus dependentBus = dependentVoltageControl.getControlledBus();
                int dependentBusNum = dependentBus.getNum();
                if (vCol[dependentBusNum] < 0 || voltageControl.getMergedControllerElements().contains(dependentBus) || reactiveRelocationDependentGroupOfRow[vCol[dependentBusNum]] >= 0) {
                    continue;
                }
                int dependentGroup = reactiveRelocationGroupStart.size();
                reactiveRelocationGroupStart.add(reactiveRelocationCandidateRows.size());
                int dependentPilotQRow = ReactiveRelocation.relocatedQRow(dependentBusNum, spanningTreeParentBranch, dummyQCol, vCol);
                reactiveRelocationCandidateRows.add(dependentPilotQRow);
                reactiveRelocationCandidateRows.addAll(memberRows);
                reactiveRelocationDependentBaseGroup.add(g);
                reactiveRelocationBaseTargetRow.add(dependentPilotQRow);
                reactiveRelocationDependentGroupOfRow[vCol[dependentBusNum]] = dependentGroup;
            }
            if (pilotSelf && GpuTrace.RL_TRACE.isOn()) {
                System.err.println("PILOTSELF_ENROL group=" + g + " pilot=" + bus.getId()
                        + " members=" + voltageControl.getMergedControllerElements().size()
                        + " fallbackRow=" + vCol[bus.getNum()]);
            }
        }
    }

    /**
     * The carrier of candidate group {@code g} under the row modes {@code modes}: the first member candidate (after
     * candidate 0, the fallback) whose row holds neither a fixed variable ({@code FIXED_VARIABLE}, or
     * {@code CROSS_FIXED_VARIABLE}: a re-merged control's identity on its coupler's row) nor the own balance of a bus
     * switched to PQ ({@code BUS_TARGET_Q}), or -1 when none is. The Java twin of the native {@code chooseCarrierRow}
     * (fixedmatrix/reactive_relocation.cuh): the host must pick the row the device reads.
     */
    public static int chooseCarrierRow(ReactiveRelocationCandidates candidates, int g, int[] modes) {
        int[] reactiveRelocationGroupStart = candidates.groupStart();
        int[] reactiveRelocationCandidateRows = candidates.candidateRows();
        for (int c = reactiveRelocationGroupStart[g] + 1; c < reactiveRelocationGroupStart[g + 1]; c++) {
            int r = reactiveRelocationCandidateRows[c];
            if (modes[r] != GpuRowMode.FIXED_VARIABLE && modes[r] != GpuRowMode.BUS_TARGET_Q
                    && modes[r] != GpuRowMode.CROSS_FIXED_VARIABLE) {
                return r;
            }
        }
        return -1;
    }

    /** The kernels map candidate c of qr group g to dq member reactiveRelocationDistrQMemberStart[g] + c - 1: true when every packed
     *  member with a row sits at that candidate (merged controller order on both sides). {@code mRow}, {@code mHasRow}
     *  and {@code memberRelocationGroup} are the packed DISTR_Q members' row, has-row flag and relocation group. */
    public static boolean reactiveRelocationMemberOrderConsistent(ReactiveRelocationCandidates candidates, int[] mRow, byte[] mHasRow,
                                                                  int[] memberRelocationGroup) {
        int[] start = reactiveRelocationDistrQMemberStart(candidates, memberRelocationGroup);
        int[] reactiveRelocationGroupStart = candidates.groupStart();
        int[] reactiveRelocationCandidateRows = candidates.candidateRows();
        if (start == null || reactiveRelocationGroupStart == null || mRow == null || memberRelocationGroup == null) {
            return false;
        }
        for (int m = 0; m < memberRelocationGroup.length; m++) {
            int g = memberRelocationGroup[m];
            if (g < 0 || mHasRow[m] == 0 || mRow[m] < 0) {
                continue;
            }
            int c = m - start[g] + 1;
            if (c < 1 || reactiveRelocationGroupStart[g] + c >= reactiveRelocationGroupStart[g + 1] || reactiveRelocationCandidateRows[reactiveRelocationGroupStart[g] + c] != mRow[m]) {
                return false;
            }
        }
        return true;
    }

    /** qr group -> its first dq member, -1 for a group without packed members ({@code memberRelocationGroup}: the packed
     *  DISTR_Q members' relocation group). */
    public static int[] reactiveRelocationDistrQMemberStart(ReactiveRelocationCandidates candidates, int[] memberRelocationGroup) {
        int[] reactiveRelocationGroupStart = candidates.groupStart();
        if (memberRelocationGroup == null || reactiveRelocationGroupStart == null || reactiveRelocationGroupStart.length < 2) {
            return null;
        }
        int[] start = new int[reactiveRelocationGroupStart.length - 1];
        java.util.Arrays.fill(start, -1);
        for (int m = 0; m < memberRelocationGroup.length; m++) {
            int g = memberRelocationGroup[m];
            if (g >= 0 && g < start.length && start[g] < 0) {
                start[g] = m;
            }
        }
        return start;
    }

    private static int[] toIntArray(List<Integer> list) {
        int[] out = new int[list.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = list.get(i);
        }
        return out;
    }
}
