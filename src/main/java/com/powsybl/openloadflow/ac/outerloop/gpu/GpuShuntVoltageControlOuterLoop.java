/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivation;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcElementMapper;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchSkips;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuDiscreteControlMemberState;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.TargetVector;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfShunt;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * DEVICE port of OLF's {@code ShuntVoltageControlOuterLoop} (the WITH_GENERATOR mode — OLF's
 * DEFAULT {@code shuntVoltageControlMode}): the same fixed two-phase flow as
 * {@link GpuSimpleTransformerVoltageControlOuterLoop} —
 *
 * <ol>
 *   <li>{@code initialize()} enables CONTINUOUS voltage control on every controller shunt BEFORE
 *       the first solve: {@link #applyVcOn} pushes each group's VC-ON activation deltas (captured
 *       by {@link #buildGroups}'s toggle-diff — the controlled bus's row becomes the V identity,
 *       the bus Q balance relocates onto the SHUNT_B row via the structure's dynamic Q-relocation,
 *       and DISTR_SHUNT_B couples shared groups) into the per-scenario row modes and targets;</li>
 *   <li>the loop's first check disables VC and dispatches each shunt's continuous b to the CLOSEST
 *       SECTIONS ({@code LfShuntImpl.dispatchB}: per sub-controller, round {@code residue/remaining}
 *       from the current position over [minPosition, size), walking the residue) → one warm
 *       re-solve with the base (VC-OFF) activation, the SHUNT_TARGET_B identities pinned at the
 *       dispatched totals and the section-driven (dG, dB) admittance deltas applied
 *       ({@link #roundAndResolve}); further checks are STABLE by construction.</li>
 * </ol>
 *
 * <p>Scenarios that outage a PACKED member shunt skip the group (such contingencies are
 * CPU-routed at qualification anyway — a voltage-controlling shunt outage never qualifies).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuShuntVoltageControlOuterLoop {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuShuntVoltageControlOuterLoop.class);

    private GpuShuntVoltageControlOuterLoop() {
    }

    /** One controller shunt of a group: its SHUNT_B row and per-section tables (dispatch targets). */
    public record Member(int bRow, List<GpuIncrementalShuntVoltageControlOuterLoop.Sub> subs, int shuntAdmittanceOverrideElement, int elemIndexAbs, int shuntNum) {
    }

    /** One controlled bus's shunt VC group: the toggle-diff VC-ON activation + its member shunts. */
    public record Group(int[] deltaRows, int[] deltaModesOn, double[] deltaTargetsOn, List<Member> members,
                 int[] deltaModesOff, double[] deltaTargetsOff) {
    }

    /** Build the groups by TOGGLING each controlled bus's merged controller shunts on the shared
     *  equation system and diffing the activation — the ON recipe is exactly what the engine itself
     *  would produce (same pattern as {@code GpuTransformerVoltageControlOuterLoop.buildTransformerControllers}).
     *  The network is restored before returning. */
    public static List<Group> buildGroups(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                   TargetVector<AcVariableType, AcEquationType> targetVector,
                                   GpuAcActivation baseActivation, double[] baseTargets,
                                   GpuAcElementMapper elementMapper, int shuntPackOffset) {
        List<Group> out = new ArrayList<>();
        java.util.Map<Integer, Integer> shuntBRows = new java.util.HashMap<>();
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            if (v.getType() == AcVariableType.SHUNT_B) {
                shuntBRows.put(v.getElementNum(), v.getRow());
            }
        }
        for (LfBus controlled : network.getControlledBuses(com.powsybl.openloadflow.network.VoltageControl.Type.SHUNT)) {
            var vc = controlled.getShuntVoltageControl()
                    .filter(c -> c.getMergeStatus() == com.powsybl.openloadflow.network.VoltageControl.MergeStatus.MAIN)
                    .orElse(null);
            if (vc == null) {
                continue;
            }
            List<LfShunt> memberShunts = vc.getMergedControllerElements().stream()
                    .filter(sh -> !sh.isDisabled())
                    .filter(sh -> !sh.getControllers().isEmpty())
                    .toList();
            boolean representable = !memberShunts.isEmpty()
                    && memberShunts.stream().allMatch(sh -> shuntBRows.get(sh.getNum()) != null);
            if (!representable) {
                GpuBatchSkips.record(GpuBatchSkips.Slot.SHUNT, LOGGER,
                        "Shunt voltage control of {} not representable on the GPU batch — skipped "
                        + "(results follow the base sections)", controlled.getId());
                continue;
            }
            List<Member> members = new ArrayList<>();
            boolean ok = true;
            for (LfShunt sh : memberShunts) {
                List<GpuIncrementalShuntVoltageControlOuterLoop.Sub> subs = new ArrayList<>();
                boolean gVaries = false;
                for (LfShunt.Controller c : sh.getControllers()) {
                    double[] sb = c.getSectionsB().stream().mapToDouble(Double::doubleValue).toArray();
                    double[] sg = new double[sb.length];
                    for (int i = 0; i < sb.length; i++) {
                        sg[i] = GpuIncrementalShuntVoltageControlOuterLoop.shuntSectionG(c, i);
                        gVaries |= sg[i] != sg[0];
                    }
                    subs.add(new GpuIncrementalShuntVoltageControlOuterLoop.Sub(sb, sg, c.getPosition(), c.getMinPosition()));
                }
                int elemIndex = elementMapper.getElementIndex(sh);
                if (elemIndex < 0 && gVaries) {              // static G follows the section: needs a packed element
                    LOGGER.info("Shunt VC controller {} has section-G variation but no packed element — group skipped",
                            sh.getId());
                    ok = false;
                    break;
                }
                members.add(new Member(shuntBRows.get(sh.getNum()), subs, elemIndex - shuntPackOffset, elemIndex, sh.getNum()));
            }
            if (!ok) {
                continue;
            }
            boolean[] wasEnabled = new boolean[memberShunts.size()];
            for (int m = 0; m < memberShunts.size(); m++) {
                wasEnabled[m] = memberShunts.get(m).isVoltageControlEnabled();
                memberShunts.get(m).setVoltageControlEnabled(false);     // the WHOLE group OFF
            }
            GpuAcActivation offAct = GpuAcActivation.fromEquationSystem(network, es);
            double[] offTargets = offAct.applyTargets(targetVector.getArray());
            for (int m = 0; m < memberShunts.size(); m++) {
                memberShunts.get(m).setVoltageControlEnabled(true);      // the WHOLE group ON
            }
            GpuAcActivation onAct = GpuAcActivation.fromEquationSystem(network, es);
            double[] onTargets = onAct.applyTargets(targetVector.getArray());
            for (int m = 0; m < memberShunts.size(); m++) {
                memberShunts.get(m).setVoltageControlEnabled(wasEnabled[m]);
            }
            // ON/OFF diffed against each other (not the caller's base): a base extracted with the
            // control already ON (the base load flow, whose network OLF's initialize switched on)
            // yields the same rows as a control-off base
            List<Integer> dRows = new ArrayList<>();
            for (int r = 0; r < baseTargets.length; r++) {
                if (onAct.rowMode()[r] != offAct.rowMode()[r] || onTargets[r] != offTargets[r]) {
                    dRows.add(r);
                }
            }
            int[] deltaRows = dRows.stream().mapToInt(Integer::intValue).toArray();
            int[] deltaModesOn = new int[deltaRows.length];
            double[] deltaTargetsOn = new double[deltaRows.length];
            int[] deltaModesOff = new int[deltaRows.length];
            double[] deltaTargetsOff = new double[deltaRows.length];
            for (int k = 0; k < deltaRows.length; k++) {
                deltaModesOn[k] = onAct.rowMode()[deltaRows[k]];
                deltaTargetsOn[k] = onTargets[deltaRows[k]];
                deltaModesOff[k] = offAct.rowMode()[deltaRows[k]];
                deltaTargetsOff[k] = offTargets[deltaRows[k]];
            }
            out.add(new Group(deltaRows, deltaModesOn, deltaTargetsOn, members, deltaModesOff, deltaTargetsOff));
        }
        return out;
    }

    /** Install the groups for the in-kernel WITH_GENERATOR ShuntVoltageControl phase (OLF order 10):
     *  {@code entryOn} = the activation the solve starts from already carries the control ON (the base
     *  load flow — OLF's initialize ran on the network); a chunk applies the ON deltas at solve entry
     *  itself. Returns true when the kernel loop owns them. */
    public static boolean uploadToDevice(long handle, List<Group> groups, boolean entryOn) {
        if (groups == null || groups.isEmpty()) {
            ShuntVoltageControlNative.setBatchDiscreteShunt(handle, 0, null, null, null, null, null, null, null,
                    null, null, null, null, null, null, null);
            return false;
        }
        int nG = groups.size();
        int[] grpPtr = new int[nG + 1];
        int[] deltaPtr = new int[nG + 1];
        List<Integer> dRow = new ArrayList<>();
        List<Integer> dModeOn = new ArrayList<>();
        List<Integer> dModeOff = new ArrayList<>();
        List<Double> dTgtOn = new ArrayList<>();
        List<Double> dTgtOff = new ArrayList<>();
        List<Integer> memBRow = new ArrayList<>();
        List<Integer> memElem = new ArrayList<>();
        List<Integer> memSubPtr = new ArrayList<>();
        List<Integer> secPtr = new ArrayList<>();
        List<Double> secB = new ArrayList<>();
        List<Integer> subMin = new ArrayList<>();
        List<Integer> subBase = new ArrayList<>();
        for (int g = 0; g < nG; g++) {
            Group grp = groups.get(g);
            grpPtr[g] = memBRow.size();
            deltaPtr[g] = dRow.size();
            for (int k = 0; k < grp.deltaRows().length; k++) {
                dRow.add(grp.deltaRows()[k]);
                dModeOn.add(grp.deltaModesOn()[k]);
                dModeOff.add(grp.deltaModesOff()[k]);
                dTgtOn.add(grp.deltaTargetsOn()[k]);
                dTgtOff.add(grp.deltaTargetsOff()[k]);
            }
            for (Member m : grp.members()) {
                memBRow.add(m.bRow());
                memElem.add(m.elemIndexAbs());
                memSubPtr.add(subBase.size());
                for (GpuIncrementalShuntVoltageControlOuterLoop.Sub sub : m.subs()) {
                    secPtr.add(secB.size());
                    for (double b : sub.sectionsB()) {
                        secB.add(b);
                    }
                    subMin.add(sub.minPos());
                    subBase.add(sub.basePos());
                }
            }
        }
        grpPtr[nG] = memBRow.size();
        deltaPtr[nG] = dRow.size();
        memSubPtr.add(subBase.size());
        secPtr.add(secB.size());
        boolean owned = ShuntVoltageControlNative.setBatchDiscreteShunt(handle, entryOn ? 1 : 0, grpPtr, deltaPtr,
                dRow.stream().mapToInt(Integer::intValue).toArray(),
                dModeOn.stream().mapToInt(Integer::intValue).toArray(), dModeOff.stream().mapToInt(Integer::intValue).toArray(),
                dTgtOn.stream().mapToDouble(Double::doubleValue).toArray(), dTgtOff.stream().mapToDouble(Double::doubleValue).toArray(),
                memBRow.stream().mapToInt(Integer::intValue).toArray(), memElem.stream().mapToInt(Integer::intValue).toArray(),
                memSubPtr.stream().mapToInt(Integer::intValue).toArray(), secPtr.stream().mapToInt(Integer::intValue).toArray(),
                secB.stream().mapToDouble(Double::doubleValue).toArray(),
                subMin.stream().mapToInt(Integer::intValue).toArray(), subBase.stream().mapToInt(Integer::intValue).toArray()) != 0;
        LOGGER.info("Uploaded {} shunt voltage-control groups ({} shunts, {} sub-controllers) to device: {}",
                nG, memBRow.size(), subBase.size(), owned ? "in-kernel loop" : "host orchestrator");
        if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
            System.err.printf("SVD_UPLOAD entryOn=%b nGroups=%d nMembers=%d nSubs=%d kernel=%b%n", entryOn, nG, memBRow.size(), subBase.size(), owned);
        }
        return owned;
    }

    /** Give the in-kernel phase's decisions back to the network (batch-of-one base load flow), exactly as
     *  OLF's loop leaves it: every dispatched controller shunt has its control disabled and
     *  {@code dispatchB()} run from the continuous b the device rounded ({@code state} =
     *  readBatchDiscreteShuntState, {@code positions} = readBatchDiscreteShuntPositions). */
    public static void applyToNetwork(LfNetwork network, List<Group> groups, double[] state, int[] positions) {
        if (groups == null || groups.isEmpty() || state == null) {
            return;
        }
        int m = 0;
        int u = 0;
        for (Group g : groups) {
            for (Member mem : g.members()) {
                int k = m++ * 2;
                int u0 = u;
                u += mem.subs().size();
                if (k + 1 >= state.length || (int) state[k] != GpuDiscreteControlMemberState.ROUNDED) {
                    continue;
                }
                LfShunt shunt = network.getShunt(mem.shuntNum());
                if (shunt == null) {
                    continue;
                }
                if (shunt.isVoltageControlEnabled()) {
                    shunt.setVoltageControlEnabled(false);
                }
                List<LfShunt.Controller> subs = shunt.getControllers();
                boolean same = positions != null;
                for (int i = 0; same && i < subs.size() && u0 + i < positions.length; i++) {
                    same = subs.get(i).getPosition() == positions[u0 + i];
                }
                if (!same) {
                    shunt.setB(state[k + 1]);
                    shunt.dispatchB();
                    for (int i = 0; positions != null && i < subs.size() && u0 + i < positions.length; i++) {
                        if (subs.get(i).getPosition() != positions[u0 + i]) {   // tie: take the device's section
                            subs.get(i).setPosition(positions[u0 + i]);
                        }
                    }
                    shunt.setB(subs.stream().mapToDouble(LfShunt.Controller::getB).sum());
                }
            }
        }
    }

    private static boolean groupLive(Group g, boolean[] slotDisabled) {
        for (Member m : g.members()) {
            if (m.elemIndexAbs() >= 0 && slotDisabled[m.elemIndexAbs()]) {
                return false;                                // member outaged: such contingencies are
            }                                                // CPU-routed; skip defensively
        }
        return true;
    }

    /** Pre-solve mutation: apply every group's VC-ON activation deltas to each real scenario's row
     *  modes and target overrides (OLF's initialize() enables VC before the first solve). */
    static void applyVcOn(int realCount, int n, List<Group> groups,
                          int[][] rowModePerScenario, int[] baseRowMode, boolean[][] disabledMask,
                          List<Integer> ovIndexList, List<Double> ovValueList) {
        for (int slot = 0; slot < realCount; slot++) {
            for (Group g : groups) {
                if (!groupLive(g, disabledMask[slot])) {
                    continue;
                }
                int[] row = rowModePerScenario[slot] == baseRowMode
                        ? baseRowMode.clone() : rowModePerScenario[slot];
                for (int k = 0; k < g.deltaRows().length; k++) {
                    row[g.deltaRows()[k]] = g.deltaModesOn()[k];
                    ovIndexList.add(slot * n + g.deltaRows()[k]);
                    ovValueList.add(g.deltaTargetsOn()[k]);
                }
                rowModePerScenario[slot] = row;
            }
        }
    }

    /** OLF LfShuntImpl.dispatchB + roundBToClosestSection, verbatim: per sub-controller, dispatch
     *  {@code residue / remaining} to the closest section (search starts at the CURRENT — base —
     *  position, candidates in [minPosition, size), strict-improvement ties keep the earlier),
     *  walking the residue. Fills the landed positions; returns the dispatched total b. */
    private static double dispatchB(Member m, double continuousB, int[] posOut) {
        double residue = continuousB;
        int remaining = m.subs().size();
        double bSum = 0;
        for (int k = 0; k < m.subs().size(); k++) {
            double toDispatch = residue / remaining--;
            GpuIncrementalShuntVoltageControlOuterLoop.Sub sub = m.subs().get(k);
            int best = sub.basePos();
            double smallest = Math.abs(toDispatch - sub.b(sub.basePos()));
            for (int p = sub.minPos(); p < sub.sectionsB().length; p++) {
                double dist = Math.abs(toDispatch - sub.b(p));
                if (dist < smallest) {
                    best = p;
                    smallest = dist;
                }
            }
            posOut[k] = best;
            residue -= sub.b(best);
            bSum += sub.b(best);
        }
        return bSum;
    }

}
