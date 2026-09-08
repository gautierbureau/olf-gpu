/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivation;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
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
 * {@link GpuBatchSimpleTvc} —
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
final class GpuBatchSimpleShuntVc {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuBatchSimpleShuntVc.class);

    private GpuBatchSimpleShuntVc() {
    }

    /** One controller shunt of a group: its SHUNT_B row and per-section tables (dispatch targets). */
    record Member(int bRow, List<GpuBatchIncrementalShuntVc.Sub> subs, int shOvElem, int elemIndexAbs) {
    }

    /** One controlled bus's shunt VC group: the toggle-diff VC-ON activation + its member shunts. */
    record Group(int[] deltaRows, int[] deltaModesOn, double[] deltaTargetsOn, List<Member> members) {
    }

    /** Build the groups by TOGGLING each controlled bus's merged controller shunts on the shared
     *  equation system and diffing the activation — the ON recipe is exactly what the engine itself
     *  would produce (same pattern as {@code GpuBatchTransformerControl.buildTransformerControllers}).
     *  The network is restored before returning. */
    static List<Group> buildGroups(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                   TargetVector<AcVariableType, AcEquationType> targetVector,
                                   GpuAcActivation baseActivation, double[] baseTargets,
                                   GpuBatchElementMapper elementMapper, int shPackOffset) {
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
                LOGGER.info("Shunt voltage control of {} not representable on the GPU batch — skipped "
                        + "(results follow the base sections)", controlled.getId());
                continue;
            }
            List<Member> members = new ArrayList<>();
            boolean ok = true;
            for (LfShunt sh : memberShunts) {
                List<GpuBatchIncrementalShuntVc.Sub> subs = new ArrayList<>();
                boolean gVaries = false;
                for (LfShunt.Controller c : sh.getControllers()) {
                    double[] sb = c.getSectionsB().stream().mapToDouble(Double::doubleValue).toArray();
                    double[] sg = new double[sb.length];
                    for (int i = 0; i < sb.length; i++) {
                        sg[i] = GpuBatchIncrementalShuntVc.shuntSectionG(c, i);
                        gVaries |= sg[i] != sg[0];
                    }
                    subs.add(new GpuBatchIncrementalShuntVc.Sub(sb, sg, c.getPosition(), c.getMinPosition()));
                }
                int elemIndex = elementMapper.getElementIndex(sh);
                if (elemIndex < 0 && gVaries) {              // static G follows the section: needs a packed element
                    LOGGER.info("Shunt VC controller {} has section-G variation but no packed element — group skipped",
                            sh.getId());
                    ok = false;
                    break;
                }
                members.add(new Member(shuntBRows.get(sh.getNum()), subs, elemIndex - shPackOffset, elemIndex));
            }
            if (!ok) {
                continue;
            }
            boolean[] wasEnabled = new boolean[memberShunts.size()];
            for (int m = 0; m < memberShunts.size(); m++) {
                wasEnabled[m] = memberShunts.get(m).isVoltageControlEnabled();
                memberShunts.get(m).setVoltageControlEnabled(true);      // the WHOLE group ON
            }
            GpuAcActivation onAct = GpuAcActivation.fromEquationSystem(network, es);
            double[] onTargets = onAct.applyTargets(targetVector.getArray());
            for (int m = 0; m < memberShunts.size(); m++) {
                memberShunts.get(m).setVoltageControlEnabled(wasEnabled[m]);
            }
            List<Integer> dRows = new ArrayList<>();
            for (int r = 0; r < baseTargets.length; r++) {
                if (onAct.rowMode()[r] != baseActivation.rowMode()[r] || onTargets[r] != baseTargets[r]) {
                    dRows.add(r);
                }
            }
            int[] deltaRows = dRows.stream().mapToInt(Integer::intValue).toArray();
            int[] deltaModesOn = new int[deltaRows.length];
            double[] deltaTargetsOn = new double[deltaRows.length];
            for (int k = 0; k < deltaRows.length; k++) {
                deltaModesOn[k] = onAct.rowMode()[deltaRows[k]];
                deltaTargetsOn[k] = onTargets[deltaRows[k]];
            }
            out.add(new Group(deltaRows, deltaModesOn, deltaTargetsOn, members));
        }
        return out;
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
            GpuBatchIncrementalShuntVc.Sub sub = m.subs().get(k);
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

    /** Post-solve rounding: read each scenario's continuous b per member, dispatch to the closest
     *  sections, restore the VC-OFF (base) activation and re-solve warm with the SHUNT_TARGET_B
     *  identities pinned at the dispatched totals, the b states seeded there, and the
     *  section-driven (dG, dB) admittance deltas applied. */
    @SuppressWarnings("java:S107")
    static double[][] roundAndResolve(long handle, int chunkSize, int realCount, int n,
            double[][] chunkResults, List<Group> groups,
            int[][] rowModePerScenario, int[] baseRowMode, boolean[][] disabledMask,
            List<Integer> baseOvIndexList, List<Double> baseOvValueList,
            int[] dsOvBus, double[] dsOvFactorDelta, double[] dsOvBaseDelta, double[] dsOvMaxDelta, double[] dsOvMinDelta,
            double[] dsAnchorShift, double[] dsSlackShift, int[] dsPhiRow, double[] dsFactor,
            int[] r1OvSlotElem, double[] r1OvVal, int[] shOvSlotElem, double[] shOvGb,
            int nziBase, int[] ziOnSlotElem, int[] ziOffSlotElem,
            double[] baseState, double[] baseTargets, NewtonRaphsonParameters parameters,
            double tol, double reactiveLimitTol, double slackDistTol, double convEpsPerEq, int[] varType,
            BatchConfig config) {
        List<Integer> ovIdx = new ArrayList<>(baseOvIndexList);
        List<Double> ovVal = new ArrayList<>(baseOvValueList);
        // DS target carry (the same three coupled pieces as every sibling orchestrator): the main
        // solve's slack distribution must ride into this re-solve, or the kernel re-derives it
        // from the re-uploaded baseTargets and takes Newton steps the CPU never takes.
        int nds = dsPhiRow != null ? dsPhiRow.length : 0;
        double[] dsAnchorUse = dsAnchorShift;
        double[] dsSlackUse = dsSlackShift;
        if (nds > 0) {
            double[] anchor = new double[chunkSize * nds];
            double[] slackC = new double[chunkSize];
            double[] seed = new double[chunkSize * nds];
            GpuBatchTransformerControl.captureDsCarry(handle, chunkSize, realCount, n, dsPhiRow, dsFactor,
                    baseTargets, dsOvBus, dsOvFactorDelta, dsOvBaseDelta, chunkResults, anchor, slackC, seed);
            dsAnchorUse = anchor;
            dsSlackUse = slackC;
            GpuBatchTransformerControl.seedDsTargets(nds, dsPhiRow, seed, chunkResults, realCount, n, ovIdx, ovVal);
        }
        List<Integer> shSe = new ArrayList<>();
        List<Double> shGb = new ArrayList<>();
        if (shOvSlotElem != null) {                          // shOv deltas are ADDITIVE — base first
            for (int i : shOvSlotElem) {
                shSe.add(i);
            }
            for (double d : shOvGb) {
                shGb.add(d);
            }
        }
        double[] x0Full = new double[chunkSize * n];
        int[][] rowModeOff = new int[chunkSize][];
        boolean[] affected = new boolean[chunkSize];
        for (int s = 0; s < chunkSize; s++) {
            double[] src = s < realCount && chunkResults[s] != null ? chunkResults[s] : baseState;
            System.arraycopy(src, 0, x0Full, s * n, n);
            rowModeOff[s] = baseRowMode;
        }
        for (int s = 0; s < realCount; s++) {
            if (chunkResults[s] == null || (int) chunkResults[s][n + 2] != GpuBatchStatus.STATUS_CONVERGED) {
                continue;
            }
            int[] row = null;
            for (Group g : groups) {
                if (!groupLive(g, disabledMask[s])) {
                    continue;
                }
                if (row == null) {
                    row = rowModePerScenario[s].clone();
                }
                for (int k = 0; k < g.deltaRows().length; k++) {
                    row[g.deltaRows()[k]] = baseRowMode[g.deltaRows()[k]];   // undo the VC-on flip
                }
                for (Member m : g.members()) {               // OLF dispatchB, per controller shunt
                    int[] pos = new int[m.subs().size()];
                    double bSum = dispatchB(m, chunkResults[s][m.bRow()], pos);
                    ovIdx.add(s * n + m.bRow());
                    ovVal.add(bSum);                         // the identity row pins b at the section total
                    x0Full[s * n + m.bRow()] = bSum;         // the CPU re-reads b from the shunt on re-solve
                    double dg = 0;
                    double db = 0;
                    for (int k = 0; k < m.subs().size(); k++) {
                        GpuBatchIncrementalShuntVc.Sub sub = m.subs().get(k);
                        dg += sub.g(pos[k]) - sub.g(sub.basePos());
                        db += sub.b(pos[k]) - sub.b(sub.basePos());
                    }
                    if (m.elemIndexAbs() >= 0 && (dg != 0 || db != 0)) {
                        shSe.add(s);
                        shSe.add(m.shOvElem());
                        shGb.add(dg);                        // static G follows the section; static B is
                        shGb.add(db);                        // shadowed by the SHUNT_B variable (harmless)
                    }
                }
                affected[s] = true;
            }
            if (row != null) {
                rowModeOff[s] = row;
            }
        }
        boolean any = false;
        for (boolean a : affected) {
            any |= a;
        }
        if (!any) {
            return chunkResults;
        }
        double[][] r2 = GpuAcNewtonSolver.solveBatchContext(handle, baseState, baseTargets,
                rowModeOff, baseRowMode, disabledMask,
                ovIdx.stream().mapToInt(Integer::intValue).toArray(),
                ovVal.stream().mapToDouble(Double::doubleValue).toArray(),
                dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta, dsAnchorUse, dsSlackUse, null, null,
                parameters.getMaxIterations(), tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                config.lsMode(), config.maxDv(), config.maxDphi(),
                config.lsMode() == 2 ? varType : null, x0Full,
                shSe.isEmpty() ? null : shSe.stream().mapToInt(Integer::intValue).toArray(),
                shGb.isEmpty() ? null : shGb.stream().mapToDouble(Double::doubleValue).toArray(),
                null, null,
                r1OvSlotElem, r1OvVal,
                config.maxOuterIterations(), config.maxPqPvSwitch(), config.dsRejectResidualFlag(), 0,
                nziBase, ziOnSlotElem, ziOffSlotElem);
        for (int s = 0; s < realCount; s++) {
            if (affected[s]) {
                chunkResults[s] = r2[s];
            }
        }
        return chunkResults;
    }
}
