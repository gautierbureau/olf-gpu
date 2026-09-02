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
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfShunt;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * DEVICE port of OLF's {@code IncrementalShuntVoltageControlOuterLoop}, sharing the incremental-family
 * machinery with {@link GpuBatchIncrementalTvc}: batched in-kernel dV/dB sensitivities (unit RHS at
 * each controller shunt's pinned {@code SHUNT_TARGET_B} identity row), then OLF's section decisions
 * replicated verbatim per scenario — the |B|-magnitude-descending shunt ordering, the per-round
 * {@code maxSectionShift} cap (OLF's {@code > maxSectionShift} guard, checked BEFORE the increment),
 * one-section-at-a-time moves via the closest-B search, and the {@code AllowedDirection} ratchet per
 * SUB-controller (an aggregated LfShunt can carry several). No sensitivity filter exists in this loop
 * (unlike the transformer one). Section moves land as per-scenario {@code SHUNT_TARGET_B} target
 * overrides + a B state seed (the pinned identity value) and additive (dG, dB) shunt-admittance
 * overrides (the static G follows the section; the static B is shadowed by the SHUNT_B variable);
 * re-solve warm and iterate until every scenario is in deadband or nothing moves.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuBatchIncrementalShuntVc {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuBatchIncrementalShuntVc.class);

    private static final int MAX_DIRECTION_CHANGE = 3;       // OLF IncrementalShuntVoltageControlOuterLoop
    private static final double MIN_TARGET_DEADBAND_KV = 0.1;

    private GpuBatchIncrementalShuntVc() {
    }

    /** One SUB-controller (OLF LfShunt.Controller): per-section B/G tables + the ratchet identity. */
    record Sub(double[] sectionsB, double[] sectionsG, int basePos, int minPos) {
        double b(int pos) {
            return sectionsB[pos];
        }

        double g(int pos) {
            return sectionsG[pos];
        }
    }

    /** One controller shunt: the SHUNT_B variable row (pinned identity — unit-RHS row, target index and
     *  state index coincide), the controlled bus V row, and its sub-controllers. */
    record Controller(int bRow, int vRow, int ctrlBusNum, double targetV, double halfDeadband,
                      List<Sub> subs, double bMagnitude, int shOvElem, int elemIndexAbs) {
    }

    private static final class Ratchet {
        int dirChangeCount;
        int currentDirection;
        int allowedDirection;

        void update(int direction) {                         // OLF ControllerContext.updateAllowedDirection
            if (dirChangeCount < MAX_DIRECTION_CHANGE) {
                if (currentDirection != 0 && currentDirection != direction) {
                    dirChangeCount++;
                }
                currentDirection = direction;
            } else {
                allowedDirection = direction;
            }
        }
    }

    static List<Controller> buildControllers(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                             GpuBatchElementMapper elementMapper, int shPackOffset) {
        List<Controller> out = new ArrayList<>();
        Map<Integer, Integer> busVRows = new LinkedHashMap<>();
        Map<Integer, Integer> shuntBRows = new LinkedHashMap<>();
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            if (v.getType() == AcVariableType.BUS_V) {
                busVRows.put(v.getElementNum(), v.getRow());
            } else if (v.getType() == AcVariableType.SHUNT_B) {
                shuntBRows.put(v.getElementNum(), v.getRow());
            }
        }
        for (LfShunt shunt : network.<LfShunt>getControllerElements(com.powsybl.openloadflow.network.VoltageControl.Type.SHUNT)) {
            var vcOpt = shunt.getVoltageControl();
            if (vcOpt.isEmpty() || shunt.getControllers().isEmpty()) {
                continue;
            }
            var vc = vcOpt.orElseThrow();
            Integer bRow = shuntBRows.get(shunt.getNum());
            Integer vRow = busVRows.get(vc.getControlledBus().getNum());
            int elemIndex = elementMapper.getElementIndex(shunt);
            if (bRow == null || vRow == null || elemIndex < 0) {
                LOGGER.info("Incremental shunt controller {} not representable on the GPU batch — skipped", shunt.getId());
                continue;
            }
            List<Sub> subs = new ArrayList<>();
            double magnitude = 0;
            for (LfShunt.Controller c : shunt.getControllers()) {
                double[] sb = c.getSectionsB().stream().mapToDouble(Double::doubleValue).toArray();
                double[] sg = new double[sb.length];
                for (int i = 0; i < sb.length; i++) {
                    sg[i] = shuntSectionG(c, i);
                }
                subs.add(new Sub(sb, sg, c.getPosition(), c.getMinPosition()));
                magnitude += c.getBMagnitude();
            }
            double halfDeadband = vc.getTargetDeadband()
                    .orElse(MIN_TARGET_DEADBAND_KV / vc.getControlledBus().getNominalV()) / 2;
            out.add(new Controller(bRow, vRow, vc.getControlledBus().getNum(), vc.getTargetValue(), halfDeadband,
                    subs, magnitude, elemIndex - shPackOffset, elemIndex));
        }
        return out;
    }

    @SuppressWarnings("unchecked")
    private static double shuntSectionG(LfShunt.Controller c, int pos) {
        // no public accessor for the per-section G table (only getG() at the current position) —
        // read-only reflection, same pattern as GpuBatchTransformerControl.tapModels
        try {
            java.lang.reflect.Field fld = LfShunt.Controller.class.getDeclaredField("sectionsG");
            fld.setAccessible(true);
            List<Double> sg = (List<Double>) fld.get(c);
            return pos < sg.size() ? sg.get(pos) : 0;
        } catch (ReflectiveOperationException | RuntimeException e) {
            return 0;                                        // no G table — sections are pure-B
        }
    }

    /** OLF LfShunt.Controller.updateSectionB: closest section to b(cur)+deltaB, one section at a time,
     *  within the ratchet's allowed range. Returns 0 (no move) or the direction. */
    private static int moveSection(Sub sub, int[] pos, double deltaB, Ratchet ratchet) {
        int cur = pos[0];
        double newB = sub.b(cur) + deltaB;
        int lo = ratchet.allowedDirection > 0 ? cur : sub.minPos();
        int hi = ratchet.allowedDirection < 0 ? cur : sub.sectionsB().length - 1;
        int best = cur;
        double smallest = Math.abs(deltaB);
        for (int p = lo; p <= hi; p++) {
            if (Math.abs(p - cur) > 1) {                     // maxSectionShift = 1 per call (OLF adjustB)
                continue;
            }
            double dist = Math.abs(newB - sub.b(p));
            if (dist < smallest) {
                best = p;
                smallest = dist;
            }
        }
        if (best == cur) {
            return 0;
        }
        pos[0] = best;
        int direction = best > cur ? 1 : -1;
        ratchet.update(direction);
        return direction;
    }

    @SuppressWarnings({"java:S107", "java:S3776"})
    static double[][] orchestrate(long handle, int chunkSize, int realCount, int n,
            double[][] chunkResults, List<Controller> ctrls,
            int[][] rowModePerScenario, int[] baseRowMode, boolean[][] disabledMask,
            int[] targetOverrideIndex, double[] targetOverrideValue,
            int[] dsOvBus, double[] dsOvFactorDelta, double[] dsOvBaseDelta, double[] dsOvMaxDelta, double[] dsOvMinDelta,
            double[] dsAnchorShift, double[] dsSlackShift,
            int[] baseR1OvSlotElem, double[] baseR1OvVal, int[] baseShOvSlotElem, double[] baseShOvGb,
            int nziBase, int[] ziOnSlotElem, int[] ziOffSlotElem,
            double[] baseState, double[] baseTargets, NewtonRaphsonParameters parameters,
            double tol, double reactiveLimitTol, double slackDistTol, double convEpsPerEq, int[] varType,
            BatchConfig config) {
        int nc = ctrls.size();
        int[][][] pos = new int[chunkSize][nc][];            // per-scenario per-sub section position
        Ratchet[][][] ratchets = new Ratchet[chunkSize][nc][];
        for (int s = 0; s < chunkSize; s++) {
            for (int c = 0; c < nc; c++) {
                List<Sub> subs = ctrls.get(c).subs();
                pos[s][c] = new int[subs.size()];
                ratchets[s][c] = new Ratchet[subs.size()];
                for (int k = 0; k < subs.size(); k++) {
                    pos[s][c][k] = subs.get(k).basePos();
                    ratchets[s][c][k] = new Ratchet();
                }
            }
        }
        // controllers grouped by controlled bus, |B| magnitude DESCENDING inside each group (OLF ordering)
        Map<Integer, List<Integer>> byBus = new LinkedHashMap<>();
        for (int c = 0; c < nc; c++) {
            byBus.computeIfAbsent(ctrls.get(c).ctrlBusNum(), k -> new ArrayList<>()).add(c);
        }
        for (List<Integer> group : byBus.values()) {
            group.sort(Comparator.comparingDouble((Integer ci) -> ctrls.get(ci).bMagnitude()).reversed());
        }
        int maxSectionShift = config.incShuntMaxSectionShift();
        int maxRounds = config.maxOuterIterations() > 0 ? config.maxOuterIterations() : 20;

        for (int round = 0; round < maxRounds; round++) {
            List<int[]> outOfDb = new ArrayList<>();
            List<Integer> busKeys = new ArrayList<>(byBus.keySet());
            boolean[] ctrlNeeded = new boolean[nc];
            for (int slot = 0; slot < realCount; slot++) {
                double[] r = chunkResults[slot];
                if (r == null || (int) r[n + 2] != GpuBatchStatus.STATUS_CONVERGED) {
                    continue;
                }
                final int slotL = slot;
                for (int g = 0; g < busKeys.size(); g++) {
                    List<Integer> group = byBus.get(busKeys.get(g));
                    Controller c0 = ctrls.get(group.get(0));
                    boolean anyLive = group.stream().anyMatch(ci -> !disabledMask[slotL][ctrls.get(ci).elemIndexAbs()]);
                    if (anyLive && Math.abs(c0.targetV() - r[c0.vRow()]) > c0.halfDeadband()) {
                        outOfDb.add(new int[] {slot, g});
                        for (int ci : group) {
                            ctrlNeeded[ci] = true;
                        }
                    }
                }
            }
            if (outOfDb.isEmpty()) {
                break;
            }
            int[] unit = new int[nc];
            int[] unitOf = new int[nc];
            int ku = 0;
            for (int c = 0; c < nc; c++) {
                if (ctrlNeeded[c]) {
                    unitOf[c] = ku;
                    unit[ku++] = ctrls.get(c).bRow();
                } else {
                    unitOf[c] = -1;
                }
            }
            int[] obs = new int[busKeys.size()];
            for (int g = 0; g < busKeys.size(); g++) {
                obs[g] = ctrls.get(byBus.get(busKeys.get(g)).get(0)).vRow();
            }
            double[] statesFlat = new double[chunkSize * n];
            for (int s = 0; s < chunkSize; s++) {
                double[] src = s < realCount && chunkResults[s] != null ? chunkResults[s] : baseState;
                System.arraycopy(src, 0, statesFlat, s * n, n);
            }
            double[] sensi = GpuAcNewtonSolver.solveBatchSensitivities(handle, statesFlat,
                    java.util.Arrays.copyOf(unit, ku), obs);
            final int kuF = ku;
            int m = obs.length;

            boolean anyMoved = false;
            boolean[] affected = new boolean[chunkSize];
            for (int[] sg : outOfDb) {
                int slot = sg[0];
                int g = sg[1];
                List<Integer> group = byBus.get(busKeys.get(g));
                Controller c0 = ctrls.get(group.get(0));
                double diffV = c0.targetV() - chunkResults[slot][c0.vRow()];
                // OLF adjustB verbatim: round-robin over the magnitude-sorted shunts, one section per
                // sub-controller per pass, per-round maxSectionShift cap, remaining tracked by sensitivity
                double remaining = diffV;
                boolean changed = true;
                int[][] shiftKey = new int[nc][];            // per-(ctrl, sub) section shifts this ROUND
                boolean roundMoved = false;
                while (changed) {
                    changed = false;
                    for (int ci : group) {
                        Controller ctl = ctrls.get(ci);
                        if (disabledMask[slot][ctl.elemIndexAbs()]) {
                            continue;                        // the controller shunt itself is outaged
                        }
                        double sv = sensi[slot * kuF * m + unitOf[ci] * m + g];
                        for (int k = 0; k < ctl.subs().size(); k++) {
                            if (Math.abs(remaining) > c0.halfDeadband()) {
                                if (shiftKey[ci] == null) {
                                    shiftKey[ci] = new int[ctl.subs().size()];
                                }
                                if (shiftKey[ci][k] > maxSectionShift) {
                                    continue;                // reached max section shift in this outer loop
                                }
                                Sub sub = ctl.subs().get(k);
                                double prevB = sub.b(pos[slot][ci][k]);
                                double deltaB = remaining / sv;
                                int[] holder = {pos[slot][ci][k]};
                                if (moveSection(sub, holder, deltaB, ratchets[slot][ci][k]) != 0) {
                                    pos[slot][ci][k] = holder[0];
                                    shiftKey[ci][k]++;
                                    remaining -= (sub.b(pos[slot][ci][k]) - prevB) * sv;
                                    changed = true;
                                    roundMoved = true;
                                }
                            }
                        }
                    }
                }
                anyMoved |= roundMoved;
                if (roundMoved) {
                    affected[slot] = true;
                }
            }
            if (!anyMoved) {
                break;
            }
            // ---- apply: SHUNT_TARGET_B targets + B state seeds + additive (dG,dB) admittance deltas ----
            List<Integer> ovIdx = new ArrayList<>();
            List<Double> ovVal = new ArrayList<>();
            for (int i = 0; i < targetOverrideIndex.length; i++) {
                ovIdx.add(targetOverrideIndex[i]);
                ovVal.add(targetOverrideValue[i]);
            }
            List<Integer> shSlot = new ArrayList<>();
            List<Double> shGb = new ArrayList<>();
            for (int i = 0; i < baseShOvSlotElem.length; i++) {
                shSlot.add(baseShOvSlotElem[i]);
            }
            for (double v : baseShOvGb) {
                shGb.add(v);
            }
            double[] x0Full = statesFlat;
            for (int s = 0; s < chunkSize; s++) {
                for (int c = 0; c < nc; c++) {
                    Controller ctl = ctrls.get(c);
                    double db = 0;
                    double dg = 0;
                    for (int k = 0; k < ctl.subs().size(); k++) {
                        Sub sub = ctl.subs().get(k);
                        db += sub.b(pos[s][c][k]) - sub.b(sub.basePos());
                        dg += sub.g(pos[s][c][k]) - sub.g(sub.basePos());
                    }
                    if (db != 0 || dg != 0) {
                        double newB = baseTargets[ctl.bRow()] + db;
                        ovIdx.add(s * n + ctl.bRow());
                        ovVal.add(newB);                     // the identity row pins B at the section total
                        x0Full[s * n + ctl.bRow()] = newB;   // the CPU re-reads b from the shunt on re-solve
                        shSlot.add(s);
                        shSlot.add(ctl.shOvElem());
                        shGb.add(dg);                        // static G follows the section; static B is
                        shGb.add(db);                        // shadowed by the SHUNT_B variable (harmless)
                    }
                }
            }
            double[][] r2 = GpuAcNewtonSolver.solveBatchContext(handle, baseState, baseTargets,
                    rowModePerScenario, baseRowMode, disabledMask,
                    ovIdx.stream().mapToInt(Integer::intValue).toArray(),
                    ovVal.stream().mapToDouble(Double::doubleValue).toArray(),
                    dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta, dsAnchorShift, dsSlackShift, null, null,
                    parameters.getMaxIterations(), tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                    config.lsMode(), config.maxDv(), config.maxDphi(),
                    config.lsMode() == 2 ? varType : null, x0Full,
                    shSlot.stream().mapToInt(Integer::intValue).toArray(),
                    shGb.stream().mapToDouble(Double::doubleValue).toArray(),
                    null, null,
                    baseR1OvSlotElem, baseR1OvVal,
                    config.maxOuterIterations(), config.maxPqPvSwitch(), config.dsRejectResidualFlag(), 0,
                    nziBase, ziOnSlotElem, ziOffSlotElem);
            for (int s = 0; s < realCount; s++) {
                if (affected[s]) {                           // untouched scenarios KEEP their converged state
                    chunkResults[s] = r2[s];                 // (no extra Newton polish the CPU never runs)
                }
            }
        }
        return chunkResults;
    }

}
