/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.iidm.network.TwoSides;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivation;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.EquationTerm;
import com.powsybl.openloadflow.equations.TargetVector;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.TransformerPhaseControl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * DEVICE port of OLF's CONTINUOUS {@code PhaseControlOuterLoop} ({@code CONTINUOUS_WITH_DISCRETISATION}):
 *
 * <ul>
 *   <li>CONTROLLER mode — the Simple-TVC shape over a1: {@code initialize()} enables phase control
 *       BEFORE the first solve (per scenario, the a1-paired row flips to the P-flow equation, row
 *       mode 3 — the cb family re-emits the controlled side's P outputs there and a1 solves free);
 *       the loop's first check switches control OFF and rounds a1 to the CLOSEST TAP
 *       ({@code roundA1ToClosestTap}) → one warm re-solve with the a1 identity pinned at the rounded
 *       tap and the tap's full pi row (per-step R/X/G/B) applied;</li>
 *   <li>LIMITER mode — from the second iteration: while the controlled side's current exceeds the
 *       limit, shift ONE tap per round in the direction that reduces it (the SIGN of dI/da1, read
 *       off OLF's own current term), exactly {@code changeTapPositions}.</li>
 * </ul>
 *
 * <p>Scope: CONTROLLER mode supports LOCAL and REMOTE control (2026-09-05 — cbPcSide carries the
 * controller's a1 column, so the controlled element emits at the right row); LIMITER stays local
 * (OLF's own loop supports only local).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuBatchContinuousPhase {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuBatchContinuousPhase.class);

    private GpuBatchContinuousPhase() {
    }

    /** kind 0 = CONTROLLER (continuous then discretized), 1 = LIMITER (one-tap current capping). */
    public record Ctl(int kind, int a1Row, EquationTerm<AcVariableType, AcEquationType> iTerm, double iLimit,
               double[][] tapRows, int baseTapIdx, int elemIndex,
               int[] deltaRows, int[] deltaModesOn, double[] deltaTargetsOn, int controlledSide, int branchNum) {
    }

    @SuppressWarnings("unchecked")
    public static List<Ctl> buildControllers(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                      TargetVector<AcVariableType, AcEquationType> targetVector,
                                      GpuAcActivation baseActivation, double[] baseTargets,
                                      GpuBatchElementMapper elementMapper) {
        Map<Integer, Integer> a1Rows = new LinkedHashMap<>();
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            if (v.getType() == AcVariableType.BRANCH_ALPHA1) {
                a1Rows.put(v.getElementNum(), v.getRow());
            }
        }
        List<Ctl> out = new ArrayList<>();
        for (LfBranch branch : network.getBranches()) {
            var pcOpt = branch.getPhaseControl();
            if (pcOpt.isEmpty() || branch.isDisabled() || !branch.isPhaseController()) {
                continue;
            }
            TransformerPhaseControl pc = pcOpt.orElseThrow();
            Integer a1Row = a1Rows.get(branch.getNum());
            int elemIndex = elementMapper.getElementIndex(branch);
            double[][] tapRows = GpuBatchTransformerControl.tapModels(branch.getPiModel());
            // REMOTE control is fine for CONTROLLER mode (the activation diff + tap rounding live on
            // the CONTROLLER; the P row re-emission is keyed by cbPcSide on the controlled element);
            // LIMITER stays local-only — OLF's own loop reads the CONTROLLER's I terms.
            boolean remoteLimiter = pc.getMode() == TransformerPhaseControl.Mode.LIMITER
                    && pc.getControlledBranch() != branch;
            if (a1Row == null || elemIndex < 0 || tapRows.length == 0 || remoteLimiter) {
                LOGGER.info("Continuous phase controller {} not representable on the GPU batch — skipped", branch.getId());
                continue;
            }
            double a1Now = branch.getPiModel().getA1();
            int baseTap = 0;
            for (int t = 1; t < tapRows.length; t++) {
                if (Math.abs(tapRows[t][7] - a1Now) < Math.abs(tapRows[baseTap][7] - a1Now)) {
                    baseTap = t;
                }
            }
            if (pc.getMode() == TransformerPhaseControl.Mode.CONTROLLER) {
                // VC-ON activation diff: what enabling the phase control changes (a1 row -> mode 3 + target)
                boolean was = branch.isPhaseControlEnabled();
                branch.setPhaseControlEnabled(true);
                GpuAcActivation onAct = GpuAcActivation.fromEquationSystem(network, es);
                double[] onTargets = onAct.applyTargets(targetVector.getArray());
                branch.setPhaseControlEnabled(was);
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
                out.add(new Ctl(0, a1Row, null, 0, tapRows, baseTap, elemIndex,
                        deltaRows, deltaModesOn, deltaTargetsOn, pc.getControlledSide() == TwoSides.ONE ? 1 : 2, branch.getNum()));
            } else if (pc.getMode() == TransformerPhaseControl.Mode.LIMITER) {
                var iTerm = (EquationTerm<AcVariableType, AcEquationType>) (pc.getControlledSide() == TwoSides.ONE
                        ? branch.getI1() : branch.getI2());
                out.add(new Ctl(1, a1Row, iTerm, pc.getTargetValue(), tapRows, baseTap, elemIndex,
                        new int[0], new int[0], new double[0], pc.getControlledSide() == TwoSides.ONE ? 1 : 2, branch.getNum()));
            }
        }
        return out;
    }

    /** Pre-solve (OLF initialize): enable every live CONTROLLER-mode phase control — the a1 row
     *  becomes the P-flow equation (row mode 3) for the first solve. */
    static void applyPhaseOn(int realCount, int n, List<Ctl> ctls,
                             int[][] rowModePerScenario, int[] baseRowMode, boolean[][] disabledMask,
                             List<Integer> ovIndexList, List<Double> ovValueList) {
        for (int slot = 0; slot < realCount; slot++) {
            for (Ctl c : ctls) {
                if (c.kind() != 0 || disabledMask[slot][c.elemIndex()]) {
                    continue;
                }
                int[] row = rowModePerScenario[slot] == baseRowMode
                        ? baseRowMode.clone() : rowModePerScenario[slot];
                for (int k = 0; k < c.deltaRows().length; k++) {
                    row[c.deltaRows()[k]] = c.deltaModesOn()[k];
                    ovIndexList.add(slot * n + c.deltaRows()[k]);
                    ovValueList.add(c.deltaTargetsOn()[k]);
                }
                rowModePerScenario[slot] = row;
            }
        }
    }

    /** Install the controllers for the in-kernel PhaseControl phase (OLF order 7). Returns true when the
     *  kernel loop OWNS them (the host orchestrator must then stay off). */
    public static boolean uploadToDevice(long handle, List<Ctl> ctls) {
        if (ctls == null || ctls.isEmpty()) {
            com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.setBatchContinuousPhase(handle,
                    null, null, null, null, null, null, null, null);
            return false;
        }
        int nc = ctls.size();
        int[] kind = new int[nc];
        int[] a1Row = new int[nc];
        int[] side = new int[nc];
        double[] limit = new double[nc];
        int[] elem = new int[nc];
        int[] tapPtr = new int[nc + 1];
        int[] baseTap = new int[nc];
        List<Double> tapData = new ArrayList<>();
        for (int c = 0; c < nc; c++) {
            Ctl ctl = ctls.get(c);
            kind[c] = ctl.kind();
            a1Row[c] = ctl.a1Row();
            side[c] = ctl.controlledSide();
            limit[c] = ctl.iLimit();
            elem[c] = ctl.elemIndex();
            tapPtr[c] = tapData.size() / 8;
            for (double[] row : ctl.tapRows()) {
                for (double d : row) {
                    tapData.add(d);
                }
            }
            baseTap[c] = ctl.baseTapIdx();
        }
        tapPtr[nc] = tapData.size() / 8;
        boolean owned = com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.setBatchContinuousPhase(handle,
                kind, a1Row, side, limit, elem, tapPtr, tapData.stream().mapToDouble(Double::doubleValue).toArray(), baseTap) != 0;
        LOGGER.info("Uploaded {} continuous phase controllers to device: {}", nc, owned ? "in-kernel loop" : "host orchestrator");
        if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
            System.err.printf("CPHASE_UPLOAD nCtrl=%d kernel=%b%n", nc, owned);
        }
        return owned;
    }

    /** Give the driver's decisions (scenario 0) back to the network the way OLF's loop leaves it: every
     *  CONTROLLER-mode control switched off, every tap at the rounded / shifted position. */
    public static void writeBack(LfNetwork network, List<Ctl> ctls, int[] taps) {
        if (ctls == null || ctls.isEmpty() || taps == null) {
            return;
        }
        for (int c = 0; c < ctls.size() && c < taps.length; c++) {
            Ctl ctl = ctls.get(c);
            LfBranch branch = network.getBranch(ctl.branchNum());
            if (branch == null) {
                continue;
            }
            if (ctl.kind() == 0 && branch.isPhaseControlEnabled()) {
                branch.setPhaseControlEnabled(false);
            }
            var pm = branch.getPiModel();
            int pos = pm.getTapPositionRange().getMinimum() + taps[c];
            if (pm.getTapPosition() != pos) {
                pm.setTapPosition(pos);
            }
        }
    }

    /** OLF PiModel.shiftOneTapPositionToChangeA1: one tap toward larger (dir>0) / smaller (dir<0) a1. */
    private static int shiftOneTap(Ctl c, int tap, int dir) {
        double a1 = c.tapRows()[tap][7];
        if (tap + 1 < c.tapRows().length
                && (dir > 0) == (c.tapRows()[tap + 1][7] > a1)) {
            return tap + 1;
        }
        if (tap - 1 >= 0 && (dir > 0) == (c.tapRows()[tap - 1][7] > a1)) {
            return tap - 1;
        }
        return tap;
    }

    @SuppressWarnings({"java:S107", "java:S3776"})
    static double[][] orchestrate(long handle, int chunkSize, int realCount, int n,
            double[][] chunkResults, List<Ctl> ctls, GpuBatchIncrementalFlowControl.EsContext esCtx,
            int[][] rowModePerScenario, int[] baseRowMode, boolean[][] disabledMask,
            List<Integer> baseOvIndexList, List<Double> baseOvValueList,
            int[] dsOvBus, double[] dsOvFactorDelta, double[] dsOvBaseDelta, double[] dsOvMaxDelta, double[] dsOvMinDelta,
            int[] dsPhiRow, double[] dsFactor,
            int[] baseR1OvSlotElem, double[] baseR1OvVal, int[] shOvSlotElem, double[] shOvGb,
            int nziBase, int[] ziOnSlotElem, int[] ziOffSlotElem,
            double[] baseState, double[] baseTargets, NewtonRaphsonParameters parameters,
            double tol, double reactiveLimitTol, double slackDistTol, double convEpsPerEq, int[] varType,
            BatchConfig config) {
        double[][] results = chunkResults;
        // DS target carry (same three coupled pieces as GpuBatchTransformerControl): the MAIN
        // solve's slack distribution must ride into the pinned re-solves, or the kernel re-derives
        // it from the re-uploaded baseTargets along a trajectory the CPU never takes.
        int nds = dsPhiRow != null ? dsPhiRow.length : 0;
        double[] dsAnchorCarry = nds > 0 ? new double[chunkSize * nds] : null;
        double[] dsSlackCarry = nds > 0 ? new double[chunkSize] : null;
        double[] dsSeed = nds > 0 ? new double[chunkSize * nds] : null;
        if (nds > 0) {
            GpuBatchTransformerControl.captureDsCarry(handle, chunkSize, realCount, n, dsPhiRow, dsFactor, baseTargets,
                    dsOvBus, dsOvFactorDelta, dsOvBaseDelta, chunkResults, dsAnchorCarry, dsSlackCarry, dsSeed);
        }
        int nc = ctls.size();
        int[][] tap = new int[chunkSize][nc];
        for (int s = 0; s < chunkSize; s++) {
            for (int c = 0; c < nc; c++) {
                tap[s][c] = ctls.get(c).baseTapIdx();
            }
        }
        int maxRounds = config.maxOuterIterations() > 0 ? config.maxOuterIterations() : 20;

        // ---- ROUND 0 (OLF firstIteration): switch off + round each continuous a1 to the closest tap ----
        boolean anyController = false;
        boolean[] affected = new boolean[chunkSize];
        for (int s = 0; s < realCount; s++) {
            double[] r = results[s];
            if (r == null || (int) r[n + 2] != GpuBatchStatus.STATUS_CONVERGED) {
                continue;
            }
            for (int c = 0; c < nc; c++) {
                Ctl ctl = ctls.get(c);
                if (disabledMask[s][ctl.elemIndex()]) {
                    continue;
                }
                // OLF's iteration-0 check returns UNSTABLE whenever ANY live phase controller exists
                // (LIMITER included, nothing moved) -> every such scenario takes one extra Newton run
                // on the CPU; re-solve it here too or the states drift by the one-iteration polish.
                affected[s] = true;
                anyController = true;
                if (ctl.kind() != 0) {
                    continue;
                }
                double a1 = r[ctl.a1Row()];
                int best = tap[s][c];
                for (int t = 0; t < ctl.tapRows().length; t++) {
                    if (Math.abs(ctl.tapRows()[t][7] - a1) < Math.abs(ctl.tapRows()[best][7] - a1)) {
                        best = t;
                    }
                }
                tap[s][c] = best;
            }
        }
        if (anyController) {                                  // pinned re-solve at the rounded taps
            results = resolvePinned(handle, chunkSize, realCount, n, results, ctls, tap, affected,
                    rowModePerScenario, baseRowMode, disabledMask, baseOvIndexList, baseOvValueList,
                    dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta,
                    nds, dsPhiRow, dsAnchorCarry, dsSlackCarry, dsSeed,
                    baseR1OvSlotElem, baseR1OvVal, shOvSlotElem, shOvGb, nziBase, ziOnSlotElem, ziOffSlotElem,
                    baseState, baseTargets, parameters, tol, reactiveLimitTol, slackDistTol, convEpsPerEq, varType, config);
            if (nds > 0) {                                   // this solve may have distributed further
                GpuBatchTransformerControl.captureDsCarry(handle, chunkSize, realCount, n, dsPhiRow, dsFactor, baseTargets,
                        dsOvBus, dsOvFactorDelta, dsOvBaseDelta, results, dsAnchorCarry, dsSlackCarry, dsSeed, affected);
            }
        }
        // ---- LIMITER rounds (OLF nextIteration): one tap per round while I exceeds the limit ----
        for (int round = 0; round < maxRounds; round++) {
            boolean[] moved = new boolean[chunkSize];
            boolean any = false;
            for (int s = 0; s < realCount; s++) {
                double[] r = results[s];
                if (r == null || (int) r[n + 2] != GpuBatchStatus.STATUS_CONVERGED) {
                    continue;
                }
                boolean loaded = false;
                for (int c = 0; c < nc; c++) {
                    Ctl ctl = ctls.get(c);
                    if (ctl.kind() != 1 || disabledMask[s][ctl.elemIndex()]) {
                        continue;
                    }
                    if (!loaded) {
                        esCtx.stateVector().set(java.util.Arrays.copyOf(r, n));
                        loaded = true;
                    }
                    if (ctl.iTerm().eval() > ctl.iLimit()) {
                        var a1Var = ctl.iTerm().getVariables().stream()
                                .filter(v -> v.getType() == AcVariableType.BRANCH_ALPHA1).findFirst().orElseThrow();
                        int dir = ctl.iTerm().der(a1Var) > 0 ? -1 : 1;   // reduce I: move a1 against the slope
                        int next = shiftOneTap(ctl, tap[s][c], dir);
                        if (next != tap[s][c]) {
                            tap[s][c] = next;
                            moved[s] = true;
                            any = true;
                        }
                    }
                }
            }
            if (!any) {
                break;
            }
            results = resolvePinned(handle, chunkSize, realCount, n, results, ctls, tap, moved,
                    rowModePerScenario, baseRowMode, disabledMask, baseOvIndexList, baseOvValueList,
                    dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta,
                    nds, dsPhiRow, dsAnchorCarry, dsSlackCarry, dsSeed,
                    baseR1OvSlotElem, baseR1OvVal, shOvSlotElem, shOvGb, nziBase, ziOnSlotElem, ziOffSlotElem,
                    baseState, baseTargets, parameters, tol, reactiveLimitTol, slackDistTol, convEpsPerEq, varType, config);
            if (nds > 0) {                                   // this solve may have distributed further
                GpuBatchTransformerControl.captureDsCarry(handle, chunkSize, realCount, n, dsPhiRow, dsFactor, baseTargets,
                        dsOvBus, dsOvFactorDelta, dsOvBaseDelta, results, dsAnchorCarry, dsSlackCarry, dsSeed, moved);
            }
        }
        return results;
    }

    /** Warm re-solve with every controller's a1 identity restored/pinned at its tap and the tap's full
     *  pi row applied; only the {@code affected} scenarios take the re-solved state. */
    @SuppressWarnings("java:S107")
    private static double[][] resolvePinned(long handle, int chunkSize, int realCount, int n,
            double[][] chunkResults, List<Ctl> ctls, int[][] tap, boolean[] affected,
            int[][] rowModePerScenario, int[] baseRowMode, boolean[][] disabledMask,
            List<Integer> baseOvIndexList, List<Double> baseOvValueList,
            int[] dsOvBus, double[] dsOvFactorDelta, double[] dsOvBaseDelta, double[] dsOvMaxDelta, double[] dsOvMinDelta,
            int nds, int[] dsPhiRow, double[] dsAnchorCarry, double[] dsSlackCarry, double[] dsSeed,
            int[] baseR1OvSlotElem, double[] baseR1OvVal, int[] shOvSlotElem, double[] shOvGb,
            int nziBase, int[] ziOnSlotElem, int[] ziOffSlotElem,
            double[] baseState, double[] baseTargets, NewtonRaphsonParameters parameters,
            double tol, double reactiveLimitTol, double slackDistTol, double convEpsPerEq, int[] varType,
            BatchConfig config) {
        List<Integer> ovIdx = new ArrayList<>(baseOvIndexList);
        List<Double> ovVal = new ArrayList<>(baseOvValueList);
        GpuBatchTransformerControl.seedDsTargets(nds, dsPhiRow, dsSeed, chunkResults, realCount, n, ovIdx, ovVal);
        List<Integer> r1Slot = new ArrayList<>();
        List<Double> r1Val = new ArrayList<>();
        for (int v : baseR1OvSlotElem) {
            r1Slot.add(v);
        }
        for (double v : baseR1OvVal) {
            r1Val.add(v);
        }
        double[] x0Full = new double[chunkSize * n];
        int[][] rowModeOff = new int[chunkSize][];
        for (int s = 0; s < chunkSize; s++) {
            double[] src = s < realCount && chunkResults[s] != null ? chunkResults[s] : baseState;
            System.arraycopy(src, 0, x0Full, s * n, n);
            rowModeOff[s] = rowModePerScenario[s];
        }
        for (int s = 0; s < realCount; s++) {
            int[] row = null;
            for (int c = 0; c < ctls.size(); c++) {
                Ctl ctl = ctls.get(c);
                if (disabledMask[s][ctl.elemIndex()]) {
                    continue;
                }
                if (ctl.kind() == 0) {                       // control OFF again: a1 identity restored
                    if (row == null) {
                        row = rowModeOff[s].clone();
                    }
                    for (int k = 0; k < ctl.deltaRows().length; k++) {
                        row[ctl.deltaRows()[k]] = baseRowMode[ctl.deltaRows()[k]];
                    }
                }
                double[] tapRow = ctl.tapRows()[tap[s][c]];
                ovIdx.add(s * n + ctl.a1Row());
                ovVal.add(tapRow[7]);                        // the identity row pins a1 at the tap's value
                x0Full[s * n + ctl.a1Row()] = tapRow[7];
                r1Slot.add(s);
                r1Slot.add(ctl.elemIndex());
                for (double d : tapRow) {
                    r1Val.add(d);                            // full 8-double pi row: per-step R/X/G/B follow
                }
            }
            if (row != null) {
                rowModeOff[s] = row;
                rowModePerScenario[s] = row;                 // controls stay OFF for subsequent rounds
            }
        }
        double[][] r2 = GpuAcNewtonSolver.solveBatchContext(handle, baseState, baseTargets,
                rowModeOff, baseRowMode, disabledMask,
                ovIdx.stream().mapToInt(Integer::intValue).toArray(),
                ovVal.stream().mapToDouble(Double::doubleValue).toArray(),
                dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta, dsAnchorCarry, dsSlackCarry, null, null,
                parameters.getMaxIterations(), tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                config.lsMode(), config.maxDv(), config.maxDphi(),
                config.lsMode() == 2 ? varType : null, x0Full,
                shOvSlotElem, shOvGb, null, null,
                r1Slot.stream().mapToInt(Integer::intValue).toArray(),
                r1Val.stream().mapToDouble(Double::doubleValue).toArray(),
                config.maxOuterIterations(), config.maxPqPvSwitch(), config.dsRejectResidualFlag(), 0,
                nziBase, ziOnSlotElem, ziOffSlotElem);
        for (int s = 0; s < realCount; s++) {
            if (affected[s]) {                               // untouched scenarios KEEP their converged state
                chunkResults[s] = r2[s];
            }
        }
        return chunkResults;
    }
}
