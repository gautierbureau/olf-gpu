/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import com.powsybl.iidm.network.TwoSides;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivation;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcElementMapper;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchOverrides;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchSkips;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuPhaseControlMode;
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
public final class GpuPhaseControlOuterLoop {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuPhaseControlOuterLoop.class);

    private GpuPhaseControlOuterLoop() {
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
                                      GpuAcElementMapper elementMapper) {
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
            double[][] tapRows = GpuTransformerVoltageControlOuterLoop.tapModels(branch.getPiModel());
            // REMOTE control is fine in BOTH modes. CONTROLLER: the activation diff + tap rounding live
            // on the CONTROLLER and the P row re-emission is keyed by cbPcSide on the controlled
            // element. LIMITER: OLF's loop is local BY CONSTRUCTION, not by restriction —
            // changeTapPositions reads controllerBranch.getI1()/getI2() and
            // isSensitivityCurrentPerA1Positive differentiates those same terms, so the controlled
            // branch is simply never consulted. The device builds its I term from the controller too,
            // so a remote LIMITER runs identically; refusing it only routed the run to the CPU.
            if (a1Row == null || elemIndex < 0 || tapRows.length == 0) {
                GpuBatchSkips.record(GpuBatchSkips.Slot.PHASE, LOGGER,
                        "Continuous phase controller {} not representable on the GPU batch — skipped", branch.getId());
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
                out.add(new Ctl(GpuPhaseControlMode.CONTROLLER, a1Row, null, 0, tapRows, baseTap, elemIndex,
                        deltaRows, deltaModesOn, deltaTargetsOn, pc.getControlledSide() == TwoSides.ONE ? 1 : 2, branch.getNum()));
            } else if (pc.getMode() == TransformerPhaseControl.Mode.LIMITER) {
                var iTerm = (EquationTerm<AcVariableType, AcEquationType>) (pc.getControlledSide() == TwoSides.ONE
                        ? branch.getI1() : branch.getI2());
                out.add(new Ctl(GpuPhaseControlMode.LIMITER, a1Row, iTerm, pc.getTargetValue(), tapRows, baseTap, elemIndex,
                        new int[0], new int[0], new double[0], pc.getControlledSide() == TwoSides.ONE ? 1 : 2, branch.getNum()));
            }
        }
        return out;
    }

    /** Pre-solve (OLF initialize): enable every live CONTROLLER-mode phase control — the a1 row
     *  becomes the P-flow equation (row mode 3) for the first solve. */
    public static void applyPhaseOn(int realCount, int n, List<Ctl> ctls,
                             int[][] rowModePerScenario, int[] baseRowMode, boolean[][] disabledMask,
                             GpuBatchOverrides overrides) {
        for (int slot = 0; slot < realCount; slot++) {
            for (Ctl c : ctls) {
                if (c.kind() != GpuPhaseControlMode.CONTROLLER || disabledMask[slot][c.elemIndex()]) {
                    continue;
                }
                int[] row = rowModePerScenario[slot] == baseRowMode
                        ? baseRowMode.clone() : rowModePerScenario[slot];
                for (int k = 0; k < c.deltaRows().length; k++) {
                    row[c.deltaRows()[k]] = c.deltaModesOn()[k];
                    overrides.add(slot * n + c.deltaRows()[k], c.deltaTargetsOn()[k]);
                }
                rowModePerScenario[slot] = row;
            }
        }
    }

    /** Install the controllers for the in-kernel PhaseControl phase (OLF order 7). Returns true when the
     *  kernel loop OWNS them (the host orchestrator must then stay off). */
    public static boolean uploadToDevice(long handle, List<Ctl> ctls) {
        if (ctls == null || ctls.isEmpty()) {
            PhaseControlNative.setBatchContinuousPhase(handle,
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
        boolean owned = PhaseControlNative.setBatchContinuousPhase(handle,
                kind, a1Row, side, limit, elem, tapPtr, tapData.stream().mapToDouble(Double::doubleValue).toArray(), baseTap) != 0;
        LOGGER.info("Uploaded {} continuous phase controllers to device: {}", nc, owned ? "in-kernel loop" : "host orchestrator");
        if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
            System.err.printf("CPHASE_UPLOAD nCtrl=%d kernel=%b%n", nc, owned);
        }
        return owned;
    }

    /** Give the driver's decisions (scenario 0) back to the network the way OLF's loop leaves it: every
     *  CONTROLLER-mode control switched off, every tap at the rounded / shifted position. */
    public static void applyToNetwork(LfNetwork network, List<Ctl> ctls, int[] taps) {
        if (ctls == null || ctls.isEmpty() || taps == null) {
            return;
        }
        for (int c = 0; c < ctls.size() && c < taps.length; c++) {
            Ctl ctl = ctls.get(c);
            LfBranch branch = network.getBranch(ctl.branchNum());
            if (branch == null) {
                continue;
            }
            if (ctl.kind() == GpuPhaseControlMode.CONTROLLER && branch.isPhaseControlEnabled()) {
                branch.setPhaseControlEnabled(false);
            }
            var pm = branch.getPiModel();
            int pos = pm.getTapPositionRange().getMinimum() + taps[c];
            if (pm.getTapPosition() != pos) {
                pm.setTapPosition(pos);
            }
        }
    }

}
