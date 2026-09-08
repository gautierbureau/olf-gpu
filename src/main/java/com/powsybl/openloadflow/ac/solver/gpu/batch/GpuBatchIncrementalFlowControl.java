/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.iidm.network.TwoSides;
import com.powsybl.math.matrix.DenseMatrix;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.EquationTerm;
import com.powsybl.openloadflow.equations.StateVector;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * DEVICE port of the FLOW-OBSERVED incremental loops — OLF's
 * {@code IncrementalTransformerReactivePowerControlOuterLoop} (rho taps driving a branch Q flow) and
 * {@code AcIncrementalPhaseControlOuterLoop} (a1 taps driving a branch P flow in CONTROLLER mode, or
 * capping a branch current in LIMITER mode). Same skeleton as the voltage-observed ports
 * ({@link GpuBatchIncrementalTvc}): batched in-kernel sensitivities with a unit RHS at each
 * controller's pinned identity row (BRANCH_TARGET_RHO1 / BRANCH_TARGET_ALPHA1), but the observable
 * is a BRANCH FLOW: OLF's own {@code EquationTerm.calculateSensi} evaluates the flow gradient dotted
 * with the sensitivity column — the orchestrator loads each scenario's state into the shared
 * equation-system state vector and calls the very same terms ({@code getQ1/getP1/getI1} & co), with
 * the device solution injected at the term's variable rows of a sparse-filled DenseMatrix, so the
 * observable math is OLF's code, not a re-derivation.
 *
 * <p>Phase sensitivities: OLF's RHS is {@code toRadians(1)} (per-degree); with the device's unit RHS
 * the tap shift collapses to {@code da = diff / rawSensi} (radians), and the {@code SENSI_EPS} gate
 * applies to the per-degree value ({@code rawSensi * toRadians(1)}).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuBatchIncrementalFlowControl {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuBatchIncrementalFlowControl.class);

    private static final int MAX_DIRECTION_CHANGE = 3;       // both OLF loops
    private static final double MIN_TARGET_DEADBAND_MVAR = 0.1;      // RPC (per-unit, OLF literal)
    private static final double MIN_TARGET_DEADBAND_P = 1.0 / 100.0; // phase: 1 MW in per-unit
    private static final double SENSI_EPS = 1e-6;            // phase loops only (RPC has no gate)

    static final int KIND_RPC = 0;                           // rho -> branch Q (closest-R1 taps)
    static final int KIND_PHASE_P = 1;                       // a1 -> branch P (closest-A1 taps)
    static final int KIND_PHASE_I = 2;                       // a1 -> branch I (exceed-A1 taps, one-sided)

    private GpuBatchIncrementalFlowControl() {
    }

    record Controller(int kind, int pinRow, int pinTapCol, EquationTerm<AcVariableType, AcEquationType> flowTerm,
                      double targetValue, double halfDeadband, double[][] tapRows, int baseTapIdx,
                      int elemIndex, int controlledElemIndex, int maxTapShift) {
        double pin(int tap) {
            return tapRows[tap][pinTapCol];
        }
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

    /** The observable-evaluation context captured at build time: the shared es state vector (scenario
     *  states are loaded into it so OLF's own flow terms evaluate/differentiate at the right point). */
    record EsContext(StateVector stateVector, int n) {
    }

    @SuppressWarnings("unchecked")
    static List<Controller> buildControllers(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                             GpuBatchElementMapper elementMapper, int rpcMaxTapShift,
                                             boolean rpc, boolean phase) {
        List<Controller> out = new ArrayList<>();
        Map<Integer, Integer> rhoRows = new LinkedHashMap<>();
        Map<Integer, Integer> a1Rows = new LinkedHashMap<>();
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            if (v.getType() == AcVariableType.BRANCH_RHO1) {
                rhoRows.put(v.getElementNum(), v.getRow());
            } else if (v.getType() == AcVariableType.BRANCH_ALPHA1) {
                a1Rows.put(v.getElementNum(), v.getRow());
            }
        }
        if (rpc) {
            for (LfBranch controlled : network.getBranches()) {
                if (!controlled.isTransformerReactivePowerControlled() || controlled.isDisabled()) {
                    continue;
                }
                var rc = controlled.getTransformerReactivePowerControl().orElseThrow();
                LfBranch controller = rc.getControllerBranch();
                Integer pinRow = rhoRows.get(controller.getNum());
                int elemIndex = elementMapper.getElementIndex(controller);
                int ctlElemIndex = elementMapper.getElementIndex(controlled);
                double[][] tapRows = GpuBatchTransformerControl.tapModels(controller.getPiModel());
                if (pinRow == null || elemIndex < 0 || ctlElemIndex < 0 || tapRows.length == 0) {
                    LOGGER.info("Incremental RPC controller {} not representable on the GPU batch — skipped", controller.getId());
                    continue;
                }
                var term = (EquationTerm<AcVariableType, AcEquationType>) (rc.getControlledSide() == TwoSides.ONE
                        ? controlled.getQ1() : controlled.getQ2());
                // OLF getHighestPriorityReactivePowerTarget: a generator reactive power control wins
                double target = controlled.getGeneratorReactivePowerControl()
                        .map(com.powsybl.openloadflow.network.GeneratorReactivePowerControl::getTargetValue)
                        .orElse(rc.getTargetValue());
                double halfDb = rc.getTargetDeadband().orElse(MIN_TARGET_DEADBAND_MVAR) / 2;
                out.add(new Controller(KIND_RPC, pinRow, 6, term, target, halfDb,
                        tapRows, closestTap(tapRows, 6, controller.getPiModel().getR1()),
                        elemIndex, ctlElemIndex, rpcMaxTapShift));
            }
        }
        if (phase) {
            for (LfBranch controller : network.getBranches()) {
                var pcOpt = controller.getPhaseControl();
                if (pcOpt.isEmpty() || controller.isDisabled() || !controller.isPhaseController()) {
                    continue;
                }
                var pc = pcOpt.orElseThrow();
                int kind = switch (pc.getMode()) {
                    case CONTROLLER -> KIND_PHASE_P;
                    case LIMITER -> KIND_PHASE_I;
                    default -> -1;
                };
                if (kind < 0) {
                    continue;
                }
                LfBranch controlled = pc.getControlledBranch();
                Integer pinRow = a1Rows.get(controller.getNum());
                int elemIndex = elementMapper.getElementIndex(controller);
                int ctlElemIndex = elementMapper.getElementIndex(controlled);
                double[][] tapRows = GpuBatchTransformerControl.tapModels(controller.getPiModel());
                if (pinRow == null || elemIndex < 0 || ctlElemIndex < 0 || tapRows.length == 0) {
                    LOGGER.info("Incremental phase controller {} not representable on the GPU batch — skipped", controller.getId());
                    continue;
                }
                EquationTerm<AcVariableType, AcEquationType> term;
                if (kind == KIND_PHASE_P) {
                    term = (EquationTerm<AcVariableType, AcEquationType>) (pc.getControlledSide() == TwoSides.ONE
                            ? controlled.getP1() : controlled.getP2());
                } else {
                    term = (EquationTerm<AcVariableType, AcEquationType>) (pc.getControlledSide() == TwoSides.ONE
                            ? controlled.getI1() : controlled.getI2());
                }
                double halfDb = Math.max(pc.getTargetDeadband(), MIN_TARGET_DEADBAND_P) / 2;
                out.add(new Controller(kind, pinRow, 7, term, pc.getTargetValue(), halfDb,
                        tapRows, closestTap(tapRows, 7, controller.getPiModel().getA1()),
                        elemIndex, ctlElemIndex, Integer.MAX_VALUE));
            }
        }
        return out;
    }

    private static int closestTap(double[][] tapRows, int col, double value) {
        int best = 0;
        for (int t = 1; t < tapRows.length; t++) {
            if (Math.abs(tapRows[t][col] - value) < Math.abs(tapRows[best][col] - value)) {
                best = t;
            }
        }
        return best;
    }

    /** OLF ClosestTapPositionFinder over the pinned column (r1 or a1). Returns 0 or the direction. */
    private static int moveClosest(Controller c, int[] tap, double delta, int maxTapShift, Ratchet ratchet) {
        int cur = tap[0];
        double newV = c.pin(cur) + delta;
        int lo = ratchet.allowedDirection > 0 ? cur : 0;
        int hi = ratchet.allowedDirection < 0 ? cur : c.tapRows().length - 1;
        int best = cur;
        double bestDist = Math.abs(newV - c.pin(cur));
        for (int i = lo; i <= hi; i++) {
            if (Math.abs(i - cur) > maxTapShift) {
                continue;
            }
            double dist = Math.abs(newV - c.pin(i));
            if (dist < bestDist) {
                best = i;
                bestDist = dist;
            }
        }
        if (best == cur) {
            return 0;
        }
        tap[0] = best;
        int direction = best > cur ? 1 : -1;
        ratchet.update(direction);
        return direction;
    }

    /** OLF FirstTapPositionAboveFinder over a1 (LIMITER mode): walk tap by tap in the shift's direction
     *  while the remaining shift still reaches past the next tap's a1, within the allowed range. */
    private static int moveExceed(Controller c, int[] tap, double deltaA1, Ratchet ratchet) {
        int start = tap[0];
        int cur = start;
        double remaining = deltaA1;
        int lo = ratchet.allowedDirection > 0 ? start : 0;
        int hi = ratchet.allowedDirection < 0 ? start : c.tapRows().length - 1;
        while (true) {
            int next = nextIndex(c, cur, remaining);
            if (next < 0 || next < lo || next > hi) {
                break;
            }
            double value = c.pin(cur);
            double nextValue = c.pin(next);
            cur = next;
            if (remaining < 0 && value + remaining > nextValue
                    || remaining > 0 && value + remaining < nextValue) {
                break;                                       // shift not enough to go to the next position
            }
            remaining -= nextValue - value;
        }
        if (cur == start) {
            return 0;
        }
        tap[0] = cur;
        int direction = cur > start ? 1 : -1;
        ratchet.update(direction);
        return direction;
    }

    private static int nextIndex(Controller c, int i, double shift) {  // OLF nextTapPositionIndex
        if (shift == 0) {
            return -1;
        }
        int last = c.tapRows().length - 1;
        if (i == 0 && descending(c, 1, 0, shift)) {
            return -1;
        }
        if (i > 0 && descending(c, i, i - 1, shift)) {
            return i - 1;
        }
        if (i < last && descending(c, i, i + 1, shift)) {
            return i + 1;
        }
        return -1;
    }

    private static boolean descending(Controller c, int n1, int n2, double shift) {
        return shift > 0 ? c.pin(n1) < c.pin(n2) : c.pin(n1) > c.pin(n2);
    }

    @SuppressWarnings({"java:S107", "java:S3776"})
    static double[][] orchestrate(long handle, int chunkSize, int realCount, int n,
            double[][] chunkResults, List<Controller> ctrls, EsContext esCtx,
            int[][] rowModePerScenario, int[] baseRowMode, boolean[][] disabledMask,
            int[] targetOverrideIndex, double[] targetOverrideValue,
            int[] dsOvBus, double[] dsOvFactorDelta, double[] dsOvBaseDelta, double[] dsOvMaxDelta, double[] dsOvMinDelta,
            int[] dsPhiRow, double[] dsFactor,
            int[] baseR1OvSlotElem, double[] baseR1OvVal, int[] shOvSlotElem, double[] shOvGb,
            int nziBase, int[] ziOnSlotElem, int[] ziOffSlotElem,
            double[] baseState, double[] baseTargets, NewtonRaphsonParameters parameters,
            double tol, double reactiveLimitTol, double slackDistTol, double convEpsPerEq, int[] varType,
            BatchConfig config) {
        int nc = ctrls.size();
        int[][] tap = new int[chunkSize][nc];
        Ratchet[][] ratchets = new Ratchet[chunkSize][nc];
        for (int s = 0; s < chunkSize; s++) {
            for (int c = 0; c < nc; c++) {
                tap[s][c] = ctrls.get(c).baseTapIdx();
                ratchets[s][c] = new Ratchet();
            }
        }
        // the union of the flow terms' variable rows: the JNI returns dx at these; a sparse-filled
        // DenseMatrix hands them to OLF's own term.calculateSensi
        List<Integer> obsList = new ArrayList<>();
        Map<Integer, Integer> obsPos = new LinkedHashMap<>();
        for (Controller c : ctrls) {
            for (Variable<AcVariableType> v : c.flowTerm().getVariables()) {
                obsPos.computeIfAbsent(v.getRow(), r -> {
                    obsList.add(r);
                    return obsList.size() - 1;
                });
            }
        }
        int[] obs = obsList.stream().mapToInt(Integer::intValue).toArray();
        int m = obs.length;
        int maxRounds = config.maxOuterIterations() > 0 ? config.maxOuterIterations() : 20;
        // DS target carry (same three coupled pieces as GpuBatchTransformerControl): the MAIN
        // solve's slack distribution must ride into every tap-move re-solve, or the kernel
        // re-derives it from the re-uploaded baseTargets along a trajectory the CPU never takes.
        int nds = dsPhiRow != null ? dsPhiRow.length : 0;
        double[] dsAnchorCarry = nds > 0 ? new double[chunkSize * nds] : null;
        double[] dsSlackCarry = nds > 0 ? new double[chunkSize] : null;
        double[] dsSeed = nds > 0 ? new double[chunkSize * nds] : null;
        if (nds > 0) {
            GpuBatchTransformerControl.captureDsCarry(handle, chunkSize, realCount, n, dsPhiRow, dsFactor, baseTargets,
                    dsOvBus, dsOvFactorDelta, dsOvBaseDelta, chunkResults, dsAnchorCarry, dsSlackCarry, dsSeed);
        }

        for (int round = 0; round < maxRounds; round++) {
            // ---- which (scenario, controller) pairs need attention on the current converged states ----
            double[][] diffOf = new double[realCount][nc];   // target - flow (RPC/P) or target - i (LIMITER)
            List<int[]> pending = new ArrayList<>();
            boolean[] ctrlNeeded = new boolean[nc];
            for (int slot = 0; slot < realCount; slot++) {
                double[] r = chunkResults[slot];
                if (r == null || (int) r[n + 2] != GpuBatchStatus.STATUS_CONVERGED) {
                    continue;
                }
                esCtx.stateVector().set(java.util.Arrays.copyOf(r, n));
                for (int c = 0; c < nc; c++) {
                    Controller ctl = ctrls.get(c);
                    if (disabledMask[slot][ctl.elemIndex()] || disabledMask[slot][ctl.controlledElemIndex()]) {
                        continue;
                    }
                    double flow = ctl.flowTerm().eval();
                    double diff = ctl.targetValue() - flow;
                    boolean needs = ctl.kind() == KIND_PHASE_I
                            ? flow > ctl.targetValue()                        // LIMITER: one-sided
                            : Math.abs(diff) > ctl.halfDeadband();
                    if (needs) {
                        diffOf[slot][c] = diff;
                        pending.add(new int[] {slot, c});
                        ctrlNeeded[c] = true;
                    }
                }
            }
            if (pending.isEmpty()) {
                break;
            }
            // ---- batched device sensitivities: unit RHS at each needed controller's pinned row ----
            int[] unit = new int[nc];
            int[] unitOf = new int[nc];
            int ku = 0;
            for (int c = 0; c < nc; c++) {
                if (ctrlNeeded[c]) {
                    unitOf[c] = ku;
                    unit[ku++] = ctrls.get(c).pinRow();
                } else {
                    unitOf[c] = -1;
                }
            }
            double[] statesFlat = new double[chunkSize * n];
            for (int s = 0; s < chunkSize; s++) {
                double[] src = s < realCount && chunkResults[s] != null ? chunkResults[s] : baseState;
                System.arraycopy(src, 0, statesFlat, s * n, n);
            }
            double[] dxAtObs = GpuAcNewtonSolver.solveBatchSensitivities(handle, statesFlat,
                    java.util.Arrays.copyOf(unit, ku), obs);

            // ---- OLF decisions per pending (scenario, controller) ----
            boolean anyMoved = false;
            boolean[] affected = new boolean[chunkSize];
            int lastSlotLoaded = -1;
            DenseMatrix dx = new DenseMatrix(n, ku);
            for (int[] sc : pending) {
                int slot = sc[0];
                int c = sc[1];
                Controller ctl = ctrls.get(c);
                if (lastSlotLoaded != slot) {
                    esCtx.stateVector().set(java.util.Arrays.copyOf(chunkResults[slot], n));
                    for (int kk = 0; kk < ku; kk++) {         // sparse-refill the observed rows for this scenario
                        for (int mi = 0; mi < m; mi++) {
                            dx.set(obs[mi], kk, dxAtObs[slot * ku * m + kk * m + mi]);
                        }
                    }
                    lastSlotLoaded = slot;
                }
                double rawSensi = ctl.flowTerm().calculateSensi(dx, unitOf[c]);
                double diff = diffOf[slot][c];
                int[] holder = {tap[slot][c]};
                int dir = 0;
                switch (ctl.kind()) {
                    case KIND_RPC -> {
                        double deltaR1 = diff / rawSensi;    // OLF RPC: no sensitivity gate
                        dir = moveClosest(ctl, holder, deltaR1, ctl.maxTapShift(), ratchets[slot][c]);
                    }
                    case KIND_PHASE_P -> {
                        if (Math.abs(rawSensi * Math.toRadians(1)) > SENSI_EPS) {
                            double da = diff / rawSensi;     // per-degree RHS and toRadians cancel out
                            dir = moveClosest(ctl, holder, da, ctl.maxTapShift(), ratchets[slot][c]);
                        }
                    }
                    case KIND_PHASE_I -> {
                        if (Math.abs(rawSensi * Math.toRadians(1)) > SENSI_EPS) {
                            double da = diff / rawSensi;     // diff = target - i < 0 here
                            dir = moveExceed(ctl, holder, da, ratchets[slot][c]);
                        }
                    }
                    default -> throw new IllegalStateException();
                }
                if (dir != 0) {
                    tap[slot][c] = holder[0];
                    anyMoved = true;
                    affected[slot] = true;
                }
            }
            if (!anyMoved) {
                break;
            }
            // ---- apply moved taps: pinned-row targets + pi rows + state seeds, re-solve warm ----
            List<Integer> ovIdx = new ArrayList<>();
            List<Double> ovVal = new ArrayList<>();
            for (int i = 0; i < targetOverrideIndex.length; i++) {
                ovIdx.add(targetOverrideIndex[i]);
                ovVal.add(targetOverrideValue[i]);
            }
            GpuBatchTransformerControl.seedDsTargets(nds, dsPhiRow, dsSeed, chunkResults, realCount, n, ovIdx, ovVal);
            List<Integer> r1Slot = new ArrayList<>();
            List<Double> r1Val = new ArrayList<>();
            for (int v : baseR1OvSlotElem) {
                r1Slot.add(v);
            }
            for (double v : baseR1OvVal) {
                r1Val.add(v);
            }
            double[] x0Full = statesFlat;
            for (int s = 0; s < chunkSize; s++) {
                for (int c = 0; c < nc; c++) {
                    Controller ctl = ctrls.get(c);
                    if (tap[s][c] != ctl.baseTapIdx()) {
                        double[] row = ctl.tapRows()[tap[s][c]];
                        ovIdx.add(s * n + ctl.pinRow());
                        ovVal.add(row[ctl.pinTapCol()]);     // identity row pins rho/a1 at the tap's value
                        x0Full[s * n + ctl.pinRow()] = row[ctl.pinTapCol()];
                        r1Slot.add(s);
                        r1Slot.add(ctl.elemIndex());
                        for (double d : row) {
                            r1Val.add(d);                    // full 8-double pi row: per-step R/X/G/B follow
                        }
                    }
                }
            }
            double[][] r2 = GpuAcNewtonSolver.solveBatchContext(handle, baseState, baseTargets,
                    rowModePerScenario, baseRowMode, disabledMask,
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
                if (affected[s]) {                           // untouched scenarios KEEP their converged state —
                    chunkResults[s] = r2[s];                 // the CPU never re-runs Newton on a STABLE scenario,
                }                                            // and the batch re-solve's forced first iteration
            }                                                // would polish them off the CPU's point (~1e-5)
            if (nds > 0) {                                   // this solve may have distributed further
                GpuBatchTransformerControl.captureDsCarry(handle, chunkSize, realCount, n, dsPhiRow, dsFactor, baseTargets,
                        dsOvBus, dsOvFactorDelta, dsOvBaseDelta, chunkResults, dsAnchorCarry, dsSlackCarry, dsSeed, affected);
            }
        }
        return chunkResults;
    }
}
