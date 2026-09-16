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
public final class GpuBatchIncrementalFlowControl {

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

    public record Controller(int kind, int pinRow, int pinTapCol, EquationTerm<AcVariableType, AcEquationType> flowTerm,
                      double targetValue, double halfDeadband, double[][] tapRows, int baseTapIdx,
                      int elemIndex, int controlledElemIndex, int maxTapShift, int controlledSide, int branchNum) {
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
    public record EsContext(StateVector stateVector, int n) {
    }

    @SuppressWarnings("unchecked")
    public static List<Controller> buildControllers(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
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
                        elemIndex, ctlElemIndex, rpcMaxTapShift, rc.getControlledSide() == TwoSides.ONE ? 1 : 2, controller.getNum()));
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
                        elemIndex, ctlElemIndex, Integer.MAX_VALUE, pc.getControlledSide() == TwoSides.ONE ? 1 : 2, controller.getNum()));
            }
        }
        return out;
    }

    /** Install the controllers for the in-kernel loops (RPC at OLF order 9, incremental phase at order 7).
     *  Returns true when the kernel loops OWN them, false when there are none or one controlled branch is
     *  not a closed branch (the host orchestrator then runs). */
    public static boolean uploadToDevice(long handle, List<Controller> ctrls) {
        if (ctrls == null || ctrls.isEmpty()) {
            com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.setBatchFlowControllers(handle,
                    null, null, null, null, null, null, null, null, null, null, null, null);
            return false;
        }
        int nc = ctrls.size();
        int[] kind = new int[nc];
        int[] pinRow = new int[nc];
        int[] pinCol = new int[nc];
        int[] side = new int[nc];
        double[] targetValue = new double[nc];
        double[] halfDb = new double[nc];
        int[] elem = new int[nc];
        int[] ctlElem = new int[nc];
        int[] tapPtr = new int[nc + 1];
        int[] baseTap = new int[nc];
        int[] maxShift = new int[nc];
        List<Double> tapData = new ArrayList<>();
        for (int c = 0; c < nc; c++) {
            Controller ctl = ctrls.get(c);
            kind[c] = ctl.kind();
            pinRow[c] = ctl.pinRow();
            pinCol[c] = ctl.pinTapCol();
            side[c] = ctl.controlledSide();
            targetValue[c] = ctl.targetValue();
            halfDb[c] = ctl.halfDeadband();
            elem[c] = ctl.elemIndex();
            ctlElem[c] = ctl.controlledElemIndex();
            tapPtr[c] = tapData.size() / 8;
            for (double[] row : ctl.tapRows()) {
                for (double d : row) {
                    tapData.add(d);
                }
            }
            baseTap[c] = ctl.baseTapIdx();
            maxShift[c] = ctl.maxTapShift() == Integer.MAX_VALUE ? 1 << 20 : ctl.maxTapShift();
        }
        tapPtr[nc] = tapData.size() / 8;
        boolean owned = com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.setBatchFlowControllers(handle,
                kind, pinRow, pinCol, side, targetValue, halfDb, elem, ctlElem, tapPtr,
                tapData.stream().mapToDouble(Double::doubleValue).toArray(), baseTap, maxShift) != 0;
        LOGGER.info("Uploaded {} flow-observed incremental controllers to device: {}", nc, owned ? "in-kernel loops" : "host orchestrator");
        if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
            System.err.printf("FLOW_UPLOAD nCtrl=%d kernel=%b%n", nc, owned);
        }
        return owned;
    }

    /** Give the driver's final taps (scenario 0) back to the network through the pi model. */
    public static void writeBackTaps(LfNetwork network, List<Controller> ctrls, int[] taps) {
        if (ctrls == null || ctrls.isEmpty() || taps == null) {
            return;
        }
        for (int c = 0; c < ctrls.size() && c < taps.length; c++) {
            Controller ctl = ctrls.get(c);
            if (taps[c] == ctl.baseTapIdx()) {
                continue;
            }
            LfBranch branch = network.getBranch(ctl.branchNum());
            if (branch != null) {
                var pm = branch.getPiModel();
                pm.setTapPosition(pm.getTapPositionRange().getMinimum() + taps[c]);
            }
        }
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




    private static boolean descending(Controller c, int n1, int n2, double shift) {
        return shift > 0 ? c.pin(n1) < c.pin(n2) : c.pin(n1) > c.pin(n2);
    }

}
