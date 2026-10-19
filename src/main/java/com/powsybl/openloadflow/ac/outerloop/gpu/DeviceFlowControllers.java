/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * The controller table of the FLOW-OBSERVED incremental loops — OLF's
 * {@code IncrementalTransformerReactivePowerControlOuterLoop} (rho taps driving a branch Q flow) and
 * {@code AcIncrementalPhaseControlOuterLoop} (a1 taps driving a branch P flow in CONTROLLER mode, or
 * capping a branch current in LIMITER mode). {@link GpuIncrementalTransformerReactivePowerControlOuterLoop} and
 * {@link GpuIncrementalPhaseControlOuterLoop} build the controllers, this class uploads them as one table
 * (reactive power controllers first) and writes the final taps back; the
 * loops run in the kernel (incremental_flow_control.cu): batched sensitivities with a unit RHS at each
 * controller's pinned identity row (BRANCH_TARGET_RHO1 / BRANCH_TARGET_ALPHA1), the controlled branch
 * flow and its gradient through the codegen'd closed-branch terms (flow_observation.cuh), OLF's tap
 * decisions and its direction ratchet.
 *
 * <p>Phase sensitivities: OLF's RHS is {@code toRadians(1)} (per-degree); with the device's unit RHS
 * the tap shift collapses to {@code da = diff / rawSensi} (radians), and the {@code SENSI_EPS} gate
 * (in incremental_flow_control.cu) applies to the per-degree value ({@code rawSensi * toRadians(1)}).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class DeviceFlowControllers {

    private static final Logger LOGGER = LoggerFactory.getLogger(DeviceFlowControllers.class);

    private DeviceFlowControllers() {
    }

    public record Controller(int kind, int pinRow, int pinTapCol, double targetValue, double halfDeadband, double[][] tapRows, int baseTapIdx,
                      int elemIndex, int controlledElemIndex, int maxTapShift, int controlledSide, int branchNum) {
    }

    /** Install the controllers for the in-kernel loops (RPC at OLF order 9, incremental phase at order 7).
     *  Returns true when the kernel loops OWN them, false when there are none or one controlled branch is
     *  not a closed branch (the host orchestrator then runs). */
    public static boolean uploadToDevice(long handle, List<Controller> ctrls) {
        if (ctrls == null || ctrls.isEmpty()) {
            PhaseControlNative.setBatchFlowControllers(handle,
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
        boolean owned = PhaseControlNative.setBatchFlowControllers(handle,
                kind, pinRow, pinCol, side, targetValue, halfDb, elem, ctlElem, tapPtr,
                tapData.stream().mapToDouble(Double::doubleValue).toArray(), baseTap, maxShift) != 0;
        LOGGER.info("Uploaded {} flow-observed incremental controllers to device: {}", nc, owned ? "in-kernel loops" : "host orchestrator");
        if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
            System.err.printf("FLOW_UPLOAD nCtrl=%d kernel=%b%n", nc, owned);
        }
        return owned;
    }

    /** Give the driver's final taps (scenario 0) back to the network through the pi model. */
    public static void applyToNetwork(LfNetwork network, List<Controller> ctrls, int[] taps) {
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

    static int closestTap(double[][] tapRows, int col, double value) {
        int best = 0;
        for (int t = 1; t < tapRows.length; t++) {
            if (Math.abs(tapRows[t][col] - value) < Math.abs(tapRows[best][col] - value)) {
                best = t;
            }
        }
        return best;
    }

}
