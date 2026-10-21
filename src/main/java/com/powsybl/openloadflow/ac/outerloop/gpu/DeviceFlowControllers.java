/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuPiModelColumn;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.util.gpu.GpuTrace;
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

    public record Controller(int kind, int pinRow, int tapColumn, double targetValue, double halfDeadband, double[][] tapRows, int baseTapIndex,
                      int elemIndex, int controlledElementIndex, int maxTapShift, int controlledSide, int branchNum) {
    }

    /** Install the controllers for the in-kernel loops (RPC at OLF order 9, incremental phase at order 7).
     *  Returns true when the kernel loops OWN them, false when there are none or one controlled branch is
     *  not a closed branch (the host orchestrator then runs). */
    public static boolean uploadToDevice(long handle, List<Controller> controllers) {
        if (controllers == null || controllers.isEmpty()) {
            PhaseControlNative.setBatchFlowControllers(handle,
                    null, null, null, null, null, null, null, null, null, null, null, null);
            return false;
        }
        int nControllers = controllers.size();
        int[] kind = new int[nControllers];
        int[] pinRow = new int[nControllers];
        int[] tapColumn = new int[nControllers];
        int[] side = new int[nControllers];
        double[] targetValue = new double[nControllers];
        double[] halfTargetDeadband = new double[nControllers];
        int[] elem = new int[nControllers];
        int[] controlledElement = new int[nControllers];
        int[] tapStart = new int[nControllers + 1];
        int[] baseTap = new int[nControllers];
        int[] maxShift = new int[nControllers];
        List<Double> tapRows = new ArrayList<>();
        for (int c = 0; c < nControllers; c++) {
            Controller controller = controllers.get(c);
            kind[c] = controller.kind();
            pinRow[c] = controller.pinRow();
            tapColumn[c] = controller.tapColumn();
            side[c] = controller.controlledSide();
            targetValue[c] = controller.targetValue();
            halfTargetDeadband[c] = controller.halfDeadband();
            elem[c] = controller.elemIndex();
            controlledElement[c] = controller.controlledElementIndex();
            tapStart[c] = tapRows.size() / GpuPiModelColumn.COUNT;
            for (double[] row : controller.tapRows()) {
                for (double d : row) {
                    tapRows.add(d);
                }
            }
            baseTap[c] = controller.baseTapIndex();
            maxShift[c] = controller.maxTapShift() == Integer.MAX_VALUE ? 1 << 20 : controller.maxTapShift();
        }
        tapStart[nControllers] = tapRows.size() / GpuPiModelColumn.COUNT;
        boolean owned = PhaseControlNative.setBatchFlowControllers(handle,
                kind, pinRow, tapColumn, side, targetValue, halfTargetDeadband, elem, controlledElement, tapStart,
                tapRows.stream().mapToDouble(Double::doubleValue).toArray(), baseTap, maxShift) != 0;
        LOGGER.info("Uploaded {} flow-observed incremental controllers to device: {}", nControllers, owned ? "in-kernel loops" : "host orchestrator");
        if (GpuTrace.BATCH_TRACE.isOn()) {
            System.err.printf("FLOW_UPLOAD nControllers=%d kernel=%b%n", nControllers, owned);
        }
        return owned;
    }

    /** Give the driver's final taps (scenario 0) back to the network through the pi model. */
    public static void applyToNetwork(LfNetwork network, List<Controller> controllers, int[] taps) {
        if (controllers == null || controllers.isEmpty() || taps == null) {
            return;
        }
        for (int c = 0; c < controllers.size() && c < taps.length; c++) {
            Controller controller = controllers.get(c);
            if (taps[c] == controller.baseTapIndex()) {
                continue;
            }
            LfBranch branch = network.getBranch(controller.branchNum());
            if (branch != null) {
                var piModel = branch.getPiModel();
                piModel.setTapPosition(piModel.getTapPositionRange().getMinimum() + taps[c]);
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
