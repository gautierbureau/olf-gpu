/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;

/**
 * The device's phase control loops (incremental flow controls and the continuous phase control).
 *
 * <p>JNI glue: {@code native/cuda/jni/jni/phase_control_jni.cu}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class PhaseControlNative {

    static {
        GpuAcNewtonSolver.isAvailable();   // loads libolfgpu once
    }

    private PhaseControlNative() {
    }

    /** Per (scenario, continuous phase controller): the tap the driver left — null without the table. */
    public static native int[] readBatchContinuousPhaseTaps(long handle);

    /** Per (scenario, flow controller): the tap the driver left — null without the table. */
    public static native int[] readBatchFlowTaps(long handle);

    /** Install the CONTINUOUS phase controllers for the in-kernel PhaseControl phase (OLF order 7);
     *  see GpuPhaseControlOuterLoop.uploadToDevice. Returns 1 when the kernel loop owns them. */
    public static native int setBatchContinuousPhase(long handle, int[] kind, int[] a1Row, int[] side, double[] limit,
            int[] elem, int[] tapStart, double[] tapRows, int[] baseTap);

    /** Install the FLOW-OBSERVED incremental controllers (RPC / incremental phase) for the in-kernel
     *  loops (OLF orders 7 and 9); see DeviceFlowControllers.uploadToDevice. Returns 1 when
     *  the kernel loops own them. */
    public static native int setBatchFlowControllers(long handle, int[] kind, int[] pinRow, int[] tapColumn, int[] side,
            double[] targetValue, double[] halfTargetDeadband, int[] elem, int[] controlledElement, int[] tapStart, double[] tapRows,
            int[] baseTap, int[] maxShift);
}
