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
 * The device's shunt voltage control loops, incremental and discrete.
 *
 * <p>JNI glue: {@code native/cuda/jni/jni/shunt_voltage_control_jni.cu}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class ShuntVoltageControlNative {

    static {
        GpuAcNewtonSolver.isAvailable();   // loads libolfgpu once
    }

    private ShuntVoltageControlNative() {
    }

    /** Per (scenario, sub-controller): the dispatched section, layout {@code s * nSubs + u}. */
    public static native int[] readBatchDiscreteShuntPositions(long handle);

    /** Per (scenario, member shunt): {state (0 off, 1 continuous, 2 dispatched), the continuous b the
     *  dispatch started from}, layout {@code (s * nMembers + m) * 2} — null without the table. */
    public static native double[] readBatchDiscreteShuntState(long handle);

    /** Per (scenario, sub-controller): the section the driver's ShuntVoltageControl phase left —
     *  null without the table. Layout {@code s * nSubs + u}, subs in upload order. */
    public static native int[] readBatchShuntPositions(long handle);

    /** Install the WITH_GENERATOR shunt voltage-control tables for the in-kernel ShuntVoltageControl phase
     *  (OLF order 10); see GpuShuntVoltageControlOuterLoop.uploadToDevice. Returns 1 when the kernel loop owns them. */
    @SuppressWarnings("java:S107")
    public static native int setBatchDiscreteShunt(long handle, int entryOn,
            int[] grpPtr, int[] deltaPtr, int[] deltaRow, int[] deltaModeOn, int[] deltaModeOff,
            double[] deltaTargetOn, double[] deltaTargetOff,
            int[] memBRow, int[] memElem, int[] memSubPtr, int[] secPtr, double[] secB, int[] subMin, int[] subBase);

    /** Install the incremental SHUNT voltage-control tables for the in-kernel loop (OLF order 10);
     *  see GpuIncrementalShuntVoltageControlOuterLoop.uploadToDevice. Returns 1 when the kernel loop owns them. */
    public static native int setBatchShuntControllers(long handle,
            int[] bRow, int[] vRow, double[] targetV, double[] halfDb, int[] elemAbs, int[] shOvElem,
            int[] groupPtr, int[] groupCtrl, int[] subPtr, int[] secPtr, double[] secB, double[] secG,
            int[] subMin, int[] subBase, int maxSectionShift);
}
