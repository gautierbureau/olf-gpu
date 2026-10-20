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
    public static native int[] readBatchShuntVoltageControlPositions(long handle);

    /** Per (scenario, member shunt): {state (0 off, 1 continuous, 2 dispatched), the continuous b the
     *  dispatch started from}, layout {@code (s * nMembers + m) * 2} — null without the table. */
    public static native double[] readBatchShuntVoltageControlState(long handle);

    /** Per (scenario, sub-controller): the section the driver's ShuntVoltageControl phase left —
     *  null without the table. Layout {@code s * nSubs + u}, subs in upload order. */
    public static native int[] readBatchIncrementalShuntVoltageControlPositions(long handle);

    /** Install the WITH_GENERATOR shunt voltage-control tables for the in-kernel ShuntVoltageControl phase
     *  (OLF order 10); see GpuShuntVoltageControlOuterLoop.uploadToDevice. Returns 1 when the kernel loop owns them. */
    @SuppressWarnings("java:S107")
    public static native int setBatchShuntVoltageControl(long handle, int entryOn,
            int[] groupMemberStart, int[] deltaStart, int[] deltaRow, int[] deltaModeOn, int[] deltaModeOff,
            double[] deltaTargetOn, double[] deltaTargetOff,
            int[] memberShuntBRow, int[] memberElement, int[] memberShuntCompensatorStart, int[] sectionStart,
            double[] sectionB, int[] shuntCompensatorMinPosition, int[] shuntCompensatorBasePosition);

    /** Install the incremental SHUNT voltage-control tables for the in-kernel loop (OLF order 10);
     *  see GpuIncrementalShuntVoltageControlOuterLoop.uploadToDevice. Returns 1 when the kernel loop owns them. */
    public static native int setBatchIncrementalShuntVoltageControl(long handle,
            int[] bRow, int[] vRow, double[] targetV, double[] halfTargetDeadband, int[] controllerElement, int[] shuntAdmittanceOverrideElement,
            int[] groupControllerStart, int[] groupController, int[] shuntCompensatorStart, int[] sectionStart, double[] sectionB, double[] sectionG,
            int[] shuntCompensatorMinPosition, int[] shuntCompensatorBasePosition, int maxSectionShift);
}
