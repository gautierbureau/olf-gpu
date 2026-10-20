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
 * The device's secondary voltage control loop (SecondaryVoltageControlOuterLoop).
 *
 * <p>JNI glue: {@code native/cuda/jni/jni/secondary_voltage_control_jni.cu}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class SecondaryVoltageControlNative {

    static {
        GpuAcNewtonSolver.isAvailable();   // loads libolfgpu once
    }

    private SecondaryVoltageControlNative() {
    }

    /** The zone-regulated generator voltage control targets the SecondaryVoltageControl phase settled
     *  on, per {@code (scenario, controller)} — what OLF's own loop leaves on the network. */
    public static native double[] readBatchSecondaryVoltageControlTargets(long handle);

    /** Install the SecondaryVoltageControl zones for the in-kernel phase (OLF order 4); see
     *  GpuSecondaryVoltageControlOuterLoop.uploadToDevice. Returns 1 when the kernel loop owns them — it refuses a zone
     *  set wider than the in-kernel dense system, or an observable on an element the batch does not pack. */
    @SuppressWarnings("java:S107")
    public static native int setBatchSecondaryVoltageControl(long handle,
            int[] controlledBusVRow, int[] busVRow, int[] reactiveLimitEntry, double[] baseTargetV,
            double[] minQ, double[] maxQ, double[] loadQ,
            int[] qPtr, int[] reactiveTermKind, int[] reactiveTermSide, int[] reactiveTermElement, int[] reactiveTermShuntBRow,
            int[] zoneStart, int[] zoneControllerBus, int[] zonePilotBusVRow, double[] zoneTargetV,
            double minPlausibleV, double maxPlausibleV);
}
