/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;

/**
 * The device outer-loop driver: host-owned loop slots, pause and resume.
 *
 * <p>JNI glue: {@code native/cuda/jni/jni/driver_jni.cu}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class DriverNative {

    static {
        GpuAcNewtonSolver.isAvailable();   // loads libolfgpu once
    }

    private DriverNative() {
    }

    /** Drop a paused driver sequence without resuming it (the host loop changed the equation layout
     *  or the row modes): the next solveBatchContext starts over on the re-extracted system. */
    public static native void cancelBatchDriverPause(long handle);

    /** Did the last solve stop at a host-owned slot? Returns {@code GpuOuterLoopKind.NOT_PAUSED} if not paused, or the
     *  {@code GpuOuterLoopKind.HOST_*} constant identifying WHICH host loop was reached. */
    public static native int readBatchDriverPaused(long handle);

    /** Resume the paused driver; {@code hostChanged[s] != 0} is the host loop's UNSTABLE verdict. */
    public static native void setBatchDriverResume(long handle, byte[] hostChanged);

    /** The next fresh solve resumes its sweep at this LOOP_HOST_* kind after its first solve — the
     *  restarted sequence re-checks the loop that went UNSTABLE, as OLF's runOuterLoop does. */
    public static native void setBatchDriverStartLoop(long handle, int kind);

    /** Enable HOST-owned outer-loop slots via a bitmask.
     *  Bit 0 = SecondaryVoltageControl (OLF order 4, pre-RL),
     *  Bit 1 = PhaseControl (OLF order 7, post-RL),
     *  Bit 2 = TransformerVoltageControl (OLF order 8),
     *  Bit 3 = TransformerReactivePowerControl (OLF order 9),
     *  Bit 4 = ShuntVoltageControl (OLF order 10).
     *  0 disables all; the driver behaves exactly as before. */
    public static native void setBatchHostLoop(long handle, int mask);
}
