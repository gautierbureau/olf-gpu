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
 * The device outer-loop driver: the run's own outer loops.
 *
 * <p>JNI glue: {@code native/cuda/jni/driver_jni.cu}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class DriverNative {

    static {
        GpuAcNewtonSolver.isAvailable();   // loads libolfgpu once
    }

    private DriverNative() {
    }

    /** The run's own outer loops: does its loop list hold OLF's DistributedSlackOuterLoop, its ReactiveLimitsOuterLoop?
     *  The driver installs the DISTRIBUTED_SLACK / REACTIVE_LIMITS phases only for a loop the run has (and a table to
     *  drive it), never from the packed tables alone. A context never told runs both, as far as its tables go. */
    public static native void setBatchOuterLoops(long handle, boolean distributedSlackOuterLoop, boolean reactiveLimitsOuterLoop);
}
