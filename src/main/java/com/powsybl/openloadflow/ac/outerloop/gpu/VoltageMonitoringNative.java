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
 * The device's voltage monitoring loop (MonitoringVoltageOuterLoop).
 *
 * <p>JNI glue: {@code native/cuda/jni/jni/voltage_monitoring_jni.cu}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class VoltageMonitoringNative {

    static {
        GpuAcNewtonSolver.isAvailable();   // loads libolfgpu once
    }

    private VoltageMonitoringNative() {
    }

    /** How many (controller, scenario) monitoring automatons fired in the last solve. */
    public static native int readBatchMonitoringActivations(long handle);

    /** Per (scenario, monitoring controller): the target the driver's MonitoringVoltage phase activated
     *  (pu), NaN while standing by — null without the table. Layout {@code s * nMonitoringVoltageControllers + c}. */
    public static native double[] readBatchMonitoringTargets(long handle);

    /**
     * MonitoringVoltage controller table (OLF outer-loop order 6) for the DRIVER phase: per
     * controller its monitored V row and {@code band = {lowThreshold, highThreshold, lowTargetV,
     * highTargetV}}, plus the CSR list ({@code deltaStart}) of (row, mode, target) writes its
     * activation applies. A null/empty {@code vRow} clears the table.
     */
    public static native void setBatchMonitoringLoops(long handle, int[] vRow, double[] band,
                                                      int[] deltaStart, int[] deltaRow,
                                                      int[] deltaMode, double[] deltaTarget);
}
