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
 * The native side of the AutomationSystem driver phase (native/cuda/outerloop/automation_system.cu).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class AutomationSystemNative {

    static {
        GpuAcNewtonSolver.isAvailable();   // loads libolfgpu once
    }

    private AutomationSystemNative() {
    }

    /**
     * Install (or clear, with null systems) the run's overload management systems. Per system: its monitored closed-branch
     * element and side (GpuTwoSides), and the slice of its trippings ({@code trippingStart}, nSystems + 1). Per tripping:
     * the disabled-mask slot of the closed branch it operates, whether it opens it (1) or closes it (0), and its current
     * threshold (pu).
     */
    public static native void setBatchAutomationSystem(long handle, int[] monitoredElement, int[] monitoredSide, int[] trippingStart,
                                                       int[] operatedMaskSlot, byte[] openAction, double[] threshold);

    /** Per scenario and tripping ({@code s * nTrippings + t}), whether the last solve left the tripping's branch disabled (1)
     *  or enabled (0); null without systems. */
    public static native byte[] readBatchAutomationSystemOperatedDisabled(long handle);
}
