/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.codes;

import com.powsybl.openloadflow.ac.solver.StateVectorScalingMode;

/**
 * OLF's {@link StateVectorScalingMode} as the device receives it ({@code stateVectorScalingMode} in the solve calls). Mirrored by
 * native/cuda/jni/common/codes/gpu_state_vector_scaling_mode.cuh.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuStateVectorScalingMode {

    /** Undamped Newton step. */
    public static final int NONE = 0;
    /** Per-scenario backtracking on the device (OLF's LineSearchStateVectorScaling). */
    public static final int LINE_SEARCH = 1;
    /** Per-scenario step clamp to maxDv / maxDphi (OLF's MaxVoltageChangeStateVectorScaling). */
    public static final int MAX_VOLTAGE_CHANGE = 2;

    private GpuStateVectorScalingMode() {
    }

    public static int of(StateVectorScalingMode mode) {
        return switch (mode) {
            case NONE -> NONE;
            case LINE_SEARCH -> LINE_SEARCH;
            case MAX_VOLTAGE_CHANGE -> MAX_VOLTAGE_CHANGE;
        };
    }
}
