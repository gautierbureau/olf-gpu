/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.codes;

/**
 * A reactive-limit entry's state as one int (reactiveLimitsSeedState / reactiveLimitsFinalState): its {@link GpuQLimitType} in the low bits and
 * its PV -> PQ switch count above, the count OLF's ReactiveLimitsOuterLoop keeps per bus (pvPqSwitchCount) and compares
 * with maxPqPvSwitch. Mirrored by native/cuda/common/codes/gpu_reactive_limits_state.cuh.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuReactiveLimitsState {

    public static final int LIMIT_TYPE_MASK = 7;
    public static final int PV_PQ_SWITCH_COUNT_SHIFT = 3;

    private GpuReactiveLimitsState() {
    }

    /** The entry's {@link GpuQLimitType}. */
    public static int limitType(int state) {
        return state & LIMIT_TYPE_MASK;
    }

    /** How many times the entry switched PV -> PQ (OLF's pvPqSwitchCount). */
    public static int pvPqSwitchCount(int state) {
        return state >> PV_PQ_SWITCH_COUNT_SHIFT;
    }
}
