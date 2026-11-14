/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.codes;

/**
 * A sensitivity factor's function as the device contracts it ({@code SensitivityBasisRequest.Contraction#factorFunction}):
 * a closed branch's active power, reactive power or current magnitude on side 1 or 2, or a bus voltage. Mirrored by
 * native/cuda/common/codes/gpu_sensitivity_function.cuh.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuSensitivityFunction {

    public static final int ACTIVE_POWER_1 = 0;
    public static final int REACTIVE_POWER_1 = 1;
    public static final int ACTIVE_POWER_2 = 2;
    public static final int REACTIVE_POWER_2 = 3;
    public static final int CURRENT_1 = 4;
    public static final int CURRENT_2 = 5;
    public static final int BUS_VOLTAGE = 6;

    private GpuSensitivityFunction() {
    }
}
