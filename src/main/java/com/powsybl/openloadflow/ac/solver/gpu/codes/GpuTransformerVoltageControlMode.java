/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.codes;

/**
 * Which of OLF's discrete transformer voltage control loops the device runs (setBatchDiscreteTvc mode), named after
 * OLF's {@code TransformerVoltageControlMode} constants. The device numbering is its own (OLF's ordinals put
 * WITH_GENERATOR first); INCREMENTAL_VOLTAGE_CONTROL runs as its own loop. Mirrored by
 * native/cuda/jni/common/codes/gpu_transformer_voltage_control_mode.cuh.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuTransformerVoltageControlMode {

    /** TransformerVoltageControlOuterLoop: INITIAL, CONTROL, COMPLETE. */
    public static final int AFTER_GENERATOR_VOLTAGE_CONTROL = 0;
    /** SimpleTransformerVoltageControlOuterLoop: control on for the first solve, then rounded. */
    public static final int WITH_GENERATOR_VOLTAGE_CONTROL = 1;

    private GpuTransformerVoltageControlMode() {
    }
}
