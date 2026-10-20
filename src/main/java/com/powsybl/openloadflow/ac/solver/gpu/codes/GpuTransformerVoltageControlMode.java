/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.codes;

import com.powsybl.openloadflow.OpenLoadFlowParameters;

/**
 * Which of OLF's discrete transformer voltage control loops the device runs (setBatchTransformerVoltageControl mode):
 * OLF's {@link OpenLoadFlowParameters.TransformerVoltageControlMode}, by ordinal. INCREMENTAL_VOLTAGE_CONTROL runs as its
 * own loop (GpuIncrementalTransformerVoltageControlOuterLoop), so the device reads only these two. Mirrored by
 * native/cuda/jni/common/codes/gpu_transformer_voltage_control_mode.cuh (GpuCodesMirrorTest checks it equals OLF's
 * ordinals).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuTransformerVoltageControlMode {

    /** SimpleTransformerVoltageControlOuterLoop: control on for the first solve, then rounded. */
    public static final int WITH_GENERATOR_VOLTAGE_CONTROL =
            OpenLoadFlowParameters.TransformerVoltageControlMode.WITH_GENERATOR_VOLTAGE_CONTROL.ordinal();
    /** TransformerVoltageControlOuterLoop: INITIAL, CONTROL, COMPLETE. */
    public static final int AFTER_GENERATOR_VOLTAGE_CONTROL =
            OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL.ordinal();

    private GpuTransformerVoltageControlMode() {
    }
}
