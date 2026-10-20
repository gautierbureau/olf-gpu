/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.codes;

/**
 * Where a member of a discrete voltage control stands, per scenario: OLF's TransformerVoltageControlOuterLoop
 * (AFTER_GENERATOR) / SimpleTransformerVoltageControlOuterLoop and ShuntVoltageControlOuterLoop (WITH_GENERATOR)
 * first solve the ratio or susceptance continuously, then round it to a tap or section and disable the control
 * (readBatchTransformerVoltageControlState / readBatchShuntVoltageControlState). Mirrored by
 * native/cuda/jni/common/codes/gpu_discrete_control_member_state.cuh.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuDiscreteControlMemberState {

    /** The member's voltage control is disabled. */
    public static final int DISABLED = 0;
    /** The ratio or susceptance is solved continuously. */
    public static final int CONTINUOUS = 1;
    /** Rounded to a tap or section and pinned there. */
    public static final int ROUNDED = 2;

    private GpuDiscreteControlMemberState() {
    }
}
