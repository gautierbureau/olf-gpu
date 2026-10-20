/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.codes;

/**
 * The variable type of each state row for the MAX_VOLTAGE_CHANGE step clamp (varType), as OLF's
 * {@code MaxVoltageChangeStateVectorScaling} sees it: it clamps on the BUS_V and BUS_PHI steps only. Device
 * numbering, named after OLF's {@code AcVariableType}. Mirrored by
 * native/cuda/jni/common/codes/gpu_step_clamp_variable.cuh.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuStepClampVariable {

    /** Clamps the angle step to maxDphi. */
    public static final int BUS_PHI = 0;
    /** Clamps the magnitude step to maxDv. */
    public static final int BUS_V = 1;
    /** Any other variable (BRANCH_RHO1, ...): scaled with the step, never a clamp source. */
    public static final int OTHER = 2;

    private GpuStepClampVariable() {
    }
}
