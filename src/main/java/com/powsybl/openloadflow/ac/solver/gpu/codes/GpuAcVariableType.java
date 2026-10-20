/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.codes;

import com.powsybl.openloadflow.ac.equations.AcVariableType;

/**
 * OLF's {@link AcVariableType}, by ordinal, for the variable types the device reads: the MAX_VOLTAGE_CHANGE step clamp
 * (varType, one per state row) clamps on the BUS_V and BUS_PHI steps only, as OLF's
 * {@code MaxVoltageChangeStateVectorScaling}. Every other ordinal is scaled with the step and never a clamp source.
 * Mirrored by native/cuda/common/codes/gpu_ac_variable_type.cuh (GpuCodesMirrorTest checks it equals OLF's ordinals).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuAcVariableType {

    /** Clamps the magnitude step to maxDv. */
    public static final int BUS_V = AcVariableType.BUS_V.ordinal();
    /** Clamps the angle step to maxDphi. */
    public static final int BUS_PHI = AcVariableType.BUS_PHI.ordinal();

    private GpuAcVariableType() {
    }
}
