/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.codes;

/**
 * OLF's {@code TransformerPhaseControl.Mode} of a continuous phase control (setBatchContinuousPhase kind[]), with
 * OLF's ordinals. Mirrored by native/cuda/common/codes/gpu_phase_control_mode.cuh.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuPhaseControlMode {

    /** Controls the active power flow: a1 is a variable of the solve, then discretised. */
    public static final int CONTROLLER = 0;
    /** Limits the current: a1 moves tap by tap while the current exceeds its target. */
    public static final int LIMITER = 1;

    private GpuPhaseControlMode() {
    }
}
