/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.codes;

/**
 * Which loop OLF lets repair an unrealistic voltage before its realistic-voltage check fails the run
 * ({@code AcLoadFlowEngine.runOuterLoops}: the last configured loop whose {@code canFixUnrealisticState()} is true,
 * under voltageRemoteControlRobustMode). Mirrored by native/cuda/common/codes/gpu_realistic_voltage_fixing_loop.cuh.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuRealisticVoltageFixingLoop {

    /** No fixing loop (robust mode off, or none configured): every solve is checked. */
    public static final int NONE = -1;
    public static final int REACTIVE_LIMITS = 0;
    /** The discrete TransformerVoltageControl. */
    public static final int TRANSFORMER_VOLTAGE_CONTROL = 1;

    private GpuRealisticVoltageFixingLoop() {
    }
}
