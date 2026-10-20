/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.codes;

/**
 * Which OLF equation a stable row holds in a lane (docs/gpu-vs-olf.md, section 2.2): OLF activates and deactivates
 * equations; the device's rows are permanent, so "OLF activates X" becomes "the row now holds X". Mirrored by
 * native/cuda/common/codes/gpu_row_mode.cuh.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuRowMode {

    /** {@code x = target}, a fixed variable: BUS_TARGET_PHI, BUS_TARGET_V of a PV bus, DUMMY_TARGET_P/Q, a fixed tap. */
    public static final int FIXED_VARIABLE = 0;
    /** The row's ordinary equation: BUS_TARGET_P, BUS_TARGET_Q, DISTR_Q, ZERO_V, ZERO_PHI... */
    public static final int EQUATION = 1;
    /** BUS_TARGET_V with the slope terms (a static var compensator with a voltage slope). */
    public static final int BUS_TARGET_V_WITH_SLOPE = 2;
    /** BRANCH_TARGET_P of the controlled branch, on a phase controller's a1 row (PhaseControl, CONTROLLER mode). */
    public static final int BRANCH_TARGET_P = 3;
    /** The bus's own BUS_TARGET_Q: after a PV -> PQ switch (target = the limit), or a remote member's own Q balance. */
    public static final int BUS_TARGET_Q = 4;
    /** Device only: a fixed variable written on another row (the cross identity, docs/gpu-vs-olf.md section 2.4). */
    public static final int CROSS_FIXED_VARIABLE = 6;

    private GpuRowMode() {
    }
}
