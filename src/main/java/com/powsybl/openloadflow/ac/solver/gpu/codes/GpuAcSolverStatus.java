/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.codes;

/**
 * A scenario's status at the end of a device solve ({@code BatchScenarioResult#convergenceStatus()}, the
 * {@code n + 2} column of the solve result). The first two are OLF's {@code AcSolverStatus}; the three negative ones
 * are the device's (docs/gpu-vs-olf.md, section 6). Mirrored by native/cuda/jni/common/codes/gpu_ac_solver_status.cuh.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuAcSolverStatus {

    /** OLF's CONVERGED: the scenario converged within the tolerance. */
    public static final int CONVERGED = 0;
    /** OLF's MAX_ITERATION_REACHED: the scenario hit the iteration cap without converging. */
    public static final int MAX_ITERATION_REACHED = 1;
    /** The scenario diverged (NaN/Inf mismatch). */
    public static final int DIVERGED = -1;
    /**
     * The scenario was not solved on the device (a shape it does not model, a budget exhausted, a robust-mode
     * unrealistic voltage) and the sequential CPU security analysis replays it. The state is the base case.
     */
    public static final int REJECTED = -2;
    /** The device's first linear solve did not solve the system (residual ratio above 1e-2: the
     *  structure does not factor in the uniform batch). Routed like a rejection, but it is the ONE
     *  reject a pivoted retry can fix, which is why it carries its own code: the batch-of-one retries
     *  in the robust single-matrix mode before any CPU fallback; every other reject goes to the CPU.
     *  The closest OLF notion is SOLVER_FAILED (the factorization throws). */
    public static final int LINEAR_SOLVE_FAILED = -3;

    private GpuAcSolverStatus() {
    }
}
