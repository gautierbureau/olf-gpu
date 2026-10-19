/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

/**
 * Shared constants of the batched N-1 path: the per-scenario convergence status codes
 * ({@link BatchScenarioResult#convergenceStatus()}) and the separator embedded in a
 * (contingency, strategy) pair's {@link BatchContingency} id.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuBatchStatus {

    /** Convergence status: the scenario converged within the tolerance. */
    public static final int STATUS_CONVERGED = 0;
    /** Convergence status: the scenario hit the iteration cap without converging. */
    public static final int STATUS_MAX_ITERATION = 1;
    /** Convergence status: the scenario diverged (NaN/Inf mismatch). */
    public static final int STATUS_DIVERGED = -1;
    /**
     * Convergence status: the contingency was not batched on the GPU (it fragments the
     * network or its branch is absent from the GPU model) and must be solved by the
     * sequential CPU security-analysis path. The state is the base case (unchanged).
     */
    public static final int STATUS_REJECTED = -2;
    /** The device's first linear solve did not solve the system (residual ratio above 1e-2: the
     *  structure does not factor in the uniform batch). Routed like a rejection, but it is the ONE
     *  reject a pivoted retry can fix, which is why it carries its own code: the batch-of-one retries
     *  in the robust single-matrix mode before any CPU fallback; every other reject goes to the CPU. */
    public static final int STATUS_LINEAR_SOLVE_FAILED = -3;

    private GpuBatchStatus() {
    }
}
