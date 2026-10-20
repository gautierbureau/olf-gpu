/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.codes;

/**
 * How a batch context factors its matrices (the mode of createBatchContext). Device only: OLF factors one KLU matrix
 * per solve. Mirrored by native/cuda/common/codes/gpu_linear_solve_mode.cuh.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuLinearSolveMode {

    /**
     * UBATCH: a single per-block matrix + CUDSS_CONFIG_UBATCH_SIZE. Its one-block ANALYSIS is cheap (~150 ms), and its
     * per-launch fixed cost amortizes with the batch (pegase9241, warm ms/scenario: chunk 32 = 17.8, 125 = 6.35,
     * 500 = 4.87), hence the memory auto-sizing of the chunk.
     */
    public static final int UBATCH = 3;
    /** The ROBUST single-matrix solve: a per-scenario loop with BTF-COLAMD ordering, global column pivoting and
     *  matching (cuDSS single-matrix API). What the batch-of-one retries in after a linear-solve reject: rte6515's
     *  full busbar structure does not factor in the uniform batch. */
    public static final int ROBUST = 6;

    private GpuLinearSolveMode() {
    }
}
