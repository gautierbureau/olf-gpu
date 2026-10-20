/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;

/**
 * The transposed solve on a batch context's factors.
 *
 * <p>JNI glue: {@code native/cuda/jni/transposed_solve_jni.cu}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class TransposedSolveNative {

    static {
        GpuAcNewtonSolver.isAvailable();   // loads libolfgpu once
    }

    private TransposedSolveNative() {
    }

    /**
     * Solves {@code Aᵀ x = b} for an explicit CSR matrix {@code A}, by building the transpose and
     * factorizing it. Validation / measurement entry for the batched-adjoint work (docs/tasks.md #40
     * step 1a) — NOT the production path.
     *
     * <p>cuDSS through 0.8.0 has no transpose solve (verified: its header carries no transpose/OP_T
     * symbol), so one GPU factorization yields ONE direction, where KLU's {@code klu_solve} /
     * {@code klu_tsolve} pair yields two from the same factors. Our Jacobian is stored as TRUE CSR
     * (row = equation, col = variable) so the plain solve serves the forward Newton; the adjoint
     * ({@code Jᵀλ = …}) is therefore the direction the forward's factors cannot provide, and until
     * cuDSS ships a transpose solve we buy it with a second factorization of an explicit transpose.
     *
     * <p>Deliberately self-contained (own handle/config/data, no Newton context) so a failure here
     * can only mean the CSR transpose or the cuDSS factorize/solve of {@code Aᵀ} is wrong — never the
     * adjoint's RHS assembly or contraction.
     *
     * @param n       matrix dimension
     * @param rowPtr  CSR row pointers of {@code A}, length {@code n+1}
     * @param colIdx  CSR column indices of {@code A}, length {@code rowPtr[n]}
     * @param vals    CSR values of {@code A}, length {@code rowPtr[n]}
     * @param b       right-hand side, length {@code n}
     * @return {@code x} such that {@code Aᵀ x = b}
     */
    public static native double[] solveTransposedExplicit(int n, int[] rowPtr, int[] colIdx,
                                                          double[] vals, double[] b);
}
