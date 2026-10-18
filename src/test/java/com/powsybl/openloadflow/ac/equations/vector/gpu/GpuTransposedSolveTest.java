/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.openloadflow.ac.solver.gpu.TransposedSolveNative;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Step 1a of the batched-adjoint work (docs/tasks.md #40): prove the TRANSPOSE machinery alone,
 * before any cotangent assembly or contraction exists to confuse a failure.
 *
 * <p>Why a transpose solve needs building at all: cuDSS through 0.8.0 has no transpose solve, so one
 * GPU factorization yields ONE direction — whereas KLU's {@code klu_solve}/{@code klu_tsolve} pair
 * yields both from the same factors, which is why the CPU adjoint gets its solve for free. Our
 * Jacobian is stored as TRUE CSR so the plain solve serves the forward Newton, leaving the adjoint
 * direction unavailable; we buy it with a second factorization of an explicit transpose.
 *
 * <p>The reference is deliberately NOT another GPU path and NOT OLF's Jacobian: it is a plain
 * dense Gaussian elimination on {@code Aᵀ} written here. An independent implementation is the point
 * — if our CSR transpose and cuDSS's factorization are both wrong in the same way, sharing code with
 * the thing under test would hide it.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuTransposedSolveTest {

    @BeforeEach
    void checkGpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(),
                "GPU full-NR native lib not available (build it with native/build-gpu.sh + cuDSS)");
    }

    /** Dense solve of {@code Mᵀ x = b} by Gaussian elimination with partial pivoting — the reference. */
    private static double[] denseSolveTransposed(int n, double[][] m, double[] b) {
        double[][] a = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i][j] = m[j][i];                           // transpose: row i of Mᵀ is column i of M
            }
            a[i][n] = b[i];
        }
        for (int col = 0; col < n; col++) {
            int piv = col;
            for (int r = col + 1; r < n; r++) {
                if (Math.abs(a[r][col]) > Math.abs(a[piv][col])) {
                    piv = r;
                }
            }
            double[] t = a[col];
            a[col] = a[piv];
            a[piv] = t;
            for (int r = col + 1; r < n; r++) {
                double f = a[r][col] / a[col][col];
                for (int c = col; c <= n; c++) {
                    a[r][c] -= f * a[col][c];
                }
            }
        }
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double s = a[i][n];
            for (int j = i + 1; j < n; j++) {
                s -= a[i][j] * x[j];
            }
            x[i] = s / a[i][i];
        }
        return x;
    }

    /**
     * A NON-SYMMETRIC sparse matrix is essential: on a symmetric matrix {@code Aᵀ == A}, so a
     * transpose bug would pass unnoticed. Diagonally dominant to keep it well conditioned, with
     * asymmetric off-diagonals so the transpose is genuinely a different matrix.
     */
    private static double[][] nonSymmetricDense(int n, long seed) {
        Random rnd = new Random(seed);
        double[][] m = new double[n][n];
        for (int i = 0; i < n; i++) {
            m[i][i] = 10 + rnd.nextDouble();
            int j1 = (i + 1) % n;
            int j2 = (i + 3) % n;
            m[i][j1] = 1 + rnd.nextDouble();                 // forward coupling
            if (j2 != i && j2 != j1) {
                m[i][j2] = -(1 + rnd.nextDouble()) * 0.5;    // different magnitude/sign the other way
            }
        }
        return m;
    }

    private static void assertTransposedSolveMatches(int n, long seed) {
        double[][] dense = nonSymmetricDense(n, seed);
        // dense -> CSR
        int[] rowPtr = new int[n + 1];
        int nnz = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dense[i][j] != 0) {
                    nnz++;
                }
            }
            rowPtr[i + 1] = nnz;
        }
        int[] colIdx = new int[nnz];
        double[] vals = new double[nnz];
        int k = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dense[i][j] != 0) {
                    colIdx[k] = j;
                    vals[k] = dense[i][j];
                    k++;
                }
            }
        }
        double[] b = new double[n];
        Random rnd = new Random(seed + 1);
        for (int i = 0; i < n; i++) {
            b[i] = rnd.nextDouble() * 2 - 1;
        }

        double[] gpu = TransposedSolveNative.solveTransposedExplicit(n, rowPtr, colIdx, vals, b);
        double[] ref = denseSolveTransposed(n, dense, b);

        double worst = 0;
        for (int i = 0; i < n; i++) {
            worst = Math.max(worst, Math.abs(gpu[i] - ref[i]));
        }
        // Residual check too: ||Aᵀx − b||inf, so we verify the SOLVE, not just agreement with a
        // reference that could share a misconception.
        double worstResidual = 0;
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int j = 0; j < n; j++) {
                s += dense[j][i] * gpu[j];                   // (Aᵀ x)_i = Σ_j A[j][i] x_j
            }
            worstResidual = Math.max(worstResidual, Math.abs(s - b[i]));
        }
        System.out.printf("TRANSPOSE-SOLVE n=%d: max |gpu-ref| = %.3e, max |Aᵀx-b| = %.3e%n",
                n, worst, worstResidual);
        assertEquals(0, worst, 1e-9, "GPU transposed solve must match the dense reference");
        assertEquals(0, worstResidual, 1e-9, "Aᵀx must reproduce b");
    }

    @Test
    void transposedSolveMatchesDenseReference() {
        assertTransposedSolveMatches(8, 42);
        assertTransposedSolveMatches(64, 7);
        assertTransposedSolveMatches(257, 1234);             // non-power-of-two, larger than a warp
    }

    /**
     * The falsification guard: on this NON-symmetric matrix, solving with {@code A} instead of
     * {@code Aᵀ} must give a DIFFERENT answer. Without this, a native implementation that quietly
     * ignored the transpose would still pass the test above if the reference were also untransposed.
     */
    @Test
    void transposeActuallyMattersOnThisMatrix() {
        int n = 32;
        double[][] dense = nonSymmetricDense(n, 99);
        double[] b = new double[n];
        for (int i = 0; i < n; i++) {
            b[i] = 1.0 / (i + 1);
        }
        double[] xT = denseSolveTransposed(n, dense, b);
        // solve with A itself = solve transposed against Aᵀ
        double[][] transposed = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                transposed[i][j] = dense[j][i];
            }
        }
        double[] xN = denseSolveTransposed(n, transposed, b);
        double diff = 0;
        for (int i = 0; i < n; i++) {
            diff = Math.max(diff, Math.abs(xT[i] - xN[i]));
        }
        org.junit.jupiter.api.Assertions.assertTrue(diff > 1e-3,
                "the test matrix must be asymmetric enough that transposing changes the solution "
                        + "(otherwise the transpose test proves nothing); got " + diff);
    }
}
