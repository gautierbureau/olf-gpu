// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The transposed solve of our batched LU, x = A^-T b on the factors of A: the solve cuDSS does not offer (through
// 0.8.0 its solve phase has no transpose), so an adjoint on cuDSS's factors pays a second factorization of an explicit
// A^T (transposed_solve_jni.cu). On our own factors it costs one more pair of triangular solves, as klu_tsolve does
// against klu_solve's factors.
//
// Both engines factorize B = A(rowOrder, colOrder), so A^T x = b is B^T y = c with c[j] = b[colOrder[j]] and
// x[rowOrder[i]] = y[i]: the right-hand side goes in through the column positions and the solution out through the
// row positions, the other way round from the plain solve.
//
// Each solve is a fixed sequence of linear operations; the transposed solve is the transpose of each, in reverse
// order, so it reads the same factors and the same schedules:
// - fast engine, B = L U: U^T forward, then L^T backward. Row i of U^T is column i of U, so the rows read the factors'
//   COLUMNS (per column of the LU, its entries above the diagonal, then below, rows ascending). The plain solve's
//   levels are longest-path levels, so reversed they are a schedule of the transposed solve, as deep: U[k][i] != 0
//   (k < i) puts row k on a higher backward level than row i, so the backward levels from the last down take U^T
//   forward, and the forward levels from the last down take L^T backward.
// - pivoting engine, each supernode s with its lane's orders P_s, Q_s inside it (P_s D_s Q_s = L_s U_s, block in
//   pivot order), its L panel B_Rs Q_s U_s^-1 and its U panel L_s^-1 P_s B_sR: an up pass, the supernodes level by
//   level from the leaves, each slot taking its descendants' U panel columns against their result, the slots taken in
//   column order Q_s and solved with U_s^T, the result kept in pivot order; then a down pass from the root, each
//   supernode taking its own L panel against its ancestors' final values, solved with L_s^T and put back on its slots
//   through its row order P_s. The plain solve with L and U, rows and columns swapped.
//
// The bodies below are what the kernels (scalar_lu.cu, supernodal_lu.cu) run for one row or one supernode of one
// lane; they compile host-side too (LU_TRANSPOSED_FN defined empty), so a host build checks them on the plans' real
// layout without a GPU. Deterministic: fixed term order, no atomics.
#ifndef OLF_GPU_LU_TRANSPOSED_SOLVE_CUH
#define OLF_GPU_LU_TRANSPOSED_SOLVE_CUH

#include <math.h>
#include <stddef.h>

#include "supernodal_lu_plan.cuh"

#ifndef LU_TRANSPOSED_FN
#define LU_TRANSPOSED_FN __host__ __device__ __forceinline__
#endif

namespace olfgpu {

// ---- fast engine: the LU entries of each column j of B's factors, (row, LU entry) with rows ascending -
// [columnStart[j], columnSplit[j]): above the diagonal (U[k][j], k < j), [columnSplit[j], columnStart[j + 1]): below
// it (L[k][j], k > j) ----

// acc - sum of lu[entry[p]] * z[row[p]] over the terms p = begin, begin + step, ... < end of a column, in that order.
LU_TRANSPOSED_FN double subtractColumnTerms(double acc, int begin, int end, int step, int a, int capacity,
                                            const int* __restrict__ columnRow, const int* __restrict__ columnEntry,
                                            const double* __restrict__ lu, const double* __restrict__ z) {
    for (int p = begin; p < end; p += step) {
        acc = fma(-lu[static_cast<size_t>(columnEntry[p]) * capacity + a],
                  z[static_cast<size_t>(columnRow[p]) * capacity + a], acc);
    }
    return acc;
}

// U^T forward, row i: z[i] = (z[i] - sum_k U[k][i] z[k]) / U[i][i], the rows k < i already final.
LU_TRANSPOSED_FN void upperTransposedRow(int i, int a, int capacity, const int* __restrict__ columnStart,
                                         const int* __restrict__ columnSplit, const int* __restrict__ columnRow,
                                         const int* __restrict__ columnEntry, const int* __restrict__ luDiag,
                                         const double* __restrict__ lu, double* __restrict__ z) {
    const double y = subtractColumnTerms(z[static_cast<size_t>(i) * capacity + a], columnStart[i], columnSplit[i], 1,
                                         a, capacity, columnRow, columnEntry, lu, z);
    z[static_cast<size_t>(i) * capacity + a] = y / lu[static_cast<size_t>(luDiag[i]) * capacity + a];
}

// L^T backward, row i, L unit lower: z[i] -= sum_k L[k][i] z[k], the rows k > i already final.
LU_TRANSPOSED_FN void lowerTransposedRow(int i, int a, int capacity, const int* __restrict__ columnStart,
                                         const int* __restrict__ columnSplit, const int* __restrict__ columnRow,
                                         const int* __restrict__ columnEntry, const double* __restrict__ lu,
                                         double* __restrict__ z) {
    z[static_cast<size_t>(i) * capacity + a] = subtractColumnTerms(z[static_cast<size_t>(i) * capacity + a],
            columnSplit[i], columnStart[i + 1], 1, a, capacity, columnRow, columnEntry, lu, z);
}

// ---- pivoting engine: one supernode s of one lane a; the block's orders are per lane, slot -> row (P_s) and
// slot -> column (Q_s) ----

// Up pass, s after its descendants: each slot j of s takes the U panel column j of every descendant t holding it
// (upperRun: its start, y: t's first slot, t's result in pivot order), the slots are read in column order Q_s and
// solved with U_s^T, and s's result stays in pivot order on its slots.
LU_TRANSPOSED_FN void supernodeUpTransposed(int s, int a, int capacity, const int* __restrict__ supernodeStart,
                                            const int* __restrict__ diagonalBase,
                                            const int* __restrict__ forwardRunStart,
                                            const int* __restrict__ forwardRunUpper,
                                            const int* __restrict__ forwardRunY,
                                            const int* __restrict__ forwardRunLength,
                                            const unsigned char* __restrict__ columnPermutation,
                                            const double* __restrict__ lu, double* __restrict__ z) {
    const int f = supernodeStart[s];
    const int b = supernodeStart[s + 1] - f;
    double v[LU_MAX_SUPERNODE];
    double g[LU_MAX_SUPERNODE];
    for (int c = 0; c < b; c++) {
        const int j = f + c;
        double x = z[static_cast<size_t>(j) * capacity + a];
        for (int run = forwardRunStart[j]; run < forwardRunStart[j + 1]; run++) {
            const double* u = lu + static_cast<size_t>(forwardRunUpper[run]) * capacity + a;
            const double* y = z + static_cast<size_t>(forwardRunY[run]) * capacity + a;
            const int length = forwardRunLength[run];
            for (int k = 0; k < length; k++) {
                x = fma(-u[static_cast<size_t>(k) * capacity], y[static_cast<size_t>(k) * capacity], x);
            }
        }
        v[c] = x;
    }
    const double* d = lu + static_cast<size_t>(diagonalBase[s]) * capacity + a;
    for (int k = 0; k < b; k++) {                            // U_s^T g = Q_s^T v
        double x = v[columnPermutation[static_cast<size_t>(f + k) * capacity + a]];
        for (int q = 0; q < k; q++) {
            x = fma(-d[static_cast<size_t>(q * b + k) * capacity], g[q], x);
        }
        g[k] = x / d[static_cast<size_t>(k * b + k) * capacity];
    }
    for (int k = 0; k < b; k++) {
        z[static_cast<size_t>(f + k) * capacity + a] = g[k];
    }
}

// Down pass, s after its ancestors: s's result takes its own L panel against the final values of the rows R_s, is
// solved with L_s^T (L_s unit lower) and lands on s's slots through its row order P_s.
LU_TRANSPOSED_FN void supernodeDownTransposed(int s, int a, int capacity, const int* __restrict__ supernodeStart,
                                              const int* __restrict__ rowStart, const int* __restrict__ rows,
                                              const int* __restrict__ diagonalBase,
                                              const int* __restrict__ lowerBase,
                                              const unsigned char* __restrict__ rowPermutation,
                                              const double* __restrict__ lu, double* __restrict__ z) {
    const int f = supernodeStart[s];
    const int b = supernodeStart[s + 1] - f;
    double v[LU_MAX_SUPERNODE];
    for (int k = 0; k < b; k++) {
        v[k] = z[static_cast<size_t>(f + k) * capacity + a];
    }
    const double* lower = lu + static_cast<size_t>(lowerBase[s]) * capacity + a;
    for (int q = rowStart[s]; q < rowStart[s + 1]; q++) {
        const double x = z[static_cast<size_t>(rows[q]) * capacity + a];
        const double* row = lower + static_cast<size_t>(q - rowStart[s]) * b * capacity;
        for (int k = 0; k < b; k++) {
            v[k] = fma(-row[static_cast<size_t>(k) * capacity], x, v[k]);
        }
    }
    const double* d = lu + static_cast<size_t>(diagonalBase[s]) * capacity + a;
    for (int k = b - 1; k >= 0; k--) {                       // L_s^T h = v, in place
        double x = v[k];
        for (int q = k + 1; q < b; q++) {
            x = fma(-d[static_cast<size_t>(q * b + k) * capacity], v[q], x);
        }
        v[k] = x;
    }
    for (int k = 0; k < b; k++) {                            // x_s = P_s^T h
        z[static_cast<size_t>(f + rowPermutation[static_cast<size_t>(f + k) * capacity + a]) * capacity + a] = v[k];
    }
}

}  // namespace olfgpu

#endif  // OLF_GPU_LU_TRANSPOSED_SOLVE_CUH
