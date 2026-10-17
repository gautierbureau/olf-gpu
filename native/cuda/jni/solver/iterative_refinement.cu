// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See iterative_refinement.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../solver/driver_steps.cuh"
#include "../common/stats.cuh"
#include "../common/launch.cuh"
#include "../solver/linear_solver_settings.cuh"
#include "iterative_refinement.cuh"

namespace olfgpu {

// Per-scenario score of the linear solve just done: max|J.dx-B| under both sign conventions and
// max|F|, one thread per scenario (n serial reads each; a few thousand per scenario, once per solve).
__global__ void linResidScoreBatch(int n, int nS, const double* rp, const double* rm, const double* F,
                                   double* score) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nS) return;
    double mp = 0.0;
    double mm = 0.0;
    double mf = 0.0;
    const double* rps = rp + (size_t) s * n;
    const double* rms = rm + (size_t) s * n;
    const double* fs = F + (size_t) s * n;
    for (int i = 0; i < n; i++) {
        mp = fmax(mp, fabs(rps[i]));
        mm = fmax(mm, fabs(rms[i]));
        mf = fmax(mf, fabs(fs[i]));
    }
    score[3 * s] = fmin(mp, mm);
    score[3 * s + 1] = mf;
    score[3 * s + 2] = (mf > 0.0 && isfinite(mp) && isfinite(mm)) ? fmin(mp, mm) / mf : 1e300;
}

// How accurately did the LINEAR solve actually solve? Per row, r_i = (J.dx)_i -/+ F_i, per scenario: every
// scenario has its OWN values block (stride nnz) and its own dx / F block (stride n), over the SHARED sparsity
// pattern. ||r|| at round-off relative to ||F|| means the solve is exact; a large ||r|| is what iterative
// refinement corrects (this is also refinement's own first step).
__global__ void linResidRowBatch(int n, int nS, int nnz, const int* rowPtr, const int* colIdx,
                                 const double* vals, const double* dx, const double* F,
                                 double* rPlus, double* rMinus) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * nS) return;
    int s = t / n;
    int i = t - s * n;
    const double* v = vals + (size_t) s * nnz;
    const double* d = dx + (size_t) s * n;
    double acc = 0.0;
    for (int k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
        acc += v[k] * d[colIdx[k]];
    }
    rPlus[t] = acc - F[t];
    rMinus[t] = acc + F[t];
}

// r = b - J dx per scenario (the refinement residual; the right-hand side as SOLVED, row repairs included)
__global__ void residualBatch(int n, int nS, int nnz, const int* rowPtr, const int* colIdx,
                              const double* vals, const double* dx, const double* b, double* r) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * nS) return;
    int s = t / n;
    int i = t - s * n;
    const double* v = vals + (size_t) s * nnz;
    const double* d = dx + (size_t) s * n;
    double acc = 0.0;
    for (int k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
        acc += v[k] * d[colIdx[k]];
    }
    r[t] = b[t] - acc;
}

// per scenario: ||r||inf / ||b||inf (0 for an inactive lane or a zero right-hand side)
__global__ void irScoreBatch(int n, int nS, const double* r, const double* b, const char* active, double* score) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nS) return;
    if (active != nullptr && !active[s]) {
        score[s] = 0.0;
        return;
    }
    const double* rs = r + (size_t) s * n;
    const double* bs = b + (size_t) s * n;
    double mr = 0.0, mb = 0.0;
    for (int i = 0; i < n; i++) {
        mr = fmax(mr, fabs(rs[i]));
        mb = fmax(mb, fabs(bs[i]));
    }
    score[s] = mb > 0.0 ? (isfinite(mr) ? mr / mb : 1e300) : 0.0;
}

__global__ void axpyLaneMaskBatch(int n, int total, double sign, const double* d, const char* lane, double* x) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;   // x += sign * d on the lanes whose mask is set
    if (t < total && lane[t / n]) x[t] += sign * d[t];
}


}  // namespace olfgpu
