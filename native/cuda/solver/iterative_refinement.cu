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
#include "state_vector_scaling.cuh"

namespace olfgpu {

// The largest of the block's values, returned to every thread (blockDim.x == LANE_NORM_BLOCK). fmax drops a NaN and
// a maximum does not depend on the order it is taken in: the result is the serial walk's, bit for bit.
static __device__ double blockMax(double v, double* shared) {
    shared[threadIdx.x] = v;
    __syncthreads();
    for (int w = LANE_NORM_BLOCK / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) {
            shared[threadIdx.x] = fmax(shared[threadIdx.x], shared[threadIdx.x + w]);
        }
        __syncthreads();
    }
    const double largest = shared[0];
    __syncthreads();                                         // the next blockMax reuses the buffer
    return largest;
}

// Per-scenario score of the linear solve just done: max|J.dx-B| under both sign conventions and
// max|F|, one block per scenario.
__global__ void linResidScoreBatch(int n, int nS, const double* rp, const double* rm, const double* F,
                                   double* score) {
    __shared__ double shared[LANE_NORM_BLOCK];
    const int s = blockIdx.x;
    if (s >= nS) return;
    double mp = 0.0;
    double mm = 0.0;
    double mf = 0.0;
    const double* rps = rp + (size_t) s * n;
    const double* rms = rm + (size_t) s * n;
    const double* fs = F + (size_t) s * n;
    for (int i = threadIdx.x; i < n; i += LANE_NORM_BLOCK) {
        mp = fmax(mp, fabs(rps[i]));
        mm = fmax(mm, fabs(rms[i]));
        mf = fmax(mf, fabs(fs[i]));
    }
    mp = blockMax(mp, shared);
    mm = blockMax(mm, shared);
    mf = blockMax(mf, shared);
    if (threadIdx.x != 0) return;
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

// r = b - J dx per scenario (the refinement residual; the right-hand side as SOLVED, row repairs included), on the
// lanes whose mask is set, every lane when it is null
__global__ void residualBatch(int n, int nS, int nnz, const int* rowPtr, const int* colIdx,
                              const double* vals, const double* dx, const double* b, const char* lanes, double* r) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * nS) return;
    int s = t / n;
    if (lanes != nullptr && !lanes[s]) return;
    int i = t - s * n;
    const double* v = vals + (size_t) s * nnz;
    const double* d = dx + (size_t) s * n;
    double acc = 0.0;
    for (int k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
        acc += v[k] * d[colIdx[k]];
    }
    r[t] = b[t] - acc;
}

namespace {

constexpr int RESIDUAL_TILE_SCENARIOS = 32;                  // the interleaved residual's block: 32 scenarios x 8 rows
constexpr int RESIDUAL_TILE_ROWS = 8;

// residualBatch on interleaved values (common/batch_layout.cuh): a warp walks one row for 32 consecutive scenarios,
// reading the values in whole sectors; b and r, scenario-major, go through shared memory 8 consecutive rows per
// scenario. Each r is b - (the row's products summed in order), as in residualBatch.
__global__ void __launch_bounds__(RESIDUAL_TILE_SCENARIOS * RESIDUAL_TILE_ROWS)
residualInterleavedBatch(int n, int nS, int stride, const int* __restrict__ rowPtr, const int* __restrict__ colIdx,
                         const double* __restrict__ vals, const double* __restrict__ dx, const double* __restrict__ b,
                         const char* __restrict__ lanes, double* __restrict__ r) {
    __shared__ double tile[RESIDUAL_TILE_SCENARIOS][RESIDUAL_TILE_ROWS + 1];
    const int s0 = blockIdx.y * RESIDUAL_TILE_SCENARIOS;
    const int i0 = blockIdx.x * RESIDUAL_TILE_ROWS;
    const int l = threadIdx.y * RESIDUAL_TILE_SCENARIOS + threadIdx.x;
    const int ls = l / RESIDUAL_TILE_ROWS;                   // the tile entry this thread moves: scenario ls, row lr
    const int lr = l % RESIDUAL_TILE_ROWS;
    if (s0 + ls < nS && i0 + lr < n) {
        tile[ls][lr] = b[static_cast<size_t>(s0 + ls) * n + i0 + lr];
    }
    __syncthreads();
    const int s = s0 + threadIdx.x;
    const int i = i0 + threadIdx.y;
    if (s < nS && i < n && (lanes == nullptr || lanes[s])) {
        const double* d = dx + static_cast<size_t>(s) * n;
        double acc = 0.0;
        for (int k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
            acc += vals[static_cast<size_t>(k) * stride + s] * d[colIdx[k]];
        }
        tile[threadIdx.x][threadIdx.y] = tile[threadIdx.x][threadIdx.y] - acc;
    }
    __syncthreads();
    if (s0 + ls < nS && i0 + lr < n && (lanes == nullptr || lanes[s0 + ls])) {
        r[static_cast<size_t>(s0 + ls) * n + i0 + lr] = tile[ls][lr];
    }
}

// linResidRowBatch on interleaved values, shaped as residualInterleavedBatch: F, r+ and r- through shared memory.
__global__ void __launch_bounds__(RESIDUAL_TILE_SCENARIOS * RESIDUAL_TILE_ROWS)
linResidRowInterleavedBatch(int n, int nS, int stride, const int* __restrict__ rowPtr, const int* __restrict__ colIdx,
                            const double* __restrict__ vals, const double* __restrict__ dx, const double* __restrict__ F,
                            double* __restrict__ rPlus, double* __restrict__ rMinus) {
    __shared__ double plus[RESIDUAL_TILE_SCENARIOS][RESIDUAL_TILE_ROWS + 1];
    __shared__ double minus[RESIDUAL_TILE_SCENARIOS][RESIDUAL_TILE_ROWS + 1];
    const int s0 = blockIdx.y * RESIDUAL_TILE_SCENARIOS;
    const int i0 = blockIdx.x * RESIDUAL_TILE_ROWS;
    const int l = threadIdx.y * RESIDUAL_TILE_SCENARIOS + threadIdx.x;
    const int ls = l / RESIDUAL_TILE_ROWS;
    const int lr = l % RESIDUAL_TILE_ROWS;
    if (s0 + ls < nS && i0 + lr < n) {
        plus[ls][lr] = F[static_cast<size_t>(s0 + ls) * n + i0 + lr];
    }
    __syncthreads();
    const int s = s0 + threadIdx.x;
    const int i = i0 + threadIdx.y;
    if (s < nS && i < n) {
        const double* d = dx + static_cast<size_t>(s) * n;
        double acc = 0.0;
        for (int k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
            acc += vals[static_cast<size_t>(k) * stride + s] * d[colIdx[k]];
        }
        const double f = plus[threadIdx.x][threadIdx.y];
        minus[threadIdx.x][threadIdx.y] = acc + f;
        plus[threadIdx.x][threadIdx.y] = acc - f;
    }
    __syncthreads();
    if (s0 + ls < nS && i0 + lr < n) {
        rPlus[static_cast<size_t>(s0 + ls) * n + i0 + lr] = plus[ls][lr];
        rMinus[static_cast<size_t>(s0 + ls) * n + i0 + lr] = minus[ls][lr];
    }
}

// residualBatch with the values and the vectors all interleaved by scenario: one thread per (row, scenario), the
// scenarios fastest, every read and write in whole sectors. Each r is b - (the row's products summed in order), as in
// residualBatch.
__global__ void residualAllInterleavedBatch(int n, int nS, int stride, const int* __restrict__ rowPtr,
                                            const int* __restrict__ colIdx, const double* __restrict__ vals,
                                            const double* __restrict__ dx, const double* __restrict__ b,
                                            const char* __restrict__ lanes, double* __restrict__ r) {
    const size_t g = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= static_cast<size_t>(n) * stride) {
        return;
    }
    const int i = static_cast<int>(g / stride);
    const int s = static_cast<int>(g - static_cast<size_t>(i) * stride);
    if (s >= nS || (lanes != nullptr && !lanes[s])) {
        return;
    }
    double acc = 0.0;
    for (int k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
        acc += vals[static_cast<size_t>(k) * stride + s] * dx[static_cast<size_t>(colIdx[k]) * stride + s];
    }
    r[g] = b[g] - acc;
}

// The interleaved score's maxima: block (32 scenarios, SCORE_ROWS row groups) over SCORE_CHUNK rows. |r| and |b| maxima
// are taken with fmax as iterativeRefinementScoreBatch takes them (a NaN entry is passed over), then merged into
// maxima[2 s] and maxima[2 s + 1] by an atomic max on their bits: non-negative doubles order as their bit patterns, and
// a maximum is the same whatever the order, so every score comes out as iterativeRefinementScoreBatch's.
constexpr int SCORE_SCENARIOS = 32;
constexpr int SCORE_ROWS = 8;
constexpr int SCORE_CHUNK = 256;

__global__ void refinementMaximaInterleavedBatch(int n, int nS, int stride, const double* __restrict__ r,
                                                 const double* __restrict__ b, const char* __restrict__ active,
                                                 unsigned long long* __restrict__ maxima) {
    __shared__ double sharedR[SCORE_ROWS][SCORE_SCENARIOS];
    __shared__ double sharedB[SCORE_ROWS][SCORE_SCENARIOS];
    const int s = blockIdx.x * SCORE_SCENARIOS + threadIdx.x;
    const bool scored = s < nS && (active == nullptr || active[s]);
    double mr = 0.0;
    double mb = 0.0;
    if (scored) {
        const int end = min(n, (blockIdx.y + 1) * SCORE_CHUNK);
        for (int i = blockIdx.y * SCORE_CHUNK + threadIdx.y; i < end; i += SCORE_ROWS) {
            mr = fmax(mr, fabs(r[static_cast<size_t>(i) * stride + s]));
            mb = fmax(mb, fabs(b[static_cast<size_t>(i) * stride + s]));
        }
    }
    sharedR[threadIdx.y][threadIdx.x] = mr;
    sharedB[threadIdx.y][threadIdx.x] = mb;
    __syncthreads();
    if (threadIdx.y == 0 && scored) {
        for (int y = 1; y < SCORE_ROWS; y++) {
            mr = fmax(mr, sharedR[y][threadIdx.x]);
            mb = fmax(mb, sharedB[y][threadIdx.x]);
        }
        atomicMax(&maxima[2 * s], static_cast<unsigned long long>(__double_as_longlong(mr)));
        atomicMax(&maxima[2 * s + 1], static_cast<unsigned long long>(__double_as_longlong(mb)));
    }
}

__global__ void refinementScoreOfMaxima(int nS, const char* __restrict__ active,
                                        const unsigned long long* __restrict__ maxima, double* __restrict__ score) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nS) {
        return;
    }
    if (active != nullptr && !active[s]) {
        score[s] = 0.0;
        return;
    }
    const double mr = __longlong_as_double(static_cast<long long>(maxima[2 * s]));
    const double mb = __longlong_as_double(static_cast<long long>(maxima[2 * s + 1]));
    score[s] = mb > 0.0 ? (isfinite(mr) ? mr / mb : 1e300) : 0.0;
}

__global__ void axpyLaneMaskInterleavedBatch(int nS, int stride, size_t total, double sign, const double* d,
                                             const char* lane, double* x) {
    const size_t g = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= total) {
        return;
    }
    const int s = static_cast<int>(g % stride);
    if (s < nS && (lane == nullptr || lane[s])) {
        x[g] += sign * d[g];
    }
}

}  // namespace

void launchRefinementScore(cudaStream_t stream, int n, int nS, const double* r, const double* b, const char* active,
                           DeviceBuf& score, BatchLayout layout, bool interleavedVectors) {
    if (!interleavedVectors) {
        iterativeRefinementScoreBatch<<<nS, LANE_NORM_BLOCK, 0, stream>>>(n, nS, r, b, active, score.as<double>());
        return;
    }
    score.ensure(static_cast<size_t>(nS) * 3 * sizeof(double));
    unsigned long long* maxima = reinterpret_cast<unsigned long long*>(score.as<double>() + nS);
    CUDA_CHECK(cudaMemsetAsync(maxima, 0, static_cast<size_t>(nS) * 2 * sizeof(unsigned long long), stream));
    const dim3 blocks((nS + SCORE_SCENARIOS - 1) / SCORE_SCENARIOS, (n + SCORE_CHUNK - 1) / SCORE_CHUNK);
    refinementMaximaInterleavedBatch<<<blocks, dim3(SCORE_SCENARIOS, SCORE_ROWS), 0, stream>>>(n, nS, layout.stride, r,
            b, active, maxima);
    refinementScoreOfMaxima<<<grid(nS), BLOCK, 0, stream>>>(nS, active, maxima, score.as<double>());
}

void launchLaneAxpy(cudaStream_t stream, int n, int nS, double sign, const double* d, const char* lane, double* x,
                    BatchLayout layout, bool interleavedVectors) {
    if (interleavedVectors) {
        const size_t total = static_cast<size_t>(n) * layout.stride;
        axpyLaneMaskInterleavedBatch<<<static_cast<unsigned int>((total + BLOCK - 1) / BLOCK), BLOCK, 0, stream>>>(nS,
                layout.stride, total, sign, d, lane, x);
    } else if (lane != nullptr) {
        axpyLaneMaskBatch<<<grid(n * nS), BLOCK, 0, stream>>>(n, n * nS, sign, d, lane, x);
    } else {
        axpy1Batch<<<grid(n * nS), BLOCK, 0, stream>>>(n * nS, d, x);
    }
}

void launchResidual(cudaStream_t stream, int n, int nS, int nnz, const int* rowPtr, const int* colIdx,
                    const double* vals, BatchLayout layout, const double* dx, const double* b, const char* lanes,
                    double* r, bool interleavedVectors) {
    if (interleavedVectors) {
        const size_t threads = static_cast<size_t>(n) * layout.stride;
        residualAllInterleavedBatch<<<static_cast<unsigned int>((threads + BLOCK - 1) / BLOCK), BLOCK, 0, stream>>>(n,
                nS, layout.stride, rowPtr, colIdx, vals, dx, b, lanes, r);
    } else if (layout.interleaved) {
        const dim3 tiles((n + RESIDUAL_TILE_ROWS - 1) / RESIDUAL_TILE_ROWS,
                         (nS + RESIDUAL_TILE_SCENARIOS - 1) / RESIDUAL_TILE_SCENARIOS);
        residualInterleavedBatch<<<tiles, dim3(RESIDUAL_TILE_SCENARIOS, RESIDUAL_TILE_ROWS), 0, stream>>>(n, nS,
                layout.stride, rowPtr, colIdx, vals, dx, b, lanes, r);
    } else {
        residualBatch<<<grid(n * nS), BLOCK, 0, stream>>>(n, nS, nnz, rowPtr, colIdx, vals, dx, b, lanes, r);
    }
}

void launchLinResidRows(cudaStream_t stream, int n, int nS, int nnz, const int* rowPtr, const int* colIdx,
                        const double* vals, BatchLayout layout, const double* dx, const double* F, double* rPlus,
                        double* rMinus) {
    if (layout.interleaved) {
        const dim3 tiles((n + RESIDUAL_TILE_ROWS - 1) / RESIDUAL_TILE_ROWS,
                         (nS + RESIDUAL_TILE_SCENARIOS - 1) / RESIDUAL_TILE_SCENARIOS);
        linResidRowInterleavedBatch<<<tiles, dim3(RESIDUAL_TILE_SCENARIOS, RESIDUAL_TILE_ROWS), 0, stream>>>(n, nS,
                layout.stride, rowPtr, colIdx, vals, dx, F, rPlus, rMinus);
    } else {
        linResidRowBatch<<<grid(n * nS), BLOCK, 0, stream>>>(n, nS, nnz, rowPtr, colIdx, vals, dx, F, rPlus, rMinus);
    }
}

// per scenario: ||r||inf / ||b||inf (0 for an inactive lane or a zero right-hand side), one block per scenario
__global__ void iterativeRefinementScoreBatch(int n, int nS, const double* r, const double* b, const char* active, double* score) {
    __shared__ double shared[LANE_NORM_BLOCK];
    const int s = blockIdx.x;
    if (s >= nS) return;
    if (active != nullptr && !active[s]) {
        if (threadIdx.x == 0) {
            score[s] = 0.0;
        }
        return;
    }
    const double* rs = r + (size_t) s * n;
    const double* bs = b + (size_t) s * n;
    double mr = 0.0, mb = 0.0;
    for (int i = threadIdx.x; i < n; i += LANE_NORM_BLOCK) {
        mr = fmax(mr, fabs(rs[i]));
        mb = fmax(mb, fabs(bs[i]));
    }
    mr = blockMax(mr, shared);
    mb = blockMax(mb, shared);
    if (threadIdx.x == 0) {
        score[s] = mb > 0.0 ? (isfinite(mr) ? mr / mb : 1e300) : 0.0;
    }
}

__global__ void axpyLaneMaskBatch(int n, int total, double sign, const double* d, const char* lane, double* x) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;   // x += sign * d on the lanes whose mask is set
    if (t < total && lane[t / n]) x[t] += sign * d[t];
}


}  // namespace olfgpu
