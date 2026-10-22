// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See convergence.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/codes/gpu_row_mode.cuh"
#include "../solver/driver_steps.cuh"
#include "convergence.cuh"

namespace olfgpu {

// Per-scenario convergence norms on DEVICE: out[s*3] = ||F||inf, out[s*3+1] = sum(F^2),
// out[s*3+2] = finite flag — so the per-iteration readback is nS*3 doubles instead of the full
// nS*n mismatch (70 MB at 500 x 18k, pageable ~5.5 GB/s + a 9.2M-element host loop). infnorm is
// exact vs the host loop; the ssq PARALLEL summation order differs from the host's sequential
// sum by last-ulp amounts — borderline ||F||2 stopping decisions can flip at ulp level (same
// class of effect as any reduction-order change; the criterion itself is unchanged).
__global__ void residualNormsBatch(int n, const double* F, const char* active, double* out,
                                   const double* fixedVariableRowMaxMismatch, const double* equationRowMaxMismatch, const int* mode,
                                   const double* alternateRowMaxMismatch, const int* alternateRowMaxMismatchSelector) {
    int s = blockIdx.x;
    __shared__ double sharedInfNorm[BLOCK];
    __shared__ double sharedSumOfSquares[BLOCK];
    __shared__ double sharedMaxRatio[BLOCK];
    __shared__ int sharedFinite[BLOCK];
    double mInf = 0.0;
    double ssq = 0.0;
    double mRat = 0.0;
    int fin = 1;
    if (active == nullptr || active[s]) {
        const double* f = F + (size_t) s * n;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            double v = f[i];
            if (!isfinite(v)) fin = 0;
            mInf = fmax(mInf, fabs(v));
            ssq += v * v;
            if (fixedVariableRowMaxMismatch != nullptr) {
                // PER_EQUATION_TYPE: identity/slope rows (mode 0/2) use the row's identity-type
                // threshold; power rows (mode 1), the phase P row (mode 3) and the BUS_TARGET_Q
                // row of a remote member switched to PQ (mode 4) the power-type one
                int m = mode[(size_t) s * n + i];
                double thr = (m == GpuRowMode::EQUATION || m == GpuRowMode::BRANCH_TARGET_P || m == GpuRowMode::BUS_TARGET_Q) ? equationRowMaxMismatch[i] : fixedVariableRowMaxMismatch[i];
                if (alternateRowMaxMismatch != nullptr) {
                    // the row's equation in THIS lane: a cross identity (mode 6) is a voltage target; a
                    // relocated child's own row hosts its balance again once its coupler's dummy row (the
                    // selector) is an identity
                    if (m == GpuRowMode::CROSS_FIXED_VARIABLE) {
                        thr = alternateRowMaxMismatch[i];
                    } else if ((m == GpuRowMode::EQUATION || m == GpuRowMode::BRANCH_TARGET_P || m == GpuRowMode::BUS_TARGET_Q) && alternateRowMaxMismatchSelector[i] >= 0 && mode[(size_t) s * n + alternateRowMaxMismatchSelector[i]] == GpuRowMode::FIXED_VARIABLE) {
                        thr = alternateRowMaxMismatch[i];
                    }
                }
                mRat = fmax(mRat, fabs(v) / thr);
            }
        }
    }
    sharedInfNorm[threadIdx.x] = mInf;
    sharedSumOfSquares[threadIdx.x] = ssq;
    sharedMaxRatio[threadIdx.x] = mRat;
    sharedFinite[threadIdx.x] = fin;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            sharedInfNorm[threadIdx.x] = fmax(sharedInfNorm[threadIdx.x], sharedInfNorm[threadIdx.x + off]);
            sharedSumOfSquares[threadIdx.x] += sharedSumOfSquares[threadIdx.x + off];
            sharedMaxRatio[threadIdx.x] = fmax(sharedMaxRatio[threadIdx.x], sharedMaxRatio[threadIdx.x + off]);
            sharedFinite[threadIdx.x] &= sharedFinite[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[s * 4 + 0] = sharedInfNorm[0];
        out[s * 4 + 1] = sharedSumOfSquares[0];
        out[s * 4 + 2] = sharedFinite[0] ? 1.0 : 0.0;
        out[s * 4 + 3] = sharedMaxRatio[0];                           // max |F_i|/thr_i (0 without thresholds)
    }
}

void readNormsBatch(BatchSolverContext* ctx, DriverPass& drv, std::vector<double>& hn) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    residualNormsBatch<<<nS, BLOCK, 0, ctx->stream>>>(n, ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->dNorms.as<double>(),
            ctx->perRowCriteria ? ctx->dFixedVariableRowMaxMismatch.as<double>() : nullptr,
            ctx->perRowCriteria ? ctx->dEquationRowMaxMismatch.as<double>() : nullptr, ctx->dMode.as<int>(),
            ctx->perRowCriteria && ctx->perRowAlternateCriteria ? ctx->dAlternateRowMaxMismatch.as<double>() : nullptr,
            ctx->perRowCriteria && ctx->perRowAlternateCriteria ? ctx->dAlternateRowMaxMismatchSelector.as<int>() : nullptr);
    hn.resize(static_cast<size_t>(nS) * 4);
    CUDA_CHECK(cudaMemcpy(hn.data(), ctx->dNorms.p, hn.size() * sizeof(double), cudaMemcpyDeviceToHost));
}

}  // namespace olfgpu
