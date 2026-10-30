// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See batched_lu_compare.cuh.
#include "batched_lu_compare.cuh"

#include <cudss.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <vector>

#include "../common/check.cuh"
#include "../solver/linear_solver_settings.cuh"

namespace olfgpu {

namespace {

constexpr int COMPARE_BLOCK = 256;
constexpr int SCORES = 5;    // max |dxOwn - dxCudss|, max |dxCudss|, max |b - J dxOwn|, max |b - J dxCudss|, max |b|

__device__ double magnitude(double v) {
    const double a = fabs(v);
    return a <= DBL_MAX ? a : INFINITY;                      // NaN counts as infinite: fmax would drop it
}

// One block per lane of the lane map.
__global__ void scoreLanes(int n, int nnz, const int* __restrict__ laneMap, const int* __restrict__ rowPtr,
                           const int* __restrict__ colIdx, const double* __restrict__ vals,
                           const double* __restrict__ b, const double* __restrict__ dxOwn,
                           const double* __restrict__ dxCudss, double* __restrict__ score) {
    __shared__ double reduction[SCORES][COMPARE_BLOCK];
    const int s = laneMap[blockIdx.x];
    const double* v = vals + static_cast<size_t>(s) * nnz;
    const double* bs = b + static_cast<size_t>(s) * n;
    const double* xo = dxOwn + static_cast<size_t>(s) * n;
    const double* xc = dxCudss + static_cast<size_t>(s) * n;
    double m[SCORES] = {0.0, 0.0, 0.0, 0.0, 0.0};
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        double ro = bs[i];
        double rc = bs[i];
        for (int k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
            ro -= v[k] * xo[colIdx[k]];
            rc -= v[k] * xc[colIdx[k]];
        }
        m[0] = fmax(m[0], magnitude(xo[i] - xc[i]));
        m[1] = fmax(m[1], magnitude(xc[i]));
        m[2] = fmax(m[2], magnitude(ro));
        m[3] = fmax(m[3], magnitude(rc));
        m[4] = fmax(m[4], magnitude(bs[i]));
    }
    for (int j = 0; j < SCORES; j++) {
        reduction[j][threadIdx.x] = m[j];
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            for (int j = 0; j < SCORES; j++) {
                reduction[j][threadIdx.x] = fmax(reduction[j][threadIdx.x], reduction[j][threadIdx.x + stride]);
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int j = 0; j < SCORES; j++) {
            score[static_cast<size_t>(blockIdx.x) * SCORES + j] = reduction[j][0];
        }
    }
}

// cuDSS's uniform batch over every slot of the context (an idle slot is an identity system), into dLuComparisonDx.
void solveWithCudss(BatchSolverContext* ctx) {
    const int n = ctx->n;
    const int nS = ctx->nScenarios;
    CudssCtx& solver = ctx->luComparisonSolver;
    if (!ctx->luComparisonAnalyzed) {
        ctx->dLuComparisonDx.alloc(static_cast<size_t>(n) * nS * sizeof(double));
        CUDSS_CHECK(cudssCreate(&solver.handle));
        CUDSS_CHECK(cudssSetStream(solver.handle, ctx->stream));
        CUDSS_CHECK(cudssConfigCreate(&solver.config));
        applyReorderAlg(solver.config, ctx->structure->nZeroImpedanceBranch > 0);
        // the internal refinement steps of the context's own solver (configured base steps on a structure with zero-
        // impedance branches): compared with our dx after the same number of plain refinement steps
        int refinement = ctx->structure->nZeroImpedanceBranch > 0 ? configuredIterativeRefinementBaseSteps() : 0;
        CUDSS_CHECK(cudssConfigSet(solver.config, CUDSS_CONFIG_IR_N_STEPS, &refinement, sizeof(refinement)));
        CUDSS_CHECK(cudssConfigSet(solver.config, CUDSS_CONFIG_UBATCH_SIZE, &ctx->nScenarios, sizeof(ctx->nScenarios)));
        CUDSS_CHECK(cudssDataCreate(solver.handle, &solver.data));
        CUDSS_CHECK(cudssMatrixCreateCsr(&solver.systemMatrix, n, n, ctx->nnz, ctx->structure->dRowPtr.as<int>(), nullptr,
                                         ctx->structure->dColIdx.as<int>(), ctx->dVals.as<double>(), OLF_CUDSS_CSR_TYPES,
                                         CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        CUDSS_CHECK(cudssMatrixCreateDn(&solver.rightHandSideMatrix, n, 1, n, ctx->dB.as<double>(), OLF_CUDSS_DN_TYPE,
                                        CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssMatrixCreateDn(&solver.solutionMatrix, n, 1, n, ctx->dLuComparisonDx.as<double>(),
                                        OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssExecute(solver.handle, CUDSS_PHASE_ANALYSIS, solver.config, solver.data, solver.systemMatrix,
                                 solver.solutionMatrix, solver.rightHandSideMatrix));
        ctx->luComparisonAnalyzed = true;
    }
    CUDSS_CHECK(cudssExecute(solver.handle, CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_SOLVE, solver.config, solver.data,
                             solver.systemMatrix, solver.solutionMatrix, solver.rightHandSideMatrix));
}

}  // namespace

void compareBatchedLuWithCudss(BatchSolverContext* ctx, int outer, int iteration) {
    BatchedLu& lu = *ctx->batchedLu;
    const int lanes = lu.width;
    if (lanes == 0) {
        return;
    }
    solveWithCudss(ctx);
    ctx->dLuComparisonScore.alloc(static_cast<size_t>(ctx->nScenarios) * SCORES * sizeof(double));
    scoreLanes<<<lanes, COMPARE_BLOCK, 0, ctx->stream>>>(ctx->n, ctx->nnz, deviceLaneMap(lu),
            ctx->structure->dRowPtr.as<int>(), ctx->structure->dColIdx.as<int>(), ctx->dVals.as<double>(),
            ctx->dB.as<double>(), ctx->dDx.as<double>(), ctx->dLuComparisonDx.as<double>(),
            ctx->dLuComparisonScore.as<double>());
    CUDA_CHECK(cudaGetLastError());
    std::vector<double> score(static_cast<size_t>(lanes) * SCORES);
    CUDA_CHECK(cudaMemcpyAsync(score.data(), ctx->dLuComparisonScore.p, score.size() * sizeof(double),
                               cudaMemcpyDeviceToHost, ctx->stream));
    const std::vector<int> clamps = readPivotClamps(lu, ctx->stream);  // synchronizes the stream
    double worstDifference = 0.0;
    double worstOwn = 0.0;
    double worstCudss = 0.0;
    int clampedLanes = 0;
    int decades[5] = {0, 0, 0, 0, 0};                        // < 1e-14, < 1e-12, < 1e-10, < 1e-8, beyond
    auto relative = [](double num, double den) { return den > 0.0 ? num / den : num; };
    for (int a = 0; a < lanes; a++) {
        const double* sc = score.data() + static_cast<size_t>(a) * SCORES;
        const double difference = relative(sc[0], sc[1]);
        worstDifference = std::max(worstDifference, difference);
        worstOwn = std::max(worstOwn, relative(sc[2], sc[4]));
        worstCudss = std::max(worstCudss, relative(sc[3], sc[4]));
        clampedLanes += clamps[a] > 0 ? 1 : 0;
        decades[difference < 1e-14 ? 0 : difference < 1e-12 ? 1 : difference < 1e-10 ? 2 : difference < 1e-8 ? 3 : 4]++;
    }
    std::fprintf(stderr, "OLF_GPU_LU_COMPARE outer=%d it=%d lanes=%d clampedLanes=%d worstRelativeDifference=%.3e "
                 "worstBackwardOwn=%.3e worstBackwardCudss=%.3e decades=[%d,%d,%d,%d,%d]\n", outer, iteration, lanes,
                 clampedLanes, worstDifference, worstOwn, worstCudss, decades[0], decades[1], decades[2], decades[3],
                 decades[4]);
}

}  // namespace olfgpu
