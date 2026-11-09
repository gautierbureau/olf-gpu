// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See sensitivity_basis.cuh.
#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/codes/gpu_linear_solve_mode.cuh"
#include "../solver/batch_context.cuh"
#include "../solver/driver_steps.cuh"
#include "sensitivity.cuh"
#include "sensitivity_basis.cuh"

namespace olfgpu {

__global__ void broadcastColumnBatch(int n, int nScenarios, const double* column, double* b) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * nScenarios) return;
    b[t] = column[t % n];
}

__global__ void gatherSensitivityBasisBatch(int nScenarios, int n, int nGather, const int* gatherRows, int column, int nColumns,
                                            const double* dx, double* out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nScenarios * nGather) return;
    int s = t / nGather;
    int f = t % nGather;
    out[((size_t) s * nColumns + column) * nGather + f] = dx[(size_t) s * n + gatherRows[f]];
}

__global__ void fillSensitivityBasisColumnBatch(int nScenarios, int nGather, int column, int nColumns, double value, double* out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nScenarios * nGather) return;
    int s = t / nGather;
    int f = t % nGather;
    out[((size_t) s * nColumns + column) * nGather + f] = value;
}

void sensitivityBasisStep(BatchSolverContext* ctx, DriverPass& drv) {
    SensitivityBasisTables& basis = ctx->sensitivityBasis;
    if (basis.nColumns <= 0 || basis.nGather <= 0) return;
    const int n = drv.n;
    const int nS = drv.nS;
    const int nColumns = basis.nColumns;
    const int nGather = basis.nGather;
    const int gatherThreads = nS * nGather;
    ctx->dSensitivityBasisOut.ensure(static_cast<size_t>(nS) * nColumns * nGather * sizeof(double));
    double* out = ctx->dSensitivityBasisOut.as<double>();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    // Every scenario of the call ended (no refill pause: the caller asks for none with a basis), in a mode that solves
    // the whole batch on factors it keeps: otherwise no column is solved.
    bool solvable = drv.nActive == 0 && (ctx->mode == GpuLinearSolveMode::UBATCH || ctx->mode == GpuLinearSolveMode::BATCHED_LU);
    if (!solvable) {
        for (int j = 0; j < nColumns; j++) {
            fillSensitivityBasisColumnBatch<<<grid(gatherThreads), BLOCK, 0, ctx->stream>>>(nS, nGather, j, nColumns, nan, out);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // The Jacobian of every slot at its converged state, factorized with the first column's solve, as the incremental
    // loops take their sensitivities: every slot active for the fill, the call's activity restored after.
    std::vector<char> ones(nS, 1);
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, ones.data(), nS * sizeof(char), cudaMemcpyHostToDevice));
    fillActiveBatch(ctx, drv, false);
    double* b = ctx->dB.as<double>();
    const size_t rhsBytes = static_cast<size_t>(nS) * n * sizeof(double);
    const bool participation = nColumns > basis.nUnits + basis.nWeighted;
    bool first = true;
    for (int j = 0; j < nColumns; j++) {
        if (j < basis.nUnits) {
            int row = basis.hUnitDeviceRows[j];
            if (row < 0) {                                       // an equation the structure has no row for: unsolved
                fillSensitivityBasisColumnBatch<<<grid(gatherThreads), BLOCK, 0, ctx->stream>>>(nS, nGather, j, nColumns, nan, out);
                continue;
            }
            setUnitRhsBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, row, b);
        } else if (j < basis.nUnits + basis.nWeighted) {
            broadcastColumnBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS,
                    basis.dWeighted.as<double>() + static_cast<size_t>(j - basis.nUnits) * n, b);
        } else if (participation && ctx->sensitivityParticipation.device() != nullptr
                   && ctx->sensitivityParticipation.width() == static_cast<size_t>(n)) {
            // each slot's own column, on device rows (setBatchSensitivityParticipation)
            CUDA_CHECK(cudaMemcpyAsync(b, ctx->sensitivityParticipation.device(), rhsBytes, cudaMemcpyDeviceToDevice, ctx->stream));
        } else {
            CUDA_CHECK(cudaMemsetAsync(b, 0, rhsBytes, ctx->stream));
        }
        CUDA_CHECK(cudaGetLastError());
        repairZeroDiagonalRightHandSide(ctx, nS, b);
        sensitivitySolve(ctx, drv, first);
        gatherSensitivityBasisBatch<<<grid(gatherThreads), BLOCK, 0, ctx->stream>>>(nS, n, nGather, basis.dGatherRows.as<int>(), j,
                nColumns, ctx->dDx.as<double>(), out);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, drv.active.data(), nS * sizeof(char), cudaMemcpyHostToDevice));   // as it was
}

}  // namespace olfgpu
