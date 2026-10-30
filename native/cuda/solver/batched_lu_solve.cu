// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See batched_lu_solve.cuh.
#include "batched_lu_solve.cuh"

#include <algorithm>
#include <numeric>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/stats.cuh"
#include "../common/trace.cuh"
#include "../diagnostics/batched_lu_compare.cuh"
#include "../lu/batched_lu.cuh"
#include "iterative_refinement.cuh"
#include "singleton_substitution.cuh"

namespace olfgpu {

void batchedLuNewtonSolve(BatchSolverContext* ctx, DriverPass& drv, int iteration) {
    const int n = drv.n;
    const int nS = drv.nS;
    BatchedLu& lu = *ctx->batchedLu;
    std::vector<int> laneMap;
    laneMap.reserve(drv.nActive);
    for (int s = 0; s < nS; s++) {
        if (drv.active[s]) {
            laneMap.push_back(s);
        }
    }
    gWidthIterSum += static_cast<long long>(laneMap.size());        // the width actually factorized
    CUDA_CHECK(cudaMemsetAsync(ctx->dDx.p, 0, static_cast<size_t>(n) * nS * sizeof(double), ctx->stream));
    {
        AnalysisTimer at_(__LINE__, 'F');
        factorizeLanes(lu, ctx->dVals.as<double>(), laneMap.data(), static_cast<int>(laneMap.size()), ctx->stream);
    }
    {
        AnalysisTimer at_(__LINE__, 'S');
        solveLanes(lu, ctx->dB.as<double>(), ctx->dDx.as<double>(), ctx->stream);
    }
    if (traceOn(Trace::LU_COMPARE) && ctx->structure->nZeroImpedanceBranch == 0) {
        compareBatchedLuWithCudss(ctx, drv.outer, iteration);
    }
    if (ctx->structure->nZeroImpedanceBranch > 0) {
        // the cuDSS batch's refinement: its base steps on every lane inside the solve (r = b - J dx, J d = r,
        // dx += d), then the adaptive steps of the lanes still above the tolerance
        const int baseSteps = std::min(configuredIterativeRefinementBaseSteps(), configuredIterativeRefinementMaxSteps());
        for (int step = 0; step < baseSteps; step++) {
            residualBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, drv.nnz, ctx->structure->dRowPtr.as<int>(),
                    ctx->structure->dColIdx.as<int>(), ctx->dVals.as<double>(), ctx->dDx.as<double>(),
                    ctx->dB.as<double>(), ctx->dR.as<double>());
            {
                AnalysisTimer at_(__LINE__, 'S');
                solveLanes(lu, ctx->dR.as<double>(), ctx->dDelta.as<double>(), ctx->stream);
            }
            axpyLaneMaskBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, n * nS, 1.0, ctx->dDelta.as<double>(),
                    ctx->dActive.as<char>(), ctx->dDx.as<double>());
        }
        if (traceOn(Trace::LU_COMPARE)) {
            compareBatchedLuWithCudss(ctx, drv.outer, iteration);
        }
        refineLanes(ctx, nS, n, drv.nnz, ctx->dVals.as<double>(), ctx->dDx.as<double>(), ctx->dB.as<double>(),
                    ctx->dR.as<double>(), ctx->dDelta.as<double>(), ctx->dActive.as<char>(), baseSteps, [&]() {
            AnalysisTimer at_(__LINE__, 'S');
            solveLanes(lu, ctx->dR.as<double>(), ctx->dDelta.as<double>(), ctx->stream);
        });
    }
    AnalysisTimer at_(__LINE__, 'S');
    substituteSingletons(ctx, nS, ctx->dB.as<double>(), ctx->dDx.as<double>());
}

void batchedLuSensitivitySolve(BatchSolverContext* ctx, int nS, bool first) {
    BatchedLu& lu = *ctx->batchedLu;
    if (first) {
        std::vector<int> laneMap(nS);
        std::iota(laneMap.begin(), laneMap.end(), 0);
        AnalysisTimer at_(__LINE__, 'F');
        factorizeLanes(lu, ctx->dVals.as<double>(), laneMap.data(), nS, ctx->stream);
    }
    AnalysisTimer at_(__LINE__, 'S');
    solveLanes(lu, ctx->dB.as<double>(), ctx->dDx.as<double>(), ctx->stream);
    substituteSingletons(ctx, nS, ctx->dB.as<double>(), ctx->dDx.as<double>());
}

}  // namespace olfgpu
