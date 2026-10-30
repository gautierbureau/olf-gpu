// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See batched_lu_solve.cuh.
#include "batched_lu_solve.cuh"

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
#include "state_vector_scaling.cuh"

namespace olfgpu {

namespace {

BatchedLuSystems systemsOf(BatchSolverContext* ctx, int nS, const char* lanes) {
    return BatchedLuSystems{ctx->batchedLu.get(), ctx->n, nS, ctx->nnz, ctx->structure->dRowPtr.as<int>(),
                            ctx->structure->dColIdx.as<int>(), ctx->dVals.as<double>(),
                            ctx->structure->nZeroImpedanceBranch > 0, lanes, ctx->stream, ctx->dR.as<double>(),
                            ctx->dDelta.as<double>(), ctx->dInternalRefinementResidual.as<double>(),
                            ctx->dInternalRefinementCorrection.as<double>(), ctx->dIterativeRefinementScore,
                            ctx->dIterativeRefinementScenarioMask};
}

}  // namespace

// cuDSS's SOLVE phase runs CUDSS_CONFIG_IR_N_STEPS (batch_context.cu: configuredIterativeRefinementBaseSteps() on a
// structure with couplers, ITERATIVE_REFINEMENT_STEPS_WITHOUT_COUPLERS otherwise) inside EVERY solve - the Newton step,
// each correction of the adaptive refinement, the sensitivity solves - and so does this. With them only after the
// Newton solve, a correction step that raised a lane's residual ended its refinement: on a network.xiidm busbar
// contingency whose factorization both solvers perturb (56 and 30 pivots at the epsilon) the step was taken with a
// backward ratio of 1.1 where cuDSS refines the same system to 3e-11.
void solveWithInternalRefinement(const BatchedLuSystems& systems, const double* b, double* x, double* r, double* d,
                                 const char* lanes) {
    const int n = systems.n;
    const int nS = systems.nS;
    {
        AnalysisTimer at_(__LINE__, 'S');
        solveLanes(*systems.lu, b, x, systems.stream);
    }
    const int steps = systems.couplers ? configuredIterativeRefinementBaseSteps() : ITERATIVE_REFINEMENT_STEPS_WITHOUT_COUPLERS;
    for (int step = 0; step < steps; step++) {
        residualBatch<<<grid(n * nS), BLOCK, 0, systems.stream>>>(n, nS, systems.nnz, systems.rowPtr, systems.colIdx,
                systems.vals, x, b, r);
        {
            AnalysisTimer at_(__LINE__, 'S');
            solveLanes(*systems.lu, r, d, systems.stream);
        }
        if (lanes != nullptr) {
            axpyLaneMaskBatch<<<grid(n * nS), BLOCK, 0, systems.stream>>>(n, n * nS, 1.0, d, lanes, x);
        } else {
            axpy1Batch<<<grid(n * nS), BLOCK, 0, systems.stream>>>(n * nS, d, x);
        }
    }
}

void solveNewtonStep(const BatchedLuSystems& systems, const double* b, double* dx, const std::function<void()>& afterSolve) {
    solveWithInternalRefinement(systems, b, dx, systems.residual, systems.correction, systems.lanes);
    if (afterSolve) {
        afterSolve();
    }
    if (!systems.couplers) {
        return;
    }
    // the cuDSS batch's adaptive refinement of the lanes still above the tolerance
    refineLanes(RefinementContext{systems.stream, systems.rowPtr, systems.colIdx, systems.score, systems.mask}, systems.nS,
                systems.n, systems.nnz, systems.vals, dx, b, systems.residual, systems.correction, systems.lanes,
                configuredIterativeRefinementBaseSteps(), [&]() {
        solveWithInternalRefinement(systems, systems.residual, systems.correction, systems.internalResidual,
                                    systems.internalCorrection, systems.lanes);
    });
}

void batchedLuNewtonSolve(BatchSolverContext* ctx, DriverPass& drv, int iteration) {
    const int n = drv.n;
    const int nS = drv.nS;
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
        factorizeLanes(*ctx->batchedLu, ctx->dVals.as<double>(), laneMap.data(), static_cast<int>(laneMap.size()), ctx->stream);
    }
    solveNewtonStep(systemsOf(ctx, nS, ctx->dActive.as<char>()), ctx->dB.as<double>(), ctx->dDx.as<double>(), [&]() {
        if (traceOn(Trace::LU_COMPARE)) {
            compareBatchedLuWithCudss(ctx, drv.outer, iteration);
        }
    });
    AnalysisTimer at_(__LINE__, 'S');
    substituteSingletons(ctx, nS, ctx->dB.as<double>(), ctx->dDx.as<double>());
}

void batchedLuSensitivitySolve(BatchSolverContext* ctx, int nS, bool first) {
    if (first) {
        std::vector<int> laneMap(nS);
        std::iota(laneMap.begin(), laneMap.end(), 0);
        AnalysisTimer at_(__LINE__, 'F');
        factorizeLanes(*ctx->batchedLu, ctx->dVals.as<double>(), laneMap.data(), nS, ctx->stream);
    }
    solveWithInternalRefinement(systemsOf(ctx, nS, nullptr), ctx->dB.as<double>(), ctx->dDx.as<double>(),
                                ctx->dR.as<double>(), ctx->dDelta.as<double>(), nullptr);
    AnalysisTimer at_(__LINE__, 'S');
    substituteSingletons(ctx, nS, ctx->dB.as<double>(), ctx->dDx.as<double>());
}

}  // namespace olfgpu
