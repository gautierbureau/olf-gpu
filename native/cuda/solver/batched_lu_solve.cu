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
                            ctx->dIterativeRefinementScenarioMask, ctx->layout};
}

// The internal refinement steps of a solve: cuDSS's base steps on a structure with couplers, none without.
int internalRefinementSteps(const BatchedLuSystems& systems) {
    return systems.couplers ? configuredIterativeRefinementBaseSteps() : ITERATIVE_REFINEMENT_STEPS_WITHOUT_COUPLERS;
}

// x = A^-1 b with steps internal refinement steps, solve(rhs, out) solving the lanes of lanes (every lane when null).
template <typename SolveFn>
void solveRefined(const BatchedLuSystems& systems, const double* b, double* x, double* r, double* d, const char* lanes,
                  int steps, SolveFn solve) {
    const int n = systems.n;
    const int nS = systems.nS;
    {
        AnalysisTimer at_(__LINE__, 'S');
        solve(b, x);
    }
    for (int step = 0; step < steps; step++) {
        launchResidual(systems.stream, n, nS, systems.nnz, systems.rowPtr, systems.colIdx, systems.vals, systems.layout,
                       x, b, lanes, r);
        {
            AnalysisTimer at_(__LINE__, 'S');
            solve(r, d);
        }
        if (lanes != nullptr) {
            axpyLaneMaskBatch<<<grid(n * nS), BLOCK, 0, systems.stream>>>(n, n * nS, 1.0, d, lanes, x);
        } else {
            axpy1Batch<<<grid(n * nS), BLOCK, 0, systems.stream>>>(n * nS, d, x);
        }
    }
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
    solveRefined(systems, b, x, r, d, lanes, internalRefinementSteps(systems), [&](const double* rhs, double* out) {
        solveLanes(*systems.lu, rhs, out, systems.stream);
    });
}

// The Newton step's own solve is a plain one, and its refinement is the adaptive one alone: every lane's backward
// residual is checked, and the lanes above the tolerance - on a coupler system, those whose repaired zero diagonals the
// static pivoting factored inaccurately - take corrections, up to the same configuredIterativeRefinementMaxSteps
// solves in all. The first correction is a plain solve too: it is the internal step those lanes took after the Newton
// solve, the same arithmetic on the same lanes. The later ones are solved with the internal steps (a correction step
// that raised a busbar lane's residual otherwise ended its refinement, see solveWithInternalRefinement). A lane the
// plain solve already solved to the tolerance takes no refinement: the internal step every lane took after it was half
// of a coupler system's triangular solves. On a busbar scope, where nearly every lane refines, the first correction
// solved with its own internal step cost a third solve per lane (network.xiidm busbar: 811 -> 1598 extra solves).
void solveNewtonStep(const BatchedLuSystems& systems, const double* b, double* dx, const std::function<void()>& afterSolve) {
    const int steps = systems.couplers ? 0 : internalRefinementSteps(systems);
    solveRefined(systems, b, dx, systems.residual, systems.correction, systems.lanes, steps,
                 [&](const double* rhs, double* out) {
        solveLanes(*systems.lu, rhs, out, systems.stream);
    });
    if (afterSolve) {
        afterSolve();
    }
    if (!systems.couplers) {
        return;
    }
    // the adaptive refinement of the lanes above the tolerance, each correction solved on those lanes only: the other
    // lanes would not take theirs
    int corrections = 0;
    refineLanes(RefinementContext{systems.stream, systems.rowPtr, systems.colIdx, systems.score, systems.mask,
                                  systems.lanes, systems.layout}, systems.nS,
                systems.n, systems.nnz, systems.vals, dx, b, systems.residual, systems.correction, systems.lanes,
                steps, [&](const std::vector<char>& refining, const char* refiningMask) {
        selectLanes(*systems.lu, refining, systems.stream);
        const int internal = corrections++ == 0 ? steps : internalRefinementSteps(systems);
        solveRefined(systems, systems.residual, systems.correction, systems.internalResidual, systems.internalCorrection,
                     refiningMask, internal, [&](const double* rhs, double* out) {
            solveSelectedLanes(*systems.lu, rhs, out, systems.stream);
        });
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
        factorizeLanes(*ctx->batchedLu, ctx->dVals.as<double>(), ctx->layout, laneMap.data(), static_cast<int>(laneMap.size()),
                       ctx->stream);
    }
    solveNewtonStep(systemsOf(ctx, nS, ctx->dActive.as<char>()), ctx->dB.as<double>(), ctx->dDx.as<double>(), [&]() {
        if (traceOn(Trace::LU_COMPARE)) {
            compareBatchedLuWithCudss(ctx, drv.outer, iteration);
        }
    });
    AnalysisTimer at_(__LINE__, 'S');
    substituteSingletons(ctx, ctx->dB.as<double>(), ctx->dDx.as<double>());
}

void batchedLuSensitivitySolve(BatchSolverContext* ctx, int nS, bool first) {
    if (first) {
        std::vector<int> laneMap(nS);
        std::iota(laneMap.begin(), laneMap.end(), 0);
        AnalysisTimer at_(__LINE__, 'F');
        factorizeLanes(*ctx->batchedLu, ctx->dVals.as<double>(), ctx->layout, laneMap.data(), nS, ctx->stream);
    }
    solveWithInternalRefinement(systemsOf(ctx, nS, nullptr), ctx->dB.as<double>(), ctx->dDx.as<double>(),
                                ctx->dR.as<double>(), ctx->dDelta.as<double>(), nullptr);
    AnalysisTimer at_(__LINE__, 'S');
    substituteSingletons(ctx, ctx->dB.as<double>(), ctx->dDx.as<double>());
}

}  // namespace olfgpu
