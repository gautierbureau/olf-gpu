// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// How well the batched linear solve solved, and the adaptive per-lane iterative refinement (see
// linear_solver_settings.cuh for why the coupler systems need it).
#ifndef OLF_GPU_SOLVER_ITERATIVE_REFINEMENT_CUH
#define OLF_GPU_SOLVER_ITERATIVE_REFINEMENT_CUH

#include <algorithm>
#include <cmath>

#include "../common/batch_layout.cuh"
#include "../solver/driver_pass.cuh"
#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/stats.cuh"
#include "linear_solver_settings.cuh"

namespace olfgpu {

// A linear solve whose residual ratio ||J.dx-B|| / ||F|| exceeds this did NOT solve the system: the
// factorization is wrong (unpivoted LU on a structure that needs pivoting), not merely inaccurate.
static const double LIN_SOLVE_FAIL_RATIO = 1e-2;

// The per-lane norm kernels below run one block of LANE_NORM_BLOCK threads per lane: grid = the lane count.
constexpr int LANE_NORM_BLOCK = 256;

__global__ void linResidScoreBatch(int n, int nS, const double* rp, const double* rm, const double* F,
                                   double* score);

__global__ void linResidRowBatch(int n, int nS, int nnz, const int* rowPtr, const int* colIdx,
                                 const double* vals, const double* dx, const double* F,
                                 double* rPlus, double* rMinus);

// linResidRowBatch with J's values laid out by layout.
void launchLinResidRows(cudaStream_t stream, int n, int nS, int nnz, const int* rowPtr, const int* colIdx,
                        const double* vals, BatchLayout layout, const double* dx, const double* F, double* rPlus,
                        double* rMinus);

// Kernels the refinement launches.
__global__ void residualBatch(int n, int nS, int nnz, const int* rowPtr, const int* colIdx,
                              const double* vals, const double* dx, const double* b, const char* lanes, double* r);

// r = b - J dx (residualBatch) with J's values laid out by layout; b, dx and r scenario-major.
void launchResidual(cudaStream_t stream, int n, int nS, int nnz, const int* rowPtr, const int* colIdx,
                    const double* vals, BatchLayout layout, const double* dx, const double* b, const char* lanes,
                    double* r);

__global__ void iterativeRefinementScoreBatch(int n, int nS, const double* r, const double* b, const char* active, double* score);

__global__ void axpyLaneMaskBatch(int n, int total, double sign, const double* d, const char* lane, double* x);

// What the refinement uses of a solver: its stream, the systems' CSR pattern, its per-lane score and mask scratch
// (a batch context's, or a standalone LU's in a test), and the lanes it solves: null for every lane (cuDSS's batch
// solves them all, a retired scenario as an identity), the active ones for our batched LU, which factorizes those
// only - the residual of any other lane is never solved nor read.
struct RefinementContext {
    cudaStream_t stream;
    const int* rowPtr;
    const int* colIdx;
    DeviceBuf& score;
    DeviceBuf& mask;
    const char* solvedLanes;
    BatchLayout layout;                                      // the values'
};

// Adaptive iterative refinement (see configuredIterativeRefinementBaseSteps / configuredIterativeRefinementMaxSteps / iterativeRefinementTolerance), decided PER LANE. internalSteps: the refinement steps the solver
// already ran inside its own solve (cuDSS's base steps, which our batched LU runs the same way). Each lane refines (r = b - J dx, J d = r,
// dx += d) while ITS OWN backward residual ratio is above tol and still falling, and undoes a step that did not
// lower it - what the same scenario does in a batch of one. solve(refining, refiningMask) computes delta = J^-1 r on
// at least the lanes still refining (host and device masks): cuDSS's batch solves every lane, our batched LU those
// only; solveLanes(lane, nLane) may instead solve the lanes still refining (and possibly more) on a narrower batch and
// return true. A lane that is done does not take a correction, nor has its residual computed again. It used to be
// decided on the WORST lane and applied to all of them, so a scenario's dx depended on which matrices shared its
// batch: network.xiidm busbar JZDQ's first pass differed by 1.2e-8 pu between a 700- and a 175-wide chunk of the same
// full topology, and the outer loops carried that over the 1 MW slack-balance threshold - 0.125 MW off the CPU in one
// chunk, 1.7e-8 MW in another.
template <typename SolveFn, typename SolveLanesFn>
void refineLanes(const RefinementContext& ctx, int lanes, int n, int nnz, const double* vals, double* dx,
                        const double* b, double* r, const double* delta, const char* active, int internalSteps,
                        SolveFn solve, SolveLanesFn solveLanes) {
    const int iterativeRefinementMaxSteps = configuredIterativeRefinementMaxSteps();
    const double tol = iterativeRefinementTolerance();
    std::vector<double> prev(lanes), sc(lanes);
    std::vector<char> lane(lanes), undo(lanes);
    // r = b - J dx on the lanes of residualLanes, and the score of those of scoredLanes (every lane when null)
    auto scores = [&](const char* residualLanes, const char* scoredLanes, std::vector<double>& out) {
        launchResidual(ctx.stream, n, lanes, nnz, ctx.rowPtr, ctx.colIdx, vals, ctx.layout, dx, b, residualLanes, r);
        iterativeRefinementScoreBatch<<<lanes, LANE_NORM_BLOCK, 0, ctx.stream>>>(n, lanes, r, b, scoredLanes, ctx.score.as<double>());
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        CUDA_CHECK(cudaMemcpy(out.data(), ctx.score.as<double>(), lanes * sizeof(double), cudaMemcpyDeviceToHost));
        gIterativeRefinementChecks++;
    };
    static const bool refinementStats = traceOn(Trace::PHASE_TIME);
    scores(ctx.solvedLanes, active, prev);
    int nLane = 0;
    for (int s = 0; s < lanes; s++) {
        lane[s] = prev[s] > tol;
        nLane += lane[s];
        if (refinementStats && prev[s] > 0.0) {             // 0: a lane not scored (or solved exactly)
            const int decade = static_cast<int>(std::floor(std::log10(prev[s]))) + 17;
            gRefinementStartScoreBuckets[std::min(REFINEMENT_SCORE_BUCKETS - 1, std::max(0, decade))]++;
        }
    }
    ctx.mask.ensure(static_cast<size_t>(lanes));
    for (int step = internalSteps; step < iterativeRefinementMaxSteps && nLane > 0; step++) {
        if (refinementStats) {
            gRefinementLanesAtCorrection[std::min(REFINEMENT_TRACKED_CORRECTIONS - 1, step - internalSteps)] += nLane;
        }
        CUDA_CHECK(cudaMemcpy(ctx.mask.p, lane.data(), lanes, cudaMemcpyHostToDevice));
        if (!solveLanes(lane, nLane)) {
            solve(lane, ctx.mask.as<char>());                // delta = J^-1 r on the lanes refining
        }
        axpyLaneMaskBatch<<<grid(n * lanes), BLOCK, 0, ctx.stream>>>(n, n * lanes, 1.0, delta, ctx.mask.as<char>(), dx);
        gIterativeRefinementExtraSolves++;
        scores(ctx.mask.as<char>(), ctx.mask.as<char>(), sc); // only the lanes refining read theirs
        int nUndo = 0;
        nLane = 0;
        for (int s = 0; s < lanes; s++) {
            undo[s] = 0;
            if (!lane[s]) continue;
            if (!(sc[s] < prev[s])) {                         // no progress on this lane: undo its step, it is done
                undo[s] = 1;
                lane[s] = 0;
                nUndo++;
            } else {
                prev[s] = sc[s];
                lane[s] = sc[s] > tol;
                nLane += lane[s];
            }
        }
        if (refinementStats) {
            gRefinementUndone += nUndo;
        }
        if (nUndo > 0) {
            CUDA_CHECK(cudaMemcpy(ctx.mask.p, undo.data(), lanes, cudaMemcpyHostToDevice));
            axpyLaneMaskBatch<<<grid(n * lanes), BLOCK, 0, ctx.stream>>>(n, n * lanes, -1.0, delta, ctx.mask.as<char>(), dx);
        }
    }
}

template <typename SolveFn>
void refineLanes(const RefinementContext& ctx, int lanes, int n, int nnz, const double* vals, double* dx,
                        const double* b, double* r, const double* delta, const char* active, int internalSteps,
                        SolveFn solve) {
    refineLanes(ctx, lanes, n, nnz, vals, dx, b, r, delta, active, internalSteps, solve,
                [](const std::vector<char>&, int) { return false; });
}

template <typename SolveFn, typename SolveLanesFn>
void refineLanes(BatchSolverContext* ctx, int lanes, int n, int nnz, const double* vals, double* dx,
                        const double* b, double* r, const double* delta, const char* active, int internalSteps,
                        SolveFn solve, SolveLanesFn solveLanes) {
    refineLanes(RefinementContext{ctx->stream, ctx->structure->dRowPtr.as<int>(), ctx->structure->dColIdx.as<int>(),
                                  ctx->dIterativeRefinementScore, ctx->dIterativeRefinementScenarioMask, nullptr,
                                  ctx->layout},
                lanes, n, nnz, vals, dx, b, r, delta, active, internalSteps, solve, solveLanes);
}

template <typename SolveFn>
void refineLanes(BatchSolverContext* ctx, int lanes, int n, int nnz, const double* vals, double* dx,
                        const double* b, double* r, const double* delta, const char* active, int internalSteps,
                        SolveFn solve) {
    refineLanes(ctx, lanes, n, nnz, vals, dx, b, r, delta, active, internalSteps, solve,
                [](const std::vector<char>&, int) { return false; });
}

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_ITERATIVE_REFINEMENT_CUH
