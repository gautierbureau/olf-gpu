// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// How well the batched linear solve solved, and the adaptive per-lane iterative refinement (see
// linear_solver_settings.cuh for why the coupler systems need it).
#ifndef OLF_GPU_SOLVER_ITERATIVE_REFINEMENT_CUH
#define OLF_GPU_SOLVER_ITERATIVE_REFINEMENT_CUH

#include "../solver/driver_pass.cuh"
#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/stats.cuh"
#include "linear_solver_settings.cuh"

namespace olfgpu {

// A linear solve whose residual ratio ||J.dx-B|| / ||F|| exceeds this did NOT solve the system: the
// factorization is wrong (unpivoted LU on a structure that needs pivoting), not merely inaccurate.
static const double LIN_SOLVE_FAIL_RATIO = 1e-2;

__global__ void linResidScoreBatch(int n, int nS, const double* rp, const double* rm, const double* F,
                                   double* score);

__global__ void linResidRowBatch(int n, int nS, int nnz, const int* rowPtr, const int* colIdx,
                                 const double* vals, const double* dx, const double* F,
                                 double* rPlus, double* rMinus);

// Kernels the refinement launches.
__global__ void residualBatch(int n, int nS, int nnz, const int* rowPtr, const int* colIdx,
                              const double* vals, const double* dx, const double* b, double* r);

__global__ void iterativeRefinementScoreBatch(int n, int nS, const double* r, const double* b, const char* active, double* score);

__global__ void axpyLaneMaskBatch(int n, int total, double sign, const double* d, const char* lane, double* x);

// Adaptive iterative refinement (see configuredIterativeRefinementBaseSteps / configuredIterativeRefinementMaxSteps / iterativeRefinementTolerance), decided PER LANE. Each lane refines (r = b - J dx, J d = r,
// dx += d) while ITS OWN backward residual ratio is above tol and still falling, and undoes a step that did not
// lower it - what the same scenario does in a batch of one. The batched solve still runs over every lane; a lane
// that is done just does not take the correction. It used to be decided on the WORST lane and applied to all of
// them, so a scenario's dx depended on which matrices shared its batch: network.xiidm busbar JZDQ's first pass
// differed by 1.2e-8 pu between a 700- and a 175-wide chunk of the same full topology, and the outer loops carried
// that over the 1 MW slack-balance threshold - 0.125 MW off the CPU in one chunk, 1.7e-8 MW in another.
template <typename SolveFn>
void refineLanes(BatchSolverContext* ctx, int lanes, int n, int nnz, const double* vals, double* dx,
                        const double* b, double* r, const double* delta, const char* active, SolveFn solve) {
    const int iterativeRefinementMaxSteps = configuredIterativeRefinementMaxSteps();
    const int iterativeRefinementBaseSteps = configuredIterativeRefinementBaseSteps();
    const double tol = iterativeRefinementTolerance();
    std::vector<double> prev(lanes), sc(lanes);
    std::vector<char> lane(lanes), undo(lanes);
    auto scores = [&](std::vector<double>& out) {
        residualBatch<<<grid(n * lanes), BLOCK, 0, ctx->stream>>>(n, lanes, nnz, ctx->structure->dRowPtr.as<int>(),
                ctx->structure->dColIdx.as<int>(), vals, dx, b, r);
        iterativeRefinementScoreBatch<<<grid(lanes), BLOCK, 0, ctx->stream>>>(n, lanes, r, b, active, ctx->dIterativeRefinementScore.as<double>());
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        CUDA_CHECK(cudaMemcpy(out.data(), ctx->dIterativeRefinementScore.as<double>(), lanes * sizeof(double), cudaMemcpyDeviceToHost));
        gIterativeRefinementChecks++;
    };
    scores(prev);
    int nLane = 0;
    for (int s = 0; s < lanes; s++) {
        lane[s] = prev[s] > tol;
        nLane += lane[s];
    }
    ctx->dIterativeRefinementScenarioMask.ensure(static_cast<size_t>(lanes));
    for (int step = iterativeRefinementBaseSteps; step < iterativeRefinementMaxSteps && nLane > 0; step++) {
        CUDA_CHECK(cudaMemcpy(ctx->dIterativeRefinementScenarioMask.p, lane.data(), lanes, cudaMemcpyHostToDevice));
        solve();                                              // delta = J^-1 r on every lane
        axpyLaneMaskBatch<<<grid(n * lanes), BLOCK, 0, ctx->stream>>>(n, n * lanes, 1.0, delta, ctx->dIterativeRefinementScenarioMask.as<char>(), dx);
        gIterativeRefinementExtraSolves++;
        scores(sc);
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
        if (nUndo > 0) {
            CUDA_CHECK(cudaMemcpy(ctx->dIterativeRefinementScenarioMask.p, undo.data(), lanes, cudaMemcpyHostToDevice));
            axpyLaneMaskBatch<<<grid(n * lanes), BLOCK, 0, ctx->stream>>>(n, n * lanes, -1.0, delta, ctx->dIterativeRefinementScenarioMask.as<char>(), dx);
        }
    }
}

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_ITERATIVE_REFINEMENT_CUH
