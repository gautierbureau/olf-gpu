// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// PhaseControlOuterLoop (continuous mode), see phase_control.cuh.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/codes/gpu_controlled_flow.cuh"
#include "../common/codes/gpu_phase_control_mode.cuh"
#include "../common/codes/gpu_pi_model_column.cuh"
#include "../common/codes/gpu_row_mode.cuh"
#include "../equations/scenario_overrides.cuh"
#include "flow_observation.cuh"
#include "phase_control.cuh"
#include "../common/trace.cuh"

namespace olfgpu {

// One thread per (scenario, controller). iter[s] = the loop's check count in this solve. First check:
// every live CONTROLLER-mode control is switched OFF (row mode 3 -> 0, the identity held at the tap
// closest to the solved a1) and the scenario is UNSTABLE when it has any live controller (OLF returns
// UNSTABLE on a non-empty controller list, LIMITER-only included — a no-change re-solve). Later checks:
// a live LIMITER whose current exceeds the limit shifts one tap against the sign of dI/da1
// (shiftOneTapPositionToChangeA1). The tap's pi row follows through the shared r1 override rebuild.
__global__ void phaseControlCheckBatch(int nPhaseControlControllerBranches, int nScenarios, int n, int nElements, int closedBranchOffset,
                              const int* kind, const int* a1Row, const int* side, const double* limit, const int* elem,
                              const int* tapStart, const double* tapRows,
                              const double* closedBranchParameters, const int* closedBranchIndices, int nClosedBranchParameterOverrides, const int* closedBranchParameterOverrideScenarioElement, const double* closedBranchParameterOverrideValues,
                              const char* disabled, const char* outerLoopCheckActive, const int* iter,
                              int* tap, int* mode, double* target, double* x, char* switched) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nPhaseControlControllerBranches * nScenarios) return;
    int c = tid % nPhaseControlControllerBranches;
    int s = tid / nPhaseControlControllerBranches;
    if (!outerLoopCheckActive[s] || disabled[(size_t) s * nElements + elem[c]]) return;
    size_t ar = (size_t) s * n + a1Row[c];
    int base = tapStart[c];
    int nTaps = tapStart[c + 1] - base;
    if (iter[s] == 0) {
        if (kind[c] == GpuPhaseControlMode::CONTROLLER) {  // switch off + round a1 to the closest tap
            double a1 = x[ar];
            int best = tap[tid];
            for (int t = 0; t < nTaps; t++) {
                if (fabs(tapRows[(size_t) (base + t) * GpuPiModelColumn::COUNT + GpuPiModelColumn::A1] - a1) < fabs(tapRows[(size_t) (base + best) * GpuPiModelColumn::COUNT + GpuPiModelColumn::A1] - a1)) best = t;
            }
            tap[tid] = best;
            mode[ar] = GpuRowMode::FIXED_VARIABLE;
            target[ar] = tapRows[(size_t) (base + best) * GpuPiModelColumn::COUNT + GpuPiModelColumn::A1];
            x[ar] = target[ar];
        }
        switched[s] = 1;                                     // OLF: UNSTABLE whenever a controller exists
        return;
    }
    if (kind[c] != GpuPhaseControlMode::LIMITER) return;
    int e = elem[c] - closedBranchOffset;                              // LIMITER is local: the controlled branch is itself
    const double* p = closedBranchParameters + e * GpuPiModelColumn::COUNT;
    double pOv[8];
    if (findTapOverride(nClosedBranchParameterOverrides, closedBranchParameterOverrideScenarioElement, closedBranchParameterOverrideValues, s, e, pOv)) p = pOv;
    double i = controlledCurrent(side[c], p, x + (size_t) s * n, closedBranchIndices + e * 12);
    if (!(i > limit[c])) return;
    double dida1 = controlledCurrentDerivativeWithRespectToA1(side[c], p, x + (size_t) s * n, closedBranchIndices + e * 12);
    int dir = dida1 > 0 ? -1 : 1;                            // reduce I: move a1 against the slope
    int cur = tap[tid];
    double a1 = tapRows[(size_t) (base + cur) * GpuPiModelColumn::COUNT + GpuPiModelColumn::A1];
    int next = cur;
    if (cur + 1 < nTaps && ((dir > 0) == (tapRows[(size_t) (base + cur + 1) * GpuPiModelColumn::COUNT + GpuPiModelColumn::A1] > a1))) next = cur + 1;
    else if (cur - 1 >= 0 && ((dir > 0) == (tapRows[(size_t) (base + cur - 1) * GpuPiModelColumn::COUNT + GpuPiModelColumn::A1] > a1))) next = cur - 1;
    if (next == cur) return;
    tap[tid] = next;
    target[ar] = tapRows[(size_t) (base + next) * GpuPiModelColumn::COUNT + GpuPiModelColumn::A1];
    x[ar] = target[ar];
    switched[s] = 1;
}

// Per-solve reset. Continuous phase control: base taps, check count 0.
void phaseControlResetSolve(BatchSolverContext* ctx, DriverPass& drv) {
    const int& nS = drv.nS;
    size_t nst = static_cast<size_t>(nS) * ctx->nPhaseControlControllerBranches;
    std::vector<int> cur(nst);
    for (int s2 = 0; s2 < nS; s2++) for (int c = 0; c < ctx->nPhaseControlControllerBranches; c++) cur[static_cast<size_t>(s2) * ctx->nPhaseControlControllerBranches + c] = ctx->hPhaseControlBaseTap[c];
    CUDA_CHECK(cudaMemcpy(ctx->dPhaseControlTap.p, cur.data(), nst * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->dPhaseControlAppliedTap.p, cur.data(), nst * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ctx->dPhaseControlCheckCount.p, 0, nS * sizeof(int)));
}

void phaseControlStep(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    std::vector<char>& outerLoopCheckActive = drv.outerLoopCheckActive;
    int nC = ctx->nPhaseControlControllerBranches;
    if (nC <= 0) return;
    size_t nst = static_cast<size_t>(nS) * nC;
    phaseControlCheckBatch<<<grid(static_cast<int>(nst)), BLOCK, 0, ctx->stream>>>(nC, nS, n, ctx->nElements, ctx->closedBranchOffset,
            ctx->dPhaseControlMode.as<int>(), ctx->dPhaseControlA1Row.as<int>(), ctx->dPhaseControlControlledSide.as<int>(), ctx->dPhaseControlCurrentLimit.as<double>(), ctx->dPhaseControlElement.as<int>(),
            ctx->dPhaseControlTapStart.as<int>(), ctx->dPhaseControlTapRows.as<double>(),
            ctx->structure->closedBranch.parameters.as<double>(), ctx->structure->closedBranch.indices.as<int>(), ctx->nClosedBranchParameterOverrides, ctx->dClosedBranchParameterOverrideScenarioElement.as<int>(), ctx->dClosedBranchParameterOverrideValues.as<double>(),
            ctx->dDisabled.as<char>(), ctx->dOuterLoopCheckActive.as<char>(), ctx->dPhaseControlCheckCount.as<int>(),
            ctx->dPhaseControlTap.as<int>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>(), ctx->dX.as<double>(), ctx->dOuterLoopUnstable.as<char>());
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    {                                                    // the loop's check count, per checked scenario
        std::vector<int> it(nS);
        CUDA_CHECK(cudaMemcpy(it.data(), ctx->dPhaseControlCheckCount.p, nS * sizeof(int), cudaMemcpyDeviceToHost));
        for (int s2 = 0; s2 < nS; s2++) if (outerLoopCheckActive[s2]) it[s2]++;
        CUDA_CHECK(cudaMemcpy(ctx->dPhaseControlCheckCount.p, it.data(), nS * sizeof(int), cudaMemcpyHostToDevice));
    }
    rebuildTapOverrides(ctx);
    if (traceOn(Trace::BATCH_TRACE)) {
        std::vector<int> cur(nst);
        CUDA_CHECK(cudaMemcpy(cur.data(), ctx->dPhaseControlTap.p, nst * sizeof(int), cudaMemcpyDeviceToHost));
        int nMoved = 0;
        for (int s2 = 0; s2 < nS; s2++) for (int c = 0; c < nC; c++) if (cur[static_cast<size_t>(s2) * nC + c] != ctx->hPhaseControlBaseTap[c]) nMoved++;
        std::fprintf(stderr, "CPHASE_KERNEL nS=%d nControllers=%d pairsOffBaseTap=%d nClosedBranchParameterOverrides=%d\n", nS, nC, nMoved, ctx->nClosedBranchParameterOverrides);
    }
}

}  // namespace olfgpu
