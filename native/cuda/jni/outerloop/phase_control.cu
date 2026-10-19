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
#include "../equations/scenario_overrides.cuh"
#include "flow_observation.cuh"
#include "phase_control.cuh"

namespace olfgpu {

// One thread per (scenario, controller). iter[s] = the loop's check count in this solve. First check:
// every live CONTROLLER-mode control is switched OFF (row mode 3 -> 0, the identity pinned at the tap
// closest to the solved a1) and the scenario is UNSTABLE when it has any live controller (OLF returns
// UNSTABLE on a non-empty controller list, LIMITER-only included — a no-change re-solve). Later checks:
// a live LIMITER whose current exceeds the limit shifts one tap against the sign of dI/da1
// (shiftOneTapPositionToChangeA1). The tap's pi row follows through the shared r1 override rebuild.
__global__ void cphCheckBatch(int nCph, int nScenarios, int n, int nElements, int cbOffset,
                              const int* kind, const int* a1Row, const int* side, const double* limit, const int* elem,
                              const int* tapPtr, const double* tapData,
                              const double* cbIn, const int* cbIx, int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal,
                              const char* disabled, const char* rlActive, const int* iter,
                              int* tap, int* mode, double* target, double* x, char* switched) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nCph * nScenarios) return;
    int c = tid % nCph;
    int s = tid / nCph;
    if (!rlActive[s] || disabled[(size_t) s * nElements + elem[c]]) return;
    size_t ar = (size_t) s * n + a1Row[c];
    int base = tapPtr[c];
    int nTaps = tapPtr[c + 1] - base;
    if (iter[s] == 0) {
        if (kind[c] == GpuPhaseControlMode::CONTROLLER) {  // switch off + round a1 to the closest tap
            double a1 = x[ar];
            int best = tap[tid];
            for (int t = 0; t < nTaps; t++) {
                if (fabs(tapData[(size_t) (base + t) * 8 + 7] - a1) < fabs(tapData[(size_t) (base + best) * 8 + 7] - a1)) best = t;
            }
            tap[tid] = best;
            mode[ar] = 0;
            target[ar] = tapData[(size_t) (base + best) * 8 + 7];
            x[ar] = target[ar];
        }
        switched[s] = 1;                                     // OLF: UNSTABLE whenever a controller exists
        return;
    }
    if (kind[c] != GpuPhaseControlMode::LIMITER) return;
    int e = elem[c] - cbOffset;                              // LIMITER is local: the controlled branch is itself
    const double* p = cbIn + e * 8;
    double pOv[8];
    if (findTapOverride(nR1Ov, r1OvSlotElem, r1OvVal, s, e, pOv)) p = pOv;
    double g[18];
    double i = flcFlowAndGrad(side[c], GpuControlledFlow::CURRENT, p, x + (size_t) s * n, cbIx + e * 12, g);
    if (!(i > limit[c])) return;
    double dida1 = g[0] == 1e300 ? 0.0 : g[4];
    int dir = dida1 > 0 ? -1 : 1;                            // reduce I: move a1 against the slope
    int cur = tap[tid];
    double a1 = tapData[(size_t) (base + cur) * 8 + 7];
    int next = cur;
    if (cur + 1 < nTaps && ((dir > 0) == (tapData[(size_t) (base + cur + 1) * 8 + 7] > a1))) next = cur + 1;
    else if (cur - 1 >= 0 && ((dir > 0) == (tapData[(size_t) (base + cur - 1) * 8 + 7] > a1))) next = cur - 1;
    if (next == cur) return;
    tap[tid] = next;
    target[ar] = tapData[(size_t) (base + next) * 8 + 7];
    x[ar] = target[ar];
    switched[s] = 1;
}

// Per-solve reset. Continuous phase control: base taps, check count 0.
void cphResetSolve(BatchSolverContext* ctx, DriverPass& drv) {
    const int& nS = drv.nS;
    size_t nst = static_cast<size_t>(nS) * ctx->nCph;
    std::vector<int> cur(nst);
    for (int s2 = 0; s2 < nS; s2++) for (int c = 0; c < ctx->nCph; c++) cur[static_cast<size_t>(s2) * ctx->nCph + c] = ctx->hCphBaseTap[c];
    CUDA_CHECK(cudaMemcpy(ctx->dCphTap.p, cur.data(), nst * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->dCphPrevTap.p, cur.data(), nst * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ctx->dCphIter.p, 0, nS * sizeof(int)));
}

void cphStep(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    std::vector<char>& rlActive = drv.rlActive;
    int nC = ctx->nCph;
    if (nC <= 0) return;
    size_t nst = static_cast<size_t>(nS) * nC;
    cphCheckBatch<<<grid(static_cast<int>(nst)), BLOCK, 0, ctx->stream>>>(nC, nS, n, ctx->nElements, ctx->cbOffset,
            ctx->dCphKind.as<int>(), ctx->dCphA1Row.as<int>(), ctx->dCphSide.as<int>(), ctx->dCphLimit.as<double>(), ctx->dCphElem.as<int>(),
            ctx->dCphTapPtr.as<int>(), ctx->dCphTapData.as<double>(),
            ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(),
            ctx->dDisabled.as<char>(), ctx->dRlActive.as<char>(), ctx->dCphIter.as<int>(),
            ctx->dCphTap.as<int>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>(), ctx->dX.as<double>(), ctx->dRlSwitched.as<char>());
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    {                                                    // the loop's check count, per checked scenario
        std::vector<int> it(nS);
        CUDA_CHECK(cudaMemcpy(it.data(), ctx->dCphIter.p, nS * sizeof(int), cudaMemcpyDeviceToHost));
        for (int s2 = 0; s2 < nS; s2++) if (rlActive[s2]) it[s2]++;
        CUDA_CHECK(cudaMemcpy(ctx->dCphIter.p, it.data(), nS * sizeof(int), cudaMemcpyHostToDevice));
    }
    tvcRebuildOverrides(ctx);
    if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
        std::vector<int> cur(nst);
        CUDA_CHECK(cudaMemcpy(cur.data(), ctx->dCphTap.p, nst * sizeof(int), cudaMemcpyDeviceToHost));
        int nMoved = 0;
        for (int s2 = 0; s2 < nS; s2++) for (int c = 0; c < nC; c++) if (cur[static_cast<size_t>(s2) * nC + c] != ctx->hCphBaseTap[c]) nMoved++;
        std::fprintf(stderr, "CPHASE_KERNEL nS=%d nCtrl=%d pairsOffBaseTap=%d nR1Ov=%d\n", nS, nC, nMoved, ctx->nR1Ov);
    }
}

}  // namespace olfgpu
