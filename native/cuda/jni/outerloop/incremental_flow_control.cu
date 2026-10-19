// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See incremental_flow_control.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/codes/gpu_linear_solve_mode.cuh"
#include "../common/codes/gpu_controlled_flow.cuh"
#include "../solver/driver_steps.cuh"
#include "incremental_transformer_voltage_control.cuh"
#include "../equations/scenario_overrides.cuh"
#include "flow_observation.cuh"
#include "sensitivity.cuh"
#include "incremental_flow_control.cuh"

namespace olfgpu {

// Per (scenario, controller): OLF's out-of-deadband test on the controlled flow (LIMITER: one-sided).
__global__ void flcCheckBatch(int nFlc, int nScenarios, int n, int nElements, int cbOffset, int kindMask,
                              const int* kind, const int* side, const double* targetValue, const double* halfDb,
                              const int* elem, const int* ctlPack,
                              const double* cbIn, const int* cbIx, int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal,
                              const char* disabled, const double* x, const char* rlActive,
                              char* outOfDb, double* diff) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nFlc * nScenarios) return;
    int c = tid % nFlc;
    int s = tid / nFlc;
    outOfDb[tid] = 0;
    if (!rlActive[s] || !(kindMask & controlledFlowMaskBit(kind[c]))) return;
    int e = ctlPack[c];
    if (disabled[(size_t) s * nElements + elem[c]] || disabled[(size_t) s * nElements + cbOffset + e]) return;
    const double* p = cbIn + e * 8;
    double pOv[8];
    if (findTapOverride(nR1Ov, r1OvSlotElem, r1OvVal, s, e, pOv)) p = pOv;
    double g[18];
    double flow = flcFlowAndGrad(side[c], kind[c], p, x + (size_t) s * n, cbIx + e * 12, g);
    double d = targetValue[c] - flow;
    diff[tid] = d;
    outOfDb[tid] = (kind[c] == GpuControlledFlow::CURRENT ? flow > targetValue[c] : fabs(d) > halfDb[c]) ? 1 : 0;
}

// Per scenario, for controller c: the controlled flow's gradient dotted with the unit-RHS solution at
// the controlled branch's variable columns (a non-variable r1/a1 column contributes nothing).
__global__ void flcGatherSensiBatch(int nScenarios, int n, int c, int nFlc, int cbOffset,
                                    const int* kind, const int* side, const int* ctlPack,
                                    const double* cbIn, const int* cbIx, int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal,
                                    const double* x, const double* dx, double* sensi) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    int e = ctlPack[c];
    const double* p = cbIn + e * 8;
    double pOv[8];
    if (findTapOverride(nR1Ov, r1OvSlotElem, r1OvVal, s, e, pOv)) p = pOv;
    const int* q = cbIx + e * 12;
    const double* dxs = dx + (size_t) s * n;
    double g[18];
    flcFlowAndGrad(side[c], kind[c], p, x + (size_t) s * n, q, g);
    double d[6] = { dxs[q[5]], dxs[q[7]], dxs[q[4]], dxs[q[6]], q[8] >= 0 ? dxs[q[8]] : 0.0, q[9] >= 0 ? dxs[q[9]] : 0.0 };
    double sv;
    if (kind[c] == GpuControlledFlow::CURRENT && g[0] == 1e300) {  // I = 0: hypot(dre·dx, dim·dx) (OLF's one-sided rule)
        double a = 0.0, b = 0.0;
        for (int k = 0; k < 5; k++) { a += g[6 + k] * d[k]; b += g[12 + k] * d[k]; }
        sv = hypot(a, b);
    } else {
        sv = 0.0;
        int nk = kind[c] == GpuControlledFlow::CURRENT ? 5 : 6;  // OLF's current sensitivity has no dr1 term
        for (int k = 0; k < nk; k++) sv += g[k] * d[k];
    }
    sensi[(size_t) s * nFlc + c] = sv;
}

// OLF's tap decisions per (scenario, controller): RPC — deltaR1 = diffQ / sensi, closest r1 within
// maxTapShift (no sensitivity gate); phase CONTROLLER — |sensi·toRadians(1)| > SENSI_EPS, da = dp / sensi,
// closest a1 (MAX_TAP_SHIFT unbounded); phase LIMITER — same gate, da = (target − i) / sensi, then
// FirstTapPositionAboveFinder: walk tap by tap in the shift's direction while the remaining shift still
// reaches past the next tap's a1. Ratchet as everywhere (updateAllowedDirection, MAX_DIRECTION_CHANGE 3).
__global__ void flcDecisionBatch(int nFlc, int nScenarios, int kindMask,
                                 const int* kind, const int* pinCol, const int* tapPtr, const double* tapData, const int* maxShift,
                                 const char* outOfDb, const double* diff, const double* sensi,
                                 int* curTap, int* ratDir, int* ratCount, int* ratAllowed, char* moved) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nFlc * nScenarios) return;
    int c = tid % nFlc;
    int s = tid / nFlc;
    if (!outOfDb[tid] || !(kindMask & controlledFlowMaskBit(kind[c]))) return;
    double sv = sensi[tid];
    if (kind[c] != GpuControlledFlow::REACTIVE_POWER && fabs(sv * 0.017453292519943295) <= 1e-6) return;  // SENSI_EPS on the per-degree value
    if (sv == 0.0) return;
    double delta = diff[tid] / sv;
    int cur = curTap[tid];
    int base = tapPtr[c];
    int nTaps = tapPtr[c + 1] - base;
    int col = pinCol[c];
    int lo = ratAllowed[tid] > 0 ? cur : 0;
    int hi = ratAllowed[tid] < 0 ? cur : nTaps - 1;
    int best = cur;
    if (kind[c] != GpuControlledFlow::CURRENT) {  // ClosestTapPositionFinder
        double newV = tapData[(size_t) (base + cur) * 8 + col] + delta;
        double bestDist = fabs(delta);
        for (int i = lo; i <= hi; i++) {
            if (abs(i - cur) > maxShift[c]) continue;
            double dist = fabs(newV - tapData[(size_t) (base + i) * 8 + col]);
            if (dist < bestDist) { best = i; bestDist = dist; }
        }
    } else {                                                 // FirstTapPositionAboveFinder
        double remaining = delta;
        int last = nTaps - 1;
        while (true) {
            int next = -1;                                   // OLF nextTapPositionIndex
            if (remaining != 0.0) {
                double vc = tapData[(size_t) (base + best) * 8 + col];
                auto descending = [&](int n1, int n2) {
                    double a = tapData[(size_t) (base + n1) * 8 + col], b = tapData[(size_t) (base + n2) * 8 + col];
                    return remaining > 0 ? a < b : a > b;
                };
                (void) vc;
                if (best == 0 && last >= 1 && descending(1, 0)) next = -1;
                else if (best > 0 && descending(best, best - 1)) next = best - 1;
                else if (best < last && descending(best, best + 1)) next = best + 1;
            }
            if (next < 0 || next < lo || next > hi) break;
            double value = tapData[(size_t) (base + best) * 8 + col];
            double nextValue = tapData[(size_t) (base + next) * 8 + col];
            best = next;
            if ((remaining < 0 && value + remaining > nextValue) || (remaining > 0 && value + remaining < nextValue)) break;
            remaining -= nextValue - value;
        }
    }
    if (best == cur) return;
    curTap[tid] = best;
    int direction = best > cur ? 1 : -1;
    if (ratCount[tid] < 3) {
        if (ratDir[tid] != 0 && ratDir[tid] != direction) ratCount[tid]++;
        ratDir[tid] = direction;
    } else {
        ratAllowed[tid] = direction;
    }
    moved[s] = 1;
}

// Per (scenario, controller) whose tap moved: pin the identity row (rho or a1) at the tap's value.
__global__ void flcApplyBatch(int nFlc, int nScenarios, int n, const int* pinRow, const int* pinCol,
                              const int* tapPtr, const double* tapData, const int* curTap, int* prevTap,
                              double* target, double* x) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nFlc * nScenarios) return;
    int c = tid % nFlc;
    int s = tid / nFlc;
    int t = curTap[tid];
    if (t == prevTap[tid]) return;
    double v = tapData[(size_t) (tapPtr[c] + t) * 8 + pinCol[c]];
    target[(size_t) s * n + pinRow[c]] = v;
    x[(size_t) s * n + pinRow[c]] = v;
    prevTap[tid] = t;
}

// Per-solve reset. Incremental flow control: base taps, fresh ratchets.
void flcResetSolve(BatchSolverContext* ctx, DriverPass& drv) {
    const int& nS = drv.nS;
    size_t nst = static_cast<size_t>(nS) * ctx->nFlc;
    std::vector<int> cur(nst);
    for (int s2 = 0; s2 < nS; s2++) for (int c = 0; c < ctx->nFlc; c++) cur[static_cast<size_t>(s2) * ctx->nFlc + c] = ctx->hFlcBaseTap[c];
    CUDA_CHECK(cudaMemcpy(ctx->dFlcCurTap.p, cur.data(), nst * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->dFlcPrevTap.p, cur.data(), nst * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ctx->dFlcRatDir.p, 0, nst * sizeof(int)));
    CUDA_CHECK(cudaMemset(ctx->dFlcRatCount.p, 0, nst * sizeof(int)));
    CUDA_CHECK(cudaMemset(ctx->dFlcRatAllowed.p, 0, nst * sizeof(int)));
}

void flcStep(BatchSolverContext* ctx, DriverPass& drv, int kindMask) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    std::vector<char>& active = drv.active;
    int nF = ctx->nFlc;
    if (nF <= 0) return;
    if (ctx->mode != GpuLinearSolveMode::UBATCH && ctx->mode != GpuLinearSolveMode::ROBUST) {
        static bool warned = false;
        if (!warned) { std::fprintf(stderr, "in-kernel flow control needs a UBATCH (3) or robust (6) mode; mode %d runs it as stable\n", ctx->mode); warned = true; }
        return;
    }
    size_t nst = static_cast<size_t>(nS) * nF;
    flcCheckBatch<<<grid(static_cast<int>(nst)), BLOCK, 0, ctx->stream>>>(nF, nS, n, ctx->nElements, ctx->cbOffset, kindMask,
            ctx->dFlcKind.as<int>(), ctx->dFlcSide.as<int>(), ctx->dFlcTargetValue.as<double>(), ctx->dFlcHalfDb.as<double>(),
            ctx->dFlcElem.as<int>(), ctx->dFlcCtlPack.as<int>(),
            ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(),
            ctx->dDisabled.as<char>(), ctx->dX.as<double>(), ctx->dRlActive.as<char>(),
            ctx->dFlcOutOfDb.as<char>(), ctx->dFlcDiff.as<double>());
    std::vector<char> outOfDb(nst);
    CUDA_CHECK(cudaMemcpyAsync(outOfDb.data(), ctx->dFlcOutOfDb.p, nst, cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    std::vector<char> needC(nF, 0);
    bool any = false;
    for (int s2 = 0; s2 < nS; s2++) for (int c = 0; c < nF; c++) if (outOfDb[static_cast<size_t>(s2) * nF + c]) { needC[c] = 1; any = true; }
    if (!any) return;
    std::vector<char> ones(nS, 1);
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, ones.data(), nS, cudaMemcpyHostToDevice));
    fillActiveBatch(ctx, drv, false);
    CUDA_CHECK(cudaMemsetAsync(ctx->dFlcSensi.p, 0, nst * sizeof(double), ctx->stream));
    bool first = true;
    for (int c = 0; c < nF; c++) {
        if (!needC[c]) continue;
        int row = ctx->st->rowMap[ctx->hFlcPinRow[c]];
        if (row < 0) continue;
        setUnitRhsBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, row, ctx->dB.as<double>());
        hazRepairRightHandSide(ctx, nS, ctx->dB.as<double>());
        sensitivitySolve(ctx, drv, first);
        flcGatherSensiBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(nS, n, c, nF, ctx->cbOffset,
                ctx->dFlcKind.as<int>(), ctx->dFlcSide.as<int>(), ctx->dFlcCtlPack.as<int>(),
                ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(),
                ctx->dX.as<double>(), ctx->dDx.as<double>(), ctx->dFlcSensi.as<double>());
    }
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, active.data(), nS, cudaMemcpyHostToDevice));
    flcDecisionBatch<<<grid(static_cast<int>(nst)), BLOCK, 0, ctx->stream>>>(nF, nS, kindMask,
            ctx->dFlcKind.as<int>(), ctx->dFlcPinCol.as<int>(), ctx->dFlcTapPtr.as<int>(), ctx->dFlcTapData.as<double>(), ctx->dFlcMaxShift.as<int>(),
            ctx->dFlcOutOfDb.as<char>(), ctx->dFlcDiff.as<double>(), ctx->dFlcSensi.as<double>(),
            ctx->dFlcCurTap.as<int>(), ctx->dFlcRatDir.as<int>(), ctx->dFlcRatCount.as<int>(), ctx->dFlcRatAllowed.as<int>(), ctx->dRlSwitched.as<char>());
    flcApplyBatch<<<grid(static_cast<int>(nst)), BLOCK, 0, ctx->stream>>>(nF, nS, n, ctx->dFlcPinRow.as<int>(), ctx->dFlcPinCol.as<int>(),
            ctx->dFlcTapPtr.as<int>(), ctx->dFlcTapData.as<double>(), ctx->dFlcCurTap.as<int>(), ctx->dFlcPrevTap.as<int>(),
            ctx->dTarget.as<double>(), ctx->dX.as<double>());
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    tvcRebuildOverrides(ctx);
    if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
        std::vector<int> cur(nst);
        CUDA_CHECK(cudaMemcpy(cur.data(), ctx->dFlcCurTap.p, nst * sizeof(int), cudaMemcpyDeviceToHost));
        int nOut = 0, nMoved = 0;
        for (char c : outOfDb) nOut += c ? 1 : 0;
        for (int s2 = 0; s2 < nS; s2++) for (int c = 0; c < nF; c++) if (cur[static_cast<size_t>(s2) * nF + c] != ctx->hFlcBaseTap[c]) nMoved++;
        std::fprintf(stderr, "FLOW_KERNEL kindMask=%d nS=%d nCtrl=%d outOfDeadband=%d pairsOffBaseTap=%d nR1Ov=%d\n", kindMask, nS, nF, nOut, nMoved, ctx->nR1Ov);
    }
}

}  // namespace olfgpu
