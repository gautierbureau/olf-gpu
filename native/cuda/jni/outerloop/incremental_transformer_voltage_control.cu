// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See incremental_transformer_voltage_control.cuh.
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
#include "../common/codes/gpu_pi_model_column.cuh"
#include "../solver/driver_steps.cuh"
#include "sensitivity.cuh"
#include "incremental_transformer_voltage_control.cuh"

namespace olfgpu {

// Per (scenario, controller): check if |targetV - V| > halfDeadband. Writes 1 to outOfDb
// for each out-of-deadband pair, 0 otherwise. Also writes diffV per pair for the tap decision.
__global__ void tvcDeadbandCheckBatch(int nTvc, int nScenarios, int n, int nElements,
        const int* tvcVRow, const double* tvcTargetV, const double* tvcHalfDb,
        const int* tvcElemIndex, const char* disabled,
        const double* x, const char* active,
        char* outOfDb, double* diffV) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nTvc * nScenarios) return;
    int c = tid % nTvc;
    int s = tid / nTvc;
    if (!active[s]) {
        outOfDb[tid] = 0;
        diffV[tid] = 0.0;
        return;
    }
    if (disabled[(size_t) s * nElements + tvcElemIndex[c]]) {    // outaged in this scenario
        outOfDb[tid] = 0;
        diffV[tid] = 0.0;                                    // never read as a group's remaining deviation
        return;
    }
    double v = x[(size_t) s * n + tvcVRow[c]];
    double dv = tvcTargetV[c] - v;
    if (fabs(dv) > tvcHalfDb[c]) {
        outOfDb[tid] = 1;
        diffV[tid] = dv;
    } else {
        outOfDb[tid] = 0;
        diffV[tid] = 0.0;
    }
}

// Per (scenario, controller): given the sensitivity dV/dρ, find the closest tap and move there.
// Updates curTap, writes the new r1 to newR1, and sets moved[s] = 1 if any scenario moved.
// Single-controller-per-bus simplification for now; the multi-controller round-robin needs the
// group structure and is a follow-up.
// OLF checkAndAdjustControlledBus, one thread per (scenario, controlled-bus group). One controller:
// adjustWithOneController (full diffV, maxTapShift). Several: adjustWithSeveralControllers — the
// one-tap-at-a-time round-robin in merged order, remainingDiffV tracked through each controller's
// sensitivity, until a pass moves nothing. Both apply MIN_SENSI_FILTER and the ratchet
// (ControllerContext.updateAllowedDirection, MAX_DIRECTION_CHANGE 3); updateTapPositionToReachNewR1
// = the closest r1 within the shift bound and the allowed range, strictly closer than the current one.
__device__ inline int tvcMoveTap(const int* tapPtr, const double* tapData, size_t st, int c, double deltaR1, int maxShift,
                                 int* curTap, int* ratchetDir, int* ratchetCount, int* ratchetAllowed) {
    int cur = curTap[st];
    int nTaps = tapPtr[c + 1] - tapPtr[c];
    int base = tapPtr[c];
    double curR1 = tapData[(size_t) (base + cur) * GpuPiModelColumn::COUNT + GpuPiModelColumn::R1];
    double targetR1 = curR1 + deltaR1;
    int lo = ratchetAllowed[st] > 0 ? cur : 0;
    int hi = ratchetAllowed[st] < 0 ? cur : nTaps - 1;
    int best = cur;
    double bestDist = fabs(targetR1 - curR1);
    for (int i = lo; i <= hi; i++) {
        if (abs(i - cur) > maxShift) continue;
        double dist = fabs(targetR1 - tapData[(size_t) (base + i) * GpuPiModelColumn::COUNT + GpuPiModelColumn::R1]);
        if (dist < bestDist) { best = i; bestDist = dist; }
    }
    if (best == cur) return 0;
    curTap[st] = best;
    int direction = best > cur ? 1 : -1;
    if (ratchetCount[st] < 3) {
        if (ratchetDir[st] != 0 && ratchetDir[st] != direction) ratchetCount[st]++;
        ratchetDir[st] = direction;
    } else {
        ratchetAllowed[st] = direction;
    }
    return direction;
}

__global__ void tvcGroupDecisionBatch(int nGroups, int nScenarios, int nTvc,
        const int* groupPtr, const int* groupCtrl,
        const char* outOfDb, const double* diffV, const double* sensi, const double* halfDb,
        const int* tapPtr, const double* tapData,
        int* curTap, int* ratchetDir, int* ratchetCount, int* ratchetAllowed,
        int maxTapShift, double minSensiFilter, char* moved) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nGroups * nScenarios) return;
    int g = tid % nGroups;
    int s = tid / nGroups;
    int k0 = groupPtr[g];
    int k1 = groupPtr[g + 1];
    // the group's live controllers (outOfDb is per controller: 0 when outaged or inside the band)
    int nLive = 0;
    int lastLive = -1;
    for (int k = k0; k < k1; k++) {
        int c = groupCtrl[k];
        if (outOfDb[(size_t) s * nTvc + c]) { nLive++; lastLive = c; }
    }
    if (nLive == 0) return;
    // OLF picks the path on the controllers LEFT after the outage (getMergedControllerElements minus the
    // disabled ones): one survivor of a shared group adjusts alone. The static group size chose the shared path
    // for it, and the shared path read the group's deviation off the FIRST member - the outaged one, whose
    // deadband check writes nothing: 0 on a fresh context (the survivor never moved), a stale value from an
    // earlier chunk on a reused one (chunk-dependent moves).
    if (nLive == 1) {                                        // adjustWithOneController
        size_t st = (size_t) s * nTvc + lastLive;
        double sv = sensi[st];
        if (fabs(sv) < minSensiFilter) return;
        if (tvcMoveTap(tapPtr, tapData, st, lastLive, diffV[st] / sv, maxTapShift,
                       curTap, ratchetDir, ratchetCount, ratchetAllowed) != 0) moved[s] = 1;
        return;
    }
    // adjustWithSeveralControllers: one tap per controller per pass, in merged order; the deviation and the
    // deadband are the controlled bus's, read off any live member (every live member of a group controls it)
    int firstLive = -1;
    for (int k = k0; k < k1 && firstLive < 0; k++) {
        if (outOfDb[(size_t) s * nTvc + groupCtrl[k]]) firstLive = groupCtrl[k];
    }
    size_t st0 = (size_t) s * nTvc + firstLive;
    double remaining = diffV[st0];
    double hdb = halfDb[firstLive];
    bool changed = true;
    bool any = false;
    while (changed) {
        changed = false;
        for (int k = k0; k < k1; k++) {
            int c = groupCtrl[k];
            size_t st = (size_t) s * nTvc + c;
            if (!outOfDb[st]) continue;                      // outaged member
            if (fabs(remaining) <= hdb) continue;
            double sv = sensi[st];
            if (fabs(sv) < minSensiFilter) continue;
            double prevR1 = tapData[(size_t) (tapPtr[c] + curTap[st]) * GpuPiModelColumn::COUNT + GpuPiModelColumn::R1];
            if (tvcMoveTap(tapPtr, tapData, st, c, remaining / sv, 1,
                           curTap, ratchetDir, ratchetCount, ratchetAllowed) != 0) {
                remaining -= (tapData[(size_t) (tapPtr[c] + curTap[st]) * GpuPiModelColumn::COUNT + GpuPiModelColumn::R1] - prevR1) * sv;
                changed = true;
                any = true;
            }
        }
    }
    if (any) moved[s] = 1;
}

// Per scenario: read the controlled bus's dV out of the unit-RHS solution for controller c.
__global__ void tvcGatherSensiBatch(int nScenarios, int n, int c, int nTvc,
                                    const int* tvcVRow, const double* dx, double* sensi) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    sensi[(size_t) s * nTvc + c] = dx[(size_t) s * n + tvcVRow[c]];
}

// Per (scenario, controller) whose tap moved this pass: pin the rho identity row's TARGET at the
// new tap's r1 and seed the rho STATE there — exactly what the host orchestrator writes before its
// warm re-solve (ovVal.add(row[6]); x0Full[rhoRow] = row[6]). Then remember the tap as applied.
__global__ void tvcApplyRhoBatch(int nTvc, int nScenarios, int n,
                                 const int* tvcRhoRow, const int* tapPtr, const double* tapData,
                                 const int* curTap, int* prevTap, double* target, double* x) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nTvc * nScenarios) return;
    int c = tid % nTvc;
    int s = tid / nTvc;
    int t = curTap[tid];
    if (t == prevTap[tid]) return;
    double r1 = tapData[(size_t) (tapPtr[c] + t) * GpuPiModelColumn::COUNT + GpuPiModelColumn::R1];
    target[(size_t) s * n + tvcRhoRow[c]] = r1;
    x[(size_t) s * n + tvcRhoRow[c]] = r1;
    prevTap[tid] = t;
}

/** Rebuild the device tap-override list = [every (scenario, controller) whose current tap differs
 *  from its base tap, as the tap's full 8-double pi row] ++ [the caller's own overrides]. The TVC
 *  entries go FIRST because findTapOverride returns the first (slot, elem) match, so a contingency
 *  tap action on the same transformer must not shadow the loop's move. Runs after every pass that
 *  moved a tap, and on a driver RESUME (the caller's list is re-uploaded at solve entry and would
 *  otherwise erase the moves the paused sequence had already made). */
void tvcRebuildOverrides(BatchSolverContext* ctx) {
    int nS = ctx->nScenarios;
    int nTvc = ctx->nTvc;
    std::vector<int> slot;
    std::vector<double> val;
    if (nTvc > 0) {
        std::vector<int> cur(static_cast<size_t>(nS) * nTvc);
        CUDA_CHECK(cudaMemcpy(cur.data(), ctx->dTvcCurTap.p, cur.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (int s = 0; s < nS; s++) {
            for (int c = 0; c < nTvc; c++) {
                int t = cur[static_cast<size_t>(s) * nTvc + c];
                if (t == ctx->hTvcBaseTap[c]) continue;
                slot.push_back(s);
                slot.push_back(ctx->hTvcElemIndex[c]);
                const double* row = &ctx->hTvcTapData[static_cast<size_t>(ctx->hTvcTapPtr[c] + t) * GpuPiModelColumn::COUNT];
                val.insert(val.end(), row, row + GpuPiModelColumn::COUNT);
            }
        }
    }
    if (ctx->nCph > 0) {                                     // the continuous phase loop's rounded/shifted taps
        std::vector<int> cur(static_cast<size_t>(nS) * ctx->nCph);
        CUDA_CHECK(cudaMemcpy(cur.data(), ctx->dCphTap.p, cur.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (int s = 0; s < nS; s++) {
            for (int c = 0; c < ctx->nCph; c++) {
                int t = cur[static_cast<size_t>(s) * ctx->nCph + c];
                if (t == ctx->hCphBaseTap[c]) continue;
                slot.push_back(s);
                slot.push_back(ctx->hCphElem[c]);
                const double* row = &ctx->hCphTapData[static_cast<size_t>(ctx->hCphTapPtr[c] + t) * GpuPiModelColumn::COUNT];
                val.insert(val.end(), row, row + GpuPiModelColumn::COUNT);
            }
        }
    }
    if (ctx->nFlc > 0) {                                     // the flow-observed loops' moved taps too
        std::vector<int> cur(static_cast<size_t>(nS) * ctx->nFlc);
        CUDA_CHECK(cudaMemcpy(cur.data(), ctx->dFlcCurTap.p, cur.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (int s = 0; s < nS; s++) {
            for (int c = 0; c < ctx->nFlc; c++) {
                int t = cur[static_cast<size_t>(s) * ctx->nFlc + c];
                if (t == ctx->hFlcBaseTap[c]) continue;
                slot.push_back(s);
                slot.push_back(ctx->hFlcElem[c]);
                const double* row = &ctx->hFlcTapData[static_cast<size_t>(ctx->hFlcTapPtr[c] + t) * GpuPiModelColumn::COUNT];
                val.insert(val.end(), row, row + GpuPiModelColumn::COUNT);
            }
        }
    }
    if (ctx->nTvd > 0) {                                     // the discrete TVC loop's fixed (rounded/clamped) taps
        std::vector<int> st(static_cast<size_t>(nS) * ctx->nTvd), cur(static_cast<size_t>(nS) * ctx->nTvd);
        CUDA_CHECK(cudaMemcpy(st.data(), ctx->dTvdMemState.p, st.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(cur.data(), ctx->dTvdMemTap.p, cur.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (int s = 0; s < nS; s++) {
            for (int m = 0; m < ctx->nTvd; m++) {
                size_t sm = static_cast<size_t>(s) * ctx->nTvd + m;
                if (st[sm] != 2 || cur[sm] == ctx->hTvdMemBaseTap[m]) continue;
                slot.push_back(s);
                slot.push_back(ctx->hTvdMemElem[m]);
                const double* row = &ctx->hTvdTapData[static_cast<size_t>(ctx->hTvdMemTapPtr[m] + cur[sm]) * GpuPiModelColumn::COUNT];
                val.insert(val.end(), row, row + GpuPiModelColumn::COUNT);
            }
        }
    }
    slot.insert(slot.end(), ctx->hR1OvBaseSlotElem.begin(), ctx->hR1OvBaseSlotElem.end());
    val.insert(val.end(), ctx->hR1OvBaseVal.begin(), ctx->hR1OvBaseVal.end());
    int nOv = static_cast<int>(slot.size() / 2);
    if (nOv > 0) {
        ctx->dR1OvSlotElem.ensure(slot.size() * sizeof(int));
        ctx->dR1OvVal.ensure(val.size() * sizeof(double));
        CUDA_CHECK(cudaMemcpy(ctx->dR1OvSlotElem.p, slot.data(), slot.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->dR1OvVal.p, val.data(), val.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    ctx->nR1Ov = nOv;
}

// Per-solve reset. Incremental TVC: every scenario starts at the base tap with a fresh ratchet.
void tvcResetSolve(BatchSolverContext* ctx, DriverPass& drv) {
    const int& nS = drv.nS;
    size_t nst = static_cast<size_t>(nS) * ctx->nTvc;
    std::vector<int> cur(nst);
    for (int s2 = 0; s2 < nS; s2++) {
        for (int c = 0; c < ctx->nTvc; c++) {
            cur[static_cast<size_t>(s2) * ctx->nTvc + c] = ctx->hTvcBaseTap[c];
        }
    }
    ctx->dTvcPrevTap.ensure(nst * sizeof(int));
    CUDA_CHECK(cudaMemcpy(ctx->dTvcCurTap.p, cur.data(), nst * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->dTvcPrevTap.p, cur.data(), nst * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ctx->dTvcRatchetDir.p, 0, nst * sizeof(int)));
    CUDA_CHECK(cudaMemset(ctx->dTvcRatchetCount.p, 0, nst * sizeof(int)));
    CUDA_CHECK(cudaMemset(ctx->dTvcRatchetAllowed.p, 0, nst * sizeof(int)));
    ctx->dTvcOutOfDb.ensure(nst * sizeof(char));
    ctx->dTvcDiffV.ensure(nst * sizeof(double));
    ctx->dTvcSensi.ensure(nst * sizeof(double));
}

void tvcStep(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    std::vector<char>& active = drv.active;
    int nTvc = ctx->nTvc;
    if (nTvc <= 0) return;
    if (ctx->mode != GpuLinearSolveMode::UBATCH && ctx->mode != GpuLinearSolveMode::ROBUST) {
        static bool warned = false;
        if (!warned) { std::fprintf(stderr, "in-kernel TVC needs a UBATCH (3) or robust (6) mode; mode %d runs it as stable\n", ctx->mode); warned = true; }
        return;
    }
    size_t nst = static_cast<size_t>(nS) * nTvc;
    tvcDeadbandCheckBatch<<<grid(static_cast<int>(nst)), BLOCK, 0, ctx->stream>>>(nTvc, nS, n, ctx->nElements,
            ctx->dTvcVRow.as<int>(), ctx->dTvcTargetV.as<double>(), ctx->dTvcHalfDb.as<double>(),
            ctx->dTvcElemIndex.as<int>(), ctx->dDisabled.as<char>(), ctx->dX.as<double>(), ctx->dOuterLoopCheckActive.as<char>(),
            ctx->dTvcOutOfDb.as<char>(), ctx->dTvcDiffV.as<double>());
    std::vector<char> outOfDb(nst);
    CUDA_CHECK(cudaMemcpyAsync(outOfDb.data(), ctx->dTvcOutOfDb.p, nst, cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    std::vector<char> needC(nTvc, 0);
    bool any = false;
    for (int s2 = 0; s2 < nS; s2++) {
        for (int c = 0; c < nTvc; c++) {
            if (outOfDb[static_cast<size_t>(s2) * nTvc + c]) { needC[c] = 1; any = true; }
        }
    }
    if (!any) return;                                    // every bus inside its deadband: STABLE
    // Sensitivities dV/drho at the CURRENT states: refill J for every scenario, factorize once,
    // then one unit-RHS solve per controller that has an out-of-deadband scenario.
    std::vector<char> ones(nS, 1);
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, ones.data(), nS, cudaMemcpyHostToDevice));
    fillActiveBatch(ctx, drv, false);
    CUDA_CHECK(cudaMemsetAsync(ctx->dTvcSensi.p, 0, nst * sizeof(double), ctx->stream));
    bool first = true;
    for (int c = 0; c < nTvc; c++) {
        if (!needC[c]) continue;
        int row = ctx->st->rowMap[ctx->hTvcRhoRow[c]];
        if (row < 0) continue;                           // rho row not an active equation: stays insensitive
        setUnitRhsBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, row, ctx->dB.as<double>());
        hazRepairRightHandSide(ctx, nS, ctx->dB.as<double>());
        sensitivitySolve(ctx, drv, first);
        tvcGatherSensiBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(nS, n, c, nTvc, ctx->dTvcVRow.as<int>(),
                ctx->dDx.as<double>(), ctx->dTvcSensi.as<double>());
    }
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, active.data(), nS, cudaMemcpyHostToDevice));   // as it was
    // Tap decision; a moved scenario is an UNSTABLE verdict (dOuterLoopUnstable), like every device loop.
    tvcGroupDecisionBatch<<<grid(ctx->nTvcGroups * nS), BLOCK, 0, ctx->stream>>>(ctx->nTvcGroups, nS, nTvc,
            ctx->dTvcGroupPtr.as<int>(), ctx->dTvcGroupCtrl.as<int>(),
            ctx->dTvcOutOfDb.as<char>(), ctx->dTvcDiffV.as<double>(), ctx->dTvcSensi.as<double>(), ctx->dTvcHalfDb.as<double>(),
            ctx->dTvcTapPtr.as<int>(), ctx->dTvcTapData.as<double>(),
            ctx->dTvcCurTap.as<int>(), ctx->dTvcRatchetDir.as<int>(), ctx->dTvcRatchetCount.as<int>(),
            ctx->dTvcRatchetAllowed.as<int>(), ctx->tvcMaxTapShift, 0.05, ctx->dOuterLoopUnstable.as<char>());
    // Apply: rho target + state seed on the moved pairs, then the pi rows through the tap-override list.
    tvcApplyRhoBatch<<<grid(static_cast<int>(nst)), BLOCK, 0, ctx->stream>>>(nTvc, nS, n,
            ctx->dTvcRhoRow.as<int>(), ctx->dTvcTapPtr.as<int>(), ctx->dTvcTapData.as<double>(),
            ctx->dTvcCurTap.as<int>(), ctx->dTvcPrevTap.as<int>(), ctx->dTarget.as<double>(), ctx->dX.as<double>());
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    tvcRebuildOverrides(ctx);
    if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
        std::vector<int> cur(nst), base(ctx->hTvcBaseTap);
        CUDA_CHECK(cudaMemcpy(cur.data(), ctx->dTvcCurTap.p, nst * sizeof(int), cudaMemcpyDeviceToHost));
        int nOut = 0, nMoved = 0;
        for (size_t k = 0; k < nst; k++) {
            nOut += outOfDb[k] ? 1 : 0;
        }
        for (int s2 = 0; s2 < nS; s2++) {
            for (int c = 0; c < nTvc; c++) {
                if (cur[static_cast<size_t>(s2) * nTvc + c] != base[c]) nMoved++;
            }
        }
        std::fprintf(stderr, "TVC_KERNEL nS=%d nCtrl=%d outOfDeadband=%d pairsOffBaseTap=%d nR1Ov=%d\n",
                     nS, nTvc, nOut, nMoved, ctx->nR1Ov);
        std::vector<double> dv(nst), se(nst);
        CUDA_CHECK(cudaMemcpy(dv.data(), ctx->dTvcDiffV.p, nst * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(se.data(), ctx->dTvcSensi.p, nst * sizeof(double), cudaMemcpyDeviceToHost));
        for (size_t k = 0; k < nst; k++) {
            if (outOfDb[k]) {
                std::fprintf(stderr, "TVC_KERNEL_PAIR s=%d c=%d diffV=%.9e sensi=%.9e tap=%d base=%d rhoRowMapped=%d\n",
                             (int) (k / nTvc), (int) (k % nTvc), dv[k], se[k], cur[k], base[k % nTvc],
                             ctx->st->rowMap[ctx->hTvcRhoRow[k % nTvc]]);
            }
        }
    }
}

}  // namespace olfgpu
