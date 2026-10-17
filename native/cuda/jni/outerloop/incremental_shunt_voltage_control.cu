// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See incremental_shunt_voltage_control.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../solver/driver_steps.cuh"
#include "sensitivity.cuh"
#include "incremental_shunt_voltage_control.cuh"

namespace olfgpu {

// Per (scenario, controlled-bus group): OLF isOutOfDeadband on the group's controlled bus, with at
// least one live (not outaged) controller shunt.
__global__ void shcDeadbandCheckBatch(int nGroups, int nScenarios, int n, int nElements,
                                      const int* groupPtr, const int* groupCtrl,
                                      const int* vRow, const double* targetV, const double* halfDb, const int* elemAbs,
                                      const char* disabled, const double* x, const char* rlActive,
                                      char* outOfDb, double* diffV) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nGroups * nScenarios) return;
    int g = tid % nGroups;
    int s = tid / nGroups;
    outOfDb[tid] = 0;
    if (!rlActive[s]) return;
    int c0 = groupCtrl[groupPtr[g]];
    bool live = false;
    for (int k = groupPtr[g]; k < groupPtr[g + 1] && !live; k++) {
        int c = groupCtrl[k];
        live = elemAbs[c] < 0 || !disabled[(size_t) s * nElements + elemAbs[c]];
    }
    double d = targetV[c0] - x[(size_t) s * n + vRow[c0]];
    diffV[tid] = d;
    outOfDb[tid] = (live && fabs(d) > halfDb[c0]) ? 1 : 0;
}

__global__ void shcGatherSensiBatch(int nScenarios, int n, int c, int nShc, const int* vRow,
                                    const double* dx, double* sensi) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    sensi[(size_t) s * nShc + c] = dx[(size_t) s * n + vRow[c]];
}

// OLF IncrementalShuntVoltageControlOuterLoop.adjustB, one thread per (scenario, group): round-robin
// over the |b|-sorted controller shunts, one section per sub-controller per pass, the per-round
// maxSectionShift cap, remainingDiffV tracked through the sensitivity; LfShunt.Controller.
// updateSectionB (closest section within one step and the ratchet's allowed range, strict <, only
// when closer than the current b); ControllerContext.updateAllowedDirection (MAX_DIRECTION_CHANGE 3).
__global__ void shcAdjustBatch(int nGroups, int nScenarios, int n, int nElements, int nShc, int nSubs, int maxSectionShift,
                               const int* groupPtr, const int* groupCtrl, const int* subPtr,
                               const int* secPtr, const double* secB, const int* subMin,
                               const double* halfDb, const int* elemAbs, const char* disabled,
                               const char* outOfDb, const double* diffV, const double* sensi,
                               int* pos, int* ratDir, int* ratCount, int* ratAllowed, int* shift,
                               char* switched) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nGroups * nScenarios) return;
    int g = tid % nGroups;
    int s = tid / nGroups;
    if (!outOfDb[tid]) return;
    int c0 = groupCtrl[groupPtr[g]];
    double hdb = halfDb[c0];
    double remaining = diffV[tid];
    for (int k = groupPtr[g]; k < groupPtr[g + 1]; k++) {   // per-round shift counters
        int c = groupCtrl[k];
        for (int u = subPtr[c]; u < subPtr[c + 1]; u++) shift[(size_t) s * nSubs + u] = 0;
    }
    bool changed = true;
    bool moved = false;
    while (changed) {
        changed = false;
        for (int k = groupPtr[g]; k < groupPtr[g + 1]; k++) {
            int c = groupCtrl[k];
            if (elemAbs[c] >= 0 && disabled[(size_t) s * nElements + elemAbs[c]]) continue;   // outaged
            double sv = sensi[(size_t) s * nShc + c];
            for (int u = subPtr[c]; u < subPtr[c + 1]; u++) {
                if (fabs(remaining) <= hdb) continue;
                size_t su = (size_t) s * nSubs + u;
                if (shift[su] > maxSectionShift) continue;   // reached max section shift this outer loop
                int cur = pos[su];
                int p0 = secPtr[u];
                int nSec = secPtr[u + 1] - p0;
                double prevB = secB[p0 + cur];
                double deltaB = remaining / sv;
                double newB = prevB + deltaB;
                int allowed = ratAllowed[su];
                int lo = allowed > 0 ? cur : subMin[u];
                int hi = allowed < 0 ? cur : nSec - 1;
                int best = cur;
                double smallest = fabs(deltaB);
                for (int p = lo; p <= hi; p++) {
                    if (abs(p - cur) > 1) continue;          // maxSectionShift = 1 per updateSectionB call
                    double dist = fabs(newB - secB[p0 + p]);
                    if (dist < smallest) { best = p; smallest = dist; }
                }
                if (best == cur) continue;
                pos[su] = best;
                int direction = best > cur ? 1 : -1;
                if (ratCount[su] < 3) {                      // ControllerContext.updateAllowedDirection
                    if (ratDir[su] != 0 && ratDir[su] != direction) ratCount[su]++;
                    ratDir[su] = direction;
                } else {
                    ratAllowed[su] = direction;
                }
                shift[su]++;
                remaining -= (secB[p0 + best] - prevB) * sv;
                changed = true;
                moved = true;
            }
        }
    }
    if (moved) switched[s] = 1;
}

// Per (scenario, controller): the section total B pins the SHUNT_B identity row (target + state
// seed), exactly the host orchestrator's ovVal.add(baseB + db) / x0Full[bRow] = newB.
__global__ void shcApplyBatch(int nShc, int nScenarios, int n, const int* bRow, const int* subPtr,
                              const int* secPtr, const double* secB, const int* subBase, const int* pos,
                              int nSubs, const double* baseB, double* appliedDb, double* target, double* x) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nShc * nScenarios) return;
    int c = tid % nShc;
    int s = tid / nShc;
    double db = 0.0;
    for (int u = subPtr[c]; u < subPtr[c + 1]; u++) {
        db += secB[secPtr[u] + pos[(size_t) s * nSubs + u]] - secB[secPtr[u] + subBase[u]];
    }
    if (db == appliedDb[tid]) return;
    double nb = baseB[tid] + db;
    target[(size_t) s * n + bRow[c]] = nb;
    x[(size_t) s * n + bRow[c]] = nb;
    appliedDb[tid] = db;
}

/** Rebuild the device shunt-override list = the caller's entries ++ one (dG, dB) per (scenario,
 *  controller) whose sections moved off their base — additive deltas (applyShuntOverride SUMS every
 *  matching entry), so the loop's move composes on top of whatever the caller had. Re-done after
 *  every pass that moved a section and on a driver RESUME. */
void shcRebuildOverrides(BatchSolverContext* ctx) {
    int nS = ctx->nScenarios;
    std::vector<int> pos(static_cast<size_t>(nS) * ctx->nShcSubs);
    CUDA_CHECK(cudaMemcpy(pos.data(), ctx->dShcPos.p, pos.size() * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> slot(ctx->hShOvBaseSlotElem);
    std::vector<double> val(ctx->hShOvBaseGb);
    for (int s = 0; s < nS; s++) {
        for (int c = 0; c < ctx->nShc; c++) {
            if (ctx->hShcShOvElem[c] < 0) continue;          // no packed element: b rides SHUNT_B only
            double db = 0.0, dg = 0.0;
            for (int u = ctx->hShcSubPtr[c]; u < ctx->hShcSubPtr[c + 1]; u++) {
                int p0 = ctx->hShcSecPtr[u];
                int pc = pos[static_cast<size_t>(s) * ctx->nShcSubs + u];
                db += ctx->hShcSecB[p0 + pc] - ctx->hShcSecB[p0 + ctx->hShcSubBase[u]];
                dg += ctx->hShcSecG[p0 + pc] - ctx->hShcSecG[p0 + ctx->hShcSubBase[u]];
            }
            if (db == 0.0 && dg == 0.0) continue;
            slot.push_back(s);
            slot.push_back(ctx->hShcShOvElem[c]);
            val.push_back(dg);
            val.push_back(db);
        }
    }
    int nOv = static_cast<int>(slot.size() / 2);
    if (nOv > 0) {
        ctx->dShOvSlotElem.ensure(slot.size() * sizeof(int));
        ctx->dShOvGb.ensure(val.size() * sizeof(double));
        CUDA_CHECK(cudaMemcpy(ctx->dShOvSlotElem.p, slot.data(), slot.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->dShOvGb.p, val.data(), val.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    ctx->nShOv = nOv;
}

// Per-solve reset. Incremental shunt VC: base sections, fresh ratchets, the identity row's target as the base b.
void shcResetSolve(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    size_t nsu = static_cast<size_t>(nS) * ctx->nShcSubs;
    std::vector<int> pos(nsu);
    for (int s2 = 0; s2 < nS; s2++) {
        for (int u = 0; u < ctx->nShcSubs; u++) pos[static_cast<size_t>(s2) * ctx->nShcSubs + u] = ctx->hShcSubBase[u];
    }
    CUDA_CHECK(cudaMemcpy(ctx->dShcPos.p, pos.data(), nsu * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ctx->dShcRatDir.p, 0, nsu * sizeof(int)));
    CUDA_CHECK(cudaMemset(ctx->dShcRatCount.p, 0, nsu * sizeof(int)));
    CUDA_CHECK(cudaMemset(ctx->dShcRatAllowed.p, 0, nsu * sizeof(int)));
    size_t nsc = static_cast<size_t>(nS) * ctx->nShc;
    CUDA_CHECK(cudaMemset(ctx->dShcAppliedDb.p, 0, nsc * sizeof(double)));
    std::vector<double> tgt(static_cast<size_t>(n) * nS), bb(nsc);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(cudaMemcpy(tgt.data(), ctx->dTarget.p, tgt.size() * sizeof(double), cudaMemcpyDeviceToHost));
    for (int s2 = 0; s2 < nS; s2++) {
        for (int c = 0; c < ctx->nShc; c++) bb[static_cast<size_t>(s2) * ctx->nShc + c] = tgt[static_cast<size_t>(s2) * n + ctx->hShcBRow[c]];
    }
    CUDA_CHECK(cudaMemcpy(ctx->dShcBaseB.p, bb.data(), nsc * sizeof(double), cudaMemcpyHostToDevice));
}

void shcStep(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    std::vector<char>& active = drv.active;
    int nShc = ctx->nShc;
    if (nShc <= 0) return;
    if (ctx->mode != 3 && ctx->mode != 6) {
        static bool warned = false;
        if (!warned) { std::fprintf(stderr, "in-kernel shunt VC needs a UBATCH (3) or robust (6) mode; mode %d runs it as stable\n", ctx->mode); warned = true; }
        return;
    }
    int nG = ctx->nShcGroups;
    size_t nsg = static_cast<size_t>(nS) * nG;
    shcDeadbandCheckBatch<<<grid(static_cast<int>(nsg)), BLOCK, 0, ctx->stream>>>(nG, nS, n, ctx->nElements,
            ctx->dShcGroupPtr.as<int>(), ctx->dShcGroupCtrl.as<int>(),
            ctx->dShcVRow.as<int>(), ctx->dShcTargetV.as<double>(), ctx->dShcHalfDb.as<double>(), ctx->dShcElemAbs.as<int>(),
            ctx->dDisabled.as<char>(), ctx->dX.as<double>(), ctx->dRlActive.as<char>(),
            ctx->dShcOutOfDb.as<char>(), ctx->dShcDiffV.as<double>());
    std::vector<char> outOfDb(nsg);
    CUDA_CHECK(cudaMemcpyAsync(outOfDb.data(), ctx->dShcOutOfDb.p, nsg, cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    std::vector<int> gPtr(nG + 1), gCtrl(ctx->nShc);
    CUDA_CHECK(cudaMemcpy(gPtr.data(), ctx->dShcGroupPtr.p, gPtr.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gCtrl.data(), ctx->dShcGroupCtrl.p, gCtrl.size() * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<char> needC(nShc, 0);
    bool any = false;
    for (int s2 = 0; s2 < nS; s2++) {
        for (int g = 0; g < nG; g++) {
            if (!outOfDb[static_cast<size_t>(s2) * nG + g]) continue;
            any = true;
            for (int k = gPtr[g]; k < gPtr[g + 1]; k++) needC[gCtrl[k]] = 1;
        }
    }
    if (!any) return;                                    // every controlled bus inside its deadband: STABLE
    std::vector<char> ones(nS, 1);
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, ones.data(), nS, cudaMemcpyHostToDevice));
    fillActiveBatch(ctx, drv, false);
    CUDA_CHECK(cudaMemsetAsync(ctx->dShcSensi.p, 0, static_cast<size_t>(nS) * nShc * sizeof(double), ctx->stream));
    bool first = true;
    for (int c = 0; c < nShc; c++) {
        if (!needC[c]) continue;
        int row = ctx->st->rowMap[ctx->hShcBRow[c]];
        if (row < 0) continue;
        setUnitRhsBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, row, ctx->dB.as<double>());
        hazRepairRightHandSide(ctx, nS, ctx->dB.as<double>());
        sensitivitySolve(ctx, drv, first);
        shcGatherSensiBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(nS, n, c, nShc, ctx->dShcVRow.as<int>(),
                ctx->dDx.as<double>(), ctx->dShcSensi.as<double>());
    }
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, active.data(), nS, cudaMemcpyHostToDevice));
    shcAdjustBatch<<<grid(static_cast<int>(nsg)), BLOCK, 0, ctx->stream>>>(nG, nS, n, ctx->nElements, nShc, ctx->nShcSubs,
            ctx->shcMaxSectionShift, ctx->dShcGroupPtr.as<int>(), ctx->dShcGroupCtrl.as<int>(), ctx->dShcSubPtr.as<int>(),
            ctx->dShcSecPtr.as<int>(), ctx->dShcSecB.as<double>(), ctx->dShcSubMin.as<int>(),
            ctx->dShcHalfDb.as<double>(), ctx->dShcElemAbs.as<int>(), ctx->dDisabled.as<char>(),
            ctx->dShcOutOfDb.as<char>(), ctx->dShcDiffV.as<double>(), ctx->dShcSensi.as<double>(),
            ctx->dShcPos.as<int>(), ctx->dShcRatDir.as<int>(), ctx->dShcRatCount.as<int>(), ctx->dShcRatAllowed.as<int>(),
            ctx->dShcShift.as<int>(), ctx->dRlSwitched.as<char>());
    shcApplyBatch<<<grid(nShc * nS), BLOCK, 0, ctx->stream>>>(nShc, nS, n, ctx->dShcBRow.as<int>(), ctx->dShcSubPtr.as<int>(),
            ctx->dShcSecPtr.as<int>(), ctx->dShcSecB.as<double>(), ctx->dShcSubBase.as<int>(), ctx->dShcPos.as<int>(),
            ctx->nShcSubs, ctx->dShcBaseB.as<double>(), ctx->dShcAppliedDb.as<double>(), ctx->dTarget.as<double>(), ctx->dX.as<double>());
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    shcRebuildOverrides(ctx);
    if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
        std::vector<int> pos(static_cast<size_t>(nS) * ctx->nShcSubs);
        CUDA_CHECK(cudaMemcpy(pos.data(), ctx->dShcPos.p, pos.size() * sizeof(int), cudaMemcpyDeviceToHost));
        int nOut = 0, nMoved = 0;
        for (char c : outOfDb) nOut += c ? 1 : 0;
        for (int s2 = 0; s2 < nS; s2++) for (int u = 0; u < ctx->nShcSubs; u++) if (pos[static_cast<size_t>(s2) * ctx->nShcSubs + u] != ctx->hShcSubBase[u]) nMoved++;
        std::fprintf(stderr, "SHUNT_KERNEL nS=%d nCtrl=%d nGroups=%d outOfDeadband=%d subsOffBase=%d nShOv=%d\n", nS, nShc, nG, nOut, nMoved, ctx->nShOv);
    }
}

}  // namespace olfgpu
