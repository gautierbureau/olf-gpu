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
#include "../common/codes/gpu_linear_solve_mode.cuh"
#include "../solver/driver_steps.cuh"
#include "../solver/scenario_override_tables.cuh"
#include "sensitivity.cuh"
#include "incremental_shunt_voltage_control.cuh"
#include "../common/trace.cuh"
#include "../solver/scenario_install.cuh"

namespace olfgpu {

// Per (scenario, controlled-bus group): OLF isOutOfDeadband on the group's controlled bus, with at
// least one live (not outaged) controller shunt.
__global__ void incrementalShuntVoltageControlDeadbandCheckBatch(int nGroups, int nScenarios, int n, int nElements,
                                      const int* groupControllerStart, const int* groupController,
                                      const int* vRow, const double* targetV, const double* halfTargetDeadband, const int* controllerElement,
                                      const char* disabled, const double* x, const char* outerLoopCheckActive,
                                      char* outOfDeadband, double* diffV) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nGroups * nScenarios) return;
    int g = tid % nGroups;
    int s = tid / nGroups;
    outOfDeadband[tid] = 0;
    if (!outerLoopCheckActive[s]) return;
    int c0 = groupController[groupControllerStart[g]];
    bool live = false;
    for (int k = groupControllerStart[g]; k < groupControllerStart[g + 1] && !live; k++) {
        int c = groupController[k];
        live = controllerElement[c] < 0 || !disabled[(size_t) s * nElements + controllerElement[c]];
    }
    double d = targetV[c0] - x[(size_t) s * n + vRow[c0]];
    diffV[tid] = d;
    outOfDeadband[tid] = (live && fabs(d) > halfTargetDeadband[c0]) ? 1 : 0;
}

__global__ void incrementalShuntVoltageControlGatherSensitivityBatch(int nScenarios, int n, int c, int nIncrementalShuntVoltageControlControllerShunts, const int* vRow,
                                    const double* dx, double* sensitivity) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    sensitivity[(size_t) s * nIncrementalShuntVoltageControlControllerShunts + c] = dx[(size_t) s * n + vRow[c]];
}

// OLF IncrementalShuntVoltageControlOuterLoop.adjustB, one thread per (scenario, group): round-robin
// over the |b|-sorted controller shunts, one section per sub-controller per pass, the per-round
// maxSectionShift cap, remainingDiffV tracked through the sensitivity; LfShunt.Controller.
// updateSectionB (closest section within one step and the ratchet's allowed range, strict <, only
// when closer than the current b); ControllerContext.updateAllowedDirection (MAX_DIRECTION_CHANGE 3).
__global__ void incrementalShuntVoltageControlAdjustBatch(int nGroups, int nScenarios, int n, int nElements, int nIncrementalShuntVoltageControlControllerShunts, int nSubs, int maxSectionShift,
                               const int* groupControllerStart, const int* groupController, const int* shuntCompensatorStart,
                               const int* sectionStart, const double* sectionB, const int* shuntCompensatorMinPosition,
                               const double* halfTargetDeadband, const int* controllerElement, const char* disabled,
                               const char* outOfDeadband, const double* diffV, const double* sensitivity,
                               int* pos, int* currentDirection, int* directionChangeCount, int* allowedDirection, int* shift,
                               char* switched) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nGroups * nScenarios) return;
    int g = tid % nGroups;
    int s = tid / nGroups;
    if (!outOfDeadband[tid]) return;
    int c0 = groupController[groupControllerStart[g]];
    double hdb = halfTargetDeadband[c0];
    double remaining = diffV[tid];
    for (int k = groupControllerStart[g]; k < groupControllerStart[g + 1]; k++) {   // per-round shift counters
        int c = groupController[k];
        for (int u = shuntCompensatorStart[c]; u < shuntCompensatorStart[c + 1]; u++) shift[(size_t) s * nSubs + u] = 0;
    }
    bool changed = true;
    bool moved = false;
    while (changed) {
        changed = false;
        for (int k = groupControllerStart[g]; k < groupControllerStart[g + 1]; k++) {
            int c = groupController[k];
            if (controllerElement[c] >= 0 && disabled[(size_t) s * nElements + controllerElement[c]]) continue;   // outaged
            double sv = sensitivity[(size_t) s * nIncrementalShuntVoltageControlControllerShunts + c];
            for (int u = shuntCompensatorStart[c]; u < shuntCompensatorStart[c + 1]; u++) {
                if (fabs(remaining) <= hdb) continue;
                size_t su = (size_t) s * nSubs + u;
                if (shift[su] > maxSectionShift) continue;   // reached max section shift this outer loop
                int cur = pos[su];
                int p0 = sectionStart[u];
                int nSec = sectionStart[u + 1] - p0;
                double prevB = sectionB[p0 + cur];
                double deltaB = remaining / sv;
                double newB = prevB + deltaB;
                int allowed = allowedDirection[su];
                int lo = allowed > 0 ? cur : shuntCompensatorMinPosition[u];
                int hi = allowed < 0 ? cur : nSec - 1;
                int best = cur;
                double smallest = fabs(deltaB);
                for (int p = lo; p <= hi; p++) {
                    if (abs(p - cur) > 1) continue;          // maxSectionShift = 1 per updateSectionB call
                    double dist = fabs(newB - sectionB[p0 + p]);
                    if (dist < smallest) { best = p; smallest = dist; }
                }
                if (best == cur) continue;
                pos[su] = best;
                int direction = best > cur ? 1 : -1;
                if (directionChangeCount[su] < 3) {                      // ControllerContext.updateAllowedDirection
                    if (currentDirection[su] != 0 && currentDirection[su] != direction) directionChangeCount[su]++;
                    currentDirection[su] = direction;
                } else {
                    allowedDirection[su] = direction;
                }
                shift[su]++;
                remaining -= (sectionB[p0 + best] - prevB) * sv;
                changed = true;
                moved = true;
            }
        }
    }
    if (moved) switched[s] = 1;
}

// Per (scenario, controller): the section total B fixes the SHUNT_B identity row (target + state
// seed), exactly the host orchestrator's ovVal.add(baseB + db) / x0Full[bRow] = newB.
__global__ void incrementalShuntVoltageControlApplyBatch(int nIncrementalShuntVoltageControlControllerShunts, int nScenarios, int n, const int* bRow, const int* shuntCompensatorStart,
                              const int* sectionStart, const double* sectionB, const int* shuntCompensatorBasePosition, const int* pos,
                              int nSubs, const double* baseB, double* appliedDeltaB, double* target, double* x) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nIncrementalShuntVoltageControlControllerShunts * nScenarios) return;
    int c = tid % nIncrementalShuntVoltageControlControllerShunts;
    int s = tid / nIncrementalShuntVoltageControlControllerShunts;
    double db = 0.0;
    for (int u = shuntCompensatorStart[c]; u < shuntCompensatorStart[c + 1]; u++) {
        db += sectionB[sectionStart[u] + pos[(size_t) s * nSubs + u]] - sectionB[sectionStart[u] + shuntCompensatorBasePosition[u]];
    }
    if (db == appliedDeltaB[tid]) return;
    double nb = baseB[tid] + db;
    target[(size_t) s * n + bRow[c]] = nb;
    x[(size_t) s * n + bRow[c]] = nb;
    appliedDeltaB[tid] = db;
}

// Per (installed scenario, controller): the base b, the SHUNT_B identity row's target as the install left it.
__global__ void incrementalShuntVoltageControlBaseBBatch(int nIncrementalShuntVoltageControlControllerShunts, int nScenarios, int n, const int* bRow,
                              const double* target, double* baseB, const char* installed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nIncrementalShuntVoltageControlControllerShunts * nScenarios) return;
    int c = tid % nIncrementalShuntVoltageControlControllerShunts;
    int s = tid / nIncrementalShuntVoltageControlControllerShunts;
    if (installed != nullptr && !installed[s]) return;   // a slot this install does not seed
    baseB[tid] = target[(size_t) s * n + bRow[c]];
}

// Per-solve reset. Incremental shunt VC: base sections, fresh ratchets, the identity row's target as the base b.
void incrementalShuntVoltageControlResetSolve(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    const size_t nu = static_cast<size_t>(ctx->incrementalShuntVoltageControl.nIncrementalShuntVoltageControlShuntCompensators);
    resetInstalledRows(ctx, ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlPosition.as<int>(), nu,
                       ctx->incrementalShuntVoltageControl.hIncrementalShuntVoltageControlShuntCompensatorBasePosition.data());
    resetInstalledRows(ctx, ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlCurrentDirection.as<int>(), nu, nullptr, 0);
    resetInstalledRows(ctx, ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlDirectionChangeCount.as<int>(), nu, nullptr, 0);
    resetInstalledRows(ctx, ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlAllowedDirection.as<int>(), nu, nullptr, 0);
    const size_t nc = static_cast<size_t>(ctx->incrementalShuntVoltageControl.nIncrementalShuntVoltageControlControllerShunts);
    resetInstalledRows(ctx, ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlAppliedDeltaB.as<double>(), nc, nullptr, 0.0);
    const int nc32 = static_cast<int>(nc);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    incrementalShuntVoltageControlBaseBBatch<<<grid(nc32 * nS), BLOCK, 0, ctx->stream>>>(nc32, nS, n, ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlShuntBRow.as<int>(),
            ctx->dTarget.as<double>(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlBaseB.as<double>(), installMaskOrNull(ctx));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
}

void incrementalShuntVoltageControlStep(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    std::vector<char>& active = drv.active;
    int nIncrementalShuntVoltageControlControllerShunts = ctx->incrementalShuntVoltageControl.nIncrementalShuntVoltageControlControllerShunts;
    if (nIncrementalShuntVoltageControlControllerShunts <= 0) return;
    if (ctx->mode != GpuLinearSolveMode::UBATCH && ctx->mode != GpuLinearSolveMode::ROBUST
            && ctx->mode != GpuLinearSolveMode::BATCHED_LU) {
        static bool warned = false;
        if (!warned) { std::fprintf(stderr, "in-kernel shunt VC needs a UBATCH (3), robust (6) or batched LU (7) mode; mode %d runs it as stable\n", ctx->mode); warned = true; }
        return;
    }
    int nG = ctx->incrementalShuntVoltageControl.nIncrementalShuntVoltageControlGroups;
    size_t nsg = static_cast<size_t>(nS) * nG;
    incrementalShuntVoltageControlDeadbandCheckBatch<<<grid(static_cast<int>(nsg)), BLOCK, 0, ctx->stream>>>(nG, nS, n, ctx->nElements,
            ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlGroupControllerStart.as<int>(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlGroupController.as<int>(),
            ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlControlledBusVRow.as<int>(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlTargetV.as<double>(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlHalfTargetDeadband.as<double>(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlElement.as<int>(),
            ctx->dDisabled.as<char>(), ctx->dX.as<double>(), ctx->dOuterLoopCheckActive.as<char>(),
            ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlOutOfDeadband.as<char>(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlDiffV.as<double>());
    std::vector<char> outOfDeadband(nsg);
    CUDA_CHECK(cudaMemcpyAsync(outOfDeadband.data(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlOutOfDeadband.p, nsg, cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    std::vector<int> gPtr(nG + 1), gCtrl(ctx->incrementalShuntVoltageControl.nIncrementalShuntVoltageControlControllerShunts);
    CUDA_CHECK(cudaMemcpy(gPtr.data(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlGroupControllerStart.p, gPtr.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gCtrl.data(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlGroupController.p, gCtrl.size() * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<char> needC(nIncrementalShuntVoltageControlControllerShunts, 0);
    bool any = false;
    for (int s2 = 0; s2 < nS; s2++) {
        for (int g = 0; g < nG; g++) {
            if (!outOfDeadband[static_cast<size_t>(s2) * nG + g]) continue;
            any = true;
            for (int k = gPtr[g]; k < gPtr[g + 1]; k++) needC[gCtrl[k]] = 1;
        }
    }
    if (!any) return;                                    // every controlled bus inside its deadband: STABLE
    std::vector<char> ones(nS, 1);
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, ones.data(), nS, cudaMemcpyHostToDevice));
    fillActiveBatch(ctx, drv, false);
    CUDA_CHECK(cudaMemsetAsync(ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlSensitivity.p, 0, static_cast<size_t>(nS) * nIncrementalShuntVoltageControlControllerShunts * sizeof(double), ctx->stream));
    bool first = true;
    for (int c = 0; c < nIncrementalShuntVoltageControlControllerShunts; c++) {
        if (!needC[c]) continue;
        int row = ctx->structure->rowMap[ctx->incrementalShuntVoltageControl.hIncrementalShuntVoltageControlShuntBRow[c]];
        if (row < 0) continue;
        setUnitRhsBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, row, ctx->dB.as<double>());
        repairZeroDiagonalRightHandSide(ctx, nS, ctx->dB.as<double>());
        sensitivitySolve(ctx, drv, first);
        incrementalShuntVoltageControlGatherSensitivityBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(nS, n, c, nIncrementalShuntVoltageControlControllerShunts, ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlControlledBusVRow.as<int>(),
                ctx->dDx.as<double>(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlSensitivity.as<double>());
    }
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, active.data(), nS, cudaMemcpyHostToDevice));
    incrementalShuntVoltageControlAdjustBatch<<<grid(static_cast<int>(nsg)), BLOCK, 0, ctx->stream>>>(nG, nS, n, ctx->nElements, nIncrementalShuntVoltageControlControllerShunts, ctx->incrementalShuntVoltageControl.nIncrementalShuntVoltageControlShuntCompensators,
            ctx->incrementalShuntVoltageControl.incrementalShuntVoltageControlMaxSectionShift, ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlGroupControllerStart.as<int>(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlGroupController.as<int>(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlShuntCompensatorStart.as<int>(),
            ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlSectionStart.as<int>(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlSectionB.as<double>(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlShuntCompensatorMinPosition.as<int>(),
            ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlHalfTargetDeadband.as<double>(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlElement.as<int>(), ctx->dDisabled.as<char>(),
            ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlOutOfDeadband.as<char>(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlDiffV.as<double>(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlSensitivity.as<double>(),
            ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlPosition.as<int>(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlCurrentDirection.as<int>(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlDirectionChangeCount.as<int>(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlAllowedDirection.as<int>(),
            ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlSectionShift.as<int>(), ctx->dOuterLoopUnstable.as<char>());
    incrementalShuntVoltageControlApplyBatch<<<grid(nIncrementalShuntVoltageControlControllerShunts * nS), BLOCK, 0, ctx->stream>>>(nIncrementalShuntVoltageControlControllerShunts, nS, n, ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlShuntBRow.as<int>(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlShuntCompensatorStart.as<int>(),
            ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlSectionStart.as<int>(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlSectionB.as<double>(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlShuntCompensatorBasePosition.as<int>(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlPosition.as<int>(),
            ctx->incrementalShuntVoltageControl.nIncrementalShuntVoltageControlShuntCompensators, ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlBaseB.as<double>(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlAppliedDeltaB.as<double>(), ctx->dTarget.as<double>(), ctx->dX.as<double>());
    publishShuntAdmittanceOverrides(ctx);
    if (traceOn(Trace::BATCH_TRACE)) {
        std::vector<int> pos(static_cast<size_t>(nS) * ctx->incrementalShuntVoltageControl.nIncrementalShuntVoltageControlShuntCompensators);
        CUDA_CHECK(cudaMemcpy(pos.data(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlPosition.p, pos.size() * sizeof(int), cudaMemcpyDeviceToHost));
        int nOut = 0, nMoved = 0;
        for (char c : outOfDeadband) nOut += c ? 1 : 0;
        for (int s2 = 0; s2 < nS; s2++) for (int u = 0; u < ctx->incrementalShuntVoltageControl.nIncrementalShuntVoltageControlShuntCompensators; u++) if (pos[static_cast<size_t>(s2) * ctx->incrementalShuntVoltageControl.nIncrementalShuntVoltageControlShuntCompensators + u] != ctx->incrementalShuntVoltageControl.hIncrementalShuntVoltageControlShuntCompensatorBasePosition[u]) nMoved++;
        std::fprintf(stderr, "SHUNT_KERNEL nS=%d nControllers=%d nGroups=%d outOfDeadband=%d subsOffBase=%d\n", nS, nIncrementalShuntVoltageControlControllerShunts, nG, nOut, nMoved);
    }
}

}  // namespace olfgpu
