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
#include "sensitivity.cuh"
#include "incremental_shunt_voltage_control.cuh"
#include "../common/trace.cuh"

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

/** Rebuild the device shunt-override list = the caller's entries ++ one (dG, dB) per (scenario,
 *  controller) whose sections moved off their base — additive deltas (applyShuntOverride SUMS every
 *  matching entry), so the loop's move composes on top of whatever the caller had. Re-done after
 *  every pass that moved a section and on a driver RESUME. */
void rebuildShuntAdmittanceOverrides(BatchSolverContext* ctx) {
    int nS = ctx->nScenarios;
    std::vector<int> sectionPosition(static_cast<size_t>(nS) * ctx->nIncrementalShuntVoltageControlShuntCompensators);
    CUDA_CHECK(cudaMemcpy(sectionPosition.data(), ctx->dIncrementalShuntVoltageControlPosition.p, sectionPosition.size() * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> scenarioElement(ctx->hShuntAdmittanceOverrideBaseScenarioElement);
    std::vector<double> delta(ctx->hShuntAdmittanceOverrideBaseDelta);
    for (int s = 0; s < nS; s++) {
        for (int c = 0; c < ctx->nIncrementalShuntVoltageControlControllerShunts; c++) {
            if (ctx->hIncrementalShuntVoltageControlShuntAdmittanceOverrideElement[c] < 0) continue;          // no packed element: b rides SHUNT_B only
            double deltaB = 0.0, deltaG = 0.0;
            for (int u = ctx->hIncrementalShuntVoltageControlShuntCompensatorStart[c]; u < ctx->hIncrementalShuntVoltageControlShuntCompensatorStart[c + 1]; u++) {
                int firstSection = ctx->hIncrementalShuntVoltageControlSectionStart[u];
                int position = sectionPosition[static_cast<size_t>(s) * ctx->nIncrementalShuntVoltageControlShuntCompensators + u];
                deltaB += ctx->hIncrementalShuntVoltageControlSectionB[firstSection + position]
                        - ctx->hIncrementalShuntVoltageControlSectionB[firstSection + ctx->hIncrementalShuntVoltageControlShuntCompensatorBasePosition[u]];
                deltaG += ctx->hIncrementalShuntVoltageControlSectionG[firstSection + position]
                        - ctx->hIncrementalShuntVoltageControlSectionG[firstSection + ctx->hIncrementalShuntVoltageControlShuntCompensatorBasePosition[u]];
            }
            if (deltaB == 0.0 && deltaG == 0.0) continue;
            scenarioElement.push_back(s);
            scenarioElement.push_back(ctx->hIncrementalShuntVoltageControlShuntAdmittanceOverrideElement[c]);
            delta.push_back(deltaG);
            delta.push_back(deltaB);
        }
    }
    int nOverrides = static_cast<int>(scenarioElement.size() / 2);
    if (nOverrides > 0) {
        ctx->dShuntAdmittanceOverrideScenarioElement.ensure(scenarioElement.size() * sizeof(int));
        ctx->dShuntAdmittanceOverrideDelta.ensure(delta.size() * sizeof(double));
        CUDA_CHECK(cudaMemcpy(ctx->dShuntAdmittanceOverrideScenarioElement.p, scenarioElement.data(), scenarioElement.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->dShuntAdmittanceOverrideDelta.p, delta.data(), delta.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    ctx->nShuntAdmittanceOverrides = nOverrides;
}

// Per-solve reset. Incremental shunt VC: base sections, fresh ratchets, the identity row's target as the base b.
void incrementalShuntVoltageControlResetSolve(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    size_t nsu = static_cast<size_t>(nS) * ctx->nIncrementalShuntVoltageControlShuntCompensators;
    std::vector<int> pos(nsu);
    for (int s2 = 0; s2 < nS; s2++) {
        for (int u = 0; u < ctx->nIncrementalShuntVoltageControlShuntCompensators; u++) pos[static_cast<size_t>(s2) * ctx->nIncrementalShuntVoltageControlShuntCompensators + u] = ctx->hIncrementalShuntVoltageControlShuntCompensatorBasePosition[u];
    }
    CUDA_CHECK(cudaMemcpy(ctx->dIncrementalShuntVoltageControlPosition.p, pos.data(), nsu * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ctx->dIncrementalShuntVoltageControlCurrentDirection.p, 0, nsu * sizeof(int)));
    CUDA_CHECK(cudaMemset(ctx->dIncrementalShuntVoltageControlDirectionChangeCount.p, 0, nsu * sizeof(int)));
    CUDA_CHECK(cudaMemset(ctx->dIncrementalShuntVoltageControlAllowedDirection.p, 0, nsu * sizeof(int)));
    size_t nsc = static_cast<size_t>(nS) * ctx->nIncrementalShuntVoltageControlControllerShunts;
    CUDA_CHECK(cudaMemset(ctx->dIncrementalShuntVoltageControlAppliedDeltaB.p, 0, nsc * sizeof(double)));
    std::vector<double> tgt(static_cast<size_t>(n) * nS), bb(nsc);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(cudaMemcpy(tgt.data(), ctx->dTarget.p, tgt.size() * sizeof(double), cudaMemcpyDeviceToHost));
    for (int s2 = 0; s2 < nS; s2++) {
        for (int c = 0; c < ctx->nIncrementalShuntVoltageControlControllerShunts; c++) bb[static_cast<size_t>(s2) * ctx->nIncrementalShuntVoltageControlControllerShunts + c] = tgt[static_cast<size_t>(s2) * n + ctx->hIncrementalShuntVoltageControlShuntBRow[c]];
    }
    CUDA_CHECK(cudaMemcpy(ctx->dIncrementalShuntVoltageControlBaseB.p, bb.data(), nsc * sizeof(double), cudaMemcpyHostToDevice));
}

void incrementalShuntVoltageControlStep(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    std::vector<char>& active = drv.active;
    int nIncrementalShuntVoltageControlControllerShunts = ctx->nIncrementalShuntVoltageControlControllerShunts;
    if (nIncrementalShuntVoltageControlControllerShunts <= 0) return;
    if (ctx->mode != GpuLinearSolveMode::UBATCH && ctx->mode != GpuLinearSolveMode::ROBUST) {
        static bool warned = false;
        if (!warned) { std::fprintf(stderr, "in-kernel shunt VC needs a UBATCH (3) or robust (6) mode; mode %d runs it as stable\n", ctx->mode); warned = true; }
        return;
    }
    int nG = ctx->nIncrementalShuntVoltageControlGroups;
    size_t nsg = static_cast<size_t>(nS) * nG;
    incrementalShuntVoltageControlDeadbandCheckBatch<<<grid(static_cast<int>(nsg)), BLOCK, 0, ctx->stream>>>(nG, nS, n, ctx->nElements,
            ctx->dIncrementalShuntVoltageControlGroupControllerStart.as<int>(), ctx->dIncrementalShuntVoltageControlGroupController.as<int>(),
            ctx->dIncrementalShuntVoltageControlControlledBusVRow.as<int>(), ctx->dIncrementalShuntVoltageControlTargetV.as<double>(), ctx->dIncrementalShuntVoltageControlHalfTargetDeadband.as<double>(), ctx->dIncrementalShuntVoltageControlElement.as<int>(),
            ctx->dDisabled.as<char>(), ctx->dX.as<double>(), ctx->dOuterLoopCheckActive.as<char>(),
            ctx->dIncrementalShuntVoltageControlOutOfDeadband.as<char>(), ctx->dIncrementalShuntVoltageControlDiffV.as<double>());
    std::vector<char> outOfDeadband(nsg);
    CUDA_CHECK(cudaMemcpyAsync(outOfDeadband.data(), ctx->dIncrementalShuntVoltageControlOutOfDeadband.p, nsg, cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    std::vector<int> gPtr(nG + 1), gCtrl(ctx->nIncrementalShuntVoltageControlControllerShunts);
    CUDA_CHECK(cudaMemcpy(gPtr.data(), ctx->dIncrementalShuntVoltageControlGroupControllerStart.p, gPtr.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gCtrl.data(), ctx->dIncrementalShuntVoltageControlGroupController.p, gCtrl.size() * sizeof(int), cudaMemcpyDeviceToHost));
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
    CUDA_CHECK(cudaMemsetAsync(ctx->dIncrementalShuntVoltageControlSensitivity.p, 0, static_cast<size_t>(nS) * nIncrementalShuntVoltageControlControllerShunts * sizeof(double), ctx->stream));
    bool first = true;
    for (int c = 0; c < nIncrementalShuntVoltageControlControllerShunts; c++) {
        if (!needC[c]) continue;
        int row = ctx->structure->rowMap[ctx->hIncrementalShuntVoltageControlShuntBRow[c]];
        if (row < 0) continue;
        setUnitRhsBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, row, ctx->dB.as<double>());
        repairZeroDiagonalRightHandSide(ctx, nS, ctx->dB.as<double>());
        sensitivitySolve(ctx, drv, first);
        incrementalShuntVoltageControlGatherSensitivityBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(nS, n, c, nIncrementalShuntVoltageControlControllerShunts, ctx->dIncrementalShuntVoltageControlControlledBusVRow.as<int>(),
                ctx->dDx.as<double>(), ctx->dIncrementalShuntVoltageControlSensitivity.as<double>());
    }
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, active.data(), nS, cudaMemcpyHostToDevice));
    incrementalShuntVoltageControlAdjustBatch<<<grid(static_cast<int>(nsg)), BLOCK, 0, ctx->stream>>>(nG, nS, n, ctx->nElements, nIncrementalShuntVoltageControlControllerShunts, ctx->nIncrementalShuntVoltageControlShuntCompensators,
            ctx->incrementalShuntVoltageControlMaxSectionShift, ctx->dIncrementalShuntVoltageControlGroupControllerStart.as<int>(), ctx->dIncrementalShuntVoltageControlGroupController.as<int>(), ctx->dIncrementalShuntVoltageControlShuntCompensatorStart.as<int>(),
            ctx->dIncrementalShuntVoltageControlSectionStart.as<int>(), ctx->dIncrementalShuntVoltageControlSectionB.as<double>(), ctx->dIncrementalShuntVoltageControlShuntCompensatorMinPosition.as<int>(),
            ctx->dIncrementalShuntVoltageControlHalfTargetDeadband.as<double>(), ctx->dIncrementalShuntVoltageControlElement.as<int>(), ctx->dDisabled.as<char>(),
            ctx->dIncrementalShuntVoltageControlOutOfDeadband.as<char>(), ctx->dIncrementalShuntVoltageControlDiffV.as<double>(), ctx->dIncrementalShuntVoltageControlSensitivity.as<double>(),
            ctx->dIncrementalShuntVoltageControlPosition.as<int>(), ctx->dIncrementalShuntVoltageControlCurrentDirection.as<int>(), ctx->dIncrementalShuntVoltageControlDirectionChangeCount.as<int>(), ctx->dIncrementalShuntVoltageControlAllowedDirection.as<int>(),
            ctx->dIncrementalShuntVoltageControlSectionShift.as<int>(), ctx->dOuterLoopUnstable.as<char>());
    incrementalShuntVoltageControlApplyBatch<<<grid(nIncrementalShuntVoltageControlControllerShunts * nS), BLOCK, 0, ctx->stream>>>(nIncrementalShuntVoltageControlControllerShunts, nS, n, ctx->dIncrementalShuntVoltageControlShuntBRow.as<int>(), ctx->dIncrementalShuntVoltageControlShuntCompensatorStart.as<int>(),
            ctx->dIncrementalShuntVoltageControlSectionStart.as<int>(), ctx->dIncrementalShuntVoltageControlSectionB.as<double>(), ctx->dIncrementalShuntVoltageControlShuntCompensatorBasePosition.as<int>(), ctx->dIncrementalShuntVoltageControlPosition.as<int>(),
            ctx->nIncrementalShuntVoltageControlShuntCompensators, ctx->dIncrementalShuntVoltageControlBaseB.as<double>(), ctx->dIncrementalShuntVoltageControlAppliedDeltaB.as<double>(), ctx->dTarget.as<double>(), ctx->dX.as<double>());
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    rebuildShuntAdmittanceOverrides(ctx);
    if (traceOn(Trace::BATCH_TRACE)) {
        std::vector<int> pos(static_cast<size_t>(nS) * ctx->nIncrementalShuntVoltageControlShuntCompensators);
        CUDA_CHECK(cudaMemcpy(pos.data(), ctx->dIncrementalShuntVoltageControlPosition.p, pos.size() * sizeof(int), cudaMemcpyDeviceToHost));
        int nOut = 0, nMoved = 0;
        for (char c : outOfDeadband) nOut += c ? 1 : 0;
        for (int s2 = 0; s2 < nS; s2++) for (int u = 0; u < ctx->nIncrementalShuntVoltageControlShuntCompensators; u++) if (pos[static_cast<size_t>(s2) * ctx->nIncrementalShuntVoltageControlShuntCompensators + u] != ctx->hIncrementalShuntVoltageControlShuntCompensatorBasePosition[u]) nMoved++;
        std::fprintf(stderr, "SHUNT_KERNEL nS=%d nControllers=%d nGroups=%d outOfDeadband=%d subsOffBase=%d nShuntAdmittanceOverrides=%d\n", nS, nIncrementalShuntVoltageControlControllerShunts, nG, nOut, nMoved, ctx->nShuntAdmittanceOverrides);
    }
}

}  // namespace olfgpu
