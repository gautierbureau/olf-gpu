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
#include "../common/codes/gpu_pi_model_column.cuh"
#include "../solver/driver_steps.cuh"
#include "incremental_transformer_voltage_control.cuh"
#include "../equations/scenario_overrides.cuh"
#include "flow_observation.cuh"
#include "sensitivity.cuh"
#include "incremental_flow_control.cuh"

namespace olfgpu {

// Per (scenario, controller): OLF's out-of-deadband test on the controlled flow (LIMITER: one-sided).
__global__ void incrementalFlowControlCheckBatch(int nIncrementalFlowControlControllerBranches, int nScenarios, int n, int nElements, int closedBranchOffset, int kindMask,
                              const int* kind, const int* side, const double* targetValue, const double* halfTargetDeadband,
                              const int* elem, const int* controlledClosedBranch,
                              const double* closedBranchParameters, const int* closedBranchIndices, int nClosedBranchParameterOverrides, const int* closedBranchParameterOverrideScenarioElement, const double* closedBranchParameterOverrideValues,
                              const char* disabled, const double* x, const char* outerLoopCheckActive,
                              char* outOfDeadband, double* diff) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nIncrementalFlowControlControllerBranches * nScenarios) return;
    int c = tid % nIncrementalFlowControlControllerBranches;
    int s = tid / nIncrementalFlowControlControllerBranches;
    outOfDeadband[tid] = 0;
    if (!outerLoopCheckActive[s] || !(kindMask & controlledFlowMaskBit(kind[c]))) return;
    int e = controlledClosedBranch[c];
    if (disabled[(size_t) s * nElements + elem[c]] || disabled[(size_t) s * nElements + closedBranchOffset + e]) return;
    const double* p = closedBranchParameters + e * GpuPiModelColumn::COUNT;
    double pOv[8];
    if (findTapOverride(nClosedBranchParameterOverrides, closedBranchParameterOverrideScenarioElement, closedBranchParameterOverrideValues, s, e, pOv)) p = pOv;
    double flow = kind[c] == GpuControlledFlow::CURRENT
            ? controlledCurrent(side[c], p, x + (size_t) s * n, closedBranchIndices + e * 12)
            : controlledFlow(side[c], kind[c], p, x + (size_t) s * n, closedBranchIndices + e * 12);
    double d = targetValue[c] - flow;
    diff[tid] = d;
    outOfDeadband[tid] = (kind[c] == GpuControlledFlow::CURRENT ? flow > targetValue[c] : fabs(d) > halfTargetDeadband[c]) ? 1 : 0;
}

// Per scenario, for controller c: the controlled flow's gradient dotted with the unit-RHS solution at
// the controlled branch's variable columns (a non-variable r1/a1 column contributes nothing).
__global__ void incrementalFlowControlGatherSensitivityBatch(int nScenarios, int n, int c, int nIncrementalFlowControlControllerBranches, int closedBranchOffset,
                                    const int* kind, const int* side, const int* controlledClosedBranch,
                                    const double* closedBranchParameters, const int* closedBranchIndices, int nClosedBranchParameterOverrides, const int* closedBranchParameterOverrideScenarioElement, const double* closedBranchParameterOverrideValues,
                                    const double* x, const double* dx, double* sensitivity) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    int e = controlledClosedBranch[c];
    const double* p = closedBranchParameters + e * GpuPiModelColumn::COUNT;
    double pOv[8];
    if (findTapOverride(nClosedBranchParameterOverrides, closedBranchParameterOverrideScenarioElement, closedBranchParameterOverrideValues, s, e, pOv)) p = pOv;
    const int* q = closedBranchIndices + e * 12;
    const double* dxs = dx + (size_t) s * n;
    double sv = kind[c] == GpuControlledFlow::CURRENT
            ? controlledCurrentSensitivity(side[c], p, x + (size_t) s * n, q, dxs)
            : controlledFlowSensitivity(side[c], kind[c], p, x + (size_t) s * n, q, dxs);
    sensitivity[(size_t) s * nIncrementalFlowControlControllerBranches + c] = sv;
}

// OLF's tap decisions per (scenario, controller): RPC — deltaR1 = diffQ / sensitivity, closest r1 within
// maxTapShift (no sensitivity gate); phase CONTROLLER — |sensitivity·toRadians(1)| > SENSI_EPS, da = dp / sensitivity,
// closest a1 (MAX_TAP_SHIFT unbounded); phase LIMITER — same gate, da = (target − i) / sensitivity, then
// FirstTapPositionAboveFinder: walk tap by tap in the shift's direction while the remaining shift still
// reaches past the next tap's a1. ControllerContext as everywhere (updateAllowedDirection, MAX_DIRECTION_CHANGE 3).
__global__ void incrementalFlowControlDecisionBatch(int nIncrementalFlowControlControllerBranches, int nScenarios, int kindMask,
                                 const int* kind, const int* tapColumn, const int* tapStart, const double* tapRows, const int* maxShift,
                                 const char* outOfDeadband, const double* diff, const double* sensitivity,
                                 int* currentTap, int* currentDirection, int* directionChangeCount, int* allowedDirection, char* moved) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nIncrementalFlowControlControllerBranches * nScenarios) return;
    int c = tid % nIncrementalFlowControlControllerBranches;
    int s = tid / nIncrementalFlowControlControllerBranches;
    if (!outOfDeadband[tid] || !(kindMask & controlledFlowMaskBit(kind[c]))) return;
    double sv = sensitivity[tid];
    if (kind[c] != GpuControlledFlow::REACTIVE_POWER && fabs(sv * 0.017453292519943295) <= 1e-6) return;  // SENSI_EPS on the per-degree value
    if (sv == 0.0) return;
    double delta = diff[tid] / sv;
    int cur = currentTap[tid];
    int base = tapStart[c];
    int nTaps = tapStart[c + 1] - base;
    int col = tapColumn[c];
    int lo = allowedDirection[tid] > 0 ? cur : 0;
    int hi = allowedDirection[tid] < 0 ? cur : nTaps - 1;
    int best = cur;
    if (kind[c] != GpuControlledFlow::CURRENT) {  // ClosestTapPositionFinder
        double newV = tapRows[(size_t) (base + cur) * GpuPiModelColumn::COUNT + col] + delta;
        double bestDist = fabs(delta);
        for (int i = lo; i <= hi; i++) {
            if (abs(i - cur) > maxShift[c]) continue;
            double dist = fabs(newV - tapRows[(size_t) (base + i) * GpuPiModelColumn::COUNT + col]);
            if (dist < bestDist) { best = i; bestDist = dist; }
        }
    } else {                                                 // FirstTapPositionAboveFinder
        double remaining = delta;
        int last = nTaps - 1;
        while (true) {
            int next = -1;                                   // OLF nextTapPositionIndex
            if (remaining != 0.0) {
                double currentTapValue = tapRows[(size_t) (base + best) * GpuPiModelColumn::COUNT + col];
                auto descending = [&](int n1, int n2) {
                    double a = tapRows[(size_t) (base + n1) * GpuPiModelColumn::COUNT + col], b = tapRows[(size_t) (base + n2) * GpuPiModelColumn::COUNT + col];
                    return remaining > 0 ? a < b : a > b;
                };
                (void) currentTapValue;
                if (best == 0 && last >= 1 && descending(1, 0)) next = -1;
                else if (best > 0 && descending(best, best - 1)) next = best - 1;
                else if (best < last && descending(best, best + 1)) next = best + 1;
            }
            if (next < 0 || next < lo || next > hi) break;
            double value = tapRows[(size_t) (base + best) * GpuPiModelColumn::COUNT + col];
            double nextValue = tapRows[(size_t) (base + next) * GpuPiModelColumn::COUNT + col];
            best = next;
            if ((remaining < 0 && value + remaining > nextValue) || (remaining > 0 && value + remaining < nextValue)) break;
            remaining -= nextValue - value;
        }
    }
    if (best == cur) return;
    currentTap[tid] = best;
    int direction = best > cur ? 1 : -1;
    if (directionChangeCount[tid] < 3) {
        if (currentDirection[tid] != 0 && currentDirection[tid] != direction) directionChangeCount[tid]++;
        currentDirection[tid] = direction;
    } else {
        allowedDirection[tid] = direction;
    }
    moved[s] = 1;
}

// Per (scenario, controller) whose tap moved: pin the identity row (rho or a1) at the tap's value.
__global__ void incrementalFlowControlApplyBatch(int nIncrementalFlowControlControllerBranches, int nScenarios, int n, const int* pinRow, const int* tapColumn,
                              const int* tapStart, const double* tapRows, const int* currentTap, int* appliedTap,
                              double* target, double* x) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nIncrementalFlowControlControllerBranches * nScenarios) return;
    int c = tid % nIncrementalFlowControlControllerBranches;
    int s = tid / nIncrementalFlowControlControllerBranches;
    int t = currentTap[tid];
    if (t == appliedTap[tid]) return;
    double v = tapRows[(size_t) (tapStart[c] + t) * GpuPiModelColumn::COUNT + tapColumn[c]];
    target[(size_t) s * n + pinRow[c]] = v;
    x[(size_t) s * n + pinRow[c]] = v;
    appliedTap[tid] = t;
}

// Per-solve reset. Incremental flow control: base taps, fresh ratchets.
void incrementalFlowControlResetSolve(BatchSolverContext* ctx, DriverPass& drv) {
    const int& nS = drv.nS;
    size_t nst = static_cast<size_t>(nS) * ctx->nIncrementalFlowControlControllerBranches;
    std::vector<int> cur(nst);
    for (int s2 = 0; s2 < nS; s2++) for (int c = 0; c < ctx->nIncrementalFlowControlControllerBranches; c++) cur[static_cast<size_t>(s2) * ctx->nIncrementalFlowControlControllerBranches + c] = ctx->hIncrementalFlowControlBaseTap[c];
    CUDA_CHECK(cudaMemcpy(ctx->dIncrementalFlowControlCurrentTap.p, cur.data(), nst * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->dIncrementalFlowControlAppliedTap.p, cur.data(), nst * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ctx->dIncrementalFlowControlCurrentDirection.p, 0, nst * sizeof(int)));
    CUDA_CHECK(cudaMemset(ctx->dIncrementalFlowControlDirectionChangeCount.p, 0, nst * sizeof(int)));
    CUDA_CHECK(cudaMemset(ctx->dIncrementalFlowControlAllowedDirection.p, 0, nst * sizeof(int)));
}

void incrementalFlowControlStep(BatchSolverContext* ctx, DriverPass& drv, int kindMask) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    std::vector<char>& active = drv.active;
    int nF = ctx->nIncrementalFlowControlControllerBranches;
    if (nF <= 0) return;
    if (ctx->mode != GpuLinearSolveMode::UBATCH && ctx->mode != GpuLinearSolveMode::ROBUST) {
        static bool warned = false;
        if (!warned) { std::fprintf(stderr, "in-kernel flow control needs a UBATCH (3) or robust (6) mode; mode %d runs it as stable\n", ctx->mode); warned = true; }
        return;
    }
    size_t nst = static_cast<size_t>(nS) * nF;
    incrementalFlowControlCheckBatch<<<grid(static_cast<int>(nst)), BLOCK, 0, ctx->stream>>>(nF, nS, n, ctx->nElements, ctx->closedBranchOffset, kindMask,
            ctx->dIncrementalFlowControlControlledFlow.as<int>(), ctx->dIncrementalFlowControlControlledSide.as<int>(), ctx->dIncrementalFlowControlTargetValue.as<double>(), ctx->dIncrementalFlowControlHalfTargetDeadband.as<double>(),
            ctx->dIncrementalFlowControlElement.as<int>(), ctx->dIncrementalFlowControlControlledClosedBranch.as<int>(),
            ctx->structure->closedBranch.parameters.as<double>(), ctx->structure->closedBranch.indices.as<int>(), ctx->nClosedBranchParameterOverrides, ctx->dClosedBranchParameterOverrideScenarioElement.as<int>(), ctx->dClosedBranchParameterOverrideValues.as<double>(),
            ctx->dDisabled.as<char>(), ctx->dX.as<double>(), ctx->dOuterLoopCheckActive.as<char>(),
            ctx->dIncrementalFlowControlOutOfDeadband.as<char>(), ctx->dIncrementalFlowControlDiff.as<double>());
    std::vector<char> outOfDeadband(nst);
    CUDA_CHECK(cudaMemcpyAsync(outOfDeadband.data(), ctx->dIncrementalFlowControlOutOfDeadband.p, nst, cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    std::vector<char> needC(nF, 0);
    bool any = false;
    for (int s2 = 0; s2 < nS; s2++) for (int c = 0; c < nF; c++) if (outOfDeadband[static_cast<size_t>(s2) * nF + c]) { needC[c] = 1; any = true; }
    if (!any) return;
    std::vector<char> ones(nS, 1);
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, ones.data(), nS, cudaMemcpyHostToDevice));
    fillActiveBatch(ctx, drv, false);
    CUDA_CHECK(cudaMemsetAsync(ctx->dIncrementalFlowControlSensitivity.p, 0, nst * sizeof(double), ctx->stream));
    bool first = true;
    for (int c = 0; c < nF; c++) {
        if (!needC[c]) continue;
        int row = ctx->structure->rowMap[ctx->hIncrementalFlowControlPinRow[c]];
        if (row < 0) continue;
        setUnitRhsBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, row, ctx->dB.as<double>());
        repairZeroDiagonalRightHandSide(ctx, nS, ctx->dB.as<double>());
        sensitivitySolve(ctx, drv, first);
        incrementalFlowControlGatherSensitivityBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(nS, n, c, nF, ctx->closedBranchOffset,
                ctx->dIncrementalFlowControlControlledFlow.as<int>(), ctx->dIncrementalFlowControlControlledSide.as<int>(), ctx->dIncrementalFlowControlControlledClosedBranch.as<int>(),
                ctx->structure->closedBranch.parameters.as<double>(), ctx->structure->closedBranch.indices.as<int>(), ctx->nClosedBranchParameterOverrides, ctx->dClosedBranchParameterOverrideScenarioElement.as<int>(), ctx->dClosedBranchParameterOverrideValues.as<double>(),
                ctx->dX.as<double>(), ctx->dDx.as<double>(), ctx->dIncrementalFlowControlSensitivity.as<double>());
    }
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, active.data(), nS, cudaMemcpyHostToDevice));
    incrementalFlowControlDecisionBatch<<<grid(static_cast<int>(nst)), BLOCK, 0, ctx->stream>>>(nF, nS, kindMask,
            ctx->dIncrementalFlowControlControlledFlow.as<int>(), ctx->dIncrementalFlowControlTapColumn.as<int>(), ctx->dIncrementalFlowControlTapStart.as<int>(), ctx->dIncrementalFlowControlTapRows.as<double>(), ctx->dIncrementalFlowControlMaxTapShift.as<int>(),
            ctx->dIncrementalFlowControlOutOfDeadband.as<char>(), ctx->dIncrementalFlowControlDiff.as<double>(), ctx->dIncrementalFlowControlSensitivity.as<double>(),
            ctx->dIncrementalFlowControlCurrentTap.as<int>(), ctx->dIncrementalFlowControlCurrentDirection.as<int>(), ctx->dIncrementalFlowControlDirectionChangeCount.as<int>(), ctx->dIncrementalFlowControlAllowedDirection.as<int>(), ctx->dOuterLoopUnstable.as<char>());
    incrementalFlowControlApplyBatch<<<grid(static_cast<int>(nst)), BLOCK, 0, ctx->stream>>>(nF, nS, n, ctx->dIncrementalFlowControlPinRow.as<int>(), ctx->dIncrementalFlowControlTapColumn.as<int>(),
            ctx->dIncrementalFlowControlTapStart.as<int>(), ctx->dIncrementalFlowControlTapRows.as<double>(), ctx->dIncrementalFlowControlCurrentTap.as<int>(), ctx->dIncrementalFlowControlAppliedTap.as<int>(),
            ctx->dTarget.as<double>(), ctx->dX.as<double>());
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    rebuildTapOverrides(ctx);
    if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
        std::vector<int> cur(nst);
        CUDA_CHECK(cudaMemcpy(cur.data(), ctx->dIncrementalFlowControlCurrentTap.p, nst * sizeof(int), cudaMemcpyDeviceToHost));
        int nOut = 0, nMoved = 0;
        for (char c : outOfDeadband) nOut += c ? 1 : 0;
        for (int s2 = 0; s2 < nS; s2++) for (int c = 0; c < nF; c++) if (cur[static_cast<size_t>(s2) * nF + c] != ctx->hIncrementalFlowControlBaseTap[c]) nMoved++;
        std::fprintf(stderr, "FLOW_KERNEL kindMask=%d nS=%d nControllers=%d outOfDeadband=%d pairsOffBaseTap=%d nClosedBranchParameterOverrides=%d\n", kindMask, nS, nF, nOut, nMoved, ctx->nClosedBranchParameterOverrides);
    }
}

}  // namespace olfgpu
