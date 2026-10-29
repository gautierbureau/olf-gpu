// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See automation_system.cuh.
#include <cuda_runtime.h>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/codes/gpu_pi_model_column.cuh"
#include "../common/codes/gpu_two_sides.cuh"
#include "../equations/generated/closed_branch_current_magnitude.cuh"
#include "../equations/scenario_overrides.cuh"
#include "flow_observation.cuh"
#include "automation_system.cuh"
#include "../solver/scenario_override_tables.cuh"

using namespace powsybl::openloadflow;

namespace olfgpu {

namespace {

// AutomationSystemOuterLoop.check on one scenario. For each system whose monitored branch is in service (OLF's isInService:
// a branch another system opened carries no current, whatever its closed-branch term computes from its buses' voltages),
// the current at that branch's side (getI1 / getI2, pu), at the scenario's tap (a contingency's tap action, a tap the
// transformer voltage control moved), against each tripping's threshold, in
// order: past it, a tripping whose branch is not already in the wanted state opens it (it is in service) or closes it (it
// is not), and the system stops at that tripping. The decisions are taken on the state the check found and applied
// together afterwards, as OLF collects branchesToOpen and branchesToClose before it touches the connectivity.
__global__ void automationSystemBatch(int nS, int n, int nElements, int nSystems,
                                      const int* monitoredElement, const int* monitoredSide,
                                      const int* trippingStart, const int* operatedMaskSlot, const char* openAction,
                                      const double* threshold, const double* closedBranchParameters, const int* closedBranchIndices,
                                      ClosedBranchParameterOverrides closedBranchParameterOverrides, const double* x, const char* outerLoopCheckActive, char* disabled, char* switched, char* unstable) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nS || !outerLoopCheckActive[s]) {
        return;
    }
    const double* xs = x + (size_t) s * n;
    char* scenarioDisabled = disabled + (size_t) s * nElements;
    int nTrippings = trippingStart[nSystems];
    char* scenarioSwitched = switched + (size_t) s * nTrippings;
    bool anyToSwitch = false;
    for (int m = 0; m < nSystems; m++) {
        int e = monitoredElement[m];
        if (scenarioDisabled[e]) {
            continue;                                                 // a closed-branch element's mask slot is its index
        }
        const double* p = closedBranchParameters + (size_t) e * GpuPiModelColumn::COUNT;
        double pOv[GpuPiModelColumn::COUNT];
        if (findTapOverride(closedBranchParameterOverrides, s, e, pOv)) {
            p = pOv;
        }
        double i = controlledCurrent(monitoredSide[m], p, xs, closedBranchIndices + (size_t) e * 12);
        for (int t = trippingStart[m]; t < trippingStart[m + 1]; t++) {
            // an operated branch is a closed-branch element: connected at both sides, in service when enabled
            if (i > threshold[t] && (openAction[t] != 0) != (scenarioDisabled[operatedMaskSlot[t]] != 0)) {
                scenarioSwitched[t] = 1;
                anyToSwitch = true;
                break;
            }
        }
    }
    if (!anyToSwitch) {
        return;
    }
    for (int t = 0; t < nTrippings; t++) {
        if (scenarioSwitched[t]) {
            scenarioSwitched[t] = 0;
            scenarioDisabled[operatedMaskSlot[t]] = openAction[t] != 0 ? 1 : 0;
        }
    }
    unstable[s] = 1;                                                  // the network changed: OLF's UNSTABLE
}

}  // namespace

void automationSystemCheck(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    if (ctx->nOverloadManagementSystems <= 0) {
        return;
    }
    ctx->dAutomationSystemSwitched.alloc((size_t) nS * ctx->nOverloadManagementSystemTrippings * sizeof(char));
    automationSystemBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(nS, n, ctx->nElements, ctx->nOverloadManagementSystems,
            ctx->dOverloadManagementSystemMonitoredElement.as<int>(), ctx->dOverloadManagementSystemMonitoredSide.as<int>(),
            ctx->dOverloadManagementSystemTrippingStart.as<int>(), ctx->dOverloadManagementSystemOperatedMaskSlot.as<int>(),
            ctx->dOverloadManagementSystemOpenAction.as<char>(), ctx->dOverloadManagementSystemThreshold.as<double>(),
            ctx->structure->closedBranch.parameters.as<double>(), ctx->structure->closedBranch.indices.as<int>(),
            closedBranchParameterOverrides(ctx), ctx->dX.as<double>(), ctx->dOuterLoopCheckActive.as<char>(), ctx->dDisabled.as<char>(),
            ctx->dAutomationSystemSwitched.as<char>(), ctx->dOuterLoopUnstable.as<char>());
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace olfgpu
