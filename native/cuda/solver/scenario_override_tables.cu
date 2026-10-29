// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See scenario_override_tables.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/codes/gpu_pi_model_column.cuh"
#include "scenario_override_tables.cuh"

namespace olfgpu {

namespace {

// Per element of [0, nElements), the controllers acting on it, in the controllers' numbering order. A controller whose
// element is outside the range acts on none.
void elementControllers(int nElements, const std::vector<int>& controllerElement, std::vector<int>& start,
                        std::vector<int>& controller) {
    start.assign(static_cast<size_t>(nElements) + 1, 0);
    for (int e : controllerElement) {
        if (e >= 0 && e < nElements) start[e + 1]++;
    }
    for (int e = 0; e < nElements; e++) start[e + 1] += start[e];
    controller.assign(std::max(start[nElements], 1), -1);
    std::vector<int> next(start.begin(), start.end() - 1);
    for (int k = 0; k < static_cast<int>(controllerElement.size()); k++) {
        int e = controllerElement[k];
        if (e >= 0 && e < nElements) controller[next[e]++] = k;
    }
}

// The tap loops' controllers in one numbering, in the order the lookup gives them precedence: incremental TVC,
// continuous phase, incremental flow, discrete TVC members. Built when the loops' sizes change (they are installed
// before the first solve).
void ensureTapControllers(BatchSolverContext* ctx) {
    std::vector<int> counts = {ctx->nIncrementalTransformerVoltageControlControllerBranches, ctx->nPhaseControlControllerBranches,
                               ctx->nIncrementalFlowControlControllerBranches, ctx->nTransformerVoltageControlMembers};
    if (counts == ctx->hClosedBranchParameterOverrideControllerCounts) return;
    ctx->hClosedBranchParameterOverrideControllerCounts = counts;
    const std::vector<int>* elements[4] = {&ctx->hIncrementalTransformerVoltageControlElement, &ctx->hPhaseControlElement,
                                           &ctx->hIncrementalFlowControlElement, &ctx->hTransformerVoltageControlMemberElement};
    std::vector<int> controllerElement;
    for (int l = 0; l < 4; l++) {
        controllerElement.insert(controllerElement.end(), elements[l]->begin(), elements[l]->begin() + counts[l]);
    }
    ctx->nClosedBranchParameterOverrideControllers = static_cast<int>(controllerElement.size());
    if (controllerElement.empty()) return;
    std::vector<int> start, controller;
    elementControllers(ctx->structure->closedBranch.elementCount, controllerElement, start, controller);
    ctx->dClosedBranchParameterOverrideElementControllerStart.upload(start.data(), start.size() * sizeof(int));
    ctx->dClosedBranchParameterOverrideElementController.upload(controller.data(), controller.size() * sizeof(int));
}

// The incremental shunt loop's controllers that have a PACK shunt element (the others act through SHUNT_B only).
void ensureShuntControllers(BatchSolverContext* ctx) {
    std::vector<int> counts = {ctx->nIncrementalShuntVoltageControlControllerShunts};
    if (counts == ctx->hShuntAdmittanceOverrideControllerCounts) return;
    ctx->hShuntAdmittanceOverrideControllerCounts = counts;
    int nControllers = counts[0];
    ctx->nShuntAdmittanceOverrideControllers = nControllers;
    if (nControllers == 0) return;
    std::vector<int> controllerElement(ctx->hIncrementalShuntVoltageControlShuntAdmittanceOverrideElement.begin(),
                                       ctx->hIncrementalShuntVoltageControlShuntAdmittanceOverrideElement.begin() + nControllers);
    std::vector<int> start, controller;
    elementControllers(ctx->structure->shunt.elementCount, controllerElement, start, controller);
    ctx->dShuntAdmittanceOverrideElementControllerStart.upload(start.data(), start.size() * sizeof(int));
    ctx->dShuntAdmittanceOverrideElementController.upload(controller.data(), controller.size() * sizeof(int));
}

// One thread per (slot, controller) of one tap loop: a tap off its base tap overrides its transformer's row with the
// tap's row; a discrete TVC member's only once its tap is FIXED (state 2).
__global__ void publishTapMovesBatch(int nScenarios, int nLoopControllers, int firstController, int nControllers,
                                     const int* currentTap, const int* memberState, const int* baseTap, const int* tapStart,
                                     const double* tapRows, char* tapMoved, double* piModel) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nLoopControllers * nScenarios) return;
    int c = tid % nLoopControllers;
    int s = tid / nLoopControllers;
    int tap = currentTap[tid];
    bool moved = tap != baseTap[c] && (memberState == nullptr || memberState[tid] == 2);
    size_t sc = static_cast<size_t>(s) * nControllers + firstController + c;
    tapMoved[sc] = moved ? 1 : 0;
    if (moved) {
        const double* row = tapRows + static_cast<size_t>(tapStart[c] + tap) * GpuPiModelColumn::COUNT;
        for (int k = 0; k < GpuPiModelColumn::COUNT; k++) {
            piModel[sc * GpuPiModelColumn::COUNT + k] = row[k];
        }
    }
}

// One thread per (slot, controller) of the incremental shunt loop: the (dG, dB) of its shunt compensators' sections
// off their base sections, summed in compensator order.
__global__ void publishSectionMovesBatch(int nScenarios, int nControllers, int nShuntCompensators, const int* shuntCompensatorStart,
                                         const int* sectionStart, const double* sectionB, const double* sectionG,
                                         const int* basePosition, const int* position, char* sectionsMoved, double* controllerDelta) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nControllers * nScenarios) return;
    int c = tid % nControllers;
    int s = tid / nControllers;
    double deltaB = 0.0, deltaG = 0.0;
    for (int u = shuntCompensatorStart[c]; u < shuntCompensatorStart[c + 1]; u++) {
        int firstSection = sectionStart[u];
        int p = position[static_cast<size_t>(s) * nShuntCompensators + u];
        deltaB += sectionB[firstSection + p] - sectionB[firstSection + basePosition[u]];
        deltaG += sectionG[firstSection + p] - sectionG[firstSection + basePosition[u]];
    }
    sectionsMoved[tid] = deltaB != 0.0 || deltaG != 0.0 ? 1 : 0;
    controllerDelta[2 * static_cast<size_t>(tid)] = deltaG;
    controllerDelta[2 * static_cast<size_t>(tid) + 1] = deltaB;
}

}  // namespace

void installClosedBranchParameterOverrides(BatchSolverContext* ctx, int nOverrides, const int* scenarioElement,
                                           const double* values) {
    int nS = ctx->nScenarios;
    ctx->closedBranchParameterOverrides.install(nS, ctx->installSlots, nOverrides, scenarioElement, values);
    ensureTapControllers(ctx);
    size_t nsc = static_cast<size_t>(nS) * ctx->nClosedBranchParameterOverrideControllers;
    if (nsc > 0) {
        ctx->dClosedBranchParameterOverrideTapMoved.ensure(nsc);
        ctx->dClosedBranchParameterOverridePiModel.ensure(nsc * GpuPiModelColumn::COUNT * sizeof(double));
        CUDA_CHECK(cudaMemset(ctx->dClosedBranchParameterOverrideTapMoved.p, 0, nsc));
    }
}

void installShuntAdmittanceOverrides(BatchSolverContext* ctx, int nOverrides, const int* scenarioElement,
                                     const double* delta) {
    int nS = ctx->nScenarios;
    ctx->shuntAdmittanceOverrides.install(nS, ctx->installSlots, nOverrides, scenarioElement, delta);
    ensureShuntControllers(ctx);
    size_t nsc = static_cast<size_t>(nS) * ctx->nShuntAdmittanceOverrideControllers;
    if (nsc > 0) {
        ctx->dShuntAdmittanceOverrideSectionsMoved.ensure(nsc);
        ctx->dShuntAdmittanceOverrideControllerDelta.ensure(nsc * 2 * sizeof(double));
        CUDA_CHECK(cudaMemset(ctx->dShuntAdmittanceOverrideSectionsMoved.p, 0, nsc));
    }
}

void publishClosedBranchParameterOverrides(BatchSolverContext* ctx) {
    ensureTapControllers(ctx);
    int nControllers = ctx->nClosedBranchParameterOverrideControllers;
    if (nControllers == 0) return;
    int nS = ctx->nScenarios;
    size_t nsc = static_cast<size_t>(nS) * nControllers;
    ctx->dClosedBranchParameterOverrideTapMoved.ensure(nsc);
    ctx->dClosedBranchParameterOverridePiModel.ensure(nsc * GpuPiModelColumn::COUNT * sizeof(double));
    struct Loop {
        int nControllers;
        DeviceBuf* currentTap;
        DeviceBuf* memberState;
        DeviceBuf* baseTap;
        DeviceBuf* tapStart;
        DeviceBuf* tapRows;
    };
    const Loop loops[4] = {
        {ctx->nIncrementalTransformerVoltageControlControllerBranches, &ctx->dIncrementalTransformerVoltageControlCurrentTap, nullptr,
         &ctx->dIncrementalTransformerVoltageControlBaseTap, &ctx->dIncrementalTransformerVoltageControlTapStart, &ctx->dIncrementalTransformerVoltageControlTapRows},
        {ctx->nPhaseControlControllerBranches, &ctx->dPhaseControlTap, nullptr,
         &ctx->dPhaseControlBaseTap, &ctx->dPhaseControlTapStart, &ctx->dPhaseControlTapRows},
        {ctx->nIncrementalFlowControlControllerBranches, &ctx->dIncrementalFlowControlCurrentTap, nullptr,
         &ctx->dIncrementalFlowControlBaseTap, &ctx->dIncrementalFlowControlTapStart, &ctx->dIncrementalFlowControlTapRows},
        {ctx->nTransformerVoltageControlMembers, &ctx->dTransformerVoltageControlMemberTap, &ctx->dTransformerVoltageControlMemberState,
         &ctx->dTransformerVoltageControlMemberBaseTap, &ctx->dTransformerVoltageControlMemberTapStart, &ctx->dTransformerVoltageControlTapRows},
    };
    int firstController = 0;
    for (const Loop& l : loops) {
        if (l.nControllers > 0) {
            publishTapMovesBatch<<<grid(l.nControllers * nS), BLOCK, 0, ctx->stream>>>(nS, l.nControllers, firstController, nControllers,
                    l.currentTap->as<int>(), l.memberState != nullptr ? l.memberState->as<int>() : nullptr, l.baseTap->as<int>(),
                    l.tapStart->as<int>(), l.tapRows->as<double>(), ctx->dClosedBranchParameterOverrideTapMoved.as<char>(),
                    ctx->dClosedBranchParameterOverridePiModel.as<double>());
        }
        firstController += l.nControllers;
    }
}

void publishShuntAdmittanceOverrides(BatchSolverContext* ctx) {
    ensureShuntControllers(ctx);
    int nControllers = ctx->nShuntAdmittanceOverrideControllers;
    if (nControllers == 0) return;
    int nS = ctx->nScenarios;
    size_t nsc = static_cast<size_t>(nS) * nControllers;
    ctx->dShuntAdmittanceOverrideSectionsMoved.ensure(nsc);
    ctx->dShuntAdmittanceOverrideControllerDelta.ensure(nsc * 2 * sizeof(double));
    publishSectionMovesBatch<<<grid(nControllers * nS), BLOCK, 0, ctx->stream>>>(nS, nControllers, ctx->nIncrementalShuntVoltageControlShuntCompensators,
            ctx->dIncrementalShuntVoltageControlShuntCompensatorStart.as<int>(), ctx->dIncrementalShuntVoltageControlSectionStart.as<int>(),
            ctx->dIncrementalShuntVoltageControlSectionB.as<double>(), ctx->dIncrementalShuntVoltageControlSectionG.as<double>(),
            ctx->dIncrementalShuntVoltageControlShuntCompensatorBasePosition.as<int>(), ctx->dIncrementalShuntVoltageControlPosition.as<int>(),
            ctx->dShuntAdmittanceOverrideSectionsMoved.as<char>(), ctx->dShuntAdmittanceOverrideControllerDelta.as<double>());
}

ClosedBranchParameterOverrides closedBranchParameterOverrides(BatchSolverContext* ctx) {
    ClosedBranchParameterOverrides o;
    o.nControllers = ctx->nClosedBranchParameterOverrideControllers;
    o.elementControllerStart = ctx->dClosedBranchParameterOverrideElementControllerStart.as<int>();
    o.elementController = ctx->dClosedBranchParameterOverrideElementController.as<int>();
    o.tapMoved = ctx->dClosedBranchParameterOverrideTapMoved.as<char>();
    o.piModel = ctx->dClosedBranchParameterOverridePiModel.as<double>();
    o.caller = ctx->closedBranchParameterOverrides.view();
    return o;
}

ShuntAdmittanceOverrides shuntAdmittanceOverrides(BatchSolverContext* ctx) {
    ShuntAdmittanceOverrides o;
    o.nControllers = ctx->nShuntAdmittanceOverrideControllers;
    o.elementControllerStart = ctx->dShuntAdmittanceOverrideElementControllerStart.as<int>();
    o.elementController = ctx->dShuntAdmittanceOverrideElementController.as<int>();
    o.sectionsMoved = ctx->dShuntAdmittanceOverrideSectionsMoved.as<char>();
    o.controllerDelta = ctx->dShuntAdmittanceOverrideControllerDelta.as<double>();
    o.caller = ctx->shuntAdmittanceOverrides.view();
    return o;
}

}  // namespace olfgpu
