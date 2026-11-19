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
#include "scenario_install.cuh"
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
    std::vector<int> counts = {ctx->incrementalTransformerVoltageControl.nIncrementalTransformerVoltageControlControllerBranches, ctx->phaseControl.nPhaseControlControllerBranches,
                               ctx->incrementalFlowControl.nIncrementalFlowControlControllerBranches, ctx->transformerVoltageControl.nTransformerVoltageControlMembers};
    if (counts == ctx->scenarioOverrides.hClosedBranchParameterOverrideControllerCounts) return;
    ctx->scenarioOverrides.hClosedBranchParameterOverrideControllerCounts = counts;
    const std::vector<int>* elements[4] = {&ctx->incrementalTransformerVoltageControl.hIncrementalTransformerVoltageControlElement, &ctx->phaseControl.hPhaseControlElement,
                                           &ctx->incrementalFlowControl.hIncrementalFlowControlElement, &ctx->transformerVoltageControl.hTransformerVoltageControlMemberElement};
    std::vector<int> controllerElement;
    for (int l = 0; l < 4; l++) {
        controllerElement.insert(controllerElement.end(), elements[l]->begin(), elements[l]->begin() + counts[l]);
    }
    ctx->scenarioOverrides.nClosedBranchParameterOverrideControllers = static_cast<int>(controllerElement.size());
    if (controllerElement.empty()) return;
    std::vector<int> start, controller;
    elementControllers(ctx->structure->closedBranch.elementCount, controllerElement, start, controller);
    ctx->scenarioOverrides.dClosedBranchParameterOverrideElementControllerStart.upload(start.data(), start.size() * sizeof(int));
    ctx->scenarioOverrides.dClosedBranchParameterOverrideElementController.upload(controller.data(), controller.size() * sizeof(int));
}

// The incremental shunt loop's controllers that have a PACK shunt element (the others act through SHUNT_B only).
void ensureShuntControllers(BatchSolverContext* ctx) {
    std::vector<int> counts = {ctx->incrementalShuntVoltageControl.nIncrementalShuntVoltageControlControllerShunts};
    if (counts == ctx->scenarioOverrides.hShuntAdmittanceOverrideControllerCounts) return;
    ctx->scenarioOverrides.hShuntAdmittanceOverrideControllerCounts = counts;
    int nControllers = counts[0];
    ctx->scenarioOverrides.nShuntAdmittanceOverrideControllers = nControllers;
    if (nControllers == 0) return;
    std::vector<int> controllerElement(ctx->incrementalShuntVoltageControl.hIncrementalShuntVoltageControlShuntAdmittanceOverrideElement.begin(),
                                       ctx->incrementalShuntVoltageControl.hIncrementalShuntVoltageControlShuntAdmittanceOverrideElement.begin() + nControllers);
    std::vector<int> start, controller;
    elementControllers(ctx->structure->shunt.elementCount, controllerElement, start, controller);
    ctx->scenarioOverrides.dShuntAdmittanceOverrideElementControllerStart.upload(start.data(), start.size() * sizeof(int));
    ctx->scenarioOverrides.dShuntAdmittanceOverrideElementController.upload(controller.data(), controller.size() * sizeof(int));
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
    ctx->scenarioOverrides.closedBranchParameterOverrides.install(nS, ctx->installSlots, nOverrides, scenarioElement, values);
    ensureTapControllers(ctx);
    size_t nsc = static_cast<size_t>(nS) * ctx->scenarioOverrides.nClosedBranchParameterOverrideControllers;
    if (nsc > 0) {
        ctx->scenarioOverrides.dClosedBranchParameterOverrideTapMoved.ensure(nsc);
        ctx->scenarioOverrides.dClosedBranchParameterOverridePiModel.ensure(nsc * GpuPiModelColumn::COUNT * sizeof(double));
        // the installed slots' taps start where their scenarios put them; a slot still solving keeps the ones its loops moved
        memsetInstalledRows(ctx, ctx->scenarioOverrides.dClosedBranchParameterOverrideTapMoved.p, ctx->scenarioOverrides.nClosedBranchParameterOverrideControllers, 0);
    }
}

void installShuntAdmittanceOverrides(BatchSolverContext* ctx, int nOverrides, const int* scenarioElement,
                                     const double* delta) {
    int nS = ctx->nScenarios;
    ctx->scenarioOverrides.shuntAdmittanceOverrides.install(nS, ctx->installSlots, nOverrides, scenarioElement, delta);
    ensureShuntControllers(ctx);
    size_t nsc = static_cast<size_t>(nS) * ctx->scenarioOverrides.nShuntAdmittanceOverrideControllers;
    if (nsc > 0) {
        ctx->scenarioOverrides.dShuntAdmittanceOverrideSectionsMoved.ensure(nsc);
        ctx->scenarioOverrides.dShuntAdmittanceOverrideControllerDelta.ensure(nsc * 2 * sizeof(double));
        // the installed slots' sections start where their scenarios put them; a slot still solving keeps its loop's moves
        memsetInstalledRows(ctx, ctx->scenarioOverrides.dShuntAdmittanceOverrideSectionsMoved.p, ctx->scenarioOverrides.nShuntAdmittanceOverrideControllers, 0);
    }
}

void publishClosedBranchParameterOverrides(BatchSolverContext* ctx) {
    ensureTapControllers(ctx);
    int nControllers = ctx->scenarioOverrides.nClosedBranchParameterOverrideControllers;
    if (nControllers == 0) return;
    int nS = ctx->nScenarios;
    size_t nsc = static_cast<size_t>(nS) * nControllers;
    ctx->scenarioOverrides.dClosedBranchParameterOverrideTapMoved.ensure(nsc);
    ctx->scenarioOverrides.dClosedBranchParameterOverridePiModel.ensure(nsc * GpuPiModelColumn::COUNT * sizeof(double));
    struct Loop {
        int nControllers;
        DeviceBuf* currentTap;
        DeviceBuf* memberState;
        DeviceBuf* baseTap;
        DeviceBuf* tapStart;
        DeviceBuf* tapRows;
    };
    const Loop loops[4] = {
        {ctx->incrementalTransformerVoltageControl.nIncrementalTransformerVoltageControlControllerBranches, &ctx->incrementalTransformerVoltageControl.dIncrementalTransformerVoltageControlCurrentTap, nullptr,
         &ctx->incrementalTransformerVoltageControl.dIncrementalTransformerVoltageControlBaseTap, &ctx->incrementalTransformerVoltageControl.dIncrementalTransformerVoltageControlTapStart, &ctx->incrementalTransformerVoltageControl.dIncrementalTransformerVoltageControlTapRows},
        {ctx->phaseControl.nPhaseControlControllerBranches, &ctx->phaseControl.dPhaseControlTap, nullptr,
         &ctx->phaseControl.dPhaseControlBaseTap, &ctx->phaseControl.dPhaseControlTapStart, &ctx->phaseControl.dPhaseControlTapRows},
        {ctx->incrementalFlowControl.nIncrementalFlowControlControllerBranches, &ctx->incrementalFlowControl.dIncrementalFlowControlCurrentTap, nullptr,
         &ctx->incrementalFlowControl.dIncrementalFlowControlBaseTap, &ctx->incrementalFlowControl.dIncrementalFlowControlTapStart, &ctx->incrementalFlowControl.dIncrementalFlowControlTapRows},
        {ctx->transformerVoltageControl.nTransformerVoltageControlMembers, &ctx->transformerVoltageControl.dTransformerVoltageControlMemberTap, &ctx->transformerVoltageControl.dTransformerVoltageControlMemberState,
         &ctx->transformerVoltageControl.dTransformerVoltageControlMemberBaseTap, &ctx->transformerVoltageControl.dTransformerVoltageControlMemberTapStart, &ctx->transformerVoltageControl.dTransformerVoltageControlTapRows},
    };
    int firstController = 0;
    for (const Loop& l : loops) {
        if (l.nControllers > 0) {
            publishTapMovesBatch<<<grid(l.nControllers * nS), BLOCK, 0, ctx->stream>>>(nS, l.nControllers, firstController, nControllers,
                    l.currentTap->as<int>(), l.memberState != nullptr ? l.memberState->as<int>() : nullptr, l.baseTap->as<int>(),
                    l.tapStart->as<int>(), l.tapRows->as<double>(), ctx->scenarioOverrides.dClosedBranchParameterOverrideTapMoved.as<char>(),
                    ctx->scenarioOverrides.dClosedBranchParameterOverridePiModel.as<double>());
        }
        firstController += l.nControllers;
    }
}

void publishShuntAdmittanceOverrides(BatchSolverContext* ctx) {
    ensureShuntControllers(ctx);
    int nControllers = ctx->scenarioOverrides.nShuntAdmittanceOverrideControllers;
    if (nControllers == 0) return;
    int nS = ctx->nScenarios;
    size_t nsc = static_cast<size_t>(nS) * nControllers;
    ctx->scenarioOverrides.dShuntAdmittanceOverrideSectionsMoved.ensure(nsc);
    ctx->scenarioOverrides.dShuntAdmittanceOverrideControllerDelta.ensure(nsc * 2 * sizeof(double));
    publishSectionMovesBatch<<<grid(nControllers * nS), BLOCK, 0, ctx->stream>>>(nS, nControllers, ctx->incrementalShuntVoltageControl.nIncrementalShuntVoltageControlShuntCompensators,
            ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlShuntCompensatorStart.as<int>(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlSectionStart.as<int>(),
            ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlSectionB.as<double>(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlSectionG.as<double>(),
            ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlShuntCompensatorBasePosition.as<int>(), ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlPosition.as<int>(),
            ctx->scenarioOverrides.dShuntAdmittanceOverrideSectionsMoved.as<char>(), ctx->scenarioOverrides.dShuntAdmittanceOverrideControllerDelta.as<double>());
}

ClosedBranchParameterOverrides closedBranchParameterOverrides(BatchSolverContext* ctx) {
    ClosedBranchParameterOverrides o;
    o.nControllers = ctx->scenarioOverrides.nClosedBranchParameterOverrideControllers;
    o.elementControllerStart = ctx->scenarioOverrides.dClosedBranchParameterOverrideElementControllerStart.as<int>();
    o.elementController = ctx->scenarioOverrides.dClosedBranchParameterOverrideElementController.as<int>();
    o.tapMoved = ctx->scenarioOverrides.dClosedBranchParameterOverrideTapMoved.as<char>();
    o.piModel = ctx->scenarioOverrides.dClosedBranchParameterOverridePiModel.as<double>();
    o.caller = ctx->scenarioOverrides.closedBranchParameterOverrides.view();
    return o;
}

ShuntAdmittanceOverrides shuntAdmittanceOverrides(BatchSolverContext* ctx) {
    ShuntAdmittanceOverrides o;
    o.nControllers = ctx->scenarioOverrides.nShuntAdmittanceOverrideControllers;
    o.elementControllerStart = ctx->scenarioOverrides.dShuntAdmittanceOverrideElementControllerStart.as<int>();
    o.elementController = ctx->scenarioOverrides.dShuntAdmittanceOverrideElementController.as<int>();
    o.sectionsMoved = ctx->scenarioOverrides.dShuntAdmittanceOverrideSectionsMoved.as<char>();
    o.controllerDelta = ctx->scenarioOverrides.dShuntAdmittanceOverrideControllerDelta.as<double>();
    o.caller = ctx->scenarioOverrides.shuntAdmittanceOverrides.view();
    return o;
}

}  // namespace olfgpu
