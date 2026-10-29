// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The per-scenario inputs of a solve, written into the install slots (ctx->installSlots, every slot of the bank when
// empty): local slot j of each array is bank slot installedSlot(ctx, j), and only the installed slots' rows change. An
// installed slot starts from the base state, targets and row modes, then takes its own.
#ifndef OLF_GPU_SOLVER_SCENARIO_INSTALL_CUH
#define OLF_GPU_SOLVER_SCENARIO_INSTALL_CUH

#include "batch_context.cuh"

namespace olfgpu {

struct ScenarioInputs {
    const double* x0 = nullptr;                              // the base state (n)
    const double* target = nullptr;                          // the base targets (n)
    const int* baseRowMode = nullptr;                        // the base row modes (n)
    // The row modes of the installed slots that differ from the base: their local slots and rows; every installed slot's
    // row when rowModeSlots is null.
    const int* rowMode = nullptr;
    const int* rowModeSlots = nullptr;
    int nRowModeSlots = 0;
    const char* disabled = nullptr;                          // per installed slot, nElements
    int nTargetOverrides = 0;                                // (local slot * n + row, value)
    const int* targetOverrideIndex = nullptr;
    const double* targetOverrideValue = nullptr;
    const double* x0Full = nullptr;                          // per installed slot, n; null: the base state and the x0 seeds
    // Per installed slot, null when none of them has it: a generator loss at a DS participant bus...
    const int* distributedSlackOverrideBus = nullptr;
    const double* distributedSlackOverrideFactorDelta = nullptr;
    const double* distributedSlackOverrideBaseTargetPDelta = nullptr;
    const double* distributedSlackOverrideMaxDelta = nullptr;
    const double* distributedSlackOverrideMinDelta = nullptr;
    // ...a strategy pair's anchor shift per DS participant, and the slack's own target move.
    const double* distributedSlackBusAnchorShift = nullptr;
    const double* slackBusAnchorShift = nullptr;
};

void installScenarios(BatchSolverContext* ctx, const ScenarioInputs& in);

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_SCENARIO_INSTALL_CUH
