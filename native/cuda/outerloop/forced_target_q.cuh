// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// forceTargetQInReactiveLimits: a generator whose reactive target is forced inside its limits has the target
// clamped at the start active power, and again after every distributed-slack move of that power.
#ifndef OLF_GPU_OUTERLOOP_FORCED_TARGET_Q_CUH
#define OLF_GPU_OUTERLOOP_FORCED_TARGET_Q_CUH

#include "../solver/driver_pass.cuh"

namespace olfgpu {

// Kernels launched by the driver and the distributed slack.
__global__ void forcedTargetQInitBatch(int nForcedTargetQ, int nS, int n, int nDistributedSlackBuses, int extrapolate, const int* forcedTargetQIndices,
                                       const double* forcedTargetQParameters, const double* forcedTargetQPoints, const double* forcedTargetQPreDistribution,
                                       SlotEntries lost, const char* distributedSlackIslanded, const int* mode,
                                       double* target, double* applied);

__global__ void forcedTargetQBatch(int nForcedTargetQ, int nS, int n, int nDistributedSlackBuses, int nDistributedSlackGenerators, int extrapolate, const int* forcedTargetQIndices,
                                   const double* forcedTargetQParameters, const double* forcedTargetQPoints, const double* forcedTargetQPreDistribution,
                                   const double* distributedSlackGeneratorMove, const char* distributedSlackGeneratorMoveValid, const double* distributedSlackGeneratorAnchor,
                                   const double* distributedSlackGeneratorInitialMove, const int* distributedSlackBusPhiRow, const int* distributedSlackBusRowOverride, SlotEntries lost, const char* distributedSlackIslanded,
                                   const int* mode, const char* outerLoopCheckActive, double* target, double* applied);

// Clamp the reactive limits of the forced-target-Q generators at their current active power, on the scenarios of
// active (nullptr: every scenario).
void launchForcedTargetQLimits(BatchSolverContext* ctx, int nS, int n, const char* active = nullptr);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_FORCED_TARGET_Q_CUH
