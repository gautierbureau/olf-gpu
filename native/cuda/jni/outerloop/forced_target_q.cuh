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
__global__ void forcedTargetQInitBatch(int nfq, int nS, int n, int nds, int extrapolate, const int* fqIdx,
                                       const double* fqIn, const double* fqPts, const double* fqPre,
                                       int nLost, const int* lost, const char* dsIslanded, const int* mode,
                                       double* target, double* applied);

__global__ void forcedTargetQBatch(int nfq, int nS, int n, int nds, int ngen, int extrapolate, const int* fqIdx,
                                   const double* fqIn, const double* fqPts, const double* fqPre,
                                   const double* genCum, const char* genCumValid, const double* dgAnchor,
                                   const double* dgStart, const int* dsPhiRow, const int* dsRowOv, int nLost, const int* lost, const char* dsIslanded,
                                   const int* mode, const char* outerLoopCheckActive, double* target, double* applied);

// Clamp the reactive limits of the forced-target-Q generators at their current active power.
void launchForcedTargetQLimits(BatchSolverContext* ctx, int nS, int n);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_FORCED_TARGET_Q_CUH
