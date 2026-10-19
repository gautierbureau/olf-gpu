// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// DistributedSlackOuterLoop as an in-kernel driver phase: the slack mismatch spread over the participating
// generators or loads by OLF's water-filling, the slack-only variant, and the slack's retained share.
#ifndef OLF_GPU_OUTERLOOP_DISTRIBUTED_SLACK_CUH
#define OLF_GPU_OUTERLOOP_DISTRIBUTED_SLACK_CUH

#include "../solver/driver_pass.cuh"
#include "../common/codes/gpu_outer_loop_kind.cuh"

namespace olfgpu {

// Kernel launched before the check, for a link on the slack bus.
__global__ void hvdcSlackBalanceBatch(int nS, int nh, int n, int nElements, int slackRowBase, const int* slackRowOv,
                                      const int* hvBus1Row, const int* hvBus2Row, const int* hvBus1Phi,
                                      const int* hvBus2Phi, const int* hvElem, const double* hvIn,
                                      const int* status, const double* d1, const double* d2, const char* ctgOut,
                                      const char* disabled, const double* x, const char* rlActive, double* qBuf);

// Per-solve resets.
void dsResetSolve(BatchSolverContext* ctx, DriverPass& drv);
void dsSlackOnlyResetSolve(BatchSolverContext* ctx, DriverPass& drv, const double* dsSlackShift);

// One check of DistributedSlackOuterLoop on every live scenario (GpuOuterLoopKind::DISTRIBUTED_SLACK).
void dsCheck(BatchSolverContext* ctx, DriverPass& drv);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_DISTRIBUTED_SLACK_CUH
