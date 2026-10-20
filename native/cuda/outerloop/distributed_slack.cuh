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
                                      const int* hvdcBus1PRow, const int* hvdcBus2PRow, const int* hvdcBus1PhiColumn,
                                      const int* hvdcBus2PhiColumn, const int* hvdcElement, const double* hvdcAcEmulationParameters,
                                      const int* status, const double* d1, const double* d2, const char* ctgOut,
                                      const char* disabled, const double* x, const char* outerLoopCheckActive, double* qBuf);

// Per-solve resets.
void distributedSlackResetSolve(BatchSolverContext* ctx, DriverPass& drv);
void slackOnlyDistributionResetSolve(BatchSolverContext* ctx, DriverPass& drv, const double* slackBusAnchorShift);

// One check of DistributedSlackOuterLoop on every live scenario (GpuOuterLoopKind::DISTRIBUTED_SLACK).
void distributedSlackCheck(BatchSolverContext* ctx, DriverPass& drv);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_DISTRIBUTED_SLACK_CUH
