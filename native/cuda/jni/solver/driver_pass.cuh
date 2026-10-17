// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// DriverPass: the per-solve state the outer-loop driver (solveBatchContext) shares with the phase steps it
// calls, one per batched solve.
#ifndef OLF_GPU_SOLVER_DRIVER_PASS_CUH
#define OLF_GPU_SOLVER_DRIVER_PASS_CUH

#include <vector>

#include "../common/device_buf.cuh"
#include "../common/loop_kinds.cuh"
#include "batch_context.cuh"

namespace olfgpu {

struct DriverPass {
    double slackDistTol = 0.0;
    double pResidueEps = 0.0;
    bool hasDsOverride = false;
    bool hasDsAnchor = false;
    bool hasDsSlackShift = false;
    DeviceBuf dDsOvBus, dDsOvFactorDelta, dDsOvBaseDelta, dDsOvMaxDelta, dDsOvMinDelta;
    DeviceBuf dDsAnchorShift, dDsSlackShift;
    int n = 0;
    int nnz = 0;
    int nS = 0;
    int outer = 0;
    std::vector<char> active;
    std::vector<char> live;
    std::vector<char> rlActive;
    std::vector<char> loopRejected;
    int loopIdx = 0;
    int loopOrder[MAX_DEVICE_LOOPS] = {};
    BatchResult res;
    bool useDynRlLimits = false;
    bool deviceFixedIter = false;
    double reactiveLimitTol = 0.0;
    int nRlVtOv = 0;
    DeviceBuf dRlVtOvSlotIdx;
    DeviceBuf dRlVtOvVal;
};

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_DRIVER_PASS_CUH
