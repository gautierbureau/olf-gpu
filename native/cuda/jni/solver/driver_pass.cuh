// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// DriverPass: the per-solve state the outer-loop driver (solveBatchContext) shares with the phase steps it
// calls, one per batched solve.
#ifndef OLF_GPU_SOLVER_DRIVER_PASS_CUH
#define OLF_GPU_SOLVER_DRIVER_PASS_CUH

#include <string>
#include <vector>

#include "../common/device_buf.cuh"
#include "../common/loop_kinds.cuh"
#include "../common/codes/gpu_state_vector_scaling_mode.cuh"
#include "batch_context.cuh"

namespace olfgpu {

struct DriverPass {
    DeviceBuf dBaseMode;                                    // the contingency-free base row mode (length n)
    size_t nTot = 0;
    double tol = 0.0;
    int maxIter = 0;
    int lsMode = GpuStateVectorScalingMode::NONE;
    double convEpsPerEq = 0.0;
    bool outerLoopTiming = false;
    int outerLoopPassSolvedScenarios = 0;
    int nActive = 0;
    std::vector<char> wasActive;
    std::vector<double> hF;
    int maxPqPvSwitch = 0;
    int reactiveLimitsDecisionLogScenario = -1;
    int skipReactiveLimitsPhase = 0;
    bool svcCoefDyn = false;
    int nReactiveLimitDeltaOverrides = 0;
    int nReactiveCapabilityCurveOverrides = 0;
    DeviceBuf dReactiveLimitDeltaOverrideScenarioEntry, dReactiveLimitDeltaOverrideValues, dReactiveLimitDeltaOverrideStart, dReactiveLimitDeltaOverrideList;
    DeviceBuf dReactiveCapabilityCurveOverrideStart, dReactiveCapabilityCurveOverrideCurve, dReactiveCapabilityCurveOverrideStepScale;
    double slackDistTol = 0.0;
    double pResidueEps = 0.0;
    bool hasDistributedSlackOverride = false;
    bool hasDistributedSlackBusAnchorShift = false;
    bool hasSlackBusAnchorShift = false;
    DeviceBuf dDistributedSlackOverrideBus, dDistributedSlackOverrideFactorDelta, dDistributedSlackOverrideBaseTargetPDelta, dDistributedSlackOverrideMaxDelta, dDistributedSlackOverrideMinDelta;
    DeviceBuf dDistributedSlackBusAnchorShift, dSlackBusAnchorShift;
    int n = 0;
    int nnz = 0;
    int nS = 0;
    int outer = 0;
    std::vector<char> active;
    std::vector<char> live;
    std::vector<char> outerLoopCheckActive;
    std::vector<char> loopRejected;
    int loopIdx = 0;
    int loopOrder[MAX_DEVICE_LOOPS] = {};
    BatchResult res;
    bool usePerScenarioReactiveLimits = false;
    bool deviceFixedIter = false;
    double reactiveLimitTol = 0.0;
    int nReactiveLimitTargetVOverrides = 0;
    DeviceBuf dReactiveLimitTargetVOverrideScenarioEntry;
    DeviceBuf dReactiveLimitTargetVOverrideValue;
};

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_DRIVER_PASS_CUH
