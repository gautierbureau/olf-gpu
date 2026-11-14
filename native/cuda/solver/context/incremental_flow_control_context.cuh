// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The part of a batched solver context that IncrementalPhaseControlOuterLoop and IncrementalTransformerReactivePowerControlOuterLoop, which share one kernel: the
// controller branches, the controlled flows, their taps and deadbands, and the per-scenario tap moves.
#ifndef OLF_GPU_SOLVER_CONTEXT_INCREMENTAL_FLOW_CONTROL_CONTEXT_CUH
#define OLF_GPU_SOLVER_CONTEXT_INCREMENTAL_FLOW_CONTROL_CONTEXT_CUH

#include <cublas_v2.h>

#include <vector>

#include "../../common/device_buf.cuh"
#include "../slot_rows.cuh"
#include "../slot_table.cuh"

namespace olfgpu {

struct IncrementalFlowControlContext {
    // FLOW-OBSERVED incremental loops (GpuOuterLoopKind::INCREMENTAL_PHASE_CONTROL /
    // INCREMENTAL_TRANSFORMER_REACTIVE_POWER_CONTROL): a rho (RPC) or a1 (phase) tap driving a CONTROLLED
    // branch's flow; kind = the GpuControlledFlow: REACTIVE_POWER (rho, closest-r1), ACTIVE_POWER (phase
    // CONTROLLER, closest-a1), CURRENT (phase LIMITER, first-above-a1, one-sided).
    int nIncrementalFlowControlControllerBranches = 0, nIncrementalPhaseControlControllerBranches = 0, nIncrementalTransformerReactivePowerControlControllerBranches = 0;
    DeviceBuf dIncrementalFlowControlControlledFlow, dIncrementalFlowControlTargetRow, dIncrementalFlowControlTapColumn, dIncrementalFlowControlControlledSide, dIncrementalFlowControlTargetValue, dIncrementalFlowControlHalfTargetDeadband, dIncrementalFlowControlElement, dIncrementalFlowControlControlledClosedBranch,
              dIncrementalFlowControlTapStart, dIncrementalFlowControlTapRows, dIncrementalFlowControlBaseTap, dIncrementalFlowControlMaxTapShift;
    DeviceBuf dIncrementalFlowControlCurrentTap, dIncrementalFlowControlAppliedTap, dIncrementalFlowControlCurrentDirection, dIncrementalFlowControlDirectionChangeCount, dIncrementalFlowControlAllowedDirection, dIncrementalFlowControlSensitivity, dIncrementalFlowControlDiff, dIncrementalFlowControlOutOfDeadband;
    std::vector<int> hIncrementalFlowControlControlledFlow, hIncrementalFlowControlTargetRow, hIncrementalFlowControlElement, hIncrementalFlowControlTapStart, hIncrementalFlowControlBaseTap;
    std::vector<double> hIncrementalFlowControlTapRows;
};

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_CONTEXT_INCREMENTAL_FLOW_CONTROL_CONTEXT_CUH
