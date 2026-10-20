// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// IncrementalPhaseControlOuterLoop and IncrementalTransformerReactivePowerControlOuterLoop as in-kernel driver
// phases: a controlled branch flow out of its deadband moves the controlling tap by the sensitivity, ratcheted.
#ifndef OLF_GPU_OUTERLOOP_INCREMENTAL_FLOW_CONTROL_CUH
#define OLF_GPU_OUTERLOOP_INCREMENTAL_FLOW_CONTROL_CUH

#include "../solver/driver_pass.cuh"
#include "../common/codes/gpu_outer_loop_kind.cuh"

namespace olfgpu {

// Per-solve reset: base taps, fresh ratchets.
void incrementalFlowControlResetSolve(BatchSolverContext* ctx, DriverPass& drv);

// One check on every live scenario: kindMask 1 = reactive power control (GpuOuterLoopKind::INCREMENTAL_TRANSFORMER_REACTIVE_POWER_CONTROL), 6 = the
// phase control kinds (GpuOuterLoopKind::INCREMENTAL_PHASE_CONTROL).
void incrementalFlowControlStep(BatchSolverContext* ctx, DriverPass& drv, int kindMask);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_INCREMENTAL_FLOW_CONTROL_CUH
