// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// IncrementalTransformerVoltageControlOuterLoop as an in-kernel driver phase: per controlled bus out of its
// deadband, dV/drho from the converged Jacobian, the closest tap within maxTapShift and the ratchet's direction.
#ifndef OLF_GPU_OUTERLOOP_INCREMENTAL_TRANSFORMER_VOLTAGE_CONTROL_CUH
#define OLF_GPU_OUTERLOOP_INCREMENTAL_TRANSFORMER_VOLTAGE_CONTROL_CUH

#include "../solver/driver_pass.cuh"
#include "../common/codes/gpu_outer_loop_kind.cuh"

namespace olfgpu {

// Per-solve reset: every scenario at the base tap with a fresh ratchet.
void incrementalTransformerVoltageControlResetSolve(BatchSolverContext* ctx, DriverPass& drv);

// One check of the loop on every live scenario (GpuOuterLoopKind::INCREMENTAL_TRANSFORMER_VOLTAGE_CONTROL).
void incrementalTransformerVoltageControlStep(BatchSolverContext* ctx, DriverPass& drv);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_INCREMENTAL_TRANSFORMER_VOLTAGE_CONTROL_CUH
