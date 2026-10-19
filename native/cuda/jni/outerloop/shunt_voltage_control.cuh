// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// ShuntVoltageControlOuterLoop (WITH_GENERATOR) as an in-kernel driver phase.
#ifndef OLF_GPU_OUTERLOOP_SHUNT_VOLTAGE_CONTROL_CUH
#define OLF_GPU_OUTERLOOP_SHUNT_VOLTAGE_CONTROL_CUH

#include "../solver/driver_pass.cuh"
#include "../common/codes/gpu_outer_loop_kind.cuh"

namespace olfgpu {

// Per-solve reset: every control at its entry state.
void svdResetSolve(BatchSolverContext* ctx, DriverPass& drv);

// One check of the loop on every live scenario (GpuOuterLoopKind::SHUNT_VOLTAGE_CONTROL).
void svdStep(BatchSolverContext* ctx, DriverPass& drv);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_SHUNT_VOLTAGE_CONTROL_CUH
