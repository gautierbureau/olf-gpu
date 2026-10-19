// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// SecondaryVoltageControlOuterLoop as an in-kernel driver phase, per (scenario, zone): the re-enable step, the
// per-zone trigger, the sensitivities on the driver's own factors, the dense zone system (batched cuBLAS LU)
// and the moved generator voltage targets.
#ifndef OLF_GPU_OUTERLOOP_SECONDARY_VOLTAGE_CONTROL_CUH
#define OLF_GPU_OUTERLOOP_SECONDARY_VOLTAGE_CONTROL_CUH

#include "../solver/driver_pass.cuh"
#include "../common/codes/gpu_outer_loop_kind.cuh"

namespace olfgpu {

// The batched LU holds an nCtl x nCtl matrix per scenario: a zone system that would not fit this is refused.
constexpr double SVC_MAT_BUDGET_BYTES = 512.0 * 1024 * 1024;   // zone-system workspace ceiling

// Per-solve reset: zone targets back at the network's own, nothing adjusted yet.
void svcResetSolve(BatchSolverContext* ctx, DriverPass& drv);

// One check of the loop on every live scenario (GpuOuterLoopKind::SECONDARY_VOLTAGE_CONTROL).
void svcStep(BatchSolverContext* ctx, DriverPass& drv);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_SECONDARY_VOLTAGE_CONTROL_CUH
