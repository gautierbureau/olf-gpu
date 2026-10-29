// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// IncrementalShuntVoltageControlOuterLoop as an in-kernel driver phase: per controlled bus out of its deadband,
// dV/db from the converged Jacobian and the section move, ratcheted, with the shunt admittance overrides.
#ifndef OLF_GPU_OUTERLOOP_INCREMENTAL_SHUNT_VOLTAGE_CONTROL_CUH
#define OLF_GPU_OUTERLOOP_INCREMENTAL_SHUNT_VOLTAGE_CONTROL_CUH

#include "../solver/driver_pass.cuh"
#include "../common/codes/gpu_outer_loop_kind.cuh"

namespace olfgpu {

// Per-solve reset: base sections, fresh ratchets, the identity row's target as the base b.
void incrementalShuntVoltageControlResetSolve(BatchSolverContext* ctx, DriverPass& drv);

// One check of the loop on every live scenario (GpuOuterLoopKind::INCREMENTAL_SHUNT_VOLTAGE_CONTROL).
void incrementalShuntVoltageControlStep(BatchSolverContext* ctx, DriverPass& drv);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_INCREMENTAL_SHUNT_VOLTAGE_CONTROL_CUH
