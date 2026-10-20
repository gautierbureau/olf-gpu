// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// MonitoringVoltageOuterLoop as an in-kernel driver phase: a standby static var compensator whose monitored bus
// leaves its band switches to voltage control at the corresponding setpoint, once per solve.
#ifndef OLF_GPU_OUTERLOOP_VOLTAGE_MONITORING_CUH
#define OLF_GPU_OUTERLOOP_VOLTAGE_MONITORING_CUH

#include "../solver/driver_pass.cuh"
#include "../common/codes/gpu_outer_loop_kind.cuh"

namespace olfgpu {

// Per-solve reset: nothing has fired yet.
void monitoringVoltageResetSolve(BatchSolverContext* ctx, DriverPass& drv);

// One check of the loop on every live scenario (GpuOuterLoopKind::MONITORING_VOLTAGE).
void monitoringVoltageStep(BatchSolverContext* ctx, DriverPass& drv);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_VOLTAGE_MONITORING_CUH
