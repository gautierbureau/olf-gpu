// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// PhaseControlOuterLoop (continuous mode) as an in-kernel driver phase: CONTROLLER-mode controls switched off at
// the first check, LIMITER-mode controls shifted one tap while their current exceeds the limit.
#ifndef OLF_GPU_OUTERLOOP_PHASE_CONTROL_CUH
#define OLF_GPU_OUTERLOOP_PHASE_CONTROL_CUH

#include "../solver/driver_pass.cuh"
#include "incremental_transformer_voltage_control.cuh"

namespace olfgpu {

// Per-solve reset: base taps, check count 0.
void cphResetSolve(BatchSolverContext* ctx, DriverPass& drv);

// One check of the loop on every live scenario (LOOP_PHASE_KERNEL).
void cphStep(BatchSolverContext* ctx, DriverPass& drv);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_PHASE_CONTROL_CUH
