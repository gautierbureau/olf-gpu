// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// AcloadFlowEngine's realistic-voltage check on an intermediate state: a scenario whose bus voltage leaves the
// plausible range is routed to the CPU, which replays it with OLF's own decision.
#ifndef OLF_GPU_SOLVER_REALISTIC_VOLTAGE_CHECK_CUH
#define OLF_GPU_SOLVER_REALISTIC_VOLTAGE_CHECK_CUH

#include "../solver/driver_pass.cuh"

namespace olfgpu {

void rejectUnrealistic(BatchSolverContext* ctx, DriverPass& drv, const std::vector<char>& gate);

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_REALISTIC_VOLTAGE_CHECK_CUH
