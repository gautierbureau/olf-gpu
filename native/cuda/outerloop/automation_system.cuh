// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// OLF's AutomationSystemOuterLoop as an in-kernel driver phase (GpuOuterLoopKind::AUTOMATION_SYSTEM): each overload management
// system reads the current at its monitored branch's side and, past a tripping's threshold, opens or closes that tripping's
// branch, by its bit in the scenario's disabled-element mask.
#ifndef OLF_GPU_OUTERLOOP_AUTOMATION_SYSTEM_CUH
#define OLF_GPU_OUTERLOOP_AUTOMATION_SYSTEM_CUH

#include "../solver/driver_pass.cuh"

namespace olfgpu {

// One check of AutomationSystem on every live scenario. A switch the scenario cannot represent (setBatchAutomationSystem:
// it would change the network's connectivity) routes the scenario to the CPU.
void automationSystemCheck(BatchSolverContext* ctx, DriverPass& drv);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_AUTOMATION_SYSTEM_CUH
