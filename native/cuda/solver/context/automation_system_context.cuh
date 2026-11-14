// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The part of a batched solver context that AutomationSystemOuterLoop: the overload management systems, their monitored elements, thresholds and trippings.
#ifndef OLF_GPU_SOLVER_CONTEXT_AUTOMATION_SYSTEM_CONTEXT_CUH
#define OLF_GPU_SOLVER_CONTEXT_AUTOMATION_SYSTEM_CONTEXT_CUH

#include <cublas_v2.h>

#include <vector>

#include "../../common/device_buf.cuh"
#include "../slot_rows.cuh"
#include "../slot_table.cuh"

namespace olfgpu {

struct AutomationSystemContext {
    // AutomationSystem (setBatchAutomationSystem, GpuOuterLoopKind::AUTOMATION_SYSTEM): OLF's AutomationSystemOuterLoop, its
    // last loop. Per overload management system: its monitored closed-branch element and side (GpuTwoSides) and the slice
    // of its trippings (dOverloadManagementSystemTrippingStart, nSystems + 1). Per tripping: the disabled-mask slot of the
    // closed branch it operates (also kept host-side for the read-back), whether it opens it, its current threshold (pu).
    int nOverloadManagementSystems = 0, nOverloadManagementSystemTrippings = 0;
    DeviceBuf dOverloadManagementSystemMonitoredElement, dOverloadManagementSystemMonitoredSide, dOverloadManagementSystemTrippingStart;
    DeviceBuf dOverloadManagementSystemOperatedMaskSlot, dOverloadManagementSystemOpenAction, dOverloadManagementSystemThreshold;
    std::vector<int> hOverloadManagementSystemOperatedMaskSlot;
    DeviceBuf dAutomationSystemSwitched;                                            // per (scenario, tripping): a check's decisions
};

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_CONTEXT_AUTOMATION_SYSTEM_CONTEXT_CUH
