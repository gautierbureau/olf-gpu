// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The part of a batched solver context that ShuntVoltageControlOuterLoop (the continuous mode): the controller groups and members, their sections, the row modes
// and targets they switch, and the rounding to sections.
#ifndef OLF_GPU_SOLVER_CONTEXT_SHUNT_VOLTAGE_CONTROL_CONTEXT_CUH
#define OLF_GPU_SOLVER_CONTEXT_SHUNT_VOLTAGE_CONTROL_CONTEXT_CUH

#include <cublas_v2.h>

#include <vector>

#include "../../common/device_buf.cuh"
#include "../slot_rows.cuh"
#include "../slot_table.cuh"

namespace olfgpu {

struct ShuntVoltageControlContext {
    // DISCRETE SHUNT VOLTAGE CONTROL — GpuOuterLoopKind::SHUNT_VOLTAGE_CONTROL (OLF order 10): ShuntVoltageControlOuterLoop
    // (WITH_GENERATOR): control ON for the first solve, LfShuntImpl.dispatchB at the first check.
    int nShuntVoltageControlMembers = 0, nShuntVoltageControlGroups = 0, nShuntVoltageControlDeltaRows = 0, nShuntVoltageControlShuntCompensators = 0, shuntVoltageControlEntryOn = 0;
    DeviceBuf dShuntVoltageControlGroupMemberStart, dShuntVoltageControlDeltaStart, dShuntVoltageControlDeltaRow, dShuntVoltageControlDeltaModeOn, dShuntVoltageControlDeltaModeOff, dShuntVoltageControlDeltaTargetOn, dShuntVoltageControlDeltaTargetOff;
    DeviceBuf dShuntVoltageControlMemberShuntBRow, dShuntVoltageControlMemberElement, dShuntVoltageControlMemberShuntCompensatorStart, dShuntVoltageControlSectionStart, dShuntVoltageControlSectionB, dShuntVoltageControlShuntCompensatorMinPosition, dShuntVoltageControlShuntCompensatorBasePosition;
    DeviceBuf dShuntVoltageControlGroupStep, dShuntVoltageControlMemberState, dShuntVoltageControlMemberContinuousB, dShuntVoltageControlShuntCompensatorPosition, dShuntVoltageControlSavedDeltaTarget;
};

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_CONTEXT_SHUNT_VOLTAGE_CONTROL_CONTEXT_CUH
