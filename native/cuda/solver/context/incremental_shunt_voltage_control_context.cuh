// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The part of a batched solver context that IncrementalShuntVoltageControlOuterLoop: the controller shunts and groups, their sections and deadbands, and the
// per-scenario section moves.
#ifndef OLF_GPU_SOLVER_CONTEXT_INCREMENTAL_SHUNT_VOLTAGE_CONTROL_CONTEXT_CUH
#define OLF_GPU_SOLVER_CONTEXT_INCREMENTAL_SHUNT_VOLTAGE_CONTROL_CONTEXT_CUH

#include <cublas_v2.h>

#include <vector>

#include "../../common/device_buf.cuh"
#include "../slot_rows.cuh"
#include "../slot_table.cuh"

namespace olfgpu {

struct IncrementalShuntVoltageControlContext {
    // INCREMENTAL SHUNT VOLTAGE CONTROL — in-kernel outer loop (GpuOuterLoopKind::INCREMENTAL_SHUNT_VOLTAGE_CONTROL). Controllers are
    // grouped per controlled bus (OLF adjustB round-robin, |b|-descending); each controller shunt has
    // sub-controllers (OLF LfShunt.Controller) with their own section B/G tables and ratchet.
    int nIncrementalShuntVoltageControlControllerShunts = 0, nIncrementalShuntVoltageControlGroups = 0, nIncrementalShuntVoltageControlShuntCompensators = 0, incrementalShuntVoltageControlMaxSectionShift = 3;
    DeviceBuf dIncrementalShuntVoltageControlShuntBRow, dIncrementalShuntVoltageControlControlledBusVRow, dIncrementalShuntVoltageControlTargetV, dIncrementalShuntVoltageControlHalfTargetDeadband, dIncrementalShuntVoltageControlElement, dIncrementalShuntVoltageControlShuntAdmittanceOverrideElement;   // per controller
    DeviceBuf dIncrementalShuntVoltageControlGroupControllerStart, dIncrementalShuntVoltageControlGroupController;                   // CSR: group -> controllers (sorted)
    DeviceBuf dIncrementalShuntVoltageControlShuntCompensatorStart;                                    // CSR: controller -> subs
    DeviceBuf dIncrementalShuntVoltageControlSectionStart, dIncrementalShuntVoltageControlSectionB, dIncrementalShuntVoltageControlSectionG, dIncrementalShuntVoltageControlShuntCompensatorMinPosition, dIncrementalShuntVoltageControlShuntCompensatorBasePosition;   // per sub: section tables
    DeviceBuf dIncrementalShuntVoltageControlPosition, dIncrementalShuntVoltageControlCurrentDirection, dIncrementalShuntVoltageControlDirectionChangeCount, dIncrementalShuntVoltageControlAllowedDirection, dIncrementalShuntVoltageControlSectionShift;   // per (scenario, sub)
    DeviceBuf dIncrementalShuntVoltageControlSensitivity, dIncrementalShuntVoltageControlBaseB, dIncrementalShuntVoltageControlAppliedDeltaB;           // per (scenario, controller)
    DeviceBuf dIncrementalShuntVoltageControlOutOfDeadband, dIncrementalShuntVoltageControlDiffV;                        // per (scenario, group)
    std::vector<int> hIncrementalShuntVoltageControlShuntBRow, hIncrementalShuntVoltageControlShuntAdmittanceOverrideElement, hIncrementalShuntVoltageControlShuntCompensatorStart, hIncrementalShuntVoltageControlSectionStart, hIncrementalShuntVoltageControlShuntCompensatorBasePosition;
    std::vector<double> hIncrementalShuntVoltageControlSectionB, hIncrementalShuntVoltageControlSectionG;
};

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_CONTEXT_INCREMENTAL_SHUNT_VOLTAGE_CONTROL_CONTEXT_CUH
