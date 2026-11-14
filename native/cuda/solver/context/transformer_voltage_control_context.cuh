// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The part of a batched solver context that TransformerVoltageControlOuterLoop (the continuous mode): the controller groups and members, the row modes and targets
// they switch, the rounding to taps, and the PV fix of a generator bus.
#ifndef OLF_GPU_SOLVER_CONTEXT_TRANSFORMER_VOLTAGE_CONTROL_CONTEXT_CUH
#define OLF_GPU_SOLVER_CONTEXT_TRANSFORMER_VOLTAGE_CONTROL_CONTEXT_CUH

#include <cublas_v2.h>

#include <vector>

#include "../../common/device_buf.cuh"
#include "../slot_rows.cuh"
#include "../slot_table.cuh"

namespace olfgpu {

struct TransformerVoltageControlContext {
    // DISCRETE (non-incremental) TRANSFORMER VOLTAGE CONTROL — GpuOuterLoopKind::TRANSFORMER_VOLTAGE_CONTROL (OLF order 8): OLF's
    // TransformerVoltageControlOuterLoop (AFTER_GENERATOR_VOLTAGE_CONTROL: INITIAL/CONTROL/COMPLETE) or
    // SimpleTransformerVoltageControlOuterLoop (WITH_GENERATOR_VOLTAGE_CONTROL). One group per controlled bus
    // (CSR over members); the group's ON/OFF activations are the toggle-diffed row deltas (CSR).
    int nTransformerVoltageControlMembers = 0, nTransformerVoltageControlGroups = 0, nTransformerVoltageControlDeltaRows = 0, transformerVoltageControlMode = GpuTransformerVoltageControlMode::AFTER_GENERATOR_VOLTAGE_CONTROL, transformerVoltageControlEntryOn = 0;
    DeviceBuf dTransformerVoltageControlGroupMemberStart, dTransformerVoltageControlGroupControlledBusVRow, dTransformerVoltageControlGroupTargetV, dTransformerVoltageControlGroupHalfTargetDeadband, dTransformerVoltageControlGroupUseInitialTapPosition;
    DeviceBuf dTransformerVoltageControlDeltaStart, dTransformerVoltageControlDeltaRow, dTransformerVoltageControlDeltaModeOn, dTransformerVoltageControlDeltaModeOff, dTransformerVoltageControlDeltaTargetOn, dTransformerVoltageControlDeltaTargetOff;
    DeviceBuf dTransformerVoltageControlMemberRhoRow, dTransformerVoltageControlMemberElement, dTransformerVoltageControlMemberBaseTap, dTransformerVoltageControlMemberTapStart, dTransformerVoltageControlMemberMinR1, dTransformerVoltageControlMemberMaxR1, dTransformerVoltageControlMemberInitialR1, dTransformerVoltageControlTapRows;
    DeviceBuf dTransformerVoltageControlStep, dTransformerVoltageControlGroupStep, dTransformerVoltageControlGroupSharedMinR1, dTransformerVoltageControlGroupSharedMaxR1, dTransformerVoltageControlGroupSharedInitialR1;   // per scenario / per (scenario, group)
    DeviceBuf dTransformerVoltageControlMemberState, dTransformerVoltageControlMemberTap, dTransformerVoltageControlMemberRoundingR1;                          // per (scenario, member)
    DeviceBuf dTransformerVoltageControlSavedDeltaTarget, dTransformerVoltageControlRejected;                                               // per (scenario, delta row) / per scenario
    std::vector<int> hTransformerVoltageControlMemberElement, hTransformerVoltageControlMemberBaseTap, hTransformerVoltageControlMemberTapStart;
    std::vector<double> hTransformerVoltageControlTapRows;
    DeviceBuf dTransformerVoltageControlFixPvRow;                                  // the fix universe: every gen-VC bus's V row
    SlotRows<int> transformerVoltageControlFixMinMask{0}, transformerVoltageControlFixMaxMask{0};   // per (slot, member): fixWords ints each
    int transformerVoltageControlFixWords = 0;
};

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_CONTEXT_TRANSFORMER_VOLTAGE_CONTROL_CONTEXT_CUH
