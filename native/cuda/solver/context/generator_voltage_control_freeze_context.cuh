// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The part of a batched solver context that The freeze of a generator voltage control at its reactive power (the candidates, their saved row mode and target).
#ifndef OLF_GPU_SOLVER_CONTEXT_GENERATOR_VOLTAGE_CONTROL_FREEZE_CONTEXT_CUH
#define OLF_GPU_SOLVER_CONTEXT_GENERATOR_VOLTAGE_CONTROL_FREEZE_CONTEXT_CUH

#include <cublas_v2.h>

#include <vector>

#include "../../common/device_buf.cuh"
#include "../slot_rows.cuh"
#include "../slot_table.cuh"

namespace olfgpu {

struct GeneratorVoltageControlFreezeContext {
    // GeneratorVoltageControlManager (AFTER_GENERATOR): generator VC controller buses under the max
    // transformer-controlled nominal voltage are FROZEN at their current Q while the transformers act
    // (freezeGenerationTargetQAndDisableGeneratorVoltageControl(q.eval()) — OLF's generation target is
    // the NET injection, so the row target is eval − loadQ), re-enabled at COMPLETE; plus
    // fixTransformerVoltageControls' PV-bus lists per member (the not-controlled side's component).
    int nGeneratorVoltageControlFreezeCandidates = 0;
    DeviceBuf dGeneratorVoltageControlFreezeVRow, dGeneratorVoltageControlFreezeLoadTargetQ;                       // per candidate
    DeviceBuf dGeneratorVoltageControlFrozen, dGeneratorVoltageControlFreezeSavedMode, dGeneratorVoltageControlFreezeSavedTarget;   // per (scenario, candidate)
    DeviceBuf dGeneratorVoltageControlFreezeReactiveLimitEntry;                                 // per candidate: RL entry of a REMOTE one, else -1
};

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_CONTEXT_GENERATOR_VOLTAGE_CONTROL_FREEZE_CONTEXT_CUH
