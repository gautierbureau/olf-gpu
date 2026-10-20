// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The outer loops the device driver runs, named after OLF's loops (getName() in the comments). Mirrors
// com.powsybl.openloadflow.ac.solver.gpu.codes.GpuOuterLoopKind (GpuCodesMirrorTest checks they agree).
#ifndef OLF_GPU_COMMON_CODES_GPU_OUTER_LOOP_KIND_CUH
#define OLF_GPU_COMMON_CODES_GPU_OUTER_LOOP_KIND_CUH

namespace GpuOuterLoopKind {
constexpr int NONE = -1;                                           // no loop: no restart sweep start, no trigger
constexpr int NOT_PAUSED = 0;                                      // readBatchDriverPaused when not paused (a pause is a HOST_* kind)
constexpr int DISTRIBUTED_SLACK = 0;                                // DistributedSlack
constexpr int FREEZING_HVDC_AC_EMULATION = 1;                       // FreezingHvdcACEmulation
constexpr int HVDC_AC_EMULATION_LIMITS = 2;                         // AcHvdcAcEmulationLimits
constexpr int MONITORING_VOLTAGE = 3;                               // VoltageMonitoring (MonitoringVoltageOuterLoop)
constexpr int REACTIVE_LIMITS = 4;                                  // ReactiveLimits
constexpr int HOST_SECONDARY_VOLTAGE_CONTROL = 5;                   // SecondaryVoltageControl, run by OLF at a driver pause
constexpr int HOST_PHASE_CONTROL = 6;                               // PhaseControl / IncrementalPhaseControl, run by OLF at a driver pause
constexpr int HOST_TRANSFORMER_VOLTAGE_CONTROL = 7;                 // the transformer voltage control loops, run by OLF at a driver pause
constexpr int HOST_TRANSFORMER_REACTIVE_POWER_CONTROL = 8;          // IncrementalTransformerReactivePowerControl, run by OLF at a driver pause
constexpr int HOST_SHUNT_VOLTAGE_CONTROL = 9;                       // the shunt voltage control loops, run by OLF at a driver pause
constexpr int INCREMENTAL_TRANSFORMER_VOLTAGE_CONTROL = 10;         // IncrementalTransformerVoltageControl
constexpr int INCREMENTAL_SHUNT_VOLTAGE_CONTROL = 11;               // IncrementalShuntVoltageControl
constexpr int INCREMENTAL_PHASE_CONTROL = 12;                       // IncrementalPhaseControl (CONTROLLER + LIMITER)
constexpr int INCREMENTAL_TRANSFORMER_REACTIVE_POWER_CONTROL = 13;  // IncrementalTransformerReactivePowerControl
constexpr int TRANSFORMER_VOLTAGE_CONTROL = 14;                     // TransformerVoltageControl (AFTER_GENERATOR) and SimpleTransformerVoltageControl (WITH_GENERATOR)
constexpr int SHUNT_VOLTAGE_CONTROL = 15;                           // ShuntVoltageControl (WITH_GENERATOR)
constexpr int SECONDARY_VOLTAGE_CONTROL = 16;                       // SecondaryVoltageControl
constexpr int COUNT = 17;                                           // number of kinds
}

// The host-loop mask bit of a HOST_* kind (bit 0 = HOST_SECONDARY_VOLTAGE_CONTROL); GpuOuterLoopKind.hostMaskBit.
inline int hostLoopMaskBit(int kind) {
    return 1 << (kind - GpuOuterLoopKind::HOST_SECONDARY_VOLTAGE_CONTROL);
}

#endif
