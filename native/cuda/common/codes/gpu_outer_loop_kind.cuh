// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The outer loops the device driver runs, named after OLF's loops (getName() in the comments). Mirrors
// com.powsybl.openloadflow.ac.solver.gpu.codes.GpuOuterLoopKind (GpuCodesMirrorTest checks they agree).
#ifndef OLF_GPU_COMMON_CODES_GPU_OUTER_LOOP_KIND_CUH
#define OLF_GPU_COMMON_CODES_GPU_OUTER_LOOP_KIND_CUH

namespace GpuOuterLoopKind {
constexpr int NONE = -1;                                           // no loop: no loop triggered the re-solve
constexpr int DISTRIBUTED_SLACK = 0;                                // DistributedSlack
constexpr int FREEZING_HVDC_AC_EMULATION = 1;                       // FreezingHvdcACEmulation
constexpr int HVDC_AC_EMULATION_LIMITS = 2;                         // AcHvdcAcEmulationLimits
constexpr int MONITORING_VOLTAGE = 3;                               // VoltageMonitoring (MonitoringVoltageOuterLoop)
constexpr int REACTIVE_LIMITS = 4;                                  // ReactiveLimits
constexpr int INCREMENTAL_TRANSFORMER_VOLTAGE_CONTROL = 5;          // IncrementalTransformerVoltageControl
constexpr int INCREMENTAL_SHUNT_VOLTAGE_CONTROL = 6;                // IncrementalShuntVoltageControl
constexpr int INCREMENTAL_PHASE_CONTROL = 7;                        // IncrementalPhaseControl (CONTROLLER + LIMITER)
constexpr int INCREMENTAL_TRANSFORMER_REACTIVE_POWER_CONTROL = 8;   // IncrementalTransformerReactivePowerControl
constexpr int TRANSFORMER_VOLTAGE_CONTROL = 9;                      // TransformerVoltageControl (AFTER_GENERATOR) and SimpleTransformerVoltageControl (WITH_GENERATOR)
constexpr int SHUNT_VOLTAGE_CONTROL = 10;                           // ShuntVoltageControl (WITH_GENERATOR)
constexpr int SECONDARY_VOLTAGE_CONTROL = 11;                       // SecondaryVoltageControl
constexpr int AREA_INTERCHANGE_CONTROL = 12;                        // AreaInterchangeControl
constexpr int COUNT = 13;                                           // number of kinds
}

#endif
