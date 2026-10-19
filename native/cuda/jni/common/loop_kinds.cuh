// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The outer loops' display names and the device's decision-log capacity.
#ifndef OLF_GPU_COMMON_LOOP_KINDS_CUH
#define OLF_GPU_COMMON_LOOP_KINDS_CUH

#include "codes/gpu_ac_emulation_status.cuh"
#include "codes/gpu_outer_loop_kind.cuh"

// The outer loops the device driver can run: codes/gpu_outer_loop_kind.cuh.
inline const char* olKindName(int k) {
    switch (k) {
        case GpuOuterLoopKind::DISTRIBUTED_SLACK: return "DistributedSlack";
        case GpuOuterLoopKind::FREEZING_HVDC_AC_EMULATION: return "FreezingHvdcACEmulation";
        case GpuOuterLoopKind::HVDC_AC_EMULATION_LIMITS: return "AcHvdcAcEmulationLimits";
        case GpuOuterLoopKind::MONITORING_VOLTAGE: return "MonitoringVoltage";
        case GpuOuterLoopKind::REACTIVE_LIMITS: return "ReactiveLimits";
        case GpuOuterLoopKind::HOST_SECONDARY_VOLTAGE_CONTROL: return "HOST:SecondaryVoltageControl";
        case GpuOuterLoopKind::HOST_PHASE_CONTROL: return "HOST:PhaseControl";
        case GpuOuterLoopKind::HOST_TRANSFORMER_VOLTAGE_CONTROL: return "HOST:TransformerVoltageControl";
        case GpuOuterLoopKind::HOST_TRANSFORMER_REACTIVE_POWER_CONTROL: return "HOST:TransformerReactivePowerControl";
        case GpuOuterLoopKind::HOST_SHUNT_VOLTAGE_CONTROL: return "HOST:ShuntVoltageControl";
        case GpuOuterLoopKind::INCREMENTAL_TRANSFORMER_VOLTAGE_CONTROL: return "TransformerVoltageControl";
        case GpuOuterLoopKind::INCREMENTAL_SHUNT_VOLTAGE_CONTROL: return "ShuntVoltageControl";
        case GpuOuterLoopKind::INCREMENTAL_PHASE_CONTROL: return "PhaseControl";
        case GpuOuterLoopKind::INCREMENTAL_TRANSFORMER_REACTIVE_POWER_CONTROL: return "TransformerReactivePowerControl";
        case GpuOuterLoopKind::TRANSFORMER_VOLTAGE_CONTROL: return "TransformerVoltageControl(after-gen)";
        case GpuOuterLoopKind::SHUNT_VOLTAGE_CONTROL: return "ShuntVoltageControl(with-gen)";
        case GpuOuterLoopKind::SECONDARY_VOLTAGE_CONTROL: return "SecondaryVoltageControl";
        default: return "?";
    }
}

constexpr int MAX_DEVICE_LOOPS = 16;



// on-device reactive-limit decision log (OLF_GPU_SW_LOG): capacity in records, width in doubles
#define SW_LOG_CAP 1048576
#define SW_LOG_W 12

#endif  // OLF_GPU_COMMON_LOOP_KINDS_CUH
