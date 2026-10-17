// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The outer loops the device driver runs, in OLF's order, and the HVDC AC-emulation status codes
// shared with the host (GpuAcNewtonSolver / DeviceLoopKind mirror these values).
#ifndef OLF_GPU_COMMON_LOOP_KINDS_CUH
#define OLF_GPU_COMMON_LOOP_KINDS_CUH

// The outer loops the device driver can run, in OLF's configured order.
enum DeviceLoopKind {
    LOOP_DS = 0,             // DistributedSlack                         (OLF order 0)
    LOOP_HVDC_FREEZE = 1,    // FreezingHvdcACEmulation                  (OLF order 1)
    LOOP_HVDC_LIMITS = 2,    // AcHvdcAcEmulationLimits                  (OLF order 2)
    LOOP_MONITORING = 3,     // MonitoringVoltage — the standby-SVC automaton (OLF order 5)
    LOOP_RL = 4,             // ReactiveLimits                           (OLF order 6)
    LOOP_HOST_SVC = 5,       // SecondaryVoltageControl — host pause     (OLF order 4, pre-RL)
    LOOP_HOST_PHASE = 6,     // PhaseControl — host pause                (OLF order 7, post-RL)
    LOOP_HOST_TVC = 7,       // TransformerVoltageControl — host pause   (OLF order 8, post-RL)
    LOOP_HOST_TRPC = 8,      // TransformerReactivePowerControl — host   (OLF order 9, post-RL)
    LOOP_HOST_SHUNT = 9,     // ShuntVoltageControl — host pause         (OLF order 10, post-RL)
    LOOP_TVC_KERNEL = 10,    // IncrementalTransformerVoltageControl — IN-KERNEL (OLF order 8)
    LOOP_SHUNT_KERNEL = 11,  // IncrementalShuntVoltageControl — IN-KERNEL (OLF order 10)
    LOOP_PHASE_KERNEL = 12,  // IncrementalPhaseControl (CONTROLLER + LIMITER) — IN-KERNEL (OLF order 7)
    LOOP_TRPC_KERNEL = 13,   // IncrementalTransformerReactivePowerControl — IN-KERNEL (OLF order 9)
    LOOP_TVD_KERNEL = 14,    // TransformerVoltageControl AFTER_GENERATOR / Simple (WITH_GENERATOR) — IN-KERNEL (OLF order 8)
    LOOP_SVD_KERNEL = 15,    // ShuntVoltageControl WITH_GENERATOR — IN-KERNEL (OLF order 10)
    LOOP_SVC_KERNEL = 16     // SecondaryVoltageControl — IN-KERNEL (OLF order 4)
};

constexpr int OL_KINDS = 17;

inline const char* olKindName(int k) {
    switch (k) {
        case 0: return "DistributedSlack";
        case 1: return "FreezingHvdcACEmulation";
        case 2: return "AcHvdcAcEmulationLimits";
        case 3: return "MonitoringVoltage";
        case 4: return "ReactiveLimits";
        case 5: return "HOST:SecondaryVoltageControl";
        case 6: return "HOST:PhaseControl";
        case 7: return "HOST:TransformerVoltageControl";
        case 8: return "HOST:TransformerReactivePowerControl";
        case 9: return "HOST:ShuntVoltageControl";
        case 10: return "TransformerVoltageControl";
        case 11: return "ShuntVoltageControl";
        case 12: return "PhaseControl";
        case 13: return "TransformerReactivePowerControl";
        case 14: return "TransformerVoltageControl(after-gen)";
        case 15: return "ShuntVoltageControl(with-gen)";
        case 16: return "SecondaryVoltageControl";
        default: return "?";
    }
}

// GpuBatchHvdcAcEmulation's status codes, shared with the host.
constexpr int HV_LINEAR = 0;
constexpr int HV_SAT_1_TO_2 = 1;
constexpr int HV_SAT_2_TO_1 = 2;
constexpr int HV_FROZEN = 3;
constexpr int MAX_DEVICE_LOOPS = 16;

#endif  // OLF_GPU_COMMON_LOOP_KINDS_CUH
