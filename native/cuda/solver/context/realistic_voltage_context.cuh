// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The part of a batched solver context that The realistic voltage check of a converged state (and its fixing loop).
#ifndef OLF_GPU_SOLVER_CONTEXT_REALISTIC_VOLTAGE_CONTEXT_CUH
#define OLF_GPU_SOLVER_CONTEXT_REALISTIC_VOLTAGE_CONTEXT_CUH

#include <cublas_v2.h>

#include <vector>

#include "../../common/device_buf.cuh"
#include "../slot_rows.cuh"
#include "../slot_table.cuh"

namespace olfgpu {

struct RealisticVoltageContext {
    // AcloadFlowEngine's realistic-voltage check, run on the INTERMEDIATE states of the driver's
    // sweep: per row, 1 = a non-fictitious bus's BUS_V at or above minNominalVoltageRealisticVoltageCheck.
    DeviceBuf dRealisticRow, dRealisticGate, dRealisticBad;
    int realisticCheck = 0;
    double realisticMinV = 0.0, realisticMaxV = 0.0;
    // AcloadFlowEngine DEFERS the check until the last outer loop that can FIX an unrealistic state
    // has run (voltageRemoteControlRobustMode; ReactiveLimits and the discrete TransformerVoltageControl
    // are the two loops that declare canFixUnrealisticState). -1 = no such loop, check every solve.
    // 0 = ReactiveLimits is the last one, 1 = the discrete TransformerVoltageControl.
    int realisticFixingLoop = GpuRealisticVoltageFixingLoop::NONE;
};

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_CONTEXT_REALISTIC_VOLTAGE_CONTEXT_CUH
