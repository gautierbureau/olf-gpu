// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The part of a batched solver context that SecondaryVoltageControlOuterLoop: the zones, their pilot and controller buses, the reactive terms of each controller,
// and the per-zone linear systems solved on the device.
#ifndef OLF_GPU_SOLVER_CONTEXT_SECONDARY_VOLTAGE_CONTROL_CONTEXT_CUH
#define OLF_GPU_SOLVER_CONTEXT_SECONDARY_VOLTAGE_CONTROL_CONTEXT_CUH

#include <cublas_v2.h>

#include <vector>

#include "../../common/device_buf.cuh"
#include "../slot_rows.cuh"
#include "../slot_table.cuh"

namespace olfgpu {

struct SecondaryVoltageControlContext {
    // SECONDARY VOLTAGE CONTROL — GpuOuterLoopKind::SECONDARY_VOLTAGE_CONTROL (OLF order 4). Zones of controller buses regulating a
    // pilot point: run entirely on the device, the small dense zone system included.
    int nSecondaryVoltageControlControllerBuses = 0, nSecondaryVoltageControlZones = 0;
    DeviceBuf dSecondaryVoltageControlControlledBusVRow, dSecondaryVoltageControlControllerBusVRow, dSecondaryVoltageControlReactiveLimitEntry, dSecondaryVoltageControlBaseTargetV, dSecondaryVoltageControlMinQ, dSecondaryVoltageControlMaxQ, dSecondaryVoltageControlLoadTargetQ;
    DeviceBuf dSecondaryVoltageControlReactiveTermStart, dSecondaryVoltageControlReactiveTermKind, dSecondaryVoltageControlReactiveTermSide, dSecondaryVoltageControlReactiveTermElement, dSecondaryVoltageControlReactiveTermShuntBRow;
    DeviceBuf dSecondaryVoltageControlZoneStart, dSecondaryVoltageControlZoneControllerBus, dSecondaryVoltageControlZonePilotBusVRow, dSecondaryVoltageControlZoneTargetV;
    DeviceBuf dSecondaryVoltageControlTargetV, dSecondaryVoltageControlK0, dSecondaryVoltageControlJk, dSecondaryVoltageControlJvpp, dSecondaryVoltageControlControllerBusEnabled, dSecondaryVoltageControlNeedsAdjustment, dSecondaryVoltageControlRejected, dSecondaryVoltageControlReactiveLimitTargetV;
    // Batched dense LU workspaces: the zone systems live in DEVICE memory, one N*N column-major matrix
    // and one RHS per scenario, plus the pointer/pivot/info arrays cuBLAS's batched API takes.
    DeviceBuf dSecondaryVoltageControlZoneMatrix, dSecondaryVoltageControlZoneRightHandSide, dSecondaryVoltageControlZoneMatrixPointers, dSecondaryVoltageControlZoneRightHandSidePointers, dSecondaryVoltageControlZonePivots, dSecondaryVoltageControlZoneFactorizationInfo, dSecondaryVoltageControlControllerBusZone;
    cublasHandle_t secondaryVoltageControlBlasHandle = nullptr;
    std::vector<int> hSecondaryVoltageControlControlledBusVRow;
    std::vector<double> hSecondaryVoltageControlBaseTargetV;
    double secondaryVoltageControlMinPlausibleTargetV = 0.8, secondaryVoltageControlMaxPlausibleTargetV = 1.2;
};

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_CONTEXT_SECONDARY_VOLTAGE_CONTROL_CONTEXT_CUH
