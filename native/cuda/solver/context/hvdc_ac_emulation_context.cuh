// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The part of a batched solver context that The HVDC lines in AC emulation: their converter rows and parameters, the limits loop, and a contingency's outage.
#ifndef OLF_GPU_SOLVER_CONTEXT_HVDC_AC_EMULATION_CONTEXT_CUH
#define OLF_GPU_SOLVER_CONTEXT_HVDC_AC_EMULATION_CONTEXT_CUH

#include <cublas_v2.h>

#include <vector>

#include "../../common/device_buf.cuh"
#include "../slot_rows.cuh"
#include "../slot_table.cuh"

namespace olfgpu {

struct HvdcAcEmulationContext {
    // HVDC AC-emulation outer loops (FreezingHvdcACEmulation + AcHvdcAcEmulationLimits) as DRIVER
    // PHASES rather than host orchestrators. OLF runs both BETWEEN DistributedSlack and
    // ReactiveLimits; run from the host after the whole solve they land after RL instead, which is
    // what put the device's last DS check on a state before the PV -> PQ switches.
    int nHvdcAcEmulation = 0;                                             // AC-emulation HVDCs in the loop tables
    int hvdcAcEmulationLimitsActive = 0;                                  // AcHvdcAcEmulationLimits configured
    DeviceBuf dHvdcElement, dHvdcBus1PRow, dHvdcBus2PRow;               // per hvdc: the rows the P TARGETS are written to
    DeviceBuf dHvdcBus1PhiColumn, dHvdcBus2PhiColumn;                        // per hvdc: the angle COLUMNS the emulation reads (#68)
    DeviceBuf dHvdcAcEmulationParameters;                                         // per hvdc x9: p0 k pMax12 pMax21 loss1 loss2 r baseD1 baseD2
    SlotRows<char> hvdcContingencyOutage{0};                              // per (slot, hvdc): outaged BY the contingency
    SlotRows<int> hvdcAcEmulationStatus{0};                               // per (slot, hvdc): LINEAR/SAT_1_2/SAT_2_1/FROZEN
    SlotRows<double> hvdcAppliedDelta1{0.0}, hvdcAppliedDelta2{0.0};     // the converter-row delta currently APPLIED
};

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_CONTEXT_HVDC_AC_EMULATION_CONTEXT_CUH
