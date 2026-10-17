// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// FreezingHvdcACEmulationOuterloop and AcHvdcAcEmulationLimitsOuterLoop as in-kernel driver phases: a link frozen
// at its base flow is released, and a link whose emulated flow exceeds its limit saturates at it.
#ifndef OLF_GPU_OUTERLOOP_HVDC_AC_EMULATION_CUH
#define OLF_GPU_OUTERLOOP_HVDC_AC_EMULATION_CUH

#include "../solver/driver_pass.cuh"

namespace olfgpu {

// One check of FreezingHvdcACEmulation (LOOP_HVDC_FREEZE) or AcHvdcAcEmulationLimits (LOOP_HVDC_LIMITS).
void hvdcStep(BatchSolverContext* ctx, DriverPass& drv, int loopKind);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_HVDC_AC_EMULATION_CUH
