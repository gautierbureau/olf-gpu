// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// FreezingHvdcACEmulationOuterloop and AcHvdcAcEmulationLimitsOuterLoop as in-kernel driver phases: a link frozen
// at its base flow is released, and a link whose emulated flow exceeds its limit saturates at it.
#ifndef OLF_GPU_OUTERLOOP_HVDC_AC_EMULATION_CUH
#define OLF_GPU_OUTERLOOP_HVDC_AC_EMULATION_CUH

#include "../solver/driver_pass.cuh"
#include "../common/codes/gpu_outer_loop_kind.cuh"

namespace olfgpu {

// One check of FreezingHvdcACEmulation (GpuOuterLoopKind::FREEZING_HVDC_AC_EMULATION) or AcHvdcAcEmulationLimits (GpuOuterLoopKind::HVDC_AC_EMULATION_LIMITS).
void hvdcStep(BatchSolverContext* ctx, DriverPass& drv, int loopKind);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_HVDC_AC_EMULATION_CUH
