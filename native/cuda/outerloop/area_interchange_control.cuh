// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// AcAreaInterchangeControlOuterLoop as an in-kernel driver phase (GpuOuterLoopKind::AREA_INTERCHANGE_CONTROL): each area's
// interchange mismatch, with its share of the slack bus mismatch, spread over the area's participating generators by OLF's
// ActivePowerDistribution; once every area is within its band, the slack bus mismatch over the buses without area, and
// what they cannot take over the areas, by their interchange margins.
#ifndef OLF_GPU_OUTERLOOP_AREA_INTERCHANGE_CONTROL_CUH
#define OLF_GPU_OUTERLOOP_AREA_INTERCHANGE_CONTROL_CUH

#include "../solver/driver_pass.cuh"

namespace olfgpu {

// Per-solve reset: nothing distributed yet, and every participating generator at its move from its initial target when the
// solve starts (dDistributedSlackGeneratorInitialMove, 0 without it).
void areaInterchangeControlResetSolve(BatchSolverContext* ctx, DriverPass& drv);

// One check of AreaInterchangeControl on every live scenario. With no area it is OLF's DistributedSlackOuterLoop check.
void areaInterchangeControlCheck(BatchSolverContext* ctx, DriverPass& drv);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_AREA_INTERCHANGE_CONTROL_CUH
