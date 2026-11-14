// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The part of a batched solver context that AreaInterchangeControlOuterLoop: the areas, their boundaries and interchange targets, the slack distribution factor of
// each area, and the loop's per-scenario scratch.
#ifndef OLF_GPU_SOLVER_CONTEXT_AREA_INTERCHANGE_CONTROL_CONTEXT_CUH
#define OLF_GPU_SOLVER_CONTEXT_AREA_INTERCHANGE_CONTROL_CONTEXT_CUH

#include <cublas_v2.h>

#include <vector>

#include "../../common/device_buf.cuh"
#include "../slot_rows.cuh"
#include "../slot_table.cuh"

namespace olfgpu {

struct AreaInterchangeControlContext {
    // AreaInterchangeControl (setBatchAreaInterchangeControl, GpuOuterLoopKind::AREA_INTERCHANGE_CONTROL): OLF's
    // AcAreaInterchangeControlOuterLoop. OLF drops the DistributedSlack loop from a run that has it and runs it after the
    // HVDC loops; it moves the distributed slack's own participants (dDistributedSlackGeneratorMove and the bus rows),
    // one area at a time. Per area: its interchange target and its share of the slack bus mismatch (pu) and the slice
    // of its boundaries (dAreaBoundaryStart, nAreas + 1). Per boundary: a GpuAreaBoundaryKind, the branch element of its
    // family, its slot in the disabled mask and its side; a tie line's half also carries {r, x, g, b, nominalV, SB} (SI),
    // OLF's SV.otherSideP inputs. Per distributed-slack bus and generator: its area, -1 for a bus without area. With no
    // area the phase is OLF's DistributedSlackOuterLoop at that place.
    bool areaInterchangeControlOuterLoop = false;
    int nAreas = 0, nAreaBoundaries = 0, slackBusArea = -1;
    double areaInterchangePMaxMismatch = 0.0;                                        // pu
    DeviceBuf dAreaInterchangeTarget, dAreaSlackDistributionFactor, dAreaBoundaryStart;
    DeviceBuf dAreaBoundaryKind, dAreaBoundaryElement, dAreaBoundaryMaskSlot, dAreaBoundarySide, dAreaBoundaryTieLineHalf;
    DeviceBuf dDistributedSlackBusArea, dDistributedSlackGeneratorArea;
    DeviceBuf dAreaInterchangeControlAreaScratch;                                   // per (scenario, area): the check's working values
    DeviceBuf dAreaInterchangeControlPreviousMove;                                  // per (scenario, generator): its move before a distribution
    DeviceBuf dAreaInterchangeControlParticipant;                                   // per (scenario, generator): still a participant
    DeviceBuf dAreaInterchangeControlDistributedActivePower;                        // per scenario: what OLF's context data accumulates (pu)
};

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_CONTEXT_AREA_INTERCHANGE_CONTROL_CONTEXT_CUH
