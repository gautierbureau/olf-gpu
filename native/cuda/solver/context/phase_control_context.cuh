// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The part of a batched solver context that PhaseControlOuterLoop: the phase shifters, their mode and current limit, and their taps.
#ifndef OLF_GPU_SOLVER_CONTEXT_PHASE_CONTROL_CONTEXT_CUH
#define OLF_GPU_SOLVER_CONTEXT_PHASE_CONTROL_CONTEXT_CUH

#include <cublas_v2.h>

#include <vector>

#include "../../common/device_buf.cuh"
#include "../slot_rows.cuh"
#include "../slot_table.cuh"

namespace olfgpu {

struct PhaseControlContext {
    // CONTINUOUS phase control (OLF PhaseControlOuterLoop, CONTINUOUS_WITH_DISCRETISATION), sharing the
    // phase slot: kind = the GpuPhaseControlMode: CONTROLLER (a1 solved free under row mode BRANCH_TARGET_P for the first
    // solve, then switched off and rounded to the closest tap at the loop's first check), LIMITER (from the second
    // check on, one tap per check against the sign of dI/da1 while I exceeds the limit — local only).
    int nPhaseControlControllerBranches = 0;
    DeviceBuf dPhaseControlMode, dPhaseControlA1Row, dPhaseControlControlledSide, dPhaseControlCurrentLimit, dPhaseControlElement, dPhaseControlTapStart, dPhaseControlTapRows, dPhaseControlBaseTap;
    DeviceBuf dPhaseControlTap, dPhaseControlAppliedTap, dPhaseControlCheckCount;                // per (scenario, controller) tap; per scenario check count
    std::vector<int> hPhaseControlElement, hPhaseControlTapStart, hPhaseControlBaseTap;
    std::vector<double> hPhaseControlTapRows;
};

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_CONTEXT_PHASE_CONTROL_CONTEXT_CUH
