// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// One outer loop's check (OLF's AcOuterLoop.check) on the scenarios standing at it: their bus balances at the
// converged states, the loop's own decision kernels, and the loop's verdict per scenario.
#ifndef OLF_GPU_SOLVER_OUTER_LOOP_CHECK_CUH
#define OLF_GPU_SOLVER_OUTER_LOOP_CHECK_CUH

#include <vector>

#include "../solver/driver_pass.cuh"

namespace olfgpu {

// Checks loopOrder[drv.loopIdx] on the scenarios drv.outerLoopCheckActive selects (each one counted in
// ctx->outerLoopChecks). outerLoopUnstable[s] = 1: the loop changed something for s, which re-solves. A scenario the
// loop routed to the CPU, or whose slack distribution failed under a failure behaviour that does not leave the
// residual on the slack bus, leaves the batch here (REJECTED).
void checkOuterLoop(BatchSolverContext* ctx, DriverPass& drv, std::vector<char>& outerLoopUnstable);

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_OUTER_LOOP_CHECK_CUH
