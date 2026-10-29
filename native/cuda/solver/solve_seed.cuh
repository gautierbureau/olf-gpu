// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The per-solve seed of a fresh (non-resumed) full load flow: the generators' slack moves cleared, the pilot Q
// targets captured, every reactive-limit state (PV or PQ) seeded (from the base or a previous solve) and stamped on the rows,
// and every in-kernel outer loop reset to its entry state.
#ifndef OLF_GPU_SOLVER_SOLVE_SEED_CUH
#define OLF_GPU_SOLVER_SOLVE_SEED_CUH

#include "../solver/driver_pass.cuh"

namespace olfgpu {

// Seed one fresh full load flow; reactiveLimitsSeedState = a previous solve's reactive-limit states (or null). A strategy
// pair's parent slack total comes from the context (slackBusAnchorShift).
void seedSolve(BatchSolverContext* ctx, DriverPass& drv, const int* reactiveLimitsSeedState);

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_SOLVE_SEED_CUH
