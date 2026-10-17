// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The inner Newton-Raphson of one outer pass (OLF NewtonRaphson.run) on every active scenario: fill, factorize
// and solve (UBATCH or robust), score and refine the linear solve, scale the step, update, and retire each
// scenario once it converges, diverges or exhausts its iterations.
#ifndef OLF_GPU_SOLVER_NEWTON_ITERATION_CUH
#define OLF_GPU_SOLVER_NEWTON_ITERATION_CUH

#include "../solver/driver_pass.cuh"

namespace olfgpu {

// One inner Newton solve of every active scenario of the pass.
void newtonSolve(BatchSolverContext* ctx, DriverPass& drv);

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_NEWTON_ITERATION_CUH
