// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The inner Newton-Raphson (OLF NewtonRaphson.run) of every scenario of the batch, each at its own iteration: fill,
// factorize and solve (UBATCH, our batched LU or robust), score and refine the linear solve, scale the step, update, and
// end each scenario's solve once it converges, diverges or exhausts its iterations. A converged scenario runs its
// outer-loop checks at once and restarts its Newton in place when one is UNSTABLE.
#ifndef OLF_GPU_SOLVER_NEWTON_ITERATION_CUH
#define OLF_GPU_SOLVER_NEWTON_ITERATION_CUH

#include "../solver/driver_pass.cuh"
#include "../solver/outer_loop_sequence.cuh"

namespace olfgpu {

// The Newton solves of every active scenario, and their outer loops (sequence), until no scenario solves.
void newtonSolve(BatchSolverContext* ctx, DriverPass& drv, OuterLoopSequence& sequence);

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_NEWTON_ITERATION_CUH
