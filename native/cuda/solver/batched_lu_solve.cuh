// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The linear solves of a mode-7 context on our batched LU (lu/batched_lu.cuh): the Newton step of the scenarios
// still iterating, and the sensitivity solves of the outer loops.
#ifndef OLF_GPU_SOLVER_BATCHED_LU_SOLVE_CUH
#define OLF_GPU_SOLVER_BATCHED_LU_SOLVE_CUH

#include "../solver/driver_pass.cuh"

namespace olfgpu {

// dDx = J^-1 dB for every active scenario, 0 for the others (their state must not move), with the adaptive
// refinement on a structure with zero-impedance couplers and the one-variable equations solved exactly
// (singleton_substitution.cuh). dVals and dB hold the iteration's system.
void batchedLuNewtonSolve(BatchSolverContext* ctx, DriverPass& drv, int iteration);

// dDx = J^-1 dB for every slot, the one-variable equations solved exactly; first: factorize every slot's dVals before.
void batchedLuSensitivitySolve(BatchSolverContext* ctx, int nS, bool first);

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_BATCHED_LU_SOLVE_CUH
