// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The one-variable equations of a Newton system solved exactly after our batched LU's solve, as KLU solves them.
//
// KLU's block-triangular form gives every chain of one-variable equations a 1x1 block, solved by substitution: a
// voltage target's identity row, then a coupler's ZERO_V (v_a - v_b = 0) once v_a is known, and so on. Those values
// come out exact - a pilot held at its target through a live coupler lands ON its target - and OLF compares them with
// strict tests (the reactive-limit switch back to PV, v > targetV). Our solve gets them to rounding only, and on a
// coupler equation the zero-diagonal repair has combined with a donor row: such a pilot came out 1 ulp above its target
// where cuDSS came out 1 ulp below, and the scenario switched back to PV where OLF does not
// (GpuSaBusbarAndIslandTest.deviceReMergedMainSwitchedToPqInTheBaseUnderALiveCouplerMatchesCpu).
//
// So after the solve, the rows of the ORIGINAL system (the repair's snapshots for a hazard row) with at most
// SINGLETON_MAX_TERMS nonzero coefficients are swept: a row whose variables are all fixed but one fixes that one,
// dx[j] = (b_r - sum_fixed a_rk dx[k]) / a_rj - the lowest such row of a variable winning - until a sweep fixes nothing.
#ifndef OLF_GPU_SOLVER_SINGLETON_SUBSTITUTION_CUH
#define OLF_GPU_SOLVER_SINGLETON_SUBSTITUTION_CUH

#include "batch_context.cuh"

namespace olfgpu {

constexpr int SINGLETON_MAX_TERMS = 3;

// dx of every lane our batched LU's last factorization solved (its lane map), for the solve of b (as solved: the
// zero-diagonal repair already applied), gets the exact value of each variable a chain of one-variable equations fixes.
// b and dx scenario-major, n per scenario; any other scenario's dx is left as it is.
void substituteSingletons(BatchSolverContext* ctx, const double* b, double* dx);

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_SINGLETON_SUBSTITUTION_CUH
