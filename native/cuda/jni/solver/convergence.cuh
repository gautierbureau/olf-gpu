// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The inner Newton's stopping criteria (OLF NewtonRaphsonStoppingCriteria / PerEquationTypeStoppingCriteria):
// the per-scenario mismatch norms, the worst row, and their read-back.
#ifndef OLF_GPU_SOLVER_CONVERGENCE_CUH
#define OLF_GPU_SOLVER_CONVERGENCE_CUH

#include "../solver/driver_pass.cuh"

namespace olfgpu {

__global__ void residualNormsBatch(int n, const double* F, const char* active, double* out,
                                   const double* fixedVariableRowMaxMismatch, const double* equationRowMaxMismatch, const int* mode,
                                   const double* alternateRowMaxMismatch, const int* alternateRowMaxMismatchSelector);

void readNormsBatch(BatchSolverContext* ctx, DriverPass& drv, std::vector<double>& hn);

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_CONVERGENCE_CUH
