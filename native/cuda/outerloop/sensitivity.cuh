// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The sensitivity solve the incremental outer loops share: unit right-hand sides solved on every scenario's
// current factors (OLF's solveTransposed counterpart), and the gather of the observed entries.
#ifndef OLF_GPU_OUTERLOOP_SENSITIVITY_CUH
#define OLF_GPU_OUTERLOOP_SENSITIVITY_CUH

#include "../solver/driver_pass.cuh"

#include <vector>

namespace olfgpu {

// Kernels the incremental loops launch around the solve.
__global__ void setUnitRhsBatch(int n, int nScenarios, int unitRow, double* b);

__global__ void gatherSensiBatch(int n, int nScenarios, int k, int m, int ki, const int* observeCols,
                                 const double* dx, double* out);

// One unit-right-hand-side solve on every scenario's current Jacobian (dVals -> dB/dDx).
void sensitivitySolve(BatchSolverContext* ctx, DriverPass& drv, bool& first);


// Sensitivities of every scenario's converged state (the SVC and diagnostics callers).
std::vector<double> solveBatchSensitivities(BatchSolverContext* ctx, const double* states,
                                            int k, const int* unitEqRows, int m, const int* observeVarCols);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_SENSITIVITY_CUH
