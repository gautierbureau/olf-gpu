// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The sensitivity basis a batched sensitivity analysis has every converged scenario solve (SensitivityBasisNative): unit,
// weighted and participation right-hand sides on the scenario's Jacobian at its converged state, the solutions read back at
// the gather rows. The incremental loops' sensitivity pattern (fillActiveBatch, rowMap, the zero-diagonal right-hand-side
// repair, sensitivitySolve), run once the driver's Newton solve has ended.
#ifndef OLF_GPU_OUTERLOOP_SENSITIVITY_BASIS_CUH
#define OLF_GPU_OUTERLOOP_SENSITIVITY_BASIS_CUH

#include "../solver/driver_pass.cuh"

namespace olfgpu {

// b (nScenarios x n, scenario-major) = the same column for every scenario.
__global__ void broadcastColumnBatch(int n, int nScenarios, const double* column, double* b);

// out[(s * nColumns + column) * nGather + f] = dx[s * n + gatherRows[f]]: one basis column's solutions at the gather rows.
__global__ void gatherSensitivityBasisBatch(int nScenarios, int n, int nGather, const int* gatherRows, int column, int nColumns,
                                            const double* dx, double* out);

// One basis column of every scenario set to value (NaN: a column the device did not solve).
__global__ void fillSensitivityBasisColumnBatch(int nScenarios, int nGather, int column, int nColumns, double value, double* out);

// Every slot's basis solutions at its current state into dSensitivityBasisOut, when the context holds a basis
// (setBatchSensitivityBasis): after the call's Newton solve, once every scenario of the call has ended. The UBATCH and
// batched-LU modes only: the robust mode refactorizes for every right-hand side, so its slots get NaN, which the host
// reads as "no basis" and solves on the CPU, as for a call that paused with scenarios still solving.
void sensitivityBasisStep(BatchSolverContext* ctx, DriverPass& drv);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_SENSITIVITY_BASIS_CUH
