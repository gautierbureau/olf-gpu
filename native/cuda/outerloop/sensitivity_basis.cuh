// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The sensitivity basis a batched sensitivity analysis has every converged scenario solve (SensitivityBasisNative): unit,
// weighted and participation right-hand sides on the scenario's Jacobian at its converged state, the solutions read back at
// the gather rows - or, with a contraction, combined and contracted with the factors' function gradients on the device, so
// that only the sensitivities come back. The incremental loops' sensitivity pattern (fillActiveBatch, rowMap, the
// zero-diagonal right-hand-side repair, sensitivitySolve), run once the driver's Newton solve has ended.
#ifndef OLF_GPU_OUTERLOOP_SENSITIVITY_BASIS_CUH
#define OLF_GPU_OUTERLOOP_SENSITIVITY_BASIS_CUH

#include <cstddef>

#include "../equations/scenario_overrides.cuh"
#include "../solver/driver_pass.cuh"

namespace olfgpu {

// b (nScenarios x n, scenario-major) = the same column for every scenario.
__global__ void broadcastColumnBatch(int n, int nScenarios, const double* column, double* b);

// out[(s * nColumns + column) * nGather + f] = dx[s * n + gatherRows[f]]: one basis column's solutions at the gather rows.
__global__ void gatherSensitivityBasisBatch(int nScenarios, int n, int nGather, const int* gatherRows, int column, int nColumns,
                                            const double* dx, double* out);

// One basis column of every scenario set to value (NaN: a column the device did not solve).
__global__ void fillSensitivityBasisColumnBatch(int nScenarios, int nGather, int column, int nColumns, double value, double* out);

// out[0, count) = value (NaN: sensitivities the device did not contract).
__global__ void fillSensitivityValuesBatch(size_t count, double value, double* out);

// Per (scenario, unit column), after the scenario's nFactors sensitivities: the row modes of the column's row and of its
// bus's active power row (FIXED_VARIABLE for none), which the coefficients were read from and the host checks them on.
__global__ void sensitivityRowModesBatch(int nScenarios, int n, int nFactors, int nUnits, const int* unitRows, const int* unitBusPRows,
                                         const int* mode, double* out);

// Per (scenario, factor): the factor's group's basis columns combined at its four gather positions, a unit column by the
// scenario's row modes (SensitivityBasisRequest.Contraction.coefficient's rule), a weighted or participation column as it
// is; then contracted with the factor's function gradient at the scenario's state: OLF's calculateSensi of a closed
// branch's flow or current (flow_observation.cuh, on the tap a loop or the caller moved) or of a bus voltage. NaN for a
// branch whose a1 or r1 is a variable (its gradient reads rows the gather does not) or a unit row the structure lacks.
__global__ void contractSensitivitiesBatch(int nScenarios, int n, int nFactors, int nUnits, int nColumns, int nGather,
                                           const int* unitRows, const int* unitBusPRows, const char* unitSlack,
                                           const int* groupColumnStart, const int* groupColumns, const int* groupColumnKind,
                                           const int* factorGroup, const int* factorFunction, const int* factorElement,
                                           const int* factorGather, const double* closedBranchParameters, const int* closedBranchIndices,
                                           ClosedBranchParameterOverrides closedBranchParameterOverrides, const int* mode,
                                           const double* x, const double* basisOut, double* out);

// Every slot's basis solutions at its current state into dSensitivityBasisOut, when the context holds a basis
// (setBatchSensitivityBasis): after the call's Newton solve, once every scenario of the call has ended. With a contraction
// (setBatchSensitivityContraction), the solutions are then contracted into the factors' sensitivities, in
// dSensitivityValuesOut with the row modes: what the host reads back instead. The UBATCH and batched-LU modes only: the
// robust mode refactorizes for every right-hand side, so its slots get NaN, which the host reads as "no basis" and solves
// on the CPU, as for a call that paused with scenarios still solving.
void sensitivityBasisStep(BatchSolverContext* ctx, DriverPass& drv);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_SENSITIVITY_BASIS_CUH
