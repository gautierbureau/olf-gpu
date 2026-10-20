// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The batched fill: one launch per term family for every active scenario, then the slack/PV identity rows, the
// zero-diagonal repair and the mismatch initialisation; with the per-scenario target overrides and the identity
// systems of retired scenarios.
#ifndef OLF_GPU_EQUATIONS_FILL_LAUNCH_CUH
#define OLF_GPU_EQUATIONS_FILL_LAUNCH_CUH

#include "../solver/driver_pass.cuh"
#include "scatter.cuh"

namespace olfgpu {

// The GROUP relocation tables of one family, or all-null when the structure has none.
ReactiveRelocationTables relocationTablesOf(Family& f, SharedStructure* structure);

// The same for the zero-impedance couplers.
ReactiveRelocationTables zeroImpedanceRelocationTables(SharedStructure* structure);

// Per-scenario slack/PV identity rows over the fixed superset pattern.
__global__ void applyRowModeBatch(int n, int nScenarios, int nnz, const int* mode, const int* rowPtr,
                                  const int* diagPos, const double* x, const double* target,
                                  double* vals, double* F, const char* active, const double* rowSlope,
                                  const int* colIdx, const int* relocatedChildOwnRow);

// F = -target for every active scenario, before the fills add the terms.
__global__ void initFBatch(int n, int nScenarios, const double* target, double* F, const char* active);

// An identity system for every retired scenario, so a singular sub-block never fails the batch.
__global__ void writeIdentityForInactive(int n, int nScenarios, int nnz, const int* rowPtr,
                                         const int* diagPos, const char* active, double* vals, double* F);

// Per-scenario target overrides on the tiled base targets (flat index -> value).
__global__ void applyTargetOverrideBatch(int nTargetOverrides, const int* targetOverrideIndex, const double* targetOverrideValue, double* target);

}  // namespace olfgpu

#endif  // OLF_GPU_EQUATIONS_FILL_LAUNCH_CUH
