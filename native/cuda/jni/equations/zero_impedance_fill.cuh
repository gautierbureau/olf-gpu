// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The zero-impedance couplers' terms of the batched Jacobian and mismatch: the dummy flows into both buses'
// balances and, on a spanning-tree edge, ZERO_V / ZERO_PHI, with the per-scenario on/off overrides.
#ifndef OLF_GPU_EQUATIONS_ZERO_IMPEDANCE_FILL_CUH
#define OLF_GPU_EQUATIONS_ZERO_IMPEDANCE_FILL_CUH

#include "../solver/driver_pass.cuh"
#include "scatter.cuh"
#include "../fixedmatrix/reactive_relocation.cuh"

namespace olfgpu {

__global__ void fillZeroImpBatch(int nZeroImpedanceBranch, int nScenarios, int n, int nnz,
                                 const double* in, const int* ix, const int* dest,
                                 const double* x, double* vals, double* F, const char* active,
                                 int nBaseClosedZeroImpedanceBranches, int nZeroImpedanceBranchCloseOverrides, const int* zeroImpedanceBranchCloseScenarioElement,
                                 int nZeroImpedanceBranchOpenOverrides, const int* zeroImpedanceBranchOpenScenarioElement,
                                 const int* zeroImpedanceAlternateDestination, const int* zeroImpedanceSelectorRow, const int* mode, int zeroImpedanceDynamicRelocation,
                                 ReactiveRelocationTables rt, int nSpanningTreeOverrides, const int* spanningTreeOverrideScenarioElementValue,
                                 const char* zeroImpedanceBranchOpenMask, const char* zeroImpedanceBranchCloseMask, const char* spanningTreeOverrideMask);

}  // namespace olfgpu

#endif  // OLF_GPU_EQUATIONS_ZERO_IMPEDANCE_FILL_CUH
