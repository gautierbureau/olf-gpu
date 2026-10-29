// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The closed- and open-branch terms of the batched Jacobian and mismatch, one thread per (element, scenario):
// the full fills, the F-only fills for the iterations that never read the Jacobian, and the closed-branch
// reactive-flow fill the reactive-limits check reads.
#ifndef OLF_GPU_EQUATIONS_BRANCH_FILL_CUH
#define OLF_GPU_EQUATIONS_BRANCH_FILL_CUH

#include "../solver/driver_pass.cuh"
#include "scatter.cuh"
#include "../fixedmatrix/reactive_relocation.cuh"
#include "scenario_overrides.cuh"

namespace olfgpu {

__global__ void fillClosedBranchBatch(int elementCount, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                                      const double* in, const int* ix, const int* dest,
                                      const double* x, double* vals, double* F,
                                      const char* disabled, const char* active,
                                      ClosedBranchParameterOverrides closedBranchParameterOverrides,
                                      const int* mode, int dynamicRelocation, const char* relocatedElement,
                                      const int* phaseControlDestination, const int* phaseControlSelector, double* qbuf,
                                      BatchLayout layout);

// The closed branches fillClosedBranchBatch leaves out under dynamic relocation (Family::relocatedElements).
__global__ void fillRelocatedClosedBranchBatch(int relocatedCount, const int* relocatedElements, int nScenarios, int n, int nnz,
                                               int nElements, int elemOffset, const double* in, const int* ix, const int* dest,
                                               const double* x, double* vals, double* F,
                                               const char* disabled, const char* active,
                                               ClosedBranchParameterOverrides closedBranchParameterOverrides,
                                               const int* alternateDestination, const int* selectorRow, const int* mode,
                                               const int* phaseControlDestination, const int* phaseControlSelector,
                                               ReactiveRelocationTables rt, double* qbuf, BatchLayout layout);

__global__ void fillOpenBranchBatch(int elementCount, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                                    const double* in, const int* ix, const int* dest,
                                    const double* x, double* vals, double* F,
                                    const char* disabled, const char* active,
                                    const int* alternateDestination, const int* selectorRow, const int* mode, int dynamicRelocation,
                                    ReactiveRelocationTables rt, BatchLayout layout);

__global__ void fillClosedBranchBatchF(int elementCount, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                                       const double* in, const int* ix, const int* dest,
                                       const double* x, double* F,
                                       const char* disabled, const char* active,
                                       ClosedBranchParameterOverrides closedBranchParameterOverrides,
                                       const int* alternateDestination, const int* selectorRow, const int* mode, int dynamicRelocation,
                                       const char* relocatedElement, const int* phaseControlDestination, const int* phaseControlSelector,
                                       ReactiveRelocationTables rt, double* qbuf, BatchLayout layout);

__global__ void fillOpenBranchBatchF(int elementCount, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                                     const double* in, const int* ix, const int* dest,
                                     const double* x, double* F,
                                     const char* disabled, const char* active,
                                     const int* alternateDestination, const int* selectorRow, const int* mode, int dynamicRelocation,
                                     ReactiveRelocationTables rt, BatchLayout layout);

}  // namespace olfgpu

#endif  // OLF_GPU_EQUATIONS_BRANCH_FILL_CUH
