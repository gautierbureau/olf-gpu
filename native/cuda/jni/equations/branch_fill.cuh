// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The closed- and open-branch terms of the batched Jacobian and mismatch, one thread per (element, scenario):
// the full fills, the F-only fills for the iterations that never read the Jacobian, and the closed-branch
// reactive-flow fill the reactive-limits check reads.
#ifndef OLF_GPU_EQUATIONS_BRANCH_FILL_CUH
#define OLF_GPU_EQUATIONS_BRANCH_FILL_CUH

#include "../solver/driver_pass.cuh"

namespace olfgpu {

__global__ void fillClosedBranchBatch(int ne, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                                      const double* in, const int* ix, const int* dest,
                                      const double* x, double* vals, double* F,
                                      const char* disabled, const char* active,
                                      int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal,
                                      const int* altDest, const int* selRow, const int* mode, int dynReloc,
                                      const int* pcDest, const int* pcSel, RelocTables rt);

__global__ void fillClosedBranchQbufBatch(int ne, int nScenarios, int n, int nElements, int elemOffset,
                                          const double* in, const int* ix, const int* dest,
                                          const double* x, double* F, const char* disabled, const char* active,
                                          int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal);

__global__ void fillOpenBranchBatch(int ne, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                                    const double* in, const int* ix, const int* dest,
                                    const double* x, double* vals, double* F,
                                    const char* disabled, const char* active,
                                    const int* altDest, const int* selRow, const int* mode, int dynReloc,
                                    RelocTables rt);

__global__ void fillClosedBranchBatchF(int ne, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                                       const double* in, const int* ix, const int* dest,
                                       const double* x, double* F,
                                       const char* disabled, const char* active,
                                       int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal,
                                       const int* altDest, const int* selRow, const int* mode, int dynReloc,
                                       const int* pcDest, const int* pcSel, RelocTables rt);

__global__ void fillOpenBranchBatchF(int ne, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                                     const double* in, const int* ix, const int* dest,
                                     const double* x, double* F,
                                     const char* disabled, const char* active,
                                     const int* altDest, const int* selRow, const int* mode, int dynReloc,
                                     RelocTables rt);

}  // namespace olfgpu

#endif  // OLF_GPU_EQUATIONS_BRANCH_FILL_CUH
