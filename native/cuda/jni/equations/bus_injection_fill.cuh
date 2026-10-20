// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The bus-injection terms of the batched Jacobian and mismatch: shunts, HVDC AC-emulation links and the
// voltage-dependent (ZIP) load model, full and F-only.
#ifndef OLF_GPU_EQUATIONS_BUS_INJECTION_FILL_CUH
#define OLF_GPU_EQUATIONS_BUS_INJECTION_FILL_CUH

#include "../solver/driver_pass.cuh"
#include "scatter.cuh"

namespace olfgpu {

__global__ void fillShuntBatch(int elementCount, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                               const double* in, const int* ix, const int* dest,
                               const double* x, double* vals, double* F,
                               const char* disabled, const char* active,
                               int nShuntAdmittanceOverrides, const int* shuntAdmittanceOverrideScenarioElement, const double* shuntAdmittanceOverrideDelta,
                               const int* alternateDestination, const int* selectorRow, const int* mode, int dynamicRelocation,
                               ReactiveRelocationTables rt);

__global__ void fillHvdcBatch(int elementCount, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                              const double* in, const int* ix, const int* dest,
                              const double* x, double* vals, double* F,
                              const char* disabled, const char* active,
                              const int* alternateDestination, const int* selectorRow, const int* mode, int dynamicRelocation);

__global__ void fillLoadModelBatch(int elementCount, int nScenarios, int n, int nnz,
                             const double* in, const int* ix, const int* dest,
                             const double* x, double* vals, double* F,
                             const int* alternateRow, const int* alternateDestination, const int* mode);

__global__ void fillShuntBatchF(int elementCount, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                                const double* in, const int* ix, const int* dest,
                                const double* x, double* F,
                                const char* disabled, const char* active,
                                int nShuntAdmittanceOverrides, const int* shuntAdmittanceOverrideScenarioElement, const double* shuntAdmittanceOverrideDelta,
                                const int* alternateDestination, const int* selectorRow, const int* mode, int dynamicRelocation,
                               ReactiveRelocationTables rt);

__global__ void fillHvdcBatchF(int elementCount, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                               const double* in, const int* ix, const int* dest,
                               const double* x, double* F,
                               const char* disabled, const char* active,
                               const int* alternateDestination, const int* selectorRow, const int* mode, int dynamicRelocation);

}  // namespace olfgpu

#endif  // OLF_GPU_EQUATIONS_BUS_INJECTION_FILL_CUH
