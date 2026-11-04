// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// fixedmatrix/: device-only: exists because the device keeps one fixed matrix for every contingency and factors
// without pivoting (docs/gpu-vs-olf.md section 2); OLF behaviour lives in the OLF-shaped packages.
//
// docs/gpu-vs-olf.md section 2.4 (a factorization without pivoting): the row repair.
// The zero-diagonal row repair of the batched matrices (the uniform batch has no pivoting): a row whose diagonal
// is exactly zero borrows a row that carries its diagonal column, on the values and on every right-hand side.
#ifndef OLF_GPU_FIXEDMATRIX_ZERO_DIAGONAL_REPAIR_CUH
#define OLF_GPU_FIXEDMATRIX_ZERO_DIAGONAL_REPAIR_CUH

#include "../solver/driver_pass.cuh"

namespace olfgpu {

__global__ void repairZeroDiagonalRowsBatch(int nZeroDiagonalRows, int nS, int nnz, int nZeroDiagonalRowValues, const int* zeroDiagonalRow, const int* zeroDiagonalRowValueStart,
                              const int* zeroDiagonalDonorStart, const int* zeroDiagonalDonor, const int* zeroDiagonalDonorOwnIndex,
                              const char* zeroDiagonalRepairSign, const int* rowPtr, const int* colIdx, const int* diagPos,
                              const char* active, const double* snap, double* vals, int* choice);

__global__ void repairZeroDiagonalRightHandSideBatch(int nZeroDiagonalRows, int nS, int n, const int* zeroDiagonalRow, const int* zeroDiagonalDonor, const int* zeroDiagonalDonorOwnIndex,
                             const char* zeroDiagonalRepairSign, const int* choice, const double* bsnap, double* B);

void repairZeroDiagonalRows(BatchSolverContext* ctx, int nS, double* vals);

void repairZeroDiagonalRightHandSide(BatchSolverContext* ctx, int nS, double* B, int sOff);

// Whether this context repairs zero diagonals: every mode but the robust one, which pivots and matches.
bool repairsZeroDiagonals(const BatchSolverContext* ctx);

}  // namespace olfgpu

#endif  // OLF_GPU_FIXEDMATRIX_ZERO_DIAGONAL_REPAIR_CUH
