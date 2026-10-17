// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The zero-diagonal row repair of the batched matrices (the uniform batch has no pivoting): a row whose diagonal
// is exactly zero borrows a row that carries its diagonal column, on the values and on every right-hand side.
#ifndef OLF_GPU_SOLVER_ZERO_DIAGONAL_REPAIR_CUH
#define OLF_GPU_SOLVER_ZERO_DIAGONAL_REPAIR_CUH

#include "../solver/driver_pass.cuh"

namespace olfgpu {

__global__ void hazRepairVals(int nHaz, int nS, int nnz, int hazTot, const int* hazRow, const int* hazOff,
                              const int* hazDonorPtr, const int* hazDonor, const int* hazDonorHaz,
                              const char* hazSign, const int* rowPtr, const int* colIdx, const int* diagPos,
                              const double* snap, double* vals, int* choice, int traceNoDonor);

__global__ void hazRepairRhs(int nHaz, int nS, int n, const int* hazRow, const int* hazDonor, const int* hazDonorHaz,
                             const char* hazSign, const int* choice, const double* bsnap, double* B);

void hazRepairValues(BatchSolverContext* ctx, int nS, double* vals);

void hazRepairRightHandSide(BatchSolverContext* ctx, int nS, double* B, int sOff);

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_ZERO_DIAGONAL_REPAIR_CUH
