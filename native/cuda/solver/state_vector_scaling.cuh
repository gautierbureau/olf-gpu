// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The Newton step applied to the state: the plain update, the line-search backtrack (OLF
// LineSearchStateVectorScaling) and the maximum voltage change clamp (MaxVoltageChangeStateVectorScaling).
#ifndef OLF_GPU_SOLVER_STATE_VECTOR_SCALING_CUH
#define OLF_GPU_SOLVER_STATE_VECTOR_SCALING_CUH

#include "../solver/driver_pass.cuh"

namespace olfgpu {

__global__ void stepFromPrevBatchMasked(int n, int total, const double* xPrev, const double* dx,
                                        const double* mu, double* x);

__global__ void negateIntoOne(int n, const double* F, double* b);

__global__ void axpy1One(int n, const double* dx, double* x);

__global__ void negateIntoBatch(int total, const double* F, double* b);

__global__ void axpy1Batch(int total, const double* dx, double* x);

__global__ void maxVoltageChangeFactorBatch(int n, const double* dx, const int* varType,
                                            const char* active, double maxDv, double maxDphi,
                                            double* factor);

__global__ void axpy1BatchScaled(int n, int total, const double* dx, const double* factor, double* x);

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_STATE_VECTOR_SCALING_CUH
