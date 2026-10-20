// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See state_vector_scaling.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/codes/gpu_step_clamp_variable.cuh"
#include "../solver/driver_steps.cuh"
#include "state_vector_scaling.cuh"

namespace olfgpu {

// Per-scenario masked backtrack step: x = xPrev + mu[s]*dx for scenarios with mu[s] >= 0;
// mu[s] < 0 leaves the scenario's state untouched (not backtracking this pass).
__global__ void stepFromPrevBatchMasked(int n, int total, const double* xPrev, const double* dx,
                                        const double* mu, double* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        double m = mu[i / n];
        if (m >= 0.0) x[i] = xPrev[i] + m * dx[i];
    }
}

__global__ void negateIntoOne(int n, const double* F, double* b) {       // single scenario slice
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = -F[i];
}

__global__ void axpy1One(int n, const double* dx, double* x) {           // single scenario slice
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += dx[i];
}

__global__ void negateIntoBatch(int total, const double* F, double* b) {  // b = -F over all scenarios
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) b[i] = -F[i];
}

__global__ void axpy1Batch(int total, const double* dx, double* x) {      // x += dx over all scenarios
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) x[i] += dx[i];
}

// Per-scenario MAX_VOLTAGE_CHANGE step factor (mirrors MaxVoltageChangeStateVectorScaling, exactly
// like the base solver's host-side clamp but one factor PER SCENARIO): the largest |Δv| (varType 1)
// stays within maxDv and the largest |Δφ| (varType 0) within maxDphi; the WHOLE step is scaled.
// One block per scenario, block-stride column loop, shared-memory max reduction.
__global__ void maxVoltageChangeFactorBatch(int n, const double* dx, const int* varType,
                                            const char* active, double maxDv, double maxDphi,
                                            double* factor) {
    int s = blockIdx.x;
    __shared__ double smDv[BLOCK];
    __shared__ double smDphi[BLOCK];
    double mdv = 0.0;
    double mdphi = 0.0;
    if (active == nullptr || active[s]) {
        const double* d = dx + (size_t) s * n;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            double a = fabs(d[i]);
            if (varType[i] == GpuStepClampVariable::BUS_V) {
                mdv = fmax(mdv, a);
            } else if (varType[i] == GpuStepClampVariable::BUS_PHI) {
                mdphi = fmax(mdphi, a);
            }
        }
    }
    smDv[threadIdx.x] = mdv;
    smDphi[threadIdx.x] = mdphi;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            smDv[threadIdx.x] = fmax(smDv[threadIdx.x], smDv[threadIdx.x + off]);
            smDphi[threadIdx.x] = fmax(smDphi[threadIdx.x], smDphi[threadIdx.x + off]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        double f = 1.0;
        if (smDv[0] > maxDv) f = fmin(f, maxDv / smDv[0]);
        if (smDphi[0] > maxDphi) f = fmin(f, maxDphi / smDphi[0]);
        factor[s] = f;
    }
}

__global__ void axpy1BatchScaled(int n, int total, const double* dx, const double* factor, double* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) x[i] += factor[i / n] * dx[i];
}

}  // namespace olfgpu
