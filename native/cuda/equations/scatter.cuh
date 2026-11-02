// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The device helpers every fill kernel shares: scattering an output into the Jacobian values / the
// mismatch through a role-folded destination, and the CSR slot lookup. The relocated destinations of
// docs/gpu-vs-olf.md section 2.3 are in ../fixedmatrix/ (relocated_scatter.cuh, reactive_relocation.cuh,
// stock_placement.cuh).
#ifndef OLF_GPU_EQUATIONS_SCATTER_CUH
#define OLF_GPU_EQUATIONS_SCATTER_CUH

#include <cuda_runtime.h>
#include "../common/codes/gpu_row_mode.cuh"

namespace olfgpu {

// A scenario's values in the batch, or null for a fill that writes the mismatch only (a check's refill: its bus
// balances): scatterAdd and addAt then skip every Jacobian output.
__device__ inline double* scenarioValues(double* vals, int s, int nnz) {
    return vals != nullptr ? vals + static_cast<size_t>(s) * nnz : nullptr;
}

static __device__ void scatterAdd(int outputCount, const int* dest, const double* o, double* vals, double* F) {
    for (int k = 0; k < outputCount; k++) {
        int d = dest[k];
        if (d >= 0) {
            if (vals != nullptr) {
                atomicAdd(&vals[d], o[k]);
            }
        } else if (d <= -2) {
            atomicAdd(&F[-2 - d], o[k]);
        }
    }
}

// Scatter one (dest, value): dest >= 0 -> J slot, dest <= -2 -> F row, -1 -> skip.
__device__ inline void addAt(int dest, double val, double* vals, double* F) {
    if (dest >= 0) {
        if (vals != nullptr) {
            atomicAdd(&vals[dest], val);
        }
    } else if (dest <= -2) {
        atomicAdd(&F[-2 - dest], val);
    }
}

__device__ inline int csrFind(const int* rowPtr, const int* colIdx, int row, int col) {
    int lo = rowPtr[row];
    int hi = rowPtr[row + 1];
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (colIdx[mid] < col) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return (lo < rowPtr[row + 1] && colIdx[lo] == col) ? lo : -1;
}

static __device__ void scatterAddF(int nres, const int* dest, const int* pos, const double* o, double* F) {
    for (int k = 0; k < nres; k++) {
        int d = dest[pos[k]];
        if (d <= -2) {
            atomicAdd(&F[-2 - d], o[k]);
        }
    }
}

}  // namespace olfgpu

#endif  // OLF_GPU_EQUATIONS_SCATTER_CUH
