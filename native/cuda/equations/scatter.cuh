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
#include "../common/batch_layout.cuh"
#include "../common/codes/gpu_row_mode.cuh"

namespace olfgpu {

// A scenario's values and mismatch in the batch (batch_layout.cuh). The values are none for a fill that writes the
// mismatch only (a check's refill: its bus balances; the fill that ends a Newton budget): scatterAdd and addAt then
// skip every Jacobian output.
using ScenarioValues = ScenarioEntries<double>;

static __device__ void scatterAdd(int outputCount, const int* dest, const double* o, ScenarioValues vals, ScenarioValues F) {
    for (int k = 0; k < outputCount; k++) {
        int d = dest[k];
        if (d >= 0) {
            if (vals) {
                atomicAdd(&vals[d], o[k]);
            }
        } else if (d <= -2) {
            atomicAdd(&F[-2 - d], o[k]);
        }
    }
}

// Scatter one (dest, value): dest >= 0 -> J slot, dest <= -2 -> F row, -1 -> skip.
__device__ inline void addAt(int dest, double val, ScenarioValues vals, ScenarioValues F) {
    if (dest >= 0) {
        if (vals) {
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

// The positions of a family's N residual outputs (scatter role 0) in its output order, read off its generated scatter
// table at compile time: where a mismatch-only fill, which computes those outputs alone, scatters them.
template <int N>
struct ResidualPositions {
    int at[N];
};

template <int N, typename Scatter, int COUNT>
__host__ __device__ constexpr ResidualPositions<N> residualPositions(const Scatter (&table)[COUNT]) {
    ResidualPositions<N> positions{};
    int k = 0;
    for (int i = 0; i < COUNT; i++) {
        if (table[i].role == 0) {
            positions.at[k++] = i;
        }
    }
    return positions;
}

// How many residual outputs (scatter role 0) a family has: checked against the N its mismatch-only fill computes.
template <typename Scatter, int COUNT>
__host__ __device__ constexpr int residualCount(const Scatter (&table)[COUNT]) {
    int k = 0;
    for (int i = 0; i < COUNT; i++) {
        k += table[i].role == 0 ? 1 : 0;
    }
    return k;
}

static __device__ void scatterAddF(int nres, const int* dest, const int* pos, const double* o, ScenarioValues F) {
    for (int k = 0; k < nres; k++) {
        int d = dest[pos[k]];
        if (d <= -2) {
            atomicAdd(&F[-2 - d], o[k]);
        }
    }
}

}  // namespace olfgpu

#endif  // OLF_GPU_EQUATIONS_SCATTER_CUH
