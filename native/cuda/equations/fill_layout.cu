// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See fill_layout.cuh.
#include "fill_layout.cuh"

#include "../common/check.cuh"

namespace olfgpu {

namespace {

constexpr int TILE = 32;                                     // a block: 32 rows x 32 scenarios, 8 threads high
constexpr int TILE_ROWS = 8;

// Block (bx, by) holds rows 32 bx .. 32 bx + 31 of scenarios 32 by .. 32 by + 31. The target is read along the rows,
// a retired scenario's not at all, the mismatch written along the scenarios.
__global__ void initInterleavedMismatchBatch(int n, int nS, int stride, const double* __restrict__ target,
                                             double* __restrict__ F, const char* __restrict__ active,
                                             double* __restrict__ qbuf) {
    __shared__ double tile[TILE][TILE + 1];
    const int r0 = blockIdx.x * TILE;
    const int s0 = blockIdx.y * TILE;
    for (int j = threadIdx.y; j < TILE; j += TILE_ROWS) {
        const int s = s0 + j;
        const int r = r0 + threadIdx.x;
        if (s < nS && r < n && active[s]) {
            tile[j][threadIdx.x] = target[static_cast<size_t>(s) * n + r];
        }
    }
    __syncthreads();
    for (int j = threadIdx.y; j < TILE; j += TILE_ROWS) {
        const int r = r0 + j;
        const int s = s0 + threadIdx.x;
        if (s < nS && r < n && active[s]) {
            F[static_cast<size_t>(r) * stride + s] = -tile[threadIdx.x][j];
            if (qbuf != nullptr) {
                qbuf[static_cast<size_t>(r) * stride + s] = 0.0;
            }
        }
    }
}

__global__ void copyToScenarioMajorBatch(int n, int nS, int stride, const double* __restrict__ from,
                                         double* __restrict__ to, const char* __restrict__ active) {
    __shared__ double tile[TILE][TILE + 1];
    const int r0 = blockIdx.x * TILE;
    const int s0 = blockIdx.y * TILE;
    for (int j = threadIdx.y; j < TILE; j += TILE_ROWS) {
        const int r = r0 + j;
        const int s = s0 + threadIdx.x;
        if (s < nS && r < n && (active == nullptr || active[s])) {     // a scenario left out is not read either
            tile[j][threadIdx.x] = from[static_cast<size_t>(r) * stride + s];
        }
    }
    __syncthreads();
    for (int j = threadIdx.y; j < TILE; j += TILE_ROWS) {
        const int s = s0 + j;
        const int r = r0 + threadIdx.x;
        if (s < nS && r < n && (active == nullptr || active[s])) {
            to[static_cast<size_t>(s) * n + r] = tile[threadIdx.x][j];
        }
    }
}

dim3 tilesOf(int n, int nS) {
    return dim3((n + TILE - 1) / TILE, (nS + TILE - 1) / TILE);
}

}  // namespace

void initInterleavedMismatch(int n, BatchLayout layout, const double* target, double* F, const char* active,
                             double* qbuf, cudaStream_t stream) {
    initInterleavedMismatchBatch<<<tilesOf(n, layout.nS), dim3(TILE, TILE_ROWS), 0, stream>>>(n, layout.nS,
            layout.stride, target, F, active, qbuf);
    CUDA_CHECK(cudaGetLastError());
}

void copyToScenarioMajor(int n, BatchLayout layout, const double* from, double* to, const char* active,
                         cudaStream_t stream) {
    copyToScenarioMajorBatch<<<tilesOf(n, layout.nS), dim3(TILE, TILE_ROWS), 0, stream>>>(n, layout.nS,
            layout.stride, from, to, active);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace olfgpu
