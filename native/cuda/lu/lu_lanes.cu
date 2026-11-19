// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See lu_lanes.cuh.
#include "lu_lanes.cuh"

namespace olfgpu {

// dst[map[k] * capacity + a] = src[laneMap[a] * stride + k]: scenario-major rows to the lane-interleaved layout,
// through a shared tile so both sides stay contiguous (k along the source, lanes along the destination). The lanes:
// 0 .. width, or the width selected ones.
__global__ void toInterleaved(int count, int stride, int width, int capacity, const int* __restrict__ laneMap,
                              const int* __restrict__ selected, const int* __restrict__ map,
                              const double* __restrict__ src, double* __restrict__ dst) {
    __shared__ double tile[LU_TILE][LU_TILE + 1];
    const int k0 = blockIdx.x * LU_TILE;
    const int c0 = blockIdx.y * LU_TILE;
    for (int r = threadIdx.y; r < LU_TILE; r += LU_TILE_ROWS) {
        const int c = c0 + r;
        const int k = k0 + threadIdx.x;
        if (c < width && k < count) {
            tile[r][threadIdx.x] = src[static_cast<size_t>(laneMap[laneOf(c, selected)]) * stride + k];
        }
    }
    __syncthreads();
    for (int r = threadIdx.y; r < LU_TILE; r += LU_TILE_ROWS) {
        const int k = k0 + r;
        const int c = c0 + threadIdx.x;
        if (c < width && k < count) {
            dst[static_cast<size_t>(map[k]) * capacity + laneOf(c, selected)] = tile[threadIdx.x][r];
        }
    }
}

// The way back: dst[laneMap[a] * stride + k] = src[map[k] * capacity + a].
__global__ void fromInterleaved(int count, int stride, int width, int capacity, const int* __restrict__ laneMap,
                                const int* __restrict__ selected, const int* __restrict__ map,
                                const double* __restrict__ src, double* __restrict__ dst) {
    __shared__ double tile[LU_TILE][LU_TILE + 1];
    const int k0 = blockIdx.x * LU_TILE;
    const int c0 = blockIdx.y * LU_TILE;
    for (int r = threadIdx.y; r < LU_TILE; r += LU_TILE_ROWS) {
        const int k = k0 + r;
        const int c = c0 + threadIdx.x;
        if (c < width && k < count) {
            tile[r][threadIdx.x] = src[static_cast<size_t>(map[k]) * capacity + laneOf(c, selected)];
        }
    }
    __syncthreads();
    for (int r = threadIdx.y; r < LU_TILE; r += LU_TILE_ROWS) {
        const int c = c0 + r;
        const int k = k0 + threadIdx.x;
        if (c < width && k < count) {
            dst[static_cast<size_t>(laneMap[laneOf(c, selected)]) * stride + k] = tile[threadIdx.x][r];
        }
    }
}

__global__ void zeroFill(int nFill, int width, int capacity, const int* __restrict__ fillEntry, double* __restrict__ lu) {
    const size_t g = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= static_cast<size_t>(nFill) * width) {
        return;
    }
    const int f = static_cast<int>(g / width);
    const int a = static_cast<int>(g - static_cast<size_t>(f) * width);
    lu[static_cast<size_t>(fillEntry[f]) * capacity + a] = 0.0;
}

}  // namespace olfgpu
