// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// What both batched LU variants (scalar_lu.cu, supernodal_lu.cu) do alike with their lanes: the transposes between the
// scenario-major and the lane-interleaved layouts, the zeroing of the fill entries, and the uploads of their plans.
#ifndef OLF_GPU_LU_LU_LANES_CUH
#define OLF_GPU_LU_LU_LANES_CUH

#include <cstddef>
#include <vector>

#include "../common/device_buf.cuh"

namespace olfgpu {

constexpr int LU_BLOCK = 128;
constexpr int LU_TILE = 32;                                  // transposes: 32 entries x 32 lanes per block
constexpr int LU_TILE_ROWS = 8;

inline unsigned int blocksFor(size_t threads) {
    return static_cast<unsigned int>((threads + LU_BLOCK - 1) / LU_BLOCK);
}

// The lane the k-th lane slot of a launch works on: lane k of the factorization, or the k-th selected lane (a position
// in its lane map) when there is a selection.
__device__ __forceinline__ int laneOf(int k, const int* __restrict__ selected) {
    return selected == nullptr ? k : selected[k];
}

// dst[map[k] * capacity + a] = src[laneMap[a] * stride + k]: scenario-major rows to the lane-interleaved layout,
// through a shared tile so both sides stay contiguous (k along the source, lanes along the destination). The lanes:
// 0 .. width, or the width selected ones.
__global__ void toInterleaved(int count, int stride, int width, int capacity, const int* __restrict__ laneMap,
                              const int* __restrict__ selected, const int* __restrict__ map,
                              const double* __restrict__ src, double* __restrict__ dst);

// The way back: dst[laneMap[a] * stride + k] = src[map[k] * capacity + a].
__global__ void fromInterleaved(int count, int stride, int width, int capacity, const int* __restrict__ laneMap,
                                const int* __restrict__ selected, const int* __restrict__ map,
                                const double* __restrict__ src, double* __restrict__ dst);

// lu[fillEntry[f] * capacity + a] = 0 for every fill entry and lane: the entries the factorization fills in.
__global__ void zeroFill(int nFill, int width, int capacity, const int* __restrict__ fillEntry, double* __restrict__ lu);

template <typename T>
void uploadVector(DeviceBuf& buf, const std::vector<T>& v) {
    if (!v.empty()) {
        buf.upload(v.data(), v.size() * sizeof(T));
    }
}

}  // namespace olfgpu

#endif  // OLF_GPU_LU_LU_LANES_CUH
