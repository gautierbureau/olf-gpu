// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// What both batched LU variants (scalar_lu.cu, supernodal_lu.cu) do alike with their lanes: the transposes between the
// scenario-major and the lane-interleaved layouts, the zeroing of the fill entries, and the uploads of their plans.
#ifndef OLF_GPU_LU_LU_LANES_CUH
#define OLF_GPU_LU_LU_LANES_CUH

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/batch_layout.cuh"
#include "../common/check.cuh"
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

// One thread per (item, lane): the item t and the lane a of this thread, false past the last one.
__device__ __forceinline__ bool itemLane(int nItems, int width, const int* __restrict__ selected, int& t, int& a) {
    const size_t g = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= static_cast<size_t>(nItems) * width) {
        return false;
    }
    t = static_cast<int>(g / width);
    a = laneOf(static_cast<int>(g - static_cast<size_t>(t) * width), selected);
    return true;
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

// The head of a factorization of width lanes, alike for both variants: the width checked against the capacity, the lane
// map kept (host and device), the per-lane flags of the factorization (laneFlags) cleared, and scenario-major values
// copied into the lane-interleaved layout (through entryOfA, the stored entry of each A entry) with the fill entries
// zeroed. Interleaved values are left where they are, read in place by the factorization. False: no lane to factorize.
template <typename Lu>
bool beginLaneFactorization(Lu& lu, const double* vals, BatchLayout layout, const int* laneMap, int width, DeviceBuf& laneFlags,
                            DeviceBuf& entryOfA, cudaStream_t stream) {
    if (width < 0 || width > lu.capacity) {
        throw std::runtime_error("batched LU width " + std::to_string(width) + " outside 0.."
                                 + std::to_string(lu.capacity));
    }
    lu.width = width;
    lu.hLaneMap.assign(laneMap, laneMap + width);
    if (width == 0) {
        return false;
    }
    CUDA_CHECK(cudaMemcpyAsync(lu.dLaneMap.p, laneMap, width * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(laneFlags.p, 0, width * sizeof(int), stream));
    if (!layout.interleaved) {
        const dim3 tileBlock(LU_TILE, LU_TILE_ROWS);
        const dim3 valuesGrid((lu.nnzA + LU_TILE - 1) / LU_TILE, (width + LU_TILE - 1) / LU_TILE);
        toInterleaved<<<valuesGrid, tileBlock, 0, stream>>>(lu.nnzA, lu.nnzA, width, lu.capacity, lu.dLaneMap.template as<int>(),
                nullptr, entryOfA.template as<int>(), vals, lu.dLu.template as<double>());
        if (lu.nFill > 0) {
            zeroFill<<<blocksFor(static_cast<size_t>(lu.nFill) * width), LU_BLOCK, 0, stream>>>(lu.nFill, width,
                    lu.capacity, lu.dFillEntry.template as<int>(), lu.dLu.template as<double>());
        }
    }
    return true;
}

}  // namespace olfgpu

#endif  // OLF_GPU_LU_LU_LANES_CUH
