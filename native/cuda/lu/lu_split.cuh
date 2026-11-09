// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Long rows split over several warps, in both engines of our batched LU (batchedLuSplitLongRows): a level or step whose
// rows are few and long leaves most of the GPU idle with one thread per (row, lane), so each row's terms are shared
// between the warps of a block, each summing a fixed slice for 32 lanes, and the first warp adds the partial sums in
// warp order - deterministic, whatever the timing.
#ifndef OLF_GPU_LU_LU_SPLIT_CUH
#define OLF_GPU_LU_LU_SPLIT_CUH

#include <cuda_runtime.h>

#include <cstddef>

#include "../common/check.cuh"
#include "../solver/linear_solver_settings.cuh"

namespace olfgpu {

constexpr int SPLIT_LANES = 32;                              // each warp takes 32 lanes of one slice of a row's terms
constexpr int SPLIT_MAX_SLICES = 32;                         // at most a block of 32 warps per (row, 32 lanes)
constexpr int SPLIT_MIN_TERMS = 4;                           // a slice gets at least this many of the longest row's terms

// The warps each (row, 32 lanes) of a level or step is split over: 1 (the unsplit kernels) unless
// batchedLuSplitLongRows is on, then the most - a power of two up to SPLIT_MAX_SLICES - that leave every slice
// SPLIT_MIN_TERMS of the longest row's terms and keep the threads within what the GPU holds resident (budget). A level
// wide enough to fill the GPU keeps one thread per (row, lane): a bigger GPU or a wider batch splits less.
inline int splitSlices(int items, int width, int longest, size_t budget) {
    if (!gLinearSolver.batchedLuSplitLongRows) {
        return 1;
    }
    const size_t threads = static_cast<size_t>(items) * ((width + SPLIT_LANES - 1) / SPLIT_LANES) * SPLIT_LANES;
    int slices = 1;
    while (slices * 2 <= SPLIT_MAX_SLICES && longest >= slices * 2 * SPLIT_MIN_TERMS && threads * slices * 2 <= budget) {
        slices *= 2;
    }
    return slices;
}

// The threads the current device holds resident: what a split may use.
inline size_t residentDeviceThreads() {
    int device = 0;
    int multiprocessors = 0;
    int threadsPerMultiprocessor = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&threadsPerMultiprocessor, cudaDevAttrMaxThreadsPerMultiProcessor, device));
    return static_cast<size_t>(multiprocessors) * threadsPerMultiprocessor;
}

}  // namespace olfgpu

#endif  // OLF_GPU_LU_LU_SPLIT_CUH
