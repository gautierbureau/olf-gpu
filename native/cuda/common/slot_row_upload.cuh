// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The rows of several bank slots of a per-slot device array (`width` values per slot, slot s's at [s * width,
// (s + 1) * width)) written in one host-to-device copy: the slot list and the rows packed into one host buffer, copied
// into a device staging buffer, and spread to their slots by one kernel. An install writes a row per installed slot into
// dozens of such arrays, a few bytes each: one blocking copy per row made it hundreds of thousands of copies per run, each
// paying a copy's fixed cost while the device waited for the next solve.
//
// The copy and the kernel go on the legacy default stream, the stream the per-row copies went on: they are ordered after
// the work already queued on every blocking stream and before the work queued after them, as the per-row copies were. A
// single row, or rows of LARGE_ROW_BYTES or more (a copy's fixed cost is small beside its bytes there, and the staging
// stays small), keep one copy per row.
#ifndef OLF_GPU_COMMON_SLOT_ROW_UPLOAD_CUH
#define OLF_GPU_COMMON_SLOT_ROW_UPLOAD_CUH

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

#include "check.cuh"
#include "device_buf.cuh"

namespace olfgpu {

template <typename T>
__global__ void scatterSlotRows(int nRows, size_t width, const int* slots, const T* rows, T* device) {
    const size_t total = static_cast<size_t>(nRows) * width;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t j = i / width;
        device[static_cast<size_t>(slots[j]) * width + (i - j * width)] = rows[i];
    }
}

class SlotRowUpload {
public:
    static constexpr size_t LARGE_ROW_BYTES = 64 * 1024;

    // Row j (its `width` values at rowOf(j)) into bank slot bankSlots[j] of `device`, for j < nRows.
    template <typename T, typename RowOf>
    void write(T* device, size_t width, int nRows, const int* bankSlots, RowOf rowOf) {
        const size_t rowBytes = width * sizeof(T);
        if (nRows <= 0 || rowBytes == 0) {
            return;
        }
        if (nRows == 1 || rowBytes >= LARGE_ROW_BYTES) {
            for (int j = 0; j < nRows; j++) {
                CUDA_CHECK(cudaMemcpy(device + static_cast<size_t>(bankSlots[j]) * width, rowOf(j), rowBytes,
                                      cudaMemcpyHostToDevice));
            }
            return;
        }
        const size_t rowsAt = (static_cast<size_t>(nRows) * sizeof(int) + 15) / 16 * 16;
        const size_t bytes = rowsAt + static_cast<size_t>(nRows) * rowBytes;
        host_.resize(bytes);
        std::memcpy(host_.data(), bankSlots, static_cast<size_t>(nRows) * sizeof(int));
        for (int j = 0; j < nRows; j++) {
            std::memcpy(host_.data() + rowsAt + static_cast<size_t>(j) * rowBytes, rowOf(j), rowBytes);
        }
        staging_.ensure(bytes);
        CUDA_CHECK(cudaMemcpy(staging_.p, host_.data(), bytes, cudaMemcpyHostToDevice));
        const size_t total = static_cast<size_t>(nRows) * width;
        const int blocks = static_cast<int>(std::min<size_t>((total + 255) / 256, 1024));
        auto* base = static_cast<unsigned char*>(staging_.p);
        scatterSlotRows<T><<<blocks, 256>>>(nRows, width, reinterpret_cast<const int*>(base),
                                            reinterpret_cast<const T*>(base + rowsAt), device);
        CUDA_CHECK(cudaGetLastError());
    }

private:
    std::vector<unsigned char> host_;
    DeviceBuf staging_;
};

}  // namespace olfgpu

#endif  // OLF_GPU_COMMON_SLOT_ROW_UPLOAD_CUH
