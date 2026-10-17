// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// DeviceBuf: a grow-only, zero-initialised device allocation freed by its destructor.
#ifndef OLF_GPU_COMMON_DEVICE_BUF_CUH
#define OLF_GPU_COMMON_DEVICE_BUF_CUH

#include <execinfo.h>

#include <cstdio>
#include <cstdlib>

#include "check.cuh"

struct DeviceBuf {
    void* p = nullptr;
    size_t cap = 0;                                          // bytes currently allocated
    ~DeviceBuf() { if (p) cudaFree(p); }
    template <typename T> T* as() { return static_cast<T*>(p); }
    /** Grow-only: keeps the buffer when it is already big enough, frees and reallocates when it is
     *  not. The old alloc() cudaMalloc'd over a live pointer on every call and leaked the previous
     *  allocation — any per-pass re-upload (the in-kernel TVC loop rebuilds the tap-override list
     *  every time a tap moves) needs a buffer that can be refilled without leaking. */
    void ensure(size_t bytes) {
        if (bytes > cap) {
            if (p) { cudaFree(p); p = nullptr; }
            CUDA_CHECK(cudaMalloc(&p, bytes));
            cap = bytes;
            // Every fresh allocation is ZEROED. A buffer read before it is
            // written holds whatever the previous owner of that device memory left: zeros in a fresh process,
            // stale content after other solves in the same one - which is how a read-before-write showed up
            // as an in-suite-only flake (2026-10-06: devicePilotLosingItsOwnGeneratorWithPinningSurvivor 3 of
            // 7 class runs, 2e-6 kV or a device failure, 0 of 5 with this on; the culprit was dRlActive read by
            // the seed launch of updatePilotVeqBatch, whose comment assumed "rlActive is still 0"). Zeroing
            // makes that assumption a guarantee instead of a property of the process's allocation history;
            // the cost is one memset per cudaMalloc, at context creation, not per iteration.
            CUDA_CHECK(cudaMemset(p, 0, bytes));
            // OLF_GPU_ALLOC_TRACE=1 prints each allocation's index and size.
            static long allocIndex = -1;
            ++allocIndex;
            static const char* trace = std::getenv("OLF_GPU_ALLOC_TRACE");
            if (trace != nullptr) {
                std::fprintf(stderr, "ALLOC #%ld %zu bytes\n", allocIndex, bytes);
                if (std::atol(trace) == allocIndex) {        // OLF_GPU_ALLOC_TRACE=<k>: host backtrace of that one
                    void* frames[24];
                    int nf = backtrace(frames, 24);
                    backtrace_symbols_fd(frames, nf, 2);
                }
            }
        }
    }
    void alloc(size_t bytes) { ensure(bytes); }
    void upload(const void* host, size_t bytes) {
        alloc(bytes);
        CUDA_CHECK(cudaMemcpy(p, host, bytes, cudaMemcpyHostToDevice));
    }
};

#endif  // OLF_GPU_COMMON_DEVICE_BUF_CUH
