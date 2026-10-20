// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The launch shape of every one-thread-per-item kernel.
#ifndef OLF_GPU_COMMON_LAUNCH_CUH
#define OLF_GPU_COMMON_LAUNCH_CUH

namespace olfgpu {

constexpr int BLOCK = 64;

inline int grid(int n) { return (n + BLOCK - 1) / BLOCK; }

}  // namespace olfgpu

#endif  // OLF_GPU_COMMON_LAUNCH_CUH
