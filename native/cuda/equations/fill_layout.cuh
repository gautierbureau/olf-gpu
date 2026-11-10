// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The fills' mismatch and closed-branch Q buffer on the interleaved layout (common/batch_layout.cuh) and the
// scenario-major arrays every other reader uses: one tile of 32 rows x 32 scenarios per block through shared memory,
// so both sides are read and written in whole sectors.
#ifndef OLF_GPU_EQUATIONS_FILL_LAYOUT_CUH
#define OLF_GPU_EQUATIONS_FILL_LAYOUT_CUH

#include <cuda_runtime.h>

#include "../common/batch_layout.cuh"

namespace olfgpu {

// initFBatch on the interleaved layout: F = -target and, when qbuf is set, qbuf = 0, on every active scenario; target
// scenario-major.
void initInterleavedMismatch(int n, BatchLayout layout, const double* target, double* F, const char* active,
                             double* qbuf, cudaStream_t stream);

// The scenario-major copy of an interleaved array of n entries per scenario: every scenario's, or the active ones'
// when active is set (the others are neither read nor written).
void copyToScenarioMajor(int n, BatchLayout layout, const double* from, double* to, const char* active,
                         cudaStream_t stream);

}  // namespace olfgpu

#endif  // OLF_GPU_EQUATIONS_FILL_LAYOUT_CUH
