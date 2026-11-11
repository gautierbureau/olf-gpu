// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See compact_solve.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../solver/driver_steps.cuh"
#include "compact_solve.cuh"

namespace olfgpu {

// Batch COMPACTION (solve-only): gather the active scenarios' per-scenario blocks into a
// contiguous compact buffer (dst[c*per+i] = src[map[c]*per+i]) so cuDSS runs a NARROWER ubatch,
// then scatter the solution back to the full-width slots. Everything else (fills, norms,
// updates) stays full-width — retirement already makes those cheap; cuDSS width was the cost.
__global__ void gatherScenarios(long per, int nC, const int* map, const double* src, double* dst) {
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= per * nC) return;
    int c = (int) (i / per);
    dst[i] = src[(long) map[c] * per + i % per];
}

__global__ void scatterScenarios(long per, int nReal, const int* map, const double* compactedSource, double* dst) {
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= per * nReal) return;                             // pad slots (c >= nReal) are DISCARDED
    int c = (int) (i / per);
    dst[(long) map[c] * per + i % per] = compactedSource[i];
}

}  // namespace olfgpu
