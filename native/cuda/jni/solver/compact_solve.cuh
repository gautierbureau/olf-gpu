// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Straggler compaction: the scenarios still iterating are gathered into a narrower batch, solved there and
// scattered back.
#ifndef OLF_GPU_SOLVER_COMPACT_SOLVE_CUH
#define OLF_GPU_SOLVER_COMPACT_SOLVE_CUH

#include "../solver/driver_pass.cuh"

namespace olfgpu {

__global__ void gatherScenarios(long per, int nC, const int* map, const double* src, double* dst);

__global__ void scatterScenarios(long per, int nReal, const int* map, const double* srcC, double* dst);

__global__ void scatterRowsInt(int nRows, int n, const int* slots, const int* src, int* dst);

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_COMPACT_SOLVE_CUH
