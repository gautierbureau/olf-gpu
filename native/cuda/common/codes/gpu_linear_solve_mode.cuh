// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// How a batch context factors its matrices (ctx->mode). Mirrors
// com.powsybl.openloadflow.ac.solver.gpu.codes.GpuLinearSolveMode (GpuCodesMirrorTest checks they agree).
#ifndef OLF_GPU_COMMON_CODES_GPU_LINEAR_SOLVE_MODE_CUH
#define OLF_GPU_COMMON_CODES_GPU_LINEAR_SOLVE_MODE_CUH

namespace GpuLinearSolveMode {
constexpr int UBATCH = 3;  // one per-block matrix, CUDSS_CONFIG_UBATCH_SIZE, no pivoting
constexpr int ROBUST = 6;  // per-scenario single-matrix solve with pivoting and matching
constexpr int BATCHED_LU = 7;  // our batched sparse LU on the scenarios still iterating
}

#endif
