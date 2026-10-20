// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// A scenario's status at the end of a device solve (res.status). Mirrors
// com.powsybl.openloadflow.ac.solver.gpu.codes.GpuAcSolverStatus (GpuCodesMirrorTest checks they agree).
#ifndef OLF_GPU_COMMON_CODES_GPU_AC_SOLVER_STATUS_CUH
#define OLF_GPU_COMMON_CODES_GPU_AC_SOLVER_STATUS_CUH

namespace GpuAcSolverStatus {
constexpr int CONVERGED = 0;               // OLF's CONVERGED
constexpr int MAX_ITERATION_REACHED = 1;   // OLF's MAX_ITERATION_REACHED
constexpr int DIVERGED = -1;               // NaN/Inf mismatch
constexpr int REJECTED = -2;               // not solved on the device: the CPU replays it
constexpr int LINEAR_SOLVE_FAILED = -3;    // first linear solve failed: the pivoted retry may fix it
}

#endif
