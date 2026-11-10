// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The linear solves of a mode-7 context on our batched LU (lu/batched_lu.cuh): the Newton step of the scenarios
// still iterating, and the sensitivity solves of the outer loops.
#ifndef OLF_GPU_SOLVER_BATCHED_LU_SOLVE_CUH
#define OLF_GPU_SOLVER_BATCHED_LU_SOLVE_CUH

#include <functional>

#include "../common/batch_layout.cuh"
#include "../solver/driver_pass.cuh"

namespace olfgpu {

struct BatchedLu;

// nS systems on our LU's last factorization, as the Newton step solves and refines them: their CSR pattern and values
// (nnz each, laid out by layout), whether the structure has zero-impedance couplers (the refinement runs on those only),
// the solved lanes' mask, the stream, and the scratch of n x nS each (the refinement's residual and correction, the
// internal steps' inside a correction) plus the refinement's per-lane score and mask. A mode-7 context's, or a test's.
struct BatchedLuSystems {
    BatchedLu* lu;
    int n;
    int nS;
    int nnz;
    const int* rowPtr;
    const int* colIdx;
    const double* vals;
    bool couplers;
    const char* lanes;
    cudaStream_t stream;
    double* residual;
    double* correction;
    double* internalResidual;
    double* internalCorrection;
    DeviceBuf& score;
    DeviceBuf& mask;
    BatchLayout layout;
    // n x layout.stride each, on interleaved values: solveNewtonStep's right-hand side and solution interleaved by
    // scenario as the values are (null: it works on the scenario-major vectors)
    double* interleavedRightHandSide = nullptr;
    double* interleavedSolution = nullptr;
};

// x = J^-1 b as cuDSS's SOLVE phase computes it under the batch context's configuration: the solve, then its internal
// refinement steps (on a structure with couplers) on the lanes of lanes (every lane when null). r and d: scratch of
// n x nS, apart from b and x.
void solveWithInternalRefinement(const BatchedLuSystems& systems, const double* b, double* x, double* r, double* d,
                                 const char* lanes);

// The Newton step's linear solve: dx = J^-1 b with the internal steps, then on a structure with couplers the adaptive
// refinement of the lanes still above the tolerance, each correction solved with the internal steps too, on those
// lanes only. afterSolve, if set, runs between the two (a trace comparing the solve with cuDSS's), dx written by then.
// b and dx are scenario-major; with the interleaved buffers set the solve works on vectors interleaved as the values
// (residual, correction and the internal ones then n x layout.stride too).
void solveNewtonStep(const BatchedLuSystems& systems, const double* b, double* dx,
                     const std::function<void()>& afterSolve = nullptr);

// dDx = J^-1 dB for every active scenario, 0 for the others (their state must not move): solveNewtonStep, then the
// one-variable equations solved exactly (singleton_substitution.cuh). dVals and dB hold the iteration's system.
void batchedLuNewtonSolve(BatchSolverContext* ctx, DriverPass& drv, int iteration);

// dDx = J^-1 dB for every slot, with cuDSS's in-solve refinement steps and the one-variable equations solved exactly;
// first: factorize every slot's dVals before.
void batchedLuSensitivitySolve(BatchSolverContext* ctx, int nS, bool first);

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_BATCHED_LU_SOLVE_CUH
