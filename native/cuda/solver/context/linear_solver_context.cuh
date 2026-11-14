// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The part of a batched solver context that the linear solve of a Newton step reads and writes: the cuDSS objects and analyses, our batched LU,
// the compaction of the solve to the scenarios still solving, the iterative refinement, the singleton columns, the
// comparison of the two solvers, and the zero-diagonal repair.
#ifndef OLF_GPU_SOLVER_CONTEXT_LINEAR_SOLVER_CONTEXT_CUH
#define OLF_GPU_SOLVER_CONTEXT_LINEAR_SOLVER_CONTEXT_CUH

#include <cublas_v2.h>

#include <vector>

#include "../../common/device_buf.cuh"
#include "../slot_rows.cuh"
#include "../slot_table.cuh"

namespace olfgpu {

struct LinearSolverContext {
    DeviceBuf dZeroDiagonalRowSnapshot, dZeroDiagonalRightHandSideSnapshot, dZeroDiagonalDonorChoice;               // zero-diagonal repair scratch (lazy)
    int zeroDiagonalRepairCapacity = 0;
    // ---- solve-only batch compaction (UBATCH, tol > 0): lazily allocated on first engage ----
    DeviceBuf dValsCompacted, dBCompacted, dDxCompacted, dCompactedScenarioMap;                      // nnz*cap / n*cap / n*cap / cap ints
    CudssCtx compactedSolver;                                            // dedicated handle/config/data for the compact width
    cudssMatrix_t compactedUniformBatchSystemMatrix = nullptr, compactedUniformBatchRightHandSideMatrix = nullptr, compactedUniformBatchSolutionMatrix = nullptr;
    cudssMatrix_t compactedUniformBatchResidualMatrix = nullptr, compactedUniformBatchDeltaMatrix = nullptr;   // adaptive refinement on the compacted solve
    cudssMatrix_t uniformBatchResidualMatrix = nullptr, uniformBatchDeltaMatrix = nullptr;     // ...and on the full-width UBATCH solve
    DeviceBuf dRCompacted, dDeltaCompacted;
    int compactedCapacity = 0;                                      // buffer capacity (nS/2)
    int compactWidth = 0;                                    // currently ANALYZED ubatch width (0 = none)
    // One ANALYZED cuDSS data object per compact width: a data object's
    // buffers belong to the batch size it was analyzed at, but a width seen before -- the next chunk walks the
    // same tiers, and so does the next run on this context -- can reuse its analysis instead of redoing it.
    std::map<int, cudssData_t> compactData;
    DeviceBuf dLinRp, dLinRm, dLinSc;                      // linear-solve guard scratch, reused across passes
    CudssCtx solver;
    bool analyzed = false;
    int mode = 0;                                            // 3 = UBATCH, 6 = per-scenario loop, robust (pivoted) config,
                                                             // 7 = our batched LU
    std::unique_ptr<BatchedLu> batchedLu;                    // BATCHED_LU: the factors, one lane per slot
    // BATCHED_LU with couplers: the residual and correction of the internal refinement steps inside a refinement's own
    // correction solve (batched_lu_solve.cu solveWithInternalRefinement)
    DeviceBuf dInternalRefinementResidual, dInternalRefinementCorrection;
    // BATCHED_LU: the exact substitution of the one-variable equations after each solve (singleton_substitution.cuh)
    DeviceBuf dSingletonHazardSlot, dSingletonColumnStart, dSingletonColumnRow, dSingletonList, dSingletonPendingList,
            dSingletonFixed, dSingletonTouched, dSingletonOwner, dSingletonProposal, dSingletonProposalValue, dSingletonCounters;
    CudssCtx luComparisonSolver;                             // OLF_GPU_LU_COMPARE: cuDSS's uniform batch on the same systems
    DeviceBuf dLuComparisonDx, dLuComparisonScore, dLuComparisonValues;   // the values' scenario-major copy, when interleaved
    bool luComparisonAnalyzed = false;
    DeviceBuf dR, dDelta, dIterativeRefinementScore, dIterativeRefinementScenarioMask;                 // adaptive refinement: residual, correction, lane scores/mask
    // UBATCH single matrices (that mode only): a SINGLE per-block CSR (n, nnz, shared rowPtr/colIdx)
    // whose value/RHS pointers are the batched buffers (dVals = nS*nnz, dB/dDx = nS*n); cuDSS
    // analyzes ONE block (cheap) and factorizes/solves all nS via CUDSS_CONFIG_UBATCH_SIZE.
    cudssMatrix_t uniformBatchSystemMatrix = nullptr, uniformBatchRightHandSideMatrix = nullptr, uniformBatchSolutionMatrix = nullptr;
};

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_CONTEXT_LINEAR_SOLVER_CONTEXT_CUH
