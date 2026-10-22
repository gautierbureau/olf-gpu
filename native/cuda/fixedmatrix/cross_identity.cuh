// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// docs/gpu-vs-olf.md section 2.4 (cross identity): a fixed variable whose own row is taken. A re-merged
// voltage-controlled bus whose own V row carries a live coupler's ZERO_V keeps its BUS_TARGET_V on the parent
// coupler's DUMMY_Q row (row mode 6, GpuRowMode::CROSS_FIXED_VARIABLE), on the bus's own V column. Device inline,
// shared by the fills (applyRowModeBatch), the reactive-limit pass and seed, the pilot voltage kernels and the
// relocated destination of a dependent pilot (reactive_relocation.cuh).
#ifndef OLF_GPU_FIXEDMATRIX_CROSS_IDENTITY_CUH
#define OLF_GPU_FIXEDMATRIX_CROSS_IDENTITY_CUH

#include <cuda_runtime.h>
#include "../common/codes/gpu_row_mode.cuh"

namespace olfgpu {

// The row holds a fixed variable this scenario: a plain identity (mode 0) or a cross identity (mode 6).
// updateDistrQTargetsBatch (outerloop/remote_voltage_control.cu) writes the same test out: through this helper the
// compiler lowers its mode 0 / 6 / 4 exits to a jump table instead of one bit-mask test.
__device__ inline bool holdsFixedVariable(int mode) {
    return mode == GpuRowMode::FIXED_VARIABLE || mode == GpuRowMode::CROSS_FIXED_VARIABLE;
}

// The fixed-variable form of a row: cross (mode 6) when it is not the variable's own row.
__device__ inline int fixedVariableMode(bool crossRow) {
    return crossRow ? GpuRowMode::CROSS_FIXED_VARIABLE : GpuRowMode::FIXED_VARIABLE;
}

// CROSS identity (tasks #70, VC_REMERGE_RELOC_VROW): this coupler DUMMY_Q row hosts the relocated child's
// reactive equation, and the child is a re-merged voltage-controlled bus whose own V row carries the live
// coupler's ZERO_V — so BUS_TARGET_V lands HERE on the child's V column (relocatedChildOwnRow[r] = its own V row =
// its V column), the slot the child's own dq/dV term already reserves in the pattern.
__device__ inline void writeCrossIdentityRow(int r, int s, int n, const int* rowPtr, const int* colIdx, const int* relocatedChildOwnRow,
                                             const double* x, const double* target, double* valsS, double* F) {
    int col = relocatedChildOwnRow != nullptr ? relocatedChildOwnRow[r] : -1;
    for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
        valsS[k] = colIdx[k] == col ? 1.0 : 0.0;
    }
    F[(size_t) s * n + r] = (col >= 0 ? x[(size_t) s * n + col] : 0.0) - target[(size_t) s * n + r];
}

// The row a re-merged LOCAL control's PV -> PQ switch and switch-back write: its relocated reactive row (the
// parent coupler's DUMMY_Q, mode[relocatedIdx]) when that row holds the cross identity, or its own reactive balance while
// its own V row (mode[ownIdx]) does not: relocatedRow then, else ownRow. Row is the caller's row type (a row number or
// an index into the all-scenario arrays).
template <typename Row>
__device__ inline Row crossIdentityTargetQRow(const int* mode, size_t relocatedIdx, size_t ownIdx, Row relocatedRow, Row ownRow) {
    if (mode[relocatedIdx] == GpuRowMode::CROSS_FIXED_VARIABLE || (mode[relocatedIdx] == GpuRowMode::BUS_TARGET_Q && mode[ownIdx] != GpuRowMode::BUS_TARGET_Q)) {
        return relocatedRow;
    }
    return ownRow;
}

}  // namespace olfgpu

#endif
