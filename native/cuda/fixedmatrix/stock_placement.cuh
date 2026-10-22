// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// docs/gpu-vs-olf.md section 2.3 (equations placed on another bus's row), couplers: the stock placement. A
// coupler-tree child's equations come back to its own row in a scenario that opens its coupler; the host-side
// alternate / selector rows of that fallback and the device-side pick of a stock pair.
#ifndef OLF_GPU_FIXEDMATRIX_STOCK_PLACEMENT_CUH
#define OLF_GPU_FIXEDMATRIX_STOCK_PLACEMENT_CUH

#include <cuda_runtime.h>
#include "../common/codes/gpu_row_mode.cuh"

namespace olfgpu {

/**
 * Pool order of a stock pair: [0] = the bus's OWN row, [1] = the parent coupler's DUMMY_Q row. Which one takes
 * the bus's raw q depends on TWO modes, not one:
 *   dummy row mode 4                      -> the dummy row: its PV -> PQ switch put its balance there, coupler alive;
 *   dummy row mode 0 and own row mode 4   -> the OWN row: the coupler is dead, so the equations are back home
 *                                            and its PQ balance is there (measured: rte6515 BUS-11_BBS_1 put mode 4 on
 *                                            BUS_V(VL-11_1), which holds zero-impedance#1's ZERO_V, and writing nothing left
 *                                            that row with a zero diagonal — 20 of the 6667 scenarios failed
 *                                            their linear solve exactly this way, all of them _BBS_1 outages);
 *   anything else                         -> nothing: while it SHARES, its DISTR_Q row (diverted home when the
 *                                            coupler dies) carries its q, and writing the raw q too double-counts.
 */
__host__ __device__ inline int stockPairPick(int m, const int* pool, int p, int frozen,
                                             const int* mode, const int* relocatedChildOwnRow, int dummyRow) {
    if (m == GpuRowMode::BUS_TARGET_Q) {
        return pool[p + 1];
    }
    if (m == GpuRowMode::FIXED_VARIABLE && relocatedChildOwnRow != nullptr) {
        int own = relocatedChildOwnRow[dummyRow];
        if (own >= 0 && mode[own] == GpuRowMode::BUS_TARGET_Q) {
            return pool[p];
        }
    }
    return frozen;
}

// The alternate candidate Q row for a closed-branch output, or -1 if this output does not relocate
// (only Q rows, rowEq==1) or the endpoint bus has no second candidate (alternateQRow was -1).
// STOCK-PLACEMENT fallback for the COUPLER-TREE relocation. A child bus whose P/Q equations were relocated
// onto its parent coupler's DUMMY_P / DUMMY_Q rows must get them BACK on its own rows in any scenario that
// OPENS that coupler: the coupler's element is then skipped (no ZERO_V / ZERO_PHI, no +-dummy coupling) and
// the host fixes the dummy rows with DUMMY_TARGET identities, so the relocated equations would be zeroed and
// the bus left with no balance at all. relocatedChildOwnRow[dummyRow] = that bus's own row.
//
// The selector is the DUMMY ROW itself, whose mode the host sets to 0 exactly when the coupler opens — but
// it must divert on mode 0 ONLY, not on mode 4: a relocated child SWITCHED TO PQ by its reactive limits carries mode 4 on that
// same dummy row, where its own Q balance legitimately lives. Encoded as -2-row in selectorRow, which the picks
// decode as "divert only while this row is an identity".
// The DEPENDENT-PILOT group of the bus owning an output's home row (see dependentPilotDestination), or -1.
inline int dependentGroupOfHomeRow(int frozenRow, int alternateRow, const int* relocatedChildOwnRow, const int* reactiveRelocationDependentGroupOfRow) {
    if (reactiveRelocationDependentGroupOfRow == nullptr) {
        return -1;
    }
    int own = alternateRow >= 0 ? alternateRow : frozenRow;
    return own >= 0 ? reactiveRelocationDependentGroupOfRow[own] : -1;
}

inline int stockPlacementAlternateRow(int frozenRow, int alt, const int* relocatedChildOwnRow) {
    return alt < 0 && frozenRow >= 0 && relocatedChildOwnRow != nullptr && relocatedChildOwnRow[frozenRow] >= 0
            ? relocatedChildOwnRow[frozenRow] : alt;
}

inline int stockPlacementSelectorRow(int frozenRow, int alt, const int* relocatedChildOwnRow) {
    return alt < 0 && frozenRow >= 0 && relocatedChildOwnRow != nullptr && relocatedChildOwnRow[frozenRow] >= 0
            ? -2 - frozenRow : frozenRow;
}

// A relocated child whose own q row is SUPPRESSED (a remote gen-VC member has no frozen dest: its q
// feeds the DISTR_Q rows) needs THREE destinations, not two:
//   the coupler's DUMMY_Q row  while it is SWITCHED TO PQ there (that row is mode 4),
//   its OWN row                while the parent coupler is dead this scenario (the dummy row is an
//                              identity, mode 0 — the stock placement),
//   nowhere at all             while it shares (mode 1: the DISTR_Q terms carry its q).
// Two of the three cannot be expressed with one alternate and one selector, so the pair travels in the
// reloc POOL, marked by a NEGATIVE relocationCandidateOffset, with the DUMMY row itself in the group slot as the selector.
// Measured without it (relocation on by default): the middle-section-loss fixtures left the child's own
// row empty once the contingency killed its coupler — one zero diagonal, dx 4e+12, no convergence.
inline bool stockPairOut(int frozenRow, int alt, const int* relocatedChildOwnRow) {
    return frozenRow < 0 && alt >= 0 && relocatedChildOwnRow != nullptr && relocatedChildOwnRow[alt] >= 0;
}

}  // namespace olfgpu

#endif  // OLF_GPU_FIXEDMATRIX_STOCK_PLACEMENT_CUH
