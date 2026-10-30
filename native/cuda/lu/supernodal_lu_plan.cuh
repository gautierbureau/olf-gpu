// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The symbolic half of our batched LU's pivoting engine: from one CSR pattern and an elimination order, everything its
// device factorization and triangular solves read, built once per structure and shared by every bank width.
//
// SUPERNODAL, with pivoting inside each supernode, as cuDSS factors a general matrix under AMD (CUDSS_PIVOT_AUTO
// resolves to CUDSS_PIVOT_LOCAL_BLOCK: the largest element of the supernode's whole diagonal block). Without it a
// pivot that is zero in one scenario's matrix is clamped and that scenario's Newton steps go wrong (a generator
// outage relocating a reactive balance: backward error 2e-2 where cuDSS stays at 4e-16).
//
// The structure is the Cholesky structure of the symmetrized pattern, its elimination tree postordered. Consecutive
// columns that form a chain of that tree with nested structure make a supernode (at most LU_MAX_SUPERNODE columns);
// all its columns share the rows below it, R, and all its rows share the same columns R to the right of it. A
// supernode s of b columns stores a dense b x b diagonal block, its L panel (R x b, row-major: L[i][k] contiguous
// in k) and its U panel (R x b, column-major: U[k][j] contiguous in k). Because every row (column) of a supernode has
// the same pattern, reordering them inside the supernode is a data operation: each scenario pivots in its own way.
//
// Factorization, supernode level by level (a supernode after every supernode of its subtree): gather every entry of
// the level's panels, entry (i, j) -= sum over the descendants t holding both i and j in R_t of L_t[i][.] . U_t[.][j]
// (a run of b_t products, descendants in ascending order: deterministic, no atomics); factor each diagonal block with
// complete pivoting; apply the block's row order and L to the U panel, its column order and U to the L panel.
#ifndef OLF_GPU_LU_SUPERNODAL_LU_PLAN_CUH
#define OLF_GPU_LU_SUPERNODAL_LU_PLAN_CUH

#include <string>
#include <vector>

namespace olfgpu {

constexpr int LU_MAX_SUPERNODE = 16;

struct SupernodalLuPlan {
    int n = 0;
    int nnzA = 0;
    // B = A(rowOrder, colOrder): the given order with its elimination tree postordered
    std::vector<int> rowOrder, colOrder;
    int nSupernodes = 0;
    std::vector<int> supernodeStart;                         // nSupernodes + 1: the supernodes' first columns
    std::vector<int> rowStart, rows;                         // per supernode: R, ascending
    std::vector<int> diagonalBase, lowerBase, upperBase;     // per supernode: where its blocks are stored
    int nStored = 0;
    long long choleskyEntries = 0;                           // L + U of the symmetrized structure, padding excluded
    std::vector<int> storedOfA;                              // per A entry: its stored entry
    // gather: per stored entry its runs (L start, U start, length), descendants in ascending order
    std::vector<int> runStart;                               // nStored + 1
    std::vector<int> runLower, runUpper, runLength;
    // levels: the supernodes of each level, and the stored entries their gather computes
    std::vector<int> levelStart, levelSupernode;
    std::vector<int> levelEntryStart, levelEntry;
    // the L and U panel rows of each level's supernodes the apply step solves: (supernode, panel row) pairs
    std::vector<int> levelPanelStart, panelSupernode, panelRow;   // panelRow < |R|: U column; >= |R|: L row - |R|
    // forward solve: per row of B, the L runs of the descendants holding it (L start, y start, length)
    std::vector<int> forwardRunStart, forwardRunLower, forwardRunY, forwardRunLength;

    long long gatherMultiplyAdds() const;
    int supernodeSize(int s) const { return supernodeStart[s + 1] - supernodeStart[s]; }
    int rowCount(int s) const { return rowStart[s + 1] - rowStart[s]; }
    int levels() const { return static_cast<int>(levelStart.size()) - 1; }
};

// The plan of A (n x n, CSR rows with ascending columns) eliminated in the order rowOrder / colOrder (each a
// permutation of 0..n-1, new index -> A's index; the same one for a symmetric order).
SupernodalLuPlan buildSupernodalLuPlan(int n, const int* rowPtr, const int* colIdx, const int* rowOrder, const int* colOrder);

// One line on the plan's shape: supernodes, their sizes, levels, stored entries and gather work.
std::string describeSupernodalLuPlan(const SupernodalLuPlan& plan);

}  // namespace olfgpu

#endif  // OLF_GPU_LU_SUPERNODAL_LU_PLAN_CUH
