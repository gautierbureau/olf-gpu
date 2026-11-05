// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The symbolic half of our batched LU's fast engine: from one CSR pattern and an elimination order, everything its
// device factorization and triangular solves read, built once per structure and shared by every bank width.
//
// The factorization is LEFT-LOOKING, one item per LU entry: U[i][j] = B[i][j] - sum_k L[i][k] U[k][j] and
// L[i][j] = (B[i][j] - sum_k L[i][k] U[k][j]) / U[j][j], k running over the pivots before min(i, j) where both factors
// are structurally nonzero, always in ascending k (deterministic, no atomics). A pivot's level is one more than the
// level of every pivot its row or column depends on; step 2 * level computes the U entries of the pivot rows of that
// level (diagonal included), step 2 * level + 1 the L entries of their columns, divided by the pivot just computed.
//
// No pivoting: the fill is the EXACT structural fill of Gaussian elimination in the given order on the unsymmetric
// pattern (a PV bus's identity V row stays a single entry, and is solved exactly), not the symmetrized superset.
#ifndef OLF_GPU_LU_SCALAR_LU_PLAN_CUH
#define OLF_GPU_LU_SCALAR_LU_PLAN_CUH

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace olfgpu {

// entryDivisor of a U entry: off the diagonal, or the pivot itself (the factorization checks it)
constexpr int SCALAR_ITEM_UPPER = -1;
constexpr int SCALAR_ITEM_PIVOT = -2;

struct ScalarLuPlan {
    int n = 0;
    int nnzA = 0;
    int nnzLu = 0;
    // B = A(rowOrder, colOrder): row i of B is row rowOrder[i] of A, column j of B is column colOrder[j] of A
    std::vector<int> rowOrder, colOrder;
    // the LU pattern of B in CSR, every row ascending: its L part, the diagonal at luDiag[i], its U part
    std::vector<int> luRowPtr, luCol, luDiag;
    std::vector<int> luSource;                               // per LU entry: its entry in A's CSR, -1 for a fill entry
    // factorization: step s computes the LU entries stepEntry[stepStart[s] .. stepStart[s + 1])
    std::vector<int> stepStart, stepEntry;
    std::vector<int> entryDivisor;                           // per LU entry: the pivot's LU entry (L), or SCALAR_ITEM_*
    std::vector<int> contributionStart;                      // per LU entry: its contributions, nnzLu + 1 offsets
    std::vector<int> contributionL, contributionU;           // LU entries L[i][k] and U[k][j], k ascending
    // triangular solves: the rows of each level, in level order (forward from the first row, backward from the last)
    std::vector<int> forwardLevelStart, forwardRow;
    std::vector<int> backwardLevelStart, backwardRow;

    long long contributions() const { return static_cast<long long>(contributionL.size()); }
    int factorSteps() const { return static_cast<int>(stepStart.size()) - 1; }
    int forwardLevels() const { return static_cast<int>(forwardLevelStart.size()) - 1; }
    int backwardLevels() const { return static_cast<int>(backwardLevelStart.size()) - 1; }
};

// The sizes a plan may reach: its LU entries and its contributions (the multiply-adds of one factorization). The plan
// indexes both with ints, so neither goes past INT_MAX whatever a caller asks.
struct ScalarLuPlanLimits {
    long long luEntries = std::numeric_limits<int>::max();
    long long contributions = std::numeric_limits<int>::max();
};

// Thrown by buildScalarLuPlan as soon as the plan would pass its limits.
struct ScalarLuPlanTooLarge : std::length_error {
    using std::length_error::length_error;
};

// The plan of A (n x n, CSR rows with ascending columns) eliminated in the order rowOrder / colOrder (each a
// permutation of 0..n-1, new index -> A's index). A diagonal entry B[i][i] missing from the pattern is added as fill:
// its pivot then is a structural zero the factorization reports. Throws ScalarLuPlanTooLarge as soon as the plan
// would pass limits, before its contribution lists are allocated.
ScalarLuPlan buildScalarLuPlan(int n, const int* rowPtr, const int* colIdx, const int* rowOrder, const int* colOrder,
                               const ScalarLuPlanLimits& limits = ScalarLuPlanLimits());

// One line on how the factorization's work spreads over its steps: the steps narrower than a warp, the widest step,
// the longest contribution list, and the share of the multiply-adds in the last tenth of the steps.
std::string describeScalarLuPlan(const ScalarLuPlan& plan);

}  // namespace olfgpu

#endif  // OLF_GPU_LU_SCALAR_LU_PLAN_CUH
