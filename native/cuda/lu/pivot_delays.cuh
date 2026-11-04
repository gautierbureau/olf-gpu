// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The fast engine's elimination order, learned from the pivots it cancels. cuDSS's analysis orders for fill, and on a
// structure with zero-impedance couplers some of its pivots cancel on every scenario, the base case included:
// - the last active-power balance of buses joined to each other by lines and transformers only, and to the rest by
//   couplers: those rows depend on angle differences only, so their block is singular until a row that ties one of the
//   angles to the rest is eliminated before the last of them - and the order puts the coupler's ZERO_PHI row after;
// - a coupler row whose zero-diagonal repair borrowed a row along a chain (a busbar section's balance borrowing the
//   coupler's ZERO_PHI row, which borrows the other end's balance): the two rows' 2x2 block is singular in any order;
// - a repaired row whose donor is eliminated before it, which takes back what the row borrowed.
// cuDSS pivots inside its supernodes and finds most of them a nonzero pivot; the fast engine cannot pivot. So after a
// factorization it did not trust on some lanes, the pivots that cancelled on a few of them - clamped, or below
// LU_PIVOT_RELATIVE_LIMIT of the largest |A| of their row and column - are each delayed right after their rescue: the
// later row with the largest entry of their column, whose elimination gives them back a value (the later column with
// the largest entry of their row when the column has none). A delayed pivot that still cancels moves after its new
// rescue. The next factorization runs on the analyzed order rebuilt with the delays; a pivot is a position of the
// analyzed order, its rescue too, so the delays outlive the orders they were learned on.
//
// network.xiidm's fixed contingency list (14,266 contingencies, 1,579 couplers in breaker mode): 34 delays, 0.4 % more
// multiply-adds and the same levels; the lanes the fast engine does not trust go from all of them to 0.9 %.
#ifndef OLF_GPU_LU_PIVOT_DELAYS_CUH
#define OLF_GPU_LU_PIVOT_DELAYS_CUH

#include <memory>
#include <vector>

#include "scalar_lu_plan.cuh"

namespace olfgpu {

constexpr int PIVOT_NOT_DELAYED = -2;
constexpr int PIVOT_DELAYED_LAST = -1;                       // no rescue: after every other pivot

// A delayed pivot that still cancels moves after its new rescue at most this many times: one that no rescue saves (a
// 2x2 block singular in any order of its two rows) does not keep the order changing.
constexpr int PIVOT_DELAY_MAX_MOVES = 4;

// Orders built from the delays, at most: a structure whose cancellations they do not remove keeps those lanes on the
// pivoting engine without a new plan at every factorization.
constexpr int PIVOT_DELAY_MAX_ORDERS = 64;

struct PivotDelays {
    std::vector<int> partner;                                // per analyzed position: the analyzed position it follows,
                                                             // PIVOT_DELAYED_LAST or PIVOT_NOT_DELAYED (empty: none yet)
    std::vector<int> moves;                                  // per analyzed position: how often it moved
    int delayed = 0;
    int orders = 0;                                          // orders built with delays
};

// The fast engine's plan: the analyzed order with the delays learned so far.
struct FastLuPlan {
    ScalarLuPlan scalar;
    std::vector<int> analyzedPosition;                       // per position of this order: its analyzed position
    std::vector<int> lowerStart, lowerRow, lowerEntry;       // per column: its L entries (row, LU entry), rows ascending
    int delayed = 0;                                         // the delays it was built with
};

// The plan of A (n x n, CSR rows with ascending columns) in the analyzed order rowOrder / colOrder (new index -> A's
// index) with the delays: each delayed pivot right after its partner (or after every other one), in analyzed order.
std::shared_ptr<const FastLuPlan> buildFastLuPlan(int n, const int* rowPtr, const int* colIdx, const int* rowOrder,
                                                  const int* colOrder, const PivotDelays& delays);

// Learns from one lane factorized on plan - lu: its plan.scalar.nnzLu factor values, a: its values of A (rowPtr[n],
// on the pattern) - each pivot that cancelled delayed after its rescue, or moved after a new one. True when the delays
// changed. A lane whose A is not finite teaches nothing.
bool learnPivotDelays(PivotDelays& delays, const FastLuPlan& plan, const int* rowPtr, const int* colIdx,
                      const double* lu, const double* a);

}  // namespace olfgpu

#endif  // OLF_GPU_LU_PIVOT_DELAYS_CUH
