// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The two plans of our batched LU, for one structure and the elimination order its analysis picked: the fast engine's
// (scalar, no pivoting, exact fill), on that order with the pivot delays it learns as it factorizes
// (pivot_delays.cuh), and the pivoting engine's (supernodal, pivoting inside each supernode per lane), on that order.
#ifndef OLF_GPU_LU_LU_PLAN_CUH
#define OLF_GPU_LU_LU_PLAN_CUH

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "pivot_delays.cuh"
#include "scalar_lu_plan.cuh"
#include "supernodal_lu_plan.cuh"

namespace olfgpu {

struct LuPlan {
    int n = 0;
    std::vector<int> rowPtr, colIdx;                         // the pattern, CSR with ascending columns
    std::vector<int> rowOrder, colOrder;                     // the analyzed order, new index -> the pattern's
    SupernodalLuPlan supernodal;
    // what a learned order's fast plan may hold and cost: the analyzed order's and the pivoting engine's together, what a
    // lane the order would spare costs today (pivot_delays.cuh)
    ScalarLuPlanLimits delayedOrderLimits;
    // the fast engine's plan, replaced each time the delays grow: a batched LU takes the current one at its next
    // factorization (every context of a structure shares its plan)
    std::mutex mutex;
    std::shared_ptr<const FastLuPlan> fast;
    PivotDelays delays;
};

// Both plans of A (n x n, CSR rows with ascending columns) in the order rowOrder / colOrder (new index -> A's index),
// the fast engine's without delays.
std::shared_ptr<LuPlan> buildLuPlan(int n, const int* rowPtr, const int* colIdx, const int* rowOrder, const int* colOrder);

// The fast engine's current plan.
std::shared_ptr<const FastLuPlan> fastLuPlanOf(LuPlan& plan);

// The fast engine's next plan, with these delays learned on the lanes of from: replaces the current one unless another
// batched LU replaced from already, or the plan would pass delayedOrderLimits - that order is not taken (the lanes it
// would spare stay on the pivoting engine), and counts toward PIVOT_DELAY_MAX_ORDERS all the same.
void publishPivotDelays(LuPlan& plan, const FastLuPlan& from, PivotDelays delays);

// One line per engine on its plan's shape.
std::string describeLuPlan(LuPlan& plan);

}  // namespace olfgpu

#endif  // OLF_GPU_LU_LU_PLAN_CUH
