// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See lu_plan.cuh.
#include "lu_plan.cuh"

#include <chrono>
#include <cstdio>
#include <utility>

#include "../common/stats.cuh"
#include "../common/trace.cuh"

namespace olfgpu {

std::shared_ptr<LuPlan> buildLuPlan(int n, const int* rowPtr, const int* colIdx, const int* rowOrder, const int* colOrder) {
    auto plan = std::make_shared<LuPlan>();
    plan->n = n;
    plan->rowPtr.assign(rowPtr, rowPtr + n + 1);
    plan->colIdx.assign(colIdx, colIdx + rowPtr[n]);
    plan->rowOrder.assign(rowOrder, rowOrder + n);
    plan->colOrder.assign(colOrder, colOrder + n);
    plan->supernodal = buildSupernodalLuPlan(n, rowPtr, colIdx, rowOrder, colOrder);
    plan->fast = buildFastLuPlan(n, rowPtr, colIdx, rowOrder, colOrder, plan->delays);
    plan->delayedOrderLimits.luEntries = static_cast<long long>(plan->fast->scalar.nnzLu) + plan->supernodal.nStored;
    plan->delayedOrderLimits.contributions = plan->fast->scalar.contributions() + plan->supernodal.gatherMultiplyAdds();
    return plan;
}

std::shared_ptr<const FastLuPlan> fastLuPlanOf(LuPlan& plan) {
    std::lock_guard<std::mutex> lock(plan.mutex);
    return plan.fast;
}

void publishPivotDelays(LuPlan& plan, const FastLuPlan& from, PivotDelays delays) {
    delays.orders++;
    const auto start = std::chrono::steady_clock::now();
    std::shared_ptr<const FastLuPlan> next;
    try {
        next = buildFastLuPlan(plan.n, plan.rowPtr.data(), plan.colIdx.data(), plan.rowOrder.data(),
                               plan.colOrder.data(), delays, plan.delayedOrderLimits);
    } catch (const ScalarLuPlanTooLarge&) {
        // costlier than the lanes it would spare: not taken
    }
    const double buildMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::lock_guard<std::mutex> lock(plan.mutex);
    if (plan.fast.get() != &from) {
        return;
    }
    if (!next) {
        plan.delays.orders++;                                // built all the same, up to PIVOT_DELAY_MAX_ORDERS
        if (traceOn(Trace::TRACE)) {
            std::fprintf(stderr, "OLF_GPU_LU_PIVOT_DELAYS refused delayed=%d order=%d: past %lld LU entries or %lld "
                         "contributions, in %.1f ms\n", delays.delayed, plan.delays.orders,
                         plan.delayedOrderLimits.luEntries, plan.delayedOrderLimits.contributions, buildMilliseconds);
        }
        return;
    }
    plan.delays = std::move(delays);
    plan.fast = std::move(next);
    gBatchedLuPivotDelayOrders++;
    if (traceOn(Trace::TRACE)) {
        std::fprintf(stderr, "OLF_GPU_LU_PIVOT_DELAYS delayed=%d order=%d luEntries=%d contributions=%lld factorSteps=%d "
                     "built in %.1f ms\n", plan.delays.delayed, plan.delays.orders, plan.fast->scalar.nnzLu,
                     plan.fast->scalar.contributions(), plan.fast->scalar.factorSteps(), buildMilliseconds);
    }
}

std::string describeLuPlan(LuPlan& plan) {
    const std::shared_ptr<const FastLuPlan> fast = fastLuPlanOf(plan);
    return "scalar: luEntries=" + std::to_string(fast->scalar.nnzLu) + " contributions="
            + std::to_string(fast->scalar.contributions()) + " factorSteps=" + std::to_string(fast->scalar.factorSteps())
            + " " + describeScalarLuPlan(fast->scalar) + " | supernodal: " + describeSupernodalLuPlan(plan.supernodal);
}

}  // namespace olfgpu
