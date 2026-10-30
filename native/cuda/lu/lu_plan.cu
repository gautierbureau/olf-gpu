// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See lu_plan.cuh.
#include "lu_plan.cuh"

namespace olfgpu {

LuPlan buildLuPlan(int n, const int* rowPtr, const int* colIdx, const int* rowOrder, const int* colOrder) {
    LuPlan plan;
    plan.scalar = buildScalarLuPlan(n, rowPtr, colIdx, rowOrder, colOrder);
    plan.supernodal = buildSupernodalLuPlan(n, rowPtr, colIdx, rowOrder, colOrder);
    return plan;
}

std::string describeLuPlan(const LuPlan& plan) {
    return "scalar: luEntries=" + std::to_string(plan.scalar.nnzLu) + " contributions="
            + std::to_string(plan.scalar.contributions()) + " factorSteps=" + std::to_string(plan.scalar.factorSteps())
            + " " + describeScalarLuPlan(plan.scalar) + " | supernodal: " + describeSupernodalLuPlan(plan.supernodal);
}

}  // namespace olfgpu
