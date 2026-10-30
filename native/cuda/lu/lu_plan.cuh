// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The two plans of our batched LU, for one structure and one elimination order: the fast engine's (scalar, no
// pivoting, exact fill) and the pivoting engine's (supernodal, pivoting inside each supernode per lane).
#ifndef OLF_GPU_LU_LU_PLAN_CUH
#define OLF_GPU_LU_LU_PLAN_CUH

#include <string>

#include "scalar_lu_plan.cuh"
#include "supernodal_lu_plan.cuh"

namespace olfgpu {

struct LuPlan {
    ScalarLuPlan scalar;
    SupernodalLuPlan supernodal;
};

// Both plans of A (n x n, CSR rows with ascending columns) in the order rowOrder / colOrder (new index -> A's index).
LuPlan buildLuPlan(int n, const int* rowPtr, const int* colIdx, const int* rowOrder, const int* colOrder);

// One line per engine on its plan's shape.
std::string describeLuPlan(const LuPlan& plan);

}  // namespace olfgpu

#endif  // OLF_GPU_LU_LU_PLAN_CUH
