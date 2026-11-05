// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Our batched sparse LU on the device, two engines. Every lane is factorized by the fast engine (scalar_lu.cuh: no
// pivoting, exact fill, one item per LU entry); a lane it does not trust - a clamped pivot or an outsized multiplier -
// is factorized again by the pivoting engine (supernodal_lu.cuh: each supernode's diagonal block pivoted per lane, as
// cuDSS pivots inside its supernodes), and solved by it. A sound power-flow Jacobian needs no pivoting in the order it
// is analyzed in (rte6515's 1000 branch outages: no lane flagged); the shapes that do - a generator outage relocating a
// controller's reactive balance onto a row without its diagonal - are rare and each costs one lane of the pivoting
// engine. On a network in breaker mode the analyzed order itself cancels a few pivots on every scenario, around the
// couplers, so the fast engine learns to delay them from the lanes it did not trust (pivot_delays.cuh): the pivoting
// engine eliminates in the analyzed order, the fast engine in that order with the delays learned so far.
//
// Values are interleaved by lane in both engines, entry * capacity + lane, and a factorization runs on the lanes of a
// lane map only: a scenario that no longer iterates costs nothing.
#ifndef OLF_GPU_LU_BATCHED_LU_CUH
#define OLF_GPU_LU_BATCHED_LU_CUH

#include <cuda_runtime.h>

#include <memory>
#include <vector>

#include "lu_plan.cuh"
#include "scalar_lu.cuh"
#include "supernodal_lu.cuh"

namespace olfgpu {

struct BatchedLu {
    int capacity = 0;
    int width = 0;                                           // lanes of the last factorization
    LuPlan* plan = nullptr;                                  // the structure's, shared with its other contexts
    std::shared_ptr<const FastLuPlan> fast;                  // the fast engine's plan, the one its factors are on
    ScalarLu scalar;
    SupernodalLu supernodal;                                 // built on its first lane
    std::vector<int> hLaneMap;                               // the last factorization's lanes
    std::vector<int> hPivoted;                               // ...those the pivoting engine factorized
    DeviceBuf laneFactors;                                   // one lane's fast factors, read to learn from
};

// Upload the fast engine's current plan and size its factors for capacity lanes; the plan must outlive the solver.
void createBatchedLu(BatchedLu& lu, LuPlan& plan, int capacity);

// Factorize scenarios laneMap[0 .. width) (host array, each < capacity) of vals, scenario s's CSR values laid out by
// layout (common/batch_layout.cuh): all of them on the fast engine (on the plan's current order), then the flagged ones
// on the pivoting engine, after the fast engine learned its next order from the first of them. Synchronizes the stream
// to read the flags, and once per lane it learns from.
void factorizeLanes(BatchedLu& lu, const double* vals, BatchLayout layout, const int* laneMap, int width,
                    cudaStream_t stream);

// Solve with the last factorization's lanes, each on the engine that factorized it: x[s * n ..] = A_s^-1 b[s * n ..]
// for every s of its lane map. Scenarios outside the lane map keep their x.
void solveLanes(BatchedLu& lu, const double* b, double* x, cudaStream_t stream);

// The last factorization's lane map on the device (the fast engine's).
const int* deviceLaneMap(BatchedLu& lu);

// Per lane of the last factorization's lane map: the pivots the pivoting engine clamped (0 for a lane the fast engine
// kept). Synchronizes the stream.
std::vector<int> readPivotClamps(BatchedLu& lu, cudaStream_t stream);

}  // namespace olfgpu

#endif  // OLF_GPU_LU_BATCHED_LU_CUH
