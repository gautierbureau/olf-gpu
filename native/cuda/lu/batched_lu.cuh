// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Our batched sparse LU on the device, two engines on one elimination order. Every lane is factorized by the fast
// engine (scalar_lu.cuh: no pivoting, exact fill, one item per LU entry); a lane it does not trust - a clamped pivot or
// an outsized multiplier - is factorized again by the pivoting engine (supernodal_lu.cuh: each supernode's diagonal
// block pivoted per lane, as cuDSS pivots inside its supernodes), and solved by it. A sound power-flow Jacobian needs
// no pivoting in the order it is analyzed in (rte6515's 1000 branch outages: no lane flagged); the shapes that do - a
// generator outage relocating a controller's reactive balance onto a row without its diagonal - are rare and each
// costs one lane of the pivoting engine.
//
// Values are interleaved by lane in both engines, entry * capacity + lane, and a factorization runs on the lanes of a
// lane map only: a scenario that no longer iterates costs nothing.
#ifndef OLF_GPU_LU_BATCHED_LU_CUH
#define OLF_GPU_LU_BATCHED_LU_CUH

#include <cuda_runtime.h>

#include <vector>

#include "lu_plan.cuh"
#include "scalar_lu.cuh"
#include "supernodal_lu.cuh"

namespace olfgpu {

struct BatchedLu {
    int capacity = 0;
    int width = 0;                                           // lanes of the last factorization
    const SupernodalLuPlan* supernodalPlan = nullptr;        // the pivoting engine is built on its first lane
    ScalarLu scalar;
    SupernodalLu supernodal;
    std::vector<int> hLaneMap;                               // the last factorization's lanes
    std::vector<int> hPivoted;                               // ...those the pivoting engine factorized
};

// Upload the fast engine's plan and size its factors for capacity lanes; the plan must outlive the solver.
void createBatchedLu(BatchedLu& lu, const LuPlan& plan, int capacity);

// Factorize scenarios laneMap[0 .. width) (host array, each < capacity) of vals, scenario s's CSR values at
// vals + s * nnz: all of them on the fast engine, then the flagged ones on the pivoting engine. Synchronizes the stream
// once, to read the flags.
void factorizeLanes(BatchedLu& lu, const double* vals, const int* laneMap, int width, cudaStream_t stream);

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
