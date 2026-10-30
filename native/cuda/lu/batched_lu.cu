// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See batched_lu.cuh.
#include "batched_lu.cuh"

#include <stdexcept>
#include <string>

#include "../common/stats.cuh"

namespace olfgpu {

void createBatchedLu(BatchedLu& lu, const LuPlan& plan, int capacity) {
    lu.capacity = capacity;
    lu.width = 0;
    lu.supernodalPlan = &plan.supernodal;
    createScalarLu(lu.scalar, plan.scalar, capacity);
}

void factorizeLanes(BatchedLu& lu, const double* vals, const int* laneMap, int width, cudaStream_t stream) {
    lu.width = width;
    lu.hLaneMap.assign(laneMap, laneMap + width);
    lu.hPivoted.clear();
    if (width == 0) {
        return;
    }
    gBatchedLuFactorizations++;
    factorizeScalarLanes(lu.scalar, vals, laneMap, width, stream);
    const std::vector<int> trouble = readScalarTrouble(lu.scalar, stream);
    for (int a = 0; a < width; a++) {
        if (trouble[a] > 0) {
            lu.hPivoted.push_back(laneMap[a]);
        }
    }
    if (lu.hPivoted.empty()) {
        return;
    }
    if (lu.supernodal.capacity == 0) {
        createSupernodalLu(lu.supernodal, *lu.supernodalPlan, lu.capacity);
    }
    gBatchedLuPivotedLanes += static_cast<long long>(lu.hPivoted.size());
    factorizeSupernodalLanes(lu.supernodal, vals, lu.hPivoted.data(), static_cast<int>(lu.hPivoted.size()), stream);
}

void solveLanes(BatchedLu& lu, const double* b, double* x, cudaStream_t stream) {
    solveScalarLanes(lu.scalar, b, x, stream);
    if (!lu.hPivoted.empty()) {
        solveSupernodalLanes(lu.supernodal, b, x, stream);  // after the fast engine: the flagged lanes' x is its own
    }
}

const int* deviceLaneMap(BatchedLu& lu) {
    return lu.scalar.dLaneMap.as<int>();
}

std::vector<int> readPivotClamps(BatchedLu& lu, cudaStream_t stream) {
    std::vector<int> clamps(lu.width, 0);
    if (lu.hPivoted.empty()) {
        return clamps;
    }
    const std::vector<int> pivotedClamps = readSupernodalClamps(lu.supernodal, stream);
    size_t next = 0;
    for (int a = 0; a < lu.width && next < lu.hPivoted.size(); a++) {
        if (lu.hLaneMap[a] == lu.hPivoted[next]) {           // both in the lane map's order
            clamps[a] = pivotedClamps[next++];
        }
    }
    return clamps;
}

}  // namespace olfgpu
