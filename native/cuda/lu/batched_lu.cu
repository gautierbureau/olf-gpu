// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See batched_lu.cuh.
#include "batched_lu.cuh"

#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include "../common/check.cuh"
#include "../common/stats.cuh"

namespace olfgpu {

namespace {

// The lanes of a factorization the fast engine did not trust that its next order is learned from: the first ones, a
// pivot that cancels on every scenario cancels on each of them.
constexpr int PIVOT_DELAY_LEARNING_LANES = 2;

void learnFromUntrustedLanes(BatchedLu& lu, const double* vals, BatchLayout layout, const std::vector<int>& untrusted,
                             cudaStream_t stream) {
    LuPlan& plan = *lu.plan;
    PivotDelays delays;
    {
        std::lock_guard<std::mutex> lock(plan.mutex);
        if (plan.fast != lu.fast || plan.delays.orders >= PIVOT_DELAY_MAX_ORDERS) {
            return;                                          // another LU moved the order on, or the learning is over
        }
        delays = plan.delays;
    }
    const int nnzA = plan.rowPtr[plan.n];
    std::vector<double> a(nnzA);
    bool changed = false;
    for (size_t t = 0; t < untrusted.size() && t < static_cast<size_t>(PIVOT_DELAY_LEARNING_LANES); t++) {
        const int lane = untrusted[t];                       // a position in the fast engine's lane map
        const std::vector<double> factors = readScalarLaneFactors(lu.scalar, lane, lu.laneFactors, stream);
        if (layout.interleaved) {                            // one value of this scenario every stride values
            CUDA_CHECK(cudaMemcpy2DAsync(a.data(), sizeof(double), vals + lu.scalar.hLaneMap[lane],
                                         static_cast<size_t>(layout.stride) * sizeof(double), sizeof(double), a.size(),
                                         cudaMemcpyDeviceToHost, stream));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(a.data(), vals + static_cast<size_t>(lu.scalar.hLaneMap[lane]) * nnzA,
                                       a.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        changed = learnPivotDelays(delays, *lu.fast, plan.rowPtr.data(), plan.colIdx.data(), factors.data(), a.data())
                  || changed;
    }
    if (changed) {
        publishPivotDelays(plan, *lu.fast, std::move(delays));
    }
}

}  // namespace

void createBatchedLu(BatchedLu& lu, LuPlan& plan, int capacity) {
    lu.capacity = capacity;
    lu.width = 0;
    lu.plan = &plan;
    lu.fast = fastLuPlanOf(plan);
    createScalarLu(lu.scalar, lu.fast->scalar, capacity);
}

void factorizeLanes(BatchedLu& lu, const double* vals, BatchLayout layout, const int* laneMap, int width,
                    cudaStream_t stream) {
    lu.width = width;
    lu.hLaneMap.assign(laneMap, laneMap + width);
    lu.hPivoted.clear();
    lu.hSelected.clear();
    lu.selectedFast = 0;
    lu.fastHasEveryLane = true;
    if (width == 0) {
        factorizeScalarLanes(lu.scalar, vals, layout, laneMap, 0, stream);   // no lane: nothing left to solve
        return;
    }
    gBatchedLuFactorizations++;
    std::shared_ptr<const FastLuPlan> current = fastLuPlanOf(*lu.plan);
    if (current != lu.fast) {                                // the delays grew since this LU's last factorization
        lu.fast = std::move(current);
        createScalarLu(lu.scalar, lu.fast->scalar, lu.capacity);
        lu.pivotingRoute.assign(lu.capacity, 0);             // the new order may keep the lanes the old one flagged
    }
    if (lu.pivotingRoute.size() != static_cast<size_t>(lu.capacity)) {
        lu.pivotingRoute.assign(lu.capacity, 0);
    }
    std::vector<int> fastLanes;                              // the lanes the fast engine factorizes
    fastLanes.reserve(width);
    for (int a = 0; a < width; a++) {
        if (!lu.pivotingRoute[laneMap[a]]) {
            fastLanes.push_back(laneMap[a]);
        }
    }
    const int fastWidth = static_cast<int>(fastLanes.size());
    factorizeScalarLanes(lu.scalar, vals, layout, fastLanes.data(), fastWidth, stream);
    const std::vector<int> trouble = readScalarTrouble(lu.scalar, stream);
    std::vector<int> untrusted;                              // positions in fastLanes
    for (int a = 0, f = 0; a < width; a++) {                 // the pivoting engine's lanes, in the lane map's order
        const int s = laneMap[a];
        if (lu.pivotingRoute[s]) {
            lu.hPivoted.push_back(s);
        } else if (trouble[f++] > 0) {
            lu.hPivoted.push_back(s);
            untrusted.push_back(f - 1);
        }
    }
    for (int s : lu.hPivoted) {
        lu.pivotingRoute[s] = 1;
    }
    if (fastWidth < width) {
        lu.fastHasEveryLane = false;
        gBatchedLuRoutedLanes += width - fastWidth;
        lu.dLaneMap.ensure(static_cast<size_t>(lu.capacity) * sizeof(int));
        CUDA_CHECK(cudaMemcpyAsync(lu.dLaneMap.p, lu.hLaneMap.data(), width * sizeof(int), cudaMemcpyHostToDevice,
                                   stream));
    }
    if (lu.hPivoted.empty()) {
        return;
    }
    learnFromUntrustedLanes(lu, vals, layout, untrusted, stream);
    if (lu.supernodal.capacity == 0) {
        createSupernodalLu(lu.supernodal, lu.plan->supernodal, lu.capacity);
    }
    gBatchedLuPivotedLanes += static_cast<long long>(lu.hPivoted.size());
    factorizeSupernodalLanes(lu.supernodal, vals, layout, lu.hPivoted.data(), static_cast<int>(lu.hPivoted.size()),
                             stream);
}

void forgetLaneRouting(BatchedLu& lu, const std::vector<int>& slots) {
    for (int s : slots) {
        if (s >= 0 && s < static_cast<int>(lu.pivotingRoute.size())) {
            lu.pivotingRoute[s] = 0;
        }
    }
}

void solveLanes(BatchedLu& lu, const double* b, double* x, cudaStream_t stream) {
    solveScalarLanes(lu.scalar, b, x, stream);
    if (!lu.hPivoted.empty()) {
        solveSupernodalLanes(lu.supernodal, b, x, stream);  // after the fast engine: the flagged lanes' x is its own
    }
}

void solveLanesTransposed(BatchedLu& lu, const double* b, double* x, cudaStream_t stream) {
    solveScalarLanesTransposed(lu.scalar, lu.fast->scalar, b, x, stream);
    if (!lu.hPivoted.empty()) {                              // after the fast engine, as in solveLanes
        solveSupernodalLanesTransposed(lu.supernodal, lu.plan->supernodal, b, x, stream);
    }
}

void selectLanes(BatchedLu& lu, const std::vector<char>& scenarios, cudaStream_t stream) {
    lu.hSelected.clear();
    const std::vector<int>& fastLanes = lu.scalar.hLaneMap;
    for (int a = 0; a < static_cast<int>(fastLanes.size()); a++) {
        if (scenarios[fastLanes[a]]) {
            lu.hSelected.push_back(a);
        }
    }
    lu.selectedFast = static_cast<int>(lu.hSelected.size());
    for (size_t p = 0; p < lu.hPivoted.size(); p++) {
        if (scenarios[lu.hPivoted[p]]) {
            lu.hSelected.push_back(static_cast<int>(p));
        }
    }
    if (!lu.hSelected.empty()) {
        lu.dSelected.ensure(lu.hSelected.size() * sizeof(int));
        CUDA_CHECK(cudaMemcpyAsync(lu.dSelected.p, lu.hSelected.data(), lu.hSelected.size() * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
    }
}

void solveSelectedLanes(BatchedLu& lu, const double* b, double* x, cudaStream_t stream) {
    solveScalarSelectedLanes(lu.scalar, lu.dSelected.as<int>(), lu.selectedFast, b, x, stream);
    const int pivoted = static_cast<int>(lu.hSelected.size()) - lu.selectedFast;
    if (pivoted > 0) {                                       // after the fast engine, as in solveLanes
        solveSupernodalSelectedLanes(lu.supernodal, lu.dSelected.as<int>() + lu.selectedFast, pivoted, b, x, stream);
    }
}

const int* deviceLaneMap(BatchedLu& lu) {
    return lu.fastHasEveryLane ? lu.scalar.dLaneMap.as<int>() : lu.dLaneMap.as<int>();
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
