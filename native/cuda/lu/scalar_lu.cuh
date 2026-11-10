// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The fast engine of our batched LU on the device: the plan of scalar_lu_plan.cuh uploaded once, the factors of every
// lane of a bank, and the factorization and triangular solves on any subset of the bank's lanes.
//
// Values are interleaved by lane, lu[entry * capacity + lane], so the 32 threads of a warp - 32 lanes of one LU entry
// - read and write contiguous memory. A factorization runs on the lanes of a lane map only (lane a of the call holds
// scenario laneMap[a]): a scenario that no longer iterates costs nothing, without an identity system in its place.
// No pivoting: a pivot below LU_PIVOT_EPSILON in magnitude is replaced by +-LU_PIVOT_EPSILON, and that or a multiplier
// beyond LU_MULTIPLIER_LIMIT (growth from a pivot too small for its column) counts as trouble on the lane.
#ifndef OLF_GPU_LU_SCALAR_LU_CUH
#define OLF_GPU_LU_SCALAR_LU_CUH

#include <cuda_runtime.h>

#include <vector>

#include "../common/batch_layout.cuh"
#include "../common/device_buf.cuh"
#include "lu_pivot.cuh"
#include "scalar_lu_plan.cuh"

namespace olfgpu {

struct ScalarLu {
    int n = 0, nnzA = 0, nnzLu = 0;
    int capacity = 0;                                        // lanes the factors hold
    int width = 0;                                           // lanes of the last factorization
    std::vector<int> stepStart, forwardLevelStart, backwardLevelStart;
    std::vector<int> stepLongest, forwardLevelLongest, backwardLevelLongest;   // per step / level: its longest term run
    size_t residentThreads = 0;                              // threads the device holds resident: what a split may use
    int nFill = 0;
    DeviceBuf dEntryOfA, dFillEntry;                         // A entry -> LU entry; the LU entries without an A entry
    DeviceBuf dLuSource;                                     // LU entry -> A entry, -1 for a fill entry
    DeviceBuf dStepEntry, dEntryDivisor, dContributionStart, dContributionL, dContributionU;
    DeviceBuf dLuRowPtr, dLuCol, dLuDiag, dForwardRow, dBackwardRow;
    DeviceBuf dRowPosition, dColPosition;                   // A row -> B row, A column -> B column
    DeviceBuf dLu;                                           // nnzLu * capacity
    DeviceBuf dWork;                                         // n * capacity: the permuted right-hand sides
    DeviceBuf dLaneMap;                                      // capacity: the last factorization's lane -> scenario
    DeviceBuf dTrouble;                                      // capacity: per lane, clamped pivots and outsized multipliers
    std::vector<int> hLaneMap;
};

// Upload the plan and size the factors for capacity lanes.
void createScalarLu(ScalarLu& lu, const ScalarLuPlan& plan, int capacity);

// Factorize scenarios laneMap[0 .. width) (host array, each < capacity) of vals, scenario s's CSR values laid out by
// layout (common/batch_layout.cuh). Asynchronous on the stream; the clamp counts restart at zero.
void factorizeScalarLanes(ScalarLu& lu, const double* vals, BatchLayout layout, const int* laneMap, int width,
                    cudaStream_t stream);

// Solve with the last factorization's lanes: x[s * n ..] = A_s^-1 b[s * n ..] for every s of its lane map. Scenarios
// outside the lane map keep their x.
void solveScalarLanes(ScalarLu& lu, const double* b, double* x, cudaStream_t stream);

// solveScalarLanes on count of the last factorization's lanes only, selected[0 .. count) (device array of positions in
// its lane map): each of them takes the operations solveScalarLanes gives it; every other scenario keeps its x.
void solveScalarSelectedLanes(ScalarLu& lu, const int* selected, int count, const double* b, double* x,
                              cudaStream_t stream);

// The last factorization's trouble counts, per lane of its lane map (synchronizes the stream): a lane with any needs
// the pivoting engine.
std::vector<int> readScalarTrouble(ScalarLu& lu, cudaStream_t stream);

// One lane of the last factorization's factors: its nnzLu values, in the plan's LU entry order (synchronizes the
// stream; scratch holds them on the device on the way).
std::vector<double> readScalarLaneFactors(ScalarLu& lu, int lane, DeviceBuf& scratch, cudaStream_t stream);

// One scenario's count values out of values interleaved by scenario: values[e * stride + scenario], gathered on the
// device into scratch and copied in one piece (synchronizes the stream).
std::vector<double> readInterleavedScenario(const double* values, int count, int stride, int scenario, DeviceBuf& scratch,
                                            cudaStream_t stream);

}  // namespace olfgpu

#endif  // OLF_GPU_LU_SCALAR_LU_CUH
