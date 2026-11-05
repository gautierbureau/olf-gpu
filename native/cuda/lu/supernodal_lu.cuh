// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The pivoting engine of our batched LU on the device: the plan of supernodal_lu_plan.cuh uploaded once, the factors
// of the lanes it is given, and the factorization and triangular solves on any subset of the bank's lanes.
//
// Values are interleaved by lane, lu[entry * capacity + lane], so the 32 threads of a warp - 32 lanes of one stored
// entry - read and write contiguous memory. A factorization runs on the lanes of a lane map only (lane a of the call
// holds scenario laneMap[a]): a scenario that no longer iterates costs nothing, without an identity system in its
// place. Each lane pivots inside each supernode on its own (complete pivoting on the diagonal block, the order kept
// per lane); a pivot still below LU_PIVOT_EPSILON in magnitude is replaced by +-LU_PIVOT_EPSILON (cuDSS's default
// absolute epsilon for double precision) and counted on its lane.
#ifndef OLF_GPU_LU_SUPERNODAL_LU_CUH
#define OLF_GPU_LU_SUPERNODAL_LU_CUH

#include <cuda_runtime.h>

#include <vector>

#include "../common/batch_layout.cuh"
#include "../common/device_buf.cuh"
#include "lu_pivot.cuh"
#include "supernodal_lu_plan.cuh"

namespace olfgpu {

struct SupernodalLu {
    int n = 0, nnzA = 0, nStored = 0, nSupernodes = 0;
    int capacity = 0;                                        // lanes the factors hold
    int width = 0;                                           // lanes of the last factorization
    std::vector<int> levelStart, levelEntryStart, levelPanelStart;
    int nFill = 0;
    DeviceBuf dStoredOfA, dFillEntry;                        // A entry -> stored entry; the stored entries without one
    DeviceBuf dSupernodeStart, dRowStart, dRows, dDiagonalBase, dLowerBase, dUpperBase, dLevelSupernode;
    DeviceBuf dRunStart, dRunLower, dRunUpper, dRunLength, dLevelEntry, dPanelSupernode, dPanelRow;
    DeviceBuf dForwardRunStart, dForwardRunLower, dForwardRunY, dForwardRunLength;
    DeviceBuf dRowPosition, dColPosition;                   // A row -> B row, A column -> B column
    DeviceBuf dLu;                                           // nStored * capacity
    DeviceBuf dRowPermutation, dColumnPermutation;          // n * capacity bytes: per lane, each supernode's slot order
    DeviceBuf dWork;                                         // n * capacity: the permuted right-hand sides
    DeviceBuf dLaneMap;                                      // capacity: the last factorization's lane -> scenario
    DeviceBuf dPivotClamps;                                  // capacity: clamped pivots of each lane's factors
    std::vector<int> hLaneMap;
};

// Upload the plan and size the factors for capacity lanes.
void createSupernodalLu(SupernodalLu& lu, const SupernodalLuPlan& plan, int capacity);

// Factorize scenarios laneMap[0 .. width) (host array, each < capacity) of vals, scenario s's CSR values laid out by
// layout (common/batch_layout.cuh). Asynchronous on the stream; the clamp counts restart at zero.
void factorizeSupernodalLanes(SupernodalLu& lu, const double* vals, BatchLayout layout, const int* laneMap, int width,
                    cudaStream_t stream);

// Solve with the last factorization's lanes: x[s * n ..] = A_s^-1 b[s * n ..] for every s of its lane map. Scenarios
// outside the lane map keep their x.
void solveSupernodalLanes(SupernodalLu& lu, const double* b, double* x, cudaStream_t stream);

// The last factorization's clamp counts, per lane of its lane map (synchronizes the stream).
std::vector<int> readSupernodalClamps(SupernodalLu& lu, cudaStream_t stream);

}  // namespace olfgpu

#endif  // OLF_GPU_LU_SUPERNODAL_LU_CUH
