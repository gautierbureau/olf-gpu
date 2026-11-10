// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See scalar_lu.cuh.
#include "scalar_lu.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "../common/check.cuh"
#include "../common/stats.cuh"
#include "lu_split.cuh"

namespace olfgpu {

namespace {

constexpr int LU_BLOCK = 128;
constexpr int TILE = 32;                                     // transposes: 32 entries x 32 lanes per block
constexpr int TILE_ROWS = 8;

unsigned int blocksFor(size_t threads) {
    return static_cast<unsigned int>((threads + LU_BLOCK - 1) / LU_BLOCK);
}

// The lane the k-th lane slot of a launch works on: lane k of the factorization, or the k-th selected lane (a position
// in its lane map) when there is a selection.
__device__ __forceinline__ int laneOf(int k, const int* __restrict__ selected) {
    return selected == nullptr ? k : selected[k];
}

// dst[map[k] * capacity + a] = src[laneMap[a] * stride + k]: scenario-major rows to the lane-interleaved layout,
// through a shared tile so both sides stay contiguous (k along the source, lanes along the destination). The lanes:
// 0 .. width, or the width selected ones.
__global__ void toInterleaved(int count, int stride, int width, int capacity, const int* __restrict__ laneMap,
                              const int* __restrict__ selected, const int* __restrict__ map,
                              const double* __restrict__ src, double* __restrict__ dst) {
    __shared__ double tile[TILE][TILE + 1];
    const int k0 = blockIdx.x * TILE;
    const int c0 = blockIdx.y * TILE;
    for (int r = threadIdx.y; r < TILE; r += TILE_ROWS) {
        const int c = c0 + r;
        const int k = k0 + threadIdx.x;
        if (c < width && k < count) {
            tile[r][threadIdx.x] = src[static_cast<size_t>(laneMap[laneOf(c, selected)]) * stride + k];
        }
    }
    __syncthreads();
    for (int r = threadIdx.y; r < TILE; r += TILE_ROWS) {
        const int k = k0 + r;
        const int c = c0 + threadIdx.x;
        if (c < width && k < count) {
            dst[static_cast<size_t>(map[k]) * capacity + laneOf(c, selected)] = tile[threadIdx.x][r];
        }
    }
}

// The way back: dst[laneMap[a] * stride + k] = src[map[k] * capacity + a].
__global__ void fromInterleaved(int count, int stride, int width, int capacity, const int* __restrict__ laneMap,
                                const int* __restrict__ selected, const int* __restrict__ map,
                                const double* __restrict__ src, double* __restrict__ dst) {
    __shared__ double tile[TILE][TILE + 1];
    const int k0 = blockIdx.x * TILE;
    const int c0 = blockIdx.y * TILE;
    for (int r = threadIdx.y; r < TILE; r += TILE_ROWS) {
        const int k = k0 + r;
        const int c = c0 + threadIdx.x;
        if (c < width && k < count) {
            tile[r][threadIdx.x] = src[static_cast<size_t>(map[k]) * capacity + laneOf(c, selected)];
        }
    }
    __syncthreads();
    for (int r = threadIdx.y; r < TILE; r += TILE_ROWS) {
        const int c = c0 + r;
        const int k = k0 + threadIdx.x;
        if (c < width && k < count) {
            dst[static_cast<size_t>(laneMap[laneOf(c, selected)]) * stride + k] = tile[threadIdx.x][r];
        }
    }
}

// out[e] = lu[e * capacity + lane]: one lane's factors out of the interleaved layout.
__global__ void gatherLane(int nnzLu, int capacity, int lane, const double* __restrict__ lu, double* __restrict__ out) {
    const size_t e = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e < static_cast<size_t>(nnzLu)) {
        out[e] = lu[e * capacity + lane];
    }
}

__global__ void zeroFill(int nFill, int width, int capacity, const int* __restrict__ fillEntry, double* __restrict__ lu) {
    const size_t g = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= static_cast<size_t>(nFill) * width) {
        return;
    }
    const int f = static_cast<int>(g / width);
    const int a = static_cast<int>(g - static_cast<size_t>(f) * width);
    lu[static_cast<size_t>(fillEntry[f]) * capacity + a] = 0.0;
}

// An entry's value once its contributions are in acc: an L entry divides by its pivot - an outsized multiplier is
// trouble - and a pivot is checked.
__device__ __forceinline__ void finishEntry(int e, int a, int capacity, const int* __restrict__ entryDivisor, double acc,
                                            double* __restrict__ lu, int* __restrict__ trouble) {
    const int d = entryDivisor[e];
    if (d >= 0) {
        acc /= lu[static_cast<size_t>(d) * capacity + a];
        if (!(fabs(acc) <= LU_MULTIPLIER_LIMIT)) {
            atomicAdd(trouble + a, 1);
        }
    } else if (d == SCALAR_ITEM_PIVOT && !(fabs(acc) >= LU_PIVOT_EPSILON)) {
        if (!isnan(acc)) {
            acc = acc < 0.0 ? -LU_PIVOT_EPSILON : LU_PIVOT_EPSILON;
        }
        atomicAdd(trouble + a, 1);
    }
    lu[static_cast<size_t>(e) * capacity + a] = acc;
}

// An entry's value before its contributions. Values interleaved by scenario (common/batch_layout.cuh) are read in
// place - src[k * stride + scenario] for the A entry k, 0 for a fill entry - every entry being read once, at the step
// that finishes it; any other layout was copied into lu first.
__device__ __forceinline__ double initialEntry(int e, int a, int capacity, const double* __restrict__ src, int stride,
                                              const int* __restrict__ luSource, const int* __restrict__ laneMap,
                                              const double* __restrict__ lu) {
    if (src == nullptr) {
        return lu[static_cast<size_t>(e) * capacity + a];
    }
    const int k = luSource[e];
    return k >= 0 ? src[static_cast<size_t>(k) * stride + laneMap[a]] : 0.0;
}

// One factorization step: every (item, lane) gathers its entry's contributions in pivot order, then finishes it.
__global__ void factorStep(int nItems, int width, int capacity, const int* __restrict__ stepEntry,
                           const int* __restrict__ entryDivisor, const int* __restrict__ contributionStart,
                           const int* __restrict__ contributionL, const int* __restrict__ contributionU,
                           const double* __restrict__ src, int stride, const int* __restrict__ luSource,
                           const int* __restrict__ laneMap, double* __restrict__ lu, int* __restrict__ trouble) {
    const size_t g = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= static_cast<size_t>(nItems) * width) {
        return;
    }
    const int t = static_cast<int>(g / width);
    const int a = static_cast<int>(g - static_cast<size_t>(t) * width);
    const int e = stepEntry[t];
    double acc = initialEntry(e, a, capacity, src, stride, luSource, laneMap, lu);
    const int cEnd = contributionStart[e + 1];
    for (int c = contributionStart[e]; c < cEnd; c++) {
        acc = fma(-lu[static_cast<size_t>(contributionL[c]) * capacity + a],
                  lu[static_cast<size_t>(contributionU[c]) * capacity + a], acc);
    }
    finishEntry(e, a, capacity, entryDivisor, acc, lu, trouble);
}

// factorStep with each entry's contributions split over blockDim.y warps (splitSlices): warp k of a block sums the
// contributions k, k + K, ... of one entry for 32 lanes, then the first warp adds the K partial sums in warp order -
// a fixed order, the same on every run - and finishes the entry.
__global__ void factorStepSplit(int nItems, int width, int capacity, const int* __restrict__ stepEntry,
                                const int* __restrict__ entryDivisor, const int* __restrict__ contributionStart,
                                const int* __restrict__ contributionL, const int* __restrict__ contributionU,
                                const double* __restrict__ src, int stride, const int* __restrict__ luSource,
                                const int* __restrict__ laneMap, double* __restrict__ lu, int* __restrict__ trouble) {
    extern __shared__ double partial[];                      // [slice][lane of the block]
    const int tiles = (width + SPLIT_LANES - 1) / SPLIT_LANES;
    const int t = blockIdx.x / tiles;
    const int a = (blockIdx.x - t * tiles) * SPLIT_LANES + threadIdx.x;
    const int slices = blockDim.y;
    const bool active = a < width;
    const int e = stepEntry[t];
    double sum = 0.0;
    if (active) {
        const int cEnd = contributionStart[e + 1];
        for (int c = contributionStart[e] + threadIdx.y; c < cEnd; c += slices) {
            sum = fma(-lu[static_cast<size_t>(contributionL[c]) * capacity + a],
                      lu[static_cast<size_t>(contributionU[c]) * capacity + a], sum);
        }
    }
    partial[threadIdx.y * SPLIT_LANES + threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.y == 0 && active) {
        double acc = initialEntry(e, a, capacity, src, stride, luSource, laneMap, lu);
        for (int k = 0; k < slices; k++) {
            acc += partial[k * SPLIT_LANES + threadIdx.x];
        }
        finishEntry(e, a, capacity, entryDivisor, acc, lu, trouble);
    }
}

// One forward level, L unit lower: z[i] -= sum_k L[i][k] z[k].
__global__ void forwardLevel(int nRows, int width, int capacity, const int* __restrict__ selected,
                             const int* __restrict__ rows, const int* __restrict__ luRowPtr,
                             const int* __restrict__ luCol, const int* __restrict__ luDiag,
                             const double* __restrict__ lu, double* __restrict__ z) {
    const size_t g = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= static_cast<size_t>(nRows) * width) {
        return;
    }
    const int t = static_cast<int>(g / width);
    const int a = laneOf(static_cast<int>(g - static_cast<size_t>(t) * width), selected);
    const int i = rows[t];
    double y = z[static_cast<size_t>(i) * capacity + a];
    const int qEnd = luDiag[i];
    for (int q = luRowPtr[i]; q < qEnd; q++) {
        y = fma(-lu[static_cast<size_t>(q) * capacity + a], z[static_cast<size_t>(luCol[q]) * capacity + a], y);
    }
    z[static_cast<size_t>(i) * capacity + a] = y;
}

// One backward level: z[i] = (z[i] - sum_j U[i][j] z[j]) / U[i][i].
__global__ void backwardLevel(int nRows, int width, int capacity, const int* __restrict__ selected,
                              const int* __restrict__ rows, const int* __restrict__ luRowPtr,
                              const int* __restrict__ luCol, const int* __restrict__ luDiag,
                              const double* __restrict__ lu, double* __restrict__ z) {
    const size_t g = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= static_cast<size_t>(nRows) * width) {
        return;
    }
    const int t = static_cast<int>(g / width);
    const int a = laneOf(static_cast<int>(g - static_cast<size_t>(t) * width), selected);
    const int i = rows[t];
    double x = z[static_cast<size_t>(i) * capacity + a];
    const int qEnd = luRowPtr[i + 1];
    for (int q = luDiag[i] + 1; q < qEnd; q++) {
        x = fma(-lu[static_cast<size_t>(q) * capacity + a], z[static_cast<size_t>(luCol[q]) * capacity + a], x);
    }
    z[static_cast<size_t>(i) * capacity + a] = x / lu[static_cast<size_t>(luDiag[i]) * capacity + a];
}

// forwardLevel (backward false) or backwardLevel (backward true) with each row's terms split over blockDim.y warps, as
// factorStepSplit splits an entry's contributions: warp k sums the terms k, k + K, ... of one row for 32 lanes, the
// first warp adds the K partial sums in warp order and finishes the row.
__global__ void levelSplit(bool backward, int nRows, int width, int capacity, const int* __restrict__ selected,
                           const int* __restrict__ rows, const int* __restrict__ luRowPtr,
                           const int* __restrict__ luCol, const int* __restrict__ luDiag,
                           const double* __restrict__ lu, double* __restrict__ z) {
    extern __shared__ double partial[];                      // [slice][lane of the block]
    const int tiles = (width + SPLIT_LANES - 1) / SPLIT_LANES;
    const int t = blockIdx.x / tiles;
    const int slot = (blockIdx.x - t * tiles) * SPLIT_LANES + threadIdx.x;
    const int slices = blockDim.y;
    const bool active = slot < width;
    const int a = active ? laneOf(slot, selected) : 0;
    const int i = rows[t];
    double sum = 0.0;
    if (active) {
        const int qEnd = backward ? luRowPtr[i + 1] : luDiag[i];
        for (int q = (backward ? luDiag[i] + 1 : luRowPtr[i]) + threadIdx.y; q < qEnd; q += slices) {
            sum = fma(-lu[static_cast<size_t>(q) * capacity + a], z[static_cast<size_t>(luCol[q]) * capacity + a], sum);
        }
    }
    partial[threadIdx.y * SPLIT_LANES + threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.y == 0 && active) {
        double y = z[static_cast<size_t>(i) * capacity + a];
        for (int k = 0; k < slices; k++) {
            y += partial[k * SPLIT_LANES + threadIdx.x];
        }
        z[static_cast<size_t>(i) * capacity + a] = backward ? y / lu[static_cast<size_t>(luDiag[i]) * capacity + a] : y;
    }
}

void launchLevel(ScalarLu& lu, bool backward, int level, const int* selected, int width, cudaStream_t stream) {
    const std::vector<int>& start = backward ? lu.backwardLevelStart : lu.forwardLevelStart;
    const int rows = start[level + 1] - start[level];
    const int* levelRows = (backward ? lu.dBackwardRow : lu.dForwardRow).as<int>() + start[level];
    const int slices = splitSlices(rows, width, (backward ? lu.backwardLevelLongest : lu.forwardLevelLongest)[level],
                                   lu.residentThreads);
    if (slices > 1) {
        gBatchedLuSplitLaunches++;
        const unsigned int blocks = static_cast<unsigned int>(rows) * ((width + SPLIT_LANES - 1) / SPLIT_LANES);
        levelSplit<<<blocks, dim3(SPLIT_LANES, slices), slices * SPLIT_LANES * sizeof(double), stream>>>(backward, rows,
                width, lu.capacity, selected, levelRows, lu.dLuRowPtr.as<int>(), lu.dLuCol.as<int>(),
                lu.dLuDiag.as<int>(), lu.dLu.as<double>(), lu.dWork.as<double>());
    } else if (backward) {
        backwardLevel<<<blocksFor(static_cast<size_t>(rows) * width), LU_BLOCK, 0, stream>>>(rows, width, lu.capacity,
                selected, levelRows, lu.dLuRowPtr.as<int>(), lu.dLuCol.as<int>(), lu.dLuDiag.as<int>(),
                lu.dLu.as<double>(), lu.dWork.as<double>());
    } else {
        forwardLevel<<<blocksFor(static_cast<size_t>(rows) * width), LU_BLOCK, 0, stream>>>(rows, width, lu.capacity,
                selected, levelRows, lu.dLuRowPtr.as<int>(), lu.dLuCol.as<int>(), lu.dLuDiag.as<int>(),
                lu.dLu.as<double>(), lu.dWork.as<double>());
    }
}

// The longest term run of each level: forward, the L part of its rows; backward, the U part right of the diagonal.
std::vector<int> levelLongest(const ScalarLuPlan& plan, bool backward) {
    const std::vector<int>& start = backward ? plan.backwardLevelStart : plan.forwardLevelStart;
    const std::vector<int>& rows = backward ? plan.backwardRow : plan.forwardRow;
    std::vector<int> longest(start.empty() ? 0 : start.size() - 1, 0);
    for (size_t l = 0; l < longest.size(); l++) {
        for (int k = start[l]; k < start[l + 1]; k++) {
            const int i = rows[k];
            const int terms = backward ? plan.luRowPtr[i + 1] - plan.luDiag[i] - 1 : plan.luDiag[i] - plan.luRowPtr[i];
            longest[l] = std::max(longest[l], terms);
        }
    }
    return longest;
}

template <typename T>
void uploadVector(DeviceBuf& buf, const std::vector<T>& v) {
    if (!v.empty()) {
        buf.upload(v.data(), v.size() * sizeof(T));
    }
}

// The triangular solves on width lanes: every lane of the last factorization (selected null) or the selected ones.
void solveScalar(ScalarLu& lu, const int* selected, int width, const double* b, double* x, cudaStream_t stream) {
    const dim3 tileBlock(TILE, TILE_ROWS);
    const dim3 vectorGrid((lu.n + TILE - 1) / TILE, (width + TILE - 1) / TILE);
    toInterleaved<<<vectorGrid, tileBlock, 0, stream>>>(lu.n, lu.n, width, lu.capacity, lu.dLaneMap.as<int>(), selected,
            lu.dRowPosition.as<int>(), b, lu.dWork.as<double>());
    for (int l = 0; l + 1 < static_cast<int>(lu.forwardLevelStart.size()); l++) {
        launchLevel(lu, false, l, selected, width, stream);
    }
    for (int l = 0; l + 1 < static_cast<int>(lu.backwardLevelStart.size()); l++) {
        launchLevel(lu, true, l, selected, width, stream);
    }
    fromInterleaved<<<vectorGrid, tileBlock, 0, stream>>>(lu.n, lu.n, width, lu.capacity, lu.dLaneMap.as<int>(),
            selected, lu.dColPosition.as<int>(), lu.dWork.as<double>(), x);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void createScalarLu(ScalarLu& lu, const ScalarLuPlan& plan, int capacity) {
    if (capacity <= 0) {
        throw std::runtime_error("batched LU capacity must be positive: " + std::to_string(capacity));
    }
    lu.n = plan.n;
    lu.nnzA = plan.nnzA;
    lu.nnzLu = plan.nnzLu;
    lu.capacity = capacity;
    lu.width = 0;
    lu.stepStart = plan.stepStart;
    lu.forwardLevelStart = plan.forwardLevelStart;
    lu.backwardLevelStart = plan.backwardLevelStart;
    lu.forwardLevelLongest = levelLongest(plan, false);
    lu.backwardLevelLongest = levelLongest(plan, true);
    lu.stepLongest.assign(plan.stepStart.empty() ? 0 : plan.stepStart.size() - 1, 0);
    for (size_t s = 0; s < lu.stepLongest.size(); s++) {
        for (int k = plan.stepStart[s]; k < plan.stepStart[s + 1]; k++) {
            const int e = plan.stepEntry[k];
            lu.stepLongest[s] = std::max(lu.stepLongest[s], plan.contributionStart[e + 1] - plan.contributionStart[e]);
        }
    }
    lu.residentThreads = residentDeviceThreads();
    std::vector<int> entryOfA(plan.nnzA, -1);
    std::vector<int> fillEntry;
    for (int e = 0; e < plan.nnzLu; e++) {
        if (plan.luSource[e] >= 0) {
            entryOfA[plan.luSource[e]] = e;
        } else {
            fillEntry.push_back(e);
        }
    }
    lu.nFill = static_cast<int>(fillEntry.size());
    std::vector<int> rowPosition(plan.n);
    std::vector<int> colPosition(plan.n);
    for (int i = 0; i < plan.n; i++) {
        rowPosition[plan.rowOrder[i]] = i;
        colPosition[plan.colOrder[i]] = i;
    }
    uploadVector(lu.dEntryOfA, entryOfA);
    uploadVector(lu.dFillEntry, fillEntry);
    uploadVector(lu.dLuSource, plan.luSource);
    uploadVector(lu.dStepEntry, plan.stepEntry);
    uploadVector(lu.dEntryDivisor, plan.entryDivisor);
    uploadVector(lu.dContributionStart, plan.contributionStart);
    uploadVector(lu.dContributionL, plan.contributionL);
    uploadVector(lu.dContributionU, plan.contributionU);
    uploadVector(lu.dLuRowPtr, plan.luRowPtr);
    uploadVector(lu.dLuCol, plan.luCol);
    uploadVector(lu.dLuDiag, plan.luDiag);
    uploadVector(lu.dForwardRow, plan.forwardRow);
    uploadVector(lu.dBackwardRow, plan.backwardRow);
    uploadVector(lu.dRowPosition, rowPosition);
    uploadVector(lu.dColPosition, colPosition);
    lu.dLu.alloc(static_cast<size_t>(plan.nnzLu) * capacity * sizeof(double));
    lu.dWork.alloc(static_cast<size_t>(plan.n) * capacity * sizeof(double));
    lu.dLaneMap.alloc(static_cast<size_t>(capacity) * sizeof(int));
    lu.dTrouble.alloc(static_cast<size_t>(capacity) * sizeof(int));
}

void factorizeScalarLanes(ScalarLu& lu, const double* vals, BatchLayout layout, const int* laneMap, int width,
                          cudaStream_t stream) {
    if (width < 0 || width > lu.capacity) {
        throw std::runtime_error("batched LU width " + std::to_string(width) + " outside 0.."
                                 + std::to_string(lu.capacity));
    }
    lu.width = width;
    lu.hLaneMap.assign(laneMap, laneMap + width);
    if (width == 0) {
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(lu.dLaneMap.p, laneMap, width * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(lu.dTrouble.p, 0, width * sizeof(int), stream));
    // Values interleaved by scenario are read in place by the steps (initialEntry); scenario-major ones are copied into
    // the lane-interleaved layout first, the fill entries zeroed.
    const double* src = layout.interleaved ? vals : nullptr;
    if (!layout.interleaved) {
        const dim3 tileBlock(TILE, TILE_ROWS);
        const dim3 valuesGrid((lu.nnzA + TILE - 1) / TILE, (width + TILE - 1) / TILE);
        toInterleaved<<<valuesGrid, tileBlock, 0, stream>>>(lu.nnzA, lu.nnzA, width, lu.capacity, lu.dLaneMap.as<int>(),
                nullptr, lu.dEntryOfA.as<int>(), vals, lu.dLu.as<double>());
        if (lu.nFill > 0) {
            zeroFill<<<blocksFor(static_cast<size_t>(lu.nFill) * width), LU_BLOCK, 0, stream>>>(lu.nFill, width,
                    lu.capacity, lu.dFillEntry.as<int>(), lu.dLu.as<double>());
        }
    }
    for (int s = 0; s + 1 < static_cast<int>(lu.stepStart.size()); s++) {
        const int items = lu.stepStart[s + 1] - lu.stepStart[s];
        if (items == 0) {
            continue;
        }
        const int slices = splitSlices(items, width, lu.stepLongest[s], lu.residentThreads);
        if (slices > 1) {
            gBatchedLuSplitLaunches++;
            const unsigned int blocks = static_cast<unsigned int>(items) * ((width + SPLIT_LANES - 1) / SPLIT_LANES);
            factorStepSplit<<<blocks, dim3(SPLIT_LANES, slices), slices * SPLIT_LANES * sizeof(double), stream>>>(items,
                    width, lu.capacity, lu.dStepEntry.as<int>() + lu.stepStart[s], lu.dEntryDivisor.as<int>(),
                    lu.dContributionStart.as<int>(), lu.dContributionL.as<int>(), lu.dContributionU.as<int>(),
                    src, layout.stride, lu.dLuSource.as<int>(), lu.dLaneMap.as<int>(), lu.dLu.as<double>(),
                    lu.dTrouble.as<int>());
        } else {
            factorStep<<<blocksFor(static_cast<size_t>(items) * width), LU_BLOCK, 0, stream>>>(items, width,
                    lu.capacity, lu.dStepEntry.as<int>() + lu.stepStart[s], lu.dEntryDivisor.as<int>(),
                    lu.dContributionStart.as<int>(), lu.dContributionL.as<int>(), lu.dContributionU.as<int>(),
                    src, layout.stride, lu.dLuSource.as<int>(), lu.dLaneMap.as<int>(), lu.dLu.as<double>(),
                    lu.dTrouble.as<int>());
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

void solveScalarLanes(ScalarLu& lu, const double* b, double* x, cudaStream_t stream) {
    if (lu.width > 0) {
        solveScalar(lu, nullptr, lu.width, b, x, stream);
    }
}

void solveScalarSelectedLanes(ScalarLu& lu, const int* selected, int count, const double* b, double* x,
                              cudaStream_t stream) {
    if (count > 0) {
        solveScalar(lu, selected, count, b, x, stream);
    }
}

std::vector<int> readScalarTrouble(ScalarLu& lu, cudaStream_t stream) {
    std::vector<int> clamps(lu.width, 0);
    if (lu.width > 0) {
        CUDA_CHECK(cudaMemcpyAsync(clamps.data(), lu.dTrouble.p, lu.width * sizeof(int), cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    return clamps;
}

std::vector<double> readScalarLaneFactors(ScalarLu& lu, int lane, DeviceBuf& scratch, cudaStream_t stream) {
    std::vector<double> factors(lu.nnzLu);
    scratch.alloc(factors.size() * sizeof(double));
    gatherLane<<<blocksFor(factors.size()), LU_BLOCK, 0, stream>>>(lu.nnzLu, lu.capacity, lane, lu.dLu.as<double>(),
                                                                   scratch.as<double>());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(factors.data(), scratch.p, factors.size() * sizeof(double), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return factors;
}

}  // namespace olfgpu
