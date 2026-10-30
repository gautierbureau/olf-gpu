// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See scalar_lu.cuh.
#include "scalar_lu.cuh"

#include <cmath>
#include <stdexcept>
#include <string>

#include "../common/check.cuh"

namespace olfgpu {

namespace {

constexpr int LU_BLOCK = 128;
constexpr int TILE = 32;                                     // transposes: 32 entries x 32 lanes per block
constexpr int TILE_ROWS = 8;

unsigned int blocksFor(size_t threads) {
    return static_cast<unsigned int>((threads + LU_BLOCK - 1) / LU_BLOCK);
}

// dst[map[k] * capacity + a] = src[laneMap[a] * stride + k]: scenario-major rows to the lane-interleaved layout,
// through a shared tile so both sides stay contiguous (k along the source, lanes along the destination).
__global__ void toInterleaved(int count, int stride, int width, int capacity, const int* __restrict__ laneMap,
                              const int* __restrict__ map, const double* __restrict__ src, double* __restrict__ dst) {
    __shared__ double tile[TILE][TILE + 1];
    const int k0 = blockIdx.x * TILE;
    const int a0 = blockIdx.y * TILE;
    for (int r = threadIdx.y; r < TILE; r += TILE_ROWS) {
        const int a = a0 + r;
        const int k = k0 + threadIdx.x;
        if (a < width && k < count) {
            tile[r][threadIdx.x] = src[static_cast<size_t>(laneMap[a]) * stride + k];
        }
    }
    __syncthreads();
    for (int r = threadIdx.y; r < TILE; r += TILE_ROWS) {
        const int k = k0 + r;
        const int a = a0 + threadIdx.x;
        if (a < width && k < count) {
            dst[static_cast<size_t>(map[k]) * capacity + a] = tile[threadIdx.x][r];
        }
    }
}

// The way back: dst[laneMap[a] * stride + k] = src[map[k] * capacity + a].
__global__ void fromInterleaved(int count, int stride, int width, int capacity, const int* __restrict__ laneMap,
                                const int* __restrict__ map, const double* __restrict__ src, double* __restrict__ dst) {
    __shared__ double tile[TILE][TILE + 1];
    const int k0 = blockIdx.x * TILE;
    const int a0 = blockIdx.y * TILE;
    for (int r = threadIdx.y; r < TILE; r += TILE_ROWS) {
        const int k = k0 + r;
        const int a = a0 + threadIdx.x;
        if (a < width && k < count) {
            tile[r][threadIdx.x] = src[static_cast<size_t>(map[k]) * capacity + a];
        }
    }
    __syncthreads();
    for (int r = threadIdx.y; r < TILE; r += TILE_ROWS) {
        const int a = a0 + r;
        const int k = k0 + threadIdx.x;
        if (a < width && k < count) {
            dst[static_cast<size_t>(laneMap[a]) * stride + k] = tile[threadIdx.x][r];
        }
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

// One factorization step: every (item, lane) gathers its entry's contributions in pivot order, then an L entry
// divides by its pivot - an outsized multiplier is trouble - and a pivot is checked.
__global__ void factorStep(int nItems, int width, int capacity, const int* __restrict__ stepEntry,
                           const int* __restrict__ entryDivisor, const int* __restrict__ contributionStart,
                           const int* __restrict__ contributionL, const int* __restrict__ contributionU,
                           double* __restrict__ lu, int* __restrict__ trouble) {
    const size_t g = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= static_cast<size_t>(nItems) * width) {
        return;
    }
    const int t = static_cast<int>(g / width);
    const int a = static_cast<int>(g - static_cast<size_t>(t) * width);
    const int e = stepEntry[t];
    double acc = lu[static_cast<size_t>(e) * capacity + a];
    const int cEnd = contributionStart[e + 1];
    for (int c = contributionStart[e]; c < cEnd; c++) {
        acc = fma(-lu[static_cast<size_t>(contributionL[c]) * capacity + a],
                  lu[static_cast<size_t>(contributionU[c]) * capacity + a], acc);
    }
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

// One forward level, L unit lower: z[i] -= sum_k L[i][k] z[k].
__global__ void forwardLevel(int nRows, int width, int capacity, const int* __restrict__ rows,
                             const int* __restrict__ luRowPtr, const int* __restrict__ luCol,
                             const int* __restrict__ luDiag, const double* __restrict__ lu, double* __restrict__ z) {
    const size_t g = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= static_cast<size_t>(nRows) * width) {
        return;
    }
    const int t = static_cast<int>(g / width);
    const int a = static_cast<int>(g - static_cast<size_t>(t) * width);
    const int i = rows[t];
    double y = z[static_cast<size_t>(i) * capacity + a];
    const int qEnd = luDiag[i];
    for (int q = luRowPtr[i]; q < qEnd; q++) {
        y = fma(-lu[static_cast<size_t>(q) * capacity + a], z[static_cast<size_t>(luCol[q]) * capacity + a], y);
    }
    z[static_cast<size_t>(i) * capacity + a] = y;
}

// One backward level: z[i] = (z[i] - sum_j U[i][j] z[j]) / U[i][i].
__global__ void backwardLevel(int nRows, int width, int capacity, const int* __restrict__ rows,
                              const int* __restrict__ luRowPtr, const int* __restrict__ luCol,
                              const int* __restrict__ luDiag, const double* __restrict__ lu, double* __restrict__ z) {
    const size_t g = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= static_cast<size_t>(nRows) * width) {
        return;
    }
    const int t = static_cast<int>(g / width);
    const int a = static_cast<int>(g - static_cast<size_t>(t) * width);
    const int i = rows[t];
    double x = z[static_cast<size_t>(i) * capacity + a];
    const int qEnd = luRowPtr[i + 1];
    for (int q = luDiag[i] + 1; q < qEnd; q++) {
        x = fma(-lu[static_cast<size_t>(q) * capacity + a], z[static_cast<size_t>(luCol[q]) * capacity + a], x);
    }
    z[static_cast<size_t>(i) * capacity + a] = x / lu[static_cast<size_t>(luDiag[i]) * capacity + a];
}

template <typename T>
void uploadVector(DeviceBuf& buf, const std::vector<T>& v) {
    if (!v.empty()) {
        buf.upload(v.data(), v.size() * sizeof(T));
    }
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

void factorizeScalarLanes(ScalarLu& lu, const double* vals, const int* laneMap, int width, cudaStream_t stream) {
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
    const dim3 tileBlock(TILE, TILE_ROWS);
    const dim3 valuesGrid((lu.nnzA + TILE - 1) / TILE, (width + TILE - 1) / TILE);
    toInterleaved<<<valuesGrid, tileBlock, 0, stream>>>(lu.nnzA, lu.nnzA, width, lu.capacity, lu.dLaneMap.as<int>(),
            lu.dEntryOfA.as<int>(), vals, lu.dLu.as<double>());
    if (lu.nFill > 0) {
        zeroFill<<<blocksFor(static_cast<size_t>(lu.nFill) * width), LU_BLOCK, 0, stream>>>(lu.nFill, width,
                lu.capacity, lu.dFillEntry.as<int>(), lu.dLu.as<double>());
    }
    for (int s = 0; s + 1 < static_cast<int>(lu.stepStart.size()); s++) {
        const int items = lu.stepStart[s + 1] - lu.stepStart[s];
        if (items == 0) {
            continue;
        }
        factorStep<<<blocksFor(static_cast<size_t>(items) * width), LU_BLOCK, 0, stream>>>(items, width, lu.capacity,
                lu.dStepEntry.as<int>() + lu.stepStart[s], lu.dEntryDivisor.as<int>(),
                lu.dContributionStart.as<int>(), lu.dContributionL.as<int>(), lu.dContributionU.as<int>(),
                lu.dLu.as<double>(), lu.dTrouble.as<int>());
    }
    CUDA_CHECK(cudaGetLastError());
}

void solveScalarLanes(ScalarLu& lu, const double* b, double* x, cudaStream_t stream) {
    const int width = lu.width;
    if (width == 0) {
        return;
    }
    const dim3 tileBlock(TILE, TILE_ROWS);
    const dim3 vectorGrid((lu.n + TILE - 1) / TILE, (width + TILE - 1) / TILE);
    toInterleaved<<<vectorGrid, tileBlock, 0, stream>>>(lu.n, lu.n, width, lu.capacity, lu.dLaneMap.as<int>(),
            lu.dRowPosition.as<int>(), b, lu.dWork.as<double>());
    for (int l = 0; l + 1 < static_cast<int>(lu.forwardLevelStart.size()); l++) {
        const int rows = lu.forwardLevelStart[l + 1] - lu.forwardLevelStart[l];
        forwardLevel<<<blocksFor(static_cast<size_t>(rows) * width), LU_BLOCK, 0, stream>>>(rows, width, lu.capacity,
                lu.dForwardRow.as<int>() + lu.forwardLevelStart[l], lu.dLuRowPtr.as<int>(), lu.dLuCol.as<int>(),
                lu.dLuDiag.as<int>(), lu.dLu.as<double>(), lu.dWork.as<double>());
    }
    for (int l = 0; l + 1 < static_cast<int>(lu.backwardLevelStart.size()); l++) {
        const int rows = lu.backwardLevelStart[l + 1] - lu.backwardLevelStart[l];
        backwardLevel<<<blocksFor(static_cast<size_t>(rows) * width), LU_BLOCK, 0, stream>>>(rows, width, lu.capacity,
                lu.dBackwardRow.as<int>() + lu.backwardLevelStart[l], lu.dLuRowPtr.as<int>(), lu.dLuCol.as<int>(),
                lu.dLuDiag.as<int>(), lu.dLu.as<double>(), lu.dWork.as<double>());
    }
    fromInterleaved<<<vectorGrid, tileBlock, 0, stream>>>(lu.n, lu.n, width, lu.capacity, lu.dLaneMap.as<int>(),
            lu.dColPosition.as<int>(), lu.dWork.as<double>(), x);
    CUDA_CHECK(cudaGetLastError());
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

}  // namespace olfgpu
