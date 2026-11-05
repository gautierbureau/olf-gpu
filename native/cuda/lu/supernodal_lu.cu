// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See supernodal_lu.cuh.
#include "supernodal_lu.cuh"

#include <cmath>
#include <stdexcept>
#include <string>

#include "../common/check.cuh"

namespace olfgpu {

namespace {

constexpr int LU_BLOCK = 128;
constexpr int TILE = 32;                                     // transposes: 32 entries x 32 lanes per block
constexpr int TILE_ROWS = 8;
constexpr int B = LU_MAX_SUPERNODE;

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

// dst[map[k] * capacity + a] = src[k * stride + laneMap[a]]: values already interleaved by scenario
// (common/batch_layout.cuh) into the lane-interleaved layout, one thread per (entry, lane), the lanes fastest.
__global__ void fromScenarioInterleaved(int count, int stride, int width, int capacity, const int* __restrict__ laneMap,
                                        const int* __restrict__ map, const double* __restrict__ src,
                                        double* __restrict__ dst) {
    const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t < static_cast<size_t>(count) * width) {
        const int a = static_cast<int>(t % width);
        const int k = static_cast<int>(t / width);
        dst[static_cast<size_t>(map[k]) * capacity + a] = src[static_cast<size_t>(k) * stride + laneMap[a]];
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

// Gather: every (entry, lane) of the level's supernodes takes its descendants' runs, in ascending order.
__global__ void gatherLevel(int nItems, int width, int capacity, const int* __restrict__ levelEntry,
                            const int* __restrict__ runStart, const int* __restrict__ runLower,
                            const int* __restrict__ runUpper, const int* __restrict__ runLength,
                            double* __restrict__ lu) {
    const size_t g = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= static_cast<size_t>(nItems) * width) {
        return;
    }
    const int t = static_cast<int>(g / width);
    const int a = static_cast<int>(g - static_cast<size_t>(t) * width);
    const int e = levelEntry[t];
    const int rEnd = runStart[e + 1];
    if (runStart[e] == rEnd) {
        return;
    }
    double acc = lu[static_cast<size_t>(e) * capacity + a];
    for (int r = runStart[e]; r < rEnd; r++) {
        const double* l = lu + static_cast<size_t>(runLower[r]) * capacity + a;
        const double* u = lu + static_cast<size_t>(runUpper[r]) * capacity + a;
        const int length = runLength[r];
        for (int k = 0; k < length; k++) {
            acc = fma(-l[static_cast<size_t>(k) * capacity], u[static_cast<size_t>(k) * capacity], acc);
        }
    }
    lu[static_cast<size_t>(e) * capacity + a] = acc;
}

// Diagonal blocks: every (supernode, lane) factors its b x b block, the slot -> row and slot -> column orders kept per
// lane, the block stored back in slot order (L unit lower, U upper). Each step pivots first on a row holding a single
// nonzero with nothing in its U panel: such a row (a voltage target's identity row) is then solved exactly, as KLU's
// singleton blocks are - its value comes out at its target to the last bit, which OLF's strict comparisons of a voltage
// with its target rely on - and eliminating with it moves nothing else. Otherwise complete pivoting, ties to the first
// position in row-major order.
__global__ void factorBlocks(int nItems, int width, int capacity, const int* __restrict__ levelSupernode,
                             const int* __restrict__ supernodeStart, const int* __restrict__ diagonalBase,
                             const int* __restrict__ rowStart, const int* __restrict__ upperBase,
                             double* __restrict__ lu, unsigned char* __restrict__ rowPermutation,
                             unsigned char* __restrict__ columnPermutation, int* __restrict__ pivotClamps) {
    const size_t g = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= static_cast<size_t>(nItems) * width) {
        return;
    }
    const int t = static_cast<int>(g / width);
    const int a = static_cast<int>(g - static_cast<size_t>(t) * width);
    const int s = levelSupernode[t];
    const int f = supernodeStart[s];
    const int b = supernodeStart[s + 1] - f;
    double* block = lu + static_cast<size_t>(diagonalBase[s]) * capacity + a;
    double d[B * B];
    unsigned char rows[B];
    unsigned char columns[B];
    for (int q = 0; q < b * b; q++) {
        d[q] = block[static_cast<size_t>(q) * capacity];
    }
    for (int q = 0; q < b; q++) {
        rows[q] = static_cast<unsigned char>(q);
        columns[q] = static_cast<unsigned char>(q);
    }
    bool panelEmpty[B];                                          // per block row: nothing in its U panel
    const int count = rowStart[s + 1] - rowStart[s];
    const double* upper = lu + static_cast<size_t>(upperBase[s]) * capacity + a;
    for (int q = 0; q < b; q++) {
        bool empty = true;
        for (int column = 0; column < count && empty; column++) {
            empty = upper[(static_cast<size_t>(column) * b + q) * capacity] == 0.0;
        }
        panelEmpty[q] = empty;
    }
    int clamps = 0;
    for (int k = 0; k < b; k++) {
        int pr = -1;
        int pc = -1;
        for (int r = k; r < b && pr < 0; r++) {
            if (!panelEmpty[rows[r]]) {
                continue;
            }
            int nonZeros = 0;
            int column = -1;
            for (int c = k; c < b; c++) {
                if (d[r * b + c] != 0.0) {
                    nonZeros++;
                    column = c;
                }
            }
            if (nonZeros == 1) {
                pr = r;
                pc = column;
            }
        }
        if (pr < 0) {
            pr = k;
            pc = k;
            double best = -1.0;
            for (int r = k; r < b; r++) {
                for (int c = k; c < b; c++) {
                    const double m = fabs(d[r * b + c]);
                    if (m > best) {
                        best = m;
                        pr = r;
                        pc = c;
                    }
                }
            }
        }
        if (pr != k) {
            for (int c = 0; c < b; c++) {
                const double x = d[k * b + c];
                d[k * b + c] = d[pr * b + c];
                d[pr * b + c] = x;
            }
            const unsigned char x = rows[k];
            rows[k] = rows[pr];
            rows[pr] = x;
        }
        if (pc != k) {
            for (int r = 0; r < b; r++) {
                const double x = d[r * b + k];
                d[r * b + k] = d[r * b + pc];
                d[r * b + pc] = x;
            }
            const unsigned char x = columns[k];
            columns[k] = columns[pc];
            columns[pc] = x;
        }
        double pivot = d[k * b + k];
        if (!(fabs(pivot) >= LU_PIVOT_EPSILON)) {
            if (!isnan(pivot)) {
                pivot = pivot < 0.0 ? -LU_PIVOT_EPSILON : LU_PIVOT_EPSILON;
                d[k * b + k] = pivot;
            }
            clamps++;
        }
        for (int r = k + 1; r < b; r++) {
            const double multiplier = d[r * b + k] / pivot;
            d[r * b + k] = multiplier;
            for (int c = k + 1; c < b; c++) {
                d[r * b + c] = fma(-multiplier, d[k * b + c], d[r * b + c]);
            }
        }
    }
    for (int q = 0; q < b * b; q++) {
        block[static_cast<size_t>(q) * capacity] = d[q];
    }
    for (int q = 0; q < b; q++) {
        rowPermutation[static_cast<size_t>(f + q) * capacity + a] = rows[q];
        columnPermutation[static_cast<size_t>(f + q) * capacity + a] = columns[q];
    }
    if (clamps > 0) {
        atomicAdd(pivotClamps + a, clamps);
    }
}

// Panels: a U column of a supernode takes the block's row order and L^-1; an L row takes its column order and U^-1.
__global__ void applyPanels(int nItems, int width, int capacity, const int* __restrict__ panelSupernode,
                            const int* __restrict__ panelRow, const int* __restrict__ supernodeStart,
                            const int* __restrict__ rowStart, const int* __restrict__ diagonalBase,
                            const int* __restrict__ lowerBase, const int* __restrict__ upperBase,
                            const unsigned char* __restrict__ rowPermutation,
                            const unsigned char* __restrict__ columnPermutation, double* __restrict__ lu) {
    const size_t g = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= static_cast<size_t>(nItems) * width) {
        return;
    }
    const int t = static_cast<int>(g / width);
    const int a = static_cast<int>(g - static_cast<size_t>(t) * width);
    const int s = panelSupernode[t];
    const int f = supernodeStart[s];
    const int b = supernodeStart[s + 1] - f;
    const int count = rowStart[s + 1] - rowStart[s];
    const int item = panelRow[t];
    const double* d = lu + static_cast<size_t>(diagonalBase[s]) * capacity + a;
    double v[B];
    double w[B];
    if (item < count) {                                      // U column: w = L^-1 P v
        double* column = lu + (static_cast<size_t>(upperBase[s]) + static_cast<size_t>(item) * b) * capacity + a;
        for (int k = 0; k < b; k++) {
            v[k] = column[static_cast<size_t>(k) * capacity];
        }
        for (int k = 0; k < b; k++) {
            double x = v[rowPermutation[static_cast<size_t>(f + k) * capacity + a]];
            for (int q = 0; q < k; q++) {
                x = fma(-d[static_cast<size_t>(k * b + q) * capacity], w[q], x);
            }
            w[k] = x;
        }
        for (int k = 0; k < b; k++) {
            column[static_cast<size_t>(k) * capacity] = w[k];
        }
    } else {                                                 // L row: w U = v Q
        double* row = lu + (static_cast<size_t>(lowerBase[s]) + static_cast<size_t>(item - count) * b) * capacity + a;
        for (int c = 0; c < b; c++) {
            v[c] = row[static_cast<size_t>(c) * capacity];
        }
        for (int c = 0; c < b; c++) {
            double x = v[columnPermutation[static_cast<size_t>(f + c) * capacity + a]];
            for (int q = 0; q < c; q++) {
                x = fma(-w[q], d[static_cast<size_t>(q * b + c) * capacity], x);
            }
            w[c] = x / d[static_cast<size_t>(c * b + c) * capacity];
        }
        for (int c = 0; c < b; c++) {
            row[static_cast<size_t>(c) * capacity] = w[c];
        }
    }
}

// Forward, one level: a supernode's rows take their descendants' L runs, then its row order and L^-1; the result
// stays in slot order.
__global__ void forwardLevel(int nItems, int width, int capacity, const int* __restrict__ levelSupernode,
                             const int* __restrict__ supernodeStart, const int* __restrict__ diagonalBase,
                             const int* __restrict__ forwardRunStart, const int* __restrict__ forwardRunLower,
                             const int* __restrict__ forwardRunY, const int* __restrict__ forwardRunLength,
                             const unsigned char* __restrict__ rowPermutation, const double* __restrict__ lu,
                             double* __restrict__ z) {
    const size_t g = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= static_cast<size_t>(nItems) * width) {
        return;
    }
    const int t = static_cast<int>(g / width);
    const int a = static_cast<int>(g - static_cast<size_t>(t) * width);
    const int s = levelSupernode[t];
    const int f = supernodeStart[s];
    const int b = supernodeStart[s + 1] - f;
    double v[B];
    double w[B];
    for (int r = 0; r < b; r++) {
        const int i = f + r;
        double x = z[static_cast<size_t>(i) * capacity + a];
        for (int run = forwardRunStart[i]; run < forwardRunStart[i + 1]; run++) {
            const double* l = lu + static_cast<size_t>(forwardRunLower[run]) * capacity + a;
            const double* y = z + static_cast<size_t>(forwardRunY[run]) * capacity + a;
            const int length = forwardRunLength[run];
            for (int k = 0; k < length; k++) {
                x = fma(-l[static_cast<size_t>(k) * capacity], y[static_cast<size_t>(k) * capacity], x);
            }
        }
        v[r] = x;
    }
    const double* d = lu + static_cast<size_t>(diagonalBase[s]) * capacity + a;
    for (int k = 0; k < b; k++) {
        double x = v[rowPermutation[static_cast<size_t>(f + k) * capacity + a]];
        for (int q = 0; q < k; q++) {
            x = fma(-d[static_cast<size_t>(k * b + q) * capacity], w[q], x);
        }
        w[k] = x;
    }
    for (int k = 0; k < b; k++) {
        z[static_cast<size_t>(f + k) * capacity + a] = w[k];
    }
}

// Backward, one level: a supernode's slots take its U panel against the solution below, then U^-1, and land on its
// columns through its column order.
__global__ void backwardLevel(int nItems, int width, int capacity, const int* __restrict__ levelSupernode,
                              const int* __restrict__ supernodeStart, const int* __restrict__ rowStart,
                              const int* __restrict__ rows, const int* __restrict__ diagonalBase,
                              const int* __restrict__ upperBase, const unsigned char* __restrict__ columnPermutation,
                              const double* __restrict__ lu, double* __restrict__ z) {
    const size_t g = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= static_cast<size_t>(nItems) * width) {
        return;
    }
    const int t = static_cast<int>(g / width);
    const int a = static_cast<int>(g - static_cast<size_t>(t) * width);
    const int s = levelSupernode[t];
    const int f = supernodeStart[s];
    const int b = supernodeStart[s + 1] - f;
    double v[B];
    for (int k = 0; k < b; k++) {
        v[k] = z[static_cast<size_t>(f + k) * capacity + a];
    }
    const double* upper = lu + static_cast<size_t>(upperBase[s]) * capacity + a;
    for (int q = rowStart[s]; q < rowStart[s + 1]; q++) {
        const double x = z[static_cast<size_t>(rows[q]) * capacity + a];
        const double* column = upper + static_cast<size_t>(q - rowStart[s]) * b * capacity;
        for (int k = 0; k < b; k++) {
            v[k] = fma(-column[static_cast<size_t>(k) * capacity], x, v[k]);
        }
    }
    const double* d = lu + static_cast<size_t>(diagonalBase[s]) * capacity + a;
    for (int k = b - 1; k >= 0; k--) {
        double x = v[k];
        for (int q = k + 1; q < b; q++) {
            x = fma(-d[static_cast<size_t>(k * b + q) * capacity], v[q], x);
        }
        v[k] = x / d[static_cast<size_t>(k * b + k) * capacity];
    }
    for (int c = 0; c < b; c++) {
        z[static_cast<size_t>(f + columnPermutation[static_cast<size_t>(f + c) * capacity + a]) * capacity + a] = v[c];
    }
}

template <typename T>
void uploadVector(DeviceBuf& buf, const std::vector<T>& v) {
    if (!v.empty()) {
        buf.upload(v.data(), v.size() * sizeof(T));
    }
}

}  // namespace

void createSupernodalLu(SupernodalLu& lu, const SupernodalLuPlan& plan, int capacity) {
    if (capacity <= 0) {
        throw std::runtime_error("batched LU capacity must be positive: " + std::to_string(capacity));
    }
    lu.n = plan.n;
    lu.nnzA = plan.nnzA;
    lu.nStored = plan.nStored;
    lu.nSupernodes = plan.nSupernodes;
    lu.capacity = capacity;
    lu.width = 0;
    lu.levelStart = plan.levelStart;
    lu.levelEntryStart = plan.levelEntryStart;
    lu.levelPanelStart = plan.levelPanelStart;
    std::vector<char> fromA(plan.nStored, 0);
    for (int e : plan.storedOfA) {
        fromA[e] = 1;
    }
    std::vector<int> fillEntry;
    for (int e = 0; e < plan.nStored; e++) {
        if (!fromA[e]) {
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
    uploadVector(lu.dStoredOfA, plan.storedOfA);
    uploadVector(lu.dFillEntry, fillEntry);
    uploadVector(lu.dSupernodeStart, plan.supernodeStart);
    uploadVector(lu.dRowStart, plan.rowStart);
    uploadVector(lu.dRows, plan.rows);
    uploadVector(lu.dDiagonalBase, plan.diagonalBase);
    uploadVector(lu.dLowerBase, plan.lowerBase);
    uploadVector(lu.dUpperBase, plan.upperBase);
    uploadVector(lu.dLevelSupernode, plan.levelSupernode);
    uploadVector(lu.dRunStart, plan.runStart);
    uploadVector(lu.dRunLower, plan.runLower);
    uploadVector(lu.dRunUpper, plan.runUpper);
    uploadVector(lu.dRunLength, plan.runLength);
    uploadVector(lu.dLevelEntry, plan.levelEntry);
    uploadVector(lu.dPanelSupernode, plan.panelSupernode);
    uploadVector(lu.dPanelRow, plan.panelRow);
    uploadVector(lu.dForwardRunStart, plan.forwardRunStart);
    uploadVector(lu.dForwardRunLower, plan.forwardRunLower);
    uploadVector(lu.dForwardRunY, plan.forwardRunY);
    uploadVector(lu.dForwardRunLength, plan.forwardRunLength);
    uploadVector(lu.dRowPosition, rowPosition);
    uploadVector(lu.dColPosition, colPosition);
    lu.dLu.alloc(static_cast<size_t>(plan.nStored) * capacity * sizeof(double));
    lu.dRowPermutation.alloc(static_cast<size_t>(plan.n) * capacity);
    lu.dColumnPermutation.alloc(static_cast<size_t>(plan.n) * capacity);
    lu.dWork.alloc(static_cast<size_t>(plan.n) * capacity * sizeof(double));
    lu.dLaneMap.alloc(static_cast<size_t>(capacity) * sizeof(int));
    lu.dPivotClamps.alloc(static_cast<size_t>(capacity) * sizeof(int));
}

void factorizeSupernodalLanes(SupernodalLu& lu, const double* vals, BatchLayout layout, const int* laneMap, int width,
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
    CUDA_CHECK(cudaMemsetAsync(lu.dPivotClamps.p, 0, width * sizeof(int), stream));
    if (layout.interleaved) {
        fromScenarioInterleaved<<<blocksFor(static_cast<size_t>(lu.nnzA) * width), LU_BLOCK, 0, stream>>>(lu.nnzA,
                layout.stride, width, lu.capacity, lu.dLaneMap.as<int>(), lu.dStoredOfA.as<int>(), vals, lu.dLu.as<double>());
    } else {
        const dim3 tileBlock(TILE, TILE_ROWS);
        const dim3 valuesGrid((lu.nnzA + TILE - 1) / TILE, (width + TILE - 1) / TILE);
        toInterleaved<<<valuesGrid, tileBlock, 0, stream>>>(lu.nnzA, lu.nnzA, width, lu.capacity, lu.dLaneMap.as<int>(),
                lu.dStoredOfA.as<int>(), vals, lu.dLu.as<double>());
    }
    if (lu.nFill > 0) {
        zeroFill<<<blocksFor(static_cast<size_t>(lu.nFill) * width), LU_BLOCK, 0, stream>>>(lu.nFill, width,
                lu.capacity, lu.dFillEntry.as<int>(), lu.dLu.as<double>());
    }
    for (int l = 0; l + 1 < static_cast<int>(lu.levelStart.size()); l++) {
        const int entries = lu.levelEntryStart[l + 1] - lu.levelEntryStart[l];
        if (entries > 0) {
            gatherLevel<<<blocksFor(static_cast<size_t>(entries) * width), LU_BLOCK, 0, stream>>>(entries, width,
                    lu.capacity, lu.dLevelEntry.as<int>() + lu.levelEntryStart[l], lu.dRunStart.as<int>(),
                    lu.dRunLower.as<int>(), lu.dRunUpper.as<int>(), lu.dRunLength.as<int>(), lu.dLu.as<double>());
        }
        const int supernodes = lu.levelStart[l + 1] - lu.levelStart[l];
        factorBlocks<<<blocksFor(static_cast<size_t>(supernodes) * width), LU_BLOCK, 0, stream>>>(supernodes, width,
                lu.capacity, lu.dLevelSupernode.as<int>() + lu.levelStart[l], lu.dSupernodeStart.as<int>(),
                lu.dDiagonalBase.as<int>(), lu.dRowStart.as<int>(), lu.dUpperBase.as<int>(), lu.dLu.as<double>(),
                lu.dRowPermutation.as<unsigned char>(),
                lu.dColumnPermutation.as<unsigned char>(), lu.dPivotClamps.as<int>());
        const int panels = lu.levelPanelStart[l + 1] - lu.levelPanelStart[l];
        if (panels > 0) {
            applyPanels<<<blocksFor(static_cast<size_t>(panels) * width), LU_BLOCK, 0, stream>>>(panels, width,
                    lu.capacity, lu.dPanelSupernode.as<int>() + lu.levelPanelStart[l],
                    lu.dPanelRow.as<int>() + lu.levelPanelStart[l], lu.dSupernodeStart.as<int>(),
                    lu.dRowStart.as<int>(), lu.dDiagonalBase.as<int>(), lu.dLowerBase.as<int>(),
                    lu.dUpperBase.as<int>(), lu.dRowPermutation.as<unsigned char>(),
                    lu.dColumnPermutation.as<unsigned char>(), lu.dLu.as<double>());
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

void solveSupernodalLanes(SupernodalLu& lu, const double* b, double* x, cudaStream_t stream) {
    const int width = lu.width;
    if (width == 0) {
        return;
    }
    const dim3 tileBlock(TILE, TILE_ROWS);
    const dim3 vectorGrid((lu.n + TILE - 1) / TILE, (width + TILE - 1) / TILE);
    toInterleaved<<<vectorGrid, tileBlock, 0, stream>>>(lu.n, lu.n, width, lu.capacity, lu.dLaneMap.as<int>(),
            lu.dRowPosition.as<int>(), b, lu.dWork.as<double>());
    const int levels = static_cast<int>(lu.levelStart.size()) - 1;
    for (int l = 0; l < levels; l++) {
        const int supernodes = lu.levelStart[l + 1] - lu.levelStart[l];
        forwardLevel<<<blocksFor(static_cast<size_t>(supernodes) * width), LU_BLOCK, 0, stream>>>(supernodes, width,
                lu.capacity, lu.dLevelSupernode.as<int>() + lu.levelStart[l], lu.dSupernodeStart.as<int>(),
                lu.dDiagonalBase.as<int>(), lu.dForwardRunStart.as<int>(), lu.dForwardRunLower.as<int>(),
                lu.dForwardRunY.as<int>(), lu.dForwardRunLength.as<int>(), lu.dRowPermutation.as<unsigned char>(),
                lu.dLu.as<double>(), lu.dWork.as<double>());
    }
    for (int l = levels - 1; l >= 0; l--) {
        const int supernodes = lu.levelStart[l + 1] - lu.levelStart[l];
        backwardLevel<<<blocksFor(static_cast<size_t>(supernodes) * width), LU_BLOCK, 0, stream>>>(supernodes, width,
                lu.capacity, lu.dLevelSupernode.as<int>() + lu.levelStart[l], lu.dSupernodeStart.as<int>(),
                lu.dRowStart.as<int>(), lu.dRows.as<int>(), lu.dDiagonalBase.as<int>(), lu.dUpperBase.as<int>(),
                lu.dColumnPermutation.as<unsigned char>(), lu.dLu.as<double>(), lu.dWork.as<double>());
    }
    fromInterleaved<<<vectorGrid, tileBlock, 0, stream>>>(lu.n, lu.n, width, lu.capacity, lu.dLaneMap.as<int>(),
            lu.dColPosition.as<int>(), lu.dWork.as<double>(), x);
    CUDA_CHECK(cudaGetLastError());
}

std::vector<int> readSupernodalClamps(SupernodalLu& lu, cudaStream_t stream) {
    std::vector<int> clamps(lu.width, 0);
    if (lu.width > 0) {
        CUDA_CHECK(cudaMemcpyAsync(clamps.data(), lu.dPivotClamps.p, lu.width * sizeof(int), cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    return clamps;
}

}  // namespace olfgpu
