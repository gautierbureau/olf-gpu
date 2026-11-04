// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See singleton_substitution.cuh.
#include "singleton_substitution.cuh"

#include <algorithm>
#include <utility>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../fixedmatrix/zero_diagonal_repair.cuh"
#include "../lu/batched_lu.cuh"

namespace olfgpu {

namespace {

constexpr int NO_OWNER = 0x7f7f7f7f;                         // a memset to 0x7f bytes: above every row index
constexpr int MAX_SWEEPS = 64;
constexpr int NO_PROPOSAL = -1;                              // no variable free: the row has nothing left to fix
constexpr int PENDING = -2;                                  // two variables or more free: a later sweep may fix one

// Where a row's ORIGINAL equation lives: the repair's snapshots for a hazard row, the values as solved otherwise.
struct OriginalSystem {
    int n = 0, nnz = 0, nZeroDiagonalRows = 0, nZeroDiagonalRowValues = 0;
    const int* rowPtr = nullptr;
    const int* colIdx = nullptr;
    const double* vals = nullptr;
    const double* b = nullptr;
    const int* hazardSlot = nullptr;                         // per row: its slot in the repair's tables, -1 for none
    const int* valueStart = nullptr;
    const double* valueSnapshot = nullptr;                   // null: no repair on this context
    const double* rightHandSideSnapshot = nullptr;
};

__device__ inline double originalValue(const OriginalSystem& o, int s, int r, int k) {
    const int h = o.hazardSlot[r];
    if (h >= 0 && o.valueSnapshot != nullptr) {
        return o.valueSnapshot[static_cast<size_t>(s) * o.nZeroDiagonalRowValues + o.valueStart[h] + (k - o.rowPtr[r])];
    }
    return o.vals[static_cast<size_t>(s) * o.nnz + k];
}

__device__ inline double originalRightHandSide(const OriginalSystem& o, int s, int r) {
    const int h = o.hazardSlot[r];
    if (h >= 0 && o.rightHandSideSnapshot != nullptr) {
        return o.rightHandSideSnapshot[static_cast<size_t>(s) * o.nZeroDiagonalRows + h];
    }
    return o.b[static_cast<size_t>(s) * o.n + r];
}

// The sweeps run over the lanes the solve solved: cell t = i * n + r is row r of solved lane i, the scenario
// laneMap[i]. The per-cell scratch (fixed, owner, touched) is indexed by cell, the systems by scenario. A sweep visits
// a list of cells: the short rows first, then those a sweep left with two free variables or more - a row with one free
// variable gets it fixed in that sweep (by itself or by a lower row) and one with none has nothing left to do, so
// neither proposes again. Of the listed rows, a sweep reads only those a variable fixed by the sweep before touches:
// any other still has the free variables it had when it was last read. The list's order is free: a cell's proposal
// is its own, and a variable's owner is the lowest row proposing it whatever the order.

// Appends t to the list when flagged, one atomic per warp; every thread of the warp calls it.
__device__ inline void appendCell(bool flagged, int t, int* __restrict__ list, int* __restrict__ count) {
    const unsigned ballot = __ballot_sync(0xffffffffu, flagged);
    const int lane = threadIdx.x & 31;
    int base = 0;
    if (lane == 0 && ballot != 0) {
        base = atomicAdd(count, __popc(ballot));
    }
    base = __shfl_sync(0xffffffffu, base, 0);
    if (flagged) {
        list[base + __popc(ballot & ((1u << lane) - 1u))] = t;
    }
}

// Lists the (row, solved lane) cells whose row's original equation has 1..SINGLETON_MAX_TERMS nonzero coefficients.
__global__ void listShortRows(OriginalSystem o, int width, const int* __restrict__ laneMap, int* __restrict__ list,
                              int* __restrict__ count) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    bool isShort = false;
    if (t < o.n * width) {
        const int i = t / o.n;
        const int r = t - i * o.n;
        const int s = laneMap[i];
        int terms = 0;
        for (int k = o.rowPtr[r]; k < o.rowPtr[r + 1] && terms <= SINGLETON_MAX_TERMS; k++) {
            terms += originalValue(o, s, r, k) != 0.0 ? 1 : 0;
        }
        isShort = terms >= 1 && terms <= SINGLETON_MAX_TERMS;
    }
    appendCell(isShort, t, list, count);
}

// Per listed cell: the one variable not yet fixed, if there is exactly one, and its value; the lowest row claims the
// variable. touched: the cells a variable fixed by the previous sweep touches (null on the first sweep, which reads
// every listed row), consumed here.
__global__ void proposeSingletons(OriginalSystem o, int listed, const int* __restrict__ list,
                                  const int* __restrict__ laneMap, const char* __restrict__ fixed,
                                  const double* __restrict__ dx, char* __restrict__ touched, int* __restrict__ owner,
                                  int* __restrict__ proposal, double* __restrict__ proposalValue) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= listed) {
        return;
    }
    const int t = list[p];
    if (touched != nullptr) {
        if (!touched[t]) {
            proposal[p] = PENDING;
            return;
        }
        touched[t] = 0;
    }
    const int i = t / o.n;
    const int r = t - i * o.n;
    const int s = laneMap[i];
    const size_t cells = static_cast<size_t>(i) * o.n;      // this lane's cells
    const size_t lane = static_cast<size_t>(s) * o.n;       // its scenario's dx
    double acc = originalRightHandSide(o, s, r);
    int unknowns = 0;
    int unknown = -1;
    double coefficient = 0.0;
    for (int k = o.rowPtr[r]; k < o.rowPtr[r + 1]; k++) {
        const double a = originalValue(o, s, r, k);
        if (a == 0.0) {
            continue;
        }
        const int j = o.colIdx[k];
        if (fixed[cells + j]) {
            acc = fma(-a, dx[lane + j], acc);
        } else {
            unknowns++;
            unknown = j;
            coefficient = a;
        }
    }
    if (unknowns == 1) {
        proposal[p] = unknown;
        proposalValue[p] = acc / coefficient;
        atomicMin(owner + cells + unknown, r);
    } else {
        proposal[p] = unknowns == 0 ? NO_PROPOSAL : PENDING;
    }
}

// The claiming row writes its variable, and marks the rows of its column (the pattern's) for the next sweep.
__global__ void applySingletons(int n, int listed, const int* __restrict__ list, const int* __restrict__ laneMap,
                                const int* __restrict__ owner, const int* __restrict__ proposal,
                                const double* __restrict__ proposalValue, const int* __restrict__ columnStart,
                                const int* __restrict__ columnRow, char* __restrict__ fixed, double* __restrict__ dx,
                                char* __restrict__ touched, int* __restrict__ fixedAny) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= listed) {
        return;
    }
    const int j = proposal[p];
    if (j < 0) {
        return;
    }
    const int t = list[p];
    const int i = t / n;
    const size_t cells = static_cast<size_t>(i) * n;
    if (owner[cells + j] != t - static_cast<int>(cells)) {
        return;
    }
    dx[static_cast<size_t>(laneMap[i]) * n + j] = proposalValue[p];
    fixed[cells + j] = 1;
    for (int k = columnStart[j]; k < columnStart[j + 1]; k++) {
        touched[cells + columnRow[k]] = 1;
    }
    *fixedAny = 1;
}

// Once every claim is applied: the claims cleared, and the cells still PENDING listed for the next sweep.
__global__ void listPendingRows(int n, int listed, const int* __restrict__ list, const int* __restrict__ proposal,
                                int* __restrict__ owner, int* __restrict__ next, int* __restrict__ nextCount) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    int t = -1;
    int j = NO_PROPOSAL;
    if (p < listed) {
        t = list[p];
        j = proposal[p];
        if (j >= 0) {
            owner[static_cast<size_t>(t / n) * n + j] = NO_OWNER;
        }
    }
    appendCell(j == PENDING, t, next, nextCount);
}

}  // namespace

void substituteSingletons(BatchSolverContext* ctx, const double* b, double* dx) {
    SharedStructure* st = ctx->structure.get();
    const int n = ctx->n;
    // the lanes the last factorization solved: a retired scenario's rows are not swept, its dx is its own
    BatchedLu& lu = *ctx->batchedLu;
    const int width = lu.width;
    if (width == 0) {
        return;
    }
    const int* laneMap = deviceLaneMap(lu);
    const size_t cells = static_cast<size_t>(n) * width;
    if (!ctx->dSingletonHazardSlot.p) {
        std::vector<int> hazardSlot(n, -1);
        for (int i = 0; i < st->nZeroDiagonalRows; i++) {
            hazardSlot[st->hZeroDiagonalRow[i]] = i;
        }
        ctx->dSingletonHazardSlot.upload(hazardSlot.data(), hazardSlot.size() * sizeof(int));
        // the pattern by column: the rows a fixed variable touches
        std::vector<int> columnStart(n + 1, 0);
        std::vector<int> columnRow(st->colIdx.size());
        for (int c : st->colIdx) {
            columnStart[c + 1]++;
        }
        for (int c = 0; c < n; c++) {
            columnStart[c + 1] += columnStart[c];
        }
        std::vector<int> fill(columnStart.begin(), columnStart.end() - 1);
        for (int r = 0; r < n; r++) {
            for (int k = st->rowPtr[r]; k < st->rowPtr[r + 1]; k++) {
                columnRow[fill[st->colIdx[k]]++] = r;
            }
        }
        ctx->dSingletonColumnStart.upload(columnStart.data(), columnStart.size() * sizeof(int));
        ctx->dSingletonColumnRow.upload(columnRow.data(), columnRow.size() * sizeof(int));
    }
    ctx->dSingletonList.ensure(cells * sizeof(int));
    ctx->dSingletonFixed.ensure(cells);
    ctx->dSingletonTouched.ensure(cells);
    ctx->dSingletonOwner.ensure(cells * sizeof(int));
    ctx->dSingletonProposal.ensure(cells * sizeof(int));
    ctx->dSingletonProposalValue.ensure(cells * sizeof(double));
    ctx->dSingletonCounters.ensure(3 * sizeof(int));        // listed, listed for the next sweep, a variable fixed
    OriginalSystem o;
    o.n = n;
    o.nnz = ctx->nnz;
    o.rowPtr = st->dRowPtr.as<int>();
    o.colIdx = st->dColIdx.as<int>();
    o.vals = ctx->dVals.as<double>();
    o.b = b;
    o.hazardSlot = ctx->dSingletonHazardSlot.as<int>();
    if (st->nZeroDiagonalRows > 0 && repairsZeroDiagonals(ctx) && ctx->zeroDiagonalRepairCapacity >= ctx->nScenarios) {
        o.nZeroDiagonalRows = st->nZeroDiagonalRows;
        o.nZeroDiagonalRowValues = st->nZeroDiagonalRowValues;
        o.valueStart = st->dZeroDiagonalRowValueStart.as<int>();
        o.valueSnapshot = ctx->dZeroDiagonalRowSnapshot.as<double>();
        o.rightHandSideSnapshot = ctx->dZeroDiagonalRightHandSideSnapshot.as<double>();
    }
    int* counters = ctx->dSingletonCounters.as<int>();
    CUDA_CHECK(cudaMemsetAsync(counters, 0, sizeof(int), ctx->stream));
    listShortRows<<<grid(static_cast<int>(cells)), BLOCK, 0, ctx->stream>>>(o, width, laneMap,
            ctx->dSingletonList.as<int>(), counters);
    CUDA_CHECK(cudaMemsetAsync(ctx->dSingletonFixed.p, 0, cells, ctx->stream));
    CUDA_CHECK(cudaMemsetAsync(ctx->dSingletonTouched.p, 0, cells, ctx->stream));
    CUDA_CHECK(cudaMemsetAsync(ctx->dSingletonOwner.p, 0x7f, cells * sizeof(int), ctx->stream));
    int listed = 0;
    CUDA_CHECK(cudaMemcpyAsync(&listed, counters, sizeof(int), cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    // the pending cells of a sweep are some of its listed ones: the second list never holds more than the first
    ctx->dSingletonPendingList.ensure(static_cast<size_t>(std::max(listed, 1)) * sizeof(int));
    int* list = ctx->dSingletonList.as<int>();
    int* next = ctx->dSingletonPendingList.as<int>();
    for (int sweep = 0; sweep < MAX_SWEEPS && listed > 0; sweep++) {
        CUDA_CHECK(cudaMemsetAsync(counters + 1, 0, 2 * sizeof(int), ctx->stream));
        proposeSingletons<<<grid(listed), BLOCK, 0, ctx->stream>>>(o, listed, list, laneMap, ctx->dSingletonFixed.as<char>(),
                dx, sweep == 0 ? nullptr : ctx->dSingletonTouched.as<char>(), ctx->dSingletonOwner.as<int>(),
                ctx->dSingletonProposal.as<int>(), ctx->dSingletonProposalValue.as<double>());
        applySingletons<<<grid(listed), BLOCK, 0, ctx->stream>>>(n, listed, list, laneMap, ctx->dSingletonOwner.as<int>(),
                ctx->dSingletonProposal.as<int>(), ctx->dSingletonProposalValue.as<double>(), ctx->dSingletonColumnStart.as<int>(),
                ctx->dSingletonColumnRow.as<int>(), ctx->dSingletonFixed.as<char>(), dx, ctx->dSingletonTouched.as<char>(),
                counters + 2);
        listPendingRows<<<grid(listed), BLOCK, 0, ctx->stream>>>(n, listed, list, ctx->dSingletonProposal.as<int>(),
                ctx->dSingletonOwner.as<int>(), next, counters + 1);
        int pendingAndFixed[2] = {0, 0};
        CUDA_CHECK(cudaMemcpyAsync(pendingAndFixed, counters + 1, 2 * sizeof(int), cudaMemcpyDeviceToHost, ctx->stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        if (!pendingAndFixed[1]) {
            break;
        }
        listed = pendingAndFixed[0];
        std::swap(list, next);
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace olfgpu
