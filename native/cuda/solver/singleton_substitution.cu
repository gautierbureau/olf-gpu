// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See singleton_substitution.cuh.
#include "singleton_substitution.cuh"

#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../fixedmatrix/zero_diagonal_repair.cuh"

namespace olfgpu {

namespace {

constexpr int NO_OWNER = 0x7f7f7f7f;                         // a memset to 0x7f bytes: above every row index
constexpr int MAX_SWEEPS = 64;

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

// Per (row, lane): 1 when the row's original equation has 1..SINGLETON_MAX_TERMS nonzero coefficients.
__global__ void markShortRows(OriginalSystem o, int nS, const char* __restrict__ active, char* __restrict__ shortRow) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= o.n * nS) {
        return;
    }
    const int s = t / o.n;
    const int r = t - s * o.n;
    int terms = 0;
    if (active[s]) {
        for (int k = o.rowPtr[r]; k < o.rowPtr[r + 1] && terms <= SINGLETON_MAX_TERMS; k++) {
            terms += originalValue(o, s, r, k) != 0.0 ? 1 : 0;
        }
    }
    shortRow[t] = terms >= 1 && terms <= SINGLETON_MAX_TERMS ? 1 : 0;
}

// Per short (row, lane): the one variable not yet fixed, if there is exactly one, and its value; the lowest row
// claims the variable.
__global__ void proposeSingletons(OriginalSystem o, int nS, const char* __restrict__ shortRow,
                                  const char* __restrict__ fixed, const double* __restrict__ dx, int* __restrict__ owner,
                                  int* __restrict__ proposal, double* __restrict__ proposalValue) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= o.n * nS) {
        return;
    }
    proposal[t] = -1;
    if (!shortRow[t]) {
        return;
    }
    const int s = t / o.n;
    const int r = t - s * o.n;
    const size_t lane = static_cast<size_t>(s) * o.n;
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
        if (fixed[lane + j]) {
            acc = fma(-a, dx[lane + j], acc);
        } else {
            unknowns++;
            unknown = j;
            coefficient = a;
        }
    }
    if (unknowns == 1) {
        proposal[t] = unknown;
        proposalValue[t] = acc / coefficient;
        atomicMin(owner + lane + unknown, r);
    }
}

// The claiming row writes its variable.
__global__ void applySingletons(int n, int nS, const int* __restrict__ owner, const int* __restrict__ proposal,
                                const double* __restrict__ proposalValue, char* __restrict__ fixed, double* __restrict__ dx,
                                int* __restrict__ fixedAny) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * nS) {
        return;
    }
    const int j = proposal[t];
    if (j < 0) {
        return;
    }
    const int s = t / n;
    const size_t lane = static_cast<size_t>(s) * n;
    if (owner[lane + j] != t - static_cast<int>(lane)) {
        return;
    }
    dx[lane + j] = proposalValue[t];
    fixed[lane + j] = 1;
    *fixedAny = 1;
}

}  // namespace

void substituteSingletons(BatchSolverContext* ctx, int nS, const double* b, double* dx) {
    SharedStructure* st = ctx->structure.get();
    const int n = ctx->n;
    const size_t cells = static_cast<size_t>(n) * nS;
    if (!ctx->dSingletonHazardSlot.p) {
        std::vector<int> hazardSlot(n, -1);
        for (int i = 0; i < st->nZeroDiagonalRows; i++) {
            hazardSlot[st->hZeroDiagonalRow[i]] = i;
        }
        ctx->dSingletonHazardSlot.upload(hazardSlot.data(), hazardSlot.size() * sizeof(int));
    }
    ctx->dSingletonShortRow.ensure(cells);
    ctx->dSingletonFixed.ensure(cells);
    ctx->dSingletonOwner.ensure(cells * sizeof(int));
    ctx->dSingletonProposal.ensure(cells * sizeof(int));
    ctx->dSingletonProposalValue.ensure(cells * sizeof(double));
    ctx->dSingletonFixedAny.ensure(sizeof(int));
    OriginalSystem o;
    o.n = n;
    o.nnz = ctx->nnz;
    o.rowPtr = st->dRowPtr.as<int>();
    o.colIdx = st->dColIdx.as<int>();
    o.vals = ctx->dVals.as<double>();
    o.b = b;
    o.hazardSlot = ctx->dSingletonHazardSlot.as<int>();
    if (st->nZeroDiagonalRows > 0 && repairsZeroDiagonals(ctx) && ctx->zeroDiagonalRepairCapacity >= nS) {
        o.nZeroDiagonalRows = st->nZeroDiagonalRows;
        o.nZeroDiagonalRowValues = st->nZeroDiagonalRowValues;
        o.valueStart = st->dZeroDiagonalRowValueStart.as<int>();
        o.valueSnapshot = ctx->dZeroDiagonalRowSnapshot.as<double>();
        o.rightHandSideSnapshot = ctx->dZeroDiagonalRightHandSideSnapshot.as<double>();
    }
    markShortRows<<<grid(static_cast<int>(cells)), BLOCK, 0, ctx->stream>>>(o, nS, ctx->dActive.as<char>(),
            ctx->dSingletonShortRow.as<char>());
    CUDA_CHECK(cudaMemsetAsync(ctx->dSingletonFixed.p, 0, cells, ctx->stream));
    for (int sweep = 0; sweep < MAX_SWEEPS; sweep++) {
        CUDA_CHECK(cudaMemsetAsync(ctx->dSingletonOwner.p, 0x7f, cells * sizeof(int), ctx->stream));
        CUDA_CHECK(cudaMemsetAsync(ctx->dSingletonFixedAny.p, 0, sizeof(int), ctx->stream));
        proposeSingletons<<<grid(static_cast<int>(cells)), BLOCK, 0, ctx->stream>>>(o, nS, ctx->dSingletonShortRow.as<char>(),
                ctx->dSingletonFixed.as<char>(), dx, ctx->dSingletonOwner.as<int>(), ctx->dSingletonProposal.as<int>(),
                ctx->dSingletonProposalValue.as<double>());
        applySingletons<<<grid(static_cast<int>(cells)), BLOCK, 0, ctx->stream>>>(n, nS, ctx->dSingletonOwner.as<int>(),
                ctx->dSingletonProposal.as<int>(), ctx->dSingletonProposalValue.as<double>(), ctx->dSingletonFixed.as<char>(), dx,
                ctx->dSingletonFixedAny.as<int>());
        int fixedAny = 0;
        CUDA_CHECK(cudaMemcpyAsync(&fixedAny, ctx->dSingletonFixedAny.p, sizeof(int), cudaMemcpyDeviceToHost, ctx->stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        if (!fixedAny) {
            break;
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace olfgpu
