// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See zero_diagonal_repair.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/codes/gpu_linear_solve_mode.cuh"
#include "../solver/driver_steps.cuh"
#include "zero_diagonal_repair.cuh"

namespace olfgpu {

// The zero-diagonal repair is for a factorization that cannot pivot. The robust mode (6) matches and pivots, which
// puts a nonzero on every diagonal by itself, so it factors the raw Jacobian; a borrowed row there only adds the
// cancellation the repair is prone to. A ZERO row that borrows a balance row which keeps its own diagonal forms,
// with that donor, a 2x2 principal block that is singular (the ZERO row has nothing on the donor's diagonal column).
// network.xiidm busbar ADNO re-couples three loop couplers of a hub, and two of their ZERO_V rows borrowed the hub's
// P balance: the pivoted factorization of that repaired matrix failed as badly as the unpivoted one (ratio 5e21),
// while the raw matrix converged to the CPU's answer.








bool repairsZeroDiagonals(const BatchSolverContext* ctx) {
    return ctx->mode != GpuLinearSolveMode::ROBUST;
}

// ---------------------------------------------------------------------------------------------
// ZERO-DIAGONAL ROW REPAIR. The batched solve runs cuDSS without pivoting or matching (neither is
// supported for uniform batches), on the promise that the diagonal-pairing row permutation leaves
// every row a nonzero diagonal. A zero-impedance CLUSTER breaks that promise twice over: the couplers'
// ZERO_V / ZERO_PHI rows never carry their own dummy, and a section whose only branches are couplers
// has P and Q rows made of dummies alone — as a plain balance (the dead-end relocation covers that
// one shape) or, with a generator on every section, as the merged group's DISTR_Q rows, which the
// relocation must not touch. cuDSS then perturbs the zero pivot and returns a wrong STEP with a
// converging trajectory: measured on rte6515's 500-busbar structure at VL-1113, ||J.dx-F||inf = 82
// for ||F||inf = 1.24 on the first step of EVERY scenario, and the answer differs from OLF's wherever
// OLF stops one iteration short (BUS-1196_BBS, 5.157e-02 kV).
//
// The repair is numeric and per scenario, applied to the assembled values right before they are
// factorized: a row whose diagonal is exactly zero borrows a row that carries its diagonal column,
//     row_r += s_r * row_d          (values AND right-hand side)
// which leaves J dx = -F with the same solution (the combination matrix is nonsingular: P rows
// borrow with +1 and the DUMMY_P row with -1, Q rows with -1 and the DUMMY_Q row with +1; on a
// spanning tree the borrowing closes only in 2-cycles P <-> ZERO_PHI and Q <-> ZERO_V, whose
// determinant 1 - s_h*s_d is 2). Candidates are structural — a dummy row's two endpoint rows (and the pilot-Q relocation
// candidates when an endpoint's Q rides there), an endpoint row's incident couplers' dummy rows —
// and the choice is numeric, so a coupler switched off or a section turned into an identity this scenario
// simply is not a donor (an identity row has nothing but its own 1). Every right-hand side handed to
// those factors goes through the same combination (the unit vectors of the sensitivity solves too).
// The pattern carries the union of each row with its candidates so the addition lands in place.
//
// Only the scenarios being filled are repaired: a retired one holds its identity system (diagonal 1), or the values
// its last fill left repaired, and gets its identity before it is solved again.
// ---------------------------------------------------------------------------------------------
// One thread per hazard-row value of a scenario being filled, so a warp copies consecutive values.
__global__ void snapshotZeroDiagonalRowsBatch(int nZeroDiagonalRowValues, int nS, int nnz, const int* zeroDiagonalRowValuePosition,
                                const char* active, const double* vals, double* snap) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nZeroDiagonalRowValues * nS) return;
    int j = tid % nZeroDiagonalRowValues;
    int s = tid / nZeroDiagonalRowValues;
    if (!active[s]) return;
    snap[tid] = vals[(size_t) s * nnz + zeroDiagonalRowValuePosition[j]];
}

__global__ void repairZeroDiagonalRowsBatch(int nZeroDiagonalRows, int nS, int nnz, int nZeroDiagonalRowValues, const int* zeroDiagonalRow, const int* zeroDiagonalRowValueStart,
                              const int* zeroDiagonalDonorStart, const int* zeroDiagonalDonor, const int* zeroDiagonalDonorOwnIndex,
                              const char* zeroDiagonalRepairSign, const int* rowPtr, const int* colIdx, const int* diagPos,
                              const char* active, const double* snap, double* vals, int* choice) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nZeroDiagonalRows * nS) return;
    int i = tid % nZeroDiagonalRows;
    int s = tid / nZeroDiagonalRows;
    int r = zeroDiagonalRow[i];
    double* v = vals + (size_t) s * nnz;
    choice[(size_t) s * nZeroDiagonalRows + i] = -1;
    if (!active[s] || v[diagPos[r]] != 0.0) return;         // retired, or a real (or identity) diagonal: nothing to do
    const double* sn = snap + (size_t) s * nZeroDiagonalRowValues;
    double sgn = zeroDiagonalRepairSign[i];
    for (int j = zeroDiagonalDonorStart[i]; j < zeroDiagonalDonorStart[i + 1]; j++) {
        int d = zeroDiagonalDonor[j];
        int hd = zeroDiagonalDonorOwnIndex[j];                             // donor's own hazard slot (its ORIGINAL row is in snap)
        int p = csrFind(rowPtr, colIdx, d, r);
        if (p < 0) continue;
        double coef = hd >= 0 ? sn[zeroDiagonalRowValueStart[hd] + (p - rowPtr[d])] : v[p];
        if (coef == 0.0) continue;                           // this donor does not carry my column this scenario
        for (int k = rowPtr[d]; k < rowPtr[d + 1]; k++) {
            double dv = hd >= 0 ? sn[zeroDiagonalRowValueStart[hd] + (k - rowPtr[d])] : v[k];
            if (dv == 0.0) continue;
            int q = csrFind(rowPtr, colIdx, r, colIdx[k]);
            if (q >= 0) v[q] += sgn * dv;
        }
        choice[(size_t) s * nZeroDiagonalRows + i] = j;
        return;
    }
}

__global__ void snapshotZeroDiagonalRightHandSideBatch(int nZeroDiagonalRows, int nS, int n, const int* zeroDiagonalRow, const double* B, double* bsnap) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nZeroDiagonalRows * nS) return;
    int i = tid % nZeroDiagonalRows;
    int s = tid / nZeroDiagonalRows;
    bsnap[tid] = B[(size_t) s * n + zeroDiagonalRow[i]];
}

__global__ void repairZeroDiagonalRightHandSideBatch(int nZeroDiagonalRows, int nS, int n, const int* zeroDiagonalRow, const int* zeroDiagonalDonor, const int* zeroDiagonalDonorOwnIndex,
                             const char* zeroDiagonalRepairSign, const int* choice, const double* bsnap, double* B) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nZeroDiagonalRows * nS) return;
    int i = tid % nZeroDiagonalRows;
    int s = tid / nZeroDiagonalRows;
    int j = choice[tid];
    if (j < 0) return;
    int d = zeroDiagonalDonor[j];
    int hd = zeroDiagonalDonorOwnIndex[j];
    double bd = hd >= 0 ? bsnap[(size_t) s * nZeroDiagonalRows + hd] : B[(size_t) s * n + d];
    B[(size_t) s * n + zeroDiagonalRow[i]] += zeroDiagonalRepairSign[i] * bd;
}

void repairZeroDiagonalRows(BatchSolverContext* ctx, int nS, double* vals) {
    auto* structure = ctx->structure.get();
    if (structure->nZeroDiagonalRows == 0 || !repairsZeroDiagonals(ctx)) return;
    if (ctx->zeroDiagonalRepairCapacity < nS) {
        ctx->dZeroDiagonalRowSnapshot.alloc((size_t) nS * structure->nZeroDiagonalRowValues * sizeof(double));
        ctx->dZeroDiagonalRightHandSideSnapshot.alloc((size_t) nS * structure->nZeroDiagonalRows * sizeof(double));
        ctx->dZeroDiagonalDonorChoice.alloc((size_t) nS * structure->nZeroDiagonalRows * sizeof(int));
        ctx->zeroDiagonalRepairCapacity = nS;
    }
    int nnz = structure->nnz;
    snapshotZeroDiagonalRowsBatch<<<grid(structure->nZeroDiagonalRowValues * nS), BLOCK, 0, ctx->stream>>>(structure->nZeroDiagonalRowValues, nS, nnz,
            structure->dZeroDiagonalRowValuePosition.template as<int>(), ctx->dActive.template as<char>(), vals,
            ctx->dZeroDiagonalRowSnapshot.template as<double>());
    repairZeroDiagonalRowsBatch<<<((structure->nZeroDiagonalRows * nS + BLOCK - 1) / BLOCK), BLOCK, 0, ctx->stream>>>(structure->nZeroDiagonalRows, nS, nnz, structure->nZeroDiagonalRowValues,
            structure->dZeroDiagonalRow.template as<int>(), structure->dZeroDiagonalRowValueStart.template as<int>(), structure->dZeroDiagonalDonorStart.template as<int>(),
                    structure->dZeroDiagonalDonor.template as<int>(),
            structure->dZeroDiagonalDonorOwnIndex.template as<int>(), structure->dZeroDiagonalRepairSign.template as<char>(), structure->dRowPtr.template as<int>(),
                    structure->dColIdx.template as<int>(),
            structure->dDiagPos.template as<int>(), ctx->dActive.template as<char>(), ctx->dZeroDiagonalRowSnapshot.template as<double>(), vals,
            ctx->dZeroDiagonalDonorChoice.template as<int>());
}

/** The same combination on a right-hand side (nS scenarios from {@code B}, choices from the last
 *  repairZeroDiagonalRows on this context; {@code sOff} selects the scenario block for a per-scenario call). */
void repairZeroDiagonalRightHandSide(BatchSolverContext* ctx, int nS, double* B, int sOff) {
    auto* structure = ctx->structure.get();
    if (structure->nZeroDiagonalRows == 0 || ctx->zeroDiagonalRepairCapacity < sOff + nS || !repairsZeroDiagonals(ctx)) return;
    int n = structure->n;
    double* bsnap = ctx->dZeroDiagonalRightHandSideSnapshot.template as<double>() + (size_t) sOff * structure->nZeroDiagonalRows;
    snapshotZeroDiagonalRightHandSideBatch<<<((structure->nZeroDiagonalRows * nS + BLOCK - 1) / BLOCK), BLOCK, 0, ctx->stream>>>(structure->nZeroDiagonalRows, nS, n, structure->dZeroDiagonalRow.template as<int>(), B, bsnap);
    repairZeroDiagonalRightHandSideBatch<<<((structure->nZeroDiagonalRows * nS + BLOCK - 1) / BLOCK), BLOCK, 0, ctx->stream>>>(structure->nZeroDiagonalRows, nS, n,
            structure->dZeroDiagonalRow.template as<int>(),
            structure->dZeroDiagonalDonor.template as<int>(), structure->dZeroDiagonalDonorOwnIndex.template as<int>(), structure->dZeroDiagonalRepairSign.template as<char>(),
            ctx->dZeroDiagonalDonorChoice.template as<int>() + (size_t) sOff * structure->nZeroDiagonalRows, bsnap, B);
}

}  // namespace olfgpu
