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
// and the choice is numeric, so a coupler switched off or a section pinned to identity this scenario
// simply is not a donor (an identity row has nothing but its own 1). Every right-hand side handed to
// those factors goes through the same combination (the unit vectors of the sensitivity solves too).
// The pattern carries the union of each row with its candidates so the addition lands in place.
// ---------------------------------------------------------------------------------------------
__global__ void hazSnapshotVals(int nHaz, int nS, int nnz, int hazTot, const int* hazRow, const int* hazOff,
                                const int* rowPtr, const double* vals, double* snap) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nHaz * nS) return;
    int i = tid % nHaz;
    int s = tid / nHaz;
    int r = hazRow[i];
    const double* v = vals + (size_t) s * nnz;
    double* o = snap + (size_t) s * hazTot + hazOff[i];
    for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
        o[k - rowPtr[r]] = v[k];
    }
}

__global__ void hazRepairVals(int nHaz, int nS, int nnz, int hazTot, const int* hazRow, const int* hazOff,
                              const int* hazDonorPtr, const int* hazDonor, const int* hazDonorHaz,
                              const char* hazSign, const int* rowPtr, const int* colIdx, const int* diagPos,
                              const double* snap, double* vals, int* choice) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nHaz * nS) return;
    int i = tid % nHaz;
    int s = tid / nHaz;
    int r = hazRow[i];
    double* v = vals + (size_t) s * nnz;
    choice[(size_t) s * nHaz + i] = -1;
    if (v[diagPos[r]] != 0.0) return;                        // a real (or identity) diagonal: nothing to do
    const double* sn = snap + (size_t) s * hazTot;
    double sgn = hazSign[i];
    for (int j = hazDonorPtr[i]; j < hazDonorPtr[i + 1]; j++) {
        int d = hazDonor[j];
        int hd = hazDonorHaz[j];                             // donor's own hazard slot (its ORIGINAL row is in snap)
        int p = csrFind(rowPtr, colIdx, d, r);
        if (p < 0) continue;
        double coef = hd >= 0 ? sn[hazOff[hd] + (p - rowPtr[d])] : v[p];
        if (coef == 0.0) continue;                           // this donor does not carry my column this scenario
        for (int k = rowPtr[d]; k < rowPtr[d + 1]; k++) {
            double dv = hd >= 0 ? sn[hazOff[hd] + (k - rowPtr[d])] : v[k];
            if (dv == 0.0) continue;
            int q = csrFind(rowPtr, colIdx, r, colIdx[k]);
            if (q >= 0) v[q] += sgn * dv;
        }
        choice[(size_t) s * nHaz + i] = j;
        return;
    }
}

__global__ void hazSnapshotB(int nHaz, int nS, int n, const int* hazRow, const double* B, double* bsnap) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nHaz * nS) return;
    int i = tid % nHaz;
    int s = tid / nHaz;
    bsnap[tid] = B[(size_t) s * n + hazRow[i]];
}

__global__ void hazRepairRhs(int nHaz, int nS, int n, const int* hazRow, const int* hazDonor, const int* hazDonorHaz,
                             const char* hazSign, const int* choice, const double* bsnap, double* B) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nHaz * nS) return;
    int i = tid % nHaz;
    int s = tid / nHaz;
    int j = choice[tid];
    if (j < 0) return;
    int d = hazDonor[j];
    int hd = hazDonorHaz[j];
    double bd = hd >= 0 ? bsnap[(size_t) s * nHaz + hd] : B[(size_t) s * n + d];
    B[(size_t) s * n + hazRow[i]] += hazSign[i] * bd;
}

void hazRepairValues(BatchSolverContext* ctx, int nS, double* vals) {
    auto* st = ctx->st.get();
    if (st->nHaz == 0 || !repairsZeroDiagonals(ctx)) return;
    if (ctx->hazCap < nS) {
        ctx->dHazSnap.alloc((size_t) nS * st->hazTot * sizeof(double));
        ctx->dHazBSnap.alloc((size_t) nS * st->nHaz * sizeof(double));
        ctx->dHazChoice.alloc((size_t) nS * st->nHaz * sizeof(int));
        ctx->hazCap = nS;
    }
    int nnz = st->nnz;
    hazSnapshotVals<<<((st->nHaz * nS + BLOCK - 1) / BLOCK), BLOCK, 0, ctx->stream>>>(st->nHaz, nS, nnz, st->hazTot,
            st->dHazRow.template as<int>(), st->dHazOff.template as<int>(), st->dRowPtr.template as<int>(), vals, ctx->dHazSnap.template as<double>());
    hazRepairVals<<<((st->nHaz * nS + BLOCK - 1) / BLOCK), BLOCK, 0, ctx->stream>>>(st->nHaz, nS, nnz, st->hazTot,
            st->dHazRow.template as<int>(), st->dHazOff.template as<int>(), st->dHazDonorPtr.template as<int>(), st->dHazDonor.template as<int>(),
            st->dHazDonorHaz.template as<int>(), st->dHazSign.template as<char>(), st->dRowPtr.template as<int>(), st->dColIdx.template as<int>(),
            st->dDiagPos.template as<int>(), ctx->dHazSnap.template as<double>(), vals, ctx->dHazChoice.template as<int>());
}

/** The same combination on a right-hand side (nS scenarios from {@code B}, choices from the last
 *  hazRepairValues on this context; {@code sOff} selects the scenario block for a per-scenario call). */
void hazRepairRightHandSide(BatchSolverContext* ctx, int nS, double* B, int sOff) {
    auto* st = ctx->st.get();
    if (st->nHaz == 0 || ctx->hazCap < sOff + nS || !repairsZeroDiagonals(ctx)) return;
    int n = st->n;
    double* bsnap = ctx->dHazBSnap.template as<double>() + (size_t) sOff * st->nHaz;
    hazSnapshotB<<<((st->nHaz * nS + BLOCK - 1) / BLOCK), BLOCK, 0, ctx->stream>>>(st->nHaz, nS, n, st->dHazRow.template as<int>(), B, bsnap);
    hazRepairRhs<<<((st->nHaz * nS + BLOCK - 1) / BLOCK), BLOCK, 0, ctx->stream>>>(st->nHaz, nS, n, st->dHazRow.template as<int>(),
            st->dHazDonor.template as<int>(), st->dHazDonorHaz.template as<int>(), st->dHazSign.template as<char>(),
            ctx->dHazChoice.template as<int>() + (size_t) sOff * st->nHaz, bsnap, B);
}

}  // namespace olfgpu
