// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// FULL Newton-Raphson AC power flow on the GPU, behind ONE JNI call. Java hands over
// the network data (per-family element parameters + the equation/variable indices the
// scatter tables resolve against), the initial state and the targets; the native side
// then keeps everything resident on the device:
//
//   - the fixed-pattern CSR Jacobian structure is built ONCE on the host from the
//     generated scatter tables (closed-branch, open-branch, shunt, HVDC AC emulation
//     + the slack/PV identity rows) — cuDSS analyses it once;
//   - each iteration, the four generated fused kernels (evalClosedBranch /
//     evalOpenBranch / evalShunt / evalHvdc) fill the CSR values AND the mismatch F
//     directly in device memory, reading the device state vector through the
//     precomputed index arrays (no host repack);
//   - cuDSS REFACTORIZATION + SOLVE produce dx on the device, a kernel applies
//     x += dx, and only the n-double mismatch crosses PCIe for the convergence check.
//
// Orientation: J is true CSR with row = equation column, col = variable row (OLF's
// indices), so cuDSS solves J dx = -F directly — no transposition games (contrast
// powsybl-math's CSC-as-CSR convention, which exists to serve solveTransposed).
//
// The host-built destination array folds the scatter ROLE in, so the kernels never
// index the constexpr scatter tables:  dest >= 0 -> J values slot;  dest == -1 ->
// skip;  dest <= -2 -> mismatch row (-2 - dest).
//
// Java: com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver
//   static native double[] solve(double[] x0, double[] target, int maxIter, double tol,
//       double[] cbIn, int[] cbIdx, double[] obIn, int[] obIdx,
//       double[] shIn, int[] shIdx, double[] hvIn, int[] hvIdx, int[] idIdx)
//     x0/target: length n (state by variable row / target by equation column)
//     cbIn nbr*8  (y g1 g2 b1 b2 ksi r1 a1)   cbIdx nbr*12 (P1 Q1 P2 Q2 ph1 v1 ph2 v2 a1col r1col qAlt1 qAlt2)
//     obIn nob*7  (y ksi g1 b1 g2 b2 r1)      obIdx nob*6 (P1 Q1 P2 Q2 v1 v2)
//     shIn nsh*2  (g b)                       shIdx nsh*3 (P Q v)
//     hvIn nhv*5  (p0 k lf1 lf2 r)            hvIdx nhv*4 (P1 P2 ph1 ph2)
//     idIdx nid*2 (eqRow varCol)              -1 = absent equation / variable
//     returns double[n + 2] = converged state, iterations, final ||F||inf
//
// This library links NVIDIA cuDSS DYNAMICALLY (never bundled, same licensing posture
// as powsybl-math-native's libmathcudss); cudart is linked statically.
#include <execinfo.h>
#include <jni.h>
#include <cuda_runtime.h>
#include <cudss.h>
#include <cublas_v2.h>

#include <algorithm>
#include <unordered_map>
#include <map>
#include <cstring>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <map>
#include <cstring>
#include <vector>

#include "closed_branch.cuh"
#include "closed_branch_scatter.h"
#include "open_branch.cuh"
#include "open_branch_scatter.h"
#include "shunt.cuh"
#include "shunt_scatter.h"
#include "hvdc.cuh"
#include "hvdc_scatter.h"
#include "vdl.cuh"

#include "common/loop_kinds.cuh"
#include "common/stats.cuh"

using namespace powsybl::openloadflow;

#include "common/check.cuh"
#include "common/launch.cuh"
#include "solver/linear_solver_settings.cuh"
#include "solver/batch_context.cuh"
#include "solver/driver_pass.cuh"
#include "solver/driver_steps.cuh"
#include "outerloop/reactive_capability_curves.cuh"
#include "outerloop/pilot_voltage.cuh"
#include "outerloop/remote_voltage_control.cuh"
#include "outerloop/shunt_voltage_control.cuh"
#include "outerloop/transformer_voltage_control.cuh"
#include "outerloop/incremental_shunt_voltage_control.cuh"
#include "outerloop/incremental_flow_control.cuh"
#include "outerloop/incremental_transformer_voltage_control.cuh"
#include "outerloop/secondary_voltage_control.cuh"

#include "structure/csr_pattern.cuh"

namespace olfgpu {

// ---------------------------------------------------------------------------------
// Device: the four family fill kernels + the loop helpers
// ---------------------------------------------------------------------------------

}  // namespace olfgpu
#include "equations/scatter.cuh"
namespace olfgpu {

__global__ void fillVdlBatch(int ne, int nScenarios, int n, int nnz,
                             const double* in, const int* ix, const int* dest,
                             const double* x, double* vals, double* F,
                             const int* altRow, const int* altDest, const int* mode);

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
                              const double* snap, double* vals, int* choice, int traceNoDonor) {
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
    if (traceNoDonor) {                                      // a zero diagonal nobody could repair: the
        printf("HAZ_NODONOR s=%d row=%d donors=%d\n", s, r,   // pivot cuDSS will perturb (device printf)
               hazDonorPtr[i + 1] - hazDonorPtr[i]);
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

// The zero-diagonal repair is for a factorization that cannot pivot. The robust mode (6) matches and pivots, which
// puts a nonzero on every diagonal by itself, so it factors the raw Jacobian; a borrowed row there only adds the
// cancellation the repair is prone to. A ZERO row that borrows a balance row which keeps its own diagonal forms,
// with that donor, a 2x2 principal block that is singular (the ZERO row has nothing on the donor's diagonal column).
// network.xiidm busbar ADNO re-couples three loop couplers of a hub, and two of their ZERO_V rows borrowed the hub's
// P balance: the pivoted factorization of that repaired matrix failed as badly as the unpivoted one (ratio 5e21),
// while the raw matrix converged to the CPU's answer.
static bool repairsZeroDiagonals(const BatchSolverContext* ctx);

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
            st->dDiagPos.template as<int>(), ctx->dHazSnap.template as<double>(), vals, ctx->dHazChoice.template as<int>(),
            std::getenv("OLF_GPU_HAZ_TRACE") != nullptr ? 1 : 0);
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


// A linear solve whose residual ratio ||J.dx-B|| / ||F|| exceeds this did NOT solve the system: the
// factorization is wrong (unpivoted LU on a structure that needs pivoting), not merely inaccurate.
static const double LIN_SOLVE_FAIL_RATIO = 1e-2;

// Per-scenario score of the linear solve just done: max|J.dx-B| under both sign conventions and
// max|F|, one thread per scenario (n serial reads each; a few thousand per scenario, once per solve).
__global__ void linResidScoreBatch(int n, int nS, const double* rp, const double* rm, const double* F,
                                   double* score) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nS) return;
    double mp = 0.0;
    double mm = 0.0;
    double mf = 0.0;
    const double* rps = rp + (size_t) s * n;
    const double* rms = rm + (size_t) s * n;
    const double* fs = F + (size_t) s * n;
    for (int i = 0; i < n; i++) {
        mp = fmax(mp, fabs(rps[i]));
        mm = fmax(mm, fabs(rms[i]));
        mf = fmax(mf, fabs(fs[i]));
    }
    score[3 * s] = fmin(mp, mm);
    score[3 * s + 1] = mf;
    score[3 * s + 2] = (mf > 0.0 && isfinite(mp) && isfinite(mm)) ? fmin(mp, mm) / mf : 1e300;
}

// How accurately did the LINEAR solve actually solve? Per row, r_i = (J.dx)_i -/+ F_i, per scenario: every
// scenario has its OWN values block (stride nnz) and its own dx / F block (stride n), over the SHARED sparsity
// pattern. ||r|| at round-off relative to ||F|| means the solve is exact; a large ||r|| is what iterative
// refinement corrects (this is also refinement's own first step).
__global__ void linResidRowBatch(int n, int nS, int nnz, const int* rowPtr, const int* colIdx,
                                 const double* vals, const double* dx, const double* F,
                                 double* rPlus, double* rMinus) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * nS) return;
    int s = t / n;
    int i = t - s * n;
    const double* v = vals + (size_t) s * nnz;
    const double* d = dx + (size_t) s * n;
    double acc = 0.0;
    for (int k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
        acc += v[k] * d[colIdx[k]];
    }
    rPlus[t] = acc - F[t];
    rMinus[t] = acc + F[t];
}

// Per-scenario masked backtrack step: x = xPrev + mu[s]*dx for scenarios with mu[s] >= 0;
// mu[s] < 0 leaves the scenario's state untouched (not backtracking this pass).
__global__ void stepFromPrevBatchMasked(int n, int total, const double* xPrev, const double* dx,
                                        const double* mu, double* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        double m = mu[i / n];
        if (m >= 0.0) x[i] = xPrev[i] + m * dx[i];
    }
}

}  // namespace olfgpu

#include "common/device_buf.cuh"
#include "structure/shared_structure.cuh"

namespace olfgpu {

// The GROUP relocation tables of one family, or all-null when the structure has none (plan-A6).
static RelocTables relocOf(Family& f, SharedStructure* st) {
    RelocTables rt;
    if (f.relocPool.p != nullptr) {
        rt.group = f.relocGroup.as<int>();
        rt.off = f.relocOff.as<int>();
        rt.pool = f.relocPool.as<int>();
        rt.dep = f.relocDep.p != nullptr ? f.relocDep.as<char>() : nullptr;
        rt.qrPtr = st->dQrPtr.as<int>();
        rt.qrRows = st->dQrRows.as<int>();
        rt.relocOwn = st->dRelocOwn.p != nullptr ? st->dRelocOwn.as<int>() : nullptr;
    }
    return rt;
}

// The zi family's relocation tables, or all-null when it has none.
// Attaches the PILOT-SELF gate to a structure-built RelocTables. Inert (and identical to before) on any
// context without such a group, so every existing candidate-table group keeps its exact behavior.
RelocTables withPilotSelf(RelocTables rt, BatchSolverContext* ctx) {
    if (ctx->hasPilotSelf && ctx->dPsSkip.p != nullptr) {
        rt.psSkip = ctx->dPsSkip.template as<char>();
        rt.psGroups = ctx->st->nQr;
    }
    if (ctx->hasDqSub) {                                     // tasks #71: per-scenario subgroups of a split group
        rt.dqSub = ctx->dDqSub.template as<char>();
        rt.qrMemberStart = ctx->dQrMemberStart.template as<int>();
        rt.nDqMembers = ctx->nDqMembers;
    }
    return rt;
}

static RelocTables ziRelocOf(SharedStructure* st) {
    RelocTables rt;
    if (st->dZiRelocPool.p != nullptr) {
        rt.group = st->dZiRelocGroup.as<int>();
        rt.off = st->dZiRelocOff.as<int>();
        rt.pool = st->dZiRelocPool.as<int>();
        rt.dep = st->dZiRelocDep.p != nullptr ? st->dZiRelocDep.as<char>() : nullptr;
        rt.qrPtr = st->dQrPtr.as<int>();
        rt.qrRows = st->dQrRows.as<int>();
        rt.relocOwn = st->dRelocOwn.p != nullptr ? st->dRelocOwn.as<int>() : nullptr;
    }
    return rt;
}

// =================================================================================
// BATCHED N-1 security analysis: many post-contingency scenarios on the SAME network
// structure, solved in one batched Newton-Raphson. The CSR pattern, the family index
// packs, the destination arrays and cuDSS's SYMBOLIC ANALYSIS are SHARED across all
// scenarios (built once); only the value/state/mismatch arrays gain a scenario
// dimension. Per-scenario element disablement (a branch outage) is a device mask the
// fill kernels honor (a disabled element simply does not scatter — its entries stay
// structurally present but numerically zero, and the diagonal-pairing guarantee keeps
// every row factorizable). Converged scenarios retire from the batch (an active mask),
// so the per-iteration cost shrinks as scenarios finish.
//
// cuDSS (OLF_GPU_BATCHED_CUDSS, default on for cuDSS >= 0.3.0): the UNIFORM BATCHED API —
// one batch CSR matrix whose scenarios share the rowPtr/colIdx pattern with per-scenario
// value sets, batched dense B/X. ANALYSIS runs ONCE on the shared pattern; each Newton
// iteration is ONE batched FACTORIZATION + ONE batched SOLVE over all scenarios. Retired
// scenarios (converged/diverged) are frozen to an identity system so a singular sub-block
// never fails the batch. On older cuDSS the fallback path repoints a single matrix and
// refactorizes+solves per scenario. Either way the symbolic analysis — the dominant
// per-case cost in sequential SA — is amortized over the entire contingency set.
// =================================================================================

// Batched fill kernels: thread = element x scenario. Skip if the scenario has retired
// (converged) or the element is outaged in that scenario. Values/mismatch are per-scenario.
}  // namespace olfgpu

#include "equations/scenario_overrides.cuh"

namespace olfgpu {

__global__ void fillClosedBranchBatch(int ne, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                                      const double* in, const int* ix, const int* dest,
                                      const double* x, double* vals, double* F,
                                      const char* disabled, const char* active,
                                      int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal,
                                      const int* altDest, const int* selRow, const int* mode, int dynReloc,
                                      const int* pcDest, const int* pcSel, RelocTables rt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ne * nScenarios) return;
    int e = tid % ne;
    int s = tid / ne;
    if (!active[s] || disabled[s * nElements + elemOffset + e]) return;
    const double* p = in + e * 8;                            // y g1 g2 b1 b2 ksi r1 a1
    const int* q = ix + e * 12;                              // P1 Q1 P2 Q2 ph1 v1 ph2 v2 a1col r1col
    double pOv[8];
    if (findTapOverride(nR1Ov, r1OvSlotElem, r1OvVal, s, e, pOv)) {
        p = pOv;
    }
    const double* xs = x + s * n;
    double v1 = xs[q[5]], v2 = xs[q[7]], ph1 = xs[q[4]], ph2 = xs[q[6]];
    double r1 = q[9] >= 0 ? xs[q[9]] : p[6];
    double a1 = q[8] >= 0 ? xs[q[8]] : p[7];
    // OLF's own form (AcNetworkVector takes sin and cos of theta1 / theta2 directly). Every closed-branch fill -
    // this one, the F-only fill and the bus-balance fill - goes through the same generated assembler, so their
    // residuals agree bit for bit. The F-only fill used to derive the angles by angle addition from per-branch
    // statics: faster, but another rounding than OLF's.
    double o[CB_NOUT];
    evalClosedBranch(p[0], p[1], p[2], p[3], p[4], v1, r1, v2, p[5], a1, ph1, ph2, 0,
                     o + 0, o + 1, o + 2, o + 3, o + 4, o + 5, o + 6, o + 7, o + 8, o + 9,
                     o + 10, o + 11, o + 12, o + 13, o + 14, o + 15, o + 16, o + 17, o + 18,
                     o + 19, o + 20, o + 21, o + 22, o + 23, o + 24, o + 25, o + 26, o + 27);
    if (dynReloc) {
        scatterAddDyn(CB_NOUT, dest + e * CB_NOUT, altDest + e * CB_NOUT, selRow + e * CB_NOUT,
                      mode + (size_t) s * n, o, vals + (size_t) s * nnz, F + (size_t) s * n,
                      relocForScenario(rt, s), e * CB_NOUT);
    } else {
        scatterAdd(CB_NOUT, dest + e * CB_NOUT, o, vals + (size_t) s * nnz, F + (size_t) s * n);
    }
    if (pcDest != nullptr) {                                 // continuous phase control: emit the controlled
        const int* pd = pcDest + e * CB_NOUT;                // side's P outputs AGAIN at the a1 row when this
        const int* psl = pcSel + e * CB_NOUT;                // scenario has the control enabled (row mode 3)
        for (int k = 0; k < CB_NOUT; k++) {
            int d = pd[k];
            if (d == -1 || mode[(size_t) s * n + psl[k]] != 3) {
                continue;
            }
            if (d >= 0) {
                atomicAdd(vals + (size_t) s * nnz + d, o[k]);
            } else {
                atomicAdd(F + (size_t) s * n + (-2 - d), o[k]);
            }
        }
    }
}

// The fused fill's residual half alone, for the reactive-limit / slack Q buffer: evalClosedBranchF evaluates
// p1 q1 p2 q2 from the SAME generated expressions and raw-angle trig as evalClosedBranch (bit-identical), with
// the frozen routing (dynReloc 0, no phase-control re-emission) and no Jacobian at all - neither its 24
// derivatives nor their atomics.
__global__ void fillClosedBranchQbufBatch(int ne, int nScenarios, int n, int nElements, int elemOffset,
                                          const double* in, const int* ix, const int* dest,
                                          const double* x, double* F, const char* disabled, const char* active,
                                          int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ne * nScenarios) return;
    int e = tid % ne;
    int s = tid / ne;
    if (!active[s] || disabled[s * nElements + elemOffset + e]) return;
    const double* p = in + e * 8;                            // y g1 g2 b1 b2 ksi r1 a1
    const int* q = ix + e * 12;
    double pOv[8];
    if (findTapOverride(nR1Ov, r1OvSlotElem, r1OvVal, s, e, pOv)) {
        p = pOv;
    }
    const double* xs = x + s * n;
    double v1 = xs[q[5]], v2 = xs[q[7]], ph1 = xs[q[4]], ph2 = xs[q[6]];
    double r1 = q[9] >= 0 ? xs[q[9]] : p[6];
    double a1 = q[8] >= 0 ? xs[q[8]] : p[7];
    double o[4];                                             // p1 q1 p2 q2: evalClosedBranch's own expressions
    evalClosedBranchF(p[0], p[1], p[2], p[3], p[4], v1, r1, v2, p[5], a1, ph1, ph2, 0, o + 0, o + 1, o + 2, o + 3);
    const int pos[4] = {0, 7, 14, 21};                       // their output slots (static_assert-pinned below)
    const int* d = dest + e * CB_NOUT;
    for (int k = 0; k < 4; k++) {
        if (d[pos[k]] <= -2) {
            atomicAdd(F + (size_t) s * n + (-2 - d[pos[k]]), o[k]);
        }
    }
}

__global__ void fillOpenBranchBatch(int ne, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                                    const double* in, const int* ix, const int* dest,
                                    const double* x, double* vals, double* F,
                                    const char* disabled, const char* active,
                                    const int* altDest, const int* selRow, const int* mode, int dynReloc,
                                    RelocTables rt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ne * nScenarios) return;
    int e = tid % ne;
    int s = tid / ne;
    if (!active[s] || disabled[s * nElements + elemOffset + e]) return;
    const double* p = in + e * 7;                            // y ksi g1 b1 g2 b2 r1
    const int* q = ix + e * 6;
    const double* xs = x + s * n;
    double v1 = q[4] >= 0 ? xs[q[4]] : 1.0;
    double v2 = q[5] >= 0 ? xs[q[5]] : 1.0;
    double o[OB_NOUT];
    evalOpenBranch(p[0], p[1], p[2], p[3], p[4], p[5], v1, p[6], v2, 0,
                   o + 0, o + 1, o + 2, o + 3, o + 4, o + 5, o + 6, o + 7);
    if (dynReloc) {
        scatterAddDyn(OB_NOUT, dest + e * OB_NOUT, altDest + e * OB_NOUT, selRow + e * OB_NOUT,
                      mode + (size_t) s * n, o, vals + (size_t) s * nnz, F + (size_t) s * n,
                      relocForScenario(rt, s), e * OB_NOUT);
    } else {
        scatterAdd(OB_NOUT, dest + e * OB_NOUT, o, vals + (size_t) s * nnz, F + (size_t) s * n);
    }
}

__global__ void fillShuntBatch(int ne, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                               const double* in, const int* ix, const int* dest,
                               const double* x, double* vals, double* F,
                               const char* disabled, const char* active,
                               int nShOv, const int* shOvSlotElem, const double* shOvGb,
                               const int* altDest, const int* selRow, const int* mode, int dynReloc,
                               RelocTables rt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ne * nScenarios) return;
    int e = tid % ne;
    int s = tid / ne;
    if (!active[s] || disabled[s * nElements + elemOffset + e]) return;
    const double* p = in + e * 2;                            // g b
    double g = p[0];
    double b = p[1];
    applyShuntOverride(nShOv, shOvSlotElem, shOvGb, s, e, g, b);
    // A voltage-controlling shunt (bcol >= 0) reads its susceptance from the per-scenario state vector; a
    // fixed shunt keeps the packed value (+ any contingency shunt-shift override).
    int bcol = ix[e * 5 + 3];
    if (bcol >= 0) b = x[(size_t) s * n + bcol];
    double v = x[(size_t) s * n + ix[e * 5 + 2]];
    double o[SH_NOUT];
    evalShunt(v, g, b, 0, o + 0, o + 1, o + 2, o + 3, o + 4);
    if (dynReloc) {
        scatterAddDyn(SH_NOUT, dest + e * SH_NOUT, altDest + e * SH_NOUT, selRow + e * SH_NOUT,
                      mode + (size_t) s * n, o, vals + (size_t) s * nnz, F + (size_t) s * n,
                      relocForScenario(rt, s), e * SH_NOUT);
    } else {
        scatterAdd(SH_NOUT, dest + e * SH_NOUT, o, vals + (size_t) s * nnz, F + (size_t) s * n);
    }
}

__global__ void fillHvdcBatch(int ne, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                              const double* in, const int* ix, const int* dest,
                              const double* x, double* vals, double* F,
                              const char* disabled, const char* active,
                              const int* altDest, const int* selRow, const int* mode, int dynReloc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ne * nScenarios) return;
    int e = tid % ne;
    int s = tid / ne;
    if (!active[s] || disabled[s * nElements + elemOffset + e]) return;
    const double* p = in + e * 5;                            // p0 k lf1 lf2 r
    const int* q = ix + e * 4;
    const double* xs = x + (size_t) s * n;
    double ph1 = xs[q[2]], ph2 = xs[q[3]];
    double o[HVDC_NOUT];
    evalHvdc(p[0], p[1], p[2], p[3], p[4], ph1, ph2, 0, o + 0, o + 1, o + 2, o + 3, o + 4, o + 5);
    if (dynReloc) {
        scatterAddDyn(HVDC_NOUT, dest + e * HVDC_NOUT, altDest + e * HVDC_NOUT, selRow + e * HVDC_NOUT,
                      mode + (size_t) s * n, o, vals + (size_t) s * nnz, F + (size_t) s * n, RelocTables{}, e * HVDC_NOUT);
    } else {
        scatterAdd(HVDC_NOUT, dest + e * HVDC_NOUT, o, vals + (size_t) s * nnz, F + (size_t) s * n);
    }
}

// Batched DISTR_Q fill: one thread per (contribution, scenario). Each contribution is a closed
// branch's side-s reactive flow (value + the 5 derivatives), scaled by the controller weight; x/vals/F are
// strided per scenario, and the kernel DROPS a contribution whose underlying closed (kind 0) / open (kind 2)
// branch is outaged in this scenario — matching the CPU, which removes that branch's q from the controller's
// reactive sum (the controller weight qPercent is unchanged, so only the term vanishes). kinds 1 (shunt) and
// 3 (zero-imp dummy) cannot be a single-branch contingency in a qualified batched scenario, so no mask check.
// ---------------------------------------------------------------------------------
// F-ONLY batch fills: evaluate just the residual values (the generated eval<X>F
// routines) and scatter only the F rows — for iterations that never consume the
// Jacobian (the final residual/convergence fill, and mode 4's solve-only
// iterations where the factors are frozen at iter 0). The residual positions in
// each family's dest stride are pinned to the generated scatter tables by
// static_asserts, so a codegen output-order change fails the build loudly.
// ---------------------------------------------------------------------------------

static_assert(CB_SCATTER[0].role == 0 && CB_SCATTER[7].role == 0
           && CB_SCATTER[14].role == 0 && CB_SCATTER[21].role == 0,
              "closed-branch residual outputs moved — update fillClosedBranchBatchF");
static_assert(OB_SCATTER[0].role == 0 && OB_SCATTER[2].role == 0
           && OB_SCATTER[4].role == 0 && OB_SCATTER[6].role == 0,
              "open-branch residual outputs moved — update fillOpenBranchBatchF");
static_assert(SH_SCATTER[0].role == 0 && SH_SCATTER[2].role == 0,
              "shunt residual outputs moved — update fillShuntBatchF");
static_assert(HVDC_SCATTER[0].role == 0 && HVDC_SCATTER[3].role == 0,
              "hvdc residual outputs moved — update fillHvdcBatchF");

__global__ void fillClosedBranchBatchF(int ne, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                                       const double* in, const int* ix, const int* dest,
                                       const double* x, double* F,
                                       const char* disabled, const char* active,
                                       int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal,
                                       const int* altDest, const int* selRow, const int* mode, int dynReloc,
                                       const int* pcDest, const int* pcSel, RelocTables rt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ne * nScenarios) return;
    int e = tid % ne;
    int s = tid / ne;
    if (!active[s] || disabled[s * nElements + elemOffset + e]) return;
    const double* p = in + e * 8;                            // y g1 g2 b1 b2 ksi r1 a1
    const int* q = ix + e * 12;                              // P1 Q1 P2 Q2 ph1 v1 ph2 v2 a1col r1col
    double pOv[8];
    bool tapOv = findTapOverride(nR1Ov, r1OvSlotElem, r1OvVal, s, e, pOv);
    if (tapOv) {
        p = pOv;
    }
    const double* xs = x + s * n;
    double v1 = xs[q[5]], v2 = xs[q[7]], ph1 = xs[q[4]], ph2 = xs[q[6]];
    double r1 = q[9] >= 0 ? xs[q[9]] : p[6];
    double a1 = q[8] >= 0 ? xs[q[8]] : p[7];
    double o[4];                                             // p1 q1 p2 q2: the fused fill's own expressions
    evalClosedBranchF(p[0], p[1], p[2], p[3], p[4], v1, r1, v2, p[5], a1, ph1, ph2, 0, o + 0, o + 1, o + 2, o + 3);
    const int pos[4] = {0, 7, 14, 21};
    if (dynReloc) {
        scatterAddFDyn(4, dest + e * CB_NOUT, altDest + e * CB_NOUT, selRow + e * CB_NOUT,
                       pos, mode + (size_t) s * n, o, F + (size_t) s * n, rt, e * CB_NOUT);
    } else {
        scatterAddF(4, dest + e * CB_NOUT, pos, o, F + (size_t) s * n);
    }
    if (pcDest != nullptr) {                                 // continuous phase control: the P residual at a1
        for (int j = 0; j < 4; j += 2) {                     // outputs 0 (p1) / 2 (p2) at table pos 0 / 14
            int d = pcDest[e * CB_NOUT + pos[j]];
            if (d < -1 && mode[(size_t) s * n + pcSel[e * CB_NOUT + pos[j]]] == 3) {
                atomicAdd(F + (size_t) s * n + (-2 - d), o[j]);
            }
        }
    }
}

__global__ void fillOpenBranchBatchF(int ne, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                                     const double* in, const int* ix, const int* dest,
                                     const double* x, double* F,
                                     const char* disabled, const char* active,
                                     const int* altDest, const int* selRow, const int* mode, int dynReloc,
                                     RelocTables rt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ne * nScenarios) return;
    int e = tid % ne;
    int s = tid / ne;
    if (!active[s] || disabled[s * nElements + elemOffset + e]) return;
    const double* p = in + e * 7;                            // y ksi g1 b1 g2 b2 r1
    const int* q = ix + e * 6;
    const double* xs = x + s * n;
    double v1 = q[4] >= 0 ? xs[q[4]] : 1.0;
    double v2 = q[5] >= 0 ? xs[q[5]] : 1.0;
    double o[4];
    evalOpenBranchF(p[0], p[1], p[2], p[3], p[4], p[5], v1, p[6], v2, 0, o + 0, o + 1, o + 2, o + 3);
    const int pos[4] = {0, 2, 4, 6};
    if (dynReloc) {
        scatterAddFDyn(4, dest + e * OB_NOUT, altDest + e * OB_NOUT, selRow + e * OB_NOUT,
                       pos, mode + (size_t) s * n, o, F + (size_t) s * n, rt, e * OB_NOUT);
    } else {
        scatterAddF(4, dest + e * OB_NOUT, pos, o, F + (size_t) s * n);
    }
}

// Voltage-dependent load model terms (OLF AbstractLoadModelEquationTerm): per entry e,
// F[row] += coef*v^n and J[(row, vOwn)] += coef*n*v^(n-1), coef = c_n * load target (pu),
// exactly OLF's f()/dfdv() (the n = 0 constant part lives in the target vector, not here).
// Identity rows are wiped by applyRowModeBatch afterwards, so a PV bus's Q-row term vanishes
// exactly like every other Q fill (and reappears when RL flips the bus PQ). vals == nullptr
// makes it the residual-only (line-search) variant.
__global__ void fillVdlBatch(int ne, int nScenarios, int n, int nnz,
                             const double* in, const int* ix, const int* dest,
                             const double* x, double* vals, double* F,
                             const int* altRow, const int* altDest, const int* mode) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ne * nScenarios) return;
    int e = t % ne;
    int s = t / ne;
    // GENERATED evalVdl (codegen family "vdl", parsed from OLF's
    // AbstractLoadModelEquationTerm.f/dfdv): one packed element per exp term, the sum over a
    // load model's terms IS the per-entry atomicAdd — exactly the Java accumulation loop.
    double c = in[e * 3];
    double expn = in[e * 3 + 1];
    double target = in[e * 3 + 2];
    double v = x[(size_t) s * n + ix[e * 2 + 1]];
    double o[2];
    evalVdl(v, target, c, expn, 0, o + 0, o + 1);
    int row = ix[e * 2];
    int slot = dest[e];
    // A load on a relocated coupler-tree child: its term sits on the parent coupler's dummy row, and goes HOME
    // (the child's own row and its reserved slot) in a lane whose dummy row is an identity - the coupler is
    // open or out of the tree there. Mode 4 on that row is the child's pinned balance, which stays.
    if (mode != nullptr && altRow != nullptr && altRow[e] >= 0 && mode[(size_t) s * n + row] == 0) {
        row = altRow[e];
        slot = altDest[e];
    }
    atomicAdd(&F[(size_t) s * n + row], o[0]);
    if (vals != nullptr) {
        atomicAdd(&vals[(size_t) s * nnz + slot], o[1]);
    }
}

__global__ void fillShuntBatchF(int ne, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                                const double* in, const int* ix, const int* dest,
                                const double* x, double* F,
                                const char* disabled, const char* active,
                                int nShOv, const int* shOvSlotElem, const double* shOvGb,
                                const int* altDest, const int* selRow, const int* mode, int dynReloc,
                               RelocTables rt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ne * nScenarios) return;
    int e = tid % ne;
    int s = tid / ne;
    if (!active[s] || disabled[s * nElements + elemOffset + e]) return;
    const double* p = in + e * 2;                            // g b
    double g = p[0];
    double b = p[1];
    applyShuntOverride(nShOv, shOvSlotElem, shOvGb, s, e, g, b);
    int bcol = ix[e * 5 + 3];
    if (bcol >= 0) b = x[(size_t) s * n + bcol];             // voltage-controlling shunt: b from state
    double v = x[(size_t) s * n + ix[e * 5 + 2]];
    double o[2];
    evalShuntF(v, g, b, 0, o + 0, o + 1);
    const int pos[2] = {0, 2};
    if (dynReloc) {
        scatterAddFDyn(2, dest + e * SH_NOUT, altDest + e * SH_NOUT, selRow + e * SH_NOUT,
                       pos, mode + (size_t) s * n, o, F + (size_t) s * n, rt, e * SH_NOUT);
    } else {
        scatterAddF(2, dest + e * SH_NOUT, pos, o, F + (size_t) s * n);
    }
}

__global__ void fillHvdcBatchF(int ne, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                               const double* in, const int* ix, const int* dest,
                               const double* x, double* F,
                               const char* disabled, const char* active,
                               const int* altDest, const int* selRow, const int* mode, int dynReloc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ne * nScenarios) return;
    int e = tid % ne;
    int s = tid / ne;
    if (!active[s] || disabled[s * nElements + elemOffset + e]) return;
    const double* p = in + e * 5;                            // p0 k lf1 lf2 r
    const int* q = ix + e * 4;
    const double* xs = x + (size_t) s * n;
    double ph1 = xs[q[2]], ph2 = xs[q[3]];
    double o[2];
    evalHvdcF(p[0], p[1], p[2], p[3], p[4], ph1, ph2, 0, o + 0, o + 1);
    const int pos[2] = {0, 3};
    if (dynReloc) {
        scatterAddFDyn(2, dest + e * HVDC_NOUT, altDest + e * HVDC_NOUT, selRow + e * HVDC_NOUT,
                       pos, mode + (size_t) s * n, o, F + (size_t) s * n, RelocTables{}, e * HVDC_NOUT);
    } else {
        scatterAddF(2, dest + e * HVDC_NOUT, pos, o, F + (size_t) s * n);
    }
}

// A DISTR_Q controller that stops controlling in ONE scenario (its bus lost its last voltage
// generator while the group's other controllers survive). OLF's updateReactiveKeys zeroes its key
// and renormalizes over the rest — the same treatment a PINNED member gets, except its target
// contribution stays its own targetQ instead of a reactive limit.
static int gpuDqTraceRow() {
    // OLF_GPU_DQ_TRACE=<deviceRow>: every DISTR_Q residual contribution landing in that row.
    static const char* e = std::getenv("OLF_GPU_DQ_TRACE");
    static int v = e != nullptr ? std::atoi(e) : -1;
    return v;
}

__global__ void fillDistrQBatch(int ndq, int nScenarios, int n, int nnz, int nElements,
                                int cbOffset, int obOffset,
                                const int* dqElem, const int* dqSide, const int* dqKind,
                                const double* dqWeight, const int* dqDest,
                                const int* dqAlt, const int* dqSel, const int* dqZiRow,
                                const double* cbIn, const int* cbIx, const double* shIn, const int* shIx,
                                const double* obIn, const int* obIx,
                                const double* x, double* vals, double* F,
                                const char* disabled, const char* active,
                                int nDqOv, const int* dqOvIdx, const double* dqOvW,
                                int nDqKeyOv, const int* dqKeyOvIdx, const double* dqKeyOvKey,
                                const int* mode,
                                int nDqInact, const int* dqInactSlotMember,
                                const int* dqKMember, const int* dqJMember, const double* mKey,
                                const int* mRlEntry, const int* mGroupStart, const int* mGroupCount,
                                const char* limitType, int nrl,
                                const char* psSkip, int psGroups,
                                const int* mQrGroup, const int* qrPtr, const int* qrRows,
                                int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal,
                                int nShOv, const int* shOvSlotElem, const double* shOvGb,
                                int dqTraceRow,
                                const char* dqSub, const int* qrMemberStart, int nDqMembersAll) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ndq * nScenarios) return;
    int c = tid % ndq;
    int s = tid / ndq;
    if (!active[s]) return;
    // STOCK PLACEMENT: a relocated child's sharing row moves to the bus's OWN row in any scenario that
    // kills its parent coupler, which pins that coupler's DUMMY_Q row to an identity (see buildDistrQDest).
    if (dqSel != nullptr && dqSel[c] >= 0 && dqAlt != nullptr
            && mode[(size_t) s * n + dqSel[c]] == 0 && dqAlt[c * 6] <= -2) {
        dqDest = dqAlt;
    }
    // an RL-PINNED remote member's row (mode 4) hosts its OWN Q balance — its DISTR_Q emissions
    // stop for this scenario (the CPU deactivates only that member's DISTR_Q equation)
    {
        int d0 = dqDest[c * 6];
        if (d0 <= -2 && mode[(size_t) s * n + (-2 - d0)] == 4) return;
        // plan-B4: the CURRENT pilot-Q carrier's row (a re-picked free role after the base free
        // member pinned) hosts the group's relocated pilot balance — its DISTR_Q row is inactive
        // this scenario, exactly like OLF deactivating the new free member's DISTR_Q equation.
        if (mQrGroup != nullptr && dqJMember[c] >= 0) {
            int g = mQrGroup[dqJMember[c]];
            // A PILOT-SELF group carries no relocated balance while its pilot is unpinned (its q feeds these
            // very DISTR_Q rows, weighted), so the carrier's row is NOT hosting anything and its DISTR_Q must
            // stay ACTIVE. Suppressing it there dropped an equation without gaining one: one short, singular.
            bool psGateOff = psSkip != nullptr && g >= 0 && psSkip[(size_t) s * psGroups + g];
            if (g >= 0 && !psGateOff && d0 <= -2
                    && -2 - d0 == qrRows[qrPtr[g] + relocChooseSub(g, qrPtr, qrRows, mode + (size_t) s * n,
                            dqSub != nullptr ? dqSub + (size_t) s * nDqMembersAll : nullptr, qrMemberStart)]) {
                return;
            }
        }
    }
    int e = dqElem[c];
    int kind = dqKind[c];
    if (kind == 0 && disabled[s * nElements + cbOffset + e]) return;   // outaged closed contributor
    if (kind == 2 && disabled[s * nElements + obOffset + e]) return;   // outaged open contributor
    // kind 3 is a COUPLER's DUMMY_Q as part of this controller's reactive injection, and its element index
    // IS that dummy column (= its row). A scenario that kills the coupler pins that row to an identity, and
    // the coupler then carries no reactive flow at all — the contribution has to go with it. Measured on the
    // middle-section-loss fixtures: the surviving section's DISTR_Q row kept a coefficient 1 on the dead
    // coupler's DUMMY_Q and the solve landed 0.34 kV off (bus vl3_b 401.35 against OLF's 401.69).
    if (kind == 3 && dqZiRow != nullptr && dqZiRow[c] >= 0 && mode[(size_t) s * n + dqZiRow[c]] == 0) return;
    double w = dqWeight[c];
    int kM = dqKMember != nullptr ? dqKMember[c] : -1;
    if (kM >= 0) {
        // DYNAMIC reactive keys (plan-B3): an RL pin disables the member's controller, which zeroes
        // its key and renormalizes qPct over the survivors (updateReactiveKeys). Replicate OLF's
        // zero-then-sum arithmetic in member order, driven by the per-(entry, scenario) limitType.
        int gs = mGroupStart[kM];
        int gc = mGroupCount[kM];
        // SUBGROUPS (tasks #71): a split group's members pair only within their own subgroup — the other
        // subgroup's keys leave this denominator and its members' q terms leave this row (below).
        const char* subS = dqSub != nullptr ? dqSub + (size_t) s * nDqMembersAll : nullptr;
        char subK = subS != nullptr ? subS[kM] : 0;
        if (subS != nullptr && dqJMember[c] >= 0 && subS[dqJMember[c]] != subK) return;
        double sum = 0.0;
        for (int m = gs; m < gs + gc; m++) {
            int re = mRlEntry[m];
            bool pinned = re >= 0 && limitType[(size_t) s * nrl + re] != 0;
            bool otherSub = subS != nullptr && subS[m] != subK;
            sum += (pinned || otherSub || dqMemberInactive(nDqInact, dqInactSlotMember, s, m))
                    ? 0.0 : dqKeyOf(nDqKeyOv, dqKeyOvIdx, dqKeyOvKey, s, m, mKey);
        }
        // the PINNED member's q terms STAY in every survivor's row (weight = the renormalized
        // qPct of the ROW OWNER); the pin is balanced on the TARGET side instead — OLF's
        // getGeneratorReactivePowerDistributionTarget adds qPct_k * targetQ_j for each frozen j
        // (updateDistrQTargetsBatch keeps the per-scenario targets in step with the pin states)
        int reK = mRlEntry[kM];
        bool kPinned = reK >= 0 && limitType[(size_t) s * nrl + reK] != 0;
        double qpk = (sum == 0.0 || kPinned || dqMemberInactive(nDqInact, dqInactSlotMember, s, kM))
                ? 0.0 : dqKeyOf(nDqKeyOv, dqKeyOvIdx, dqKeyOvKey, s, kM, mKey) / sum;
        w = qpk - (dqJMember[c] == kM ? 1.0 : 0.0);
        if (kind == 3 && dqSide[c] != 0) {
            // The host folds the SIDE SIGN into a kind-3 weight (+dummyQ where the controller is
            // bus1, -dummyQ where it is bus2 — exactly createReactiveTerms' term); this live
            // recomputation rebuilds qPct_k - delta_jk and must put that sign back. Without it a
            // coupler whose BOTH ends are controllers of the same group cancels to zero at
            // qPct = 0.5: measured on rte6515 BUS-1014_COUPLER_1, where DUMMY_Q then appeared in NO
            // equation at all and the Jacobian had a structurally zero column, against OLF's
            // coefficient of +1.0 in DISTR_Q of VL-1014_1.
            w = -w;
        }
    }
    for (int k = 0; k < nDqOv; k++) {                        // per-scenario qPct renormalization (gen outage)
        if (dqOvIdx[k * 2] == s && dqOvIdx[k * 2 + 1] == c) {
            w = dqOvW[k];
        }
    }
    const int* dst = dqDest + c * 6;
    const double* xs = x + (size_t) s * n;
    double* valsS = vals + (size_t) s * nnz;
    double* FS = F + (size_t) s * n;

    if (kind == 1) {                             // shunt q: {q -> F, dq/dv -> J(row, vcol)}
        const double* sp = shIn + e * 2;
        int vcol = shIx[e * 5 + 2];
        double g = sp[0];
        double b = sp[1];
        applyShuntOverride(nShOv, shOvSlotElem, shOvGb, s, e, g, b);
        double so[SH_NOUT];
        evalShunt(xs[vcol], g, b, 0, so + 0, so + 1, so + 2, so + 3, so + 4);
        if (dst[0] <= -2 && (-2 - dst[0]) == dqTraceRow) {
            printf("DQF kind=1(shunt) elem=%d w=%.6f contrib=%.9f\n", e, w, w * so[2]);
        }
        if (dst[0] <= -2) atomicAdd(&FS[-2 - dst[0]], w * so[2]);
        if (dst[1] >= 0) atomicAdd(&valsS[dst[1]], w * so[3]);
        return;
    }
    if (kind == 3) {                             // zero-impedance DUMMY_Q: linear w*dummyQ
        if (dst[0] <= -2 && (-2 - dst[0]) == dqTraceRow) {
            printf("DQF kind=3(ziDummyQ) col=%d w=%.6f contrib=%.9f\n", e, w, w * xs[e]);
        }
        if (dst[0] <= -2) atomicAdd(&FS[-2 - dst[0]], w * xs[e]);
        if (dst[1] >= 0) atomicAdd(&valsS[dst[1]], w);
        return;
    }
    if (kind == 2) {                             // half-open branch: connected-side q + dq/dv
        const double* op = obIn + e * 7;
        const int* oq = obIx + e * 6;
        int vcol = dqSide[c] == 0 ? oq[4] : oq[5];
        double v1 = dqSide[c] == 0 ? xs[vcol] : 1.0;
        double v2 = dqSide[c] == 1 ? xs[vcol] : 1.0;
        double oo[OB_NOUT];
        evalOpenBranch(op[0], op[1], op[2], op[3], op[4], op[5], v1, op[6], v2, 0,
                       oo + 0, oo + 1, oo + 2, oo + 3, oo + 4, oo + 5, oo + 6, oo + 7);
        int qi = dqSide[c] == 0 ? 6 : 2;
        if (dst[0] <= -2 && (-2 - dst[0]) == dqTraceRow) {
            printf("DQF kind=2(halfOpen) elem=%d w=%.6f contrib=%.9f\n", e, w, w * oo[qi]);
        }
        if (dst[0] <= -2) atomicAdd(&FS[-2 - dst[0]], w * oo[qi]);
        if (dst[1] >= 0) atomicAdd(&valsS[dst[1]], w * oo[qi + 1]);
        return;
    }
    // kind 0: closed-branch q1/q2 + its derivatives.
    // The per-scenario pi override MUST be honoured here, exactly as fillRemoteMemberQBatch does.
    // It is not a tap-only channel: an outaged branch on this path is nulled by zeroing its
    // admittance through it, NOT by the disabled mask. Reading the base pi instead kept three
    // transformers into an outaged busbar live against a far bus frozen at v=1.0 and injected
    // +0.813908 of phantom reactive into one DISTR_Q row (rte6515 BUS-1265_BBS, measured residual
    // -0.814738) - which moved the sharing solution 2.1 kV and then tripped a spurious RL pin.
    const double* p0 = cbIn + e * 8;
    double p[8];
    for (int i = 0; i < 8; i++) {
        p[i] = p0[i];
    }
    double pOv[8];
    if (findTapOverride(nR1Ov, r1OvSlotElem, r1OvVal, s, e, pOv)) {
        for (int i = 0; i < 8; i++) {
            p[i] = pOv[i];
        }
    }
    const int* q = cbIx + e * 12;
    double v1 = xs[q[5]], v2 = xs[q[7]], ph1 = xs[q[4]], ph2 = xs[q[6]];
    double r1 = q[9] >= 0 ? xs[q[9]] : p[6];
    double a1 = q[8] >= 0 ? xs[q[8]] : p[7];
    double o[CB_NOUT];
    evalClosedBranch(p[0], p[1], p[2], p[3], p[4], v1, r1, v2, p[5], a1, ph1, ph2, 0,
                     o + 0, o + 1, o + 2, o + 3, o + 4, o + 5, o + 6, o + 7, o + 8, o + 9,
                     o + 10, o + 11, o + 12, o + 13, o + 14, o + 15, o + 16, o + 17, o + 18,
                     o + 19, o + 20, o + 21, o + 22, o + 23, o + 24, o + 25, o + 26, o + 27);
    int qi = dqSide[c] == 0 ? 7 : 21;            // side-1 q1[7]..dq1dr1[13]; side-2 q2[21]..dq2dr1[27]
    double out6[6] = {o[qi], o[qi + 1], o[qi + 2], o[qi + 3], o[qi + 4], o[qi + 6]};
    for (int m = 0; m < 6; m++) {
        int d = dst[m];
        double val = w * out6[m];
        if (d <= -2 && (-2 - d) == dqTraceRow) {
            printf("DQF kind=0(closed) elem=%d w=%.6f v1=%.9f v2=%.9f y=%.6f contrib=%.9f\n",
                   e, w, v1, v2, p[0], val);
        }
        if (d >= 0) atomicAdd(&valsS[d], val);
        else if (d <= -2) atomicAdd(&FS[-2 - d], val);
    }
}

// Batched DISTR_RHO fill: one thread per (contribution, scenario). Linear coef*r1 into the controller's
// BRANCH_RHO1 row — strided per scenario, active-gated, no mask (controller-outage contingencies are
// rejected to the CPU because the 1/count coefficients reform).
__global__ void fillDistrRhoBatch(int ndr, int nScenarios, int n, int nnz,
                                  const int* drCol, const double* drCoef, const int* drDest,
                                  const double* x, double* vals, double* F, const char* active,
                                  int nDrOv, const int* drOvIdx, const double* drOvC,
                                  const int* drQrGroup, const int* mode, const int* qrPtr, const int* qrRows) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ndr * nScenarios) return;
    int c = tid % ndr;
    int s = tid / ndr;
    if (!active[s]) return;
    // the CURRENT pilot-Q carrier's row hosts the relocated pilot balance — its DISTR_RHO row is
    // inactive PER SCENARIO (same rule as fillDistrQBatch; the
    // machine's drOv zeroing covers machine steps, but a bare batch solve — the batch-of-1
    // sequential route on a clamp-in-the-base network — has no drOv and needs the in-kernel skip)
    if (drQrGroup != nullptr && drQrGroup[c] >= 0) {
        int g = drQrGroup[c];
        if (-2 - drDest[c * 2] == qrRows[qrPtr[g] + relocChoose(g, qrPtr, qrRows, mode + (size_t) s * n)]) return;
    }
    double coef = drCoef[c];
    for (int k = 0; k < nDrOv; k++) {                        // per-scenario 1/count' renormalization
        if (drOvIdx[k * 2] == s && drOvIdx[k * 2 + 1] == c) {
            coef = drOvC[k];
        }
    }
    int dF = drDest[c * 2 + 0];
    int dJ = drDest[c * 2 + 1];
    if (dF <= -2) atomicAdd(&F[(size_t) s * n + (-2 - dF)], coef * x[(size_t) s * n + drCol[c]]);
    if (dJ >= 0) atomicAdd(&vals[(size_t) s * nnz + dJ], coef);
}

// Batched DISTR_SHUNT_B fill: the SHUNT_B analogue of fillDistrRhoBatch. Linear coef*b into the controller's
// SHUNT_B row, strided per scenario, active-gated.
__global__ void fillDistrShuntBBatch(int ndb, int nScenarios, int n, int nnz,
                                     const int* dbCol, const double* dbCoef, const int* dbDest,
                                     const double* x, double* vals, double* F, const char* active) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ndb * nScenarios) return;
    int c = tid % ndb;
    int s = tid / ndb;
    if (!active[s]) return;
    double coef = dbCoef[c];
    int dF = dbDest[c * 2 + 0];
    int dJ = dbDest[c * 2 + 1];
    if (dF <= -2) atomicAdd(&F[(size_t) s * n + (-2 - dF)], coef * x[(size_t) s * n + dbCol[c]]);
    if (dJ >= 0) atomicAdd(&vals[(size_t) s * nnz + dJ], coef);
}

// Batched zero-impedance fill: one thread per (branch, scenario). The Jacobian entries are
// constants, only F depends on the state — DUMMY_P/Q ±1 couplings into both buses' P/Q balances and, on a spanning-tree edge,
// the ZERO_V (v1−rho·v2) / ZERO_PHI (ph1−ph2) coupling. Active-gated, no mask (bus-coupler outage is rejected).
__global__ void fillZeroImpBatch(int nzi, int nScenarios, int n, int nnz,
                                 const double* in, const int* ix, const int* dest,
                                 const double* x, double* vals, double* F, const char* active,
                                 int nziBase, int nZiOn, const int* ziOnSlotElem,
                                 int nZiOff, const int* ziOffSlotElem,
                                 const int* ziAlt, const int* ziSel, const int* mode, int ziDyn,
                                 RelocTables rt, int nZiTree, const int* ziTreeSlotElemVal,
                                 const char* ziOffMask, const char* ziOnMask, const char* ziTreeMask) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nzi * nScenarios) return;
    int e = tid % nzi;
    int s = tid / nzi;
    if (!active[s]) return;
    size_t me = (size_t) s * nzi + e;                        // the dense (scenario, element) masks, when built
    if (ziOffMask != nullptr) {
        if (ziOffMask[me]) return;                           // per-scenario switch OUTAGE
    } else {
        for (int k = 0; k < nZiOff; k++) {                   // per-scenario switch OUTAGE (node-breaker
            if (ziOffSlotElem[2 * k] == s && ziOffSlotElem[2 * k + 1] == e) {   // contingency propagation)
                return;
            }
        }
    }
    if (e >= nziBase) {
        // BASE-OPEN closable switch element: contributes ONLY in scenarios that CLOSE it —
        // sparse (slot, elem) enable list, a handful of entries per run.
        bool on = false;
        if (ziOnMask != nullptr) {
            on = ziOnMask[me] != 0;
        } else {
            for (int k = 0; k < nZiOn; k++) {
                if (ziOnSlotElem[2 * k] == s && ziOnSlotElem[2 * k + 1] == e) {
                    on = true;
                    break;
                }
            }
        }
        if (!on) return;
    }
    double rho = in[e];
    const int* q = ix + e * 11;                              // b1P b1Q b2P b2Q dpCol dqCol v1 v2 ph1 ph2 tree
    const double* xs = x + (size_t) s * n;
    double* valsS = vals + (size_t) s * nnz;
    double* FS = F + (size_t) s * n;
    const int* d = dest + e * 14;
    double dummyP = xs[q[4]], dummyQ = xs[q[5]];
    int d2 = d[2], d3 = d[3], d6 = d[6], d7 = d[7];
    int d0 = d[0], d1 = d[1], d4 = d[4], d5 = d[5];
    if (ziDyn) {                                             // dynamic Q-routing (ob/zi channels)
        const int* ms = mode + (size_t) s * n;
        RelocTables rtS = relocForScenario(rt, s);
        d2 = ziDynDest(d2, e * 8 + 0, ziAlt, ziSel, ms, rtS);
        d3 = ziDynDest(d3, e * 8 + 1, ziAlt, ziSel, ms, rtS);
        d6 = ziDynDest(d6, e * 8 + 2, ziAlt, ziSel, ms, rtS);
        d7 = ziDynDest(d7, e * 8 + 3, ziAlt, ziSel, ms, rtS);
        d0 = ziDynDest(d0, e * 8 + 4, ziAlt, ziSel, ms, rtS);   // +-dummyP: back to the bus's own PHI row
        d1 = ziDynDest(d1, e * 8 + 5, ziAlt, ziSel, ms, rtS);   // when its parent coupler opens
        d4 = ziDynDest(d4, e * 8 + 6, ziAlt, ziSel, ms, rtS);
        d5 = ziDynDest(d5, e * 8 + 7, ziAlt, ziSel, ms, rtS);
    }
    addAt(d0, dummyP, valsS, FS);                          // F[bus1 P] += dummyP
    addAt(d1, -dummyP, valsS, FS);                         // F[bus2 P] -= dummyP
    addAt(d2, dummyQ, valsS, FS);                            // F[bus1 Q] += dummyQ
    addAt(d3, -dummyQ, valsS, FS);                           // F[bus2 Q] -= dummyQ
    addAt(d4, 1.0, valsS, FS);                             // J(bus1 P, dummyP)
    addAt(d5, -1.0, valsS, FS);                            // J(bus2 P, dummyP)
    addAt(d6, 1.0, valsS, FS);                               // J(bus1 Q, dummyQ)
    addAt(d7, -1.0, valsS, FS);                              // J(bus2 Q, dummyQ)
    int isTree = q[10] & 1;                                  // the tree at EXTRACTION...
    if (ziTreeMask != nullptr) {
        char t = ziTreeMask[me];                             // ...overridden where OLF's has moved by then
        if (t >= 0) isTree = t;
    } else {
        for (int k = 0; k < nZiTree; k++) {
            if (ziTreeSlotElemVal[3 * k] == s && ziTreeSlotElemVal[3 * k + 1] == e) {
                isTree = ziTreeSlotElemVal[3 * k + 2];
                break;
            }
        }
    }
    if (isTree) {                                            // spanning-tree edge: voltage/angle coupling
        double v1 = xs[q[6]], v2 = xs[q[7]], ph1 = xs[q[8]], ph2 = xs[q[9]];
        addAt(d[8], v1 - rho * v2, valsS, FS);               // F[ZERO_V row]
        addAt(d[9], 1.0, valsS, FS);                         // J(ZERO_V, v1)
        addAt(d[10], -rho, valsS, FS);                       // J(ZERO_V, v2)
        addAt(d[11], ph1 - ph2, valsS, FS);                  // F[ZERO_PHI row]
        addAt(d[12], 1.0, valsS, FS);                        // J(ZERO_PHI, ph1)
        addAt(d[13], -1.0, valsS, FS);                       // J(ZERO_PHI, ph2)
    }
}

// Per-scenario slack/PV identity rows (rowMode is per scenario: a contingency can change
// which buses are PV/slack): a row in identity mode (mode 0) is zeroed, its diagonal set to 1 and its
// mismatch overwritten with x - target. It runs after the fills and overwrites them, which keeps the
// structure (and cuDSS's analysis) immutable across PV->PQ switching.
__global__ void applyRowModeBatch(int n, int nScenarios, int nnz, const int* mode, const int* rowPtr,
                                  const int* diagPos, const double* x, const double* target,
                                  double* vals, double* F, const char* active, const double* rowSlope,
                                  const int* colIdx, const int* relocOwn) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n * nScenarios) return;
    int r = tid % n;
    int s = tid / n;
    if (!active[s]) return;
    int m = mode[(size_t) s * n + r];
    double* valsS = vals + (size_t) s * nnz;
    if (m == 6) {
        // CROSS identity (tasks #70, VC_REMERGE_RELOC_VROW): this coupler DUMMY_Q row hosts the relocated child's
        // reactive equation, and the child is a re-homed voltage-controlled bus whose own V row carries the live
        // coupler's ZERO_V — so BUS_TARGET_V lands HERE on the child's V column (relocOwn[r] = its own V row =
        // its V column), the slot the child's own dq/dV term already reserves in the pattern.
        int col = relocOwn != nullptr ? relocOwn[r] : -1;
        for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
            valsS[k] = colIdx[k] == col ? 1.0 : 0.0;
        }
        F[(size_t) s * n + r] = (col >= 0 ? x[(size_t) s * n + col] : 0.0) - target[(size_t) s * n + r];
    } else if (m == 0) {                                     // identity: v/phi/rho/a1/b pinned to target
        for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
            valsS[k] = 0.0;
        }
        valsS[diagPos[r]] = 1.0;
        F[(size_t) s * n + r] = x[(size_t) s * n + r] - target[(size_t) s * n + r];
    } else if (m == 2) {                                     // SLOPE-SVC: V + slope*sumQ = target — the natural
        double slope = rowSlope[r];                          // reactive fill stands, scaled, plus the v term
        for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
            valsS[k] *= slope;
        }
        valsS[diagPos[r]] += 1.0;
        size_t idx = (size_t) s * n + r;
        F[idx] = slope * (F[idx] + target[idx]) + x[idx] - target[idx];
    }
}

__global__ void initFBatch(int n, int nScenarios, const double* target, double* F, const char* active) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n * nScenarios) return;
    int s = tid / n;
    if (!active[s]) return;
    F[tid] = -target[tid];
}

// Seed the per-(controller, scenario) limitType from the BASE state, so a controller that OLF had already
// switched to PQ in the base case (packed by the extractor as a voltage-CAPABLE PQ bus, base row mode 1)
// starts pinned at the RIGHT limit and can be switched BACK to PV by a relieving contingency. A base-PV
// controller (mode 0) starts at limitType 0. For a base-PQ controller, v below its target ⇒ it is producing
// max reactive yet cannot reach target ⇒ maxQ-pinned (2); v above target ⇒ minQ-pinned (1). switchCount
// starts at 0 either way (a separate memset).
const int* psEntryPtr(BatchSolverContext* ctx) {
    return ctx->hasPilotSelf && ctx->dPsPilotEntry.p != nullptr ? ctx->dPsPilotEntry.template as<int>() : nullptr;
}


__global__ void initRlLimitTypeBatch(int nrl, int nScenarios, int n, const int* rlVrow, const int* rlQrow,
                                     const double* rlVtarget, const char* rlMonitored,
                                     const int* baseMode, const double* x, char* limitType,
                                     int nRlVtOv, const int* rlVtOvSlotIdx, const double* rlVtOvVal, const double* svcRlVt,
                                     const char* rlRemote, const int* rlCmpVrow,
                                     const double* target, const double* rlMinQb, const double* rlMaxQb,
                                     const double* rlLoadQb, const char* rlBaseLt,
                                     const int* psEntry, int nQr, const char* rlGenLost) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nrl * nScenarios) return;
    int i = tid % nrl;
    int s = tid / nrl;
    // A bus that LOST every generator this scenario (GenOutage, setBatchRlGenLost) is no controller: OLF has no
    // voltage control there at all. Seeded FROZEN, the entry is skipped by the reactive-limit check, reads as
    // "not enabled" to the pilot-self layout and, for a pilot that was one of its own controllers, opens the
    // pilot-self gate so the bus's reactive balance is emitted (on the carrier, then on its own row once every
    // survivor pins). Left at 0 it read as a live controller: the gate suppressed the balance and the outaged
    // pilot's row, flipped to host it, stayed EMPTY (rte6515 GEN-1522: VL-1522_0 held at its 1.122 target).
    if (rlGenLost != nullptr && rlGenLost[(size_t) s * nrl + i]) {
        limitType[(size_t) s * nrl + i] = RL_LIMIT_FROZEN;
        return;
    }
    int vrow = rlVrow[i];
    // the bus's own V variable (VOLTAGE reads) and the row its pin state lives on, which the coupler-tree
    // relocation can move to the parent coupler's DUMMY_Q (see dRlQrow)
    int qrow = rlQrow != nullptr ? rlQrow[i] : vrow;
    if (qrow != vrow && baseMode[qrow] == 0) {
        qrow = vrow;                                         // that coupler is open in the BASE: equations at home
    }
    size_t idx = (size_t) s * n + vrow;
    size_t qix = (size_t) s * n + qrow;
    size_t st = (size_t) s * nrl + i;
    // OLF's own reason for the base pin, when the host carried it: it distinguishes a
    // realistic-voltage pin (3/4, target = generation targetQ) from a Q-limit pin (1/2, target =
    // the limit). The nearest-limit guess below turns the former into the latter, and the
    // curve-limit drift then re-freezes it AT the limit (rte6515 VL-1251_1: -0.0035 -> -1.0000).
    char olfKind = rlBaseLt != nullptr ? rlBaseLt[i] : 0;
    if (rlRemote != nullptr && rlRemote[i]) {
        // REMOTE group member (plan-B3): base mode 1 at its row is the UNPINNED DISTR_Q share —
        // only the canonical mode 4 marks a base pin. The pilot's V equals the group target at the
        // base solution (the survivors keep controlling), so the v-vs-target heuristic is
        // DEGENERATE here — key the direction on WHICH limit the extracted BUS_TARGET_Q target
        // equals instead (target = limit − loadQ was written by the base activation).
        limitType[st] = 0;
        if (isEnrolledPilot(psEntry, nQr, i) && olfKind != RL_BASE_RELOCATED) {
            // An ENROLLED pilot-self pilot (rotation) keeps its V row an identity while PINNED: the pin's
            // balance rides the carrier (or its own row flipped to a power row once the group lost control),
            // so the mode-4 marker below never appears on its row. OLF's own kind is the only record of its
            // base pin — and without it the seeding left the pilot UNPINNED (psSkip = 1, raw q dropped) on a
            // base that had it pinned: every scenario of the [5000,6000) busbar window failed its first
            // linear solve (976 x LINEAR_SOLVE_FAILED).
            limitType[st] = olfKind >= 1 && olfKind <= 4 ? olfKind : 0;
            return;
        }
        if (baseMode[qrow] == 4) {
            double tgt = target[qix];
            limitType[st] = olfKind != 0 ? olfKind
                    : (fabs(tgt - (rlMaxQb[i] - rlLoadQb[i])) <= fabs(tgt - (rlMinQb[i] - rlLoadQb[i])) ? 2 : 1);
        }
        if (olfKind == RL_BASE_RELOCATED) {
            // This entry does NOT own the pin on its V row (GpuAcOuterLoopPacks.RL_BASE_RELOCATED): either its
            // own reactive balance was relocated off that row, or the row carries ANOTHER bus's relocated
            // balance. Both happen on a pilot that is one of its own controllers while a sibling still holds
            // the voltage. Freeze the entry: the reactive-limits pass and the switch-back both key on the V
            // row, and pinning or unpinning a row this entry does not own left the Jacobian rank-deficient.
            limitType[st] = RL_LIMIT_FROZEN;
        }
        return;
    }
    // Read the BASE row mode (length n, contingency-free) — NOT the per-scenario mode: a controller flipped to
    // PQ by a per-scenario gen-loss OVERRIDE (its generator is gone) has base mode 0 and must get limitType 0 so
    // the reactive-limits switch-back leaves it PQ. Only a controller that is PQ in the BASE (an actual Q-limit
    // switch during the base loadflow) gets a pinned limit type so a relieving contingency can switch it back.
    if (baseMode[qrow] == 1 && (rlMonitored[i] & 1) == 0) {       // PQ in the base case (a real Q-limit switch —
        // NOT a monitoring SVC standing by: that one seeds UNPINNED so only the VoltageMonitoring
        // orchestrator may activate it, exactly like the CPU's null qLimitType)
        double vt = rlTargetOf(nRlVtOv, rlVtOvSlotIdx, rlVtOvVal, svcRlVt, nrl, s, i, rlVtarget[i]);
        limitType[st] = olfKind != 0 ? olfKind
                : ((x[idx] < vt) ? 2 : 1);                  // v below target ⇒ maxQ pinned; above ⇒ minQ pinned
    } else {
        limitType[st] = 0;                                  // base PV (incl. a per-scenario gen-override flip)
    }
}


// Evaluate the per-scenario DYNAMIC reactive limits (OLF: bus.getMinQ()/getMaxQ() re-read each RL
// check, curve gens at their CURRENT targetP): start from the static pack values and add each
// participating curve gen's delta eval(p_now) - eval(p_base). p_now = basePpu + share * (the bus's
// DS-moved row target - its base) — exact while intra-bus proportionality holds (the pack's DS
// model). At the base point the delta is bit-zero (same formula both sides).
// rlCurveLimitsBatch reads each generator's OWN cumulative P from the distributed-slack water-fill
// (dGenCum), which clamps PER GENERATOR, instead of splitting its bus's movement by a share fixed at
// pack time. The share is exact only while the intra-bus split stays proportional, and a clamp breaks
// that: on rte6515 GEN-6172 at VL-6139_0 the 451 MW unit takes +0.0442 where its share says +0.0890,
// which puts the bus's Q box out by 2.67 MVAr and the bus 0.0379 kV off. The share reconstruction
// survives ONLY as the fallback for entries genCum cannot serve (no DS slot, the per-generator
// water-fill did not run, or the outaged bus's own entries) — there is no switch back to it.
// OLF_GPU_CV_PROBE=<rl entry>: trace the curve-limit evaluation for that entry (device printf).
static const int cvProbeEntry = std::getenv("OLF_GPU_CV_PROBE") == nullptr ? -1
        : (std::string(std::getenv("OLF_GPU_CV_PROBE")) == "all" ? -2
           : std::atoi(std::getenv("OLF_GPU_CV_PROBE")));
// The host readback ([CV-HOST]) takes a COMMA list of entries (the device printf keeps the first).
static std::vector<int> cvProbeEntries() {
    std::vector<int> v;
    const char* e = std::getenv("OLF_GPU_CV_PROBE");
    if (e == nullptr || std::string(e) == "all") return v;
    std::string str(e);
    size_t pos = 0;
    while (pos <= str.size()) {
        size_t c = str.find(',', pos);
        if (c == std::string::npos) c = str.size();
        if (c > pos) v.push_back(std::atoi(str.substr(pos, c - pos).c_str()));
        pos = c + 1;
    }
    return v;
}
// OLF_GPU_CV_PROBE_SLOT=<slot>: the scenario the probe prints for (default 0 — the base load flow's single slot).
// "ctg" resolves to the chunk-LOCAL slot the host set for OLF_GPU_LOOP_TRACE_CTG (kernel scenario indices
// are chunk-local; a global contingency index never matches one).
static const bool cvProbeSlotByCtg = std::getenv("OLF_GPU_CV_PROBE_SLOT") != nullptr
        && std::string(std::getenv("OLF_GPU_CV_PROBE_SLOT")) == "ctg";
static const int cvProbeSlot = std::getenv("OLF_GPU_CV_PROBE_SLOT") == nullptr || cvProbeSlotByCtg ? 0
        : std::atoi(std::getenv("OLF_GPU_CV_PROBE_SLOT"));

}  // namespace olfgpu
#include "outerloop/slack_participants.cuh"
namespace olfgpu {








// The AC-emulation VSC entries of the curve channel: OLF evaluates such a station's curve at the live flow of its
// link in the linear mode (LfVscConverterStationImpl.getReactiveLimitsActivePower: minus the side's emulation term),
// and at its target P otherwise (saturated, frozen, disabled link), which the HVDC loop holds as baseD + appliedD.
// hv == nullptr: no such entry. nhv == 0: the HVDC loops are not on this context (then nothing was packed).
}  // namespace olfgpu
#include "outerloop/reactive_capability_curve.cuh"
namespace olfgpu {

}  // namespace olfgpu
#include "outerloop/forced_target_q.cuh"
namespace olfgpu {

















// on-device reactive-limit decision log (OLF_GPU_SW_LOG): capacity in records, width in doubles
#define SW_LOG_CAP 1048576
#define SW_LOG_W 12


__global__ void reactiveLimitsBatch(const double* svcRlVt, int nrl, int nScenarios, int n, double tol, int maxSwitch,
                                    const int* rlVrow, const int* rlQrow, const double* rlMinQ, const double* rlMaxQ,
                                    const double* rlLoadQ, const double* rlVtarget, const double* qBuf,
                                    const double* x, const char* rlActive, int* mode, double* target,
                                    char* limitType, char* switchCount, char* switched,
                                    char* passSwitched, char* savedMode,
                                    int nRlVtOv, const int* rlVtOvSlotIdx, const double* rlVtOvVal,
                                    const double* rowSlope, double* savedTarget, int cvStride,
                                    const char* rlRemote, const int* rlCmpVrow,
                                    const double* rlSvcBmin, const double* rlSvcBmax, const double* rlV2Base,
                                    const double* svcBminDyn, const double* svcBmaxDyn,
                                    const char* rlIslanded,
                                    const double* rlInitTgtQ, double vLoRobust, double vHiRobust,
                                    int robust,
                                    int swLogSlot, double* swLog, int* swLogCnt, int passId,
                                    const int* psEntry, int nQr,
                                    const char* rlLocalOv, const double* rlLocalVt,
                                    const char* rlNoUnpin, char* unpinBad, const char* rlMonitored,
                                    const int* rlCmpOv) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nrl * nScenarios) return;
    int i = tid % nrl;
    int s = tid / nrl;
    if (!rlActive[s]) return;
    // A controller the contingency put in ANOTHER synchronous component is not a controller at all:
    // OLF disables that bus and its outer loops never see it. Its rows here are FROZEN at a flat
    // 1<0, so the q read out of qBuf is a placeholder, not a measurement — and switching on it pins
    // a PV bus that OLF leaves alone, which perturbs the MAIN component through any remote-control
    // or DISTR_Q group the controller belongs to. Measured on LINE-3320-350: the device pinned
    // VL-6432_0, an ISLANDED bus, giving it 20 pinned controllers against OLF's 19 and one extra
    // LOW_VOLTAGE violation at VL-1624.
    if (rlIslanded != nullptr && rlIslanded[(size_t) s * nrl + i]) return;
    // A controller FROZEN by the generator voltage control manager has its voltage control OFF, so it
    // is not one of this loop's controllers at all — OLF iterates buses with voltage control and never
    // reaches it. Leave the type (and the row) to the transformer phase that set it.
    if (limitType[(size_t) s * nrl + i] == RL_LIMIT_FROZEN) return;
    // SVC reactive limits are VOLTAGE-DEPENDENT: LfStaticVarCompensatorImpl evaluates
    // Bmin/Bmax*(v*nomV)^2 at the LIVE bus V on every CPU check (and a pinned bus re-freezes at
    // the moved limit via busesWithUpdatedQLimits). rlMinQ/rlMaxQ hold the base-V aggregates;
    // the per-entry svcB coefficients add the quadratic delta vs the packed base v^2 — exact for
    // the SVC part, identically zero on SVC-less buses.
    bool svcLim = rlSvcBmin[i] != 0.0 || rlSvcBmax[i] != 0.0;
    // ...minus a lost SVC's share, per scenario (svcBminDyn: built only when an outage takes one)
    double svcBMin = svcBminDyn != nullptr ? svcBminDyn[(size_t) s * nrl + i] : rlSvcBmin[i];
    double svcBMax = svcBmaxDyn != nullptr ? svcBmaxDyn[(size_t) s * nrl + i] : rlSvcBmax[i];
    // records one decision of the traced scenario; the branches below call it right where they act
    // swLogSlot >= 0: every check of that ONE scenario. swLogSlot == -2 ("all"): only the checks
    // that DECIDED something, for every scenario - the whole batch's decision set in ~40 records per
    // scenario, which is what two runs have to be compared on when the contingency that flips is not
    // known in advance.
    auto swLogPut = [&](int act, double q, double mn, double mx, double vOwn, double cmpV, double vt) {
        if (swLog == nullptr) return;
        if (swLogSlot == -2 ? act == 0 : s != swLogSlot) return;
        int k = atomicAdd(swLogCnt, 1);
        if (k >= SW_LOG_CAP) return;
        double* r = swLog + (size_t) k * SW_LOG_W;
        r[0] = passId; r[1] = i; r[2] = act; r[3] = q; r[4] = mn; r[5] = mx;
        r[6] = vOwn; r[7] = cmpV; r[8] = vt; r[9] = limitType[(size_t) s * nrl + i];
        r[10] = switchCount[(size_t) s * nrl + i]; r[11] = s;
    };
    // A RE-HOMED voltage control (GpuBatchQualification.reHomedControls): this scenario the bus is a plain PV
    // bus, so its entry — packed as a remote member of the base group — takes the LOCAL path below, at its
    // own voltage target. OLF checks it as the PV bus it now is (BUS-6113_BBS_1: VL-6113_1 pinned at RL#1).
    bool localOv = rlLocalOv != nullptr && rlLocalOv[(size_t) s * nrl + i];
    if (rlRemote != nullptr && rlRemote[i] && !localOv) {
        // REMOTE group member (plan-B3). Unpinned = its DISTR_Q row (mode 1, limitType 0);
        // pinned = mode 4 ("row hosts its own Q balance"). BOTH switch conditions test the
        // PILOT's voltage (rlCmpVrow) against the GROUP target — B0-verified CPU semantics.
        // The FREE member pins the same way (plan-B4): the fill kernels re-pick the pilot-Q
        // carrier by mode-scan over the relocation candidates, and updateDistrQTargetsBatch
        // hands the pilot's Q target to the new carrier's row. When the LAST live member pins,
        // updatePilotVeqBatch flips the pilot's V identity to a power row (control lost) and
        // restores it on the first unpin — the all-pinned corner runs on-device too.
        size_t ridx = (size_t) s * n + rlVrow[i];            // the bus's own V variable: every VOLTAGE read
        // ...and the row its reactive balance and pin live on. The coupler-tree relocation moves it to the parent
        // coupler's DUMMY_Q row — but PER SCENARIO: a contingency that OPENS that coupler sends the child's
        // equations back to its own rows and the host pins the dummy row with a DUMMY_TARGET identity, which is
        // exactly what mode 0 there means (the same test the zi scatter's stock-placement fallback uses).
        size_t qidx = ridx;
        if (rlQrow != nullptr && rlQrow[i] != rlVrow[i]) {
            size_t relocIdx = (size_t) s * n + rlQrow[i];
            qidx = mode[relocIdx] == 0 ? ridx : relocIdx;
        }
        size_t rst = (size_t) s * nrl + i;
        size_t rli = cvStride ? rst : (size_t) i;
        // The quadratic delta exists only for SVC entries: rlV2Base is the packed base v², NaN when
        // the extraction ran on a never-solved network (the base load flow's own context) — 0 × NaN
        // poisoned every entry's limits there and nothing ever pinned at base.
        double rdv2 = svcLim ? x[ridx] * x[ridx] - rlV2Base[i] : 0.0;
        double rliMin = rlMinQ[rli] + (svcLim ? svcBMin * rdv2 : 0.0);
        double rliMax = rlMaxQ[rli] + (svcLim ? svcBMax * rdv2 : 0.0);
        // rlRemote 2 = a PILOT that is one of its own controllers: unpinned, its row is the group's
        // voltage IDENTITY (mode 0), not a DISTR_Q share; pinned, its own Q balance (mode 4, the
        // alternate dest). Everything else (q via the remote q pack, the pilot's V as compare row)
        // is a remote member's. As a LOCAL entry its pin landed on mode 1, an EMPTY row.
        int pvForm = rlRemote[i] == 2 ? 0 : 1;
        bool enrolledPilot = rlRemote[i] == 2 && isEnrolledPilot(psEntry, nQr, i);
        if (limitType[rst] == 0) {
            if (mode[qidx] != pvForm) return;                // islanded / frozen this scenario
            double q = qBuf[qidx] + rlLoadQ[rli];            // own q via fillRemoteMemberQBatch
            // voltageRemoteControlRobustMode (ReactiveLimitsOuterLoop.checkControllerBus): the criteria
            // below test this REMOTE controller's OWN voltage — the same x row the quadratic SVC delta
            // reads — against the realistic band shrunk by the 1.02 margin.
            // OLF's robust criterion applies to REMOTE controllers only, and "remote" is
            // ReactiveLimitsOuterLoop.isGeneratorRemoteController: the bus's OWN voltage control, not the
            // merged group's. Two classes are local in that respect and both are exempt here:
            //   2 = a pilot that controls its own bus;
            //   3 = a member of a MERGED group whose own (DEPENDENT) control points at ITSELF, which is
            //       every merged member — OLF skips the whole robust block for them (tasks #66; the
            //       device used to freeze rte6515's VL-2669_1..4 with MIN_REALISTIC_V where OLF froze
            //       nothing, replacing the DISTR_Q sharing with mode-4 own-Q balances in the STRUCTURE).
            bool robustRemote = rlRemote[i] != 2 && rlRemote[i] != 3;
            bool unrealisticLo = robust && robustRemote && x[ridx] < vLoRobust;
            bool unrealisticHi = robust && robustRemote && x[ridx] > vHiRobust;
            if (q < rliMin - tol) {
                savedTarget[rst] = target[qidx];             // the DISTR_Q row target (NOT 0 in general:
                                                             // see updateDistrQTargetsBatch)
                if (!enrolledPilot) {
                    mode[qidx] = 4;
                    target[qidx] = rliMin - rlLoadQ[rli];
                }
                limitType[rst] = 1;
                switchCount[rst]++;
                switched[s] = 1;
                passSwitched[rst] = 1;
            } else if (q > rliMax + tol) {
                savedTarget[rst] = target[qidx];
                if (!enrolledPilot) {
                    mode[qidx] = 4;
                    target[qidx] = rliMax - rlLoadQ[rli];
                }
                limitType[rst] = 2;
                switchCount[rst]++;
                switched[s] = 1;
                passSwitched[rst] = 1;
            } else if (unrealisticLo || unrealisticHi) {
                // Q inside its limits but the controller's own voltage is already unrealistic: OLF
                // stops maintaining the REMOTE voltage and freezes the generation at its INITIAL
                // target Q, with a realistic-V limit type (3 = MIN_REALISTIC_V, 4 = MAX_REALISTIC_V)
                // that the switch-back reads like a min/max limit but the Q-limit drift ignores.
                savedTarget[rst] = target[qidx];
                if (!enrolledPilot) {
                    mode[qidx] = 4;
                    target[qidx] = rlInitTgtQ[i] - rlLoadQ[rli];
                }
                limitType[rst] = unrealisticLo ? 3 : 4;
                switchCount[rst]++;
                switched[s] = 1;
                passSwitched[rst] = 1;
            }
            // OLF also sets the controller's V to a safe 1 pu for the next computation, under BOTH
            // robust criteria. That WRITES the state vector every other entry is still reading here,
            // so it runs as its own pass (rlRobustVResetBatch) once this one has finished.
        } else {                                            // pinned — switch-back tests the PILOT
            // a controller re-homed onto a NEW pilot this scenario reads that pilot's voltage (rlCmpOv)
            double v = x[(size_t) s * n + (rlCmpOv != nullptr && rlCmpOv[rst] >= 0 ? rlCmpOv[rst] : rlCmpVrow[i])];
            double vt = rlTargetOf(nRlVtOv, rlVtOvSlotIdx, rlVtOvVal, svcRlVt, nrl, s, i, rlVtarget[i]);
            bool minPin = limitType[rst] == 1 || limitType[rst] == 3;   // LfBus.QLimitType.isMinLimit()
            // OLF's checkPqBusWithQLimitType PROPOSES the unpin whenever the pilot's voltage asks for
            // it; the per-bus cap (maxPqPvSwitch) then blocks the switch itself in switchPqPv. Only
            // the ELSE of that test re-freezes a moved limit — so a capped bus that still WANTS to
            // unpin re-freezes on neither arm, while a capped bus that does not must still follow its
            // limit. Gating this whole block on the cap froze such a bus's target at a stale limit:
            // network.xiidm / GBMK, CQEM_0 and CQEM_5 kept -0.181175517 / -0.184950765 where OLF had
            // re-frozen at -0.181173957 / -0.184949017 (a reactive-capability curve moving with the
            // distributed slack), 2.079e-05 kV network-wide.
            bool wantsBack = (!minPin && v > vt) || (minPin && v < vt);
            bool back = wantsBack && switchCount[rst] < maxSwitch;
            if (back && rlNoUnpin != nullptr && rlNoUnpin[rst]) {
                // a pinned member of a re-homed cluster with no enabled controller: OLF would re-activate the
                // cluster's BUS_TARGET_V, which this table cannot host for a lost pilot — reject, CPU replay
                unpinBad[s] = 1;
                back = false;
            }
            if (back) {
                if (!enrolledPilot) {
                    mode[qidx] = pvForm;                     // back to the DISTR_Q share row (the V identity for a pilot)
                    target[qidx] = savedTarget[rst];
                }
                limitType[rst] = 0;
                switched[s] = 1;
                passSwitched[rst] = 2;
            } else if (!wantsBack && (cvStride || svcLim) && limitType[rst] <= 2) {   // curve/SVC limit drift while pinned
                // OLF re-freezes at a moved limit only for MIN_Q/MAX_Q (checkPqBusWithQLimitType tests
                // the type explicitly); a realistic-V freeze holds its initial target Q.
                double lim = limitType[rst] == 1 ? rliMin : rliMax;
                if (!enrolledPilot && fabs(lim - (target[qidx] + rlLoadQ[rli])) > tol) {
                    target[qidx] = lim - rlLoadQ[rli];
                    switched[s] = 1;
                    passSwitched[rst] = 3;
                }
            }
        }
        {
            double qRec = qBuf[qidx] + rlLoadQ[rli];
            int cmpR = rlCmpOv != nullptr && rlCmpOv[rst] >= 0 ? rlCmpOv[rst] : (rlCmpVrow != nullptr ? rlCmpVrow[i] : -1);
            double cmpRec = cmpR >= 0 ? x[(size_t) s * n + cmpR] : x[ridx];
            swLogPut(passSwitched[rst], qRec, rliMin, rliMax, x[ridx], cmpRec,
                     rlTargetOf(nRlVtOv, rlVtOvSlotIdx, rlVtOvVal, svcRlVt, nrl, s, i, rlVtarget[i]));
        }
        return;
    }
    int vrow = rlVrow[i];
    size_t idx = (size_t) s * n + vrow;
    // The q this check READS is accumulated by fillRemoteMemberQBatch at rlQrow, which for a relocated
    // child is the parent coupler's DUMMY_Q row and NOT its own V row (see the rlQrow comment in
    // GpuAcOuterLoopPacks: filling the own row "left the check reading ZERO for every relocated member,
    // so it could never pin one"). That pack is STRUCTURAL -- it keeps writing there even in a scenario
    // that pins the dummy to a DUMMY_TARGET identity -- and qBuf is a separate accumulator, so reading it
    // there is valid whatever the row's equation MODE. Only the pin WRITE belongs on the live row, which
    // is idx. Conflating the two reproduced that base failure per scenario: measured on rte6515
    // BUS-6113_BBS_1, VL-6113_1 is re-homed LOCAL (reHomePv rlEntry=969) and OLF pins it MIN_Q at
    // q=-3.231 against a -0.774 limit, while the device read q~0 and never pinned. Its THIRD slack check
    // then measured 0.013280180 where OLF measures 0.006846832 -- straddling the 1 MW gate -- so the
    // device took a distribution OLF does not and reported 1.328 MW more distributed active power: the
    // worst such difference on the whole busbar scope.
    size_t qIdx = (rlQrow != nullptr && rlQrow[i] >= 0 && rlQrow[i] != vrow)
            ? (size_t) s * n + rlQrow[i] : idx;
    size_t st = (size_t) s * nrl + i;
    size_t li = cvStride ? st : (size_t) i;                 // curve limits are per (scenario, entry)
    bool slopeRow = rowSlope[vrow] != 0;                    // SLOPE-SVC bus: its PV form is row mode 2
    double dv2 = svcLim ? x[idx] * x[idx] - rlV2Base[i] : 0.0;   // see the remote branch: NaN base v² otherwise
    double liMin = rlMinQ[li] + (svcLim ? svcBMin * dv2 : 0.0);
    double liMax = rlMaxQ[li] + (svcLim ? svcBMax * dv2 : 0.0);
    // A DEGENERATE reactive range (min >= max) belongs to a bus with no reactive capability of its
    // own: the CONTROLLED (pilot) bus of a remote group carries a generator voltage control but no
    // generators, so bus.getMinQ()/getMaxQ() are a sum over nothing and both are 0. OLF never checks
    // such a bus - ReactiveLimitsOuterLoop walks getControllerElements(GENERATOR) and the pilot is
    // not one of them - while here its q is the bus's own reactive balance, which for a pure transit
    // node is zero up to rounding, so the pin test |q| > tol with tol = convEps (1e-12 under
    // UNIFORM_CRITERIA) fires on the LAST BITS. Measured on network.xiidm: two runs of the same
    // 2000-contingency window differed on 430 reactive-limit decisions and every one was such an
    // entry, q at +-1.2e-12 against limits of exactly 0. The pin flips the pilot's row to its own Q
    // balance for a pass, which perturbs the whole group, and once per ~2000 contingencies it
    // changes a real downstream pin (8.178e-05 kV on the KUMB group).
    // Neither direction is this entry's to decide. The pin is refused (above), and the switch-back
    // is not its either: the pilot's row follows its MEMBERS — updatePilotVeqBatch drops the V
    // identity when the last member pins and gives it back when one unpins, base-lost groups
    // included. Blocking both directions used to leave network.xiidm/LKPH 0.32 kV off because
    // nothing restored the row then; now something does, in the same pass OLF does it, and this
    // entry's own voltage test (a pin type guessed for an empty range) fired a pass late on LNER.
    // ...but ONLY such a bus: rlMonitored bit 2 (GpuAcOuterLoopPacks) marks an entry that is NOT one of its voltage
    // control's controller elements, the set OLF iterates (LfNetwork.getControllerElements(GENERATOR)). A real
    // generator bus whose capability curve collapses to [0, 0] at its current P is a controller OLF still checks:
    // network.xiidm/FBQV, FBQF_0 pinned MIN_Q at 0 with V 1.009793 under its 1.010221 target - OLF switched it
    // back to PV at the first reactive-limit pass, the device never tested it, and CQEM_0 then pinned a pass late
    // (the one contingency of 24,376 whose outer-loop sequence differed from the 1-thread CPU's).
    if (liMin >= liMax && (rlMonitored == nullptr || (rlMonitored[i] & 2) != 0)) {
        swLogPut(passSwitched[st], qBuf[qIdx] + rlLoadQ[li], liMin, liMax, x[idx], x[idx],
                 rlTargetOf(nRlVtOv, rlVtOvSlotIdx, rlVtOvVal, svcRlVt, nrl, s, i, rlVtarget[i]));
        return;
    }
    // A re-homed LOCAL control whose own V row carries a live coupler's ZERO_V keeps its identity (mode 6, on its
    // own V column) and its pinned balance (mode 4) on the RELOCATED row — the parent coupler's DUMMY_Q (tasks #70,
    // VC_REMERGE_RELOC_VROW). The voltage is still read at the bus's own column (idx).
    size_t pinIdx = idx;
    if (localOv && rlQrow != nullptr && rlQrow[i] >= 0 && rlQrow[i] != vrow) {
        size_t r2 = (size_t) s * n + rlQrow[i];
        if (mode[r2] == 6 || (mode[r2] == 4 && mode[idx] != 4)) {
            pinIdx = r2;
        }
    }
    const bool crossRow = pinIdx != idx;
    if ((mode[pinIdx] == 0 || mode[pinIdx] == 2 || mode[pinIdx] == 6) && limitType[st] == 0) {   // currently PV — check the reactive limits
        double q = qBuf[qIdx] + rlLoadQ[li];
        if (q < liMin - tol) {
            savedTarget[st] = target[pinIdx];                // remember the row target for the switch-back
            savedMode[st] = static_cast<char>(mode[pinIdx]);   // and the PV form (0, 2 for a slope SVC, 6 cross)
            mode[pinIdx] = localOv ? 4 : 1;              // a re-homed member: own Q through its ALTERNATE dest
            target[pinIdx] = liMin - rlLoadQ[li];
            limitType[st] = 1;
            switchCount[st]++;
            switched[s] = 1;
            passSwitched[st] = 1;
        } else if (q > liMax + tol) {
            savedTarget[st] = target[pinIdx];
            savedMode[st] = static_cast<char>(mode[pinIdx]);
            mode[pinIdx] = localOv ? 4 : 1;              // a re-homed member: own Q through its ALTERNATE dest
            target[pinIdx] = liMax - rlLoadQ[li];
            limitType[st] = 2;
            switchCount[st]++;
            switched[s] = 1;
            passSwitched[st] = 1;
        }
    } else if (limitType[st] != 0) {                    // pinned at a limit — check switchback
        double v = x[idx];
        double vt = localOv ? rlLocalVt[(size_t) s * nrl + i]
                  : rlTargetOf(nRlVtOv, rlVtOvSlotIdx, rlVtOvVal, svcRlVt, nrl, s, i, rlVtarget[i]);
        // OLF never back-switches a slope-SVC bus: checkPqBus runs with
        // canSwitchToPv = !bus.hasGeneratorsWithSlope() ("we don't support switching PQ to PV
        // for bus with one controller with slope") — once pinned, a slope row stays PQ.
        bool maxPin = limitType[st] == 2 || limitType[st] == 4;       // LfBus.QLimitType.isMaxLimit()
        bool minPin = limitType[st] == 1 || limitType[st] == 3;       // LfBus.QLimitType.isMinLimit()
        // the cap bounds the SWITCH, not the limit refresh below (see the remote branch)
        bool wantsBack = !slopeRow
                && ((maxPin && v > vt)                               // maxQ-pinned but v above target
                 || (minPin && v < vt));                             // minQ-pinned but v below target
        bool back = wantsBack && switchCount[st] < maxSwitch;
        if (back) {
            mode[pinIdx] = crossRow ? 6 : 0;                // PQ → PV: resume voltage control (cross identity stays cross)
            target[pinIdx] = vt;
            limitType[st] = 0;
            switched[s] = 1;
            passSwitched[st] = 2;
        } else if (!wantsBack && (cvStride || svcLim) && limitType[st] <= 2) {   // a realistic-V freeze keeps its target Q
            // Q-limit UPDATE while pinned (OLF checkPqBusWithQLimitType): a CURVE limit that moved
            // (DS moved the gen's P) or an SVC limit that moved (B*v^2 at the live V) re-freezes
            // the pinned Q at the NEW limit when it drifted by more than the mismatch tol.
            // Unreachable with static limits — gated on the channels.
            double lim = limitType[st] == 1 ? liMin : liMax;
            if (fabs(lim - (target[pinIdx] + rlLoadQ[li])) > tol) {
                target[pinIdx] = lim - rlLoadQ[li];
                switched[s] = 1;
                passSwitched[st] = 3;
            }
        }
    }
    swLogPut(passSwitched[st], qBuf[qIdx] + rlLoadQ[li], liMin, liMax, x[idx], x[idx],
             rlTargetOf(nRlVtOv, rlVtOvSlotIdx, rlVtOvVal, svcRlVt, nrl, s, i, rlVtarget[i]));
}

// AcloadFlowEngine.isStateUnrealistic, on the states INSIDE the sweep. OLF re-checks after every
// solver run once the last loop that can fix an unrealistic state has run, and a state outside
// [minRealisticVoltage, maxRealisticVoltage] on a non-fictitious bus at or above
// minNominalVoltageRealisticVoltageCheck ends the outer-loop cycle with UNREALISTIC_STATE — the run
// FAILS and reports no violations. The device only ever checked the FINAL restored state, so a
// scenario that passed through an unrealistic state and recovered came back CONVERGED where OLF
// reports FAILED. Flagged here after every inner solve and routed to the CPU replay, which applies
// OLF's own rule about WHICH solve the check belongs to (it depends on the configured loop order and
// on robust mode) instead of the device guessing at it. Rejecting is always sound: it changes the
// route, never the answer.
__global__ void realisticCheckBatch(int n, int nScenarios, const char* rowChecked, const double* x,
                                    const char* live, double minV, double maxV, char* bad) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * nScenarios) return;
    int i = t % n;
    int s = t / n;
    if (!live[s] || !rowChecked[i]) return;
    double v = x[(size_t) s * n + i];
    if (v < minV || v > maxV) {
        bad[s] = 1;
    }
}

// voltageRemoteControlRobustMode, second half: "Set V to a safe 1 pu for the next computation".
// ReactiveLimitsOuterLoop does this to every REMOTE voltage controller it switched PV->PQ in this
// check whose OWN voltage is outside the realistic band — whether it switched on a reactive limit
// (the first robust block) or on the realistic-V criterion itself. It is a WRITE to the state
// vector the check kernel reads, so it gets its own pass: one thread per (entry, scenario), reading
// the same pre-pass x, which makes the result independent of the order the entries happen to run in.
// x is the next inner solve's warm start, so this is OLF's controllerBus.setV(1) verbatim.
__global__ void rlRobustVResetBatch(int nrl, int nScenarios, int n, const int* rlVrow, const char* rlRemote,
                                    const char* rlActive, const char* passSwitched, double* x,
                                    double vLoRobust, double vHiRobust) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nrl * nScenarios) return;
    int i = tid % nrl;
    int s = tid / nrl;
    if (!rlActive[s]) return;
    // robust mode is remote-controller only, in OLF's sense: not a class-2 pilot and not a class-3
    // merged member whose own control is local (both are exempt from the gate that sets 3/4 at all)
    if (rlRemote == nullptr || rlRemote[i] != 1) return;
    if (passSwitched[(size_t) s * nrl + i] != 1) return;     // switched PV -> PQ in THIS check
    size_t idx = (size_t) s * n + rlVrow[i];
    if (x[idx] < vLoRobust || x[idx] > vHiRobust) {
        x[idx] = 1.0;
    }
}

// OLF's GLOBAL forced-PV election (ReactiveLimitsOuterLoop.switchPvPq): when EVERY bus still PV
// would switch PV→PQ in this check (remainingPvBusCount == 0), the strongest candidate stays PV —
// highest controlled-bus nominal V, then highest bus target P (LIVE: the DistributedSlack may have
// moved it), then lowest id. One thread per scenario, after the pin kernel: it UNDOES the elected
// pin (row mode and target back, limit type and switch count back) and re-derives the scenario's
// "changed" flag from what the pass still did. Before this the device pinned every candidate and a
// per-scenario DETECTOR routed the scenario to the CPU (1.9 kV wrong on the forced-PV network).
__global__ void rlForcedPvElectBatch(int nrl, int nScenarios, int n,
                                     const int* rlVrow, const char* rlRemote, const char* rlIslanded,
                                     const double* nomV, const int* pRow, const double* staticP, const int* idRank,
                                     char* limitType, char* switchCount, char* passSwitched, const char* savedMode,
                                     const double* savedTarget, int* mode, double* target,
                                     const char* rlActive, char* switched) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    if (!rlActive[s]) return;
    int remaining = 0;
    int cand = 0;
    int best = -1;
    double bestV = -1.0;
    double bestP = -1e300;
    int bestId = 0x7fffffff;
    for (int i = 0; i < nrl; i++) {
        size_t st = (size_t) s * nrl + i;
        if (rlIslanded != nullptr && rlIslanded[st]) continue;   // not a controller in this component
        bool remote = rlRemote != nullptr && rlRemote[i];
        int m = mode[(size_t) s * n + rlVrow[i]];
        if (passSwitched[st] == 1) {
            cand++;
            double v = nomV[i];
            double p = pRow[i] >= 0 ? target[(size_t) s * n + pRow[i]] : staticP[i];
            bool better = best < 0 || v > bestV
                    || (v == bestV && (p > bestP || (p == bestP && idRank[i] < bestId)));
            if (better) {
                best = i;
                bestV = v;
                bestP = p;
                bestId = idRank[i];
            }
        } else if (limitType[st] == 0 && (rlRemote != nullptr && rlRemote[i] == 2 ? m == 0
                                          : remote ? m == 1 : (m == 0 || m == 2))) {
            remaining++;                                     // still PV after this pass
        }
    }
    if (cand == 0 || remaining > 0 || best < 0) return;
    size_t st = (size_t) s * nrl + best;
    size_t idx = (size_t) s * n + rlVrow[best];
    mode[idx] = (rlRemote != nullptr && rlRemote[best] == 2) ? 0
              : (rlRemote != nullptr && rlRemote[best]) ? 1 : savedMode[st];
    target[idx] = savedTarget[st];
    limitType[st] = 0;
    switchCount[st]--;
    passSwitched[st] = 0;
    char any = 0;
    for (int i = 0; i < nrl && !any; i++) {
        any = passSwitched[(size_t) s * nrl + i] != 0;
    }
    switched[s] = any;
}










__global__ void negateIntoOne(int n, const double* F, double* b) {       // single scenario slice
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = -F[i];
}

__global__ void axpy1One(int n, const double* dx, double* x) {           // single scenario slice
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += dx[i];
}

// r = b - J dx per scenario (the refinement residual; the right-hand side as SOLVED, row repairs included)
__global__ void residualBatch(int n, int nS, int nnz, const int* rowPtr, const int* colIdx,
                              const double* vals, const double* dx, const double* b, double* r) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * nS) return;
    int s = t / n;
    int i = t - s * n;
    const double* v = vals + (size_t) s * nnz;
    const double* d = dx + (size_t) s * n;
    double acc = 0.0;
    for (int k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
        acc += v[k] * d[colIdx[k]];
    }
    r[t] = b[t] - acc;
}

// per scenario: ||r||inf / ||b||inf (0 for an inactive lane or a zero right-hand side)
__global__ void irScoreBatch(int n, int nS, const double* r, const double* b, const char* active, double* score) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nS) return;
    if (active != nullptr && !active[s]) {
        score[s] = 0.0;
        return;
    }
    const double* rs = r + (size_t) s * n;
    const double* bs = b + (size_t) s * n;
    double mr = 0.0, mb = 0.0;
    for (int i = 0; i < n; i++) {
        mr = fmax(mr, fabs(rs[i]));
        mb = fmax(mb, fabs(bs[i]));
    }
    score[s] = mb > 0.0 ? (isfinite(mr) ? mr / mb : 1e300) : 0.0;
}

__global__ void axpyLaneMaskBatch(int n, int total, double sign, const double* d, const char* lane, double* x) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;   // x += sign * d on the lanes whose mask is set
    if (t < total && lane[t / n]) x[t] += sign * d[t];
}

__global__ void negateIntoBatch(int total, const double* F, double* b) {  // b = -F over all scenarios
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) b[i] = -F[i];
}

// Warm-continuation reconcile: a SEEDED pinned controller (limitType != 0) must also carry its
// consequences — the BUS_V row flips to the reactive equation (mode 1) with the pinned Q target —
// else the solve holds the PV setpoint while the RL bookkeeping thinks the bus is at a limit.
// A PINNED pilot that is one of its own controllers (rlRemote 2) while a member of the same group
// (same compare row = the pilot's V) still controls: OLF then keeps the pilot's V identity AND
// activates the pilot's own Q balance, hosting it on the reference member's row - a relocation this
// group has no table for on the device. Flag the scenario; the host rejects it to the CPU replay.
// The other mixed state (members pinned, pilot controlling) is an ordinary layout: pilot identity,
// members at their own Q. One thread per (scenario, class-2 entry).
__global__ void rlPilotSelfMixedGate(int nrl, int nScenarios, const char* rlRemote, const int* rlCmpVrow,
                                     const char* limitType, const char* rlIslanded, char* bad,
                                     const int* psEntry, int nQr,
                                int nDqMembers, const int* dqMemberRlEntry, const char* dqSub) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nrl * nScenarios) return;
    int i = t % nrl;
    int s = t / nrl;
    if (rlRemote[i] != 2 || limitType[(size_t) s * nrl + i] == 0) return;   // only a PINNED pilot
    // ...that is still ON this component: an islanded pilot keeps its base limit type but has no pin and no
    // balance (OLF removed the bus), and an islanded member is not "still controlling" either.
    if (rlIslanded != nullptr && rlIslanded[(size_t) s * nrl + i]) return;
    if (isEnrolledPilot(psEntry, nQr, i)) return;            // enrolled (rotation): the carrier hosts its balance
    for (int j = 0; j < nrl; j++) {                          // ... with a member still controlling
        if (j != i && rlCmpVrow[j] == rlCmpVrow[i] && rlRemote[j] != 0
                && limitType[(size_t) s * nrl + j] == 0
                && !(rlIslanded != nullptr && rlIslanded[(size_t) s * nrl + j])) {
            if (dqSub != nullptr && dqMemberRlEntry != nullptr) {   // a member of ANOTHER subgroup is not this
                bool other = false;                                  // pilot's controller any more (tasks #71)
                for (int m = 0; m < nDqMembers; m++) {
                    if (dqMemberRlEntry[m] == j && dqSub[(size_t) s * nDqMembers + m] != 0) { other = true; break; }
                }
                if (other) continue;
            }
            bad[s] = 1;
            return;
        }
    }
}

__global__ void seedRlModeFromState(int nrl, int nScenarios, int n, const int* rlVrow, const int* rlQrow,
                                    const double* rlMinQ, const double* rlMaxQ, const double* rlLoadQ,
                                    const char* limitType, int* mode, double* target, int cvStride,
                                    const char* rlRemote, double* savedTarget, char* savedMode,
                                    const char* rlBaseLt, const int* psEntry, int nQr,
                                    const double* rlVtarget, int nRlVtOv, const int* rlVtOvSlotIdx, const double* rlVtOvVal,
                                    const double* svcRlVt, const char* rlLocalOv, const double* rlLocalVt,
                                    const double* pinTargetIn, const char* rlIslanded) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nrl * nScenarios) return;
    int i = tid % nrl;
    int s = tid / nrl;
    // A controller this scenario islanded: its bus is out of the system and its rows are FROZEN identities. Stamping
    // the continued pin turned its V row into a reactive balance at the frozen target, an island equation OLF does not
    // have, which the first solve then iterated on (rte6515 LINE-3477-2893: 0.0075 MW on every strategy that moves P).
    if (rlIslanded != nullptr && rlIslanded[(size_t) s * nrl + i]) return;
    char lt = limitType[(size_t) s * nrl + i];
    // A LOCAL controller pinned in the BASE case that the previous solve released (its exported state says PV): the
    // base activation still has it pinned, and OLF's strategy starts with the bus back on voltage control, exactly as
    // the reactive-limit switch-back left it - the PV form (6 on a cross row) at its voltage target. Left alone it
    // stayed pinned at the base limit: a strategy pair of rte6515 LINE-3273-1812 solved two generators the parent had
    // released at their base Q limits, 0.016 kV at VL-2968 even for a strategy that changes nothing.
    if (lt == 0 && rlBaseLt != nullptr && rlBaseLt[i] >= 1 && rlBaseLt[i] <= 4 && (rlRemote == nullptr || !rlRemote[i])
            && !isEnrolledPilot(psEntry, nQr, i)) {
        size_t st = (size_t) s * nrl + i;
        bool localOv = rlLocalOv != nullptr && rlLocalOv[st];
        int vrow = rlVrow[i];
        int pinRow = vrow;
        if (localOv && rlQrow != nullptr && rlQrow[i] >= 0 && rlQrow[i] != vrow) {
            size_t r2 = (size_t) s * n + rlQrow[i];
            if (mode[r2] == 6 || (mode[r2] == 4 && mode[(size_t) s * n + vrow] != 4)) {
                pinRow = rlQrow[i];
            }
        }
        size_t pidx = (size_t) s * n + pinRow;
        if (mode[pidx] == 1 || mode[pidx] == 4) {            // still holding the base pin
            mode[pidx] = pinRow != vrow ? 6 : 0;
            target[pidx] = localOv ? rlLocalVt[st] : rlTargetOf(nRlVtOv, rlVtOvSlotIdx, rlVtOvVal, svcRlVt, nrl, s, i, rlVtarget[i]);
        }
        return;
    }
    // ...and the same for a REMOTE member, as the reactive-limits pass releases one: its row goes back to its unpinned
    // form - the DISTR_Q share (mode 1), whose target the DISTR_Q pass recomputes, or a self-controlling pilot's voltage
    // identity (mode 0). Left pinned at the base limit, its group shared the pilot's reactive balance one member short
    // (network.xiidm LLJP, the member on bus 5779: mode 4 at -0.2954 where the parent had released it, 5.46 kV).
    if (lt == 0 && rlBaseLt != nullptr && rlBaseLt[i] >= 1 && rlBaseLt[i] <= 4 && rlRemote != nullptr && rlRemote[i]
            && !isEnrolledPilot(psEntry, nQr, i)) {
        int vrow = rlVrow[i];
        int qrow = rlQrow != nullptr && rlQrow[i] >= 0 && rlQrow[i] != vrow && mode[(size_t) s * n + rlQrow[i]] != 0
                ? rlQrow[i] : vrow;
        size_t qidx = (size_t) s * n + qrow;
        if (mode[qidx] == 4) {                               // still holding the base pin
            mode[qidx] = rlRemote[i] == 2 ? 0 : 1;
            target[qidx] = rlRemote[i] == 2
                    ? rlTargetOf(nRlVtOv, rlVtOvSlotIdx, rlVtOvVal, svcRlVt, nrl, s, i, rlVtarget[i]) : 0.0;
        }
        return;
    }
    // An entry that does not OWN the pin on its V row must not have one stamped there: its balance lives on
    // another row (or that row carries another bus's), and the row it does own was already set correctly by
    // the base activation. RL_LIMIT_FROZEN alone is not enough to skip on, because the run-time freeze uses
    // the same type on entries that DO own their row and must be restored here on a resumed solve.
    if (rlBaseLt != nullptr && rlBaseLt[i] == RL_BASE_RELOCATED) return;
    if (isEnrolledPilot(psEntry, nQr, i)) return;           // its pin never lives on its own V row (rotation)
    if (lt != 0) {
        // the pin's row (see dRlQrow), with the relocation's per-scenario diversion: mode 0 at the relocated row
        // means its coupler is open this scenario and the equations are back on the bus's own rows
        int seedRow = rlQrow != nullptr ? rlQrow[i] : rlVrow[i];
        if (seedRow != rlVrow[i] && mode[(size_t) s * n + seedRow] == 0) {
            seedRow = rlVrow[i];
        }
        size_t idx = (size_t) s * n + seedRow;
        size_t li = cvStride ? (size_t) s * nrl + i : (size_t) i;
        // What the row looked like BEFORE this seed is what a later switch-back restores. Leaving it
        // unwritten let the restore read whatever the previous solve left in this slot: the buffers are
        // per (entry, SLOT) and a context is reused chunk after chunk, so an entry pinned at seed time
        // and unpinned during the contingency took another scenario's target. Invisible wherever the
        // row's target is recomputed right after (a DISTR_Q share), a small wrong answer where it is
        // not — measured on network.xiidm as an intermittent 8.178e-05 kV shared by every scenario the
        // stale slot reached (the KUMF / KUMD pair swapping reactive power).
        savedTarget[(size_t) s * nrl + i] = target[idx];
        savedMode[(size_t) s * nrl + i] = static_cast<char>(mode[idx]);
        // a pinned REMOTE member's row is mode 4 (own Q balance), a pinned local one mode 1
        mode[idx] = (rlRemote != nullptr && rlRemote[i]) ? 4 : 1;
        // the pinned target the previous solve ENDED with, when it exported one: OLF restores the bus's generation
        // target Q from the parent's end, which is not the current limit when that limit moved by less than the
        // re-freeze tolerance (rte6515 LINE-3273-1812, VL-6095: 0.7518298 frozen, the curve at 0.7518918)
        double pin = pinTargetIn != nullptr ? pinTargetIn[(size_t) s * nrl + i] : NAN;
        target[idx] = !isnan(pin) ? pin : (lt == 1 ? rlMinQ[li] : rlMaxQ[li]) - rlLoadQ[li];
    }
}

// The pinned target of every pinned entry at the end of a solve, read on the row the seed above writes it to (NaN for
// an unpinned entry): what a continued solve (a strategy pair) restores, as OLF restores the parent's target Q.
__global__ void exportRlPinTargets(int nrl, int nScenarios, int n, const int* rlVrow, const int* rlQrow,
                                   const char* limitType, const int* mode, const double* target, const char* rlIslanded,
                                   int nQr, const int* psEntry, const int* qrPtr, const int* qrRows, const char* veqOff,
                                   const int* qrDepBase, double* out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nrl * nScenarios) return;
    int i = tid % nrl;
    int s = tid / nrl;
    char lt = limitType[(size_t) s * nrl + i];
    // an enrolled pilot's pin is its group's flip: the flipped row's target when the group is flipped (what a
    // continued solve's seed flip restores), nothing otherwise (its balance rides a carrier, never its own rows)
    for (int g = 0; psEntry != nullptr && g < nQr; g++) {
        if (psEntry[g] != i) continue;
        bool flipped = lt != 0 && veqOff != nullptr && veqOff[(size_t) s * nQr + g] && (qrDepBase == nullptr || qrDepBase[g] < 0);
        out[tid] = flipped ? target[(size_t) s * n + qrRows[qrPtr[g]]] : NAN;
        return;
    }
    // an islanded controller's row holds its bus's frozen value, not a Q: a pair that brings the bus back seeds the
    // pin from its limit, the target the bus kept while it was out
    if (lt == 0 || (rlIslanded != nullptr && rlIslanded[(size_t) s * nrl + i])) {
        out[tid] = NAN;
        return;
    }
    int seedRow = rlQrow != nullptr ? rlQrow[i] : rlVrow[i];
    if (seedRow != rlVrow[i] && mode[(size_t) s * n + seedRow] == 0) {
        seedRow = rlVrow[i];
    }
    out[tid] = target[(size_t) s * n + seedRow];
}

__global__ void axpy1Batch(int total, const double* dx, double* x) {      // x += dx over all scenarios
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) x[i] += dx[i];
}

// Per-scenario MAX_VOLTAGE_CHANGE step factor (mirrors MaxVoltageChangeStateVectorScaling, exactly
// like the base solver's host-side clamp but one factor PER SCENARIO): the largest |Δv| (varType 1)
// stays within maxDv and the largest |Δφ| (varType 0) within maxDphi; the WHOLE step is scaled.
// One block per scenario, block-stride column loop, shared-memory max reduction.
__global__ void maxVoltageChangeFactorBatch(int n, const double* dx, const int* varType,
                                            const char* active, double maxDv, double maxDphi,
                                            double* factor) {
    int s = blockIdx.x;
    __shared__ double smDv[BLOCK];
    __shared__ double smDphi[BLOCK];
    double mdv = 0.0;
    double mdphi = 0.0;
    if (active == nullptr || active[s]) {
        const double* d = dx + (size_t) s * n;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            double a = fabs(d[i]);
            if (varType[i] == 1) {
                mdv = fmax(mdv, a);
            } else if (varType[i] == 0) {
                mdphi = fmax(mdphi, a);
            }
        }
    }
    smDv[threadIdx.x] = mdv;
    smDphi[threadIdx.x] = mdphi;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            smDv[threadIdx.x] = fmax(smDv[threadIdx.x], smDv[threadIdx.x + off]);
            smDphi[threadIdx.x] = fmax(smDphi[threadIdx.x], smDphi[threadIdx.x + off]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        double f = 1.0;
        if (smDv[0] > maxDv) f = fmin(f, maxDv / smDv[0]);
        if (smDphi[0] > maxDphi) f = fmin(f, maxDphi / smDphi[0]);
        factor[s] = f;
    }
}

__global__ void axpy1BatchScaled(int n, int total, const double* dx, const double* factor, double* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) x[i] += factor[i / n] * dx[i];
}

// Per-scenario convergence norms on DEVICE: out[s*3] = ||F||inf, out[s*3+1] = sum(F^2),
// out[s*3+2] = finite flag — so the per-iteration readback is nS*3 doubles instead of the full
// nS*n mismatch (70 MB at 500 x 18k, pageable ~5.5 GB/s + a 9.2M-element host loop). infnorm is
// exact vs the host loop; the ssq PARALLEL summation order differs from the host's sequential
// sum by last-ulp amounts — borderline ||F||2 stopping decisions can flip at ulp level (same
// class of effect as any reduction-order change; the criterion itself is unchanged).
// OLF_GPU_WORST_ROW=1 (with OLF_GPU_BATCH_TRACE): name the row carrying the largest residual at a
// solve's start. A norm says HOW FAR the state is; only the row says WHY — an identity row sitting at
// an exact 1.0 is a different animal from a power row at 1.0.
__global__ void worstRowBatch(int n, const double* F, const int* mode, const double* target,
                              const double* x, int s) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int bi = -1;
    double bv = -1.0;
    for (int i = 0; i < n; i++) {
        double v = fabs(F[(size_t) s * n + i]);
        if (v > bv) {
            bv = v;
            bi = i;
        }
    }
    if (bi >= 0) {
        printf("[WORST_ROW] s=%d row=%d |F|=%.12f mode=%d target=%.9f x=%.9f\n",
               s, bi, bv, mode[(size_t) s * n + bi], target[(size_t) s * n + bi], x[(size_t) s * n + bi]);
    }
}

__global__ void residualNormsBatch(int n, const double* F, const char* active, double* out,
                                   const double* thrIdent, const double* thrPower, const int* mode,
                                   const double* thrAlt, const int* thrSel) {
    int s = blockIdx.x;
    __shared__ double smInf[BLOCK];
    __shared__ double smSsq[BLOCK];
    __shared__ double smRat[BLOCK];
    __shared__ int smFin[BLOCK];
    double mInf = 0.0;
    double ssq = 0.0;
    double mRat = 0.0;
    int fin = 1;
    if (active == nullptr || active[s]) {
        const double* f = F + (size_t) s * n;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            double v = f[i];
            if (!isfinite(v)) fin = 0;
            mInf = fmax(mInf, fabs(v));
            ssq += v * v;
            if (thrIdent != nullptr) {
                // PER_EQUATION_TYPE: identity/slope rows (mode 0/2) use the row's identity-type
                // threshold; power rows (mode 1), the phase P row (mode 3) and the pinned
                // remote-member Q row (mode 4) the power-type one
                int m = mode[(size_t) s * n + i];
                double thr = (m == 1 || m == 3 || m == 4) ? thrPower[i] : thrIdent[i];
                if (thrAlt != nullptr) {
                    // the row's equation in THIS lane: a cross identity (mode 6) is a voltage target; a
                    // relocated child's own row hosts its balance again once its coupler's dummy row (the
                    // selector) is an identity
                    if (m == 6) {
                        thr = thrAlt[i];
                    } else if ((m == 1 || m == 3 || m == 4) && thrSel[i] >= 0 && mode[(size_t) s * n + thrSel[i]] == 0) {
                        thr = thrAlt[i];
                    }
                }
                mRat = fmax(mRat, fabs(v) / thr);
            }
        }
    }
    smInf[threadIdx.x] = mInf;
    smSsq[threadIdx.x] = ssq;
    smRat[threadIdx.x] = mRat;
    smFin[threadIdx.x] = fin;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            smInf[threadIdx.x] = fmax(smInf[threadIdx.x], smInf[threadIdx.x + off]);
            smSsq[threadIdx.x] += smSsq[threadIdx.x + off];
            smRat[threadIdx.x] = fmax(smRat[threadIdx.x], smRat[threadIdx.x + off]);
            smFin[threadIdx.x] &= smFin[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[s * 4 + 0] = smInf[0];
        out[s * 4 + 1] = smSsq[0];
        out[s * 4 + 2] = smFin[0] ? 1.0 : 0.0;
        out[s * 4 + 3] = smRat[0];                           // max |F_i|/thr_i (0 without thresholds)
    }
}

// Batch COMPACTION (solve-only): gather the active scenarios' per-scenario blocks into a
// contiguous compact buffer (dst[c*per+i] = src[map[c]*per+i]) so cuDSS runs a NARROWER ubatch,
// then scatter the solution back to the full-width slots. Everything else (fills, norms,
// updates) stays full-width — retirement already makes those cheap; cuDSS width was the cost.
__global__ void gatherScenarios(long per, int nC, const int* map, const double* src, double* dst) {
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= per * nC) return;
    int c = (int) (i / per);
    dst[i] = src[(long) map[c] * per + i % per];
}

__global__ void scatterScenarios(long per, int nReal, const int* map, const double* srcC, double* dst) {
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= per * nReal) return;                             // pad slots (c >= nReal) are DISCARDED
    int c = (int) (i / per);
    dst[(long) map[c] * per + i % per] = srcC[i];
}

// Tile the base row (row 0, already in d[0..n)) across all nS scenarios: d[s*n+i] = d[i] for
// s in 1..nS-1. The base state/targets/row-mode are identical for every contingency, so the host
// uploads ONE n-row and this replicates it on-device — no 32x JNI marshalling / H2D of the base.
template <typename T>
__global__ void tileBaseRow(int total, int n, T* d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n && idx < total) {                       // rows 1..nS-1 copy from row 0; row 0 is the source
        d[idx] = d[idx % n];
    }
}

// Put nRows compact rows (src, row-major, n each) at their scenario slots of a [nS][n] buffer.
__global__ void scatterRowsInt(int nRows, int n, const int* slots, const int* src, int* dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nRows * n) {
        int r = idx / n;
        dst[(size_t) slots[r] * n + idx % n] = src[idx];
    }
}

// Make a RETIRED scenario's system an identity (diagonal 1, mismatch 0) so it stays
// well-posed inside the uniform batched factorization (a singular sub-block would fail the
// whole batch) and its state is frozen (dx = 0). A scenario retired THIS iteration still
// holds its filled values, so the row is zeroed before the diagonal is set — this is
// correct whether the scenario was just filled or has been inactive (zero) for a while.
__global__ void writeIdentityForInactive(int n, int nScenarios, int nnz, const int* rowPtr,
                                         const int* diagPos, const char* active, double* vals, double* F) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n * nScenarios) return;
    int r = tid % n;
    int s = tid / n;
    if (active[s]) return;
    double* valsS = vals + (size_t) s * nnz;
    for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
        valsS[k] = 0.0;
    }
    valsS[diagPos[r]] = 1.0;
    F[(size_t) s * n + r] = 0.0;
}

}  // namespace olfgpu

#include "diagnostics/matrix_scan.cuh"
#include "diagnostics/driver_traces.cuh"

namespace olfgpu {

static bool repairsZeroDiagonals(const BatchSolverContext* ctx) {
    return ctx->mode != 6;
}

static void setupBatchMatricesUbatch(BatchSolverContext* ctx) {
    int n = ctx->n;
    int nScenarios = ctx->nScenarios;
    // ---- UBATCH: ONE per-block CSR (n, nnz, shared rowPtr/colIdx) with batched value/RHS pointers;
    //      cuDSS analyzes a single block (cheap) and factorizes/solves all nS via UBATCH_SIZE.
    //      Modes 4/5 share this exact setup and differ only in the per-iteration factorize policy
    //      (mode 4 = factor at chunk iter 0 only; mode 5 = factor once + refactorize after). ----
    {
        int ub = nScenarios;
        CUDSS_CHECK(cudssConfigSet(ctx->ds.config, CUDSS_CONFIG_UBATCH_SIZE, &ub, sizeof(ub)));
        CUDSS_CHECK(cudssMatrixCreateCsr(&ctx->matUb, n, n, ctx->nnz,
                ctx->st->dRowPtr.as<int>(), nullptr, ctx->st->dColIdx.as<int>(), ctx->dVals.as<double>(),
                OLF_CUDSS_CSR_TYPES,
                CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        CUDSS_CHECK(cudssMatrixCreateDn(&ctx->matUbB, n, 1, n, ctx->dB.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssMatrixCreateDn(&ctx->matUbX, n, 1, n, ctx->dDx.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        // adaptive refinement: the residual and the correction, the same single-block dense shape as B and X
        if (!ctx->dR.p) {
            ctx->dR.alloc(static_cast<size_t>(n) * ctx->nScenarios * sizeof(double));
            ctx->dDelta.alloc(static_cast<size_t>(n) * ctx->nScenarios * sizeof(double));
            // zeroed: a solve covers UBATCH_SIZE = nScenarios lanes while a short last chunk fills nS < nScenarios
            // of them - the padding lanes' residual must be a clean 0, not whatever the allocation held
            CUDA_CHECK(cudaMemset(ctx->dR.p, 0, static_cast<size_t>(n) * ctx->nScenarios * sizeof(double)));
            CUDA_CHECK(cudaMemset(ctx->dDelta.p, 0, static_cast<size_t>(n) * ctx->nScenarios * sizeof(double)));
        }
        if (!ctx->dIrScore.p) {
            ctx->dIrScore.alloc(static_cast<size_t>(ctx->nScenarios) * sizeof(double));
        }
        CUDSS_CHECK(cudssMatrixCreateDn(&ctx->matUbR, n, 1, n, ctx->dR.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssMatrixCreateDn(&ctx->matUbDelta, n, 1, n, ctx->dDelta.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
    }
}

static void setupBatchMatricesFallback(BatchSolverContext* ctx) {
    int n = ctx->n;
    // ---- fallback: a single matrix repointed per scenario inside the solve loop ----
    CUDSS_CHECK(cudssMatrixCreateCsr(&ctx->ds.matA, n, n, ctx->nnz,
                                     ctx->st->dRowPtr.as<int>(), nullptr, ctx->st->dColIdx.as<int>(), ctx->dVals.as<double>(),
                                     OLF_CUDSS_CSR_TYPES,
                                     CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
    CUDSS_CHECK(cudssMatrixCreateDn(&ctx->ds.matB, n, 1, n, ctx->dB.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(&ctx->ds.matX, n, 1, n, ctx->dDx.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
}

BatchSolverContext* createBatchContext(const SharedStructurePtr& st, int nScenarios, int mode,
                                       int nrl, const int* rlVrow, const double* rlMinQ,
                                       const double* rlMaxQ, const double* rlLoadQ, const double* rlVtarget,
                                       const signed char* rlMonitored, const double* rowSlope,
                                       int nds, const int* dsPhiRow, const double* dsFactor,
                                       int dsSlackPhiRow, double dsSlackTargetP,
                                       const double* dsBaseTargetP, const double* dsMaxDelta, const double* dsMinDelta,
                                       int dsMode, const double* dsMarginUp, const double* dsMarginDown,
                                       double dsSlackMarginUp, double dsSlackMarginDown) {
    auto ctx = std::make_unique<BatchSolverContext>();
    ctx->st = st;
    int n = st->n;
    ctx->n = n;
    ctx->nnz = st->nnz;
    ctx->nScenarios = nScenarios;
    ctx->mode = mode;
    ctx->nElements = st->nElements;
    ctx->cbOffset = st->cbOffset;
    ctx->obOffset = st->obOffset;
    ctx->shOffset = st->shOffset;
    ctx->hvOffset = st->hvOffset;
    ctx->ndq = st->ndq;
    ctx->ndr = st->ndr;
    ctx->ndb = st->ndb;
    ctx->nzi = st->nzi;
    const std::vector<int>& rowMap = st->rowMap;
    const std::vector<int>& rowPtr = st->rowPtr;
    const std::vector<int>& colIdx = st->colIdx;

    // ---- per-scenario device state ----
    size_t nS = static_cast<size_t>(n) * nScenarios;
    size_t nnzS = static_cast<size_t>(ctx->nnz) * nScenarios;
    ctx->dX.alloc(nS * sizeof(double));
    ctx->dXLastConv.alloc(nS * sizeof(double));
    ctx->dTarget.alloc(nS * sizeof(double));
    ctx->dF.alloc(nS * sizeof(double));
    ctx->dB.alloc(nS * sizeof(double));
    ctx->dDx.alloc(nS * sizeof(double));
    ctx->dVals.alloc(nnzS * sizeof(double));
    ctx->dNorms.alloc(static_cast<size_t>(nScenarios) * 4 * sizeof(double));
    ctx->dMode.alloc(nS * sizeof(int));
    ctx->dDisabled.alloc(static_cast<size_t>(ctx->nElements) * nScenarios * sizeof(char));
    ctx->dActive.alloc(static_cast<size_t>(nScenarios) * sizeof(char));

    // ---- reactive-limits pack: the controller BUS_V rows mapped through the diagonal-pairing
    //      permutation (= identity in the batch path, no families, but mapped for correctness) ----
    ctx->nrl = nrl;
    ctx->dQbuf.alloc(static_cast<size_t>(n) * nScenarios * sizeof(double));
    ctx->dRlActive.alloc(static_cast<size_t>(nScenarios) * sizeof(char));
    ctx->dRlSwitched.alloc(static_cast<size_t>(nScenarios) * sizeof(char));
    // The seed launch of updatePilotVeqBatch / updateDistrQTargetsBatch (before the first solve) READS
    // dRlActive and relies on "not active yet" = 0; the RL pass is what first writes it. Explicit, not
    // left to the allocator's zeroing.
    CUDA_CHECK(cudaMemset(ctx->dRlActive.p, 0, static_cast<size_t>(nScenarios) * sizeof(char)));
    CUDA_CHECK(cudaMemset(ctx->dRlSwitched.p, 0, static_cast<size_t>(nScenarios) * sizeof(char)));
    // OUTSIDE the `nrl > 0` block: a run with reactive limits DISABLED has no entries at all, and that is
    // exactly when every pilot-self group needs the gate — its pilot can never pin, so it controls throughout.
    // Built inside, the gate did not exist for such a run: the pilots' raw q was emitted at their carrier rows
    // on top of the DISTR_Q terms and rte6515's base (limits off) started at |f(x0)| 117.908 against OLF's
    // 117.845 and never converged.
    // Derive the PILOT-SELF groups from the candidate table itself: the fallback row repeated among the
    // member rows means the controlled bus is one of its own controllers. The pilot's RL entry is the one
    // whose (row-mapped) V row is that row; without an entry the pilot can never pin and the group needs
    // no gate. nQr is small (groups), nrl a few thousand, and this runs once per context.
    int nQrG = ctx->st != nullptr ? ctx->st->nQr : 0;
    if (nQrG > 0) {
        std::vector<int> psEntry(nQrG, -1);
        for (int g = 0; g < nQrG; g++) {
            int p0 = ctx->st->hQrPtr[g];
            int cnt = ctx->st->hQrPtr[g + 1] - p0;
            int fallbackRow = ctx->st->hQrRows[p0];
            bool pilotSelf = false;
            for (int c = 1; c < cnt; c++) {
                if (ctx->st->hQrRows[p0 + c] == fallbackRow) {
                    pilotSelf = true;
                    break;
                }
            }
            if (!pilotSelf) {
                continue;                                // not a pilot-self group: -1, no gate (raw q as always)
            }
            // PS_NO_ENTRY (-2), not -1: a pilot-self pilot with NO reactive-limit entry can never be PINNED
            // (limits off for the run, or an empty q range), so it controls for the whole solve and its raw q
            // must ALWAYS be skipped — the gate is needed MORE, not less, than for a pinnable one. Read as
            // "no gate" (-1, the non-pilot-self value) it was emitted at the carrier row on top of the very
            // DISTR_Q terms it feeds: with reactive limits disabled on rte6515 the base started at
            // |f(x0)| 117.908 against OLF's 117.845 and the device's load flow never converged.
            psEntry[g] = PS_NO_ENTRY;
            ctx->hasPilotSelf = true;
            for (int i = 0; i < nrl; i++) {
                if (rowMap[rlVrow[i]] == fallbackRow) {
                    psEntry[g] = i;
                    break;
                }
            }
        }
        if (ctx->hasPilotSelf) {
            ctx->dPsPilotEntry.upload(psEntry.data(), psEntry.size() * sizeof(int));
            ctx->dPsSkip.alloc(static_cast<size_t>(nQrG) * ctx->nScenarios * sizeof(char));
            if (std::getenv("OLF_GPU_RL_TRACE") != nullptr) {
                int cnt = 0, noEntry = 0;
                for (int g = 0; g < nQrG; g++) {
                    cnt += psEntry[g] >= 0 ? 1 : 0;
                    noEntry += psEntry[g] == PS_NO_ENTRY ? 1 : 0;
                }
                std::fprintf(stderr, "PILOTSELF_GATE groups=%d(+%d never-pinnable) of %d nrl=%d\n",
                             cnt, noEntry, nQrG, nrl);
            }
        }
    }
    if (nrl > 0) {
        std::vector<int> rlVrowM(nrl);
        for (int i = 0; i < nrl; i++) {
            rlVrowM[i] = rowMap[rlVrow[i]];
        }
        ctx->dRlVrow.upload(rlVrowM.data(), nrl * sizeof(int));
        // A bus whose power equations the COUPLER-TREE relocation moved onto its parent coupler's dummy columns has
        // its reactive balance — and its BUS_TARGET_Q, which GpuAcActivationExtractor already puts there — at that
        // coupler's DUMMY_Q row, while its OWN V row carries the coupler's ZERO_V. The reactive-limit pass must read
        // q and flip the pin at THAT row; its VOLTAGE reads stay on the bus's own row (dRlVrow). Without the split
        // the check compared a voltage residual against reactive limits: measured on rte6515's full busbar scope
        // with the relocation enabled, entries read -78.0 and 4767.3 MVAr and 20 buses were pinned that OLF never
        // pins, after which the pinned solve burned all 30 iterations.
        std::vector<int> rlQrowM(rlVrowM);
        if (ctx->nzi > 0 && ctx->st->dZiIdx.p != nullptr) {
            std::vector<int> zidx(static_cast<size_t>(ctx->nzi) * 11);
            CUDA_CHECK(cudaMemcpy(zidx.data(), ctx->st->dZiIdx.p, zidx.size() * sizeof(int), cudaMemcpyDeviceToHost));
            std::map<int, int> qRowOfVRow;
            for (int e = 0; e < ctx->nzi; e++) {
                const int* z = zidx.data() + static_cast<size_t>(e) * 11;
                int childSide = (z[10] >> 1) & 3;
                int ownVCol = childSide == 1 ? z[6] : (childSide == 2 ? z[7] : -1);
                if (childSide == 0 || z[5] < 0 || ownVCol < 0) {
                    continue;
                }
                qRowOfVRow[rowMap[ownVCol]] = rowMap[z[5]];
            }
            int moved = 0;
            for (int i = 0; i < nrl; i++) {
                auto it = qRowOfVRow.find(rlVrowM[i]);
                if (it != qRowOfVRow.end()) {
                    rlQrowM[i] = it->second;
                    moved++;
                }
            }
            if (moved > 0 && std::getenv("OLF_GPU_RL_TRACE") != nullptr) {
                std::fprintf(stderr, "RL_QROW %d of %d entries follow the coupler relocation\n", moved, nrl);
            }
        }
        ctx->dRlQrow.upload(rlQrowM.data(), nrl * sizeof(int));
        ctx->dRlMinQ.upload(rlMinQ, nrl * sizeof(double));
        ctx->dRlMaxQ.upload(rlMaxQ, nrl * sizeof(double));
        ctx->dRlLoadQ.upload(rlLoadQ, nrl * sizeof(double));
        ctx->dRlVtarget.upload(rlVtarget, nrl * sizeof(double));
        ctx->dRlMonitored.upload(rlMonitored, nrl * sizeof(char));
        ctx->allPvPinnable = 1;
        for (int i = 0; i < nrl; i++) {
            if (!(rlMonitored[i] & 1) && !((rlMinQ[i] > -1e7 || rlMaxQ[i] < 1e7) && rlMinQ[i] < rlMaxQ[i])) {
                ctx->allPvPinnable = 0;                      // an unlimited PV bus: the rule never fires
            }
        }
        ctx->dRlSavedTarget.alloc(static_cast<size_t>(nrl) * nScenarios * sizeof(double));
        ctx->dRlPassSwitched.alloc(static_cast<size_t>(nrl) * nScenarios * sizeof(char));
        // OLF_GPU_SW_LOG=<slot>: record one scenario's reactive-limit decisions ON THE DEVICE (one
        // atomicAdd per decision, copied out once at the end) instead of reading the whole batch's
        // buffers over PCIe every pass. The host-side traces cost hundreds of MB per pass, which
        // serializes the stream and changes the very kernel scheduling whose effect is being traced:
        // a run-to-run flip disappears while you watch it.
        if (std::getenv("OLF_GPU_SW_LOG") != nullptr) {
            ctx->dSwLog.alloc(SW_LOG_CAP * SW_LOG_W * sizeof(double));
            ctx->dSwLogCnt.alloc(sizeof(int));
            CUDA_CHECK(cudaMemset(ctx->dSwLogCnt.p, 0, sizeof(int)));
        }
        ctx->dRlSavedMode.alloc(static_cast<size_t>(nrl) * nScenarios * sizeof(char));
        ctx->dRlLimitType.alloc(static_cast<size_t>(nrl) * nScenarios * sizeof(char));
        ctx->dRlSwitchCount.alloc(static_cast<size_t>(nrl) * nScenarios * sizeof(char));
        std::vector<double> svcZero(nrl, 0.0);              // neutral until setBatchRlSvcLimits installs
        ctx->dRlSvcBmin.upload(svcZero.data(), nrl * sizeof(double));
        ctx->dRlSvcBmax.upload(svcZero.data(), nrl * sizeof(double));
        ctx->dRlV2Base.upload(svcZero.data(), nrl * sizeof(double));
    }
    ctx->dRowSlope.upload(rowSlope, n * sizeof(double));
    ctx->nds = nds;
    ctx->dsSlackTargetP = dsSlackTargetP;
    ctx->dsMode = dsMode;
    ctx->dsSlackMarginUp = dsSlackMarginUp;
    ctx->dsSlackMarginDown = dsSlackMarginDown;
    if (dsMode == 1 && nds > 0) {
        ctx->dDsMarginUp.upload(dsMarginUp, nds * sizeof(double));
        ctx->dDsMarginDown.upload(dsMarginDown, nds * sizeof(double));
    }
    ctx->dsSlackPhiRow = dsSlackPhiRow >= 0 ? rowMap[dsSlackPhiRow] : -1;
    ctx->dsSlackOnly = (nds == 0 && ctx->dsSlackPhiRow >= 0);
    if (ctx->dsSlackOnly) {
        ctx->dDsSlackAbsorbed.alloc(static_cast<size_t>(nScenarios) * sizeof(double));
    }
    if (nds > 0) {
        std::vector<int> dsPhiRowM(nds);
        for (int i = 0; i < nds; i++) {
            dsPhiRowM[i] = rowMap[dsPhiRow[i]];
        }
        ctx->dDsPhiRow.upload(dsPhiRowM.data(), nds * sizeof(int));
        ctx->dDsFactor.upload(dsFactor, nds * sizeof(double));
        ctx->dDsBaseTargetP.upload(dsBaseTargetP, nds * sizeof(double));
        ctx->dDsMaxDelta.upload(dsMaxDelta, nds * sizeof(double));
        ctx->dDsMinDelta.upload(dsMinDelta, nds * sizeof(double));
        double factorSum = 0.0;
        for (int i = 0; i < nds; i++) {
            factorSum += dsFactor[i];
        }
        ctx->dsFactorSum = factorSum > 0 ? factorSum : 1.0;
        ctx->dDsRetain.alloc(static_cast<size_t>(nScenarios) * sizeof(double));
        ctx->dDsGateLast.alloc(static_cast<size_t>(nScenarios) * 8 * sizeof(double));   // see the gateLast write
        ctx->dDsRetainApplied.alloc(static_cast<size_t>(nScenarios) * sizeof(double));
    }

    // ---- cuDSS handle/config/data (shared by both solve paths) ----
    CUDA_CHECK(cudaStreamCreate(&ctx->stream));
    CUDSS_CHECK(cudssCreate(&ctx->ds.handle));
    CUDSS_CHECK(cudssSetStream(ctx->ds.handle, ctx->stream));
    CUDSS_CHECK(cudssConfigCreate(&ctx->ds.config));
    int ir = ctx->st->nzi > 0 ? irBaseSteps() : IR_N_STEPS;
    CUDSS_CHECK(cudssConfigSet(ctx->ds.config, CUDSS_CONFIG_IR_N_STEPS, &ir, sizeof(ir)));
    if (mode == 6) {
        applyRobustSingleMatrixConfig(ctx->ds.config);       // its own ordering: never set the ordering twice
    } else {
        applyReorderAlg(ctx->ds.config, ctx->st->nzi > 0);
    }
    CUDSS_CHECK(cudssDataCreate(ctx->ds.handle, &ctx->ds.data));

    if (mode == 3) {
        setupBatchMatricesUbatch(ctx.get());
    } else {
        setupBatchMatricesFallback(ctx.get());
    }
    return ctx.release();
}

// Per-scenario target overrides: each entry writes target[ovIndex[k]] = ovValue[k], where ovIndex is the
// FLAT dTarget index (scenarioSlot*n + row). Used by generator contingencies to set a scenario's bus P/Q
// target (remove the lost generator's P, set BUS_TARGET_Q on a PV→PQ flip) after the base row is tiled.
__global__ void applyTargetOverrideBatch(int nOv, const int* ovIndex, const double* ovValue, double* target) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < nOv) {
        target[ovIndex[k]] = ovValue[k];
    }
}

static void uploadBatchOverrides(BatchSolverContext* ctx, int lsMode, double maxDv, double maxDphi,
                                 const int* varType, int nShOv, const int* shOvSlotElem, const double* shOvGb,
                                 int nziBase, int nZiOff, const int* ziOffSlotElem,
                                 int nZiOn, const int* ziOnSlotElem,
                                 int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal) {
    int n = ctx->n;
    int nS = ctx->nScenarios;
    bool clampStep = lsMode == 2 && varType != nullptr;
    if (clampStep && !ctx->dVarType.p) {
        ctx->dVarType.upload(varType, static_cast<size_t>(n) * sizeof(int));
        ctx->dStepFactor.alloc(static_cast<size_t>(nS) * sizeof(double));
    }
    if (lsMode == 1) {                                       // LINE_SEARCH backtracking buffers
        if (!ctx->dXprevLs.p) ctx->dXprevLs.alloc(static_cast<size_t>(n) * nS * sizeof(double));
        if (!ctx->dStepFactor.p) ctx->dStepFactor.alloc(static_cast<size_t>(nS) * sizeof(double));
    }
    ctx->clampStep = clampStep;
    ctx->maxDv = maxDv;
    ctx->maxDphi = maxDphi;
    // Per-scenario shunt admittance overrides (section shifts / reconnection). PACK-indexed
    // elements (shunt family index, not the global element index). Empty = a 1-entry dummy
    // upload so the kernels' pointers are always valid.
    ctx->nShOv = nShOv;
    ctx->hShOvBaseSlotElem.assign(shOvSlotElem, shOvSlotElem + (size_t) nShOv * 2);
    ctx->hShOvBaseGb.assign(shOvGb, shOvGb + (size_t) nShOv * 2);
    if (nShOv > 0 || ctx->nShc > 0) {
        // room for the in-kernel shunt loop to append one (dG, dB) per (scenario, controller)
        size_t capEntries = std::max<size_t>(static_cast<size_t>(nShOv) + (ctx->nShc > 0 ? static_cast<size_t>(nS) * ctx->nShc : 0), 1);
        ctx->dShOvSlotElem.ensure(capEntries * 2 * sizeof(int));
        ctx->dShOvGb.ensure(capEntries * 2 * sizeof(double));
        if (nShOv > 0) {
            CUDA_CHECK(cudaMemcpy(ctx->dShOvSlotElem.p, shOvSlotElem, static_cast<size_t>(nShOv) * 2 * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(ctx->dShOvGb.p, shOvGb, static_cast<size_t>(nShOv) * 2 * sizeof(double), cudaMemcpyHostToDevice));
        }
    } else if (!ctx->dShOvSlotElem.p) {
        int dummyI[2] = {-1, -1};
        double dummyD[2] = {0, 0};
        ctx->dShOvSlotElem.upload(dummyI, sizeof(dummyI));
        ctx->dShOvGb.upload(dummyD, sizeof(dummyD));
    }
    ctx->nziBase = nziBase > 0 ? nziBase : ctx->nzi;         // default: every zi element always enabled
    ctx->nZiOff = nZiOff;
    if (nZiOff > 0) {
        ctx->dZiOffSlotElem.upload(ziOffSlotElem, static_cast<size_t>(nZiOff) * 2 * sizeof(int));
    } else if (!ctx->dZiOffSlotElem.p) {
        int dummyZiOff[2] = {-1, -1};
        ctx->dZiOffSlotElem.upload(dummyZiOff, sizeof(dummyZiOff));
    }
    ctx->nZiOn = nZiOn;
    if (nZiOn > 0) {
        ctx->dZiOnSlotElem.upload(ziOnSlotElem, static_cast<size_t>(nZiOn) * 2 * sizeof(int));
    } else if (!ctx->dZiOnSlotElem.p) {
        int dummyZi[2] = {-1, -1};
        ctx->dZiOnSlotElem.upload(dummyZi, sizeof(dummyZi));
    }
    if (ctx->nzi > 0) {                                      // the dense (scenario, element) forms (see dZiOffMask)
        size_t len = static_cast<size_t>(ctx->nScenarios) * ctx->nzi;
        std::vector<char> off(len, 0), on(len, 0);
        for (int k = 0; k < nZiOff; k++) {
            int sl = ziOffSlotElem[2 * k], e = ziOffSlotElem[2 * k + 1];
            if (sl >= 0 && sl < ctx->nScenarios && e >= 0 && e < ctx->nzi) off[(size_t) sl * ctx->nzi + e] = 1;
        }
        for (int k = 0; k < nZiOn; k++) {
            int sl = ziOnSlotElem[2 * k], e = ziOnSlotElem[2 * k + 1];
            if (sl >= 0 && sl < ctx->nScenarios && e >= 0 && e < ctx->nzi) on[(size_t) sl * ctx->nzi + e] = 1;
        }
        ctx->dZiOffMask.upload(off.data(), len);
        ctx->dZiOnMask.upload(on.data(), len);
        if (!ctx->dZiTreeMask.p) {
            std::vector<char> tree(len, (char) -1);
            ctx->dZiTreeMask.upload(tree.data(), len);
        }
        ctx->hasZiMasks = true;
        if (std::getenv("OLF_GPU_PHASE_TIME") != nullptr) {
            std::fprintf(stderr, "ZI_LISTS nzi=%d nScenarios=%d nZiOff=%d nZiOn=%d\n", ctx->nzi, ctx->nScenarios, nZiOff, nZiOn);
        }
    }
    ctx->nR1Ov = nR1Ov;
    ctx->hR1OvBaseSlotElem.assign(r1OvSlotElem, r1OvSlotElem + (size_t) nR1Ov * 2);
    ctx->hR1OvBaseVal.assign(r1OvVal, r1OvVal + (size_t) nR1Ov * 8);
    if (nR1Ov > 0 || ctx->nTvc > 0 || ctx->nFlc > 0 || ctx->nCph > 0) {
        // Room for the in-kernel TVC loop to merge one moved tap per (scenario, controller) in
        // front of the caller's list without reallocating mid-solve.
        size_t capEntries = static_cast<size_t>(nR1Ov) + static_cast<size_t>(nS) * (ctx->nTvc + ctx->nFlc + ctx->nCph + ctx->nTvd);
        capEntries = std::max<size_t>(capEntries, 1);
        ctx->dR1OvSlotElem.ensure(capEntries * 2 * sizeof(int));
        ctx->dR1OvVal.ensure(capEntries * 8 * sizeof(double));
        if (nR1Ov > 0) {
            CUDA_CHECK(cudaMemcpy(ctx->dR1OvSlotElem.p, r1OvSlotElem, static_cast<size_t>(nR1Ov) * 2 * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(ctx->dR1OvVal.p, r1OvVal, static_cast<size_t>(nR1Ov) * 8 * sizeof(double), cudaMemcpyHostToDevice));
        }
    } else if (!ctx->dR1OvSlotElem.p) {
        int dummyI[2] = {-1, -1};
        double dummyD[1] = {0};
        ctx->dR1OvSlotElem.upload(dummyI, sizeof(dummyI));
        ctx->dR1OvVal.upload(dummyD, sizeof(dummyD));
    }
}

// SecondaryVoltageControl's tryToReEnableHelpfulControllerBuses, applied to the DEVICE pin state at a
// driver pause: a zone whose controllers are only PARTLY at a reactive limit un-pins the limited ones so
// they can help the zone realign. OLF does it with setGeneratorVoltageControlEnabledAndRecomputeTargetQ
// (back to PV) + setQLimitType(null); here that is the row's saved PV form, the zone's CURRENT voltage
// target, and limitType 0. The host orchestrator got this for free by re-seeding rlStateIn on its own
// re-solve — a RESUME skips the RL seeding, so without this channel the un-pin never reached the device.
__global__ void rlUnpinBatch(int nPairs, int nS, int n, int nrl, const int* pairs, const double* newTarget,
                             const int* rlVrow, char* limitType, const char* savedMode, int* mode, double* target) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nPairs) return;
    int s = pairs[2 * k];
    int i = pairs[2 * k + 1];
    if (s < 0 || s >= nS || i < 0 || i >= nrl) return;
    size_t st = (size_t) s * nrl + i;
    if (limitType[st] == 0) return;                          // not pinned: nothing to give back
    // A member FROZEN by the generator voltage control manager is not a reactive-limit pin: giving it
    // back would restore a V identity on a row that holds the group's DISTR_Q share, and would undo a
    // decision the transformer phase owns and will undo itself at COMPLETE.
    if (limitType[st] == RL_LIMIT_FROZEN) return;
    size_t idx = (size_t) s * n + rlVrow[i];
    char sm = savedMode[st];
    mode[idx] = (sm == 2) ? 2 : 0;                           // the PV form saved at pin time (2 = slope SVC)
    target[idx] = newTarget[k];
    limitType[st] = 0;
}

















// Adaptive iterative refinement (see irBaseSteps / irMaxSteps / irTol), decided PER LANE. Each lane refines (r = b - J dx, J d = r,
// dx += d) while ITS OWN backward residual ratio is above tol and still falling, and undoes a step that did not
// lower it - what the same scenario does in a batch of one. The batched solve still runs over every lane; a lane
// that is done just does not take the correction. It used to be decided on the WORST lane and applied to all of
// them, so a scenario's dx depended on which matrices shared its batch: network.xiidm busbar JZDQ's first pass
// differed by 1.2e-8 pu between a 700- and a 175-wide chunk of the same full topology, and the outer loops carried
// that over the 1 MW slack-balance threshold - 0.125 MW off the CPU in one chunk, 1.7e-8 MW in another.
template <typename SolveFn>
static void refineLanes(BatchSolverContext* ctx, int lanes, int n, int nnz, const double* vals, double* dx,
                        const double* b, double* r, const double* delta, const char* active, SolveFn solve) {
    const int irMax = irMaxSteps();
    const int irBase = irBaseSteps();
    const double tol = irTol();
    std::vector<double> prev(lanes), sc(lanes);
    std::vector<char> lane(lanes), undo(lanes);
    auto scores = [&](std::vector<double>& out) {
        residualBatch<<<grid(n * lanes), BLOCK, 0, ctx->stream>>>(n, lanes, nnz, ctx->st->dRowPtr.as<int>(),
                ctx->st->dColIdx.as<int>(), vals, dx, b, r);
        irScoreBatch<<<grid(lanes), BLOCK, 0, ctx->stream>>>(n, lanes, r, b, active, ctx->dIrScore.as<double>());
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        CUDA_CHECK(cudaMemcpy(out.data(), ctx->dIrScore.as<double>(), lanes * sizeof(double), cudaMemcpyDeviceToHost));
        gIrChecks++;
    };
    scores(prev);
    int nLane = 0;
    for (int s = 0; s < lanes; s++) {
        lane[s] = prev[s] > tol;
        nLane += lane[s];
    }
    ctx->dIrLane.ensure(static_cast<size_t>(lanes));
    for (int step = irBase; step < irMax && nLane > 0; step++) {
        CUDA_CHECK(cudaMemcpy(ctx->dIrLane.p, lane.data(), lanes, cudaMemcpyHostToDevice));
        solve();                                              // delta = J^-1 r on every lane
        axpyLaneMaskBatch<<<grid(n * lanes), BLOCK, 0, ctx->stream>>>(n, n * lanes, 1.0, delta, ctx->dIrLane.as<char>(), dx);
        gIrExtraSteps++;
        scores(sc);
        int nUndo = 0;
        nLane = 0;
        for (int s = 0; s < lanes; s++) {
            undo[s] = 0;
            if (!lane[s]) continue;
            if (!(sc[s] < prev[s])) {                         // no progress on this lane: undo its step, it is done
                undo[s] = 1;
                lane[s] = 0;
                nUndo++;
            } else {
                prev[s] = sc[s];
                lane[s] = sc[s] > tol;
                nLane += lane[s];
            }
        }
        if (nUndo > 0) {
            CUDA_CHECK(cudaMemcpy(ctx->dIrLane.p, undo.data(), lanes, cudaMemcpyHostToDevice));
            axpyLaneMaskBatch<<<grid(n * lanes), BLOCK, 0, ctx->stream>>>(n, n * lanes, -1.0, delta, ctx->dIrLane.as<char>(), dx);
        }
    }
}

}  // namespace olfgpu

#include "outerloop/phase_control.cuh"

namespace olfgpu {

}  // namespace olfgpu
#include "outerloop/voltage_monitoring.cuh"
namespace olfgpu {

}  // namespace olfgpu
#include "outerloop/hvdc_ac_emulation.cuh"
namespace olfgpu {

}  // namespace olfgpu
#include "outerloop/distributed_slack.cuh"
namespace olfgpu {

}  // namespace olfgpu
#include "outerloop/sensitivity.cuh"
namespace olfgpu {



void fillActiveBatch(BatchSolverContext* ctx, DriverPass& drv, bool fOnly) {
    const int& n = drv.n;
    const int& nnz = drv.nnz;
    const int& nS = drv.nS;
    const int& outer = drv.outer;
    const bool& deviceFixedIter = drv.deviceFixedIter;
    if (ctx->hasPilotSelf && ctx->dPsSkip.p != nullptr && ctx->st->nQr > 0) {
        // Recompute the PILOT-SELF gate from the CURRENT pin states BEFORE any family fill: such a pilot
        // has a Q balance to relocate only once a reactive limit has pinned it, and the RL pass moves
        // pins between passes. The closed/open-branch and shunt fills are what route the pilot's raw q
        // through it (ziDynDest), so it must be current when THEY run — launched after them (before the
        // DISTR_Q fill only) the FIRST fill of every context read an unset gate and put the unpinned
        // pilot's raw q on the carrier row on top of its DISTR_Q terms: row 0 of the 4-bus
        // pilot-as-controller fixture assembled -1.451 against OLF's -5.715 (= OLF + 1.0 * q_pilot), and
        // rte6515's base started from |f(x0)| 117.908 against OLF's 117.845.
        pilotSelfGateBatch<<<grid(ctx->st->nQr * nS), BLOCK, 0, ctx->stream>>>(
                ctx->st->nQr, nS, ctx->nrl, ctx->dPsPilotEntry.as<int>(),
                ctx->dRlLimitType.as<char>(),
                ctx->nRlIslanded > 0 ? ctx->dRlIslanded.as<char>() : nullptr, ctx->dPsSkip.as<char>());
    }
    if (!fOnly) {
        CUDA_CHECK(cudaMemsetAsync(ctx->dVals.p, 0, static_cast<size_t>(nnz) * nS * sizeof(double), ctx->stream));
    }
    initFBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, ctx->dTarget.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>());
    if (ctx->st->cb.ne) {
        if (fOnly) {
            fillClosedBranchBatchF<<<grid(ctx->st->cb.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->cb.ne, nS, n, nnz, ctx->nElements, ctx->cbOffset, ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->st->cb.dest.as<int>(), ctx->dX.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(), ctx->st->cb.altDest.as<int>(), ctx->st->cb.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->cb.dynReloc ? 1 : 0, ctx->st->cb.phaseCtl ? ctx->st->cb.pcDest.as<int>() : nullptr, ctx->st->cb.phaseCtl ? ctx->st->cb.pcSel.as<int>() : nullptr, withPilotSelf(relocOf(ctx->st->cb, ctx->st.get()), ctx));
        } else {
            fillClosedBranchBatch<<<grid(ctx->st->cb.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->cb.ne, nS, n, nnz, ctx->nElements, ctx->cbOffset, ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->st->cb.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(), ctx->st->cb.altDest.as<int>(), ctx->st->cb.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->cb.dynReloc ? 1 : 0, ctx->st->cb.phaseCtl ? ctx->st->cb.pcDest.as<int>() : nullptr, ctx->st->cb.phaseCtl ? ctx->st->cb.pcSel.as<int>() : nullptr, withPilotSelf(relocOf(ctx->st->cb, ctx->st.get()), ctx));
        }
    }
    {
        static const char* cbProbe = std::getenv("OLF_GPU_CB_PROBE");
        static bool cbProbed = false;
        if (cbProbe != nullptr && !cbProbed && nS > 1 && ctx->st->cb.ne) {
            cbProbed = true;
            cbRowProbe<<<1, 1, 0, ctx->stream>>>(nS, std::atoi(cbProbe), ctx->st->cb.in.as<double>(),
                    ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(),
                    ctx->nElements, ctx->cbOffset, ctx->dDisabled.as<char>(), ctx->dActive.as<char>());
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        }
    }
    if (ctx->st->ob.ne) {
        if (fOnly) {
            fillOpenBranchBatchF<<<grid(ctx->st->ob.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->ob.ne, nS, n, nnz, ctx->nElements, ctx->obOffset, ctx->st->ob.in.as<double>(), ctx->st->ob.ix.as<int>(), ctx->st->ob.dest.as<int>(), ctx->dX.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->st->ob.altDest.as<int>(), ctx->st->ob.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->ob.dynReloc ? 1 : 0, withPilotSelf(relocOf(ctx->st->ob, ctx->st.get()), ctx));
        } else {
            fillOpenBranchBatch<<<grid(ctx->st->ob.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->ob.ne, nS, n, nnz, ctx->nElements, ctx->obOffset, ctx->st->ob.in.as<double>(), ctx->st->ob.ix.as<int>(), ctx->st->ob.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->st->ob.altDest.as<int>(), ctx->st->ob.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->ob.dynReloc ? 1 : 0, withPilotSelf(relocOf(ctx->st->ob, ctx->st.get()), ctx));
        }
    }
    if (ctx->st->sh.ne) {
        if (fOnly) {
            fillShuntBatchF<<<grid(ctx->st->sh.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->sh.ne, nS, n, nnz, ctx->nElements, ctx->shOffset, ctx->st->sh.in.as<double>(), ctx->st->sh.ix.as<int>(), ctx->st->sh.dest.as<int>(), ctx->dX.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nShOv, ctx->dShOvSlotElem.as<int>(), ctx->dShOvGb.as<double>(), ctx->st->sh.altDest.as<int>(), ctx->st->sh.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->sh.dynReloc ? 1 : 0, withPilotSelf(relocOf(ctx->st->sh, ctx->st.get()), ctx));
        } else {
            fillShuntBatch<<<grid(ctx->st->sh.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->sh.ne, nS, n, nnz, ctx->nElements, ctx->shOffset, ctx->st->sh.in.as<double>(), ctx->st->sh.ix.as<int>(), ctx->st->sh.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nShOv, ctx->dShOvSlotElem.as<int>(), ctx->dShOvGb.as<double>(), ctx->st->sh.altDest.as<int>(), ctx->st->sh.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->sh.dynReloc ? 1 : 0, withPilotSelf(relocOf(ctx->st->sh, ctx->st.get()), ctx));
        }
    }
    // voltage-dependent load model: own guarded block (NOT inside the shunt guard — a modeled
    // load needs no shunt present); residual always, Jacobian slot unless the F-only pass
    if (ctx->st->nvdl) {
        fillVdlBatch<<<grid(ctx->st->nvdl * nS), BLOCK, 0, ctx->stream>>>(ctx->st->nvdl, nS, n, nnz, ctx->st->dVdlIn.as<double>(), ctx->st->dVdlIdx.as<int>(), ctx->st->dVdlDest.as<int>(), ctx->dX.as<double>(), fOnly ? nullptr : ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->st->vdlDyn ? ctx->st->dVdlAltRow.as<int>() : nullptr, ctx->st->vdlDyn ? ctx->st->dVdlAltDest.as<int>() : nullptr, ctx->st->vdlDyn ? ctx->dMode.as<int>() : nullptr);
    }
    if (ctx->st->hv.ne) {
        if (fOnly) {
            fillHvdcBatchF<<<grid(ctx->st->hv.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->hv.ne, nS, n, nnz, ctx->nElements, ctx->hvOffset, ctx->st->hv.in.as<double>(), ctx->st->hv.ix.as<int>(), ctx->st->hv.dest.as<int>(), ctx->dX.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->st->hv.altDest.as<int>(), ctx->st->hv.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->hv.dynReloc ? 1 : 0);
        } else {
            fillHvdcBatch<<<grid(ctx->st->hv.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->hv.ne, nS, n, nnz, ctx->nElements, ctx->hvOffset, ctx->st->hv.in.as<double>(), ctx->st->hv.ix.as<int>(), ctx->st->hv.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->st->hv.altDest.as<int>(), ctx->st->hv.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->hv.dynReloc ? 1 : 0);
        }
    }
int dqTraceRow = gpuDqTraceRow();
if (ctx->ndq) fillDistrQBatch<<<grid(ctx->ndq * nS), BLOCK, 0, ctx->stream>>>(ctx->ndq, nS, n, nnz, ctx->nElements, ctx->cbOffset, ctx->obOffset, ctx->st->dDqElem.as<int>(), ctx->st->dDqSide.as<int>(), ctx->st->dDqKind.as<int>(), ctx->st->dDqWeight.as<double>(), ctx->st->dDqDest.as<int>(), ctx->st->dDqAlt.as<int>(), ctx->st->dDqSel.as<int>(), ctx->st->dDqZiRow.as<int>(), ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->st->sh.in.as<double>(), ctx->st->sh.ix.as<int>(), ctx->st->ob.in.as<double>(), ctx->st->ob.ix.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nDqOv, ctx->nDqOv ? ctx->dDqOvIdx.as<int>() : nullptr, ctx->nDqOv ? ctx->dDqOvW.as<double>() : nullptr,
            ctx->nDqKeyOv, ctx->nDqKeyOv ? ctx->dDqKeyOvIdx.as<int>() : nullptr, ctx->nDqKeyOv ? ctx->dDqKeyOvKey.as<double>() : nullptr, ctx->dMode.as<int>(), ctx->nDqInact,
            ctx->nDqInact ? ctx->dDqInact.as<int>() : nullptr,
            ctx->nDqDyn ? ctx->dDqKMember.as<int>() : nullptr, ctx->nDqDyn || ctx->hasDqQrGroups ? ctx->dDqJMember.as<int>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberKey.as<double>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberRlEntry.as<int>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberGroupStart.as<int>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberGroupCount.as<int>() : nullptr, ctx->dRlLimitType.as<char>(), ctx->nrl, ctx->hasPilotSelf && ctx->dPsSkip.p != nullptr ? ctx->dPsSkip.as<char>() : nullptr, ctx->st->nQr, ctx->hasDqQrGroups ? ctx->dDqMemberQrGroup.as<int>() : nullptr, ctx->st->nQr > 0 ? ctx->st->dQrPtr.as<int>() : nullptr, ctx->st->nQr > 0 ? ctx->st->dQrRows.as<int>() : nullptr, ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(), ctx->nShOv, ctx->dShOvSlotElem.as<int>(), ctx->dShOvGb.as<double>(), dqTraceRow,
                ctx->hasDqSub ? ctx->dDqSub.as<char>() : nullptr, ctx->hasDqSub ? ctx->dQrMemberStart.as<int>() : nullptr, ctx->nDqMembers);
    if (ctx->ndr) fillDistrRhoBatch<<<grid(ctx->ndr * nS), BLOCK, 0, ctx->stream>>>(ctx->ndr, nS, n, nnz, ctx->st->dDrCol.as<int>(), ctx->st->dDrCoef.as<double>(), ctx->st->dDrDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->nDrOv, ctx->nDrOv ? ctx->dDrOvIdx.as<int>() : nullptr, ctx->nDrOv ? ctx->dDrOvC.as<double>() : nullptr, ctx->st->drQr ? ctx->st->dDrQrGroup.as<int>() : nullptr, ctx->dMode.as<int>(), ctx->st->dQrPtr.as<int>(), ctx->st->dQrRows.as<int>());
    if (ctx->ndb) fillDistrShuntBBatch<<<grid(ctx->ndb * nS), BLOCK, 0, ctx->stream>>>(ctx->ndb, nS, n, nnz, ctx->st->dDbCol.as<int>(), ctx->st->dDbCoef.as<double>(), ctx->st->dDbDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>());
    if (ctx->nzi) fillZeroImpBatch<<<grid(ctx->nzi * nS), BLOCK, 0, ctx->stream>>>(ctx->nzi, nS, n, nnz, ctx->st->dZiIn.as<double>(), ctx->st->dZiIdx.as<int>(), ctx->st->dZiDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->nziBase, ctx->nZiOn, ctx->dZiOnSlotElem.as<int>(), ctx->nZiOff, ctx->dZiOffSlotElem.as<int>(), ctx->st->dZiAltDest.as<int>(), ctx->st->dZiSelRow.as<int>(), ctx->dMode.as<int>(), ctx->st->ziDyn ? 1 : 0, withPilotSelf(ziRelocOf(ctx->st.get()), ctx), ctx->nZiTree, ctx->nZiTree ? ctx->dZiTreeSlotElemVal.as<int>() : nullptr, ctx->hasZiMasks ? ctx->dZiOffMask.as<char>() : nullptr, ctx->hasZiMasks ? ctx->dZiOnMask.as<char>() : nullptr, ctx->hasZiMasks ? ctx->dZiTreeMask.as<char>() : nullptr);
    applyRowModeBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, nnz, ctx->dMode.as<int>(), ctx->st->dRowPtr.as<int>(), ctx->st->dDiagPos.as<int>(), ctx->dX.as<double>(), ctx->dTarget.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->dRowSlope.as<double>(),
                ctx->st->dColIdx.as<int>(), ctx->st->dRelocOwn.p != nullptr ? ctx->st->dRelocOwn.as<int>() : nullptr);
    // OLF_GPU_VALS_SCAN_PREREPAIR: scan/dump the fill BEFORE the zero-diagonal repair rewrites hazard rows
    // (a row that reads as a borrowed donor after the repair was EMPTY before it — the defect is upstream).
    static const bool scanPreRepair = std::getenv("OLF_GPU_VALS_SCAN_PREREPAIR") != nullptr;
    if (scanPreRepair) scanValsFinite(ctx, nS, n, nnz, outer);
    hazRepairValues(ctx, nS, ctx->dVals.as<double>());
    if (!scanPreRepair) scanValsFinite(ctx, nS, n, nnz, outer);   // OLF_GPU_VALS_SCAN: is the matrix sound at all?
    if (!deviceFixedIter) CUDA_CHECK(cudaStreamSynchronize(ctx->stream));   // fixed-iter: stream-ordered into the solve, no host sync
}








static auto readNormsBatch(BatchSolverContext* ctx, DriverPass& drv, std::vector<double>& hn) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    residualNormsBatch<<<nS, BLOCK, 0, ctx->stream>>>(n, ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->dNorms.as<double>(),
            ctx->perRowCriteria ? ctx->dThrIdent.as<double>() : nullptr,
            ctx->perRowCriteria ? ctx->dThrPower.as<double>() : nullptr, ctx->dMode.as<int>(),
            ctx->perRowCriteria && ctx->perRowAlt ? ctx->dThrAlt.as<double>() : nullptr,
            ctx->perRowCriteria && ctx->perRowAlt ? ctx->dThrSel.as<int>() : nullptr);
    hn.resize(static_cast<size_t>(nS) * 4);
    CUDA_CHECK(cudaMemcpy(hn.data(), ctx->dNorms.p, hn.size() * sizeof(double), cudaMemcpyDeviceToHost));
}

static auto rejectUnrealistic(BatchSolverContext* ctx, DriverPass& drv, const std::vector<char>& gate) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    const int& outer = drv.outer;
    std::vector<char>& active = drv.active;
    std::vector<char>& live = drv.live;
    int& loopIdx = drv.loopIdx;
    int (&loopOrder)[MAX_DEVICE_LOOPS] = drv.loopOrder;
    BatchResult& res = drv.res;
    if (!ctx->dRealisticGate.p) ctx->dRealisticGate.alloc(nS * sizeof(char));
    if (!ctx->dRealisticBad.p) ctx->dRealisticBad.alloc(nS * sizeof(char));
    CUDA_CHECK(cudaMemcpyAsync(ctx->dRealisticGate.p, gate.data(), nS * sizeof(char),
                               cudaMemcpyHostToDevice, ctx->stream));
    CUDA_CHECK(cudaMemsetAsync(ctx->dRealisticBad.p, 0, nS * sizeof(char), ctx->stream));
    realisticCheckBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, ctx->dRealisticRow.as<char>(),
            ctx->dX.as<double>(), ctx->dRealisticGate.as<char>(),
            ctx->realisticMinV, ctx->realisticMaxV, ctx->dRealisticBad.as<char>());
    std::vector<char> bad(nS);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(cudaMemcpy(bad.data(), ctx->dRealisticBad.p, nS * sizeof(char), cudaMemcpyDeviceToHost));
    static const bool traceUnrealistic = std::getenv("OLF_GPU_UNREALISTIC_TRACE") != nullptr;
    for (int s = 0; s < nS; s++) {
        if (bad[s]) {
            if (traceUnrealistic) {
                std::fprintf(stderr, "GPU_UNREALISTIC_SWEEP slot=%d outer=%d loop=%d band=[%.4f, %.4f]\n",
                             s, outer, loopOrder[loopIdx], ctx->realisticMinV, ctx->realisticMaxV);
            }
            traceReject(s, "unrealistic-state", outer, 0);
            res.status[s] = -2;                          // STATUS_REJECTED -> the CPU replays it
            live[s] = 0;
            active[s] = 0;
        }
    }
}







BatchResult solveBatchContext(BatchSolverContext* ctx, const double* x0, const double* target,
                              const int* rowMode, const int* rowModeSlots, int nRowModeSlots,
                              const int* baseRowMode, const char* disabled,
                              int nOv, const int* ovIndex, const double* ovValue,
                              const int* dsOvBus, const double* dsOvFactorDelta, const double* dsOvBaseDelta,
                              const double* dsOvMaxDelta, const double* dsOvMinDelta, const double* dsAnchorShift,
                              const double* dsSlackShift,
                              int nRlVtOv, const int* rlVtOvSlotIdx, const double* rlVtOvVal,
                              int nRlQOv, const int* rlQOvSlotEntry, const double* rlQOvMinMax,
                              int nCvOv, const int* cvOvSlotEntry, const double* cvOvStepScale,
                              int maxIter, double tol,
                              double reactiveLimitTol, double slackDistTol, double convEpsPerEq,
                              int lsMode, double maxDv, double maxDphi, const int* varType,
                              const double* x0Full, int nShOv, const int* shOvSlotElem, const double* shOvGb,
                              const int* rlStateIn, int* rlStateOut,
                              int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal,
                              int maxOuterProductive, int rlMaxSwitch, int dsRejectResidual, int skipRlPhase,
                              int nziBase, int nZiOn, const int* ziOnSlotElem,
                              int nZiOff, const int* ziOffSlotElem) {
    DriverPass drv;
    drv.n = ctx->n;
    drv.nnz = ctx->nnz;
    drv.nS = ctx->nScenarios;
    drv.nRlVtOv = nRlVtOv;
    const int& n = drv.n;
    const int& nnz = drv.nnz;
    const int& nS = drv.nS;
    // MAX_VOLTAGE_CHANGE (lsMode 2): per-scenario step clamp, same rule as the base solver's
    // (varType 1 = V vs maxDv, 0 = PHI vs maxDphi; whole step scaled). Buffers allocated lazily.
    uploadBatchOverrides(ctx, lsMode, maxDv, maxDphi, varType, nShOv, shOvSlotElem, shOvGb,
                         nziBase, nZiOff, ziOffSlotElem, nZiOn, ziOnSlotElem, nR1Ov, r1OvSlotElem, r1OvVal);
    size_t nTot = static_cast<size_t>(n) * nS;
    DeviceBuf dOvIndex, dOvValue;                            // function-scoped: outlive the async override kernel
    DeviceBuf dBaseMode;                                     // the contingency-free base row mode (length n)
    DeviceBuf dModeSlots, dModeRows;                         // compact row modes: the scenarios that differ from the base
    DeviceBuf dX0SeedIdx, dX0SeedVal;                        // setBatchX0Seeds, function-scoped like the overrides
    // Per-scenario distributed-slack reweight (generator loss at a DS-participant bus): one affected DS bus per
    // scenario (dsOvBus[s], -1 if none) whose factor/base/headroom/footroom shrink by the lost gen's share, and
    // a per-scenario factor-sum drop so the other buses pick up that share. nullptr when no DS scenarios.
    DeviceBuf& dDsOvBus = drv.dDsOvBus;
    DeviceBuf& dDsOvFactorDelta = drv.dDsOvFactorDelta;
    DeviceBuf& dDsOvBaseDelta = drv.dDsOvBaseDelta;
    DeviceBuf& dDsOvMaxDelta = drv.dDsOvMaxDelta;
    DeviceBuf& dDsOvMinDelta = drv.dDsOvMinDelta;
    drv.hasDsOverride = (dsOvBus != nullptr && ctx->nds > 0);
    const bool& hasDsOverride = drv.hasDsOverride;
    // Per-scenario DS ANCHOR SHIFT (pair/strategy legs): shift_i = parent's final target at participant i
    // minus the extraction-time base — re-anchors the water-fill's cumulative reference at the PARENT
    // contingency's distribution, mirroring OLF's setGeneratorsInitialTargetPToTargetP before a strategy
    // re-solve. nullptr = anchored at the base (plain contingencies).
    DeviceBuf& dRlVtOvSlotIdx = drv.dRlVtOvSlotIdx;
    DeviceBuf& dRlVtOvVal = drv.dRlVtOvVal;
    if (nRlVtOv > 0) {
        dRlVtOvSlotIdx.upload(rlVtOvSlotIdx, 2L * nRlVtOv * sizeof(int));
        dRlVtOvVal.upload(rlVtOvVal, (size_t) nRlVtOv * sizeof(double));
    }
    DeviceBuf& dDsAnchorShift = drv.dDsAnchorShift;
    DeviceBuf& dDsSlackShift = drv.dDsSlackShift;
    drv.hasDsAnchor = (dsAnchorShift != nullptr && ctx->nds > 0);
    const bool& hasDsAnchor = drv.hasDsAnchor;
    if (hasDsAnchor) {
        dDsAnchorShift.upload(dsAnchorShift, static_cast<size_t>(nS) * ctx->nds * sizeof(double));
    }
    // The slack's own target move, independent of the participants' anchors: a strategy pair's parent share,
    // and also a contingency's load lost at the slack (a plain scenario, no anchor shift).
    drv.hasDsSlackShift = (dsSlackShift != nullptr && ctx->nds > 0);
    const bool& hasDsSlackShift = drv.hasDsSlackShift;
    if (hasDsSlackShift) {
        dDsSlackShift.upload(dsSlackShift, static_cast<size_t>(nS) * sizeof(double));
    }

    // x0/target are the BASE row (length n), identical for every scenario: upload once and tile across
    // nS on-device. rowMode is PER-SCENARIO (length n*nS): a contingency can freeze islanded buses
    // (Phase 0b) or flip PV↔PQ via reactive limits (Phase 1), so it is NOT tiled — uploaded as-is and
    // mutated in place on-device by the outer-loop kernels. Only the disabled mask is also per-scenario.
    // x0: either the BASE row tiled across nS (default), or a FULL per-scenario n*nS state
    // (x0Full != null) — the warm-restart hook for multi-call outer sequencing (e.g. the
    // transformer voltage-control orchestrator re-solving scenarios from their previous states).
    // RESUME: the driver is continuing from a LOOP_HOST pause, so x/targets/modes already hold the
    // sequence's current state on the device — re-uploading the base rows would throw away the
    // distributed slack, the HVDC deltas and the monitoring activations it has accumulated.
    const bool resume = ctx->driverResume != 0;
    if (!resume || ctx->olPasses.size() != static_cast<size_t>(nS)) {
        ctx->olFirst.assign(nS, 0);
        ctx->olIters.assign(nS, 0);
        ctx->olPasses.assign(nS, 0);
    }
    ctx->driverResume = 0;
    ctx->driverPaused = 0;
    if (resume && (ctx->nTvc > 0 || ctx->nFlc > 0 || ctx->nCph > 0 || ctx->nTvd > 0)) {
        tvcRebuildOverrides(ctx);                            // uploadBatchOverrides just erased the moves
    }
    if (resume && ctx->nShc > 0) {
        shcRebuildOverrides(ctx);
    }
    dBaseMode.upload(baseRowMode, n * sizeof(int));         // base (contingency-free) row mode for the RL limit-type seed
    if (!resume) {
        if (x0Full != nullptr) {
            CUDA_CHECK(cudaMemcpy(ctx->dX.p, x0Full, (size_t) nTot * sizeof(double), cudaMemcpyHostToDevice));
        } else {
            CUDA_CHECK(cudaMemcpy(ctx->dX.p, x0, n * sizeof(double), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemcpy(ctx->dTarget.p, target, n * sizeof(double), cudaMemcpyHostToDevice));
        // The BASE targets, kept aside before tiling and the per-scenario overrides: the pilot-Q
        // capture below reads the base carrier's row, and a scenario may have overridden that row
        // (a member islanded by a bus loss freezes it to a flat voltage identity). Capturing from the
        // base keeps the captured value the pilot's reactive target whatever the scenario did to the
        // row. (The carrier's target itself is stamped host-side when the base carrier is frozen -
        // GpuBatchChunkSolver - because the device stamp runs only inside the outer-loop passes.)
        if (!ctx->dTargetBase.p) ctx->dTargetBase.alloc((size_t) n * sizeof(double));
        CUDA_CHECK(cudaMemcpy(ctx->dTargetBase.p, target, n * sizeof(double), cudaMemcpyHostToDevice));
        if (rowModeSlots == nullptr) {
            CUDA_CHECK(cudaMemcpy(ctx->dMode.p, rowMode, nTot * sizeof(int), cudaMemcpyHostToDevice));
        } else {
            // COMPACT: every scenario starts from the base row mode, then only those that differ (islanded buses,
            // switch outages, ...) get their own row — a few rows instead of nS * n ints through PCIe.
            CUDA_CHECK(cudaMemcpy(ctx->dMode.p, baseRowMode, n * sizeof(int), cudaMemcpyHostToDevice));
            tileBaseRow<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n * nS, n, ctx->dMode.as<int>());
            if (nRowModeSlots > 0) {
                dModeSlots.upload(rowModeSlots, nRowModeSlots * sizeof(int));
                dModeRows.upload(rowMode, (size_t) nRowModeSlots * n * sizeof(int));
                scatterRowsInt<<<grid(nRowModeSlots * n), BLOCK, 0, ctx->stream>>>(nRowModeSlots, n,
                        dModeSlots.as<int>(), dModeRows.as<int>(), ctx->dMode.as<int>());
            }
        }
        if (x0Full == nullptr) {
            tileBaseRow<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n * nS, n, ctx->dX.as<double>());
            int nSeeds = static_cast<int>(ctx->x0SeedIdx.size());
            if (nSeeds > 0) {                                // a few per-scenario starting values over the base
                dX0SeedIdx.upload(ctx->x0SeedIdx.data(), nSeeds * sizeof(int));
                dX0SeedVal.upload(ctx->x0SeedVal.data(), nSeeds * sizeof(double));
                applyTargetOverrideBatch<<<grid(nSeeds), BLOCK, 0, ctx->stream>>>(nSeeds, dX0SeedIdx.as<int>(),
                        dX0SeedVal.as<double>(), ctx->dX.as<double>());
            }
        }
        ctx->x0SeedIdx.clear();                              // consumed by this solve either way
        ctx->x0SeedVal.clear();
        tileBaseRow<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n * nS, n, ctx->dTarget.as<double>());
        if (nOv > 0) {                                       // per-scenario target overrides (generator loss)
            dOvIndex.upload(ovIndex, nOv * sizeof(int));
            dOvValue.upload(ovValue, nOv * sizeof(double));
            applyTargetOverrideBatch<<<grid(nOv), BLOCK, 0, ctx->stream>>>(nOv, dOvIndex.as<int>(), dOvValue.as<double>(), ctx->dTarget.as<double>());
        }
    }
    if (hasDsOverride) {                                     // per-scenario generator-loss DS reweight
        dDsOvBus.upload(dsOvBus, nS * sizeof(int));
        dDsOvFactorDelta.upload(dsOvFactorDelta, nS * sizeof(double));
        dDsOvBaseDelta.upload(dsOvBaseDelta, nS * sizeof(double));
        dDsOvMaxDelta.upload(dsOvMaxDelta, nS * sizeof(double));
        dDsOvMinDelta.upload(dsOvMinDelta, nS * sizeof(double));
    }
    if (!resume) {
        // NOT on a resume: the HVDC AC-emulation driver phases MUTATE element bits on the device during
        // the paused sweep (a frozen or saturated emulation has its coupling deactivated), and the host's
        // mirror of the mask is the pre-solve one. Re-uploading it mid-sequence silently undoes those
        // loops' work — measured on a 400-contingency rte6515 slice: 5 differing violations against 1,
        // with NO host decision involved at all (the split alone did it). Contingency outages do not
        // change during a sweep, so the device copy stays authoritative across the pause.
        CUDA_CHECK(cudaMemcpy(ctx->dDisabled.p, disabled,
                              static_cast<size_t>(ctx->nElements) * nS * sizeof(char), cudaMemcpyHostToDevice));
    }

    std::vector<char>& active = drv.active;
    active.assign(nS, 1);

    BatchResult& res = drv.res;
    res.x.resize(nTot);
    res.iters.assign(nS, 0);
    res.norm.assign(nS, 0.0);
    res.status.assign(nS, 1);                                // default: max_iter (overwritten on converge/diverge)

    std::vector<double> hF(nTot);

    // Device-side fixed-iteration mode: tol <= 0 means "run exactly maxIter Newton steps
    // on the device with NO per-iteration host sync and NO convergence retirement" (the
    // a fixed-iteration NR loop). tol > 0 keeps the per-iteration retirement loop.
    drv.deviceFixedIter = (tol <= 0.0);
    const bool& deviceFixedIter = drv.deviceFixedIter;
    // A scenario that exhausts the budget is reported as MAX_ITERATION_REACHED by the device itself (the
    // precomputer's exhausted set), which is OLF's own answer: no early exit hands stragglers to the CPU.

    // Full-loadflow mode (tol > 0) with local PV controllers runs the reactive-limits (PV→PQ) OUTER loop:
    // converge the inner Newton, switch any generator over its Q limit, re-converge the switched scenarios,
    // repeat until a pass switches nothing. Fixed-iter mode (tol <= 0, the throughput benchmark) and
    // family-free networks with no controllers do ONE pass (no outer loop) — byte-identical to before.
    bool fullLoadflow = !deviceFixedIter && (ctx->nrl > 0 || ctx->nds > 0 || ctx->dsSlackOnly
                                             || ctx->nmon > 0 || ctx->nTvc > 0 || ctx->nShc > 0 || ctx->nFlc > 0 || ctx->nCph > 0
                                             || ctx->nTvd > 0 || ctx->nSvd > 0 || ctx->nSvc > 0);
    const int RL_MAX_OUTER = maxOuterProductive > 0 ? maxOuterProductive : 20;   // OLF maxOuterLoopIterations
    // per-bus PV→PQ count cap (OLF maxPqPvSwitch): 0 is OLF's "never switch back" and is honoured as such;
    // only a negative value (no cap given) takes OLF's default of 3.
    const int RL_MAX_SWITCH = rlMaxSwitch >= 0 ? rlMaxSwitch : 3;
    static const char* swLogEnv = std::getenv("OLF_GPU_SW_LOG");
    const int swLogSlot = swLogEnv == nullptr ? -1 : (std::string(swLogEnv) == "all" ? -2 : std::atoi(swLogEnv));
    // OLF_GPU_ITER_LOG=1: EVERY scenario's stopping decision at every Newton iteration (ITER_LOG), the score of
    // every step's linear solve (LIN_LOG, and LIN_WORST where it is poor) and its record at the end of every outer
    // pass (PASS_LOG: loop, re-solve, switch, iterations, status, DS gate), printed once when the driver returns.
    // Built from the norms the host already reads each iteration, the pass-0 guard's two scoring kernels run on
    // every step (one 3*nS copy) and ONE copy of the DS gate records per pass - none of the whole-vector copies
    // the slot traces make: a run-to-run divergence of one scenario among hundreds is found by diffing two logs.
    static const bool iterLog = std::getenv("OLF_GPU_ITER_LOG") != nullptr;
    std::vector<std::string> iterLogLines;
    // Outer-loop convergence tolerances (pu) — passed in so the device uses the SAME criteria as the CPU
    // loadflow that built the base case (maxReactivePowerMismatch / slackBusPMaxMismatch, per-unitized), NOT
    // hardcoded constants. With matching parameters and the same equations the GPU outer-loop fixed point
    // matches the CPU's to the inner-Newton precision; a loose tolerance (e.g. the 1 MW slack default) leaves
    // a redistribution band, so a test wanting machine-precision agreement passes tight values on both sides.
    drv.reactiveLimitTol = reactiveLimitTol;
    const double& REACTIVE_LIMIT_TOL = drv.reactiveLimitTol;
    drv.slackDistTol = slackDistTol;
    const double& SLACK_DIST_TOL = drv.slackDistTol;
    drv.pResidueEps = 1e-5;                                  // OLF ActivePowerDistribution.P_RESIDUE_EPS: the
                                                            // water-filling distributes the mismatch down to this
    // RL_MAX_OUTER bounds the PRODUCTIVE outer iterations (an outer loop that re-solved), matching OLF's
    // outerLoopTotalIterations budget. The raw pass loop gets a generous safety bound on top, because the
    // sequential driver also spends non-productive passes (phase transitions / stability re-checks) that OLF
    // does not count.
    // The reactive-limit limits become PER-SCENARIO as soon as either channel is live: curve limits at
    // the runtime P, or a generator lost at a bus that keeps voltage control taking its capability away.
    DeviceBuf dRlQOvSlotEntry, dRlQOvMinMax, dRlQOvPtr, dRlQOvList;
    if (nRlQOv > 0) {
        dRlQOvSlotEntry.upload(rlQOvSlotEntry, static_cast<size_t>(nRlQOv) * 2 * sizeof(int));
        dRlQOvMinMax.upload(rlQOvMinMax, static_cast<size_t>(nRlQOv) * 5 * sizeof(double));   // {dMinQ, dMaxQ, dLoadQ, dBmin, dBmax}
        // ...indexed by slot (CSR, entry order kept), so the dynamic-limits kernel reads its scenario's own
        // overrides instead of scanning the chunk's.
        std::vector<int> qPtr(nS + 1, 0);
        std::vector<int> qList(nRlQOv);
        for (int k = 0; k < nRlQOv; k++) qPtr[rlQOvSlotEntry[2 * k] + 1]++;
        for (int t = 0; t < nS; t++) qPtr[t + 1] += qPtr[t];
        std::vector<int> qCursor(qPtr.begin(), qPtr.end() - 1);
        for (int k = 0; k < nRlQOv; k++) qList[qCursor[rlQOvSlotEntry[2 * k]]++] = k;
        dRlQOvPtr.upload(qPtr.data(), qPtr.size() * sizeof(int));
        dRlQOvList.upload(qList.data(), qList.size() * sizeof(int));
    }
    // A lost voltage-controlling SVC also takes its share of the entry's quadratic limit coefficients (an SVC's
    // limit is its susceptance times v^2): per (scenario, entry) coefficients, read by the reactive-limits check
    // in place of the run-constant ones. Built only when some override carries such a share.
    bool svcCoefDyn = false;
    for (int k = 0; k < nRlQOv && !svcCoefDyn; k++) {
        svcCoefDyn = rlQOvMinMax[5 * k + 3] != 0.0 || rlQOvMinMax[5 * k + 4] != 0.0;
    }
    if (svcCoefDyn && ctx->nrl > 0) {
        size_t nrl = static_cast<size_t>(ctx->nrl);
        std::vector<double> bmin(nrl), bmax(nrl);
        CUDA_CHECK(cudaMemcpy(bmin.data(), ctx->dRlSvcBmin.p, nrl * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(bmax.data(), ctx->dRlSvcBmax.p, nrl * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<double> dynMin(static_cast<size_t>(nS) * nrl), dynMax(static_cast<size_t>(nS) * nrl);
        for (int t = 0; t < nS; t++) {
            std::copy(bmin.begin(), bmin.end(), dynMin.begin() + static_cast<size_t>(t) * nrl);
            std::copy(bmax.begin(), bmax.end(), dynMax.begin() + static_cast<size_t>(t) * nrl);
        }
        for (int k = 0; k < nRlQOv; k++) {
            size_t at = static_cast<size_t>(rlQOvSlotEntry[2 * k]) * nrl + rlQOvSlotEntry[2 * k + 1];
            dynMin[at] += rlQOvMinMax[5 * k + 3];
            dynMax[at] += rlQOvMinMax[5 * k + 4];
        }
        if (!ctx->dRlSvcBminDyn.p) {
            ctx->dRlSvcBminDyn.alloc(static_cast<size_t>(ctx->nScenarios) * nrl * sizeof(double));
            ctx->dRlSvcBmaxDyn.alloc(static_cast<size_t>(ctx->nScenarios) * nrl * sizeof(double));
        }
        CUDA_CHECK(cudaMemcpy(ctx->dRlSvcBminDyn.p, dynMin.data(), dynMin.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->dRlSvcBmaxDyn.p, dynMax.data(), dynMax.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    // Per-scenario curve repairs, CSR by scenario: the dynamic-limits kernel runs one thread per
    // (reactive-limit entry, scenario) and walks the curve entries inside, so a flat {slot, entry}
    // scan would be paid ncv times per thread. Bucketing by slot makes it a handful of comparisons.
    DeviceBuf dCvOvPtr, dCvOvEntry, dCvOvVal;
    if (nCvOv > 0) {
        std::vector<int> ptr(nS + 1, 0);
        for (int k = 0; k < nCvOv; k++) ptr[cvOvSlotEntry[2 * k] + 1]++;
        for (int t = 0; t < nS; t++) ptr[t + 1] += ptr[t];
        std::vector<int> cursor(ptr.begin(), ptr.end() - 1);
        std::vector<int> entry(nCvOv);
        std::vector<double> val(static_cast<size_t>(nCvOv) * 2);
        for (int k = 0; k < nCvOv; k++) {
            int at = cursor[cvOvSlotEntry[2 * k]]++;
            entry[at] = cvOvSlotEntry[2 * k + 1];
            val[2 * at] = cvOvStepScale[2 * k];
            val[2 * at + 1] = cvOvStepScale[2 * k + 1];
        }
        dCvOvPtr.upload(ptr.data(), ptr.size() * sizeof(int));
        dCvOvEntry.upload(entry.data(), entry.size() * sizeof(int));
        dCvOvVal.upload(val.data(), val.size() * sizeof(double));
    }
    drv.useDynRlLimits = ctx->nrl > 0 && (ctx->ncv > 0 || nRlQOv > 0 || ctx->nfqLim > 0);
    const bool& useDynRlLimits = drv.useDynRlLimits;
    if (useDynRlLimits && (!ctx->dRlMinQDyn.p || !ctx->dRlLoadQDyn.p)) {   // curves allocate these on their own
                                                            // table upload; a Q override alone must too, and a
                                                            // guard that names only ONE of them silently leaves
                                                            // the others null on a curve network (measured: the
                                                            // whole precompute fell back to the CPU)
        ctx->dRlLoadQDyn.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nrl * sizeof(double));
        ctx->dRlMinQDyn.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nrl * sizeof(double));
        ctx->dRlMaxQDyn.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nrl * sizeof(double));
    }
    int maxOuter = fullLoadflow ? 8 * RL_MAX_OUTER : 1;
    // The two kernels that lay a group's rows out from the CURRENT pin states (the RL pass runs them after
    // its pins; the seeding below runs them once so the FIRST fill sees the seeded pins on the rows).
    // `staticLimits`: use the STATIC reactive-limit arrays rather than the curve-evaluated ones. At SEED time the
    // dynamic arrays are still zero (rlCurveLimitsBatch runs in the reactive-limits phase), so a seed-time flip
    // stamped the pilot's own-Q target as 0 instead of (limit - loadQ) and the whole slack phase ran against it —
    // measured on rte6515 BUS-6112_BBS_2: row 12834 converged at target 0 through outer 32, then the RL phase
    // re-stamped -0.444 and the solve had to catch up, one pass behind OLF. The re-stamp branch in the kernel
    // keeps the value current once the curve limits exist, so the static value is a correct starting point.
    std::vector<char>& rlActive = drv.rlActive;
    rlActive.assign(nS, 0);
    std::vector<char> rlSwitched(nS), wasActive(nS);
    // OLF_GPU_RL_AUDIT: per-scenario decision history, kept in memory and printed only for the
    // scenarios the end-of-solve audit flags. The failing scenario is not known until the solve is
    // over and the flip is rare, so the only way to see its trajectory is to record every scenario's
    // and throw away the rest.
    const bool rlAuditOn = std::getenv("OLF_GPU_RL_AUDIT") != nullptr;
    std::vector<std::vector<std::string>> auditLog(rlAuditOn ? nS : 0);
    // ...but NOT on a resume: these are per-SOLVE state, and a resumed driver is continuing the SAME
    // solve. Re-seeding here would discard the RL pins, monitoring activations and absorbed slack the
    // sequence has already established — which is exactly what makes a resume cheaper than the
    // orchestrator pattern it replaces: that one resets and re-runs the whole driver every round.
    if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr && nS == 1) {
        std::fprintf(stderr, "CTX_FLAGS nS=1 nrl=%d nRlRemote=%d hasDqQrGroups=%d nQr=%d ndq=%d nds=%d ngen=%d nzi=%d nziBase=%d nRmq=%d resume=%d fullLoadflow=%d\n",
                     ctx->nrl, ctx->nRlRemote, ctx->hasDqQrGroups ? 1 : 0, ctx->st->nQr, ctx->ndq, ctx->nds, ctx->ngen, ctx->nzi, ctx->nziBase, ctx->nRmq, resume ? 1 : 0, fullLoadflow ? 1 : 0);
    }
    if (fullLoadflow && !resume && ctx->nSvc > 0) {
        // BEFORE the reactive-limit seeding below, which consults it: no zone has moved a target yet.
        std::vector<double> nanv(static_cast<size_t>(nS) * std::max(ctx->nrl, 1), std::nan(""));
        CUDA_CHECK(cudaMemcpy(ctx->dSvcRlVt.p, nanv.data(), nanv.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    if (!resume) {                                           // no inner solve has converged yet: the start
        CUDA_CHECK(cudaMemcpyAsync(ctx->dXLastConv.p, ctx->dX.p, static_cast<size_t>(nS) * n * sizeof(double),
                                   cudaMemcpyDeviceToDevice, ctx->stream));
    }
    if (fullLoadflow && !resume) {                           // per-solve PV↔PQ state: 0 switches, limitType from base
        // Each generator's own slack move is per-solve state, cleared before ANYTHING reads it: the seed-time curve
        // limits below evaluate a curve at base + genCum wherever genCumValid is set. Cleared only after the seed,
        // a reused context evaluated scenario slot s at the move of the contingency the PREVIOUS chunk solved in
        // that slot. The write-back also reads it when the DS phase never fires (NaN targetP otherwise).
        if (ctx->ngen > 0 && ctx->dGenCum.p) {
            CUDA_CHECK(cudaMemsetAsync(ctx->dGenCum.p, 0, static_cast<size_t>(ctx->ngen) * nS * sizeof(double), ctx->stream));
            CUDA_CHECK(cudaMemsetAsync(ctx->dGenCumValid.p, 0, (size_t) nS * sizeof(char), ctx->stream));
        }
        if (ctx->hasDqQrGroups && ctx->st->nQr > 0) {        // plan-B4: capture each group's pilot-Q target from
            size_t need = static_cast<size_t>(ctx->st->nQr) * nS * sizeof(double);   // the BASE carrier's row,
            if (!ctx->dQrPilotQ.p) ctx->dQrPilotQ.alloc(need);   // BEFORE any RL mode seeding rewrites it
            if (!ctx->dQrSavedVT.p) ctx->dQrSavedVT.alloc(need);
            if (!ctx->dQrVeqOff.p) ctx->dQrVeqOff.alloc(static_cast<size_t>(ctx->st->nQr) * nS * sizeof(char));
            CUDA_CHECK(cudaMemsetAsync(ctx->dQrVeqOff.p, 0,      // per-solve state, like limitType
                    static_cast<size_t>(ctx->st->nQr) * nS * sizeof(char), ctx->stream));
            capturePilotQBatch<<<grid(ctx->st->nQr * nS), BLOCK, 0, ctx->stream>>>(ctx->st->nQr, nS, n,
                    ctx->st->dQrPtr.as<int>(), ctx->st->dQrRows.as<int>(), dBaseMode.as<int>(),
                    ctx->dTargetBase.as<double>(), ctx->dQrPilotQ.as<double>(),
                    ctx->st->dQrBaseTargetRow.p ? ctx->st->dQrBaseTargetRow.as<int>() : nullptr);
            if (ctx->nQrPqDelta > 0) {
                applyPilotQDeltasBatch<<<grid(ctx->nQrPqDelta), BLOCK, 0, ctx->stream>>>(ctx->nQrPqDelta,
                        ctx->dQrPqDeltaKey.as<int>(), ctx->dQrPqDeltaVal.as<double>(), ctx->st->nQr, n,
                        ctx->st->dQrPtr.as<int>(), ctx->st->dQrRows.as<int>(), ctx->dMode.as<int>(),
                        ctx->hasDqSub ? ctx->dDqSub.as<char>() : nullptr, ctx->nDqMembers,
                        ctx->hasDqSub ? ctx->dQrMemberStart.as<int>() : nullptr,
                        ctx->dQrPilotQ.as<double>(), ctx->dTarget.as<double>());
            }
            if (!ctx->dQrPilotRlEntry.p) {
                // Per group, the pilot's own LOCAL reactive-limit entry: a pilot whose group lost control
                // in the BASE (every member pinned there) enters the pack as a local entry with an empty
                // reactive range, and that entry carries the group's V target — the value the pilot row
                // takes back when a member unpins (updatePilotVeqBatch). Built once per context.
                int nQr = ctx->st->nQr;
                std::vector<int> ptr(nQr + 1);
                CUDA_CHECK(cudaMemcpy(ptr.data(), ctx->st->dQrPtr.p, ptr.size() * sizeof(int), cudaMemcpyDeviceToHost));
                std::vector<int> rows(std::max(ptr[nQr], 1));
                CUDA_CHECK(cudaMemcpy(rows.data(), ctx->st->dQrRows.p, static_cast<size_t>(ptr[nQr]) * sizeof(int), cudaMemcpyDeviceToHost));
                std::vector<int> vrow(std::max(ctx->nrl, 1));
                std::vector<char> rem(std::max(ctx->nrl, 1), 0);
                if (ctx->nrl > 0) {
                    CUDA_CHECK(cudaMemcpy(vrow.data(), ctx->dRlVrow.p, static_cast<size_t>(ctx->nrl) * sizeof(int), cudaMemcpyDeviceToHost));
                    if (ctx->dRlRemote.p) {
                        CUDA_CHECK(cudaMemcpy(rem.data(), ctx->dRlRemote.p, static_cast<size_t>(ctx->nrl), cudaMemcpyDeviceToHost));
                    }
                }
                std::vector<int> pe(std::max(nQr, 1), -1);
                for (int g = 0; g < nQr; g++) {
                    int pilotRow = rows[ptr[g]];
                    for (int i = 0; i < ctx->nrl; i++) {
                        if (!rem[i] && vrow[i] == pilotRow) { pe[g] = i; break; }
                    }
                }
                ctx->dQrPilotRlEntry.upload(pe.data(), pe.size() * sizeof(int));
            }
            if (nS == 1 && std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
                CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                int nQr = ctx->st->nQr;
                std::vector<int> ptr(nQr + 1);
                CUDA_CHECK(cudaMemcpy(ptr.data(), ctx->st->dQrPtr.p, ptr.size() * sizeof(int), cudaMemcpyDeviceToHost));
                std::vector<int> rows(ptr[nQr]);
                CUDA_CHECK(cudaMemcpy(rows.data(), ctx->st->dQrRows.p, rows.size() * sizeof(int), cudaMemcpyDeviceToHost));
                std::vector<double> pq(nQr), tg(n);
                std::vector<int> md(n);
                CUDA_CHECK(cudaMemcpy(pq.data(), ctx->dQrPilotQ.p, nQr * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(tg.data(), ctx->dTarget.p, n * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(md.data(), dBaseMode.p, n * sizeof(int), cudaMemcpyDeviceToHost));
                for (int g = 0; g < nQr; g++) {
                    std::fprintf(stderr, "PILOTQ_CAPTURE group=%d captured=%.9f candidates:", g, pq[g]);
                    for (int c = ptr[g]; c < ptr[g + 1]; c++) {
                        std::fprintf(stderr, " row%d(mode=%d,target=%.9f)", rows[c], md[rows[c]], tg[rows[c]]);
                    }
                    std::fprintf(stderr, "\n");
                }
            }
        }
        if (rlStateIn != nullptr && ctx->nrl > 0) {
            // WARM CONTINUATION: seed each (controller, scenario)'s pinned limit type AND switch
            // count from the state the caller passes. A re-solve inside one load flow passes both; a
            // strategy pair passes its parent's pins with zero counts, since OLF's reactive-limit
            // loop counts afresh in every load flow.
            std::vector<char> lt(static_cast<size_t>(ctx->nrl) * nS), sc(static_cast<size_t>(ctx->nrl) * nS);
            for (size_t k = 0; k < lt.size(); k++) {
                lt[k] = static_cast<char>(rlStateIn[k] & 7);
                sc[k] = static_cast<char>(rlStateIn[k] >> 3);
            }
            CUDA_CHECK(cudaMemcpyAsync(ctx->dRlLimitType.p, lt.data(), lt.size(), cudaMemcpyHostToDevice, ctx->stream));
            CUDA_CHECK(cudaMemcpy(ctx->dRlSwitchCount.p, sc.data(), sc.size(), cudaMemcpyHostToDevice));
            // The limits the continued pins are STAMPED at: OLF restores each pinned bus's generation target Q from
            // the parent's end, and a GeneratorAction only invalidates the bus's target P, so the first solve keeps
            // the limit at the parent's P (dgSeedAnchor) and the reactive-limit check refreshes it afterwards.
            if (useDynRlLimits) {                            // curve limits at the CURRENT (seeded) targets,
                rlCurveLimitsBatch<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(ctx->nrl, nS, ctx->ncv, n, ctx->cvExtrapolate,
                        ctx->dCvIdx.as<int>(), ctx->dCvIn.as<double>(), ctx->dCvPts.as<double>(),
                        ctx->dCvPtr.as<int>(), ctx->dCvList.as<int>(), cvHvdcView(ctx),
                        ctx->dRlMinQ.as<double>(), ctx->dRlMaxQ.as<double>(), ctx->dRlLoadQ.as<double>(),
                        ctx->dTarget.as<double>(),
                        nRlQOv, nRlQOv > 0 ? dRlQOvSlotEntry.as<int>() : nullptr,
                        nRlQOv > 0 ? dRlQOvMinMax.as<double>() : nullptr,
                        nRlQOv > 0 ? dRlQOvPtr.as<int>() : nullptr, nRlQOv > 0 ? dRlQOvList.as<int>() : nullptr,
                        nCvOv > 0 ? dCvOvPtr.as<int>() : nullptr,
                        nCvOv > 0 ? dCvOvEntry.as<int>() : nullptr,
                        nCvOv > 0 ? dCvOvVal.as<double>() : nullptr,
                        ctx->ngen > 0 ? ctx->dGenCum.as<double>() : nullptr,
                        ctx->ngen > 0 ? ctx->dGenCumValid.as<char>() : nullptr, ctx->hasDgSeedAnchor ? ctx->dDgSeedAnchor.as<double>()
                                : (ctx->hasDgAnchor ? ctx->dDgAnchor.as<double>() : nullptr),
                        ctx->hasDgStart ? ctx->dDgStart.as<double>() : nullptr, ctx->ngen,
                        ctx->nds > 0 ? ctx->dDsPhiRow.as<int>() : nullptr,
                        ctx->nDsRowOv > 0 ? ctx->dDsRowOv.as<int>() : nullptr, ctx->nds,
                        ctx->ngen > 0 ? ctx->dgBus.as<int>() : nullptr,
                        ctx->dRlMinQDyn.as<double>(), ctx->dRlMaxQDyn.as<double>(),
                        ctx->dRlLoadQDyn.as<double>(), cvProbeEntry, cvProbeSlotByCtg ? ctx->loopTraceSlot : cvProbeSlot);
                launchForcedTargetQLimits(ctx, nS, n);
            }
            seedRlModeFromState<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(ctx->nrl, nS, n,
                    ctx->dRlVrow.as<int>(), ctx->dRlQrow.p ? ctx->dRlQrow.as<int>() : nullptr,
                    useDynRlLimits ? ctx->dRlMinQDyn.as<double>() : ctx->dRlMinQ.as<double>(),
                    useDynRlLimits ? ctx->dRlMaxQDyn.as<double>() : ctx->dRlMaxQ.as<double>(),
                    useDynRlLimits ? ctx->dRlLoadQDyn.as<double>() : ctx->dRlLoadQ.as<double>(), ctx->dRlLimitType.as<char>(),
                    ctx->dMode.as<int>(), ctx->dTarget.as<double>(), useDynRlLimits ? 1 : 0,
                    ctx->nRlRemote > 0 ? ctx->dRlRemote.as<char>() : nullptr,
                    ctx->dRlSavedTarget.as<double>(), ctx->dRlSavedMode.as<char>(),
                    ctx->hasRlBaseLt ? ctx->dRlBaseLimitType.as<char>() : nullptr,
                    psEntryPtr(ctx), ctx->st->nQr,
                    ctx->dRlVtarget.as<double>(), nRlVtOv, nRlVtOv > 0 ? dRlVtOvSlotIdx.as<int>() : nullptr,
                    nRlVtOv > 0 ? dRlVtOvVal.as<double>() : nullptr, ctx->nSvc > 0 ? ctx->dSvcRlVt.as<double>() : nullptr,
                    ctx->nRlLocal > 0 ? ctx->dRlLocalOv.as<char>() : nullptr,
                    ctx->nRlLocal > 0 ? ctx->dRlLocalVt.as<double>() : nullptr,
                    ctx->hasRlPinTgtIn ? ctx->dRlPinTgtIn.as<double>() : nullptr,
                    ctx->nRlIslanded > 0 ? ctx->dRlIslanded.as<char>() : nullptr);
        } else {
            CUDA_CHECK(cudaMemsetAsync(ctx->dRlSwitchCount.p, 0, static_cast<size_t>(ctx->nrl) * nS * sizeof(char), ctx->stream));
            if (ctx->nrl > 0) {                              // seed the pinned limit for base-PQ controllers (else all PV)
                initRlLimitTypeBatch<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(ctx->nrl, nS, n, ctx->dRlVrow.as<int>(),
                        ctx->dRlQrow.p ? ctx->dRlQrow.as<int>() : nullptr,
                        ctx->dRlVtarget.as<double>(), ctx->dRlMonitored.as<char>(), dBaseMode.as<int>(), ctx->dX.as<double>(),
                        ctx->dRlLimitType.as<char>(),
                        nRlVtOv, nRlVtOv > 0 ? dRlVtOvSlotIdx.as<int>() : nullptr,
                        nRlVtOv > 0 ? dRlVtOvVal.as<double>() : nullptr,
                        ctx->nSvc > 0 ? ctx->dSvcRlVt.as<double>() : nullptr,
                        ctx->nRlRemote > 0 ? ctx->dRlRemote.as<char>() : nullptr,
                        ctx->nRlRemote > 0 ? ctx->dRlCmpVrow.as<int>() : nullptr,
                        ctx->dTarget.as<double>(), ctx->dRlMinQ.as<double>(), ctx->dRlMaxQ.as<double>(),
                        ctx->dRlLoadQ.as<double>(),
                        ctx->hasRlBaseLt ? ctx->dRlBaseLimitType.as<char>() : nullptr,
                        psEntryPtr(ctx), ctx->st->nQr,
                        ctx->nRlGenLost > 0 ? ctx->dRlGenLost.as<char>() : nullptr);
                if (ctx->nRlGenLost > 0 && std::getenv("OLF_GPU_GENLOST_TRACE") != nullptr) {
                    // the seeded types of the lost-generator entries, read back (device printf is not captured)
                    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                    std::vector<char> hm((size_t) nS * ctx->nrl), hl((size_t) nS * ctx->nrl);
                    CUDA_CHECK(cudaMemcpy(hm.data(), ctx->dRlGenLost.p, hm.size(), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(hl.data(), ctx->dRlLimitType.p, hl.size(), cudaMemcpyDeviceToHost));
                    for (size_t k = 0; k < hm.size(); k++) {
                        if (hm[k]) {
                            std::fprintf(stderr, "GENLOST_SEED slot=%d entry=%d limitType=%d nS=%d nrl=%d\n",
                                         (int) (k / ctx->nrl), (int) (k % ctx->nrl), (int) hl[k], nS, ctx->nrl);
                        }
                    }
                }
            }
        }
        // The seeded pin states must reach the ROWS before the first fill: a pinned enrolled pilot's balance
        // goes on the carrier's target (its DISTR_Q target otherwise), and a group that lost control in the
        // base has its pilot row flipped already. Both kernels are idempotent; the RL pass runs them again.
        // Every scenario takes the seed stamps: updateDistrQTargetsBatch returns on !rlActive[s], and rlActive is
        // only written by the outer-loop passes - at seed time it still held the context's zeros (or the previous
        // chunk's last pass), so the carrier stamp below never ran before the FIRST solve. A pilot-self pilot that
        // lost every generator with two live survivors then solved its carrier row against the plan's renormalized
        // DISTR_Q seed (0.5 x its -0.6 balance = -0.3) instead of its balance: measured as 0.258 kV at b2 on
        // devicePilotControllerLosingItsOwnGeneratorMatchesCpu (OLF's BUS_TARGET_Q of the pilot 0.3 pu off at the
        // device state), and rte6515's 50 refused pilot outages. One survivor hid it: its seed degenerates to the
        // pilot's own target. The passes upload their own rlActive before they run.
        {
            std::vector<char> allOn(nS, 1);
            CUDA_CHECK(cudaMemcpyAsync(ctx->dRlActive.p, allOn.data(), nS, cudaMemcpyHostToDevice, ctx->stream));
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        }
        // The dynamic reactive limits BEFORE the seed stamps, on the cold path too. They were first computed in the
        // reactive-limit pass, after the first solve, so the stamps read the arrays unfilled: a gen-lost pilot's
        // carrier target came out 0 - 0 instead of 0 - loadQ, and a survivor's sharing target summed a pinned
        // peer's (limit - loadQ) as 0 - 0 - measured on devicePilotControllerLosingItsOwnGeneratorMatchesCpu
        // as rows 2 and 4 both at target 0 during the solve (OLF: -0.6 and -0.3). The warm path already ran
        // this before seeding; here it runs after initRlLimitTypeBatch has set the scenario's limit types.
        if (useDynRlLimits) {                            // curve limits at the CURRENT (seeded) targets,
            rlCurveLimitsBatch<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(ctx->nrl, nS, ctx->ncv, n, ctx->cvExtrapolate,
                    ctx->dCvIdx.as<int>(), ctx->dCvIn.as<double>(), ctx->dCvPts.as<double>(),
                        ctx->dCvPtr.as<int>(), ctx->dCvList.as<int>(), cvHvdcView(ctx),
                    ctx->dRlMinQ.as<double>(), ctx->dRlMaxQ.as<double>(), ctx->dRlLoadQ.as<double>(),
                    ctx->dTarget.as<double>(),
                    nRlQOv, nRlQOv > 0 ? dRlQOvSlotEntry.as<int>() : nullptr,
                    nRlQOv > 0 ? dRlQOvMinMax.as<double>() : nullptr,
                        nRlQOv > 0 ? dRlQOvPtr.as<int>() : nullptr, nRlQOv > 0 ? dRlQOvList.as<int>() : nullptr,
                    nCvOv > 0 ? dCvOvPtr.as<int>() : nullptr,
                    nCvOv > 0 ? dCvOvEntry.as<int>() : nullptr,
                    nCvOv > 0 ? dCvOvVal.as<double>() : nullptr,
                    ctx->ngen > 0 ? ctx->dGenCum.as<double>() : nullptr,
                    ctx->ngen > 0 ? ctx->dGenCumValid.as<char>() : nullptr, ctx->hasDgAnchor ? ctx->dDgAnchor.as<double>() : nullptr, ctx->hasDgStart ? ctx->dDgStart.as<double>() : nullptr, ctx->ngen,
                    ctx->nds > 0 ? ctx->dDsPhiRow.as<int>() : nullptr,
                    ctx->nDsRowOv > 0 ? ctx->dDsRowOv.as<int>() : nullptr, ctx->nds,
                    ctx->ngen > 0 ? ctx->dgBus.as<int>() : nullptr,
                    ctx->dRlMinQDyn.as<double>(), ctx->dRlMaxQDyn.as<double>(),
                    ctx->dRlLoadQDyn.as<double>(), cvProbeEntry, cvProbeSlotByCtg ? ctx->loopTraceSlot : cvProbeSlot);
            launchForcedTargetQLimits(ctx, nS, n);
        }
        launchPilotVeqInner(ctx, drv, true, true);                     // seed: base limits, or the previous solve's flip
        launchDistrQTargetsInner(ctx, drv);
        if (ctx->nds > 0) {
            dsResetSolve(ctx, drv);
        }
        if (ctx->nmon > 0) {
            monitoringResetSolve(ctx, drv);
        }
        if (ctx->nTvc > 0) {
            tvcResetSolve(ctx, drv);
        }
        if (ctx->nCph > 0) {
            cphResetSolve(ctx, drv);
        }
        if (ctx->nTvd > 0) {
            tvdResetSolve(ctx, drv);
        }
        if (ctx->nSvc > 0) {
            svcResetSolve(ctx, drv);
        }
        if (ctx->nSvd > 0) {
            svdResetSolve(ctx, drv);
        }
        if (ctx->nFlc > 0) {
            flcResetSolve(ctx, drv);
        }
        if (ctx->nShc > 0) {
            shcResetSolve(ctx, drv);
        }
        if (ctx->dsSlackOnly) {
            dsSlackOnlyResetSolve(ctx, drv, dsSlackShift);
        }
    }

    // SEQUENTIAL outer loops, exactly like OLF's AcloadFlowEngine: fully stabilize the reactive-limits loop
    // (switch + re-solve until it switches nothing), THEN fully stabilize the distributed-slack loop
    // (distribute + re-solve until balanced), and repeat the whole RL->DS sequence until a full sequence
    // triggers no inner Newton re-solve. Running RL and DS in the SAME pass converges to a different (still
    // in-tolerance) point when switching interacts with slack distribution. phase 0 = reactive limits, 1 = slack.
    std::vector<char>& live = drv.live;
    live = active;                          // scenarios still iterating outer loops (dropped on failure)
    // OLF's DefaultAcOuterLoopConfig orders DistributedSlack BEFORE ReactiveLimits (innermost first), and the
    // engine stabilizes them in that order. Match it: phase 0 = distributed slack, phase 1 = reactive limits.
    // (Switching PV->PQ before the slack is balanced leaves a large active imbalance that the undamped device
    // Newton diverges on — DS must stabilize first.)
    // The outer loops this solve drives, IN OLF'S CONFIGURED ORDER. OLF's engine
    // (AcloadFlowEngine.runOuterLoops) walks its configured list in order, drives each loop to
    // STABLE — re-solving after every unstable check — then moves to the next, restarts from the
    // top, and stops when a full sweep changes nothing. The driver below already has exactly those
    // semantics; what it lacked was the LIST: it hard-coded two loops, so every other loop had to be
    // bolted on as a host orchestrator running AFTER the whole solve, landing out of order. Naming
    // the order explicitly is what lets the remaining loops move into their proper slot.
    //
    // OLF's contingency order: DistributedSlack(1), FreezingHvdcACEmulation(2),
    // AcHvdcAcEmulationLimits(3), SecondaryVoltageControl(5), MonitoringVoltage(6),
    // ReactiveLimits(7). All but SecondaryVoltageControl are driver phases here, in that order.
    // The loops OLF runs AFTER ReactiveLimits (PhaseControl, transformer and shunt voltage control)
    // are still host orchestrators: their position relative to RL is right, but OLF restarts the
    // WHOLE sweep whenever one of them changes something, so an earlier loop is re-checked and here
    // it is not. Folding them needs the discrete tap/section tables on the device.
    int (&loopOrder)[MAX_DEVICE_LOOPS] = drv.loopOrder;
    int nLoops = 0;
    if (ctx->nds > 0 || ctx->dsSlackOnly) {
        loopOrder[nLoops++] = LOOP_DS;
    }
    if (ctx->nhv > 0) {
        loopOrder[nLoops++] = LOOP_HVDC_FREEZE;              // OLF: FreezingHvdcACEmulation
        if (ctx->hvLimitsActive) {
            loopOrder[nLoops++] = LOOP_HVDC_LIMITS;          // OLF: AcHvdcAcEmulationLimits
        }
    }
    if (ctx->nSvc > 0) {
        loopOrder[nLoops++] = LOOP_SVC_KERNEL;               // OLF order 4: SecondaryVoltageControl, IN-KERNEL
    } else if (ctx->hostLoopMask & (1 << 0)) {
        loopOrder[nLoops++] = LOOP_HOST_SVC;                 // OLF order 4: SecondaryVoltageControl, host pause
    }
    if (ctx->nmon > 0) {
        loopOrder[nLoops++] = LOOP_MONITORING;               // OLF order 5: MonitoringVoltage
    }
    if (ctx->nrl > 0) {
        loopOrder[nLoops++] = LOOP_RL;                       // OLF order 6: ReactiveLimits
    }
    if (ctx->nFlcPhase > 0 || ctx->nCph > 0) {
        loopOrder[nLoops++] = LOOP_PHASE_KERNEL;             // OLF order 7: (Incremental)PhaseControl, IN-KERNEL
    } else if (ctx->hostLoopMask & (1 << 1)) {
        loopOrder[nLoops++] = LOOP_HOST_PHASE;               // OLF order 7: PhaseControl
    }
    if (ctx->nTvc > 0) {
        loopOrder[nLoops++] = LOOP_TVC_KERNEL;               // OLF order 8: TransformerVoltageControl, IN-KERNEL
    } else if (ctx->nTvd > 0) {
        loopOrder[nLoops++] = LOOP_TVD_KERNEL;               // OLF order 8: AFTER_GENERATOR / Simple TVC, IN-KERNEL
    } else if (ctx->hostLoopMask & (1 << 2)) {
        loopOrder[nLoops++] = LOOP_HOST_TVC;                 // OLF order 8: TransformerVoltageControl, host pause
    }
    if (ctx->nFlcRpc > 0) {
        loopOrder[nLoops++] = LOOP_TRPC_KERNEL;              // OLF order 9: TransformerReactivePowerControl, IN-KERNEL
    } else if (ctx->hostLoopMask & (1 << 3)) {
        loopOrder[nLoops++] = LOOP_HOST_TRPC;                // OLF order 9: TransformerReactivePowerControl
    }
    if (ctx->nShc > 0) {
        loopOrder[nLoops++] = LOOP_SHUNT_KERNEL;             // OLF order 10: ShuntVoltageControl, IN-KERNEL
    } else if (ctx->nSvd > 0) {
        loopOrder[nLoops++] = LOOP_SVD_KERNEL;               // OLF order 10: WITH_GENERATOR shunt VC, IN-KERNEL
    } else if (ctx->hostLoopMask & (1 << 4)) {
        loopOrder[nLoops++] = LOOP_HOST_SHUNT;               // OLF order 10: ShuntVoltageControl
    }
    if (nLoops == 0) {
        loopOrder[nLoops++] = LOOP_RL;                       // nothing to drive: the old code's phase-1 fallback
    }
    // Where the realistic-voltage check STARTS applying in this sweep. AcloadFlowEngine keeps
    // checkUnrealisticStates false up to and including the last loop that can fix an unrealistic state
    // (that is the whole point of robust mode: give ReactiveLimits / the discrete
    // TransformerVoltageControl a chance to repair the state instead of failing the run on it), then
    // turns it on for the rest of the sweep and resets it at the top of the next one. A device pass at
    // index i applies loopOrder[i] to the state its solve produced, so "past the fixing loop" is
    // i > realisticFixingIdx. Not found, or no fixing loop at all, means check every solve — the
    // conservative direction, since a check that fires too early only routes to the CPU.
    int realisticFixingIdx = -1;
    for (int i = 0; i < nLoops; i++) {
        bool isFixing = ctx->realisticFixingLoop == 0
                ? loopOrder[i] == LOOP_RL
                : (ctx->realisticFixingLoop == 1
                        && (loopOrder[i] == LOOP_HOST_TVC
                            || (loopOrder[i] == LOOP_TVD_KERNEL && ctx->tvdMode != 1)));
        if (isFixing) {
            realisticFixingIdx = i;
        }
    }
    int& loopIdx = drv.loopIdx;
    loopIdx = 0;
    bool seqChanged = false;                                 // did any outer loop re-solve during the current RL->DS sequence
    if (!resume && ctx->startLoopKind >= 0) {
        // Restart after a host loop mutated the system: the first pass solves, then the sweep picks
        // up at THAT loop's slot (OLF's runOuterLoop re-checks it until STABLE, and only then does
        // the sweep restart from the first loop — seqChanged makes that restart happen).
        for (int i = 0; i < nLoops; i++) {
            if (loopOrder[i] == ctx->startLoopKind) {
                loopIdx = i;
                seqChanged = true;
            }
        }
        ctx->startLoopKind = -1;
    }
    std::vector<int> prodIters(nS, 0);                       // PER-SCENARIO productive outer iterations: OLF's
                                                             // maxOuterLoopIterations is a per-RUN budget — a global
                                                             // counter capped busy chunks collectively (violation-diff
                                                             // root cause: a scenario solved solo matched the CPU
                                                             // exactly but diverged inside a full 500-chunk)
    // A phase transition after a NO-SWITCH pass must NOT re-solve: OLF's runOuterLoop re-solves only on an
    // UNSTABLE check — a STABLE loop is followed by checking the NEXT loop against the SAME state. The device
    // Newton forces >=1 iteration per solve (matching OLF's per-run behavior), so an extra no-op pass would
    // move an already-converged state off the CPU's by ~the stopping residual (measured 4e-4 kV on the
    // DS->RL transition with a distributed base). solvePass=false skips the inner Newton for such a pass.
    bool solvePass = true;
    if (resume) {
        // Continue the paused sequence: the loop the host owned has just run, so step PAST it and
        // let its verdict drive the pass exactly as a device loop's would — UNSTABLE re-solves only
        // the scenarios it changed, STABLE advances without a solve (OLF's rule).
        loopIdx = ctx->pausedLoopIdx;
        seqChanged = ctx->pausedSeqChanged != 0;
        prodIters = ctx->pausedProdIters;
        live = ctx->pausedLive;
        if (ctx->pausedStatus.size() == res.status.size()) {
            res.status = ctx->pausedStatus;                  // the state is the converged one the pause left: a
        }                                                    // STABLE host verdict then re-CHECKS RL on it (rlActive)
        bool anyChanged = false;
        for (int s = 0; s < nS; s++) {
            char ch = s < (int) ctx->hostChanged.size() ? ctx->hostChanged[s] : 0;
            active[s] = ch ? 1 : 0;
            anyChanged |= ch != 0;
        }
        if (anyChanged) {
            seqChanged = true;
            solvePass = true;                                // the host loop moved targets -> re-solve
            // ...and it CONSUMES the outer-iteration budget, exactly as a device loop's change does.
            // Without this a host loop that keeps going UNSTABLE never retires a scenario: each resume
            // starts a fresh `outer` budget, so the pause/resume cycle spins forever (measured: a
            // 2000-contingency rte6515 slice with SecondaryVoltageControl did not finish in 28 min).
            for (int s = 0; s < nS; s++) {
                if (active[s] && ++prodIters[s] >= RL_MAX_OUTER) {
                    traceReject(s, "host-loop-outer-budget", -1, 0);
                    res.status[s] = -2;                      // budget exhausted -> CPU replays it
                    live[s] = 0;
                    active[s] = 0;
                }
            }
        } else {
            solvePass = false;
            for (int s = 0; s < nS; s++) {
                active[s] = live[s];                         // stable: re-CHECK every live scenario
            }
            if (++loopIdx >= nLoops) {                       // ...and advance, as the stable branch does
                if (!seqChanged) {
                    loopIdx = nLoops;                        // a full sweep changed nothing -> converged
                } else {
                    seqChanged = false;
                    loopIdx = 0;
                }
            }
        }
    }
    // OLF_GPU_SLOT_DUMP=<prefix>: the traced scenario's (OLF_GPU_LOOP_TRACE_CTG) row targets, initial state and row
    // modes as the driver starts - after every host override and every device seed - one file per solve
    // (<prefix>.<k>), "<row> <target> <x> <mode>" with every double exact. Two runs of one scenario that should be
    // identical are diffed row by row: the rows that differ name the writer.
    if (!resume && ctx->loopTraceSlot >= 0 && ctx->loopTraceSlot < nS) {
        static const char* slotDump = std::getenv("OLF_GPU_SLOT_DUMP");
        static int slotDumps = 0;
        if (slotDump != nullptr) {
            const size_t off = static_cast<size_t>(ctx->loopTraceSlot) * n;
            std::vector<double> ht(n), hx(n);
            std::vector<int> hm(n);
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            CUDA_CHECK(cudaMemcpy(ht.data(), ctx->dTarget.as<double>() + off, n * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hx.data(), ctx->dX.as<double>() + off, n * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hm.data(), ctx->dMode.as<int>() + off, n * sizeof(int), cudaMemcpyDeviceToHost));
            std::string path = std::string(slotDump) + "." + std::to_string(slotDumps++);
            if (FILE* f = std::fopen(path.c_str(), "w")) {
                for (int r = 0; r < n; r++) {
                    std::fprintf(f, "%d %.17g %.17g %d\n", r, ht[r], hx[r], hm[r]);
                }
                std::fclose(f);
            }
        }
    }
    static const bool olTime = std::getenv("OLF_GPU_PHASE_TIME") != nullptr || rlReasonsOn();
    int olPassSolved = 0;
    bool olFirstSolveDone = resume;                          // a resumed call's passes are all re-solves
    int olTriggerKind = resume ? ctx->pausedLoopKind : -1;  // the loop whose check made the next pass re-solve
    int& outer = drv.outer;
    for (outer = 0; outer < maxOuter && loopIdx < nLoops; outer++) {
    gOuterPasses++;
    if (olTime) CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    const double olT0 = olTime ? olNowMs() : 0.0;
    const long long olAI0 = gActiveIterSum, olWI0 = gWidthIterSum, olNI0 = gNewtonIters;
    const bool olSolving = solvePass;
    if (solvePass) {
        gSolvePasses++;
    }
    std::copy(active.begin(), active.end(), wasActive.begin());   // scenarios (re)solved in this inner pass
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, active.data(), nS * sizeof(char), cudaMemcpyHostToDevice));
    int nActive = 0;
    for (char a : active) {
        if (a) nActive++;
    }
    // ---- batched fill for all active scenarios (J + F, or F-ONLY: the final residual fill and
    //      mode 4's solve-only iterations evaluate just the residual and skip the vals memset —
    //      the J buffer is never read before the next full fill re-zeroes it). A lambda so the
    //      LINE_SEARCH backtrack below can re-evaluate at a reduced step. ----
    // ---- IncrementalTransformerVoltageControl as a DRIVER PHASE (OLF order 8). OLF's own rule, per
    //      controlled bus: if |targetV - V| > halfDeadband, take dV/drho from the Jacobian at the converged
    //      state (unit RHS at the rho identity row, read the bus V entry), deltaR1 = diffV / sensi, move to
    //      the tap whose r1 is closest to r1 + deltaR1 within maxTapShift and the ratchet's allowed
    //      direction (MAX_DIRECTION_CHANGE 3), skip controllers with |sensi| < MIN_SENSI_FILTER (0.05).
    //      The sensitivity is the same forward solve GpuBatchIncrementalTvc used to make through
    //      solveBatchSensitivities — one host round trip per round — done here against the factors the
    //      driver already holds, with no host copy of the states. Single controller per controlled bus
    //      (the JNI upload refuses shared buses, which stay on the host orchestrator). ----
    // One unit-right-hand-side sensitivity solve on every scenario's CURRENT Jacobian (dVals, filled for all
    // scenarios by the caller; dB = the right-hand sides, dDx = the solutions), for the in-kernel outer loops.
    // The UBATCH mode (3) solves the whole batch in one call on the factors they hold - the analysis once per
    // context, the factorization with the first controller's solve. The robust mode (6) holds ONE matrix object
    // repointed per scenario and keeps no factors from one scenario to the next, so it factors and solves scenario
    // by scenario - a refactorization per (scenario, controller), affordable at the robust retry's chunk sizes and
    // the price of the loops running there at all: they used to report themselves STABLE in mode 6, so a scenario
    // retried in it (or a base load flow that fell into it) with a transformer, shunt, flow or secondary voltage
    // control out of its deadband got a wrong answer and nothing flagged it.
    std::vector<char>& loopRejected = drv.loopRejected;
    loopRejected.assign(nS, 0);                   // a device loop routed the scenario to the CPU this pass
    // ---- IncrementalShuntVoltageControl as a DRIVER PHASE (OLF order 10): deadband per controlled bus,
    //      dV/dB from a unit RHS at each needed controller's SHUNT_B identity row (the driver's own
    //      factors), OLF's adjustB round-robin per group, the section total pinned at the identity row
    //      and the static (dG, dB) through the shunt-override list. Groups with several controller
    //      shunts ARE handled (the round-robin runs in the group kernel). ----
    // ---- the FLOW-OBSERVED incremental loops as DRIVER PHASES: kindMask selects the phase controllers
    //      (OLF order 7: kinds 1+2) or the RPC controllers (OLF order 9: kind 0). Same skeleton as the
    //      voltage-observed phases; the observable is the controlled branch's flow gradient dotted with
    //      the unit-RHS solution (OLF's EquationTerm.calculateSensi), evaluated by the codegen'd branch
    //      functions on the device. ----
    // ---- CONTINUOUS PhaseControl as a DRIVER PHASE (OLF order 7), see cphCheckBatch. ----
    // ---- the DISCRETE voltage-control loops as DRIVER PHASES: TransformerVoltageControl (AFTER_GENERATOR /
    //      WITH_GENERATOR, OLF order 8) and ShuntVoltageControl (WITH_GENERATOR, OLF order 10). No sensitivities:
    //      OLF's decisions read the converged state only (deadband, clamp bounds, closest tap / section). ----
    // ---- SecondaryVoltageControl as a DRIVER PHASE (OLF order 4). The whole loop: the re-enable step and
    //      the per-zone trigger on the paused state's own qBuf, one unit-RHS solve per controlled bus on the
    //      driver's own factors for the K and pilot sensitivities, the small dense zone system per scenario,
    //      and the moved targets applied to the controlled buses' V rows (plus the switch-back comparison
    //      target for a pinned controller). Nothing goes back to the host, so the loop sits in OLF's order
    //      instead of after the sweep. ----
    if (solvePass) {

    // LINE_SEARCH per-scenario backtracking state (lastNorm2 / haveLast, per pass — a phase re-entry retargets the system, voiding the step relation)
    // Per scenario, the linear-solve residual ratio of this pass's LAST solve — scored only when the pass runs
    // its Newton budget out (see the it == maxIter branch), so a healthy run never pays for it.
    std::vector<double> lastSolveRatio(nS, 0.0);
    std::vector<double> lsLastSsq(lsMode == 1 ? nS : 0, 0.0);
    std::vector<char> lsHaveLast(lsMode == 1 ? nS : 0, 0);
    std::vector<double> hMu(lsMode == 1 ? nS : 0, -1.0);

    for (int it = 0; it <= maxIter && nActive > 0; it++) {
        gLastPassIters = it;
        gNewtonIters++;
        bool fOnly = it == maxIter;
        fillActiveBatch(ctx, drv, fOnly);

        // ---- residual check + retirement ----
        // Normal mode (tol > 0): every iteration copies the batched mismatch to the host (the
        // only thing crossing PCIe) and retires converged/diverged scenarios.
        // Device fixed-iteration mode (tol <= 0): NO per-iteration host sync and NO retirement
        // — every scenario runs exactly maxIter steps on the device; the mismatch is read back
        // ONCE, on the final pass, only to report ||F||inf. This drops the per-iteration D->H
        // copy of the full batched mismatch. Tradeoff: a scenario going non-finite is not
        // isolated, so it can poison a uniform-batched factorization — use tol > 0 if
        // scenarios may diverge.
        if (!deviceFixedIter || it == maxIter) {
            std::vector<double> hNorms;
            if (!deviceFixedIter) {
                // Device-side reduction: read back nS*4 norms, not the 70-MB mismatch.
                readNormsBatch(ctx, drv, hNorms);
                // ---- LINE_SEARCH backtrack (per scenario): the full step applied at it-1 may have
                //      RAISED that scenario's ||F||2 — walk it back to x = xPrev + mu*dx with
                //      mu = stepFold^-k until the 2-norm improves, mirroring
                //      OLF's LineSearchStateVectorScaling. Judged on the 2-norm
                //      (ssq, monotonic-equivalent); a non-finite mismatch backtracks too. Scenarios
                //      not backtracking keep their state (mu sentinel -1 masks the step kernel). ----
                if (lsMode == 1) {
                    auto needsBt = [&](int s2) {
                        return active[s2] && lsHaveLast[s2]
                                && (hNorms[s2 * 4 + 2] == 0.0 || hNorms[s2 * 4 + 1] >= lsLastSsq[s2]);
                    };
                    bool anyBt = false;
                    for (int s2 = 0; s2 < nS; s2++) {
                        if (needsBt(s2)) anyBt = true;
                    }
                    for (int k = 1; anyBt && k <= ctx->lsMaxIter; k++) {
                        double mu = 1.0 / std::pow(ctx->lsStepFold, k);
                        for (int s2 = 0; s2 < nS; s2++) hMu[s2] = needsBt(s2) ? mu : -1.0;
                        CUDA_CHECK(cudaMemcpyAsync(ctx->dStepFactor.p, hMu.data(), nS * sizeof(double), cudaMemcpyHostToDevice, ctx->stream));
                        stepFromPrevBatchMasked<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, n * nS,
                                ctx->dXprevLs.as<double>(), ctx->dDx.as<double>(), ctx->dStepFactor.as<double>(), ctx->dX.as<double>());
                        fillActiveBatch(ctx, drv, false);              // full fill: J must sit at the accepted x (like the seq path)
                        readNormsBatch(ctx, drv, hNorms);
                        anyBt = false;
                        for (int s2 = 0; s2 < nS; s2++) {
                            if (needsBt(s2)) anyBt = true;
                        }
                    }
                    for (int s2 = 0; s2 < nS; s2++) {
                        if (active[s2]) {
                            lsLastSsq[s2] = hNorms[s2 * 4 + 1];
                            lsHaveLast[s2] = 1;
                        }
                    }
                }
            } else {
                CUDA_CHECK(cudaMemcpy(hF.data(), ctx->dF.p, nTot * sizeof(double), cudaMemcpyDeviceToHost));
            }
            // A pass about to report exhaustion: score its LAST linear solve (dVals/dDx/dB still hold it - the
            // it == maxIter fill is F-only). A wrong factorization sits at 1e5..1e13, a sound solve at 1e-9, so this
            // separates "OLF's budget, spent moving" from "the step was junk" without the per-pass cost of scoring
            // every pass (measured: 34 s on a 19 s rte6515 run).
            if (it == maxIter && !deviceFixedIter && nActive > 0) {
                if (!ctx->dLinRp.p) {
                    ctx->dLinRp.alloc((size_t) n * nS * sizeof(double));
                    ctx->dLinRm.alloc((size_t) n * nS * sizeof(double));
                    ctx->dLinSc.alloc((size_t) 3 * nS * sizeof(double));
                }
                linResidRowBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, nnz, ctx->st->dRowPtr.as<int>(),
                        ctx->st->dColIdx.as<int>(), ctx->dVals.as<double>(), ctx->dDx.as<double>(),
                        ctx->dB.as<double>(), ctx->dLinRp.as<double>(), ctx->dLinRm.as<double>());
                linResidScoreBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(n, nS, ctx->dLinRp.as<double>(),
                        ctx->dLinRm.as<double>(), ctx->dF.as<double>(), ctx->dLinSc.as<double>());
                std::vector<double> hsc((size_t) 3 * nS);
                CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                CUDA_CHECK(cudaMemcpy(hsc.data(), ctx->dLinSc.p, (size_t) 3 * nS * sizeof(double), cudaMemcpyDeviceToHost));
                for (int s2 = 0; s2 < nS; s2++) {
                    lastSolveRatio[s2] = hsc[3 * s2 + 2];
                }
            }
            bool anyChanged = false;
            // Inner-Newton convergence test. Default (convEpsPerEq <= 0): the device infinity-norm floor
            // ‖F‖∞ < tol (stricter than OLF's usual criterion). When convEpsPerEq > 0: OLF's EXACT criterion
            // — DefaultNewtonRaphsonStoppingCriteria stops at ‖F‖₂ < sqrt(convEpsPerEq² · n) — so a batched SA
            // can be run with the SAME stopping decision as the CPU loadflow for an apples-to-apples timing
            // comparison. ‖F‖∞ is still recorded in res.norm[s] for reporting either way. n counts the padded
            // system rows; for batched scenarios (no islanding — those go to the CPU) that equals OLF's active
            // equation count, since disabled-branch rows keep the bus equations and identity pads contribute 0.
            const double olf2NormThreshold = convEpsPerEq > 0.0 ? std::sqrt(convEpsPerEq * convEpsPerEq * n) : 0.0;
            auto thr2Of = [&](int sIdx) {
                return convEpsPerEq > 0.0 && !ctx->hNActive.empty()
                        ? std::sqrt(convEpsPerEq * convEpsPerEq * ctx->hNActive[sIdx]) : olf2NormThreshold;
            };
            for (int s = 0; s < nS; s++) {
                if (!active[s]) continue;
                double norm = 0.0;
                double ssq = 0.0;
                bool finite = true;
                double maxRatio = 0.0;
                if (!deviceFixedIter) {
                    norm = hNorms[(size_t) s * 4];
                    ssq = hNorms[(size_t) s * 4 + 1];
                    finite = hNorms[(size_t) s * 4 + 2] != 0.0;
                    maxRatio = hNorms[(size_t) s * 4 + 3];
                } else {
                    for (int i = 0; i < n; i++) {
                        double v = hF[(size_t) s * n + i];
                        if (!std::isfinite(v)) {
                            finite = false;
                            break;
                        }
                        norm = std::max(norm, std::fabs(v));
                        ssq += v * v;
                    }
                }
                // PER_EQUATION_TYPE (per-row thresholds set on the context): OLF's
                // PerEquationTypeStoppingCriteria — stop when EVERY row clears its type threshold.
                bool converged = ctx->perRowCriteria ? (finite && maxRatio < 1.0)
                        : convEpsPerEq > 0.0 ? (std::sqrt(ssq) < thr2Of(s)) : (norm < tol);
                if (iterLog) {
                    char line[192];
                    std::snprintf(line, sizeof line, "ITER_LOG outer=%d s=%d it=%d norm2=%.17e thr2=%.17e conv=%d",
                                  outer, s, it, std::sqrt(ssq), thr2Of(s), converged ? 1 : 0);
                    iterLogLines.emplace_back(line);
                }
                // OLF_GPU_CONV_TRACE=<slot>|all: the stopping DECISION itself - the 2-norm, the
                // threshold it is held against and the active row count behind that threshold. Two
                // arms that agree on every residual and still stop one Newton step apart differ
                // here and nowhere else, and the exit norms alone cannot show it.
                static const char* convTraceEnv = std::getenv("OLF_GPU_CONV_TRACE");
                static const int convTraceSlot = convTraceEnv == nullptr ? -1
                        : (std::string(convTraceEnv) == "all" ? -2 : std::atoi(convTraceEnv));
                if (convTraceSlot == -2 || convTraceSlot == s) {
                    std::fprintf(stderr, "CONV_TRACE s=%d it=%d norm2=%.17e thr2=%.17e nActive=%d inf=%.17e perRow=%d maxRatio=%.6e converged=%d\n",
                                 s, it, std::sqrt(ssq), thr2Of(s), ctx->hNActive.empty() ? n : ctx->hNActive[s], norm,
                                 ctx->perRowCriteria ? 1 : 0, maxRatio, converged ? 1 : 0);
                    // the eight worst rows, by index: WHERE the residual sits, so a step that lands
                    // two orders of magnitude from the other arm's can be traced to its rows
                    std::vector<double> hFs(n);
                    CUDA_CHECK(cudaMemcpy(hFs.data(), ctx->dF.as<double>() + (size_t) s * n, (size_t) n * sizeof(double), cudaMemcpyDeviceToHost));
                    std::vector<int> ord(n);
                    for (int i = 0; i < n; i++) ord[i] = i;
                    std::partial_sort(ord.begin(), ord.begin() + std::min(n, 8), ord.end(),
                                      [&](int a, int b) { return std::fabs(hFs[a]) > std::fabs(hFs[b]); });
                    std::fprintf(stderr, "CONV_TOP s=%d it=%d :", s, it);
                    for (int k = 0; k < std::min(n, 8); k++) {
                        std::fprintf(stderr, " [%d]=%.9e", ord[k], hFs[ord[k]]);
                    }
                    std::fprintf(stderr, "\n");
                }
                static const char* sdEnv = std::getenv("OLF_GPU_STATE_DUMP");
                if (sdEnv != nullptr && s == 0 && it <= std::atoi(sdEnv) && n <= 64) {
                    stateDumpBatch<<<1, 1, 0, ctx->stream>>>(n, ctx->dF.as<double>(), ctx->dMode.as<int>(),
                            ctx->dTarget.as<double>(), ctx->dX.as<double>(), s, it);
                    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                }
                static const char* wrEnv = std::getenv("OLF_GPU_WORST_ROW");
                if (wrEnv != nullptr && s == 0 && it <= std::atoi(wrEnv)) {
                    worstRowBatch<<<1, 1, 0, ctx->stream>>>(n, ctx->dF.as<double>(), ctx->dMode.as<int>(),
                            ctx->dTarget.as<double>(), ctx->dX.as<double>(), s);
                    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                }
                if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
                    std::fprintf(stderr, "BATCH_TRACE s=%d it=%d infnorm=%.17e norm2=%.17e thr=%.17e conv=%d\n",
                                 s, it, norm, std::sqrt(ssq), thr2Of(s), converged ? 1 : 0);
                }
                if (s == ctx->loopTraceSlot) {
                    // The traced scenario's Newton step, host-side: the norms the convergence test saw,
                    // the MAX_VOLTAGE_CHANGE factor the step was scaled by, and the row that carries the
                    // largest residual with what sits on it. OLF's NR_EXIT prints the same norms after
                    // each of ITS iterations; a different iteration COUNT on the same system is where
                    // two converged answers part (network.xiidm/IZUZ: OLF 1 iteration, device 2).
                    std::vector<double> hF(n), hT(n), hX(n);
                    std::vector<int> hM(n);
                    double fac = 1.0;
                    CUDA_CHECK(cudaMemcpy(hF.data(), ctx->dF.as<double>() + (size_t) s * n, n * sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(hT.data(), ctx->dTarget.as<double>() + (size_t) s * n, n * sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(hX.data(), ctx->dX.as<double>() + (size_t) s * n, n * sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(hM.data(), ctx->dMode.as<int>() + (size_t) s * n, n * sizeof(int), cudaMemcpyDeviceToHost));
                    if (ctx->clampStep && ctx->dStepFactor.p) {
                        CUDA_CHECK(cudaMemcpy(&fac, ctx->dStepFactor.as<double>() + s, sizeof(double), cudaMemcpyDeviceToHost));
                    }
                    int wr = 0;
                    for (int r = 1; r < n; r++) if (std::fabs(hF[r]) > std::fabs(hF[wr])) wr = r;
                    // ...and the RAW step's largest entry (dDx still holds it), which is what the
                    // MAX_VOLTAGE_CHANGE factor was computed from
                    std::vector<double> hDx(n);
                    CUDA_CHECK(cudaMemcpy(hDx.data(), ctx->dDx.as<double>() + (size_t) s * n, n * sizeof(double), cudaMemcpyDeviceToHost));
                    int wd = 0;
                    for (int r = 1; r < n; r++) if (std::fabs(hDx[r]) > std::fabs(hDx[wd])) wd = r;
                    std::fprintf(stderr, "SLOT_TRACE s=%d it=%d infnorm=%.17e norm2=%.17e thr=%.17e conv=%d stepFactor=%.9f worstRow=%d F=%.12e mode=%d target=%.9f x=%.9f maxDxRow=%d dx=%.9f\n",
                                 s, it, norm, std::sqrt(ssq), thr2Of(s), converged ? 1 : 0, fac, wr, hF[wr], hM[wr], hT[wr], hX[wr], wd, hDx[wd]);
                }
                res.iters[s] = it;
                res.norm[s] = norm;
                if (deviceFixedIter) {
                    res.status[s] = finite ? 1 : -1;         // ran maxIter; no convergence test
                } else if (!finite) {                        // a diverged scenario must not poison the batch
                    res.status[s] = -1;
                    active[s] = 0;
                    live[s] = 0;                             // failed: stop iterating its outer loops
                    nActive--;
                    anyChanged = true;
                } else if (converged && it > 0 && it < maxIter) {
                    // Force >=1 iteration, like CPU NewtonRaphson: a warm-started
                    // state may already meet the global 2-norm threshold yet leave a just-flipped identity row
                    // (PV target) unpinned — one Newton step pins it. Retiring at it==0 skips that solve.
                    // And NOT at the budget: OLF's NewtonRaphson.run counts a step before testing it and
                    // reports MAX_ITERATION_REACHED whenever iterations >= maxIterations, so a test that
                    // passes after the LAST allowed step is not CONVERGED there — the it == maxIter branch
                    // below reports it at OLF's budget (with a sound last solve).
                    res.status[s] = 0;
                    active[s] = 0;
                    nActive--;
                    anyChanged = true;
                    CUDA_CHECK(cudaMemcpyAsync(ctx->dXLastConv.as<double>() + (size_t) s * n, ctx->dX.as<double>() + (size_t) s * n,
                                               n * sizeof(double), cudaMemcpyDeviceToDevice, ctx->stream));
                } else if (it == maxIter) {
                    // exhausted iterations. With fixed inner iterations those are the K steps asked for, not OLF's
                    // Newton budget, so the scenario is REJECTED (the CPU replays it) rather than reported
                    // MAX_ITERATION_REACHED from the device.
                    //
                    // A BAD LAST SOLVE is a reject, not OLF's answer: the pass ran its whole budget on steps the
                    // factorization got wrong (rte6515 busbar sections [5000,6000): 885 scenarios sat at
                    // ||F||inf ~ 56 for 30 iterations, reported as MAX_ITERATION_REACHED, 10,067 violations
                    // against the CPU's 71,553). Reported max_iter has to mean "OLF's budget, spent on sound
                    // steps". On the BATCH OF ONE (the base load flow) -3 takes the pivoted device retry first;
                    // a chunk scenario replays on the CPU either way (needsCpuFallback covers both).
                    bool badSolve = !(lastSolveRatio[s] <= LIN_SOLVE_FAIL_RATIO);
                    if (badSolve) {
                        traceReject(s, "max-iter-bad-solve", outer, it);
                        if (std::getenv("OLF_GPU_REJECT_TRACE") != nullptr) {
                            std::fprintf(stderr, "GPU_REJECT slot=%d reason=max-iter-bad-solve ratio=%.3e norm=%.6e\n",
                                         s, lastSolveRatio[s], norm);
                        }
                    }
                    res.status[s] = badSolve ? (nS == 1 ? -3 : -2) : 1;
                    active[s] = 0;
                    live[s] = 0;                             // did not converge: stop iterating its outer loops
                    nActive--;
                    anyChanged = true;
                }
            }
            if (!deviceFixedIter && anyChanged) {
                CUDA_CHECK(cudaMemcpy(ctx->dActive.p, active.data(), nS * sizeof(char), cudaMemcpyHostToDevice));
            }
        }
        if (deviceFixedIter) {
            if (it == maxIter) {
                break;                                       // final residual recorded; done
            }
        } else if (nActive == 0) {
            break;
        }

        // Scenarios still being FACTORIZED this iteration. Summed over iterations this is the batch's
        // real work measure: the CPU pays each contingency its own iteration count, so a batch is only
        // as efficient as its retirement — a batch that keeps every scenario alive until the last
        // straggler converges pays the WORST case for all of them.
        gActiveIterSum += nActive;
        if (lsMode == 1) {                                   // LINE_SEARCH: remember the pre-step iterates
            CUDA_CHECK(cudaMemcpyAsync(ctx->dXprevLs.p, ctx->dX.p, static_cast<size_t>(n) * nS * sizeof(double),
                                       cudaMemcpyDeviceToDevice, ctx->stream));
        }
        if (ctx->mode == 3) {
        // ---- UBATCH solve: ANALYZE one block once (cheap), then a linear solve over all nS systems
        //      via CUDSS_CONFIG_UBATCH_SIZE on the single per-block matrix, FACTORIZED every iteration
        //      (honest Newton). ----
        writeIdentityForInactive<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, nnz, ctx->st->dRowPtr.as<int>(),
                ctx->st->dDiagPos.as<int>(), ctx->dActive.as<char>(), ctx->dVals.as<double>(), ctx->dF.as<double>());
        negateIntoBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n * nS, ctx->dF.as<double>(), ctx->dB.as<double>());
        hazRepairRightHandSide(ctx, nS, ctx->dB.as<double>());
        if (!ctx->analyzed) {
            bool timeIt = std::getenv("OLF_GPU_TRACE") != nullptr;
            long long t0 = 0;
            if (timeIt) {
                CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                t0 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
            }
            gAnalysesMain++;
            { AnalysisTimer at_(__LINE__); CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_ANALYSIS, ctx->ds.config, ctx->ds.data, ctx->matUb, ctx->matUbX, ctx->matUbB)); }
            double ms = -1.0;
            if (timeIt) {
                CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                ms = (std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count() - t0) / 1.0e6;
            }
            traceLuNnz(ctx->ds.handle, ctx->ds.data, "ubatch", ctx->nScenarios, ms);
            ctx->analyzed = true;
        }
        // ONE cudssExecute per iteration (phases are OR-able bit flags): halves the per-iteration
        // host-side phase overhead vs separate FACTORIZATION + SOLVE calls.
        int ubPhase = CUDSS_PHASE_SOLVE | CUDSS_PHASE_FACTORIZATION;
        // ---- solve-only COMPACTION: when few scenarios remain active, gather their vals/B into
        //      compact buffers and solve a NARROWER ubatch on the dedicated dsC context, then
        //      scatter dx back. mode 3 only (factorizes every iteration — stateless across width
        //      changes); width quantized to power-of-2 tiers so the cheap single-block re-ANALYSIS
        //      runs only on tier change. Pad slots duplicate the first active scenario (keeps the
        //      batch non-singular) and are discarded by the scatter. dDx is zeroed first so
        //      inactive scenarios' frozen states are not perturbed by stale solutions (the
        //      full-width update kernel adds dx over ALL scenarios). ----
        // Compact once at most half of the chunk is active. Above it every pass factorizes the full width, padded
        // lanes included (network.xiidm: 278,144 lane-iterations for 234,088 useful). Compact widths are powers of
        // two, so a tier change re-analyzes rarely.
        constexpr double COMPACT_FRAC = 0.5;
        const int compactLimit = std::max(1, std::min(nS, static_cast<int>(COMPACT_FRAC * nS)));
        bool compactSolve = !deviceFixedIter && ctx->mode == 3 && nActive > 0 && nActive <= compactLimit
                && nActive < nS;
        if (!compactSolve) {
            gWidthIterSum += nS;                             // full-width factorization this iteration
        }
        int iterLogWidth = nS;                               // OLF_GPU_ITER_LOG: the width this iteration factorized
        if (compactSolve) {
            if (ctx->compactCap == 0) {                       // lazy first-engage setup
                ctx->compactCap = std::max(1, compactLimit);
                ctx->dValsC.alloc(static_cast<size_t>(nnz) * ctx->compactCap * sizeof(double));
                ctx->dBC.alloc(static_cast<size_t>(n) * ctx->compactCap * sizeof(double));
                ctx->dDxC.alloc(static_cast<size_t>(n) * ctx->compactCap * sizeof(double));
                ctx->dMapC.alloc(static_cast<size_t>(ctx->compactCap) * sizeof(int));
                CUDSS_CHECK(cudssCreate(&ctx->dsC.handle));
                CUDSS_CHECK(cudssSetStream(ctx->dsC.handle, ctx->stream));
                CUDSS_CHECK(cudssConfigCreate(&ctx->dsC.config));
                applyReorderAlg(ctx->dsC.config, ctx->st->nzi > 0);
                int irC = ctx->st->nzi > 0 ? irBaseSteps() : IR_N_STEPS;
                CUDSS_CHECK(cudssConfigSet(ctx->dsC.config, CUDSS_CONFIG_IR_N_STEPS, &irC, sizeof(irC)));
                CUDSS_CHECK(cudssMatrixCreateCsr(&ctx->matUbC, n, n, nnz,
                        ctx->st->dRowPtr.as<int>(), nullptr, ctx->st->dColIdx.as<int>(), ctx->dValsC.as<double>(),
                        OLF_CUDSS_CSR_TYPES, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
                CUDSS_CHECK(cudssMatrixCreateDn(&ctx->matUbCB, n, 1, n, ctx->dBC.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
                CUDSS_CHECK(cudssMatrixCreateDn(&ctx->matUbCX, n, 1, n, ctx->dDxC.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
                ctx->dRC.alloc(static_cast<size_t>(n) * ctx->compactCap * sizeof(double));
                ctx->dDeltaC.alloc(static_cast<size_t>(n) * ctx->compactCap * sizeof(double));
                CUDSS_CHECK(cudssMatrixCreateDn(&ctx->matUbCR, n, 1, n, ctx->dRC.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
                CUDSS_CHECK(cudssMatrixCreateDn(&ctx->matUbCDelta, n, 1, n, ctx->dDeltaC.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
            }
            std::vector<int> cmap;
            cmap.reserve(ctx->compactCap);
            for (int cs = 0; cs < nS; cs++) {
                if (active[cs]) {
                    cmap.push_back(cs);
                }
            }
            int nReal = static_cast<int>(cmap.size());
            int w = 1;
            while (w < nReal) {
                w <<= 1;
            }
            if (w > ctx->compactCap) {
                w = ctx->compactCap;
            }
            while (static_cast<int>(cmap.size()) < w) {
                cmap.push_back(cmap[0]);
            }
            gWidthIterSum += w;                              // compacted width actually factorized
            iterLogWidth = -w;
            gCompactIters++;
            CUDA_CHECK(cudaMemcpyAsync(ctx->dMapC.p, cmap.data(), w * sizeof(int), cudaMemcpyHostToDevice, ctx->stream));
            gatherScenarios<<<grid(nnz * w), BLOCK, 0, ctx->stream>>>(nnz, w, ctx->dMapC.as<int>(), ctx->dVals.as<double>(), ctx->dValsC.as<double>());
            gatherScenarios<<<grid(n * w), BLOCK, 0, ctx->stream>>>(n, w, ctx->dMapC.as<int>(), ctx->dB.as<double>(), ctx->dBC.as<double>());
            // A new batch size needs a FRESH data object and a fresh ANALYSIS. Setting
            // CUDSS_CONFIG_UBATCH_SIZE on the existing data and skipping the analysis is accepted by
            // cuDSS and even looks faster on a bench -- measured 2026-09-16 at -11% warm with exact
            // violation parity -- and then fails with an illegal memory access on a bigger network:
            // the data object's buffers are sized for the batch size it was analyzed at. The analysis
            // is cheap (~64 ms) and costs the same at any batch size, but its STATE belongs to one.
            cudssData_t compactData = nullptr;
            {
                // Tier change to a width analyzed before: its own data object, the batch size set back on the
                // (shared) config. A new width: a fresh data object and its single-block ANALYSIS, kept.
                cudssData_t& cached = ctx->compactData[w];
                if (cached == nullptr) {
                    CUDSS_CHECK(cudssDataCreate(ctx->dsC.handle, &cached));
                    CUDSS_CHECK(cudssConfigSet(ctx->dsC.config, CUDSS_CONFIG_UBATCH_SIZE, &w, sizeof(w)));
                    gAnalysesCompact++;
                    { AnalysisTimer at_(__LINE__); CUDSS_CHECK(cudssExecute(ctx->dsC.handle, CUDSS_PHASE_ANALYSIS, ctx->dsC.config, cached, ctx->matUbC, ctx->matUbCX, ctx->matUbCB)); }
                } else if (w != ctx->compactWidth) {
                    CUDSS_CHECK(cudssConfigSet(ctx->dsC.config, CUDSS_CONFIG_UBATCH_SIZE, &w, sizeof(w)));
                }
                ctx->compactWidth = w;
                compactData = cached;
            }
            { AnalysisTimer at_(__LINE__, 'F'); CUDSS_CHECK(cudssExecute(ctx->dsC.handle, CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_SOLVE, ctx->dsC.config, compactData, ctx->matUbC, ctx->matUbCX, ctx->matUbCB)); }
            if (ctx->st->nzi > 0 && ctx->dRC.p != nullptr && ctx->dIrScore.p != nullptr && ctx->dIrScore.cap >= static_cast<size_t>(w) * sizeof(double)) {
                // adaptive refinement on the compacted lanes (all w are real or a duplicate of a real one)
                refineLanes(ctx, w, n, nnz, ctx->dValsC.as<double>(), ctx->dDxC.as<double>(), ctx->dBC.as<double>(),
                        ctx->dRC.as<double>(), ctx->dDeltaC.as<double>(), nullptr, [&]() {
                    AnalysisTimer at_(__LINE__, 'S');
                    CUDSS_CHECK(cudssExecute(ctx->dsC.handle, CUDSS_PHASE_SOLVE, ctx->dsC.config, compactData, ctx->matUbC, ctx->matUbCDelta, ctx->matUbCR));
                });
            }
            CUDA_CHECK(cudaMemsetAsync(ctx->dDx.p, 0, static_cast<size_t>(n) * nS * sizeof(double), ctx->stream));
            scatterScenarios<<<grid(n * nReal), BLOCK, 0, ctx->stream>>>(n, nReal, ctx->dMapC.as<int>(), ctx->dDxC.as<double>(), ctx->dDx.as<double>());
        } else {
            { AnalysisTimer at_(__LINE__, ((ubPhase) & (CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_REFACTORIZATION)) ? 'G' : ((ubPhase) & CUDSS_PHASE_ANALYSIS) ? 'A' : 'S'); CUDSS_CHECK(cudssExecute(ctx->ds.handle, ubPhase, ctx->ds.config, ctx->ds.data, ctx->matUb, ctx->matUbX, ctx->matUbB)); }
            if (ctx->st->nzi > 0 && ctx->dR.p != nullptr && ctx->matUbR != nullptr && ctx->dIrScore.p != nullptr) {
                // adaptive refinement on the full-width UBATCH solve (see refineLanes)
                refineLanes(ctx, nS, n, nnz, ctx->dVals.as<double>(), ctx->dDx.as<double>(), ctx->dB.as<double>(),
                        ctx->dR.as<double>(), ctx->dDelta.as<double>(), ctx->dActive.as<char>(), [&]() {
                    AnalysisTimer at_(__LINE__, 'S');
                    CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_SOLVE, ctx->ds.config, ctx->ds.data, ctx->matUb, ctx->matUbDelta, ctx->matUbR));
                });
            }
        }
        // OLF_GPU_DX_DUMP=<path>: the FIRST solve's first Newton step, written out whole — the
        // assembled Jacobian (CSR triplets), the mismatch F it was built at, and the step dx it
        // produced. The CPU arm writes the same three under OLF_NR_DX_DUMP, and the two are
        // comparable WITHOUT a row map: the device's COLUMN index is OLF's own variable row
        // (GpuBatchPrecomputer stores v.getRow() verbatim), so dx compares term by term, and a
        // per-variable-column fingerprint of |J| localises any Jacobian difference to a variable.
        // Only F needs the row map, and its sorted multiset is permutation-invariant anyway.
        {
            static const char* dxPath = std::getenv("OLF_GPU_DX_DUMP");   // read once, not per iteration
            static bool dumped = false;
            // With OLF_GPU_LOOP_TRACE_CTG set, the dump is that contingency's slot in its own chunk
            // (the host sets loopTraceSlot to -1 for the chunks that do not hold it); slot 0 of the
            // first solve otherwise, as before.
            const int dumpSlot = ctx->loopTraceSlot >= 0 ? ctx->loopTraceSlot : (std::getenv("OLF_GPU_LOOP_TRACE_CTG") != nullptr ? -1 : 0);
            // OLF_GPU_DX_DUMP_OUTER=<k>: the first step of outer pass k instead of the first solve's
            static const int dxOuter = std::getenv("OLF_GPU_DX_DUMP_OUTER") != nullptr ? std::atoi(std::getenv("OLF_GPU_DX_DUMP_OUTER")) : -1;
            if (dxPath != nullptr && !dumped && it == 0 && dumpSlot >= 0 && (dxOuter < 0 || outer == dxOuter)) {
                dumped = true;
                dumpSlotSystem(ctx, n, nnz, dumpSlot, dxPath);
            }
        }
        // OLF_GPU_ROW_PROBE=<row>: that row in EVERY scenario at EVERY iteration - its residual,
        // its own state, and its non-zero Jacobian entries by column. The DX dump answers "what is
        // the shared base structure" and the state dump only fits a toy system; neither says what
        // one row of a 13000-row SCENARIO looks like at convergence, which is the question when a
        // bus's voltage has not moved by a single bit from the base while the CPU's has.
        {
            static const char* rowProbeEnv = std::getenv("OLF_GPU_ROW_PROBE");
            static int rowProbe = rowProbeEnv != nullptr ? std::atoi(rowProbeEnv) : -1;
            // With OLF_GPU_LOOP_TRACE_CTG set, only the traced contingency's slot in the chunk that holds it: every
            // chunk has a slot of that number, so printing them all mixed other contingencies' rows into the trace
            // (rte6515 GEN-676: slot 275 of every chunk read as GEN-676's row, base weights and all).
            const bool probeCtg = std::getenv("OLF_GPU_LOOP_TRACE_CTG") != nullptr;
            if (rowProbe >= 0 && rowProbe < n && (!probeCtg || ctx->loopTraceSlot >= 0)) {
                CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                std::vector<int> hrp(n + 1);
                CUDA_CHECK(cudaMemcpy(hrp.data(), ctx->st->dRowPtr.p, (size_t) (n + 1) * sizeof(int), cudaMemcpyDeviceToHost));
                int k0 = hrp[rowProbe];
                int k1 = hrp[rowProbe + 1];
                std::vector<int> hci(k1 - k0);
                CUDA_CHECK(cudaMemcpy(hci.data(), ctx->st->dColIdx.as<int>() + k0, (size_t) (k1 - k0) * sizeof(int), cudaMemcpyDeviceToHost));
                std::vector<int> hmode(nS);
                for (int s2 = probeCtg ? ctx->loopTraceSlot : 0; s2 < (probeCtg ? ctx->loopTraceSlot + 1 : nS); s2++) {
                    double fv = 0;
                    double xv = 0;
                    int mv = 0;
                    CUDA_CHECK(cudaMemcpy(&fv, ctx->dF.as<double>() + (size_t) s2 * n + rowProbe, sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(&xv, ctx->dX.as<double>() + (size_t) s2 * n + rowProbe, sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(&mv, ctx->dMode.as<int>() + (size_t) s2 * n + rowProbe, sizeof(int), cudaMemcpyDeviceToHost));
                    double tv = 0;
                    CUDA_CHECK(cudaMemcpy(&tv, ctx->dTarget.as<double>() + (size_t) s2 * n + rowProbe, sizeof(double), cudaMemcpyDeviceToHost));
                    std::vector<double> hv(k1 - k0);
                    CUDA_CHECK(cudaMemcpy(hv.data(), ctx->dVals.as<double>() + (size_t) s2 * nnz + k0, (size_t) (k1 - k0) * sizeof(double), cudaMemcpyDeviceToHost));
                    std::fprintf(stderr, "ROW_PROBE it=%d s=%d row=%d mode=%d F=%.12e x=%.12e target=%.12e nnz=%d :", it, s2, rowProbe, mv, fv, xv, tv, k1 - k0);
                    for (int k = 0; k < k1 - k0; k++) {
                        if (hv[k] != 0.0) {
                            std::fprintf(stderr, " [%d]=%.6e", hci[k], hv[k]);
                        }
                    }
                    std::fprintf(stderr, "\n");
                }
            }
        }
        static const bool linResidOn = std::getenv("OLF_GPU_LIN_RESID") != nullptr;
        if (linResidOn) {
            DeviceBuf rp;
            DeviceBuf rm;
            rp.alloc((size_t) n * nS * sizeof(double));
            rm.alloc((size_t) n * nS * sizeof(double));
            linResidRowBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, nnz, ctx->st->dRowPtr.as<int>(),
                    ctx->st->dColIdx.as<int>(), ctx->dVals.as<double>(), ctx->dDx.as<double>(),
                    ctx->dB.as<double>(), rp.as<double>(), rm.as<double>());   // B = the (repaired) RHS
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            std::vector<double> hrp((size_t) n * nS);
            std::vector<double> hrm((size_t) n * nS);
            std::vector<double> hf((size_t) n * nS);
            std::vector<double> hdx((size_t) n * nS);
            CUDA_CHECK(cudaMemcpy(hrp.data(), rp.p, (size_t) n * nS * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hrm.data(), rm.p, (size_t) n * nS * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hf.data(), ctx->dF.p, (size_t) n * nS * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hdx.data(), ctx->dDx.p, (size_t) n * nS * sizeof(double), cudaMemcpyDeviceToHost));
            for (int s = 0; s < nS; s++) {
                if (!active[s]) continue;
                double mp = 0;
                double mm = 0;
                double mf = 0;
                double mdx = 0;
                double ssq = 0;                              // ||F||2 too: OLF's default stopping test is on the 2-norm
                int argp = -1;
                int argm = -1;
                for (int i = 0; i < n; i++) {
                    size_t t = (size_t) s * n + i;
                    if (std::fabs(hrp[t]) > mp) { mp = std::fabs(hrp[t]); argp = i; }
                    if (std::fabs(hrm[t]) > mm) { mm = std::fabs(hrm[t]); argm = i; }
                    mf = std::max(mf, std::fabs(hf[t]));
                    ssq += hf[t] * hf[t];
                    mdx = std::max(mdx, std::fabs(hdx[t]));
                }
                double best = std::min(mp, mm);
                int argw = mp < mm ? argp : argm;
                std::fprintf(stderr, "LIN_RESID it=%d slot=%d n=%d ||J.dx-F||inf=%.6e ||F||inf=%.6e ||F||2=%.17e "
                             "||dx||inf=%.6e ratio=%.3e worstRow=%d\n",
                             it, s, n, best, mf, std::sqrt(ssq), mdx, mf > 0 ? best / mf : 0.0, argw);
            }
        }
        // LINEAR-SOLVE GUARD, first iteration of the first pass of every batched solve: a
        // factorization the UNPIVOTED uniform batch got wrong shows as a linear residual ratio of
        // 1e6..1e13 (rte6515's full busbar scope, 366 retained couplers: min pivots of 1e-18 under
        // any unpivoted ordering) while a sound one sits at 1e-9 (repaired rows, IR=2) and the hardest
        // measured case at 4e-4 (IR=0). Newton then never moves and the scenario surfaced as an
        // honest-looking MAX_ITERATION_REACHED; for the batch of ONE that is the BASE load flow of a
        // whole security analysis, which then reported NOTHING (6876 sections, GPU={}). Reject it
        // here (status -2): the base throws into the CPU fallback, a chunk scenario replays on the
        // CPU - never wrong, only un-accelerated. One SpMV per scenario, once per solve. Checked at
        // the SECOND iteration of the first pass: a first step from the flat start can be garbage
        // on a row that the first refactorization then heals (deviceRemoteSingletonMemberPin:
        // ratio 0.17, dx 2e13, converged to 1e-12 all the same), while a wrong factorization stays
        // at 1e6..1e13 on every iteration.
        // Pass 0 only: two n*nS kernels and a sync per pass cost 34 s of a 19 s rte6515 run (measured), and a
        // factorization that goes wrong LATER - after a reactive-limits pin rewrites its rows - is caught by the
        // no-progress reject at the end of that pass instead (see the it == maxIter branch).
        if (outer == 0 && it == 1 && !deviceFixedIter) {   // it=1: a flat-start first step may be garbage and heal (singleton-pin fixture); a wrong factorization never moves
            if (!ctx->dLinRp.p) {
                ctx->dLinRp.alloc((size_t) n * nS * sizeof(double));
                ctx->dLinRm.alloc((size_t) n * nS * sizeof(double));
                ctx->dLinSc.alloc((size_t) 3 * nS * sizeof(double));
            }
            DeviceBuf& rp = ctx->dLinRp;
            DeviceBuf& rm = ctx->dLinRm;
            DeviceBuf& sc = ctx->dLinSc;
            linResidRowBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, nnz, ctx->st->dRowPtr.as<int>(),
                    ctx->st->dColIdx.as<int>(), ctx->dVals.as<double>(), ctx->dDx.as<double>(),
                    ctx->dB.as<double>(), rp.as<double>(), rm.as<double>());
            linResidScoreBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(n, nS, rp.as<double>(), rm.as<double>(),
                    ctx->dF.as<double>(), sc.as<double>());
            std::vector<double> hsc((size_t) 3 * nS);
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            CUDA_CHECK(cudaMemcpy(hsc.data(), sc.p, (size_t) 3 * nS * sizeof(double), cudaMemcpyDeviceToHost));
            for (int s = 0; s < nS; s++) {
                if (!active[s]) continue;
                double ratio = hsc[3 * s + 2];
                if (!(ratio <= LIN_SOLVE_FAIL_RATIO)) {         // NaN counts as failed
                    if (std::getenv("OLF_GPU_REJECT_TRACE") != nullptr) {
                        std::fprintf(stderr, "GPU_REJECT slot=%d reason=linear-solve ratio=%.3e ||J.dx-B||inf=%.3e ||F||inf=%.3e\n",
                                     s, ratio, hsc[3 * s], hsc[3 * s + 1]);
                    }
                    // OLF_GPU_DX_DUMP_ON_REJECT=<dir>: the rejected slot's system as the guard scored it, for a reject
                    // that only shows in a full run (the rte6515 busbar flap: one -3 per run on a different section
                    // each time, the identical ratio 3.06e+06 every time, never reproduced alone). The dump happens
                    // AFTER the verdict, so it cannot perturb what it catches.
                    static const char* rejectDumpDir = std::getenv("OLF_GPU_DX_DUMP_ON_REJECT");
                    static int rejectDumps = 0;
                    if (rejectDumpDir != nullptr && rejectDumps < 8) {
                        std::string p = std::string(rejectDumpDir) + "/reject-" + std::to_string(rejectDumps++)
                                + "-slot" + std::to_string(s) + ".dump";
                        dumpSlotSystem(ctx, n, nnz, s, p.c_str());
                    }
                    traceReject(s, "linear-solve", outer, it);
                    res.status[s] = -3;                      // STATUS_LINEAR_SOLVE_FAILED: the one reject a pivoted retry can fix
                    res.iters[s] = it;
                    active[s] = 0;
                    live[s] = 0;
                    nActive--;
                }
            }
        }
        // OLF_GPU_ITER_LOG: every step's linear solve scored like the pass-0 guard above, on every pass
        std::vector<double> iterLogLin;
        if (iterLog && !deviceFixedIter) {
            if (!ctx->dLinRp.p) {
                ctx->dLinRp.alloc((size_t) n * nS * sizeof(double));
                ctx->dLinRm.alloc((size_t) n * nS * sizeof(double));
                ctx->dLinSc.alloc((size_t) 3 * nS * sizeof(double));
            }
            linResidRowBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, nnz, ctx->st->dRowPtr.as<int>(),
                    ctx->st->dColIdx.as<int>(), ctx->dVals.as<double>(), ctx->dDx.as<double>(),
                    ctx->dB.as<double>(), ctx->dLinRp.as<double>(), ctx->dLinRm.as<double>());
            linResidScoreBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(n, nS, ctx->dLinRp.as<double>(), ctx->dLinRm.as<double>(),
                    ctx->dF.as<double>(), ctx->dLinSc.as<double>());
            iterLogLin.resize((size_t) 3 * nS);
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            CUDA_CHECK(cudaMemcpy(iterLogLin.data(), ctx->dLinSc.p, (size_t) 3 * nS * sizeof(double), cudaMemcpyDeviceToHost));
        }
        if (ctx->clampStep) {
            maxVoltageChangeFactorBatch<<<nS, BLOCK, 0, ctx->stream>>>(n, ctx->dDx.as<double>(), ctx->dVarType.as<int>(), ctx->dActive.as<char>(), ctx->maxDv, ctx->maxDphi, ctx->dStepFactor.as<double>());
            traceStepFactor(ctx, nS, it, active);
            axpy1BatchScaled<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, n * nS, ctx->dDx.as<double>(), ctx->dStepFactor.as<double>(), ctx->dX.as<double>());
        } else {
            axpy1Batch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n * nS, ctx->dDx.as<double>(), ctx->dX.as<double>());
        }
        if (!iterLogLin.empty()) {
            std::vector<double> fac(nS, 1.0);
            if (ctx->clampStep) {
                CUDA_CHECK(cudaMemcpy(fac.data(), ctx->dStepFactor.p, nS * sizeof(double), cudaMemcpyDeviceToHost));
            }
            for (int s = 0; s < nS; s++) {
                if (!active[s]) continue;
                char line[256];
                std::snprintf(line, sizeof line, "LIN_LOG outer=%d s=%d it=%d ratio=%.6e resid=%.6e fmax=%.6e factor=%.9f width=%d",
                              outer, s, it, iterLogLin[3 * s + 2], iterLogLin[3 * s], iterLogLin[3 * s + 1], fac[s], iterLogWidth);
                iterLogLines.emplace_back(line);
                if (!(iterLogLin[3 * s + 2] <= 1e-4)) {
                    // WHERE a poor solve leaves its residual: the worst row under the sign convention the ratio
                    // used, its row mode, and whether it is a zero-diagonal hazard row (with this lane's donor)
                    std::vector<double> rp(n), rm(n);
                    std::vector<int> md(n);
                    CUDA_CHECK(cudaMemcpy(rp.data(), ctx->dLinRp.as<double>() + (size_t) s * n, n * sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(rm.data(), ctx->dLinRm.as<double>() + (size_t) s * n, n * sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(md.data(), ctx->dMode.as<int>() + (size_t) s * n, n * sizeof(int), cudaMemcpyDeviceToHost));
                    double mp = 0.0;
                    double mm = 0.0;
                    for (int i = 0; i < n; i++) {
                        mp = std::max(mp, std::fabs(rp[i]));
                        mm = std::max(mm, std::fabs(rm[i]));
                    }
                    const std::vector<double>& r = mp <= mm ? rp : rm;   // the sign convention the score took
                    std::vector<int> ord(n);
                    for (int i = 0; i < n; i++) ord[i] = i;
                    std::partial_sort(ord.begin(), ord.begin() + std::min(n, 4), ord.end(),
                                      [&](int a, int b) { return std::fabs(r[a]) > std::fabs(r[b]); });
                    const auto& hz = ctx->st->hHazRow;
                    std::vector<int> choice;
                    if (!hz.empty() && ctx->dHazChoice.p && ctx->hazCap >= nS) {
                        choice.resize(hz.size());
                        CUDA_CHECK(cudaMemcpy(choice.data(), ctx->dHazChoice.as<int>() + (size_t) s * hz.size(),
                                              hz.size() * sizeof(int), cudaMemcpyDeviceToHost));
                    }
                    std::string worst = "LIN_WORST outer=" + std::to_string(outer) + " s=" + std::to_string(s)
                            + " it=" + std::to_string(it);
                    for (int k = 0; k < std::min(n, 4); k++) {
                        int row = ord[k];
                        int hi = -1;
                        for (size_t h = 0; h < hz.size(); h++) {
                            if (hz[h] == row) hi = static_cast<int>(h);
                        }
                        char b[128];
                        std::snprintf(b, sizeof b, " [row=%d r=%.3e mode=%d haz=%d donor=%d]", row, r[row], md[row], hi,
                                      hi >= 0 && !choice.empty() ? choice[hi] : -9);
                        worst += b;
                    }
                    iterLogLines.emplace_back(worst);
                }
            }
        }
        if (!deviceFixedIter) CUDA_CHECK(cudaStreamSynchronize(ctx->stream));   // fixed-iter: no per-iter host sync (final readback syncs)
        } else {
        // ---- fallback: analyze once (shared pattern), refactorize + solve per scenario ----
        for (int s = 0; s < nS; s++) {
            if (!active[s]) continue;
            double* valsS = ctx->dVals.as<double>() + (size_t) s * nnz;
            double* bS = ctx->dB.as<double>() + (size_t) s * n;
            double* dxS = ctx->dDx.as<double>() + (size_t) s * n;
            double* fS = ctx->dF.as<double>() + (size_t) s * n;
            negateIntoOne<<<grid(n), BLOCK, 0, ctx->stream>>>(n, fS, bS);
            hazRepairRightHandSide(ctx, 1, bS, s);
            CUDSS_CHECK(cudssMatrixSetValues(ctx->ds.matA, valsS));
            CUDSS_CHECK(cudssMatrixSetValues(ctx->ds.matB, bS));
            CUDSS_CHECK(cudssMatrixSetValues(ctx->ds.matX, dxS));
            if (!ctx->analyzed) {
                gAnalysesMain++;
                { AnalysisTimer at_(__LINE__); CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_ANALYSIS, ctx->ds.config, ctx->ds.data, ctx->ds.matA, ctx->ds.matX, ctx->ds.matB)); }
                traceLuNnz(ctx->ds.handle, ctx->ds.data, "per-scenario", ctx->nScenarios);
                ctx->analyzed = true;
            }
            { AnalysisTimer at_(__LINE__, 'F'); CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_SOLVE,
                    ctx->ds.config, ctx->ds.data, ctx->ds.matA, ctx->ds.matX, ctx->ds.matB)); }
            double* xS = ctx->dX.as<double>() + (size_t) s * n;
            if (ctx->clampStep) {                            // OLF's MAX_VOLTAGE_CHANGE scaling, like every other mode:
                // unclamped, the robust mode took full Newton steps and its base load flow on network.xiidm's
                // busbar scope settled 0.32 kV away from OLF's
                maxVoltageChangeFactorBatch<<<1, BLOCK, 0, ctx->stream>>>(n, dxS, ctx->dVarType.as<int>(), nullptr,
                        ctx->maxDv, ctx->maxDphi, ctx->dStepFactor.as<double>() + s);
                axpy1BatchScaled<<<grid(n), BLOCK, 0, ctx->stream>>>(n, n, dxS, ctx->dStepFactor.as<double>() + s, xS);
            } else {
                axpy1One<<<grid(n), BLOCK, 0, ctx->stream>>>(n, dxS, xS);
            }
        }
        if (!deviceFixedIter) CUDA_CHECK(cudaStreamSynchronize(ctx->stream));   // fixed-iter: no per-iter host sync
        }
    }   // ---- end inner Newton loop ----
    if (olTime) {
        int nSolved = 0;
        for (int s = 0; s < nS; s++) nSolved += wasActive[s] ? 1 : 0;
        olPassSolved = nSolved;
    }
    for (int s = 0; s < nS; s++) {                           // per-scenario Newton work by pass (cheap, host)
        if (wasActive[s]) {
            if (ctx->olPasses[s] == 0) ctx->olFirst[s] = res.iters[s];
            ctx->olIters[s] += res.iters[s];
            ctx->olPasses[s]++;
        }
    }
    }   // ---- end if (solvePass) ----
    if (olTime) {
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        double ms = olNowMs() - olT0;
        long long ai = gActiveIterSum - olAI0, wi = gWidthIterSum - olWI0;
        if (olSolving && !olFirstSolveDone) {
            gOlFirstSolveMs += ms; gOlFirstSolveScen += olPassSolved; gOlFirstLaneIters += ai; gOlFirstWidthIters += wi;
            gOlFirstPasses++;
            olFirstSolveDone = true;
        } else if (olSolving) {
            int k = olTriggerKind >= 0 && olTriggerKind < OL_KINDS ? olTriggerKind : 0;
            gOlReSolveMs[k] += ms; gOlReSolveScen[k] += olPassSolved; gOlReSolveLaneIters[k] += ai;
            gOlReSolveWidthIters[k] += wi; gOlReSolvePasses[k]++;
            if (rlReasonsOn() && k == LOOP_RL) {
                std::fprintf(stderr, "GPU_RL_PASS outer=%d width=%d active=%d ms=%.1f batchIters=%lld laneIters=%lld widthIters=%lld\n",
                             outer, nS, olPassSolved, ms, gNewtonIters - olNI0, ai, wi);
            }
        }
    }
    // the CHECK part of the pass, charged to the loop being checked when the pass body ends (break/continue too)
    struct OlCheckTimer {
        bool on; double t0; int kind;
        ~OlCheckTimer() {
            if (on && kind >= 0 && kind < OL_KINDS) {
                gOlCheckMs[kind] += olNowMs() - t0;
                gOlCheckPasses[kind]++;
            }
        }
    } olCheck{olTime, olTime ? olNowMs() : 0.0, -1};

    // ---- AcloadFlowEngine's realistic-voltage check, on this INTERMEDIATE state ----
    // OLF re-checks after every solver run once the last loop that can fix an unrealistic state has
    // run, and an unrealistic state ends the outer-loop cycle with UNREALISTIC_STATE: the run FAILS
    // and reports no violations. Which solve the check belongs to depends on the configured loop
    // order and on robust mode, so rather than reproduce that ordering the device routes any
    // scenario that is EVER unrealistic to the CPU replay, which applies OLF's rule itself. That is
    // stricter than OLF (a state it would have let a later loop fix also leaves the batch), so it
    // can cost batching share and can never change an answer.
    // One check over the scenarios gate selects: any whose state leaves the band goes to the CPU replay.
    if (ctx->realisticCheck && solvePass && loopIdx > realisticFixingIdx) {
        std::vector<char> gate(nS);
        for (int s = 0; s < nS; s++) {
            gate[s] = (wasActive[s] && res.status[s] == 0 && live[s]) ? 1 : 0;
        }
        rejectUnrealistic(ctx, drv, gate);
    }

    if (!fullLoadflow) {
        break;                                              // fixed-iter / no controllers: one pass only
    }

    // ---- REACTIVE LIMITS pass: switch any local PV controller over its Q limit to PQ ----
    // Only scenarios (re)solved this pass that converged can have new violations.
    for (int s = 0; s < nS; s++) {
        rlActive[s] = (wasActive[s] && res.status[s] == 0) ? 1 : 0;
    }
    CUDA_CHECK(cudaMemcpy(ctx->dRlActive.p, rlActive.data(), nS * sizeof(char), cudaMemcpyHostToDevice));
    // generator Q per controller = (Σ branch q + shunt q) at its BUS_V row + load Q. Compute the first
    // part by a fill into a ZEROED buffer: no −target init, no applyRowModeBatch, so qBuf[row] is the raw
    // bus reactive balance. Disabled branches are skipped (outage); dVals is scratch (the next inner
    // iteration memsets and refills it).
    CUDA_CHECK(cudaMemsetAsync(ctx->dQbuf.p, 0, nTot * sizeof(double), ctx->stream));
    // Reactive-limit Q buffer: physical bus-Q for PV-limit detection — keep the FROZEN routing
    // (dynReloc=0), the relocation only concerns the Newton residual, not this diagnostic Q.
    // Nothing reads a Jacobian here, and the fused J+F fill paid for one anyway: 46 of the 98 closed-branch
    // fills of a 485-contingency network.xiidm chunk, at 38 ms each, were this call. This kernel runs the fused
    // fill's own residual expressions and drops only the 24 Jacobian scatters (an angle-addition variant once
    // moved transformer-control cases by ~3e-5 kV: the bus balance must be the Newton residual's arithmetic).
    if (ctx->st->cb.ne) fillClosedBranchQbufBatch<<<grid(ctx->st->cb.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->cb.ne, nS, n, ctx->nElements, ctx->cbOffset, ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->st->cb.dest.as<int>(), ctx->dX.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dRlActive.as<char>(), ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>());
    if (ctx->st->ob.ne) fillOpenBranchBatch<<<grid(ctx->st->ob.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->ob.ne, nS, n, nnz, ctx->nElements, ctx->obOffset, ctx->st->ob.in.as<double>(), ctx->st->ob.ix.as<int>(), ctx->st->ob.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dRlActive.as<char>(), nullptr, nullptr, nullptr, 0, RelocTables{});
    if (ctx->st->sh.ne) fillShuntBatch<<<grid(ctx->st->sh.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->sh.ne, nS, n, nnz, ctx->nElements, ctx->shOffset, ctx->st->sh.in.as<double>(), ctx->st->sh.ix.as<int>(), ctx->st->sh.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dRlActive.as<char>(), ctx->nShOv, ctx->dShOvSlotElem.as<int>(), ctx->dShOvGb.as<double>(), nullptr, nullptr, nullptr, 0, RelocTables{});
    if (ctx->st->nvdl) fillVdlBatch<<<grid(ctx->st->nvdl * nS), BLOCK, 0, ctx->stream>>>(ctx->st->nvdl, nS, n, nnz, ctx->st->dVdlIn.as<double>(), ctx->st->dVdlIdx.as<int>(), ctx->st->dVdlDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), nullptr, nullptr, nullptr);
    // Zero-impedance dummy transfers MUST be in the Q/P buffer too: the distributed-slack retain
    // reads the slack row's balance from dQbuf — without the coupler's dummy-P the DS sees a
    // phantom mismatch equal to the through-flow and distributes it FOREVER (measured: +392 MW
    // runaway on the first switch-close scenario until the per-scenario budget rejected it).
    if (ctx->nzi) fillZeroImpBatch<<<grid(ctx->nzi * nS), BLOCK, 0, ctx->stream>>>(ctx->nzi, nS, n, nnz, ctx->st->dZiIn.as<double>(), ctx->st->dZiIdx.as<int>(), ctx->st->dZiDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), ctx->dRlActive.as<char>(), ctx->nziBase, ctx->nZiOn, ctx->dZiOnSlotElem.as<int>(), ctx->nZiOff, ctx->dZiOffSlotElem.as<int>(), nullptr, nullptr, nullptr, 0, RelocTables{}, ctx->nZiTree, ctx->nZiTree ? ctx->dZiTreeSlotElemVal.as<int>() : nullptr, ctx->hasZiMasks ? ctx->dZiOffMask.as<char>() : nullptr, ctx->hasZiMasks ? ctx->dZiOnMask.as<char>() : nullptr, ctx->hasZiMasks ? ctx->dZiTreeMask.as<char>() : nullptr);
    // Remote-member OWN q (plan-B3): fill each member's own reactive injection into its (suppressed,
    // hence free) qBuf row so the RL kernel evaluates remote entries like local ones.
    if (ctx->nRmq > 0) {
        // OLF_GPU_RMQ_TRACE=<deviceRow>: every contribution this kernel adds to that member's
        // diagnostic q row, named by kind and element. The RL check pins on q = qBuf[row] + loadQ,
        // so a q that disagrees with OLF is answered here and nowhere else.
        static const char* rmqTraceEnv = std::getenv("OLF_GPU_RMQ_TRACE");
        int rmqTraceRow = rmqTraceEnv != nullptr ? std::atoi(rmqTraceEnv) : -1;
        fillRemoteMemberQBatch<<<grid(ctx->nRmq * nS), BLOCK, 0, ctx->stream>>>(ctx->nRmq, nS, n,
                ctx->nElements, ctx->cbOffset, ctx->obOffset,
                ctx->dRmqRow.as<int>(), ctx->dRmqElem.as<int>(), ctx->dRmqSide.as<int>(),
                ctx->dRmqKind.as<int>(), ctx->dRmqWeight.as<double>(),
                ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->st->sh.in.as<double>(), ctx->st->sh.ix.as<int>(),
                ctx->st->ob.in.as<double>(), ctx->st->ob.ix.as<int>(),
                ctx->dX.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dRlActive.as<char>(),
                ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(),
                ctx->nShOv, ctx->dShOvSlotElem.as<int>(), ctx->dShOvGb.as<double>(),
                rmqTraceRow);
    }
    if (ctx->nhv > 0 && ctx->dsSlackPhiRow >= 0) {           // a link on the slack bus, as OLF's slack mismatch sees it
        hvdcSlackBalanceBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(nS, ctx->nhv, n, ctx->nElements, ctx->dsSlackPhiRow,
                ctx->hasDsSlackRowOv ? ctx->dDsSlackRowOv.as<int>() : nullptr,
                ctx->dHvBus1Row.as<int>(), ctx->dHvBus2Row.as<int>(), ctx->dHvBus1Phi.as<int>(), ctx->dHvBus2Phi.as<int>(),
                ctx->dHvElem.as<int>(), ctx->dHvIn.as<double>(), ctx->dHvStatus.as<int>(),
                ctx->dHvAppliedD1.as<double>(), ctx->dHvAppliedD2.as<double>(), ctx->dHvCtgOut.as<char>(),
                ctx->dDisabled.as<char>(), ctx->dX.as<double>(), ctx->dRlActive.as<char>(), ctx->dQbuf.as<double>());
    }
    CUDA_CHECK(cudaMemsetAsync(ctx->dRlSwitched.p, 0, nS * sizeof(char), ctx->stream));
    if (outer == 0 && !resume && ctx->loopTraceSlot >= 0 && ctx->loopTraceSlot < nS) {
        // OLF_GPU_SLOT_DUMP, second point: the traced scenario's state and bus balances (qBuf) at the FIRST
        // outer-loop check, i.e. after the first solve - "<prefix>.<k>.pass0", "<row> <x> <qBuf>".
        static const char* slotDump0 = std::getenv("OLF_GPU_SLOT_DUMP");
        static int slotDumps0 = 0;
        if (slotDump0 != nullptr) {
            const size_t off = static_cast<size_t>(ctx->loopTraceSlot) * n;
            std::vector<double> hx(n), hq(n);
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            CUDA_CHECK(cudaMemcpy(hx.data(), ctx->dX.as<double>() + off, n * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hq.data(), ctx->dQbuf.as<double>() + off, n * sizeof(double), cudaMemcpyDeviceToHost));
            std::string path = std::string(slotDump0) + "." + std::to_string(slotDumps0++) + ".pass0";
            if (FILE* f = std::fopen(path.c_str(), "w")) {
                for (int r = 0; r < n; r++) {
                    std::fprintf(f, "%d %.17g %.17g\n", r, hx[r], hq[r]);
                }
                std::fclose(f);
            }
        }
    }
    // Apply ONLY the current phase's outer loop this pass (sequential, like OLF) — never both at once.
    bool inDsPhase = (loopOrder[loopIdx] == LOOP_DS);
    olCheck.kind = loopOrder[loopIdx];
    olTriggerKind = loopOrder[loopIdx];                     // a re-solve in the next pass was triggered by this check
    if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
        const char* kindName = "?";
        switch (loopOrder[loopIdx]) {
            case LOOP_DS:           kindName = "DistributedSlack"; break;
            case LOOP_HVDC_FREEZE:  kindName = "FreezingHvdcACEmulation"; break;
            case LOOP_HVDC_LIMITS:  kindName = "AcHvdcAcEmulationLimits"; break;
            case LOOP_HOST_SVC:     kindName = "HOST:SecondaryVoltageControl"; break;
            case LOOP_MONITORING:   kindName = "MonitoringVoltage"; break;
            case LOOP_RL:           kindName = "ReactiveLimits"; break;
            case LOOP_HOST_PHASE:   kindName = "HOST:PhaseControl"; break;
            case LOOP_HOST_TVC:     kindName = "HOST:TransformerVoltageControl"; break;
            case LOOP_HOST_TRPC:    kindName = "HOST:TransformerReactivePowerControl"; break;
            case LOOP_HOST_SHUNT:   kindName = "HOST:ShuntVoltageControl"; break;
            case LOOP_TVC_KERNEL:   kindName = "TransformerVoltageControl"; break;
            case LOOP_SHUNT_KERNEL: kindName = "ShuntVoltageControl"; break;
            case LOOP_PHASE_KERNEL: kindName = ctx->nCph > 0 ? "PhaseControl" : "IncrementalPhaseControl"; break;
            case LOOP_TRPC_KERNEL:  kindName = "TransformerReactivePowerControl"; break;
            case LOOP_TVD_KERNEL:   kindName = ctx->tvdMode == 1 ? "SimpleTransformerVoltageControl" : "TransformerVoltageControl"; break;
            case LOOP_SVD_KERNEL:   kindName = "ShuntVoltageControl"; break;
            case LOOP_SVC_KERNEL:   kindName = "SecondaryVoltageControl"; break;
        }
        int nSw = 0;
        {
            std::vector<char> sw(nS);
            CUDA_CHECK(cudaMemcpy(sw.data(), ctx->dRlSwitched.p, nS, cudaMemcpyDeviceToHost));
            for (char c : sw) nSw += c ? 1 : 0;
        }
        int nPinned = 0;
        if (ctx->nrl > 0) {
            std::vector<char> lt(static_cast<size_t>(ctx->nrl) * nS);
            CUDA_CHECK(cudaMemcpy(lt.data(), ctx->dRlLimitType.p, lt.size(), cudaMemcpyDeviceToHost));
            for (char c : lt) nPinned += c ? 1 : 0;
        }
        std::fprintf(stderr, "BATCH_TRACE pass done: outer=%d loop=%d/%d kind=%s solved=%d changedScenarios=%d iters=%d pinned=%d",
                     outer, loopIdx, nLoops, kindName, solvePass ? 1 : 0, nSw, gLastPassIters, nPinned);
        {
            // OLF_GPU_ROW_PROBE=<row>: that row's slot-0 TARGET as each pass leaves it — the inner
            // probe only sees iterations, and a target a loop phase mutates between solves (the
            // HVDC freeze/unfreeze, a DS anchor reset) is invisible there.
            static const char* passProbeEnv = std::getenv("OLF_GPU_ROW_PROBE");
            static int passProbe = passProbeEnv != nullptr ? std::atoi(passProbeEnv) : -1;
            if (passProbe >= 0 && passProbe < n) {
                double tv = 0;
                double fv = 0;
                CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                CUDA_CHECK(cudaMemcpy(&tv, ctx->dTarget.as<double>() + passProbe, sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&fv, ctx->dF.as<double>() + passProbe, sizeof(double), cudaMemcpyDeviceToHost));
                std::fprintf(stderr, " probeRow=%d target=%.12e F=%.12e", passProbe, tv, fv);
            }
        }
        std::fprintf(stderr, "\n");
    }
    if (inDsPhase && getenv("OLF_GPU_DS_DEBUG") && nS > 0) {
        // Is scenario 0 actually CONVERGED at the moment the DS loop reads its mismatch? OLF checks
        // its slack mismatch on a converged state; if ours is only part-way there, the mismatch it
        // measures is not the same quantity, and a threshold at the same scale flips.
        std::vector<double> f0(n);
        CUDA_CHECK(cudaMemcpy(f0.data(), ctx->dF.as<double>(), n * sizeof(double), cudaMemcpyDeviceToHost));
        double ssq = 0.0;
        double inf = 0.0;
        for (int i = 0; i < n; i++) {
            ssq += f0[i] * f0[i];
            inf = fmax(inf, fabs(f0[i]));
        }
        double slackM = std::numeric_limits<double>::quiet_NaN();
        double retain0 = std::numeric_limits<double>::quiet_NaN();
        if (ctx->dsSlackPhiRow >= 0) {
            std::vector<double> q0(n);
            CUDA_CHECK(cudaMemcpy(q0.data(), ctx->dQbuf.as<double>(), n * sizeof(double), cudaMemcpyDeviceToHost));
            slackM = q0[ctx->dsSlackPhiRow] - ctx->dsSlackTargetP;
            if (ctx->dDsRetain.p) CUDA_CHECK(cudaMemcpy(&retain0, ctx->dDsRetain.p, sizeof(double), cudaMemcpyDeviceToHost));
        }
        std::fprintf(stderr, "[DS-CONV] outer=%d innerNewtonIters=%d ||F||2=%.9e ||F||inf=%.9e slackMismatchFromAnchor=%.9f retain=%.9f rlActive=%d\n",
                     outer, gLastPassIters, std::sqrt(ssq), inf, slackM, retain0, (int) rlActive[0]);
    }
    int loopKind = loopOrder[loopIdx];
    if (loopKind >= LOOP_HOST_SVC && loopKind <= LOOP_HOST_SHUNT) {   // any LOOP_HOST_* kind: pause for the host
        // Hand the sequence back. Everything the device holds — x, targets, row modes, RL pins,
        // monitoring activations, absorbed slack, HVDC statuses — stays put in the context; only
        // these host-side locals need saving.
        ctx->pausedLoopIdx = loopIdx;
        ctx->pausedSeqChanged = seqChanged ? 1 : 0;
        ctx->pausedProdIters = prodIters;
        ctx->pausedLive = live;
        ctx->pausedActive = active;
        ctx->pausedStatus = res.status;
        ctx->driverPaused = 1;
        ctx->pausedLoopKind = loopKind;                      // WHICH host loop to run
        break;
    }
    if (loopKind == LOOP_TVC_KERNEL) {
        tvcStep(ctx, drv);
    } else if (loopKind == LOOP_SHUNT_KERNEL) {
        shcStep(ctx, drv);
    } else if (loopKind == LOOP_PHASE_KERNEL) {
        if (ctx->nCph > 0) cphStep(ctx, drv); else flcStep(ctx, drv, 6);       // continuous, or incremental kinds 1 + 2
    } else if (loopKind == LOOP_TRPC_KERNEL) {
        flcStep(ctx, drv, 1);                                          // kind 0 (RPC)
    } else if (loopKind == LOOP_TVD_KERNEL) {
        tvdStep(ctx, drv);
    } else if (loopKind == LOOP_SVD_KERNEL) {
        svdStep(ctx, drv);
    } else if (loopKind == LOOP_SVC_KERNEL) {
        svcStep(ctx, drv);
    } else if (loopKind == LOOP_MONITORING) {
        monitoringStep(ctx, drv);
    } else if (loopKind == LOOP_HVDC_FREEZE || loopKind == LOOP_HVDC_LIMITS) {
        hvdcStep(ctx, drv, loopKind);
    } else if (inDsPhase) {                                // distributed-slack outer loop: spread the slack mismatch
        dsCheck(ctx, drv);
    } else {                                               // reactive-limits outer loop: switch PV->PQ over Q limit
        if (loopKind == LOOP_RL && ctx->nrl > 0 && !skipRlPhase) {                  // skipRlPhase: a monitoring interleave solve — the
            if (useDynRlLimits) {                            // curve limits at the CURRENT (post-DS) targets,
                rlCurveLimitsBatch<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(ctx->nrl, nS, ctx->ncv, n, ctx->cvExtrapolate,
                        ctx->dCvIdx.as<int>(), ctx->dCvIn.as<double>(), ctx->dCvPts.as<double>(),
                        ctx->dCvPtr.as<int>(), ctx->dCvList.as<int>(), cvHvdcView(ctx),
                        ctx->dRlMinQ.as<double>(), ctx->dRlMaxQ.as<double>(), ctx->dRlLoadQ.as<double>(),
                        ctx->dTarget.as<double>(),
                        nRlQOv, nRlQOv > 0 ? dRlQOvSlotEntry.as<int>() : nullptr,
                        nRlQOv > 0 ? dRlQOvMinMax.as<double>() : nullptr,
                        nRlQOv > 0 ? dRlQOvPtr.as<int>() : nullptr, nRlQOv > 0 ? dRlQOvList.as<int>() : nullptr,
                        nCvOv > 0 ? dCvOvPtr.as<int>() : nullptr,
                        nCvOv > 0 ? dCvOvEntry.as<int>() : nullptr,
                        nCvOv > 0 ? dCvOvVal.as<double>() : nullptr,
                        ctx->ngen > 0 ? ctx->dGenCum.as<double>() : nullptr,
                        ctx->ngen > 0 ? ctx->dGenCumValid.as<char>() : nullptr, ctx->hasDgAnchor ? ctx->dDgAnchor.as<double>() : nullptr, ctx->hasDgStart ? ctx->dDgStart.as<double>() : nullptr, ctx->ngen,
                        ctx->nds > 0 ? ctx->dDsPhiRow.as<int>() : nullptr,
                        ctx->nDsRowOv > 0 ? ctx->dDsRowOv.as<int>() : nullptr, ctx->nds,
                        ctx->ngen > 0 ? ctx->dgBus.as<int>() : nullptr,
                        ctx->dRlMinQDyn.as<double>(), ctx->dRlMaxQDyn.as<double>(),
                        ctx->dRlLoadQDyn.as<double>(), cvProbeEntry, cvProbeSlotByCtg ? ctx->loopTraceSlot : cvProbeSlot);
                launchForcedTargetQLimits(ctx, nS, n);
                // Host-side readback of the probed entry's dynamic limits (device printf is not reliably
                // captured by the test runner): the limit the RL pass that follows will pin at.
                int cvHostSlot = cvProbeSlotByCtg ? ctx->loopTraceSlot : cvProbeSlot;
                static const std::vector<int> cvHostEntries = cvProbeEntries();
                if (cvHostSlot >= 0 && cvHostSlot < nS) {
                    for (int pe : cvHostEntries) {
                        if (pe < 0 || pe >= ctx->nrl) continue;
                        double lim[2];
                        size_t off = static_cast<size_t>(cvHostSlot) * ctx->nrl + pe;
                        CUDA_CHECK(cudaMemcpy(&lim[0], ctx->dRlMinQDyn.as<double>() + off, sizeof(double), cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaMemcpy(&lim[1], ctx->dRlMaxQDyn.as<double>() + off, sizeof(double), cudaMemcpyDeviceToHost));
                        fprintf(stderr, "[CV-HOST] outer=%d slot=%d entry=%d minQDyn=%.12f maxQDyn=%.12f\n",
                                outer, cvHostSlot, pe, lim[0], lim[1]);
                    }
                }
            }
            CUDA_CHECK(cudaMemsetAsync(ctx->dRlPassSwitched.p, 0, static_cast<size_t>(ctx->nrl) * nS, ctx->stream));
            reactiveLimitsBatch<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(
                    ctx->nSvc > 0 ? ctx->dSvcRlVt.as<double>() : nullptr,
                    ctx->nrl, nS, n, REACTIVE_LIMIT_TOL, RL_MAX_SWITCH,
                    ctx->dRlVrow.as<int>(),
                    ctx->dRlQrow.p ? ctx->dRlQrow.as<int>() : nullptr,
                    useDynRlLimits ? ctx->dRlMinQDyn.as<double>() : ctx->dRlMinQ.as<double>(),
                    useDynRlLimits ? ctx->dRlMaxQDyn.as<double>() : ctx->dRlMaxQ.as<double>(),
                    useDynRlLimits ? ctx->dRlLoadQDyn.as<double>() : ctx->dRlLoadQ.as<double>(),
                    ctx->dRlVtarget.as<double>(), ctx->dQbuf.as<double>(), ctx->dX.as<double>(), ctx->dRlActive.as<char>(),
                    ctx->dMode.as<int>(), ctx->dTarget.as<double>(),
                    ctx->dRlLimitType.as<char>(), ctx->dRlSwitchCount.as<char>(), ctx->dRlSwitched.as<char>(),
                    ctx->dRlPassSwitched.as<char>(), ctx->dRlSavedMode.as<char>(),
                    nRlVtOv, nRlVtOv > 0 ? dRlVtOvSlotIdx.as<int>() : nullptr,
                    nRlVtOv > 0 ? dRlVtOvVal.as<double>() : nullptr,
                    ctx->dRowSlope.as<double>(), ctx->dRlSavedTarget.as<double>(), useDynRlLimits ? 1 : 0,
                    ctx->nRlRemote > 0 ? ctx->dRlRemote.as<char>() : nullptr,
                    ctx->nRlRemote > 0 ? ctx->dRlCmpVrow.as<int>() : nullptr,
                    ctx->dRlSvcBmin.as<double>(), ctx->dRlSvcBmax.as<double>(), ctx->dRlV2Base.as<double>(),
                    svcCoefDyn ? ctx->dRlSvcBminDyn.as<double>() : nullptr, svcCoefDyn ? ctx->dRlSvcBmaxDyn.as<double>() : nullptr,
                    ctx->nRlIslanded > 0 ? ctx->dRlIslanded.as<char>() : nullptr,
                    ctx->rlRobust ? ctx->dRlInitTgtQ.as<double>() : nullptr,
                    ctx->rlVLoRobust, ctx->rlVHiRobust, ctx->rlRobust,
                    swLogSlot, ctx->dSwLog.p ? ctx->dSwLog.as<double>() : nullptr,
                    ctx->dSwLogCnt.p ? ctx->dSwLogCnt.as<int>() : nullptr, outer,
                    psEntryPtr(ctx), ctx->st->nQr,
                    ctx->nRlLocal > 0 ? ctx->dRlLocalOv.as<char>() : nullptr,
                    ctx->nRlLocal > 0 ? ctx->dRlLocalVt.as<double>() : nullptr,
                    ctx->nRlNoUnpin > 0 ? ctx->dRlNoUnpin.as<char>() : nullptr, ctx->nRlNoUnpin > 0 ? ctx->dRlUnpinBad.as<char>() : nullptr,
                    ctx->dRlMonitored.p ? ctx->dRlMonitored.as<char>() : nullptr,
                    ctx->nRlCmp > 0 ? ctx->dRlCmpOv.as<int>() : nullptr);
psgStep(ctx, nS, n, outer, "afterRL");
            if (ctx->nRlPilotSelf > 0) {
                // a pilot that is its own controller pinned while a member of its group is not (or
                // the reverse): the device has no row for OLF's layout of that state - CPU replay
                if (!ctx->dRlMixedBad.p) ctx->dRlMixedBad.alloc(nS * sizeof(char));
                CUDA_CHECK(cudaMemsetAsync(ctx->dRlMixedBad.p, 0, nS * sizeof(char), ctx->stream));
                rlPilotSelfMixedGate<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(ctx->nrl, nS,
                        ctx->dRlRemote.as<char>(), ctx->dRlCmpVrow.as<int>(), ctx->dRlLimitType.as<char>(),
                        ctx->nRlIslanded > 0 ? ctx->dRlIslanded.as<char>() : nullptr,
                        ctx->dRlMixedBad.as<char>(),
                        psEntryPtr(ctx), ctx->st->nQr,
                    ctx->nDqMembers, ctx->dDqMemberRlEntry.p != nullptr ? ctx->dDqMemberRlEntry.as<int>() : nullptr, ctx->hasDqSub ? ctx->dDqSub.as<char>() : nullptr);
                std::vector<char> mixed(nS);
                CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                CUDA_CHECK(cudaMemcpy(mixed.data(), ctx->dRlMixedBad.p, nS * sizeof(char), cudaMemcpyDeviceToHost));
                for (int s = 0; s < nS; s++) {
                    if (mixed[s] && live[s]) {
                        traceReject(s, "pilot-self-mixed-pin", outer, 0);
                        res.status[s] = -2;
                        active[s] = 0;
                        live[s] = 0;
                    }
                }
            }
            if (ctx->nRlNoUnpin > 0) {
                std::vector<char> ub(nS);
                CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                CUDA_CHECK(cudaMemcpy(ub.data(), ctx->dRlUnpinBad.p, nS * sizeof(char), cudaMemcpyDeviceToHost));
                for (int s = 0; s < nS; s++) {
                    if (ub[s] && live[s]) {
                        traceReject(s, "unpin-unrepresentable", outer, 0);
                        res.status[s] = -2;
                        active[s] = 0;
                        live[s] = 0;
                    }
                }
            }
psgStep(ctx, nS, n, outer, "afterMixedGate");
            if (ctx->rlRobust && ctx->nRlRemote > 0) {       // "set V to a safe 1 pu", as its own pass
                rlRobustVResetBatch<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(
                        ctx->nrl, nS, n, ctx->dRlVrow.as<int>(), ctx->dRlRemote.as<char>(),
                        ctx->dRlActive.as<char>(), ctx->dRlPassSwitched.as<char>(), ctx->dX.as<double>(),
                        ctx->rlVLoRobust, ctx->rlVHiRobust);
            }
psgStep(ctx, nS, n, outer, "afterRobust");
            if (ctx->hasRlElection && ctx->allPvPinnable) {  // the rule can only fire when every PV is pinnable
                rlForcedPvElectBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(ctx->nrl, nS, n,
                        ctx->dRlVrow.as<int>(),
                        ctx->nRlRemote > 0 ? ctx->dRlRemote.as<char>() : nullptr,
                        ctx->nRlIslanded > 0 ? ctx->dRlIslanded.as<char>() : nullptr,
                        ctx->dRlElectNomV.as<double>(), ctx->dRlElectPRow.as<int>(),
                        ctx->dRlElectStaticP.as<double>(), ctx->dRlElectIdRank.as<int>(),
                        ctx->dRlLimitType.as<char>(), ctx->dRlSwitchCount.as<char>(), ctx->dRlPassSwitched.as<char>(),
                        ctx->dRlSavedMode.as<char>(), ctx->dRlSavedTarget.as<double>(),
                        ctx->dMode.as<int>(), ctx->dTarget.as<double>(), ctx->dRlActive.as<char>(), ctx->dRlSwitched.as<char>());
            }
psgStep(ctx, nS, n, outer, "afterElect");
            // The pilot's V-identity flip/restore runs BEFORE the DISTR_Q targets: the carrier hand-off picks the
            // carrier with relocChoose, which reads the pilot's own row (candidate 1 of a pilot-self group).
            // Measured with the order reversed (rte6515 busbar base, group VL-5997): the member that just
            // switched back was handed a target of 0 while the pilot's row was still flipped, its row then
            // hosted the pilot's balance against that 0, and it re-pinned every other pass until maxSwitch.
            // A pilot that LOST its own generator (GenOutage.flipsPilotRowWhenAllPinned) is laid out here, BEFORE
            // the pilot-self flip below: that kernel's "control lost" layout puts the pilot's row in mode 1, which
            // for a pilot-that-is-its-own-controller is an EMPTY row (no q term is routed to it - the RL kernel's
            // own note), so the voltage stayed exactly at the identity's value while OLF solved the pilot's
            // BUS_TARGET_Q = 0 there. Measured on rte6515 GEN-1522: VL-1522_0 at 1.122 (the target) against OLF's
            // 1.0811, 3.2 kV at VL-1525 and 1.98e-2 MW of distributed power. Mode 4 - the own-Q balance the
            // member ALTERNATE dest routes the pilot's q to - is what the outage composer armed; run first, the
            // pilot-self kernel then finds the row already off its identity and leaves it alone.
            if (ctx->nPilotFlip > 0) {                      // pilot stopped controlling AND every survivor pinned
                pilotRowAllPinnedBatch<<<grid(ctx->nPilotFlip), BLOCK, 0, ctx->stream>>>(
                        ctx->nPilotFlip, nS, n, ctx->nrl,
                        ctx->dPfSlot.as<int>(), ctx->dPfRow.as<int>(), ctx->dPfQTarget.as<double>(),
                        ctx->dPfOff.as<int>(), ctx->dPfCount.as<int>(), ctx->dPfRlPool.as<int>(),
                        ctx->dRlLimitType.as<char>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>(),
                        ctx->dPfFlipped.as<char>(), ctx->dPfSavedVT.as<double>(),
                        ctx->dRlActive.as<char>(), ctx->dRlSwitched.as<char>(),
                        ctx->hasRlBaseLt ? ctx->dRlBaseLimitType.as<char>() : nullptr);
            }
psgStep(ctx, nS, n, outer, "afterAllPinned");
            if (ctx->hasDqQrGroups && ctx->st->nQr > 0) {    // all-pinned corner: pilot vEq flip/restore
                launchPilotVeqInner(ctx, drv);
            }
psgStep(ctx, nS, n, outer, "afterPilotVeq");
            if (ctx->nRlRemote > 0 && ctx->nDqMembers > 0) {
                // keep the survivors' DISTR_Q row targets in step with the pin states (plan-B3)
                launchDistrQTargetsInner(ctx, drv);
            }
psgStep(ctx, nS, n, outer, "afterDistrQ");
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(cudaMemcpy(rlSwitched.data(), ctx->dRlSwitched.p, nS * sizeof(char), cudaMemcpyDeviceToHost));
    if (rlReasonsOn() && loopOrder[loopIdx] == LOOP_RL && ctx->nrl > 0 && ctx->dRlPassSwitched.p) {
        std::vector<char> ps(static_cast<size_t>(ctx->nrl) * nS);
        CUDA_CHECK(cudaMemcpy(ps.data(), ctx->dRlPassSwitched.p, ps.size(), cudaMemcpyDeviceToHost));
        for (int s = 0; s < nS; s++) {
            if (!rlActive[s]) continue;
            int mask = 0;
            for (int i = 0; i < ctx->nrl; i++) {
                char c = ps[static_cast<size_t>(s) * ctx->nrl + i];
                if (c >= 1 && c <= 3) {
                    mask |= 1 << (c - 1);
                    gRlEntryCodes[static_cast<int>(c)]++;
                }
            }
            if (rlSwitched[s]) {
                gRlCheckScen[mask]++;
            } else {
                gRlStableScen++;
            }
        }
    }
    for (int s = 0; s < nS; s++) {
        if (loopRejected[s]) {                              // a device loop routed the scenario to the CPU fallback
            traceReject(s, "device-loop-routed", outer, 0);
            res.status[s] = -2;
            live[s] = 0;
            active[s] = 0;
            rlSwitched[s] = 0;
            loopRejected[s] = 0;
        }
    }
    // OLF_GPU_RL_TRACE_VROW=<v row>: one LOCAL entry's pin state at every check (the trace below
    // otherwise names remote entries only) - with its live limits and row target, which is what a
    // curve-limit drift moves.
    static const char* rlTraceVrowEnv = std::getenv("OLF_GPU_RL_TRACE_VROW");
    static const int rlTraceVrow = rlTraceVrowEnv != nullptr ? std::atoi(rlTraceVrowEnv) : -1;
    // OLF_GPU_SW_TRACE=<slot>: every controller this RL pass CHANGED for one scenario, with the
    // quantities its decision turned on. A scenario that ends one outer pass short of the CPU's
    // trajectory did so because one of these decisions went the other way; without the list there is
    // no way to name WHICH controller, and the margin of that one decision is the whole question.
    static const char* swTraceEnv = std::getenv("OLF_GPU_SW_TRACE");
    static const int swTraceSlotEnv = swTraceEnv != nullptr ? std::atoi(swTraceEnv) : -1;
    // the contingency-id slot of OLF_GPU_LOOP_TRACE_CTG wins (chunk composition moves fixed slots)
    const int swTraceSlot = ctx->loopTraceSlot >= 0 ? ctx->loopTraceSlot : swTraceSlotEnv;
    if (loopOrder[loopIdx] == LOOP_RL && (rlAuditOn || (swTraceSlot >= 0 && swTraceSlot < nS)) && ctx->nrl > 0) {
        std::vector<char> ps(static_cast<size_t>(ctx->nrl) * nS), lt2(static_cast<size_t>(ctx->nrl) * nS);
        std::vector<int> vr2(ctx->nrl);
        std::vector<double> qb2(static_cast<size_t>(n) * nS), lq2(ctx->nrl), x2(static_cast<size_t>(n) * nS);
        CUDA_CHECK(cudaMemcpy(ps.data(), ctx->dRlPassSwitched.p, ps.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lt2.data(), ctx->dRlLimitType.p, lt2.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vr2.data(), ctx->dRlVrow.p, vr2.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(qb2.data(), ctx->dQbuf.p, qb2.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lq2.data(), ctx->dRlLoadQ.p, lq2.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(x2.data(), ctx->dX.p, x2.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<double> mn2(static_cast<size_t>(ctx->nrl) * (ctx->ncv > 0 ? nS : 1));
        std::vector<double> mx2(mn2.size());
        CUDA_CHECK(cudaMemcpy(mn2.data(), ctx->ncv > 0 ? ctx->dRlMinQDyn.p : ctx->dRlMinQ.p, mn2.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mx2.data(), ctx->ncv > 0 ? ctx->dRlMaxQDyn.p : ctx->dRlMaxQ.p, mx2.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<int> cv2(ctx->nrl, -1);
        std::vector<double> vt2(ctx->nrl, 0.0);
        if (ctx->dRlCmpVrow.p) {
            CUDA_CHECK(cudaMemcpy(cv2.data(), ctx->dRlCmpVrow.p, cv2.size() * sizeof(int), cudaMemcpyDeviceToHost));
        }
        if (ctx->dRlVtarget.p) {
            CUDA_CHECK(cudaMemcpy(vt2.data(), ctx->dRlVtarget.p, vt2.size() * sizeof(double), cudaMemcpyDeviceToHost));
        }
        for (int s2 = 0; s2 < nS; s2++) {
            bool toStderr = (s2 == swTraceSlot);
            if (!toStderr && !(rlAuditOn && rlActive[s2])) continue;
            int nsw = 0;
            std::string rec;
            for (int i2 = 0; i2 < ctx->nrl; i2++) {
                size_t st2 = (size_t) s2 * ctx->nrl + i2;
                // OLF_GPU_SW_TRACE_ENTRIES=<e,...>: those entries print EVERY RL pass for the traced slot, switched
                // or not - a decision the device did NOT take is the one to compare against the CPU's.
                static const std::vector<int> swEntries = [] {
                    std::vector<int> v;
                    const char* e = std::getenv("OLF_GPU_SW_TRACE_ENTRIES");
                    if (e == nullptr) return v;
                    std::string str(e);
                    size_t pos = 0;
                    while (pos <= str.size()) {
                        size_t c = str.find(',', pos);
                        if (c == std::string::npos) c = str.size();
                        if (c > pos) v.push_back(std::atoi(str.substr(pos, c - pos).c_str()));
                        pos = c + 1;
                    }
                    return v;
                }();
                static const bool swAll = std::getenv("OLF_GPU_SW_TRACE_ENTRIES") != nullptr
                                          && std::string(std::getenv("OLF_GPU_SW_TRACE_ENTRIES")) == "all";
                bool forced = toStderr && (swAll || std::find(swEntries.begin(), swEntries.end(), i2) != swEntries.end());
                if (!ps[st2] && !forced) continue;
                size_t li2 = ctx->ncv > 0 ? st2 : (size_t) i2;
                double cmpV2 = (cv2[i2] >= 0) ? x2[(size_t) s2 * n + cv2[i2]] : 0.0;
                double qq = qb2[(size_t) s2 * n + vr2[i2]] + lq2[i2];
                // the q the decision READ: at rlQrow when the entry's balance was relocated (a coupler-tree
                // child reads its parent's DUMMY_Q row), not at its V row - printing the V-row value logged a
                // relocated child as q=0 while the kernel pinned it on its real q
                static const std::vector<int> qr2 = [&] {
                    std::vector<int> v(ctx->nrl, -1);
                    if (ctx->dRlQrow.p) CUDA_CHECK(cudaMemcpy(v.data(), ctx->dRlQrow.p, v.size() * sizeof(int), cudaMemcpyDeviceToHost));
                    return v;
                }();
                int qrow2 = qr2[i2] >= 0 ? qr2[i2] : vr2[i2];
                double qRead = qb2[(size_t) s2 * n + qrow2] + lq2[i2];
                nsw++;
                char buf[400];
                std::snprintf(buf, sizeof buf, "SW_PASS s=%d outer=%d entry=%d vrow=%d act=%d lt=%d q=%.15e qRow=%d qRead=%.15e min=%.15e max=%.15e v=%.15e cmpV=%.15e vt=%.15e margin=%.3e",
                             s2, outer, i2, vr2[i2], (int) ps[st2], (int) lt2[st2],
                             qq, qrow2, qRead, mn2[li2], mx2[li2],
                             x2[(size_t) s2 * n + vr2[i2]], cmpV2, vt2[i2], cmpV2 - vt2[i2]);
                if (toStderr) {
                    // the switch-back's other gates, read back for the traced entry: the row slope (a slope-SVC
                    // row never switches back) and the switch count against the cap
                    double slope = 0.0;
                    CUDA_CHECK(cudaMemcpy(&slope, ctx->dRowSlope.as<double>() + vr2[i2], sizeof(double), cudaMemcpyDeviceToHost));
                    char sc = 0;
                    CUDA_CHECK(cudaMemcpy(&sc, ctx->dRlSwitchCount.as<char>() + st2, 1, cudaMemcpyDeviceToHost));
                    char lov = 0;
                    double lvt = 0.0;
                    if (ctx->nRlLocal > 0) {
                        CUDA_CHECK(cudaMemcpy(&lov, ctx->dRlLocalOv.as<char>() + st2, 1, cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaMemcpy(&lvt, ctx->dRlLocalVt.as<double>() + st2, sizeof(double), cudaMemcpyDeviceToHost));
                    }
                    char mon = 0;
                    CUDA_CHECK(cudaMemcpy(&mon, ctx->dRlMonitored.as<char>() + i2, 1, cudaMemcpyDeviceToHost));
                    char isl = 0;
                    if (ctx->nRlIslanded > 0) {
                        CUDA_CHECK(cudaMemcpy(&isl, ctx->dRlIslanded.as<char>() + st2, 1, cudaMemcpyDeviceToHost));
                    }
                    char act = 0;
                    CUDA_CHECK(cudaMemcpy(&act, ctx->dRlActive.as<char>() + s2, 1, cudaMemcpyDeviceToHost));
                    std::fprintf(stderr, "%s slope=%.6e swCount=%d localOv=%d localVt=%.12f monitored=%d islanded=%d rlActive=%d\n", buf, slope, (int) sc,
                                 (int) lov, lvt, (int) mon, (int) isl, (int) act);
                }
                if (rlAuditOn && nsw <= 12) {
                    rec += buf;
                    rec += "\n";
                }
            }
            if (rlAuditOn) {
                char hdr[160];
                std::snprintf(hdr, sizeof hdr, "AUDIT_PASS s=%d outer=%d loop=%d rlActive=%d nsw=%d",
                              s2, outer, loopOrder[loopIdx], (int) rlActive[s2], nsw);
                auditLog[s2].push_back(std::string(hdr) + "\n" + rec);
            }
        }
    }
    if (!inDsPhase && (ctx->nRlRemote > 0 || nS == 1 || rlTraceVrow >= 0) && ctx->nrl > 0
            && (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr || rlTraceVrow >= 0)) {   // RL trace
        std::vector<char> lt(static_cast<size_t>(ctx->nrl) * nS);
        std::vector<char> rem(ctx->nrl);
        std::vector<int> vr(ctx->nrl);
        std::vector<double> qb(static_cast<size_t>(n) * nS);
        std::vector<double> lq(ctx->nrl);
        CUDA_CHECK(cudaMemcpy(lt.data(), ctx->dRlLimitType.p, lt.size(), cudaMemcpyDeviceToHost));
        if (ctx->nRlRemote > 0) {
        CUDA_CHECK(cudaMemcpy(rem.data(), ctx->dRlRemote.p, rem.size(), cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaMemcpy(vr.data(), ctx->dRlVrow.p, vr.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(qb.data(), ctx->dQbuf.p, qb.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lq.data(), ctx->dRlLoadQ.p, lq.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<int> hMode(static_cast<size_t>(n) * nS);
        CUDA_CHECK(cudaMemcpy(hMode.data(), ctx->dMode.p, hMode.size() * sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<double> hMaxQ(static_cast<size_t>(ctx->nrl) * (ctx->ncv > 0 ? nS : 1));
        CUDA_CHECK(cudaMemcpy(hMaxQ.data(), ctx->ncv > 0 ? ctx->dRlMaxQDyn.p : ctx->dRlMaxQ.p, hMaxQ.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<char> hRlAct(nS);
        CUDA_CHECK(cudaMemcpy(hRlAct.data(), ctx->dRlActive.p, nS, cudaMemcpyDeviceToHost));
        // The STATE the pin decision is taken at. Without it, a q that disagrees with OLF cannot be
        // told apart from a q evaluated at a different (or not yet converged) operating point - the
        // same quantity at a different V is not evidence of a wrong formula.
        std::vector<double> hXrl(static_cast<size_t>(n) * nS);
        CUDA_CHECK(cudaMemcpy(hXrl.data(), ctx->dX.p, hXrl.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<double> hMinQ(hMaxQ.size());
        CUDA_CHECK(cudaMemcpy(hMinQ.data(), ctx->ncv > 0 ? ctx->dRlMinQDyn.p : ctx->dRlMinQ.p, hMinQ.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<double> hTgt(static_cast<size_t>(n) * nS);
        CUDA_CHECK(cudaMemcpy(hTgt.data(), ctx->dTarget.p, hTgt.size() * sizeof(double), cudaMemcpyDeviceToHost));
        // The switch-back COMPARISON itself: the controlled (pilot) bus's voltage against the group
        // target, and the MARGIN between them. A pin that flips between runs is only "the last bits"
        // if that margin is at the noise floor - print it rather than assume it.
        std::vector<int> cmpVr(ctx->nrl);
        std::vector<double> vTgt(ctx->nrl);
        if (ctx->dRlCmpVrow.p) {
            CUDA_CHECK(cudaMemcpy(cmpVr.data(), ctx->dRlCmpVrow.p, cmpVr.size() * sizeof(int), cudaMemcpyDeviceToHost));
        }
        if (ctx->dRlVtarget.p) {
            CUDA_CHECK(cudaMemcpy(vTgt.data(), ctx->dRlVtarget.p, vTgt.size() * sizeof(double), cudaMemcpyDeviceToHost));
        }
        for (int s2 = 0; s2 < nS; s2++) {
            for (int i2 = 0; i2 < ctx->nrl; i2++) {
                if (rlTraceVrow >= 0 && vr[i2] == rlTraceVrow) {
                    double cmpV = (ctx->dRlCmpVrow.p && cmpVr[i2] >= 0) ? hXrl[(size_t) s2 * n + cmpVr[i2]] : 0.0;
                    std::fprintf(stderr, "RL_VROW s=%d entry=%d lt=%d q=%.9f minQ=%.9f maxQ=%.9f target=%.9f mode=%d v=%.9f loadQ=%.9f cmpV=%.12f vt=%.12f margin=%.3e\n",
                                 s2, i2, (int) lt[(size_t) s2 * ctx->nrl + i2], qb[(size_t) s2 * n + vr[i2]] + lq[i2],
                                 hMinQ[ctx->ncv > 0 ? (size_t) s2 * ctx->nrl + i2 : (size_t) i2],
                                 hMaxQ[ctx->ncv > 0 ? (size_t) s2 * ctx->nrl + i2 : (size_t) i2],
                                 hTgt[(size_t) s2 * n + vr[i2]], hMode[(size_t) s2 * n + vr[i2]],
                                 hXrl[(size_t) s2 * n + vr[i2]], lq[i2], cmpV, vTgt[i2], cmpV - vTgt[i2]);
                }
                if ((rem[i2] || nS == 1) && std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
                    std::fprintf(stderr, "RL_TRACE s=%d entry=%d remote=%d lt=%d switched=%d q=%.9f mode=%d tol=%.3e maxQ=%.6f ncv=%d rlActive=%d v=%.9f qbuf=%.9f\n",
                                 s2, i2, (int) rem[i2], (int) lt[(size_t) s2 * ctx->nrl + i2], (int) rlSwitched[s2],
                                 qb[(size_t) s2 * n + vr[i2]] + lq[i2], hMode[(size_t) s2 * n + vr[i2]], reactiveLimitTol,
                                 hMaxQ[ctx->ncv > 0 ? (size_t) s2 * ctx->nrl + i2 : (size_t) i2], ctx->ncv, (int) hRlAct[s2],
                                 hXrl[(size_t) s2 * n + vr[i2]], qb[(size_t) s2 * n + vr[i2]]);
                }
            }
        }
        // PSG: every pilot-self group's layout this pass — the pilot's pin state, its own row's mode/target,
        // whether updatePilotVeqBatch flipped it (veqOff), and every candidate row with its mode, so a
        // "control lost" verdict can be checked against the members it was taken over.
        if (ctx->hasPilotSelf && ctx->dPsPilotEntry.p && ctx->st->nQr > 0 && !ctx->st->hQrPtr.empty()
                && std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
            std::vector<int> pe(ctx->st->nQr, -1);
            std::vector<char> veq(static_cast<size_t>(ctx->st->nQr) * nS, 0);
            CUDA_CHECK(cudaMemcpy(pe.data(), ctx->dPsPilotEntry.p, pe.size() * sizeof(int), cudaMemcpyDeviceToHost));
            if (ctx->dQrVeqOff.p) {
                CUDA_CHECK(cudaMemcpy(veq.data(), ctx->dQrVeqOff.p, veq.size(), cudaMemcpyDeviceToHost));
            }
            for (int s2 = 0; s2 < nS; s2++) {
                for (int g = 0; g < ctx->st->nQr; g++) {
                    int e = pe[g];
                    if (e < 0) continue;
                    int p0 = ctx->st->hQrPtr[g];
                    int cnt = ctx->st->hQrPtr[g + 1] - p0;
                    int prow = ctx->st->hQrRows[p0];
                    std::string cands;
                    int pick = 0;
                    for (int c = 1; c < cnt; c++) {
                        int r = ctx->st->hQrRows[p0 + c];
                        int m = hMode[(size_t) s2 * n + r];
                        if (pick == 0 && m != 0 && m != 4) pick = c;
                        char b[48];
                        std::snprintf(b, sizeof b, " %d:m%d/t%.4f", r, m, hTgt[(size_t) s2 * n + r]);
                        cands += b;
                    }
                    std::fprintf(stderr, "PSG s=%d outer=%d g=%d entry=%d lt=%d pilotRow=%d mode=%d target=%.9f veqOff=%d q=%.9f pick=%d cands=[%s ]\n",
                                 s2, outer, g, e, (int) lt[(size_t) s2 * ctx->nrl + e], prow, hMode[(size_t) s2 * n + prow],
                                 hTgt[(size_t) s2 * n + prow], (int) veq[(size_t) s2 * ctx->st->nQr + g],
                                 qb[(size_t) s2 * n + vr[e]] + lq[e], pick, cands.c_str());
                }
            }
        }
    }
    if (inDsPhase && (ctx->nds > 0 || ctx->dsSlackOnly) && getenv("OLF_GPU_DS_DEBUG")) {
        // EVERY scenario, not just 0: the question "did OLF's DS fire on this check and did ours?"
        // is per scenario, and the scenario that disagrees is rarely slot 0.
        std::vector<double> qbufH(static_cast<size_t>(n) * nS);
        std::vector<double> retainH(nS, 0.0);
        std::vector<double> absH(nS, 0.0);
        CUDA_CHECK(cudaMemcpy(qbufH.data(), ctx->dQbuf.p, sizeof(double) * qbufH.size(), cudaMemcpyDeviceToHost));
        if (ctx->nds > 0 && ctx->dDsRetain.p) {
            CUDA_CHECK(cudaMemcpy(retainH.data(), ctx->dDsRetain.p, sizeof(double) * nS, cudaMemcpyDeviceToHost));
        }
        if (ctx->dsSlackOnly && ctx->dDsSlackAbsorbed.p) {
            CUDA_CHECK(cudaMemcpy(absH.data(), ctx->dDsSlackAbsorbed.p, sizeof(double) * nS, cudaMemcpyDeviceToHost));
        }
        for (int s = 0; s < nS; s++) {
            double slackP = qbufH[static_cast<size_t>(s) * n + ctx->dsSlackPhiRow];
            double rem = (slackP - ctx->dsSlackTargetP) - retainH[s] - absH[s];
            std::fprintf(stderr, "[DS-DEV] outer=%d s=%d live=%d slackP=%.9f slackTargetP=%.9f retain=%.9f rem=%.9f tol=%.3e fires=%d\n",
                    outer, s, (int) live[s], slackP, ctx->dsSlackTargetP, retainH[s], rem, SLACK_DIST_TOL,
                    std::fabs(rem) > SLACK_DIST_TOL ? 1 : 0);
        }
    }
    if (inDsPhase && (ctx->nds > 0 || ctx->dsSlackOnly) && dsRejectResidual) {
        // slackDistributionFailureBehavior != LEAVE_ON_SLACK_BUS: a scenario whose participants all
        // saturated must NOT be reported converged with the residual absorbed at the slack — route it
        // to the CPU fallback (status -2), which replays it with the run's real failure behavior.
        std::vector<char> dsResidual(nS);
        CUDA_CHECK(cudaMemcpy(dsResidual.data(), ctx->dDsResidual.p, nS * sizeof(char), cudaMemcpyDeviceToHost));
        for (int s = 0; s < nS; s++) {
            if (dsResidual[s] && live[s]) {
                traceReject(s, "ds-residual-saturated", outer, 0);
                res.status[s] = -2;
                active[s] = 0;
                live[s] = 0;
                rlSwitched[s] = 0;
            }
        }
    }
    int nChanged = 0;
    for (int s = 0; s < nS; s++) {
        if (rlSwitched[s]) {
            nChanged++;
        }
    }
    // OLF_GPU_LOOP_TRACE=<slot>: the device side of OLF's OL_CHECK - which loop this pass ran, whether
    // it found the slot UNSTABLE (a re-solve follows) and whether the whole phase is still unstable.
    // Two arms that agree on every residual and still stop apart differ in this SEQUENCE.
    static const char* loopTraceEnv = std::getenv("OLF_GPU_LOOP_TRACE");
    // a slot resolved per chunk from a contingency id (setBatchLoopTraceSlot) wins over the env slot:
    // chunk composition moves with the resolved chunk size, so a fixed slot traces a different
    // scenario from one run to the next (IZUZ: slot 170, then 18)
    const int loopTraceSlot = ctx->loopTraceSlot >= 0 ? ctx->loopTraceSlot : (loopTraceEnv != nullptr ? std::atoi(loopTraceEnv) : -1);
    if (loopTraceSlot >= 0 && loopTraceSlot < nS) {
        double gate[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};   // see the gateLast write
        if (ctx->dDsGateLast.p) {
            CUDA_CHECK(cudaMemcpy(gate, ctx->dDsGateLast.as<double>() + 8 * loopTraceSlot, 8 * sizeof(double), cudaMemcpyDeviceToHost));
        }
        std::fprintf(stderr, "LOOP_PASS s=%d outer=%d loopIdx=%d kind=%d solved=%d switched=%d nChanged=%d live=%d status=%d"
                     " dsRem=%.12f dsFire=%d nrIters=%d dsSlackP=%.12f dsSlackTgt=%.12f dsSlackMismatch=%.12f"
                     " dsRetain=%.12f fsGlobal=%.12f dsPlaced=%.12f\n",
                     loopTraceSlot, outer, loopIdx, loopOrder[loopIdx], solvePass ? 1 : 0,
                     (int) rlSwitched[loopTraceSlot], nChanged, (int) live[loopTraceSlot], res.status[loopTraceSlot],
                     gate[0], (int) gate[1], res.iters[loopTraceSlot],
                     gate[2], gate[3], gate[4], gate[5], gate[6], gate[7]);
    }
    if (iterLog) {
        std::vector<double> gates;
        if (ctx->dDsGateLast.p) {
            gates.resize(static_cast<size_t>(nS) * 8);
            CUDA_CHECK(cudaMemcpy(gates.data(), ctx->dDsGateLast.p, gates.size() * sizeof(double), cudaMemcpyDeviceToHost));
        }
        for (int s = 0; s < nS; s++) {
            char line[256];
            std::snprintf(line, sizeof line, "PASS_LOG outer=%d loopIdx=%d kind=%d solved=%d s=%d live=%d switched=%d iters=%d"
                          " status=%d dsRem=%.17e dsFire=%d dsPlaced=%.17e",
                          outer, loopIdx, loopOrder[loopIdx], solvePass ? 1 : 0, s, (int) live[s], (int) rlSwitched[s],
                          res.iters[s], res.status[s], gates.empty() ? 0.0 : gates[8 * s],
                          gates.empty() ? 0 : (int) gates[8 * s + 1], gates.empty() ? 0.0 : gates[8 * s + 7]);
            iterLogLines.emplace_back(line);
        }
    }
    if (nChanged > 0) {                                     // this outer loop is still unstable
        for (int s = 0; s < nS; s++) {
            if (rlSwitched[s] && ++prodIters[s] >= RL_MAX_OUTER) {
                // This scenario exhausted ITS outer-iteration budget mid-sequence. Never report it
                // converged on a half-finished RL/DS fixpoint — reject to the CPU fallback, which
                // replays it under OLF's own maxOuterLoopIterations/UNSTABLE handling.
                traceReject(s, "outer-loop-budget", outer, 0);
                res.status[s] = -2;
                rlSwitched[s] = 0;
                live[s] = 0;
            }
            active[s] = rlSwitched[s];                      // re-solve only the scenarios it changed, stay in this phase
        }
        seqChanged = true;
        solvePass = true;                                   // mutated targets/modes -> a real re-solve is due
    } else {                                               // this outer loop stabilized
        // The last loop that can fix an unrealistic state is done for this sweep: AcloadFlowEngine checks the
        // state it leaves (runOuterLoop, after its do-while), whether or not the loop re-solved. The solve gate
        // above only sees the solves of the LATER loops, so with that loop last in the order (the discrete
        // TransformerVoltageControl under the validation parameters) only the final state was ever checked.
        if (ctx->realisticCheck && realisticFixingIdx >= 0 && loopIdx == realisticFixingIdx) {
            std::vector<char> gate(nS);
            for (int s = 0; s < nS; s++) {
                gate[s] = (res.status[s] == 0 && live[s]) ? 1 : 0;
            }
            rejectUnrealistic(ctx, drv, gate);
        }
        if (++loopIdx >= nLoops) {                          // ...move to the next loop in OLF's order
            if (!seqChanged) {
                break;                                      // a full sweep re-solved nothing -> converged
            }
            seqChanged = false;
            loopIdx = 0;                                    // restart the sweep at the first loop
        }
        for (int s = 0; s < nS; s++) {
            active[s] = live[s];                            // re-CHECK every live scenario in the next phase...
        }
        solvePass = false;                                  // ...against the SAME state: STABLE never re-solves (OLF)
    }
    }   // ---- end sequential outer-loop driver ----

    if (iterLog) {
        for (const std::string& line : iterLogLines) {
            std::fprintf(stderr, "%s\n", line.c_str());
        }
        std::fprintf(stderr, "ITER_LOG_END nS=%d lines=%zu\n", nS, iterLogLines.size());
    }

    if (ctx->dSwLog.p && (swLogSlot == -2 || (swLogSlot >= 0 && swLogSlot < nS))) {
        int cnt = 0;
        CUDA_CHECK(cudaMemcpy(&cnt, ctx->dSwLogCnt.p, sizeof(int), cudaMemcpyDeviceToHost));
        int keep = cnt > SW_LOG_CAP ? SW_LOG_CAP : cnt;
        std::vector<double> lg((size_t) keep * SW_LOG_W);
        if (keep > 0) {
            CUDA_CHECK(cudaMemcpy(lg.data(), ctx->dSwLog.p, lg.size() * sizeof(double), cudaMemcpyDeviceToHost));
        }
        for (int k = 0; k < keep; k++) {
            const double* r = lg.data() + (size_t) k * SW_LOG_W;
            std::fprintf(stderr, "SW_LOG s=%d pass=%d entry=%d act=%d lt=%d sc=%d q=%.17e min=%.17e max=%.17e v=%.17e cmpV=%.17e vt=%.17e\n",
                         (int) r[11], (int) r[0], (int) r[1], (int) r[2], (int) r[9], (int) r[10],
                         r[3], r[4], r[5], r[6], r[7], r[8]);
        }
        std::fprintf(stderr, "SW_LOG_END s=%d records=%d (cap %d)\n", swLogSlot, cnt, SW_LOG_CAP);
        CUDA_CHECK(cudaMemset(ctx->dSwLogCnt.p, 0, sizeof(int)));
    }
    // ---- OLF_GPU_RL_AUDIT=<pu threshold>: audit the FINAL state against the reactive-limit rule
    //      itself. OLF's outer loop ends only after a check found NOTHING to switch, so in OLF's
    //      answer every UNPINNED controller sits inside its reactive limits (within the switch
    //      tolerance). A device scenario that returns with an unpinned controller beyond its limit
    //      did not reach that fixpoint - it stopped one check short - and that is a defect no
    //      comparison with the CPU is needed to see. Re-fills the diagnostic q at the final state
    //      for every live scenario, refreshes the curve limits at the final targets, and prints one
    //      line per violating (scenario, entry).
    if (ctx->nrl > 0 && std::getenv("OLF_GPU_RL_AUDIT") != nullptr) {
        double auditTol = std::atof(std::getenv("OLF_GPU_RL_AUDIT"));
        if (auditTol <= 0.0) auditTol = 1e-6;
        std::vector<char> allLive(nS), auditDumped(nS, 0);
        for (int s = 0; s < nS; s++) allLive[s] = (res.status[s] == 0) ? 1 : 0;
        CUDA_CHECK(cudaMemcpy(ctx->dRlActive.p, allLive.data(), nS * sizeof(char), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemsetAsync(ctx->dQbuf.p, 0, nTot * sizeof(double), ctx->stream));
        if (ctx->st->cb.ne) fillClosedBranchBatch<<<grid(ctx->st->cb.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->cb.ne, nS, n, nnz, ctx->nElements, ctx->cbOffset, ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->st->cb.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dRlActive.as<char>(), ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(), nullptr, nullptr, nullptr, 0, nullptr, nullptr, RelocTables{});
        if (ctx->st->ob.ne) fillOpenBranchBatch<<<grid(ctx->st->ob.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->ob.ne, nS, n, nnz, ctx->nElements, ctx->obOffset, ctx->st->ob.in.as<double>(), ctx->st->ob.ix.as<int>(), ctx->st->ob.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dRlActive.as<char>(), nullptr, nullptr, nullptr, 0, RelocTables{});
        if (ctx->st->sh.ne) fillShuntBatch<<<grid(ctx->st->sh.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->sh.ne, nS, n, nnz, ctx->nElements, ctx->shOffset, ctx->st->sh.in.as<double>(), ctx->st->sh.ix.as<int>(), ctx->st->sh.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dRlActive.as<char>(), ctx->nShOv, ctx->dShOvSlotElem.as<int>(), ctx->dShOvGb.as<double>(), nullptr, nullptr, nullptr, 0, RelocTables{});
        if (ctx->st->nvdl) fillVdlBatch<<<grid(ctx->st->nvdl * nS), BLOCK, 0, ctx->stream>>>(ctx->st->nvdl, nS, n, nnz, ctx->st->dVdlIn.as<double>(), ctx->st->dVdlIdx.as<int>(), ctx->st->dVdlDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), nullptr, nullptr, nullptr);
        if (ctx->nzi) fillZeroImpBatch<<<grid(ctx->nzi * nS), BLOCK, 0, ctx->stream>>>(ctx->nzi, nS, n, nnz, ctx->st->dZiIn.as<double>(), ctx->st->dZiIdx.as<int>(), ctx->st->dZiDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), ctx->dRlActive.as<char>(), ctx->nziBase, ctx->nZiOn, ctx->dZiOnSlotElem.as<int>(), ctx->nZiOff, ctx->dZiOffSlotElem.as<int>(), nullptr, nullptr, nullptr, 0, RelocTables{}, ctx->nZiTree, ctx->nZiTree ? ctx->dZiTreeSlotElemVal.as<int>() : nullptr, ctx->hasZiMasks ? ctx->dZiOffMask.as<char>() : nullptr, ctx->hasZiMasks ? ctx->dZiOnMask.as<char>() : nullptr, ctx->hasZiMasks ? ctx->dZiTreeMask.as<char>() : nullptr);
        if (ctx->nRmq > 0) {
            fillRemoteMemberQBatch<<<grid(ctx->nRmq * nS), BLOCK, 0, ctx->stream>>>(ctx->nRmq, nS, n,
                    ctx->nElements, ctx->cbOffset, ctx->obOffset,
                    ctx->dRmqRow.as<int>(), ctx->dRmqElem.as<int>(), ctx->dRmqSide.as<int>(),
                    ctx->dRmqKind.as<int>(), ctx->dRmqWeight.as<double>(),
                    ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->st->sh.in.as<double>(), ctx->st->sh.ix.as<int>(),
                    ctx->st->ob.in.as<double>(), ctx->st->ob.ix.as<int>(),
                    ctx->dX.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dRlActive.as<char>(),
                    ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(),
                    ctx->nShOv, ctx->dShOvSlotElem.as<int>(), ctx->dShOvGb.as<double>(), -1);
        }
        // The audit's verdict comes from OLF's OWN check, not a second implementation of it: run the
        // real reactiveLimitsBatch on the final state. It either changes nothing - the state IS the
        // fixpoint the loop claimed - or it makes exactly the change the loop still owed, and then
        // rlSwitched names the scenarios that were returned one check short. Running it on the real
        // buffers is safe for the same reason: a fixpoint is a no-op.
        if (useDynRlLimits) {                                // limits at the FINAL targets
            rlCurveLimitsBatch<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(ctx->nrl, nS, ctx->ncv, n, ctx->cvExtrapolate,
                    ctx->dCvIdx.as<int>(), ctx->dCvIn.as<double>(), ctx->dCvPts.as<double>(),
                        ctx->dCvPtr.as<int>(), ctx->dCvList.as<int>(), cvHvdcView(ctx),
                    ctx->dRlMinQ.as<double>(), ctx->dRlMaxQ.as<double>(), ctx->dRlLoadQ.as<double>(),
                    ctx->dTarget.as<double>(),
                    nRlQOv, nRlQOv > 0 ? dRlQOvSlotEntry.as<int>() : nullptr,
                    nRlQOv > 0 ? dRlQOvMinMax.as<double>() : nullptr,
                        nRlQOv > 0 ? dRlQOvPtr.as<int>() : nullptr, nRlQOv > 0 ? dRlQOvList.as<int>() : nullptr,
                    nCvOv > 0 ? dCvOvPtr.as<int>() : nullptr,
                    nCvOv > 0 ? dCvOvEntry.as<int>() : nullptr,
                    nCvOv > 0 ? dCvOvVal.as<double>() : nullptr,
                    ctx->ngen > 0 ? ctx->dGenCum.as<double>() : nullptr,
                    ctx->ngen > 0 ? ctx->dGenCumValid.as<char>() : nullptr, ctx->hasDgAnchor ? ctx->dDgAnchor.as<double>() : nullptr, ctx->hasDgStart ? ctx->dDgStart.as<double>() : nullptr, ctx->ngen,
                    ctx->nds > 0 ? ctx->dDsPhiRow.as<int>() : nullptr,
                    ctx->nDsRowOv > 0 ? ctx->dDsRowOv.as<int>() : nullptr, ctx->nds,
                    ctx->ngen > 0 ? ctx->dgBus.as<int>() : nullptr,
                    ctx->dRlMinQDyn.as<double>(), ctx->dRlMaxQDyn.as<double>(),
                    ctx->dRlLoadQDyn.as<double>(), -1, 0);
            launchForcedTargetQLimits(ctx, nS, n);
        }
        CUDA_CHECK(cudaMemsetAsync(ctx->dRlPassSwitched.p, 0, static_cast<size_t>(ctx->nrl) * nS, ctx->stream));
        CUDA_CHECK(cudaMemsetAsync(ctx->dRlSwitched.p, 0, nS, ctx->stream));
        // The check writes pin state, row modes and row targets. The context is REUSED by the next
        // chunk, so the audit stashes everything the kernel can touch and puts it back: an
        // instrument that changes the next chunk's answer is not an instrument. dDx/dB are dead
        // here (the solve is over) and are exactly n*nS wide, so they carry the two big ones.
        CUDA_CHECK(cudaMemcpyAsync(ctx->dDx.p, ctx->dTarget.p, (size_t) n * nS * sizeof(double), cudaMemcpyDeviceToDevice, ctx->stream));
        CUDA_CHECK(cudaMemcpyAsync(ctx->dB.p, ctx->dMode.p, (size_t) n * nS * sizeof(int), cudaMemcpyDeviceToDevice, ctx->stream));
        std::vector<char> keepLt(static_cast<size_t>(ctx->nrl) * nS), keepSc(static_cast<size_t>(ctx->nrl) * nS), keepSm(static_cast<size_t>(ctx->nrl) * nS);
        std::vector<double> keepSt(static_cast<size_t>(ctx->nrl) * nS);
        CUDA_CHECK(cudaMemcpy(keepLt.data(), ctx->dRlLimitType.p, keepLt.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(keepSc.data(), ctx->dRlSwitchCount.p, keepSc.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(keepSm.data(), ctx->dRlSavedMode.p, keepSm.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(keepSt.data(), ctx->dRlSavedTarget.p, keepSt.size() * sizeof(double), cudaMemcpyDeviceToHost));
        reactiveLimitsBatch<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(
                ctx->nSvc > 0 ? ctx->dSvcRlVt.as<double>() : nullptr,
                ctx->nrl, nS, n, REACTIVE_LIMIT_TOL, RL_MAX_SWITCH,
                ctx->dRlVrow.as<int>(),
                ctx->dRlQrow.p ? ctx->dRlQrow.as<int>() : nullptr,
                useDynRlLimits ? ctx->dRlMinQDyn.as<double>() : ctx->dRlMinQ.as<double>(),
                useDynRlLimits ? ctx->dRlMaxQDyn.as<double>() : ctx->dRlMaxQ.as<double>(),
                useDynRlLimits ? ctx->dRlLoadQDyn.as<double>() : ctx->dRlLoadQ.as<double>(),
                ctx->dRlVtarget.as<double>(), ctx->dQbuf.as<double>(), ctx->dX.as<double>(), ctx->dRlActive.as<char>(),
                ctx->dMode.as<int>(), ctx->dTarget.as<double>(),
                ctx->dRlLimitType.as<char>(), ctx->dRlSwitchCount.as<char>(), ctx->dRlSwitched.as<char>(),
                ctx->dRlPassSwitched.as<char>(), ctx->dRlSavedMode.as<char>(),
                nRlVtOv, nRlVtOv > 0 ? dRlVtOvSlotIdx.as<int>() : nullptr,
                nRlVtOv > 0 ? dRlVtOvVal.as<double>() : nullptr,
                ctx->dRowSlope.as<double>(), ctx->dRlSavedTarget.as<double>(), useDynRlLimits ? 1 : 0,
                ctx->nRlRemote > 0 ? ctx->dRlRemote.as<char>() : nullptr,
                ctx->nRlRemote > 0 ? ctx->dRlCmpVrow.as<int>() : nullptr,
                ctx->dRlSvcBmin.as<double>(), ctx->dRlSvcBmax.as<double>(), ctx->dRlV2Base.as<double>(),
                    svcCoefDyn ? ctx->dRlSvcBminDyn.as<double>() : nullptr, svcCoefDyn ? ctx->dRlSvcBmaxDyn.as<double>() : nullptr,
                ctx->nRlIslanded > 0 ? ctx->dRlIslanded.as<char>() : nullptr,
                ctx->rlRobust ? ctx->dRlInitTgtQ.as<double>() : nullptr,
                ctx->rlVLoRobust, ctx->rlVHiRobust, ctx->rlRobust,
                swLogSlot, ctx->dSwLog.p ? ctx->dSwLog.as<double>() : nullptr,
                ctx->dSwLogCnt.p ? ctx->dSwLogCnt.as<int>() : nullptr, 999999,
                    psEntryPtr(ctx), ctx->st->nQr,
                    ctx->nRlLocal > 0 ? ctx->dRlLocalOv.as<char>() : nullptr,
                    ctx->nRlLocal > 0 ? ctx->dRlLocalVt.as<double>() : nullptr,
                    ctx->nRlNoUnpin > 0 ? ctx->dRlNoUnpin.as<char>() : nullptr, ctx->nRlNoUnpin > 0 ? ctx->dRlUnpinBad.as<char>() : nullptr,
                    ctx->dRlMonitored.p ? ctx->dRlMonitored.as<char>() : nullptr,
                    ctx->nRlCmp > 0 ? ctx->dRlCmpOv.as<int>() : nullptr);
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        std::vector<char> aSw(nS), aPs(static_cast<size_t>(ctx->nrl) * nS);
        CUDA_CHECK(cudaMemcpy(aSw.data(), ctx->dRlSwitched.p, nS, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(aPs.data(), ctx->dRlPassSwitched.p, aPs.size(), cudaMemcpyDeviceToHost));
        std::vector<char> altPost(static_cast<size_t>(ctx->nrl) * nS);
        CUDA_CHECK(cudaMemcpy(altPost.data(), ctx->dRlLimitType.p, altPost.size(), cudaMemcpyDeviceToHost));
        // put every touched buffer back before anything else reads them
        CUDA_CHECK(cudaMemcpyAsync(ctx->dTarget.p, ctx->dDx.p, (size_t) n * nS * sizeof(double), cudaMemcpyDeviceToDevice, ctx->stream));
        CUDA_CHECK(cudaMemcpyAsync(ctx->dMode.p, ctx->dB.p, (size_t) n * nS * sizeof(int), cudaMemcpyDeviceToDevice, ctx->stream));
        CUDA_CHECK(cudaMemcpy(ctx->dRlLimitType.p, keepLt.data(), keepLt.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->dRlSwitchCount.p, keepSc.data(), keepSc.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->dRlSavedMode.p, keepSm.data(), keepSm.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->dRlSavedTarget.p, keepSt.data(), keepSt.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemsetAsync(ctx->dRlSwitched.p, 0, nS, ctx->stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        std::vector<double> aq(static_cast<size_t>(n) * nS), ax(static_cast<size_t>(n) * nS), alq(ctx->nrl);
        std::vector<char> alt(static_cast<size_t>(ctx->nrl) * nS), aisl;
        std::vector<int> avr(ctx->nrl), amode(static_cast<size_t>(n) * nS);
        CUDA_CHECK(cudaMemcpy(aq.data(), ctx->dQbuf.p, aq.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ax.data(), ctx->dX.p, ax.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(alq.data(), ctx->dRlLoadQ.p, alq.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(alt.data(), ctx->dRlLimitType.p, alt.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(avr.data(), ctx->dRlVrow.p, avr.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(amode.data(), ctx->dMode.p, amode.size() * sizeof(int), cudaMemcpyDeviceToHost));
        if (ctx->nRlIslanded > 0) {
            aisl.resize(static_cast<size_t>(ctx->nrl) * nS);
            CUDA_CHECK(cudaMemcpy(aisl.data(), ctx->dRlIslanded.p, aisl.size(), cudaMemcpyDeviceToHost));
        }
        std::vector<double> amn(static_cast<size_t>(ctx->nrl) * (ctx->ncv > 0 ? nS : 1)), amx;
        amx.resize(amn.size());
        CUDA_CHECK(cudaMemcpy(amn.data(), ctx->ncv > 0 ? ctx->dRlMinQDyn.p : ctx->dRlMinQ.p, amn.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(amx.data(), ctx->ncv > 0 ? ctx->dRlMaxQDyn.p : ctx->dRlMaxQ.p, amx.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<int> acv(ctx->nrl, -1);
        std::vector<double> avt(ctx->nrl, 0.0);
        if (ctx->dRlCmpVrow.p) {
            CUDA_CHECK(cudaMemcpy(acv.data(), ctx->dRlCmpVrow.p, acv.size() * sizeof(int), cudaMemcpyDeviceToHost));
        }
        if (ctx->dRlVtarget.p) {
            CUDA_CHECK(cudaMemcpy(avt.data(), ctx->dRlVtarget.p, avt.size() * sizeof(double), cudaMemcpyDeviceToHost));
        }
        for (int s = 0; s < nS; s++) {
            if (!allLive[s] || !aSw[s]) continue;
            for (int i = 0; i < ctx->nrl; i++) {
                size_t st = (size_t) s * ctx->nrl + i;
                if (!aPs[st]) continue;
                size_t li = ctx->ncv > 0 ? st : (size_t) i;
                double cmpV = (acv[i] >= 0) ? ax[(size_t) s * n + acv[i]] : 0.0;
                // act 1 = a pin this check owed, 2 = an unpin, 3 = a moved limit it owed
                std::fprintf(stderr, "RL_OWED s=%d entry=%d vrow=%d act=%d ltBefore=%d ltAfter=%d q=%.15e min=%.15e max=%.15e v=%.15e cmpV=%.15e vt=%.15e margin=%.3e\n",
                             s, i, avr[i], (int) aPs[st], (int) alt[st], (int) altPost[st],
                             aq[(size_t) s * n + avr[i]] + alq[i], amn[li], amx[li],
                             ax[(size_t) s * n + avr[i]], cmpV, avt[i], cmpV - avt[i]);
                if (!auditDumped.empty() && !auditDumped[s]) {
                    auditDumped[s] = 1;
                    for (const std::string& r : auditLog[s]) {
                        std::fprintf(stderr, "AUDIT_HIST s=%d | %s", s, r.c_str());
                    }
                }
            }
        }
        for (int s = 0; s < nS; s++) {
            if (!allLive[s]) continue;
            for (int i = 0; i < ctx->nrl; i++) {
                size_t st = (size_t) s * ctx->nrl + i;
                if (alt[st] != 0) continue;                  // pinned: its row holds the limit by construction
                if (!aisl.empty() && aisl[st]) continue;     // not a controller in this component
                int m = amode[(size_t) s * n + avr[i]];
                if (m == 4) continue;                        // row owns its own Q balance this scenario
                size_t li = ctx->ncv > 0 ? st : (size_t) i;
                double q = aq[(size_t) s * n + avr[i]] + alq[i];
                double over = q < amn[li] ? amn[li] - q : (q > amx[li] ? q - amx[li] : 0.0);
                if (over > auditTol) {
                    std::fprintf(stderr, "RL_AUDIT s=%d entry=%d vrow=%d mode=%d q=%.15e min=%.15e max=%.15e over=%.6e v=%.15e\n",
                                 s, i, avr[i], m, q, amn[li], amx[li], over, ax[(size_t) s * n + avr[i]]);
                    if (!auditDumped.empty() && !auditDumped[s]) {
                        auditDumped[s] = 1;
                        for (const std::string& r : auditLog[s]) {
                            std::fprintf(stderr, "AUDIT_HIST s=%d | %s", s, r.c_str());
                        }
                    }
                }
            }
        }
    }
    if (getenv("OLF_GPU_DBG_DS") && ctx->nds > 0) {
        std::vector<double> tFinal(static_cast<size_t>(n) * nS);
        CUDA_CHECK(cudaMemcpy(tFinal.data(), ctx->dTarget.p, tFinal.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<char> ltDump(static_cast<size_t>(std::max(ctx->nrl, 1)) * nS, 0);
        if (ctx->nrl > 0) {
            CUDA_CHECK(cudaMemcpy(ltDump.data(), ctx->dRlLimitType.p, static_cast<size_t>(ctx->nrl) * nS, cudaMemcpyDeviceToHost));
        }
        std::vector<int> dsRows(ctx->nds);
        std::vector<double> dsBase(ctx->nds);
        CUDA_CHECK(cudaMemcpy(dsRows.data(), ctx->dDsPhiRow.p, ctx->nds * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dsBase.data(), ctx->dDsBaseTargetP.p, ctx->nds * sizeof(double), cudaMemcpyDeviceToHost));
        for (int s = 0; s < nS; s++) {
            double tot = 0;
            fprintf(stderr, "GPU-DS s=%d", s);
            for (int i = 0; i < ctx->nds; i++) {
                double d = tFinal[(size_t) s * n + dsRows[i]] - dsBase[i];
                tot += d;
                fprintf(stderr, " row%d=%+.4f", dsRows[i], d * 100.0);
            }
            fprintf(stderr, " | totMW=%+.4f | lt:", tot * 100.0);
            for (int i = 0; i < ctx->nrl; i++) fprintf(stderr, "%d", ltDump[(size_t) i * nS + s]);
            fprintf(stderr, " status=%d\n", res.status[s]);
        }
    }
    if (rlStateOut != nullptr && ctx->nrl > 0) {             // export per-(controller, scenario) RL state
        std::vector<char> lt(static_cast<size_t>(ctx->nrl) * nS), sc(static_cast<size_t>(ctx->nrl) * nS);
        CUDA_CHECK(cudaMemcpy(lt.data(), ctx->dRlLimitType.p, lt.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sc.data(), ctx->dRlSwitchCount.p, sc.size(), cudaMemcpyDeviceToHost));
        for (size_t k = 0; k < lt.size(); k++) {
            rlStateOut[k] = (lt[k] & 7) | (sc[k] << 3);
        }
    }
    if (!ctx->driverPaused && std::getenv("OLF_GPU_PHASE_TIME") != nullptr && ctx->olPasses.size() == static_cast<size_t>(nS)) {
        for (int s = 0; s < nS; s++) {
            if (ctx->olPasses[s] == 0) continue;
            if (res.status[s] != 0) {                         // not finished converged on device (CPU replay,
                gOlNotConverged++;                           // max-iter, ...): its work is counted apart
                gOlNotConvergedIters += ctx->olIters[s];
                continue;
            }
            int k = std::min(ctx->olPasses[s], 15);
            gOlHistCount[k]++;
            gOlHistFirst[k] += ctx->olFirst[s];
            gOlHistTotal[k] += ctx->olIters[s];
        }
    }
    CUDA_CHECK(cudaMemcpy(res.x.data(), ctx->dX.p, nTot * sizeof(double), cudaMemcpyDeviceToHost));
    return res;
}

}  // namespace olfgpu

using namespace olfgpu;

extern "C" JNIEXPORT jlong JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_createStructure(
        JNIEnv* env, jclass, jint n,
        jdoubleArray jcbIn, jintArray jcbIdx, jdoubleArray jobIn, jintArray jobIdx,
        jdoubleArray jshIn, jintArray jshIdx, jdoubleArray jhvIn, jintArray jhvIdx,
        jintArray jdqElem, jintArray jdqSide, jdoubleArray jdqWeight, jintArray jdqRow, jintArray jdqKind,
        jdoubleArray jziIn, jintArray jziIdx,
        jintArray jdrRow, jintArray jdrCol, jdoubleArray jdrCoef,
        jintArray jdbRow, jintArray jdbCol, jdoubleArray jdbCoef, jintArray jcbPcSide,
        jdoubleArray jvdlIn, jintArray jvdlIdx, jintArray jqrPtr, jintArray jqrRows,
        jintArray jobQAlt, jintArray jziQAlt, jintArray jqrDepBase, jintArray jqrBaseTargetRow, jintArray jdepGroupOfRow) {
    jdouble* cbIn = env->GetDoubleArrayElements(jcbIn, nullptr);
    jint* cbIdx = env->GetIntArrayElements(jcbIdx, nullptr);
    jdouble* obIn = env->GetDoubleArrayElements(jobIn, nullptr);
    jint* obIdx = env->GetIntArrayElements(jobIdx, nullptr);
    jdouble* shIn = env->GetDoubleArrayElements(jshIn, nullptr);
    jint* shIdx = env->GetIntArrayElements(jshIdx, nullptr);
    jdouble* hvIn = env->GetDoubleArrayElements(jhvIn, nullptr);
    jint* hvIdx = env->GetIntArrayElements(jhvIdx, nullptr);
    jint* dqElem = env->GetIntArrayElements(jdqElem, nullptr);
    jint* dqSide = env->GetIntArrayElements(jdqSide, nullptr);
    jdouble* dqWeight = env->GetDoubleArrayElements(jdqWeight, nullptr);
    jint* dqRow = env->GetIntArrayElements(jdqRow, nullptr);
    jint* dqKind = env->GetIntArrayElements(jdqKind, nullptr);
    jdouble* ziIn = env->GetDoubleArrayElements(jziIn, nullptr);
    jint* ziIdx = env->GetIntArrayElements(jziIdx, nullptr);
    jint* drRow = env->GetIntArrayElements(jdrRow, nullptr);
    jint* drCol = env->GetIntArrayElements(jdrCol, nullptr);
    jdouble* drCoef = env->GetDoubleArrayElements(jdrCoef, nullptr);
    jint* dbRow = env->GetIntArrayElements(jdbRow, nullptr);
    jint* dbCol = env->GetIntArrayElements(jdbCol, nullptr);
    jdouble* dbCoef = env->GetDoubleArrayElements(jdbCoef, nullptr);
    jint* cbPcSide = env->GetIntArrayElements(jcbPcSide, nullptr);
    jint* qrPtr = env->GetIntArrayElements(jqrPtr, nullptr);
    jint* qrRows = env->GetIntArrayElements(jqrRows, nullptr);
    int nQr = env->GetArrayLength(jqrPtr) - 1;               // groups; qrPtr = [0] when none
    int nQrRows = env->GetArrayLength(jqrRows);
    jdouble* vdlIn = env->GetDoubleArrayElements(jvdlIn, nullptr);
    jint* vdlIdx = env->GetIntArrayElements(jvdlIdx, nullptr);
    jint* obQAlt = env->GetIntArrayElements(jobQAlt, nullptr);
    jint* ziQAlt = env->GetIntArrayElements(jziQAlt, nullptr);
    jint* qrDepBase = env->GetIntArrayElements(jqrDepBase, nullptr);
    jint* qrBaseTargetRow = env->GetIntArrayElements(jqrBaseTargetRow, nullptr);
    jint* depGroupOfRow = env->GetIntArrayElements(jdepGroupOfRow, nullptr);

    int ncb = env->GetArrayLength(jcbIdx) / 12;             // cbIdx stride 12 (…a1col r1col qAlt1 qAlt2)
    int nob = env->GetArrayLength(jobIdx) / 6;
    int nsh = env->GetArrayLength(jshIdx) / 5;             // shIdx stride 5 (P Q v bcol qAlt)
    int nhv = env->GetArrayLength(jhvIdx) / 4;
    int ndq = env->GetArrayLength(jdqElem);
    int nzi = env->GetArrayLength(jziIdx) / 11;
    int ndr = env->GetArrayLength(jdrRow);
    int ndb = env->GetArrayLength(jdbRow);
    int nvdl = env->GetArrayLength(jvdlIdx) / 2;             // vdlIdx stride 2 (row, vCol)

    std::string error;
    SharedStructurePtr* st = nullptr;
    try {
        // DEFENSIVE input validation (an intermittent SIGSEGV at buildStructure+0xe43 — three
        // hs_err logs 2026-09-06 — is a one-past-the-end host read that only faults at a page
        // boundary; a named exception routes to the CPU fallback and identifies the bad input).
        auto checkRange = [&](const char* what, const jint* a, size_t len, size_t stride,
                              std::initializer_list<size_t> slots, int lo, int hi) {
            for (size_t e = 0; e < len; e++) {
                for (size_t sl : slots) {
                    int v = a[e * stride + sl];
                    if (v < lo || v >= hi) {
                        throw std::runtime_error(std::string("bad ") + what + " at elem " + std::to_string(e)
                                + " slot " + std::to_string(sl) + ": " + std::to_string(v)
                                + " (allowed [" + std::to_string(lo) + ", " + std::to_string(hi) + "))");
                    }
                }
            }
        };
        // rows/cols may be -1 = absent; anything else must be a valid row/col index < n
        checkRange("cbIdx", cbIdx, ncb, 12, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, -1, n);
        // qAlt slots ([10]/[11] cb, [4] sh) also carry the GROUP encoding -2-groupId (plan-A6)
        checkRange("cbIdx.qAlt", cbIdx, ncb, 12, {10, 11}, -1 - nQr, n);
        checkRange("obIdx", obIdx, nob, 6, {0, 1, 2, 3, 4, 5}, -1, n);
        checkRange("obQAlt", obQAlt, nob, 2, {0, 1}, -1 - nQr, n);
        checkRange("shIdx", shIdx, nsh, 5, {0, 1, 2, 3}, -1, n);
        checkRange("shIdx.qAlt", shIdx, nsh, 5, {4}, -1 - nQr, n);
        checkRange("qrRows", qrRows, nQrRows, 1, {0}, 0, n);
        for (int g = 0; g < nQr; g++) {
            if (qrPtr[g] < 0 || qrPtr[g + 1] <= qrPtr[g] + 1 || qrPtr[g + 1] > nQrRows) {
                throw std::runtime_error("bad qrPtr group " + std::to_string(g));
            }
        }
        checkRange("hvIdx", hvIdx, nhv, 4, {0, 1, 2, 3}, -1, n);
        checkRange("dqRow", dqRow, ndq, 1, {0}, 0, n);
        checkRange("ziIdx", ziIdx, nzi, 11, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, -1, n);
        // slot 10 is the FLAGS word, not a row: bit 0 spanning-tree, bits 1-2 the relocated child side,
        // bit 3 "the side came from the coupler-tree walk". Checking it against n rejected a valid
        // structure on any network with fewer than 14 rows once the relocation went on by default
        // ("bad ziIdx at elem 0 slot 10: 13 (allowed [-1, 10))", four fixtures).
        checkRange("ziIdx.flags", ziIdx, nzi, 11, {10}, 0, 16);
        checkRange("ziQAlt", ziQAlt, nzi, 2, {0, 1}, -1 - nQr, n);
        checkRange("drRow", drRow, ndr, 1, {0}, 0, n);
        checkRange("drCol", drCol, ndr, 1, {0}, 0, n);
        checkRange("dbRow", dbRow, ndb, 1, {0}, 0, n);
        checkRange("dbCol", dbCol, ndb, 1, {0}, 0, n);
        checkRange("vdlIdx", vdlIdx, nvdl, 2, {0, 1}, 0, n);
        for (int e = 0; e < ncb; e++) {                      // encoded (a1Col << 1) | side, or -1
            if (cbPcSide[e] >= 0 && (cbPcSide[e] >> 1) >= n) {
                throw std::runtime_error("bad cbPcSide at elem " + std::to_string(e)
                        + ": a1Col " + std::to_string(cbPcSide[e] >> 1) + " >= n " + std::to_string(n));
            }
        }
        st = new SharedStructurePtr(buildStructure(n, ncb, cbIn, cbIdx, nob, obIn, obIdx,
                                                   nsh, shIn, shIdx, nhv, hvIn, hvIdx,
                                                   ndq, dqElem, dqSide, dqWeight, dqRow, dqKind,
                                                   nzi, ziIn, ziIdx,
                                                   ndr, drRow, drCol, drCoef,
                                                   ndb, dbRow, dbCol, dbCoef, cbPcSide,
                                                   nvdl, vdlIn, vdlIdx, nQr, qrPtr, qrRows,
                                                   obQAlt, ziQAlt, qrDepBase, qrBaseTargetRow, depGroupOfRow));
    } catch (const std::exception& e) {
        error = e.what();
    } catch (...) {
        error = "unknown native error";
    }

    env->ReleaseDoubleArrayElements(jcbIn, cbIn, JNI_ABORT);
    env->ReleaseIntArrayElements(jcbIdx, cbIdx, JNI_ABORT);
    env->ReleaseIntArrayElements(jcbPcSide, cbPcSide, JNI_ABORT);
    env->ReleaseIntArrayElements(jqrPtr, qrPtr, JNI_ABORT);
    env->ReleaseIntArrayElements(jqrRows, qrRows, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jobIn, obIn, JNI_ABORT);
    env->ReleaseIntArrayElements(jobIdx, obIdx, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jshIn, shIn, JNI_ABORT);
    env->ReleaseIntArrayElements(jshIdx, shIdx, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jhvIn, hvIn, JNI_ABORT);
    env->ReleaseIntArrayElements(jhvIdx, hvIdx, JNI_ABORT);
    env->ReleaseIntArrayElements(jdqElem, dqElem, JNI_ABORT);
    env->ReleaseIntArrayElements(jdqSide, dqSide, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jdqWeight, dqWeight, JNI_ABORT);
    env->ReleaseIntArrayElements(jdqRow, dqRow, JNI_ABORT);
    env->ReleaseIntArrayElements(jdqKind, dqKind, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jziIn, ziIn, JNI_ABORT);
    env->ReleaseIntArrayElements(jziIdx, ziIdx, JNI_ABORT);
    env->ReleaseIntArrayElements(jdrRow, drRow, JNI_ABORT);
    env->ReleaseIntArrayElements(jdrCol, drCol, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jdrCoef, drCoef, JNI_ABORT);
    env->ReleaseIntArrayElements(jdbRow, dbRow, JNI_ABORT);
    env->ReleaseIntArrayElements(jdbCol, dbCol, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jdbCoef, dbCoef, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jvdlIn, vdlIn, JNI_ABORT);
    env->ReleaseIntArrayElements(jvdlIdx, vdlIdx, JNI_ABORT);
    env->ReleaseIntArrayElements(jobQAlt, obQAlt, JNI_ABORT);
    env->ReleaseIntArrayElements(jziQAlt, ziQAlt, JNI_ABORT);
    env->ReleaseIntArrayElements(jqrDepBase, qrDepBase, JNI_ABORT);
    env->ReleaseIntArrayElements(jqrBaseTargetRow, qrBaseTargetRow, JNI_ABORT);
    env->ReleaseIntArrayElements(jdepGroupOfRow, depGroupOfRow, JNI_ABORT);

    if (!error.empty()) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      ("GPU structure creation failed: " + error).c_str());
        return 0;
    }
    return reinterpret_cast<jlong>(st);
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_destroyStructure(
        JNIEnv*, jclass, jlong handle) {
    // Drops the Java-side owner reference only: contexts hold their own shared_ptr copies,
    // so the device-resident structure lives until the last context is destroyed.
    delete reinterpret_cast<SharedStructurePtr*>(handle);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_createBatchContext(
        JNIEnv* env, jclass, jlong structureHandle, jint nScenarios, jint mode,
        jintArray jrlVrow, jdoubleArray jrlMinQ, jdoubleArray jrlMaxQ, jdoubleArray jrlLoadQ,
        jdoubleArray jrlVtarget, jbyteArray jrlMonitored, jdoubleArray jrowSlope,
        jintArray jdsPhiRow, jdoubleArray jdsFactor,
        jint jdsSlackPhiRow, jdouble jdsSlackTargetP,
        jdoubleArray jdsBaseTargetP, jdoubleArray jdsMaxDelta, jdoubleArray jdsMinDelta,
        jint jdsMode, jdoubleArray jdsMarginUp, jdoubleArray jdsMarginDown,
        jdouble jdsSlackMarginUp, jdouble jdsSlackMarginDown) {
    auto* st = reinterpret_cast<SharedStructurePtr*>(structureHandle);
    jint* rlVrow = env->GetIntArrayElements(jrlVrow, nullptr);
    jdouble* rlMinQ = env->GetDoubleArrayElements(jrlMinQ, nullptr);
    jdouble* rlMaxQ = env->GetDoubleArrayElements(jrlMaxQ, nullptr);
    jdouble* rlLoadQ = env->GetDoubleArrayElements(jrlLoadQ, nullptr);
    jdouble* rlVtarget = env->GetDoubleArrayElements(jrlVtarget, nullptr);
    jbyte* rlMonitored = env->GetByteArrayElements(jrlMonitored, nullptr);
    jdouble* rowSlope = env->GetDoubleArrayElements(jrowSlope, nullptr);
    jint* dsPhiRow = env->GetIntArrayElements(jdsPhiRow, nullptr);
    jdouble* dsFactor = env->GetDoubleArrayElements(jdsFactor, nullptr);
    jdouble* dsBaseTargetP = env->GetDoubleArrayElements(jdsBaseTargetP, nullptr);
    jdouble* dsMaxDelta = env->GetDoubleArrayElements(jdsMaxDelta, nullptr);
    jdouble* dsMinDelta = env->GetDoubleArrayElements(jdsMinDelta, nullptr);
    jdouble* dsMarginUp = env->GetDoubleArrayElements(jdsMarginUp, nullptr);
    jdouble* dsMarginDown = env->GetDoubleArrayElements(jdsMarginDown, nullptr);

    int nrl = env->GetArrayLength(jrlVrow);
    int nds = env->GetArrayLength(jdsPhiRow);

    std::string error;
    BatchSolverContext* ctx = nullptr;
    try {
        if (st == nullptr || !*st) {
            throw std::runtime_error("invalid structure handle");
        }
        ctx = createBatchContext(*st, nScenarios, mode,
                                 nrl, rlVrow, rlMinQ, rlMaxQ, rlLoadQ, rlVtarget, rlMonitored, rowSlope,
                                 nds, dsPhiRow, dsFactor, jdsSlackPhiRow, jdsSlackTargetP,
                                 dsBaseTargetP, dsMaxDelta, dsMinDelta,
                                 jdsMode, dsMarginUp, dsMarginDown, jdsSlackMarginUp, jdsSlackMarginDown);
    } catch (const std::exception& e) {
        error = e.what();
    } catch (...) {
        error = "unknown native error";
    }

    env->ReleaseIntArrayElements(jrlVrow, rlVrow, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jrlMinQ, rlMinQ, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jrlMaxQ, rlMaxQ, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jrlLoadQ, rlLoadQ, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jrlVtarget, rlVtarget, JNI_ABORT);
    env->ReleaseByteArrayElements(jrlMonitored, rlMonitored, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jrowSlope, rowSlope, JNI_ABORT);
    env->ReleaseIntArrayElements(jdsPhiRow, dsPhiRow, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jdsFactor, dsFactor, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jdsBaseTargetP, dsBaseTargetP, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jdsMaxDelta, dsMaxDelta, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jdsMinDelta, dsMinDelta, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jdsMarginUp, dsMarginUp, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jdsMarginDown, dsMarginDown, JNI_ABORT);

    if (!error.empty()) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      ("GPU batched context creation failed: " + error).c_str());
        return 0;
    }
    return reinterpret_cast<jlong>(ctx);
}

// OLF_GPU_PHASE_TIME: split one solveBatchContext JNI call into the three things it does —
// marshalling the Java arrays in, the solve itself, and packing the per-scenario result arrays out.
// The Java-side timer sees only the total, and on a 642-scenario chunk the total carries roughly a
// second of fixed cost that the Newton iterations do not explain.
static bool jniPhaseTime() {
    static const bool on = std::getenv("OLF_GPU_PHASE_TIME") != nullptr;
    return on;
}

static long long nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

static long long marshalNs = 0;
static long long solveNs = 0;
static long long packNs = 0;
static long long calls = 0;

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_solveBatchContext(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jx0Base, jdoubleArray jtargetBase,
        jobjectArray jrowMode, jintArray jbaseRowMode, jobjectArray jdisabledMask,
        jintArray jtargetOverrideIndex, jdoubleArray jtargetOverrideValue,
        jintArray jdsOverrideBus, jdoubleArray jdsOverrideFactorDelta, jdoubleArray jdsOverrideBaseDelta,
        jdoubleArray jdsOverrideMaxDelta, jdoubleArray jdsOverrideMinDelta, jdoubleArray jdsAnchorShift,
        jdoubleArray jdsSlackShift, jintArray jrlVtOvSlotIdx, jdoubleArray jrlVtOvVal,
        jintArray jrlQOvSlotEntry, jdoubleArray jrlQOvMinMax,
        jintArray jcvOvSlotEntry, jdoubleArray jcvOvStepScale, jint maxIter, jdouble tol,
        jdouble reactiveLimitTol, jdouble slackDistTol, jdouble convEpsPerEq,
        jint lsMode, jdouble maxDv, jdouble maxDphi, jintArray jvarType, jdoubleArray jx0Full,
        jintArray jShuntOvSlotElem, jdoubleArray jShuntOvGb,
        jintArray jRlStateIn, jintArray jRlStateOut,
        jintArray jR1OvSlotElem, jdoubleArray jR1OvVal,
        jint maxOuterProductive, jint rlMaxSwitch, jint dsRejectResidual, jint skipRlPhase,
        jint nziBase, jintArray jZiOnSlotElem, jintArray jZiOffSlotElem) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        ctx->solveCount++;
    }
    // CONTRACT: reactive-limit entries require OLF's forced-PV election keys. switchPvPq never lets
    // a network reach zero PV buses ("all PV buses should switch PQ, strongest one will stay PV"),
    // and rlForcedPvElectBatch is the only thing on the device that reproduces it. A context that
    // installed RL entries without the keys would pin every candidate and silently diverge from the
    // CPU (measured 1.9 kV on the forced-PV network). GpuBatchOuterLoopTables.install always sets
    // them, so reaching this is a table-installation bug — say so instead of routing the scenario
    // away and hiding it.
    if (ctx != nullptr && ctx->nrl > 0 && !ctx->hasRlElection) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                      "solveBatchContext: reactive-limit entries installed without the forced-PV "
                      "election keys (setBatchRlElection) — OLF's strongest-bus-stays-PV rule "
                      "cannot be reproduced on the device");
        return nullptr;
    }
    long long tMarshal0 = jniPhaseTime() ? nowNs() : 0;
    std::vector<int> r1OvSlotElemVec;
    std::vector<double> r1OvValVec;
    if (jR1OvSlotElem != nullptr) {
        jsize rn = env->GetArrayLength(jR1OvSlotElem);
        r1OvSlotElemVec.resize(rn);
        env->GetIntArrayRegion(jR1OvSlotElem, 0, rn, r1OvSlotElemVec.data());
        r1OvValVec.resize(static_cast<size_t>(rn) * 4);      // full 8-double cbIn row per (slot, elem) entry
        env->GetDoubleArrayRegion(jR1OvVal, 0, rn * 4, r1OvValVec.data());
    }
    std::vector<int> ziOffSlotElemVec;
    if (jZiOffSlotElem != nullptr) {
        jsize zn = env->GetArrayLength(jZiOffSlotElem);
        ziOffSlotElemVec.resize(zn);
        env->GetIntArrayRegion(jZiOffSlotElem, 0, zn, ziOffSlotElemVec.data());
    }
    std::vector<int> ziOnSlotElemVec;
    if (jZiOnSlotElem != nullptr) {
        jsize zn = env->GetArrayLength(jZiOnSlotElem);
        ziOnSlotElemVec.resize(zn);
        env->GetIntArrayRegion(jZiOnSlotElem, 0, zn, ziOnSlotElemVec.data());
    }
    std::vector<int> rlStateInVec;
    if (jRlStateIn != nullptr) {
        jsize rn = env->GetArrayLength(jRlStateIn);
        rlStateInVec.resize(rn);
        env->GetIntArrayRegion(jRlStateIn, 0, rn, rlStateInVec.data());
    }
    std::vector<int> rlStateOutVec;
    if (jRlStateOut != nullptr) {
        rlStateOutVec.resize(env->GetArrayLength(jRlStateOut));
    }
    std::vector<int> shOvSlotElemVec;
    std::vector<double> shOvGbVec;
    if (jShuntOvSlotElem != nullptr) {
        jsize sn = env->GetArrayLength(jShuntOvSlotElem);
        shOvSlotElemVec.resize(sn);
        env->GetIntArrayRegion(jShuntOvSlotElem, 0, sn, shOvSlotElemVec.data());
        shOvGbVec.resize(sn);
        env->GetDoubleArrayRegion(jShuntOvGb, 0, sn, shOvGbVec.data());
    }
    std::vector<double> x0FullVec;
    if (jx0Full != nullptr) {
        jsize fn = env->GetArrayLength(jx0Full);
        x0FullVec.resize(fn);
        env->GetDoubleArrayRegion(jx0Full, 0, fn, x0FullVec.data());
    }
    std::vector<int> varTypeVec;
    if (jvarType != nullptr) {
        jsize vn = env->GetArrayLength(jvarType);
        varTypeVec.resize(vn);
        env->GetIntArrayRegion(jvarType, 0, vn, varTypeVec.data());
    }

    std::string error;
    BatchResult res;
    try {
        if (ctx == nullptr) {
            throw std::runtime_error("invalid batch solver context");
        }
        int n = ctx->n;
        int nS = ctx->nScenarios;
        int nElem = ctx->nElements;
        if (env->GetArrayLength(jx0Base) != n) {
            throw std::runtime_error("base state length mismatch with batch context");
        }
        if (env->GetArrayLength(jdisabledMask) != nS) {
            throw std::runtime_error("scenario count mismatch with batch context");
        }

        if (env->GetArrayLength(jrowMode) != nS) {
            throw std::runtime_error("rowMode scenario count mismatch with batch context");
        }
        // x0/target are the BASE row (length n), identical for every scenario — read once and tiled
        // across nS on-device. rowMode is PER-SCENARIO ([nS][n]) — read all nS rows. Only the disabled
        // mask is also per-scenario.
        std::vector<double> x0(n);
        std::vector<double> target(n);
        env->GetDoubleArrayRegion(jx0Base, 0, n, x0.data());
        env->GetDoubleArrayRegion(jtargetBase, 0, n, target.data());
        // COMPACT row modes: a scenario whose row IS the base array (the Java side shares that one array for
        // every scenario it does not modify) is not copied; only the others travel, with their slots.
        std::vector<int> rowModeSlots;
        std::vector<int> rowMode;
        for (int s = 0; s < nS; s++) {
            auto mRow = (jintArray) env->GetObjectArrayElement(jrowMode, s);
            if (env->IsSameObject(mRow, jbaseRowMode)) {
                env->DeleteLocalRef(mRow);
                continue;
            }
            if (env->GetArrayLength(mRow) != n) {
                env->DeleteLocalRef(mRow);
                throw std::runtime_error("rowMode row length mismatch with batch context");
            }
            rowModeSlots.push_back(s);
            rowMode.resize(rowModeSlots.size() * (size_t) n);
            env->GetIntArrayRegion(mRow, 0, n, rowMode.data() + (rowModeSlots.size() - 1) * (size_t) n);
            env->DeleteLocalRef(mRow);
        }
        std::vector<int> baseRowMode(n);                     // contingency-free base row mode (length n)
        env->GetIntArrayRegion(jbaseRowMode, 0, n, baseRowMode.data());
        // One region copy per scenario: a jboolean is an unsigned byte holding 0 or 1, the mask's own encoding.
        std::unique_ptr<char[]> disabledBuf(new char[static_cast<size_t>(nElem) * nS]);
        char* disabledData = disabledBuf.get();
        for (int s = 0; s < nS; s++) {
            auto dRow = (jbooleanArray) env->GetObjectArrayElement(jdisabledMask, s);
            int dLen = env->GetArrayLength(dRow);
            int len = dLen < nElem ? dLen : nElem;
            char* dst = disabledData + (size_t) s * nElem;
            env->GetBooleanArrayRegion(dRow, 0, len, reinterpret_cast<jboolean*>(dst));
            if (len < nElem) {
                std::memset(dst + len, 0, nElem - len);
            }
            env->DeleteLocalRef(dRow);
        }

        int nOv = env->GetArrayLength(jtargetOverrideIndex);
        std::vector<int> ovIndex(nOv);
        std::vector<double> ovValue(nOv);
        if (nOv > 0) {
            env->GetIntArrayRegion(jtargetOverrideIndex, 0, nOv, ovIndex.data());
            env->GetDoubleArrayRegion(jtargetOverrideValue, 0, nOv, ovValue.data());
        }

        // Per-scenario distributed-slack reweight: present (length nS) only when a generator-loss scenario
        // sits on a DS-participant bus; empty array → no DS override (pass nullptr to the solver).
        int nDsOv = env->GetArrayLength(jdsOverrideBus);
        std::vector<int> dsOvBus(nDsOv);
        std::vector<double> dsOvFactorDelta(nDsOv), dsOvBaseDelta(nDsOv), dsOvMaxDelta(nDsOv), dsOvMinDelta(nDsOv);
        if (nDsOv > 0) {
            env->GetIntArrayRegion(jdsOverrideBus, 0, nDsOv, dsOvBus.data());
            env->GetDoubleArrayRegion(jdsOverrideFactorDelta, 0, nDsOv, dsOvFactorDelta.data());
            env->GetDoubleArrayRegion(jdsOverrideBaseDelta, 0, nDsOv, dsOvBaseDelta.data());
            env->GetDoubleArrayRegion(jdsOverrideMaxDelta, 0, nDsOv, dsOvMaxDelta.data());
            env->GetDoubleArrayRegion(jdsOverrideMinDelta, 0, nDsOv, dsOvMinDelta.data());
        }
        std::vector<double> dsAnchorVec;
        if (jdsAnchorShift != nullptr) {
            dsAnchorVec.resize(env->GetArrayLength(jdsAnchorShift));
            env->GetDoubleArrayRegion(jdsAnchorShift, 0, static_cast<jsize>(dsAnchorVec.size()), dsAnchorVec.data());
        }
        std::vector<double> dsSlackShiftVec;
        if (jdsSlackShift != nullptr) {
            dsSlackShiftVec.resize(env->GetArrayLength(jdsSlackShift));
            env->GetDoubleArrayRegion(jdsSlackShift, 0, static_cast<jsize>(dsSlackShiftVec.size()), dsSlackShiftVec.data());
        }
        std::vector<int> rlVtOvSlotIdxVec;
        std::vector<double> rlVtOvValVec;
        std::vector<int> rlQOvSlotEntryVec;
        std::vector<double> rlQOvMinMaxVec;
        if (jrlQOvSlotEntry != nullptr && jrlQOvMinMax != nullptr) {
            rlQOvSlotEntryVec.resize(env->GetArrayLength(jrlQOvSlotEntry));
            env->GetIntArrayRegion(jrlQOvSlotEntry, 0, static_cast<jsize>(rlQOvSlotEntryVec.size()), rlQOvSlotEntryVec.data());
            rlQOvMinMaxVec.resize(env->GetArrayLength(jrlQOvMinMax));
            env->GetDoubleArrayRegion(jrlQOvMinMax, 0, static_cast<jsize>(rlQOvMinMaxVec.size()), rlQOvMinMaxVec.data());
        }
        std::vector<int> cvOvSlotEntryVec;
        std::vector<double> cvOvStepScaleVec;
        if (jcvOvSlotEntry != nullptr && jcvOvStepScale != nullptr) {
            cvOvSlotEntryVec.resize(env->GetArrayLength(jcvOvSlotEntry));
            env->GetIntArrayRegion(jcvOvSlotEntry, 0, static_cast<jsize>(cvOvSlotEntryVec.size()), cvOvSlotEntryVec.data());
            cvOvStepScaleVec.resize(env->GetArrayLength(jcvOvStepScale));
            env->GetDoubleArrayRegion(jcvOvStepScale, 0, static_cast<jsize>(cvOvStepScaleVec.size()), cvOvStepScaleVec.data());
        }
        if (jrlVtOvSlotIdx != nullptr && jrlVtOvVal != nullptr) {
            rlVtOvSlotIdxVec.resize(env->GetArrayLength(jrlVtOvSlotIdx));
            env->GetIntArrayRegion(jrlVtOvSlotIdx, 0, static_cast<jsize>(rlVtOvSlotIdxVec.size()), rlVtOvSlotIdxVec.data());
            rlVtOvValVec.resize(env->GetArrayLength(jrlVtOvVal));
            env->GetDoubleArrayRegion(jrlVtOvVal, 0, static_cast<jsize>(rlVtOvValVec.size()), rlVtOvValVec.data());
        }

        if (jniPhaseTime()) {
            marshalNs += nowNs() - tMarshal0;
        }
        long long tSolve0 = jniPhaseTime() ? nowNs() : 0;
        static const int kNoModeSlots[1] = {0};              // non-null: COMPACT mode even when no scenario differs
        res = solveBatchContext(ctx, x0.data(), target.data(), rowMode.data(),
                                rowModeSlots.empty() ? kNoModeSlots : rowModeSlots.data(),
                                static_cast<int>(rowModeSlots.size()), baseRowMode.data(), disabledData,
                                nOv, ovIndex.data(), ovValue.data(),
                                nDsOv > 0 ? dsOvBus.data() : nullptr, dsOvFactorDelta.data(),
                                dsOvBaseDelta.data(), dsOvMaxDelta.data(), dsOvMinDelta.data(),
                                dsAnchorVec.empty() ? nullptr : dsAnchorVec.data(),
                                dsSlackShiftVec.empty() ? nullptr : dsSlackShiftVec.data(),
                                static_cast<int>(rlVtOvValVec.size()),
                                rlVtOvSlotIdxVec.empty() ? nullptr : rlVtOvSlotIdxVec.data(),
                                rlVtOvValVec.empty() ? nullptr : rlVtOvValVec.data(),
                                static_cast<int>(rlQOvMinMaxVec.size() / 5),
                                rlQOvSlotEntryVec.empty() ? nullptr : rlQOvSlotEntryVec.data(),
                                rlQOvMinMaxVec.empty() ? nullptr : rlQOvMinMaxVec.data(),
                                static_cast<int>(cvOvStepScaleVec.size() / 2),
                                cvOvSlotEntryVec.empty() ? nullptr : cvOvSlotEntryVec.data(),
                                cvOvStepScaleVec.empty() ? nullptr : cvOvStepScaleVec.data(), maxIter, tol,
                                reactiveLimitTol, slackDistTol, convEpsPerEq,
                                lsMode, maxDv, maxDphi, varTypeVec.empty() ? nullptr : varTypeVec.data(),
                                x0FullVec.empty() ? nullptr : x0FullVec.data(),
                                static_cast<int>(shOvSlotElemVec.size() / 2),
                                shOvSlotElemVec.empty() ? nullptr : shOvSlotElemVec.data(),
                                shOvGbVec.empty() ? nullptr : shOvGbVec.data(),
                                rlStateInVec.empty() ? nullptr : rlStateInVec.data(),
                                rlStateOutVec.empty() ? nullptr : rlStateOutVec.data(),
                                static_cast<int>(r1OvSlotElemVec.size() / 2),
                                r1OvSlotElemVec.empty() ? nullptr : r1OvSlotElemVec.data(),
                                r1OvValVec.empty() ? nullptr : r1OvValVec.data(),
                                maxOuterProductive, rlMaxSwitch, dsRejectResidual, skipRlPhase,
                                nziBase, static_cast<int>(ziOnSlotElemVec.size() / 2),
                                ziOnSlotElemVec.empty() ? nullptr : ziOnSlotElemVec.data(),
                                static_cast<int>(ziOffSlotElemVec.size() / 2),
                                ziOffSlotElemVec.empty() ? nullptr : ziOffSlotElemVec.data());
        if (jniPhaseTime()) {
            solveNs += nowNs() - tSolve0;
        }
        if (jRlStateOut != nullptr && !rlStateOutVec.empty()) {
            env->SetIntArrayRegion(jRlStateOut, 0, static_cast<jsize>(rlStateOutVec.size()), rlStateOutVec.data());
        }
    } catch (const std::exception& e) {
        error = e.what();
    } catch (...) {
        error = "unknown native error";
    }

    if (!error.empty()) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      ("GPU batched Newton-Raphson failed: " + error).c_str());
        return nullptr;
    }

    // Pack per-scenario results: [state(n), iterations, finalMismatch, convergenceStatus].
    long long tPack0 = jniPhaseTime() ? nowNs() : 0;
    int n = ctx->n;
    int nS = ctx->nScenarios;
    jclass doubleArrayClass = env->FindClass("[D");
    // A null from either allocation means the Java HEAP is exhausted and an OutOfMemoryError is pending: return
    // and let it surface in Java. Writing into the null row instead was a SIGSEGV in JNIHandles::resolve that
    // killed the JVM - network.xiidm's full busbar scope, the robust retry packing its results on top of the
    // ~14000 precomputed states.
    jobjectArray out = env->NewObjectArray(nS, doubleArrayClass, nullptr);
    if (out == nullptr) {
        return nullptr;
    }
    for (int s = 0; s < nS; s++) {
        jdoubleArray row = env->NewDoubleArray(n + 3);
        if (row == nullptr) {
            return nullptr;
        }
        env->SetDoubleArrayRegion(row, 0, n, res.x.data() + (size_t) s * n);
        double tail[3] = {static_cast<double>(res.iters[s]), res.norm[s], static_cast<double>(res.status[s])};
        env->SetDoubleArrayRegion(row, n, 3, tail);
        env->SetObjectArrayElement(out, s, row);
        env->DeleteLocalRef(row);
    }
    if (jniPhaseTime()) {
        packNs += nowNs() - tPack0;
        calls++;
        fprintf(stderr, "JNI_PHASE_TIME calls=%lld marshal=%lld ms solve=%lld ms pack=%lld ms "
                "| outerPasses=%lld (solving=%lld) newtonIters=%lld activeIterSum=%lld widthIterSum=%lld compactIters=%lld"
                " | cuDSS analyses main=%lld compaction=%lld | irChecks=%ld irExtraSolves=%ld\n",
                calls, marshalNs / 1000000, solveNs / 1000000, packNs / 1000000,
                gOuterPasses, gSolvePasses, gNewtonIters, gActiveIterSum, gWidthIterSum, gCompactIters,
                gAnalysesMain, gAnalysesCompact, gIrChecks, gIrExtraSteps);
        double tot[4] = {0, 0, 0, 0};
        long long cnt[4] = {0, 0, 0, 0};
        for (const auto& kv : gAnalysisSites) {
            int ph = kv.first / 100000, line = kv.first % 100000;
            int k = ph == 'A' ? 0 : ph == 'F' ? 1 : ph == 'S' ? 2 : 3;
            tot[k] += kv.second.second;
            cnt[k] += kv.second.first;
            fprintf(stderr, "CUDSS_PHASE_SITE %c line=%d count=%lld ms=%.1f\n", ph, line, kv.second.first, kv.second.second);
        }
        for (int k = 1; k < 16; k++) {
            if (gOlHistCount[k] == 0) continue;
            fprintf(stderr, "GPU_OL_HIST passes=%d%s scenarios=%lld firstNrIters=%lld totalNrIters=%lld\n", k,
                    k == 15 ? "+" : "", gOlHistCount[k], gOlHistFirst[k], gOlHistTotal[k]);
        }
        fprintf(stderr, "GPU_OL_HIST notConvergedOnDevice=%lld iters=%lld\n", gOlNotConverged, gOlNotConvergedIters);
        fprintf(stderr, "GPU_OL_TIME first-solve passes=%lld ms=%.1f scenarioSolves=%lld laneIters=%lld widthIters=%lld\n",
                gOlFirstPasses, gOlFirstSolveMs, gOlFirstSolveScen, gOlFirstLaneIters, gOlFirstWidthIters);
        for (int k = 0; k < OL_KINDS; k++) {
            if (gOlReSolvePasses[k] > 0) {
                fprintf(stderr, "GPU_OL_TIME re-solve trigger=%s passes=%lld ms=%.1f scenarioSolves=%lld laneIters=%lld widthIters=%lld\n",
                        olKindName(k), gOlReSolvePasses[k], gOlReSolveMs[k], gOlReSolveScen[k], gOlReSolveLaneIters[k], gOlReSolveWidthIters[k]);
            }
        }
        for (int k = 0; k < OL_KINDS; k++) {
            if (gOlCheckPasses[k] > 0) {
                fprintf(stderr, "GPU_OL_TIME check loop=%s passes=%lld ms=%.1f\n", olKindName(k), gOlCheckPasses[k], gOlCheckMs[k]);
            }
        }
        if (rlReasonsOn()) {
            static const char* names[8] = {"group re-stamp only", "pin", "unpin", "pin+unpin", "limit refresh",
                                           "pin+refresh", "unpin+refresh", "pin+unpin+refresh"};
            fprintf(stderr, "GPU_RL_REASONS stable checks=%lld | entries: pins=%lld unpins=%lld refreshes=%lld\n",
                    gRlStableScen, gRlEntryCodes[1], gRlEntryCodes[2], gRlEntryCodes[3]);
            for (int m = 0; m < 8; m++) {
                if (gRlCheckScen[m] > 0) fprintf(stderr, "GPU_RL_REASONS unstable because %s: %lld\n", names[m], gRlCheckScen[m]);
            }
        }
        fprintf(stderr, "CUDSS_PHASE_TIME analysis=%lld x %.1f ms | factorization=%lld x %.1f ms | solve=%lld x %.1f ms"
                " | factorization+solve=%lld x %.1f ms\n", cnt[0], tot[0], cnt[1], tot[1], cnt[2], tot[2], cnt[3], tot[3]);
    }
    return out;
}

// Read back the batch context's per-scenario TARGET buffer (n * nScenarios doubles) after a solve —
// the device outer loops (distributed slack) mutate targets in place, and the strategy/pair phase needs
// the PARENT contingency's final distributed targets to re-anchor its own distribution (OLF's
// setGeneratorsInitialTargetPToTargetP semantics).
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchTargets(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return nullptr;
    }
    size_t nTot = static_cast<size_t>(ctx->n) * ctx->nScenarios;
    std::vector<double> host(nTot);
    cudaMemcpy(host.data(), ctx->dTarget.p, nTot * sizeof(double), cudaMemcpyDeviceToHost);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(nTot));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(nTot), host.data());
    return out;
}

// The per-scenario ROW MODES as the device ENDED with them, the companion to readBatchTargets.
// A row's equation is (mode, target) together, and the outer-loop kernels rewrite BOTH: the host's
// rowModePerScenario is only what SETUP asked for, so reading a target without its final mode can
// say "the target is right" about a row that is no longer the equation it was set up to be. That is
// the whole failure shape of the device-target-writer class, where a wrong row converges cleanly and
// nothing flags it.
extern "C" JNIEXPORT jintArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchRowModes(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return nullptr;
    }
    size_t nTot = static_cast<size_t>(ctx->n) * ctx->nScenarios;
    std::vector<int> host(nTot);
    cudaMemcpy(host.data(), ctx->dMode.p, nTot * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(nTot));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(nTot), host.data());
    return out;
}

// Batched SENSITIVITY solve for the incremental outer-loop family (device port of OLF's
// SensitivityContext): J at the given per-scenario states (the same forward system the Newton
// solves — OLF's "solveTransposed" IS the forward direction in powsybl-math's transposed
// storage), factorized once, solved for a unit RHS at each controller's pinned-variable
// identity row (BRANCH_TARGET_RHO1 & co, storage row = rowMap[eqRow]); the requested variable
// columns of each solution are gathered per scenario. MUST be called on a context that has just
// completed solveBatchContext with the SAME per-scenario overrides/row modes — the fill reads
// them from the live device buffers. UBATCH mode (3) only.
std::vector<double> solveBatchSensitivities(BatchSolverContext* ctx, const double* states,
                                            int k, const int* unitEqRows, int m, const int* observeVarCols) {
    if (ctx->mode != 3) {
        throw std::runtime_error("solveBatchSensitivities: UBATCH mode (3) only");
    }
    int n = ctx->st->n;
    int nS = ctx->nScenarios;
    int nnz = ctx->st->nnz;

    CUDA_CHECK(cudaMemcpyAsync(ctx->dX.p, states, (size_t) n * nS * sizeof(double), cudaMemcpyHostToDevice, ctx->stream));
    std::vector<char> allActive(nS, 1);
    CUDA_CHECK(cudaMemcpyAsync(ctx->dActive.p, allActive.data(), nS * sizeof(char), cudaMemcpyHostToDevice, ctx->stream));

    // ---- J fill at the converged states (same launches as the solve loop's full fill) ----
    CUDA_CHECK(cudaMemsetAsync(ctx->dVals.p, 0, (size_t) nnz * nS * sizeof(double), ctx->stream));
    initFBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, ctx->dTarget.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>());
    if (ctx->st->cb.ne) {
        fillClosedBranchBatch<<<grid(ctx->st->cb.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->cb.ne, nS, n, nnz, ctx->nElements, ctx->cbOffset, ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->st->cb.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(), ctx->st->cb.altDest.as<int>(), ctx->st->cb.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->cb.dynReloc ? 1 : 0, ctx->st->cb.phaseCtl ? ctx->st->cb.pcDest.as<int>() : nullptr, ctx->st->cb.phaseCtl ? ctx->st->cb.pcSel.as<int>() : nullptr, withPilotSelf(relocOf(ctx->st->cb, ctx->st.get()), ctx));
    }
    if (ctx->st->ob.ne) {
        fillOpenBranchBatch<<<grid(ctx->st->ob.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->ob.ne, nS, n, nnz, ctx->nElements, ctx->obOffset, ctx->st->ob.in.as<double>(), ctx->st->ob.ix.as<int>(), ctx->st->ob.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->st->ob.altDest.as<int>(), ctx->st->ob.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->ob.dynReloc ? 1 : 0, withPilotSelf(relocOf(ctx->st->ob, ctx->st.get()), ctx));
    }
    if (ctx->st->sh.ne) {
        fillShuntBatch<<<grid(ctx->st->sh.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->sh.ne, nS, n, nnz, ctx->nElements, ctx->shOffset, ctx->st->sh.in.as<double>(), ctx->st->sh.ix.as<int>(), ctx->st->sh.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nShOv, ctx->dShOvSlotElem.as<int>(), ctx->dShOvGb.as<double>(), ctx->st->sh.altDest.as<int>(), ctx->st->sh.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->sh.dynReloc ? 1 : 0, withPilotSelf(relocOf(ctx->st->sh, ctx->st.get()), ctx));
    }
    if (ctx->st->nvdl) {
        fillVdlBatch<<<grid(ctx->st->nvdl * nS), BLOCK, 0, ctx->stream>>>(ctx->st->nvdl, nS, n, nnz, ctx->st->dVdlIn.as<double>(), ctx->st->dVdlIdx.as<int>(), ctx->st->dVdlDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->st->vdlDyn ? ctx->st->dVdlAltRow.as<int>() : nullptr, ctx->st->vdlDyn ? ctx->st->dVdlAltDest.as<int>() : nullptr, ctx->st->vdlDyn ? ctx->dMode.as<int>() : nullptr);
    }
    if (ctx->st->hv.ne) {
        fillHvdcBatch<<<grid(ctx->st->hv.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->hv.ne, nS, n, nnz, ctx->nElements, ctx->hvOffset, ctx->st->hv.in.as<double>(), ctx->st->hv.ix.as<int>(), ctx->st->hv.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->st->hv.altDest.as<int>(), ctx->st->hv.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->hv.dynReloc ? 1 : 0);
    }
    int dqTraceRow = gpuDqTraceRow();
    if (ctx->hasPilotSelf && ctx->dPsSkip.p != nullptr && ctx->st->nQr > 0) {
        // Recompute the PILOT-SELF gate from the CURRENT pin states: such a pilot has a Q balance to
        // relocate only once a reactive limit has pinned it, and the RL pass moves pins between passes.
        pilotSelfGateBatch<<<grid(ctx->st->nQr * nS), BLOCK, 0, ctx->stream>>>(
                ctx->st->nQr, nS, ctx->nrl, ctx->dPsPilotEntry.as<int>(),
                ctx->dRlLimitType.as<char>(),
                ctx->nRlIslanded > 0 ? ctx->dRlIslanded.as<char>() : nullptr, ctx->dPsSkip.as<char>());
    }
    if (ctx->ndq) fillDistrQBatch<<<grid(ctx->ndq * nS), BLOCK, 0, ctx->stream>>>(ctx->ndq, nS, n, nnz, ctx->nElements, ctx->cbOffset, ctx->obOffset, ctx->st->dDqElem.as<int>(), ctx->st->dDqSide.as<int>(), ctx->st->dDqKind.as<int>(), ctx->st->dDqWeight.as<double>(), ctx->st->dDqDest.as<int>(), ctx->st->dDqAlt.as<int>(), ctx->st->dDqSel.as<int>(), ctx->st->dDqZiRow.as<int>(), ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->st->sh.in.as<double>(), ctx->st->sh.ix.as<int>(), ctx->st->ob.in.as<double>(), ctx->st->ob.ix.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nDqOv, ctx->nDqOv ? ctx->dDqOvIdx.as<int>() : nullptr, ctx->nDqOv ? ctx->dDqOvW.as<double>() : nullptr,
                ctx->nDqKeyOv, ctx->nDqKeyOv ? ctx->dDqKeyOvIdx.as<int>() : nullptr, ctx->nDqKeyOv ? ctx->dDqKeyOvKey.as<double>() : nullptr, ctx->dMode.as<int>(), ctx->nDqInact,
                ctx->nDqInact ? ctx->dDqInact.as<int>() : nullptr,
                ctx->nDqDyn ? ctx->dDqKMember.as<int>() : nullptr, ctx->nDqDyn || ctx->hasDqQrGroups ? ctx->dDqJMember.as<int>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberKey.as<double>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberRlEntry.as<int>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberGroupStart.as<int>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberGroupCount.as<int>() : nullptr, ctx->dRlLimitType.as<char>(), ctx->nrl, ctx->hasPilotSelf && ctx->dPsSkip.p != nullptr ? ctx->dPsSkip.as<char>() : nullptr, ctx->st->nQr, ctx->hasDqQrGroups ? ctx->dDqMemberQrGroup.as<int>() : nullptr, ctx->st->nQr > 0 ? ctx->st->dQrPtr.as<int>() : nullptr, ctx->st->nQr > 0 ? ctx->st->dQrRows.as<int>() : nullptr, ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(), ctx->nShOv, ctx->dShOvSlotElem.as<int>(), ctx->dShOvGb.as<double>(), dqTraceRow,
                    ctx->hasDqSub ? ctx->dDqSub.as<char>() : nullptr, ctx->hasDqSub ? ctx->dQrMemberStart.as<int>() : nullptr, ctx->nDqMembers);
    if (ctx->ndr) fillDistrRhoBatch<<<grid(ctx->ndr * nS), BLOCK, 0, ctx->stream>>>(ctx->ndr, nS, n, nnz, ctx->st->dDrCol.as<int>(), ctx->st->dDrCoef.as<double>(), ctx->st->dDrDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->nDrOv, ctx->nDrOv ? ctx->dDrOvIdx.as<int>() : nullptr, ctx->nDrOv ? ctx->dDrOvC.as<double>() : nullptr, ctx->st->drQr ? ctx->st->dDrQrGroup.as<int>() : nullptr, ctx->dMode.as<int>(), ctx->st->dQrPtr.as<int>(), ctx->st->dQrRows.as<int>());
    if (ctx->ndb) fillDistrShuntBBatch<<<grid(ctx->ndb * nS), BLOCK, 0, ctx->stream>>>(ctx->ndb, nS, n, nnz, ctx->st->dDbCol.as<int>(), ctx->st->dDbCoef.as<double>(), ctx->st->dDbDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>());
    if (ctx->nzi) fillZeroImpBatch<<<grid(ctx->nzi * nS), BLOCK, 0, ctx->stream>>>(ctx->nzi, nS, n, nnz, ctx->st->dZiIn.as<double>(), ctx->st->dZiIdx.as<int>(), ctx->st->dZiDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->nziBase, ctx->nZiOn, ctx->dZiOnSlotElem.as<int>(), ctx->nZiOff, ctx->dZiOffSlotElem.as<int>(), ctx->st->dZiAltDest.as<int>(), ctx->st->dZiSelRow.as<int>(), ctx->dMode.as<int>(), ctx->st->ziDyn ? 1 : 0, withPilotSelf(ziRelocOf(ctx->st.get()), ctx), ctx->nZiTree, ctx->nZiTree ? ctx->dZiTreeSlotElemVal.as<int>() : nullptr, ctx->hasZiMasks ? ctx->dZiOffMask.as<char>() : nullptr, ctx->hasZiMasks ? ctx->dZiOnMask.as<char>() : nullptr, ctx->hasZiMasks ? ctx->dZiTreeMask.as<char>() : nullptr);
    applyRowModeBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, nnz, ctx->dMode.as<int>(), ctx->st->dRowPtr.as<int>(), ctx->st->dDiagPos.as<int>(), ctx->dX.as<double>(), ctx->dTarget.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->dRowSlope.as<double>(),
                    ctx->st->dColIdx.as<int>(), ctx->st->dRelocOwn.p != nullptr ? ctx->st->dRelocOwn.as<int>() : nullptr);
    hazRepairValues(ctx, nS, ctx->dVals.as<double>());

    if (!ctx->analyzed) {
        gAnalysesMain++;
        { AnalysisTimer at_(__LINE__); CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_ANALYSIS, ctx->ds.config, ctx->ds.data, ctx->matUb, ctx->matUbX, ctx->matUbB)); }
        ctx->analyzed = true;
    }
    DeviceBuf dObs, dOut;
    dObs.upload(observeVarCols, m * sizeof(int));
    dOut.alloc((size_t) nS * k * m * sizeof(double));
    for (int ki = 0; ki < k; ki++) {
        int row = ctx->st->rowMap[unitEqRows[ki]];           // equation row -> CSR storage row
        if (row < 0) {
            throw std::runtime_error("solveBatchSensitivities: unit row not an active equation");
        }
        setUnitRhsBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, row, ctx->dB.as<double>());
        hazRepairRightHandSide(ctx, nS, ctx->dB.as<double>());
        int phase = CUDSS_PHASE_SOLVE | (ki == 0 ? CUDSS_PHASE_FACTORIZATION : 0);
        { AnalysisTimer at_(__LINE__, ((phase) & (CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_REFACTORIZATION)) ? 'G' : ((phase) & CUDSS_PHASE_ANALYSIS) ? 'A' : 'S'); CUDSS_CHECK(cudssExecute(ctx->ds.handle, phase, ctx->ds.config, ctx->ds.data, ctx->matUb, ctx->matUbX, ctx->matUbB)); }
        gatherSensiBatch<<<grid(nS * m), BLOCK, 0, ctx->stream>>>(n, nS, k, m, ki, dObs.as<int>(),
                ctx->dDx.as<double>(), dOut.as<double>());
    }
    std::vector<double> out((size_t) nS * k * m);
    CUDA_CHECK(cudaMemcpyAsync(out.data(), dOut.p, out.size() * sizeof(double), cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    return out;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_solveBatchSensitivities(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jstates, jintArray junitRows, jintArray jobserveCols) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return nullptr;
    }
    std::vector<double> states(env->GetArrayLength(jstates));
    env->GetDoubleArrayRegion(jstates, 0, (jsize) states.size(), states.data());
    std::vector<int> unitRows(env->GetArrayLength(junitRows));
    env->GetIntArrayRegion(junitRows, 0, (jsize) unitRows.size(), unitRows.data());
    std::vector<int> observeCols(env->GetArrayLength(jobserveCols));
    env->GetIntArrayRegion(jobserveCols, 0, (jsize) observeCols.size(), observeCols.data());
    std::string error;
    std::vector<double> out;
    try {
        if (states.size() != (size_t) ctx->st->n * ctx->nScenarios) {
            throw std::runtime_error("states size mismatch");
        }
        out = solveBatchSensitivities(ctx, states.data(),
                (int) unitRows.size(), unitRows.data(), (int) observeCols.size(), observeCols.data());
    } catch (const std::exception& e) {
        error = e.what();
    }
    if (!error.empty()) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      ("GPU batched sensitivity solve failed: " + error).c_str());
        return nullptr;
    }
    jdoubleArray jout = env->NewDoubleArray((jsize) out.size());
    env->SetDoubleArrayRegion(jout, 0, (jsize) out.size(), out.data());
    return jout;
}

// Per-scenario slack-only-DS absorbed totals after a solve (see slackOnlyDsCheck) — the pair phase
// seeds its scenarios with the parent contingency's value (the device analogue of OLF's
// setGeneratorsInitialTargetPToTargetP before a strategy re-solve). Null when not slack-only.
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchSlackAbsorbed(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || !ctx->dsSlackOnly) {
        return nullptr;
    }
    std::vector<double> host(ctx->nScenarios);
    cudaMemcpy(host.data(), ctx->dDsSlackAbsorbed.p, ctx->nScenarios * sizeof(double), cudaMemcpyDeviceToHost);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(ctx->nScenarios));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(ctx->nScenarios), host.data());
    return out;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchDsGenCum(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->ngen <= 0 || !ctx->dGenCum.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nScenarios) * ctx->ngen;
    std::vector<double> host(cnt);
    cudaMemcpy(host.data(), ctx->dGenCum.p, cnt * sizeof(double), cudaMemcpyDeviceToHost);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(cnt));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}

// Per scenario of the last solveBatchContext call (and its host-loop resumes): {solve passes, first pass's
// Newton iterations, total Newton iterations}, flattened s*3. The per-contingency counterpart of
// GPU_OL_HIST, for matching OLF's CPU_OL_STATS contingency by contingency.
extern "C" JNIEXPORT jintArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchOlStats(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->olPasses.size() != static_cast<size_t>(ctx->nScenarios)) {
        return nullptr;
    }
    std::vector<jint> host(static_cast<size_t>(ctx->nScenarios) * 3);
    for (int s = 0; s < ctx->nScenarios; s++) {
        host[s * 3] = ctx->olPasses[s];
        host[s * 3 + 1] = ctx->olFirst[s];
        host[s * 3 + 2] = ctx->olIters[s];
    }
    jintArray out = env->NewIntArray(static_cast<jsize>(host.size()));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(host.size()), host.data());
    return out;
}

// How many solveBatchContext calls this context has run. The per-solve DS read-backs (retain, genCum,
// slack absorbed) are reset at every solve, so a caller summing them must know they span its solve.
extern "C" JNIEXPORT jlong JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchSolveCount(
        JNIEnv*, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    return ctx == nullptr ? -1 : static_cast<jlong>(ctx->solveCount);
}

// One scenario's state at its last converged inner solve of the last solve (its starting state if none converged).
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchLastConvergedState(
        JNIEnv* env, jclass, jlong handle, jint slot) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || !ctx->dXLastConv.p || slot < 0 || slot >= ctx->nScenarios) {
        return nullptr;
    }
    int n = ctx->st->n;
    std::vector<double> host(n);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(cudaMemcpy(host.data(), ctx->dXLastConv.as<double>() + (size_t) slot * n, n * sizeof(double),
                          cudaMemcpyDeviceToHost));
    jdoubleArray out = env->NewDoubleArray(n);
    env->SetDoubleArrayRegion(out, 0, n, host.data());
    return out;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchDsRetain(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || !ctx->dDsRetainApplied.p) {
        return nullptr;
    }
    std::vector<double> host(ctx->nScenarios);
    cudaMemcpy(host.data(), ctx->dDsRetainApplied.p, ctx->nScenarios * sizeof(double), cudaMemcpyDeviceToHost);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(ctx->nScenarios));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(ctx->nScenarios), host.data());
    return out;
}

// LINE_SEARCH backtracking parameters for a batch context (OLF LineSearchStateVectorScaling's
// maxIteration and stepFold): consumed by solves called with lsMode == 1. Context-scoped like
// setBatchRowCriteria, so the solveBatchContext signature stays unchanged.
// Per-scenario starting values for the next solve given no full per-scenario state: flat index s*n+row -> value,
// written over the base state tiled on the device. Replaces sending nScenarios * n doubles when only a few rows
// differ (the frozen rows of islanded buses). Null or empty arrays clear them.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchX0Seeds(
        JNIEnv* env, jclass, jlong handle, jintArray jIdx, jdoubleArray jVal) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->x0SeedIdx.clear();
    ctx->x0SeedVal.clear();
    if (jIdx == nullptr || jVal == nullptr) {
        return;
    }
    jsize k = env->GetArrayLength(jIdx);
    ctx->x0SeedIdx.resize(k);
    ctx->x0SeedVal.resize(k);
    env->GetIntArrayRegion(jIdx, 0, k, ctx->x0SeedIdx.data());
    env->GetDoubleArrayRegion(jVal, 0, k, ctx->x0SeedVal.data());
}

// The linear-solver settings of the run (GpuLoadFlowParameters): read by every context built after this call and by
// the adaptive refinement of every solve. -1 reorder / pivotType = AUTO; NaN threshold = the library's.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setLinearSolverSettings(
        JNIEnv*, jclass, jint reorder, jint matching, jint pivotType, jdouble pivotThreshold,
        jint irBase, jint irMax, jdouble irTol) {
    gLinearSolver.reorder = reorder;
    gLinearSolver.matching = matching;
    gLinearSolver.pivotType = pivotType;
    gLinearSolver.pivotThreshold = pivotThreshold;
    gLinearSolver.irBase = irBase;
    gLinearSolver.irMax = irMax;
    gLinearSolver.irTol = irTol;
}

// The installed linear-solver settings, in setLinearSolverSettings's order (tests: the parameters reach the device).
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readLinearSolverSettings(JNIEnv* env, jclass) {
    const double v[7] = {static_cast<double>(gLinearSolver.reorder), static_cast<double>(gLinearSolver.matching),
                         static_cast<double>(gLinearSolver.pivotType), gLinearSolver.pivotThreshold,
                         static_cast<double>(gLinearSolver.irBase), static_cast<double>(gLinearSolver.irMax),
                         gLinearSolver.irTol};
    jdoubleArray out = env->NewDoubleArray(7);
    env->SetDoubleArrayRegion(out, 0, 7, v);
    return out;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchLineSearch(
        JNIEnv*, jclass, jlong handle, jint lsMaxIter, jdouble lsStepFold) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->lsMaxIter = lsMaxIter > 0 ? lsMaxIter : 10;
    ctx->lsStepFold = lsStepFold > 1.0 ? lsStepFold : 4.0 / 3.0;
}

// PER_EQUATION_TYPE stopping criterion: install (or clear, with nulls) the per-row threshold
// arrays on a batch context — thrIdent for identity/slope rows, thrPower for power rows, both
// length n. The convergence test then requires EVERY row's |F| under its threshold (OLF's
// PerEquationTypeStoppingCriteria), replacing the 2-norm/infnorm rule for this context.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRowCriteria(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jThrIdent, jdoubleArray jThrPower) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jThrIdent == nullptr || jThrPower == nullptr) {
        ctx->perRowCriteria = false;
        ctx->perRowAlt = false;
        return;
    }
    int n = ctx->st->n;
    jdouble* thrIdent = env->GetDoubleArrayElements(jThrIdent, nullptr);
    jdouble* thrPower = env->GetDoubleArrayElements(jThrPower, nullptr);
    ctx->dThrIdent.upload(thrIdent, static_cast<size_t>(n) * sizeof(double));
    ctx->dThrPower.upload(thrPower, static_cast<size_t>(n) * sizeof(double));
    env->ReleaseDoubleArrayElements(jThrIdent, thrIdent, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jThrPower, thrPower, JNI_ABORT);
    ctx->perRowCriteria = true;
}

// The per-row ALTERNATE threshold and its selector row (-1: none) of the PER_EQUATION_TYPE criterion, read by
// residualNormsBatch for a cross identity (mode 6) and for a relocated child's row once its coupler's dummy row
// is an identity. Nulls clear them. Install after setBatchRowCriteria, which clears them too.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRowCriteriaAlt(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jThrAlt, jintArray jThrSel) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jThrAlt == nullptr || jThrSel == nullptr) {
        ctx->perRowAlt = false;
        return;
    }
    int n = ctx->st->n;
    jdouble* thrAlt = env->GetDoubleArrayElements(jThrAlt, nullptr);
    jint* thrSel = env->GetIntArrayElements(jThrSel, nullptr);
    ctx->dThrAlt.upload(thrAlt, static_cast<size_t>(n) * sizeof(double));
    ctx->dThrSel.upload(thrSel, static_cast<size_t>(n) * sizeof(int));
    env->ReleaseDoubleArrayElements(jThrAlt, thrAlt, JNI_ABORT);
    env->ReleaseIntArrayElements(jThrSel, thrSel, JNI_ABORT);
    ctx->perRowAlt = true;
}

// Reactive-capability CURVES at runtime P: install (or clear, with nulls) the per-gen curve channel
// on a batch context — see rlCurveLimitsBatch. cvIdx stride 5 {rlIdx, pRow, ptsOff, npts, dsGenSlot}; cvIn
// stride 5 {basePpu, share, baseRowTargetP, baseEvalMinPu, baseEvalMaxPu}; cvPts stride 3
// {p_MW, minQ_MW, maxQ_MW} sorted by p; cvHv (null when no entry is an AC-emulation VSC) stride 2 {hvdc, side}.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlCurves(
        JNIEnv* env, jclass, jlong handle, jintArray jCvIdx, jdoubleArray jCvIn, jdoubleArray jCvPts,
        jintArray jCvHv, jint extrapolate) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->cvHasHvdc = false;
    if (jCvIdx == nullptr || jCvIn == nullptr || jCvPts == nullptr) {
        ctx->ncv = 0;
        return;
    }
    int ncv = env->GetArrayLength(jCvIdx) / 5;   // stride 5: {rlIdx, pRow, ptsOff, npts, dsGenSlot}
    if (ncv == 0) {
        ctx->ncv = 0;
        return;
    }
    if (jCvHv != nullptr) {                      // stride 2 per entry: {hvdc, side}, -1 for a plain generator
        jint* hv = env->GetIntArrayElements(jCvHv, nullptr);
        ctx->dCvHv.upload(hv, static_cast<size_t>(ncv) * 2 * sizeof(int));
        env->ReleaseIntArrayElements(jCvHv, hv, JNI_ABORT);
        ctx->cvHasHvdc = true;
    }
    jint* cvIdx = env->GetIntArrayElements(jCvIdx, nullptr);
    jdouble* cvIn = env->GetDoubleArrayElements(jCvIn, nullptr);
    jdouble* cvPts = env->GetDoubleArrayElements(jCvPts, nullptr);
    ctx->dCvIdx.upload(cvIdx, static_cast<size_t>(ncv) * 5 * sizeof(int));
    ctx->dCvIn.upload(cvIn, static_cast<size_t>(ncv) * 5 * sizeof(double));
    ctx->dCvPts.upload(cvPts, static_cast<size_t>(env->GetArrayLength(jCvPts)) * sizeof(double));
    // The curves of each reactive-limit entry, CSR (pack order kept within an entry): rlCurveLimitsBatch runs
    // one thread per (entry, scenario) and walked EVERY curve to find its own.
    {
        std::vector<int> ptr(static_cast<size_t>(ctx->nrl) + 1, 0);
        std::vector<int> list(ncv);
        for (int g = 0; g < ncv; g++) {
            int e = cvIdx[g * 5];
            if (e < 0 || e >= ctx->nrl) {
                env->ReleaseIntArrayElements(jCvIdx, cvIdx, JNI_ABORT);
                env->ReleaseDoubleArrayElements(jCvIn, cvIn, JNI_ABORT);
                env->ReleaseDoubleArrayElements(jCvPts, cvPts, JNI_ABORT);
                ctx->ncv = 0;
                env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                              "GPU batch: a curve's reactive-limit entry is outside the pack");
                return;
            }
            ptr[e + 1]++;
        }
        for (int e = 0; e < ctx->nrl; e++) ptr[e + 1] += ptr[e];
        std::vector<int> cursor(ptr.begin(), ptr.end() - 1);
        for (int g = 0; g < ncv; g++) list[cursor[cvIdx[g * 5]]++] = g;
        ctx->dCvPtr.upload(ptr.data(), ptr.size() * sizeof(int));
        ctx->dCvList.upload(list.data(), list.size() * sizeof(int));
    }
    ctx->dRlMinQDyn.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nrl * sizeof(double));
    ctx->dRlMaxQDyn.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nrl * sizeof(double));
    ctx->dRlLoadQDyn.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nrl * sizeof(double));
    env->ReleaseIntArrayElements(jCvIdx, cvIdx, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jCvIn, cvIn, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jCvPts, cvPts, JNI_ABORT);
    ctx->ncv = ncv;
    ctx->cvExtrapolate = extrapolate;
}

// forceTargetQInReactiveLimits channel (GpuBatchForcedTargetQ): fqIdx stride 7 {qRow, dsGenSlot, ptsOff, npts,
// dsBusIdx, baseQRowMode, rlIdx}; fqIn stride 5 {basePpu, rawTargetQPu, baseClampedPu, share, baseRowTargetP}; fqPts
// stride 3 {p_MW, minQ_MW, maxQ_MW} sorted by p. Null clears the channel (a cached context reused by a run
// without it).
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchForcedTargetQ(
        JNIEnv* env, jclass, jlong handle, jintArray jIdx, jdoubleArray jIn, jdoubleArray jPts, jint extrapolate) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->fqHasPre = false;
    if (jIdx == nullptr || jIn == nullptr || jPts == nullptr || env->GetArrayLength(jIdx) == 0) {
        ctx->nfq = 0;
        ctx->nfqLim = 0;
        return;
    }
    int nfq = env->GetArrayLength(jIdx) / 7;
    jint* idx = env->GetIntArrayElements(jIdx, nullptr);
    int nfqLim = 0;
    for (int e = 0; e < nfq; e++) {
        nfqLim += idx[e * 7 + 6] >= 0 ? 1 : 0;
    }
    jdouble* in = env->GetDoubleArrayElements(jIn, nullptr);
    jdouble* pts = env->GetDoubleArrayElements(jPts, nullptr);
    ctx->dFqIdx.upload(idx, static_cast<size_t>(nfq) * 7 * sizeof(int));
    ctx->dFqIn.upload(in, static_cast<size_t>(nfq) * 5 * sizeof(double));
    ctx->dFqPts.upload(pts, static_cast<size_t>(env->GetArrayLength(jPts)) * sizeof(double));
    ctx->dFqApplied.alloc(static_cast<size_t>(ctx->nScenarios) * nfq * sizeof(double));
    env->ReleaseIntArrayElements(jIdx, idx, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jIn, in, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jPts, pts, JNI_ABORT);
    ctx->nfq = nfq;
    ctx->nfqLim = nfqLim;
    ctx->fqExtrapolate = extrapolate;
}

// Per-chunk loss pre-distribution moves of the forced generators ([nS * nfq], pu), or null when no scenario of the
// chunk pre-distributes. Set before each solve that carries contingencies.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchForcedTargetQPre(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jPre) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jPre == nullptr || ctx->nfq == 0) {
        ctx->fqHasPre = false;
        return;
    }
    jdouble* pre = env->GetDoubleArrayElements(jPre, nullptr);
    ctx->dFqPre.upload(pre, static_cast<size_t>(env->GetArrayLength(jPre)) * sizeof(double));
    env->ReleaseDoubleArrayElements(jPre, pre, JNI_ABORT);
    ctx->fqHasPre = true;
}

// Remote-member RL entries (plan-B1): install (or clear) the per-entry remote kinds + compare rows
// on a batch context. The RL kernel SKIPS remote entries until the B3 transition machinery lands.
// Slack-only distribution: the slack generators' initial target per scenario (relative to the slack's target P)
// for the next solves - a strategy pair's parent total; null clears it (zero: a contingency's base).
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchSlackOnlyAnchor(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jAnchor) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jAnchor == nullptr || !ctx->dsSlackOnly) {
        ctx->dsSlackHasAnchor = false;
        return;
    }
    jdouble* a = env->GetDoubleArrayElements(jAnchor, nullptr);
    ctx->dDsSlackAnchor.upload(a, static_cast<size_t>(env->GetArrayLength(jAnchor)) * sizeof(double));
    env->ReleaseDoubleArrayElements(jAnchor, a, JNI_ABORT);
    ctx->dsSlackHasAnchor = true;
}

// Per scenario and participating generator (s * ngen + g, pu): anchor = its move from its base target P to where the
// slack's water-fill starts it (a strategy pair: the parent's final target, or a GeneratorAction's value; null = the
// base), start = its move from that anchor at the solve's start (a contingency's loss pre-distribution; null = none).
// The curve and forced-target-Q channels evaluate a generator at anchor + (genCum once a pass ran, else start).
// seedAnchor (pairs): the anchor WITHOUT the strategy's GeneratorActions - where the continued pins are stamped.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchGenAnchor(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jAnchor, jdoubleArray jStart, jdoubleArray jSeedAnchor) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    size_t want = static_cast<size_t>(ctx->nScenarios) * std::max(ctx->ngen, 0);
    auto put = [&](jdoubleArray arr, DeviceBuf& buf, bool& has) {
        has = false;
        if (arr == nullptr || ctx->ngen <= 0) {
            return true;
        }
        if (static_cast<size_t>(env->GetArrayLength(arr)) != want) {
            env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "setBatchGenAnchor: expected nScenarios * ngen");
            return false;
        }
        jdouble* a = env->GetDoubleArrayElements(arr, nullptr);
        buf.upload(a, want * sizeof(double));
        env->ReleaseDoubleArrayElements(arr, a, JNI_ABORT);
        has = true;
        return true;
    };
    if (put(jAnchor, ctx->dDgAnchor, ctx->hasDgAnchor) && put(jStart, ctx->dDgStart, ctx->hasDgStart)) {
        put(jSeedAnchor, ctx->dDgSeedAnchor, ctx->hasDgSeedAnchor);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlElection(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jNomV, jintArray jPRow, jdoubleArray jStaticP, jintArray jIdRank) {
    BatchSolverContext* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->hasRlElection = 0;
    if (ctx->nrl == 0 || jNomV == nullptr || jPRow == nullptr || jStaticP == nullptr || jIdRank == nullptr) {
        return;
    }
    if (env->GetArrayLength(jNomV) != ctx->nrl || env->GetArrayLength(jPRow) != ctx->nrl
            || env->GetArrayLength(jStaticP) != ctx->nrl || env->GetArrayLength(jIdRank) != ctx->nrl) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "setBatchRlElection: array length != nrl");
        return;
    }
    jdouble* nv = env->GetDoubleArrayElements(jNomV, nullptr);
    jint* pr = env->GetIntArrayElements(jPRow, nullptr);
    jdouble* sp = env->GetDoubleArrayElements(jStaticP, nullptr);
    jint* ir = env->GetIntArrayElements(jIdRank, nullptr);
    ctx->dRlElectNomV.upload(nv, static_cast<size_t>(ctx->nrl) * sizeof(double));
    ctx->dRlElectPRow.upload(pr, static_cast<size_t>(ctx->nrl) * sizeof(int));
    ctx->dRlElectStaticP.upload(sp, static_cast<size_t>(ctx->nrl) * sizeof(double));
    ctx->dRlElectIdRank.upload(ir, static_cast<size_t>(ctx->nrl) * sizeof(int));
    env->ReleaseDoubleArrayElements(jNomV, nv, JNI_ABORT);
    env->ReleaseIntArrayElements(jPRow, pr, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jStaticP, sp, JNI_ABORT);
    env->ReleaseIntArrayElements(jIdRank, ir, JNI_ABORT);
    ctx->hasRlElection = 1;
}

// voltageRemoteControlRobustMode (OLF's default): install the robust-mode channel on a batch context —
// the per-entry initial generation target Q that a realistic-V freeze pins at, and the two thresholds
// ReactiveLimitsOuterLoop tests the controller's OWN voltage against (minRealisticVoltage ·
// REALISTIC_VOLTAGE_MARGIN and maxRealisticVoltage / REALISTIC_VOLTAGE_MARGIN, margin 1.02). robust = 0
// clears the channel, so a cached context reused by a run with the mode off stops applying it.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlRobust(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jInitTgtQ, jdouble vLo, jdouble vHi, jint robust) {
    BatchSolverContext* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->rlRobust = 0;
    if (ctx->nrl == 0 || robust == 0 || jInitTgtQ == nullptr) {
        return;
    }
    if (env->GetArrayLength(jInitTgtQ) != ctx->nrl) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "setBatchRlRobust: array length != nrl");
        return;
    }
    jdouble* q0 = env->GetDoubleArrayElements(jInitTgtQ, nullptr);
    ctx->dRlInitTgtQ.upload(q0, static_cast<size_t>(ctx->nrl) * sizeof(double));
    env->ReleaseDoubleArrayElements(jInitTgtQ, q0, JNI_ABORT);
    ctx->rlVLoRobust = vLo;
    ctx->rlVHiRobust = vHi;
    ctx->rlRobust = 1;
}

// AcloadFlowEngine's realistic-voltage check: install (or clear) the per-row "this row is a
// checked bus voltage" mask and the two bounds. enabled = 0 clears it on a cached context.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRealisticCheck(
        JNIEnv* env, jclass, jlong handle, jbyteArray jRowChecked, jdouble minV, jdouble maxV, jint enabled,
        jint fixingLoop) {
    BatchSolverContext* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->realisticCheck = 0;
    if (enabled == 0 || jRowChecked == nullptr) {
        return;
    }
    if (env->GetArrayLength(jRowChecked) != ctx->n) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "setBatchRealisticCheck: array length != n");
        return;
    }
    jbyte* rc = env->GetByteArrayElements(jRowChecked, nullptr);
    ctx->dRealisticRow.upload(rc, static_cast<size_t>(ctx->n) * sizeof(char));
    env->ReleaseByteArrayElements(jRowChecked, rc, JNI_ABORT);
    ctx->realisticMinV = minV;
    ctx->realisticMaxV = maxV;
    ctx->realisticFixingLoop = fixingLoop;
    ctx->realisticCheck = 1;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlSvcLimits(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jBmin, jdoubleArray jBmax, jdoubleArray jV2Base) {
    BatchSolverContext* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nrl == 0) {
        return;
    }
    if (env->GetArrayLength(jBmin) != ctx->nrl || env->GetArrayLength(jBmax) != ctx->nrl
            || env->GetArrayLength(jV2Base) != ctx->nrl) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      "setBatchRlSvcLimits: array length != nrl");
        return;
    }
    jdouble* bmin = env->GetDoubleArrayElements(jBmin, nullptr);
    jdouble* bmax = env->GetDoubleArrayElements(jBmax, nullptr);
    jdouble* v2b = env->GetDoubleArrayElements(jV2Base, nullptr);
    ctx->dRlSvcBmin.upload(bmin, static_cast<size_t>(ctx->nrl) * sizeof(double));
    ctx->dRlSvcBmax.upload(bmax, static_cast<size_t>(ctx->nrl) * sizeof(double));
    ctx->dRlV2Base.upload(v2b, static_cast<size_t>(ctx->nrl) * sizeof(double));
    env->ReleaseDoubleArrayElements(jBmin, bmin, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jBmax, bmax, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jV2Base, v2b, JNI_ABORT);
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlBaseLimitType(
        JNIEnv* env, jclass, jlong handle, jbyteArray jKinds) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jKinds == nullptr || env->GetArrayLength(jKinds) == 0) {
        ctx->hasRlBaseLt = 0;
        return;
    }
    if (env->GetArrayLength(jKinds) != ctx->nrl) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      "setBatchRlBaseLimitType: array length != nrl");
        return;
    }
    jbyte* kinds = env->GetByteArrayElements(jKinds, nullptr);
    ctx->dRlBaseLimitType.alloc(static_cast<size_t>(ctx->nrl));
    CUDA_CHECK(cudaMemcpy(ctx->dRlBaseLimitType.p, kinds, static_cast<size_t>(ctx->nrl), cudaMemcpyHostToDevice));
    env->ReleaseByteArrayElements(jKinds, kinds, JNI_ABORT);
    ctx->hasRlBaseLt = 1;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlRemote(
        JNIEnv* env, jclass, jlong handle, jbyteArray jRemote, jintArray jCmpVrow,
        jintArray jRmqRow, jintArray jRmqElem, jintArray jRmqSide, jintArray jRmqKind, jdoubleArray jRmqWeight,
        jintArray jDqKMember, jintArray jDqJMember, jdoubleArray jMKey, jintArray jMRlEntry,
        jintArray jMGroupStart, jintArray jMGroupCount, jintArray jMRow, jbyteArray jMHasRow,
        jintArray jMQrGroup, jdoubleArray jMTargetQ, jintArray jMTgtPtr, jintArray jMTgtPeer) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jRemote == nullptr || jCmpVrow == nullptr || env->GetArrayLength(jRemote) == 0) {
        ctx->nRlRemote = 0;
        ctx->nRmq = 0;
        ctx->nDqDyn = 0;
        ctx->hasDqQrGroups = 0;
        return;
    }
    int nr = env->GetArrayLength(jRemote);
    if (nr != ctx->nrl || env->GetArrayLength(jCmpVrow) != ctx->nrl) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      "setBatchRlRemote: array length != nrl");
        return;
    }
    jbyte* rem = env->GetByteArrayElements(jRemote, nullptr);
    jint* cmp = env->GetIntArrayElements(jCmpVrow, nullptr);
    int count = 0;
    int selfCount = 0;
    for (int i = 0; i < nr; i++) {
        if (rem[i]) {
            count++;
        }
        if (rem[i] == 2) {
            selfCount++;
        }
    }
    ctx->nRlPilotSelf = selfCount;
    ctx->dRlRemote.upload(rem, static_cast<size_t>(nr) * sizeof(char));
    ctx->dRlCmpVrow.upload(cmp, static_cast<size_t>(nr) * sizeof(int));
    env->ReleaseByteArrayElements(jRemote, rem, JNI_ABORT);
    env->ReleaseIntArrayElements(jCmpVrow, cmp, JNI_ABORT);
    ctx->nRlRemote = count;
    ctx->nRmq = 0;
    if (jRmqRow != nullptr && env->GetArrayLength(jRmqRow) > 0) {
        int nq = env->GetArrayLength(jRmqRow);
        jint* qr = env->GetIntArrayElements(jRmqRow, nullptr);
        jint* qe = env->GetIntArrayElements(jRmqElem, nullptr);
        jint* qsd = env->GetIntArrayElements(jRmqSide, nullptr);
        jint* qk = env->GetIntArrayElements(jRmqKind, nullptr);
        jdouble* qw = env->GetDoubleArrayElements(jRmqWeight, nullptr);
        ctx->dRmqRow.upload(qr, static_cast<size_t>(nq) * sizeof(int));
        ctx->dRmqElem.upload(qe, static_cast<size_t>(nq) * sizeof(int));
        ctx->dRmqSide.upload(qsd, static_cast<size_t>(nq) * sizeof(int));
        ctx->dRmqKind.upload(qk, static_cast<size_t>(nq) * sizeof(int));
        ctx->dRmqWeight.upload(qw, static_cast<size_t>(nq) * sizeof(double));
        env->ReleaseIntArrayElements(jRmqRow, qr, JNI_ABORT);
        env->ReleaseIntArrayElements(jRmqElem, qe, JNI_ABORT);
        env->ReleaseIntArrayElements(jRmqSide, qsd, JNI_ABORT);
        env->ReleaseIntArrayElements(jRmqKind, qk, JNI_ABORT);
        env->ReleaseDoubleArrayElements(jRmqWeight, qw, JNI_ABORT);
        ctx->nRmq = nq;
    }
    ctx->nDqDyn = 0;
    ctx->hasDqQrGroups = 0;
    // Keyed on the MEMBER list, not on the DISTR_Q rows: a SINGLETON remote group has members (and a
    // pilot-Q relocation group, so its pilot's control-loss flip must be armed) but no DISTR_Q row at
    // all — gating on dqKMember left hasDqQrGroups at 0 for it, and the base load flow on the driver
    // kept the pilot's V equation after pinning its only member (2.4e-3 pu on the pilot's Q).
    if (jMKey != nullptr && env->GetArrayLength(jMKey) > 0) {
        int ndqc = jDqKMember != nullptr ? env->GetArrayLength(jDqKMember) : 0;
        int nm = env->GetArrayLength(jMKey);
        jint* km = ndqc > 0 ? env->GetIntArrayElements(jDqKMember, nullptr) : nullptr;
        jint* jm = ndqc > 0 ? env->GetIntArrayElements(jDqJMember, nullptr) : nullptr;
        jdouble* mk = env->GetDoubleArrayElements(jMKey, nullptr);
        jint* mre = env->GetIntArrayElements(jMRlEntry, nullptr);
        jint* mgs = env->GetIntArrayElements(jMGroupStart, nullptr);
        jint* mgc = env->GetIntArrayElements(jMGroupCount, nullptr);
        jint* mrw = env->GetIntArrayElements(jMRow, nullptr);
        jbyte* mhr = env->GetByteArrayElements(jMHasRow, nullptr);
        ctx->dDqMemberRow.upload(mrw, static_cast<size_t>(nm) * sizeof(int));
        ctx->dDqMemberHasRow.upload(mhr, static_cast<size_t>(nm) * sizeof(char));
        env->ReleaseIntArrayElements(jMRow, mrw, JNI_ABORT);
        env->ReleaseByteArrayElements(jMHasRow, mhr, JNI_ABORT);
        if (jMTargetQ == nullptr || env->GetArrayLength(jMTargetQ) != nm) {
            env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                          "setBatchRlRemote: dqMemberTargetQ missing or length != members");
            return;
        }
        jdouble* mtq = env->GetDoubleArrayElements(jMTargetQ, nullptr);
        ctx->dDqMemberTargetQ.upload(mtq, static_cast<size_t>(nm) * sizeof(double));
        env->ReleaseDoubleArrayElements(jMTargetQ, mtq, JNI_ABORT);
        if (jMTgtPtr == nullptr || env->GetArrayLength(jMTgtPtr) != nm + 1 || jMTgtPeer == nullptr) {
            env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                          "setBatchRlRemote: dqMemberTgtPtr missing or length != members + 1");
            return;
        }
        jint* mtp = env->GetIntArrayElements(jMTgtPtr, nullptr);
        ctx->dDqMemberTgtPtr.upload(mtp, static_cast<size_t>(nm + 1) * sizeof(int));
        env->ReleaseIntArrayElements(jMTgtPtr, mtp, JNI_ABORT);
        int nPeer = env->GetArrayLength(jMTgtPeer);
        if (nPeer > 0) {
            jint* mpe = env->GetIntArrayElements(jMTgtPeer, nullptr);
            ctx->dDqMemberTgtPeer.upload(mpe, static_cast<size_t>(nPeer) * sizeof(int));
            env->ReleaseIntArrayElements(jMTgtPeer, mpe, JNI_ABORT);
        }
        ctx->nDqMembers = nm;
        if (ndqc > 0) {
            ctx->dDqKMember.upload(km, static_cast<size_t>(ndqc) * sizeof(int));
            ctx->dDqJMember.upload(jm, static_cast<size_t>(ndqc) * sizeof(int));
        }
        ctx->dDqMemberKey.upload(mk, static_cast<size_t>(nm) * sizeof(double));
        ctx->dDqMemberRlEntry.upload(mre, static_cast<size_t>(nm) * sizeof(int));
        ctx->dDqMemberGroupStart.upload(mgs, static_cast<size_t>(nm) * sizeof(int));
        ctx->dDqMemberGroupCount.upload(mgc, static_cast<size_t>(nm) * sizeof(int));
        if (ndqc > 0) {
            env->ReleaseIntArrayElements(jDqKMember, km, JNI_ABORT);
            env->ReleaseIntArrayElements(jDqJMember, jm, JNI_ABORT);
        }
        env->ReleaseDoubleArrayElements(jMKey, mk, JNI_ABORT);
        env->ReleaseIntArrayElements(jMRlEntry, mre, JNI_ABORT);
        env->ReleaseIntArrayElements(jMGroupStart, mgs, JNI_ABORT);
        env->ReleaseIntArrayElements(jMGroupCount, mgc, JNI_ABORT);
        ctx->nDqDyn = ndqc;
        ctx->hasDqQrGroups = 0;
        if (jMQrGroup != nullptr && env->GetArrayLength(jMQrGroup) == nm) {
            jint* mqg = env->GetIntArrayElements(jMQrGroup, nullptr);
            std::vector<char> genVc(ctx->st->nQr > 0 ? ctx->st->nQr : 1, 0);
            for (int i = 0; i < nm; i++) {
                if (mqg[i] >= 0) {
                    ctx->hasDqQrGroups = 1;                  // plan-B4: relocation-linked group present
                    genVc[mqg[i]] = 1;                       // this qr group is a remote gen-VC group
                }
            }
            for (int g = 0; g < ctx->st->nQr; g++) {         // a DEPENDENT pilot's group is as generator-controlled
                if (g < static_cast<int>(ctx->st->hQrDepBase.size()) && ctx->st->hQrDepBase[g] >= 0) {   // as its base
                    genVc[g] = genVc[ctx->st->hQrDepBase[g]];
                }
            }
            ctx->dDqMemberQrGroup.upload(mqg, static_cast<size_t>(nm) * sizeof(int));
            if (ctx->hasDqQrGroups) {
                ctx->dQrGenVc.upload(genVc.data(), genVc.size() * sizeof(char));
            }
            env->ReleaseIntArrayElements(jMQrGroup, mqg, JNI_ABORT);
        }
  }
}

// plan-B4 (JAVA hybrid): install ONLY the carrier-row suppression tables — dqJMember (per DISTR_Q
// contribution: row-owner member) + dqMemberQrGroup (per member: pilot's relocation group) — so a
// pushed mode-4 free-member row gates the re-picked carrier's DISTR_Q row off in fillDistrQBatch.
// Does NOT touch nDqDyn: the hybrid's weights stay static-pack + dqOv (OLF renormalizes and the
// engine pushes live values). Nulls clear.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchZiTreeOverrides(
        JNIEnv* env, jclass, jlong handle, jintArray jSlotElemVal) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    int len = jSlotElemVal == nullptr ? 0 : env->GetArrayLength(jSlotElemVal);
    ctx->nZiTree = len / 3;
    if (ctx->nZiTree > 0) {
        jint* p = env->GetIntArrayElements(jSlotElemVal, nullptr);
        ctx->dZiTreeSlotElemVal.upload(p, static_cast<size_t>(len) * sizeof(int));
        env->ReleaseIntArrayElements(jSlotElemVal, p, JNI_ABORT);
    }
    // Built whenever the masks are enabled, NOT only once the off/on setter has run: the two setters' order is
    // the caller's, and gating this one on hasZiMasks dropped every tree override of a chunk whose tree list
    // arrived first - the extraction-time tree then stood in for OLF's, and a coupler child like LLJ_2 on
    // network.xiidm lost its ZERO_V coupling in 2,308 scenarios (V 0.787, FAILED as unrealistic).
    if (ctx->nzi > 0) {                                      // the dense form (see dZiTreeMask)
        ctx->hasZiMasks = true;
        size_t mlen = static_cast<size_t>(ctx->nScenarios) * ctx->nzi;
        std::vector<char> tree(mlen, (char) -1);
        if (len > 0) {
            jint* p = env->GetIntArrayElements(jSlotElemVal, nullptr);
            for (int k = 0; k < len / 3; k++) {
                int sl = p[3 * k], e = p[3 * k + 1];
                if (sl >= 0 && sl < ctx->nScenarios && e >= 0 && e < ctx->nzi) tree[(size_t) sl * ctx->nzi + e] = (char) (p[3 * k + 2] != 0 ? 1 : 0);
            }
            env->ReleaseIntArrayElements(jSlotElemVal, p, JNI_ABORT);
        }
        ctx->dZiTreeMask.upload(tree.data(), mlen);
        if (std::getenv("OLF_GPU_PHASE_TIME") != nullptr) {
            std::fprintf(stderr, "ZI_LISTS nZiTree=%d\n", len / 3);
        }
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchLoopTraceSlot(
        JNIEnv*, jclass, jlong handle, jint slot) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        ctx->loopTraceSlot = slot;
    }
}

// DISTR_Q weight overrides: install (or clear, with nulls) the per-scenario qPct renormalization
// pairs {s, contrib} + new weights on a batch context (gen outage at a remote-VC controller bus).
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDqInactiveMembers(
        JNIEnv* env, jclass, jlong handle, jintArray jSlotMember) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jSlotMember == nullptr || env->GetArrayLength(jSlotMember) == 0) {
        ctx->nDqInact = 0;
        return;
    }
    int n2 = env->GetArrayLength(jSlotMember);
    jint* sm = env->GetIntArrayElements(jSlotMember, nullptr);
    ctx->dDqInact.upload(sm, static_cast<size_t>(n2) * sizeof(int));
    env->ReleaseIntArrayElements(jSlotMember, sm, JNI_ABORT);
    ctx->nDqInact = n2 / 2;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDqSubgroups(
        JNIEnv* env, jclass, jlong handle, jbyteArray jDense, jintArray jQrMemberStart) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jDense == nullptr || env->GetArrayLength(jDense) == 0) {
        ctx->hasDqSub = false;
        return;
    }
    jbyte* d = env->GetByteArrayElements(jDense, nullptr);
    ctx->dDqSub.upload(d, static_cast<size_t>(env->GetArrayLength(jDense)));
    env->ReleaseByteArrayElements(jDense, d, JNI_ABORT);
    jint* q = env->GetIntArrayElements(jQrMemberStart, nullptr);
    ctx->dQrMemberStart.upload(q, static_cast<size_t>(env->GetArrayLength(jQrMemberStart)) * sizeof(int));
    env->ReleaseIntArrayElements(jQrMemberStart, q, JNI_ABORT);
    ctx->hasDqSub = true;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchFrozenRows(
        JNIEnv* env, jclass, jlong handle, jintArray jSlotRow) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jSlotRow == nullptr || env->GetArrayLength(jSlotRow) == 0) {
        ctx->nFrozenRow = 0;
        return;
    }
    int n2 = env->GetArrayLength(jSlotRow);
    jint* sr = env->GetIntArrayElements(jSlotRow, nullptr);
    ctx->dFrozenRow.upload(sr, static_cast<size_t>(n2) * sizeof(int));
    env->ReleaseIntArrayElements(jSlotRow, sr, JNI_ABORT);
    ctx->nFrozenRow = n2 / 2;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDsLostGenerators(
        JNIEnv* env, jclass, jlong handle, jintArray jSlotGen) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jSlotGen == nullptr || env->GetArrayLength(jSlotGen) == 0) {
        ctx->nDsLostGen = 0;
        return;
    }
    int n2 = env->GetArrayLength(jSlotGen);
    jint* sg = env->GetIntArrayElements(jSlotGen, nullptr);
    ctx->dDsLostGen.upload(sg, static_cast<size_t>(n2) * sizeof(int));
    env->ReleaseIntArrayElements(jSlotGen, sg, JNI_ABORT);
    ctx->nDsLostGen = n2 / 2;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlPinTargets(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jTargets) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jTargets == nullptr || env->GetArrayLength(jTargets) == 0) {
        ctx->hasRlPinTgtIn = false;
        return;
    }
    int len = env->GetArrayLength(jTargets);
    jdouble* t = env->GetDoubleArrayElements(jTargets, nullptr);
    ctx->dRlPinTgtIn.upload(t, static_cast<size_t>(len) * sizeof(double));
    env->ReleaseDoubleArrayElements(jTargets, t, JNI_ABORT);
    ctx->hasRlPinTgtIn = true;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchRlPinTargets(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nrl <= 0) {
        return nullptr;
    }
    int nS = ctx->nScenarios;
    int n = ctx->st->n;
    DeviceBuf out;
    out.alloc(static_cast<size_t>(ctx->nrl) * nS * sizeof(double));
    exportRlPinTargets<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(ctx->nrl, nS, n, ctx->dRlVrow.as<int>(),
            ctx->dRlQrow.p ? ctx->dRlQrow.as<int>() : nullptr, ctx->dRlLimitType.as<char>(), ctx->dMode.as<int>(),
            ctx->dTarget.as<double>(), ctx->nRlIslanded > 0 ? ctx->dRlIslanded.as<char>() : nullptr,
            ctx->hasDqQrGroups ? ctx->st->nQr : 0, psEntryPtr(ctx),
            ctx->hasDqQrGroups && ctx->st->nQr > 0 ? ctx->st->dQrPtr.as<int>() : nullptr,
            ctx->hasDqQrGroups && ctx->st->nQr > 0 ? ctx->st->dQrRows.as<int>() : nullptr,
            ctx->hasDqQrGroups && ctx->st->nQr > 0 ? ctx->dQrVeqOff.as<char>() : nullptr,
            ctx->st->dQrDepBase.p ? ctx->st->dQrDepBase.as<int>() : nullptr, out.as<double>());
    std::vector<double> host(static_cast<size_t>(ctx->nrl) * nS);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(cudaMemcpy(host.data(), out.p, host.size() * sizeof(double), cudaMemcpyDeviceToHost));
    jdoubleArray arr = env->NewDoubleArray(static_cast<jsize>(host.size()));
    env->SetDoubleArrayRegion(arr, 0, static_cast<jsize>(host.size()), host.data());
    return arr;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDsOutGenerators(
        JNIEnv* env, jclass, jlong handle, jintArray jSlotGen) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jSlotGen == nullptr || env->GetArrayLength(jSlotGen) == 0) {
        ctx->nDsOutGen = 0;
        return;
    }
    int n2 = env->GetArrayLength(jSlotGen);
    jint* sg = env->GetIntArrayElements(jSlotGen, nullptr);
    ctx->dDsOutGen.upload(sg, static_cast<size_t>(n2) * sizeof(int));
    env->ReleaseIntArrayElements(jSlotGen, sg, JNI_ABORT);
    ctx->nDsOutGen = n2 / 2;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDqWeightOverrides(
        JNIEnv* env, jclass, jlong handle, jintArray jIdx, jdoubleArray jW) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jIdx == nullptr || jW == nullptr || env->GetArrayLength(jW) == 0) {
        ctx->nDqOv = 0;
        return;
    }
    int nOv = env->GetArrayLength(jW);
    jint* idx = env->GetIntArrayElements(jIdx, nullptr);
    jdouble* w = env->GetDoubleArrayElements(jW, nullptr);
    ctx->dDqOvIdx.upload(idx, static_cast<size_t>(nOv) * 2 * sizeof(int));
    ctx->dDqOvW.upload(w, static_cast<size_t>(nOv) * sizeof(double));
    env->ReleaseIntArrayElements(jIdx, idx, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jW, w, JNI_ABORT);
    ctx->nDqOv = nOv;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDqKeyOverrides(
        JNIEnv* env, jclass, jlong handle, jintArray jIdx, jdoubleArray jKey) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jIdx == nullptr || jKey == nullptr || env->GetArrayLength(jKey) == 0) {
        ctx->nDqKeyOv = 0;
        return;
    }
    int nOv = env->GetArrayLength(jKey);
    jint* idx = env->GetIntArrayElements(jIdx, nullptr);
    jdouble* key = env->GetDoubleArrayElements(jKey, nullptr);
    ctx->dDqKeyOvIdx.upload(idx, static_cast<size_t>(nOv) * 2 * sizeof(int));
    ctx->dDqKeyOvKey.upload(key, static_cast<size_t>(nOv) * sizeof(double));
    env->ReleaseIntArrayElements(jIdx, idx, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jKey, key, JNI_ABORT);
    ctx->nDqKeyOv = nOv;
}

// Per-GENERATOR distributed-slack participants: the factors and limits OLF's water-fill actually
// saturates on, grouped by bus (a bus's row value is the sum over its contiguous slice).
// The slack bus's participating generators' total rooms: the bound of a slack-only distribution (slackOnlyDsCheck).
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDsSlackRooms(
        JNIEnv*, jclass, jlong handle, jdouble roomUp, jdouble roomDown) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        ctx->dsSlackRoomUp = roomUp;
        ctx->dsSlackRoomDown = roomDown;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDsGenerators(
        JNIEnv* env, jclass, jlong handle, jintArray jBus, jdoubleArray jFactor,
        jdoubleArray jMaxDelta, jdoubleArray jMinDelta, jdoubleArray jMarginUp, jdoubleArray jMarginDown,
        jintArray jBusGenStart, jintArray jBusGenEnd) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jBus == nullptr || env->GetArrayLength(jBus) == 0) {
        ctx->ngen = 0;
        return;
    }
    int ngen = env->GetArrayLength(jBus);
    auto upI = [&](jintArray a, DeviceBuf& buf) {
        jint* p = env->GetIntArrayElements(a, nullptr);
        buf.upload(p, (size_t) env->GetArrayLength(a) * sizeof(int));
        env->ReleaseIntArrayElements(a, p, JNI_ABORT);
    };
    auto upD = [&](jdoubleArray a, DeviceBuf& buf) {
        if (a == nullptr || env->GetArrayLength(a) == 0) {
            return;
        }
        jdouble* p = env->GetDoubleArrayElements(a, nullptr);
        buf.upload(p, (size_t) env->GetArrayLength(a) * sizeof(double));
        env->ReleaseDoubleArrayElements(a, p, JNI_ABORT);
    };
    upI(jBus, ctx->dgBus);
    upD(jFactor, ctx->dgFactor);
    upD(jMaxDelta, ctx->dgMaxDelta);
    upD(jMinDelta, ctx->dgMinDelta);
    upD(jMarginUp, ctx->dgMarginUp);
    upD(jMarginDown, ctx->dgMarginDown);
    upI(jBusGenStart, ctx->dBusGenStart);
    upI(jBusGenEnd, ctx->dBusGenEnd);
    ctx->dGenCum.alloc((size_t) ctx->nScenarios * ngen * sizeof(double));
    ctx->dGenCumValid.alloc((size_t) ctx->nScenarios * sizeof(char));
    ctx->dGenFactorWork.alloc((size_t) ctx->nScenarios * ngen * sizeof(double));
    ctx->dGenRemoved.alloc((size_t) ctx->nScenarios * ngen * sizeof(char));
    ctx->ngen = ngen;
}

// The HVDC AC-emulation outer loops' tables + their per-scenario starting state (the statuses and
// applied converter-row deltas the host's pre-solve freeze produced). Install to run those loops as
// DRIVER PHASES in OLF's order; clear (null) to leave them to the host orchestrator.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchHvdcLoops(
        JNIEnv* env, jclass, jlong handle, jintArray jElem, jintArray jBus1Row, jintArray jBus2Row,
        jintArray jBus1Phi, jintArray jBus2Phi,
        jdoubleArray jIn, jbyteArray jCtgOut, jintArray jStatus,
        jdoubleArray jAppliedD1, jdoubleArray jAppliedD2, jint limitsActive) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jElem == nullptr || env->GetArrayLength(jElem) == 0) {
        ctx->nhv = 0;
        return;
    }
    int nh = env->GetArrayLength(jElem);
    auto upI = [&](jintArray a, DeviceBuf& b) {
        jint* q = env->GetIntArrayElements(a, nullptr);
        b.upload(q, (size_t) env->GetArrayLength(a) * sizeof(int));
        env->ReleaseIntArrayElements(a, q, JNI_ABORT);
    };
    auto upD = [&](jdoubleArray a, DeviceBuf& b) {
        jdouble* q = env->GetDoubleArrayElements(a, nullptr);
        b.upload(q, (size_t) env->GetArrayLength(a) * sizeof(double));
        env->ReleaseDoubleArrayElements(a, q, JNI_ABORT);
    };
    upI(jElem, ctx->dHvElem);
    upI(jBus1Row, ctx->dHvBus1Row);
    upI(jBus2Row, ctx->dHvBus2Row);
    upI(jBus1Phi, ctx->dHvBus1Phi);
    upI(jBus2Phi, ctx->dHvBus2Phi);
    upD(jIn, ctx->dHvIn);
    upI(jStatus, ctx->dHvStatus);
    upD(jAppliedD1, ctx->dHvAppliedD1);
    upD(jAppliedD2, ctx->dHvAppliedD2);
    jbyte* co = env->GetByteArrayElements(jCtgOut, nullptr);
    ctx->dHvCtgOut.upload(co, (size_t) env->GetArrayLength(jCtgOut));
    env->ReleaseByteArrayElements(jCtgOut, co, JNI_ABORT);
    ctx->nhv = nh;
    ctx->hvLimitsActive = limitsActive;
}

/** Enable HOST-owned outer-loop slots via a bitmask.
 *  Bit 0 = SecondaryVoltageControl (OLF order 4, pre-RL),
 *  Bit 1 = PhaseControl (OLF order 7, post-RL),
 *  Bit 2 = TransformerVoltageControl (OLF order 8),
 *  Bit 3 = TransformerReactivePowerControl (OLF order 9),
 *  Bit 4 = ShuntVoltageControl (OLF order 10).
 *  0 disables all host loops; the driver behaves exactly as if the mechanism did not exist. */
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchHostLoop(
        JNIEnv*, jclass, jlong handle, jint mask) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        ctx->hostLoopMask = mask;
    }
}

/** Upload the incremental TVC controller data for the in-kernel outer loop.
 *  rhoRow/vRow/elemIndex: [nCtrl] int; targetV/halfDb: [nCtrl] double;
 *  tapPtr: [nCtrl+1] int (CSR into tapData); tapData: [totalTaps*8] double;
 *  baseTap: [nCtrl] int; groupPtr: [nGroups+1] int (CSR into groupCtrl);
 *  groupCtrl: [nCtrl] int (controller indices per group); maxTapShift: OLF's per-pass bound.
 *  Returns 1 when the kernel loop OWNS the controllers (shared controlled buses included — the
 *  one-tap-at-a-time round robin runs in tvcGroupDecisionBatch), 0 when there are none. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchTvcControllers(
        JNIEnv* env, jclass, jlong handle,
        jintArray jRhoRow, jintArray jVRow, jdoubleArray jTargetV, jdoubleArray jHalfDb,
        jintArray jElemIndex, jintArray jTapPtr, jdoubleArray jTapData, jintArray jBaseTap,
        jintArray jGroupPtr, jintArray jGroupCtrl, jint maxTapShift) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nTvc = 0;
    if (jRhoRow == nullptr || env->GetArrayLength(jRhoRow) == 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    auto readD = [&](jdoubleArray a, std::vector<double>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetDoubleArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> groupPtr;
    readI(jGroupPtr, groupPtr);                              // shared controlled buses run in-kernel too
    std::vector<int> vRow, groupCtrl;
    std::vector<double> targetV, halfDb;
    readI(jRhoRow, ctx->hTvcRhoRow);
    readI(jVRow, vRow);
    readD(jTargetV, targetV);
    readD(jHalfDb, halfDb);
    readI(jElemIndex, ctx->hTvcElemIndex);
    readI(jTapPtr, ctx->hTvcTapPtr);
    readD(jTapData, ctx->hTvcTapData);
    readI(jBaseTap, ctx->hTvcBaseTap);
    readI(jGroupCtrl, groupCtrl);
    int nc = static_cast<int>(ctx->hTvcRhoRow.size());
    ctx->dTvcRhoRow.upload(ctx->hTvcRhoRow.data(), nc * sizeof(int));
    ctx->dTvcVRow.upload(vRow.data(), nc * sizeof(int));
    ctx->dTvcTargetV.upload(targetV.data(), nc * sizeof(double));
    ctx->dTvcHalfDb.upload(halfDb.data(), nc * sizeof(double));
    ctx->dTvcElemIndex.upload(ctx->hTvcElemIndex.data(), nc * sizeof(int));
    ctx->dTvcTapPtr.upload(ctx->hTvcTapPtr.data(), ctx->hTvcTapPtr.size() * sizeof(int));
    ctx->dTvcTapData.upload(ctx->hTvcTapData.data(), ctx->hTvcTapData.size() * sizeof(double));
    ctx->dTvcBaseTap.upload(ctx->hTvcBaseTap.data(), nc * sizeof(int));
    ctx->nTvcGroups = static_cast<int>(groupPtr.size()) - 1;
    ctx->dTvcGroupPtr.upload(groupPtr.data(), groupPtr.size() * sizeof(int));
    ctx->dTvcGroupCtrl.upload(groupCtrl.data(), groupCtrl.size() * sizeof(int));
    ctx->tvcMaxTapShift = maxTapShift > 0 ? maxTapShift : 1;
    size_t nst = static_cast<size_t>(ctx->nScenarios) * nc;
    ctx->dTvcCurTap.ensure(nst * sizeof(int));
    ctx->dTvcPrevTap.ensure(nst * sizeof(int));
    ctx->dTvcRatchetDir.ensure(nst * sizeof(int));
    ctx->dTvcRatchetCount.ensure(nst * sizeof(int));
    ctx->dTvcRatchetAllowed.ensure(nst * sizeof(int));
    ctx->nTvc = nc;
    return 1;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchMonitoringTargets(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nmon == 0 || !ctx->dMonActivated.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nmon) * ctx->nScenarios;
    std::vector<double> host(cnt);
    cudaMemcpy(host.data(), ctx->dMonActivated.p, cnt * sizeof(double), cudaMemcpyDeviceToHost);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(cnt));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}

// The zone-regulated generator voltage control TARGETS the SVC phase settled on, per
// (scenario, controller) — what OLF's SecondaryVoltageControlOuterLoop leaves on the network. The
// batched SA restores per scenario and needs none of this; the base load flow has to give them back.
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchSecondaryVcTargets(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nSvc <= 0 || !ctx->dSvcTargetV.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nSvc) * ctx->nScenarios;
    std::vector<double> host(cnt);
    cudaMemcpy(host.data(), ctx->dSvcTargetV.p, cnt * sizeof(double), cudaMemcpyDeviceToHost);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(cnt));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchTvcTaps(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nTvc <= 0 || !ctx->dTvcCurTap.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nTvc) * ctx->nScenarios;
    std::vector<int> host(cnt);
    cudaMemcpy(host.data(), ctx->dTvcCurTap.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(cnt));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}

/** Upload the incremental SHUNT voltage-control tables for the in-kernel loop. Per controller:
 *  bRow (SHUNT_B identity row), vRow (controlled bus), targetV, halfDb, elemAbs (global element, -1),
 *  shOvElem (shunt-pack element for the static (dG, dB) override, -1); groups: groupPtr[nGroups+1] ->
 *  groupCtrl (controllers, |b|-descending); subs: subPtr[nCtrl+1] -> per sub secPtr[nSubs+1] into
 *  secB/secG, subMin, subBase. Returns 1 when the kernel loop owns them, 0 when there are none. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchShuntControllers(
        JNIEnv* env, jclass, jlong handle,
        jintArray jBRow, jintArray jVRow, jdoubleArray jTargetV, jdoubleArray jHalfDb, jintArray jElemAbs, jintArray jShOvElem,
        jintArray jGroupPtr, jintArray jGroupCtrl, jintArray jSubPtr, jintArray jSecPtr, jdoubleArray jSecB, jdoubleArray jSecG,
        jintArray jSubMin, jintArray jSubBase, jint maxSectionShift) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nShc = 0;
    if (jBRow == nullptr || env->GetArrayLength(jBRow) == 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    auto readD = [&](jdoubleArray a, std::vector<double>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetDoubleArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> vRow, elemAbs, groupPtr, groupCtrl, subMin;
    std::vector<double> targetV, halfDb;
    readI(jBRow, ctx->hShcBRow);
    readI(jVRow, vRow);
    readD(jTargetV, targetV);
    readD(jHalfDb, halfDb);
    readI(jElemAbs, elemAbs);
    readI(jShOvElem, ctx->hShcShOvElem);
    readI(jGroupPtr, groupPtr);
    readI(jGroupCtrl, groupCtrl);
    readI(jSubPtr, ctx->hShcSubPtr);
    readI(jSecPtr, ctx->hShcSecPtr);
    readD(jSecB, ctx->hShcSecB);
    readD(jSecG, ctx->hShcSecG);
    readI(jSubMin, subMin);
    readI(jSubBase, ctx->hShcSubBase);
    int nc = static_cast<int>(ctx->hShcBRow.size());
    int nSubs = static_cast<int>(ctx->hShcSubBase.size());
    ctx->dShcBRow.upload(ctx->hShcBRow.data(), nc * sizeof(int));
    ctx->dShcVRow.upload(vRow.data(), nc * sizeof(int));
    ctx->dShcTargetV.upload(targetV.data(), nc * sizeof(double));
    ctx->dShcHalfDb.upload(halfDb.data(), nc * sizeof(double));
    ctx->dShcElemAbs.upload(elemAbs.data(), nc * sizeof(int));
    ctx->dShcShOvElem.upload(ctx->hShcShOvElem.data(), nc * sizeof(int));
    ctx->dShcGroupPtr.upload(groupPtr.data(), groupPtr.size() * sizeof(int));
    ctx->dShcGroupCtrl.upload(groupCtrl.data(), groupCtrl.size() * sizeof(int));
    ctx->dShcSubPtr.upload(ctx->hShcSubPtr.data(), ctx->hShcSubPtr.size() * sizeof(int));
    ctx->dShcSecPtr.upload(ctx->hShcSecPtr.data(), ctx->hShcSecPtr.size() * sizeof(int));
    ctx->dShcSecB.upload(ctx->hShcSecB.data(), ctx->hShcSecB.size() * sizeof(double));
    ctx->dShcSecG.upload(ctx->hShcSecG.data(), ctx->hShcSecG.size() * sizeof(double));
    ctx->dShcSubMin.upload(subMin.data(), nSubs * sizeof(int));
    ctx->dShcSubBase.upload(ctx->hShcSubBase.data(), nSubs * sizeof(int));
    ctx->nShcGroups = static_cast<int>(groupPtr.size()) - 1;
    ctx->nShcSubs = nSubs;
    ctx->shcMaxSectionShift = maxSectionShift >= 0 ? maxSectionShift : 3;
    size_t nsu = static_cast<size_t>(ctx->nScenarios) * nSubs;
    size_t nsc = static_cast<size_t>(ctx->nScenarios) * nc;
    size_t nsg = static_cast<size_t>(ctx->nScenarios) * ctx->nShcGroups;
    ctx->dShcPos.ensure(nsu * sizeof(int));
    ctx->dShcRatDir.ensure(nsu * sizeof(int));
    ctx->dShcRatCount.ensure(nsu * sizeof(int));
    ctx->dShcRatAllowed.ensure(nsu * sizeof(int));
    ctx->dShcShift.ensure(nsu * sizeof(int));
    ctx->dShcSensi.ensure(nsc * sizeof(double));
    ctx->dShcBaseB.ensure(nsc * sizeof(double));
    ctx->dShcAppliedDb.ensure(nsc * sizeof(double));
    ctx->dShcOutOfDb.ensure(nsg * sizeof(char));
    ctx->dShcDiffV.ensure(nsg * sizeof(double));
    ctx->nShc = nc;
    return 1;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchShuntPositions(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nShc <= 0 || !ctx->dShcPos.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nShcSubs) * ctx->nScenarios;
    std::vector<int> host(cnt);
    cudaMemcpy(host.data(), ctx->dShcPos.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(cnt));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}

/** Upload the FLOW-OBSERVED incremental controllers (RPC / incremental phase). Per controller: kind
 *  (0 RPC, 1 phase CONTROLLER, 2 phase LIMITER), pinRow (the rho/a1 identity row), pinCol (6 r1 / 7 a1),
 *  side (1/2) of the controlled flow, targetValue, halfDb, elem (controller's global element),
 *  ctlElem (controlled branch's global element — a closed branch), tapPtr/tapData ([y g1 g2 b1 b2 ksi r1
 *  a1] per tap), baseTap, maxShift. Returns 1 when the kernel loops own them. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchFlowControllers(
        JNIEnv* env, jclass, jlong handle,
        jintArray jKind, jintArray jPinRow, jintArray jPinCol, jintArray jSide, jdoubleArray jTargetValue, jdoubleArray jHalfDb,
        jintArray jElem, jintArray jCtlElem, jintArray jTapPtr, jdoubleArray jTapData, jintArray jBaseTap, jintArray jMaxShift) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nFlc = 0;
    ctx->nFlcPhase = 0;
    ctx->nFlcRpc = 0;
    if (jKind == nullptr || env->GetArrayLength(jKind) == 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    auto readD = [&](jdoubleArray a, std::vector<double>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetDoubleArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> pinCol, side, ctlElem, maxShift;
    std::vector<double> targetValue, halfDb;
    readI(jKind, ctx->hFlcKind);
    readI(jPinRow, ctx->hFlcPinRow);
    readI(jPinCol, pinCol);
    readI(jSide, side);
    readD(jTargetValue, targetValue);
    readD(jHalfDb, halfDb);
    readI(jElem, ctx->hFlcElem);
    readI(jCtlElem, ctlElem);
    readI(jTapPtr, ctx->hFlcTapPtr);
    readD(jTapData, ctx->hFlcTapData);
    readI(jBaseTap, ctx->hFlcBaseTap);
    readI(jMaxShift, maxShift);
    int nc = static_cast<int>(ctx->hFlcKind.size());
    std::vector<int> ctlPack(nc);
    for (int c = 0; c < nc; c++) {
        ctlPack[c] = ctlElem[c] - ctx->cbOffset;             // the controlled branch's index in the closed-branch pack
        if (ctlPack[c] < 0 || ctlPack[c] >= ctx->st->cb.ne) {
            return 0;                                        // not a closed branch: the host orchestrator keeps it
        }
        if (ctx->hFlcKind[c] == 0) ctx->nFlcRpc++; else ctx->nFlcPhase++;
    }
    ctx->dFlcKind.upload(ctx->hFlcKind.data(), nc * sizeof(int));
    ctx->dFlcPinRow.upload(ctx->hFlcPinRow.data(), nc * sizeof(int));
    ctx->dFlcPinCol.upload(pinCol.data(), nc * sizeof(int));
    ctx->dFlcSide.upload(side.data(), nc * sizeof(int));
    ctx->dFlcTargetValue.upload(targetValue.data(), nc * sizeof(double));
    ctx->dFlcHalfDb.upload(halfDb.data(), nc * sizeof(double));
    ctx->dFlcElem.upload(ctx->hFlcElem.data(), nc * sizeof(int));
    ctx->dFlcCtlPack.upload(ctlPack.data(), nc * sizeof(int));
    ctx->dFlcTapPtr.upload(ctx->hFlcTapPtr.data(), ctx->hFlcTapPtr.size() * sizeof(int));
    ctx->dFlcTapData.upload(ctx->hFlcTapData.data(), ctx->hFlcTapData.size() * sizeof(double));
    ctx->dFlcBaseTap.upload(ctx->hFlcBaseTap.data(), nc * sizeof(int));
    ctx->dFlcMaxShift.upload(maxShift.data(), nc * sizeof(int));
    size_t nst = static_cast<size_t>(ctx->nScenarios) * nc;
    ctx->dFlcCurTap.ensure(nst * sizeof(int));
    ctx->dFlcPrevTap.ensure(nst * sizeof(int));
    ctx->dFlcRatDir.ensure(nst * sizeof(int));
    ctx->dFlcRatCount.ensure(nst * sizeof(int));
    ctx->dFlcRatAllowed.ensure(nst * sizeof(int));
    ctx->dFlcSensi.ensure(nst * sizeof(double));
    ctx->dFlcDiff.ensure(nst * sizeof(double));
    ctx->dFlcOutOfDb.ensure(nst * sizeof(char));
    ctx->nFlc = nc;
    return 1;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchFlowTaps(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nFlc <= 0 || !ctx->dFlcCurTap.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nFlc) * ctx->nScenarios;
    std::vector<int> host(cnt);
    cudaMemcpy(host.data(), ctx->dFlcCurTap.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(cnt));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}

/** Upload the CONTINUOUS phase controllers (kind 0 CONTROLLER / 1 LIMITER, a1 identity row, controlled
 *  side, current limit, global element, tap rows, base tap). Returns 1 when the kernel loop owns them. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchContinuousPhase(
        JNIEnv* env, jclass, jlong handle,
        jintArray jKind, jintArray jA1Row, jintArray jSide, jdoubleArray jLimit, jintArray jElem,
        jintArray jTapPtr, jdoubleArray jTapData, jintArray jBaseTap) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nCph = 0;
    if (jKind == nullptr || env->GetArrayLength(jKind) == 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    auto readD = [&](jdoubleArray a, std::vector<double>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetDoubleArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> kind, a1Row, side;
    std::vector<double> limit;
    readI(jKind, kind);
    readI(jA1Row, a1Row);
    readI(jSide, side);
    readD(jLimit, limit);
    readI(jElem, ctx->hCphElem);
    readI(jTapPtr, ctx->hCphTapPtr);
    readD(jTapData, ctx->hCphTapData);
    readI(jBaseTap, ctx->hCphBaseTap);
    int nc = static_cast<int>(kind.size());
    for (int c = 0; c < nc; c++) {
        int e = ctx->hCphElem[c] - ctx->cbOffset;
        if (e < 0 || e >= ctx->st->cb.ne) {
            return 0;                                        // not a closed branch: the host orchestrator keeps it
        }
    }
    ctx->dCphKind.upload(kind.data(), nc * sizeof(int));
    ctx->dCphA1Row.upload(a1Row.data(), nc * sizeof(int));
    ctx->dCphSide.upload(side.data(), nc * sizeof(int));
    ctx->dCphLimit.upload(limit.data(), nc * sizeof(double));
    ctx->dCphElem.upload(ctx->hCphElem.data(), nc * sizeof(int));
    ctx->dCphTapPtr.upload(ctx->hCphTapPtr.data(), ctx->hCphTapPtr.size() * sizeof(int));
    ctx->dCphTapData.upload(ctx->hCphTapData.data(), ctx->hCphTapData.size() * sizeof(double));
    ctx->dCphBaseTap.upload(ctx->hCphBaseTap.data(), nc * sizeof(int));
    size_t nst = static_cast<size_t>(ctx->nScenarios) * nc;
    ctx->dCphTap.ensure(nst * sizeof(int));
    ctx->dCphPrevTap.ensure(nst * sizeof(int));
    ctx->dCphIter.ensure(static_cast<size_t>(ctx->nScenarios) * sizeof(int));
    ctx->nCph = nc;
    return 1;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchContinuousPhaseTaps(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nCph <= 0 || !ctx->dCphTap.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nCph) * ctx->nScenarios;
    std::vector<int> host(cnt);
    cudaMemcpy(host.data(), ctx->dCphTap.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(cnt));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}

/** Install the SecondaryVoltageControl zones for the in-kernel phase (OLF order 4). Controllers are
 *  addressed by their controlled bus's V row (the sensitivity's unit row and where a moved target lands)
 *  and their own bus V row (the reactive injection the K comes from); each carries a CSR slice of Q
 *  observables (kind 0/1 = a closed branch's side-1/side-2 reactive flow, 2 = a shunt) and the zones a
 *  CSR slice of controllers. Returns 1 when the kernel loop owns them — it refuses a zone set wider than
 *  the in-kernel dense system, or an observable on an element the batch does not pack. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchSecondaryVc(
        JNIEnv* env, jclass, jlong handle,
        jintArray jCtldVRow, jintArray jBusVRow, jintArray jRlIdx, jdoubleArray jBaseTargetV,
        jdoubleArray jMinQ, jdoubleArray jMaxQ, jdoubleArray jLoadQ,
        jintArray jQPtr, jintArray jQKind, jintArray jQElem, jintArray jQBRow,
        jintArray jZonePtr, jintArray jZoneCtl, jintArray jZonePilotRow, jdoubleArray jZoneTargetV,
        jdouble minPlausibleV, jdouble maxPlausibleV) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nSvc = 0;
    ctx->nSvcZones = 0;
    if (jCtldVRow == nullptr || env->GetArrayLength(jCtldVRow) == 0 || ctx->nrl <= 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    auto readD = [&](jdoubleArray a, std::vector<double>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetDoubleArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> busVRow, rlIdx, qPtr, qKind, qElem, qBRow, zonePtr, zoneCtl, zonePilotRow;
    std::vector<double> minQ, maxQ, loadQ, zoneTargetV;
    readI(jCtldVRow, ctx->hSvcCtldVRow);
    readI(jBusVRow, busVRow);
    readI(jRlIdx, rlIdx);
    readD(jBaseTargetV, ctx->hSvcBaseTargetV);
    readD(jMinQ, minQ);
    readD(jMaxQ, maxQ);
    readD(jLoadQ, loadQ);
    readI(jQPtr, qPtr);
    readI(jQKind, qKind);
    readI(jQElem, qElem);
    readI(jQBRow, qBRow);
    readI(jZonePtr, zonePtr);
    readI(jZoneCtl, zoneCtl);
    readI(jZonePilotRow, zonePilotRow);
    readD(jZoneTargetV, zoneTargetV);
    int nC = static_cast<int>(ctx->hSvcCtldVRow.size());
    int nZ = static_cast<int>(zonePtr.size()) - 1;
    if (nZ <= 0) {
        return 0;
    }
    // OLF solves ONE system over every enabled controller of every zone, so the size is the total
    // controller count. The batched LU holds an N*N matrix per scenario, which is what actually bounds
    // this: refuse when that workspace would not fit rather than at an arbitrary controller count.
    double matBytes = static_cast<double>(ctx->nScenarios) * nC * nC * sizeof(double);
    if (matBytes > SVC_MAT_BUDGET_BYTES) {
        std::fprintf(stderr, "SVC in-kernel declined: %d controllers x %d scenarios needs %.1f MB of zone-system "
                     "workspace (budget %.0f MB)\n", nC, ctx->nScenarios, matBytes / 1048576.0,
                     SVC_MAT_BUDGET_BYTES / 1048576.0);
        return 0;
    }
    for (int e : qElem) {
        if (e < 0) {
            return 0;                                        // an observable the batch does not pack
        }
    }
    ctx->dSvcCtldVRow.upload(ctx->hSvcCtldVRow.data(), nC * sizeof(int));
    ctx->dSvcBusVRow.upload(busVRow.data(), nC * sizeof(int));
    ctx->dSvcRlIdx.upload(rlIdx.data(), nC * sizeof(int));
    ctx->dSvcBaseTargetV.upload(ctx->hSvcBaseTargetV.data(), nC * sizeof(double));
    ctx->dSvcMinQ.upload(minQ.data(), nC * sizeof(double));
    ctx->dSvcMaxQ.upload(maxQ.data(), nC * sizeof(double));
    ctx->dSvcLoadQ.upload(loadQ.data(), nC * sizeof(double));
    ctx->dSvcQPtr.upload(qPtr.data(), qPtr.size() * sizeof(int));
    ctx->dSvcQKind.upload(qKind.data(), std::max<size_t>(qKind.size(), 1) * sizeof(int));
    ctx->dSvcQElem.upload(qElem.data(), std::max<size_t>(qElem.size(), 1) * sizeof(int));
    ctx->dSvcQBRow.upload(qBRow.data(), std::max<size_t>(qBRow.size(), 1) * sizeof(int));
    ctx->dSvcZonePtr.upload(zonePtr.data(), zonePtr.size() * sizeof(int));
    ctx->dSvcZoneCtl.upload(zoneCtl.data(), zoneCtl.size() * sizeof(int));
    ctx->dSvcZonePilotRow.upload(zonePilotRow.data(), nZ * sizeof(int));
    ctx->dSvcZoneTargetV.upload(zoneTargetV.data(), nZ * sizeof(double));
    size_t nS = ctx->nScenarios;
    ctx->dSvcTargetV.ensure(nS * nC * sizeof(double));
    ctx->dSvcK0.ensure(nS * nC * sizeof(double));
    ctx->dSvcJk.ensure(nS * nC * nC * sizeof(double));
    ctx->dSvcJvpp.ensure(static_cast<size_t>(nS) * static_cast<size_t>(nZ) * nC * sizeof(double));   // per (zone pilot, controlled bus)
    ctx->dSvcEnabled.ensure(nS * nC * sizeof(char));
    ctx->dSvcNeed.ensure(nS * sizeof(char));
    ctx->dSvcReject.ensure(nS * sizeof(char));
    ctx->dSvcRlVt.ensure(nS * std::max(ctx->nrl, 1) * sizeof(double));
    ctx->dSvcMat.ensure(nS * static_cast<size_t>(nC) * nC * sizeof(double));
    ctx->dSvcRhs.ensure(nS * nC * sizeof(double));
    ctx->dSvcMatPtr.ensure(nS * sizeof(double*));
    ctx->dSvcRhsPtr.ensure(nS * sizeof(double*));
    ctx->dSvcPiv.ensure(nS * nC * sizeof(int));
    ctx->dSvcInfo.ensure(nS * sizeof(int));
    {
        std::vector<int> zoneOf(nC, 0);                      // controller -> its zone, for the assembly
        for (int z = 0; z < nZ; z++) {
            for (int k = zonePtr[z]; k < zonePtr[z + 1]; k++) {
                zoneOf[zoneCtl[k]] = z;
            }
        }
        ctx->dSvcCtlZone.upload(zoneOf.data(), zoneOf.size() * sizeof(int));
    }
    {
        // NaN = "this zone has not moved the controller's target", which is what rlTargetOf tests. The
        // reactive-limit SEEDING reads this buffer before the per-solve reset further down runs, so an
        // uninitialised one hands the switch-back a garbage comparison voltage and un-pins controllers
        // that should stay pinned (measured on ieee14's z1: pins 4 -> 2 instead of 4 -> 6, and 1.0e-10 kV).
        std::vector<double> nanv(nS * std::max(ctx->nrl, 1), std::nan(""));
        CUDA_CHECK(cudaMemcpy(ctx->dSvcRlVt.p, nanv.data(), nanv.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    ctx->svcMinPlausibleV = minPlausibleV;
    ctx->svcMaxPlausibleV = maxPlausibleV;
    ctx->nSvcZones = nZ;
    ctx->nSvc = nC;
    return 1;
}

/** Un-pin reactive-limit entries at a driver pause (SecondaryVoltageControl's re-enable step):
 *  {@code pairs} is (scenario, RL entry) flattened, {@code newTarget} the voltage target each row goes
 *  back to. No-op for an entry that is not pinned. */
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlUnpin(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs, jdoubleArray jTargets) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nrl <= 0 || jPairs == nullptr || jTargets == nullptr) {
        return;
    }
    int nPairs = env->GetArrayLength(jTargets);
    if (nPairs == 0 || env->GetArrayLength(jPairs) != 2 * nPairs) {
        return;
    }
    std::vector<int> pairs(2 * nPairs);
    std::vector<double> targets(nPairs);
    env->GetIntArrayRegion(jPairs, 0, 2 * nPairs, pairs.data());
    env->GetDoubleArrayRegion(jTargets, 0, nPairs, targets.data());
    DeviceBuf dPairs;
    DeviceBuf dTargets;
    dPairs.upload(pairs.data(), pairs.size() * sizeof(int));
    dTargets.upload(targets.data(), targets.size() * sizeof(double));
    rlUnpinBatch<<<grid(nPairs), BLOCK, 0, ctx->stream>>>(nPairs, ctx->nScenarios, ctx->n, ctx->nrl,
            dPairs.as<int>(), dTargets.as<double>(), ctx->dRlVrow.as<int>(), ctx->dRlLimitType.as<char>(),
            ctx->dRlSavedMode.as<char>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>());
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
}

/** Did the last solve stop at a host-owned slot? Returns 0 if not paused, or the
 *  DeviceLoopKind value (LOOP_HOST_SVC=5, LOOP_HOST_PHASE=6, ...) of the paused slot. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchDriverPaused(
        JNIEnv*, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    return (ctx == nullptr || !ctx->driverPaused) ? 0 : ctx->pausedLoopKind;
}

/** Resume the paused driver: {@code hostChanged[s] != 0} is the host loop's UNSTABLE verdict for
 *  that scenario, which drives the next pass exactly as a device loop's verdict would. */
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDriverResume(
        JNIEnv* env, jclass, jlong handle, jbyteArray jHostChanged) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->hostChanged.assign(ctx->nScenarios, 0);
    if (jHostChanged != nullptr) {
        jsize len = std::min<jsize>(env->GetArrayLength(jHostChanged), ctx->nScenarios);
        env->GetByteArrayRegion(jHostChanged, 0, len, reinterpret_cast<jbyte*>(ctx->hostChanged.data()));
    }
    ctx->driverResume = 1;
}

/** Drop a paused driver sequence without resuming it: the host loop changed the equation layout or
 *  the row modes, so the next solveBatchContext must START OVER on the re-extracted system. */
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_cancelBatchDriverPause(
        JNIEnv*, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->driverPaused = 0;
    ctx->driverResume = 0;
    ctx->pausedLoopKind = 0;
}

/** The next FRESH solve resumes its sweep at this loop kind after its first solve (a restarted
 *  sequence: OLF re-checks the loop that went UNSTABLE before restarting the sweep). The context may
 *  have been rebuilt since the pause, so the Java side carries the kind across. */
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDriverStartLoop(
        JNIEnv*, jclass, jlong handle, jint kind) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->startLoopKind = kind;
}

/** Write absolute target values at (scenario, row) positions — how a host-owned loop applies its
 *  decision without disturbing the device-side targets the sequence has already accumulated. */
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchTargetDeltas(
        JNIEnv* env, jclass, jlong handle, jintArray jRows, jdoubleArray jVals) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || jRows == nullptr) {
        return;
    }
    jsize cnt = env->GetArrayLength(jRows);
    if (cnt == 0) {
        return;
    }
    std::vector<int> rows(cnt);
    std::vector<double> vals(cnt);
    env->GetIntArrayRegion(jRows, 0, cnt, rows.data());
    env->GetDoubleArrayRegion(jVals, 0, cnt, vals.data());
    DeviceBuf dR;
    DeviceBuf dV;
    dR.upload(rows.data(), (size_t) cnt * sizeof(int));
    dV.upload(vals.data(), (size_t) cnt * sizeof(double));
    applyTargetOverrideBatch<<<grid(cnt), BLOCK, 0, ctx->stream>>>(cnt, dR.as<int>(), dV.as<double>(),
                                                                  ctx->dTarget.as<double>());
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
}

/** MonitoringVoltage controller table (OLF order 6). Empty array clears it. */
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchMonitoringLoops(
        JNIEnv* env, jclass, jlong handle, jintArray jVRow, jdoubleArray jBand,
        jintArray jDeltaPtr, jintArray jDeltaRow, jintArray jDeltaMode, jdoubleArray jDeltaTarget) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jVRow == nullptr || env->GetArrayLength(jVRow) == 0) {
        ctx->nmon = 0;
        return;
    }
    auto upI = [&](jintArray a, DeviceBuf& b) {
        jint* q = env->GetIntArrayElements(a, nullptr);
        b.upload(q, (size_t) env->GetArrayLength(a) * sizeof(int));
        env->ReleaseIntArrayElements(a, q, JNI_ABORT);
    };
    auto upD = [&](jdoubleArray a, DeviceBuf& b) {
        jdouble* q = env->GetDoubleArrayElements(a, nullptr);
        b.upload(q, (size_t) env->GetArrayLength(a) * sizeof(double));
        env->ReleaseDoubleArrayElements(a, q, JNI_ABORT);
    };
    upI(jVRow, ctx->dMonVRow);
    upD(jBand, ctx->dMonBand);
    upI(jDeltaPtr, ctx->dMonDeltaPtr);
    upI(jDeltaRow, ctx->dMonDeltaRow);
    upI(jDeltaMode, ctx->dMonDeltaMode);
    upD(jDeltaTarget, ctx->dMonDeltaTarget);
    ctx->nmon = env->GetArrayLength(jVRow);
    ctx->dMonActivated.alloc(static_cast<size_t>(ctx->nmon) * ctx->nScenarios * sizeof(double));
}

/** How many (controller, scenario) monitoring automatons FIRED in the last solve. The host keeps
 *  the ACTIVATIONS counter meaningful now that the loop is a driver phase rather than an
 *  orchestrator — it is the coverage guard that a test case really exercises an activation. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchMonitoringActivations(
        JNIEnv*, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nmon == 0 || !ctx->dMonActivated.p) {
        return 0;
    }
    size_t cnt = static_cast<size_t>(ctx->nmon) * ctx->nScenarios;
    std::vector<double> host(cnt);
    cudaMemcpy(host.data(), ctx->dMonActivated.p, cnt * sizeof(double), cudaMemcpyDeviceToHost);
    jint fired = 0;
    for (double v : host) {
        if (!std::isnan(v)) {
            fired++;
        }
    }
    return fired;
}

/** Read the HVDC statuses back after a solve, so the host keeps its own view in step. */
extern "C" JNIEXPORT jintArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchHvdcStatus(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nhv == 0) {
        return nullptr;
    }
    size_t len = (size_t) ctx->nScenarios * ctx->nhv;
    std::vector<int> host(len);
    if (cudaMemcpy(host.data(), ctx->dHvStatus.p, len * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return nullptr;
    }
    jintArray out = env->NewIntArray((jsize) len);
    env->SetIntArrayRegion(out, 0, (jsize) len, host.data());
    return out;
}

// Reactive-limits controllers STRANDED in another synchronous component: install (or clear, with
// null) the per-scenario mask as {slot, controller} pairs.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlNoUnpin(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jPairs == nullptr || env->GetArrayLength(jPairs) == 0 || ctx->nrl == 0) {
        ctx->nRlNoUnpin = 0;
        return;
    }
    size_t len = static_cast<size_t>(ctx->nScenarios) * ctx->nrl;
    std::vector<char> mask(len, 0);
    int nPairs = env->GetArrayLength(jPairs) / 2;
    jint* pairs = env->GetIntArrayElements(jPairs, nullptr);
    for (int k = 0; k < nPairs; k++) {
        int s = pairs[2 * k];
        int i = pairs[2 * k + 1];
        if (s >= 0 && s < ctx->nScenarios && i >= 0 && i < ctx->nrl) {
            mask[(size_t) s * ctx->nrl + i] = 1;
        }
    }
    env->ReleaseIntArrayElements(jPairs, pairs, JNI_ABORT);
    ctx->dRlNoUnpin.upload(mask.data(), len);
    std::vector<char> zero(ctx->nScenarios, 0);
    ctx->dRlUnpinBad.upload(zero.data(), ctx->nScenarios);
    ctx->nRlNoUnpin = nPairs;
}

// (scenario, entry) pairs whose bus lost EVERY generator this scenario: seeded FROZEN by initRlLimitTypeBatch.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlGenLost(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jPairs == nullptr || env->GetArrayLength(jPairs) == 0 || ctx->nrl == 0) {
        ctx->nRlGenLost = 0;
        return;
    }
    size_t len = static_cast<size_t>(ctx->nScenarios) * ctx->nrl;
    std::vector<char> mask(len, 0);
    int nPairs = env->GetArrayLength(jPairs) / 2;
    jint* pairs = env->GetIntArrayElements(jPairs, nullptr);
    for (int k = 0; k < nPairs; k++) {
        int s = pairs[2 * k];
        int i = pairs[2 * k + 1];
        if (s >= 0 && s < ctx->nScenarios && i >= 0 && i < ctx->nrl) {
            mask[(size_t) s * ctx->nrl + i] = 1;
        }
    }
    if (std::getenv("OLF_GPU_GENLOST_TRACE") != nullptr) {
        for (int k = 0; k < nPairs; k++) {
            std::fprintf(stderr, "GENLOST slot=%d entry=%d\n", pairs[2 * k], pairs[2 * k + 1]);
        }
    }
    env->ReleaseIntArrayElements(jPairs, pairs, JNI_ABORT);
    ctx->dRlGenLost.upload(mask.data(), len);
    ctx->nRlGenLost = nPairs;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlIslanded(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jPairs == nullptr || env->GetArrayLength(jPairs) == 0 || ctx->nrl == 0) {
        ctx->nRlIslanded = 0;
        return;
    }
    size_t maskLen = static_cast<size_t>(ctx->nScenarios) * ctx->nrl;
    std::vector<char> mask(maskLen, 0);
    int nPairs = env->GetArrayLength(jPairs) / 2;
    jint* pairs = env->GetIntArrayElements(jPairs, nullptr);
    for (int k = 0; k < nPairs; k++) {
        int s = pairs[2 * k];
        int i = pairs[2 * k + 1];
        if (s >= 0 && s < ctx->nScenarios && i >= 0 && i < ctx->nrl) {
            mask[(size_t) s * ctx->nrl + i] = 1;
        }
    }
    env->ReleaseIntArrayElements(jPairs, pairs, JNI_ABORT);
    ctx->dRlIslanded.upload(mask.data(), maskLen);
    ctx->nRlIslanded = nPairs;
}

// Re-homed voltage controls (GpuBatchQualification.reHomedControls): per (slot, entry) pair the entry is
// checked by the reactive-limits pass as a LOCAL controller at the given voltage target (pu) — the bus is
// a plain PV bus this scenario, whatever its base packing. Null clears the context's list.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlLocal(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs, jdoubleArray jVt) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jPairs == nullptr || jVt == nullptr || env->GetArrayLength(jPairs) == 0 || ctx->nrl == 0) {
        ctx->nRlLocal = 0;
        return;
    }
    size_t len = static_cast<size_t>(ctx->nScenarios) * ctx->nrl;
    std::vector<char> mask(len, 0);
    std::vector<double> vt(len, 0.0);
    int nPairs = env->GetArrayLength(jPairs) / 2;
    jint* pairs = env->GetIntArrayElements(jPairs, nullptr);
    jdouble* vts = env->GetDoubleArrayElements(jVt, nullptr);
    for (int k = 0; k < nPairs; k++) {
        int s = pairs[2 * k];
        int i = pairs[2 * k + 1];
        if (s >= 0 && s < ctx->nScenarios && i >= 0 && i < ctx->nrl) {
            mask[(size_t) s * ctx->nrl + i] = 1;
            vt[(size_t) s * ctx->nrl + i] = vts[k];
        }
    }
    env->ReleaseIntArrayElements(jPairs, pairs, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jVt, vts, JNI_ABORT);
    ctx->dRlLocalOv.upload(mask.data(), len);
    ctx->dRlLocalVt.upload(vt.data(), len * sizeof(double));
    ctx->nRlLocal = nPairs;
}

// Remote controllers OLF re-merges onto a NEW pilot after a zero-impedance split: per (scenario, entry) the V row
// the reactive-limit switch-back compares against the (per-scenario) group target - the new pilot's, not the base
// pilot's whose row is frozen with its island. Null or empty clears the context's list.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlCmpOverrides(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs, jintArray jCmp) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jPairs == nullptr || jCmp == nullptr || env->GetArrayLength(jPairs) == 0 || ctx->nrl == 0) {
        ctx->nRlCmp = 0;
        return;
    }
    size_t len = static_cast<size_t>(ctx->nScenarios) * ctx->nrl;
    std::vector<int> cmp(len, -1);
    int nPairs = env->GetArrayLength(jPairs) / 2;
    jint* pairs = env->GetIntArrayElements(jPairs, nullptr);
    jint* rows = env->GetIntArrayElements(jCmp, nullptr);
    for (int k = 0; k < nPairs; k++) {
        int s = pairs[2 * k];
        int i = pairs[2 * k + 1];
        if (s >= 0 && s < ctx->nScenarios && i >= 0 && i < ctx->nrl) {
            cmp[(size_t) s * ctx->nrl + i] = rows[k];
        }
    }
    env->ReleaseIntArrayElements(jPairs, pairs, JNI_ABORT);
    env->ReleaseIntArrayElements(jCmp, rows, JNI_ABORT);
    ctx->dRlCmpOv.upload(cmp.data(), len * sizeof(int));
    ctx->nRlCmp = nPairs;
}

// Per-scenario member targetQ overrides for the DISTR_Q target kernel: {key, value} where key is
// slot * nDqMembers + memberIndex. Set when an outaged bus is a packed group member whose own targetQ
// moved (a reactive LOAD loss). Nulls clear.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDqMemberTargetQOverrides(
        JNIEnv* env, jclass, jlong handle, jintArray jKeys, jdoubleArray jVals) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jKeys == nullptr || env->GetArrayLength(jKeys) == 0) {
        ctx->nDqTqOv = 0;
        return;
    }
    int ne = env->GetArrayLength(jKeys);
    jint* keys = env->GetIntArrayElements(jKeys, nullptr);
    jdouble* vals = env->GetDoubleArrayElements(jVals, nullptr);
    ctx->dDqTqOvKey.upload(keys, static_cast<size_t>(ne) * sizeof(int));
    ctx->dDqTqOvVal.upload(vals, static_cast<size_t>(ne) * sizeof(double));
    ctx->nDqTqOv = ne;
    env->ReleaseIntArrayElements(jKeys, keys, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jVals, vals, JNI_ABORT);
}

// Per-scenario moves of a pilot's captured reactive target (applyPilotQDeltasBatch): {key, delta} with key
// slot * nQr + group, one entry per key. Nulls clear.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchPilotQDeltas(
        JNIEnv* env, jclass, jlong handle, jintArray jKeys, jdoubleArray jVals) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jKeys == nullptr || env->GetArrayLength(jKeys) == 0) {
        ctx->nQrPqDelta = 0;
        return;
    }
    int ne = env->GetArrayLength(jKeys);
    jint* keys = env->GetIntArrayElements(jKeys, nullptr);
    jdouble* vals = env->GetDoubleArrayElements(jVals, nullptr);
    ctx->dQrPqDeltaKey.upload(keys, static_cast<size_t>(ne) * sizeof(int));
    ctx->dQrPqDeltaVal.upload(vals, static_cast<size_t>(ne) * sizeof(double));
    ctx->nQrPqDelta = ne;
    env->ReleaseIntArrayElements(jKeys, keys, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jVals, vals, JNI_ABORT);
}

// Per slot, the row the distributed-slack kernels read the slack on (host row, -1 = the base slack row): a
// contingency that isolates the base slack makes OLF re-pick one, whose angle row becomes the reference. Null clears.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDsSlackRows(
        JNIEnv* env, jclass, jlong handle, jintArray jRows) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jRows == nullptr || env->GetArrayLength(jRows) == 0) {
        ctx->hasDsSlackRowOv = 0;
        return;
    }
    int ne = env->GetArrayLength(jRows);
    std::vector<int> rows(ne);
    env->GetIntArrayRegion(jRows, 0, ne, rows.data());
    for (int& r : rows) {
        r = r >= 0 ? ctx->st->rowMap[r] : -1;
    }
    ctx->dDsSlackRowOv.upload(rows.data(), static_cast<size_t>(ne) * sizeof(int));
    ctx->hasDsSlackRowOv = 1;
}

// Pilot-row flip entries (see pilotRowAllPinnedBatch): one per (scenario, pilot) where the outage
// made the pilot stop being one of its own controllers. slots/rows/qTargets are per entry; rlCounts
// says how many surviving-controller RL entries each one owns and rlPool concatenates them in order.
// Nulls clear. Flip state (flipped/savedVTarget) is allocated zeroed here, so each solve starts with
// every pilot row unflipped.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchPilotAllPinnedFlip(
        JNIEnv* env, jclass, jlong handle, jintArray jSlots, jintArray jRows, jdoubleArray jQTargets,
        jintArray jRlCounts, jintArray jRlPool) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jSlots == nullptr || env->GetArrayLength(jSlots) == 0) {
        ctx->nPilotFlip = 0;
        return;
    }
    int ne = env->GetArrayLength(jSlots);
    jint* slots = env->GetIntArrayElements(jSlots, nullptr);
    jint* rows = env->GetIntArrayElements(jRows, nullptr);
    jdouble* qt = env->GetDoubleArrayElements(jQTargets, nullptr);
    jint* cnt = env->GetIntArrayElements(jRlCounts, nullptr);
    jint* pool = env->GetIntArrayElements(jRlPool, nullptr);
    int poolLen = env->GetArrayLength(jRlPool);
    std::vector<int> off(ne);
    int acc = 0;
    for (int i = 0; i < ne; i++) {
        off[i] = acc;
        acc += cnt[i];
    }
    ctx->dPfSlot.upload(slots, static_cast<size_t>(ne) * sizeof(int));
    ctx->dPfRow.upload(rows, static_cast<size_t>(ne) * sizeof(int));
    ctx->dPfQTarget.upload(qt, static_cast<size_t>(ne) * sizeof(double));
    ctx->dPfCount.upload(cnt, static_cast<size_t>(ne) * sizeof(int));
    ctx->dPfOff.upload(off.data(), static_cast<size_t>(ne) * sizeof(int));
    ctx->dPfRlPool.upload(pool, static_cast<size_t>(poolLen > 0 ? poolLen : 1) * sizeof(int));
    std::vector<char> zeroC(ne, 0);
    std::vector<double> zeroD(ne, 0.0);
    ctx->dPfFlipped.upload(zeroC.data(), static_cast<size_t>(ne) * sizeof(char));
    ctx->dPfSavedVT.upload(zeroD.data(), static_cast<size_t>(ne) * sizeof(double));
    ctx->nPilotFlip = ne;
    env->ReleaseIntArrayElements(jSlots, slots, JNI_ABORT);
    env->ReleaseIntArrayElements(jRows, rows, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jQTargets, qt, JNI_ABORT);
    env->ReleaseIntArrayElements(jRlCounts, cnt, JNI_ABORT);
    env->ReleaseIntArrayElements(jRlPool, pool, JNI_ABORT);
}

// Distributed-slack participants STRANDED in another synchronous component: install (or clear, with
// null) the per-scenario mask as {slot, participant} pairs. A contingency that fragments the network
// leaves some participating buses outside the main component; OLF drops them from the participant
// set, so the device must renormalize over the survivors — see distributeSlackBatch.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDsIslanded(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jPairs == nullptr || env->GetArrayLength(jPairs) == 0 || ctx->nds == 0) {
        ctx->nDsIslanded = 0;
        return;
    }
    // Expanded to a dense (scenario, participant) mask on the host: the kernel reads it once per
    // participant inside loops it already walks, so a scan over a pair list would be the only cost
    // the mask adds to scenarios that have no islanded participant at all.
    size_t maskLen = static_cast<size_t>(ctx->nScenarios) * ctx->nds;
    std::vector<char> mask(maskLen, 0);
    int nPairs = env->GetArrayLength(jPairs) / 2;
    jint* pairs = env->GetIntArrayElements(jPairs, nullptr);
    for (int k = 0; k < nPairs; k++) {
        int s = pairs[2 * k];
        int i = pairs[2 * k + 1];
        if (s >= 0 && s < ctx->nScenarios && i >= 0 && i < ctx->nds) {
            mask[(size_t) s * ctx->nds + i] = 1;
        }
    }
    env->ReleaseIntArrayElements(jPairs, pairs, JNI_ABORT);
    ctx->dDsIslanded.upload(mask.data(), maskLen);
    ctx->nDsIslanded = nPairs;
}

// Per-scenario P-balance row of a DS participant this scenario MOVES: {slot, participant, row}
// triples, expanded to a dense (scenario, participant) row array (-1 = the structural dsPhiRow).
// The rows arrive in the same space dsPhiRow is given in and are taken through the structure's
// rowMap here, exactly as createBatchContext maps dsPhiRow itself — so the two are always in one
// space whatever rowMap is.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDsRowOverrides(
        JNIEnv* env, jclass, jlong handle, jintArray jTriples) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jTriples == nullptr || env->GetArrayLength(jTriples) == 0 || ctx->nds == 0) {
        ctx->nDsRowOv = 0;
        return;
    }
    size_t len = static_cast<size_t>(ctx->nScenarios) * ctx->nds;
    std::vector<int> rows(len, -1);
    int nTriples = env->GetArrayLength(jTriples) / 3;
    jint* t = env->GetIntArrayElements(jTriples, nullptr);
    const std::vector<int>& rowMap = ctx->st->rowMap;
    int nOk = 0;
    for (int k = 0; k < nTriples; k++) {
        int s = t[3 * k];
        int i = t[3 * k + 1];
        int row = t[3 * k + 2];
        if (s >= 0 && s < ctx->nScenarios && i >= 0 && i < ctx->nds
                && row >= 0 && row < static_cast<int>(rowMap.size()) && rowMap[row] >= 0) {
            rows[(size_t) s * ctx->nds + i] = rowMap[row];
            nOk++;
        }
    }
    env->ReleaseIntArrayElements(jTriples, t, JNI_ABORT);
    ctx->dDsRowOv.upload(rows.data(), len * sizeof(int));
    ctx->nDsRowOv = nOk;
}

// DISTR_RHO coefficient overrides: install (or clear, with nulls) the per-scenario 1/count'
// renormalization pairs {s, contrib} + new coefficients on a batch context (a shared-TVC group
// member outaged in that scenario — plan-group-member-dynamics A5).
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDrCoefOverrides(
        JNIEnv* env, jclass, jlong handle, jintArray jIdx, jdoubleArray jC) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jIdx == nullptr || jC == nullptr || env->GetArrayLength(jC) == 0) {
        ctx->nDrOv = 0;
        return;
    }
    int nOv = env->GetArrayLength(jC);
    jint* idx = env->GetIntArrayElements(jIdx, nullptr);
    jdouble* c = env->GetDoubleArrayElements(jC, nullptr);
    ctx->dDrOvIdx.upload(idx, static_cast<size_t>(nOv) * 2 * sizeof(int));
    ctx->dDrOvC.upload(c, static_cast<size_t>(nOv) * sizeof(double));
    env->ReleaseIntArrayElements(jIdx, idx, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jC, c, JNI_ABORT);
    ctx->nDrOv = nOv;
}

// Per-scenario ACTIVE equation counts for the L2 stopping threshold (see hNActive). Per-chunk state.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchActiveRowCounts(
        JNIEnv* env, jclass, jlong handle, jintArray jNActive) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->hNActive.clear();
    if (jNActive != nullptr) {
        int len = env->GetArrayLength(jNActive);
        ctx->hNActive.resize(len);
        env->GetIntArrayRegion(jNActive, 0, len, ctx->hNActive.data());
    }
}


/** Install the DISCRETE transformer voltage-control tables (OLF order 8): mode 0 = AFTER_GENERATOR
 *  (TransformerVoltageControlOuterLoop), 1 = WITH_GENERATOR (SimpleTransformerVoltageControlOuterLoop);
 *  entryOn = the uploaded activation already carries the control ON (a base load flow whose network
 *  OLF's initialize switched on). Groups = controlled buses (CSR over members), each with its toggle-
 *  diffed ON/OFF row deltas (CSR); members with rho row, element, base tap, tap table, r1 range and
 *  initial r1. Returns 1 when the kernel loop owns them. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDiscreteTvc(
        JNIEnv* env, jclass, jlong handle, jint mode, jint entryOn,
        jintArray jGrpPtr, jintArray jGrpVRow, jdoubleArray jGrpTargetV, jdoubleArray jGrpHalfDb, jintArray jGrpUseInit,
        jintArray jDeltaPtr, jintArray jDeltaRow, jintArray jDeltaModeOn, jintArray jDeltaModeOff,
        jdoubleArray jDeltaTargetOn, jdoubleArray jDeltaTargetOff,
        jintArray jMemRhoRow, jintArray jMemElem, jintArray jMemBaseTap, jintArray jMemTapPtr,
        jdoubleArray jMemMin, jdoubleArray jMemMax, jdoubleArray jMemInit, jdoubleArray jTapData) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nTvd = 0;
    ctx->nTvdFz = 0;
    ctx->tvdFixTables = false;
    ctx->tvdFixWords = 0;
    if (jMemRhoRow == nullptr || env->GetArrayLength(jMemRhoRow) == 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    auto readD = [&](jdoubleArray a, std::vector<double>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetDoubleArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> grpPtr, grpVRow, grpUseInit, deltaPtr, deltaRow, deltaModeOn, deltaModeOff, memRhoRow;
    std::vector<double> grpTargetV, grpHalfDb, deltaTargetOn, deltaTargetOff, memMin, memMax, memInit;
    readI(jGrpPtr, grpPtr);
    readI(jGrpVRow, grpVRow);
    readD(jGrpTargetV, grpTargetV);
    readD(jGrpHalfDb, grpHalfDb);
    readI(jGrpUseInit, grpUseInit);
    readI(jDeltaPtr, deltaPtr);
    readI(jDeltaRow, deltaRow);
    readI(jDeltaModeOn, deltaModeOn);
    readI(jDeltaModeOff, deltaModeOff);
    readD(jDeltaTargetOn, deltaTargetOn);
    readD(jDeltaTargetOff, deltaTargetOff);
    readI(jMemRhoRow, memRhoRow);
    readI(jMemElem, ctx->hTvdMemElem);
    readI(jMemBaseTap, ctx->hTvdMemBaseTap);
    readI(jMemTapPtr, ctx->hTvdMemTapPtr);
    readD(jMemMin, memMin);
    readD(jMemMax, memMax);
    readD(jMemInit, memInit);
    readD(jTapData, ctx->hTvdTapData);
    int nM = static_cast<int>(memRhoRow.size());
    int nG = static_cast<int>(grpPtr.size()) - 1;
    int nD = static_cast<int>(deltaRow.size());
    for (int m = 0; m < nM; m++) {
        if (ctx->hTvdMemTapPtr[m + 1] - ctx->hTvdMemTapPtr[m] <= 0) {
            return 0;                                        // no tap table: the host orchestrator keeps the control
        }
    }
    ctx->dTvdGrpPtr.upload(grpPtr.data(), grpPtr.size() * sizeof(int));
    ctx->dTvdGrpVRow.upload(grpVRow.data(), nG * sizeof(int));
    ctx->dTvdGrpTargetV.upload(grpTargetV.data(), nG * sizeof(double));
    ctx->dTvdGrpHalfDb.upload(grpHalfDb.data(), nG * sizeof(double));
    ctx->dTvdGrpUseInit.upload(grpUseInit.data(), nG * sizeof(int));
    ctx->dTvdDeltaPtr.upload(deltaPtr.data(), deltaPtr.size() * sizeof(int));
    ctx->dTvdDeltaRow.upload(deltaRow.data(), std::max(nD, 1) * sizeof(int));
    ctx->dTvdDeltaModeOn.upload(deltaModeOn.data(), std::max(nD, 1) * sizeof(int));
    ctx->dTvdDeltaModeOff.upload(deltaModeOff.data(), std::max(nD, 1) * sizeof(int));
    ctx->dTvdDeltaTargetOn.upload(deltaTargetOn.data(), std::max(nD, 1) * sizeof(double));
    ctx->dTvdDeltaTargetOff.upload(deltaTargetOff.data(), std::max(nD, 1) * sizeof(double));
    ctx->dTvdMemRhoRow.upload(memRhoRow.data(), nM * sizeof(int));
    ctx->dTvdMemElem.upload(ctx->hTvdMemElem.data(), nM * sizeof(int));
    ctx->dTvdMemBaseTap.upload(ctx->hTvdMemBaseTap.data(), nM * sizeof(int));
    ctx->dTvdMemTapPtr.upload(ctx->hTvdMemTapPtr.data(), ctx->hTvdMemTapPtr.size() * sizeof(int));
    ctx->dTvdMemMin.upload(memMin.data(), nM * sizeof(double));
    ctx->dTvdMemMax.upload(memMax.data(), nM * sizeof(double));
    ctx->dTvdMemInit.upload(memInit.data(), nM * sizeof(double));
    ctx->dTvdTapData.upload(ctx->hTvdTapData.data(), ctx->hTvdTapData.size() * sizeof(double));
    size_t nS = ctx->nScenarios;
    ctx->dTvdScnStep.ensure(nS * sizeof(int));
    ctx->dTvdGrpStep.ensure(nS * nG * sizeof(int));
    ctx->dTvdGrpShMin.ensure(nS * nG * sizeof(double));
    ctx->dTvdGrpShMax.ensure(nS * nG * sizeof(double));
    ctx->dTvdGrpShInit.ensure(nS * nG * sizeof(double));
    ctx->dTvdMemState.ensure(nS * nM * sizeof(int));
    ctx->dTvdMemTap.ensure(nS * nM * sizeof(int));
    ctx->dTvdMemRoundFrom.ensure(nS * nM * sizeof(double));
    ctx->dTvdSaved.ensure(nS * std::max(nD, 1) * sizeof(double));
    ctx->dTvdReject.ensure(nS * sizeof(char));
    ctx->nTvdGroups = nG;
    ctx->nTvdDelta = nD;
    ctx->tvdMode = mode;
    ctx->tvdEntryOn = entryOn;
    ctx->nTvd = nM;
    return 1;
}

/** GeneratorVoltageControlManager tables for the AFTER_GENERATOR phase installed by setBatchDiscreteTvc:
 *  the freeze candidates (controller bus V row, load Q) and fixTransformerVoltageControls' per-member
 *  PV-candidate row lists (CSR, [nM+1]; null = no connectivity fix). Returns 1 when installed. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDiscreteTvcFreeze(
        JNIEnv* env, jclass, jlong handle, jintArray jFzVRow, jdoubleArray jFzLoadQ, jintArray jFzRlEntry,
        jintArray jFixPvRow) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nTvd <= 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> vRow, fixRow;
    std::vector<double> loadQ;
    if (jFzVRow != nullptr) {
        readI(jFzVRow, vRow);
        loadQ.resize(env->GetArrayLength(jFzLoadQ));
        env->GetDoubleArrayRegion(jFzLoadQ, 0, static_cast<jsize>(loadQ.size()), loadQ.data());
    }
    int nFz = static_cast<int>(vRow.size());
    if (nFz > 0) {
        std::vector<int> rlEntry;
        if (jFzRlEntry != nullptr) {
            readI(jFzRlEntry, rlEntry);
        }
        if (static_cast<int>(rlEntry.size()) != nFz) {
            rlEntry.assign(nFz, -1);                         // no remote candidate
        }
        ctx->dTvdFzVRow.upload(vRow.data(), nFz * sizeof(int));
        ctx->dTvdFzLoadQ.upload(loadQ.data(), nFz * sizeof(double));
        ctx->dTvdFzRlEntry.upload(rlEntry.data(), nFz * sizeof(int));
        size_t nsc = static_cast<size_t>(ctx->nScenarios) * nFz;
        ctx->dTvdFzFrozen.ensure(nsc * sizeof(int));
        ctx->dTvdFzSavedMode.ensure(nsc * sizeof(int));
        ctx->dTvdFzSavedTarget.ensure(nsc * sizeof(double));
    }
    ctx->nTvdFz = nFz;
    ctx->tvdFixTables = false;                               // armed by setBatchDiscreteTvcFixMasks
    ctx->tvdFixWords = 0;
    if (jFixPvRow != nullptr) {
        readI(jFixPvRow, fixRow);
        if (!fixRow.empty()) {
            ctx->dTvdFixPvRow.upload(fixRow.data(), fixRow.size() * sizeof(int));
        }
    }
    return 1;
}

// The per-(scenario, member) connectivity masks of fixTransformerVoltageControls. Uploaded per CHUNK,
// unlike the tables above: OLF re-runs the check on the POST-CONTINGENCY network, so the component of a
// not-controlled side is a property of the scenario, not of the run.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDiscreteTvcFixMasks(
        JNIEnv* env, jclass, jlong handle, jintArray jMin, jintArray jMax, jint words) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nTvd <= 0) {
        return;
    }
    ctx->tvdFixTables = false;
    ctx->tvdFixWords = 0;
    if (jMin == nullptr || jMax == nullptr || words <= 0) {
        return;
    }
    size_t want = static_cast<size_t>(ctx->nScenarios) * ctx->nTvd * words;
    if (static_cast<size_t>(env->GetArrayLength(jMin)) != want
            || static_cast<size_t>(env->GetArrayLength(jMax)) != want) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      "setBatchDiscreteTvcFixMasks: array length != nScenarios * nMembers * words");
        return;
    }
    std::vector<int> mn(want), mx(want);
    env->GetIntArrayRegion(jMin, 0, static_cast<jsize>(want), mn.data());
    env->GetIntArrayRegion(jMax, 0, static_cast<jsize>(want), mx.data());
    ctx->dTvdFixMinMask.upload(mn.data(), want * sizeof(int));
    ctx->dTvdFixMaxMask.upload(mx.data(), want * sizeof(int));
    ctx->tvdFixWords = words;
    ctx->tvdFixTables = true;
}

/** Per (scenario, freeze candidate): {frozen (0/1), the row target while frozen (gen − load)} — layout
 *  {@code (s * nFz + c) * 2}; null without candidates. */
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchDiscreteTvcFreeze(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nTvd <= 0 || ctx->nTvdFz <= 0 || !ctx->dTvdFzFrozen.p) {
        return nullptr;
    }
    int n = ctx->n;
    size_t cnt = static_cast<size_t>(ctx->nTvdFz) * ctx->nScenarios;
    std::vector<int> fz(cnt), vrow(ctx->nTvdFz);
    std::vector<double> tgt(static_cast<size_t>(n) * ctx->nScenarios), out(cnt * 2);
    cudaMemcpy(fz.data(), ctx->dTvdFzFrozen.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(vrow.data(), ctx->dTvdFzVRow.p, vrow.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tgt.data(), ctx->dTarget.p, tgt.size() * sizeof(double), cudaMemcpyDeviceToHost);
    for (int s = 0; s < ctx->nScenarios; s++) {
        for (int c = 0; c < ctx->nTvdFz; c++) {
            size_t i = static_cast<size_t>(s) * ctx->nTvdFz + c;
            out[i * 2] = fz[i];
            out[i * 2 + 1] = tgt[static_cast<size_t>(s) * n + vrow[c]];
        }
    }
    jdoubleArray res = env->NewDoubleArray(static_cast<jsize>(out.size()));
    env->SetDoubleArrayRegion(res, 0, static_cast<jsize>(out.size()), out.data());
    return res;
}

/** Per (scenario, member): {state (0 off, 1 continuous, 2 fixed), tap index, the r1 fed to the
 *  rounding} — layout {@code (s * nM + m) * 3}; null without the table. */
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchDiscreteTvcState(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nTvd <= 0 || !ctx->dTvdMemState.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nTvd) * ctx->nScenarios;
    std::vector<int> st(cnt), tap(cnt);
    std::vector<double> rf(cnt), out(cnt * 3);
    cudaMemcpy(st.data(), ctx->dTvdMemState.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tap.data(), ctx->dTvdMemTap.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(rf.data(), ctx->dTvdMemRoundFrom.p, cnt * sizeof(double), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < cnt; i++) {
        out[i * 3] = st[i];
        out[i * 3 + 1] = tap[i];
        out[i * 3 + 2] = rf[i];
    }
    jdoubleArray res = env->NewDoubleArray(static_cast<jsize>(out.size()));
    env->SetDoubleArrayRegion(res, 0, static_cast<jsize>(out.size()), out.data());
    return res;
}

/** Install the DISCRETE (WITH_GENERATOR) shunt voltage-control tables (OLF order 10): groups =
 *  controlled buses (CSR over member shunts) with their toggle-diffed ON/OFF row deltas; members with
 *  SHUNT_B row, element (-1 = unpacked, never outaged) and sub-controllers (CSR) whose section-B tables,
 *  min and base positions drive LfShuntImpl.dispatchB. Returns 1 when the kernel loop owns them. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDiscreteShunt(
        JNIEnv* env, jclass, jlong handle, jint entryOn,
        jintArray jGrpPtr, jintArray jDeltaPtr, jintArray jDeltaRow, jintArray jDeltaModeOn, jintArray jDeltaModeOff,
        jdoubleArray jDeltaTargetOn, jdoubleArray jDeltaTargetOff,
        jintArray jMemBRow, jintArray jMemElem, jintArray jMemSubPtr,
        jintArray jSecPtr, jdoubleArray jSecB, jintArray jSubMin, jintArray jSubBase) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nSvd = 0;
    if (jMemBRow == nullptr || env->GetArrayLength(jMemBRow) == 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    auto readD = [&](jdoubleArray a, std::vector<double>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetDoubleArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> grpPtr, deltaPtr, deltaRow, deltaModeOn, deltaModeOff, memBRow, memElem, memSubPtr, secPtr, subMin, subBase;
    std::vector<double> deltaTargetOn, deltaTargetOff, secB;
    readI(jGrpPtr, grpPtr);
    readI(jDeltaPtr, deltaPtr);
    readI(jDeltaRow, deltaRow);
    readI(jDeltaModeOn, deltaModeOn);
    readI(jDeltaModeOff, deltaModeOff);
    readD(jDeltaTargetOn, deltaTargetOn);
    readD(jDeltaTargetOff, deltaTargetOff);
    readI(jMemBRow, memBRow);
    readI(jMemElem, memElem);
    readI(jMemSubPtr, memSubPtr);
    readI(jSecPtr, secPtr);
    readD(jSecB, secB);
    readI(jSubMin, subMin);
    readI(jSubBase, subBase);
    int nM = static_cast<int>(memBRow.size());
    int nG = static_cast<int>(grpPtr.size()) - 1;
    int nD = static_cast<int>(deltaRow.size());
    int nSubs = static_cast<int>(subBase.size());
    ctx->dSvdGrpPtr.upload(grpPtr.data(), grpPtr.size() * sizeof(int));
    ctx->dSvdDeltaPtr.upload(deltaPtr.data(), deltaPtr.size() * sizeof(int));
    ctx->dSvdDeltaRow.upload(deltaRow.data(), std::max(nD, 1) * sizeof(int));
    ctx->dSvdDeltaModeOn.upload(deltaModeOn.data(), std::max(nD, 1) * sizeof(int));
    ctx->dSvdDeltaModeOff.upload(deltaModeOff.data(), std::max(nD, 1) * sizeof(int));
    ctx->dSvdDeltaTargetOn.upload(deltaTargetOn.data(), std::max(nD, 1) * sizeof(double));
    ctx->dSvdDeltaTargetOff.upload(deltaTargetOff.data(), std::max(nD, 1) * sizeof(double));
    ctx->dSvdMemBRow.upload(memBRow.data(), nM * sizeof(int));
    ctx->dSvdMemElem.upload(memElem.data(), nM * sizeof(int));
    ctx->dSvdMemSubPtr.upload(memSubPtr.data(), memSubPtr.size() * sizeof(int));
    ctx->dSvdSecPtr.upload(secPtr.data(), secPtr.size() * sizeof(int));
    ctx->dSvdSecB.upload(secB.data(), std::max<size_t>(secB.size(), 1) * sizeof(double));
    ctx->dSvdSubMin.upload(subMin.data(), std::max(nSubs, 1) * sizeof(int));
    ctx->dSvdSubBase.upload(subBase.data(), std::max(nSubs, 1) * sizeof(int));
    size_t nS = ctx->nScenarios;
    ctx->dSvdGrpStep.ensure(nS * nG * sizeof(int));
    ctx->dSvdMemState.ensure(nS * nM * sizeof(int));
    ctx->dSvdMemContB.ensure(nS * nM * sizeof(double));
    ctx->dSvdSubPos.ensure(nS * std::max(nSubs, 1) * sizeof(int));
    ctx->dSvdSaved.ensure(nS * std::max(nD, 1) * sizeof(double));
    ctx->nSvdGroups = nG;
    ctx->nSvdDelta = nD;
    ctx->nSvdSubs = nSubs;
    ctx->svdEntryOn = entryOn;
    ctx->nSvd = nM;
    return 1;
}

/** Per (scenario, member shunt): {state (0 off, 1 continuous, 2 dispatched), the continuous b the
 *  dispatch started from} — layout {@code (s * nM + m) * 2}; null without the table. */
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchDiscreteShuntState(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nSvd <= 0 || !ctx->dSvdMemState.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nSvd) * ctx->nScenarios;
    std::vector<int> st(cnt);
    std::vector<double> cb(cnt), out(cnt * 2);
    cudaMemcpy(st.data(), ctx->dSvdMemState.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cb.data(), ctx->dSvdMemContB.p, cnt * sizeof(double), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < cnt; i++) {
        out[i * 2] = st[i];
        out[i * 2 + 1] = cb[i];
    }
    jdoubleArray res = env->NewDoubleArray(static_cast<jsize>(out.size()));
    env->SetDoubleArrayRegion(res, 0, static_cast<jsize>(out.size()), out.data());
    return res;
}

/** Per (scenario, sub-controller): the dispatched section — layout {@code s * nSubs + u}. */
extern "C" JNIEXPORT jintArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchDiscreteShuntPositions(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nSvd <= 0 || ctx->nSvdSubs <= 0 || !ctx->dSvdSubPos.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nSvdSubs) * ctx->nScenarios;
    std::vector<int> host(cnt);
    cudaMemcpy(host.data(), ctx->dSvdSubPos.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(cnt));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_destroyBatchContext(
        JNIEnv*, jclass, jlong handle) {
    delete reinterpret_cast<BatchSolverContext*>(handle);
}

// VALUES refresh addressed by the SHARED STRUCTURE handle (no solver context needed): the
// sequential batch-of-1 path reconciles tap/shunt moves the same way the seq context did —
// one extraction has ONE value set, and every context sharing the structure sees the update.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_refreshStructureValues(
        JNIEnv* env, jclass, jlong structureHandle, jdoubleArray jcbIn, jdoubleArray jobIn,
        jdoubleArray jshIn, jdoubleArray jhvIn, jdoubleArray jziIn) {
    auto* stp = reinterpret_cast<SharedStructurePtr*>(structureHandle);
    jdouble* cbIn = env->GetDoubleArrayElements(jcbIn, nullptr);
    jdouble* obIn = env->GetDoubleArrayElements(jobIn, nullptr);
    jdouble* shIn = env->GetDoubleArrayElements(jshIn, nullptr);
    jdouble* hvIn = env->GetDoubleArrayElements(jhvIn, nullptr);
    jdouble* ziIn = env->GetDoubleArrayElements(jziIn, nullptr);
    std::string error;
    try {
        if (stp == nullptr || !*stp) {
            throw std::runtime_error("invalid structure handle");
        }
        refreshStructureValues(**stp, env->GetArrayLength(jcbIn) / 8, cbIn, env->GetArrayLength(jobIn) / 7, obIn,
                               env->GetArrayLength(jshIn) / 2, shIn, env->GetArrayLength(jhvIn) / 5, hvIn,
                               env->GetArrayLength(jziIn), ziIn);
    } catch (const std::exception& e) {
        error = e.what();
    } catch (...) {
        error = "unknown native error";
    }
    env->ReleaseDoubleArrayElements(jcbIn, cbIn, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jobIn, obIn, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jshIn, shIn, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jhvIn, hvIn, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jziIn, ziIn, JNI_ABORT);
    if (!error.empty()) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      ("GPU structure values refresh failed: " + error).c_str());
    }
}

extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_recommendBatchSize(
        JNIEnv* env, jclass, jlong structureHandle, jint maxCap) {
    auto* stp = reinterpret_cast<SharedStructurePtr*>(structureHandle);
    int rec = 1;
    try {
        if (stp == nullptr || !*stp) {
            throw std::runtime_error("invalid structure handle");
        }
        SharedStructure& st = **stp;
        // Per-scenario device footprint: the dominant term is cuDSS's LU numeric factors
        // (the exact fill from a probe analysis — cached on the structure — or the heuristic),
        // plus the value/state arrays (vals + x/target/F/B/dx). The LU index structure is
        // shared across the batch.
        constexpr double SAFETY = 0.7;                       // leave headroom for cuDSS workspace + fragmentation
        double luBytes = luBytesPerScenario(st);
        size_t freeBytes = 0;
        size_t totalBytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));  // after the probe freed its buffers
        // 7n doubles per scenario, not 5: x/target/F/B/dx PLUS dQbuf (the per-scenario reactive
        // accumulator) and dXprevLs (the line-search previous state, allocated under the RTE
        // parameters' MAX_VOLTAGE_CHANGE scaling).
        double perScenarioBytes = luBytes
                + (static_cast<double>(st.nnz) + 7.0 * st.n) * sizeof(double)
                + static_cast<double>(st.n) * sizeof(int)    // dMode
                + static_cast<double>(st.nElements);         // dDisabled (char)
        double budget = static_cast<double>(freeBytes) * SAFETY;
        long fit = perScenarioBytes > 0 ? static_cast<long>(budget / perScenarioBytes) : maxCap;
        rec = static_cast<int>(std::max<long>(1, std::min<long>(maxCap, fit)));
    } catch (...) {
        rec = 1;                                             // a query failure yields the safest chunk
    }
    return rec;
}

// ---------------------------------------------------------------------------------
// ADJOINT step 1a: solve Aᵀ x = b for an EXPLICIT CSR matrix, by building the transpose
// and factorizing it.
//
// Why this exists: cuDSS (through 0.8.0 — verified, its header has no transpose/OP_T
// symbol at all) offers only the non-transposed solve against its factors, so one GPU
// factorization yields ONE direction where KLU's klu_solve/klu_tsolve pair yields two.
// Our Jacobian is stored as TRUE CSR (row = equation, col = variable) so the plain solve
// serves the forward Newton; the adjoint (Jᵀλ = ...) is therefore the direction we cannot
// get from the forward's factors, and until cuDSS ships a transpose solve we buy it with a
// second factorization of an explicitly transposed matrix.
//
// This entry point is deliberately SELF-CONTAINED (its own handle/config/data, no reuse of
// the Newton context): it isolates exactly one question — is our CSR transpose + cuDSS
// factorize/solve of Aᵀ correct? — so a failure here cannot be confused with the adjoint's
// RHS assembly or contraction. It is a validation//measurement entry, not the production
// path; the production seam should reuse the forward's structure and swap to a native
// transpose solve the moment cuDSS provides one.
//
// CSR transpose is the standard counting sort: count entries per column of A (= rows of Aᵀ),
// prefix-sum into row pointers, then scatter. O(nnz + n), host-side, and for a FIXED pattern
// the resulting scatter is a permutation that can be computed once and reused per solve.
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_solveTransposedExplicit(
        JNIEnv* env, jclass, jint n, jintArray jRowPtr, jintArray jColIdx,
        jdoubleArray jVals, jdoubleArray jB) {
    try {
        const int nn = n;
        std::vector<int> rowPtr(nn + 1);
        env->GetIntArrayRegion(jRowPtr, 0, nn + 1, rowPtr.data());
        const int nnz = rowPtr[nn];
        std::vector<int> colIdx(nnz);
        std::vector<double> vals(nnz);
        std::vector<double> b(nn);
        env->GetIntArrayRegion(jColIdx, 0, nnz, colIdx.data());
        env->GetDoubleArrayRegion(jVals, 0, nnz, vals.data());
        env->GetDoubleArrayRegion(jB, 0, nn, b.data());

        // --- CSR(A) -> CSR(Aᵀ) by counting sort ---
        std::vector<int> tRowPtr(nn + 1, 0);
        std::vector<int> tColIdx(nnz);
        std::vector<double> tVals(nnz);
        for (int k = 0; k < nnz; k++) {
            tRowPtr[colIdx[k] + 1]++;                        // column of A becomes row of Aᵀ
        }
        for (int r = 0; r < nn; r++) {
            tRowPtr[r + 1] += tRowPtr[r];
        }
        std::vector<int> cursor(tRowPtr.begin(), tRowPtr.end() - 1);
        for (int r = 0; r < nn; r++) {
            for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
                const int dst = cursor[colIdx[k]]++;
                tColIdx[dst] = r;                            // row of A becomes column of Aᵀ
                tVals[dst] = vals[k];
            }
        }

        DeviceBuf dRowPtr;
        DeviceBuf dColIdx;
        DeviceBuf dVals;
        DeviceBuf dB;
        DeviceBuf dX;
        dRowPtr.upload(tRowPtr.data(), (nn + 1) * sizeof(int));
        dColIdx.upload(tColIdx.data(), nnz * sizeof(int));
        dVals.upload(tVals.data(), nnz * sizeof(double));
        dB.upload(b.data(), nn * sizeof(double));
        dX.alloc(nn * sizeof(double));

        CudssCtx ds;
        CUDSS_CHECK(cudssCreate(&ds.handle));
        CUDSS_CHECK(cudssConfigCreate(&ds.config));
        applyReorderAlg(ds.config, false);   // adjoint: its own matrix, matching enabled
        applyMatchingAlg(ds.config);
        CUDSS_CHECK(cudssDataCreate(ds.handle, &ds.data));
        CUDSS_CHECK(cudssMatrixCreateCsr(&ds.matA, nn, nn, nnz, dRowPtr.as<int>(), nullptr,
                                         dColIdx.as<int>(), dVals.as<double>(), OLF_CUDSS_CSR_TYPES,
                                         CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        CUDSS_CHECK(cudssMatrixCreateDn(&ds.matB, nn, 1, nn, dB.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssMatrixCreateDn(&ds.matX, nn, 1, nn, dX.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        gAnalysesMain++;
        { AnalysisTimer at_(__LINE__); CUDSS_CHECK(cudssExecute(ds.handle, CUDSS_PHASE_ANALYSIS, ds.config, ds.data, ds.matA, ds.matX, ds.matB)); }
        { AnalysisTimer at_(__LINE__, 'F'); CUDSS_CHECK(cudssExecute(ds.handle, CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_SOLVE,
                                 ds.config, ds.data, ds.matA, ds.matX, ds.matB)); }
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> x(nn);
        CUDA_CHECK(cudaMemcpy(x.data(), dX.p, nn * sizeof(double), cudaMemcpyDeviceToHost));
        jdoubleArray out = env->NewDoubleArray(nn);
        env->SetDoubleArrayRegion(out, 0, nn, x.data());
        return out;
    } catch (const std::exception& e) {
        jclass ex = env->FindClass("java/lang/RuntimeException");
        if (ex) env->ThrowNew(ex, e.what());
        return nullptr;
    }
}
