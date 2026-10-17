// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See sensitivity.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/stats.cuh"
#include "../equations/branch_fill.cuh"
#include "../equations/bus_injection_fill.cuh"
#include "../equations/distribution_fill.cuh"
#include "../equations/fill_launch.cuh"
#include "../equations/zero_impedance_fill.cuh"
#include "../solver/driver_steps.cuh"
#include "../solver/zero_diagonal_repair.cuh"
#include "pilot_voltage.cuh"
#include "sensitivity.cuh"

namespace olfgpu {

// Distributed-slack pass: thread = (participating bus, scenario). The slack mismatch is
// qBuf[slackPhiRow] (= Σ branch p at the slack) − slackTargetP. If it exceeds the tolerance, each
// participating bus takes factor·mismatch into its BUS_TARGET_P target (factor normalized to sum 1, so
// the whole mismatch is absorbed), mirroring ActivePowerDistribution (one-shot, no Pmin/Pmax saturation
// — exact vs OLF when no generator saturates). A distribution marks its scenario for another inner solve.
// Freeze, on the FIRST distribution pass of a solve, the active-power share the slack/reference bus must
// RETAIN when its own generators participate: retain = (1 − Σfactor)·mismatch0. Σfactor over the emitted
// (non-slack) buses is < 1 exactly when the slack participates (its factor sits in the normalization
// denominator but emits no row); 1 − Σfactor is the slack's participation fraction ρ. Without this the
// geometric outer loop would drive the slack mismatch to 0 and leak the slack's whole share to the
// non-slack buses. ρ = 0 (non-participating slack) ⇒ retain = 0 ⇒ the slack keeps nothing extra.
// The slack/reference bus retains ρ = (1 − Σfactor) of the TOTAL active imbalance currently being shared,
// where total = the slack's residual mismatch (qBuf at slackPhiRow − slackTargetP) PLUS what the non-slack
// participating buses have already absorbed (Σ of their cumulative target deltas). Re-evaluated EVERY outer
// pass: freezing ρ·mismatch0 at the first pass ignores the AC loss change during redistribution, so the
// slack would keep ρ·mismatch0 and the non-slack side would over-absorb by ρ·Δlosses; tracking the current
// total instead makes the non-slack buses absorb exactly (1 − ρ)·total_final, matching OLF's per-pass
// distribution of the full mismatch. ρ = 0 (non-participating slack, incl. all batched gen-loss scenarios)
// ⇒ retain = 0, so this is a no-op there.
__global__ void setUnitRhsBatch(int n, int nScenarios, int unitRow, double* b) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * nScenarios) return;
    b[t] = (t % n == unitRow) ? 1.0 : 0.0;
}

__global__ void gatherSensiBatch(int n, int nScenarios, int k, int m, int ki, const int* observeCols,
                                 const double* dx, double* out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;                     // t = s*m + mi
    if (t >= nScenarios * m) return;
    int sIdx = t / m;
    int mi = t % m;
    out[(size_t) sIdx * k * m + (size_t) ki * m + mi] = dx[(size_t) sIdx * n + observeCols[mi]];
}

void sensitivitySolve(BatchSolverContext* ctx, DriverPass& drv, bool& first) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    if (ctx->mode == 6) {
        for (int s2 = 0; s2 < nS; s2++) {
            CUDSS_CHECK(cudssMatrixSetValues(ctx->ds.matA, ctx->dVals.as<double>() + (size_t) s2 * ctx->nnz));
            CUDSS_CHECK(cudssMatrixSetValues(ctx->ds.matB, ctx->dB.as<double>() + (size_t) s2 * n));
            CUDSS_CHECK(cudssMatrixSetValues(ctx->ds.matX, ctx->dDx.as<double>() + (size_t) s2 * n));
            if (!ctx->analyzed) {
                gAnalysesMain++;
                { AnalysisTimer at_(__LINE__); CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_ANALYSIS, ctx->ds.config, ctx->ds.data, ctx->ds.matA, ctx->ds.matX, ctx->ds.matB)); }
                ctx->analyzed = true;
            }
            { AnalysisTimer at_(__LINE__, 'G'); CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_SOLVE,
                    ctx->ds.config, ctx->ds.data, ctx->ds.matA, ctx->ds.matX, ctx->ds.matB)); }
        }
        first = false;
        return;
    }
    if (!ctx->analyzed) {
        gAnalysesMain++;
        { AnalysisTimer at_(__LINE__); CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_ANALYSIS, ctx->ds.config, ctx->ds.data, ctx->matUb, ctx->matUbX, ctx->matUbB)); }
        ctx->analyzed = true;
    }
    int phase = CUDSS_PHASE_SOLVE | (first ? CUDSS_PHASE_FACTORIZATION : 0);
    first = false;
    { AnalysisTimer at_(__LINE__, ((phase) & (CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_REFACTORIZATION)) ? 'G' : ((phase) & CUDSS_PHASE_ANALYSIS) ? 'A' : 'S'); CUDSS_CHECK(cudssExecute(ctx->ds.handle, phase, ctx->ds.config, ctx->ds.data, ctx->matUb, ctx->matUbX, ctx->matUbB)); }
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

}  // namespace olfgpu
