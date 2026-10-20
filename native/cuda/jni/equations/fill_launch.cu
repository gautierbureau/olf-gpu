// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See fill_launch.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/codes/gpu_row_mode.cuh"
#include "scatter.cuh"
#include "../solver/driver_steps.cuh"
#include "branch_fill.cuh"
#include "bus_injection_fill.cuh"
#include "distribution_fill.cuh"
#include "zero_impedance_fill.cuh"
#include "../outerloop/pilot_voltage.cuh"
#include "fill_launch.cuh"

namespace olfgpu {

// The GROUP relocation tables of one family, or all-null when the structure has none (plan-A6).
RelocTables relocOf(Family& f, SharedStructure* st) {
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

RelocTables ziRelocOf(SharedStructure* st) {
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
    if (m == GpuRowMode::CROSS_FIXED_VARIABLE) {
        // CROSS identity (tasks #70, VC_REMERGE_RELOC_VROW): this coupler DUMMY_Q row hosts the relocated child's
        // reactive equation, and the child is a re-homed voltage-controlled bus whose own V row carries the live
        // coupler's ZERO_V — so BUS_TARGET_V lands HERE on the child's V column (relocOwn[r] = its own V row =
        // its V column), the slot the child's own dq/dV term already reserves in the pattern.
        int col = relocOwn != nullptr ? relocOwn[r] : -1;
        for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
            valsS[k] = colIdx[k] == col ? 1.0 : 0.0;
        }
        F[(size_t) s * n + r] = (col >= 0 ? x[(size_t) s * n + col] : 0.0) - target[(size_t) s * n + r];
    } else if (m == GpuRowMode::FIXED_VARIABLE) {  // identity: v/phi/rho/a1/b pinned to target
        for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
            valsS[k] = 0.0;
        }
        valsS[diagPos[r]] = 1.0;
        F[(size_t) s * n + r] = x[(size_t) s * n + r] - target[(size_t) s * n + r];
    } else if (m == GpuRowMode::BUS_TARGET_V_WITH_SLOPE) {  // SLOPE-SVC: V + slope*sumQ = target — the natural
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

// Per-scenario target overrides: each entry writes target[ovIndex[k]] = ovValue[k], where ovIndex is the
// FLAT dTarget index (scenarioSlot*n + row). Used by generator contingencies to set a scenario's bus P/Q
// target (remove the lost generator's P, set BUS_TARGET_Q on a PV→PQ flip) after the base row is tiled.
__global__ void applyTargetOverrideBatch(int nOv, const int* ovIndex, const double* ovValue, double* target) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < nOv) {
        target[ovIndex[k]] = ovValue[k];
    }
}

void fillActiveBatch(BatchSolverContext* ctx, DriverPass& drv, bool fOnly) {
    const int& n = drv.n;
    const int& nnz = drv.nnz;
    const int& nS = drv.nS;
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
    if (ctx->st->nLoadModelTerms) {
        fillLoadModelBatch<<<grid(ctx->st->nLoadModelTerms * nS), BLOCK, 0, ctx->stream>>>(ctx->st->nLoadModelTerms, nS, n, nnz, ctx->st->dLoadModelParameters.as<double>(), ctx->st->dLoadModelIndices.as<int>(), ctx->st->dLoadModelDestination.as<int>(), ctx->dX.as<double>(), fOnly ? nullptr : ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->st->loadModelRelocated ? ctx->st->dLoadModelAlternateRow.as<int>() : nullptr, ctx->st->loadModelRelocated ? ctx->st->dLoadModelAlternateDestination.as<int>() : nullptr, ctx->st->loadModelRelocated ? ctx->dMode.as<int>() : nullptr);
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
    if (ctx->nDistrRho) fillDistrRhoBatch<<<grid(ctx->nDistrRho * nS), BLOCK, 0, ctx->stream>>>(ctx->nDistrRho, nS, n, nnz, ctx->st->dDistrRhoColumn.as<int>(), ctx->st->dDistrRhoCoefficient.as<double>(), ctx->st->dDistrRhoDestination.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->nDistrRhoCoefficientOverrides, ctx->nDistrRhoCoefficientOverrides ? ctx->dDistrRhoOverrideScenarioTerm.as<int>() : nullptr, ctx->nDistrRhoCoefficientOverrides ? ctx->dDistrRhoOverrideCoefficient.as<double>() : nullptr, ctx->st->distrRhoQr ? ctx->st->dDistrRhoQrGroup.as<int>() : nullptr, ctx->dMode.as<int>(), ctx->st->dQrPtr.as<int>(), ctx->st->dQrRows.as<int>());
    if (ctx->nDistrShuntB) fillDistrShuntBBatch<<<grid(ctx->nDistrShuntB * nS), BLOCK, 0, ctx->stream>>>(ctx->nDistrShuntB, nS, n, nnz, ctx->st->dDistrShuntBColumn.as<int>(), ctx->st->dDistrShuntBCoefficient.as<double>(), ctx->st->dDistrShuntBDestination.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>());
    if (ctx->nzi) fillZeroImpBatch<<<grid(ctx->nzi * nS), BLOCK, 0, ctx->stream>>>(ctx->nzi, nS, n, nnz, ctx->st->dZiIn.as<double>(), ctx->st->dZiIdx.as<int>(), ctx->st->dZiDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->nziBase, ctx->nZiOn, ctx->dZiOnSlotElem.as<int>(), ctx->nZiOff, ctx->dZiOffSlotElem.as<int>(), ctx->st->dZiAltDest.as<int>(), ctx->st->dZiSelRow.as<int>(), ctx->dMode.as<int>(), ctx->st->ziDyn ? 1 : 0, withPilotSelf(ziRelocOf(ctx->st.get()), ctx), ctx->nZiTree, ctx->nZiTree ? ctx->dZiTreeSlotElemVal.as<int>() : nullptr, ctx->hasZiMasks ? ctx->dZiOffMask.as<char>() : nullptr, ctx->hasZiMasks ? ctx->dZiOnMask.as<char>() : nullptr, ctx->hasZiMasks ? ctx->dZiTreeMask.as<char>() : nullptr);
    applyRowModeBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, nnz, ctx->dMode.as<int>(), ctx->st->dRowPtr.as<int>(), ctx->st->dDiagPos.as<int>(), ctx->dX.as<double>(), ctx->dTarget.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->dRowSlope.as<double>(),
                ctx->st->dColIdx.as<int>(), ctx->st->dRelocOwn.p != nullptr ? ctx->st->dRelocOwn.as<int>() : nullptr);
    hazRepairValues(ctx, nS, ctx->dVals.as<double>());
    if (!deviceFixedIter) CUDA_CHECK(cudaStreamSynchronize(ctx->stream));   // fixed-iter: stream-ordered into the solve, no host sync
}

}  // namespace olfgpu
