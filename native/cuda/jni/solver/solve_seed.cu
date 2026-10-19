// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See solve_seed.cuh.
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
#include "../outerloop/distributed_slack.cuh"
#include "../outerloop/forced_target_q.cuh"
#include "../outerloop/incremental_flow_control.cuh"
#include "../outerloop/incremental_shunt_voltage_control.cuh"
#include "../outerloop/incremental_transformer_voltage_control.cuh"
#include "../outerloop/phase_control.cuh"
#include "../outerloop/pilot_voltage.cuh"
#include "../outerloop/reactive_capability_curves.cuh"
#include "../outerloop/reactive_limits_seed.cuh"
#include "../outerloop/remote_voltage_control.cuh"
#include "../outerloop/secondary_voltage_control.cuh"
#include "../outerloop/shunt_voltage_control.cuh"
#include "../outerloop/transformer_voltage_control.cuh"
#include "../outerloop/voltage_monitoring.cuh"
#include "driver_steps.cuh"
#include "solve_seed.cuh"

namespace olfgpu {

// The per-solve seed of a fresh (non-resumed) full load flow: the generators' slack moves cleared, the pilot Q targets
// captured, every reactive-limit pin seeded (from the base or a previous solve) and stamped on the rows, and every
// in-kernel loop reset to its entry state.
void seedSolve(BatchSolverContext* ctx, DriverPass& drv, const int* rlStateIn, const double* dsSlackShift) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    const bool& useDynRlLimits = drv.useDynRlLimits;
    const int& nRlVtOv = drv.nRlVtOv;
    const int& nRlQOv = drv.nRlQOv;
    const int& nCvOv = drv.nCvOv;
    DeviceBuf& dBaseMode = drv.dBaseMode;
    DeviceBuf& dRlVtOvSlotIdx = drv.dRlVtOvSlotIdx;
    DeviceBuf& dRlVtOvVal = drv.dRlVtOvVal;
    DeviceBuf& dRlQOvSlotEntry = drv.dRlQOvSlotEntry;
    DeviceBuf& dRlQOvMinMax = drv.dRlQOvMinMax;
    DeviceBuf& dRlQOvPtr = drv.dRlQOvPtr;
    DeviceBuf& dRlQOvList = drv.dRlQOvList;
    DeviceBuf& dCvOvPtr = drv.dCvOvPtr;
    DeviceBuf& dCvOvEntry = drv.dCvOvEntry;
    DeviceBuf& dCvOvVal = drv.dCvOvVal;
    // Each generator's own slack move is per-solve state, cleared before ANYTHING reads it: the seed-time curve
    // limits below evaluate a curve at base + genCum wherever genCumValid is set. Cleared only after the seed,
    // a reused context evaluated scenario slot s at the move of the contingency the PREVIOUS chunk solved in
    // that slot. applyToNetwork also reads it when the DS phase never fires (NaN targetP otherwise).
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

}  // namespace olfgpu
