// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See forced_target_q.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "reactive_capability_curve.cuh"
#include "slack_participants.cuh"
#include "forced_target_q.cuh"

namespace olfgpu {

// OLF LfGeneratorImpl.getTargetQ under forceTargetQInReactiveLimits: the IIDM target Q, clamped into the
// generator's curve limits at target P p (below minQ -> minQ, above maxQ -> maxQ).
__device__ double fqClamped(double pPu, double raw, const double* pts, int npts, int extrapolate) {
    double mn = rlCurveEval(pPu * 100.0, pts, npts, 0, extrapolate) / 100.0;
    double mx = rlCurveEval(pPu * 100.0, pts, npts, 1, extrapolate) / 100.0;
    return raw < mn ? mn : (raw > mx ? mx : raw);
}

// Whether entry e takes part in scenario s: its generator is still there, its bus is not islanded, and - for a
// Q-row entry - its Q row still is the bus's Q balance (a frozen or repurposed row keeps what the scenario put there).
__device__ bool fqEntryLive(int s, int e, int n, int nds, const int* fqIdx, int nLost, const int* lost,
                            const char* dsIslanded, const int* mode) {
    int slot = fqIdx[e * 7 + 1];
    for (int k = 0; k < nLost; k++) {
        if (lost[2 * k] == s && lost[2 * k + 1] == slot) return false;
    }
    if (dsIslanded != nullptr && dsIslanded[(size_t) s * nds + fqIdx[e * 7 + 4]]) return false;
    return fqIdx[e * 7] < 0 || mode[(size_t) s * n + fqIdx[e * 7]] == fqIdx[e * 7 + 5];
}

// The generator's target P now: its base P plus its own water-fill move (genCum), or - when the per-generator
// water-fill has not run - its share of its bus's row movement. genCum ALREADY carries the loss pre-distribution:
// like OLF's ActivePowerDistribution (which redistributes previousMismatch + the new mismatch from the INITIAL
// targets), the device re-derives the whole distribution every pass - measured on the B8-G fixture: pre 0.0909,
// genCum 0.0947 = 0.0909 + the pass's own 0.0038. Adding the pre-distribution again put B3's max limit at -0.2718
// against OLF's re-freeze at -0.2173 (0.84 kV at B3). Before the first pass the init kernel uses base + pre.
__device__ double fqPNow(int s, int e, int n, int nds, int ngen, int nfq, const int* fqIdx, const double* fqIn,
                         const double* fqPre, const double* genCum, const char* genCumValid, const double* dgAnchor,
                         const double* dgStart, const int* dsPhiRow, const int* dsRowOv, const double* target) {
    int slot = fqIdx[e * 7 + 1];
    if (slot >= 0 && (dgAnchor != nullptr || dgStart != nullptr
                      || genCum != nullptr && genCumValid != nullptr && genCumValid[s])) {
        return fqIn[e * 5] + genPNowMove(s, slot, ngen, genCum, genCumValid, dgAnchor, dgStart);
    }
    int pRow = dsRowOf(dsPhiRow, dsRowOv, s, nds, fqIdx[e * 7 + 4]);
    return fqIn[e * 5] + fqIn[e * 5 + 3] * (target[(size_t) s * n + pRow] - fqIn[e * 5 + 4]);
}

// Solve start: each scenario's Q targets carry the clamp at the BASE P (the base load flow's targets); a scenario
// whose loss pre-distribution moved the generator starts from the clamp at the moved P, as OLF re-reads it when
// the pre-distribution invalidates the bus's generation target (AbstractLfBus.invalidateGenerationTargetP).
// One thread per scenario, entries in order: two generators on one bus add to one row deterministically.
// Q-row entries only (qRow >= 0); limit entries are recomputed with the dynamic limits.
__global__ void forcedTargetQInitBatch(int nfq, int nS, int n, int nds, int extrapolate, const int* fqIdx,
                                       const double* fqIn, const double* fqPts, const double* fqPre,
                                       int nLost, const int* lost, const char* dsIslanded, const int* mode,
                                       double* target, double* applied) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nS) return;
    for (int e = 0; e < nfq; e++) {
        double base = fqIn[e * 5 + 2];
        applied[(size_t) s * nfq + e] = base;
        if (fqIdx[e * 7] < 0 || !fqEntryLive(s, e, n, nds, fqIdx, nLost, lost, dsIslanded, mode)) continue;
        double p = fqIn[e * 5] + (fqPre != nullptr ? fqPre[(size_t) s * nfq + e] : 0.0);
        double q = fqClamped(p, fqIn[e * 5 + 1], fqPts + fqIdx[e * 7 + 2] * 3, fqIdx[e * 7 + 3], extrapolate);
        target[(size_t) s * n + fqIdx[e * 7]] += q - base;
        applied[(size_t) s * nfq + e] = q;
    }
}

// After a distributed-slack pass: the Q target of a Q-row entry follows the clamp at the generator's new P.
__global__ void forcedTargetQBatch(int nfq, int nS, int n, int nds, int ngen, int extrapolate, const int* fqIdx,
                                   const double* fqIn, const double* fqPts, const double* fqPre,
                                   const double* genCum, const char* genCumValid, const double* dgAnchor,
                                   const double* dgStart, const int* dsPhiRow, const int* dsRowOv, int nLost, const int* lost, const char* dsIslanded,
                                   const int* mode, const char* rlActive, double* target, double* applied) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nS || !rlActive[s]) return;
    for (int e = 0; e < nfq; e++) {
        if (fqIdx[e * 7] < 0 || !fqEntryLive(s, e, n, nds, fqIdx, nLost, lost, dsIslanded, mode)) continue;
        double p = fqPNow(s, e, n, nds, ngen, nfq, fqIdx, fqIn, fqPre, genCum, genCumValid, dgAnchor, dgStart, dsPhiRow, dsRowOv, target);
        double q = fqClamped(p, fqIn[e * 5 + 1], fqPts + fqIdx[e * 7 + 2] * 3, fqIdx[e * 7 + 3], extrapolate);
        size_t a = (size_t) s * nfq + e;
        target[(size_t) s * n + fqIdx[e * 7]] += q - applied[a];
        applied[a] = q;
    }
}

// A non-controlling generator on a VOLTAGE-CONTROLLED bus: AbstractLfBus.getLimitQ counts its (clamped) target Q
// in BOTH the bus's min and max limits, re-read at the current P on every reactive-limit check. Runs right after
// rlCurveLimitsBatch rebuilt the dynamic limits from their static values (which carry the BASE clamp), so it adds
// the clamp's move since the base. Limit entries only (rlIdx >= 0).
__global__ void forcedTargetQLimitsBatch(int nfq, int nS, int n, int nds, int ngen, int nrl, int extrapolate,
                                         const int* fqIdx, const double* fqIn, const double* fqPts,
                                         const double* fqPre, const double* genCum, const char* genCumValid,
                                         const double* dgAnchor, const double* dgStart, const int* dsPhiRow, const int* dsRowOv, int nLost, const int* lost,
                                         const char* dsIslanded, const int* mode, const double* target,
                                         double* minQDyn, double* maxQDyn) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nS) return;
    for (int e = 0; e < nfq; e++) {
        int rl = fqIdx[e * 7 + 6];
        if (rl < 0 || !fqEntryLive(s, e, n, nds, fqIdx, nLost, lost, dsIslanded, mode)) continue;
        double p = fqPNow(s, e, n, nds, ngen, nfq, fqIdx, fqIn, fqPre, genCum, genCumValid, dgAnchor, dgStart, dsPhiRow, dsRowOv, target);
        double d = fqClamped(p, fqIn[e * 5 + 1], fqPts + fqIdx[e * 7 + 2] * 3, fqIdx[e * 7 + 3], extrapolate)
                - fqIn[e * 5 + 2];
        minQDyn[(size_t) s * nrl + rl] += d;
        maxQDyn[(size_t) s * nrl + rl] += d;
    }
}

// forceTargetQInReactiveLimits on voltage-controlled buses: after every rebuild of the dynamic reactive limits.
void launchForcedTargetQLimits(BatchSolverContext* ctx, int nS, int n) {
    if (ctx->nfqLim <= 0) return;
    forcedTargetQLimitsBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(ctx->nfq, nS, n, ctx->nds, ctx->ngen, ctx->nrl,
            ctx->fqExtrapolate, ctx->dFqIdx.as<int>(), ctx->dFqIn.as<double>(), ctx->dFqPts.as<double>(),
            ctx->fqHasPre ? ctx->dFqPre.as<double>() : nullptr,
            ctx->ngen > 0 ? ctx->dGenCum.as<double>() : nullptr,
            ctx->ngen > 0 ? ctx->dGenCumValid.as<char>() : nullptr, ctx->hasDgAnchor ? ctx->dDgAnchor.as<double>() : nullptr, ctx->hasDgStart ? ctx->dDgStart.as<double>() : nullptr,
            ctx->nds > 0 ? ctx->dDsPhiRow.as<int>() : nullptr, ctx->nDsRowOv > 0 ? ctx->dDsRowOv.as<int>() : nullptr,
            ctx->nDsLostGen, ctx->nDsLostGen > 0 ? ctx->dDsLostGen.as<int>() : nullptr,
            ctx->nDsIslanded > 0 ? ctx->dDsIslanded.as<char>() : nullptr, ctx->dMode.as<int>(),
            ctx->dTarget.as<double>(), ctx->dRlMinQDyn.as<double>(), ctx->dRlMaxQDyn.as<double>());
}

}  // namespace olfgpu
