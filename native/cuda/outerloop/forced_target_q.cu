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
__device__ double forcedTargetQClamped(double pPu, double raw, const double* pts, int npts, int extrapolate) {
    double mn = evaluateReactiveCapabilityCurve(pPu * 100.0, pts, npts, 0, extrapolate) / 100.0;
    double mx = evaluateReactiveCapabilityCurve(pPu * 100.0, pts, npts, 1, extrapolate) / 100.0;
    return raw < mn ? mn : (raw > mx ? mx : raw);
}

// Whether entry e takes part in scenario s: its generator is still there, its bus is not islanded, and - for a
// Q-row entry - its Q row still is the bus's Q balance (a frozen or repurposed row keeps what the scenario put there).
__device__ bool forcedTargetQEntryLive(int s, int e, int n, int nDistributedSlackBuses, const int* forcedTargetQIndices, int nLost, const int* lost,
                            const char* distributedSlackIslanded, const int* mode) {
    int distributedSlackGenerator = forcedTargetQIndices[e * 7 + 1];
    for (int k = 0; k < nLost; k++) {
        if (lost[2 * k] == s && lost[2 * k + 1] == distributedSlackGenerator) return false;
    }
    if (distributedSlackIslanded != nullptr && distributedSlackIslanded[(size_t) s * nDistributedSlackBuses + forcedTargetQIndices[e * 7 + 4]]) return false;
    return forcedTargetQIndices[e * 7] < 0 || mode[(size_t) s * n + forcedTargetQIndices[e * 7]] == forcedTargetQIndices[e * 7 + 5];
}

// The generator's target P now: its base P plus its own water-fill move (distributedSlackGeneratorMove), or - when the per-generator
// water-fill has not run - its share of its bus's row movement. distributedSlackGeneratorMove ALREADY carries the loss pre-distribution:
// like OLF's ActivePowerDistribution (which redistributes previousMismatch + the new mismatch from the INITIAL
// targets), the device re-derives the whole distribution every pass - measured on the B8-G fixture: pre 0.0909,
// distributedSlackGeneratorMove 0.0947 = 0.0909 + the pass's own 0.0038. Adding the pre-distribution again put B3's max limit at -0.2718
// against OLF's re-freeze at -0.2173 (0.84 kV at B3). Before the first pass the init kernel uses base + pre.
__device__ double forcedTargetQGeneratorTargetP(int s, int e, int n, int nDistributedSlackBuses, int nDistributedSlackGenerators, int nForcedTargetQ, const int* forcedTargetQIndices, const double* forcedTargetQParameters,
                         const double* forcedTargetQPreDistribution, const double* distributedSlackGeneratorMove, const char* distributedSlackGeneratorMoveValid, const double* distributedSlackGeneratorAnchor,
                         const double* distributedSlackGeneratorInitialMove, const int* distributedSlackBusPhiRow, const int* distributedSlackBusRowOverride, const double* target) {
    int distributedSlackGenerator = forcedTargetQIndices[e * 7 + 1];
    if (distributedSlackGenerator >= 0 && (distributedSlackGeneratorAnchor != nullptr || distributedSlackGeneratorInitialMove != nullptr
                      || distributedSlackGeneratorMove != nullptr && distributedSlackGeneratorMoveValid != nullptr && distributedSlackGeneratorMoveValid[s])) {
        return forcedTargetQParameters[e * 5] + distributedSlackGeneratorTargetPMove(s, distributedSlackGenerator, nDistributedSlackGenerators, distributedSlackGeneratorMove, distributedSlackGeneratorMoveValid, distributedSlackGeneratorAnchor, distributedSlackGeneratorInitialMove);
    }
    int pRow = distributedSlackBusRowOf(distributedSlackBusPhiRow, distributedSlackBusRowOverride, s, nDistributedSlackBuses, forcedTargetQIndices[e * 7 + 4]);
    return forcedTargetQParameters[e * 5] + forcedTargetQParameters[e * 5 + 3] * (target[(size_t) s * n + pRow] - forcedTargetQParameters[e * 5 + 4]);
}

// Solve start: each scenario's Q targets carry the clamp at the BASE P (the base load flow's targets); a scenario
// whose loss pre-distribution moved the generator starts from the clamp at the moved P, as OLF re-reads it when
// the pre-distribution invalidates the bus's generation target (AbstractLfBus.invalidateGenerationTargetP).
// One thread per scenario, entries in order: two generators on one bus add to one row deterministically.
// Q-row entries only (qRow >= 0); limit entries are recomputed with the dynamic limits.
__global__ void forcedTargetQInitBatch(int nForcedTargetQ, int nS, int n, int nDistributedSlackBuses, int extrapolate, const int* forcedTargetQIndices,
                                       const double* forcedTargetQParameters, const double* forcedTargetQPoints, const double* forcedTargetQPreDistribution,
                                       int nLost, const int* lost, const char* distributedSlackIslanded, const int* mode,
                                       double* target, double* applied) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nS) return;
    for (int e = 0; e < nForcedTargetQ; e++) {
        double base = forcedTargetQParameters[e * 5 + 2];
        applied[(size_t) s * nForcedTargetQ + e] = base;
        if (forcedTargetQIndices[e * 7] < 0 || !forcedTargetQEntryLive(s, e, n, nDistributedSlackBuses, forcedTargetQIndices, nLost, lost, distributedSlackIslanded, mode)) continue;
        double p = forcedTargetQParameters[e * 5] + (forcedTargetQPreDistribution != nullptr ? forcedTargetQPreDistribution[(size_t) s * nForcedTargetQ + e] : 0.0);
        double q = forcedTargetQClamped(p, forcedTargetQParameters[e * 5 + 1], forcedTargetQPoints + forcedTargetQIndices[e * 7 + 2] * 3, forcedTargetQIndices[e * 7 + 3], extrapolate);
        target[(size_t) s * n + forcedTargetQIndices[e * 7]] += q - base;
        applied[(size_t) s * nForcedTargetQ + e] = q;
    }
}

// After a distributed-slack pass: the Q target of a Q-row entry follows the clamp at the generator's new P.
__global__ void forcedTargetQBatch(int nForcedTargetQ, int nS, int n, int nDistributedSlackBuses, int nDistributedSlackGenerators, int extrapolate, const int* forcedTargetQIndices,
                                   const double* forcedTargetQParameters, const double* forcedTargetQPoints, const double* forcedTargetQPreDistribution,
                                   const double* distributedSlackGeneratorMove, const char* distributedSlackGeneratorMoveValid, const double* distributedSlackGeneratorAnchor,
                                   const double* distributedSlackGeneratorInitialMove, const int* distributedSlackBusPhiRow, const int* distributedSlackBusRowOverride, int nLost, const int* lost, const char* distributedSlackIslanded,
                                   const int* mode, const char* outerLoopCheckActive, double* target, double* applied) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nS || !outerLoopCheckActive[s]) return;
    for (int e = 0; e < nForcedTargetQ; e++) {
        if (forcedTargetQIndices[e * 7] < 0 || !forcedTargetQEntryLive(s, e, n, nDistributedSlackBuses, forcedTargetQIndices, nLost, lost, distributedSlackIslanded, mode)) continue;
        double p = forcedTargetQGeneratorTargetP(s, e, n, nDistributedSlackBuses, nDistributedSlackGenerators, nForcedTargetQ, forcedTargetQIndices, forcedTargetQParameters, forcedTargetQPreDistribution, distributedSlackGeneratorMove, distributedSlackGeneratorMoveValid, distributedSlackGeneratorAnchor, distributedSlackGeneratorInitialMove, distributedSlackBusPhiRow, distributedSlackBusRowOverride, target);
        double q = forcedTargetQClamped(p, forcedTargetQParameters[e * 5 + 1], forcedTargetQPoints + forcedTargetQIndices[e * 7 + 2] * 3, forcedTargetQIndices[e * 7 + 3], extrapolate);
        size_t a = (size_t) s * nForcedTargetQ + e;
        target[(size_t) s * n + forcedTargetQIndices[e * 7]] += q - applied[a];
        applied[a] = q;
    }
}

// A non-controlling generator on a VOLTAGE-CONTROLLED bus: AbstractLfBus.getLimitQ counts its (clamped) target Q
// in BOTH the bus's min and max limits, re-read at the current P on every reactive-limit check. Runs right after
// reactiveCapabilityCurveLimitsBatch rebuilt the dynamic limits from their static values (which carry the BASE clamp), so it adds
// the clamp's move since the base. Limit entries only (reactiveLimitEntry >= 0).
__global__ void forcedTargetQLimitsBatch(int nForcedTargetQ, int nS, int n, int nDistributedSlackBuses, int nDistributedSlackGenerators, int nReactiveLimitEntries, int extrapolate,
                                         const int* forcedTargetQIndices, const double* forcedTargetQParameters, const double* forcedTargetQPoints,
                                         const double* forcedTargetQPreDistribution, const double* distributedSlackGeneratorMove, const char* distributedSlackGeneratorMoveValid,
                                         const double* distributedSlackGeneratorAnchor, const double* distributedSlackGeneratorInitialMove, const int* distributedSlackBusPhiRow, const int* distributedSlackBusRowOverride, int nLost, const int* lost,
                                         const char* distributedSlackIslanded, const int* mode, const double* target,
                                         double* minQDyn, double* maxQDyn, const char* active) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nS) return;
    if (active != nullptr && !active[s]) return;
    for (int e = 0; e < nForcedTargetQ; e++) {
        int reactiveLimitEntry = forcedTargetQIndices[e * 7 + 6];
        if (reactiveLimitEntry < 0 || !forcedTargetQEntryLive(s, e, n, nDistributedSlackBuses, forcedTargetQIndices, nLost, lost, distributedSlackIslanded, mode)) continue;
        double p = forcedTargetQGeneratorTargetP(s, e, n, nDistributedSlackBuses, nDistributedSlackGenerators, nForcedTargetQ, forcedTargetQIndices, forcedTargetQParameters, forcedTargetQPreDistribution, distributedSlackGeneratorMove, distributedSlackGeneratorMoveValid, distributedSlackGeneratorAnchor, distributedSlackGeneratorInitialMove, distributedSlackBusPhiRow, distributedSlackBusRowOverride, target);
        double d = forcedTargetQClamped(p, forcedTargetQParameters[e * 5 + 1], forcedTargetQPoints + forcedTargetQIndices[e * 7 + 2] * 3, forcedTargetQIndices[e * 7 + 3], extrapolate)
                - forcedTargetQParameters[e * 5 + 2];
        minQDyn[(size_t) s * nReactiveLimitEntries + reactiveLimitEntry] += d;
        maxQDyn[(size_t) s * nReactiveLimitEntries + reactiveLimitEntry] += d;
    }
}

// forceTargetQInReactiveLimits on voltage-controlled buses: after every rebuild of the dynamic reactive limits.
void launchForcedTargetQLimits(BatchSolverContext* ctx, int nS, int n, const char* active) {
    if (ctx->nForcedTargetQLimitEntries <= 0) return;
    forcedTargetQLimitsBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(ctx->nForcedTargetQ, nS, n, ctx->nDistributedSlackBuses, ctx->nDistributedSlackGenerators, ctx->nReactiveLimitEntries,
            ctx->forcedTargetQExtrapolate, ctx->dForcedTargetQIndices.as<int>(), ctx->dForcedTargetQParameters.as<double>(), ctx->dForcedTargetQPoints.as<double>(),
            ctx->hasForcedTargetQPreDistribution ? ctx->dForcedTargetQPreDistribution.as<double>() : nullptr,
            ctx->nDistributedSlackGenerators > 0 ? ctx->dDistributedSlackGeneratorMove.as<double>() : nullptr,
            ctx->nDistributedSlackGenerators > 0 ? ctx->dDistributedSlackGeneratorMoveValid.as<char>() : nullptr, ctx->hasDistributedSlackGeneratorAnchor ? ctx->dDistributedSlackGeneratorAnchor.as<double>() : nullptr, ctx->hasDistributedSlackGeneratorInitialMove ? ctx->dDistributedSlackGeneratorInitialMove.as<double>() : nullptr,
            ctx->nDistributedSlackBuses > 0 ? ctx->dDistributedSlackBusPhiRow.as<int>() : nullptr, ctx->nDistributedSlackBusRowOverrides > 0 ? ctx->dDistributedSlackBusRowOverride.as<int>() : nullptr,
            ctx->nDistributedSlackLostGenerators, ctx->nDistributedSlackLostGenerators > 0 ? ctx->dDistributedSlackLostGenerators.as<int>() : nullptr,
            ctx->nDistributedSlackIslanded > 0 ? ctx->dDistributedSlackIslanded.as<char>() : nullptr, ctx->dMode.as<int>(),
            ctx->dTarget.as<double>(), ctx->dReactiveLimitMinQPerScenario.as<double>(), ctx->dReactiveLimitMaxQPerScenario.as<double>(), active);
}

}  // namespace olfgpu
