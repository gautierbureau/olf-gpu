// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See reactive_capability_curves.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/codes/gpu_ac_emulation_status.cuh"
#include "../common/codes/gpu_two_sides.cuh"
#include "../equations/generated/hvdc.cuh"
#include "reactive_capability_curve.cuh"
#include "slack_participants.cuh"
#include "reactive_capability_curves.cuh"

using namespace powsybl::openloadflow;

namespace olfgpu {

__device__ inline double hvdcStationReactiveLimitsActivePower(const CurveHvdcAcEmulationView& c, int s, int n, int h, int side) {
    size_t scenarioHvdc = (size_t) s * c.nHvdcAcEmulation + h;
    const double* p = c.hvdcAcEmulationParameters + (size_t) h * 9;
    bool off = c.contingencyOutage[scenarioHvdc] || c.disabled[(size_t) s * c.nElements + c.hvdcElement[h]];
    if (c.status[scenarioHvdc] == GpuAcEmulationStatus::LINEAR_MODE && !off) {
        const double* xs = c.x + (size_t) s * n;
        double ph1 = xs[c.bus1PhiColumn[h]];
        double ph2 = xs[c.bus2PhiColumn[h]];
        return side == GpuTwoSides::ONE ? -hvdc::p1(p[0], p[1], p[4], p[5], p[6], ph1, ph2)
                                        : -hvdc::p2(p[0], p[1], p[4], p[5], p[6], ph1, ph2);
    }
    return side == GpuTwoSides::ONE ? c.appliedDelta1[scenarioHvdc] + p[7] : c.appliedDelta2[scenarioHvdc] + p[8];
}

// Evaluate the per-scenario DYNAMIC reactive limits (OLF: bus.getMinQ()/getMaxQ() re-read each RL
// check, curve gens at their CURRENT targetP): start from the static pack values and add each
// participating curve gen's delta eval(p_now) - eval(p_base). p_now = basePpu + share * (the bus's
// DS-moved row target - its base) — exact while intra-bus proportionality holds (the pack's DS
// model). At the base point the delta is bit-zero (same formula both sides).
// reactiveCapabilityCurveLimitsBatch reads each generator's OWN cumulative P from the distributed-slack water-fill
// (dDistributedSlackGeneratorMove), which clamps PER GENERATOR, instead of splitting its bus's movement by a share fixed at
// pack time. The share is exact only while the intra-bus split stays proportional, and a clamp breaks
// that: on rte6515 GEN-6172 at VL-6139_0 the 451 MW unit takes +0.0442 where its share says +0.0890,
// which puts the bus's Q box out by 2.67 MVAr and the bus 0.0379 kV off. The share reconstruction
// survives ONLY as the fallback for entries distributedSlackGeneratorMove cannot serve (no DS slot, the per-generator
// water-fill did not run, or the outaged bus's own entries) — there is no switch back to it.
__global__ void reactiveCapabilityCurveLimitsBatch(int nReactiveLimitEntries, int nScenarios, int nReactiveCapabilityCurves, int n, int reactiveCapabilityCurveExtrapolate,
                                   const int* reactiveCapabilityCurveIndices, const double* reactiveCapabilityCurveParameters, const double* reactiveCapabilityCurvePoints,
                                   const int* reactiveLimitEntryCurveStart, const int* reactiveLimitEntryCurves, CurveHvdcAcEmulationView reactiveCapabilityCurveHvdcStation,
                                   const double* reactiveLimitMinQ, const double* reactiveLimitMaxQ, const double* reactiveLimitLoadQ,
                                   const double* target,
                                   int nQOv, const int* qOvSlotEntry, const double* qOvMinMax,
                                   const int* qOvPtr, const int* qOvList,
                                   const int* reactiveCapabilityCurveOverrideStart, const int* reactiveCapabilityCurveOverrideCurve, const double* reactiveCapabilityCurveOverrideStepScale,
                                   const double* distributedSlackGeneratorMove, const char* distributedSlackGeneratorMoveValid, const double* distributedSlackGeneratorAnchor, const double* distributedSlackGeneratorInitialMove, int nDistributedSlackGenerators,
                                   const int* distributedSlackBusPhiRow, const int* distributedSlackBusRowOverride, int nDistributedSlackBuses, const int* distributedSlackGeneratorBus,
                                   double* minQDyn, double* maxQDyn, double* loadQDyn) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nReactiveLimitEntries * nScenarios) return;
    int i = tid % nReactiveLimitEntries;
    int s = tid / nReactiveLimitEntries;
    double mn = reactiveLimitMinQ[i];
    double mx = reactiveLimitMaxQ[i];
    double lq = reactiveLimitLoadQ[i];
    int ovLo = reactiveCapabilityCurveOverrideStart != nullptr ? reactiveCapabilityCurveOverrideStart[s] : 0;           // this scenario's curve repairs, if any
    int ovHi = reactiveCapabilityCurveOverrideStart != nullptr ? reactiveCapabilityCurveOverrideStart[s + 1] : 0;       // (CSR by scenario: a handful of entries)
    // This entry's OWN curves, CSR by entry (built at upload, pack order kept so the sums accumulate in the
    // same order). The flat scan over every curve cost nReactiveLimitEntries * nS * nReactiveCapabilityCurves reads per launch - quadratic in the
    // network's size, and this kernel runs before every reactive-limit decision of every chunk.
    // (nReactiveCapabilityCurves == 0: no curve channel on this context, or a cached context's stale table from an earlier run - the
    // pointers are not read; the kernel still runs for the Q overrides and the forced target Q below.)
    for (int c = nReactiveCapabilityCurves > 0 ? reactiveLimitEntryCurveStart[i] : 0; c < (nReactiveCapabilityCurves > 0 ? reactiveLimitEntryCurveStart[i + 1] : 0); c++) {
        int g = reactiveLimitEntryCurves[c];
        // A generator outage at this bus skews the p_now reconstruction below. pStep is the part of
        // the row target the OUTAGE moved (the lost P, and any shed load) — not slack power, so it
        // moves nobody's curve; shareScale renormalizes this generator's slack share over the
        // generators still there; shareScale < 0 marks the lost generator's OWN entry, whose whole
        // capability left with it (its static base share is taken off by the qOv channel below).
        double pStep = 0;
        double shareScale = 1;
        for (int k = ovLo; k < ovHi; k++) {
            if (reactiveCapabilityCurveOverrideCurve[k] == g) {
                pStep = reactiveCapabilityCurveOverrideStepScale[2 * k];
                shareScale = reactiveCapabilityCurveOverrideStepScale[2 * k + 1];
                break;
            }
        }
        if (shareScale < 0) continue;
        int pRow = reactiveCapabilityCurveIndices[g * 5 + 1];
        const double* pts = reactiveCapabilityCurvePoints + reactiveCapabilityCurveIndices[g * 5 + 2] * 3;
        int npts = reactiveCapabilityCurveIndices[g * 5 + 3];
        int distributedSlackGenerator = reactiveCapabilityCurveIndices[g * 5 + 4];
        // The row that HOSTS this bus's active balance THIS scenario: the slack pack's row for the bus
        // (a coupler-relocated child's is its parent coupler's DUMMY_P row), moved back to the bus's own
        // PHI row by distributedSlackBusRowOverride when the coupler dies. The static pRow is the bus's own PHI row, which for a
        // relocated child hosts ZERO_PHI (target 0, never moved): read there, the share reconstruction
        // put the child's generator at its BASE P while its pilot sibling sat at the pre-distributed
        // one. Measured on rte6515 BUS-4467_BBS: VL-6231_1 pinned at 0.576013 where OLF (and the pilot
        // VL-6231_0) had 0.566571 - the same unit halves at the same P - and the 9.4e-3 pu of reactive
        // power that put on the coupler moved the slack mismatch by 9.6e-6 pu, the whole 9.56e-4 MW
        // distributed-power gap.
        if (distributedSlackGenerator >= 0 && distributedSlackGeneratorBus != nullptr && distributedSlackBusPhiRow != nullptr
                && distributedSlackGeneratorBus[distributedSlackGenerator] >= 0) {   // -1: the slack's own
            pRow = distributedSlackBusRowOf(distributedSlackBusPhiRow, distributedSlackBusRowOverride, s, nDistributedSlackBuses, distributedSlackGeneratorBus[distributedSlackGenerator]);
        }
        // This generator's OWN movement, straight from the distributed-slack water-fill, which clamps
        // PER GENERATOR. Splitting its bus's movement by a fixed share is exact only while the
        // intra-bus split stays proportional, and a clamp breaks that: measured on rte6515 GEN-6172 at
        // VL-6139_0, the 451 MW unit takes +0.0442 where its share says +0.0890 and the remainder goes
        // to its four siblings. The share reconstruction stays as the fallback for an entry with no
        // slot, and for the outaged bus's own entries, whose pStep/shareScale repair is expressed in
        // those terms.
        // distributedSlackGeneratorMove already reflects the outage — the lost generator is marked out of the water-fill and
        // the bus's anchor carries its P — so pStep and shareScale, which exist only to repair the
        // share reconstruction on the outaged bus, are not needed when it is available. The lost
        // generator's OWN entry never reaches here (shareScale < 0 continues above). An AC-emulation
        // VSC is not in the slack distribution: its P is its link's (CurveHvdcAcEmulationView).
        double pNow;
        int hvdcIndex = reactiveCapabilityCurveHvdcStation.hvdcStation != nullptr ? reactiveCapabilityCurveHvdcStation.hvdcStation[2 * g] : -1;
        if (hvdcIndex >= 0 && reactiveCapabilityCurveHvdcStation.nHvdcAcEmulation > 0) {
            pNow = hvdcStationReactiveLimitsActivePower(reactiveCapabilityCurveHvdcStation, s, n, hvdcIndex, reactiveCapabilityCurveHvdcStation.hvdcStation[2 * g + 1]);
        } else if (distributedSlackGenerator >= 0 && (distributedSlackGeneratorAnchor != nullptr || distributedSlackGeneratorInitialMove != nullptr
                                 || distributedSlackGeneratorMove != nullptr && distributedSlackGeneratorMoveValid != nullptr && distributedSlackGeneratorMoveValid[s])) {
            // the generator's own P: its anchor, plus this solve's water-fill or, before any pass, where the
            // solve started it (distributedSlackGeneratorTargetPMove). A share of its bus row's move is not that once anything clamps.
            pNow = reactiveCapabilityCurveParameters[g * 5] + distributedSlackGeneratorTargetPMove(s, distributedSlackGenerator, nDistributedSlackGenerators, distributedSlackGeneratorMove, distributedSlackGeneratorMoveValid, distributedSlackGeneratorAnchor, distributedSlackGeneratorInitialMove);
        } else {
            pNow = reactiveCapabilityCurveParameters[g * 5]
                    + reactiveCapabilityCurveParameters[g * 5 + 1] * shareScale * (target[(size_t) s * n + pRow] - reactiveCapabilityCurveParameters[g * 5 + 2] - pStep);
        }
        double mnAdd = evaluateReactiveCapabilityCurve(pNow * 100.0, pts, npts, 0, reactiveCapabilityCurveExtrapolate) / 100.0 - reactiveCapabilityCurveParameters[g * 5 + 3];
        double mxAdd = evaluateReactiveCapabilityCurve(pNow * 100.0, pts, npts, 1, reactiveCapabilityCurveExtrapolate) / 100.0 - reactiveCapabilityCurveParameters[g * 5 + 4];
        mn += mnAdd;
        mx += mxAdd;
    }
    // A generator lost at a bus that KEEPS voltage control takes its reactive capability with it. The
    // static aggregate is run-constant and still holds it, while OLF's bus.getMinQ()/getMaxQ() sum only
    // the generators still there, so the scenario's own delta comes in here. Without curves this is the
    // ONLY reason the dynamic limits exist, which is why the seeding above starts from the static pair.
    // ...the scenario's OWN overrides, CSR by slot: the flat {slot, entry} scan cost nReactiveLimitEntries * nS * nQOv reads per
    // launch, with nQOv about nS on a generator-outage chunk (one override per scenario).
    for (int c = qOvPtr != nullptr ? qOvPtr[s] : 0; c < (qOvPtr != nullptr ? qOvPtr[s + 1] : 0); c++) {
        int k = qOvList[c];
        if (qOvSlotEntry[2 * k + 1] == i) {
            mn += qOvMinMax[5 * k];
            mx += qOvMinMax[5 * k + 1];
            lq += qOvMinMax[5 * k + 2];                      // a SHED load's reactive power: the pinned
        }                                                    // target and the controller's own q both
    }                                                        // measure against the bus's load target Q
    minQDyn[(size_t) s * nReactiveLimitEntries + i] = mn;
    maxQDyn[(size_t) s * nReactiveLimitEntries + i] = mx;
    loadQDyn[(size_t) s * nReactiveLimitEntries + i] = lq;
}

// The curve kernel's view of the AC-emulation loop state (see CurveHvdcAcEmulationView).
CurveHvdcAcEmulationView curveHvdcAcEmulationView(BatchSolverContext* ctx) {
    CurveHvdcAcEmulationView c{};
    if (!ctx->reactiveCapabilityCurveHasHvdc) {
        return c;
    }
    c.hvdcStation = ctx->dReactiveCapabilityCurveHvdcStation.as<int>();
    c.nHvdcAcEmulation = ctx->nHvdcAcEmulation;
    if (ctx->nHvdcAcEmulation > 0) {
        c.status = ctx->dHvdcAcEmulationStatus.as<int>();
        c.appliedDelta1 = ctx->dHvdcAppliedDelta1.as<double>();
        c.appliedDelta2 = ctx->dHvdcAppliedDelta2.as<double>();
        c.hvdcAcEmulationParameters = ctx->dHvdcAcEmulationParameters.as<double>();
        c.bus1PhiColumn = ctx->dHvdcBus1PhiColumn.as<int>();
        c.bus2PhiColumn = ctx->dHvdcBus2PhiColumn.as<int>();
        c.hvdcElement = ctx->dHvdcElement.as<int>();
        c.disabled = ctx->dDisabled.as<char>();
        c.nElements = ctx->nElements;
        c.contingencyOutage = ctx->dHvdcContingencyOutage.as<char>();
        c.x = ctx->dX.as<double>();
    }
    return c;
}

}  // namespace olfgpu
