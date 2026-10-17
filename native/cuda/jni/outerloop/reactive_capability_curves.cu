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
#include "hvdc.cuh"
#include "reactive_capability_curve.cuh"
#include "slack_participants.cuh"
#include "reactive_capability_curves.cuh"

using namespace powsybl::openloadflow;

namespace olfgpu {

__device__ inline double cvHvdcActivePower(const CvHvdc& c, int s, int n, int h, int side) {
    size_t sh = (size_t) s * c.nhv + h;
    const double* p = c.in + (size_t) h * 9;
    bool off = c.ctgOut[sh] || c.disabled[(size_t) s * c.nElements + c.elem[h]];
    if (c.status[sh] == HV_LINEAR && !off) {
        const double* xs = c.x + (size_t) s * n;
        double ph1 = xs[c.phi1[h]];
        double ph2 = xs[c.phi2[h]];
        return side == 1 ? -hvdc::p1(p[0], p[1], p[4], p[5], p[6], ph1, ph2)
                         : -hvdc::p2(p[0], p[1], p[4], p[5], p[6], ph1, ph2);
    }
    return side == 1 ? c.d1[sh] + p[7] : c.d2[sh] + p[8];
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
__global__ void rlCurveLimitsBatch(int nrl, int nScenarios, int ncv, int n, int cvExtrapolate,
                                   const int* cvIdx, const double* cvIn, const double* cvPts,
                                   const int* cvPtr, const int* cvList, CvHvdc cvHv,
                                   const double* rlMinQ, const double* rlMaxQ, const double* rlLoadQ,
                                   const double* target,
                                   int nQOv, const int* qOvSlotEntry, const double* qOvMinMax,
                                   const int* qOvPtr, const int* qOvList,
                                   const int* cvOvPtr, const int* cvOvEntry, const double* cvOvVal,
                                   const double* genCum, const char* genCumValid, const double* dgAnchor, const double* dgStart, int ngen,
                                   const int* dsPhiRow, const int* dsRowOv, int nds, const int* dgBus,
                                   double* minQDyn, double* maxQDyn, double* loadQDyn, int cvProbeEntry, int cvProbeSlot) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nrl * nScenarios) return;
    int i = tid % nrl;
    int s = tid / nrl;
    double mn = rlMinQ[i];
    double mx = rlMaxQ[i];
    double lq = rlLoadQ[i];
    int ovLo = cvOvPtr != nullptr ? cvOvPtr[s] : 0;           // this scenario's curve repairs, if any
    int ovHi = cvOvPtr != nullptr ? cvOvPtr[s + 1] : 0;       // (CSR by scenario: a handful of entries)
    // This entry's OWN curves, CSR by entry (built at upload, pack order kept so the sums accumulate in the
    // same order). The flat scan over every curve cost nrl * nS * ncv reads per launch - quadratic in the
    // network's size, and this kernel runs before every reactive-limit decision of every chunk.
    // (ncv == 0: no curve channel on this context, or a cached context's stale table from an earlier run - the
    // pointers are not read; the kernel still runs for the Q overrides and the forced target Q below.)
    for (int c = ncv > 0 ? cvPtr[i] : 0; c < (ncv > 0 ? cvPtr[i + 1] : 0); c++) {
        int g = cvList[c];
        // A generator outage at this bus skews the p_now reconstruction below. pStep is the part of
        // the row target the OUTAGE moved (the lost P, and any shed load) — not slack power, so it
        // moves nobody's curve; shareScale renormalizes this generator's slack share over the
        // generators still there; shareScale < 0 marks the lost generator's OWN entry, whose whole
        // capability left with it (its static base share is taken off by the qOv channel below).
        double pStep = 0;
        double shareScale = 1;
        for (int k = ovLo; k < ovHi; k++) {
            if (cvOvEntry[k] == g) {
                pStep = cvOvVal[2 * k];
                shareScale = cvOvVal[2 * k + 1];
                break;
            }
        }
        if (shareScale < 0) continue;
        int pRow = cvIdx[g * 5 + 1];
        const double* pts = cvPts + cvIdx[g * 5 + 2] * 3;
        int npts = cvIdx[g * 5 + 3];
        int slot = cvIdx[g * 5 + 4];
        // The row that HOSTS this bus's active balance THIS scenario: the slack pack's row for the bus
        // (a coupler-relocated child's is its parent coupler's DUMMY_P row), moved back to the bus's own
        // PHI row by dsRowOv when the coupler dies. The static pRow is the bus's own PHI row, which for a
        // relocated child hosts ZERO_PHI (target 0, never moved): read there, the share reconstruction
        // put the child's generator at its BASE P while its pilot sibling sat at the pre-distributed
        // one. Measured on rte6515 BUS-4467_BBS: VL-6231_1 pinned at 0.576013 where OLF (and the pilot
        // VL-6231_0) had 0.566571 - the same unit halves at the same P - and the 9.4e-3 pu of reactive
        // power that put on the coupler moved the slack mismatch by 9.6e-6 pu, the whole 9.56e-4 MW
        // distributed-power gap.
        if (slot >= 0 && dgBus != nullptr && dsPhiRow != nullptr && dgBus[slot] >= 0) {   // -1: the slack's own
            pRow = dsRowOf(dsPhiRow, dsRowOv, s, nds, dgBus[slot]);
        }
        // This generator's OWN movement, straight from the distributed-slack water-fill, which clamps
        // PER GENERATOR. Splitting its bus's movement by a fixed share is exact only while the
        // intra-bus split stays proportional, and a clamp breaks that: measured on rte6515 GEN-6172 at
        // VL-6139_0, the 451 MW unit takes +0.0442 where its share says +0.0890 and the remainder goes
        // to its four siblings. The share reconstruction stays as the fallback for an entry with no
        // slot, and for the outaged bus's own entries, whose pStep/shareScale repair is expressed in
        // those terms.
        // genCum already reflects the outage — the lost generator is marked out of the water-fill and
        // the bus's anchor carries its P — so pStep and shareScale, which exist only to repair the
        // share reconstruction on the outaged bus, are not needed when it is available. The lost
        // generator's OWN entry never reaches here (shareScale < 0 continues above). An AC-emulation
        // VSC is not in the slack distribution: its P is its link's (CvHvdc).
        double pNow;
        int hvh = cvHv.hv != nullptr ? cvHv.hv[2 * g] : -1;
        if (hvh >= 0 && cvHv.nhv > 0) {
            pNow = cvHvdcActivePower(cvHv, s, n, hvh, cvHv.hv[2 * g + 1]);
        } else if (slot >= 0 && (dgAnchor != nullptr || dgStart != nullptr
                                 || genCum != nullptr && genCumValid != nullptr && genCumValid[s])) {
            // the generator's own P: its anchor, plus this solve's water-fill or, before any pass, where the
            // solve started it (genPNowMove). A share of its bus row's move is not that once anything clamps.
            pNow = cvIn[g * 5] + genPNowMove(s, slot, ngen, genCum, genCumValid, dgAnchor, dgStart);
        } else {
            pNow = cvIn[g * 5]
                    + cvIn[g * 5 + 1] * shareScale * (target[(size_t) s * n + pRow] - cvIn[g * 5 + 2] - pStep);
        }
        double mnAdd = rlCurveEval(pNow * 100.0, pts, npts, 0, cvExtrapolate) / 100.0 - cvIn[g * 5 + 3];
        double mxAdd = rlCurveEval(pNow * 100.0, pts, npts, 1, cvExtrapolate) / 100.0 - cvIn[g * 5 + 4];
        if ((cvProbeEntry == -2 || (cvProbeEntry >= 0 && i == cvProbeEntry)) && s == cvProbeSlot) {
            // OLF_GPU_CV_PROBE=<rl entry>: the P this curve is evaluated AT, and where that P came
            // from. OLF re-reads bus.getMinQ() at the generator's CURRENT targetP; the device
            // reconstructs it, and the two only agree while the reconstruction does.
            printf("[CV-PROBE] s=%d entry=%d g=%d slot=%d genCumValid=%d basePpu=%.12f genCum=%.12f"
                   " share=%.12f baseRowTargetP=%.12f rowTarget=%.12f pStep=%.12f shareScale=%.6f"
                   " pNow=%.12f mnAdd=%.12e mxAdd=%.12e\n",
                   s, i, g, slot, (int) (genCumValid != nullptr && genCumValid[s]), cvIn[g * 5],
                   (genCum != nullptr && slot >= 0) ? genCum[(size_t) s * ngen + slot] : 0.0,
                   cvIn[g * 5 + 1], cvIn[g * 5 + 2], target[(size_t) s * n + pRow], pStep, shareScale,
                   pNow, mnAdd, mxAdd);
        }
        mn += mnAdd;
        mx += mxAdd;
    }
    // A generator lost at a bus that KEEPS voltage control takes its reactive capability with it. The
    // static aggregate is run-constant and still holds it, while OLF's bus.getMinQ()/getMaxQ() sum only
    // the generators still there, so the scenario's own delta comes in here. Without curves this is the
    // ONLY reason the dynamic limits exist, which is why the seeding above starts from the static pair.
    // ...the scenario's OWN overrides, CSR by slot: the flat {slot, entry} scan cost nrl * nS * nQOv reads per
    // launch, with nQOv about nS on a generator-outage chunk (one override per scenario).
    for (int c = qOvPtr != nullptr ? qOvPtr[s] : 0; c < (qOvPtr != nullptr ? qOvPtr[s + 1] : 0); c++) {
        int k = qOvList[c];
        if (qOvSlotEntry[2 * k + 1] == i) {
            mn += qOvMinMax[5 * k];
            mx += qOvMinMax[5 * k + 1];
            lq += qOvMinMax[5 * k + 2];                      // a SHED load's reactive power: the pinned
        }                                                    // target and the controller's own q both
    }                                                        // measure against the bus's load target Q
    minQDyn[(size_t) s * nrl + i] = mn;
    maxQDyn[(size_t) s * nrl + i] = mx;
    loadQDyn[(size_t) s * nrl + i] = lq;
}

// The curve kernel's view of the AC-emulation loop state (see CvHvdc).
CvHvdc cvHvdcView(BatchSolverContext* ctx) {
    CvHvdc c{};
    if (!ctx->cvHasHvdc) {
        return c;
    }
    c.hv = ctx->dCvHv.as<int>();
    c.nhv = ctx->nhv;
    if (ctx->nhv > 0) {
        c.status = ctx->dHvStatus.as<int>();
        c.d1 = ctx->dHvAppliedD1.as<double>();
        c.d2 = ctx->dHvAppliedD2.as<double>();
        c.in = ctx->dHvIn.as<double>();
        c.phi1 = ctx->dHvBus1Phi.as<int>();
        c.phi2 = ctx->dHvBus2Phi.as<int>();
        c.elem = ctx->dHvElem.as<int>();
        c.disabled = ctx->dDisabled.as<char>();
        c.nElements = ctx->nElements;
        c.ctgOut = ctx->dHvCtgOut.as<char>();
        c.x = ctx->dX.as<double>();
    }
    return c;
}

}  // namespace olfgpu
