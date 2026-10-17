// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Device helpers shared by the reactive-limit curve evaluation and the forced target Q: a capability curve
// read at an active power, and a generator's active power move this scenario.
#ifndef OLF_GPU_OUTERLOOP_REACTIVE_CAPABILITY_CURVE_CUH
#define OLF_GPU_OUTERLOOP_REACTIVE_CAPABILITY_CURVE_CUH

#include <cmath>
#include <cuda_runtime.h>

namespace olfgpu {

// Reactive-limits (PV↔PQ) pass: thread = (controller, scenario), state per (controller, scenario) in
// limitType (0 PV / 1 minQ / 2 maxQ) + switchCount (PV→PQ transitions, capped to block oscillation).
//  - PV (mode 0): generator q = (Σ branch q at the BUS_V row, in qBuf) + loadQ. If q < minQ−tol, pin to
//    minQ; if q > maxQ+tol, pin to maxQ → flip the row PV→PQ (mode 0→1, target = qLimit − loadQ), record
//    the limit type, increment the switch count.
//  - PQ (mode 1, pinned): if it has not yet switched MAX_SWITCH times, resume voltage control (PQ→PV) when
//    the voltage indicates the limit is no longer binding: a maxQ-pinned bus whose v rose ABOVE target, or
//    a minQ-pinned bus whose v fell BELOW target (mirrors ReactiveLimitsOuterLoop). Restore mode 1→0,
//    target → Vtarget. The count cap (matching OLF's maxPqPvSwitch) bounds oscillation. SLOPE-SVC buses
//    never back-switch (OLF: canSwitchToPv = !hasGeneratorsWithSlope()), and SVC limits are re-evaluated
//    at the live V (B*v^2 quadratic channel) at every check plus the pinned re-freeze.
// Either switch marks the scenario for another inner Newton solve.
// ReactiveCapabilityCurveImpl.getReactiveLimit, verbatim: exact point hit -> the point's value;
// inside [minP,maxP] -> floor/ceiling interpolation; outside -> CLAMP to the first/last point, or
// (extrapolate != 0, OLF's extrapolateReactiveLimits) the ReactiveCapabilityCurveUtil
// .extrapolateReactiveLimitsSlope math — the BOUNDARY two-point slope continued past the range,
// with the intersecting-limits corner (extrapolated minQ > maxQ) collapsing BOTH to their mean.
// sel 0 = minQ, 1 = maxQ. p in MW, result in MVAr.
static __device__ double rlCurveEval(double pMW, const double* pts, int npts, int sel, int extrapolate) {
    for (int k = 0; k < npts; k++) {
        if (pts[k * 3] == pMW) {
            return pts[k * 3 + 1 + sel];                     // first case: exact curve point
        }
    }
    if (pMW >= pts[0] && pMW <= pts[(npts - 1) * 3]) {       // second case: interpolate floor..ceiling
        int lo = 0;
        while (lo + 1 < npts && pts[(lo + 1) * 3] < pMW) {
            lo++;
        }
        double p1 = pts[lo * 3];
        double q1 = pts[lo * 3 + 1 + sel];
        double p2 = pts[(lo + 1) * 3];
        double q2 = pts[(lo + 1) * 3 + 1 + sel];
        return q1 + (q2 - q1) / (p2 - p1) * (pMW - p1);
    }
    if (!extrapolate) {
        return pMW < pts[0] ? pts[1 + sel] : pts[(npts - 1) * 3 + 1 + sel];   // third case: clamp
    }
    int bi = pMW < pts[0] ? 0 : npts - 1;                    // pBound = the boundary point,
    int si = pMW < pts[0] ? 1 : npts - 2;                    // pbis = its inward neighbor
    double pB = pts[bi * 3];
    double pS = pts[si * 3];
    double slopeMin = (pts[si * 3 + 1] - pts[bi * 3 + 1]) / (pS - pB);
    double slopeMax = (pts[si * 3 + 2] - pts[bi * 3 + 2]) / (pS - pB);
    double minQ = pts[bi * 3 + 1] + slopeMin * (pMW - pB);
    double maxQ = pts[bi * 3 + 2] + slopeMax * (pMW - pB);
    if (minQ <= maxQ) {
        return sel ? maxQ : minQ;
    }
    return (minQ + maxQ) / 2;                                // intersecting-limits corner: the mean, BOTH
}

// A participating generator's move from its base target P in scenario s: its anchor (a strategy pair's parent target,
// or a GeneratorAction's value; dgAnchor, null = the base) plus its own move from that anchor - this solve's water-fill
// (genCum) once a pass ran, else where the solve STARTED it (dgStart: a contingency's loss pre-distribution, which
// moves each generator by its own clamped share, not by a share of its bus's move).
__device__ inline double genPNowMove(int s, int g, int ngen, const double* genCum, const char* genCumValid,
                                     const double* dgAnchor, const double* dgStart) {
    size_t k = (size_t) s * ngen + g;
    double a = dgAnchor != nullptr ? dgAnchor[k] : 0.0;
    if (genCum != nullptr && genCumValid != nullptr && genCumValid[s]) {
        return a + genCum[k];
    }
    return a + (dgStart != nullptr ? dgStart[k] : 0.0);
}

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_REACTIVE_CAPABILITY_CURVE_CUH
