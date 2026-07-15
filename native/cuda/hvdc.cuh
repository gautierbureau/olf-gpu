// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// HVDC AC-emulation active-power residuals and (approximated) Jacobian entries, parsed from the hand-written OLF equation-term Java (the source of truth).
//
// GENERATED — DO NOT EDIT. Regenerate with: python3 codegen/java2cuda/java2cuda.py (parsed from the OLF Java).
#pragma once
#include <math.h>

#ifndef HVDC_FN
#define HVDC_FN __host__ __device__   // also compiles host-side for the diff test
#endif

namespace powsybl { namespace openloadflow {



// FUSED per-element kernel body: one routine computes the two HVDC AC-emulation
// injections p1/p2 + their four (approximated, loss-derivative-neglecting) angle
// derivatives for one hvdc, writing into Struct-of-Arrays at index i. The control
// side is piecewise on the sign of rawP = p0 + k*(ph1 - ph2) — branchless ternaries,
// no trig; rawP is shared as a kernel local.
HVDC_FN void evalHvdc(
        double p0, double k, double lossFactor1, double lossFactor2, double r,
        double ph1, double ph2,
        int i,
        double* p1,
        double* dp1dph1,
        double* dp1dph2,
        double* p2,
        double* dp2dph1,
        double* dp2dph2) {
    double rawP = p0 + k * (ph1 - ph2);
    p1[i] = (rawP >= 0) ? rawP : -((1 - lossFactor2) * (((1 - lossFactor1) * fabs(rawP)) - (r * ((1 - lossFactor1) * fabs(rawP)) * ((1 - lossFactor1) * fabs(rawP)) / (1 * 1))));
    dp1dph1[i] = (rawP >= 0) ? k : k * ((1 - lossFactor1) * (1 - lossFactor2));
    dp1dph2[i] = -((rawP >= 0) ? k : k * ((1 - lossFactor1) * (1 - lossFactor2)));
    p2[i] = (rawP < 0) ? -rawP : -((1 - lossFactor1) * (((1 - lossFactor2) * fabs(rawP)) - (r * ((1 - lossFactor2) * fabs(rawP)) * ((1 - lossFactor2) * fabs(rawP)) / (1 * 1))));
    dp2dph1[i] = (rawP < 0) ? -k : -k * ((1 - lossFactor1) * (1 - lossFactor2));
    dp2dph2[i] = -((rawP < 0) ? -k : -k * ((1 - lossFactor1) * (1 - lossFactor2)));
}

// F-ONLY variant: the residual values alone (no Jacobian outputs) — for fill
// sites that never factorize: residual/convergence checks, line-search
// backtracks, frozen-Jacobian iterations. Same parsed OLF formulas.
HVDC_FN void evalHvdcF(
        double p0, double k, double lossFactor1, double lossFactor2, double r,
        double ph1, double ph2,
        int i,
        double* p1,
        double* p2) {
    double rawP = p0 + k * (ph1 - ph2);
    p1[i] = (rawP >= 0) ? rawP : -((1 - lossFactor2) * (((1 - lossFactor1) * fabs(rawP)) - (r * ((1 - lossFactor1) * fabs(rawP)) * ((1 - lossFactor1) * fabs(rawP)) / (1 * 1))));
    p2[i] = (rawP < 0) ? -rawP : -((1 - lossFactor1) * (((1 - lossFactor2) * fabs(rawP)) - (r * ((1 - lossFactor2) * fabs(rawP)) * ((1 - lossFactor2) * fabs(rawP)) / (1 * 1))));
}

}}  // namespace powsybl::openloadflow
