// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// HVDC AC-emulation active-power residuals and (approximated) Jacobian entries, parsed from the hand-written OLF equation-term Java (the source of truth).
//
// GENERATED — DO NOT EDIT. Regenerate with: python3 codegen/java2cuda/java2cuda.py (parsed from the OLF Java).
#pragma once
#include <math.h>

#ifndef HVDC_AC_EMULATION_FN
#define HVDC_AC_EMULATION_FN __host__ __device__   // also compiles host-side for the diff test
#endif

namespace powsybl { namespace openloadflow {


// One device function per OLF Java method — SAME name, SAME signature, SAME body
// (the CPU methods are the optimized precomputed-trig form; the family namespace
// mirrors the Java class scoping). The eval* routines below only assemble calls,
// exactly like AcNetworkVector on the CPU.
namespace hvdc_ac_emulation {

HVDC_AC_EMULATION_FN inline double p1(double p0, double k, double lossFactor1, double lossFactor2, double r, double ph1, double ph2) {
    return ((p0 + k * (ph1 - ph2)) >= 0) ? (p0 + k * (ph1 - ph2)) : -((1 - lossFactor2) * (((1 - lossFactor1) * fabs(p0 + k * (ph1 - ph2))) - (r * ((1 - lossFactor1) * fabs(p0 + k * (ph1 - ph2))) * ((1 - lossFactor1) * fabs(p0 + k * (ph1 - ph2))) / (1 * 1))));
}

HVDC_AC_EMULATION_FN inline double dp1dph1(double p0, double k, double lossFactor1, double lossFactor2, double ph1, double ph2) {
    return ((p0 + k * (ph1 - ph2)) >= 0) ? k : k * ((1 - lossFactor1) * (1 - lossFactor2));
}

HVDC_AC_EMULATION_FN inline double dp1dph2(double p0, double k, double lossFactor1, double lossFactor2, double ph1, double ph2) {
    return -(((p0 + k * (ph1 - ph2)) >= 0) ? k : k * ((1 - lossFactor1) * (1 - lossFactor2)));
}

HVDC_AC_EMULATION_FN inline double p2(double p0, double k, double lossFactor1, double lossFactor2, double r, double ph1, double ph2) {
    return ((p0 + k * (ph1 - ph2)) < 0) ? -(p0 + k * (ph1 - ph2)) : -((1 - lossFactor1) * (((1 - lossFactor2) * fabs(p0 + k * (ph1 - ph2))) - (r * ((1 - lossFactor2) * fabs(p0 + k * (ph1 - ph2))) * ((1 - lossFactor2) * fabs(p0 + k * (ph1 - ph2))) / (1 * 1))));
}

HVDC_AC_EMULATION_FN inline double dp2dph1(double p0, double k, double lossFactor1, double lossFactor2, double ph1, double ph2) {
    return ((p0 + k * (ph1 - ph2)) < 0) ? -k : -k * ((1 - lossFactor1) * (1 - lossFactor2));
}

HVDC_AC_EMULATION_FN inline double dp2dph2(double p0, double k, double lossFactor1, double lossFactor2, double ph1, double ph2) {
    return -(((p0 + k * (ph1 - ph2)) < 0) ? -k : -k * ((1 - lossFactor1) * (1 - lossFactor2)));
}

// FUSED per-element kernel body: one routine computes the two HVDC AC-emulation
// injections p1/p2 + their four (approximated, loss-derivative-neglecting) angle
// derivatives for one HVDC line, writing into Struct-of-Arrays at index i. The control
// side is piecewise on the sign of rawP = p0 + k*(ph1 - ph2) — branchless ternaries,
// no trig; rawP is shared as a kernel local.
HVDC_AC_EMULATION_FN inline void updateHvdcAcEmulation(
        double p0, double k, double lossFactor1, double lossFactor2, double r,
        double ph1, double ph2,
        int i,
        double* p1_out,
        double* dp1dph1_out,
        double* dp1dph2_out,
        double* p2_out,
        double* dp2dph1_out,
        double* dp2dph2_out) {
    p1_out[i] = p1(p0, k, lossFactor1, lossFactor2, r, ph1, ph2);
    dp1dph1_out[i] = dp1dph1(p0, k, lossFactor1, lossFactor2, ph1, ph2);
    dp1dph2_out[i] = dp1dph2(p0, k, lossFactor1, lossFactor2, ph1, ph2);
    p2_out[i] = p2(p0, k, lossFactor1, lossFactor2, r, ph1, ph2);
    dp2dph1_out[i] = dp2dph1(p0, k, lossFactor1, lossFactor2, ph1, ph2);
    dp2dph2_out[i] = dp2dph2(p0, k, lossFactor1, lossFactor2, ph1, ph2);
}

// VALUES-ONLY variant (OLF's withDerivatives = false): the residual values alone,
// no Jacobian outputs — for fill sites that never factorize: residual/convergence
// checks, line-search backtracks, frozen-Jacobian iterations.
HVDC_AC_EMULATION_FN inline void updateHvdcAcEmulationWithoutDerivatives(
        double p0, double k, double lossFactor1, double lossFactor2, double r,
        double ph1, double ph2,
        int i,
        double* p1_out,
        double* p2_out) {
    p1_out[i] = p1(p0, k, lossFactor1, lossFactor2, r, ph1, ph2);
    p2_out[i] = p2(p0, k, lossFactor1, lossFactor2, r, ph1, ph2);
}

}  // namespace hvdc_ac_emulation

using hvdc_ac_emulation::updateHvdcAcEmulation;
using hvdc_ac_emulation::updateHvdcAcEmulationWithoutDerivatives;

}}  // namespace powsybl::openloadflow
