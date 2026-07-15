// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Closed-branch current magnitudes and their Jacobian entries, parsed from the hand-written OLF equation-term Java (the source of truth).
//
// GENERATED — DO NOT EDIT. Regenerate with: python3 codegen/java2cuda/java2cuda.py (parsed from the OLF Java).
#pragma once
#include <math.h>

#ifndef CBI_FN
#define CBI_FN __host__ __device__   // also compiles host-side for the diff test
#endif

namespace powsybl { namespace openloadflow {

#ifndef OLF_CONST_R2
#define OLF_CONST_R2
constexpr double R2 = 1.0;
#endif
#ifndef OLF_CONST_A2
#define OLF_CONST_A2
constexpr double A2 = 0.0;
#endif

// FUSED per-element kernel body: one routine computes the two closed-branch
// current magnitudes i1/i2 + their ten chain-rule derivatives for one branch,
// writing into Struct-of-Arrays at index i. The complex-current components
// reI1/imI1/reI2/imI2 are computed once as kernel locals and every derivative is
// (re*dred<var> + im*dimd<var>) / hypot(re, im) — OLF's own chain-rule shape.
// NEW GPU target (the residual pipeline emitted no CUDA for current magnitude);
// no derivative w.r.t. r1 exists, matching OLF (calculateSensi throws on dr1).
CBI_FN void evalClosedBranchI(
        double y, double g1, double g2, double b1, double b2,
        double v1, double r1, double v2, double ksi, double a1, double ph1, double ph2,
        int i,
        double* i1,
        double* di1dv1,
        double* di1dv2,
        double* di1dph1,
        double* di1dph2,
        double* di1da1,
        double* i2,
        double* di2dv1,
        double* di2dv2,
        double* di2dph1,
        double* di2dph2,
        double* di2da1) {
    double reI1 = r1 * (r1 * v1 * (g1 * cos(ph1) - b1 * sin(ph1) + y * sin(ph1 + ksi)) - y * R2 * v2 * sin(ksi - a1 + A2 + ph2));
    double imI1 = r1 * (r1 * v1 * (g1 * sin(ph1) + b1 * cos(ph1) - y * cos(ph1 + ksi)) + y * R2 * v2 * cos(ksi - a1 + A2 + ph2));
    double reI2 = R2 * (R2 * v2 * (g2 * cos(ph2) - b2 * sin(ph2) + y * sin(ph2 + ksi)) - y * r1 * v1 * sin(ksi + a1 - A2 + ph1));
    double imI2 = R2 * (R2 * v2 * (g2 * sin(ph2) + b2 * cos(ph2) - y * cos(ph2 + ksi)) + y * r1 * v1 * cos(ksi + a1 - A2 + ph1));
    i1[i] = (hypot(reI1, imI1));
    di1dv1[i] = (reI1 * (r1 * r1 * (g1 * cos(ph1) - b1 * sin(ph1) + y * sin(ph1 + ksi))) + imI1 * (r1 * r1 * (g1 * sin(ph1) + b1 * cos(ph1) - y * cos(ph1 + ksi)))) / (hypot(reI1, imI1));
    di1dv2[i] = (reI1 * (r1 * (-y * R2 * sin(ksi - a1 + A2 + ph2))) + imI1 * (r1 * (y * R2 * cos(ksi - a1 + A2 + ph2)))) / (hypot(reI1, imI1));
    di1dph1[i] = (reI1 * (r1 * r1 * v1 * (-g1 * sin(ph1) - b1 * cos(ph1) + y * cos(ph1 + ksi))) + imI1 * (r1 * r1 * v1 * (g1 * cos(ph1) - b1 * sin(ph1) + y * sin(ph1 + ksi)))) / (hypot(reI1, imI1));
    di1dph2[i] = (reI1 * (r1 * (-y * R2 * v2 * cos(ksi - a1 + A2 + ph2))) + imI1 * (r1 * (-y * R2 * v2 * sin(ksi - a1 + A2 + ph2)))) / (hypot(reI1, imI1));
    di1da1[i] = -((reI1 * (r1 * (-y * R2 * v2 * cos(ksi - a1 + A2 + ph2))) + imI1 * (r1 * (-y * R2 * v2 * sin(ksi - a1 + A2 + ph2)))) / (hypot(reI1, imI1)));
    i2[i] = (hypot(reI2, imI2));
    di2dv1[i] = (reI2 * (R2 * (-y * r1 * sin(ksi + a1 - A2 + ph1))) + imI2 * (R2 * (y * r1 * cos(ksi + a1 - A2 + ph1)))) / (hypot(reI2, imI2));
    di2dv2[i] = (reI2 * (R2 * R2 * (g2 * cos(ph2) - b2 * sin(ph2) + y * sin(ph2 + ksi))) + imI2 * (R2 * R2 * (g2 * sin(ph2) + b2 * cos(ph2) - y * cos(ph2 + ksi)))) / (hypot(reI2, imI2));
    di2dph1[i] = (reI2 * (R2 * (-y * r1 * v1 * cos(ksi + a1 - A2 + ph1))) + imI2 * (R2 * (-y * r1 * v1 * sin(ksi + a1 - A2 + ph1)))) / (hypot(reI2, imI2));
    di2dph2[i] = (reI2 * (R2 * R2 * v2 * (-g2 * sin(ph2) - b2 * cos(ph2) + y * cos(ph2 + ksi))) + imI2 * (R2 * R2 * v2 * (g2 * cos(ph2) - b2 * sin(ph2) + y * sin(ph2 + ksi)))) / (hypot(reI2, imI2));
    di2da1[i] = ((reI2 * (R2 * (-y * r1 * v1 * cos(ksi + a1 - A2 + ph1))) + imI2 * (R2 * (-y * r1 * v1 * sin(ksi + a1 - A2 + ph1)))) / (hypot(reI2, imI2)));
}

// F-ONLY variant: the residual values alone (no Jacobian outputs) — for fill
// sites that never factorize: residual/convergence checks, line-search
// backtracks, frozen-Jacobian iterations. Same parsed OLF formulas.
CBI_FN void evalClosedBranchIF(
        double y, double g1, double g2, double b1, double b2,
        double v1, double r1, double v2, double ksi, double a1, double ph1, double ph2,
        int i,
        double* i1,
        double* i2) {
    double reI1 = r1 * (r1 * v1 * (g1 * cos(ph1) - b1 * sin(ph1) + y * sin(ph1 + ksi)) - y * R2 * v2 * sin(ksi - a1 + A2 + ph2));
    double imI1 = r1 * (r1 * v1 * (g1 * sin(ph1) + b1 * cos(ph1) - y * cos(ph1 + ksi)) + y * R2 * v2 * cos(ksi - a1 + A2 + ph2));
    double reI2 = R2 * (R2 * v2 * (g2 * cos(ph2) - b2 * sin(ph2) + y * sin(ph2 + ksi)) - y * r1 * v1 * sin(ksi + a1 - A2 + ph1));
    double imI2 = R2 * (R2 * v2 * (g2 * sin(ph2) + b2 * cos(ph2) - y * cos(ph2 + ksi)) + y * r1 * v1 * cos(ksi + a1 - A2 + ph1));
    i1[i] = (hypot(reI1, imI1));
    i2[i] = (hypot(reI2, imI2));
}

}}  // namespace powsybl::openloadflow
