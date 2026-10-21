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

// One device function per OLF Java method — SAME name, SAME signature, SAME body
// (the CPU methods are the optimized precomputed-trig form; the family namespace
// mirrors the Java class scoping). The eval* routines below only assemble calls,
// exactly like AcNetworkVector on the CPU.
namespace closed_branch_i {

CBI_FN inline double i1(double y, double ksi, double g1, double b1, double v1, double ph1, double r1, double a1, double v2, double ph2) {
    return (hypot((r1 * (r1 * v1 * (g1 * cos(ph1) - b1 * sin(ph1) + y * sin(ph1 + ksi)) - y * R2 * v2 * sin(ksi - a1 + A2 + ph2))), (r1 * (r1 * v1 * (g1 * sin(ph1) + b1 * cos(ph1) - y * cos(ph1 + ksi)) + y * R2 * v2 * cos(ksi - a1 + A2 + ph2)))));
}

CBI_FN inline double di1dv1(double y, double ksi, double g1, double b1, double v1, double ph1, double r1, double a1, double v2, double ph2) {
    return ((r1 * (r1 * v1 * (g1 * cos(ph1) - b1 * sin(ph1) + y * sin(ph1 + ksi)) - y * R2 * v2 * sin(ksi - a1 + A2 + ph2))) * (r1 * r1 * (g1 * cos(ph1) - b1 * sin(ph1) + y * sin(ph1 + ksi))) + (r1 * (r1 * v1 * (g1 * sin(ph1) + b1 * cos(ph1) - y * cos(ph1 + ksi)) + y * R2 * v2 * cos(ksi - a1 + A2 + ph2))) * (r1 * r1 * (g1 * sin(ph1) + b1 * cos(ph1) - y * cos(ph1 + ksi)))) / (hypot((r1 * (r1 * v1 * (g1 * cos(ph1) - b1 * sin(ph1) + y * sin(ph1 + ksi)) - y * R2 * v2 * sin(ksi - a1 + A2 + ph2))), (r1 * (r1 * v1 * (g1 * sin(ph1) + b1 * cos(ph1) - y * cos(ph1 + ksi)) + y * R2 * v2 * cos(ksi - a1 + A2 + ph2)))));
}

CBI_FN inline double di1dv2(double y, double ksi, double g1, double b1, double v1, double ph1, double r1, double a1, double v2, double ph2) {
    return ((r1 * (r1 * v1 * (g1 * cos(ph1) - b1 * sin(ph1) + y * sin(ph1 + ksi)) - y * R2 * v2 * sin(ksi - a1 + A2 + ph2))) * (r1 * (-y * R2 * sin(ksi - a1 + A2 + ph2))) + (r1 * (r1 * v1 * (g1 * sin(ph1) + b1 * cos(ph1) - y * cos(ph1 + ksi)) + y * R2 * v2 * cos(ksi - a1 + A2 + ph2))) * (r1 * (y * R2 * cos(ksi - a1 + A2 + ph2)))) / (hypot((r1 * (r1 * v1 * (g1 * cos(ph1) - b1 * sin(ph1) + y * sin(ph1 + ksi)) - y * R2 * v2 * sin(ksi - a1 + A2 + ph2))), (r1 * (r1 * v1 * (g1 * sin(ph1) + b1 * cos(ph1) - y * cos(ph1 + ksi)) + y * R2 * v2 * cos(ksi - a1 + A2 + ph2)))));
}

CBI_FN inline double di1dph1(double y, double ksi, double g1, double b1, double v1, double ph1, double r1, double a1, double v2, double ph2) {
    return ((r1 * (r1 * v1 * (g1 * cos(ph1) - b1 * sin(ph1) + y * sin(ph1 + ksi)) - y * R2 * v2 * sin(ksi - a1 + A2 + ph2))) * (r1 * r1 * v1 * (-g1 * sin(ph1) - b1 * cos(ph1) + y * cos(ph1 + ksi))) + (r1 * (r1 * v1 * (g1 * sin(ph1) + b1 * cos(ph1) - y * cos(ph1 + ksi)) + y * R2 * v2 * cos(ksi - a1 + A2 + ph2))) * (r1 * r1 * v1 * (g1 * cos(ph1) - b1 * sin(ph1) + y * sin(ph1 + ksi)))) / (hypot((r1 * (r1 * v1 * (g1 * cos(ph1) - b1 * sin(ph1) + y * sin(ph1 + ksi)) - y * R2 * v2 * sin(ksi - a1 + A2 + ph2))), (r1 * (r1 * v1 * (g1 * sin(ph1) + b1 * cos(ph1) - y * cos(ph1 + ksi)) + y * R2 * v2 * cos(ksi - a1 + A2 + ph2)))));
}

CBI_FN inline double di1dph2(double y, double ksi, double g1, double b1, double v1, double ph1, double r1, double a1, double v2, double ph2) {
    return ((r1 * (r1 * v1 * (g1 * cos(ph1) - b1 * sin(ph1) + y * sin(ph1 + ksi)) - y * R2 * v2 * sin(ksi - a1 + A2 + ph2))) * (r1 * (-y * R2 * v2 * cos(ksi - a1 + A2 + ph2))) + (r1 * (r1 * v1 * (g1 * sin(ph1) + b1 * cos(ph1) - y * cos(ph1 + ksi)) + y * R2 * v2 * cos(ksi - a1 + A2 + ph2))) * (r1 * (-y * R2 * v2 * sin(ksi - a1 + A2 + ph2)))) / (hypot((r1 * (r1 * v1 * (g1 * cos(ph1) - b1 * sin(ph1) + y * sin(ph1 + ksi)) - y * R2 * v2 * sin(ksi - a1 + A2 + ph2))), (r1 * (r1 * v1 * (g1 * sin(ph1) + b1 * cos(ph1) - y * cos(ph1 + ksi)) + y * R2 * v2 * cos(ksi - a1 + A2 + ph2)))));
}

CBI_FN inline double di1da1(double y, double ksi, double g1, double b1, double v1, double ph1, double r1, double a1, double v2, double ph2) {
    return -(((r1 * (r1 * v1 * (g1 * cos(ph1) - b1 * sin(ph1) + y * sin(ph1 + ksi)) - y * R2 * v2 * sin(ksi - a1 + A2 + ph2))) * (r1 * (-y * R2 * v2 * cos(ksi - a1 + A2 + ph2))) + (r1 * (r1 * v1 * (g1 * sin(ph1) + b1 * cos(ph1) - y * cos(ph1 + ksi)) + y * R2 * v2 * cos(ksi - a1 + A2 + ph2))) * (r1 * (-y * R2 * v2 * sin(ksi - a1 + A2 + ph2)))) / (hypot((r1 * (r1 * v1 * (g1 * cos(ph1) - b1 * sin(ph1) + y * sin(ph1 + ksi)) - y * R2 * v2 * sin(ksi - a1 + A2 + ph2))), (r1 * (r1 * v1 * (g1 * sin(ph1) + b1 * cos(ph1) - y * cos(ph1 + ksi)) + y * R2 * v2 * cos(ksi - a1 + A2 + ph2))))));
}

CBI_FN inline double i2(double y, double ksi, double g2, double b2, double v1, double ph1, double r1, double a1, double v2, double ph2) {
    return (hypot((R2 * (R2 * v2 * (g2 * cos(ph2) - b2 * sin(ph2) + y * sin(ph2 + ksi)) - y * r1 * v1 * sin(ksi + a1 - A2 + ph1))), (R2 * (R2 * v2 * (g2 * sin(ph2) + b2 * cos(ph2) - y * cos(ph2 + ksi)) + y * r1 * v1 * cos(ksi + a1 - A2 + ph1)))));
}

CBI_FN inline double di2dv1(double y, double ksi, double g2, double b2, double v1, double ph1, double r1, double a1, double v2, double ph2) {
    return ((R2 * (R2 * v2 * (g2 * cos(ph2) - b2 * sin(ph2) + y * sin(ph2 + ksi)) - y * r1 * v1 * sin(ksi + a1 - A2 + ph1))) * (R2 * (-y * r1 * sin(ksi + a1 - A2 + ph1))) + (R2 * (R2 * v2 * (g2 * sin(ph2) + b2 * cos(ph2) - y * cos(ph2 + ksi)) + y * r1 * v1 * cos(ksi + a1 - A2 + ph1))) * (R2 * (y * r1 * cos(ksi + a1 - A2 + ph1)))) / (hypot((R2 * (R2 * v2 * (g2 * cos(ph2) - b2 * sin(ph2) + y * sin(ph2 + ksi)) - y * r1 * v1 * sin(ksi + a1 - A2 + ph1))), (R2 * (R2 * v2 * (g2 * sin(ph2) + b2 * cos(ph2) - y * cos(ph2 + ksi)) + y * r1 * v1 * cos(ksi + a1 - A2 + ph1)))));
}

CBI_FN inline double di2dv2(double y, double ksi, double g2, double b2, double v1, double ph1, double r1, double a1, double v2, double ph2) {
    return ((R2 * (R2 * v2 * (g2 * cos(ph2) - b2 * sin(ph2) + y * sin(ph2 + ksi)) - y * r1 * v1 * sin(ksi + a1 - A2 + ph1))) * (R2 * R2 * (g2 * cos(ph2) - b2 * sin(ph2) + y * sin(ph2 + ksi))) + (R2 * (R2 * v2 * (g2 * sin(ph2) + b2 * cos(ph2) - y * cos(ph2 + ksi)) + y * r1 * v1 * cos(ksi + a1 - A2 + ph1))) * (R2 * R2 * (g2 * sin(ph2) + b2 * cos(ph2) - y * cos(ph2 + ksi)))) / (hypot((R2 * (R2 * v2 * (g2 * cos(ph2) - b2 * sin(ph2) + y * sin(ph2 + ksi)) - y * r1 * v1 * sin(ksi + a1 - A2 + ph1))), (R2 * (R2 * v2 * (g2 * sin(ph2) + b2 * cos(ph2) - y * cos(ph2 + ksi)) + y * r1 * v1 * cos(ksi + a1 - A2 + ph1)))));
}

CBI_FN inline double di2dph1(double y, double ksi, double g2, double b2, double v1, double ph1, double r1, double a1, double v2, double ph2) {
    return ((R2 * (R2 * v2 * (g2 * cos(ph2) - b2 * sin(ph2) + y * sin(ph2 + ksi)) - y * r1 * v1 * sin(ksi + a1 - A2 + ph1))) * (R2 * (-y * r1 * v1 * cos(ksi + a1 - A2 + ph1))) + (R2 * (R2 * v2 * (g2 * sin(ph2) + b2 * cos(ph2) - y * cos(ph2 + ksi)) + y * r1 * v1 * cos(ksi + a1 - A2 + ph1))) * (R2 * (-y * r1 * v1 * sin(ksi + a1 - A2 + ph1)))) / (hypot((R2 * (R2 * v2 * (g2 * cos(ph2) - b2 * sin(ph2) + y * sin(ph2 + ksi)) - y * r1 * v1 * sin(ksi + a1 - A2 + ph1))), (R2 * (R2 * v2 * (g2 * sin(ph2) + b2 * cos(ph2) - y * cos(ph2 + ksi)) + y * r1 * v1 * cos(ksi + a1 - A2 + ph1)))));
}

CBI_FN inline double di2dph2(double y, double ksi, double g2, double b2, double v1, double ph1, double r1, double a1, double v2, double ph2) {
    return ((R2 * (R2 * v2 * (g2 * cos(ph2) - b2 * sin(ph2) + y * sin(ph2 + ksi)) - y * r1 * v1 * sin(ksi + a1 - A2 + ph1))) * (R2 * R2 * v2 * (-g2 * sin(ph2) - b2 * cos(ph2) + y * cos(ph2 + ksi))) + (R2 * (R2 * v2 * (g2 * sin(ph2) + b2 * cos(ph2) - y * cos(ph2 + ksi)) + y * r1 * v1 * cos(ksi + a1 - A2 + ph1))) * (R2 * R2 * v2 * (g2 * cos(ph2) - b2 * sin(ph2) + y * sin(ph2 + ksi)))) / (hypot((R2 * (R2 * v2 * (g2 * cos(ph2) - b2 * sin(ph2) + y * sin(ph2 + ksi)) - y * r1 * v1 * sin(ksi + a1 - A2 + ph1))), (R2 * (R2 * v2 * (g2 * sin(ph2) + b2 * cos(ph2) - y * cos(ph2 + ksi)) + y * r1 * v1 * cos(ksi + a1 - A2 + ph1)))));
}

CBI_FN inline double di2da1(double y, double ksi, double g2, double b2, double v1, double ph1, double r1, double a1, double v2, double ph2) {
    return (((R2 * (R2 * v2 * (g2 * cos(ph2) - b2 * sin(ph2) + y * sin(ph2 + ksi)) - y * r1 * v1 * sin(ksi + a1 - A2 + ph1))) * (R2 * (-y * r1 * v1 * cos(ksi + a1 - A2 + ph1))) + (R2 * (R2 * v2 * (g2 * sin(ph2) + b2 * cos(ph2) - y * cos(ph2 + ksi)) + y * r1 * v1 * cos(ksi + a1 - A2 + ph1))) * (R2 * (-y * r1 * v1 * sin(ksi + a1 - A2 + ph1)))) / (hypot((R2 * (R2 * v2 * (g2 * cos(ph2) - b2 * sin(ph2) + y * sin(ph2 + ksi)) - y * r1 * v1 * sin(ksi + a1 - A2 + ph1))), (R2 * (R2 * v2 * (g2 * sin(ph2) + b2 * cos(ph2) - y * cos(ph2 + ksi)) + y * r1 * v1 * cos(ksi + a1 - A2 + ph1))))));
}

// ClosedBranchSide1CurrentMagnitudeEquationTerm's private helpers, as OLF scopes them (per class: both sides define their own theta).
namespace ClosedBranchSide1CurrentMagnitudeEquationTerm {

CBI_FN inline double dreI1dph1(double y, double ksi, double g1, double b1, double v1, double ph1, double r1) {
    return r1 * r1 * v1 * (-g1 * sin(ph1) - b1 * cos(ph1) + y * cos(ph1 + ksi));
}

CBI_FN inline double dreI1dph2(double y, double ksi, double r1, double a1, double v2, double ph2) {
    return r1 * (-y * R2 * v2 * cos(ksi - a1 + A2 + ph2));
}

CBI_FN inline double dreI1dv1(double y, double ksi, double g1, double b1, double ph1, double r1) {
    return r1 * r1 * (g1 * cos(ph1) - b1 * sin(ph1) + y * sin(ph1 + ksi));
}

CBI_FN inline double dreI1dv2(double y, double ksi, double r1, double a1, double ph2) {
    return r1 * (-y * R2 * sin(ksi - a1 + A2 + ph2));
}

CBI_FN inline double dimI1dph1(double y, double ksi, double g1, double b1, double v1, double ph1, double r1) {
    return r1 * r1 * v1 * (g1 * cos(ph1) - b1 * sin(ph1) + y * sin(ph1 + ksi));
}

CBI_FN inline double dimI1dph2(double y, double ksi, double r1, double a1, double v2, double ph2) {
    return r1 * (-y * R2 * v2 * sin(ksi - a1 + A2 + ph2));
}

CBI_FN inline double dimI1dv1(double y, double ksi, double g1, double b1, double ph1, double r1) {
    return r1 * r1 * (g1 * sin(ph1) + b1 * cos(ph1) - y * cos(ph1 + ksi));
}

CBI_FN inline double dimI1dv2(double y, double ksi, double r1, double a1, double ph2) {
    return r1 * (y * R2 * cos(ksi - a1 + A2 + ph2));
}

}  // namespace ClosedBranchSide1CurrentMagnitudeEquationTerm

// ClosedBranchSide2CurrentMagnitudeEquationTerm's private helpers, as OLF scopes them (per class: both sides define their own theta).
namespace ClosedBranchSide2CurrentMagnitudeEquationTerm {

CBI_FN inline double theta(double ksi, double ph1, double a1) {
    return ksi + a1 - A2 + ph1;
}

CBI_FN inline double dreI2dph1(double y, double ksi, double v1, double ph1, double r1, double a1) {
    return R2 * (-y * r1 * v1 * cos(ksi + a1 - A2 + ph1));
}

CBI_FN inline double dreI2dph2(double y, double ksi, double g2, double b2, double v2, double ph2) {
    return R2 * R2 * v2 * (-g2 * sin(ph2) - b2 * cos(ph2) + y * cos(ph2 + ksi));
}

CBI_FN inline double dreI2dv1(double y, double r1, double theta) {
    return R2 * (-y * r1 * sin(theta));
}

CBI_FN inline double dreI2dv2(double y, double ksi, double g2, double b2, double ph2) {
    return R2 * R2 * (g2 * cos(ph2) - b2 * sin(ph2) + y * sin(ph2 + ksi));
}

CBI_FN inline double dimI2dph1(double y, double v1, double r1, double theta) {
    return R2 * (-y * r1 * v1 * sin(theta));
}

CBI_FN inline double dimI2dph2(double y, double ksi, double g2, double b2, double v2, double ph2) {
    return R2 * R2 * v2 * (g2 * cos(ph2) - b2 * sin(ph2) + y * sin(ph2 + ksi));
}

CBI_FN inline double dimI2dv1(double y, double ksi, double ph1, double r1, double a1) {
    return R2 * (y * r1 * cos(ksi + a1 - A2 + ph1));
}

CBI_FN inline double dimI2dv2(double y, double ksi, double g2, double b2, double ph2) {
    return R2 * R2 * (g2 * sin(ph2) + b2 * cos(ph2) - y * cos(ph2 + ksi));
}

}  // namespace ClosedBranchSide2CurrentMagnitudeEquationTerm

// FUSED per-element kernel body: one routine computes the two closed-branch
// current magnitudes i1/i2 + their ten chain-rule derivatives for one branch,
// writing into Struct-of-Arrays at index i. The complex-current components
// reI1/imI1/reI2/imI2 are computed once as kernel locals and every derivative is
// (re*dred<var> + im*dimd<var>) / hypot(re, im) — OLF's own chain-rule shape.
// NEW GPU target (the residual pipeline emitted no CUDA for current magnitude);
// no derivative w.r.t. r1 exists, matching OLF (calculateSensi throws on dr1).
CBI_FN inline void evalClosedBranchI(
        double y, double g1, double g2, double b1, double b2,
        double v1, double r1, double v2, double ksi, double a1, double ph1, double ph2,
        int i,
        double* i1_out,
        double* di1dv1_out,
        double* di1dv2_out,
        double* di1dph1_out,
        double* di1dph2_out,
        double* di1da1_out,
        double* i2_out,
        double* di2dv1_out,
        double* di2dv2_out,
        double* di2dph1_out,
        double* di2dph2_out,
        double* di2da1_out) {
    i1_out[i] = i1(y, ksi, g1, b1, v1, ph1, r1, a1, v2, ph2);
    di1dv1_out[i] = di1dv1(y, ksi, g1, b1, v1, ph1, r1, a1, v2, ph2);
    di1dv2_out[i] = di1dv2(y, ksi, g1, b1, v1, ph1, r1, a1, v2, ph2);
    di1dph1_out[i] = di1dph1(y, ksi, g1, b1, v1, ph1, r1, a1, v2, ph2);
    di1dph2_out[i] = di1dph2(y, ksi, g1, b1, v1, ph1, r1, a1, v2, ph2);
    di1da1_out[i] = di1da1(y, ksi, g1, b1, v1, ph1, r1, a1, v2, ph2);
    i2_out[i] = i2(y, ksi, g2, b2, v1, ph1, r1, a1, v2, ph2);
    di2dv1_out[i] = di2dv1(y, ksi, g2, b2, v1, ph1, r1, a1, v2, ph2);
    di2dv2_out[i] = di2dv2(y, ksi, g2, b2, v1, ph1, r1, a1, v2, ph2);
    di2dph1_out[i] = di2dph1(y, ksi, g2, b2, v1, ph1, r1, a1, v2, ph2);
    di2dph2_out[i] = di2dph2(y, ksi, g2, b2, v1, ph1, r1, a1, v2, ph2);
    di2da1_out[i] = di2da1(y, ksi, g2, b2, v1, ph1, r1, a1, v2, ph2);
}

// F-ONLY variant: the residual values alone (no Jacobian outputs) — for fill
// sites that never factorize: residual/convergence checks, line-search
// backtracks, frozen-Jacobian iterations.
CBI_FN inline void evalClosedBranchIF(
        double y, double g1, double g2, double b1, double b2,
        double v1, double r1, double v2, double ksi, double a1, double ph1, double ph2,
        int i,
        double* i1_out,
        double* i2_out) {
    i1_out[i] = i1(y, ksi, g1, b1, v1, ph1, r1, a1, v2, ph2);
    i2_out[i] = i2(y, ksi, g2, b2, v1, ph1, r1, a1, v2, ph2);
}

}  // namespace closed_branch_i

using closed_branch_i::evalClosedBranchI;
using closed_branch_i::evalClosedBranchIF;

}}  // namespace powsybl::openloadflow
