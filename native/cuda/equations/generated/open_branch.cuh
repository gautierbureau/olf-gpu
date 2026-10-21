// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Open-branch AC active/reactive flow residuals and Jacobian entries, parsed from the hand-written OLF equation-term Java (the source of truth).
//
// GENERATED — DO NOT EDIT. Regenerate with: python3 codegen/java2cuda/java2cuda.py (parsed from the OLF Java).
#pragma once
#include <math.h>

#ifndef OB_FN
#define OB_FN __host__ __device__   // also compiles host-side for the diff test
#endif

namespace powsybl { namespace openloadflow {

#ifndef OLF_CONST_R2
#define OLF_CONST_R2
constexpr double R2 = 1.0;
#endif

// One device function per OLF Java method — SAME name, SAME signature, SAME body
// (the CPU methods are the optimized precomputed-trig form; the family namespace
// mirrors the Java class scoping). The eval* routines below only assemble calls,
// exactly like AcNetworkVector on the CPU.
namespace open_branch {

OB_FN inline double p2(double y, double cosKsi, double sinKsi, double g1, double b1, double g2, double v2) {
    return R2 * R2 * v2 * v2 * (g2 + y * y * g1 / ((g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi)) + (b1 * b1 + g1 * g1) * y * sinKsi / ((g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi)));
}

OB_FN inline double dp2dv2(double y, double cosKsi, double sinKsi, double g1, double b1, double g2, double v2) {
    return 2 * R2 * R2 * v2 * (g2 + y * y * g1 / ((g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi)) + (b1 * b1 + g1 * g1) * y * sinKsi / ((g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi)));
}

OB_FN inline double q2(double y, double cosKsi, double sinKsi, double g1, double b1, double b2, double v2) {
    return -R2 * R2 * v2 * v2 * (b2 + y * y * b1 / ((g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi)) - (b1 * b1 + g1 * g1) * y * cosKsi / ((g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi)));
}

OB_FN inline double dq2dv2(double y, double cosKsi, double sinKsi, double g1, double b1, double b2, double v2) {
    return -2 * v2 * R2 * R2 * (b2 + y * y * b1 / ((g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi)) - (b1 * b1 + g1 * g1) * y * cosKsi / ((g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi)));
}

OB_FN inline double p1(double y, double cosKsi, double sinKsi, double g1, double g2, double b2, double v1, double r1) {
    return r1 * r1 * v1 * v1 * (g1 + y * y * g2 / ((g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi)) + (b2 * b2 + g2 * g2) * y * sinKsi / ((g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi)));
}

OB_FN inline double dp1dv1(double y, double cosKsi, double sinKsi, double g1, double g2, double b2, double v1, double r1) {
    return 2 * r1 * r1 * v1 * (g1 + y * y * g2 / ((g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi)) + (b2 * b2 + g2 * g2) * y * sinKsi / ((g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi)));
}

OB_FN inline double q1(double y, double cosKsi, double sinKsi, double b1, double g2, double b2, double v1, double r1) {
    return -r1 * r1 * v1 * v1 * (b1 + y * y * b2 / ((g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi)) - (b2 * b2 + g2 * g2) * y * cosKsi / ((g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi)));
}

OB_FN inline double dq1dv1(double y, double cosKsi, double sinKsi, double b1, double g2, double b2, double v1, double r1) {
    return -2 * v1 * r1 * r1 * (b1 + y * y * b2 / ((g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi)) - (b2 * b2 + g2 * g2) * y * cosKsi / ((g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi)));
}

// FUSED per-element kernel body: one routine computes the four open-branch flows
// (p2/q2 for a side-1-open branch, p1/q1 for a side-2-open one) + their four
// voltage derivatives, writing into Struct-of-Arrays at index i. sin/cos of ksi
// are computed once and shared. A real branch has only ONE side open: the host
// stamps that side's pair and resolves the other pair's scatter rows to -1
// (absent), exactly like the closed-branch fill.
OB_FN inline void evalOpenBranch(
        double y, double ksi, double g1, double b1, double g2, double b2,
        double v1, double r1, double v2,
        int i,
        double* p2_out,
        double* dp2dv2_out,
        double* q2_out,
        double* dq2dv2_out,
        double* p1_out,
        double* dp1dv1_out,
        double* q1_out,
        double* dq1dv1_out) {
    double sinKsi = sin(ksi);
    double cosKsi = cos(ksi);
    p2_out[i] = p2(y, cosKsi, sinKsi, g1, b1, g2, v2);
    dp2dv2_out[i] = dp2dv2(y, cosKsi, sinKsi, g1, b1, g2, v2);
    q2_out[i] = q2(y, cosKsi, sinKsi, g1, b1, b2, v2);
    dq2dv2_out[i] = dq2dv2(y, cosKsi, sinKsi, g1, b1, b2, v2);
    p1_out[i] = p1(y, cosKsi, sinKsi, g1, g2, b2, v1, r1);
    dp1dv1_out[i] = dp1dv1(y, cosKsi, sinKsi, g1, g2, b2, v1, r1);
    q1_out[i] = q1(y, cosKsi, sinKsi, b1, g2, b2, v1, r1);
    dq1dv1_out[i] = dq1dv1(y, cosKsi, sinKsi, b1, g2, b2, v1, r1);
}

// F-ONLY variant: the residual values alone (no Jacobian outputs) — for fill
// sites that never factorize: residual/convergence checks, line-search
// backtracks, frozen-Jacobian iterations.
OB_FN inline void evalOpenBranchF(
        double y, double ksi, double g1, double b1, double g2, double b2,
        double v1, double r1, double v2,
        int i,
        double* p2_out,
        double* q2_out,
        double* p1_out,
        double* q1_out) {
    double sinKsi = sin(ksi);
    double cosKsi = cos(ksi);
    p2_out[i] = p2(y, cosKsi, sinKsi, g1, b1, g2, v2);
    q2_out[i] = q2(y, cosKsi, sinKsi, g1, b1, b2, v2);
    p1_out[i] = p1(y, cosKsi, sinKsi, g1, g2, b2, v1, r1);
    q1_out[i] = q1(y, cosKsi, sinKsi, b1, g2, b2, v1, r1);
}

}  // namespace open_branch

using open_branch::evalOpenBranch;
using open_branch::evalOpenBranchF;

}}  // namespace powsybl::openloadflow
