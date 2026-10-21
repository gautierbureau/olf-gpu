// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Closed-branch AC flow residuals and Jacobian entries, parsed from the hand-written OLF equation-term Java (the source of truth).
//
// GENERATED — DO NOT EDIT. Regenerate with: python3 codegen/java2cuda/java2cuda.py (parsed from the OLF Java).
#pragma once
#include <math.h>

#ifndef CLOSED_BRANCH_FN
#define CLOSED_BRANCH_FN __host__ __device__   // also compiles host-side for the diff test
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
namespace closed_branch {

CLOSED_BRANCH_FN inline double p1(double y, double sinKsi, double g1, double v1, double r1, double v2, double sinTheta) {
    return r1 * v1 * (g1 * r1 * v1 + y * r1 * v1 * sinKsi - y * R2 * v2 * sinTheta);
}

CLOSED_BRANCH_FN inline double dp1dv1(double y, double sinKsi, double g1, double v1, double r1, double v2, double sinTheta) {
    return r1 * (2 * g1 * r1 * v1 + 2 * y * r1 * v1 * sinKsi - y * R2 * v2 * sinTheta);
}

CLOSED_BRANCH_FN inline double dp1dv2(double y, double v1, double r1, double sinTheta) {
    return -y * r1 * R2 * v1 * sinTheta;
}

CLOSED_BRANCH_FN inline double dp1dph1(double y, double v1, double r1, double v2, double cosTheta) {
    return y * r1 * R2 * v1 * v2 * cosTheta;
}

CLOSED_BRANCH_FN inline double dp1dph2(double y, double v1, double r1, double v2, double cosTheta) {
    return -(y * r1 * R2 * v1 * v2 * cosTheta);
}

CLOSED_BRANCH_FN inline double dp1da1(double y, double v1, double r1, double v2, double cosTheta) {
    return (y * r1 * R2 * v1 * v2 * cosTheta);
}

CLOSED_BRANCH_FN inline double dp1dr1(double y, double sinKsi, double g1, double v1, double r1, double v2, double sinTheta) {
    return v1 * (2 * r1 * v1 * (g1 + y * sinKsi) - y * R2 * v2 * sinTheta);
}

CLOSED_BRANCH_FN inline double q1(double y, double cosKsi, double b1, double v1, double r1, double v2, double cosTheta) {
    return r1 * v1 * (-b1 * r1 * v1 + y * r1 * v1 * cosKsi - y * R2 * v2 * cosTheta);
}

CLOSED_BRANCH_FN inline double dq1dv1(double y, double cosKsi, double b1, double v1, double r1, double v2, double cosTheta) {
    return r1 * (-2 * b1 * r1 * v1 + 2 * y * r1 * v1 * cosKsi - y * R2 * v2 * cosTheta);
}

CLOSED_BRANCH_FN inline double dq1dv2(double y, double v1, double r1, double cosTheta) {
    return -y * r1 * R2 * v1 * cosTheta;
}

CLOSED_BRANCH_FN inline double dq1dph1(double y, double v1, double r1, double v2, double sinTheta) {
    return -y * r1 * R2 * v1 * v2 * sinTheta;
}

CLOSED_BRANCH_FN inline double dq1dph2(double y, double v1, double r1, double v2, double sinTheta) {
    return -(-y * r1 * R2 * v1 * v2 * sinTheta);
}

CLOSED_BRANCH_FN inline double dq1da1(double y, double v1, double r1, double v2, double sinTheta) {
    return (-y * r1 * R2 * v1 * v2 * sinTheta);
}

CLOSED_BRANCH_FN inline double dq1dr1(double y, double cosKsi, double b1, double v1, double r1, double v2, double cosTheta) {
    return v1 * (2 * r1 * v1 * (-b1 + y * cosKsi) - y * R2 * v2 * cosTheta);
}

CLOSED_BRANCH_FN inline double p2(double y, double sinKsi, double g2, double v1, double r1, double v2, double sinTheta) {
    return R2 * v2 * (g2 * R2 * v2 - y * r1 * v1 * sinTheta + y * R2 * v2 * sinKsi);
}

CLOSED_BRANCH_FN inline double dp2dv1(double y, double r1, double v2, double sinTheta) {
    return -y * r1 * R2 * v2 * sinTheta;
}

CLOSED_BRANCH_FN inline double dp2dv2(double y, double sinKsi, double g2, double v1, double r1, double v2, double sinTheta) {
    return R2 * (2 * g2 * R2 * v2 - y * r1 * v1 * sinTheta + 2 * y * R2 * v2 * sinKsi);
}

CLOSED_BRANCH_FN inline double dp2dph1(double y, double v1, double r1, double v2, double cosTheta) {
    return -y * r1 * R2 * v1 * v2 * cosTheta;
}

CLOSED_BRANCH_FN inline double dp2dph2(double y, double v1, double r1, double v2, double cosTheta) {
    return -(-y * r1 * R2 * v1 * v2 * cosTheta);
}

CLOSED_BRANCH_FN inline double dp2da1(double y, double v1, double r1, double v2, double cosTheta) {
    return (-y * r1 * R2 * v1 * v2 * cosTheta);
}

CLOSED_BRANCH_FN inline double dp2dr1(double y, double v1, double v2, double sinTheta) {
    return -y * R2 * v1 * v2 * sinTheta;
}

CLOSED_BRANCH_FN inline double q2(double y, double cosKsi, double b2, double v1, double r1, double v2, double cosTheta) {
    return R2 * v2 * (-b2 * R2 * v2 - y * r1 * v1 * cosTheta + y * R2 * v2 * cosKsi);
}

CLOSED_BRANCH_FN inline double dq2dv1(double y, double r1, double v2, double cosTheta) {
    return -y * r1 * R2 * v2 * cosTheta;
}

CLOSED_BRANCH_FN inline double dq2dv2(double y, double cosKsi, double b2, double v1, double r1, double v2, double cosTheta) {
    return R2 * (-2 * b2 * R2 * v2 - y * r1 * v1 * cosTheta + 2 * y * R2 * v2 * cosKsi);
}

CLOSED_BRANCH_FN inline double dq2dph1(double y, double v1, double r1, double v2, double sinTheta) {
    return y * r1 * R2 * v1 * v2 * sinTheta;
}

CLOSED_BRANCH_FN inline double dq2dph2(double y, double v1, double r1, double v2, double sinTheta) {
    return -(y * r1 * R2 * v1 * v2 * sinTheta);
}

CLOSED_BRANCH_FN inline double dq2da1(double y, double v1, double r1, double v2, double sinTheta) {
    return (y * r1 * R2 * v1 * v2 * sinTheta);
}

CLOSED_BRANCH_FN inline double dq2dr1(double y, double v1, double v2, double cosTheta) {
    return -y * R2 * v1 * v2 * cosTheta;
}

// FUSED per-element kernel body: one routine computes p1/q1/p2/q2 + all 24 Jacobian
// entries for one branch, writing into Struct-of-Arrays at index i. sin/cos of ksi,
// theta1, theta2 are computed once and shared as kernel locals — the per-element
// device routine the GPU assembly kernel calls (one thread = one branch, inline
// trig, SoA output — a fused per-branch kernel).
CLOSED_BRANCH_FN inline void updateClosedBranch(
        double y, double g1, double g2, double b1, double b2,
        double v1, double r1, double v2, double ksi, double a1, double ph1, double ph2,
        int i,
        double* p1_out,
        double* dp1dv1_out,
        double* dp1dv2_out,
        double* dp1dph1_out,
        double* dp1dph2_out,
        double* dp1da1_out,
        double* dp1dr1_out,
        double* q1_out,
        double* dq1dv1_out,
        double* dq1dv2_out,
        double* dq1dph1_out,
        double* dq1dph2_out,
        double* dq1da1_out,
        double* dq1dr1_out,
        double* p2_out,
        double* dp2dv1_out,
        double* dp2dv2_out,
        double* dp2dph1_out,
        double* dp2dph2_out,
        double* dp2da1_out,
        double* dp2dr1_out,
        double* q2_out,
        double* dq2dv1_out,
        double* dq2dv2_out,
        double* dq2dph1_out,
        double* dq2dph2_out,
        double* dq2da1_out,
        double* dq2dr1_out) {
    double theta1 = ksi - a1 + A2 - ph1 + ph2;
    double theta2 = ksi + a1 - A2 + ph1 - ph2;
    double sinKsi = sin(ksi);
    double sinTheta1 = sin(theta1);
    double cosTheta1 = cos(theta1);
    double cosKsi = cos(ksi);
    double sinTheta2 = sin(theta2);
    double cosTheta2 = cos(theta2);
    p1_out[i] = p1(y, sinKsi, g1, v1, r1, v2, sinTheta1);
    dp1dv1_out[i] = dp1dv1(y, sinKsi, g1, v1, r1, v2, sinTheta1);
    dp1dv2_out[i] = dp1dv2(y, v1, r1, sinTheta1);
    dp1dph1_out[i] = dp1dph1(y, v1, r1, v2, cosTheta1);
    dp1dph2_out[i] = dp1dph2(y, v1, r1, v2, cosTheta1);
    dp1da1_out[i] = dp1da1(y, v1, r1, v2, cosTheta1);
    dp1dr1_out[i] = dp1dr1(y, sinKsi, g1, v1, r1, v2, sinTheta1);
    q1_out[i] = q1(y, cosKsi, b1, v1, r1, v2, cosTheta1);
    dq1dv1_out[i] = dq1dv1(y, cosKsi, b1, v1, r1, v2, cosTheta1);
    dq1dv2_out[i] = dq1dv2(y, v1, r1, cosTheta1);
    dq1dph1_out[i] = dq1dph1(y, v1, r1, v2, sinTheta1);
    dq1dph2_out[i] = dq1dph2(y, v1, r1, v2, sinTheta1);
    dq1da1_out[i] = dq1da1(y, v1, r1, v2, sinTheta1);
    dq1dr1_out[i] = dq1dr1(y, cosKsi, b1, v1, r1, v2, cosTheta1);
    p2_out[i] = p2(y, sinKsi, g2, v1, r1, v2, sinTheta2);
    dp2dv1_out[i] = dp2dv1(y, r1, v2, sinTheta2);
    dp2dv2_out[i] = dp2dv2(y, sinKsi, g2, v1, r1, v2, sinTheta2);
    dp2dph1_out[i] = dp2dph1(y, v1, r1, v2, cosTheta2);
    dp2dph2_out[i] = dp2dph2(y, v1, r1, v2, cosTheta2);
    dp2da1_out[i] = dp2da1(y, v1, r1, v2, cosTheta2);
    dp2dr1_out[i] = dp2dr1(y, v1, v2, sinTheta2);
    q2_out[i] = q2(y, cosKsi, b2, v1, r1, v2, cosTheta2);
    dq2dv1_out[i] = dq2dv1(y, r1, v2, cosTheta2);
    dq2dv2_out[i] = dq2dv2(y, cosKsi, b2, v1, r1, v2, cosTheta2);
    dq2dph1_out[i] = dq2dph1(y, v1, r1, v2, sinTheta2);
    dq2dph2_out[i] = dq2dph2(y, v1, r1, v2, sinTheta2);
    dq2da1_out[i] = dq2da1(y, v1, r1, v2, sinTheta2);
    dq2dr1_out[i] = dq2dr1(y, v1, v2, cosTheta2);
}

// VALUES-ONLY variant (OLF's withDerivatives = false): the residual values alone,
// no Jacobian outputs — for fill sites that never factorize: residual/convergence
// checks, line-search backtracks, frozen-Jacobian iterations.
CLOSED_BRANCH_FN inline void updateClosedBranchWithoutDerivatives(
        double y, double g1, double g2, double b1, double b2,
        double v1, double r1, double v2, double ksi, double a1, double ph1, double ph2,
        int i,
        double* p1_out,
        double* q1_out,
        double* p2_out,
        double* q2_out) {
    double theta1 = ksi - a1 + A2 - ph1 + ph2;
    double theta2 = ksi + a1 - A2 + ph1 - ph2;
    double sinKsi = sin(ksi);
    double sinTheta1 = sin(theta1);
    double cosTheta1 = cos(theta1);
    double cosKsi = cos(ksi);
    double sinTheta2 = sin(theta2);
    double cosTheta2 = cos(theta2);
    p1_out[i] = p1(y, sinKsi, g1, v1, r1, v2, sinTheta1);
    q1_out[i] = q1(y, cosKsi, b1, v1, r1, v2, cosTheta1);
    p2_out[i] = p2(y, sinKsi, g2, v1, r1, v2, sinTheta2);
    q2_out[i] = q2(y, cosKsi, b2, v1, r1, v2, cosTheta2);
}

}  // namespace closed_branch

using closed_branch::updateClosedBranch;
using closed_branch::updateClosedBranchWithoutDerivatives;

}}  // namespace powsybl::openloadflow
