// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Closed-branch AC flow residuals and Jacobian entries, parsed from the hand-written OLF equation-term Java (the source of truth).
//
// GENERATED — DO NOT EDIT. Regenerate with: python3 codegen/java2cuda/java2cuda.py (parsed from the OLF Java).
//
// FUSED per-element kernel body: one routine computes p1/q1/p2/q2 + all 24 Jacobian
// entries for one branch, writing into Struct-of-Arrays at index i. sin/cos of ksi,
// theta1, theta2 are computed once and shared as kernel locals — the per-element
// device routine the GPU assembly kernel calls (one thread = one branch, inline
// trig, SoA output — a fused per-branch kernel).
#pragma once
#include <math.h>

#ifndef CB_FN
#define CB_FN __host__ __device__   // also compiles host-side for the diff test
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

CB_FN void evalClosedBranch(
        double y, double g1, double g2, double b1, double b2,
        double v1, double r1, double v2, double ksi, double a1, double ph1, double ph2,
        int i,
        double* p1,
        double* dp1dv1,
        double* dp1dv2,
        double* dp1dph1,
        double* dp1dph2,
        double* dp1da1,
        double* dp1dr1,
        double* q1,
        double* dq1dv1,
        double* dq1dv2,
        double* dq1dph1,
        double* dq1dph2,
        double* dq1da1,
        double* dq1dr1,
        double* p2,
        double* dp2dv1,
        double* dp2dv2,
        double* dp2dph1,
        double* dp2dph2,
        double* dp2da1,
        double* dp2dr1,
        double* q2,
        double* dq2dv1,
        double* dq2dv2,
        double* dq2dph1,
        double* dq2dph2,
        double* dq2da1,
        double* dq2dr1) {
    double theta1 = ksi - a1 + A2 - ph1 + ph2;
    double theta2 = ksi + a1 - A2 + ph1 - ph2;
    double sinKsi = sin(ksi);
    double sinTheta1 = sin(theta1);
    double cosTheta1 = cos(theta1);
    double cosKsi = cos(ksi);
    double sinTheta2 = sin(theta2);
    double cosTheta2 = cos(theta2);
    p1[i] = r1 * v1 * (g1 * r1 * v1 + y * r1 * v1 * sinKsi - y * R2 * v2 * sinTheta1);
    dp1dv1[i] = r1 * (2 * g1 * r1 * v1 + 2 * y * r1 * v1 * sinKsi - y * R2 * v2 * sinTheta1);
    dp1dv2[i] = -y * r1 * R2 * v1 * sinTheta1;
    dp1dph1[i] = y * r1 * R2 * v1 * v2 * cosTheta1;
    dp1dph2[i] = -(y * r1 * R2 * v1 * v2 * cosTheta1);
    dp1da1[i] = (y * r1 * R2 * v1 * v2 * cosTheta1);
    dp1dr1[i] = v1 * (2 * r1 * v1 * (g1 + y * sinKsi) - y * R2 * v2 * sinTheta1);
    q1[i] = r1 * v1 * (-b1 * r1 * v1 + y * r1 * v1 * cosKsi - y * R2 * v2 * cosTheta1);
    dq1dv1[i] = r1 * (-2 * b1 * r1 * v1 + 2 * y * r1 * v1 * cosKsi - y * R2 * v2 * cosTheta1);
    dq1dv2[i] = -y * r1 * R2 * v1 * cosTheta1;
    dq1dph1[i] = -y * r1 * R2 * v1 * v2 * sinTheta1;
    dq1dph2[i] = -(-y * r1 * R2 * v1 * v2 * sinTheta1);
    dq1da1[i] = (-y * r1 * R2 * v1 * v2 * sinTheta1);
    dq1dr1[i] = v1 * (2 * r1 * v1 * (-b1 + y * cosKsi) - y * R2 * v2 * cosTheta1);
    p2[i] = R2 * v2 * (g2 * R2 * v2 - y * r1 * v1 * sinTheta2 + y * R2 * v2 * sinKsi);
    dp2dv1[i] = -y * r1 * R2 * v2 * sinTheta2;
    dp2dv2[i] = R2 * (2 * g2 * R2 * v2 - y * r1 * v1 * sinTheta2 + 2 * y * R2 * v2 * sinKsi);
    dp2dph1[i] = -y * r1 * R2 * v1 * v2 * cosTheta2;
    dp2dph2[i] = -(-y * r1 * R2 * v1 * v2 * cosTheta2);
    dp2da1[i] = (-y * r1 * R2 * v1 * v2 * cosTheta2);
    dp2dr1[i] = -y * R2 * v1 * v2 * sinTheta2;
    q2[i] = R2 * v2 * (-b2 * R2 * v2 - y * r1 * v1 * cosTheta2 + y * R2 * v2 * cosKsi);
    dq2dv1[i] = -y * r1 * R2 * v2 * cosTheta2;
    dq2dv2[i] = R2 * (-2 * b2 * R2 * v2 - y * r1 * v1 * cosTheta2 + 2 * y * R2 * v2 * cosKsi);
    dq2dph1[i] = y * r1 * R2 * v1 * v2 * sinTheta2;
    dq2dph2[i] = -(y * r1 * R2 * v1 * v2 * sinTheta2);
    dq2da1[i] = (y * r1 * R2 * v1 * v2 * sinTheta2);
    dq2dr1[i] = -y * R2 * v1 * v2 * cosTheta2;
}

}}  // namespace powsybl::openloadflow
