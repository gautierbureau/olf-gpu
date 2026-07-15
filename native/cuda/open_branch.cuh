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

// FUSED per-element kernel body: one routine computes the four open-branch flows
// (p2/q2 for a side-1-open branch, p1/q1 for a side-2-open one) + their four
// voltage derivatives, writing into Struct-of-Arrays at index i. sin/cos of ksi
// are computed once and shared. A real branch has only ONE side open: the host
// stamps that side's pair and resolves the other pair's scatter rows to -1
// (absent), exactly like the closed-branch fill.
OB_FN void evalOpenBranch(
        double y, double ksi, double g1, double b1, double g2, double b2,
        double v1, double r1, double v2,
        int i,
        double* p2,
        double* dp2dv2,
        double* q2,
        double* dq2dv2,
        double* p1,
        double* dp1dv1,
        double* q1,
        double* dq1dv1) {
    double sinKsi = sin(ksi);
    double cosKsi = cos(ksi);
    double shunt = (g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi);
    double shunt2 = (g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi);
    p2[i] = R2 * R2 * v2 * v2 * (g2 + y * y * g1 / shunt + (b1 * b1 + g1 * g1) * y * sinKsi / shunt);
    dp2dv2[i] = 2 * R2 * R2 * v2 * (g2 + y * y * g1 / shunt + (b1 * b1 + g1 * g1) * y * sinKsi / shunt);
    q2[i] = -R2 * R2 * v2 * v2 * (b2 + y * y * b1 / shunt - (b1 * b1 + g1 * g1) * y * cosKsi / shunt);
    dq2dv2[i] = -2 * v2 * R2 * R2 * (b2 + y * y * b1 / shunt - (b1 * b1 + g1 * g1) * y * cosKsi / shunt);
    p1[i] = r1 * r1 * v1 * v1 * (g1 + y * y * g2 / shunt2 + (b2 * b2 + g2 * g2) * y * sinKsi / shunt2);
    dp1dv1[i] = 2 * r1 * r1 * v1 * (g1 + y * y * g2 / shunt2 + (b2 * b2 + g2 * g2) * y * sinKsi / shunt2);
    q1[i] = -r1 * r1 * v1 * v1 * (b1 + y * y * b2 / shunt2 - (b2 * b2 + g2 * g2) * y * cosKsi / shunt2);
    dq1dv1[i] = -2 * v1 * r1 * r1 * (b1 + y * y * b2 / shunt2 - (b2 * b2 + g2 * g2) * y * cosKsi / shunt2);
}

// F-ONLY variant: the residual values alone (no Jacobian outputs) — for fill
// sites that never factorize: residual/convergence checks, line-search
// backtracks, frozen-Jacobian iterations. Same parsed OLF formulas.
OB_FN void evalOpenBranchF(
        double y, double ksi, double g1, double b1, double g2, double b2,
        double v1, double r1, double v2,
        int i,
        double* p2,
        double* q2,
        double* p1,
        double* q1) {
    double sinKsi = sin(ksi);
    double cosKsi = cos(ksi);
    double shunt = (g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi);
    double shunt2 = (g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi);
    p2[i] = R2 * R2 * v2 * v2 * (g2 + y * y * g1 / shunt + (b1 * b1 + g1 * g1) * y * sinKsi / shunt);
    q2[i] = -R2 * R2 * v2 * v2 * (b2 + y * y * b1 / shunt - (b1 * b1 + g1 * g1) * y * cosKsi / shunt);
    p1[i] = r1 * r1 * v1 * v1 * (g1 + y * y * g2 / shunt2 + (b2 * b2 + g2 * g2) * y * sinKsi / shunt2);
    q1[i] = -r1 * r1 * v1 * v1 * (b1 + y * y * b2 / shunt2 - (b2 * b2 + g2 * g2) * y * cosKsi / shunt2);
}

}}  // namespace powsybl::openloadflow
