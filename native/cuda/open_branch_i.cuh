// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Open-branch current magnitudes and their voltage derivatives, parsed from the hand-written OLF equation-term Java (the source of truth).
//
// GENERATED — DO NOT EDIT. Regenerate with: python3 codegen/java2cuda/java2cuda.py (parsed from the OLF Java).
#pragma once
#include <math.h>

#ifndef OBI_FN
#define OBI_FN __host__ __device__   // also compiles host-side for the diff test
#endif

namespace powsybl { namespace openloadflow {

#ifndef OLF_CONST_R2
#define OLF_CONST_R2
constexpr double R2 = 1.0;
#endif

// FUSED per-element kernel body: one routine computes the two open-branch current
// magnitudes (i2 for a side-1-open branch, i1 for a side-2-open one) + their two
// voltage derivatives, writing into Struct-of-Arrays at index i. The magnitude is
// rotation-invariant in the bus angle, so only di/dv exists (FD-audited ~0 for the
// angles). The complex components reI/imI are shared kernel locals. NEW GPU target
// (the residual pipeline emitted no CUDA for current magnitude).
OBI_FN void evalOpenBranchI(
        double y, double ksi, double g1, double b1, double g2, double b2,
        double v1, double ph1, double r1, double v2, double ph2,
        int i,
        double* i2,
        double* di2dv2,
        double* i1,
        double* di1dv1) {
    double sinKsi = sin(ksi);
    double cosKsi = cos(ksi);
    double shunt = (g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi);
    double shunt2 = (g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi);
    double reI1 = r1 * r1 * v1 * ((g1 + (y * y * g2 + (b2 * b2 + g2 * g2) * y * sinKsi) / shunt2) * cos(ph1) - (b1 + (y * y * b2 - (b2 * b2 + g2 * g2) * y * cosKsi) / shunt2) * sin(ph1));
    double imI1 = r1 * r1 * v1 * ((g1 + (y * y * g2 + (b2 * b2 + g2 * g2) * y * sinKsi) / shunt2) * sin(ph1) + (b1 + (y * y * b2 - (b2 * b2 + g2 * g2) * y * cosKsi) / shunt2) * cos(ph1));
    double reI2 = R2 * R2 * v2 * ((g2 + (y * y * g1 + (b1 * b1 + g1 * g1) * y * sinKsi) / shunt) * cos(ph2) - (b2 + (y * y * b1 - (b1 * b1 + g1 * g1) * y * cosKsi) / shunt) * sin(ph2));
    double imI2 = R2 * R2 * v2 * ((g2 + (y * y * g1 + (b1 * b1 + g1 * g1) * y * sinKsi) / shunt) * sin(ph2) + (b2 + (y * y * b1 - (b1 * b1 + g1 * g1) * y * cosKsi) / shunt) * cos(ph2));
    i2[i] = hypot(reI2, imI2);
    di2dv2[i] = (reI2 * (R2 * R2 * ((g2 + (y * y * g1 + (b1 * b1 + g1 * g1) * y * sinKsi) / shunt) * cos(ph2) - (b2 + (y * y * b1 - (b1 * b1 + g1 * g1) * y * cosKsi) / shunt) * sin(ph2))) + imI2 * (R2 * R2 * ((g2 + (y * y * g1 + (b1 * b1 + g1 * g1) * y * sinKsi) / shunt) * sin(ph2) + (b2 + (y * y * b1 - (b1 * b1 + g1 * g1) * y * cosKsi) / shunt) * cos(ph2)))) / (hypot(reI2, imI2));
    i1[i] = hypot(reI1, imI1);
    di1dv1[i] = (reI1 * (r1 * r1 * ((g1 + (y * y * g2 + (b2 * b2 + g2 * g2) * y * sinKsi) / shunt2) * cos(ph1) - (b1 + (y * y * b2 - (b2 * b2 + g2 * g2) * y * cosKsi) / shunt2) * sin(ph1))) + imI1 * (r1 * r1 * ((g1 + (y * y * g2 + (b2 * b2 + g2 * g2) * y * sinKsi) / shunt2) * sin(ph1) + (b1 + (y * y * b2 - (b2 * b2 + g2 * g2) * y * cosKsi) / shunt2) * cos(ph1)))) / (hypot(reI1, imI1));
}

// F-ONLY variant: the residual values alone (no Jacobian outputs) — for fill
// sites that never factorize: residual/convergence checks, line-search
// backtracks, frozen-Jacobian iterations. Same parsed OLF formulas.
OBI_FN void evalOpenBranchIF(
        double y, double ksi, double g1, double b1, double g2, double b2,
        double v1, double ph1, double r1, double v2, double ph2,
        int i,
        double* i2,
        double* i1) {
    double sinKsi = sin(ksi);
    double cosKsi = cos(ksi);
    double shunt = (g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi);
    double shunt2 = (g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi);
    double reI1 = r1 * r1 * v1 * ((g1 + (y * y * g2 + (b2 * b2 + g2 * g2) * y * sinKsi) / shunt2) * cos(ph1) - (b1 + (y * y * b2 - (b2 * b2 + g2 * g2) * y * cosKsi) / shunt2) * sin(ph1));
    double imI1 = r1 * r1 * v1 * ((g1 + (y * y * g2 + (b2 * b2 + g2 * g2) * y * sinKsi) / shunt2) * sin(ph1) + (b1 + (y * y * b2 - (b2 * b2 + g2 * g2) * y * cosKsi) / shunt2) * cos(ph1));
    double reI2 = R2 * R2 * v2 * ((g2 + (y * y * g1 + (b1 * b1 + g1 * g1) * y * sinKsi) / shunt) * cos(ph2) - (b2 + (y * y * b1 - (b1 * b1 + g1 * g1) * y * cosKsi) / shunt) * sin(ph2));
    double imI2 = R2 * R2 * v2 * ((g2 + (y * y * g1 + (b1 * b1 + g1 * g1) * y * sinKsi) / shunt) * sin(ph2) + (b2 + (y * y * b1 - (b1 * b1 + g1 * g1) * y * cosKsi) / shunt) * cos(ph2));
    i2[i] = hypot(reI2, imI2);
    i1[i] = hypot(reI1, imI1);
}

}}  // namespace powsybl::openloadflow
