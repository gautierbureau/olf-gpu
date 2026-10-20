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

// One device function per OLF Java method — SAME name, SAME signature, SAME body
// (the CPU methods are the optimized precomputed-trig form; the family namespace
// mirrors the Java class scoping). The eval* routines below only assemble calls,
// exactly like AcNetworkVector on the CPU.
namespace open_branch_i {

OBI_FN inline double i2(double y, double cosKsi, double sinKsi, double g1, double b1, double g2, double b2, double v2, double ph2) {
    return hypot((R2 * R2 * v2 * ((g2 + (y * y * g1 + (b1 * b1 + g1 * g1) * y * sinKsi) / ((g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi))) * cos(ph2) - (b2 + (y * y * b1 - (b1 * b1 + g1 * g1) * y * cosKsi) / ((g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi))) * sin(ph2))), (R2 * R2 * v2 * ((g2 + (y * y * g1 + (b1 * b1 + g1 * g1) * y * sinKsi) / ((g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi))) * sin(ph2) + (b2 + (y * y * b1 - (b1 * b1 + g1 * g1) * y * cosKsi) / ((g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi))) * cos(ph2))));
}

OBI_FN inline double di2dv2(double y, double cosKsi, double sinKsi, double g1, double b1, double g2, double b2, double v2, double ph2) {
    return ((R2 * R2 * v2 * ((g2 + (y * y * g1 + (b1 * b1 + g1 * g1) * y * sinKsi) / ((g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi))) * cos(ph2) - (b2 + (y * y * b1 - (b1 * b1 + g1 * g1) * y * cosKsi) / ((g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi))) * sin(ph2))) * (R2 * R2 * ((g2 + (y * y * g1 + (b1 * b1 + g1 * g1) * y * sinKsi) / ((g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi))) * cos(ph2) - (b2 + (y * y * b1 - (b1 * b1 + g1 * g1) * y * cosKsi) / ((g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi))) * sin(ph2))) + (R2 * R2 * v2 * ((g2 + (y * y * g1 + (b1 * b1 + g1 * g1) * y * sinKsi) / ((g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi))) * sin(ph2) + (b2 + (y * y * b1 - (b1 * b1 + g1 * g1) * y * cosKsi) / ((g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi))) * cos(ph2))) * (R2 * R2 * ((g2 + (y * y * g1 + (b1 * b1 + g1 * g1) * y * sinKsi) / ((g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi))) * sin(ph2) + (b2 + (y * y * b1 - (b1 * b1 + g1 * g1) * y * cosKsi) / ((g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi))) * cos(ph2)))) / (hypot((R2 * R2 * v2 * ((g2 + (y * y * g1 + (b1 * b1 + g1 * g1) * y * sinKsi) / ((g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi))) * cos(ph2) - (b2 + (y * y * b1 - (b1 * b1 + g1 * g1) * y * cosKsi) / ((g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi))) * sin(ph2))), (R2 * R2 * v2 * ((g2 + (y * y * g1 + (b1 * b1 + g1 * g1) * y * sinKsi) / ((g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi))) * sin(ph2) + (b2 + (y * y * b1 - (b1 * b1 + g1 * g1) * y * cosKsi) / ((g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi))) * cos(ph2)))));
}

OBI_FN inline double i1(double y, double cosKsi, double sinKsi, double g1, double b1, double g2, double b2, double v1, double ph1, double r1) {
    return hypot((r1 * r1 * v1 * ((g1 + (y * y * g2 + (b2 * b2 + g2 * g2) * y * sinKsi) / ((g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi))) * cos(ph1) - (b1 + (y * y * b2 - (b2 * b2 + g2 * g2) * y * cosKsi) / ((g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi))) * sin(ph1))), (r1 * r1 * v1 * ((g1 + (y * y * g2 + (b2 * b2 + g2 * g2) * y * sinKsi) / ((g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi))) * sin(ph1) + (b1 + (y * y * b2 - (b2 * b2 + g2 * g2) * y * cosKsi) / ((g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi))) * cos(ph1))));
}

OBI_FN inline double di1dv1(double y, double cosKsi, double sinKsi, double g1, double b1, double g2, double b2, double v1, double ph1, double r1) {
    return ((r1 * r1 * v1 * ((g1 + (y * y * g2 + (b2 * b2 + g2 * g2) * y * sinKsi) / ((g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi))) * cos(ph1) - (b1 + (y * y * b2 - (b2 * b2 + g2 * g2) * y * cosKsi) / ((g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi))) * sin(ph1))) * (r1 * r1 * ((g1 + (y * y * g2 + (b2 * b2 + g2 * g2) * y * sinKsi) / ((g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi))) * cos(ph1) - (b1 + (y * y * b2 - (b2 * b2 + g2 * g2) * y * cosKsi) / ((g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi))) * sin(ph1))) + (r1 * r1 * v1 * ((g1 + (y * y * g2 + (b2 * b2 + g2 * g2) * y * sinKsi) / ((g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi))) * sin(ph1) + (b1 + (y * y * b2 - (b2 * b2 + g2 * g2) * y * cosKsi) / ((g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi))) * cos(ph1))) * (r1 * r1 * ((g1 + (y * y * g2 + (b2 * b2 + g2 * g2) * y * sinKsi) / ((g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi))) * sin(ph1) + (b1 + (y * y * b2 - (b2 * b2 + g2 * g2) * y * cosKsi) / ((g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi))) * cos(ph1)))) / (hypot((r1 * r1 * v1 * ((g1 + (y * y * g2 + (b2 * b2 + g2 * g2) * y * sinKsi) / ((g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi))) * cos(ph1) - (b1 + (y * y * b2 - (b2 * b2 + g2 * g2) * y * cosKsi) / ((g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi))) * sin(ph1))), (r1 * r1 * v1 * ((g1 + (y * y * g2 + (b2 * b2 + g2 * g2) * y * sinKsi) / ((g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi))) * sin(ph1) + (b1 + (y * y * b2 - (b2 * b2 + g2 * g2) * y * cosKsi) / ((g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi))) * cos(ph1)))));
}

// FUSED per-element kernel body: one routine computes the two open-branch current
// magnitudes (i2 for a side-1-open branch, i1 for a side-2-open one) + their two
// voltage derivatives, writing into Struct-of-Arrays at index i. The magnitude is
// rotation-invariant in the bus angle, so only di/dv exists (FD-audited ~0 for the
// angles). The complex components reI/imI are shared kernel locals. NEW GPU target
// (the residual pipeline emitted no CUDA for current magnitude).
OBI_FN inline void evalOpenBranchI(
        double y, double ksi, double g1, double b1, double g2, double b2,
        double v1, double ph1, double r1, double v2, double ph2,
        int i,
        double* i2_out,
        double* di2dv2_out,
        double* i1_out,
        double* di1dv1_out) {
    double sinKsi = sin(ksi);
    double cosKsi = cos(ksi);
    i2_out[i] = i2(y, cosKsi, sinKsi, g1, b1, g2, b2, v2, ph2);
    di2dv2_out[i] = di2dv2(y, cosKsi, sinKsi, g1, b1, g2, b2, v2, ph2);
    i1_out[i] = i1(y, cosKsi, sinKsi, g1, b1, g2, b2, v1, ph1, r1);
    di1dv1_out[i] = di1dv1(y, cosKsi, sinKsi, g1, b1, g2, b2, v1, ph1, r1);
}

// F-ONLY variant: the residual values alone (no Jacobian outputs) — for fill
// sites that never factorize: residual/convergence checks, line-search
// backtracks, frozen-Jacobian iterations.
OBI_FN inline void evalOpenBranchIF(
        double y, double ksi, double g1, double b1, double g2, double b2,
        double v1, double ph1, double r1, double v2, double ph2,
        int i,
        double* i2_out,
        double* i1_out) {
    double sinKsi = sin(ksi);
    double cosKsi = cos(ksi);
    i2_out[i] = i2(y, cosKsi, sinKsi, g1, b1, g2, b2, v2, ph2);
    i1_out[i] = i1(y, cosKsi, sinKsi, g1, b1, g2, b2, v1, ph1, r1);
}

// PRECOMPUTED-TRIG variant — the caller supplies the trig (per-element
// statics + one sincos via angle addition in the fill kernel), matching
// how AcNetworkVector feeds the same methods on the CPU.
OBI_FN inline void evalOpenBranchIPre(
        double y, double g1, double b1, double g2, double b2,
        double v1, double ph1, double r1, double v2, double ph2,
        double sinKsi, double cosKsi,
        int i,
        double* i2_out,
        double* di2dv2_out,
        double* i1_out,
        double* di1dv1_out) {
    i2_out[i] = i2(y, cosKsi, sinKsi, g1, b1, g2, b2, v2, ph2);
    di2dv2_out[i] = di2dv2(y, cosKsi, sinKsi, g1, b1, g2, b2, v2, ph2);
    i1_out[i] = i1(y, cosKsi, sinKsi, g1, b1, g2, b2, v1, ph1, r1);
    di1dv1_out[i] = di1dv1(y, cosKsi, sinKsi, g1, b1, g2, b2, v1, ph1, r1);
}

// PRECOMPUTED-TRIG + F-ONLY variant: residual values alone, trig supplied
// by the caller.
OBI_FN inline void evalOpenBranchIPreF(
        double y, double g1, double b1, double g2, double b2,
        double v1, double ph1, double r1, double v2, double ph2,
        double sinKsi, double cosKsi,
        int i,
        double* i2_out,
        double* i1_out) {
    i2_out[i] = i2(y, cosKsi, sinKsi, g1, b1, g2, b2, v2, ph2);
    i1_out[i] = i1(y, cosKsi, sinKsi, g1, b1, g2, b2, v1, ph1, r1);
}

}  // namespace open_branch_i

using open_branch_i::evalOpenBranchI;
using open_branch_i::evalOpenBranchIF;
using open_branch_i::evalOpenBranchIPre;
using open_branch_i::evalOpenBranchIPreF;

}}  // namespace powsybl::openloadflow
