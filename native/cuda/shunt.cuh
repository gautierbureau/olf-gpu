// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Shunt-compensator AC flow residuals and Jacobian entries, parsed from the hand-written OLF equation-term Java (the source of truth).
//
// GENERATED — DO NOT EDIT. Regenerate with: python3 codegen/java2cuda/java2cuda.py (parsed from the OLF Java).
#pragma once
#include <math.h>

#ifndef SH_FN
#define SH_FN __host__ __device__   // also compiles host-side for the diff test
#endif

namespace powsybl { namespace openloadflow {


// One device function per OLF Java method — SAME name, SAME signature, SAME body
// (the CPU methods are the optimized precomputed-trig form; the family namespace
// mirrors the Java class scoping). The eval* routines below only assemble calls,
// exactly like AcNetworkVector on the CPU.
namespace shunt {

SH_FN inline double p(double v, double g) {
    return g * v * v;
}

SH_FN inline double dpdv(double v, double g) {
    return 2 * g * v;
}

SH_FN inline double q(double v, double b) {
    return -b * v * v;
}

SH_FN inline double dqdv(double v, double b) {
    return -2 * b * v;
}

SH_FN inline double dqdb(double v) {
    return -v * v;
}

// FUSED per-element kernel body: one routine computes the shunt injections p/q
// + their three non-zero derivatives for one shunt, writing into Struct-of-Arrays
// at index i. No trig at all — the cheapest fill of the families; dqdb only
// matters when the shunt is controllable (SHUNT_B is a skip column in plain PF).
SH_FN inline void evalShunt(
        double v, double g, double b,
        int i,
        double* p_out,
        double* dpdv_out,
        double* q_out,
        double* dqdv_out,
        double* dqdb_out) {
    p_out[i] = p(v, g);
    dpdv_out[i] = dpdv(v, g);
    q_out[i] = q(v, b);
    dqdv_out[i] = dqdv(v, b);
    dqdb_out[i] = dqdb(v);
}

// F-ONLY variant: the residual values alone (no Jacobian outputs) — for fill
// sites that never factorize: residual/convergence checks, line-search
// backtracks, frozen-Jacobian iterations.
SH_FN inline void evalShuntF(
        double v, double g, double b,
        int i,
        double* p_out,
        double* q_out) {
    p_out[i] = p(v, g);
    q_out[i] = q(v, b);
}

}  // namespace shunt

using shunt::evalShunt;
using shunt::evalShuntF;

}}  // namespace powsybl::openloadflow
