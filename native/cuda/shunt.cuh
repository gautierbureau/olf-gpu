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



// FUSED per-element kernel body: one routine computes the shunt injections p/q
// + their three non-zero derivatives for one shunt, writing into Struct-of-Arrays
// at index i. No trig at all — the cheapest fill of the families; dqdb only
// matters when the shunt is controllable (SHUNT_B is a skip column in plain PF).
SH_FN void evalShunt(
        double v, double g, double b,
        int i,
        double* p,
        double* dpdv,
        double* q,
        double* dqdv,
        double* dqdb) {
    p[i] = g * v * v;
    dpdv[i] = 2 * g * v;
    q[i] = -b * v * v;
    dqdv[i] = -2 * b * v;
    dqdb[i] = -v * v;
}

// F-ONLY variant: the residual values alone (no Jacobian outputs) — for fill
// sites that never factorize: residual/convergence checks, line-search
// backtracks, frozen-Jacobian iterations. Same parsed OLF formulas.
SH_FN void evalShuntF(
        double v, double g, double b,
        int i,
        double* p,
        double* q) {
    p[i] = g * v * v;
    q[i] = -b * v * v;
}

}}  // namespace powsybl::openloadflow
