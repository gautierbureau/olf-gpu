// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Voltage-dependent (ZIP/exponential) load model per-exp-term value and derivative, parsed from the hand-written OLF equation-term Java (the source of truth); the sum over terms happens through the per-entry scatter, exactly OLF's accumulation loop.
//
// GENERATED — DO NOT EDIT. Regenerate with: python3 codegen/java2cuda/java2cuda.py (parsed from the OLF Java).
#pragma once
#include <math.h>

#ifndef VDL_FN
#define VDL_FN __host__ __device__   // also compiles host-side for the diff test
#endif

namespace powsybl { namespace openloadflow {


// One device function per OLF Java method — SAME name, SAME signature, SAME body
// (the CPU methods are the optimized precomputed-trig form; the family namespace
// mirrors the Java class scoping). The eval* routines below only assemble calls,
// exactly like AcNetworkVector on the CPU.
namespace vdl {

VDL_FN inline double f(double v, double target, double c, double n) {
    return (c * pow(v, n)) * target;
}

VDL_FN inline double dfdv(double v, double target, double c, double n) {
    return (c * n * pow(v, n - 1)) * target;
}

// FUSED per-EXP-TERM kernel body: one routine computes this term's contribution
// f = c*target*v^n to its bus P/Q residual and dfdv = c*n*target*v^(n-1) to the
// own-V Jacobian slot; the sum over a load model's terms is the per-entry
// scatter (atomicAdd), exactly the accumulation loop in
// AbstractLoadModelEquationTerm.f/dfdv.
VDL_FN inline void evalVdl(
        double v, double target, double c, double n,
        int i,
        double* f_out,
        double* dfdv_out) {
    f_out[i] = f(v, target, c, n);
    dfdv_out[i] = dfdv(v, target, c, n);
}

// F-ONLY variant: the residual values alone (no Jacobian outputs) — for fill
// sites that never factorize: residual/convergence checks, line-search
// backtracks, frozen-Jacobian iterations.
VDL_FN inline void evalVdlF(
        double v, double target, double c, double n,
        int i,
        double* f_out) {
    f_out[i] = f(v, target, c, n);
}

}  // namespace vdl

using vdl::evalVdl;
using vdl::evalVdlF;

}}  // namespace powsybl::openloadflow
