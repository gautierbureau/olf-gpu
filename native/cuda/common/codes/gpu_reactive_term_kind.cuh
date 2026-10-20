// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The kind of a reactive injection term in a DISTR_Q contribution (distrQKind) or a remote member's Q balance
// (remoteMemberReactiveTermKind): the terms OLF's createReactiveTerms sums. Mirrors
// com.powsybl.openloadflow.ac.solver.gpu.codes.GpuReactiveTermKind (GpuCodesMirrorTest checks they agree).
#ifndef OLF_GPU_COMMON_CODES_GPU_REACTIVE_TERM_KIND_CUH
#define OLF_GPU_COMMON_CODES_GPU_REACTIVE_TERM_KIND_CUH

namespace GpuReactiveTermKind {
constexpr int CLOSED_BRANCH_REACTIVE_FLOW = 0;      // ClosedBranchSide1/2ReactiveFlowEquationTerm
constexpr int SHUNT_COMPENSATOR_REACTIVE_FLOW = 1;  // ShuntCompensatorReactiveFlowEquationTerm
constexpr int OPEN_BRANCH_REACTIVE_FLOW = 2;        // OpenBranchSide1/2ReactiveFlowEquationTerm
constexpr int DUMMY_Q = 3;                          // a zero-impedance branch's DUMMY_Q variable (linear)
}

#endif
