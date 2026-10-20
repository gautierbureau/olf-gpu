# Copyright (c) 2026, RTE (http://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0
#
# java2cuda spec for the closed-branch CURRENT-MAGNITUDE family — METADATA ONLY.
#
# NEW GPU target: the residual pipeline emitted no CUDA for this family (nothing on
# the GPU consumes current magnitude today) — this kernel is groundwork for future
# on-device current-limit monitoring, and prove.py's drop-in check reports it as new.
#
# OLF computes i = hypot(reI, imI) with the chain rule through private component
# helpers; the old pipeline needed a special "structured magnitudes emission" to
# reproduce that shape in generated Java. Here the methods are just parsed:
# - the value i1 is OVERLOADED (public (..., a1, ..., ph2) over a private
#   (..., theta) core): the parser picks the public one and inlines the private
#   overload through it;
# - each side has its OWN private theta(...) (same name/arity, different body):
#   per-file helper scope resolves them, like HVDC's isController;
# - reI/imI are hoisted into kernel locals (one instance each — every output calls
#   them with identical arguments), so each derivative is
#   (reI*dreId<var> + imI*dimId<var>) / hypot(reI, imI) over shared locals —
#   OLF's own chain-rule factoring, without any structured-emission machinery;
# - di1da1 = -di1dph2 (a1 enters theta opposite to ph2): a sibling call, inlined;
# - r1 is NOT a differentiation variable: OLF deliberately does not implement
#   di/dr1 (calculateSensi throws for dr1 != 0), so no dr1 output exists — unlike
#   the P/Q closed-branch family.
# Raw-trig family: no precomputed sin/cos parameters — the angles stay inline
# (sin(ph1 + ksi), the theta expressions), shared through the hoisted components.

MODULE = {
    "family": "closed_branch_i",

    # --- parse inputs: the hand-written OLF Java (source of truth) --------------
    "java_sources": [
        "src/main/java/com/powsybl/openloadflow/ac/equations/ClosedBranchSide1CurrentMagnitudeEquationTerm.java",
        "src/main/java/com/powsybl/openloadflow/ac/equations/ClosedBranchSide2CurrentMagnitudeEquationTerm.java",
    ],

    "residuals": [("i1", None), ("i2", None)],
    "variables": ["v1", "v2", "ph1", "ph2", "a1"],
    "theta_methods": {},
    "precomputed": {},
    "hoist_helpers": ["reI1", "imI1", "reI2", "imI2"],

    "base_symbols": ["v1", "v2", "ph1", "ph2", "a1", "r1", "ksi", "y",
                     "g1", "g2", "b1", "b2"],
    "constants": {"R2": 1, "A2": 0},

    # --- CUDA emission identity (NEW — no residual-pipeline counterpart) --------
    "cuda_path": "native/cuda/equations/generated/closed_branch_i.cuh",
    "cuda_namespace": "powsybl::openloadflow",
    "cuda_routine": "evalClosedBranchI",
    "cuda_fn_macro": "CBI_FN",
    "cuda_params": [["y", "g1", "g2", "b1", "b2"],
                    ["v1", "r1", "v2", "ksi", "a1", "ph1", "ph2"]],
    "cuda_consts": ["constexpr double R2 = 1.0;", "constexpr double A2 = 0.0;"],
    "doc": "Closed-branch current magnitudes and their Jacobian entries, parsed from "
           "the hand-written OLF equation-term Java (the source of truth).",
    "cuda_fused_doc": (
        "// FUSED per-element kernel body: one routine computes the two closed-branch\n"
        "// current magnitudes i1/i2 + their ten chain-rule derivatives for one branch,\n"
        "// writing into Struct-of-Arrays at index i. The complex-current components\n"
        "// reI1/imI1/reI2/imI2 are computed once as kernel locals and every derivative is\n"
        "// (re*dred<var> + im*dimd<var>) / hypot(re, im) — OLF's own chain-rule shape.\n"
        "// NEW GPU target (the residual pipeline emitted no CUDA for current magnitude);\n"
        "// no derivative w.r.t. r1 exists, matching OLF (calculateSensi throws on dr1)."),
}
