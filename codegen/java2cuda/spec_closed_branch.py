# Copyright (c) 2026, RTE (http://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0
#
# java2cuda spec for the closed-branch family — METADATA ONLY, no formulas.
#
# The UNTOUCHED OLF Java (src/main/java, byte-identical to upstream main) is the
# single source of truth for BOTH the values and the hand-written Jacobian. This
# file only tells the parser where to look and how the family is shaped:
#   - which Java files hold the static formula methods,
#   - which residuals exist and which side's branch angle their sinTheta/cosTheta
#     refer to,
#   - which precomputed-trig parameters map back to which raw-angle expressions
#     (so the fused kernel can inline the trig and share it as kernel locals),
#   - the CUDA emission identity + the scatter stamp (where each output lands in
#     OLF's fixed-pattern Jacobian; validated by codegen/check_scatter.py).

MODULE = {
    "family": "closed_branch",

    # --- parse inputs: the hand-written OLF Java (source of truth) --------------
    "java_sources": [
        "src/main/java/com/powsybl/openloadflow/ac/equations/AbstractClosedBranchAcFlowEquationTerm.java",
        "src/main/java/com/powsybl/openloadflow/ac/equations/ClosedBranchSide1ActiveFlowEquationTerm.java",
        "src/main/java/com/powsybl/openloadflow/ac/equations/ClosedBranchSide1ReactiveFlowEquationTerm.java",
        "src/main/java/com/powsybl/openloadflow/ac/equations/ClosedBranchSide2ActiveFlowEquationTerm.java",
        "src/main/java/com/powsybl/openloadflow/ac/equations/ClosedBranchSide2ReactiveFlowEquationTerm.java",
    ],

    # Residuals and the side whose branch angle their sinTheta/cosTheta mean.
    # The fused outputs are, per residual: the value method, then d<res>d<var> for
    # each var below IN ORDER — a variable with no method in the Java is a zero
    # derivative and is dropped (OLF declares der() only for depended-on variables).
    "residuals": [("p1", 1), ("q1", 1), ("p2", 2), ("q2", 2)],
    "variables": ["v1", "v2", "ph1", "ph2", "a1", "r1"],

    # The branch-angle helper methods, parsed from the Java too (theta1/theta2 in
    # AbstractClosedBranchAcFlowEquationTerm) — NOT re-authored here.
    "theta_methods": {1: "theta1", 2: "theta2"},

    # Precomputed-trig parameters -> raw-angle expressions. `theta` resolves to the
    # residual side's parsed angle. This is how the scalar precomputed-trig API is
    # folded back into a fused kernel with inline trig (CSE then shares the calls).
    "precomputed": {"sinKsi": "sin(ksi)", "cosKsi": "cos(ksi)",
                    "sinTheta": "sin(theta)", "cosTheta": "cos(theta)"},

    # Raw kernel inputs + model constants (kept symbolic; constexpr in the .cuh).
    "base_symbols": ["v1", "v2", "ph1", "ph2", "a1", "r1", "ksi", "y",
                     "g1", "g2", "b1", "b2"],
    "constants": {"R2": 1, "A2": 0},

    # --- CUDA emission identity ------------------------------------------------
    "cuda_path": "native/cuda/closed_branch.cuh",
    "scatter_path": "native/cuda/closed_branch.scatter.json",
    "scatter_header_path": "native/cuda/closed_branch_scatter.h",
    "cuda_namespace": "powsybl::openloadflow",
    "cuda_routine": "evalClosedBranch",
    "cuda_fn_macro": "CB_FN",
    "scatter_prefix": "CB",
    "cuda_params": [["y", "g1", "g2", "b1", "b2"],
                    ["v1", "r1", "v2", "ksi", "a1", "ph1", "ph2"]],
    "cuda_consts": ["constexpr double R2 = 1.0;", "constexpr double A2 = 0.0;"],
    "doc": "Closed-branch AC flow residuals and Jacobian entries, parsed from the "
           "hand-written OLF equation-term Java (the source of truth).",
    "cuda_fused_doc": (
        "// FUSED per-element kernel body: one routine computes p1/q1/p2/q2 + all 24 Jacobian\n"
        "// entries for one branch, writing into Struct-of-Arrays at index i. sin/cos of ksi,\n"
        "// theta1, theta2 are computed once and shared as kernel locals — the per-element\n"
        "// device routine the GPU assembly kernel calls (one thread = one branch, inline\n"
        "// trig, SoA output — a fused per-branch kernel)."),

    # --- Scatter stamp (where each output lands in OLF's equation system) -------
    #
    # Each residual feeds one bus power-balance equation; each derivative d<res>d<var>
    # scatters into J[row = res's equation, col = var's variable]. Endpoints (bus1 /
    # bus2 / branch) are symbolic — the host resolves them per branch to flat
    # row/col indices. Names are the real OLF AcEquationType / AcVariableType enums;
    # codegen/check_scatter.py asserts this stamp against the actual term-class
    # wiring (the vector evaluator's getDerivatives() and AcEquationSystemCreator).
    "stamp": {
        "sources": {
            "columns": "src/main/java/com/powsybl/openloadflow/ac/equations/vector/"
                       "AbstractClosedBranchEquationTermArrayEvaluator.java",
            "equations": "src/main/java/com/powsybl/openloadflow/ac/equations/"
                         "AcEquationSystemCreator.java",
        },
        # residual -> (AcEquationType, endpoint) it contributes to (the F row)
        "residual_equation": {
            "p1": ("BUS_TARGET_P", "bus1"),
            "q1": ("BUS_TARGET_Q", "bus1"),
            "p2": ("BUS_TARGET_P", "bus2"),
            "q2": ("BUS_TARGET_Q", "bus2"),
        },
        # differentiation variable -> (AcVariableType, endpoint) it is (the J column)
        "variable_column": {
            "v1": ("BUS_V", "bus1"),
            "v2": ("BUS_V", "bus2"),
            "ph1": ("BUS_PHI", "bus1"),
            "ph2": ("BUS_PHI", "bus2"),
            "a1": ("BRANCH_ALPHA1", "branch"),
            "r1": ("BRANCH_RHO1", "branch"),
        },
    },
}
