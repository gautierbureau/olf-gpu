# Copyright (c) 2026, RTE (http://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0
#
# java2cuda spec for the open-branch active/reactive family — METADATA ONLY.
#
# A branch with ONE side open carries no through-flow: the connected side sees only
# the series + the open side's shunt admittance, so the flow is radial (no branch
# angle — no theta methods) and each output depends on the connected side's voltage
# only. The `shunt` denominator helper lives in the two abstract bases (with
# positionally-renamed parameters, g1/b1 vs g2/b2 — alpha-equivalent, so the parser
# keeps one and inlines by position). Residual order matches the committed
# residual-pipeline kernel: side-1-open pair (p2/q2) first, then side-2-open (p1/q1);
# a variable without a derivative method in the Java (e.g. dp2dv1) is a zero
# derivative and is dropped, exactly like the pipeline's zero-derivative dropping.

MODULE = {
    "family": "open_branch",

    # --- parse inputs: the hand-written OLF Java (source of truth) --------------
    "java_sources": [
        "src/main/java/com/powsybl/openloadflow/ac/equations/AbstractOpenSide1BranchAcFlowEquationTerm.java",
        "src/main/java/com/powsybl/openloadflow/ac/equations/AbstractOpenSide2BranchAcFlowEquationTerm.java",
        "src/main/java/com/powsybl/openloadflow/ac/equations/OpenBranchSide1ActiveFlowEquationTerm.java",
        "src/main/java/com/powsybl/openloadflow/ac/equations/OpenBranchSide1ReactiveFlowEquationTerm.java",
        "src/main/java/com/powsybl/openloadflow/ac/equations/OpenBranchSide2ActiveFlowEquationTerm.java",
        "src/main/java/com/powsybl/openloadflow/ac/equations/OpenBranchSide2ReactiveFlowEquationTerm.java",
    ],

    # No branch angle: side is None everywhere and there are no theta methods.
    "residuals": [("p2", None), ("q2", None), ("p1", None), ("q1", None)],
    "variables": ["v1", "v2"],
    "theta_methods": {},
    "precomputed": {"sinKsi": "sin(ksi)", "cosKsi": "cos(ksi)"},
    # The shunt denominator helper is hoisted into kernel locals, one per distinct
    # argument tuple (shunt1 = the g1/b1 instance, shunt2 = g2/b2) — the same
    # factoring OLF's Java does with its `double shunt = shunt(...)` local.
    "hoist_helpers": ["shunt"],

    "base_symbols": ["v1", "v2", "r1", "ksi", "y", "g1", "g2", "b1", "b2"],
    "constants": {"R2": 1},

    # --- CUDA emission identity ------------------------------------------------
    "cuda_path": "native/cuda/equations/generated/open_branch.cuh",
    "scatter_path": "codegen/scatter/open_branch.scatter.json",
    "scatter_header_path": "native/cuda/equations/generated/open_branch_scatter.h",
    "cuda_namespace": "powsybl::openloadflow",
    "cuda_routine": "evalOpenBranch",
    "cuda_fn_macro": "OB_FN",
    "scatter_prefix": "OB",
    "cuda_params": [["y", "ksi", "g1", "b1", "g2", "b2"],
                    ["v1", "r1", "v2"]],
    "cuda_consts": ["constexpr double R2 = 1.0;"],
    "doc": "Open-branch AC active/reactive flow residuals and Jacobian entries, "
           "parsed from the hand-written OLF equation-term Java (the source of truth).",
    "cuda_fused_doc": (
        "// FUSED per-element kernel body: one routine computes the four open-branch flows\n"
        "// (p2/q2 for a side-1-open branch, p1/q1 for a side-2-open one) + their four\n"
        "// voltage derivatives, writing into Struct-of-Arrays at index i. sin/cos of ksi\n"
        "// are computed once and shared. A real branch has only ONE side open: the host\n"
        "// stamps that side's pair and resolves the other pair's scatter rows to -1\n"
        "// (absent), exactly like the closed-branch fill."),

    # --- Scatter stamp (where each output lands in OLF's equation system) -------
    #
    # The open-branch terms are SCALAR equation terms (no vector-array evaluator):
    # each term class declares its variable as
    # v<N>Var = variableSet.getVariable(bus<N>.getNum(), AcVariableType.BUS_V).
    # check_scatter.py parses every named declaration across these files (cross-file
    # conflicts fail) and checks the rows against the creator's addTerm(open<XX>).
    "stamp": {
        "sources": {
            "columns": [
                "src/main/java/com/powsybl/openloadflow/ac/equations/"
                "OpenBranchSide1ActiveFlowEquationTerm.java",
                "src/main/java/com/powsybl/openloadflow/ac/equations/"
                "OpenBranchSide1ReactiveFlowEquationTerm.java",
                "src/main/java/com/powsybl/openloadflow/ac/equations/"
                "OpenBranchSide2ActiveFlowEquationTerm.java",
                "src/main/java/com/powsybl/openloadflow/ac/equations/"
                "OpenBranchSide2ReactiveFlowEquationTerm.java",
            ],
            "equations": "src/main/java/com/powsybl/openloadflow/ac/equations/"
                         "AcEquationSystemCreator.java",
        },
        "term_prefix": "open",                 # creator wires addTerm(openP1) etc.
        "residual_equation": {
            "p2": ("BUS_TARGET_P", "bus2"),
            "q2": ("BUS_TARGET_Q", "bus2"),
            "p1": ("BUS_TARGET_P", "bus1"),
            "q1": ("BUS_TARGET_Q", "bus1"),
        },
        "variable_column": {
            "v1": ("BUS_V", "bus1"),
            "v2": ("BUS_V", "bus2"),
        },
    },
}
