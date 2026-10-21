# Copyright (c) 2026, RTE (http://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0
#
# java2cuda spec for the shunt-compensator family — METADATA ONLY.
#
# The simplest family: p = g*v^2 and q = -b*v^2, no trig, no helpers, no branch
# angle. dp/db has no method in the Java (p does not depend on b) and is dropped;
# dqdb only matters when the shunt is controllable (SHUNT_B is a skip column in
# plain AC PF, resolved by the host exactly like BRANCH_ALPHA1/RHO1).

MODULE = {
    "family": "shunt",

    # --- parse inputs: the hand-written OLF Java (source of truth) --------------
    "java_sources": [
        "src/main/java/com/powsybl/openloadflow/ac/equations/AbstractShuntCompensatorEquationTerm.java",
        "src/main/java/com/powsybl/openloadflow/ac/equations/ShuntCompensatorActiveFlowEquationTerm.java",
        "src/main/java/com/powsybl/openloadflow/ac/equations/ShuntCompensatorReactiveFlowEquationTerm.java",
    ],

    "residuals": [("p", None), ("q", None)],
    "variables": ["v", "b"],
    "theta_methods": {},
    "precomputed": {},

    "base_symbols": ["v", "g", "b"],
    "constants": {},

    # --- CUDA emission identity ------------------------------------------------
    "cuda_path": "native/cuda/equations/generated/shunt.cuh",
    "scatter_path": "codegen/scatter/shunt.scatter.json",
    "scatter_header_path": "native/cuda/equations/generated/shunt_scatter.h",
    "cuda_namespace": "powsybl::openloadflow",
    "cuda_routine": "updateShunt",
    "cuda_fn_macro": "SH_FN",
    "scatter_prefix": "SH",
    "cuda_params": [["v", "g", "b"]],
    "cuda_consts": [],
    "doc": "Shunt-compensator AC flow residuals and Jacobian entries, parsed from "
           "the hand-written OLF equation-term Java (the source of truth).",
    "cuda_fused_doc": (
        "// FUSED per-element kernel body: one routine computes the shunt injections p/q\n"
        "// + their three non-zero derivatives for one shunt, writing into Struct-of-Arrays\n"
        "// at index i. No trig at all — the cheapest fill of the families; dqdb only\n"
        "// matters when the shunt is controllable (SHUNT_B is a skip column in plain PF)."),

    # --- Scatter stamp (where each output lands in OLF's equation system) -------
    #
    # Scalar terms: vVar is declared in the shared abstract base, bVar in the
    # reactive term (when the shunt is controllable). The creator wires bare-named
    # terms addTerm(p)/addTerm(q) inside createShuntEquation — term_prefix "" +
    # equations_method scope the check to that method. The shunt is a ONE-bus
    # element: endpoints "bus" (code 0) and "shunt" (code 2, the SHUNT_B skip
    # column, like BRANCH_ALPHA1/RHO1).
    "stamp": {
        "sources": {
            "columns": [
                "src/main/java/com/powsybl/openloadflow/ac/equations/"
                "AbstractShuntCompensatorEquationTerm.java",
                "src/main/java/com/powsybl/openloadflow/ac/equations/"
                "ShuntCompensatorReactiveFlowEquationTerm.java",
            ],
            "equations": "src/main/java/com/powsybl/openloadflow/ac/equations/"
                         "AcEquationSystemCreator.java",
        },
        "term_prefix": "",
        "equations_method": "createShuntEquation",
        "endpoints": ["bus", "shunt"],
        "residual_equation": {
            "p": ("BUS_TARGET_P", "bus"),
            "q": ("BUS_TARGET_Q", "bus"),
        },
        "variable_column": {
            "v": ("BUS_V", "bus"),
            "b": ("SHUNT_B", "shunt"),
        },
    },
}
