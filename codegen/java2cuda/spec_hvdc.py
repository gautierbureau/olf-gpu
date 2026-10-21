# Copyright (c) 2026, RTE (http://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0
#
# java2cuda spec for the HVDC AC-emulation family — METADATA ONLY.
#
# The trickiest family, and the one whose old-pipeline special cases disappear here:
# - the values are PIECEWISE on the sign of rawP = p0 + k*(ph1 - ph2): the Java `?:`
#   ternary is ALSO valid C, so it passes straight through to the kernel;
# - each side has its OWN private boolean `isController` (>= 0 vs < 0, same
#   name/arity, different bodies): resolved by per-file helper scope (local wins),
#   never by a global pool;
# - the Jacobian is an INTENTIONAL approximation (the cable-loss derivative is
#   neglected — the Java comment says so), so the FD audit is skipped exactly like
#   the residual pipeline skipped its autodiff audit ("explicit family");
# - `getHvdcLineLosses` bottoms out in powsybl-iidm's HvdcUtils (another repo, not
#   parseable here): its closed form r*p*p/(v*v) is declared below as an extern
#   method — the ONE formula in this spec, guarded by the golden equivalence proof
#   (prove_equivalence.py supplied the exact same closed form).
# - rawP is hoisted into a kernel local (rawP1), matching the committed kernel's
#   CSE temp x0 and OLF's own `double rawP = rawP(...)` local.

MODULE = {
    "family": "hvdc",

    # --- parse inputs: the hand-written OLF Java (source of truth) --------------
    "java_sources": [
        "src/main/java/com/powsybl/openloadflow/ac/equations/AbstractHvdcAcEmulationFlowEquationTerm.java",
        "src/main/java/com/powsybl/openloadflow/ac/equations/HvdcAcEmulationSide1ActiveFlowEquationTerm.java",
        "src/main/java/com/powsybl/openloadflow/ac/equations/HvdcAcEmulationSide2ActiveFlowEquationTerm.java",
    ],

    "residuals": [("p1", None), ("p2", None)],
    "variables": ["ph1", "ph2"],
    "theta_methods": {},
    "precomputed": {},
    "hoist_helpers": ["rawP"],

    # powsybl-iidm's HvdcUtils.getHvdcLineLosses closed form (nominalV = the 2nd arg;
    # OLF calls it with 1). Declared, not parsed: the source lives in another repo.
    "extern_methods": [
        {"name": "getHvdcLineLosses", "params": ["p", "v", "r"],
         "body": "r * p * p / (v * v)"},
    ],

    # The angle derivatives neglect the cable-loss derivative on purpose (OLF comment:
    # "derivative of cable loss is neglected") — an FD check would rightly fail.
    "approx_jacobian": True,

    "base_symbols": ["p0", "k", "lossFactor1", "lossFactor2", "r", "ph1", "ph2"],
    "constants": {},

    # --- CUDA emission identity ------------------------------------------------
    "cuda_path": "native/cuda/equations/generated/hvdc.cuh",
    "scatter_path": "codegen/scatter/hvdc.scatter.json",
    "scatter_header_path": "native/cuda/equations/generated/hvdc_scatter.h",
    "cuda_namespace": "powsybl::openloadflow",
    "cuda_routine": "updateHvdcAcEmulation",
    "cuda_fn_macro": "HVDC_FN",
    "scatter_prefix": "HVDC",
    "cuda_params": [["p0", "k", "lossFactor1", "lossFactor2", "r"],
                    ["ph1", "ph2"]],
    "cuda_consts": [],
    "doc": "HVDC AC-emulation active-power residuals and (approximated) Jacobian "
           "entries, parsed from the hand-written OLF equation-term Java (the source "
           "of truth).",
    "cuda_fused_doc": (
        "// FUSED per-element kernel body: one routine computes the two HVDC AC-emulation\n"
        "// injections p1/p2 + their four (approximated, loss-derivative-neglecting) angle\n"
        "// derivatives for one hvdc, writing into Struct-of-Arrays at index i. The control\n"
        "// side is piecewise on the sign of rawP = p0 + k*(ph1 - ph2) — branchless ternaries,\n"
        "// no trig; rawP is shared as a kernel local."),

    # --- Scatter stamp (where each output lands in OLF's equation system) -------
    "stamp": {
        "sources": {
            "columns": [
                "src/main/java/com/powsybl/openloadflow/ac/equations/"
                "AbstractHvdcAcEmulationFlowEquationTerm.java",
            ],
            "equations": "src/main/java/com/powsybl/openloadflow/ac/equations/"
                         "AcEquationSystemCreator.java",
        },
        "term_prefix": "",
        "equations_method": "createHvdcAcEmulationEquations",
        "endpoints": ["bus1", "bus2"],
        "residual_equation": {
            "p1": ("BUS_TARGET_P", "bus1"),
            "p2": ("BUS_TARGET_P", "bus2"),
        },
        "variable_column": {
            "ph1": ("BUS_PHI", "bus1"),
            "ph2": ("BUS_PHI", "bus2"),
        },
    },
}
