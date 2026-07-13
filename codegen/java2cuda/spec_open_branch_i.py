# Copyright (c) 2026, RTE (http://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0
#
# java2cuda spec for the open-branch CURRENT-MAGNITUDE family — METADATA ONLY.
#
# NEW GPU target: the residual pipeline emitted no CUDA for this family — see
# spec_closed_branch_i.py; prove.py's drop-in check reports it as new.
#
# Mixed-trig family: the methods take precomputed cosKsi/sinKsi (mapped back to
# kernel locals) but keep the connected side's angle raw (cos(ph2), sin(ph2)).
# The magnitude is rotation-invariant in the bus angle, so OLF implements only the
# voltage derivative (di2dv2 / di1dv1) — ph1/ph2 are still listed as variables
# below so the FD audit CONFIRMS the dropped angle derivatives are ~0 (the
# invariance claim), exactly like open-branch P/Q's cross-voltage drops.
# The value methods i1/i2 are PRIVATE in OLF (eval() is the only Java caller —
# the old pipeline promoted them with extract_golden --output); the parser just
# resolves them by name like any method. reI/imI components are hoisted; their
# gres/bres/shunt internals (private per-file helpers + the abstract bases'
# alpha-equivalent shunt) inline into the hoisted locals.

MODULE = {
    "family": "open_branch_i",

    # --- parse inputs: the hand-written OLF Java (source of truth) --------------
    "java_sources": [
        "src/main/java/com/powsybl/openloadflow/ac/equations/AbstractOpenSide1BranchAcFlowEquationTerm.java",
        "src/main/java/com/powsybl/openloadflow/ac/equations/AbstractOpenSide2BranchAcFlowEquationTerm.java",
        "src/main/java/com/powsybl/openloadflow/ac/equations/OpenBranchSide1CurrentMagnitudeEquationTerm.java",
        "src/main/java/com/powsybl/openloadflow/ac/equations/OpenBranchSide2CurrentMagnitudeEquationTerm.java",
    ],

    # Same side order as open-branch P/Q: side-1-open (i2) first, then side-2-open.
    "residuals": [("i2", None), ("i1", None)],
    "variables": ["v1", "v2", "ph1", "ph2"],
    "theta_methods": {},
    "precomputed": {"sinKsi": "sin(ksi)", "cosKsi": "cos(ksi)"},
    # shunt FIRST: it is a dependency of the reI/imI locals (hoisted expressions
    # resolve their kept calls against earlier-listed helpers' locals). gres/bres
    # cannot be hoisted (each side defines a different one under the same
    # name/arity) — they inline into the components, which is where OLF uses them.
    "hoist_helpers": ["shunt", "reI1", "imI1", "reI2", "imI2"],

    "base_symbols": ["v1", "v2", "ph1", "ph2", "r1", "ksi", "y",
                     "g1", "g2", "b1", "b2"],
    "constants": {"R2": 1},

    # --- CUDA emission identity (NEW — no residual-pipeline counterpart) --------
    "cuda_path": "native/cuda/open_branch_i.cuh",
    "cuda_namespace": "powsybl::openloadflow",
    "cuda_routine": "evalOpenBranchI",
    "cuda_fn_macro": "OBI_FN",
    "cuda_params": [["y", "ksi", "g1", "b1", "g2", "b2"],
                    ["v1", "ph1", "r1", "v2", "ph2"]],
    "cuda_consts": ["constexpr double R2 = 1.0;"],
    "doc": "Open-branch current magnitudes and their voltage derivatives, parsed from "
           "the hand-written OLF equation-term Java (the source of truth).",
    "cuda_fused_doc": (
        "// FUSED per-element kernel body: one routine computes the two open-branch current\n"
        "// magnitudes (i2 for a side-1-open branch, i1 for a side-2-open one) + their two\n"
        "// voltage derivatives, writing into Struct-of-Arrays at index i. The magnitude is\n"
        "// rotation-invariant in the bus angle, so only di/dv exists (FD-audited ~0 for the\n"
        "// angles). The complex components reI/imI are shared kernel locals. NEW GPU target\n"
        "// (the residual pipeline emitted no CUDA for current magnitude)."),
}
