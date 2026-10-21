# Copyright (c) 2026, RTE (http://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0
#
# java2cuda spec for the VOLTAGE-DEPENDENT LOAD (ZIP/exponential) family — METADATA ONLY.
#
# OLF's AbstractLoadModelEquationTerm hand-writes the value AND the derivative as
# ACCUMULATION LOOPS over the load model's exp terms:
#     f    = target * sum(c * v^n)          (n != 0 terms)
#     dfdv = target * sum(c * n * v^(n-1))
# The GPU packs ONE element per exp term and performs the sum through the per-entry
# scatter (atomicAdd into the bus P/Q row) — exactly the Java loop's sigma — so the
# transpiled unit is the PER-TERM body, with the record accessors (expTerm.c(),
# expTerm.n()) becoming the scalar parameters c and n (extract_golden's REDUCTION
# pass). The n = 0 constant part lives in the target vector, mirrored at pack time.
#
# NO scatter stamp: the load-model landing rows are DYNAMIC per entry (the bus P row for
# expTermsP entries, the Q row for expTermsQ — carried by vdlIdx), so the fill
# kernel scatters through its own index pack, not a fixed-role table. The wiring
# guarantee stays with the packer + the 1e-12 parity suites; what this family adds
# is the FORMULA guarantee (golden manifest + finite-difference audit + rendered ==
# committed), like every other family.

MODULE = {
    "family": "load_model",

    # --- parse inputs: the hand-written OLF Java (source of truth) --------------
    "java_sources": [
        "src/main/java/com/powsybl/openloadflow/ac/equations/AbstractLoadModelEquationTerm.java",
    ],

    "residuals": [("f", None)],
    "variables": ["v"],
    "theta_methods": {},
    "precomputed": {},

    "base_symbols": ["v", "target", "c", "n"],
    "constants": {},

    # --- CUDA emission identity ------------------------------------------------
    "cuda_path": "native/cuda/equations/generated/load_model.cuh",
    "cuda_namespace": "powsybl::openloadflow",
    "cuda_routines": [""],                                # updateLoadModelWithoutDerivatives: no caller
    "cuda_routine": "updateLoadModel",
    "cuda_fn_macro": "LOAD_MODEL_FN",
    "scatter_prefix": "LOAD_MODEL",
    "cuda_params": [["v", "target", "c", "n"]],
    "cuda_consts": [],
    "doc": "Voltage-dependent (ZIP/exponential) load model per-exp-term value and "
           "derivative, parsed from the hand-written OLF equation-term Java (the "
           "source of truth); the sum over terms happens through the per-entry "
           "scatter, exactly OLF's accumulation loop.",
    "cuda_fused_doc": (
        "// FUSED per-EXP-TERM kernel body: one routine computes this term's contribution\n"
        "// f = c*target*v^n to its bus P/Q residual and dfdv = c*n*target*v^(n-1) to the\n"
        "// own-V Jacobian slot; the sum over a load model's terms is the per-entry\n"
        "// scatter (atomicAdd), exactly the accumulation loop in\n"
        "// AbstractLoadModelEquationTerm.f/dfdv."),
    # no "stamp": the landing rows are dynamic per entry (vdlIdx), not fixed roles
}
