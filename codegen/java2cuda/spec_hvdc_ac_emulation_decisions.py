# Copyright (c) 2026, RTE (http://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0
#
# java2cuda spec for the HVDC AC emulation limits outer loops (linear mode, saturation) — METADATA ONLY, a RULE family (rules.py).
#
# OLF's HvdcAcEmulationDecisions holds the rules as static functions of numbers. The functions listed here are the ones that are an
# expression of their arguments; the ones with a branch or a loop, or over arrays, are not transpiled (and their
# reference vectors are not checked).

MODULE = {
    "family": "hvdc_ac_emulation_decisions",

    # --- parse inputs: the OLF Java (source of truth) ---------------------------
    "java_class": "HvdcAcEmulationDecisions",
    "java_source": "src/main/java/com/powsybl/openloadflow/lf/outerloop/HvdcAcEmulationDecisions.java",
    # in the order they are emitted: a function is declared before the ones that call it
    "methods": ["linearModeDecision", "saturationModeDecision", "decision"],

    # --- the reference vectors OLF's OuterLoopRuleVectorsTest maintains ---------
    "vectors": "src/test/resources/outerLoopRuleVectors/HvdcAcEmulationDecisions.txt",

    # --- CUDA emission identity ------------------------------------------------
    "cuda_path": "native/cuda/outerloop/generated/hvdc_ac_emulation_decisions.cuh",
    "cuda_namespace": "powsybl::openloadflow::hvdc_ac_emulation_decisions",
    "cuda_fn_macro": "HVDC_AC_EMULATION_DECISIONS_FN",
    "doc": "The rules of the HVDC AC emulation limits outer loops (linear mode, saturation), parsed from the OLF Java (HvdcAcEmulationDecisions, the source of truth).",
}
