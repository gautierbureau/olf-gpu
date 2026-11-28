# Copyright (c) 2026, RTE (http://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0
#
# java2cuda spec for the area interchange control's area rules — METADATA ONLY, a RULE family (rules.py).
#
# OLF's AreaInterchangeControlDecisions holds the rules as static functions of numbers. The functions listed here are the ones that are an
# expression of their arguments; the ones with a branch or a loop, or over arrays, are not transpiled (and their
# reference vectors are not checked).

MODULE = {
    "family": "area_interchange_control_decisions",

    # --- parse inputs: the OLF Java (source of truth) ---------------------------
    "java_class": "AreaInterchangeControlDecisions",
    "java_source": "src/main/java/com/powsybl/openloadflow/lf/outerloop/AreaInterchangeControlDecisions.java",
    # in the order they are emitted: a function is declared before the ones that call it
    "methods": ["interchangeMismatch", "interchangeMismatchWithSlack", "isWithinMaxMismatch", "slackMargin", "areaShare", "isAboveResidue", "remainingAfter"],

    # another class's constants the rules read, as their values (the reference vectors check them)
    "extern_constants": {"PerUnit.SB": "100.0", "ActivePowerDistribution.P_RESIDUE_EPS": "1e-5"},

    # --- the reference vectors OLF's OuterLoopRuleVectorsTest maintains ---------
    "vectors": "src/test/resources/outerLoopRuleVectors/AreaInterchangeControlDecisions.txt",

    # --- CUDA emission identity ------------------------------------------------
    "cuda_path": "native/cuda/outerloop/generated/area_interchange_control_decisions.cuh",
    "cuda_namespace": "powsybl::openloadflow::area_interchange_control_decisions",
    "cuda_fn_macro": "AREA_INTERCHANGE_CONTROL_DECISIONS_FN",
    "doc": "The rules of the area interchange control's area rules, parsed from the OLF Java (AreaInterchangeControlDecisions, the source of truth).",
}
