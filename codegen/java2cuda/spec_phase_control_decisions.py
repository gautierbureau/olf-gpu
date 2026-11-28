# Copyright (c) 2026, RTE (http://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0
#
# java2cuda spec for the phase control outer loops (a current limiter, an active power controller) — METADATA ONLY, a RULE family (rules.py).
#
# OLF's PhaseControlDecisions holds the rules as static functions of numbers. The functions listed here are the ones that are an
# expression of their arguments; the ones with a branch or a loop, or over arrays, are not transpiled (and their
# reference vectors are not checked).

MODULE = {
    "family": "phase_control_decisions",

    # --- parse inputs: the OLF Java (source of truth) ---------------------------
    "java_class": "PhaseControlDecisions",
    "java_source": "src/main/java/com/powsybl/openloadflow/lf/outerloop/PhaseControlDecisions.java",
    # in the order they are emitted: a function is declared before the ones that call it
    "methods": ["isAboveLimit", "discreteLimiterDecision", "halfTargetDeadband", "isOutOfDeadband", "needsSensitivity", "isSensitive", "phaseShift", "incrementalDecision", "isSignificantCrossImpact"],

    # another class's constants the rules read, as their values (the reference vectors check them)
    "extern_constants": {"PerUnit.SB": "100.0"},

    # --- the reference vectors OLF's OuterLoopRuleVectorsTest maintains ---------
    "vectors": "src/test/resources/outerLoopRuleVectors/PhaseControlDecisions.txt",

    # --- CUDA emission identity ------------------------------------------------
    "cuda_path": "native/cuda/outerloop/generated/phase_control_decisions.cuh",
    "cuda_namespace": "powsybl::openloadflow::phase_control_decisions",
    "cuda_fn_macro": "PHASE_CONTROL_DECISIONS_FN",
    "doc": "The rules of the phase control outer loops (a current limiter, an active power controller), parsed from the OLF Java (PhaseControlDecisions, the source of truth).",
}
