# Copyright (c) 2026, RTE (http://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0
#
# java2cuda spec for the incremental voltage controls' shared rules (deadband, insensitivity, the direction ratchet) — METADATA ONLY, a RULE family (rules.py).
#
# OLF's IncrementalControlDecisions holds the rules as static functions of numbers. The functions listed here are the ones that are an
# expression of their arguments; the ones with a branch or a loop, or over arrays, are not transpiled (and their
# reference vectors are not checked).

MODULE = {
    "family": "incremental_control_decisions",

    # --- parse inputs: the OLF Java (source of truth) ---------------------------
    "java_class": "IncrementalControlDecisions",
    "java_source": "src/main/java/com/powsybl/openloadflow/lf/outerloop/IncrementalControlDecisions.java",
    # in the order they are emitted: a function is declared before the ones that call it
    "methods": ["directionOf", "directionChangeCountAfter", "currentDirectionAfter", "allowedDirectionAfter", "halfVoltageTargetDeadband", "isOutOfDeadband", "isInsensitive"],

    # another class's constants the rules read, as their values (the reference vectors check them)
    "extern_constants": {"DiscretePositionDecisions.ALLOWED_INCREASE": "0", "DiscretePositionDecisions.ALLOWED_DECREASE": "1"},

    # --- the reference vectors OLF's OuterLoopRuleVectorsTest maintains ---------
    "vectors": "src/test/resources/outerLoopRuleVectors/IncrementalControlDecisions.txt",

    # --- CUDA emission identity ------------------------------------------------
    "cuda_path": "native/cuda/outerloop/generated/incremental_control_decisions.cuh",
    "cuda_namespace": "powsybl::openloadflow::incremental_control_decisions",
    "cuda_fn_macro": "INCREMENTAL_CONTROL_DECISIONS_FN",
    "doc": "The rules of the incremental voltage controls' shared rules (deadband, insensitivity, the direction ratchet), parsed from the OLF Java (IncrementalControlDecisions, the source of truth).",
}
