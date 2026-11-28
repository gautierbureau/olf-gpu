# Copyright (c) 2026, RTE (http://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0
#
# java2cuda spec for the secondary voltage control's zone rules (k, progress, clipping) — METADATA ONLY, a RULE family (rules.py).
#
# OLF's SecondaryVoltageControlDecisions holds the rules as static functions of numbers. The functions listed here are the ones that are an
# expression of their arguments; the ones with a branch or a loop, or over arrays, are not transpiled (and their
# reference vectors are not checked).

MODULE = {
    "family": "secondary_voltage_control_decisions",

    # --- parse inputs: the OLF Java (source of truth) ---------------------------
    "java_class": "SecondaryVoltageControlDecisions",
    "java_source": "src/main/java/com/powsybl/openloadflow/ac/outerloop/SecondaryVoltageControlDecisions.java",
    # in the order they are emitted: a function is declared before the ones that call it
    "methods": ["qToK", "hasProgressed", "noProgressCountAfter", "isStalled", "isStepClipped", "clippedStep", "isOutOfPlausibleRange", "clipToPlausibleRange"],

    # --- the reference vectors OLF's OuterLoopRuleVectorsTest maintains ---------
    "vectors": "src/test/resources/outerLoopRuleVectors/SecondaryVoltageControlDecisions.txt",

    # --- CUDA emission identity ------------------------------------------------
    "cuda_path": "native/cuda/outerloop/generated/secondary_voltage_control_decisions.cuh",
    "cuda_namespace": "powsybl::openloadflow::secondary_voltage_control_decisions",
    "cuda_fn_macro": "SECONDARY_VOLTAGE_CONTROL_DECISIONS_FN",
    "doc": "The rules of the secondary voltage control's zone rules (k, progress, clipping), parsed from the OLF Java (SecondaryVoltageControlDecisions, the source of truth).",
}
