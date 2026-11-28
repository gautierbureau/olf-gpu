# Copyright (c) 2026, RTE (http://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0
#
# java2cuda spec for the automation system outer loop (an overload management system's trippings) — METADATA ONLY, a RULE family (rules.py).
#
# OLF's AutomationSystemDecisions holds the rules as static functions of numbers. The functions listed here are the ones that are an
# expression of their arguments; the ones with a branch or a loop, or over arrays, are not transpiled (and their
# reference vectors are not checked).

MODULE = {
    "family": "automation_system_decisions",

    # --- parse inputs: the OLF Java (source of truth) ---------------------------
    "java_class": "AutomationSystemDecisions",
    "java_source": "src/main/java/com/powsybl/openloadflow/ac/outerloop/AutomationSystemDecisions.java",
    # in the order they are emitted: a function is declared before the ones that call it
    "methods": ["trippingDecision"],

    # --- the reference vectors OLF's OuterLoopRuleVectorsTest maintains ---------
    "vectors": "src/test/resources/outerLoopRuleVectors/AutomationSystemDecisions.txt",

    # --- CUDA emission identity ------------------------------------------------
    "cuda_path": "native/cuda/outerloop/generated/automation_system_decisions.cuh",
    "cuda_namespace": "powsybl::openloadflow::automation_system_decisions",
    "cuda_fn_macro": "AUTOMATION_SYSTEM_DECISIONS_FN",
    "doc": "The rules of the automation system outer loop (an overload management system's trippings), parsed from the OLF Java (AutomationSystemDecisions, the source of truth).",
}
