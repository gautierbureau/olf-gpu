# Copyright (c) 2026, RTE (http://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0
#
# java2cuda spec for the discrete transformer voltage control's ratio rules — METADATA ONLY, a RULE family (rules.py).
#
# OLF's TransformerRatioDecisions holds the rules as static functions of numbers. The functions listed here are the ones that are an
# expression of their arguments; the ones with a branch or a loop, or over arrays, are not transpiled (and their
# reference vectors are not checked).

MODULE = {
    "family": "transformer_ratio_decisions",

    # --- parse inputs: the OLF Java (source of truth) ---------------------------
    "java_class": "TransformerRatioDecisions",
    "java_source": "src/main/java/com/powsybl/openloadflow/ac/outerloop/tap/TransformerRatioDecisions.java",
    # in the order they are emitted: a function is declared before the ones that call it
    "methods": ["isOutOfRange", "extremeR1", "updatedR1", "shareFromMax", "shareFromMin", "groupInitialR1"],

    # --- the reference vectors OLF's OuterLoopRuleVectorsTest maintains ---------
    "vectors": "src/test/resources/outerLoopRuleVectors/TransformerRatioDecisions.txt",

    # --- CUDA emission identity ------------------------------------------------
    "cuda_path": "native/cuda/outerloop/generated/transformer_ratio_decisions.cuh",
    "cuda_namespace": "powsybl::openloadflow::transformer_ratio_decisions",
    "cuda_fn_macro": "TRANSFORMER_RATIO_DECISIONS_FN",
    "doc": "The rules of the discrete transformer voltage control's ratio rules, parsed from the OLF Java (TransformerRatioDecisions, the source of truth).",
}
