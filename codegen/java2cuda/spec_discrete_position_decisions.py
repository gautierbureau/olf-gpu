# Copyright (c) 2026, RTE (http://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0
#
# java2cuda spec for the discrete controllers' allowed position range — METADATA ONLY, a RULE family (rules.py).
#
# OLF's DiscretePositionDecisions holds the rules as static functions of numbers. The functions listed here are the ones that are an
# expression of their arguments; the ones with a branch or a loop, or over arrays, are not transpiled (and their
# reference vectors are not checked).

MODULE = {
    "family": "discrete_position_decisions",

    # --- parse inputs: the OLF Java (source of truth) ---------------------------
    "java_class": "DiscretePositionDecisions",
    "java_source": "src/main/java/com/powsybl/openloadflow/network/DiscretePositionDecisions.java",
    # in the order they are emitted: a function is declared before the ones that call it
    "methods": ["allowedFrom", "allowedTo"],

    # --- the reference vectors OLF's OuterLoopRuleVectorsTest maintains ---------
    "vectors": "src/test/resources/outerLoopRuleVectors/DiscretePositionDecisions.txt",

    # --- CUDA emission identity ------------------------------------------------
    "cuda_path": "native/cuda/outerloop/generated/discrete_position_decisions.cuh",
    "cuda_namespace": "powsybl::openloadflow::discrete_position_decisions",
    "cuda_fn_macro": "DISCRETE_POSITION_DECISIONS_FN",
    "doc": "The rules of the discrete controllers' allowed position range, parsed from the OLF Java (DiscretePositionDecisions, the source of truth).",
}
