# Copyright (c) 2026, RTE (http://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0
#
# java2cuda spec for the reactive limits decisions — METADATA ONLY, a RULE family (rules.py).
#
# OLF's ReactiveLimitsDecisions holds the rules of ReactiveLimitsOuterLoop as static functions of
# numbers: a controlling bus out of its reactive limits (or, for a remote controller in robust
# mode, out of the realistic voltage band) switches to PQ; a bus switched to PQ switches back on
# the controlled voltage or follows a limit that moved. The functions listed here are the ones
# that are an expression of their arguments; the ones that walk a ReactiveLimitsVector (the
# election of the bus kept PV, the cap on switches back) have loops and are not transpiled.

MODULE = {
    "family": "reactive_limits_decisions",

    # --- parse inputs: the OLF Java (source of truth) ---------------------------
    "java_class": "ReactiveLimitsDecisions",
    "java_source": "src/main/java/com/powsybl/openloadflow/ac/outerloop/ReactiveLimitsDecisions.java",
    # in the order they are emitted: a function is declared before the ones that call it
    "methods": ["pvBusDecision", "resetsV", "frozenQ", "isMinLimit", "pqBusDecision", "isSwitchToPq", "limitTypeOf"],

    # --- the reference vectors OLF's ReactiveLimitsDecisionsVectorsTest maintains ---
    "vectors": "src/test/resources/reactiveLimitsDecisionsVectors.txt",

    # --- CUDA emission identity ------------------------------------------------
    "cuda_path": "native/cuda/outerloop/generated/reactive_limits_decisions.cuh",
    "cuda_namespace": "powsybl::openloadflow::reactive_limits_decisions",
    "cuda_fn_macro": "REACTIVE_LIMITS_DECISIONS_FN",
    "doc": "The decisions of OLF's ReactiveLimitsOuterLoop, parsed from the OLF Java "
           "(ReactiveLimitsDecisions, the source of truth).",
}
