# Copyright (c) 2026, RTE (http://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0
#
# java2cuda spec for the voltage monitoring outer loop (a stand-by automaton switching its bus PV) — METADATA ONLY, a RULE family (rules.py).
#
# OLF's MonitoringVoltageDecisions holds the rules as static functions of numbers. The functions listed here are the ones that are an
# expression of their arguments; the ones with a branch or a loop, or over arrays, are not transpiled (and their
# reference vectors are not checked).

MODULE = {
    "family": "monitoring_voltage_decisions",

    # --- parse inputs: the OLF Java (source of truth) ---------------------------
    "java_class": "MonitoringVoltageDecisions",
    "java_source": "src/main/java/com/powsybl/openloadflow/ac/outerloop/MonitoringVoltageDecisions.java",
    # in the order they are emitted: a function is declared before the ones that call it
    "methods": ["decision", "switchesAtHigh", "switchesAtLow"],

    # --- the reference vectors OLF's OuterLoopRuleVectorsTest maintains ---------
    "vectors": "src/test/resources/outerLoopRuleVectors/MonitoringVoltageDecisions.txt",

    # --- CUDA emission identity ------------------------------------------------
    "cuda_path": "native/cuda/outerloop/generated/monitoring_voltage_decisions.cuh",
    "cuda_namespace": "powsybl::openloadflow::monitoring_voltage_decisions",
    "cuda_fn_macro": "MONITORING_VOLTAGE_DECISIONS_FN",
    "doc": "The rules of the voltage monitoring outer loop (a stand-by automaton switching its bus PV), parsed from the OLF Java (MonitoringVoltageDecisions, the source of truth).",
}
