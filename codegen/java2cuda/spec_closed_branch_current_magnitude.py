# Copyright (c) 2026, RTE (http://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0
#
# java2cuda spec for the closed-branch CURRENT-MAGNITUDE family — METADATA ONLY.
#
# Read by the current-limiter loops (outerloop/flow_observation.cuh): the current i1/i2,
# di1da1/di2da1 (PhaseControlOuterLoop's direction) and calculateSensi, which the device
# mirrors line for line on these methods. No assembler routine: the methods are called
# one by one, and "class_methods" emits the private components calculateSensi's one-sided
# rule at I = 0 combines.
#
# - the value i1 is OVERLOADED (public (..., a1, ..., ph2) over a private
#   (..., theta) core): the parser picks the public one and inlines the private
#   overload through it;
# - each side has its OWN private theta(...) (same name/arity, different body):
#   per-file helper scope resolves them, like HVDC's isController;
# - di1da1 = -di1dph2 (a1 enters theta opposite to ph2): a sibling call, inlined;
# - r1 is NOT a differentiation variable: OLF deliberately does not implement
#   di/dr1 (calculateSensi throws for dr1 != 0), so no dr1 output exists — unlike
#   the P/Q closed-branch family.
# Raw-trig family: no precomputed sin/cos parameters — the angles stay inline.

MODULE = {
    "family": "closed_branch_current_magnitude",

    # --- parse inputs: the hand-written OLF Java (source of truth) --------------
    "java_sources": [
        "src/main/java/com/powsybl/openloadflow/ac/equations/ClosedBranchSide1CurrentMagnitudeEquationTerm.java",
        "src/main/java/com/powsybl/openloadflow/ac/equations/ClosedBranchSide2CurrentMagnitudeEquationTerm.java",
    ],

    "residuals": [("i1", None), ("i2", None)],
    "variables": ["v1", "v2", "ph1", "ph2", "a1"],
    "theta_methods": {},
    "precomputed": {},
    # the components calculateSensi's one-sided rule at I = 0 combines (flow_observation.cuh mirrors it)
    "class_methods": {
        "ClosedBranchSide1CurrentMagnitudeEquationTerm": [
            "dreI1dph1", "dreI1dph2", "dreI1dv1", "dreI1dv2", "dimI1dph1", "dimI1dph2", "dimI1dv1", "dimI1dv2"],
        "ClosedBranchSide2CurrentMagnitudeEquationTerm": [
            "theta", "dreI2dph1", "dreI2dph2", "dreI2dv1", "dreI2dv2", "dimI2dph1", "dimI2dph2", "dimI2dv1", "dimI2dv2"],
    },

    "base_symbols": ["v1", "v2", "ph1", "ph2", "a1", "r1", "ksi", "y",
                     "g1", "g2", "b1", "b2"],
    "constants": {"R2": 1, "A2": 0},

    # --- CUDA emission identity ------------------------------------------------
    "cuda_path": "native/cuda/equations/generated/closed_branch_current_magnitude.cuh",
    "cuda_namespace": "powsybl::openloadflow",
    "cuda_routines": [],
    "cuda_fn_macro": "CLOSED_BRANCH_CURRENT_MAGNITUDE_FN",
    "cuda_params": [["y", "g1", "g2", "b1", "b2"],
                    ["v1", "r1", "v2", "ksi", "a1", "ph1", "ph2"]],
    "cuda_consts": ["constexpr double R2 = 1.0;", "constexpr double A2 = 0.0;"],
    "doc": "Closed-branch current magnitudes and their derivatives, parsed from "
           "the hand-written OLF equation-term Java (the source of truth).",
}
