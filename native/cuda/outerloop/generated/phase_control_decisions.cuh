// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The rules of the phase control outer loops (a current limiter, an active power controller), parsed from the OLF Java (PhaseControlDecisions, the source of truth).
//
// GENERATED — DO NOT EDIT. Regenerate with: python3 codegen/java2cuda/java2cuda.py (parsed from the OLF Java).
#pragma once
#include <math.h>

#ifndef PHASE_CONTROL_DECISIONS_FN
#define PHASE_CONTROL_DECISIONS_FN __host__ __device__   // also compiles host-side for the vectors check
#endif

namespace powsybl { namespace openloadflow { namespace phase_control_decisions {

// PhaseControlDecisions's constants.
constexpr int DECISION_NONE = 0;
constexpr int DECISION_SHIFT_ONE_TAP_UP = 1;
constexpr int DECISION_SHIFT_ONE_TAP_DOWN = 2;
constexpr int DECISION_REACH_PHASE_SHIFT = 3;
constexpr int DECISION_EXCEED_PHASE_SHIFT = 4;
constexpr int DECISION_INSENSITIVE = 5;
constexpr double MIN_TARGET_DEADBAND = 1 / 100.0;
constexpr double SENSI_EPS = 1e-6;

// Java library functions, each exactly Java's.
PHASE_CONTROL_DECISIONS_FN inline double java_max(double a, double b) {
    return a != a ? a : (a == 0.0 && b == 0.0 && signbit(a)) ? b : (a >= b ? a : b);
}
PHASE_CONTROL_DECISIONS_FN inline double java_to_radians(double angdeg) {
    return angdeg * 0.017453292519943295;
}
PHASE_CONTROL_DECISIONS_FN inline double java_to_degrees(double angrad) {
    return angrad * 57.29577951308232;
}

// One device function per OLF Java method — SAME name, SAME signature, SAME statements.

PHASE_CONTROL_DECISIONS_FN inline bool isAboveLimit(double i, double limit) {
    return i > limit;
}

PHASE_CONTROL_DECISIONS_FN inline int discreteLimiterDecision(double i, double limit, bool sensitivityPositive) {
    return !isAboveLimit(i, limit) ? DECISION_NONE
         : sensitivityPositive ? DECISION_SHIFT_ONE_TAP_DOWN
         : DECISION_SHIFT_ONE_TAP_UP;
}

PHASE_CONTROL_DECISIONS_FN inline double halfTargetDeadband(double targetDeadband) {
    return java_max(targetDeadband, MIN_TARGET_DEADBAND) / 2;
}

PHASE_CONTROL_DECISIONS_FN inline bool isOutOfDeadband(double p, double targetP, double halfTargetDeadband) {
    return fabs(p - targetP) > halfTargetDeadband;
}

PHASE_CONTROL_DECISIONS_FN inline bool needsSensitivity(bool limiter, double value, double target, double halfTargetDeadband) {
    return limiter ? isAboveLimit(value, target) : isOutOfDeadband(value, target, halfTargetDeadband);
}

PHASE_CONTROL_DECISIONS_FN inline bool isSensitive(double sensitivity) {
    return fabs(sensitivity) > SENSI_EPS;
}

PHASE_CONTROL_DECISIONS_FN inline double phaseShift(double value, double target, double sensitivity) {
    return java_to_radians((target - value) / sensitivity);
}

PHASE_CONTROL_DECISIONS_FN inline int incrementalDecision(bool limiter, double value, double target, double halfTargetDeadband, double sensitivity) {
    return !needsSensitivity(limiter, value, target, halfTargetDeadband) ? DECISION_NONE
         : !isSensitive(sensitivity) ? DECISION_INSENSITIVE
         : limiter ? DECISION_EXCEED_PHASE_SHIFT
         : DECISION_REACH_PHASE_SHIFT;
}

PHASE_CONTROL_DECISIONS_FN inline bool isSignificantCrossImpact(double phaseShift, double crossSensitivity, double otherI, double otherLimit, double margin) {
    return java_to_degrees(phaseShift) * crossSensitivity > margin * (otherI - otherLimit);
}

}}}  // namespace powsybl::openloadflow::phase_control_decisions
