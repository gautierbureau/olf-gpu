// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The decisions of OLF's ReactiveLimitsOuterLoop, parsed from the OLF Java (ReactiveLimitsDecisions, the source of truth).
//
// GENERATED — DO NOT EDIT. Regenerate with: python3 codegen/java2cuda/java2cuda.py (parsed from the OLF Java).
#pragma once
#include <math.h>

#ifndef REACTIVE_LIMITS_DECISIONS_FN
#define REACTIVE_LIMITS_DECISIONS_FN __host__ __device__   // also compiles host-side for the vectors check
#endif

namespace powsybl { namespace openloadflow { namespace reactive_limits_decisions {

// ReactiveLimitsDecisions's constants.
constexpr int DECISION_NONE = 0;
constexpr int DECISION_TO_PQ_MIN_Q = 1;
constexpr int DECISION_TO_PQ_MAX_Q = 2;
constexpr int DECISION_TO_PQ_MIN_REALISTIC_V = 3;
constexpr int DECISION_TO_PQ_MAX_REALISTIC_V = 4;
constexpr int DECISION_TO_PV = 5;
constexpr int DECISION_REFREEZE_AT_LIMIT = 6;
constexpr int DECISION_TO_PV_BLOCKED = 7;
constexpr int DECISION_KEPT_PV = 8;
constexpr int LIMIT_NONE = 0;
constexpr int LIMIT_MIN_Q = 1;
constexpr int LIMIT_MAX_Q = 2;
constexpr int LIMIT_MIN_REALISTIC_V = 3;
constexpr int LIMIT_MAX_REALISTIC_V = 4;

// One device function per OLF Java method — SAME name, SAME signature, SAME statements.

REACTIVE_LIMITS_DECISIONS_FN inline int pvBusDecision(double q, double minQ, double maxQ, double tol, bool robustRemote, double v, double vLow, double vHigh) {
    return q < minQ - tol ? DECISION_TO_PQ_MIN_Q
         : q > maxQ + tol ? DECISION_TO_PQ_MAX_Q
         : robustRemote && v < vLow ? DECISION_TO_PQ_MIN_REALISTIC_V
         : robustRemote && v > vHigh ? DECISION_TO_PQ_MAX_REALISTIC_V
         : DECISION_NONE;
}

REACTIVE_LIMITS_DECISIONS_FN inline bool resetsV(int decision, bool robustRemote, double v, double vLow, double vHigh) {
    return decision != DECISION_NONE && robustRemote && (v < vLow || v > vHigh);
}

REACTIVE_LIMITS_DECISIONS_FN inline double frozenQ(int decision, double minQ, double maxQ, double initialTargetQ) {
    return decision == DECISION_TO_PQ_MIN_Q ? minQ
         : decision == DECISION_TO_PQ_MAX_Q ? maxQ
         : initialTargetQ;
}

REACTIVE_LIMITS_DECISIONS_FN inline bool isMinLimit(int qLimitType) {
    return qLimitType == LIMIT_MIN_Q || qLimitType == LIMIT_MIN_REALISTIC_V;
}

REACTIVE_LIMITS_DECISIONS_FN inline int pqBusDecision(int qLimitType, double controlledV, double targetV, bool canSwitchToPv, double minQ, double maxQ, double frozenTargetQ, double tol) {
    bool atMin = isMinLimit(qLimitType);
    bool wantsBack = atMin ? controlledV < targetV : controlledV > targetV;
    double limit = atMin ? minQ : maxQ;
    bool limitMoved = (qLimitType == LIMIT_MIN_Q || qLimitType == LIMIT_MAX_Q) && fabs(limit - frozenTargetQ) > tol;
    return wantsBack && canSwitchToPv ? DECISION_TO_PV
         : limitMoved ? DECISION_REFREEZE_AT_LIMIT
         : DECISION_NONE;
}

REACTIVE_LIMITS_DECISIONS_FN inline bool isSwitchToPq(int decision) {
    return decision >= DECISION_TO_PQ_MIN_Q && decision <= DECISION_TO_PQ_MAX_REALISTIC_V;
}

REACTIVE_LIMITS_DECISIONS_FN inline int limitTypeOf(int switchToPqDecision) {
    return switchToPqDecision;
}

}}}  // namespace powsybl::openloadflow::reactive_limits_decisions
