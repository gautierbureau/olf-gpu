// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The rules of the incremental voltage controls' shared rules (deadband, insensitivity, the direction ratchet), parsed from the OLF Java (IncrementalControlDecisions, the source of truth).
//
// GENERATED — DO NOT EDIT. Regenerate with: python3 codegen/java2cuda/java2cuda.py (parsed from the OLF Java).
#pragma once
#include <math.h>

#ifndef INCREMENTAL_CONTROL_DECISIONS_FN
#define INCREMENTAL_CONTROL_DECISIONS_FN __host__ __device__   // also compiles host-side for the vectors check
#endif

namespace powsybl { namespace openloadflow { namespace incremental_control_decisions {

// IncrementalControlDecisions's constants.
constexpr int DIRECTION_NONE = 0;
constexpr int DIRECTION_INCREASE = 1;
constexpr int DIRECTION_DECREASE = -1;

// One device function per OLF Java method — SAME name, SAME signature, SAME statements.

INCREMENTAL_CONTROL_DECISIONS_FN inline int directionOf(int from, int to) {
    return to - from > 0 ? DIRECTION_INCREASE : DIRECTION_DECREASE;
}

INCREMENTAL_CONTROL_DECISIONS_FN inline int directionChangeCountAfter(int directionChangeCount, int maxDirectionChange, int currentDirection, int direction) {
    return directionChangeCount < maxDirectionChange && currentDirection != DIRECTION_NONE && currentDirection != direction
         ? directionChangeCount + 1
         : directionChangeCount;
}

INCREMENTAL_CONTROL_DECISIONS_FN inline int currentDirectionAfter(int directionChangeCount, int maxDirectionChange, int currentDirection, int direction) {
    return directionChangeCount < maxDirectionChange ? direction : currentDirection;
}

INCREMENTAL_CONTROL_DECISIONS_FN inline int allowedDirectionAfter(int directionChangeCount, int maxDirectionChange, int allowedDirection, int direction) {
    return directionChangeCount < maxDirectionChange ? allowedDirection
         : direction == DIRECTION_INCREASE ? 0
         : 1;
}

INCREMENTAL_CONTROL_DECISIONS_FN inline double halfVoltageTargetDeadband(double targetDeadband, double nominalV, double minTargetDeadbandKv) {
    return (isnan(targetDeadband) ? minTargetDeadbandKv / nominalV : targetDeadband) / 2;
}

INCREMENTAL_CONTROL_DECISIONS_FN inline bool isOutOfDeadband(double diffV, double halfTargetDeadband) {
    return fabs(diffV) > halfTargetDeadband;
}

INCREMENTAL_CONTROL_DECISIONS_FN inline bool isInsensitive(double sensitivity, double minSensitivity) {
    return fabs(sensitivity) < minSensitivity;
}

}}}  // namespace powsybl::openloadflow::incremental_control_decisions
