// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The rules of the discrete controllers' allowed position range, parsed from the OLF Java (DiscretePositionDecisions, the source of truth).
//
// GENERATED — DO NOT EDIT. Regenerate with: python3 codegen/java2cuda/java2cuda.py (parsed from the OLF Java).
#pragma once
#include <math.h>

#ifndef DISCRETE_POSITION_DECISIONS_FN
#define DISCRETE_POSITION_DECISIONS_FN __host__ __device__   // also compiles host-side for the vectors check
#endif

namespace powsybl { namespace openloadflow { namespace discrete_position_decisions {

// DiscretePositionDecisions's constants.
constexpr int ALLOWED_INCREASE = 0;
constexpr int ALLOWED_DECREASE = 1;
constexpr int ALLOWED_BOTH = 2;

// One device function per OLF Java method — SAME name, SAME signature, SAME statements.

DISCRETE_POSITION_DECISIONS_FN inline int allowedFrom(int allowedDirection, int current, int min) {
    return allowedDirection == ALLOWED_INCREASE ? current : min;
}

DISCRETE_POSITION_DECISIONS_FN inline int allowedTo(int allowedDirection, int current, int count) {
    return allowedDirection == ALLOWED_DECREASE ? current : count - 1;
}

}}}  // namespace powsybl::openloadflow::discrete_position_decisions
