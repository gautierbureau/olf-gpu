// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The rules of the voltage monitoring outer loop (a stand-by automaton switching its bus PV), parsed from the OLF Java (MonitoringVoltageDecisions, the source of truth).
//
// GENERATED — DO NOT EDIT. Regenerate with: python3 codegen/java2cuda/java2cuda.py (parsed from the OLF Java).
#pragma once
#include <math.h>

#ifndef MONITORING_VOLTAGE_DECISIONS_FN
#define MONITORING_VOLTAGE_DECISIONS_FN __host__ __device__   // also compiles host-side for the vectors check
#endif

namespace powsybl { namespace openloadflow { namespace monitoring_voltage_decisions {

// MonitoringVoltageDecisions's constants.
constexpr int DECISION_NONE = 0;
constexpr int DECISION_TO_PV_HIGH = 1;
constexpr int DECISION_TO_PV_LOW = 2;
constexpr int DECISION_TO_PV_HIGH_THEN_LOW = 3;

// One device function per OLF Java method — SAME name, SAME signature, SAME statements.

MONITORING_VOLTAGE_DECISIONS_FN inline int decision(double controlledV, double lowThreshold, double highThreshold) {
    bool above = controlledV > highThreshold;
    bool below = controlledV < lowThreshold;
    return above && below ? DECISION_TO_PV_HIGH_THEN_LOW
         : above ? DECISION_TO_PV_HIGH
         : below ? DECISION_TO_PV_LOW
         : DECISION_NONE;
}

MONITORING_VOLTAGE_DECISIONS_FN inline bool switchesAtHigh(int decision) {
    return decision == DECISION_TO_PV_HIGH || decision == DECISION_TO_PV_HIGH_THEN_LOW;
}

MONITORING_VOLTAGE_DECISIONS_FN inline bool switchesAtLow(int decision) {
    return decision == DECISION_TO_PV_LOW || decision == DECISION_TO_PV_HIGH_THEN_LOW;
}

}}}  // namespace powsybl::openloadflow::monitoring_voltage_decisions
