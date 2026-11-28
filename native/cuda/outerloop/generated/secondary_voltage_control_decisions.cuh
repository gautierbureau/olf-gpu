// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The rules of the secondary voltage control's zone rules (k, progress, clipping), parsed from the OLF Java (SecondaryVoltageControlDecisions, the source of truth).
//
// GENERATED — DO NOT EDIT. Regenerate with: python3 codegen/java2cuda/java2cuda.py (parsed from the OLF Java).
#pragma once
#include <math.h>

#ifndef SECONDARY_VOLTAGE_CONTROL_DECISIONS_FN
#define SECONDARY_VOLTAGE_CONTROL_DECISIONS_FN __host__ __device__   // also compiles host-side for the vectors check
#endif

namespace powsybl { namespace openloadflow { namespace secondary_voltage_control_decisions {

// SecondaryVoltageControlDecisions's constants.
constexpr int MAX_NO_PROGRESS_COUNT = 2;
constexpr double DV_EPS = 1E-4;
constexpr double DK_DIFF_MAX_EPS = 1E-3;
constexpr double K_SATURATION_EPS = 5E-2;
constexpr double MAX_TARGET_VOLTAGE_STEP_KV = 2;
constexpr double MIN_PROGRESS_RATIO = 1E-2;

// One device function per OLF Java method — SAME name, SAME signature, SAME statements.

SECONDARY_VOLTAGE_CONTROL_DECISIONS_FN inline double qToK(double q, double minQ, double maxQ) {
    return (2 * q - maxQ - minQ) / (maxQ - minQ);
}

SECONDARY_VOLTAGE_CONTROL_DECISIONS_FN inline bool hasProgressed(double pilotDvAbs, double dkDiffMax, double previousPilotDvAbs, double previousDkDiffMax) {
    return pilotDvAbs < previousPilotDvAbs * (1 - MIN_PROGRESS_RATIO)
         || dkDiffMax < previousDkDiffMax * (1 - MIN_PROGRESS_RATIO);
}

SECONDARY_VOLTAGE_CONTROL_DECISIONS_FN inline int noProgressCountAfter(bool hasPrevious, double pilotDvAbs, double dkDiffMax, double previousPilotDvAbs, double previousDkDiffMax, int noProgressCount) {
    return !hasPrevious || hasProgressed(pilotDvAbs, dkDiffMax, previousPilotDvAbs, previousDkDiffMax) ? 0 : noProgressCount + 1;
}

SECONDARY_VOLTAGE_CONTROL_DECISIONS_FN inline bool isStalled(int noProgressCount) {
    return noProgressCount >= MAX_NO_PROGRESS_COUNT;
}

SECONDARY_VOLTAGE_CONTROL_DECISIONS_FN inline bool isStepClipped(double dv, double nominalV) {
    double stepKv = dv * nominalV;
    return stepKv > MAX_TARGET_VOLTAGE_STEP_KV || stepKv < -MAX_TARGET_VOLTAGE_STEP_KV;
}

SECONDARY_VOLTAGE_CONTROL_DECISIONS_FN inline double clippedStep(double dv, double nominalV) {
    double stepKv = dv * nominalV;
    return stepKv > MAX_TARGET_VOLTAGE_STEP_KV ? MAX_TARGET_VOLTAGE_STEP_KV / nominalV
         : stepKv < -MAX_TARGET_VOLTAGE_STEP_KV ? -MAX_TARGET_VOLTAGE_STEP_KV / nominalV
         : dv;
}

SECONDARY_VOLTAGE_CONTROL_DECISIONS_FN inline bool isOutOfPlausibleRange(double targetV, double minPlausibleTargetVoltage, double maxPlausibleTargetVoltage) {
    return targetV < minPlausibleTargetVoltage || targetV > maxPlausibleTargetVoltage;
}

SECONDARY_VOLTAGE_CONTROL_DECISIONS_FN inline double clipToPlausibleRange(double targetV, double minPlausibleTargetVoltage, double maxPlausibleTargetVoltage) {
    return targetV < minPlausibleTargetVoltage ? minPlausibleTargetVoltage
         : targetV > maxPlausibleTargetVoltage ? maxPlausibleTargetVoltage
         : targetV;
}

}}}  // namespace powsybl::openloadflow::secondary_voltage_control_decisions
