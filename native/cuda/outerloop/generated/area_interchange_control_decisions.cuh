// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The rules of the area interchange control's area rules, parsed from the OLF Java (AreaInterchangeControlDecisions, the source of truth).
//
// GENERATED — DO NOT EDIT. Regenerate with: python3 codegen/java2cuda/java2cuda.py (parsed from the OLF Java).
#pragma once
#include <math.h>

#ifndef AREA_INTERCHANGE_CONTROL_DECISIONS_FN
#define AREA_INTERCHANGE_CONTROL_DECISIONS_FN __host__ __device__   // also compiles host-side for the vectors check
#endif

namespace powsybl { namespace openloadflow { namespace area_interchange_control_decisions {

// AreaInterchangeControlDecisions's constants.

// Java library functions, each exactly Java's.
AREA_INTERCHANGE_CONTROL_DECISIONS_FN inline double java_signum(double d) {
    return (d == 0.0 || d != d) ? d : copysign(1.0, d);
}

// One device function per OLF Java method — SAME name, SAME signature, SAME statements.

AREA_INTERCHANGE_CONTROL_DECISIONS_FN inline double interchangeMismatch(double interchange, double interchangeTarget) {
    return interchange - interchangeTarget;
}

AREA_INTERCHANGE_CONTROL_DECISIONS_FN inline double interchangeMismatchWithSlack(double interchange, double interchangeTarget, double slackDistributionFactor, double slackBusActivePowerMismatch) {
    return interchange - interchangeTarget + slackDistributionFactor * slackBusActivePowerMismatch;
}

AREA_INTERCHANGE_CONTROL_DECISIONS_FN inline bool isWithinMaxMismatch(double mismatch, double maxMismatchMw) {
    return fabs(mismatch) <= maxMismatchMw / 100.0 || fabs(mismatch) <= 1e-5;
}

AREA_INTERCHANGE_CONTROL_DECISIONS_FN inline double slackMargin(double mismatch, double interchangeMismatchWithSlack, double areaInterchangePMaxMismatchMw) {
    return java_signum(mismatch) * interchangeMismatchWithSlack + areaInterchangePMaxMismatchMw / 100.0;
}

AREA_INTERCHANGE_CONTROL_DECISIONS_FN inline double areaShare(double mismatch, double factor) {
    return fabs(mismatch * factor) > 1e-5
         ? mismatch * factor
         : java_signum(mismatch) * 1.01 * 1e-5;
}

AREA_INTERCHANGE_CONTROL_DECISIONS_FN inline bool isAboveResidue(double mismatch) {
    return fabs(mismatch) > 1e-5;
}

AREA_INTERCHANGE_CONTROL_DECISIONS_FN inline double remainingAfter(double remainingMismatch, double areaShare, double areaRemaining) {
    return remainingMismatch - areaShare + areaRemaining;
}

}}}  // namespace powsybl::openloadflow::area_interchange_control_decisions
