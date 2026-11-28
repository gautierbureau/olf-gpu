// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The rules of the HVDC AC emulation limits outer loops (linear mode, saturation), parsed from the OLF Java (HvdcAcEmulationDecisions, the source of truth).
//
// GENERATED — DO NOT EDIT. Regenerate with: python3 codegen/java2cuda/java2cuda.py (parsed from the OLF Java).
#pragma once
#include <math.h>

#ifndef HVDC_AC_EMULATION_DECISIONS_FN
#define HVDC_AC_EMULATION_DECISIONS_FN __host__ __device__   // also compiles host-side for the vectors check
#endif

namespace powsybl { namespace openloadflow { namespace hvdc_ac_emulation_decisions {

// HvdcAcEmulationDecisions's constants.
constexpr int STATUS_LINEAR_MODE = 0;
constexpr int STATUS_SATURATION_MODE_FROM_CS1_TO_CS2 = 1;
constexpr int STATUS_SATURATION_MODE_FROM_CS2_TO_CS1 = 2;
constexpr int STATUS_FROZEN = 3;
constexpr int DECISION_NONE = 0;
constexpr int DECISION_TO_SATURATION_FROM_CS1_TO_CS2 = 1;
constexpr int DECISION_TO_SATURATION_FROM_CS2_TO_CS1 = 2;
constexpr int DECISION_TO_LINEAR_MODE = 3;

// One device function per OLF Java method — SAME name, SAME signature, SAME statements.

HVDC_AC_EMULATION_DECISIONS_FN inline int linearModeDecision(double p1, double p2, double pMaxFromCS1toCS2, double pMaxFromCS2toCS1) {
    return p1 >= 0 ? (p1 > pMaxFromCS1toCS2 ? DECISION_TO_SATURATION_FROM_CS1_TO_CS2 : DECISION_NONE)
         : p2 > pMaxFromCS2toCS1 ? DECISION_TO_SATURATION_FROM_CS2_TO_CS1
         : DECISION_NONE;
}

HVDC_AC_EMULATION_DECISIONS_FN inline int saturationModeDecision(int status, double p1, double p2, double pMaxFromCS1toCS2, double pMaxFromCS2toCS1) {
    return p1 > 0 && p1 < pMaxFromCS1toCS2 || p2 > 0 && p2 < pMaxFromCS2toCS1 ? DECISION_TO_LINEAR_MODE
         : p1 < 0 && status == STATUS_SATURATION_MODE_FROM_CS1_TO_CS2 ? DECISION_TO_SATURATION_FROM_CS2_TO_CS1
         : p2 < 0 && status == STATUS_SATURATION_MODE_FROM_CS2_TO_CS1 ? DECISION_TO_SATURATION_FROM_CS1_TO_CS2
         : DECISION_NONE;
}

HVDC_AC_EMULATION_DECISIONS_FN inline int decision(int status, double p1, double p2, double pMaxFromCS1toCS2, double pMaxFromCS2toCS1) {
    return status == STATUS_LINEAR_MODE ? linearModeDecision(p1, p2, pMaxFromCS1toCS2, pMaxFromCS2toCS1)
         : status == STATUS_SATURATION_MODE_FROM_CS1_TO_CS2 || status == STATUS_SATURATION_MODE_FROM_CS2_TO_CS1
         ? saturationModeDecision(status, p1, p2, pMaxFromCS1toCS2, pMaxFromCS2toCS1)
         : DECISION_NONE;
}

}}}  // namespace powsybl::openloadflow::hvdc_ac_emulation_decisions
