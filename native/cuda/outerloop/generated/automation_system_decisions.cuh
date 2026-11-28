// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The rules of the automation system outer loop (an overload management system's trippings), parsed from the OLF Java (AutomationSystemDecisions, the source of truth).
//
// GENERATED — DO NOT EDIT. Regenerate with: python3 codegen/java2cuda/java2cuda.py (parsed from the OLF Java).
#pragma once
#include <math.h>

#ifndef AUTOMATION_SYSTEM_DECISIONS_FN
#define AUTOMATION_SYSTEM_DECISIONS_FN __host__ __device__   // also compiles host-side for the vectors check
#endif

namespace powsybl { namespace openloadflow { namespace automation_system_decisions {

// AutomationSystemDecisions's constants.
constexpr int DECISION_NONE = 0;
constexpr int DECISION_OPEN = 1;
constexpr int DECISION_CLOSE = 2;

// One device function per OLF Java method — SAME name, SAME signature, SAME statements.

AUTOMATION_SYSTEM_DECISIONS_FN inline int trippingDecision(double i, double threshold, bool branchOpen, bool disabled, bool inService) {
    return !(i > threshold && branchOpen != disabled) ? DECISION_NONE
         : branchOpen && inService ? DECISION_OPEN
         : !branchOpen && !inService ? DECISION_CLOSE
         : DECISION_NONE;
}

}}}  // namespace powsybl::openloadflow::automation_system_decisions
