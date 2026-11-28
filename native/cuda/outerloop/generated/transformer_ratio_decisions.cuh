// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The rules of the discrete transformer voltage control's ratio rules, parsed from the OLF Java (TransformerRatioDecisions, the source of truth).
//
// GENERATED — DO NOT EDIT. Regenerate with: python3 codegen/java2cuda/java2cuda.py (parsed from the OLF Java).
#pragma once
#include <math.h>

#ifndef TRANSFORMER_RATIO_DECISIONS_FN
#define TRANSFORMER_RATIO_DECISIONS_FN __host__ __device__   // also compiles host-side for the vectors check
#endif

namespace powsybl { namespace openloadflow { namespace transformer_ratio_decisions {

// TransformerRatioDecisions's constants.

// One device function per OLF Java method — SAME name, SAME signature, SAME statements.

TRANSFORMER_RATIO_DECISIONS_FN inline bool isOutOfRange(double r1, double minR1, double maxR1) {
    return r1 < minR1 || r1 > maxR1;
}

TRANSFORMER_RATIO_DECISIONS_FN inline double extremeR1(double r1, double minR1, double maxR1) {
    return r1 > maxR1 ? maxR1 : minR1;
}

TRANSFORMER_RATIO_DECISIONS_FN inline double updatedR1(double computedR1, double individualInitialR1, double individualMinR1, double individualMaxR1, double minR1, double maxR1, double initialR1) {
    return computedR1 >= initialR1 ?
         individualInitialR1 + (computedR1 - initialR1) * (individualMaxR1 - individualInitialR1) / (maxR1 - initialR1)
         :
         individualInitialR1 - (initialR1 - computedR1) * (individualInitialR1 - individualMinR1) / (initialR1 - minR1);
}

TRANSFORMER_RATIO_DECISIONS_FN inline double shareFromMax(double r1, double minR1, double maxR1) {
    return maxR1 != minR1 ? (maxR1 - r1) / (maxR1 - minR1) : 1;
}

TRANSFORMER_RATIO_DECISIONS_FN inline double shareFromMin(double r1, double minR1, double maxR1) {
    return maxR1 != minR1 ? (r1 - minR1) / (maxR1 - minR1) : 1;
}

TRANSFORMER_RATIO_DECISIONS_FN inline double groupInitialR1(double maxR1, double minR1, double sumShareFromMax, double sumShareFromMin, int count) {
    return (minR1 * sumShareFromMax + maxR1 * sumShareFromMin) / count;
}

}}}  // namespace powsybl::openloadflow::transformer_ratio_decisions
