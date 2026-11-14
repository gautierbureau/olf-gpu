// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The part of a batched solver context that NewtonRaphson reads beyond the system itself: its state vector scaling (MAX_VOLTAGE_CHANGE,
// LINE_SEARCH) and its stopping criteria (the norms, the per-row thresholds of PER_EQUATION_TYPE).
#ifndef OLF_GPU_SOLVER_CONTEXT_NEWTON_RAPHSON_CONTEXT_CUH
#define OLF_GPU_SOLVER_CONTEXT_NEWTON_RAPHSON_CONTEXT_CUH

#include <cublas_v2.h>

#include <vector>

#include "../../common/device_buf.cuh"
#include "../slot_rows.cuh"
#include "../slot_table.cuh"

namespace olfgpu {

struct NewtonRaphsonContext {
    bool clampStep = false;                                  // MAX_VOLTAGE_CHANGE per-scenario step clamp (set per solve)
    double maxDv = 0.0, maxDphi = 0.0;
    DeviceBuf dVarType, dStepFactor;                         // n ints / nS factors (lazy, first clamped solve)
    DeviceBuf dLineSearchPreviousX;                                      // n*nS pre-step iterates (lazy, LINE_SEARCH solves)
    int lineSearchStateVectorScalingMaxIteration = 10;                                      // LINE_SEARCH backtracks (setBatchLineSearch)
    double lineSearchStateVectorScalingStepFold = 4.0 / 3.0;
    DeviceBuf dNorms;                                        // nS*4 device-side convergence norms (see residualNormsBatch)
    DeviceBuf dFixedVariableRowMaxMismatch, dEquationRowMaxMismatch;                          // n each: per-row PER_EQUATION_TYPE thresholds (empty = 2-norm/infnorm criterion)
    bool perRowCriteria = false;
    DeviceBuf dAlternateRowMaxMismatch, dAlternateRowMaxMismatchSelector;                              // n each: the alternate threshold and its selector row (see residualNormsBatch)
    bool perRowAlternateCriteria = false;
    // Per-scenario ACTIVE equation counts (setBatchActiveRowCounts): OLF's L2 stopping threshold is
    // sqrt(eps^2 * nActive) and the CPU REMOVES equations for outaged elements carrying variables
    // (rho/a1 identities) and islanded buses — the padded device n over-counts by those rows, a ~%
    // threshold skew that lands stops up to the criterion apart. Empty = use n for every scenario.
    std::vector<int> hNActive;
};

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_CONTEXT_NEWTON_RAPHSON_CONTEXT_CUH
