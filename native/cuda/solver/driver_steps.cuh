// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The driver's own steps that the outer-loop phases call back into: the batched fill, the zero-diagonal repair
// of a matrix and of a right-hand side, the pilot-voltage and DISTR_Q target stamps, and the reject trace.
#ifndef OLF_GPU_SOLVER_DRIVER_STEPS_CUH
#define OLF_GPU_SOLVER_DRIVER_STEPS_CUH

#include <cstdio>
#include <cstdlib>
#include "../equations/scatter.cuh"
#include "../fixedmatrix/reactive_relocation.cuh"
#include "driver_pass.cuh"
#include "../common/trace.cuh"

namespace olfgpu {

// Fill J and F (or F only) for every active scenario.
void fillActiveBatch(BatchSolverContext* ctx, DriverPass& drv, bool fOnly);

// The zero-diagonal row repair on the assembled values, and the same combination on a right-hand side (nS scenarios
// from B, choices from the last repairZeroDiagonalRows; sOff selects the scenario block of a per-scenario call).
void repairZeroDiagonalRows(BatchSolverContext* ctx, int nS, double* vals);
void repairZeroDiagonalRightHandSide(BatchSolverContext* ctx, int nS, double* B, int sOff = 0);

// The relocation tables of a family with the pilot-self entries of this context, and those entries.
ReactiveRelocationTables withPilotSelf(ReactiveRelocationTables rt, BatchSolverContext* ctx);
const int* pilotSelfReactiveLimitEntryOrNull(BatchSolverContext* ctx);

// Lay a remote-voltage-control group's rows out from the current pin states.
void launchPilotTargetVEquationUpdate(BatchSolverContext* ctx, DriverPass& drv, bool staticLimits = false, bool continuedPins = false);
void launchDistrQTargetsInner(BatchSolverContext* ctx, DriverPass& drv);

// OLF_GPU_REJECT_TRACE=1: WHY a scenario left the batch for the CPU replay (status -2). The census
// counts them under one tag ("RUNTIME:status=-2"); this names the gate, per slot, so the Java side
// can pair it with the contingency id (GpuBatchPrecomputer prints the id under the same switch).
static inline void traceReject(int scenario, const char* reason, int outer, int it) {
    static const bool on = traceOn(Trace::REJECT_TRACE);
    if (on) {
        std::fprintf(stderr, "GPU_REJECT slot=%d reason=%s outer=%d it=%d\n", scenario, reason, outer, it);
    }
}

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_DRIVER_STEPS_CUH
