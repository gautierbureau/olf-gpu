// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Driver traces (off the hot path, each behind its OLF_GPU_* switch): the per-row state dump, the closed-branch
// row the device actually evaluates, one pilot-self group's layout after each reactive-limits kernel, and the
// MAX_VOLTAGE_CHANGE clamp factor per iteration.
#ifndef OLF_GPU_DIAGNOSTICS_DRIVER_TRACES_CUH
#define OLF_GPU_DIAGNOSTICS_DRIVER_TRACES_CUH

#include <vector>

#include "../solver/batch_context.cuh"

namespace olfgpu {

// OLF_GPU_STATE_DUMP: every row's mode, target, state and residual of slot s at iteration it.
__global__ void stateDumpBatch(int n, const double* F, const int* mode, const double* target,
                               const double* x, int s, int it);

// OLF_GPU_CB_PROBE: the pi row closed branch e is evaluated with in every slot, tap overrides applied.
__global__ void cbRowProbe(int nS, int e, const double* cbIn, int nR1Ov,
                           const int* r1OvSlotElem, const double* r1OvVal,
                           int nElements, int elemOffset, const char* disabled, const char* active);

// OLF_GPU_PSG_STEP / OLF_GPU_ROW_TRACE: a pilot-self group's (or rows') layout after a reactive-limits kernel.
void psgStep(BatchSolverContext* ctx, int nS, int n, int outer, const char* tag);

// OLF_GPU_STEP_TRACE: the MAX_VOLTAGE_CHANGE clamp factor applied at iteration it, per active slot.
void traceStepFactor(BatchSolverContext* ctx, int nS, int it, const std::vector<char>& active);

}  // namespace olfgpu

#endif  // OLF_GPU_DIAGNOSTICS_DRIVER_TRACES_CUH
