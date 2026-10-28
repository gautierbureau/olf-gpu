// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Driver traces (off the hot path, each behind its OLF_GPU_* switch): the per-row state dump and the driver's
// per-pass and end-of-solve traces.
#ifndef OLF_GPU_DIAGNOSTICS_DRIVER_TRACES_CUH
#define OLF_GPU_DIAGNOSTICS_DRIVER_TRACES_CUH

#include <string>
#include <vector>

#include "../solver/batch_context.cuh"
#include "../solver/driver_pass.cuh"

namespace olfgpu {

// OLF_GPU_STATE_DUMP: every row's mode, target, state and residual of slot s at iteration it.
__global__ void stateDumpBatch(int n, const double* F, const int* mode, const double* target,
                               const double* x, int s, int it);

// The driver's per-pass and end-of-solve traces, each behind its OLF_GPU_* switch.
// OLF_GPU_BATCH_TRACE: the pass's per-scenario state before the loop check.
void traceBatchPass(BatchSolverContext* ctx, DriverPass& drv, int nLoops, int nChecked);

// OLF_GPU_RL_AUDIT / OLF_GPU_SW_TRACE: every reactive-limit switch of the pass, per scenario.
void traceReactiveLimitsSwitches(BatchSolverContext* ctx, DriverPass& drv, std::vector<std::vector<std::string>>& auditLog, bool reactiveLimitsAuditOn, int reactiveLimitsTraceScenario);

// OLF_GPU_BATCH_TRACE: the reactive-limit entries' state after the check.
void traceReactiveLimitsState(BatchSolverContext* ctx, DriverPass& drv, bool inDistributedSlackPhase, double reactiveLimitTol, std::vector<char>& outerLoopUnstable);

// OLF_GPU_LOOP_TRACE: one scenario's loop sequence (which slot found it unstable).
void traceLoopSlot(BatchSolverContext* ctx, DriverPass& drv, int loopTraceSlot, int nChanged, std::vector<char>& outerLoopUnstable, bool solved);

// OLF_GPU_SW_LOG: the on-device reactive-limit decision log of one scenario.
void traceSwitchLog(BatchSolverContext* ctx, DriverPass& drv);

// OLF_GPU_RL_AUDIT: the end-of-solve audit of the reactive-limit decisions.
void traceReactiveLimitsAudit(BatchSolverContext* ctx, DriverPass& drv, std::vector<std::vector<std::string>>& auditLog);

// OLF_GPU_PHASE_TIME: the per-scenario solve-pass histogram.
void tracePhaseTime(BatchSolverContext* ctx, DriverPass& drv);

}  // namespace olfgpu

#endif  // OLF_GPU_DIAGNOSTICS_DRIVER_TRACES_CUH
