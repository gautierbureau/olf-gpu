// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Driver traces (off the hot path, each behind its OLF_GPU_* switch): the per-row state dump, the closed-branch
// row the device actually evaluates, one pilot-self group's layout after each reactive-limits kernel, and the
// MAX_VOLTAGE_CHANGE clamp factor per iteration.
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

// OLF_GPU_CB_PROBE: the pi row closed branch e is evaluated with in every slot, tap overrides applied.
__global__ void cbRowProbe(int nS, int e, const double* cbIn, int nR1Ov,
                           const int* r1OvSlotElem, const double* r1OvVal,
                           int nElements, int elemOffset, const char* disabled, const char* active);

// OLF_GPU_PSG_STEP / OLF_GPU_ROW_TRACE: a pilot-self group's (or rows') layout after a reactive-limits kernel.
void psgStep(BatchSolverContext* ctx, int nS, int n, int outer, const char* tag);

// OLF_GPU_STEP_TRACE: the MAX_VOLTAGE_CHANGE clamp factor applied at iteration it, per active slot.
void traceStepFactor(BatchSolverContext* ctx, int nS, int it, const std::vector<char>& active);


// The driver's per-pass and end-of-solve traces, each behind its OLF_GPU_* switch.
// OLF_GPU_BATCH_TRACE / OLF_GPU_ROW_PROBE: the pass's per-scenario state before the loop check.
void traceBatchPass(BatchSolverContext* ctx, DriverPass& drv, int nLoops, bool solvePass);

// OLF_GPU_DS_DEBUG: is scenario 0 converged when the distributed-slack check reads its mismatch?
void traceDsConvergence(BatchSolverContext* ctx, DriverPass& drv, bool inDsPhase);

// OLF_GPU_RL_REASONS: why each reactive-limits check was unstable.
void traceRlReasons(BatchSolverContext* ctx, DriverPass& drv, const std::vector<char>& rlSwitched);

// OLF_GPU_RL_AUDIT / OLF_GPU_SW_TRACE: every reactive-limit switch of the pass, per scenario.
void traceRlSwitches(BatchSolverContext* ctx, DriverPass& drv, std::vector<std::vector<std::string>>& auditLog, bool rlAuditOn, int swTraceSlot);

// OLF_GPU_BATCH_TRACE / OLF_GPU_RL_TRACE_VROW: the reactive-limit entries' state after the check.
void traceRlState(BatchSolverContext* ctx, DriverPass& drv, bool inDsPhase, double reactiveLimitTol, std::vector<char>& rlSwitched, int rlTraceVrow);

// OLF_GPU_DS_DEBUG: the distributed-slack check's gate per scenario.
void traceDsCheck(BatchSolverContext* ctx, DriverPass& drv, bool inDsPhase);

// OLF_GPU_LOOP_TRACE: one scenario's loop sequence (which slot found it unstable).
void traceLoopSlot(BatchSolverContext* ctx, DriverPass& drv, int loopTraceSlot, int nChanged, std::vector<char>& rlSwitched, bool solvePass);

// OLF_GPU_ITER_LOG: the pass record (loop, re-solve, switch, iterations, status, DS gate).
void tracePassLog(BatchSolverContext* ctx, DriverPass& drv, std::vector<char>& rlSwitched, bool solvePass);

// OLF_GPU_SW_LOG: the on-device reactive-limit decision log of one scenario.
void traceSwitchLog(BatchSolverContext* ctx, DriverPass& drv);

// OLF_GPU_RL_AUDIT: the end-of-solve audit of the reactive-limit decisions.
void traceRlAudit(BatchSolverContext* ctx, DriverPass& drv, std::vector<std::vector<std::string>>& auditLog);

// OLF_GPU_DBG_DS: the distributed-slack state at the end of the solve.
void traceDsFinal(BatchSolverContext* ctx, DriverPass& drv);

// OLF_GPU_PHASE_TIME: the per-scenario solve-pass histogram.
void tracePhaseTime(BatchSolverContext* ctx, DriverPass& drv);

}  // namespace olfgpu

#endif  // OLF_GPU_DIAGNOSTICS_DRIVER_TRACES_CUH
