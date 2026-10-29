// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See outer_loop_sequence.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <vector>

#include "../common/stats.cuh"
#include "../common/codes/gpu_ac_solver_status.cuh"
#include "../common/codes/gpu_outer_loop_kind.cuh"
#include "../common/codes/gpu_realistic_voltage_fixing_loop.cuh"
#include "../common/codes/gpu_transformer_voltage_control_mode.cuh"
#include "../diagnostics/driver_traces.cuh"
#include "driver_steps.cuh"
#include "outer_loop_check.cuh"
#include "realistic_voltage_check.cuh"
#include "outer_loop_sequence.cuh"
#include "../common/trace.cuh"

namespace olfgpu {

void initOuterLoopSequence(OuterLoopSequence& seq, BatchSolverContext* ctx, const DriverPass& drv, bool checksOn,
                           int maxOuterLoopIterations) {
    seq.checksOn = checksOn;
    // Where the realistic-voltage check STARTS applying in this sweep. AcloadFlowEngine keeps
    // checkUnrealisticStates false up to and including the last loop that can fix an unrealistic state
    // (that is the whole point of robust mode: give ReactiveLimits / the discrete
    // TransformerVoltageControl a chance to repair the state instead of failing the run on it), then
    // turns it on for the rest of the sweep and resets it at the top of the next one. A check at index i
    // applies loopOrder[i] to the state the scenario's solve produced, so "past the fixing loop" is
    // i > realisticFixingIdx. Not found, or no fixing loop at all, means check every solve — the
    // conservative direction, since a check that fires too early only routes to the CPU.
    seq.realisticFixingIdx = -1;
    for (int i = 0; i < drv.nLoops; i++) {
        bool isFixing = ctx->realisticFixingLoop == GpuRealisticVoltageFixingLoop::REACTIVE_LIMITS
                ? drv.loopOrder[i] == GpuOuterLoopKind::REACTIVE_LIMITS
                : (ctx->realisticFixingLoop == GpuRealisticVoltageFixingLoop::TRANSFORMER_VOLTAGE_CONTROL
                        && drv.loopOrder[i] == GpuOuterLoopKind::TRANSFORMER_VOLTAGE_CONTROL
                        && ctx->transformerVoltageControlMode != GpuTransformerVoltageControlMode::WITH_GENERATOR_VOLTAGE_CONTROL);
        if (isFixing) {
            seq.realisticFixingIdx = i;
        }
    }
    seq.maxOuterLoopIterations = maxOuterLoopIterations;
    // OLF_GPU_LOOP_TRACE=<slot>: the device side of OLF's OL_CHECK - which loop a round ran, whether
    // it found the slot UNSTABLE (a re-solve follows) and how many scenarios that loop found unstable.
    // Two arms that agree on every residual and still stop apart differ in this SEQUENCE.
    // A slot resolved per chunk from a contingency id (setBatchLoopTraceSlot) wins over the env slot:
    // chunk composition moves with the resolved chunk size, so a fixed slot traces a different
    // scenario from one run to the next (IZUZ: slot 170, then 18)
    static const char* loopTraceEnv = traceValue(Trace::LOOP_TRACE);
    seq.loopTraceSlot = ctx->loopTraceSlot >= 0 ? ctx->loopTraceSlot : (loopTraceEnv != nullptr ? std::atoi(loopTraceEnv) : -1);
}

void checkSolvedScenarios(BatchSolverContext* ctx, DriverPass& drv, OuterLoopSequence& seq, const std::vector<char>& solved,
                          std::vector<char>& restarted) {
    const int nS = drv.nS;
    const int nLoops = drv.nLoops;
    const int (&loopOrder)[MAX_DEVICE_LOOPS] = drv.loopOrder;
    std::vector<char>& active = drv.active;
    std::vector<char>& live = drv.live;
    std::vector<char>& outerLoopCheckActive = drv.outerLoopCheckActive;
    BatchResult& res = drv.res;
    int& loopIdx = drv.loopIdx;
    restarted.assign(nS, 0);

    // ---- AcloadFlowEngine's realistic-voltage check, on this INTERMEDIATE state ----
    // OLF re-checks after every solver run once the last loop that can fix an unrealistic state has
    // run, and an unrealistic state ends the outer-loop cycle with UNREALISTIC_STATE: the run FAILS
    // and reports no violations. Which solve the check belongs to depends on the configured loop
    // order and on robust mode, so rather than reproduce that ordering the device routes any
    // scenario that is EVER unrealistic to the CPU replay, which applies OLF's rule itself. That is
    // stricter than OLF (a state it would have let a later loop fix also leaves the batch), so it
    // can cost batching share and can never change an answer.
    if (ctx->realisticCheck) {
        std::vector<char> gate(nS);
        bool anyGate = false;
        for (int s = 0; s < nS; s++) {
            gate[s] = (solved[s] && res.status[s] == GpuAcSolverStatus::CONVERGED && live[s] && seq.scenarioLoopIdx[s] > seq.realisticFixingIdx) ? 1 : 0;
            anyGate = anyGate || gate[s];
        }
        if (anyGate) {
            rejectUnrealistic(ctx, drv, gate);
        }
    }

    if (!seq.checksOn) {
        return;                                             // fixed-iter / no controllers: one solve only
    }

    // ---- the checks: each scenario that just converged, at its own loop, then on along the next loops while STABLE ----
    for (int s = 0; s < nS; s++) {
        seq.checkPending[s] = (solved[s] && res.status[s] == GpuAcSolverStatus::CONVERGED && live[s]) ? 1 : 0;
    }
    std::vector<char> outerLoopUnstable(nS);
    // A scenario wraps to the first loop at most once per call (its sweep restarts only after a re-solve, and a
    // re-solve takes it out of these checks), so this runs the loops at most twice.
    while (std::find(seq.checkPending.begin(), seq.checkPending.end(), 1) != seq.checkPending.end()) {
    for (int i = 0; i < nLoops; i++) {
        bool anyChecked = false;
        for (int s = 0; s < nS; s++) {
            outerLoopCheckActive[s] = (seq.checkPending[s] && seq.scenarioLoopIdx[s] == i) ? 1 : 0;
            anyChecked = anyChecked || outerLoopCheckActive[s];
        }
        if (!anyChecked) {
            continue;
        }
        loopIdx = i;
        const double checkStartMilliseconds = drv.outerLoopTiming ? nowMilliseconds() : 0.0;
        checkOuterLoop(ctx, drv, outerLoopUnstable);
        int nChanged = 0;
        for (int s = 0; s < nS; s++) {
            if (outerLoopUnstable[s]) {
                nChanged++;
            }
        }
        if (seq.loopTraceSlot >= 0 && seq.loopTraceSlot < nS && outerLoopCheckActive[seq.loopTraceSlot]) {
            traceLoopSlot(ctx, drv, seq.loopTraceSlot, nChanged, outerLoopUnstable, solved[seq.loopTraceSlot] != 0);
        }
        for (int s = 0; s < nS; s++) {
            if (!outerLoopCheckActive[s]) {
                continue;
            }
            if (!live[s]) {
                seq.checkPending[s] = 0;                    // the check routed it to the CPU, whatever its verdict
                continue;
            }
            if (!outerLoopUnstable[s]) {
                continue;                                   // STABLE: on to the next loop, below
            }
            seq.checkPending[s] = 0;                        // UNSTABLE: re-solve, then this loop checks again
            if (++seq.productiveIterations[s] >= seq.maxOuterLoopIterations) {
                // This scenario exhausted ITS outer-iteration budget mid-sequence. Never report it
                // converged on a half-finished RL/DS fixpoint — reject to the CPU fallback, which
                // replays it under OLF's own maxOuterLoopIterations/UNSTABLE handling.
                traceReject(s, "outer-loop-budget", drv.outer, 0);
                res.status[s] = GpuAcSolverStatus::REJECTED;
                live[s] = 0;
                continue;
            }
            active[s] = 1;
            restarted[s] = 1;
            seq.sweepResolved[s] = 1;
            seq.lastUnstableLoopIdx[s] = i;
            seq.solveTrigger[s] = loopOrder[i];
        }
        // The last loop that can fix an unrealistic state is done for this sweep: AcloadFlowEngine checks the
        // state it leaves (runOuterLoop, after its do-while), whether or not the loop re-solved. The solve gate
        // above only sees the solves of the LATER loops, so with that loop last in the order (the discrete
        // TransformerVoltageControl under the validation parameters) only the final state was ever checked.
        if (ctx->realisticCheck && i == seq.realisticFixingIdx) {
            std::vector<char> gate(nS);
            for (int s = 0; s < nS; s++) {
                gate[s] = (outerLoopCheckActive[s] && !outerLoopUnstable[s] && res.status[s] == GpuAcSolverStatus::CONVERGED && live[s]) ? 1 : 0;
            }
            rejectUnrealistic(ctx, drv, gate);
        }
        for (int s = 0; s < nS; s++) {                      // STABLE: the next loop, against the same state
            if (!outerLoopCheckActive[s] || !seq.checkPending[s] || outerLoopUnstable[s]) {
                continue;
            }
            int next = i + 1;
            if (!live[s]) {
                next = -1;                                  // unrealistic at the fixing loop: routed to the CPU
            } else if (next == nLoops) {                    // the end of the sweep
                if (seq.sweepResolved[s]) {
                    seq.sweepResolved[s] = 0;
                    next = 0;                               // the next sweep, from the first loop
                } else {
                    next = -1;                              // a whole sweep re-solved nothing: done
                }
            }
            if (next >= 0 && next == seq.lastUnstableLoopIdx[s]) {
                next = -1;                                  // back at the loop last UNSTABLE: done
            }
            if (next < 0) {
                seq.checkPending[s] = 0;
            } else {
                seq.scenarioLoopIdx[s] = next;
            }
        }
        if (drv.outerLoopTiming) {
            gOuterLoopCheckMilliseconds[loopOrder[i]] += nowMilliseconds() - checkStartMilliseconds;
            gOuterLoopCheckRounds[loopOrder[i]]++;
        }
    }
    }
}

}  // namespace olfgpu
