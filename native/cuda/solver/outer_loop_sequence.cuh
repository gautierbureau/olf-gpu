// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// OLF's AcloadFlowEngine.runOuterLoops, scenario by scenario, inside the batched Newton loop. Each scenario stands at
// its own loop of loopOrder. When its solve converges it is checked at that loop (runOuterLoop re-checks the loop that
// made it re-solve), a STABLE check moves it on to the next loop against the SAME state, and an UNSTABLE one restarts
// its Newton at once, while the other scenarios go on with theirs. Its outer loops end when a sweep re-solved nothing,
// or when a new sweep reaches the loop it was last UNSTABLE at (OLF's lastUnstableOuterLoop: every loop since has been
// checked STABLE on this state).
#ifndef OLF_GPU_SOLVER_OUTER_LOOP_SEQUENCE_CUH
#define OLF_GPU_SOLVER_OUTER_LOOP_SEQUENCE_CUH

#include <vector>

#include "driver_pass.cuh"

namespace olfgpu {

struct OuterLoopSequence {
    bool checksOn = true;                                    // false: the run has no outer loop, every solve ends it
    int realisticFixingIdx = -1;                             // the last loop that can fix an unrealistic state, -1: none
    int maxOuterLoopIterations = 20;                         // OLF maxOuterLoopIterations, per scenario
    int loopTraceSlot = -1;                                  // OLF_GPU_LOOP_TRACE
    std::vector<int> scenarioLoopIdx;                        // the loop each scenario stands at
    std::vector<int> lastUnstableLoopIdx;                    // the loop each scenario was last UNSTABLE at, -1: none yet
    std::vector<char> sweepResolved;                         // the scenario re-solved during its current sweep
    std::vector<char> checkPending;                          // a check is due on the scenario's current state
    // PER-SCENARIO productive outer iterations: OLF's maxOuterLoopIterations is a per-RUN budget — a global counter
    // capped busy chunks collectively (violation-diff root cause: a scenario solved solo matched the CPU exactly but
    // diverged inside a full 500-chunk)
    std::vector<int> productiveIterations;
    std::vector<int> solveTrigger;                           // the loop kind whose check started the current solve, -1: the first
};

// The sequence of drv.loopOrder for every scenario, each at its first loop; checksOn false for a run with no outer loop.
void initOuterLoopSequence(OuterLoopSequence& seq, BatchSolverContext* ctx, const DriverPass& drv, bool checksOn,
                           int maxOuterLoopIterations);

// The scenarios whose solve just CONVERGED (solved[s]): AcloadFlowEngine's realistic-voltage check on the solves past
// the fixing loop, then their checks, each at its own loop and on along the next ones while STABLE. A scenario that an
// UNSTABLE check makes re-solve is active again and marked in restarted; the others have ended their outer loops, or
// were routed to the CPU.
void checkSolvedScenarios(BatchSolverContext* ctx, DriverPass& drv, OuterLoopSequence& seq, const std::vector<char>& solved,
                          std::vector<char>& restarted);

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_OUTER_LOOP_SEQUENCE_CUH
