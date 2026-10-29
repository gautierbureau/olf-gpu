// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// SlotSolveState: what each slot of the bank carries through the batched solve - whether its scenario is solving, its
// result so far, its Newton iteration and where it stands in OLF's outer-loop sequence - and BatchResult, what a batched
// solve hands back. The state lives on the context, so a solve call starts afresh only the slots it installs and leaves
// the others where they are: a call that returns while some scenarios still solve (to refill the slots of those that
// ended) is followed by one that goes on with them.
#ifndef OLF_GPU_SOLVER_SLOT_SOLVE_STATE_CUH
#define OLF_GPU_SOLVER_SLOT_SOLVE_STATE_CUH

#include <algorithm>
#include <vector>

#include "../common/codes/gpu_ac_solver_status.cuh"

namespace olfgpu {

struct BatchResult {
    std::vector<double> x;                                   // nScenarios * n
    std::vector<int> iters;                                  // nScenarios
    std::vector<double> norm;                                // nScenarios
    std::vector<int> status;                                 // nScenarios: a GpuAcSolverStatus
};

struct SlotSolveState {
    std::vector<char> active;                                // the scenario is solving: its Newton runs this batched iteration
    std::vector<char> live;                                  // the scenario is still iterating its outer loops (dropped on failure)
    BatchResult result;                                      // its status, iterations and norm so far (x is read at the end)
    // Its Newton: the iteration of its current solve, the solves it has ended, the batched iterations since it was
    // installed (the loop's safety cap), the linear-solve residual ratio of its last solve and the LINE_SEARCH state of
    // its current solve.
    std::vector<int> iteration, solves, batchedIterations;
    std::vector<double> lastSolveRatio, lineSearchLastSumOfSquares;
    std::vector<char> lineSearchHasLastTestResult;
    // Its place in OLF's outer-loop sequence (OuterLoopSequence).
    std::vector<int> loopIdx, lastUnstableLoopIdx, productiveIterations, solveTrigger;
    std::vector<char> sweepResolved, checkPending;
    // The slot holds a scenario the solve has not handed back yet, installed as the installSerial-th of the context.
    std::vector<char> holding;
    std::vector<long long> installSerial;
    long long installs = 0;

    // A solve call starts: each slot it installs (`installed`, bank slots in the install's order) begins its scenario -
    // solving, at the first iteration of a first solve, at the first loop of the sequence, with no result yet. A slot
    // neither installed nor solving holds no scenario: not live, no result. A slot still solving keeps its state.
    void start(int nS, const std::vector<int>& installed) {
        if (active.size() != static_cast<size_t>(nS)) {
            active.assign(nS, 0);
            live.assign(nS, 0);
            result.iters.assign(nS, 0);
            result.norm.assign(nS, 0.0);
            result.status.assign(nS, GpuAcSolverStatus::MAX_ITERATION_REACHED);
            iteration.assign(nS, 0);
            solves.assign(nS, 0);
            batchedIterations.assign(nS, 0);
            lastSolveRatio.assign(nS, 0.0);
            lineSearchLastSumOfSquares.assign(nS, 0.0);
            lineSearchHasLastTestResult.assign(nS, 0);
            loopIdx.assign(nS, 0);
            lastUnstableLoopIdx.assign(nS, -1);
            productiveIterations.assign(nS, 0);
            solveTrigger.assign(nS, -1);
            sweepResolved.assign(nS, 0);
            checkPending.assign(nS, 0);
            holding.assign(nS, 0);
            installSerial.assign(nS, 0);
        }
        for (int s = 0; s < nS; s++) {
            if (!active[s]) {
                live[s] = 0;
                result.iters[s] = 0;
                result.norm[s] = 0.0;
                result.status[s] = GpuAcSolverStatus::MAX_ITERATION_REACHED;
            }
        }
        for (int s : installed) {
            holding[s] = 1;
            installSerial[s] = installs++;
            active[s] = 1;
            live[s] = 1;
            result.iters[s] = 0;
            result.norm[s] = 0.0;
            result.status[s] = GpuAcSolverStatus::MAX_ITERATION_REACHED;   // until it converges or fails
            iteration[s] = 0;
            solves[s] = 0;
            batchedIterations[s] = 0;
            lastSolveRatio[s] = 0.0;
            lineSearchLastSumOfSquares[s] = 0.0;
            lineSearchHasLastTestResult[s] = 0;
            loopIdx[s] = 0;
            lastUnstableLoopIdx[s] = -1;
            productiveIterations[s] = 0;
            solveTrigger[s] = -1;
            sweepResolved[s] = 0;
            checkPending[s] = 0;
        }
    }

    // How many slots hold a scenario that has ended (it no longer solves) and is not handed back yet.
    int ended() const {
        int count = 0;
        for (size_t s = 0; s < active.size(); s++) {
            count += holding[s] && !active[s] ? 1 : 0;
        }
        return count;
    }

    // The slots whose scenarios have ended, in the order they were installed, handed back: they hold no scenario any more.
    std::vector<int> handBack() {
        std::vector<int> slots;
        for (size_t s = 0; s < active.size(); s++) {
            if (holding[s] && !active[s]) {
                slots.push_back(static_cast<int>(s));
                holding[s] = 0;
            }
        }
        std::sort(slots.begin(), slots.end(), [this](int a, int b) { return installSerial[a] < installSerial[b]; });
        return slots;
    }
};

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_SLOT_SOLVE_STATE_CUH
