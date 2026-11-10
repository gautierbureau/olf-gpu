// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Run-wide counters and timers of the batched solver, printed by the OLF_GPU_PHASE_TIME trace. Declared here,
// defined once in stats.cu.
#ifndef OLF_GPU_COMMON_STATS_CUH
#define OLF_GPU_COMMON_STATS_CUH

#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <map>
#include <utility>

#include "loop_kinds.cuh"
#include "codes/gpu_outer_loop_kind.cuh"
#include "trace.cuh"

// OLF_GPU_PHASE_TIME accounting: the batched Newton loops (one per solveBatchContext call) and the scenarios' restarts
// in place (an UNSTABLE check made the scenario re-solve).
extern long long gNewtonLoops;
extern long long gRestarts;

// OLF_GPU_PHASE_TIME: where the Newton loops' time and work go. The scenarios' solves by what started them, their
// first solve or the check of the GpuOuterLoopKind (index) that TRIGGERED a re-solve; the checks by the loop CHECKED.
extern double gNewtonLoopMilliseconds, gOuterLoopCheckMilliseconds[GpuOuterLoopKind::COUNT];
extern long long gNewtonLoopWidthIterations;
extern long long gOuterLoopFirstSolveScenarios, gOuterLoopReSolveScenarios[GpuOuterLoopKind::COUNT];
extern long long gOuterLoopFirstSolveLaneIterations, gOuterLoopReSolveLaneIterations[GpuOuterLoopKind::COUNT];
extern long long gOuterLoopCheckRounds[GpuOuterLoopKind::COUNT];

inline double nowMilliseconds() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

extern long long gNewtonIters;
extern int gNewtonLoopIteration;                             // the batched iteration the Newton loop is at
extern long long gActiveIterSum;
extern long long gWidthIterSum;
extern long long gCompactIters;
extern long long gBatchedLuFactorizations;                   // factorizations of our batched LU
extern long long gBatchedLuPivotedLanes;                     // ...and the lanes of them its pivoting engine factorized
extern long long gBatchedLuPivotDelayOrders;                 // fast-engine orders built with learned pivot delays
extern long long gBatchedLuSplitLaunches;                    // fast-engine steps and levels launched with their long rows split
extern long long gBatchedLuSupernodalSplitLaunches;          // pivoting-engine forward levels with long rows split
extern long long gBatchedLuSupernodalRowLaunches;            // pivoting-engine backward levels solved a row per warp
extern long long gBatchedLuSupernodalForwardRowLaunches;     // pivoting-engine split forward levels summed a row per block
extern long long gBatchedLuSupernodalRowFactorizations;      // pivoting-engine diagonal blocks factorized a row per warp
extern long long gBatchedLuRoutedLanes;                      // lanes sent straight to the pivoting engine, past the fast one
extern long long gBatchedLuFastLaneSolves;                   // lanes the fast engine's triangular solves ran on
// Factorizations by kind (the compacted ones are gCompactIters): full-width uniform-batch Newton iterations, robust ones
// (one scenario each), and those of the in-kernel loops' sensitivities, with their unit right-hand-side solves. A check
// round is one outer loop checked on the scenarios standing at it.
extern long long gFullFactorizations;
extern long long gRobustFactorizations;
extern long long gSensitivityFactorizations;
extern long long gSensitivitySolves;
extern long long gRefinementFactorizations;   // compacted factorizations the full-width refinement took
extern long long gCheckRounds;
// Per scenario at the end of a completed batch call: how many SOLVE PASSES it took part in (1 = the state came
// right out of the first inner Newton; each further pass is an outer loop that changed something and
// re-solved), the first pass's iterations and the total. Histogram key = passes (capped at 15).
extern long long gOuterLoopPassHistogramCount[16], gOuterLoopPassHistogramFirstSolveIterations[16], gOuterLoopPassHistogramTotalIterations[16];
extern long long gOuterLoopNotConverged, gOuterLoopNotConvergedIterations;
// cuDSS symbolic analyses run: the main contexts (a batch, the base load flow) and the straggler-compaction
// context (one per compact WIDTH it is analyzed at).
extern long long gAnalysesMain;
extern long long gAnalysesCompact;
// Adaptive iterative refinement: the refinement solves actually run and the residual checks (one per batched
// iteration at least).
extern long gIterativeRefinementExtraSolves;
extern long gIterativeRefinementChecks;
// ...and what the lanes need (OLF_GPU_PHASE_TIME only): the backward-residual score of every scored lane right after the
// Newton solve, by decade (bucket k: 1e(k-17) <= score < 1e(k-16); the first holds everything below 1e-16 and 0, the
// last everything from 1e-6 up), the lanes still refining at each correction (index: the correction, from 0), and the
// corrections undone because they did not lower a lane's score.
constexpr int REFINEMENT_SCORE_BUCKETS = 12;
constexpr int REFINEMENT_TRACKED_CORRECTIONS = 8;
extern long long gRefinementStartScoreBuckets[REFINEMENT_SCORE_BUCKETS];
extern long long gRefinementLanesAtCorrection[REFINEMENT_TRACKED_CORRECTIONS];
extern long long gRefinementUndone;

// Wall time of every symbolic analysis, per CALL SITE (source line), with a device sync on both sides so the
// reordering's host work and its device kernels are both inside. Only under OLF_GPU_PHASE_TIME (the syncs are
// not free); printed by the JNI_PHASE_TIME line as ANALYSIS_TIME.
// Same for factorizations (F) and solves (S); key = phase letter * 100000 + line.
extern std::map<int, std::pair<long long, double>> gAnalysisSites;   // key -> (count, ms)
struct AnalysisTimer {
    int line;
    bool on;
    std::chrono::steady_clock::time_point t0;
    explicit AnalysisTimer(int l, char ph = 'A') : line(ph * 100000 + l), on(traceOn(Trace::PHASE_TIME)) {
        if (on) {
            cudaDeviceSynchronize();
            t0 = std::chrono::steady_clock::now();
        }
    }
    ~AnalysisTimer() {
        if (on) {
            cudaDeviceSynchronize();
            auto& e = gAnalysisSites[line];
            e.first++;
            e.second += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
    }
};

#endif  // OLF_GPU_COMMON_STATS_CUH
