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

// OLF_GPU_PHASE_TIME accounting: how much of a solveBatchContext call is device outer-loop passes
// vs Newton iterations. A pass that re-solves is the expensive one; a pass that only re-checks is not.
extern long long gOuterPasses;

// OLF_GPU_PHASE_TIME: where the outer-loop driver's time goes. Index = the GpuOuterLoopKind that TRIGGERED a
// re-solve (the loop whose check changed something), or the loop being CHECKED; the first solve of a call is apart.
extern double gOuterLoopFirstSolveMilliseconds, gOuterLoopReSolveMilliseconds[GpuOuterLoopKind::COUNT], gOuterLoopCheckMilliseconds[GpuOuterLoopKind::COUNT];
extern long long gOuterLoopFirstSolveScenarios, gOuterLoopReSolveScenarios[GpuOuterLoopKind::COUNT], gOuterLoopReSolvePasses[GpuOuterLoopKind::COUNT];
extern long long gOuterLoopFirstSolveLaneIterations, gOuterLoopReSolveLaneIterations[GpuOuterLoopKind::COUNT], gOuterLoopFirstSolveWidthIterations, gOuterLoopReSolveWidthIterations[GpuOuterLoopKind::COUNT];
extern long long gOuterLoopCheckPasses[GpuOuterLoopKind::COUNT], gOuterLoopFirstSolvePasses;

inline double nowMilliseconds() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

extern long long gNewtonIters;
extern int gLastPassIters;                                   // inner-Newton steps taken by the pass just finished
extern long long gSolvePasses;
extern long long gActiveIterSum;
extern long long gWidthIterSum;
extern long long gCompactIters;
extern long long gBatchedLuFactorizations;                   // factorizations of our batched LU
extern long long gBatchedLuPivotedLanes;                     // ...and the lanes of them its pivoting engine factorized
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
