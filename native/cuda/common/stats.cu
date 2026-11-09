// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The single definition of every run-wide counter declared in stats.cuh.
#include "stats.cuh"
#include "codes/gpu_outer_loop_kind.cuh"

long long gNewtonLoops = 0;
long long gRestarts = 0;

double gNewtonLoopMilliseconds = 0, gOuterLoopCheckMilliseconds[GpuOuterLoopKind::COUNT] = {0};
long long gNewtonLoopWidthIterations = 0;
long long gOuterLoopFirstSolveScenarios = 0, gOuterLoopReSolveScenarios[GpuOuterLoopKind::COUNT] = {0};
long long gOuterLoopFirstSolveLaneIterations = 0, gOuterLoopReSolveLaneIterations[GpuOuterLoopKind::COUNT] = {0};
long long gOuterLoopCheckRounds[GpuOuterLoopKind::COUNT] = {0};

long long gNewtonIters = 0;
int gNewtonLoopIteration = 0;
long long gActiveIterSum = 0;
long long gWidthIterSum = 0;
long long gCompactIters = 0;
long long gBatchedLuFactorizations = 0;
long long gBatchedLuPivotedLanes = 0;
long long gBatchedLuPivotDelayOrders = 0;
long long gBatchedLuSplitLaunches = 0;
long long gBatchedLuSupernodalSplitLaunches = 0;
long long gFullFactorizations = 0;
long long gRobustFactorizations = 0;
long long gSensitivityFactorizations = 0;
long long gSensitivitySolves = 0;
long long gRefinementFactorizations = 0;
long long gCheckRounds = 0;
long long gOuterLoopPassHistogramCount[16] = {0}, gOuterLoopPassHistogramFirstSolveIterations[16] = {0}, gOuterLoopPassHistogramTotalIterations[16] = {0};
long long gOuterLoopNotConverged = 0, gOuterLoopNotConvergedIterations = 0;
long long gAnalysesMain = 0;
long long gAnalysesCompact = 0;
long gIterativeRefinementExtraSolves = 0;
long gIterativeRefinementChecks = 0;

std::map<int, std::pair<long long, double>> gAnalysisSites;
