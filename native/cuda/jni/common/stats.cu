// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The single definition of every run-wide counter declared in stats.cuh.
#include "stats.cuh"

long long gOuterPasses = 0;

double gOlFirstSolveMs = 0, gOlReSolveMs[OL_KINDS] = {0}, gOlCheckMs[OL_KINDS] = {0};
long long gOlFirstSolveScen = 0, gOlReSolveScen[OL_KINDS] = {0}, gOlReSolvePasses[OL_KINDS] = {0};
long long gOlFirstLaneIters = 0, gOlReSolveLaneIters[OL_KINDS] = {0}, gOlFirstWidthIters = 0, gOlReSolveWidthIters[OL_KINDS] = {0};
long long gOlCheckPasses[OL_KINDS] = {0}, gOlFirstPasses = 0;

long long gRlCheckScen[8] = {0}, gRlStableScen = 0, gRlEntryCodes[4] = {0};

long long gNewtonIters = 0;
int gLastPassIters = 0;
long long gSolvePasses = 0;
long long gActiveIterSum = 0;
long long gWidthIterSum = 0;
long long gCompactIters = 0;
long long gOlHistCount[16] = {0}, gOlHistFirst[16] = {0}, gOlHistTotal[16] = {0};
long long gOlNotConverged = 0, gOlNotConvergedIters = 0;
long long gAnalysesMain = 0;
long long gAnalysesCompact = 0;
long gIrExtraSteps = 0;
long gIrChecks = 0;

std::map<int, std::pair<long long, double>> gAnalysisSites;
