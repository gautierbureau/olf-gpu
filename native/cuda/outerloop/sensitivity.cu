// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See sensitivity.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/stats.cuh"
#include "../common/codes/gpu_linear_solve_mode.cuh"
#include "../equations/branch_fill.cuh"
#include "../equations/bus_injection_fill.cuh"
#include "../equations/distribution_fill.cuh"
#include "../equations/fill_launch.cuh"
#include "../equations/zero_impedance_fill.cuh"
#include "../solver/batched_lu_solve.cuh"
#include "../solver/driver_steps.cuh"
#include "../fixedmatrix/zero_diagonal_repair.cuh"
#include "pilot_voltage.cuh"
#include "sensitivity.cuh"

namespace olfgpu {

// Distributed-slack pass: thread = (participating bus, scenario). The slack mismatch is
// qBuf[slackPhiRow] (= Σ branch p at the slack) − slackTargetP. If it exceeds the tolerance, each
// participating bus takes factor·mismatch into its BUS_TARGET_P target (factor normalized to sum 1, so
// the whole mismatch is absorbed), mirroring ActivePowerDistribution (one-shot, no Pmin/Pmax saturation
// — exact vs OLF when no generator saturates). A distribution marks its scenario for another inner solve.
// Freeze, on the FIRST distribution pass of a solve, the active-power share the slack/reference bus must
// RETAIN when its own generators participate: retain = (1 − Σfactor)·mismatch0. Σfactor over the emitted
// (non-slack) buses is < 1 exactly when the slack participates (its factor sits in the normalization
// denominator but emits no row); 1 − Σfactor is the slack's participation fraction ρ. Without this the
// geometric outer loop would drive the slack mismatch to 0 and leak the slack's whole share to the
// non-slack buses. ρ = 0 (non-participating slack) ⇒ retain = 0 ⇒ the slack keeps nothing extra.
// The slack/reference bus retains ρ = (1 − Σfactor) of the TOTAL active imbalance currently being shared,
// where total = the slack's residual mismatch (qBuf at slackPhiRow − slackTargetP) PLUS what the non-slack
// participating buses have already absorbed (Σ of their cumulative target deltas). Re-evaluated EVERY outer
// pass: freezing ρ·mismatch0 at the first pass ignores the AC loss change during redistribution, so the
// slack would keep ρ·mismatch0 and the non-slack side would over-absorb by ρ·Δlosses; tracking the current
// total instead makes the non-slack buses absorb exactly (1 − ρ)·total_final, matching OLF's per-pass
// distribution of the full mismatch. ρ = 0 (non-participating slack, incl. all batched gen-loss scenarios)
// ⇒ retain = 0, so this is a no-op there.
__global__ void setUnitRhsBatch(int n, int nScenarios, int unitRow, double* b) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * nScenarios) return;
    b[t] = (t % n == unitRow) ? 1.0 : 0.0;
}

// One unit-right-hand-side sensitivity solve on every scenario's CURRENT Jacobian (dVals, filled for all
// scenarios by the caller; dB = the right-hand sides, dDx = the solutions), for the in-kernel outer loops.
// The UBATCH mode (3) solves the whole batch in one call on the factors they hold - the analysis once per
// context, the factorization with the first controller's solve. The robust mode (6) holds ONE matrix object
// repointed per scenario and keeps no factors from one scenario to the next, so it factors and solves scenario
// by scenario - a refactorization per (scenario, controller), affordable at the robust retry's chunk sizes and
// the price of the loops running there at all: they used to report themselves STABLE in mode 6, so a scenario
// retried in it (or a base load flow that fell into it) with a transformer, shunt, flow or secondary voltage
// control out of its deadband got a wrong answer and nothing flagged it.
void sensitivitySolve(BatchSolverContext* ctx, DriverPass& drv, bool& first) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    if (ctx->mode == GpuLinearSolveMode::ROBUST) {
        for (int s2 = 0; s2 < nS; s2++) {
            CUDSS_CHECK(cudssMatrixSetValues(ctx->solver.systemMatrix, ctx->dVals.as<double>() + (size_t) s2 * ctx->nnz));
            CUDSS_CHECK(cudssMatrixSetValues(ctx->solver.rightHandSideMatrix, ctx->dB.as<double>() + (size_t) s2 * n));
            CUDSS_CHECK(cudssMatrixSetValues(ctx->solver.solutionMatrix, ctx->dDx.as<double>() + (size_t) s2 * n));
            if (!ctx->analyzed) {
                gAnalysesMain++;
                { AnalysisTimer at_(__LINE__); CUDSS_CHECK(cudssExecute(ctx->solver.handle, CUDSS_PHASE_ANALYSIS, ctx->solver.config, ctx->solver.data, ctx->solver.systemMatrix, ctx->solver.solutionMatrix, ctx->solver.rightHandSideMatrix)); }
                ctx->analyzed = true;
            }
            { AnalysisTimer at_(__LINE__, 'G'); CUDSS_CHECK(cudssExecute(ctx->solver.handle, CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_SOLVE,
                    ctx->solver.config, ctx->solver.data, ctx->solver.systemMatrix, ctx->solver.solutionMatrix, ctx->solver.rightHandSideMatrix)); }
        }
        first = false;
        return;
    }
    if (ctx->mode == GpuLinearSolveMode::BATCHED_LU) {
        batchedLuSensitivitySolve(ctx, nS, first);
        first = false;
        return;
    }
    if (!ctx->analyzed) {
        gAnalysesMain++;
        { AnalysisTimer at_(__LINE__); CUDSS_CHECK(cudssExecute(ctx->solver.handle, CUDSS_PHASE_ANALYSIS, ctx->solver.config, ctx->solver.data, ctx->uniformBatchSystemMatrix, ctx->uniformBatchSolutionMatrix, ctx->uniformBatchRightHandSideMatrix)); }
        ctx->analyzed = true;
    }
    int phase = CUDSS_PHASE_SOLVE | (first ? CUDSS_PHASE_FACTORIZATION : 0);
    first = false;
    { AnalysisTimer at_(__LINE__, ((phase) & (CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_REFACTORIZATION)) ? 'G' : ((phase) & CUDSS_PHASE_ANALYSIS) ? 'A' : 'S'); CUDSS_CHECK(cudssExecute(ctx->solver.handle, phase, ctx->solver.config, ctx->solver.data, ctx->uniformBatchSystemMatrix, ctx->uniformBatchSolutionMatrix, ctx->uniformBatchRightHandSideMatrix)); }
}

}  // namespace olfgpu
