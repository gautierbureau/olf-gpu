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

__global__ void gatherSensiBatch(int n, int nScenarios, int k, int m, int ki, const int* observeCols,
                                 const double* dx, double* out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;                     // t = s*m + mi
    if (t >= nScenarios * m) return;
    int sIdx = t / m;
    int mi = t % m;
    out[(size_t) sIdx * k * m + (size_t) ki * m + mi] = dx[(size_t) sIdx * n + observeCols[mi]];
}

void sensitivitySolve(BatchSolverContext* ctx, DriverPass& drv, bool& first) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    if (ctx->mode == 6) {
        for (int s2 = 0; s2 < nS; s2++) {
            CUDSS_CHECK(cudssMatrixSetValues(ctx->ds.matA, ctx->dVals.as<double>() + (size_t) s2 * ctx->nnz));
            CUDSS_CHECK(cudssMatrixSetValues(ctx->ds.matB, ctx->dB.as<double>() + (size_t) s2 * n));
            CUDSS_CHECK(cudssMatrixSetValues(ctx->ds.matX, ctx->dDx.as<double>() + (size_t) s2 * n));
            if (!ctx->analyzed) {
                gAnalysesMain++;
                { AnalysisTimer at_(__LINE__); CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_ANALYSIS, ctx->ds.config, ctx->ds.data, ctx->ds.matA, ctx->ds.matX, ctx->ds.matB)); }
                ctx->analyzed = true;
            }
            { AnalysisTimer at_(__LINE__, 'G'); CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_SOLVE,
                    ctx->ds.config, ctx->ds.data, ctx->ds.matA, ctx->ds.matX, ctx->ds.matB)); }
        }
        first = false;
        return;
    }
    if (!ctx->analyzed) {
        gAnalysesMain++;
        { AnalysisTimer at_(__LINE__); CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_ANALYSIS, ctx->ds.config, ctx->ds.data, ctx->matUb, ctx->matUbX, ctx->matUbB)); }
        ctx->analyzed = true;
    }
    int phase = CUDSS_PHASE_SOLVE | (first ? CUDSS_PHASE_FACTORIZATION : 0);
    first = false;
    { AnalysisTimer at_(__LINE__, ((phase) & (CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_REFACTORIZATION)) ? 'G' : ((phase) & CUDSS_PHASE_ANALYSIS) ? 'A' : 'S'); CUDSS_CHECK(cudssExecute(ctx->ds.handle, phase, ctx->ds.config, ctx->ds.data, ctx->matUb, ctx->matUbX, ctx->matUbB)); }
}

}  // namespace olfgpu
