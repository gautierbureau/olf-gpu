// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Driver traces, see driver_traces.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../equations/scenario_overrides.cuh"
#include "driver_traces.cuh"

namespace olfgpu {

// OLF_GPU_STATE_DUMP=<maxIt>: for slot 0 and iterations 0..maxIt, EVERY row's mode, target, state
// and residual. Only usable on a small system, which is exactly where a reproducer lives — on a
// 16-row network it shows at a glance which variable ran away and which equation let it.
__global__ void stateDumpBatch(int n, const double* F, const int* mode, const double* target,
                               const double* x, int s, int it) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int i = 0; i < n; i++) {
        size_t k = (size_t) s * n + i;
        printf("[STATE] it=%d row=%d mode=%d target=%.9e x=%.9e F=%.9e\n",
               it, i, mode[k], target[k], x[k], F[k]);
    }
}

// OLF_GPU_CB_PROBE=<cb element index>: the pi row the device ACTUALLY evaluates for that closed
// branch in EVERY slot, after findTapOverride, alongside its disabled/active bits. The host knows
// what it MEANT to send (GpuBatchSideOpen.danglingCbRow); this says what the kernel read. A
// side-open that reduces to a pure shunt must show y = 0 with the equivalent on the live side's
// g/b — anything else and the override never reached the fill.
__global__ void cbRowProbe(int nS, int e, const double* cbIn, int nR1Ov,
                           const int* r1OvSlotElem, const double* r1OvVal,
                           int nElements, int elemOffset, const char* disabled, const char* active) {
    for (int s = 0; s < nS; s++) {
        const double* p = cbIn + (size_t) e * 8;
        double pOv[8];
        bool ov = findTapOverride(nR1Ov, r1OvSlotElem, r1OvVal, s, e, pOv);
        if (ov) {
            p = pOv;
        }
        printf("[CB_ROW] s=%d e=%d ov=%d dis=%d act=%d y=%.12e g1=%.12e g2=%.12e b1=%.12e b2=%.12e "
               "ksi=%.12e r1=%.12f a1=%.12f\n",
               s, e, ov ? 1 : 0, (int) disabled[(size_t) s * nElements + elemOffset + e], (int) active[s],
               p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
    }
}

// OLF_GPU_PSG_STEP=<group>: one pilot-self group's layout after EACH kernel of the reactive-limits pass
// (scenario 0), so a row that changes between two traces can be charged to the kernel that ran between them.
void psgStep(BatchSolverContext* ctx, int nS, int n, int outer, const char* tag) {
    // OLF_GPU_ROW_TRACE=<row,row,...>: those rows' mode and target for the slot OLF_GPU_LOOP_TRACE_CTG resolved,
    // after each kernel of the reactive-limits pass (same hook points as PSG_STEP, any scenario).
    static const char* rowEnv = std::getenv("OLF_GPU_ROW_TRACE");
    if (rowEnv != nullptr && ctx->loopTraceSlot >= 0 && ctx->loopTraceSlot < nS) {
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        std::string str(rowEnv);
        size_t pos = 0;
        while (pos <= str.size()) {
            size_t c = str.find(',', pos);
            if (c == std::string::npos) c = str.size();
            if (c > pos) {
                int r = std::atoi(str.substr(pos, c - pos).c_str());
                if (r >= 0 && r < n) {
                    int m = 0;
                    double t = 0;
                    size_t idx = (size_t) ctx->loopTraceSlot * n + r;
                    CUDA_CHECK(cudaMemcpy(&m, static_cast<const int*>(ctx->dMode.p) + idx, sizeof(int), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(&t, static_cast<const double*>(ctx->dTarget.p) + idx, sizeof(double), cudaMemcpyDeviceToHost));
                    std::fprintf(stderr, "ROW_TRACE outer=%d %-14s slot=%d row=%d mode=%d target=%.12f\n",
                                 outer, tag, ctx->loopTraceSlot, r, m, t);
                }
            }
            pos = c + 1;
        }
        // OLF_GPU_ROW_TRACE_GROUP=<g>: that pilot-self group's gate and its pilot entry's limit type, same slot
        static const char* grpEnv = std::getenv("OLF_GPU_ROW_TRACE_GROUP");
        if (grpEnv != nullptr && ctx->hasPilotSelf && ctx->dPsSkip.p != nullptr && ctx->dPsPilotEntry.p != nullptr) {
            int g = std::atoi(grpEnv);
            if (g >= 0 && g < ctx->st->nQr) {
                char sk = 0;
                int pe = -1;
                char lt = -1;
                CUDA_CHECK(cudaMemcpy(&sk, static_cast<const char*>(ctx->dPsSkip.p) + (size_t) ctx->loopTraceSlot * ctx->st->nQr + g, 1, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&pe, static_cast<const int*>(ctx->dPsPilotEntry.p) + g, sizeof(int), cudaMemcpyDeviceToHost));
                if (pe >= 0 && pe < ctx->nrl) {
                    CUDA_CHECK(cudaMemcpy(&lt, static_cast<const char*>(ctx->dRlLimitType.p) + (size_t) ctx->loopTraceSlot * ctx->nrl + pe, 1, cudaMemcpyDeviceToHost));
                }
                std::fprintf(stderr, "GROUP_TRACE outer=%d %-14s slot=%d g=%d psSkip=%d pilotEntry=%d lt=%d\n",
                             outer, tag, ctx->loopTraceSlot, g, (int) sk, pe, (int) lt);
            }
        }
    }
    static const char* env = std::getenv("OLF_GPU_PSG_STEP");
    if (env == nullptr || !ctx->hasPilotSelf || !ctx->dPsPilotEntry.p || ctx->st->nQr <= 0 || ctx->st->hQrPtr.empty()) return;
    int g = std::atoi(env);
    if (g < 0 || g >= ctx->st->nQr) return;
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    std::vector<int> pe(ctx->st->nQr);
    CUDA_CHECK(cudaMemcpy(pe.data(), ctx->dPsPilotEntry.p, pe.size() * sizeof(int), cudaMemcpyDeviceToHost));
    int e = pe[g];
    if (e < 0) return;
    int p0 = ctx->st->hQrPtr[g], cnt = ctx->st->hQrPtr[g + 1] - p0;
    std::vector<int> hm(n);
    std::vector<double> ht(n);
    CUDA_CHECK(cudaMemcpy(hm.data(), ctx->dMode.p, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ht.data(), ctx->dTarget.p, n * sizeof(double), cudaMemcpyDeviceToHost));
    char lt = 0, veq = 0;
    CUDA_CHECK(cudaMemcpy(&lt, static_cast<const char*>(ctx->dRlLimitType.p) + e, 1, cudaMemcpyDeviceToHost));
    if (ctx->dQrVeqOff.p) CUDA_CHECK(cudaMemcpy(&veq, static_cast<const char*>(ctx->dQrVeqOff.p) + g, 1, cudaMemcpyDeviceToHost));
    std::string cands;
    for (int c = 1; c < cnt; c++) {
        int r = ctx->st->hQrRows[p0 + c];
        char b[48];
        std::snprintf(b, sizeof b, " %d:m%d/t%.4f", r, hm[r], ht[r]);
        cands += b;
    }
    int prow = ctx->st->hQrRows[p0];
    std::fprintf(stderr, "PSG_STEP outer=%d %-14s g=%d entry=%d lt=%d pilotRow=%d mode=%d target=%.9f veqOff=%d cands=[%s ]\n",
                 outer, tag, g, e, (int) lt, prow, hm[prow], ht[prow], (int) veq, cands.c_str());
}

// OLF_GPU_STEP_TRACE=1: the MAX_VOLTAGE_CHANGE clamp factor actually applied, per (iteration, slot).
// The decisive number for "why does this scenario need more than its iteration budget": a factor near
// 1 means Newton is taking its full step and the slowness is elsewhere, while a factor of a few
// percent means the step is throttled and 30 iterations simply cannot cover the distance. Nothing
// else printed it — ||dx||inf from LIN_RESID is NOT a proxy, because the largest component is
// typically a DUMMY_P/DUMMY_Q flow correction and those carry varType 2, which the clamp IGNORES
// (only BUS_V=1 and BUS_PHI=0 enter the maxima). See tasks #67.
void traceStepFactor(BatchSolverContext* ctx, int nS, int it, const std::vector<char>& active) {
    static const bool on = std::getenv("OLF_GPU_STEP_TRACE") != nullptr;
    if (!on || ctx->dStepFactor.p == nullptr) {
        return;
    }
    std::vector<double> hf(nS);
    CUDA_CHECK(cudaMemcpy(hf.data(), ctx->dStepFactor.p, (size_t) nS * sizeof(double), cudaMemcpyDeviceToHost));
    // WHICH variable binds the clamp, and by how much. f alone says the step is throttled; the
    // argmax says WHERE the wild direction is, which is the only way to tell an ill-conditioned
    // direction from a legitimately large move.
    int n = ctx->n;
    std::vector<double> hdx((size_t) n * nS);
    std::vector<int> hvt(n);
    CUDA_CHECK(cudaMemcpy(hdx.data(), ctx->dDx.p, (size_t) n * nS * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvt.data(), ctx->dVarType.p, (size_t) n * sizeof(int), cudaMemcpyDeviceToHost));
    for (int s = 0; s < nS; s++) {
        if (!active.empty() && !active[s]) {
            continue;
        }
        double mdv = 0;
        double mdphi = 0;
        int argv = -1;
        int argphi = -1;
        for (int i = 0; i < n; i++) {
            double av = std::fabs(hdx[(size_t) s * n + i]);
            if (hvt[i] == 1 && av > mdv) { mdv = av; argv = i; }
            if (hvt[i] == 0 && av > mdphi) { mdphi = av; argphi = i; }
        }
        std::fprintf(stderr, "STEP_FACTOR it=%d slot=%d f=%.9g maxDv=%.6g atVrow=%d maxDphi=%.6g atPhirow=%d\n",
                     it, s, hf[s], mdv, argv, mdphi, argphi);
    }
}

}  // namespace olfgpu
