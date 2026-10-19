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
#include "../common/launch.cuh"
#include "../common/stats.cuh"
#include "../common/codes/gpu_ac_solver_status.cuh"
#include "../equations/branch_fill.cuh"
#include "../equations/bus_injection_fill.cuh"
#include "../equations/distribution_fill.cuh"
#include "../equations/fill_launch.cuh"
#include "../equations/zero_impedance_fill.cuh"
#include "../outerloop/forced_target_q.cuh"
#include "../outerloop/reactive_capability_curves.cuh"
#include "../outerloop/reactive_limits.cuh"
#include "../outerloop/remote_voltage_control.cuh"
#include "../solver/convergence.cuh"
#include "../solver/driver_steps.cuh"
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

// OLF_GPU_BATCH_TRACE / OLF_GPU_ROW_PROBE: the pass's per-scenario state before the loop check.
void traceBatchPass(BatchSolverContext* ctx, DriverPass& drv, int nLoops, bool solvePass) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    const int& outer = drv.outer;
    int& loopIdx = drv.loopIdx;
    int (&loopOrder)[MAX_DEVICE_LOOPS] = drv.loopOrder;
    if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
        const char* kindName = "?";
        switch (loopOrder[loopIdx]) {
            case LOOP_DS:           kindName = "DistributedSlack"; break;
            case LOOP_HVDC_FREEZE:  kindName = "FreezingHvdcACEmulation"; break;
            case LOOP_HVDC_LIMITS:  kindName = "AcHvdcAcEmulationLimits"; break;
            case LOOP_HOST_SVC:     kindName = "HOST:SecondaryVoltageControl"; break;
            case LOOP_MONITORING:   kindName = "MonitoringVoltage"; break;
            case LOOP_RL:           kindName = "ReactiveLimits"; break;
            case LOOP_HOST_PHASE:   kindName = "HOST:PhaseControl"; break;
            case LOOP_HOST_TVC:     kindName = "HOST:TransformerVoltageControl"; break;
            case LOOP_HOST_TRPC:    kindName = "HOST:TransformerReactivePowerControl"; break;
            case LOOP_HOST_SHUNT:   kindName = "HOST:ShuntVoltageControl"; break;
            case LOOP_TVC_KERNEL:   kindName = "TransformerVoltageControl"; break;
            case LOOP_SHUNT_KERNEL: kindName = "ShuntVoltageControl"; break;
            case LOOP_PHASE_KERNEL: kindName = ctx->nCph > 0 ? "PhaseControl" : "IncrementalPhaseControl"; break;
            case LOOP_TRPC_KERNEL:  kindName = "TransformerReactivePowerControl"; break;
            case LOOP_TVD_KERNEL:   kindName = ctx->tvdMode == 1 ? "SimpleTransformerVoltageControl" : "TransformerVoltageControl"; break;
            case LOOP_SVD_KERNEL:   kindName = "ShuntVoltageControl"; break;
            case LOOP_SVC_KERNEL:   kindName = "SecondaryVoltageControl"; break;
        }
        int nSw = 0;
        {
            std::vector<char> sw(nS);
            CUDA_CHECK(cudaMemcpy(sw.data(), ctx->dRlSwitched.p, nS, cudaMemcpyDeviceToHost));
            for (char c : sw) nSw += c ? 1 : 0;
        }
        int nPinned = 0;
        if (ctx->nrl > 0) {
            std::vector<char> lt(static_cast<size_t>(ctx->nrl) * nS);
            CUDA_CHECK(cudaMemcpy(lt.data(), ctx->dRlLimitType.p, lt.size(), cudaMemcpyDeviceToHost));
            for (char c : lt) nPinned += c ? 1 : 0;
        }
        std::fprintf(stderr, "BATCH_TRACE pass done: outer=%d loop=%d/%d kind=%s solved=%d changedScenarios=%d iters=%d pinned=%d",
                     outer, loopIdx, nLoops, kindName, solvePass ? 1 : 0, nSw, gLastPassIters, nPinned);
        {
            // OLF_GPU_ROW_PROBE=<row>: that row's slot-0 TARGET as each pass leaves it — the inner
            // probe only sees iterations, and a target a loop phase mutates between solves (the
            // HVDC freeze/unfreeze, a DS anchor reset) is invisible there.
            static const char* passProbeEnv = std::getenv("OLF_GPU_ROW_PROBE");
            static int passProbe = passProbeEnv != nullptr ? std::atoi(passProbeEnv) : -1;
            if (passProbe >= 0 && passProbe < n) {
                double tv = 0;
                double fv = 0;
                CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                CUDA_CHECK(cudaMemcpy(&tv, ctx->dTarget.as<double>() + passProbe, sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&fv, ctx->dF.as<double>() + passProbe, sizeof(double), cudaMemcpyDeviceToHost));
                std::fprintf(stderr, " probeRow=%d target=%.12e F=%.12e", passProbe, tv, fv);
            }
        }
        std::fprintf(stderr, "\n");
    }
}

// OLF_GPU_DS_DEBUG: is scenario 0 converged when the distributed-slack check reads its mismatch?
void traceDsConvergence(BatchSolverContext* ctx, DriverPass& drv, bool inDsPhase) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    const int& outer = drv.outer;
    std::vector<char>& rlActive = drv.rlActive;
    if (inDsPhase && getenv("OLF_GPU_DS_DEBUG") && nS > 0) {
        // Is scenario 0 actually CONVERGED at the moment the DS loop reads its mismatch? OLF checks
        // its slack mismatch on a converged state; if ours is only part-way there, the mismatch it
        // measures is not the same quantity, and a threshold at the same scale flips.
        std::vector<double> f0(n);
        CUDA_CHECK(cudaMemcpy(f0.data(), ctx->dF.as<double>(), n * sizeof(double), cudaMemcpyDeviceToHost));
        double ssq = 0.0;
        double inf = 0.0;
        for (int i = 0; i < n; i++) {
            ssq += f0[i] * f0[i];
            inf = fmax(inf, fabs(f0[i]));
        }
        double slackM = std::numeric_limits<double>::quiet_NaN();
        double retain0 = std::numeric_limits<double>::quiet_NaN();
        if (ctx->dsSlackPhiRow >= 0) {
            std::vector<double> q0(n);
            CUDA_CHECK(cudaMemcpy(q0.data(), ctx->dQbuf.as<double>(), n * sizeof(double), cudaMemcpyDeviceToHost));
            slackM = q0[ctx->dsSlackPhiRow] - ctx->dsSlackTargetP;
            if (ctx->dDsRetain.p) CUDA_CHECK(cudaMemcpy(&retain0, ctx->dDsRetain.p, sizeof(double), cudaMemcpyDeviceToHost));
        }
        std::fprintf(stderr, "[DS-CONV] outer=%d innerNewtonIters=%d ||F||2=%.9e ||F||inf=%.9e slackMismatchFromAnchor=%.9f retain=%.9f rlActive=%d\n",
                     outer, gLastPassIters, std::sqrt(ssq), inf, slackM, retain0, (int) rlActive[0]);
    }
}

// OLF_GPU_RL_REASONS: why each reactive-limits check was unstable.
void traceRlReasons(BatchSolverContext* ctx, DriverPass& drv, const std::vector<char>& rlSwitched) {
    const int& nS = drv.nS;
    int& loopIdx = drv.loopIdx;
    int (&loopOrder)[MAX_DEVICE_LOOPS] = drv.loopOrder;
    std::vector<char>& rlActive = drv.rlActive;
    if (rlReasonsOn() && loopOrder[loopIdx] == LOOP_RL && ctx->nrl > 0 && ctx->dRlPassSwitched.p) {
        std::vector<char> ps(static_cast<size_t>(ctx->nrl) * nS);
        CUDA_CHECK(cudaMemcpy(ps.data(), ctx->dRlPassSwitched.p, ps.size(), cudaMemcpyDeviceToHost));
        for (int s = 0; s < nS; s++) {
            if (!rlActive[s]) continue;
            int mask = 0;
            for (int i = 0; i < ctx->nrl; i++) {
                char c = ps[static_cast<size_t>(s) * ctx->nrl + i];
                if (c >= 1 && c <= 3) {
                    mask |= 1 << (c - 1);
                    gRlEntryCodes[static_cast<int>(c)]++;
                }
            }
            if (rlSwitched[s]) {
                gRlCheckScen[mask]++;
            } else {
                gRlStableScen++;
            }
        }
    }
}

// OLF_GPU_RL_AUDIT / OLF_GPU_SW_TRACE: every reactive-limit switch of the pass, per scenario.
void traceRlSwitches(BatchSolverContext* ctx, DriverPass& drv, std::vector<std::vector<std::string>>& auditLog, bool rlAuditOn, int swTraceSlot) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    const int& outer = drv.outer;
    int& loopIdx = drv.loopIdx;
    int (&loopOrder)[MAX_DEVICE_LOOPS] = drv.loopOrder;
    std::vector<char>& rlActive = drv.rlActive;
    if (loopOrder[loopIdx] == LOOP_RL && (rlAuditOn || (swTraceSlot >= 0 && swTraceSlot < nS)) && ctx->nrl > 0) {
        std::vector<char> ps(static_cast<size_t>(ctx->nrl) * nS), lt2(static_cast<size_t>(ctx->nrl) * nS);
        std::vector<int> vr2(ctx->nrl);
        std::vector<double> qb2(static_cast<size_t>(n) * nS), lq2(ctx->nrl), x2(static_cast<size_t>(n) * nS);
        CUDA_CHECK(cudaMemcpy(ps.data(), ctx->dRlPassSwitched.p, ps.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lt2.data(), ctx->dRlLimitType.p, lt2.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vr2.data(), ctx->dRlVrow.p, vr2.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(qb2.data(), ctx->dQbuf.p, qb2.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lq2.data(), ctx->dRlLoadQ.p, lq2.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(x2.data(), ctx->dX.p, x2.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<double> mn2(static_cast<size_t>(ctx->nrl) * (ctx->ncv > 0 ? nS : 1));
        std::vector<double> mx2(mn2.size());
        CUDA_CHECK(cudaMemcpy(mn2.data(), ctx->ncv > 0 ? ctx->dRlMinQDyn.p : ctx->dRlMinQ.p, mn2.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mx2.data(), ctx->ncv > 0 ? ctx->dRlMaxQDyn.p : ctx->dRlMaxQ.p, mx2.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<int> cv2(ctx->nrl, -1);
        std::vector<double> vt2(ctx->nrl, 0.0);
        if (ctx->dRlCmpVrow.p) {
            CUDA_CHECK(cudaMemcpy(cv2.data(), ctx->dRlCmpVrow.p, cv2.size() * sizeof(int), cudaMemcpyDeviceToHost));
        }
        if (ctx->dRlVtarget.p) {
            CUDA_CHECK(cudaMemcpy(vt2.data(), ctx->dRlVtarget.p, vt2.size() * sizeof(double), cudaMemcpyDeviceToHost));
        }
        for (int s2 = 0; s2 < nS; s2++) {
            bool toStderr = (s2 == swTraceSlot);
            if (!toStderr && !(rlAuditOn && rlActive[s2])) continue;
            int nsw = 0;
            std::string rec;
            for (int i2 = 0; i2 < ctx->nrl; i2++) {
                size_t st2 = (size_t) s2 * ctx->nrl + i2;
                // OLF_GPU_SW_TRACE_ENTRIES=<e,...>: those entries print EVERY RL pass for the traced slot, switched
                // or not - a decision the device did NOT take is the one to compare against the CPU's.
                static const std::vector<int> swEntries = [] {
                    std::vector<int> v;
                    const char* e = std::getenv("OLF_GPU_SW_TRACE_ENTRIES");
                    if (e == nullptr) return v;
                    std::string str(e);
                    size_t pos = 0;
                    while (pos <= str.size()) {
                        size_t c = str.find(',', pos);
                        if (c == std::string::npos) c = str.size();
                        if (c > pos) v.push_back(std::atoi(str.substr(pos, c - pos).c_str()));
                        pos = c + 1;
                    }
                    return v;
                }();
                static const bool swAll = std::getenv("OLF_GPU_SW_TRACE_ENTRIES") != nullptr
                                          && std::string(std::getenv("OLF_GPU_SW_TRACE_ENTRIES")) == "all";
                bool forced = toStderr && (swAll || std::find(swEntries.begin(), swEntries.end(), i2) != swEntries.end());
                if (!ps[st2] && !forced) continue;
                size_t li2 = ctx->ncv > 0 ? st2 : (size_t) i2;
                double cmpV2 = (cv2[i2] >= 0) ? x2[(size_t) s2 * n + cv2[i2]] : 0.0;
                double qq = qb2[(size_t) s2 * n + vr2[i2]] + lq2[i2];
                // the q the decision READ: at rlQrow when the entry's balance was relocated (a coupler-tree
                // child reads its parent's DUMMY_Q row), not at its V row - printing the V-row value logged a
                // relocated child as q=0 while the kernel pinned it on its real q
                static const std::vector<int> qr2 = [&] {
                    std::vector<int> v(ctx->nrl, -1);
                    if (ctx->dRlQrow.p) CUDA_CHECK(cudaMemcpy(v.data(), ctx->dRlQrow.p, v.size() * sizeof(int), cudaMemcpyDeviceToHost));
                    return v;
                }();
                int qrow2 = qr2[i2] >= 0 ? qr2[i2] : vr2[i2];
                double qRead = qb2[(size_t) s2 * n + qrow2] + lq2[i2];
                nsw++;
                char buf[400];
                std::snprintf(buf, sizeof buf, "SW_PASS s=%d outer=%d entry=%d vrow=%d act=%d lt=%d q=%.15e qRow=%d qRead=%.15e min=%.15e max=%.15e v=%.15e cmpV=%.15e vt=%.15e margin=%.3e",
                             s2, outer, i2, vr2[i2], (int) ps[st2], (int) lt2[st2],
                             qq, qrow2, qRead, mn2[li2], mx2[li2],
                             x2[(size_t) s2 * n + vr2[i2]], cmpV2, vt2[i2], cmpV2 - vt2[i2]);
                if (toStderr) {
                    // the switch-back's other gates, read back for the traced entry: the row slope (a slope-SVC
                    // row never switches back) and the switch count against the cap
                    double slope = 0.0;
                    CUDA_CHECK(cudaMemcpy(&slope, ctx->dRowSlope.as<double>() + vr2[i2], sizeof(double), cudaMemcpyDeviceToHost));
                    char sc = 0;
                    CUDA_CHECK(cudaMemcpy(&sc, ctx->dRlSwitchCount.as<char>() + st2, 1, cudaMemcpyDeviceToHost));
                    char lov = 0;
                    double lvt = 0.0;
                    if (ctx->nRlLocal > 0) {
                        CUDA_CHECK(cudaMemcpy(&lov, ctx->dRlLocalOv.as<char>() + st2, 1, cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaMemcpy(&lvt, ctx->dRlLocalVt.as<double>() + st2, sizeof(double), cudaMemcpyDeviceToHost));
                    }
                    char mon = 0;
                    CUDA_CHECK(cudaMemcpy(&mon, ctx->dRlMonitored.as<char>() + i2, 1, cudaMemcpyDeviceToHost));
                    char isl = 0;
                    if (ctx->nRlIslanded > 0) {
                        CUDA_CHECK(cudaMemcpy(&isl, ctx->dRlIslanded.as<char>() + st2, 1, cudaMemcpyDeviceToHost));
                    }
                    char act = 0;
                    CUDA_CHECK(cudaMemcpy(&act, ctx->dRlActive.as<char>() + s2, 1, cudaMemcpyDeviceToHost));
                    std::fprintf(stderr, "%s slope=%.6e swCount=%d localOv=%d localVt=%.12f monitored=%d islanded=%d rlActive=%d\n", buf, slope, (int) sc,
                                 (int) lov, lvt, (int) mon, (int) isl, (int) act);
                }
                if (rlAuditOn && nsw <= 12) {
                    rec += buf;
                    rec += "\n";
                }
            }
            if (rlAuditOn) {
                char hdr[160];
                std::snprintf(hdr, sizeof hdr, "AUDIT_PASS s=%d outer=%d loop=%d rlActive=%d nsw=%d",
                              s2, outer, loopOrder[loopIdx], (int) rlActive[s2], nsw);
                auditLog[s2].push_back(std::string(hdr) + "\n" + rec);
            }
        }
    }
}

// OLF_GPU_BATCH_TRACE / OLF_GPU_RL_TRACE_VROW: the reactive-limit entries' state after the check.
void traceRlState(BatchSolverContext* ctx, DriverPass& drv, bool inDsPhase, double reactiveLimitTol, std::vector<char>& rlSwitched, int rlTraceVrow) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    const int& outer = drv.outer;
    if (!inDsPhase && (ctx->nRlRemote > 0 || nS == 1 || rlTraceVrow >= 0) && ctx->nrl > 0
            && (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr || rlTraceVrow >= 0)) {   // RL trace
        std::vector<char> lt(static_cast<size_t>(ctx->nrl) * nS);
        std::vector<char> rem(ctx->nrl);
        std::vector<int> vr(ctx->nrl);
        std::vector<double> qb(static_cast<size_t>(n) * nS);
        std::vector<double> lq(ctx->nrl);
        CUDA_CHECK(cudaMemcpy(lt.data(), ctx->dRlLimitType.p, lt.size(), cudaMemcpyDeviceToHost));
        if (ctx->nRlRemote > 0) {
        CUDA_CHECK(cudaMemcpy(rem.data(), ctx->dRlRemote.p, rem.size(), cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaMemcpy(vr.data(), ctx->dRlVrow.p, vr.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(qb.data(), ctx->dQbuf.p, qb.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lq.data(), ctx->dRlLoadQ.p, lq.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<int> hMode(static_cast<size_t>(n) * nS);
        CUDA_CHECK(cudaMemcpy(hMode.data(), ctx->dMode.p, hMode.size() * sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<double> hMaxQ(static_cast<size_t>(ctx->nrl) * (ctx->ncv > 0 ? nS : 1));
        CUDA_CHECK(cudaMemcpy(hMaxQ.data(), ctx->ncv > 0 ? ctx->dRlMaxQDyn.p : ctx->dRlMaxQ.p, hMaxQ.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<char> hRlAct(nS);
        CUDA_CHECK(cudaMemcpy(hRlAct.data(), ctx->dRlActive.p, nS, cudaMemcpyDeviceToHost));
        // The STATE the pin decision is taken at. Without it, a q that disagrees with OLF cannot be
        // told apart from a q evaluated at a different (or not yet converged) operating point - the
        // same quantity at a different V is not evidence of a wrong formula.
        std::vector<double> hXrl(static_cast<size_t>(n) * nS);
        CUDA_CHECK(cudaMemcpy(hXrl.data(), ctx->dX.p, hXrl.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<double> hMinQ(hMaxQ.size());
        CUDA_CHECK(cudaMemcpy(hMinQ.data(), ctx->ncv > 0 ? ctx->dRlMinQDyn.p : ctx->dRlMinQ.p, hMinQ.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<double> hTgt(static_cast<size_t>(n) * nS);
        CUDA_CHECK(cudaMemcpy(hTgt.data(), ctx->dTarget.p, hTgt.size() * sizeof(double), cudaMemcpyDeviceToHost));
        // The switch-back COMPARISON itself: the controlled (pilot) bus's voltage against the group
        // target, and the MARGIN between them. A pin that flips between runs is only "the last bits"
        // if that margin is at the noise floor - print it rather than assume it.
        std::vector<int> cmpVr(ctx->nrl);
        std::vector<double> vTgt(ctx->nrl);
        if (ctx->dRlCmpVrow.p) {
            CUDA_CHECK(cudaMemcpy(cmpVr.data(), ctx->dRlCmpVrow.p, cmpVr.size() * sizeof(int), cudaMemcpyDeviceToHost));
        }
        if (ctx->dRlVtarget.p) {
            CUDA_CHECK(cudaMemcpy(vTgt.data(), ctx->dRlVtarget.p, vTgt.size() * sizeof(double), cudaMemcpyDeviceToHost));
        }
        for (int s2 = 0; s2 < nS; s2++) {
            for (int i2 = 0; i2 < ctx->nrl; i2++) {
                if (rlTraceVrow >= 0 && vr[i2] == rlTraceVrow) {
                    double cmpV = (ctx->dRlCmpVrow.p && cmpVr[i2] >= 0) ? hXrl[(size_t) s2 * n + cmpVr[i2]] : 0.0;
                    std::fprintf(stderr, "RL_VROW s=%d entry=%d lt=%d q=%.9f minQ=%.9f maxQ=%.9f target=%.9f mode=%d v=%.9f loadQ=%.9f cmpV=%.12f vt=%.12f margin=%.3e\n",
                                 s2, i2, (int) lt[(size_t) s2 * ctx->nrl + i2], qb[(size_t) s2 * n + vr[i2]] + lq[i2],
                                 hMinQ[ctx->ncv > 0 ? (size_t) s2 * ctx->nrl + i2 : (size_t) i2],
                                 hMaxQ[ctx->ncv > 0 ? (size_t) s2 * ctx->nrl + i2 : (size_t) i2],
                                 hTgt[(size_t) s2 * n + vr[i2]], hMode[(size_t) s2 * n + vr[i2]],
                                 hXrl[(size_t) s2 * n + vr[i2]], lq[i2], cmpV, vTgt[i2], cmpV - vTgt[i2]);
                }
                if ((rem[i2] || nS == 1) && std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
                    std::fprintf(stderr, "RL_TRACE s=%d entry=%d remote=%d lt=%d switched=%d q=%.9f mode=%d tol=%.3e maxQ=%.6f ncv=%d rlActive=%d v=%.9f qbuf=%.9f\n",
                                 s2, i2, (int) rem[i2], (int) lt[(size_t) s2 * ctx->nrl + i2], (int) rlSwitched[s2],
                                 qb[(size_t) s2 * n + vr[i2]] + lq[i2], hMode[(size_t) s2 * n + vr[i2]], reactiveLimitTol,
                                 hMaxQ[ctx->ncv > 0 ? (size_t) s2 * ctx->nrl + i2 : (size_t) i2], ctx->ncv, (int) hRlAct[s2],
                                 hXrl[(size_t) s2 * n + vr[i2]], qb[(size_t) s2 * n + vr[i2]]);
                }
            }
        }
        // PSG: every pilot-self group's layout this pass — the pilot's pin state, its own row's mode/target,
        // whether updatePilotVeqBatch flipped it (veqOff), and every candidate row with its mode, so a
        // "control lost" verdict can be checked against the members it was taken over.
        if (ctx->hasPilotSelf && ctx->dPsPilotEntry.p && ctx->st->nQr > 0 && !ctx->st->hQrPtr.empty()
                && std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
            std::vector<int> pe(ctx->st->nQr, -1);
            std::vector<char> veq(static_cast<size_t>(ctx->st->nQr) * nS, 0);
            CUDA_CHECK(cudaMemcpy(pe.data(), ctx->dPsPilotEntry.p, pe.size() * sizeof(int), cudaMemcpyDeviceToHost));
            if (ctx->dQrVeqOff.p) {
                CUDA_CHECK(cudaMemcpy(veq.data(), ctx->dQrVeqOff.p, veq.size(), cudaMemcpyDeviceToHost));
            }
            for (int s2 = 0; s2 < nS; s2++) {
                for (int g = 0; g < ctx->st->nQr; g++) {
                    int e = pe[g];
                    if (e < 0) continue;
                    int p0 = ctx->st->hQrPtr[g];
                    int cnt = ctx->st->hQrPtr[g + 1] - p0;
                    int prow = ctx->st->hQrRows[p0];
                    std::string cands;
                    int pick = 0;
                    for (int c = 1; c < cnt; c++) {
                        int r = ctx->st->hQrRows[p0 + c];
                        int m = hMode[(size_t) s2 * n + r];
                        if (pick == 0 && m != 0 && m != 4) pick = c;
                        char b[48];
                        std::snprintf(b, sizeof b, " %d:m%d/t%.4f", r, m, hTgt[(size_t) s2 * n + r]);
                        cands += b;
                    }
                    std::fprintf(stderr, "PSG s=%d outer=%d g=%d entry=%d lt=%d pilotRow=%d mode=%d target=%.9f veqOff=%d q=%.9f pick=%d cands=[%s ]\n",
                                 s2, outer, g, e, (int) lt[(size_t) s2 * ctx->nrl + e], prow, hMode[(size_t) s2 * n + prow],
                                 hTgt[(size_t) s2 * n + prow], (int) veq[(size_t) s2 * ctx->st->nQr + g],
                                 qb[(size_t) s2 * n + vr[e]] + lq[e], pick, cands.c_str());
                }
            }
        }
    }
}

// OLF_GPU_DS_DEBUG: the distributed-slack check's gate per scenario.
void traceDsCheck(BatchSolverContext* ctx, DriverPass& drv, bool inDsPhase) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    const int& outer = drv.outer;
    std::vector<char>& live = drv.live;
    const double& SLACK_DIST_TOL = drv.slackDistTol;
    if (inDsPhase && (ctx->nds > 0 || ctx->dsSlackOnly) && getenv("OLF_GPU_DS_DEBUG")) {
        // EVERY scenario, not just 0: the question "did OLF's DS fire on this check and did ours?"
        // is per scenario, and the scenario that disagrees is rarely slot 0.
        std::vector<double> qbufH(static_cast<size_t>(n) * nS);
        std::vector<double> retainH(nS, 0.0);
        std::vector<double> absH(nS, 0.0);
        CUDA_CHECK(cudaMemcpy(qbufH.data(), ctx->dQbuf.p, sizeof(double) * qbufH.size(), cudaMemcpyDeviceToHost));
        if (ctx->nds > 0 && ctx->dDsRetain.p) {
            CUDA_CHECK(cudaMemcpy(retainH.data(), ctx->dDsRetain.p, sizeof(double) * nS, cudaMemcpyDeviceToHost));
        }
        if (ctx->dsSlackOnly && ctx->dDsSlackAbsorbed.p) {
            CUDA_CHECK(cudaMemcpy(absH.data(), ctx->dDsSlackAbsorbed.p, sizeof(double) * nS, cudaMemcpyDeviceToHost));
        }
        for (int s = 0; s < nS; s++) {
            double slackP = qbufH[static_cast<size_t>(s) * n + ctx->dsSlackPhiRow];
            double rem = (slackP - ctx->dsSlackTargetP) - retainH[s] - absH[s];
            std::fprintf(stderr, "[DS-DEV] outer=%d s=%d live=%d slackP=%.9f slackTargetP=%.9f retain=%.9f rem=%.9f tol=%.3e fires=%d\n",
                    outer, s, (int) live[s], slackP, ctx->dsSlackTargetP, retainH[s], rem, SLACK_DIST_TOL,
                    std::fabs(rem) > SLACK_DIST_TOL ? 1 : 0);
        }
    }
}

// OLF_GPU_LOOP_TRACE: one scenario's loop sequence (which slot found it unstable).
void traceLoopSlot(BatchSolverContext* ctx, DriverPass& drv, int loopTraceSlot, int nChanged, std::vector<char>& rlSwitched, bool solvePass) {
    const int& nS = drv.nS;
    const int& outer = drv.outer;
    int& loopIdx = drv.loopIdx;
    int (&loopOrder)[MAX_DEVICE_LOOPS] = drv.loopOrder;
    std::vector<char>& live = drv.live;
    BatchResult& res = drv.res;
    if (loopTraceSlot >= 0 && loopTraceSlot < nS) {
        double gate[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};   // see the gateLast write
        if (ctx->dDsGateLast.p) {
            CUDA_CHECK(cudaMemcpy(gate, ctx->dDsGateLast.as<double>() + 8 * loopTraceSlot, 8 * sizeof(double), cudaMemcpyDeviceToHost));
        }
        std::fprintf(stderr, "LOOP_PASS s=%d outer=%d loopIdx=%d kind=%d solved=%d switched=%d nChanged=%d live=%d status=%d"
                     " dsRem=%.12f dsFire=%d nrIters=%d dsSlackP=%.12f dsSlackTgt=%.12f dsSlackMismatch=%.12f"
                     " dsRetain=%.12f fsGlobal=%.12f dsPlaced=%.12f\n",
                     loopTraceSlot, outer, loopIdx, loopOrder[loopIdx], solvePass ? 1 : 0,
                     (int) rlSwitched[loopTraceSlot], nChanged, (int) live[loopTraceSlot], res.status[loopTraceSlot],
                     gate[0], (int) gate[1], res.iters[loopTraceSlot],
                     gate[2], gate[3], gate[4], gate[5], gate[6], gate[7]);
    }
}

// OLF_GPU_ITER_LOG: the pass record (loop, re-solve, switch, iterations, status, DS gate).
void tracePassLog(BatchSolverContext* ctx, DriverPass& drv, std::vector<char>& rlSwitched, bool solvePass) {
    const int& nS = drv.nS;
    const int& outer = drv.outer;
    int& loopIdx = drv.loopIdx;
    int (&loopOrder)[MAX_DEVICE_LOOPS] = drv.loopOrder;
    std::vector<char>& live = drv.live;
    BatchResult& res = drv.res;
    const bool& iterLog = drv.iterLog;
    std::vector<std::string>& iterLogLines = drv.iterLogLines;
    if (iterLog) {
        std::vector<double> gates;
        if (ctx->dDsGateLast.p) {
            gates.resize(static_cast<size_t>(nS) * 8);
            CUDA_CHECK(cudaMemcpy(gates.data(), ctx->dDsGateLast.p, gates.size() * sizeof(double), cudaMemcpyDeviceToHost));
        }
        for (int s = 0; s < nS; s++) {
            char line[256];
            std::snprintf(line, sizeof line, "PASS_LOG outer=%d loopIdx=%d kind=%d solved=%d s=%d live=%d switched=%d iters=%d"
                          " status=%d dsRem=%.17e dsFire=%d dsPlaced=%.17e",
                          outer, loopIdx, loopOrder[loopIdx], solvePass ? 1 : 0, s, (int) live[s], (int) rlSwitched[s],
                          res.iters[s], res.status[s], gates.empty() ? 0.0 : gates[8 * s],
                          gates.empty() ? 0 : (int) gates[8 * s + 1], gates.empty() ? 0.0 : gates[8 * s + 7]);
            iterLogLines.emplace_back(line);
        }
    }
}

// OLF_GPU_SW_LOG: the on-device reactive-limit decision log of one scenario.
void traceSwitchLog(BatchSolverContext* ctx, DriverPass& drv) {
    const int& nS = drv.nS;
    const int& swLogSlot = drv.swLogSlot;
    if (ctx->dSwLog.p && (swLogSlot == -2 || (swLogSlot >= 0 && swLogSlot < nS))) {
        int cnt = 0;
        CUDA_CHECK(cudaMemcpy(&cnt, ctx->dSwLogCnt.p, sizeof(int), cudaMemcpyDeviceToHost));
        int keep = cnt > SW_LOG_CAP ? SW_LOG_CAP : cnt;
        std::vector<double> lg((size_t) keep * SW_LOG_W);
        if (keep > 0) {
            CUDA_CHECK(cudaMemcpy(lg.data(), ctx->dSwLog.p, lg.size() * sizeof(double), cudaMemcpyDeviceToHost));
        }
        for (int k = 0; k < keep; k++) {
            const double* r = lg.data() + (size_t) k * SW_LOG_W;
            std::fprintf(stderr, "SW_LOG s=%d pass=%d entry=%d act=%d lt=%d sc=%d q=%.17e min=%.17e max=%.17e v=%.17e cmpV=%.17e vt=%.17e\n",
                         (int) r[11], (int) r[0], (int) r[1], (int) r[2], (int) r[9], (int) r[10],
                         r[3], r[4], r[5], r[6], r[7], r[8]);
        }
        std::fprintf(stderr, "SW_LOG_END s=%d records=%d (cap %d)\n", swLogSlot, cnt, SW_LOG_CAP);
        CUDA_CHECK(cudaMemset(ctx->dSwLogCnt.p, 0, sizeof(int)));
    }
}

// OLF_GPU_RL_AUDIT: the end-of-solve audit of the reactive-limit decisions.
void traceRlAudit(BatchSolverContext* ctx, DriverPass& drv, std::vector<std::vector<std::string>>& auditLog) {
    const double& REACTIVE_LIMIT_TOL = drv.reactiveLimitTol;
    const int& RL_MAX_SWITCH = drv.rlMaxSwitch;
    const bool& svcCoefDyn = drv.svcCoefDyn;
    const int& nRlVtOv = drv.nRlVtOv;
    DeviceBuf& dRlVtOvSlotIdx = drv.dRlVtOvSlotIdx;
    DeviceBuf& dRlVtOvVal = drv.dRlVtOvVal;
    DeviceBuf& dCvOvVal = drv.dCvOvVal;
    const int& n = drv.n;
    const int& nS = drv.nS;
    const int& nnz = drv.nnz;
    const size_t& nTot = drv.nTot;
    BatchResult& res = drv.res;
    const int& swLogSlot = drv.swLogSlot;
    const bool& useDynRlLimits = drv.useDynRlLimits;
    const int& nRlQOv = drv.nRlQOv;
    const int& nCvOv = drv.nCvOv;
    DeviceBuf& dRlQOvSlotEntry = drv.dRlQOvSlotEntry;
    DeviceBuf& dRlQOvMinMax = drv.dRlQOvMinMax;
    DeviceBuf& dRlQOvPtr = drv.dRlQOvPtr;
    DeviceBuf& dRlQOvList = drv.dRlQOvList;
    DeviceBuf& dCvOvPtr = drv.dCvOvPtr;
    DeviceBuf& dCvOvEntry = drv.dCvOvEntry;
    if (ctx->nrl > 0 && std::getenv("OLF_GPU_RL_AUDIT") != nullptr) {
        double auditTol = std::atof(std::getenv("OLF_GPU_RL_AUDIT"));
        if (auditTol <= 0.0) auditTol = 1e-6;
        std::vector<char> allLive(nS), auditDumped(nS, 0);
        for (int s = 0; s < nS; s++) allLive[s] = (res.status[s] == GpuAcSolverStatus::CONVERGED) ? 1 : 0;
        CUDA_CHECK(cudaMemcpy(ctx->dRlActive.p, allLive.data(), nS * sizeof(char), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemsetAsync(ctx->dQbuf.p, 0, nTot * sizeof(double), ctx->stream));
        if (ctx->st->cb.ne) fillClosedBranchBatch<<<grid(ctx->st->cb.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->cb.ne, nS, n, nnz, ctx->nElements, ctx->cbOffset, ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->st->cb.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dRlActive.as<char>(), ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(), nullptr, nullptr, nullptr, 0, nullptr, nullptr, RelocTables{});
        if (ctx->st->ob.ne) fillOpenBranchBatch<<<grid(ctx->st->ob.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->ob.ne, nS, n, nnz, ctx->nElements, ctx->obOffset, ctx->st->ob.in.as<double>(), ctx->st->ob.ix.as<int>(), ctx->st->ob.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dRlActive.as<char>(), nullptr, nullptr, nullptr, 0, RelocTables{});
        if (ctx->st->sh.ne) fillShuntBatch<<<grid(ctx->st->sh.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->sh.ne, nS, n, nnz, ctx->nElements, ctx->shOffset, ctx->st->sh.in.as<double>(), ctx->st->sh.ix.as<int>(), ctx->st->sh.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dRlActive.as<char>(), ctx->nShOv, ctx->dShOvSlotElem.as<int>(), ctx->dShOvGb.as<double>(), nullptr, nullptr, nullptr, 0, RelocTables{});
        if (ctx->st->nvdl) fillVdlBatch<<<grid(ctx->st->nvdl * nS), BLOCK, 0, ctx->stream>>>(ctx->st->nvdl, nS, n, nnz, ctx->st->dVdlIn.as<double>(), ctx->st->dVdlIdx.as<int>(), ctx->st->dVdlDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), nullptr, nullptr, nullptr);
        if (ctx->nzi) fillZeroImpBatch<<<grid(ctx->nzi * nS), BLOCK, 0, ctx->stream>>>(ctx->nzi, nS, n, nnz, ctx->st->dZiIn.as<double>(), ctx->st->dZiIdx.as<int>(), ctx->st->dZiDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), ctx->dRlActive.as<char>(), ctx->nziBase, ctx->nZiOn, ctx->dZiOnSlotElem.as<int>(), ctx->nZiOff, ctx->dZiOffSlotElem.as<int>(), nullptr, nullptr, nullptr, 0, RelocTables{}, ctx->nZiTree, ctx->nZiTree ? ctx->dZiTreeSlotElemVal.as<int>() : nullptr, ctx->hasZiMasks ? ctx->dZiOffMask.as<char>() : nullptr, ctx->hasZiMasks ? ctx->dZiOnMask.as<char>() : nullptr, ctx->hasZiMasks ? ctx->dZiTreeMask.as<char>() : nullptr);
        if (ctx->nRmq > 0) {
            fillRemoteMemberQBatch<<<grid(ctx->nRmq * nS), BLOCK, 0, ctx->stream>>>(ctx->nRmq, nS, n,
                    ctx->nElements, ctx->cbOffset, ctx->obOffset,
                    ctx->dRmqRow.as<int>(), ctx->dRmqElem.as<int>(), ctx->dRmqSide.as<int>(),
                    ctx->dRmqKind.as<int>(), ctx->dRmqWeight.as<double>(),
                    ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->st->sh.in.as<double>(), ctx->st->sh.ix.as<int>(),
                    ctx->st->ob.in.as<double>(), ctx->st->ob.ix.as<int>(),
                    ctx->dX.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dRlActive.as<char>(),
                    ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(),
                    ctx->nShOv, ctx->dShOvSlotElem.as<int>(), ctx->dShOvGb.as<double>(), -1);
        }
        // The audit's verdict comes from OLF's OWN check, not a second implementation of it: run the
        // real reactiveLimitsBatch on the final state. It either changes nothing - the state IS the
        // fixpoint the loop claimed - or it makes exactly the change the loop still owed, and then
        // rlSwitched names the scenarios that were returned one check short. Running it on the real
        // buffers is safe for the same reason: a fixpoint is a no-op.
        if (useDynRlLimits) {                                // limits at the FINAL targets
            rlCurveLimitsBatch<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(ctx->nrl, nS, ctx->ncv, n, ctx->cvExtrapolate,
                    ctx->dCvIdx.as<int>(), ctx->dCvIn.as<double>(), ctx->dCvPts.as<double>(),
                        ctx->dCvPtr.as<int>(), ctx->dCvList.as<int>(), cvHvdcView(ctx),
                    ctx->dRlMinQ.as<double>(), ctx->dRlMaxQ.as<double>(), ctx->dRlLoadQ.as<double>(),
                    ctx->dTarget.as<double>(),
                    nRlQOv, nRlQOv > 0 ? dRlQOvSlotEntry.as<int>() : nullptr,
                    nRlQOv > 0 ? dRlQOvMinMax.as<double>() : nullptr,
                        nRlQOv > 0 ? dRlQOvPtr.as<int>() : nullptr, nRlQOv > 0 ? dRlQOvList.as<int>() : nullptr,
                    nCvOv > 0 ? dCvOvPtr.as<int>() : nullptr,
                    nCvOv > 0 ? dCvOvEntry.as<int>() : nullptr,
                    nCvOv > 0 ? dCvOvVal.as<double>() : nullptr,
                    ctx->ngen > 0 ? ctx->dGenCum.as<double>() : nullptr,
                    ctx->ngen > 0 ? ctx->dGenCumValid.as<char>() : nullptr, ctx->hasDgAnchor ? ctx->dDgAnchor.as<double>() : nullptr, ctx->hasDgStart ? ctx->dDgStart.as<double>() : nullptr, ctx->ngen,
                    ctx->nds > 0 ? ctx->dDsPhiRow.as<int>() : nullptr,
                    ctx->nDsRowOv > 0 ? ctx->dDsRowOv.as<int>() : nullptr, ctx->nds,
                    ctx->ngen > 0 ? ctx->dgBus.as<int>() : nullptr,
                    ctx->dRlMinQDyn.as<double>(), ctx->dRlMaxQDyn.as<double>(),
                    ctx->dRlLoadQDyn.as<double>(), -1, 0);
            launchForcedTargetQLimits(ctx, nS, n);
        }
        CUDA_CHECK(cudaMemsetAsync(ctx->dRlPassSwitched.p, 0, static_cast<size_t>(ctx->nrl) * nS, ctx->stream));
        CUDA_CHECK(cudaMemsetAsync(ctx->dRlSwitched.p, 0, nS, ctx->stream));
        // The check writes pin state, row modes and row targets. The context is REUSED by the next
        // chunk, so the audit stashes everything the kernel can touch and puts it back: an
        // instrument that changes the next chunk's answer is not an instrument. dDx/dB are dead
        // here (the solve is over) and are exactly n*nS wide, so they carry the two big ones.
        CUDA_CHECK(cudaMemcpyAsync(ctx->dDx.p, ctx->dTarget.p, (size_t) n * nS * sizeof(double), cudaMemcpyDeviceToDevice, ctx->stream));
        CUDA_CHECK(cudaMemcpyAsync(ctx->dB.p, ctx->dMode.p, (size_t) n * nS * sizeof(int), cudaMemcpyDeviceToDevice, ctx->stream));
        std::vector<char> keepLt(static_cast<size_t>(ctx->nrl) * nS), keepSc(static_cast<size_t>(ctx->nrl) * nS), keepSm(static_cast<size_t>(ctx->nrl) * nS);
        std::vector<double> keepSt(static_cast<size_t>(ctx->nrl) * nS);
        CUDA_CHECK(cudaMemcpy(keepLt.data(), ctx->dRlLimitType.p, keepLt.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(keepSc.data(), ctx->dRlSwitchCount.p, keepSc.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(keepSm.data(), ctx->dRlSavedMode.p, keepSm.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(keepSt.data(), ctx->dRlSavedTarget.p, keepSt.size() * sizeof(double), cudaMemcpyDeviceToHost));
        reactiveLimitsBatch<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(
                ctx->nSvc > 0 ? ctx->dSvcRlVt.as<double>() : nullptr,
                ctx->nrl, nS, n, REACTIVE_LIMIT_TOL, RL_MAX_SWITCH,
                ctx->dRlVrow.as<int>(),
                ctx->dRlQrow.p ? ctx->dRlQrow.as<int>() : nullptr,
                useDynRlLimits ? ctx->dRlMinQDyn.as<double>() : ctx->dRlMinQ.as<double>(),
                useDynRlLimits ? ctx->dRlMaxQDyn.as<double>() : ctx->dRlMaxQ.as<double>(),
                useDynRlLimits ? ctx->dRlLoadQDyn.as<double>() : ctx->dRlLoadQ.as<double>(),
                ctx->dRlVtarget.as<double>(), ctx->dQbuf.as<double>(), ctx->dX.as<double>(), ctx->dRlActive.as<char>(),
                ctx->dMode.as<int>(), ctx->dTarget.as<double>(),
                ctx->dRlLimitType.as<char>(), ctx->dRlSwitchCount.as<char>(), ctx->dRlSwitched.as<char>(),
                ctx->dRlPassSwitched.as<char>(), ctx->dRlSavedMode.as<char>(),
                nRlVtOv, nRlVtOv > 0 ? dRlVtOvSlotIdx.as<int>() : nullptr,
                nRlVtOv > 0 ? dRlVtOvVal.as<double>() : nullptr,
                ctx->dRowSlope.as<double>(), ctx->dRlSavedTarget.as<double>(), useDynRlLimits ? 1 : 0,
                ctx->nRlRemote > 0 ? ctx->dRlRemote.as<char>() : nullptr,
                ctx->nRlRemote > 0 ? ctx->dRlCmpVrow.as<int>() : nullptr,
                ctx->dRlSvcBmin.as<double>(), ctx->dRlSvcBmax.as<double>(), ctx->dRlV2Base.as<double>(),
                    svcCoefDyn ? ctx->dRlSvcBminDyn.as<double>() : nullptr, svcCoefDyn ? ctx->dRlSvcBmaxDyn.as<double>() : nullptr,
                ctx->nRlIslanded > 0 ? ctx->dRlIslanded.as<char>() : nullptr,
                ctx->rlRobust ? ctx->dRlInitTgtQ.as<double>() : nullptr,
                ctx->rlVLoRobust, ctx->rlVHiRobust, ctx->rlRobust,
                swLogSlot, ctx->dSwLog.p ? ctx->dSwLog.as<double>() : nullptr,
                ctx->dSwLogCnt.p ? ctx->dSwLogCnt.as<int>() : nullptr, 999999,
                    psEntryPtr(ctx), ctx->st->nQr,
                    ctx->nRlLocal > 0 ? ctx->dRlLocalOv.as<char>() : nullptr,
                    ctx->nRlLocal > 0 ? ctx->dRlLocalVt.as<double>() : nullptr,
                    ctx->nRlNoUnpin > 0 ? ctx->dRlNoUnpin.as<char>() : nullptr, ctx->nRlNoUnpin > 0 ? ctx->dRlUnpinBad.as<char>() : nullptr,
                    ctx->dRlMonitored.p ? ctx->dRlMonitored.as<char>() : nullptr,
                    ctx->nRlCmp > 0 ? ctx->dRlCmpOv.as<int>() : nullptr);
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        std::vector<char> aSw(nS), aPs(static_cast<size_t>(ctx->nrl) * nS);
        CUDA_CHECK(cudaMemcpy(aSw.data(), ctx->dRlSwitched.p, nS, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(aPs.data(), ctx->dRlPassSwitched.p, aPs.size(), cudaMemcpyDeviceToHost));
        std::vector<char> altPost(static_cast<size_t>(ctx->nrl) * nS);
        CUDA_CHECK(cudaMemcpy(altPost.data(), ctx->dRlLimitType.p, altPost.size(), cudaMemcpyDeviceToHost));
        // put every touched buffer back before anything else reads them
        CUDA_CHECK(cudaMemcpyAsync(ctx->dTarget.p, ctx->dDx.p, (size_t) n * nS * sizeof(double), cudaMemcpyDeviceToDevice, ctx->stream));
        CUDA_CHECK(cudaMemcpyAsync(ctx->dMode.p, ctx->dB.p, (size_t) n * nS * sizeof(int), cudaMemcpyDeviceToDevice, ctx->stream));
        CUDA_CHECK(cudaMemcpy(ctx->dRlLimitType.p, keepLt.data(), keepLt.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->dRlSwitchCount.p, keepSc.data(), keepSc.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->dRlSavedMode.p, keepSm.data(), keepSm.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->dRlSavedTarget.p, keepSt.data(), keepSt.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemsetAsync(ctx->dRlSwitched.p, 0, nS, ctx->stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        std::vector<double> aq(static_cast<size_t>(n) * nS), ax(static_cast<size_t>(n) * nS), alq(ctx->nrl);
        std::vector<char> alt(static_cast<size_t>(ctx->nrl) * nS), aisl;
        std::vector<int> avr(ctx->nrl), amode(static_cast<size_t>(n) * nS);
        CUDA_CHECK(cudaMemcpy(aq.data(), ctx->dQbuf.p, aq.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ax.data(), ctx->dX.p, ax.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(alq.data(), ctx->dRlLoadQ.p, alq.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(alt.data(), ctx->dRlLimitType.p, alt.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(avr.data(), ctx->dRlVrow.p, avr.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(amode.data(), ctx->dMode.p, amode.size() * sizeof(int), cudaMemcpyDeviceToHost));
        if (ctx->nRlIslanded > 0) {
            aisl.resize(static_cast<size_t>(ctx->nrl) * nS);
            CUDA_CHECK(cudaMemcpy(aisl.data(), ctx->dRlIslanded.p, aisl.size(), cudaMemcpyDeviceToHost));
        }
        std::vector<double> amn(static_cast<size_t>(ctx->nrl) * (ctx->ncv > 0 ? nS : 1)), amx;
        amx.resize(amn.size());
        CUDA_CHECK(cudaMemcpy(amn.data(), ctx->ncv > 0 ? ctx->dRlMinQDyn.p : ctx->dRlMinQ.p, amn.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(amx.data(), ctx->ncv > 0 ? ctx->dRlMaxQDyn.p : ctx->dRlMaxQ.p, amx.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<int> acv(ctx->nrl, -1);
        std::vector<double> avt(ctx->nrl, 0.0);
        if (ctx->dRlCmpVrow.p) {
            CUDA_CHECK(cudaMemcpy(acv.data(), ctx->dRlCmpVrow.p, acv.size() * sizeof(int), cudaMemcpyDeviceToHost));
        }
        if (ctx->dRlVtarget.p) {
            CUDA_CHECK(cudaMemcpy(avt.data(), ctx->dRlVtarget.p, avt.size() * sizeof(double), cudaMemcpyDeviceToHost));
        }
        for (int s = 0; s < nS; s++) {
            if (!allLive[s] || !aSw[s]) continue;
            for (int i = 0; i < ctx->nrl; i++) {
                size_t st = (size_t) s * ctx->nrl + i;
                if (!aPs[st]) continue;
                size_t li = ctx->ncv > 0 ? st : (size_t) i;
                double cmpV = (acv[i] >= 0) ? ax[(size_t) s * n + acv[i]] : 0.0;
                // act 1 = a pin this check owed, 2 = an unpin, 3 = a moved limit it owed
                std::fprintf(stderr, "RL_OWED s=%d entry=%d vrow=%d act=%d ltBefore=%d ltAfter=%d q=%.15e min=%.15e max=%.15e v=%.15e cmpV=%.15e vt=%.15e margin=%.3e\n",
                             s, i, avr[i], (int) aPs[st], (int) alt[st], (int) altPost[st],
                             aq[(size_t) s * n + avr[i]] + alq[i], amn[li], amx[li],
                             ax[(size_t) s * n + avr[i]], cmpV, avt[i], cmpV - avt[i]);
                if (!auditDumped.empty() && !auditDumped[s]) {
                    auditDumped[s] = 1;
                    for (const std::string& r : auditLog[s]) {
                        std::fprintf(stderr, "AUDIT_HIST s=%d | %s", s, r.c_str());
                    }
                }
            }
        }
        for (int s = 0; s < nS; s++) {
            if (!allLive[s]) continue;
            for (int i = 0; i < ctx->nrl; i++) {
                size_t st = (size_t) s * ctx->nrl + i;
                if (alt[st] != 0) continue;                  // pinned: its row holds the limit by construction
                if (!aisl.empty() && aisl[st]) continue;     // not a controller in this component
                int m = amode[(size_t) s * n + avr[i]];
                if (m == 4) continue;                        // row owns its own Q balance this scenario
                size_t li = ctx->ncv > 0 ? st : (size_t) i;
                double q = aq[(size_t) s * n + avr[i]] + alq[i];
                double over = q < amn[li] ? amn[li] - q : (q > amx[li] ? q - amx[li] : 0.0);
                if (over > auditTol) {
                    std::fprintf(stderr, "RL_AUDIT s=%d entry=%d vrow=%d mode=%d q=%.15e min=%.15e max=%.15e over=%.6e v=%.15e\n",
                                 s, i, avr[i], m, q, amn[li], amx[li], over, ax[(size_t) s * n + avr[i]]);
                    if (!auditDumped.empty() && !auditDumped[s]) {
                        auditDumped[s] = 1;
                        for (const std::string& r : auditLog[s]) {
                            std::fprintf(stderr, "AUDIT_HIST s=%d | %s", s, r.c_str());
                        }
                    }
                }
            }
        }
    }
}

// OLF_GPU_DBG_DS: the distributed-slack state at the end of the solve.
void traceDsFinal(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    BatchResult& res = drv.res;
    if (getenv("OLF_GPU_DBG_DS") && ctx->nds > 0) {
        std::vector<double> tFinal(static_cast<size_t>(n) * nS);
        CUDA_CHECK(cudaMemcpy(tFinal.data(), ctx->dTarget.p, tFinal.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<char> ltDump(static_cast<size_t>(std::max(ctx->nrl, 1)) * nS, 0);
        if (ctx->nrl > 0) {
            CUDA_CHECK(cudaMemcpy(ltDump.data(), ctx->dRlLimitType.p, static_cast<size_t>(ctx->nrl) * nS, cudaMemcpyDeviceToHost));
        }
        std::vector<int> dsRows(ctx->nds);
        std::vector<double> dsBase(ctx->nds);
        CUDA_CHECK(cudaMemcpy(dsRows.data(), ctx->dDsPhiRow.p, ctx->nds * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dsBase.data(), ctx->dDsBaseTargetP.p, ctx->nds * sizeof(double), cudaMemcpyDeviceToHost));
        for (int s = 0; s < nS; s++) {
            double tot = 0;
            fprintf(stderr, "GPU-DS s=%d", s);
            for (int i = 0; i < ctx->nds; i++) {
                double d = tFinal[(size_t) s * n + dsRows[i]] - dsBase[i];
                tot += d;
                fprintf(stderr, " row%d=%+.4f", dsRows[i], d * 100.0);
            }
            fprintf(stderr, " | totMW=%+.4f | lt:", tot * 100.0);
            for (int i = 0; i < ctx->nrl; i++) fprintf(stderr, "%d", ltDump[(size_t) i * nS + s]);
            fprintf(stderr, " status=%d\n", res.status[s]);
        }
    }
}

// OLF_GPU_PHASE_TIME: the per-scenario solve-pass histogram.
void tracePhaseTime(BatchSolverContext* ctx, DriverPass& drv) {
    const int& nS = drv.nS;
    BatchResult& res = drv.res;
    if (!ctx->driverPaused && std::getenv("OLF_GPU_PHASE_TIME") != nullptr && ctx->olPasses.size() == static_cast<size_t>(nS)) {
        for (int s = 0; s < nS; s++) {
            if (ctx->olPasses[s] == 0) continue;
            if (res.status[s] != GpuAcSolverStatus::CONVERGED) {                         // not finished converged on device (CPU replay,
                gOlNotConverged++;                           // max-iter, ...): its work is counted apart
                gOlNotConvergedIters += ctx->olIters[s];
                continue;
            }
            int k = std::min(ctx->olPasses[s], 15);
            gOlHistCount[k]++;
            gOlHistFirst[k] += ctx->olFirst[s];
            gOlHistTotal[k] += ctx->olIters[s];
        }
    }
}

}  // namespace olfgpu
