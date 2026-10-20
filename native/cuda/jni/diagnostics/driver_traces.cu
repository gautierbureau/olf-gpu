// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Driver traces, see driver_traces.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../equations/scenario_overrides.cuh"
#include "../common/launch.cuh"
#include "../common/stats.cuh"
#include "../common/codes/gpu_ac_solver_status.cuh"
#include "../common/codes/gpu_outer_loop_kind.cuh"
#include "../common/codes/gpu_transformer_voltage_control_mode.cuh"
#include "../common/codes/gpu_row_mode.cuh"
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

// OLF_GPU_BATCH_TRACE: the pass's per-scenario state before the loop check.
void traceBatchPass(BatchSolverContext* ctx, DriverPass& drv, int nLoops, bool solvePass) {
    const int& nS = drv.nS;
    const int& outer = drv.outer;
    int& loopIdx = drv.loopIdx;
    int (&loopOrder)[MAX_DEVICE_LOOPS] = drv.loopOrder;
    if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
        const char* kindName = "?";
        switch (loopOrder[loopIdx]) {
            case GpuOuterLoopKind::DISTRIBUTED_SLACK:           kindName = "DistributedSlack"; break;
            case GpuOuterLoopKind::FREEZING_HVDC_AC_EMULATION:  kindName = "FreezingHvdcACEmulation"; break;
            case GpuOuterLoopKind::HVDC_AC_EMULATION_LIMITS:  kindName = "AcHvdcAcEmulationLimits"; break;
            case GpuOuterLoopKind::HOST_SECONDARY_VOLTAGE_CONTROL:     kindName = "HOST:SecondaryVoltageControl"; break;
            case GpuOuterLoopKind::MONITORING_VOLTAGE:   kindName = "MonitoringVoltage"; break;
            case GpuOuterLoopKind::REACTIVE_LIMITS:           kindName = "ReactiveLimits"; break;
            case GpuOuterLoopKind::HOST_PHASE_CONTROL:   kindName = "HOST:PhaseControl"; break;
            case GpuOuterLoopKind::HOST_TRANSFORMER_VOLTAGE_CONTROL:     kindName = "HOST:TransformerVoltageControl"; break;
            case GpuOuterLoopKind::HOST_TRANSFORMER_REACTIVE_POWER_CONTROL:    kindName = "HOST:TransformerReactivePowerControl"; break;
            case GpuOuterLoopKind::HOST_SHUNT_VOLTAGE_CONTROL:   kindName = "HOST:ShuntVoltageControl"; break;
            case GpuOuterLoopKind::INCREMENTAL_TRANSFORMER_VOLTAGE_CONTROL:   kindName = "TransformerVoltageControl"; break;
            case GpuOuterLoopKind::INCREMENTAL_SHUNT_VOLTAGE_CONTROL: kindName = "ShuntVoltageControl"; break;
            case GpuOuterLoopKind::INCREMENTAL_PHASE_CONTROL: kindName = ctx->nPhaseControlControllerBranches > 0 ? "PhaseControl" : "IncrementalPhaseControl"; break;
            case GpuOuterLoopKind::INCREMENTAL_TRANSFORMER_REACTIVE_POWER_CONTROL:  kindName = "TransformerReactivePowerControl"; break;
            case GpuOuterLoopKind::TRANSFORMER_VOLTAGE_CONTROL:   kindName = ctx->transformerVoltageControlMode == GpuTransformerVoltageControlMode::WITH_GENERATOR_VOLTAGE_CONTROL ? "SimpleTransformerVoltageControl" : "TransformerVoltageControl"; break;
            case GpuOuterLoopKind::SHUNT_VOLTAGE_CONTROL:   kindName = "ShuntVoltageControl"; break;
            case GpuOuterLoopKind::SECONDARY_VOLTAGE_CONTROL:   kindName = "SecondaryVoltageControl"; break;
        }
        int nUnstableScenarios = 0;
        {
            std::vector<char> sw(nS);
            CUDA_CHECK(cudaMemcpy(sw.data(), ctx->dOuterLoopUnstable.p, nS, cudaMemcpyDeviceToHost));
            for (char c : sw) nUnstableScenarios += c ? 1 : 0;
        }
        int nPinned = 0;
        if (ctx->nReactiveLimitEntries > 0) {
            std::vector<char> lt(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS);
            CUDA_CHECK(cudaMemcpy(lt.data(), ctx->dReactiveLimitQLimitType.p, lt.size(), cudaMemcpyDeviceToHost));
            for (char c : lt) nPinned += c ? 1 : 0;
        }
        std::fprintf(stderr, "BATCH_TRACE pass done: outer=%d loop=%d/%d kind=%s solved=%d changedScenarios=%d iters=%d pinned=%d",
                     outer, loopIdx, nLoops, kindName, solvePass ? 1 : 0, nUnstableScenarios, gLastPassIters, nPinned);
        std::fprintf(stderr, "\n");
    }
}

// OLF_GPU_RL_AUDIT / OLF_GPU_SW_TRACE: every reactive-limit switch of the pass, per scenario.
void traceReactiveLimitsSwitches(BatchSolverContext* ctx, DriverPass& drv, std::vector<std::vector<std::string>>& auditLog, bool reactiveLimitsAuditOn, int reactiveLimitsTraceScenario) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    const int& outer = drv.outer;
    int& loopIdx = drv.loopIdx;
    int (&loopOrder)[MAX_DEVICE_LOOPS] = drv.loopOrder;
    std::vector<char>& outerLoopCheckActive = drv.outerLoopCheckActive;
    if (loopOrder[loopIdx] == GpuOuterLoopKind::REACTIVE_LIMITS && (reactiveLimitsAuditOn || (reactiveLimitsTraceScenario >= 0 && reactiveLimitsTraceScenario < nS)) && ctx->nReactiveLimitEntries > 0) {
        std::vector<char> ps(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS), lt2(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS);
        std::vector<int> vr2(ctx->nReactiveLimitEntries);
        std::vector<double> qb2(static_cast<size_t>(n) * nS), lq2(ctx->nReactiveLimitEntries), x2(static_cast<size_t>(n) * nS);
        CUDA_CHECK(cudaMemcpy(ps.data(), ctx->dReactiveLimitCheckDecision.p, ps.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lt2.data(), ctx->dReactiveLimitQLimitType.p, lt2.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vr2.data(), ctx->dReactiveLimitVoltageRow.p, vr2.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(qb2.data(), ctx->dQbuf.p, qb2.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lq2.data(), ctx->dReactiveLimitLoadQ.p, lq2.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(x2.data(), ctx->dX.p, x2.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<double> mn2(static_cast<size_t>(ctx->nReactiveLimitEntries) * (ctx->nReactiveCapabilityCurves > 0 ? nS : 1));
        std::vector<double> mx2(mn2.size());
        CUDA_CHECK(cudaMemcpy(mn2.data(), ctx->nReactiveCapabilityCurves > 0 ? ctx->dReactiveLimitMinQPerScenario.p : ctx->dReactiveLimitMinQ.p, mn2.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mx2.data(), ctx->nReactiveCapabilityCurves > 0 ? ctx->dReactiveLimitMaxQPerScenario.p : ctx->dReactiveLimitMaxQ.p, mx2.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<int> cv2(ctx->nReactiveLimitEntries, -1);
        std::vector<double> vt2(ctx->nReactiveLimitEntries, 0.0);
        if (ctx->dReactiveLimitComparedVoltageRow.p) {
            CUDA_CHECK(cudaMemcpy(cv2.data(), ctx->dReactiveLimitComparedVoltageRow.p, cv2.size() * sizeof(int), cudaMemcpyDeviceToHost));
        }
        if (ctx->dReactiveLimitTargetV.p) {
            CUDA_CHECK(cudaMemcpy(vt2.data(), ctx->dReactiveLimitTargetV.p, vt2.size() * sizeof(double), cudaMemcpyDeviceToHost));
        }
        for (int s2 = 0; s2 < nS; s2++) {
            bool toStderr = (s2 == reactiveLimitsTraceScenario);
            if (!toStderr && !(reactiveLimitsAuditOn && outerLoopCheckActive[s2])) continue;
            int nReactiveLimitsDecisions = 0;
            std::string rec;
            for (int i2 = 0; i2 < ctx->nReactiveLimitEntries; i2++) {
                size_t st2 = (size_t) s2 * ctx->nReactiveLimitEntries + i2;
                if (!ps[st2]) continue;
                size_t li2 = ctx->nReactiveCapabilityCurves > 0 ? st2 : (size_t) i2;
                double cmpV2 = (cv2[i2] >= 0) ? x2[(size_t) s2 * n + cv2[i2]] : 0.0;
                double qq = qb2[(size_t) s2 * n + vr2[i2]] + lq2[i2];
                // the q the decision READ: at reactiveLimitReactivePowerRow when the entry's balance was relocated (a coupler-tree
                // child reads its parent's DUMMY_Q row), not at its V row - printing the V-row value logged a
                // relocated child as q=0 while the kernel pinned it on its real q
                static const std::vector<int> qr2 = [&] {
                    std::vector<int> v(ctx->nReactiveLimitEntries, -1);
                    if (ctx->dReactiveLimitReactivePowerRow.p) CUDA_CHECK(cudaMemcpy(v.data(), ctx->dReactiveLimitReactivePowerRow.p, v.size() * sizeof(int), cudaMemcpyDeviceToHost));
                    return v;
                }();
                int qrow2 = qr2[i2] >= 0 ? qr2[i2] : vr2[i2];
                double qRead = qb2[(size_t) s2 * n + qrow2] + lq2[i2];
                nReactiveLimitsDecisions++;
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
                    CUDA_CHECK(cudaMemcpy(&sc, ctx->dReactiveLimitPvPqSwitchCount.as<char>() + st2, 1, cudaMemcpyDeviceToHost));
                    char lov = 0;
                    double lvt = 0.0;
                    if (ctx->nReactiveLimitLocalOverrides > 0) {
                        CUDA_CHECK(cudaMemcpy(&lov, ctx->dReactiveLimitLocalOverride.as<char>() + st2, 1, cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaMemcpy(&lvt, ctx->dReactiveLimitLocalTargetV.as<double>() + st2, sizeof(double), cudaMemcpyDeviceToHost));
                    }
                    char mon = 0;
                    CUDA_CHECK(cudaMemcpy(&mon, ctx->dReactiveLimitFlags.as<char>() + i2, 1, cudaMemcpyDeviceToHost));
                    char isl = 0;
                    if (ctx->nReactiveLimitIslanded > 0) {
                        CUDA_CHECK(cudaMemcpy(&isl, ctx->dReactiveLimitIslanded.as<char>() + st2, 1, cudaMemcpyDeviceToHost));
                    }
                    char act = 0;
                    CUDA_CHECK(cudaMemcpy(&act, ctx->dOuterLoopCheckActive.as<char>() + s2, 1, cudaMemcpyDeviceToHost));
                    std::fprintf(stderr, "%s slope=%.6e swCount=%d localOv=%d localVt=%.12f monitored=%d islanded=%d outerLoopCheckActive=%d\n", buf, slope, (int) sc,
                                 (int) lov, lvt, (int) mon, (int) isl, (int) act);
                }
                if (reactiveLimitsAuditOn && nReactiveLimitsDecisions <= 12) {
                    rec += buf;
                    rec += "\n";
                }
            }
            if (reactiveLimitsAuditOn) {
                char hdr[160];
                std::snprintf(hdr, sizeof hdr, "AUDIT_PASS s=%d outer=%d loop=%d outerLoopCheckActive=%d nReactiveLimitsDecisions=%d",
                              s2, outer, loopOrder[loopIdx], (int) outerLoopCheckActive[s2], nReactiveLimitsDecisions);
                auditLog[s2].push_back(std::string(hdr) + "\n" + rec);
            }
        }
    }
}

// OLF_GPU_BATCH_TRACE: the reactive-limit entries' state after the check.
void traceReactiveLimitsState(BatchSolverContext* ctx, DriverPass& drv, bool inDistributedSlackPhase, double reactiveLimitTol, std::vector<char>& outerLoopUnstable) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    const int& outer = drv.outer;
    if (!inDistributedSlackPhase && (ctx->nReactiveLimitRemoteEntries > 0 || nS == 1) && ctx->nReactiveLimitEntries > 0
            && std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {   // RL trace
        std::vector<char> lt(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS);
        std::vector<char> rem(ctx->nReactiveLimitEntries);
        std::vector<int> vr(ctx->nReactiveLimitEntries);
        std::vector<double> qb(static_cast<size_t>(n) * nS);
        std::vector<double> lq(ctx->nReactiveLimitEntries);
        CUDA_CHECK(cudaMemcpy(lt.data(), ctx->dReactiveLimitQLimitType.p, lt.size(), cudaMemcpyDeviceToHost));
        if (ctx->nReactiveLimitRemoteEntries > 0) {
        CUDA_CHECK(cudaMemcpy(rem.data(), ctx->dReactiveLimitVoltageControlRole.p, rem.size(), cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaMemcpy(vr.data(), ctx->dReactiveLimitVoltageRow.p, vr.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(qb.data(), ctx->dQbuf.p, qb.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lq.data(), ctx->dReactiveLimitLoadQ.p, lq.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<int> hMode(static_cast<size_t>(n) * nS);
        CUDA_CHECK(cudaMemcpy(hMode.data(), ctx->dMode.p, hMode.size() * sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<double> hMaxQ(static_cast<size_t>(ctx->nReactiveLimitEntries) * (ctx->nReactiveCapabilityCurves > 0 ? nS : 1));
        CUDA_CHECK(cudaMemcpy(hMaxQ.data(), ctx->nReactiveCapabilityCurves > 0 ? ctx->dReactiveLimitMaxQPerScenario.p : ctx->dReactiveLimitMaxQ.p, hMaxQ.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<char> hOuterLoopCheckActive(nS);
        CUDA_CHECK(cudaMemcpy(hOuterLoopCheckActive.data(), ctx->dOuterLoopCheckActive.p, nS, cudaMemcpyDeviceToHost));
        // The STATE the pin decision is taken at. Without it, a q that disagrees with OLF cannot be
        // told apart from a q evaluated at a different (or not yet converged) operating point - the
        // same quantity at a different V is not evidence of a wrong formula.
        std::vector<double> hXrl(static_cast<size_t>(n) * nS);
        CUDA_CHECK(cudaMemcpy(hXrl.data(), ctx->dX.p, hXrl.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<double> hTgt(static_cast<size_t>(n) * nS);
        CUDA_CHECK(cudaMemcpy(hTgt.data(), ctx->dTarget.p, hTgt.size() * sizeof(double), cudaMemcpyDeviceToHost));
        for (int s2 = 0; s2 < nS; s2++) {
            for (int i2 = 0; i2 < ctx->nReactiveLimitEntries; i2++) {
                if (rem[i2] || nS == 1) {
                    std::fprintf(stderr, "RL_TRACE s=%d entry=%d remote=%d lt=%d switched=%d q=%.9f mode=%d tol=%.3e maxQ=%.6f nReactiveCapabilityCurves=%d outerLoopCheckActive=%d v=%.9f qbuf=%.9f\n",
                                 s2, i2, (int) rem[i2], (int) lt[(size_t) s2 * ctx->nReactiveLimitEntries + i2], (int) outerLoopUnstable[s2],
                                 qb[(size_t) s2 * n + vr[i2]] + lq[i2], hMode[(size_t) s2 * n + vr[i2]], reactiveLimitTol,
                                 hMaxQ[ctx->nReactiveCapabilityCurves > 0 ? (size_t) s2 * ctx->nReactiveLimitEntries + i2 : (size_t) i2], ctx->nReactiveCapabilityCurves, (int) hOuterLoopCheckActive[s2],
                                 hXrl[(size_t) s2 * n + vr[i2]], qb[(size_t) s2 * n + vr[i2]]);
                }
            }
        }
        // PSG: every pilot-self group's layout this pass — the pilot's pin state, its own row's mode/target,
        // whether updatePilotTargetVEquationBatch flipped it (pilotTargetVDeactivated), and every candidate row with its mode, so a
        // "control lost" verdict can be checked against the members it was taken over.
        if (ctx->hasPilotSelf && ctx->dPilotSelfReactiveLimitEntry.p && ctx->st->nReactiveRelocationGroups > 0 && !ctx->st->hReactiveRelocationGroupStart.empty()
                && std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
            std::vector<int> pe(ctx->st->nReactiveRelocationGroups, -1);
            std::vector<char> hPilotTargetVDeactivated(static_cast<size_t>(ctx->st->nReactiveRelocationGroups) * nS, 0);
            CUDA_CHECK(cudaMemcpy(pe.data(), ctx->dPilotSelfReactiveLimitEntry.p, pe.size() * sizeof(int), cudaMemcpyDeviceToHost));
            if (ctx->dReactiveRelocationPilotTargetVDeactivated.p) {
                CUDA_CHECK(cudaMemcpy(hPilotTargetVDeactivated.data(), ctx->dReactiveRelocationPilotTargetVDeactivated.p, hPilotTargetVDeactivated.size(), cudaMemcpyDeviceToHost));
            }
            for (int s2 = 0; s2 < nS; s2++) {
                for (int g = 0; g < ctx->st->nReactiveRelocationGroups; g++) {
                    int e = pe[g];
                    if (e < 0) continue;
                    int p0 = ctx->st->hReactiveRelocationGroupStart[g];
                    int cnt = ctx->st->hReactiveRelocationGroupStart[g + 1] - p0;
                    int prow = ctx->st->hReactiveRelocationCandidateRows[p0];
                    std::string cands;
                    int pick = 0;
                    for (int c = 1; c < cnt; c++) {
                        int r = ctx->st->hReactiveRelocationCandidateRows[p0 + c];
                        int m = hMode[(size_t) s2 * n + r];
                        if (pick == 0 && m != GpuRowMode::FIXED_VARIABLE && m != GpuRowMode::BUS_TARGET_Q) pick = c;
                        char b[48];
                        std::snprintf(b, sizeof b, " %d:m%d/t%.4f", r, m, hTgt[(size_t) s2 * n + r]);
                        cands += b;
                    }
                    std::fprintf(stderr, "PSG s=%d outer=%d g=%d entry=%d lt=%d pilotRow=%d mode=%d target=%.9f pilotTargetVDeactivated=%d q=%.9f pick=%d cands=[%s ]\n",
                                 s2, outer, g, e, (int) lt[(size_t) s2 * ctx->nReactiveLimitEntries + e], prow, hMode[(size_t) s2 * n + prow],
                                 hTgt[(size_t) s2 * n + prow], (int) hPilotTargetVDeactivated[(size_t) s2 * ctx->st->nReactiveRelocationGroups + g],
                                 qb[(size_t) s2 * n + vr[e]] + lq[e], pick, cands.c_str());
                }
            }
        }
    }
}

// OLF_GPU_LOOP_TRACE: one scenario's loop sequence (which slot found it unstable).
void traceLoopSlot(BatchSolverContext* ctx, DriverPass& drv, int loopTraceSlot, int nChanged, std::vector<char>& outerLoopUnstable, bool solvePass) {
    const int& nS = drv.nS;
    const int& outer = drv.outer;
    int& loopIdx = drv.loopIdx;
    int (&loopOrder)[MAX_DEVICE_LOOPS] = drv.loopOrder;
    std::vector<char>& live = drv.live;
    BatchResult& res = drv.res;
    if (loopTraceSlot >= 0 && loopTraceSlot < nS) {
        double gate[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};   // see the gateLast write
        if (ctx->dDistributedSlackLastPassAccounting.p) {
            CUDA_CHECK(cudaMemcpy(gate, ctx->dDistributedSlackLastPassAccounting.as<double>() + 8 * loopTraceSlot, 8 * sizeof(double), cudaMemcpyDeviceToHost));
        }
        std::fprintf(stderr, "LOOP_PASS s=%d outer=%d loopIdx=%d kind=%d solved=%d switched=%d nChanged=%d live=%d status=%d"
                     " dsRem=%.12f dsFire=%d nrIters=%d dsSlackP=%.12f dsSlackTgt=%.12f dsSlackMismatch=%.12f"
                     " slackRetainedShare=%.12f fsGlobal=%.12f dsPlaced=%.12f\n",
                     loopTraceSlot, outer, loopIdx, loopOrder[loopIdx], solvePass ? 1 : 0,
                     (int) outerLoopUnstable[loopTraceSlot], nChanged, (int) live[loopTraceSlot], res.status[loopTraceSlot],
                     gate[0], (int) gate[1], res.iters[loopTraceSlot],
                     gate[2], gate[3], gate[4], gate[5], gate[6], gate[7]);
    }
}

// OLF_GPU_SW_LOG: the on-device reactive-limit decision log of one scenario.
void traceSwitchLog(BatchSolverContext* ctx, DriverPass& drv) {
    const int& nS = drv.nS;
    const int& reactiveLimitsDecisionLogScenario = drv.reactiveLimitsDecisionLogScenario;
    if (ctx->dReactiveLimitsDecisionLog.p && (reactiveLimitsDecisionLogScenario == -2 || (reactiveLimitsDecisionLogScenario >= 0 && reactiveLimitsDecisionLogScenario < nS))) {
        int cnt = 0;
        CUDA_CHECK(cudaMemcpy(&cnt, ctx->dReactiveLimitsDecisionLogCount.p, sizeof(int), cudaMemcpyDeviceToHost));
        int keep = cnt > REACTIVE_LIMITS_DECISION_LOG_CAPACITY ? REACTIVE_LIMITS_DECISION_LOG_CAPACITY : cnt;
        std::vector<double> lg((size_t) keep * REACTIVE_LIMITS_DECISION_LOG_WIDTH);
        if (keep > 0) {
            CUDA_CHECK(cudaMemcpy(lg.data(), ctx->dReactiveLimitsDecisionLog.p, lg.size() * sizeof(double), cudaMemcpyDeviceToHost));
        }
        for (int k = 0; k < keep; k++) {
            const double* r = lg.data() + (size_t) k * REACTIVE_LIMITS_DECISION_LOG_WIDTH;
            std::fprintf(stderr, "SW_LOG s=%d pass=%d entry=%d act=%d lt=%d sc=%d q=%.17e min=%.17e max=%.17e v=%.17e cmpV=%.17e vt=%.17e\n",
                         (int) r[11], (int) r[0], (int) r[1], (int) r[2], (int) r[9], (int) r[10],
                         r[3], r[4], r[5], r[6], r[7], r[8]);
        }
        std::fprintf(stderr, "SW_LOG_END s=%d records=%d (cap %d)\n", reactiveLimitsDecisionLogScenario, cnt, REACTIVE_LIMITS_DECISION_LOG_CAPACITY);
        CUDA_CHECK(cudaMemset(ctx->dReactiveLimitsDecisionLogCount.p, 0, sizeof(int)));
    }
}

// OLF_GPU_RL_AUDIT: the end-of-solve audit of the reactive-limit decisions.
void traceReactiveLimitsAudit(BatchSolverContext* ctx, DriverPass& drv, std::vector<std::vector<std::string>>& auditLog) {
    const double& REACTIVE_LIMIT_TOL = drv.reactiveLimitTol;
    const int& MAX_PQ_PV_SWITCH = drv.maxPqPvSwitch;
    const bool& svcCoefDyn = drv.svcCoefDyn;
    const int& nReactiveLimitTargetVOverrides = drv.nReactiveLimitTargetVOverrides;
    DeviceBuf& dReactiveLimitTargetVOverrideScenarioEntry = drv.dReactiveLimitTargetVOverrideScenarioEntry;
    DeviceBuf& dReactiveLimitTargetVOverrideValue = drv.dReactiveLimitTargetVOverrideValue;
    DeviceBuf& dReactiveCapabilityCurveOverrideStepScale = drv.dReactiveCapabilityCurveOverrideStepScale;
    const int& n = drv.n;
    const int& nS = drv.nS;
    const int& nnz = drv.nnz;
    const size_t& nTot = drv.nTot;
    BatchResult& res = drv.res;
    const int& reactiveLimitsDecisionLogScenario = drv.reactiveLimitsDecisionLogScenario;
    const bool& usePerScenarioReactiveLimits = drv.usePerScenarioReactiveLimits;
    const int& nReactiveLimitDeltaOverrides = drv.nReactiveLimitDeltaOverrides;
    const int& nReactiveCapabilityCurveOverrides = drv.nReactiveCapabilityCurveOverrides;
    DeviceBuf& dReactiveLimitDeltaOverrideScenarioEntry = drv.dReactiveLimitDeltaOverrideScenarioEntry;
    DeviceBuf& dReactiveLimitDeltaOverrideValues = drv.dReactiveLimitDeltaOverrideValues;
    DeviceBuf& dReactiveLimitDeltaOverrideStart = drv.dReactiveLimitDeltaOverrideStart;
    DeviceBuf& dReactiveLimitDeltaOverrideList = drv.dReactiveLimitDeltaOverrideList;
    DeviceBuf& dReactiveCapabilityCurveOverrideStart = drv.dReactiveCapabilityCurveOverrideStart;
    DeviceBuf& dReactiveCapabilityCurveOverrideCurve = drv.dReactiveCapabilityCurveOverrideCurve;
    if (ctx->nReactiveLimitEntries > 0 && std::getenv("OLF_GPU_RL_AUDIT") != nullptr) {
        double auditTol = std::atof(std::getenv("OLF_GPU_RL_AUDIT"));
        if (auditTol <= 0.0) auditTol = 1e-6;
        std::vector<char> allLive(nS), auditDumped(nS, 0);
        for (int s = 0; s < nS; s++) allLive[s] = (res.status[s] == GpuAcSolverStatus::CONVERGED) ? 1 : 0;
        CUDA_CHECK(cudaMemcpy(ctx->dOuterLoopCheckActive.p, allLive.data(), nS * sizeof(char), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemsetAsync(ctx->dQbuf.p, 0, nTot * sizeof(double), ctx->stream));
        if (ctx->st->closedBranch.elementCount) fillClosedBranchBatch<<<grid(ctx->st->closedBranch.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->st->closedBranch.elementCount, nS, n, nnz, ctx->nElements, ctx->closedBranchOffset, ctx->st->closedBranch.parameters.as<double>(), ctx->st->closedBranch.indices.as<int>(), ctx->st->closedBranch.destination.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dOuterLoopCheckActive.as<char>(), ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(), nullptr, nullptr, nullptr, 0, nullptr, nullptr, ReactiveRelocationTables{});
        if (ctx->st->openBranch.elementCount) fillOpenBranchBatch<<<grid(ctx->st->openBranch.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->st->openBranch.elementCount, nS, n, nnz, ctx->nElements, ctx->openBranchOffset, ctx->st->openBranch.parameters.as<double>(), ctx->st->openBranch.indices.as<int>(), ctx->st->openBranch.destination.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dOuterLoopCheckActive.as<char>(), nullptr, nullptr, nullptr, 0, ReactiveRelocationTables{});
        if (ctx->st->shunt.elementCount) fillShuntBatch<<<grid(ctx->st->shunt.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->st->shunt.elementCount, nS, n, nnz, ctx->nElements, ctx->shuntOffset, ctx->st->shunt.parameters.as<double>(), ctx->st->shunt.indices.as<int>(), ctx->st->shunt.destination.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dOuterLoopCheckActive.as<char>(), ctx->nShuntAdmittanceOverrides, ctx->dShuntAdmittanceOverrideScenarioElement.as<int>(), ctx->dShuntAdmittanceOverrideDelta.as<double>(), nullptr, nullptr, nullptr, 0, ReactiveRelocationTables{});
        if (ctx->st->nLoadModelTerms) fillLoadModelBatch<<<grid(ctx->st->nLoadModelTerms * nS), BLOCK, 0, ctx->stream>>>(ctx->st->nLoadModelTerms, nS, n, nnz, ctx->st->dLoadModelParameters.as<double>(), ctx->st->dLoadModelIndices.as<int>(), ctx->st->dLoadModelDestination.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), nullptr, nullptr, nullptr);
        if (ctx->nZeroImpedanceBranch) fillZeroImpBatch<<<grid(ctx->nZeroImpedanceBranch * nS), BLOCK, 0, ctx->stream>>>(ctx->nZeroImpedanceBranch, nS, n, nnz, ctx->st->dZeroImpedanceBranchParameters.as<double>(), ctx->st->dZeroImpedanceBranchIndices.as<int>(), ctx->st->dZeroImpedanceBranchDestination.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), ctx->dOuterLoopCheckActive.as<char>(), ctx->nBaseClosedZeroImpedanceBranches, ctx->nZeroImpedanceBranchCloseOverrides, ctx->dZeroImpedanceBranchCloseScenarioElement.as<int>(), ctx->nZeroImpedanceBranchOpenOverrides, ctx->dZeroImpedanceBranchOpenScenarioElement.as<int>(), nullptr, nullptr, nullptr, 0, ReactiveRelocationTables{}, ctx->nSpanningTreeOverrides, ctx->nSpanningTreeOverrides ? ctx->dSpanningTreeOverrideScenarioElementValue.as<int>() : nullptr, ctx->hasZeroImpedanceBranchMasks ? ctx->dZeroImpedanceBranchOpenMask.as<char>() : nullptr, ctx->hasZeroImpedanceBranchMasks ? ctx->dZeroImpedanceBranchCloseMask.as<char>() : nullptr, ctx->hasZeroImpedanceBranchMasks ? ctx->dSpanningTreeOverrideMask.as<char>() : nullptr);
        if (ctx->nRemoteMemberReactiveTerms > 0) {
            fillRemoteMemberQBatch<<<grid(ctx->nRemoteMemberReactiveTerms * nS), BLOCK, 0, ctx->stream>>>(ctx->nRemoteMemberReactiveTerms, nS, n,
                    ctx->nElements, ctx->closedBranchOffset, ctx->openBranchOffset,
                    ctx->dRemoteMemberReactiveTermRow.as<int>(), ctx->dRemoteMemberReactiveTermElement.as<int>(), ctx->dRemoteMemberReactiveTermSide.as<int>(),
                    ctx->dRemoteMemberReactiveTermKind.as<int>(), ctx->dRemoteMemberReactiveTermWeight.as<double>(),
                    ctx->st->closedBranch.parameters.as<double>(), ctx->st->closedBranch.indices.as<int>(), ctx->st->shunt.parameters.as<double>(), ctx->st->shunt.indices.as<int>(),
                    ctx->st->openBranch.parameters.as<double>(), ctx->st->openBranch.indices.as<int>(),
                    ctx->dX.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dOuterLoopCheckActive.as<char>(),
                    ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(),
                    ctx->nShuntAdmittanceOverrides, ctx->dShuntAdmittanceOverrideScenarioElement.as<int>(), ctx->dShuntAdmittanceOverrideDelta.as<double>());
        }
        // The audit's verdict comes from OLF's OWN check, not a second implementation of it: run the
        // real reactiveLimitsBatch on the final state. It either changes nothing - the state IS the
        // fixpoint the loop claimed - or it makes exactly the change the loop still owed, and then
        // outerLoopUnstable names the scenarios that were returned one check short. Running it on the real
        // buffers is safe for the same reason: a fixpoint is a no-op.
        if (usePerScenarioReactiveLimits) {                                // limits at the FINAL targets
            reactiveCapabilityCurveLimitsBatch<<<grid(ctx->nReactiveLimitEntries * nS), BLOCK, 0, ctx->stream>>>(ctx->nReactiveLimitEntries, nS, ctx->nReactiveCapabilityCurves, n, ctx->reactiveCapabilityCurveExtrapolate,
                    ctx->dReactiveCapabilityCurveIndices.as<int>(), ctx->dReactiveCapabilityCurveParameters.as<double>(), ctx->dReactiveCapabilityCurvePoints.as<double>(),
                        ctx->dReactiveLimitEntryCurveStart.as<int>(), ctx->dReactiveLimitEntryCurves.as<int>(), curveHvdcAcEmulationView(ctx),
                    ctx->dReactiveLimitMinQ.as<double>(), ctx->dReactiveLimitMaxQ.as<double>(), ctx->dReactiveLimitLoadQ.as<double>(),
                    ctx->dTarget.as<double>(),
                    nReactiveLimitDeltaOverrides, nReactiveLimitDeltaOverrides > 0 ? dReactiveLimitDeltaOverrideScenarioEntry.as<int>() : nullptr,
                    nReactiveLimitDeltaOverrides > 0 ? dReactiveLimitDeltaOverrideValues.as<double>() : nullptr,
                        nReactiveLimitDeltaOverrides > 0 ? dReactiveLimitDeltaOverrideStart.as<int>() : nullptr, nReactiveLimitDeltaOverrides > 0 ? dReactiveLimitDeltaOverrideList.as<int>() : nullptr,
                    nReactiveCapabilityCurveOverrides > 0 ? dReactiveCapabilityCurveOverrideStart.as<int>() : nullptr,
                    nReactiveCapabilityCurveOverrides > 0 ? dReactiveCapabilityCurveOverrideCurve.as<int>() : nullptr,
                    nReactiveCapabilityCurveOverrides > 0 ? dReactiveCapabilityCurveOverrideStepScale.as<double>() : nullptr,
                    ctx->nDistributedSlackGenerators > 0 ? ctx->dDistributedSlackGeneratorMove.as<double>() : nullptr,
                    ctx->nDistributedSlackGenerators > 0 ? ctx->dDistributedSlackGeneratorMoveValid.as<char>() : nullptr, ctx->hasDistributedSlackGeneratorAnchor ? ctx->dDistributedSlackGeneratorAnchor.as<double>() : nullptr, ctx->hasDistributedSlackGeneratorInitialMove ? ctx->dDistributedSlackGeneratorInitialMove.as<double>() : nullptr, ctx->nDistributedSlackGenerators,
                    ctx->nDistributedSlackBuses > 0 ? ctx->dDistributedSlackBusPhiRow.as<int>() : nullptr,
                    ctx->nDistributedSlackBusRowOverrides > 0 ? ctx->dDistributedSlackBusRowOverride.as<int>() : nullptr, ctx->nDistributedSlackBuses,
                    ctx->nDistributedSlackGenerators > 0 ? ctx->distributedSlackGeneratorBus.as<int>() : nullptr,
                    ctx->dReactiveLimitMinQPerScenario.as<double>(), ctx->dReactiveLimitMaxQPerScenario.as<double>(),
                    ctx->dReactiveLimitLoadQPerScenario.as<double>());
            launchForcedTargetQLimits(ctx, nS, n);
        }
        CUDA_CHECK(cudaMemsetAsync(ctx->dReactiveLimitCheckDecision.p, 0, static_cast<size_t>(ctx->nReactiveLimitEntries) * nS, ctx->stream));
        CUDA_CHECK(cudaMemsetAsync(ctx->dOuterLoopUnstable.p, 0, nS, ctx->stream));
        // The check writes pin state, row modes and row targets. The context is REUSED by the next
        // chunk, so the audit stashes everything the kernel can touch and puts it back: an
        // instrument that changes the next chunk's answer is not an instrument. dDx/dB are dead
        // here (the solve is over) and are exactly n*nS wide, so they carry the two big ones.
        CUDA_CHECK(cudaMemcpyAsync(ctx->dDx.p, ctx->dTarget.p, (size_t) n * nS * sizeof(double), cudaMemcpyDeviceToDevice, ctx->stream));
        CUDA_CHECK(cudaMemcpyAsync(ctx->dB.p, ctx->dMode.p, (size_t) n * nS * sizeof(int), cudaMemcpyDeviceToDevice, ctx->stream));
        std::vector<char> keepLt(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS), keepSc(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS), keepSm(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS);
        std::vector<double> keepSt(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS);
        CUDA_CHECK(cudaMemcpy(keepLt.data(), ctx->dReactiveLimitQLimitType.p, keepLt.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(keepSc.data(), ctx->dReactiveLimitPvPqSwitchCount.p, keepSc.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(keepSm.data(), ctx->dReactiveLimitSavedRowMode.p, keepSm.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(keepSt.data(), ctx->dReactiveLimitSavedTarget.p, keepSt.size() * sizeof(double), cudaMemcpyDeviceToHost));
        reactiveLimitsBatch<<<grid(ctx->nReactiveLimitEntries * nS), BLOCK, 0, ctx->stream>>>(
                ctx->nSecondaryVoltageControlControllerBuses > 0 ? ctx->dSecondaryVoltageControlReactiveLimitTargetV.as<double>() : nullptr,
                ctx->nReactiveLimitEntries, nS, n, REACTIVE_LIMIT_TOL, MAX_PQ_PV_SWITCH,
                ctx->dReactiveLimitVoltageRow.as<int>(),
                ctx->dReactiveLimitReactivePowerRow.p ? ctx->dReactiveLimitReactivePowerRow.as<int>() : nullptr,
                usePerScenarioReactiveLimits ? ctx->dReactiveLimitMinQPerScenario.as<double>() : ctx->dReactiveLimitMinQ.as<double>(),
                usePerScenarioReactiveLimits ? ctx->dReactiveLimitMaxQPerScenario.as<double>() : ctx->dReactiveLimitMaxQ.as<double>(),
                usePerScenarioReactiveLimits ? ctx->dReactiveLimitLoadQPerScenario.as<double>() : ctx->dReactiveLimitLoadQ.as<double>(),
                ctx->dReactiveLimitTargetV.as<double>(), ctx->dQbuf.as<double>(), ctx->dX.as<double>(), ctx->dOuterLoopCheckActive.as<char>(),
                ctx->dMode.as<int>(), ctx->dTarget.as<double>(),
                ctx->dReactiveLimitQLimitType.as<char>(), ctx->dReactiveLimitPvPqSwitchCount.as<char>(), ctx->dOuterLoopUnstable.as<char>(),
                ctx->dReactiveLimitCheckDecision.as<char>(), ctx->dReactiveLimitSavedRowMode.as<char>(),
                nReactiveLimitTargetVOverrides, nReactiveLimitTargetVOverrides > 0 ? dReactiveLimitTargetVOverrideScenarioEntry.as<int>() : nullptr,
                nReactiveLimitTargetVOverrides > 0 ? dReactiveLimitTargetVOverrideValue.as<double>() : nullptr,
                ctx->dRowSlope.as<double>(), ctx->dReactiveLimitSavedTarget.as<double>(), usePerScenarioReactiveLimits ? 1 : 0,
                ctx->nReactiveLimitRemoteEntries > 0 ? ctx->dReactiveLimitVoltageControlRole.as<char>() : nullptr,
                ctx->nReactiveLimitRemoteEntries > 0 ? ctx->dReactiveLimitComparedVoltageRow.as<int>() : nullptr,
                ctx->dReactiveLimitSvcMinB.as<double>(), ctx->dReactiveLimitSvcMaxB.as<double>(), ctx->dReactiveLimitBaseVSquared.as<double>(),
                    svcCoefDyn ? ctx->dReactiveLimitSvcMinBPerScenario.as<double>() : nullptr, svcCoefDyn ? ctx->dReactiveLimitSvcMaxBPerScenario.as<double>() : nullptr,
                ctx->nReactiveLimitIslanded > 0 ? ctx->dReactiveLimitIslanded.as<char>() : nullptr,
                ctx->reactiveLimitsRobustMode ? ctx->dReactiveLimitInitialTargetQ.as<double>() : nullptr,
                ctx->reactiveLimitsRobustMinRealisticVoltage, ctx->reactiveLimitsRobustMaxRealisticVoltage, ctx->reactiveLimitsRobustMode,
                reactiveLimitsDecisionLogScenario, ctx->dReactiveLimitsDecisionLog.p ? ctx->dReactiveLimitsDecisionLog.as<double>() : nullptr,
                ctx->dReactiveLimitsDecisionLogCount.p ? ctx->dReactiveLimitsDecisionLogCount.as<int>() : nullptr, 999999,
                    pilotSelfReactiveLimitEntryOrNull(ctx), ctx->st->nReactiveRelocationGroups,
                    ctx->nReactiveLimitLocalOverrides > 0 ? ctx->dReactiveLimitLocalOverride.as<char>() : nullptr,
                    ctx->nReactiveLimitLocalOverrides > 0 ? ctx->dReactiveLimitLocalTargetV.as<double>() : nullptr,
                    ctx->nReactiveLimitPqPvSwitchForbidden > 0 ? ctx->dReactiveLimitPqPvSwitchForbidden.as<char>() : nullptr, ctx->nReactiveLimitPqPvSwitchForbidden > 0 ? ctx->dReactiveLimitPqPvSwitchRejected.as<char>() : nullptr,
                    ctx->dReactiveLimitFlags.p ? ctx->dReactiveLimitFlags.as<char>() : nullptr,
                    ctx->nReactiveLimitComparedVoltageRowOverrides > 0 ? ctx->dReactiveLimitComparedVoltageRowOverride.as<int>() : nullptr);
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        std::vector<char> auditOuterLoopUnstable(nS), aPs(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS);
        CUDA_CHECK(cudaMemcpy(auditOuterLoopUnstable.data(), ctx->dOuterLoopUnstable.p, nS, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(aPs.data(), ctx->dReactiveLimitCheckDecision.p, aPs.size(), cudaMemcpyDeviceToHost));
        std::vector<char> altPost(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS);
        CUDA_CHECK(cudaMemcpy(altPost.data(), ctx->dReactiveLimitQLimitType.p, altPost.size(), cudaMemcpyDeviceToHost));
        // put every touched buffer back before anything else reads them
        CUDA_CHECK(cudaMemcpyAsync(ctx->dTarget.p, ctx->dDx.p, (size_t) n * nS * sizeof(double), cudaMemcpyDeviceToDevice, ctx->stream));
        CUDA_CHECK(cudaMemcpyAsync(ctx->dMode.p, ctx->dB.p, (size_t) n * nS * sizeof(int), cudaMemcpyDeviceToDevice, ctx->stream));
        CUDA_CHECK(cudaMemcpy(ctx->dReactiveLimitQLimitType.p, keepLt.data(), keepLt.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->dReactiveLimitPvPqSwitchCount.p, keepSc.data(), keepSc.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->dReactiveLimitSavedRowMode.p, keepSm.data(), keepSm.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->dReactiveLimitSavedTarget.p, keepSt.data(), keepSt.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemsetAsync(ctx->dOuterLoopUnstable.p, 0, nS, ctx->stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        std::vector<double> aq(static_cast<size_t>(n) * nS), ax(static_cast<size_t>(n) * nS), alq(ctx->nReactiveLimitEntries);
        std::vector<char> alt(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS), aisl;
        std::vector<int> avr(ctx->nReactiveLimitEntries), amode(static_cast<size_t>(n) * nS);
        CUDA_CHECK(cudaMemcpy(aq.data(), ctx->dQbuf.p, aq.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ax.data(), ctx->dX.p, ax.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(alq.data(), ctx->dReactiveLimitLoadQ.p, alq.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(alt.data(), ctx->dReactiveLimitQLimitType.p, alt.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(avr.data(), ctx->dReactiveLimitVoltageRow.p, avr.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(amode.data(), ctx->dMode.p, amode.size() * sizeof(int), cudaMemcpyDeviceToHost));
        if (ctx->nReactiveLimitIslanded > 0) {
            aisl.resize(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS);
            CUDA_CHECK(cudaMemcpy(aisl.data(), ctx->dReactiveLimitIslanded.p, aisl.size(), cudaMemcpyDeviceToHost));
        }
        std::vector<double> amn(static_cast<size_t>(ctx->nReactiveLimitEntries) * (ctx->nReactiveCapabilityCurves > 0 ? nS : 1)), amx;
        amx.resize(amn.size());
        CUDA_CHECK(cudaMemcpy(amn.data(), ctx->nReactiveCapabilityCurves > 0 ? ctx->dReactiveLimitMinQPerScenario.p : ctx->dReactiveLimitMinQ.p, amn.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(amx.data(), ctx->nReactiveCapabilityCurves > 0 ? ctx->dReactiveLimitMaxQPerScenario.p : ctx->dReactiveLimitMaxQ.p, amx.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<int> acv(ctx->nReactiveLimitEntries, -1);
        std::vector<double> avt(ctx->nReactiveLimitEntries, 0.0);
        if (ctx->dReactiveLimitComparedVoltageRow.p) {
            CUDA_CHECK(cudaMemcpy(acv.data(), ctx->dReactiveLimitComparedVoltageRow.p, acv.size() * sizeof(int), cudaMemcpyDeviceToHost));
        }
        if (ctx->dReactiveLimitTargetV.p) {
            CUDA_CHECK(cudaMemcpy(avt.data(), ctx->dReactiveLimitTargetV.p, avt.size() * sizeof(double), cudaMemcpyDeviceToHost));
        }
        for (int s = 0; s < nS; s++) {
            if (!allLive[s] || !auditOuterLoopUnstable[s]) continue;
            for (int i = 0; i < ctx->nReactiveLimitEntries; i++) {
                size_t st = (size_t) s * ctx->nReactiveLimitEntries + i;
                if (!aPs[st]) continue;
                size_t li = ctx->nReactiveCapabilityCurves > 0 ? st : (size_t) i;
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
            for (int i = 0; i < ctx->nReactiveLimitEntries; i++) {
                size_t st = (size_t) s * ctx->nReactiveLimitEntries + i;
                if (alt[st] != 0) continue;                  // pinned: its row holds the limit by construction
                if (!aisl.empty() && aisl[st]) continue;     // not a controller in this component
                int m = amode[(size_t) s * n + avr[i]];
                if (m == GpuRowMode::BUS_TARGET_Q) continue;  // row owns its own Q balance this scenario
                size_t li = ctx->nReactiveCapabilityCurves > 0 ? st : (size_t) i;
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

// OLF_GPU_PHASE_TIME: the per-scenario solve-pass histogram.
void tracePhaseTime(BatchSolverContext* ctx, DriverPass& drv) {
    const int& nS = drv.nS;
    BatchResult& res = drv.res;
    if (!ctx->driverPaused && std::getenv("OLF_GPU_PHASE_TIME") != nullptr && ctx->outerLoopSolvePasses.size() == static_cast<size_t>(nS)) {
        for (int s = 0; s < nS; s++) {
            if (ctx->outerLoopSolvePasses[s] == 0) continue;
            if (res.status[s] != GpuAcSolverStatus::CONVERGED) {                         // not finished converged on device (CPU replay,
                gOuterLoopNotConverged++;                           // max-iter, ...): its work is counted apart
                gOuterLoopNotConvergedIterations += ctx->outerLoopIterations[s];
                continue;
            }
            int k = std::min(ctx->outerLoopSolvePasses[s], 15);
            gOuterLoopPassHistogramCount[k]++;
            gOuterLoopPassHistogramFirstSolveIterations[k] += ctx->outerLoopFirstSolveIterations[s];
            gOuterLoopPassHistogramTotalIterations[k] += ctx->outerLoopIterations[s];
        }
    }
}

}  // namespace olfgpu
