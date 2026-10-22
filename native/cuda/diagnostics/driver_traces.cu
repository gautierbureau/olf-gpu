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
#include "../common/trace.cuh"

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
    if (traceOn(Trace::BATCH_TRACE)) {
        const char* kindName = "?";
        switch (loopOrder[loopIdx]) {
            case GpuOuterLoopKind::DISTRIBUTED_SLACK:           kindName = "DistributedSlack"; break;
            case GpuOuterLoopKind::FREEZING_HVDC_AC_EMULATION:  kindName = "FreezingHvdcACEmulation"; break;
            case GpuOuterLoopKind::HVDC_AC_EMULATION_LIMITS:  kindName = "AcHvdcAcEmulationLimits"; break;
            case GpuOuterLoopKind::MONITORING_VOLTAGE:   kindName = "MonitoringVoltage"; break;
            case GpuOuterLoopKind::REACTIVE_LIMITS:           kindName = "ReactiveLimits"; break;
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
            std::vector<char> unstable(nS);
            CUDA_CHECK(cudaMemcpy(unstable.data(), ctx->dOuterLoopUnstable.p, nS, cudaMemcpyDeviceToHost));
            for (char c : unstable) nUnstableScenarios += c ? 1 : 0;
        }
        int nPinned = 0;
        if (ctx->nReactiveLimitEntries > 0) {
            std::vector<char> qLimitType(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS);
            CUDA_CHECK(cudaMemcpy(qLimitType.data(), ctx->dReactiveLimitQLimitType.p, qLimitType.size(), cudaMemcpyDeviceToHost));
            for (char c : qLimitType) nPinned += c ? 1 : 0;
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
        std::vector<char> checkDecision(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS), qLimitType(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS);
        std::vector<int> voltageRow(ctx->nReactiveLimitEntries);
        std::vector<double> rowQ(static_cast<size_t>(n) * nS), loadQ(ctx->nReactiveLimitEntries), stateVector(static_cast<size_t>(n) * nS);
        CUDA_CHECK(cudaMemcpy(checkDecision.data(), ctx->dReactiveLimitCheckDecision.p, checkDecision.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(qLimitType.data(), ctx->dReactiveLimitQLimitType.p, qLimitType.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(voltageRow.data(), ctx->dReactiveLimitVoltageRow.p, voltageRow.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(rowQ.data(), ctx->dQbuf.p, rowQ.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(loadQ.data(), ctx->dReactiveLimitLoadQ.p, loadQ.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(stateVector.data(), ctx->dX.p, stateVector.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<double> minQ(static_cast<size_t>(ctx->nReactiveLimitEntries) * (ctx->nReactiveCapabilityCurves > 0 ? nS : 1));
        std::vector<double> maxQ(minQ.size());
        CUDA_CHECK(cudaMemcpy(minQ.data(), ctx->nReactiveCapabilityCurves > 0 ? ctx->dReactiveLimitMinQPerScenario.p : ctx->dReactiveLimitMinQ.p, minQ.size() * sizeof(double),
                cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(maxQ.data(), ctx->nReactiveCapabilityCurves > 0 ? ctx->dReactiveLimitMaxQPerScenario.p : ctx->dReactiveLimitMaxQ.p, maxQ.size() * sizeof(double),
                cudaMemcpyDeviceToHost));
        std::vector<int> comparedVoltageRow(ctx->nReactiveLimitEntries, -1);
        std::vector<double> targetV(ctx->nReactiveLimitEntries, 0.0);
        if (ctx->dReactiveLimitComparedVoltageRow.p) {
            CUDA_CHECK(cudaMemcpy(comparedVoltageRow.data(), ctx->dReactiveLimitComparedVoltageRow.p, comparedVoltageRow.size() * sizeof(int), cudaMemcpyDeviceToHost));
        }
        if (ctx->dReactiveLimitTargetV.p) {
            CUDA_CHECK(cudaMemcpy(targetV.data(), ctx->dReactiveLimitTargetV.p, targetV.size() * sizeof(double), cudaMemcpyDeviceToHost));
        }
        for (int s = 0; s < nS; s++) {
            bool toStderr = (s == reactiveLimitsTraceScenario);
            if (!toStderr && !(reactiveLimitsAuditOn && outerLoopCheckActive[s])) continue;
            int nReactiveLimitsDecisions = 0;
            std::string record;
            for (int entry = 0; entry < ctx->nReactiveLimitEntries; entry++) {
                size_t scenarioEntry = (size_t) s * ctx->nReactiveLimitEntries + entry;
                if (!checkDecision[scenarioEntry]) continue;
                size_t limitIndex = ctx->nReactiveCapabilityCurves > 0 ? scenarioEntry : (size_t) entry;
                double comparedV = (comparedVoltageRow[entry] >= 0) ? stateVector[(size_t) s * n + comparedVoltageRow[entry]] : 0.0;
                double q = rowQ[(size_t) s * n + voltageRow[entry]] + loadQ[entry];
                // the q the decision READ: at reactiveLimitReactivePowerRow when the entry's balance was relocated (a coupler-tree
                // child reads its parent's DUMMY_Q row), not at its V row - printing the V-row value logged a
                // relocated child as q=0 while the kernel pinned it on its real q
                static const std::vector<int> reactivePowerRow = [&] {
                    std::vector<int> v(ctx->nReactiveLimitEntries, -1);
                    if (ctx->dReactiveLimitReactivePowerRow.p) CUDA_CHECK(cudaMemcpy(v.data(), ctx->dReactiveLimitReactivePowerRow.p, v.size() * sizeof(int), cudaMemcpyDeviceToHost));
                    return v;
                }();
                int qRow = reactivePowerRow[entry] >= 0 ? reactivePowerRow[entry] : voltageRow[entry];
                double qRead = rowQ[(size_t) s * n + qRow] + loadQ[entry];
                nReactiveLimitsDecisions++;
                char buf[400];
                std::snprintf(buf, sizeof buf, "SW_PASS s=%d outer=%d entry=%d vrow=%d act=%d lt=%d q=%.15e qRow=%d qRead=%.15e min=%.15e max=%.15e v=%.15e cmpV=%.15e vt=%.15e margin=%.3e",
                             s, outer, entry, voltageRow[entry], (int) checkDecision[scenarioEntry], (int) qLimitType[scenarioEntry],
                             q, qRow, qRead, minQ[limitIndex], maxQ[limitIndex],
                             stateVector[(size_t) s * n + voltageRow[entry]], comparedV, targetV[entry], comparedV - targetV[entry]);
                if (toStderr) {
                    // the switch-back's other gates, read back for the traced entry: the row slope (a slope-SVC
                    // row never switches back) and the switch count against the cap
                    double slope = 0.0;
                    CUDA_CHECK(cudaMemcpy(&slope, ctx->dRowSlope.as<double>() + voltageRow[entry], sizeof(double), cudaMemcpyDeviceToHost));
                    char switchCount = 0;
                    CUDA_CHECK(cudaMemcpy(&switchCount, ctx->dReactiveLimitPvPqSwitchCount.as<char>() + scenarioEntry, 1, cudaMemcpyDeviceToHost));
                    char localOverride = 0;
                    double localTargetV = 0.0;
                    if (ctx->nReactiveLimitLocalOverrides > 0) {
                        CUDA_CHECK(cudaMemcpy(&localOverride, ctx->dReactiveLimitLocalOverride.as<char>() + scenarioEntry, 1, cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaMemcpy(&localTargetV, ctx->dReactiveLimitLocalTargetV.as<double>() + scenarioEntry, sizeof(double), cudaMemcpyDeviceToHost));
                    }
                    char monitored = 0;
                    CUDA_CHECK(cudaMemcpy(&monitored, ctx->dReactiveLimitFlags.as<char>() + entry, 1, cudaMemcpyDeviceToHost));
                    char islanded = 0;
                    if (ctx->nReactiveLimitIslanded > 0) {
                        CUDA_CHECK(cudaMemcpy(&islanded, ctx->dReactiveLimitIslanded.as<char>() + scenarioEntry, 1, cudaMemcpyDeviceToHost));
                    }
                    char checkActive = 0;
                    CUDA_CHECK(cudaMemcpy(&checkActive, ctx->dOuterLoopCheckActive.as<char>() + s, 1, cudaMemcpyDeviceToHost));
                    std::fprintf(stderr, "%s slope=%.6e swCount=%d localOv=%d localVt=%.12f monitored=%d islanded=%d outerLoopCheckActive=%d\n", buf, slope, (int) switchCount,
                                 (int) localOverride, localTargetV, (int) monitored, (int) islanded, (int) checkActive);
                }
                if (reactiveLimitsAuditOn && nReactiveLimitsDecisions <= 12) {
                    record += buf;
                    record += "\n";
                }
            }
            if (reactiveLimitsAuditOn) {
                char header[160];
                std::snprintf(header, sizeof header, "AUDIT_PASS s=%d outer=%d loop=%d outerLoopCheckActive=%d nReactiveLimitsDecisions=%d",
                              s, outer, loopOrder[loopIdx], (int) outerLoopCheckActive[s], nReactiveLimitsDecisions);
                auditLog[s].push_back(std::string(header) + "\n" + record);
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
            && traceOn(Trace::BATCH_TRACE)) {   // RL trace
        std::vector<char> qLimitType(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS);
        std::vector<char> voltageControlRole(ctx->nReactiveLimitEntries);
        std::vector<int> voltageRow(ctx->nReactiveLimitEntries);
        std::vector<double> rowQ(static_cast<size_t>(n) * nS);
        std::vector<double> loadQ(ctx->nReactiveLimitEntries);
        CUDA_CHECK(cudaMemcpy(qLimitType.data(), ctx->dReactiveLimitQLimitType.p, qLimitType.size(), cudaMemcpyDeviceToHost));
        if (ctx->nReactiveLimitRemoteEntries > 0) {
        CUDA_CHECK(cudaMemcpy(voltageControlRole.data(), ctx->dReactiveLimitVoltageControlRole.p, voltageControlRole.size(), cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaMemcpy(voltageRow.data(), ctx->dReactiveLimitVoltageRow.p, voltageRow.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(rowQ.data(), ctx->dQbuf.p, rowQ.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(loadQ.data(), ctx->dReactiveLimitLoadQ.p, loadQ.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<int> hMode(static_cast<size_t>(n) * nS);
        CUDA_CHECK(cudaMemcpy(hMode.data(), ctx->dMode.p, hMode.size() * sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<double> hMaxQ(static_cast<size_t>(ctx->nReactiveLimitEntries) * (ctx->nReactiveCapabilityCurves > 0 ? nS : 1));
        CUDA_CHECK(cudaMemcpy(hMaxQ.data(), ctx->nReactiveCapabilityCurves > 0 ? ctx->dReactiveLimitMaxQPerScenario.p : ctx->dReactiveLimitMaxQ.p, hMaxQ.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<char> hOuterLoopCheckActive(nS);
        CUDA_CHECK(cudaMemcpy(hOuterLoopCheckActive.data(), ctx->dOuterLoopCheckActive.p, nS, cudaMemcpyDeviceToHost));
        // The STATE the pin decision is taken at. Without it, a q that disagrees with OLF cannot be
        // told apart from a q evaluated at a different (or not yet converged) operating point - the
        // same quantity at a different V is not evidence of a wrong formula.
        std::vector<double> hStateVector(static_cast<size_t>(n) * nS);
        CUDA_CHECK(cudaMemcpy(hStateVector.data(), ctx->dX.p, hStateVector.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<double> hTarget(static_cast<size_t>(n) * nS);
        CUDA_CHECK(cudaMemcpy(hTarget.data(), ctx->dTarget.p, hTarget.size() * sizeof(double), cudaMemcpyDeviceToHost));
        for (int s = 0; s < nS; s++) {
            for (int entry = 0; entry < ctx->nReactiveLimitEntries; entry++) {
                if (voltageControlRole[entry] || nS == 1) {
                    std::fprintf(stderr, "RL_TRACE s=%d entry=%d remote=%d lt=%d switched=%d q=%.9f mode=%d tol=%.3e maxQ=%.6f nReactiveCapabilityCurves=%d outerLoopCheckActive=%d v=%.9f qbuf=%.9f\n",
                                 s, entry, (int) voltageControlRole[entry], (int) qLimitType[(size_t) s * ctx->nReactiveLimitEntries + entry], (int) outerLoopUnstable[s],
                                 rowQ[(size_t) s * n + voltageRow[entry]] + loadQ[entry], hMode[(size_t) s * n + voltageRow[entry]], reactiveLimitTol,
                                 hMaxQ[ctx->nReactiveCapabilityCurves > 0 ? (size_t) s * ctx->nReactiveLimitEntries + entry : (size_t) entry], ctx->nReactiveCapabilityCurves, (int) hOuterLoopCheckActive[s],
                                 hStateVector[(size_t) s * n + voltageRow[entry]], rowQ[(size_t) s * n + voltageRow[entry]]);
                }
            }
        }
        // PSG: every pilot-self group's layout this pass — the pilot's pin state, its own row's mode/target,
        // whether updatePilotTargetVEquationBatch flipped it (pilotTargetVDeactivated), and every candidate row with its mode, so a
        // "control lost" verdict can be checked against the members it was taken over.
        if (ctx->hasPilotSelf && ctx->dPilotSelfReactiveLimitEntry.p && ctx->structure->nReactiveRelocationGroups > 0 && !ctx->structure->hReactiveRelocationGroupStart.empty()
                && traceOn(Trace::BATCH_TRACE)) {
            std::vector<int> pilotSelfEntry(ctx->structure->nReactiveRelocationGroups, -1);
            std::vector<char> hPilotTargetVDeactivated(static_cast<size_t>(ctx->structure->nReactiveRelocationGroups) * nS, 0);
            CUDA_CHECK(cudaMemcpy(pilotSelfEntry.data(), ctx->dPilotSelfReactiveLimitEntry.p, pilotSelfEntry.size() * sizeof(int), cudaMemcpyDeviceToHost));
            if (ctx->dReactiveRelocationPilotTargetVDeactivated.p) {
                CUDA_CHECK(cudaMemcpy(hPilotTargetVDeactivated.data(), ctx->dReactiveRelocationPilotTargetVDeactivated.p, hPilotTargetVDeactivated.size(), cudaMemcpyDeviceToHost));
            }
            for (int s = 0; s < nS; s++) {
                for (int g = 0; g < ctx->structure->nReactiveRelocationGroups; g++) {
                    int e = pilotSelfEntry[g];
                    if (e < 0) continue;
                    int p0 = ctx->structure->hReactiveRelocationGroupStart[g];
                    int count = ctx->structure->hReactiveRelocationGroupStart[g + 1] - p0;
                    int pilotRow = ctx->structure->hReactiveRelocationCandidateRows[p0];
                    std::string candidates;
                    int pick = 0;
                    for (int c = 1; c < count; c++) {
                        int r = ctx->structure->hReactiveRelocationCandidateRows[p0 + c];
                        int m = hMode[(size_t) s * n + r];
                        if (pick == 0 && m != GpuRowMode::FIXED_VARIABLE && m != GpuRowMode::BUS_TARGET_Q) pick = c;
                        char b[48];
                        std::snprintf(b, sizeof b, " %d:m%d/t%.4f", r, m, hTarget[(size_t) s * n + r]);
                        candidates += b;
                    }
                    std::fprintf(stderr, "PSG s=%d outer=%d g=%d entry=%d lt=%d pilotRow=%d mode=%d target=%.9f pilotTargetVDeactivated=%d q=%.9f pick=%d cands=[%s ]\n",
                                 s, outer, g, e, (int) qLimitType[(size_t) s * ctx->nReactiveLimitEntries + e], pilotRow, hMode[(size_t) s * n + pilotRow],
                                 hTarget[(size_t) s * n + pilotRow], (int) hPilotTargetVDeactivated[(size_t) s * ctx->structure->nReactiveRelocationGroups + g],
                                 rowQ[(size_t) s * n + voltageRow[e]] + loadQ[e], pick, candidates.c_str());
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
        int count = 0;
        CUDA_CHECK(cudaMemcpy(&count, ctx->dReactiveLimitsDecisionLogCount.p, sizeof(int), cudaMemcpyDeviceToHost));
        int keep = count > REACTIVE_LIMITS_DECISION_LOG_CAPACITY ? REACTIVE_LIMITS_DECISION_LOG_CAPACITY : count;
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
        std::fprintf(stderr, "SW_LOG_END s=%d records=%d (cap %d)\n", reactiveLimitsDecisionLogScenario, count, REACTIVE_LIMITS_DECISION_LOG_CAPACITY);
        CUDA_CHECK(cudaMemset(ctx->dReactiveLimitsDecisionLogCount.p, 0, sizeof(int)));
    }
}

// OLF_GPU_RL_AUDIT: the end-of-solve audit of the reactive-limit decisions.
void traceReactiveLimitsAudit(BatchSolverContext* ctx, DriverPass& drv, std::vector<std::vector<std::string>>& auditLog) {
    const double& REACTIVE_LIMIT_TOL = drv.reactiveLimitTol;
    const int& MAX_PQ_PV_SWITCH = drv.maxPqPvSwitch;
    const bool& hasPerScenarioSvcSusceptanceLimits = drv.hasPerScenarioSvcSusceptanceLimits;
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
    if (ctx->nReactiveLimitEntries > 0 && traceOn(Trace::RL_AUDIT)) {
        double auditTol = std::atof(traceValue(Trace::RL_AUDIT));
        if (auditTol <= 0.0) auditTol = 1e-6;
        std::vector<char> allLive(nS), auditDumped(nS, 0);
        for (int s = 0; s < nS; s++) allLive[s] = (res.status[s] == GpuAcSolverStatus::CONVERGED) ? 1 : 0;
        CUDA_CHECK(cudaMemcpy(ctx->dOuterLoopCheckActive.p, allLive.data(), nS * sizeof(char), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemsetAsync(ctx->dQbuf.p, 0, nTot * sizeof(double), ctx->stream));
        if (ctx->structure->closedBranch.elementCount) fillClosedBranchBatch<<<grid(ctx->structure->closedBranch.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->closedBranch.elementCount, nS, n, nnz, ctx->nElements, ctx->closedBranchOffset, ctx->structure->closedBranch.parameters.as<double>(), ctx->structure->closedBranch.indices.as<int>(), ctx->structure->closedBranch.destination.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dOuterLoopCheckActive.as<char>(), ctx->nClosedBranchParameterOverrides, ctx->dClosedBranchParameterOverrideScenarioElement.as<int>(), ctx->dClosedBranchParameterOverrideValues.as<double>(), nullptr, nullptr, nullptr, 0, nullptr, nullptr, ReactiveRelocationTables{});
        if (ctx->structure->openBranch.elementCount) fillOpenBranchBatch<<<grid(ctx->structure->openBranch.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->openBranch.elementCount, nS, n, nnz, ctx->nElements, ctx->openBranchOffset, ctx->structure->openBranch.parameters.as<double>(), ctx->structure->openBranch.indices.as<int>(), ctx->structure->openBranch.destination.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dOuterLoopCheckActive.as<char>(), nullptr, nullptr, nullptr, 0, ReactiveRelocationTables{});
        if (ctx->structure->shunt.elementCount) fillShuntBatch<<<grid(ctx->structure->shunt.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->shunt.elementCount, nS, n, nnz, ctx->nElements, ctx->shuntOffset, ctx->structure->shunt.parameters.as<double>(), ctx->structure->shunt.indices.as<int>(), ctx->structure->shunt.destination.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dOuterLoopCheckActive.as<char>(), ctx->nShuntAdmittanceOverrides, ctx->dShuntAdmittanceOverrideScenarioElement.as<int>(), ctx->dShuntAdmittanceOverrideDelta.as<double>(), nullptr, nullptr, nullptr, 0, ReactiveRelocationTables{});
        if (ctx->structure->nLoadModelTerms) fillLoadModelBatch<<<grid(ctx->structure->nLoadModelTerms * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->nLoadModelTerms, nS, n, nnz, ctx->structure->dLoadModelParameters.as<double>(), ctx->structure->dLoadModelIndices.as<int>(), ctx->structure->dLoadModelDestination.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), nullptr, nullptr, nullptr);
        if (ctx->nZeroImpedanceBranch) fillZeroImpBatch<<<grid(ctx->nZeroImpedanceBranch * nS), BLOCK, 0, ctx->stream>>>(ctx->nZeroImpedanceBranch, nS, n, nnz, ctx->structure->dZeroImpedanceBranchParameters.as<double>(), ctx->structure->dZeroImpedanceBranchIndices.as<int>(), ctx->structure->dZeroImpedanceBranchDestination.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), ctx->dOuterLoopCheckActive.as<char>(), ctx->nBaseClosedZeroImpedanceBranches, ctx->nZeroImpedanceBranchCloseOverrides, ctx->dZeroImpedanceBranchCloseScenarioElement.as<int>(), ctx->nZeroImpedanceBranchOpenOverrides, ctx->dZeroImpedanceBranchOpenScenarioElement.as<int>(), nullptr, nullptr, nullptr, 0, ReactiveRelocationTables{}, ctx->nSpanningTreeOverrides, ctx->nSpanningTreeOverrides ? ctx->dSpanningTreeOverrideScenarioElementValue.as<int>() : nullptr, ctx->hasZeroImpedanceBranchMasks ? ctx->dZeroImpedanceBranchOpenMask.as<char>() : nullptr, ctx->hasZeroImpedanceBranchMasks ? ctx->dZeroImpedanceBranchCloseMask.as<char>() : nullptr, ctx->hasZeroImpedanceBranchMasks ? ctx->dSpanningTreeOverrideMask.as<char>() : nullptr);
        if (ctx->nRemoteMemberReactiveTerms > 0) {
            fillRemoteMemberQBatch<<<grid(ctx->nRemoteMemberReactiveTerms * nS), BLOCK, 0, ctx->stream>>>(ctx->nRemoteMemberReactiveTerms, nS, n,
                    ctx->nElements, ctx->closedBranchOffset, ctx->openBranchOffset,
                    ctx->dRemoteMemberReactiveTermRow.as<int>(), ctx->dRemoteMemberReactiveTermElement.as<int>(), ctx->dRemoteMemberReactiveTermSide.as<int>(),
                    ctx->dRemoteMemberReactiveTermKind.as<int>(), ctx->dRemoteMemberReactiveTermWeight.as<double>(),
                    ctx->structure->closedBranch.parameters.as<double>(), ctx->structure->closedBranch.indices.as<int>(), ctx->structure->shunt.parameters.as<double>(), ctx->structure->shunt.indices.as<int>(),
                    ctx->structure->openBranch.parameters.as<double>(), ctx->structure->openBranch.indices.as<int>(),
                    ctx->dX.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dOuterLoopCheckActive.as<char>(),
                    ctx->nClosedBranchParameterOverrides, ctx->dClosedBranchParameterOverrideScenarioElement.as<int>(), ctx->dClosedBranchParameterOverrideValues.as<double>(),
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
        std::vector<char> keepQLimitType(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS), keepSwitchCount(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS), keepSavedRowMode(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS);
        std::vector<double> keepSavedTarget(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS);
        CUDA_CHECK(cudaMemcpy(keepQLimitType.data(), ctx->dReactiveLimitQLimitType.p, keepQLimitType.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(keepSwitchCount.data(), ctx->dReactiveLimitPvPqSwitchCount.p, keepSwitchCount.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(keepSavedRowMode.data(), ctx->dReactiveLimitSavedRowMode.p, keepSavedRowMode.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(keepSavedTarget.data(), ctx->dReactiveLimitSavedTarget.p, keepSavedTarget.size() * sizeof(double), cudaMemcpyDeviceToHost));
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
                    hasPerScenarioSvcSusceptanceLimits ? ctx->dReactiveLimitSvcMinBPerScenario.as<double>() : nullptr, hasPerScenarioSvcSusceptanceLimits ? ctx->dReactiveLimitSvcMaxBPerScenario.as<double>() : nullptr,
                ctx->nReactiveLimitIslanded > 0 ? ctx->dReactiveLimitIslanded.as<char>() : nullptr,
                ctx->reactiveLimitsRobustMode ? ctx->dReactiveLimitInitialTargetQ.as<double>() : nullptr,
                ctx->reactiveLimitsRobustMinRealisticVoltage, ctx->reactiveLimitsRobustMaxRealisticVoltage, ctx->reactiveLimitsRobustMode,
                reactiveLimitsDecisionLogScenario, ctx->dReactiveLimitsDecisionLog.p ? ctx->dReactiveLimitsDecisionLog.as<double>() : nullptr,
                ctx->dReactiveLimitsDecisionLogCount.p ? ctx->dReactiveLimitsDecisionLogCount.as<int>() : nullptr, 999999,
                    pilotSelfReactiveLimitEntryOrNull(ctx), ctx->structure->nReactiveRelocationGroups,
                    ctx->nReactiveLimitLocalOverrides > 0 ? ctx->dReactiveLimitLocalOverride.as<char>() : nullptr,
                    ctx->nReactiveLimitLocalOverrides > 0 ? ctx->dReactiveLimitLocalTargetV.as<double>() : nullptr,
                    ctx->nReactiveLimitPqPvSwitchForbidden > 0 ? ctx->dReactiveLimitPqPvSwitchForbidden.as<char>() : nullptr, ctx->nReactiveLimitPqPvSwitchForbidden > 0 ? ctx->dReactiveLimitPqPvSwitchRejected.as<char>() : nullptr,
                    ctx->dReactiveLimitFlags.p ? ctx->dReactiveLimitFlags.as<char>() : nullptr,
                    ctx->nReactiveLimitComparedVoltageRowOverrides > 0 ? ctx->dReactiveLimitComparedVoltageRowOverride.as<int>() : nullptr,
                    ctx->nReMergedVoltageTargets > 0 ? ctx->dReactiveLimitHoldsReMergedVoltageTarget.as<char>() : nullptr,
                    ctx->nReMergedVoltageTargets > 0 ? ctx->dReactiveLimitReMergedVoltageRow.as<int>() : nullptr);
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        std::vector<char> auditOuterLoopUnstable(nS), auditCheckDecision(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS);
        CUDA_CHECK(cudaMemcpy(auditOuterLoopUnstable.data(), ctx->dOuterLoopUnstable.p, nS, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(auditCheckDecision.data(), ctx->dReactiveLimitCheckDecision.p, auditCheckDecision.size(), cudaMemcpyDeviceToHost));
        std::vector<char> auditQLimitTypeAfter(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS);
        CUDA_CHECK(cudaMemcpy(auditQLimitTypeAfter.data(), ctx->dReactiveLimitQLimitType.p, auditQLimitTypeAfter.size(), cudaMemcpyDeviceToHost));
        // put every touched buffer back before anything else reads them
        CUDA_CHECK(cudaMemcpyAsync(ctx->dTarget.p, ctx->dDx.p, (size_t) n * nS * sizeof(double), cudaMemcpyDeviceToDevice, ctx->stream));
        CUDA_CHECK(cudaMemcpyAsync(ctx->dMode.p, ctx->dB.p, (size_t) n * nS * sizeof(int), cudaMemcpyDeviceToDevice, ctx->stream));
        CUDA_CHECK(cudaMemcpy(ctx->dReactiveLimitQLimitType.p, keepQLimitType.data(), keepQLimitType.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->dReactiveLimitPvPqSwitchCount.p, keepSwitchCount.data(), keepSwitchCount.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->dReactiveLimitSavedRowMode.p, keepSavedRowMode.data(), keepSavedRowMode.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->dReactiveLimitSavedTarget.p, keepSavedTarget.data(), keepSavedTarget.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemsetAsync(ctx->dOuterLoopUnstable.p, 0, nS, ctx->stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        std::vector<double> auditRowQ(static_cast<size_t>(n) * nS), auditStateVector(static_cast<size_t>(n) * nS), auditLoadQ(ctx->nReactiveLimitEntries);
        std::vector<char> auditQLimitTypeBefore(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS), auditIslanded;
        std::vector<int> auditVoltageRow(ctx->nReactiveLimitEntries), auditMode(static_cast<size_t>(n) * nS);
        CUDA_CHECK(cudaMemcpy(auditRowQ.data(), ctx->dQbuf.p, auditRowQ.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(auditStateVector.data(), ctx->dX.p, auditStateVector.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(auditLoadQ.data(), ctx->dReactiveLimitLoadQ.p, auditLoadQ.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(auditQLimitTypeBefore.data(), ctx->dReactiveLimitQLimitType.p, auditQLimitTypeBefore.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(auditVoltageRow.data(), ctx->dReactiveLimitVoltageRow.p, auditVoltageRow.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(auditMode.data(), ctx->dMode.p, auditMode.size() * sizeof(int), cudaMemcpyDeviceToHost));
        if (ctx->nReactiveLimitIslanded > 0) {
            auditIslanded.resize(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS);
            CUDA_CHECK(cudaMemcpy(auditIslanded.data(), ctx->dReactiveLimitIslanded.p, auditIslanded.size(), cudaMemcpyDeviceToHost));
        }
        std::vector<double> auditMinQ(static_cast<size_t>(ctx->nReactiveLimitEntries) * (ctx->nReactiveCapabilityCurves > 0 ? nS : 1)), auditMaxQ;
        auditMaxQ.resize(auditMinQ.size());
        CUDA_CHECK(cudaMemcpy(auditMinQ.data(), ctx->nReactiveCapabilityCurves > 0 ? ctx->dReactiveLimitMinQPerScenario.p : ctx->dReactiveLimitMinQ.p, auditMinQ.size() * sizeof(double),
                cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(auditMaxQ.data(), ctx->nReactiveCapabilityCurves > 0 ? ctx->dReactiveLimitMaxQPerScenario.p : ctx->dReactiveLimitMaxQ.p, auditMaxQ.size() * sizeof(double),
                cudaMemcpyDeviceToHost));
        std::vector<int> auditComparedVoltageRow(ctx->nReactiveLimitEntries, -1);
        std::vector<double> auditTargetV(ctx->nReactiveLimitEntries, 0.0);
        if (ctx->dReactiveLimitComparedVoltageRow.p) {
            CUDA_CHECK(cudaMemcpy(auditComparedVoltageRow.data(), ctx->dReactiveLimitComparedVoltageRow.p, auditComparedVoltageRow.size() * sizeof(int), cudaMemcpyDeviceToHost));
        }
        if (ctx->dReactiveLimitTargetV.p) {
            CUDA_CHECK(cudaMemcpy(auditTargetV.data(), ctx->dReactiveLimitTargetV.p, auditTargetV.size() * sizeof(double), cudaMemcpyDeviceToHost));
        }
        for (int s = 0; s < nS; s++) {
            if (!allLive[s] || !auditOuterLoopUnstable[s]) continue;
            for (int i = 0; i < ctx->nReactiveLimitEntries; i++) {
                size_t scenarioEntry = (size_t) s * ctx->nReactiveLimitEntries + i;
                if (!auditCheckDecision[scenarioEntry]) continue;
                size_t limitIndex = ctx->nReactiveCapabilityCurves > 0 ? scenarioEntry : (size_t) i;
                double comparedV = (auditComparedVoltageRow[i] >= 0) ? auditStateVector[(size_t) s * n + auditComparedVoltageRow[i]] : 0.0;
                // act 1 = a pin this check owed, 2 = an unpin, 3 = a moved limit it owed
                std::fprintf(stderr, "RL_OWED s=%d entry=%d vrow=%d act=%d ltBefore=%d ltAfter=%d q=%.15e min=%.15e max=%.15e v=%.15e cmpV=%.15e vt=%.15e margin=%.3e\n",
                             s, i, auditVoltageRow[i], (int) auditCheckDecision[scenarioEntry], (int) auditQLimitTypeBefore[scenarioEntry], (int) auditQLimitTypeAfter[scenarioEntry],
                             auditRowQ[(size_t) s * n + auditVoltageRow[i]] + auditLoadQ[i], auditMinQ[limitIndex], auditMaxQ[limitIndex],
                             auditStateVector[(size_t) s * n + auditVoltageRow[i]], comparedV, auditTargetV[i], comparedV - auditTargetV[i]);
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
                size_t scenarioEntry = (size_t) s * ctx->nReactiveLimitEntries + i;
                if (auditQLimitTypeBefore[scenarioEntry] != 0) continue;                  // pinned: its row holds the limit by construction
                if (!auditIslanded.empty() && auditIslanded[scenarioEntry]) continue;     // not a controller in this component
                int m = auditMode[(size_t) s * n + auditVoltageRow[i]];
                if (m == GpuRowMode::BUS_TARGET_Q) continue;  // row owns its own Q balance this scenario
                size_t limitIndex = ctx->nReactiveCapabilityCurves > 0 ? scenarioEntry : (size_t) i;
                double q = auditRowQ[(size_t) s * n + auditVoltageRow[i]] + auditLoadQ[i];
                double over = q < auditMinQ[limitIndex] ? auditMinQ[limitIndex] - q : (q > auditMaxQ[limitIndex] ? q - auditMaxQ[limitIndex] : 0.0);
                if (over > auditTol) {
                    std::fprintf(stderr, "RL_AUDIT s=%d entry=%d vrow=%d mode=%d q=%.15e min=%.15e max=%.15e over=%.6e v=%.15e\n",
                                 s, i, auditVoltageRow[i], m, q, auditMinQ[limitIndex], auditMaxQ[limitIndex], over, auditStateVector[(size_t) s * n + auditVoltageRow[i]]);
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
    if (traceOn(Trace::PHASE_TIME) && ctx->outerLoopSolvePasses.size() == static_cast<size_t>(nS)) {
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
