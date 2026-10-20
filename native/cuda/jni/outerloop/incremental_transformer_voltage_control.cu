// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See incremental_transformer_voltage_control.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/codes/gpu_linear_solve_mode.cuh"
#include "../common/codes/gpu_pi_model_column.cuh"
#include "../solver/driver_steps.cuh"
#include "sensitivity.cuh"
#include "incremental_transformer_voltage_control.cuh"

namespace olfgpu {

// Per (scenario, controller): check if |targetV - V| > halfDeadband. Writes 1 to outOfDeadband
// for each out-of-deadband pair, 0 otherwise. Also writes diffV per pair for the tap decision.
__global__ void incrementalTransformerVoltageControlDeadbandCheckBatch(int nIncrementalTransformerVoltageControlControllerBranches, int nScenarios, int n, int nElements,
        const int* incrementalTransformerVoltageControlControlledBusVRow, const double* incrementalTransformerVoltageControlTargetV, const double* incrementalTransformerVoltageControlHalfTargetDeadband,
        const int* incrementalTransformerVoltageControlElement, const char* disabled,
        const double* x, const char* active,
        char* outOfDeadband, double* diffV) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nIncrementalTransformerVoltageControlControllerBranches * nScenarios) return;
    int c = tid % nIncrementalTransformerVoltageControlControllerBranches;
    int s = tid / nIncrementalTransformerVoltageControlControllerBranches;
    if (!active[s]) {
        outOfDeadband[tid] = 0;
        diffV[tid] = 0.0;
        return;
    }
    if (disabled[(size_t) s * nElements + incrementalTransformerVoltageControlElement[c]]) {    // outaged in this scenario
        outOfDeadband[tid] = 0;
        diffV[tid] = 0.0;                                    // never read as a group's remaining deviation
        return;
    }
    double v = x[(size_t) s * n + incrementalTransformerVoltageControlControlledBusVRow[c]];
    double dv = incrementalTransformerVoltageControlTargetV[c] - v;
    if (fabs(dv) > incrementalTransformerVoltageControlHalfTargetDeadband[c]) {
        outOfDeadband[tid] = 1;
        diffV[tid] = dv;
    } else {
        outOfDeadband[tid] = 0;
        diffV[tid] = 0.0;
    }
}

// Per (scenario, controller): given the sensitivity dV/dρ, find the closest tap and move there.
// Updates currentTap, writes the new r1 to newR1, and sets moved[s] = 1 if any scenario moved.
// Single-controller-per-bus simplification for now; the multi-controller round-robin needs the
// group structure and is a follow-up.
// OLF checkAndAdjustControlledBus, one thread per (scenario, controlled-bus group). One controller:
// adjustWithOneController (full diffV, maxTapShift). Several: adjustWithSeveralControllers — the
// one-tap-at-a-time round-robin in merged order, remainingDiffV tracked through each controller's
// sensitivity, until a pass moves nothing. Both apply MIN_SENSI_FILTER and the ratchet
// (ControllerContext.updateAllowedDirection, MAX_DIRECTION_CHANGE 3); updateTapPositionToReachNewR1
// = the closest r1 within the shift bound and the allowed range, strictly closer than the current one.
__device__ inline int incrementalTransformerVoltageControlMoveTap(const int* tapStart, const double* tapRows, size_t st, int c, double deltaR1, int maxShift,
                                 int* currentTap, int* currentDirection, int* directionChangeCount, int* allowedDirection) {
    int cur = currentTap[st];
    int nTaps = tapStart[c + 1] - tapStart[c];
    int base = tapStart[c];
    double curR1 = tapRows[(size_t) (base + cur) * GpuPiModelColumn::COUNT + GpuPiModelColumn::R1];
    double targetR1 = curR1 + deltaR1;
    int lo = allowedDirection[st] > 0 ? cur : 0;
    int hi = allowedDirection[st] < 0 ? cur : nTaps - 1;
    int best = cur;
    double bestDist = fabs(targetR1 - curR1);
    for (int i = lo; i <= hi; i++) {
        if (abs(i - cur) > maxShift) continue;
        double dist = fabs(targetR1 - tapRows[(size_t) (base + i) * GpuPiModelColumn::COUNT + GpuPiModelColumn::R1]);
        if (dist < bestDist) { best = i; bestDist = dist; }
    }
    if (best == cur) return 0;
    currentTap[st] = best;
    int direction = best > cur ? 1 : -1;
    if (directionChangeCount[st] < 3) {
        if (currentDirection[st] != 0 && currentDirection[st] != direction) directionChangeCount[st]++;
        currentDirection[st] = direction;
    } else {
        allowedDirection[st] = direction;
    }
    return direction;
}

__global__ void incrementalTransformerVoltageControlGroupDecisionBatch(int nGroups, int nScenarios, int nIncrementalTransformerVoltageControlControllerBranches,
        const int* groupControllerStart, const int* groupController,
        const char* outOfDeadband, const double* diffV, const double* sensitivity, const double* halfTargetDeadband,
        const int* tapStart, const double* tapRows,
        int* currentTap, int* currentDirection, int* directionChangeCount, int* allowedDirection,
        int maxTapShift, double minSensiFilter, char* moved) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nGroups * nScenarios) return;
    int g = tid % nGroups;
    int s = tid / nGroups;
    int k0 = groupControllerStart[g];
    int k1 = groupControllerStart[g + 1];
    // the group's live controllers (outOfDeadband is per controller: 0 when outaged or inside the band)
    int nLive = 0;
    int lastLive = -1;
    for (int k = k0; k < k1; k++) {
        int c = groupController[k];
        if (outOfDeadband[(size_t) s * nIncrementalTransformerVoltageControlControllerBranches + c]) { nLive++; lastLive = c; }
    }
    if (nLive == 0) return;
    // OLF picks the path on the controllers LEFT after the outage (getMergedControllerElements minus the
    // disabled ones): one survivor of a shared group adjusts alone. The static group size chose the shared path
    // for it, and the shared path read the group's deviation off the FIRST member - the outaged one, whose
    // deadband check writes nothing: 0 on a fresh context (the survivor never moved), a stale value from an
    // earlier chunk on a reused one (chunk-dependent moves).
    if (nLive == 1) {                                        // adjustWithOneController
        size_t st = (size_t) s * nIncrementalTransformerVoltageControlControllerBranches + lastLive;
        double sv = sensitivity[st];
        if (fabs(sv) < minSensiFilter) return;
        if (incrementalTransformerVoltageControlMoveTap(tapStart, tapRows, st, lastLive, diffV[st] / sv, maxTapShift,
                       currentTap, currentDirection, directionChangeCount, allowedDirection) != 0) moved[s] = 1;
        return;
    }
    // adjustWithSeveralControllers: one tap per controller per pass, in merged order; the deviation and the
    // deadband are the controlled bus's, read off any live member (every live member of a group controls it)
    int firstLive = -1;
    for (int k = k0; k < k1 && firstLive < 0; k++) {
        if (outOfDeadband[(size_t) s * nIncrementalTransformerVoltageControlControllerBranches + groupController[k]]) firstLive = groupController[k];
    }
    size_t st0 = (size_t) s * nIncrementalTransformerVoltageControlControllerBranches + firstLive;
    double remaining = diffV[st0];
    double hdb = halfTargetDeadband[firstLive];
    bool changed = true;
    bool any = false;
    while (changed) {
        changed = false;
        for (int k = k0; k < k1; k++) {
            int c = groupController[k];
            size_t st = (size_t) s * nIncrementalTransformerVoltageControlControllerBranches + c;
            if (!outOfDeadband[st]) continue;                      // outaged member
            if (fabs(remaining) <= hdb) continue;
            double sv = sensitivity[st];
            if (fabs(sv) < minSensiFilter) continue;
            double prevR1 = tapRows[(size_t) (tapStart[c] + currentTap[st]) * GpuPiModelColumn::COUNT + GpuPiModelColumn::R1];
            if (incrementalTransformerVoltageControlMoveTap(tapStart, tapRows, st, c, remaining / sv, 1,
                           currentTap, currentDirection, directionChangeCount, allowedDirection) != 0) {
                remaining -= (tapRows[(size_t) (tapStart[c] + currentTap[st]) * GpuPiModelColumn::COUNT + GpuPiModelColumn::R1] - prevR1) * sv;
                changed = true;
                any = true;
            }
        }
    }
    if (any) moved[s] = 1;
}

// Per scenario: read the controlled bus's dV out of the unit-RHS solution for controller c.
__global__ void incrementalTransformerVoltageControlGatherSensitivityBatch(int nScenarios, int n, int c, int nIncrementalTransformerVoltageControlControllerBranches,
                                    const int* incrementalTransformerVoltageControlControlledBusVRow, const double* dx, double* sensitivity) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    sensitivity[(size_t) s * nIncrementalTransformerVoltageControlControllerBranches + c] = dx[(size_t) s * n + incrementalTransformerVoltageControlControlledBusVRow[c]];
}

// Per (scenario, controller) whose tap moved this pass: pin the rho identity row's TARGET at the
// new tap's r1 and seed the rho STATE there — exactly what the host orchestrator writes before its
// warm re-solve (ovVal.add(row[6]); x0Full[rhoRow] = row[6]). Then remember the tap as applied.
__global__ void incrementalTransformerVoltageControlApplyRhoBatch(int nIncrementalTransformerVoltageControlControllerBranches, int nScenarios, int n,
                                 const int* incrementalTransformerVoltageControlRhoRow, const int* tapStart, const double* tapRows,
                                 const int* currentTap, int* appliedTap, double* target, double* x) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nIncrementalTransformerVoltageControlControllerBranches * nScenarios) return;
    int c = tid % nIncrementalTransformerVoltageControlControllerBranches;
    int s = tid / nIncrementalTransformerVoltageControlControllerBranches;
    int t = currentTap[tid];
    if (t == appliedTap[tid]) return;
    double r1 = tapRows[(size_t) (tapStart[c] + t) * GpuPiModelColumn::COUNT + GpuPiModelColumn::R1];
    target[(size_t) s * n + incrementalTransformerVoltageControlRhoRow[c]] = r1;
    x[(size_t) s * n + incrementalTransformerVoltageControlRhoRow[c]] = r1;
    appliedTap[tid] = t;
}

/** Rebuild the device tap-override list = [every (scenario, controller) whose current tap differs
 *  from its base tap, as the tap's full 8-double pi row] ++ [the caller's own overrides]. The TVC
 *  entries go FIRST because findTapOverride returns the first (slot, elem) match, so a contingency
 *  tap action on the same transformer must not shadow the loop's move. Runs after every pass that
 *  moved a tap, and on a driver RESUME (the caller's list is re-uploaded at solve entry and would
 *  otherwise erase the moves the paused sequence had already made). */
void rebuildTapOverrides(BatchSolverContext* ctx) {
    int nS = ctx->nScenarios;
    int nIncrementalTransformerVoltageControlControllerBranches = ctx->nIncrementalTransformerVoltageControlControllerBranches;
    std::vector<int> slot;
    std::vector<double> val;
    if (nIncrementalTransformerVoltageControlControllerBranches > 0) {
        std::vector<int> cur(static_cast<size_t>(nS) * nIncrementalTransformerVoltageControlControllerBranches);
        CUDA_CHECK(cudaMemcpy(cur.data(), ctx->dIncrementalTransformerVoltageControlCurrentTap.p, cur.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (int s = 0; s < nS; s++) {
            for (int c = 0; c < nIncrementalTransformerVoltageControlControllerBranches; c++) {
                int t = cur[static_cast<size_t>(s) * nIncrementalTransformerVoltageControlControllerBranches + c];
                if (t == ctx->hIncrementalTransformerVoltageControlBaseTap[c]) continue;
                slot.push_back(s);
                slot.push_back(ctx->hIncrementalTransformerVoltageControlElement[c]);
                const double* row = &ctx->hIncrementalTransformerVoltageControlTapRows[static_cast<size_t>(ctx->hIncrementalTransformerVoltageControlTapStart[c] + t) * GpuPiModelColumn::COUNT];
                val.insert(val.end(), row, row + GpuPiModelColumn::COUNT);
            }
        }
    }
    if (ctx->nPhaseControlControllerBranches > 0) {                                     // the continuous phase loop's rounded/shifted taps
        std::vector<int> cur(static_cast<size_t>(nS) * ctx->nPhaseControlControllerBranches);
        CUDA_CHECK(cudaMemcpy(cur.data(), ctx->dPhaseControlTap.p, cur.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (int s = 0; s < nS; s++) {
            for (int c = 0; c < ctx->nPhaseControlControllerBranches; c++) {
                int t = cur[static_cast<size_t>(s) * ctx->nPhaseControlControllerBranches + c];
                if (t == ctx->hPhaseControlBaseTap[c]) continue;
                slot.push_back(s);
                slot.push_back(ctx->hPhaseControlElement[c]);
                const double* row = &ctx->hPhaseControlTapRows[static_cast<size_t>(ctx->hPhaseControlTapStart[c] + t) * GpuPiModelColumn::COUNT];
                val.insert(val.end(), row, row + GpuPiModelColumn::COUNT);
            }
        }
    }
    if (ctx->nIncrementalFlowControlControllerBranches > 0) {                                     // the flow-observed loops' moved taps too
        std::vector<int> cur(static_cast<size_t>(nS) * ctx->nIncrementalFlowControlControllerBranches);
        CUDA_CHECK(cudaMemcpy(cur.data(), ctx->dIncrementalFlowControlCurrentTap.p, cur.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (int s = 0; s < nS; s++) {
            for (int c = 0; c < ctx->nIncrementalFlowControlControllerBranches; c++) {
                int t = cur[static_cast<size_t>(s) * ctx->nIncrementalFlowControlControllerBranches + c];
                if (t == ctx->hIncrementalFlowControlBaseTap[c]) continue;
                slot.push_back(s);
                slot.push_back(ctx->hIncrementalFlowControlElement[c]);
                const double* row = &ctx->hIncrementalFlowControlTapRows[static_cast<size_t>(ctx->hIncrementalFlowControlTapStart[c] + t) * GpuPiModelColumn::COUNT];
                val.insert(val.end(), row, row + GpuPiModelColumn::COUNT);
            }
        }
    }
    if (ctx->nTransformerVoltageControlMembers > 0) {                                     // the discrete TVC loop's fixed (rounded/clamped) taps
        std::vector<int> st(static_cast<size_t>(nS) * ctx->nTransformerVoltageControlMembers), cur(static_cast<size_t>(nS) * ctx->nTransformerVoltageControlMembers);
        CUDA_CHECK(cudaMemcpy(st.data(), ctx->dTransformerVoltageControlMemberState.p, st.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(cur.data(), ctx->dTransformerVoltageControlMemberTap.p, cur.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (int s = 0; s < nS; s++) {
            for (int m = 0; m < ctx->nTransformerVoltageControlMembers; m++) {
                size_t sm = static_cast<size_t>(s) * ctx->nTransformerVoltageControlMembers + m;
                if (st[sm] != 2 || cur[sm] == ctx->hTransformerVoltageControlMemberBaseTap[m]) continue;
                slot.push_back(s);
                slot.push_back(ctx->hTransformerVoltageControlMemberElement[m]);
                const double* row = &ctx->hTransformerVoltageControlTapRows[static_cast<size_t>(ctx->hTransformerVoltageControlMemberTapStart[m] + cur[sm]) * GpuPiModelColumn::COUNT];
                val.insert(val.end(), row, row + GpuPiModelColumn::COUNT);
            }
        }
    }
    slot.insert(slot.end(), ctx->hR1OvBaseSlotElem.begin(), ctx->hR1OvBaseSlotElem.end());
    val.insert(val.end(), ctx->hR1OvBaseVal.begin(), ctx->hR1OvBaseVal.end());
    int nOv = static_cast<int>(slot.size() / 2);
    if (nOv > 0) {
        ctx->dR1OvSlotElem.ensure(slot.size() * sizeof(int));
        ctx->dR1OvVal.ensure(val.size() * sizeof(double));
        CUDA_CHECK(cudaMemcpy(ctx->dR1OvSlotElem.p, slot.data(), slot.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->dR1OvVal.p, val.data(), val.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    ctx->nR1Ov = nOv;
}

// Per-solve reset. Incremental TVC: every scenario starts at the base tap with a fresh ratchet.
void incrementalTransformerVoltageControlResetSolve(BatchSolverContext* ctx, DriverPass& drv) {
    const int& nS = drv.nS;
    size_t nst = static_cast<size_t>(nS) * ctx->nIncrementalTransformerVoltageControlControllerBranches;
    std::vector<int> cur(nst);
    for (int s2 = 0; s2 < nS; s2++) {
        for (int c = 0; c < ctx->nIncrementalTransformerVoltageControlControllerBranches; c++) {
            cur[static_cast<size_t>(s2) * ctx->nIncrementalTransformerVoltageControlControllerBranches + c] = ctx->hIncrementalTransformerVoltageControlBaseTap[c];
        }
    }
    ctx->dIncrementalTransformerVoltageControlAppliedTap.ensure(nst * sizeof(int));
    CUDA_CHECK(cudaMemcpy(ctx->dIncrementalTransformerVoltageControlCurrentTap.p, cur.data(), nst * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->dIncrementalTransformerVoltageControlAppliedTap.p, cur.data(), nst * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ctx->dIncrementalTransformerVoltageControlCurrentDirection.p, 0, nst * sizeof(int)));
    CUDA_CHECK(cudaMemset(ctx->dIncrementalTransformerVoltageControlDirectionChangeCount.p, 0, nst * sizeof(int)));
    CUDA_CHECK(cudaMemset(ctx->dIncrementalTransformerVoltageControlAllowedDirection.p, 0, nst * sizeof(int)));
    ctx->dIncrementalTransformerVoltageControlOutOfDeadband.ensure(nst * sizeof(char));
    ctx->dIncrementalTransformerVoltageControlDiffV.ensure(nst * sizeof(double));
    ctx->dIncrementalTransformerVoltageControlSensitivity.ensure(nst * sizeof(double));
}

void incrementalTransformerVoltageControlStep(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    std::vector<char>& active = drv.active;
    int nIncrementalTransformerVoltageControlControllerBranches = ctx->nIncrementalTransformerVoltageControlControllerBranches;
    if (nIncrementalTransformerVoltageControlControllerBranches <= 0) return;
    if (ctx->mode != GpuLinearSolveMode::UBATCH && ctx->mode != GpuLinearSolveMode::ROBUST) {
        static bool warned = false;
        if (!warned) { std::fprintf(stderr, "in-kernel TVC needs a UBATCH (3) or robust (6) mode; mode %d runs it as stable\n", ctx->mode); warned = true; }
        return;
    }
    size_t nst = static_cast<size_t>(nS) * nIncrementalTransformerVoltageControlControllerBranches;
    incrementalTransformerVoltageControlDeadbandCheckBatch<<<grid(static_cast<int>(nst)), BLOCK, 0, ctx->stream>>>(nIncrementalTransformerVoltageControlControllerBranches, nS, n, ctx->nElements,
            ctx->dIncrementalTransformerVoltageControlControlledBusVRow.as<int>(), ctx->dIncrementalTransformerVoltageControlTargetV.as<double>(), ctx->dIncrementalTransformerVoltageControlHalfTargetDeadband.as<double>(),
            ctx->dIncrementalTransformerVoltageControlElement.as<int>(), ctx->dDisabled.as<char>(), ctx->dX.as<double>(), ctx->dOuterLoopCheckActive.as<char>(),
            ctx->dIncrementalTransformerVoltageControlOutOfDeadband.as<char>(), ctx->dIncrementalTransformerVoltageControlDiffV.as<double>());
    std::vector<char> outOfDeadband(nst);
    CUDA_CHECK(cudaMemcpyAsync(outOfDeadband.data(), ctx->dIncrementalTransformerVoltageControlOutOfDeadband.p, nst, cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    std::vector<char> needC(nIncrementalTransformerVoltageControlControllerBranches, 0);
    bool any = false;
    for (int s2 = 0; s2 < nS; s2++) {
        for (int c = 0; c < nIncrementalTransformerVoltageControlControllerBranches; c++) {
            if (outOfDeadband[static_cast<size_t>(s2) * nIncrementalTransformerVoltageControlControllerBranches + c]) { needC[c] = 1; any = true; }
        }
    }
    if (!any) return;                                    // every bus inside its deadband: STABLE
    // Sensitivities dV/drho at the CURRENT states: refill J for every scenario, factorize once,
    // then one unit-RHS solve per controller that has an out-of-deadband scenario.
    std::vector<char> ones(nS, 1);
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, ones.data(), nS, cudaMemcpyHostToDevice));
    fillActiveBatch(ctx, drv, false);
    CUDA_CHECK(cudaMemsetAsync(ctx->dIncrementalTransformerVoltageControlSensitivity.p, 0, nst * sizeof(double), ctx->stream));
    bool first = true;
    for (int c = 0; c < nIncrementalTransformerVoltageControlControllerBranches; c++) {
        if (!needC[c]) continue;
        int row = ctx->st->rowMap[ctx->hIncrementalTransformerVoltageControlRhoRow[c]];
        if (row < 0) continue;                           // rho row not an active equation: stays insensitive
        setUnitRhsBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, row, ctx->dB.as<double>());
        repairZeroDiagonalRightHandSide(ctx, nS, ctx->dB.as<double>());
        sensitivitySolve(ctx, drv, first);
        incrementalTransformerVoltageControlGatherSensitivityBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(nS, n, c, nIncrementalTransformerVoltageControlControllerBranches, ctx->dIncrementalTransformerVoltageControlControlledBusVRow.as<int>(),
                ctx->dDx.as<double>(), ctx->dIncrementalTransformerVoltageControlSensitivity.as<double>());
    }
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, active.data(), nS, cudaMemcpyHostToDevice));   // as it was
    // Tap decision; a moved scenario is an UNSTABLE verdict (dOuterLoopUnstable), like every device loop.
    incrementalTransformerVoltageControlGroupDecisionBatch<<<grid(ctx->nIncrementalTransformerVoltageControlGroups * nS), BLOCK, 0, ctx->stream>>>(ctx->nIncrementalTransformerVoltageControlGroups, nS, nIncrementalTransformerVoltageControlControllerBranches,
            ctx->dIncrementalTransformerVoltageControlGroupControllerStart.as<int>(), ctx->dIncrementalTransformerVoltageControlGroupController.as<int>(),
            ctx->dIncrementalTransformerVoltageControlOutOfDeadband.as<char>(), ctx->dIncrementalTransformerVoltageControlDiffV.as<double>(), ctx->dIncrementalTransformerVoltageControlSensitivity.as<double>(), ctx->dIncrementalTransformerVoltageControlHalfTargetDeadband.as<double>(),
            ctx->dIncrementalTransformerVoltageControlTapStart.as<int>(), ctx->dIncrementalTransformerVoltageControlTapRows.as<double>(),
            ctx->dIncrementalTransformerVoltageControlCurrentTap.as<int>(), ctx->dIncrementalTransformerVoltageControlCurrentDirection.as<int>(), ctx->dIncrementalTransformerVoltageControlDirectionChangeCount.as<int>(),
            ctx->dIncrementalTransformerVoltageControlAllowedDirection.as<int>(), ctx->incrementalTransformerVoltageControlMaxTapShift, 0.05, ctx->dOuterLoopUnstable.as<char>());
    // Apply: rho target + state seed on the moved pairs, then the pi rows through the tap-override list.
    incrementalTransformerVoltageControlApplyRhoBatch<<<grid(static_cast<int>(nst)), BLOCK, 0, ctx->stream>>>(nIncrementalTransformerVoltageControlControllerBranches, nS, n,
            ctx->dIncrementalTransformerVoltageControlRhoRow.as<int>(), ctx->dIncrementalTransformerVoltageControlTapStart.as<int>(), ctx->dIncrementalTransformerVoltageControlTapRows.as<double>(),
            ctx->dIncrementalTransformerVoltageControlCurrentTap.as<int>(), ctx->dIncrementalTransformerVoltageControlAppliedTap.as<int>(), ctx->dTarget.as<double>(), ctx->dX.as<double>());
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    rebuildTapOverrides(ctx);
    if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
        std::vector<int> cur(nst), base(ctx->hIncrementalTransformerVoltageControlBaseTap);
        CUDA_CHECK(cudaMemcpy(cur.data(), ctx->dIncrementalTransformerVoltageControlCurrentTap.p, nst * sizeof(int), cudaMemcpyDeviceToHost));
        int nOut = 0, nMoved = 0;
        for (size_t k = 0; k < nst; k++) {
            nOut += outOfDeadband[k] ? 1 : 0;
        }
        for (int s2 = 0; s2 < nS; s2++) {
            for (int c = 0; c < nIncrementalTransformerVoltageControlControllerBranches; c++) {
                if (cur[static_cast<size_t>(s2) * nIncrementalTransformerVoltageControlControllerBranches + c] != base[c]) nMoved++;
            }
        }
        std::fprintf(stderr, "TVC_KERNEL nS=%d nControllers=%d outOfDeadband=%d pairsOffBaseTap=%d nR1Ov=%d\n",
                     nS, nIncrementalTransformerVoltageControlControllerBranches, nOut, nMoved, ctx->nR1Ov);
        std::vector<double> dv(nst), se(nst);
        CUDA_CHECK(cudaMemcpy(dv.data(), ctx->dIncrementalTransformerVoltageControlDiffV.p, nst * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(se.data(), ctx->dIncrementalTransformerVoltageControlSensitivity.p, nst * sizeof(double), cudaMemcpyDeviceToHost));
        for (size_t k = 0; k < nst; k++) {
            if (outOfDeadband[k]) {
                std::fprintf(stderr, "TVC_KERNEL_PAIR s=%d c=%d diffV=%.9e sensitivity=%.9e tap=%d base=%d rhoRowMapped=%d\n",
                             (int) (k / nIncrementalTransformerVoltageControlControllerBranches), (int) (k % nIncrementalTransformerVoltageControlControllerBranches), dv[k], se[k], cur[k], base[k % nIncrementalTransformerVoltageControlControllerBranches],
                             ctx->st->rowMap[ctx->hIncrementalTransformerVoltageControlRhoRow[k % nIncrementalTransformerVoltageControlControllerBranches]]);
            }
        }
    }
}

}  // namespace olfgpu
