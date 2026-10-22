// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See shunt_voltage_control.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/codes/gpu_discrete_control_member_state.cuh"
#include "../common/codes/gpu_row_mode.cuh"
#include "../solver/driver_steps.cuh"
#include "shunt_voltage_control.cuh"
#include "../common/trace.cuh"

namespace olfgpu {

// ShuntVoltageControlOuterLoop (WITH_GENERATOR) as an in-kernel driver phase: control ON for the first solve,
// LfShuntImpl.dispatchB per controller shunt and switched off at the first check (UNSTABLE). Group activations
// and member layouts follow the discrete transformer control (transformer_voltage_control.cu).

struct ShuntVoltageControlTables {
    int nGroups, nMembers, nDeltaRows, nShuntCompensators;
    const int* groupMemberStart;
    const int* deltaStart; const int* deltaRow; const int* deltaModeOn; const int* deltaModeOff;
    const double* deltaTargetOn; const double* deltaTargetOff;
    const int* memberShuntBRow; const int* memberElement; const int* memberShuntCompensatorStart;
    const int* sectionStart; const double* sectionB; const int* shuntCompensatorMinPosition; const int* shuntCompensatorBasePosition;
    int* groupStep; int* memberState; double* memberContinuousB; int* shuntCompensatorPosition; double* savedDeltaTarget;
};

// A shunt group is live when none of its (packed) member shunts is outaged — such contingencies are
// CPU-routed at qualification anyway (a voltage-controlling shunt outage never qualifies).
__device__ inline bool shuntVoltageControlGroupLive(const ShuntVoltageControlTables& T, int s, int nElements, int g, const char* disabled) {
    for (int m = T.groupMemberStart[g]; m < T.groupMemberStart[g + 1]; m++) {
        if (T.memberElement[m] >= 0 && disabled[(size_t) s * nElements + T.memberElement[m]]) return false;
    }
    return true;
}

// ON: every captured delta (the pilot's V identity, the member SHUNT_B power rows — pilot Q relocation and
// DISTR_SHUNT_B — as the toggled equation system produced them); OFF: the pilot back to its saved/OFF
// target, every member's SHUNT_B identity held at its dispatched section total.
__device__ inline void shuntVoltageControlLayoutGroup(const ShuntVoltageControlTables& T, int s, int n, int g, bool on,
                                      int* mode, double* target, double* x) {
    size_t sn = (size_t) s * n;
    for (int k = T.deltaStart[g]; k < T.deltaStart[g + 1]; k++) {
        int r = T.deltaRow[k];
        int member = -1;
        for (int m = T.groupMemberStart[g]; m < T.groupMemberStart[g + 1]; m++) if (T.memberShuntBRow[m] == r) member = m;
        size_t sk = (size_t) s * T.nDeltaRows + k;
        if (on) {
            if (member < 0 && isnan(T.savedDeltaTarget[sk])) T.savedDeltaTarget[sk] = target[sn + r];
            mode[sn + r] = T.deltaModeOn[k];
            target[sn + r] = T.deltaTargetOn[k];
        } else if (member < 0) {
            mode[sn + r] = T.deltaModeOff[k];
            target[sn + r] = isnan(T.savedDeltaTarget[sk]) ? T.deltaTargetOff[k] : T.savedDeltaTarget[sk];
            T.savedDeltaTarget[sk] = nan("");
        }
    }
    if (!on) {
        for (int m = T.groupMemberStart[g]; m < T.groupMemberStart[g + 1]; m++) {
            if (T.memberState[(size_t) s * T.nMembers + m] != GpuDiscreteControlMemberState::ROUNDED) continue;
            double bSum = 0.0;
            for (int u = T.memberShuntCompensatorStart[m]; u < T.memberShuntCompensatorStart[m + 1]; u++) bSum += T.sectionB[T.sectionStart[u] + T.shuntCompensatorPosition[(size_t) s * T.nShuntCompensators + u]];
            size_t br = sn + T.memberShuntBRow[m];
            mode[br] = GpuRowMode::FIXED_VARIABLE;
            target[br] = bSum;
            x[br] = bSum;
        }
    }
}

__global__ void shuntVoltageControlEntryBatch(ShuntVoltageControlTables T, int nScenarios, int n, int nElements, int entryOn,
                              const char* disabled, int* mode, double* target, double* x) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    size_t sn = (size_t) s * n;
    for (int g = 0; g < T.nGroups; g++) {
        if (!shuntVoltageControlGroupLive(T, s, nElements, g, disabled)) continue;
        for (int m = T.groupMemberStart[g]; m < T.groupMemberStart[g + 1]; m++) {
            T.memberState[(size_t) s * T.nMembers + m] = entryOn ? (mode[sn + T.memberShuntBRow[m]] == GpuRowMode::EQUATION ? GpuDiscreteControlMemberState::CONTINUOUS : GpuDiscreteControlMemberState::DISABLED) : GpuDiscreteControlMemberState::CONTINUOUS;
        }
        if (!entryOn) shuntVoltageControlLayoutGroup(T, s, n, g, true, mode, target, x);
    }
}

// The loop's first check: OLF disables every controller shunt and dispatches its continuous b to the
// closest sections (LfShuntImpl.dispatchB: per sub-controller, residue / remaining to the closest section
// from the CURRENT position over [minPosition, size), strictly closer only), UNSTABLE.
__global__ void shuntVoltageControlCheckBatch(ShuntVoltageControlTables T, int nScenarios, int n, int nElements, const char* disabled, const char* outerLoopCheckActive,
                              int* mode, double* target, double* x, char* switched) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios || !outerLoopCheckActive[s]) return;
    size_t sn = (size_t) s * n;
    bool anyCtl = false;
    for (int g = 0; g < T.nGroups; g++) {
        size_t sg = (size_t) s * T.nGroups + g;
        if (T.groupStep[sg] != 0) continue;
        T.groupStep[sg] = 2;
        for (int m = T.groupMemberStart[g]; m < T.groupMemberStart[g + 1]; m++) {
            if (T.memberElement[m] < 0 || !disabled[(size_t) s * nElements + T.memberElement[m]]) anyCtl = true;
        }
        if (!shuntVoltageControlGroupLive(T, s, nElements, g, disabled)) continue;
        bool touched = false;
        for (int m = T.groupMemberStart[g]; m < T.groupMemberStart[g + 1]; m++) {
            size_t sm = (size_t) s * T.nMembers + m;
            if (T.memberState[sm] != GpuDiscreteControlMemberState::CONTINUOUS) continue;
            double residue = x[sn + T.memberShuntBRow[m]];
            T.memberContinuousB[sm] = residue;
            int remaining = T.memberShuntCompensatorStart[m + 1] - T.memberShuntCompensatorStart[m];
            for (int u = T.memberShuntCompensatorStart[m]; u < T.memberShuntCompensatorStart[m + 1]; u++) {
                double toDispatch = residue / remaining--;
                int p0 = T.sectionStart[u];
                int nSec = T.sectionStart[u + 1] - p0;
                int best = T.shuntCompensatorBasePosition[u];
                double smallest = fabs(toDispatch - T.sectionB[p0 + best]);
                for (int p = T.shuntCompensatorMinPosition[u]; p < nSec; p++) {
                    double dist = fabs(toDispatch - T.sectionB[p0 + p]);
                    if (dist < smallest) {
                        best = p;
                        smallest = dist;
                    }
                }
                T.shuntCompensatorPosition[(size_t) s * T.nShuntCompensators + u] = best;
                residue -= T.sectionB[p0 + best];
            }
            T.memberState[sm] = GpuDiscreteControlMemberState::ROUNDED;
            touched = true;
        }
        if (touched) shuntVoltageControlLayoutGroup(T, s, n, g, false, mode, target, x);
    }
    if (anyCtl) switched[s] = 1;                             // OLF: UNSTABLE whenever a live controller shunt exists
}

static ShuntVoltageControlTables shuntVoltageControlTables(BatchSolverContext* ctx) {
    ShuntVoltageControlTables T;
    T.nGroups = ctx->nShuntVoltageControlGroups; T.nMembers = ctx->nShuntVoltageControlMembers; T.nDeltaRows = ctx->nShuntVoltageControlDeltaRows; T.nShuntCompensators = ctx->nShuntVoltageControlShuntCompensators;
    T.groupMemberStart = ctx->dShuntVoltageControlGroupMemberStart.as<int>();
    T.deltaStart = ctx->dShuntVoltageControlDeltaStart.as<int>(); T.deltaRow = ctx->dShuntVoltageControlDeltaRow.as<int>();
    T.deltaModeOn = ctx->dShuntVoltageControlDeltaModeOn.as<int>(); T.deltaModeOff = ctx->dShuntVoltageControlDeltaModeOff.as<int>();
    T.deltaTargetOn = ctx->dShuntVoltageControlDeltaTargetOn.as<double>(); T.deltaTargetOff = ctx->dShuntVoltageControlDeltaTargetOff.as<double>();
    T.memberShuntBRow = ctx->dShuntVoltageControlMemberShuntBRow.as<int>(); T.memberElement = ctx->dShuntVoltageControlMemberElement.as<int>(); T.memberShuntCompensatorStart = ctx->dShuntVoltageControlMemberShuntCompensatorStart.as<int>();
    T.sectionStart = ctx->dShuntVoltageControlSectionStart.as<int>(); T.sectionB = ctx->dShuntVoltageControlSectionB.as<double>(); T.shuntCompensatorMinPosition = ctx->dShuntVoltageControlShuntCompensatorMinPosition.as<int>(); T.shuntCompensatorBasePosition = ctx->dShuntVoltageControlShuntCompensatorBasePosition.as<int>();
    T.groupStep = ctx->dShuntVoltageControlGroupStep.as<int>(); T.memberState = ctx->dShuntVoltageControlMemberState.as<int>(); T.memberContinuousB = ctx->dShuntVoltageControlMemberContinuousB.as<double>();
    T.shuntCompensatorPosition = ctx->dShuntVoltageControlShuntCompensatorPosition.as<int>(); T.savedDeltaTarget = ctx->dShuntVoltageControlSavedDeltaTarget.as<double>();
    return T;
}

void shuntVoltageControlStep(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    if (ctx->nShuntVoltageControlMembers <= 0) return;
    shuntVoltageControlCheckBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(shuntVoltageControlTables(ctx), nS, n, ctx->nElements, ctx->dDisabled.as<char>(),
            ctx->dOuterLoopCheckActive.as<char>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>(), ctx->dX.as<double>(), ctx->dOuterLoopUnstable.as<char>());
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    if (traceOn(Trace::BATCH_TRACE)) {
        size_t nsm = static_cast<size_t>(nS) * ctx->nShuntVoltageControlMembers;
        std::vector<int> memberState(nsm);
        CUDA_CHECK(cudaMemcpy(memberState.data(), ctx->dShuntVoltageControlMemberState.p, nsm * sizeof(int), cudaMemcpyDeviceToHost));
        int nCont = 0, nFixed = 0;
        for (size_t k = 0; k < nsm; k++) { nCont += memberState[k] == GpuDiscreteControlMemberState::CONTINUOUS; nFixed += memberState[k] == GpuDiscreteControlMemberState::ROUNDED; }
        std::fprintf(stderr, "SVD_KERNEL nS=%d nMembers=%d continuous=%d dispatched=%d\n", nS, ctx->nShuntVoltageControlMembers, nCont, nFixed);
    }
}

// Per-solve reset. Discrete shunt VC: every control at its entry state.
void shuntVoltageControlResetSolve(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    size_t nsm = static_cast<size_t>(nS) * ctx->nShuntVoltageControlMembers;
    size_t nsg = static_cast<size_t>(nS) * ctx->nShuntVoltageControlGroups;
    size_t nsd = static_cast<size_t>(nS) * std::max(ctx->nShuntVoltageControlDeltaRows, 1);
    size_t nsu = static_cast<size_t>(nS) * std::max(ctx->nShuntVoltageControlShuntCompensators, 1);
    std::vector<int> pos(nsu);
    for (int s2 = 0; s2 < nS; s2++) for (int u = 0; u < ctx->nShuntVoltageControlShuntCompensators; u++) pos[static_cast<size_t>(s2) * ctx->nShuntVoltageControlShuntCompensators + u] = 0;
    std::vector<double> nanv(std::max(nsm, nsd), std::nan(""));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(cudaMemcpy(ctx->dShuntVoltageControlShuntCompensatorPosition.p, pos.data(), nsu * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ctx->dShuntVoltageControlMemberState.p, 0, nsm * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(ctx->dShuntVoltageControlMemberContinuousB.p, nanv.data(), nsm * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->dShuntVoltageControlSavedDeltaTarget.p, nanv.data(), nsd * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ctx->dShuntVoltageControlGroupStep.p, 0, nsg * sizeof(int)));
    shuntVoltageControlEntryBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(shuntVoltageControlTables(ctx), nS, n, ctx->nElements, ctx->shuntVoltageControlEntryOn,
            ctx->dDisabled.as<char>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>(), ctx->dX.as<double>());
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
}

}  // namespace olfgpu
