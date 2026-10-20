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

namespace olfgpu {

// ShuntVoltageControlOuterLoop (WITH_GENERATOR) as an in-kernel driver phase: control ON for the first solve,
// LfShuntImpl.dispatchB per controller shunt and switched off at the first check (UNSTABLE). Group activations
// and member layouts follow the discrete transformer control (transformer_voltage_control.cu).

struct SvdTables {
    int nG, nM, nDelta, nSubs;
    const int* grpPtr;
    const int* deltaPtr; const int* deltaRow; const int* deltaModeOn; const int* deltaModeOff;
    const double* deltaTargetOn; const double* deltaTargetOff;
    const int* memBRow; const int* memElem; const int* memSubPtr;
    const int* secPtr; const double* secB; const int* subMin; const int* subBase;
    int* grpStep; int* memState; double* memContB; int* subPos; double* saved;
};

// A shunt group is live when none of its (packed) member shunts is outaged — such contingencies are
// CPU-routed at qualification anyway (a voltage-controlling shunt outage never qualifies).
__device__ inline bool svdGroupLive(const SvdTables& T, int s, int nElements, int g, const char* disabled) {
    for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) {
        if (T.memElem[m] >= 0 && disabled[(size_t) s * nElements + T.memElem[m]]) return false;
    }
    return true;
}

// ON: every captured delta (the pilot's V identity, the member SHUNT_B power rows — pilot Q relocation and
// DISTR_SHUNT_B — as the toggled equation system produced them); OFF: the pilot back to its saved/OFF
// target, every member's SHUNT_B identity pinned at its dispatched section total.
__device__ inline void svdLayoutGroup(const SvdTables& T, int s, int n, int g, bool on,
                                      int* mode, double* target, double* x) {
    size_t sn = (size_t) s * n;
    for (int k = T.deltaPtr[g]; k < T.deltaPtr[g + 1]; k++) {
        int r = T.deltaRow[k];
        int member = -1;
        for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) if (T.memBRow[m] == r) member = m;
        size_t sk = (size_t) s * T.nDelta + k;
        if (on) {
            if (member < 0 && isnan(T.saved[sk])) T.saved[sk] = target[sn + r];
            mode[sn + r] = T.deltaModeOn[k];
            target[sn + r] = T.deltaTargetOn[k];
        } else if (member < 0) {
            mode[sn + r] = T.deltaModeOff[k];
            target[sn + r] = isnan(T.saved[sk]) ? T.deltaTargetOff[k] : T.saved[sk];
            T.saved[sk] = nan("");
        }
    }
    if (!on) {
        for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) {
            if (T.memState[(size_t) s * T.nM + m] != GpuDiscreteControlMemberState::ROUNDED) continue;
            double bSum = 0.0;
            for (int u = T.memSubPtr[m]; u < T.memSubPtr[m + 1]; u++) bSum += T.secB[T.secPtr[u] + T.subPos[(size_t) s * T.nSubs + u]];
            size_t br = sn + T.memBRow[m];
            mode[br] = GpuRowMode::FIXED_VARIABLE;
            target[br] = bSum;
            x[br] = bSum;
        }
    }
}

__global__ void svdEntryBatch(SvdTables T, int nScenarios, int n, int nElements, int entryOn,
                              const char* disabled, int* mode, double* target, double* x) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    size_t sn = (size_t) s * n;
    for (int g = 0; g < T.nG; g++) {
        if (!svdGroupLive(T, s, nElements, g, disabled)) continue;
        for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) {
            T.memState[(size_t) s * T.nM + m] = entryOn ? (mode[sn + T.memBRow[m]] == GpuRowMode::EQUATION ? GpuDiscreteControlMemberState::CONTINUOUS : GpuDiscreteControlMemberState::DISABLED) : GpuDiscreteControlMemberState::CONTINUOUS;
        }
        if (!entryOn) svdLayoutGroup(T, s, n, g, true, mode, target, x);
    }
}

// The loop's first check: OLF disables every controller shunt and dispatches its continuous b to the
// closest sections (LfShuntImpl.dispatchB: per sub-controller, residue / remaining to the closest section
// from the CURRENT position over [minPosition, size), strictly closer only), UNSTABLE.
__global__ void svdCheckBatch(SvdTables T, int nScenarios, int n, int nElements, const char* disabled, const char* outerLoopCheckActive,
                              int* mode, double* target, double* x, char* switched) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios || !outerLoopCheckActive[s]) return;
    size_t sn = (size_t) s * n;
    bool anyCtl = false;
    for (int g = 0; g < T.nG; g++) {
        size_t sg = (size_t) s * T.nG + g;
        if (T.grpStep[sg] != 0) continue;
        T.grpStep[sg] = 2;
        for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) {
            if (T.memElem[m] < 0 || !disabled[(size_t) s * nElements + T.memElem[m]]) anyCtl = true;
        }
        if (!svdGroupLive(T, s, nElements, g, disabled)) continue;
        bool touched = false;
        for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) {
            size_t sm = (size_t) s * T.nM + m;
            if (T.memState[sm] != GpuDiscreteControlMemberState::CONTINUOUS) continue;
            double residue = x[sn + T.memBRow[m]];
            T.memContB[sm] = residue;
            int remaining = T.memSubPtr[m + 1] - T.memSubPtr[m];
            for (int u = T.memSubPtr[m]; u < T.memSubPtr[m + 1]; u++) {
                double toDispatch = residue / remaining--;
                int p0 = T.secPtr[u];
                int nSec = T.secPtr[u + 1] - p0;
                int best = T.subBase[u];
                double smallest = fabs(toDispatch - T.secB[p0 + best]);
                for (int p = T.subMin[u]; p < nSec; p++) {
                    double dist = fabs(toDispatch - T.secB[p0 + p]);
                    if (dist < smallest) {
                        best = p;
                        smallest = dist;
                    }
                }
                T.subPos[(size_t) s * T.nSubs + u] = best;
                residue -= T.secB[p0 + best];
            }
            T.memState[sm] = GpuDiscreteControlMemberState::ROUNDED;
            touched = true;
        }
        if (touched) svdLayoutGroup(T, s, n, g, false, mode, target, x);
    }
    if (anyCtl) switched[s] = 1;                             // OLF: UNSTABLE whenever a live controller shunt exists
}

static SvdTables svdTables(BatchSolverContext* ctx) {
    SvdTables T;
    T.nG = ctx->nSvdGroups; T.nM = ctx->nSvd; T.nDelta = ctx->nSvdDelta; T.nSubs = ctx->nSvdSubs;
    T.grpPtr = ctx->dSvdGrpPtr.as<int>();
    T.deltaPtr = ctx->dSvdDeltaPtr.as<int>(); T.deltaRow = ctx->dSvdDeltaRow.as<int>();
    T.deltaModeOn = ctx->dSvdDeltaModeOn.as<int>(); T.deltaModeOff = ctx->dSvdDeltaModeOff.as<int>();
    T.deltaTargetOn = ctx->dSvdDeltaTargetOn.as<double>(); T.deltaTargetOff = ctx->dSvdDeltaTargetOff.as<double>();
    T.memBRow = ctx->dSvdMemBRow.as<int>(); T.memElem = ctx->dSvdMemElem.as<int>(); T.memSubPtr = ctx->dSvdMemSubPtr.as<int>();
    T.secPtr = ctx->dSvdSecPtr.as<int>(); T.secB = ctx->dSvdSecB.as<double>(); T.subMin = ctx->dSvdSubMin.as<int>(); T.subBase = ctx->dSvdSubBase.as<int>();
    T.grpStep = ctx->dSvdGrpStep.as<int>(); T.memState = ctx->dSvdMemState.as<int>(); T.memContB = ctx->dSvdMemContB.as<double>();
    T.subPos = ctx->dSvdSubPos.as<int>(); T.saved = ctx->dSvdSaved.as<double>();
    return T;
}

void svdStep(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    if (ctx->nSvd <= 0) return;
    svdCheckBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(svdTables(ctx), nS, n, ctx->nElements, ctx->dDisabled.as<char>(),
            ctx->dOuterLoopCheckActive.as<char>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>(), ctx->dX.as<double>(), ctx->dOuterLoopUnstable.as<char>());
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
        size_t nsm = static_cast<size_t>(nS) * ctx->nSvd;
        std::vector<int> st(nsm);
        CUDA_CHECK(cudaMemcpy(st.data(), ctx->dSvdMemState.p, nsm * sizeof(int), cudaMemcpyDeviceToHost));
        int nCont = 0, nFixed = 0;
        for (size_t k = 0; k < nsm; k++) { nCont += st[k] == GpuDiscreteControlMemberState::CONTINUOUS; nFixed += st[k] == GpuDiscreteControlMemberState::ROUNDED; }
        std::fprintf(stderr, "SVD_KERNEL nS=%d nMembers=%d continuous=%d dispatched=%d\n", nS, ctx->nSvd, nCont, nFixed);
    }
}

// Per-solve reset. Discrete shunt VC: every control at its entry state.
void svdResetSolve(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    size_t nsm = static_cast<size_t>(nS) * ctx->nSvd;
    size_t nsg = static_cast<size_t>(nS) * ctx->nSvdGroups;
    size_t nsd = static_cast<size_t>(nS) * std::max(ctx->nSvdDelta, 1);
    size_t nsu = static_cast<size_t>(nS) * std::max(ctx->nSvdSubs, 1);
    std::vector<int> pos(nsu);
    for (int s2 = 0; s2 < nS; s2++) for (int u = 0; u < ctx->nSvdSubs; u++) pos[static_cast<size_t>(s2) * ctx->nSvdSubs + u] = 0;
    std::vector<double> nanv(std::max(nsm, nsd), std::nan(""));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(cudaMemcpy(ctx->dSvdSubPos.p, pos.data(), nsu * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ctx->dSvdMemState.p, 0, nsm * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(ctx->dSvdMemContB.p, nanv.data(), nsm * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->dSvdSaved.p, nanv.data(), nsd * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ctx->dSvdGrpStep.p, 0, nsg * sizeof(int)));
    svdEntryBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(svdTables(ctx), nS, n, ctx->nElements, ctx->svdEntryOn,
            ctx->dDisabled.as<char>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>(), ctx->dX.as<double>());
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
}

}  // namespace olfgpu
