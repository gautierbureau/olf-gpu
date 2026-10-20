// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See secondary_voltage_control.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/codes/gpu_q_limit_type.cuh"
#include "../common/codes/gpu_linear_solve_mode.cuh"
#include "../common/codes/gpu_controlled_flow.cuh"
#include "../common/codes/gpu_row_mode.cuh"
#include <cublas_v2.h>
#include "closed_branch.cuh"
#include "../solver/driver_steps.cuh"
#include "../equations/scenario_overrides.cuh"
#include "flow_observation.cuh"
#include "sensitivity.cuh"
#include "secondary_voltage_control.cuh"

namespace olfgpu {

// OLF's SecondaryVoltageControlOuterLoop, per (scenario, zone), with nothing left on the host. The
// observables are the SAME quantities OLF's EquationTerms compute, reached the device's way: a
// controller bus's reactive injection is the sum of its CLOSED-BRANCH side flows (gradient from the
// codegen'd evalClosedBranch, exactly as flcGatherSensiBatch does it) plus its shunts' q = -b*v^2
// (ShuntCompensatorReactiveFlowEquationTerm: dq/dv = -2bv, dq/db = -v^2), and the pilot's calculated V
// is the BUS_V variable itself, so its sensitivity is dx at the pilot row.
constexpr double SVC_DV_EPS = 1e-4;                          // OLF DV_EPS (0.1 kV)
constexpr double SVC_DK_DIFF_MAX_EPS = 1e-3;                 // OLF DK_DIFF_MAX_EPS (1 MVar)

struct SvcTables {
    int nCtl, nZones, nrl;
    const int* ctldVRow; const int* busVRow; const int* rlIdx;
    const double* baseTargetV; const double* minQ; const double* maxQ; const double* loadQ;
    const int* qPtr; const int* qKind; const int* qElem; const int* qBRow;
    const int* zonePtr; const int* zoneCtl; const int* zonePilotRow; const double* zoneTargetV;
    double* targetV; double* k0; double* jk; double* jvpp; double* rlVt;
    char* enabled; char* need; char* reject;
    double minPlausibleV, maxPlausibleV;
};

__device__ inline double svcQToK(double q, double minQ, double maxQ) {
    return (2.0 * q - maxQ - minQ) / (maxQ - minQ);          // OLF qToK
}

// OLF check(): tryToReEnableHelpfulControllerBuses FIRST (a zone with SOME but not all controllers at a
// reactive limit gives every limited one back its voltage control), then the per-zone trigger on the
// pilot deadband and the K spread. The re-enable runs whether or not the zone then adjusts — OLF mutates
// the network before it decides.
__global__ void svcTriggerBatch(SvcTables T, int nScenarios, int n, const char* rlActive,
                                const double* qBuf, const double* x, char* limitType, const char* savedMode,
                                int* mode, double* target) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= T.nZones * nScenarios) return;
    int z = tid % T.nZones;
    int s = tid / T.nZones;
    if (!rlActive[s]) return;
    int lo = T.zonePtr[z];
    int hi = T.zonePtr[z + 1];
    // ---- tryToReEnableHelpfulControllerBuses ----
    int nPinned = 0;
    for (int k = lo; k < hi; k++) {
        if (limitType[(size_t) s * T.nrl + T.rlIdx[T.zoneCtl[k]]] != GpuQLimitType::NONE) nPinned++;
    }
    if (nPinned > 0 && nPinned < hi - lo) {
        for (int k = lo; k < hi; k++) {
            int c = T.zoneCtl[k];
            size_t st = (size_t) s * T.nrl + T.rlIdx[c];
            if (limitType[st] == GpuQLimitType::NONE) continue;
            size_t idx = (size_t) s * n + T.busVRow[c];
            char sm = savedMode[st];
            mode[idx] = (sm == GpuRowMode::BUS_TARGET_V_WITH_SLOPE) ? GpuRowMode::BUS_TARGET_V_WITH_SLOPE : GpuRowMode::FIXED_VARIABLE;                  // back to its PV form
            target[idx] = T.targetV[(size_t) s * T.nCtl + c];
            limitType[st] = GpuQLimitType::NONE;
        }
    }
    // ---- the zone's trigger: pilot deadband, or K spread while not all controllers are at a limit ----
    double pilotDv = T.zoneTargetV[z] - x[(size_t) s * n + T.zonePilotRow[z]];
    double kMin = 1e300;
    double kMax = -1e300;
    bool allAbove = true;
    bool allBelow = true;
    for (int k = lo; k < hi; k++) {
        int c = T.zoneCtl[k];
        double q = qBuf[(size_t) s * n + T.busVRow[c]] + T.loadQ[c];
        double kk = svcQToK(q, T.minQ[c], T.maxQ[c]);
        T.k0[(size_t) s * T.nCtl + c] = kk;
        kMin = fmin(kMin, kk);
        kMax = fmax(kMax, kk);
        allAbove &= kk > 1.0;
        allBelow &= kk < -1.0;
        T.enabled[(size_t) s * T.nCtl + c] = limitType[(size_t) s * T.nrl + T.rlIdx[c]] == GpuQLimitType::NONE ? 1 : 0;
    }
    // OLF filters the zone OUT of the check when it has no controller whose voltage control is enabled
    // (findAnyControlledBusWithAtLeastOneControllerBusWithVoltageControlEnabled): every controller at a
    // reactive limit means the zone cannot act, so it is not even looked at. Without this the pilot
    // deadband still flags the scenario, and the sensitivity solve that follows adjusts nothing while
    // leaving the driver's factorization state disturbed — 1.0e-10 kV on ieee14's z1.
    int nEnabled = 0;
    for (int k = lo; k < hi; k++) {
        if (T.enabled[(size_t) s * T.nCtl + T.zoneCtl[k]]) nEnabled++;
    }
    bool allAtLimits = allAbove || allBelow;
    if (nEnabled > 0 && (fabs(pilotDv) > SVC_DV_EPS || (!allAtLimits && fabs(kMax - kMin) > SVC_DK_DIFF_MAX_EPS))) {
        T.need[s] = 1;
    }
}

// After the unit-RHS solve for CONTROLLED bus i: OLF's calculateSensiK for every controller j — the sum
// over j's own Q observables of (term gradient . dx_i), scaled by 2/(maxQ-minQ) — and calculateSensiVpp,
// which is dx_i at the zone's pilot row.
__global__ void svcGatherBatch(SvcTables T, int nScenarios, int n, int i, int nElements, int cbOffset, int shOffset,
                               const double* cbIn, const int* cbIx, const double* shIn, const int* shIx,
                               int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal,
                               int nShOv, const int* shOvSlotElem, const double* shOvGb,
                               const char* disabled, const double* x, const double* dx) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= T.nCtl * nScenarios) return;
    int j = tid % T.nCtl;
    int s = tid / T.nCtl;
    const double* xs = x + (size_t) s * n;
    const double* dxs = dx + (size_t) s * n;
    double sq = 0.0;
    for (int k = T.qPtr[j]; k < T.qPtr[j + 1]; k++) {
        int elem = T.qElem[k];
        // NO isDisabled test here, deliberately: OLF's calculateSensiQ sums the controller bus's branch
        // terms as they stand, an outaged one included. Replicating the quirk keeps us on its trajectory.
        if (T.qKind[k] == 2) {                               // shunt: q = -b v^2
            int e = elem - shOffset;
            if (e < 0) continue;
            double g = shIn[e * 2];
            double b = shIn[e * 2 + 1];
            applyShuntOverride(nShOv, shOvSlotElem, shOvGb, s, e, g, b);
            int vCol = shIx[e * 5 + 2];
            int bCol = shIx[e * 5 + 3];                      // SHUNT_B row when b is a solved variable
            if (bCol >= 0) b = xs[bCol];
            double v = xs[vCol];
            sq += (-2.0 * b * v) * dxs[vCol] + (bCol >= 0 ? (-v * v) * dxs[bCol] : 0.0);
        } else {
            int e = elem - cbOffset;
            if (e < 0) continue;
            const double* p = cbIn + e * 8;
            double pOv[8];
            if (findTapOverride(nR1Ov, r1OvSlotElem, r1OvVal, s, e, pOv)) p = pOv;
            const int* q = cbIx + e * 12;
            double g[18];
            flcFlowAndGrad(T.qKind[k] == 0 ? 1 : 2, GpuControlledFlow::REACTIVE_POWER, p, xs, q, g);  // the reactive flow on side 1 / 2
            double d[6] = { dxs[q[5]], dxs[q[7]], dxs[q[4]], dxs[q[6]],
                            q[8] >= 0 ? dxs[q[8]] : 0.0, q[9] >= 0 ? dxs[q[9]] : 0.0 };
            for (int m = 0; m < 6; m++) sq += g[m] * d[m];
        }
    }
    T.jk[((size_t) s * T.nCtl + i) * T.nCtl + j] = 2.0 * sq / (T.maxQ[j] - T.minQ[j]);
    if (j == 0) {
        // the pilot sensitivity of controlled bus i at EVERY zone's pilot: OLF's pilot row of zone z (createJvpp
        // over allControlledBuses) carries each controlled bus's response at THAT pilot, other zones' buses
        // included - a controller in one zone moves another zone's pilot through the network. Stored for i's
        // own zone only, the pilot row of a multi-zone system read the other zones' columns at their own pilots.
        for (int z = 0; z < T.nZones; z++) {
            T.jvpp[((size_t) s * T.nZones + z) * T.nCtl + i] = dxs[T.zonePilotRow[z]];
        }
    }
}

// OLF processSecondaryVoltageControl's zone system, ASSEMBLED per scenario into device memory for a
// batched dense LU. OLF builds one system over every ENABLED controller of every zone — A is block
// diagonal by zone but Jk is full, since a controller in one zone moves a controlled bus in another
// through the network — so the size is the total controller count, not a per-zone one. A DISABLED
// controller (at a reactive limit) is padded out with an identity row and a zero right-hand side,
// which leaves the enabled block exactly the system OLF solves and gives the disabled one dv = 0.
// Column-major, because that is what cuBLAS reads.
__global__ void svcAssembleBatch(SvcTables T, int nScenarios, int n, const char* rlActive,
                                 const int* ctlZone, const double* x, double* mat, double* rhs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= T.nCtl * nScenarios) return;
    int i = tid % T.nCtl;
    int s = tid / T.nCtl;
    int nC = T.nCtl;
    double* m = mat + (size_t) s * nC * nC;
    double* r = rhs + (size_t) s * nC;
    bool live = rlActive[s] && T.need[s];
    if (!live || !T.enabled[(size_t) s * nC + i]) {
        for (int j = 0; j < nC; j++) {
            m[(size_t) j * nC + i] = (i == j) ? 1.0 : 0.0;   // identity row: this controller does not move
        }
        r[i] = 0.0;
        return;
    }
    int z = ctlZone[i];
    int lo = T.zonePtr[z];
    int hi = T.zonePtr[z + 1];
    int cnt = 0;
    int last = -1;
    for (int k = lo; k < hi; k++) {
        int c = T.zoneCtl[k];
        if (T.enabled[(size_t) s * nC + c]) {
            cnt++;
            last = c;
        }
    }
    if (i == last) {                                         // the zone's LAST controlled bus carries the pilot row
        for (int j = 0; j < nC; j++) {                       // every enabled controlled bus's response at pilot z
            m[(size_t) j * nC + i] = T.enabled[(size_t) s * nC + j]
                    ? T.jvpp[((size_t) s * T.nZones + z) * nC + j] : 0.0;
        }
        r[i] = T.zoneTargetV[z] - x[(size_t) s * n + T.zonePilotRow[z]];
        return;
    }
    for (int j = 0; j < nC; j++) {                           // b = A . Jk^T, restricted to enabled columns
        if (!T.enabled[(size_t) s * nC + j]) {
            m[(size_t) j * nC + i] = 0.0;
            continue;
        }
        double v = 0.0;
        for (int k = lo; k < hi; k++) {
            int c = T.zoneCtl[k];
            if (!T.enabled[(size_t) s * nC + c]) continue;
            double a = (c == i) ? 1.0 - 1.0 / cnt : -1.0 / cnt;
            v += a * T.jk[((size_t) s * nC + j) * nC + c];
        }
        m[(size_t) j * nC + i] = v;
    }
    double rv = 0.0;                                         // rhs = -A . k0
    for (int k = lo; k < hi; k++) {
        int c = T.zoneCtl[k];
        if (!T.enabled[(size_t) s * nC + c]) continue;
        double a = (c == i) ? 1.0 - 1.0 / cnt : -1.0 / cnt;
        rv += a * T.k0[(size_t) s * nC + c];
    }
    r[i] = -rv;
}

// The pointer arrays cuBLAS's batched API takes, one entry per scenario.
__global__ void svcPointersBatch(int nScenarios, int nC, double* mat, double* rhs,
                                 double** matPtr, double** rhsPtr) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    matPtr[s] = mat + (size_t) s * nC * nC;
    rhsPtr[s] = rhs + (size_t) s * nC;
}

// After the batched solve: OLF's plausibility test on the new targets, then commit. A singular zone
// system (cuBLAS info != 0) or an implausible target sends the scenario to the CPU, which is where
// OLF's own MatrixException / FAILED verdict would land it.
__global__ void svcCommitBatch(SvcTables T, int nScenarios, const char* rlActive,
                               const int* info, const double* rhs, char* switched) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios || !rlActive[s] || !T.need[s]) return;
    int nC = T.nCtl;
    if (info[s] != 0) {
        T.reject[s] = 1;
        return;
    }
    const double* dv = rhs + (size_t) s * nC;
    for (int i = 0; i < nC; i++) {
        if (!T.enabled[(size_t) s * nC + i]) continue;
        double nt = T.targetV[(size_t) s * nC + i] + dv[i];
        if (!(nt >= T.minPlausibleV && nt <= T.maxPlausibleV)) {
            T.reject[s] = 1;
            return;
        }
    }
    bool moved = false;
    for (int i = 0; i < nC; i++) {
        if (!T.enabled[(size_t) s * nC + i]) continue;
        T.targetV[(size_t) s * nC + i] += dv[i];
        moved = true;
    }
    if (moved) {
        switched[s] = 1;                                     // OLF: UNSTABLE, the zone adjusted
    }
}

// Give the moved zone targets to the system: an UNPINNED controller's controlled-bus V identity takes the
// new target; a PINNED one keeps its Q pin but its switch-back must compare against the moved target.
__global__ void svcApplyBatch(SvcTables T, int nScenarios, int n, const char* rlActive,
                              const char* limitType, double* target) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= T.nCtl * nScenarios) return;
    int c = tid % T.nCtl;
    int s = tid / T.nCtl;
    if (!rlActive[s]) return;
    double tv = T.targetV[(size_t) s * T.nCtl + c];
    if (tv == T.baseTargetV[c]) return;
    T.rlVt[(size_t) s * T.nrl + T.rlIdx[c]] = tv;
    if (limitType[(size_t) s * T.nrl + T.rlIdx[c]] == GpuQLimitType::NONE) {
        target[(size_t) s * n + T.ctldVRow[c]] = tv;
    }
}

static SvcTables svcTables(BatchSolverContext* ctx) {
    SvcTables T;
    T.nCtl = ctx->nSvc; T.nZones = ctx->nSvcZones; T.nrl = ctx->nrl;
    T.ctldVRow = ctx->dSvcCtldVRow.as<int>(); T.busVRow = ctx->dSvcBusVRow.as<int>(); T.rlIdx = ctx->dSvcRlIdx.as<int>();
    T.baseTargetV = ctx->dSvcBaseTargetV.as<double>(); T.minQ = ctx->dSvcMinQ.as<double>();
    T.maxQ = ctx->dSvcMaxQ.as<double>(); T.loadQ = ctx->dSvcLoadQ.as<double>();
    T.qPtr = ctx->dSvcQPtr.as<int>(); T.qKind = ctx->dSvcQKind.as<int>();
    T.qElem = ctx->dSvcQElem.as<int>(); T.qBRow = ctx->dSvcQBRow.as<int>();
    T.zonePtr = ctx->dSvcZonePtr.as<int>(); T.zoneCtl = ctx->dSvcZoneCtl.as<int>();
    T.zonePilotRow = ctx->dSvcZonePilotRow.as<int>(); T.zoneTargetV = ctx->dSvcZoneTargetV.as<double>();
    T.targetV = ctx->dSvcTargetV.as<double>(); T.k0 = ctx->dSvcK0.as<double>();
    T.jk = ctx->dSvcJk.as<double>(); T.jvpp = ctx->dSvcJvpp.as<double>(); T.rlVt = ctx->dSvcRlVt.as<double>();
    T.enabled = ctx->dSvcEnabled.as<char>(); T.need = ctx->dSvcNeed.as<char>(); T.reject = ctx->dSvcReject.as<char>();
    T.minPlausibleV = ctx->svcMinPlausibleV; T.maxPlausibleV = ctx->svcMaxPlausibleV;
    return T;
}

// Per-solve reset. Secondary voltage control: zone targets back at the network's own, nothing adjusted yet.
void svcResetSolve(BatchSolverContext* ctx, DriverPass& drv) {
    const int& nS = drv.nS;
    size_t nsc = static_cast<size_t>(nS) * ctx->nSvc;
    std::vector<double> tv(nsc);
    for (int s2 = 0; s2 < nS; s2++) {
        for (int c = 0; c < ctx->nSvc; c++) tv[static_cast<size_t>(s2) * ctx->nSvc + c] = ctx->hSvcBaseTargetV[c];
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(cudaMemcpy(ctx->dSvcTargetV.p, tv.data(), nsc * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ctx->dSvcNeed.p, 0, nS));
    CUDA_CHECK(cudaMemset(ctx->dSvcReject.p, 0, nS));
    CUDA_CHECK(cudaMemset(ctx->dSvcEnabled.p, 1, nsc));
}

void svcStep(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    const int& outer = drv.outer;
    std::vector<char>& active = drv.active;
    std::vector<char>& loopRejected = drv.loopRejected;
    int nC = ctx->nSvc;
    if (nC <= 0) return;
    if (ctx->mode != GpuLinearSolveMode::UBATCH && ctx->mode != GpuLinearSolveMode::ROBUST) {
        static bool warned = false;
        if (!warned) { std::fprintf(stderr, "in-kernel SecondaryVc needs a UBATCH (3) or robust (6) mode; mode %d runs it as stable\n", ctx->mode); warned = true; }
        return;
    }
    CUDA_CHECK(cudaMemsetAsync(ctx->dSvcNeed.p, 0, nS, ctx->stream));
    svcTriggerBatch<<<grid(ctx->nSvcZones * nS), BLOCK, 0, ctx->stream>>>(svcTables(ctx), nS, n,
            ctx->dRlActive.as<char>(), ctx->dQbuf.as<double>(), ctx->dX.as<double>(),
            ctx->dRlLimitType.as<char>(), ctx->dRlSavedMode.as<char>(),
            ctx->dMode.as<int>(), ctx->dTarget.as<double>());
    std::vector<char> need(nS);
    CUDA_CHECK(cudaMemcpyAsync(need.data(), ctx->dSvcNeed.p, nS, cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    bool any = false;
    for (char c : need) any |= c != 0;
    if (!any) return;                                    // every zone inside its bands: OLF's STABLE
    std::vector<char> ones(nS, 1);
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, ones.data(), nS, cudaMemcpyHostToDevice));
    fillActiveBatch(ctx, drv, false);
    bool first = true;
    for (int i = 0; i < nC; i++) {
        int row = ctx->st->rowMap[ctx->hSvcCtldVRow[i]];
        if (row < 0) continue;                           // controlled bus not a live identity: insensitive
        setUnitRhsBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, row, ctx->dB.as<double>());
        hazRepairRightHandSide(ctx, nS, ctx->dB.as<double>());
        sensitivitySolve(ctx, drv, first);
        svcGatherBatch<<<grid(nC * nS), BLOCK, 0, ctx->stream>>>(svcTables(ctx), nS, n, i,
                ctx->nElements, ctx->cbOffset, ctx->shOffset,
                ctx->st->cb.parameters.as<double>(), ctx->st->cb.indices.as<int>(),
                ctx->st->sh.parameters.as<double>(), ctx->st->sh.indices.as<int>(),
                ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(),
                ctx->nShOv, ctx->dShOvSlotElem.as<int>(), ctx->dShOvGb.as<double>(),
                ctx->dDisabled.as<char>(), ctx->dX.as<double>(), ctx->dDx.as<double>());
    }
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, active.data(), nS, cudaMemcpyHostToDevice));   // as it was
    // ---- the zone systems: assembled into device memory, solved by cuBLAS's BATCHED dense LU ----
    if (ctx->svcBlas == nullptr) {
        CUBLAS_CHECK(cublasCreate(&ctx->svcBlas));
    }
    CUBLAS_CHECK(cublasSetStream(ctx->svcBlas, ctx->stream));
    svcAssembleBatch<<<grid(nC * nS), BLOCK, 0, ctx->stream>>>(svcTables(ctx), nS, n,
            ctx->dRlActive.as<char>(), ctx->dSvcCtlZone.as<int>(), ctx->dX.as<double>(),
            ctx->dSvcMat.as<double>(), ctx->dSvcRhs.as<double>());
    svcPointersBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(nS, nC,
            ctx->dSvcMat.as<double>(), ctx->dSvcRhs.as<double>(),
            reinterpret_cast<double**>(ctx->dSvcMatPtr.p), reinterpret_cast<double**>(ctx->dSvcRhsPtr.p));
    CUBLAS_CHECK(cublasDgetrfBatched(ctx->svcBlas, nC,
            reinterpret_cast<double**>(ctx->dSvcMatPtr.p), nC,
            ctx->dSvcPiv.as<int>(), ctx->dSvcInfo.as<int>(), nS));
    int trsInfo = 0;
    CUBLAS_CHECK(cublasDgetrsBatched(ctx->svcBlas, CUBLAS_OP_N, nC, 1,
            reinterpret_cast<const double* const*>(ctx->dSvcMatPtr.p), nC,
            ctx->dSvcPiv.as<int>(),
            reinterpret_cast<double**>(ctx->dSvcRhsPtr.p), nC, &trsInfo, nS));
    if (trsInfo != 0) {
        throw std::runtime_error("cublasDgetrsBatched rejected its arguments");
    }
    svcCommitBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(svcTables(ctx), nS,
            ctx->dRlActive.as<char>(), ctx->dSvcInfo.as<int>(), ctx->dSvcRhs.as<double>(),
            ctx->dRlSwitched.as<char>());
    svcApplyBatch<<<grid(nC * nS), BLOCK, 0, ctx->stream>>>(svcTables(ctx), nS, n,
            ctx->dRlActive.as<char>(), ctx->dRlLimitType.as<char>(), ctx->dTarget.as<double>());
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    std::vector<char> rej(nS);
    CUDA_CHECK(cudaMemcpy(rej.data(), ctx->dSvcReject.p, nS, cudaMemcpyDeviceToHost));
    int nRej = 0;
    for (int s2 = 0; s2 < nS; s2++) {
        if (rej[s2]) { loopRejected[s2] = 1; nRej++; traceReject(s2, "svc-kernel-reject", outer, 0); }
    }
    if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
        int nNeed = 0;
        for (char c : need) nNeed += c ? 1 : 0;
        std::fprintf(stderr, "SVC_KERNEL nS=%d nCtl=%d nZones=%d adjusting=%d rejected=%d\n",
                     nS, nC, ctx->nSvcZones, nNeed, nRej);
    }
}

}  // namespace olfgpu
