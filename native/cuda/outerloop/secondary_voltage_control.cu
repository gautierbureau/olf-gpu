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
#include "../common/codes/gpu_reactive_term_kind.cuh"
#include "../common/codes/gpu_row_mode.cuh"
#include "../common/codes/gpu_two_sides.cuh"
#include <cublas_v2.h>
#include "../equations/generated/closed_branch.cuh"
#include "../solver/driver_steps.cuh"
#include "../equations/scenario_overrides.cuh"
#include "flow_observation.cuh"
#include "sensitivity.cuh"
#include "secondary_voltage_control.cuh"

namespace olfgpu {

// OLF's SecondaryVoltageControlOuterLoop, per (scenario, zone), with nothing left on the host. The
// observables are the SAME quantities OLF's EquationTerms compute, reached the device's way: a
// controller bus's reactive injection is the sum of its CLOSED-BRANCH side flows (gradient from the
// codegen'd updateClosedBranch, exactly as incrementalFlowControlGatherSensitivityBatch does it) plus its shunts' q = -b*v^2
// (ShuntCompensatorReactiveFlowEquationTerm: dq/dv = -2bv, dq/db = -v^2), and the pilot's calculated V
// is the BUS_V variable itself, so its sensitivity is dx at the pilot row.
constexpr double SECONDARY_VOLTAGE_CONTROL_DV_EPS = 1e-4;                          // OLF DV_EPS (0.1 kV)
constexpr double SECONDARY_VOLTAGE_CONTROL_DK_DIFF_MAX_EPS = 1e-3;                 // OLF DK_DIFF_MAX_EPS (1 MVar)

struct SecondaryVoltageControlTables {
    int nControllerBuses, nZones, nReactiveLimitEntries;
    const int* controlledBusVRow; const int* controllerBusVRow; const int* reactiveLimitEntry;
    const double* baseTargetV; const double* minQ; const double* maxQ; const double* loadTargetQ;
    const int* reactiveTermStart; const int* reactiveTermKind; const int* reactiveTermSide; const int* reactiveTermElement; const int* reactiveTermShuntBRow;
    const int* zoneStart; const int* zoneControllerBus; const int* zonePilotBusVRow; const double* zoneTargetV;
    double* controlledBusTargetV; double* k0; double* jk; double* jvpp; double* secondaryVoltageControlReactiveLimitTargetV;
    char* controllerBusEnabled; char* needsAdjustment; char* rejected;
    double minPlausibleV, maxPlausibleV;
};

__device__ inline double secondaryVoltageControlQToK(double q, double minQ, double maxQ) {
    return (2.0 * q - maxQ - minQ) / (maxQ - minQ);          // OLF qToK
}

// OLF check(): tryToReEnableHelpfulControllerBuses FIRST (a zone with SOME but not all controllers at a
// reactive limit gives every limited one back its voltage control), then the per-zone trigger on the
// pilot deadband and the K spread. The re-enable runs whether or not the zone then adjusts — OLF mutates
// the network before it decides.
__global__ void secondaryVoltageControlTriggerBatch(SecondaryVoltageControlTables T, int nScenarios, int n, const char* outerLoopCheckActive,
                                const double* qBuf, const double* x, char* limitType, const char* savedMode,
                                int* mode, double* target) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= T.nZones * nScenarios) return;
    int z = tid % T.nZones;
    int s = tid / T.nZones;
    if (!outerLoopCheckActive[s]) return;
    int lo = T.zoneStart[z];
    int hi = T.zoneStart[z + 1];
    // ---- tryToReEnableHelpfulControllerBuses ----
    int nPinned = 0;
    for (int k = lo; k < hi; k++) {
        if (limitType[(size_t) s * T.nReactiveLimitEntries + T.reactiveLimitEntry[T.zoneControllerBus[k]]] != GpuQLimitType::NONE) nPinned++;
    }
    if (nPinned > 0 && nPinned < hi - lo) {
        for (int k = lo; k < hi; k++) {
            int c = T.zoneControllerBus[k];
            size_t scenarioEntry = (size_t) s * T.nReactiveLimitEntries + T.reactiveLimitEntry[c];
            if (limitType[scenarioEntry] == GpuQLimitType::NONE) continue;
            size_t idx = (size_t) s * n + T.controllerBusVRow[c];
            char sm = savedMode[scenarioEntry];
            mode[idx] = (sm == GpuRowMode::BUS_TARGET_V_WITH_SLOPE) ? GpuRowMode::BUS_TARGET_V_WITH_SLOPE : GpuRowMode::FIXED_VARIABLE;                  // back to its PV form
            target[idx] = T.controlledBusTargetV[(size_t) s * T.nControllerBuses + c];
            limitType[scenarioEntry] = GpuQLimitType::NONE;
        }
    }
    // ---- the zone's trigger: pilot deadband, or K spread while not all controllers are at a limit ----
    double pilotDv = T.zoneTargetV[z] - x[(size_t) s * n + T.zonePilotBusVRow[z]];
    double kMin = 1e300;
    double kMax = -1e300;
    bool allAbove = true;
    bool allBelow = true;
    for (int k = lo; k < hi; k++) {
        int c = T.zoneControllerBus[k];
        double q = qBuf[(size_t) s * n + T.controllerBusVRow[c]] + T.loadTargetQ[c];
        double kk = secondaryVoltageControlQToK(q, T.minQ[c], T.maxQ[c]);
        T.k0[(size_t) s * T.nControllerBuses + c] = kk;
        kMin = fmin(kMin, kk);
        kMax = fmax(kMax, kk);
        allAbove &= kk > 1.0;
        allBelow &= kk < -1.0;
        T.controllerBusEnabled[(size_t) s * T.nControllerBuses + c] = limitType[(size_t) s * T.nReactiveLimitEntries + T.reactiveLimitEntry[c]] == GpuQLimitType::NONE ? 1 : 0;
    }
    // OLF filters the zone OUT of the check when it has no controller whose voltage control is enabled
    // (findAnyControlledBusWithAtLeastOneControllerBusWithVoltageControlEnabled): every controller at a
    // reactive limit means the zone cannot act, so it is not even looked at. Without this the pilot
    // deadband still flags the scenario, and the sensitivity solve that follows adjusts nothing while
    // leaving the driver's factorization state disturbed — 1.0e-10 kV on ieee14's z1.
    int nEnabled = 0;
    for (int k = lo; k < hi; k++) {
        if (T.controllerBusEnabled[(size_t) s * T.nControllerBuses + T.zoneControllerBus[k]]) nEnabled++;
    }
    bool allAtLimits = allAbove || allBelow;
    if (nEnabled > 0 && (fabs(pilotDv) > SECONDARY_VOLTAGE_CONTROL_DV_EPS || (!allAtLimits && fabs(kMax - kMin) > SECONDARY_VOLTAGE_CONTROL_DK_DIFF_MAX_EPS))) {
        T.needsAdjustment[s] = 1;
    }
}

// After the unit-RHS solve for CONTROLLED bus i: OLF's calculateSensiK for every controller j — the sum
// over j's own Q observables of (term gradient . dx_i), scaled by 2/(maxQ-minQ) — and calculateSensiVpp,
// which is dx_i at the zone's pilot row.
__global__ void secondaryVoltageControlGatherSensitivityBatch(SecondaryVoltageControlTables T, int nScenarios, int n, int i, int nElements, int closedBranchOffset, int shuntOffset,
                               const double* closedBranchParameters, const int* closedBranchIndices, const double* shuntParameters, const int* shuntIndices,
                               int nClosedBranchParameterOverrides, const int* closedBranchParameterOverrideScenarioElement, const double* closedBranchParameterOverrideValues,
                               int nShuntAdmittanceOverrides, const int* shuntAdmittanceOverrideScenarioElement, const double* shuntAdmittanceOverrideDelta,
                               const char* disabled, const double* x, const double* dx) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= T.nControllerBuses * nScenarios) return;
    int j = tid % T.nControllerBuses;
    int s = tid / T.nControllerBuses;
    const double* xs = x + (size_t) s * n;
    const double* dxs = dx + (size_t) s * n;
    double sq = 0.0;
    for (int k = T.reactiveTermStart[j]; k < T.reactiveTermStart[j + 1]; k++) {
        int elem = T.reactiveTermElement[k];
        // NO isDisabled test here, deliberately: OLF's calculateSensiQ sums the controller bus's branch
        // terms as they stand, an outaged one included. Replicating the quirk keeps us on its trajectory.
        if (T.reactiveTermKind[k] == GpuReactiveTermKind::SHUNT_COMPENSATOR_REACTIVE_FLOW) {  // shunt: q = -b v^2
            int e = elem - shuntOffset;
            if (e < 0) continue;
            double g = shuntParameters[e * 2];
            double b = shuntParameters[e * 2 + 1];
            applyShuntOverride(nShuntAdmittanceOverrides, shuntAdmittanceOverrideScenarioElement, shuntAdmittanceOverrideDelta, s, e, g, b);
            int vCol = shuntIndices[e * 5 + 2];
            int bCol = shuntIndices[e * 5 + 3];                      // SHUNT_B row when b is a solved variable
            if (bCol >= 0) b = xs[bCol];
            double v = xs[vCol];
            sq += (-2.0 * b * v) * dxs[vCol] + (bCol >= 0 ? (-v * v) * dxs[bCol] : 0.0);
        } else {
            int e = elem - closedBranchOffset;
            if (e < 0) continue;
            const double* p = closedBranchParameters + e * 8;
            double pOv[8];
            if (findTapOverride(nClosedBranchParameterOverrides, closedBranchParameterOverrideScenarioElement, closedBranchParameterOverrideValues, s, e, pOv)) p = pOv;
            const int* q = closedBranchIndices + e * 12;
            sq += controlledFlowSensitivity(T.reactiveTermSide[k], GpuControlledFlow::REACTIVE_POWER, p, xs, q, dxs);  // getQ1/getQ2(branch).calculateSensi
        }
    }
    T.jk[((size_t) s * T.nControllerBuses + i) * T.nControllerBuses + j] = 2.0 * sq / (T.maxQ[j] - T.minQ[j]);
    if (j == 0) {
        // the pilot sensitivity of controlled bus i at EVERY zone's pilot: OLF's pilot row of zone z (createJvpp
        // over allControlledBuses) carries each controlled bus's response at THAT pilot, other zones' buses
        // included - a controller in one zone moves another zone's pilot through the network. Stored for i's
        // own zone only, the pilot row of a multi-zone system read the other zones' columns at their own pilots.
        for (int z = 0; z < T.nZones; z++) {
            T.jvpp[((size_t) s * T.nZones + z) * T.nControllerBuses + i] = dxs[T.zonePilotBusVRow[z]];
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
__global__ void secondaryVoltageControlAssembleBatch(SecondaryVoltageControlTables T, int nScenarios, int n, const char* outerLoopCheckActive,
                                 const int* controllerBusZone, const double* x, double* mat, double* rhs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= T.nControllerBuses * nScenarios) return;
    int i = tid % T.nControllerBuses;
    int s = tid / T.nControllerBuses;
    int nC = T.nControllerBuses;
    double* m = mat + (size_t) s * nC * nC;
    double* r = rhs + (size_t) s * nC;
    bool live = outerLoopCheckActive[s] && T.needsAdjustment[s];
    if (!live || !T.controllerBusEnabled[(size_t) s * nC + i]) {
        for (int j = 0; j < nC; j++) {
            m[(size_t) j * nC + i] = (i == j) ? 1.0 : 0.0;   // identity row: this controller does not move
        }
        r[i] = 0.0;
        return;
    }
    int z = controllerBusZone[i];
    int lo = T.zoneStart[z];
    int hi = T.zoneStart[z + 1];
    int count = 0;
    int last = -1;
    for (int k = lo; k < hi; k++) {
        int c = T.zoneControllerBus[k];
        if (T.controllerBusEnabled[(size_t) s * nC + c]) {
            count++;
            last = c;
        }
    }
    if (i == last) {                                         // the zone's LAST controlled bus carries the pilot row
        for (int j = 0; j < nC; j++) {                       // every enabled controlled bus's response at pilot z
            m[(size_t) j * nC + i] = T.controllerBusEnabled[(size_t) s * nC + j]
                    ? T.jvpp[((size_t) s * T.nZones + z) * nC + j] : 0.0;
        }
        r[i] = T.zoneTargetV[z] - x[(size_t) s * n + T.zonePilotBusVRow[z]];
        return;
    }
    for (int j = 0; j < nC; j++) {                           // b = A . Jk^T, restricted to enabled columns
        if (!T.controllerBusEnabled[(size_t) s * nC + j]) {
            m[(size_t) j * nC + i] = 0.0;
            continue;
        }
        double v = 0.0;
        for (int k = lo; k < hi; k++) {
            int c = T.zoneControllerBus[k];
            if (!T.controllerBusEnabled[(size_t) s * nC + c]) continue;
            double a = (c == i) ? 1.0 - 1.0 / count : -1.0 / count;
            v += a * T.jk[((size_t) s * nC + j) * nC + c];
        }
        m[(size_t) j * nC + i] = v;
    }
    double rv = 0.0;                                         // rhs = -A . k0
    for (int k = lo; k < hi; k++) {
        int c = T.zoneControllerBus[k];
        if (!T.controllerBusEnabled[(size_t) s * nC + c]) continue;
        double a = (c == i) ? 1.0 - 1.0 / count : -1.0 / count;
        rv += a * T.k0[(size_t) s * nC + c];
    }
    r[i] = -rv;
}

// The pointer arrays cuBLAS's batched API takes, one entry per scenario.
__global__ void secondaryVoltageControlPointersBatch(int nScenarios, int nC, double* mat, double* rhs,
                                 double** matPtr, double** rhsPtr) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    matPtr[s] = mat + (size_t) s * nC * nC;
    rhsPtr[s] = rhs + (size_t) s * nC;
}

// After the batched solve: OLF's plausibility test on the new targets, then commit. A singular zone
// system (cuBLAS info != 0) or an implausible target sends the scenario to the CPU, which is where
// OLF's own MatrixException / FAILED verdict would land it.
__global__ void secondaryVoltageControlCommitBatch(SecondaryVoltageControlTables T, int nScenarios, const char* outerLoopCheckActive,
                               const int* info, const double* rhs, char* switched) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios || !outerLoopCheckActive[s] || !T.needsAdjustment[s]) return;
    int nC = T.nControllerBuses;
    if (info[s] != 0) {
        T.rejected[s] = 1;
        return;
    }
    const double* dv = rhs + (size_t) s * nC;
    for (int i = 0; i < nC; i++) {
        if (!T.controllerBusEnabled[(size_t) s * nC + i]) continue;
        double nt = T.controlledBusTargetV[(size_t) s * nC + i] + dv[i];
        if (!(nt >= T.minPlausibleV && nt <= T.maxPlausibleV)) {
            T.rejected[s] = 1;
            return;
        }
    }
    bool moved = false;
    for (int i = 0; i < nC; i++) {
        if (!T.controllerBusEnabled[(size_t) s * nC + i]) continue;
        T.controlledBusTargetV[(size_t) s * nC + i] += dv[i];
        moved = true;
    }
    if (moved) {
        switched[s] = 1;                                     // OLF: UNSTABLE, the zone adjusted
    }
}

// Give the moved zone targets to the system: an UNPINNED controller's controlled-bus V identity takes the
// new target; a PINNED one keeps its Q pin but its switch-back must compare against the moved target.
__global__ void secondaryVoltageControlApplyBatch(SecondaryVoltageControlTables T, int nScenarios, int n, const char* outerLoopCheckActive,
                              const char* limitType, double* target) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= T.nControllerBuses * nScenarios) return;
    int c = tid % T.nControllerBuses;
    int s = tid / T.nControllerBuses;
    if (!outerLoopCheckActive[s]) return;
    double tv = T.controlledBusTargetV[(size_t) s * T.nControllerBuses + c];
    if (tv == T.baseTargetV[c]) return;
    T.secondaryVoltageControlReactiveLimitTargetV[(size_t) s * T.nReactiveLimitEntries + T.reactiveLimitEntry[c]] = tv;
    if (limitType[(size_t) s * T.nReactiveLimitEntries + T.reactiveLimitEntry[c]] == GpuQLimitType::NONE) {
        target[(size_t) s * n + T.controlledBusVRow[c]] = tv;
    }
}

static SecondaryVoltageControlTables secondaryVoltageControlTables(BatchSolverContext* ctx) {
    SecondaryVoltageControlTables T;
    T.nControllerBuses = ctx->nSecondaryVoltageControlControllerBuses; T.nZones = ctx->nSecondaryVoltageControlZones; T.nReactiveLimitEntries = ctx->nReactiveLimitEntries;
    T.controlledBusVRow = ctx->dSecondaryVoltageControlControlledBusVRow.as<int>(); T.controllerBusVRow = ctx->dSecondaryVoltageControlControllerBusVRow.as<int>(); T.reactiveLimitEntry = ctx->dSecondaryVoltageControlReactiveLimitEntry.as<int>();
    T.baseTargetV = ctx->dSecondaryVoltageControlBaseTargetV.as<double>(); T.minQ = ctx->dSecondaryVoltageControlMinQ.as<double>();
    T.maxQ = ctx->dSecondaryVoltageControlMaxQ.as<double>(); T.loadTargetQ = ctx->dSecondaryVoltageControlLoadTargetQ.as<double>();
    T.reactiveTermStart = ctx->dSecondaryVoltageControlReactiveTermStart.as<int>(); T.reactiveTermKind = ctx->dSecondaryVoltageControlReactiveTermKind.as<int>();
    T.reactiveTermSide = ctx->dSecondaryVoltageControlReactiveTermSide.as<int>();
    T.reactiveTermElement = ctx->dSecondaryVoltageControlReactiveTermElement.as<int>(); T.reactiveTermShuntBRow = ctx->dSecondaryVoltageControlReactiveTermShuntBRow.as<int>();
    T.zoneStart = ctx->dSecondaryVoltageControlZoneStart.as<int>(); T.zoneControllerBus = ctx->dSecondaryVoltageControlZoneControllerBus.as<int>();
    T.zonePilotBusVRow = ctx->dSecondaryVoltageControlZonePilotBusVRow.as<int>(); T.zoneTargetV = ctx->dSecondaryVoltageControlZoneTargetV.as<double>();
    T.controlledBusTargetV = ctx->dSecondaryVoltageControlTargetV.as<double>(); T.k0 = ctx->dSecondaryVoltageControlK0.as<double>();
    T.jk = ctx->dSecondaryVoltageControlJk.as<double>(); T.jvpp = ctx->dSecondaryVoltageControlJvpp.as<double>(); T.secondaryVoltageControlReactiveLimitTargetV = ctx->dSecondaryVoltageControlReactiveLimitTargetV.as<double>();
    T.controllerBusEnabled = ctx->dSecondaryVoltageControlControllerBusEnabled.as<char>(); T.needsAdjustment = ctx->dSecondaryVoltageControlNeedsAdjustment.as<char>(); T.rejected = ctx->dSecondaryVoltageControlRejected.as<char>();
    T.minPlausibleV = ctx->secondaryVoltageControlMinPlausibleTargetV; T.maxPlausibleV = ctx->secondaryVoltageControlMaxPlausibleTargetV;
    return T;
}

// Per-solve reset. Secondary voltage control: zone targets back at the network's own, nothing adjusted yet.
void secondaryVoltageControlResetSolve(BatchSolverContext* ctx, DriverPass& drv) {
    const int& nS = drv.nS;
    size_t nsc = static_cast<size_t>(nS) * ctx->nSecondaryVoltageControlControllerBuses;
    std::vector<double> tv(nsc);
    for (int s2 = 0; s2 < nS; s2++) {
        for (int c = 0; c < ctx->nSecondaryVoltageControlControllerBuses; c++) tv[static_cast<size_t>(s2) * ctx->nSecondaryVoltageControlControllerBuses + c] = ctx->hSecondaryVoltageControlBaseTargetV[c];
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(cudaMemcpy(ctx->dSecondaryVoltageControlTargetV.p, tv.data(), nsc * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ctx->dSecondaryVoltageControlNeedsAdjustment.p, 0, nS));
    CUDA_CHECK(cudaMemset(ctx->dSecondaryVoltageControlRejected.p, 0, nS));
    CUDA_CHECK(cudaMemset(ctx->dSecondaryVoltageControlControllerBusEnabled.p, 1, nsc));
}

void secondaryVoltageControlStep(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    const int& outer = drv.outer;
    std::vector<char>& active = drv.active;
    std::vector<char>& loopRejected = drv.loopRejected;
    int nC = ctx->nSecondaryVoltageControlControllerBuses;
    if (nC <= 0) return;
    if (ctx->mode != GpuLinearSolveMode::UBATCH && ctx->mode != GpuLinearSolveMode::ROBUST) {
        static bool warned = false;
        if (!warned) { std::fprintf(stderr, "in-kernel SecondaryVc needs a UBATCH (3) or robust (6) mode; mode %d runs it as stable\n", ctx->mode); warned = true; }
        return;
    }
    CUDA_CHECK(cudaMemsetAsync(ctx->dSecondaryVoltageControlNeedsAdjustment.p, 0, nS, ctx->stream));
    secondaryVoltageControlTriggerBatch<<<grid(ctx->nSecondaryVoltageControlZones * nS), BLOCK, 0, ctx->stream>>>(secondaryVoltageControlTables(ctx), nS, n,
            ctx->dOuterLoopCheckActive.as<char>(), ctx->dQbuf.as<double>(), ctx->dX.as<double>(),
            ctx->dReactiveLimitQLimitType.as<char>(), ctx->dReactiveLimitSavedRowMode.as<char>(),
            ctx->dMode.as<int>(), ctx->dTarget.as<double>());
    std::vector<char> need(nS);
    CUDA_CHECK(cudaMemcpyAsync(need.data(), ctx->dSecondaryVoltageControlNeedsAdjustment.p, nS, cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    bool any = false;
    for (char c : need) any |= c != 0;
    if (!any) return;                                    // every zone inside its bands: OLF's STABLE
    std::vector<char> ones(nS, 1);
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, ones.data(), nS, cudaMemcpyHostToDevice));
    fillActiveBatch(ctx, drv, false);
    bool first = true;
    for (int i = 0; i < nC; i++) {
        int row = ctx->structure->rowMap[ctx->hSecondaryVoltageControlControlledBusVRow[i]];
        if (row < 0) continue;                           // controlled bus not a live identity: insensitive
        setUnitRhsBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, row, ctx->dB.as<double>());
        repairZeroDiagonalRightHandSide(ctx, nS, ctx->dB.as<double>());
        sensitivitySolve(ctx, drv, first);
        secondaryVoltageControlGatherSensitivityBatch<<<grid(nC * nS), BLOCK, 0, ctx->stream>>>(secondaryVoltageControlTables(ctx), nS, n, i,
                ctx->nElements, ctx->closedBranchOffset, ctx->shuntOffset,
                ctx->structure->closedBranch.parameters.as<double>(), ctx->structure->closedBranch.indices.as<int>(),
                ctx->structure->shunt.parameters.as<double>(), ctx->structure->shunt.indices.as<int>(),
                ctx->nClosedBranchParameterOverrides, ctx->dClosedBranchParameterOverrideScenarioElement.as<int>(), ctx->dClosedBranchParameterOverrideValues.as<double>(),
                ctx->nShuntAdmittanceOverrides, ctx->dShuntAdmittanceOverrideScenarioElement.as<int>(), ctx->dShuntAdmittanceOverrideDelta.as<double>(),
                ctx->dDisabled.as<char>(), ctx->dX.as<double>(), ctx->dDx.as<double>());
    }
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, active.data(), nS, cudaMemcpyHostToDevice));   // as it was
    // ---- the zone systems: assembled into device memory, solved by cuBLAS's BATCHED dense LU ----
    if (ctx->secondaryVoltageControlBlasHandle == nullptr) {
        CUBLAS_CHECK(cublasCreate(&ctx->secondaryVoltageControlBlasHandle));
    }
    CUBLAS_CHECK(cublasSetStream(ctx->secondaryVoltageControlBlasHandle, ctx->stream));
    secondaryVoltageControlAssembleBatch<<<grid(nC * nS), BLOCK, 0, ctx->stream>>>(secondaryVoltageControlTables(ctx), nS, n,
            ctx->dOuterLoopCheckActive.as<char>(), ctx->dSecondaryVoltageControlControllerBusZone.as<int>(), ctx->dX.as<double>(),
            ctx->dSecondaryVoltageControlZoneMatrix.as<double>(), ctx->dSecondaryVoltageControlZoneRightHandSide.as<double>());
    secondaryVoltageControlPointersBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(nS, nC,
            ctx->dSecondaryVoltageControlZoneMatrix.as<double>(), ctx->dSecondaryVoltageControlZoneRightHandSide.as<double>(),
            reinterpret_cast<double**>(ctx->dSecondaryVoltageControlZoneMatrixPointers.p), reinterpret_cast<double**>(ctx->dSecondaryVoltageControlZoneRightHandSidePointers.p));
    CUBLAS_CHECK(cublasDgetrfBatched(ctx->secondaryVoltageControlBlasHandle, nC,
            reinterpret_cast<double**>(ctx->dSecondaryVoltageControlZoneMatrixPointers.p), nC,
            ctx->dSecondaryVoltageControlZonePivots.as<int>(), ctx->dSecondaryVoltageControlZoneFactorizationInfo.as<int>(), nS));
    int trsInfo = 0;
    CUBLAS_CHECK(cublasDgetrsBatched(ctx->secondaryVoltageControlBlasHandle, CUBLAS_OP_N, nC, 1,
            reinterpret_cast<const double* const*>(ctx->dSecondaryVoltageControlZoneMatrixPointers.p), nC,
            ctx->dSecondaryVoltageControlZonePivots.as<int>(),
            reinterpret_cast<double**>(ctx->dSecondaryVoltageControlZoneRightHandSidePointers.p), nC, &trsInfo, nS));
    if (trsInfo != 0) {
        throw std::runtime_error("cublasDgetrsBatched rejected its arguments");
    }
    secondaryVoltageControlCommitBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(secondaryVoltageControlTables(ctx), nS,
            ctx->dOuterLoopCheckActive.as<char>(), ctx->dSecondaryVoltageControlZoneFactorizationInfo.as<int>(), ctx->dSecondaryVoltageControlZoneRightHandSide.as<double>(),
            ctx->dOuterLoopUnstable.as<char>());
    secondaryVoltageControlApplyBatch<<<grid(nC * nS), BLOCK, 0, ctx->stream>>>(secondaryVoltageControlTables(ctx), nS, n,
            ctx->dOuterLoopCheckActive.as<char>(), ctx->dReactiveLimitQLimitType.as<char>(), ctx->dTarget.as<double>());
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    std::vector<char> rej(nS);
    CUDA_CHECK(cudaMemcpy(rej.data(), ctx->dSecondaryVoltageControlRejected.p, nS, cudaMemcpyDeviceToHost));
    int nRej = 0;
    for (int s2 = 0; s2 < nS; s2++) {
        if (rej[s2]) { loopRejected[s2] = 1; nRej++; traceReject(s2, "svc-kernel-reject", outer, 0); }
    }
    if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
        int nNeed = 0;
        for (char c : need) nNeed += c ? 1 : 0;
        std::fprintf(stderr, "SVC_KERNEL nS=%d nControllerBuses=%d nZones=%d adjusting=%d rejected=%d\n",
                     nS, nC, ctx->nSecondaryVoltageControlZones, nNeed, nRej);
    }
}

}  // namespace olfgpu
