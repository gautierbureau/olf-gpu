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
#include "../common/trace.cuh"
#include "../solver/scenario_override_tables.cuh"
#include "../solver/scenario_install.cuh"

namespace olfgpu {

// OLF's SecondaryVoltageControlOuterLoop, per (scenario, zone), with nothing left on the host. The
// observables are the SAME quantities OLF's EquationTerms compute, reached the device's way: a
// controller bus's reactive injection is the sum of its CLOSED-BRANCH side flows (gradient from the
// codegen'd updateClosedBranch, exactly as incrementalFlowControlGatherSensitivityBatch does it) plus its shunts' q = -b*v^2
// (ShuntCompensatorReactiveFlowEquationTerm: dq/dv = -2bv, dq/db = -v^2), and the pilot's calculated V
// is the BUS_V variable itself, so its sensitivity is dx at the pilot row.
constexpr double SECONDARY_VOLTAGE_CONTROL_DV_EPS = 1e-4;                          // OLF DV_EPS (0.1 kV)
constexpr double SECONDARY_VOLTAGE_CONTROL_DK_DIFF_MAX_EPS = 1e-3;                 // OLF DK_DIFF_MAX_EPS (1 MVar)
constexpr double SECONDARY_VOLTAGE_CONTROL_K_SATURATION_EPS = 5e-2;               // OLF K_SATURATION_EPS
constexpr double SECONDARY_VOLTAGE_CONTROL_MAX_TARGET_VOLTAGE_STEP_KV = 2;        // OLF MAX_TARGET_VOLTAGE_STEP_KV
constexpr double SECONDARY_VOLTAGE_CONTROL_MIN_PROGRESS_RATIO = 1e-2;             // OLF MIN_PROGRESS_RATIO
constexpr int SECONDARY_VOLTAGE_CONTROL_MAX_NO_PROGRESS_COUNT = 2;                // OLF MAX_NO_PROGRESS_COUNT

struct SecondaryVoltageControlTables {
    int nControllerBuses, nZones, nReactiveLimitEntries;
    const int* controlledBusVRow; const int* controllerBusVRow; const int* reactiveLimitEntry;
    const double* baseTargetV; const double* minQ; const double* maxQ; const double* loadTargetQ;
    const int* reactiveTermStart; const int* reactiveTermKind; const int* reactiveTermSide; const int* reactiveTermElement; const int* reactiveTermShuntBRow;
    const int* zoneStart; const int* zoneControllerBus; const int* zonePilotBusVRow; const double* zoneTargetV;
    double* controlledBusTargetV; double* k0; double* jk; double* jvpp; double* secondaryVoltageControlReactiveLimitTargetV;
    char* controllerBusEnabled; char* needsAdjustment; char* rejected;
    const double* controlledBusNominalV;
    double* zoneLastPilotDv; double* zoneLastDk; int* zoneNoProgress; char* zoneStalled;
    double minPlausibleV, maxPlausibleV;
};

// A controller at a reactive limit (OLF: its generator voltage control disabled, its QLimitType set). Only a run with
// OLF's ReactiveLimitsOuterLoop switches one to PQ, and only that run keys the controller by a reactive-limit entry: entry -1 is
// a controller OLF never switches PV -> PQ.
__device__ inline bool secondaryVoltageControlControllerAtLimit(const SecondaryVoltageControlTables& T, const char* limitType, int s, int c) {
    int e = T.reactiveLimitEntry[c];
    return e >= 0 && limitType[(size_t) s * T.nReactiveLimitEntries + e] != GpuQLimitType::NONE;
}

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
    int nSwitchedToPq = 0;
    for (int k = lo; k < hi; k++) {
        if (secondaryVoltageControlControllerAtLimit(T, limitType, s, T.zoneControllerBus[k])) nSwitchedToPq++;
    }
    if (nSwitchedToPq > 0 && nSwitchedToPq < hi - lo) {
        for (int k = lo; k < hi; k++) {
            int c = T.zoneControllerBus[k];
            if (!secondaryVoltageControlControllerAtLimit(T, limitType, s, c)) continue;
            size_t scenarioEntry = (size_t) s * T.nReactiveLimitEntries + T.reactiveLimitEntry[c];
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
    double kSum = 0.0;
    bool allAbove = true;
    bool allBelow = true;
    for (int k = lo; k < hi; k++) {
        int c = T.zoneControllerBus[k];
        double q = qBuf[(size_t) s * n + T.controllerBusVRow[c]] + T.loadTargetQ[c];
        double kk = secondaryVoltageControlQToK(q, T.minQ[c], T.maxQ[c]);
        T.k0[(size_t) s * T.nControllerBuses + c] = kk;
        kMin = fmin(kMin, kk);
        kMax = fmax(kMax, kk);
        kSum += kk;
        allAbove &= kk > 1.0;
        allBelow &= kk < -1.0;
        T.controllerBusEnabled[(size_t) s * T.nControllerBuses + c] = secondaryVoltageControlControllerAtLimit(T, limitType, s, c) ? 0 : 1;
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
    if (nEnabled == 0) {
        return;
    }
    // OLF processSecondaryVoltageControl: a zone given up as stalled is not looked at again in this load flow (its
    // controllers stay in the zone system the other zones' adjustment solves)
    size_t zs = (size_t) s * T.nZones + z;
    if (T.zoneStalled[zs]) {
        return;
    }
    double kAverage = kSum / (hi - lo);
    double dkDiffMax = kMax - kMin;
    // OLF shouldAdjustSecondaryVoltageControl: the pilot deadband only where the zone's average k leaves room in the
    // direction the pilot needs, and once the pilot target is reached, the k alignment only short of saturation
    bool adjust;
    if (fabs(pilotDv) > SECONDARY_VOLTAGE_CONTROL_DV_EPS) {
        adjust = (pilotDv > 0 && kAverage < 1 - SECONDARY_VOLTAGE_CONTROL_K_SATURATION_EPS)
                || (pilotDv < 0 && kAverage > -1 + SECONDARY_VOLTAGE_CONTROL_K_SATURATION_EPS);
    } else if (allAtLimits) {
        adjust = false;
    } else if (kAverage <= -1 + SECONDARY_VOLTAGE_CONTROL_K_SATURATION_EPS || kAverage >= 1 - SECONDARY_VOLTAGE_CONTROL_K_SATURATION_EPS) {
        adjust = false;
    } else {
        adjust = fabs(dkDiffMax) > SECONDARY_VOLTAGE_CONTROL_DK_DIFF_MAX_EPS;
    }
    if (!adjust) {
        // the zone does not need to be adjusted anymore: its progress tracking starts again
        T.zoneLastPilotDv[zs] = NAN;
        T.zoneLastDk[zs] = NAN;
        T.zoneNoProgress[zs] = 0;
        return;
    }
    // OLF isStalled: the zone's error must decrease by MIN_PROGRESS_RATIO from one adjustment to the next, on the pilot
    // point dv or on the k spread; MAX_NO_PROGRESS_COUNT adjustments in a row without progress give the zone up
    double previousPilotDv = T.zoneLastPilotDv[zs];
    double previousDk = T.zoneLastDk[zs];
    double pilotDvAbs = fabs(pilotDv);
    T.zoneLastPilotDv[zs] = pilotDvAbs;
    T.zoneLastDk[zs] = dkDiffMax;
    if (isnan(previousPilotDv) || pilotDvAbs < previousPilotDv * (1 - SECONDARY_VOLTAGE_CONTROL_MIN_PROGRESS_RATIO)
            || dkDiffMax < previousDk * (1 - SECONDARY_VOLTAGE_CONTROL_MIN_PROGRESS_RATIO)) {
        T.zoneNoProgress[zs] = 0;
        T.needsAdjustment[s] = 1;
        return;
    }
    int count = ++T.zoneNoProgress[zs];
    if (count < SECONDARY_VOLTAGE_CONTROL_MAX_NO_PROGRESS_COUNT) {
        T.needsAdjustment[s] = 1;
    } else {
        T.zoneStalled[zs] = 1;
    }
}

// After the unit-RHS solve for CONTROLLED bus i: OLF's calculateSensiK for every controller j — the sum
// over j's own Q observables of (term gradient . dx_i), scaled by 2/(maxQ-minQ) — and calculateSensiVpp,
// which is dx_i at the zone's pilot row.
__global__ void secondaryVoltageControlGatherSensitivityBatch(SecondaryVoltageControlTables T, int nScenarios, int n, int i, int nElements, int closedBranchOffset, int shuntOffset,
                               const double* closedBranchParameters, const int* closedBranchIndices, const double* shuntParameters, const int* shuntIndices,
                               ClosedBranchParameterOverrides closedBranchParameterOverrides,
                               ShuntAdmittanceOverrides shuntAdmittanceOverrides,
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
            applyShuntOverride(shuntAdmittanceOverrides, s, e, g, b);
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
            if (findTapOverride(closedBranchParameterOverrides, s, e, pOv)) p = pOv;
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

// After the batched solve: OLF clips each controlled bus's target voltage step to MAX_TARGET_VOLTAGE_STEP_KV, and the new
// target to the plausible range, then commits. A singular zone system (cuBLAS info != 0) sends the scenario to the CPU,
// which is where OLF's own MatrixException would land it.
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
    bool moved = false;
    for (int i = 0; i < nC; i++) {
        if (!T.controllerBusEnabled[(size_t) s * nC + i]) continue;
        double nominalV = T.controlledBusNominalV[i];
        double step = dv[i];
        double stepKv = step * nominalV;                     // OLF clipTargetVoltageChange
        if (stepKv > SECONDARY_VOLTAGE_CONTROL_MAX_TARGET_VOLTAGE_STEP_KV) {
            step = SECONDARY_VOLTAGE_CONTROL_MAX_TARGET_VOLTAGE_STEP_KV / nominalV;
        } else if (stepKv < -SECONDARY_VOLTAGE_CONTROL_MAX_TARGET_VOLTAGE_STEP_KV) {
            step = -SECONDARY_VOLTAGE_CONTROL_MAX_TARGET_VOLTAGE_STEP_KV / nominalV;
        }
        double nt = T.controlledBusTargetV[(size_t) s * nC + i] + step;
        if (nt < T.minPlausibleV) {                          // OLF clipTargetVoltageToPlausibleRange
            nt = T.minPlausibleV;
        } else if (nt > T.maxPlausibleV) {
            nt = T.maxPlausibleV;
        }
        T.controlledBusTargetV[(size_t) s * nC + i] = nt;
        moved = true;
    }
    if (moved) {
        switched[s] = 1;                                     // OLF: UNSTABLE, the zone adjusted
    }
}

// Give the moved zone targets to the system: a PV controller's controlled-bus V identity takes the
// new target; one SWITCHED TO PQ keeps its frozen target Q but its switch-back must compare against the moved target. Without
// the ReactiveLimits loop no controller has an entry and the BUS_TARGET_V row is the only place the target lives.
__global__ void secondaryVoltageControlApplyBatch(SecondaryVoltageControlTables T, int nScenarios, int n, const char* outerLoopCheckActive,
                              const char* limitType, double* target) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= T.nControllerBuses * nScenarios) return;
    int c = tid % T.nControllerBuses;
    int s = tid / T.nControllerBuses;
    if (!outerLoopCheckActive[s]) return;
    double tv = T.controlledBusTargetV[(size_t) s * T.nControllerBuses + c];
    if (tv == T.baseTargetV[c]) return;
    if (T.reactiveLimitEntry[c] >= 0) {
        T.secondaryVoltageControlReactiveLimitTargetV[(size_t) s * T.nReactiveLimitEntries + T.reactiveLimitEntry[c]] = tv;
    }
    if (!secondaryVoltageControlControllerAtLimit(T, limitType, s, c)) {
        target[(size_t) s * n + T.controlledBusVRow[c]] = tv;
    }
}

static SecondaryVoltageControlTables secondaryVoltageControlTables(BatchSolverContext* ctx) {
    SecondaryVoltageControlTables T;
    T.nControllerBuses = ctx->secondaryVoltageControl.nSecondaryVoltageControlControllerBuses; T.nZones = ctx->secondaryVoltageControl.nSecondaryVoltageControlZones; T.nReactiveLimitEntries = ctx->reactiveLimits.nReactiveLimitEntries;
    T.controlledBusVRow = ctx->secondaryVoltageControl.dSecondaryVoltageControlControlledBusVRow.as<int>(); T.controllerBusVRow = ctx->secondaryVoltageControl.dSecondaryVoltageControlControllerBusVRow.as<int>(); T.reactiveLimitEntry = ctx->secondaryVoltageControl.dSecondaryVoltageControlReactiveLimitEntry.as<int>();
    T.baseTargetV = ctx->secondaryVoltageControl.dSecondaryVoltageControlBaseTargetV.as<double>(); T.minQ = ctx->secondaryVoltageControl.dSecondaryVoltageControlMinQ.as<double>();
    T.maxQ = ctx->secondaryVoltageControl.dSecondaryVoltageControlMaxQ.as<double>(); T.loadTargetQ = ctx->secondaryVoltageControl.dSecondaryVoltageControlLoadTargetQ.as<double>();
    T.reactiveTermStart = ctx->secondaryVoltageControl.dSecondaryVoltageControlReactiveTermStart.as<int>(); T.reactiveTermKind = ctx->secondaryVoltageControl.dSecondaryVoltageControlReactiveTermKind.as<int>();
    T.reactiveTermSide = ctx->secondaryVoltageControl.dSecondaryVoltageControlReactiveTermSide.as<int>();
    T.reactiveTermElement = ctx->secondaryVoltageControl.dSecondaryVoltageControlReactiveTermElement.as<int>(); T.reactiveTermShuntBRow = ctx->secondaryVoltageControl.dSecondaryVoltageControlReactiveTermShuntBRow.as<int>();
    T.zoneStart = ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneStart.as<int>(); T.zoneControllerBus = ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneControllerBus.as<int>();
    T.zonePilotBusVRow = ctx->secondaryVoltageControl.dSecondaryVoltageControlZonePilotBusVRow.as<int>(); T.zoneTargetV = ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneTargetV.as<double>();
    T.controlledBusTargetV = ctx->secondaryVoltageControl.dSecondaryVoltageControlTargetV.as<double>(); T.k0 = ctx->secondaryVoltageControl.dSecondaryVoltageControlK0.as<double>();
    T.jk = ctx->secondaryVoltageControl.dSecondaryVoltageControlJk.as<double>(); T.jvpp = ctx->secondaryVoltageControl.dSecondaryVoltageControlJvpp.as<double>(); T.secondaryVoltageControlReactiveLimitTargetV = ctx->secondaryVoltageControl.dSecondaryVoltageControlReactiveLimitTargetV.as<double>();
    T.controllerBusEnabled = ctx->secondaryVoltageControl.dSecondaryVoltageControlControllerBusEnabled.as<char>(); T.needsAdjustment = ctx->secondaryVoltageControl.dSecondaryVoltageControlNeedsAdjustment.as<char>(); T.rejected = ctx->secondaryVoltageControl.dSecondaryVoltageControlRejected.as<char>();
    T.minPlausibleV = ctx->secondaryVoltageControl.secondaryVoltageControlMinPlausibleTargetV; T.maxPlausibleV = ctx->secondaryVoltageControl.secondaryVoltageControlMaxPlausibleTargetV;
    T.controlledBusNominalV = ctx->secondaryVoltageControl.dSecondaryVoltageControlControlledBusNominalV.as<double>();
    T.zoneLastPilotDv = ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneLastPilotDv.as<double>(); T.zoneLastDk = ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneLastDk.as<double>();
    T.zoneNoProgress = ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneNoProgress.as<int>(); T.zoneStalled = ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneStalled.as<char>();
    return T;
}

// Per-solve reset. Secondary voltage control: zone targets back at the network's own, nothing adjusted yet.
void secondaryVoltageControlResetSolve(BatchSolverContext* ctx, DriverPass&) {
    const size_t nc = static_cast<size_t>(ctx->secondaryVoltageControl.nSecondaryVoltageControlControllerBuses);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    resetInstalledRows(ctx, ctx->secondaryVoltageControl.dSecondaryVoltageControlTargetV.as<double>(), nc, ctx->secondaryVoltageControl.hSecondaryVoltageControlBaseTargetV.data());
    resetInstalledRows(ctx, ctx->secondaryVoltageControl.dSecondaryVoltageControlNeedsAdjustment.as<char>(), 1, nullptr, char(0));
    resetInstalledRows(ctx, ctx->secondaryVoltageControl.dSecondaryVoltageControlRejected.as<char>(), 1, nullptr, char(0));
    resetInstalledRows(ctx, ctx->secondaryVoltageControl.dSecondaryVoltageControlControllerBusEnabled.as<char>(), nc, nullptr, char(1));
    // OLF's ContextData is created per load flow (SecondaryVoltageControlOuterLoop.initialize)
    const size_t nz = static_cast<size_t>(ctx->secondaryVoltageControl.nSecondaryVoltageControlZones);
    resetInstalledRows(ctx, ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneLastPilotDv.as<double>(), nz, nullptr, std::nan(""));
    resetInstalledRows(ctx, ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneLastDk.as<double>(), nz, nullptr, std::nan(""));
    resetInstalledRows(ctx, ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneNoProgress.as<int>(), nz, nullptr, 0);
    resetInstalledRows(ctx, ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneStalled.as<char>(), nz, nullptr, char(0));
}

void secondaryVoltageControlStep(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    const int& outer = drv.outer;
    std::vector<char>& active = drv.active;
    std::vector<char>& loopRejected = drv.loopRejected;
    int nC = ctx->secondaryVoltageControl.nSecondaryVoltageControlControllerBuses;
    if (nC <= 0) return;
    if (ctx->linearSolver.mode != GpuLinearSolveMode::UBATCH && ctx->linearSolver.mode != GpuLinearSolveMode::ROBUST
            && ctx->linearSolver.mode != GpuLinearSolveMode::BATCHED_LU) {
        static bool warned = false;
        if (!warned) { std::fprintf(stderr, "in-kernel SecondaryVc needs a UBATCH (3), robust (6) or batched LU (7) mode; mode %d runs it as stable\n", ctx->linearSolver.mode); warned = true; }
        return;
    }
    CUDA_CHECK(cudaMemsetAsync(ctx->secondaryVoltageControl.dSecondaryVoltageControlNeedsAdjustment.p, 0, nS, ctx->stream));
    secondaryVoltageControlTriggerBatch<<<grid(ctx->secondaryVoltageControl.nSecondaryVoltageControlZones * nS), BLOCK, 0, ctx->stream>>>(secondaryVoltageControlTables(ctx), nS, n,
            ctx->dOuterLoopCheckActive.as<char>(), ctx->dQbuf.as<double>(), ctx->dX.as<double>(),
            ctx->reactiveLimits.dReactiveLimitQLimitType.as<char>(), ctx->reactiveLimits.dReactiveLimitSavedRowMode.as<char>(),
            ctx->dMode.as<int>(), ctx->dTarget.as<double>());
    std::vector<char> need(nS);
    CUDA_CHECK(cudaMemcpyAsync(need.data(), ctx->secondaryVoltageControl.dSecondaryVoltageControlNeedsAdjustment.p, nS, cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    bool any = false;
    for (char c : need) any |= c != 0;
    if (!any) return;                                    // every zone inside its bands: OLF's STABLE
    std::vector<char> ones(nS, 1);
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, ones.data(), nS, cudaMemcpyHostToDevice));
    fillActiveBatch(ctx, drv, false);
    bool first = true;
    for (int i = 0; i < nC; i++) {
        int row = ctx->structure->rowMap[ctx->secondaryVoltageControl.hSecondaryVoltageControlControlledBusVRow[i]];
        if (row < 0) continue;                           // controlled bus not a live identity: insensitive
        setUnitRhsBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, row, ctx->dB.as<double>());
        repairZeroDiagonalRightHandSide(ctx, nS, ctx->dB.as<double>());
        sensitivitySolve(ctx, drv, first);
        secondaryVoltageControlGatherSensitivityBatch<<<grid(nC * nS), BLOCK, 0, ctx->stream>>>(secondaryVoltageControlTables(ctx), nS, n, i,
                ctx->nElements, ctx->closedBranchOffset, ctx->shuntOffset,
                ctx->structure->closedBranch.parameters.as<double>(), ctx->structure->closedBranch.indices.as<int>(),
                ctx->structure->shunt.parameters.as<double>(), ctx->structure->shunt.indices.as<int>(),
                closedBranchParameterOverrides(ctx),
                shuntAdmittanceOverrides(ctx),
                ctx->dDisabled.as<char>(), ctx->dX.as<double>(), ctx->dDx.as<double>());
    }
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, active.data(), nS, cudaMemcpyHostToDevice));   // as it was
    // ---- the zone systems: assembled into device memory, solved by cuBLAS's BATCHED dense LU ----
    if (ctx->secondaryVoltageControl.secondaryVoltageControlBlasHandle == nullptr) {
        CUBLAS_CHECK(cublasCreate(&ctx->secondaryVoltageControl.secondaryVoltageControlBlasHandle));
    }
    CUBLAS_CHECK(cublasSetStream(ctx->secondaryVoltageControl.secondaryVoltageControlBlasHandle, ctx->stream));
    secondaryVoltageControlAssembleBatch<<<grid(nC * nS), BLOCK, 0, ctx->stream>>>(secondaryVoltageControlTables(ctx), nS, n,
            ctx->dOuterLoopCheckActive.as<char>(), ctx->secondaryVoltageControl.dSecondaryVoltageControlControllerBusZone.as<int>(), ctx->dX.as<double>(),
            ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneMatrix.as<double>(), ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneRightHandSide.as<double>());
    secondaryVoltageControlPointersBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(nS, nC,
            ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneMatrix.as<double>(), ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneRightHandSide.as<double>(),
            reinterpret_cast<double**>(ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneMatrixPointers.p), reinterpret_cast<double**>(ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneRightHandSidePointers.p));
    CUBLAS_CHECK(cublasDgetrfBatched(ctx->secondaryVoltageControl.secondaryVoltageControlBlasHandle, nC,
            reinterpret_cast<double**>(ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneMatrixPointers.p), nC,
            ctx->secondaryVoltageControl.dSecondaryVoltageControlZonePivots.as<int>(), ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneFactorizationInfo.as<int>(), nS));
    int trsInfo = 0;
    CUBLAS_CHECK(cublasDgetrsBatched(ctx->secondaryVoltageControl.secondaryVoltageControlBlasHandle, CUBLAS_OP_N, nC, 1,
            reinterpret_cast<const double* const*>(ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneMatrixPointers.p), nC,
            ctx->secondaryVoltageControl.dSecondaryVoltageControlZonePivots.as<int>(),
            reinterpret_cast<double**>(ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneRightHandSidePointers.p), nC, &trsInfo, nS));
    if (trsInfo != 0) {
        throw std::runtime_error("cublasDgetrsBatched rejected its arguments");
    }
    secondaryVoltageControlCommitBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(secondaryVoltageControlTables(ctx), nS,
            ctx->dOuterLoopCheckActive.as<char>(), ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneFactorizationInfo.as<int>(), ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneRightHandSide.as<double>(),
            ctx->dOuterLoopUnstable.as<char>());
    secondaryVoltageControlApplyBatch<<<grid(nC * nS), BLOCK, 0, ctx->stream>>>(secondaryVoltageControlTables(ctx), nS, n,
            ctx->dOuterLoopCheckActive.as<char>(), ctx->reactiveLimits.dReactiveLimitQLimitType.as<char>(), ctx->dTarget.as<double>());
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    std::vector<char> rej(nS);
    CUDA_CHECK(cudaMemcpy(rej.data(), ctx->secondaryVoltageControl.dSecondaryVoltageControlRejected.p, nS, cudaMemcpyDeviceToHost));
    int nRej = 0;
    for (int s2 = 0; s2 < nS; s2++) {
        if (rej[s2]) { loopRejected[s2] = 1; nRej++; traceReject(s2, "svc-kernel-reject", outer, 0); }
    }
    if (traceOn(Trace::BATCH_TRACE)) {
        int nNeed = 0;
        for (char c : need) nNeed += c ? 1 : 0;
        std::fprintf(stderr, "SVC_KERNEL nS=%d nControllerBuses=%d nZones=%d adjusting=%d rejected=%d\n",
                     nS, nC, ctx->secondaryVoltageControl.nSecondaryVoltageControlZones, nNeed, nRej);
    }
}

}  // namespace olfgpu
