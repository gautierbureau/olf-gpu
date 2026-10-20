// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See reactive_limits_seed.cuh.
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
#include "../common/codes/gpu_voltage_control_role.cuh"
#include "../common/codes/gpu_reactive_limits_entry_flag.cuh"
#include "../common/codes/gpu_row_mode.cuh"
#include "../equations/scatter.cuh"
#include "../fixedmatrix/reactive_relocation.cuh"
#include "../fixedmatrix/cross_identity.cuh"
#include "../equations/scenario_overrides.cuh"
#include "reactive_limits_seed.cuh"

namespace olfgpu {

__global__ void initReactiveLimitQLimitTypeBatch(int nReactiveLimitEntries, int nScenarios, int n, const int* reactiveLimitVoltageRow, const int* reactiveLimitReactivePowerRow,
                                     const double* reactiveLimitTargetV, const char* reactiveLimitFlags,
                                     const int* baseMode, const double* x, char* limitType,
                                     int nReactiveLimitTargetVOverrides, const int* reactiveLimitTargetVOverrideScenarioEntry, const double* reactiveLimitTargetVOverrideValue, const double* secondaryVoltageControlReactiveLimitTargetV,
                                     const char* reactiveLimitVoltageControlRole, const int* reactiveLimitComparedVoltageRow,
                                     const double* target, const double* reactiveLimitBaseMinQ, const double* reactiveLimitBaseMaxQ,
                                     const double* reactiveLimitBaseLoadQ, const char* reactiveLimitBaseQLimitType,
                                     const int* pilotSelfReactiveLimitEntry, int nReactiveRelocationGroups, const char* reactiveLimitGeneratorsLost) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nReactiveLimitEntries * nScenarios) return;
    int i = tid % nReactiveLimitEntries;
    int s = tid / nReactiveLimitEntries;
    // A bus that LOST every generator this scenario (GenOutage, setBatchReactiveLimitGeneratorsLost) is no controller: OLF has no
    // voltage control there at all. Seeded FROZEN, the entry is skipped by the reactive-limit check, reads as
    // "not enabled" to the pilot-self layout and, for a pilot that was one of its own controllers, opens the
    // pilot-self gate so the bus's reactive balance is emitted (on the carrier, then on its own row once every
    // survivor pins). Left at 0 it read as a live controller: the gate suppressed the balance and the outaged
    // pilot's row, flipped to host it, stayed EMPTY (rte6515 GEN-1522: VL-1522_0 held at its 1.122 target).
    if (reactiveLimitGeneratorsLost != nullptr && reactiveLimitGeneratorsLost[(size_t) s * nReactiveLimitEntries + i]) {
        limitType[(size_t) s * nReactiveLimitEntries + i] = GpuQLimitType::FROZEN;
        return;
    }
    int vrow = reactiveLimitVoltageRow[i];
    // the bus's own V variable (VOLTAGE reads) and the row its pin state lives on, which the coupler-tree
    // relocation can move to the parent coupler's DUMMY_Q (see dReactiveLimitReactivePowerRow)
    int qrow = reactiveLimitReactivePowerRow != nullptr ? reactiveLimitReactivePowerRow[i] : vrow;
    if (qrow != vrow && baseMode[qrow] == GpuRowMode::FIXED_VARIABLE) {
        qrow = vrow;                                         // that coupler is open in the BASE: equations at home
    }
    size_t idx = (size_t) s * n + vrow;
    size_t qix = (size_t) s * n + qrow;
    size_t scenarioEntry = (size_t) s * nReactiveLimitEntries + i;
    // OLF's own reason for the base pin, when the host carried it: it distinguishes a
    // realistic-voltage pin (3/4, target = generation targetQ) from a Q-limit pin (1/2, target =
    // the limit). The nearest-limit guess below turns the former into the latter, and the
    // curve-limit drift then re-freezes it AT the limit (rte6515 VL-1251_1: -0.0035 -> -1.0000).
    char olfKind = reactiveLimitBaseQLimitType != nullptr ? reactiveLimitBaseQLimitType[i] : 0;
    if (reactiveLimitVoltageControlRole != nullptr && reactiveLimitVoltageControlRole[i] != GpuVoltageControlRole::LOCAL) {
        // REMOTE group member (plan-B3): base mode 1 at its row is the UNPINNED DISTR_Q share —
        // only the canonical mode 4 marks a base pin. The pilot's V equals the group target at the
        // base solution (the survivors keep controlling), so the v-vs-target heuristic is
        // DEGENERATE here — key the direction on WHICH limit the extracted BUS_TARGET_Q target
        // equals instead (target = limit − loadQ was written by the base activation).
        limitType[scenarioEntry] = GpuQLimitType::NONE;
        if (isEnrolledPilot(pilotSelfReactiveLimitEntry, nReactiveRelocationGroups, i)) {
            // An ENROLLED pilot-self pilot (rotation) keeps its V row an identity while PINNED: the pin's
            // balance rides the carrier (or its own row flipped to a power row once the group lost control),
            // so the mode-4 marker below never appears on its row. OLF's own kind is the only record of its
            // base pin — and without it the seeding left the pilot UNPINNED (pilotSelfCarrierSkip = 1, raw q dropped) on a
            // base that had it pinned: every scenario of the [5000,6000) busbar window failed its first
            // linear solve (976 x LINEAR_SOLVE_FAILED).
            limitType[scenarioEntry] = olfKind >= GpuQLimitType::MIN_Q && olfKind <= GpuQLimitType::MAX_REALISTIC_V ? olfKind : GpuQLimitType::NONE;
            return;
        }
        if (baseMode[qrow] == GpuRowMode::BUS_TARGET_Q) {
            double tgt = target[qix];
            limitType[scenarioEntry] = olfKind != GpuQLimitType::NONE ? olfKind
                    : (fabs(tgt - (reactiveLimitBaseMaxQ[i] - reactiveLimitBaseLoadQ[i])) <= fabs(tgt - (reactiveLimitBaseMinQ[i] - reactiveLimitBaseLoadQ[i])) ? GpuQLimitType::MAX_Q : GpuQLimitType::MIN_Q);
        }
        return;
    }
    // Read the BASE row mode (length n, contingency-free) — NOT the per-scenario mode: a controller flipped to
    // PQ by a per-scenario gen-loss OVERRIDE (its generator is gone) has base mode 0 and must get limitType 0 so
    // the reactive-limits switch-back leaves it PQ. Only a controller that is PQ in the BASE (an actual Q-limit
    // switch during the base loadflow) gets a pinned limit type so a relieving contingency can switch it back.
    if (baseMode[qrow] == GpuRowMode::EQUATION && (reactiveLimitFlags[i] & GpuReactiveLimitsEntryFlag::STAND_BY_AUTOMATON) == 0) {       // PQ in the base case (a real Q-limit switch —
        // NOT a monitoring SVC standing by: that one seeds UNPINNED so only the VoltageMonitoring
        // orchestrator may activate it, exactly like the CPU's null qLimitType)
        double targetV = reactiveLimitTargetVOf(nReactiveLimitTargetVOverrides, reactiveLimitTargetVOverrideScenarioEntry, reactiveLimitTargetVOverrideValue, secondaryVoltageControlReactiveLimitTargetV, nReactiveLimitEntries, s, i, reactiveLimitTargetV[i]);
        limitType[scenarioEntry] = olfKind != GpuQLimitType::NONE ? olfKind
                : ((x[idx] < targetV) ? GpuQLimitType::MAX_Q : GpuQLimitType::MIN_Q);                  // v below target ⇒ maxQ pinned; above ⇒ minQ pinned
    } else {
        limitType[scenarioEntry] = GpuQLimitType::NONE;  // base PV (incl. a per-scenario gen-override flip)
    }
}

__global__ void seedReactiveLimitRowModeFromState(int nReactiveLimitEntries, int nScenarios, int n, const int* reactiveLimitVoltageRow, const int* reactiveLimitReactivePowerRow,
                                    const double* reactiveLimitMinQ, const double* reactiveLimitMaxQ, const double* reactiveLimitLoadQ,
                                    const char* limitType, int* mode, double* target, int perScenarioReactiveLimits,
                                    const char* reactiveLimitVoltageControlRole, double* savedTarget, char* savedMode,
                                    const char* reactiveLimitBaseQLimitType, const int* pilotSelfReactiveLimitEntry, int nReactiveRelocationGroups,
                                    const double* reactiveLimitTargetV, int nReactiveLimitTargetVOverrides, const int* reactiveLimitTargetVOverrideScenarioEntry, const double* reactiveLimitTargetVOverrideValue,
                                    const double* secondaryVoltageControlReactiveLimitTargetV, const char* reactiveLimitLocalOverride, const double* reactiveLimitLocalTargetV,
                                    const double* pinTargetIn, const char* reactiveLimitIslanded) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nReactiveLimitEntries * nScenarios) return;
    int i = tid % nReactiveLimitEntries;
    int s = tid / nReactiveLimitEntries;
    // A controller this scenario islanded: its bus is out of the system and its rows are FROZEN identities. Stamping
    // the continued pin turned its V row into a reactive balance at the frozen target, an island equation OLF does not
    // have, which the first solve then iterated on (rte6515 LINE-3477-2893: 0.0075 MW on every strategy that moves P).
    if (reactiveLimitIslanded != nullptr && reactiveLimitIslanded[(size_t) s * nReactiveLimitEntries + i]) return;
    char qLimitType = limitType[(size_t) s * nReactiveLimitEntries + i];
    // A LOCAL controller pinned in the BASE case that the previous solve released (its exported state says PV): the
    // base activation still has it pinned, and OLF's strategy starts with the bus back on voltage control, exactly as
    // the reactive-limit switch-back left it - the PV form (6 on a cross row) at its voltage target. Left alone it
    // stayed pinned at the base limit: a strategy pair of rte6515 LINE-3273-1812 solved two generators the parent had
    // released at their base Q limits, 0.016 kV at VL-2968 even for a strategy that changes nothing.
    if (qLimitType == GpuQLimitType::NONE && reactiveLimitBaseQLimitType != nullptr && reactiveLimitBaseQLimitType[i] >= GpuQLimitType::MIN_Q && reactiveLimitBaseQLimitType[i] <= GpuQLimitType::MAX_REALISTIC_V && (reactiveLimitVoltageControlRole == nullptr || reactiveLimitVoltageControlRole[i] == GpuVoltageControlRole::LOCAL)
            && !isEnrolledPilot(pilotSelfReactiveLimitEntry, nReactiveRelocationGroups, i)) {
        size_t scenarioEntry = (size_t) s * nReactiveLimitEntries + i;
        bool localOv = reactiveLimitLocalOverride != nullptr && reactiveLimitLocalOverride[scenarioEntry];
        int vrow = reactiveLimitVoltageRow[i];
        int pinRow = vrow;
        if (localOv && reactiveLimitReactivePowerRow != nullptr && reactiveLimitReactivePowerRow[i] >= 0 && reactiveLimitReactivePowerRow[i] != vrow) {
            size_t r2 = (size_t) s * n + reactiveLimitReactivePowerRow[i];
            pinRow = crossIdentityPinRow(mode, r2, (size_t) s * n + vrow, reactiveLimitReactivePowerRow[i], vrow);
        }
        size_t pidx = (size_t) s * n + pinRow;
        if (mode[pidx] == GpuRowMode::EQUATION || mode[pidx] == GpuRowMode::BUS_TARGET_Q) {  // still holding the base pin
            mode[pidx] = fixedVariableMode(pinRow != vrow);
            target[pidx] = localOv ? reactiveLimitLocalTargetV[scenarioEntry] : reactiveLimitTargetVOf(nReactiveLimitTargetVOverrides, reactiveLimitTargetVOverrideScenarioEntry, reactiveLimitTargetVOverrideValue, secondaryVoltageControlReactiveLimitTargetV, nReactiveLimitEntries, s, i, reactiveLimitTargetV[i]);
        }
        return;
    }
    // ...and the same for a REMOTE member, as the reactive-limits pass releases one: its row goes back to its unpinned
    // form - the DISTR_Q share (mode 1), whose target the DISTR_Q pass recomputes, or a self-controlling pilot's voltage
    // identity (mode 0). Left pinned at the base limit, its group shared the pilot's reactive balance one member short
    // (network.xiidm LLJP, the member on bus 5779: mode 4 at -0.2954 where the parent had released it, 5.46 kV).
    if (qLimitType == GpuQLimitType::NONE && reactiveLimitBaseQLimitType != nullptr && reactiveLimitBaseQLimitType[i] >= GpuQLimitType::MIN_Q && reactiveLimitBaseQLimitType[i] <= GpuQLimitType::MAX_REALISTIC_V && reactiveLimitVoltageControlRole != nullptr && reactiveLimitVoltageControlRole[i] != GpuVoltageControlRole::LOCAL
            && !isEnrolledPilot(pilotSelfReactiveLimitEntry, nReactiveRelocationGroups, i)) {
        int vrow = reactiveLimitVoltageRow[i];
        int qrow = reactiveLimitReactivePowerRow != nullptr && reactiveLimitReactivePowerRow[i] >= 0 && reactiveLimitReactivePowerRow[i] != vrow && mode[(size_t) s * n + reactiveLimitReactivePowerRow[i]] != GpuRowMode::FIXED_VARIABLE
                ? reactiveLimitReactivePowerRow[i] : vrow;
        size_t qidx = (size_t) s * n + qrow;
        if (mode[qidx] == GpuRowMode::BUS_TARGET_Q) {  // still holding the base pin
            mode[qidx] = reactiveLimitVoltageControlRole[i] == GpuVoltageControlRole::PILOT_SELF ? GpuRowMode::FIXED_VARIABLE : GpuRowMode::EQUATION;
            target[qidx] = reactiveLimitVoltageControlRole[i] == GpuVoltageControlRole::PILOT_SELF
                    ? reactiveLimitTargetVOf(nReactiveLimitTargetVOverrides, reactiveLimitTargetVOverrideScenarioEntry, reactiveLimitTargetVOverrideValue, secondaryVoltageControlReactiveLimitTargetV, nReactiveLimitEntries, s, i, reactiveLimitTargetV[i]) : 0.0;
        }
        return;
    }
    if (isEnrolledPilot(pilotSelfReactiveLimitEntry, nReactiveRelocationGroups, i)) return;           // its pin never lives on its own V row (rotation)
    if (qLimitType != GpuQLimitType::NONE) {
        // the pin's row (see dReactiveLimitReactivePowerRow), with the relocation's per-scenario diversion: mode 0 at the relocated row
        // means its coupler is open this scenario and the equations are back on the bus's own rows
        int seedRow = reactiveLimitReactivePowerRow != nullptr ? reactiveLimitReactivePowerRow[i] : reactiveLimitVoltageRow[i];
        if (seedRow != reactiveLimitVoltageRow[i] && mode[(size_t) s * n + seedRow] == GpuRowMode::FIXED_VARIABLE) {
            seedRow = reactiveLimitVoltageRow[i];
        }
        size_t idx = (size_t) s * n + seedRow;
        size_t li = perScenarioReactiveLimits ? (size_t) s * nReactiveLimitEntries + i : (size_t) i;
        // What the row looked like BEFORE this seed is what a later switch-back restores. Leaving it
        // unwritten let the restore read whatever the previous solve left in this slot: the buffers are
        // per (entry, SLOT) and a context is reused chunk after chunk, so an entry pinned at seed time
        // and unpinned during the contingency took another scenario's target. Invisible wherever the
        // row's target is recomputed right after (a DISTR_Q share), a small wrong answer where it is
        // not — measured on network.xiidm as an intermittent 8.178e-05 kV shared by every scenario the
        // stale slot reached (the KUMF / KUMD pair swapping reactive power).
        savedTarget[(size_t) s * nReactiveLimitEntries + i] = target[idx];
        savedMode[(size_t) s * nReactiveLimitEntries + i] = static_cast<char>(mode[idx]);
        // a pinned REMOTE member's row is mode 4 (own Q balance), a pinned local one mode 1
        mode[idx] = (reactiveLimitVoltageControlRole != nullptr && reactiveLimitVoltageControlRole[i] != GpuVoltageControlRole::LOCAL) ? GpuRowMode::BUS_TARGET_Q : GpuRowMode::EQUATION;
        // the pinned target the previous solve ENDED with, when it exported one: OLF restores the bus's generation
        // target Q from the parent's end, which is not the current limit when that limit moved by less than the
        // re-freeze tolerance (rte6515 LINE-3273-1812, VL-6095: 0.7518298 frozen, the curve at 0.7518918)
        double pin = pinTargetIn != nullptr ? pinTargetIn[(size_t) s * nReactiveLimitEntries + i] : NAN;
        target[idx] = !isnan(pin) ? pin : (qLimitType == GpuQLimitType::MIN_Q ? reactiveLimitMinQ[li] : reactiveLimitMaxQ[li]) - reactiveLimitLoadQ[li];
    }
}

// The pinned target of every pinned entry at the end of a solve, read on the row the seed above writes it to (NaN for
// an unpinned entry): what a continued solve (a strategy pair) restores, as OLF restores the parent's target Q.
__global__ void exportReactiveLimitTargetQ(int nReactiveLimitEntries, int nScenarios, int n, const int* reactiveLimitVoltageRow, const int* reactiveLimitReactivePowerRow,
                                   const char* limitType, const int* mode, const double* target, const char* reactiveLimitIslanded,
                                   int nReactiveRelocationGroups, const int* pilotSelfReactiveLimitEntry, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows, const char* pilotTargetVDeactivated,
                                   const int* reactiveRelocationDependentBaseGroup, double* out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nReactiveLimitEntries * nScenarios) return;
    int i = tid % nReactiveLimitEntries;
    int s = tid / nReactiveLimitEntries;
    char qLimitType = limitType[(size_t) s * nReactiveLimitEntries + i];
    // an enrolled pilot's pin is its group's flip: the flipped row's target when the group is flipped (what a
    // continued solve's seed flip restores), nothing otherwise (its balance rides a carrier, never its own rows)
    for (int g = 0; pilotSelfReactiveLimitEntry != nullptr && g < nReactiveRelocationGroups; g++) {
        if (pilotSelfReactiveLimitEntry[g] != i) continue;
        bool flipped = qLimitType != GpuQLimitType::NONE && pilotTargetVDeactivated != nullptr && pilotTargetVDeactivated[(size_t) s * nReactiveRelocationGroups + g] && (reactiveRelocationDependentBaseGroup == nullptr || reactiveRelocationDependentBaseGroup[g] < 0);
        out[tid] = flipped ? target[(size_t) s * n + reactiveRelocationCandidateRows[reactiveRelocationGroupStart[g]]] : NAN;
        return;
    }
    // an islanded controller's row holds its bus's frozen value, not a Q: a pair that brings the bus back seeds the
    // pin from its limit, the target the bus kept while it was out
    if (qLimitType == GpuQLimitType::NONE || (reactiveLimitIslanded != nullptr && reactiveLimitIslanded[(size_t) s * nReactiveLimitEntries + i])) {
        out[tid] = NAN;
        return;
    }
    int seedRow = reactiveLimitReactivePowerRow != nullptr ? reactiveLimitReactivePowerRow[i] : reactiveLimitVoltageRow[i];
    if (seedRow != reactiveLimitVoltageRow[i] && mode[(size_t) s * n + seedRow] == GpuRowMode::FIXED_VARIABLE) {
        seedRow = reactiveLimitVoltageRow[i];
    }
    out[tid] = target[(size_t) s * n + seedRow];
}

}  // namespace olfgpu
