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
#include "../equations/scatter.cuh"
#include "../equations/scenario_overrides.cuh"
#include "reactive_limits_seed.cuh"

namespace olfgpu {

__global__ void initRlLimitTypeBatch(int nrl, int nScenarios, int n, const int* rlVrow, const int* rlQrow,
                                     const double* rlVtarget, const char* rlMonitored,
                                     const int* baseMode, const double* x, char* limitType,
                                     int nRlVtOv, const int* rlVtOvSlotIdx, const double* rlVtOvVal, const double* svcRlVt,
                                     const char* rlRemote, const int* rlCmpVrow,
                                     const double* target, const double* rlMinQb, const double* rlMaxQb,
                                     const double* rlLoadQb, const char* rlBaseLt,
                                     const int* psEntry, int nQr, const char* rlGenLost) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nrl * nScenarios) return;
    int i = tid % nrl;
    int s = tid / nrl;
    // A bus that LOST every generator this scenario (GenOutage, setBatchRlGenLost) is no controller: OLF has no
    // voltage control there at all. Seeded FROZEN, the entry is skipped by the reactive-limit check, reads as
    // "not enabled" to the pilot-self layout and, for a pilot that was one of its own controllers, opens the
    // pilot-self gate so the bus's reactive balance is emitted (on the carrier, then on its own row once every
    // survivor pins). Left at 0 it read as a live controller: the gate suppressed the balance and the outaged
    // pilot's row, flipped to host it, stayed EMPTY (rte6515 GEN-1522: VL-1522_0 held at its 1.122 target).
    if (rlGenLost != nullptr && rlGenLost[(size_t) s * nrl + i]) {
        limitType[(size_t) s * nrl + i] = RL_LIMIT_FROZEN;
        return;
    }
    int vrow = rlVrow[i];
    // the bus's own V variable (VOLTAGE reads) and the row its pin state lives on, which the coupler-tree
    // relocation can move to the parent coupler's DUMMY_Q (see dRlQrow)
    int qrow = rlQrow != nullptr ? rlQrow[i] : vrow;
    if (qrow != vrow && baseMode[qrow] == 0) {
        qrow = vrow;                                         // that coupler is open in the BASE: equations at home
    }
    size_t idx = (size_t) s * n + vrow;
    size_t qix = (size_t) s * n + qrow;
    size_t st = (size_t) s * nrl + i;
    // OLF's own reason for the base pin, when the host carried it: it distinguishes a
    // realistic-voltage pin (3/4, target = generation targetQ) from a Q-limit pin (1/2, target =
    // the limit). The nearest-limit guess below turns the former into the latter, and the
    // curve-limit drift then re-freezes it AT the limit (rte6515 VL-1251_1: -0.0035 -> -1.0000).
    char olfKind = rlBaseLt != nullptr ? rlBaseLt[i] : 0;
    if (rlRemote != nullptr && rlRemote[i]) {
        // REMOTE group member (plan-B3): base mode 1 at its row is the UNPINNED DISTR_Q share —
        // only the canonical mode 4 marks a base pin. The pilot's V equals the group target at the
        // base solution (the survivors keep controlling), so the v-vs-target heuristic is
        // DEGENERATE here — key the direction on WHICH limit the extracted BUS_TARGET_Q target
        // equals instead (target = limit − loadQ was written by the base activation).
        limitType[st] = 0;
        if (isEnrolledPilot(psEntry, nQr, i)) {
            // An ENROLLED pilot-self pilot (rotation) keeps its V row an identity while PINNED: the pin's
            // balance rides the carrier (or its own row flipped to a power row once the group lost control),
            // so the mode-4 marker below never appears on its row. OLF's own kind is the only record of its
            // base pin — and without it the seeding left the pilot UNPINNED (psSkip = 1, raw q dropped) on a
            // base that had it pinned: every scenario of the [5000,6000) busbar window failed its first
            // linear solve (976 x LINEAR_SOLVE_FAILED).
            limitType[st] = olfKind >= 1 && olfKind <= 4 ? olfKind : 0;
            return;
        }
        if (baseMode[qrow] == 4) {
            double tgt = target[qix];
            limitType[st] = olfKind != 0 ? olfKind
                    : (fabs(tgt - (rlMaxQb[i] - rlLoadQb[i])) <= fabs(tgt - (rlMinQb[i] - rlLoadQb[i])) ? 2 : 1);
        }
        return;
    }
    // Read the BASE row mode (length n, contingency-free) — NOT the per-scenario mode: a controller flipped to
    // PQ by a per-scenario gen-loss OVERRIDE (its generator is gone) has base mode 0 and must get limitType 0 so
    // the reactive-limits switch-back leaves it PQ. Only a controller that is PQ in the BASE (an actual Q-limit
    // switch during the base loadflow) gets a pinned limit type so a relieving contingency can switch it back.
    if (baseMode[qrow] == 1 && (rlMonitored[i] & 1) == 0) {       // PQ in the base case (a real Q-limit switch —
        // NOT a monitoring SVC standing by: that one seeds UNPINNED so only the VoltageMonitoring
        // orchestrator may activate it, exactly like the CPU's null qLimitType)
        double vt = rlTargetOf(nRlVtOv, rlVtOvSlotIdx, rlVtOvVal, svcRlVt, nrl, s, i, rlVtarget[i]);
        limitType[st] = olfKind != 0 ? olfKind
                : ((x[idx] < vt) ? 2 : 1);                  // v below target ⇒ maxQ pinned; above ⇒ minQ pinned
    } else {
        limitType[st] = 0;                                  // base PV (incl. a per-scenario gen-override flip)
    }
}

__global__ void seedRlModeFromState(int nrl, int nScenarios, int n, const int* rlVrow, const int* rlQrow,
                                    const double* rlMinQ, const double* rlMaxQ, const double* rlLoadQ,
                                    const char* limitType, int* mode, double* target, int cvStride,
                                    const char* rlRemote, double* savedTarget, char* savedMode,
                                    const char* rlBaseLt, const int* psEntry, int nQr,
                                    const double* rlVtarget, int nRlVtOv, const int* rlVtOvSlotIdx, const double* rlVtOvVal,
                                    const double* svcRlVt, const char* rlLocalOv, const double* rlLocalVt,
                                    const double* pinTargetIn, const char* rlIslanded) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nrl * nScenarios) return;
    int i = tid % nrl;
    int s = tid / nrl;
    // A controller this scenario islanded: its bus is out of the system and its rows are FROZEN identities. Stamping
    // the continued pin turned its V row into a reactive balance at the frozen target, an island equation OLF does not
    // have, which the first solve then iterated on (rte6515 LINE-3477-2893: 0.0075 MW on every strategy that moves P).
    if (rlIslanded != nullptr && rlIslanded[(size_t) s * nrl + i]) return;
    char lt = limitType[(size_t) s * nrl + i];
    // A LOCAL controller pinned in the BASE case that the previous solve released (its exported state says PV): the
    // base activation still has it pinned, and OLF's strategy starts with the bus back on voltage control, exactly as
    // the reactive-limit switch-back left it - the PV form (6 on a cross row) at its voltage target. Left alone it
    // stayed pinned at the base limit: a strategy pair of rte6515 LINE-3273-1812 solved two generators the parent had
    // released at their base Q limits, 0.016 kV at VL-2968 even for a strategy that changes nothing.
    if (lt == 0 && rlBaseLt != nullptr && rlBaseLt[i] >= 1 && rlBaseLt[i] <= 4 && (rlRemote == nullptr || !rlRemote[i])
            && !isEnrolledPilot(psEntry, nQr, i)) {
        size_t st = (size_t) s * nrl + i;
        bool localOv = rlLocalOv != nullptr && rlLocalOv[st];
        int vrow = rlVrow[i];
        int pinRow = vrow;
        if (localOv && rlQrow != nullptr && rlQrow[i] >= 0 && rlQrow[i] != vrow) {
            size_t r2 = (size_t) s * n + rlQrow[i];
            if (mode[r2] == 6 || (mode[r2] == 4 && mode[(size_t) s * n + vrow] != 4)) {
                pinRow = rlQrow[i];
            }
        }
        size_t pidx = (size_t) s * n + pinRow;
        if (mode[pidx] == 1 || mode[pidx] == 4) {            // still holding the base pin
            mode[pidx] = pinRow != vrow ? 6 : 0;
            target[pidx] = localOv ? rlLocalVt[st] : rlTargetOf(nRlVtOv, rlVtOvSlotIdx, rlVtOvVal, svcRlVt, nrl, s, i, rlVtarget[i]);
        }
        return;
    }
    // ...and the same for a REMOTE member, as the reactive-limits pass releases one: its row goes back to its unpinned
    // form - the DISTR_Q share (mode 1), whose target the DISTR_Q pass recomputes, or a self-controlling pilot's voltage
    // identity (mode 0). Left pinned at the base limit, its group shared the pilot's reactive balance one member short
    // (network.xiidm LLJP, the member on bus 5779: mode 4 at -0.2954 where the parent had released it, 5.46 kV).
    if (lt == 0 && rlBaseLt != nullptr && rlBaseLt[i] >= 1 && rlBaseLt[i] <= 4 && rlRemote != nullptr && rlRemote[i]
            && !isEnrolledPilot(psEntry, nQr, i)) {
        int vrow = rlVrow[i];
        int qrow = rlQrow != nullptr && rlQrow[i] >= 0 && rlQrow[i] != vrow && mode[(size_t) s * n + rlQrow[i]] != 0
                ? rlQrow[i] : vrow;
        size_t qidx = (size_t) s * n + qrow;
        if (mode[qidx] == 4) {                               // still holding the base pin
            mode[qidx] = rlRemote[i] == 2 ? 0 : 1;
            target[qidx] = rlRemote[i] == 2
                    ? rlTargetOf(nRlVtOv, rlVtOvSlotIdx, rlVtOvVal, svcRlVt, nrl, s, i, rlVtarget[i]) : 0.0;
        }
        return;
    }
    if (isEnrolledPilot(psEntry, nQr, i)) return;           // its pin never lives on its own V row (rotation)
    if (lt != 0) {
        // the pin's row (see dRlQrow), with the relocation's per-scenario diversion: mode 0 at the relocated row
        // means its coupler is open this scenario and the equations are back on the bus's own rows
        int seedRow = rlQrow != nullptr ? rlQrow[i] : rlVrow[i];
        if (seedRow != rlVrow[i] && mode[(size_t) s * n + seedRow] == 0) {
            seedRow = rlVrow[i];
        }
        size_t idx = (size_t) s * n + seedRow;
        size_t li = cvStride ? (size_t) s * nrl + i : (size_t) i;
        // What the row looked like BEFORE this seed is what a later switch-back restores. Leaving it
        // unwritten let the restore read whatever the previous solve left in this slot: the buffers are
        // per (entry, SLOT) and a context is reused chunk after chunk, so an entry pinned at seed time
        // and unpinned during the contingency took another scenario's target. Invisible wherever the
        // row's target is recomputed right after (a DISTR_Q share), a small wrong answer where it is
        // not — measured on network.xiidm as an intermittent 8.178e-05 kV shared by every scenario the
        // stale slot reached (the KUMF / KUMD pair swapping reactive power).
        savedTarget[(size_t) s * nrl + i] = target[idx];
        savedMode[(size_t) s * nrl + i] = static_cast<char>(mode[idx]);
        // a pinned REMOTE member's row is mode 4 (own Q balance), a pinned local one mode 1
        mode[idx] = (rlRemote != nullptr && rlRemote[i]) ? 4 : 1;
        // the pinned target the previous solve ENDED with, when it exported one: OLF restores the bus's generation
        // target Q from the parent's end, which is not the current limit when that limit moved by less than the
        // re-freeze tolerance (rte6515 LINE-3273-1812, VL-6095: 0.7518298 frozen, the curve at 0.7518918)
        double pin = pinTargetIn != nullptr ? pinTargetIn[(size_t) s * nrl + i] : NAN;
        target[idx] = !isnan(pin) ? pin : (lt == 1 ? rlMinQ[li] : rlMaxQ[li]) - rlLoadQ[li];
    }
}

// The pinned target of every pinned entry at the end of a solve, read on the row the seed above writes it to (NaN for
// an unpinned entry): what a continued solve (a strategy pair) restores, as OLF restores the parent's target Q.
__global__ void exportRlPinTargets(int nrl, int nScenarios, int n, const int* rlVrow, const int* rlQrow,
                                   const char* limitType, const int* mode, const double* target, const char* rlIslanded,
                                   int nQr, const int* psEntry, const int* qrPtr, const int* qrRows, const char* veqOff,
                                   const int* qrDepBase, double* out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nrl * nScenarios) return;
    int i = tid % nrl;
    int s = tid / nrl;
    char lt = limitType[(size_t) s * nrl + i];
    // an enrolled pilot's pin is its group's flip: the flipped row's target when the group is flipped (what a
    // continued solve's seed flip restores), nothing otherwise (its balance rides a carrier, never its own rows)
    for (int g = 0; psEntry != nullptr && g < nQr; g++) {
        if (psEntry[g] != i) continue;
        bool flipped = lt != 0 && veqOff != nullptr && veqOff[(size_t) s * nQr + g] && (qrDepBase == nullptr || qrDepBase[g] < 0);
        out[tid] = flipped ? target[(size_t) s * n + qrRows[qrPtr[g]]] : NAN;
        return;
    }
    // an islanded controller's row holds its bus's frozen value, not a Q: a pair that brings the bus back seeds the
    // pin from its limit, the target the bus kept while it was out
    if (lt == 0 || (rlIslanded != nullptr && rlIslanded[(size_t) s * nrl + i])) {
        out[tid] = NAN;
        return;
    }
    int seedRow = rlQrow != nullptr ? rlQrow[i] : rlVrow[i];
    if (seedRow != rlVrow[i] && mode[(size_t) s * n + seedRow] == 0) {
        seedRow = rlVrow[i];
    }
    out[tid] = target[(size_t) s * n + seedRow];
}

}  // namespace olfgpu
