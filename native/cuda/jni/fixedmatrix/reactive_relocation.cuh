// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// docs/gpu-vs-olf.md section 2.3 (equations placed on another bus's row), the pilot Q carrier: the candidate
// tables (ReactiveRelocationTables) and the per-scenario carrier choice (chooseCarrierRow, dependentPilotDestination).
#ifndef OLF_GPU_FIXEDMATRIX_REACTIVE_RELOCATION_CUH
#define OLF_GPU_FIXEDMATRIX_REACTIVE_RELOCATION_CUH

#include <cuda_runtime.h>
#include "../common/codes/gpu_row_mode.cuh"
#include "cross_identity.cuh"

namespace olfgpu {

// GROUP pilot-Q relocation tables (plan-A6), bundled to keep the fill signatures stable. All
// nullptr when the structure has no relocation groups.
struct ReactiveRelocationTables {
    const int* relocationGroup = nullptr;                              // per output: group id, -1 = not grouped
    const int* relocationCandidateOffset = nullptr;                                // per output: offset into pool
    const int* relocationCandidateDestinations = nullptr;                               // one slot (or -2-row) per (output, candidate)
    const int* reactiveRelocationGroupStart = nullptr;                              // group -> candidate range
    const int* reactiveRelocationCandidateRows = nullptr;                             // candidate DEVICE rows ([0] = fallback)
    // PILOT-SELF gate: per (scenario, group), 1 = this group's pilot is one of its OWN controllers and is
    // NOT pinned this scenario, so it has no Q balance to relocate (its q feeds the DISTR_Q rows weighted)
    // and its raw q must NOT be emitted at the carrier row. Null = no such group, every pick as before.
    const char* pilotSelfCarrierSkip = nullptr;
    int pilotSelfCarrierSkipStride = 0;                                        // stride of pilotSelfCarrierSkip (= nReactiveRelocationGroups)
    // Per row: the relocated child's OWN row for a coupler dummy row, -1 elsewhere. A stock pair needs it to
    // tell "the coupler is dead AND this bus is PINNED at home" (its own q belongs on its own row) from
    // "the coupler is dead and it still SHARES" (its DISTR_Q, diverted home, carries its q).
    const int* relocatedChildOwnRow = nullptr;
    // Per (scenario, dq member): the member's SUBGROUP this scenario (0 = the base pilot's group). A contingency
    // that splits a merged group behind its surviving main leaves two OLF groups on ONE candidate table (tasks
    // #71): keys and q contributions pair only within a subgroup, and the base pilot's carrier is picked among
    // subgroup-0 members only. Null = no split anywhere in the batch.
    const char* distrQSubgroup = nullptr;
    const int* reactiveRelocationDistrQMemberStart = nullptr;                      // qr group -> its first dq member (candidate 1)
    int nDistrQMembers = 0;
    const char* relocationDependentPilot = nullptr;                               // per output: dependent-pilot routing (see dependentPilotDestination)
};

// GROUP pilot-Q relocation (plan-A6): candidates = [fallback row, member rows in merged order];
// pick the FIRST member row whose per-scenario mode is neither identity (0: outaged/clamped/off)
// nor pinned (4) — exactly OLF's "first enabled" free-role rule — else the fallback (control off).
// pilotSelfCarrierSkip is per (scenario, group); the picks read it for the current scenario, exactly as they read `mode`
// already offset. Offset a by-value copy rather than indexing with s inside every pick.
__device__ inline ReactiveRelocationTables relocationTablesForScenario(ReactiveRelocationTables rt, int s) {
    if (rt.pilotSelfCarrierSkip != nullptr) {
        rt.pilotSelfCarrierSkip += (size_t) s * rt.pilotSelfCarrierSkipStride;
    }
    if (rt.distrQSubgroup != nullptr) {
        rt.distrQSubgroup += (size_t) s * rt.nDistrQMembers;
    }
    return rt;
}

__device__ inline int chooseCarrierRow(int g, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows, const int* mode) {
    int p0 = reactiveRelocationGroupStart[g];
    int count = reactiveRelocationGroupStart[g + 1] - p0;
    for (int c = 1; c < count; c++) {
        int m = mode[reactiveRelocationCandidateRows[p0 + c]];
        if (m != GpuRowMode::FIXED_VARIABLE && m != GpuRowMode::BUS_TARGET_Q && m != GpuRowMode::CROSS_FIXED_VARIABLE) {
            return c;
        }
    }
    return 0;
}

// chooseCarrierRow restricted to the base pilot's SUBGROUP (tasks #71): candidate c is member reactiveRelocationDistrQMemberStart[g] + c - 1,
// and a member the scenario moved to another subgroup can host neither the pilot's balance nor be its carrier.
// sub is already offset to the scenario (relocationTablesForScenario / the caller).
__device__ inline int chooseCarrierRowInSubgroup(int g, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows, const int* mode,
                                     const char* sub, const int* reactiveRelocationDistrQMemberStart) {
    if (sub == nullptr || reactiveRelocationDistrQMemberStart == nullptr || reactiveRelocationDistrQMemberStart[g] < 0) {
        return chooseCarrierRow(g, reactiveRelocationGroupStart, reactiveRelocationCandidateRows, mode);
    }
    int p0 = reactiveRelocationGroupStart[g];
    int count = reactiveRelocationGroupStart[g + 1] - p0;
    for (int c = 1; c < count; c++) {
        if (sub[reactiveRelocationDistrQMemberStart[g] + c - 1] != 0) continue;
        int m = mode[reactiveRelocationCandidateRows[p0 + c]];
        if (m != GpuRowMode::FIXED_VARIABLE && m != GpuRowMode::BUS_TARGET_Q && m != GpuRowMode::CROSS_FIXED_VARIABLE) {
            return c;
        }
    }
    return 0;
}

// The pilot of an ENROLLED pilot-self group (rotation property): its Q is re-routed through the candidate
// table, so a pin must NOT turn its own row into a Q balance — the row keeps BUS_TARGET_V while the pinned
// balance rides the carrier (updateDistrQTargetsBatch puts the pin limit there). Measured before this:
// the pilot's row went mode 4 with target -0.444 and NO q content (its q had gone to the carrier), F = -target.
// pilotEntryOfPilotSelfGroup marker: a PILOT-SELF group whose pilot has no reactive-limit entry (see pilotSelfGateBatch).
// Distinct from -1 ("not a pilot-self group"), and negative so every `pe >= 0` test still reads it as "no entry".
static constexpr int PILOT_SELF_NO_REACTIVE_LIMIT_ENTRY = -2;

__device__ inline bool isEnrolledPilot(const int* pilotSelfReactiveLimitEntry, int nReactiveRelocationGroups, int i) {
    if (pilotSelfReactiveLimitEntry == nullptr) return false;
    for (int g = 0; g < nReactiveRelocationGroups; g++) {
        if (pilotSelfReactiveLimitEntry[g] == i) return true;
    }
    return false;
}

// DEPENDENT-PILOT routing (ExtractionColumns: a merged remote group's dependent controlled bus has a candidate
// group of its own, the same member rows as its base group's). The bus keeps its two-way reactive row - its
// parent coupler's dummy row while that coupler lives, its own row once the coupler is dead - UNLESS its own
// row has become the voltage identity (OLF re-elected it the main of the survivors): then its balance rides the
// first free member row of its group, exactly as the base pilot's does, and so does a cross identity on the
// dummy row (mode 6, the main under a live coupler). A root (no relocation) reads its own row the same way.
__device__ inline int dependentPilotDestination(int slot, int frozen, int alternateDestination, int selectorRow, const int* mode, ReactiveRelocationTables rt) {
    int fr = selectorRow <= -2 ? -2 - selectorRow : selectorRow;
    int mf = mode[fr];
    if (selectorRow <= -2) {                                      // a relocated child: fr is the parent's dummy row
        if (mf == 1 || mf == 4) return frozen;               // the coupler lives (or the pin sits there)
        if (mf == 0) {                                       // the coupler is dead: home, unless home is the identity
            int mo = mode[rt.relocatedChildOwnRow[fr]];
            if (!holdsFixedVariable(mo)) return alternateDestination;
        }
    } else if (mf != 0 && mf != 6) {                         // a root: its own row, unless that is the identity
        return frozen;
    }
    return rt.relocationCandidateDestinations[rt.relocationCandidateOffset[slot] + chooseCarrierRowInSubgroup(rt.relocationGroup[slot], rt.reactiveRelocationGroupStart, rt.reactiveRelocationCandidateRows, mode, rt.distrQSubgroup, rt.reactiveRelocationDistrQMemberStart)];
}

}  // namespace olfgpu

#endif  // OLF_GPU_FIXEDMATRIX_REACTIVE_RELOCATION_CUH
