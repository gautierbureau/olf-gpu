// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The device helpers every fill kernel shares: scattering an output into the Jacobian values / the
// mismatch through a role-folded destination, the dynamic Q-relocation choice of that destination
// (RelocTables, relocChoose, depPick, ziDynDest), and the CSR slot lookup.
#ifndef OLF_GPU_EQUATIONS_SCATTER_CUH
#define OLF_GPU_EQUATIONS_SCATTER_CUH

#include <cuda_runtime.h>

/**
 * Pool order of a stock pair: [0] = the bus's OWN row, [1] = the parent coupler's DUMMY_Q row. Which one takes
 * the bus's raw q depends on TWO modes, not one:
 *   dummy row mode 4                      -> the dummy row: a reactive limit pins it there, coupler alive;
 *   dummy row mode 0 and own row mode 4   -> the OWN row: the coupler is dead, so the equations are back home
 *                                            and the pin is there (measured: rte6515 BUS-11_BBS_1 put mode 4 on
 *                                            BUS_V(VL-11_1), which holds zi#1's ZERO_V, and writing nothing left
 *                                            that row with a zero diagonal — 20 of the 6667 scenarios failed
 *                                            their linear solve exactly this way, all of them _BBS_1 outages);
 *   anything else                         -> nothing: while it SHARES, its DISTR_Q row (diverted home when the
 *                                            coupler dies) carries its q, and writing the raw q too double-counts.
 */
__host__ __device__ inline int stockPairPick(int m, const int* pool, int p, int frozen,
                                             const int* mode, const int* relocOwn, int dummyRow) {
    if (m == 4) {
        return pool[p + 1];
    }
    if (m == 0 && relocOwn != nullptr) {
        int own = relocOwn[dummyRow];
        if (own >= 0 && mode[own] == 4) {
            return pool[p];
        }
    }
    return frozen;
}

__device__ void scatterAdd(int nout, const int* dest, const double* o, double* vals, double* F) {
    for (int k = 0; k < nout; k++) {
        int d = dest[k];
        if (d >= 0) {
            atomicAdd(&vals[d], o[k]);
        } else if (d <= -2) {
            atomicAdd(&F[-2 - d], o[k]);
        }
    }
}

// Dynamic Q-relocation scatter: when an output's frozen Q row is an identity this scenario
// (mode[selRow] == 0), route it to the alternate candidate row instead. mode is this scenario's
// row-mode slice (length n). dest/altDest/selRow share the per-output layout of scatterAdd.
// Mode 4 ("pinned remote gen-VC member" — the member row hosts its OWN Q balance, plan-B2) also
// selects the alternate: those entries carry a SUPPRESSED frozen dest and their own row as the
// alternate, with selRow = the member row itself. TVC/shunt select rows never carry mode 4.
// GROUP pilot-Q relocation tables (plan-A6), bundled to keep the fill signatures stable. All
// nullptr when the structure has no relocation groups.
struct RelocTables {
    const int* group = nullptr;                              // per output: group id, -1 = not grouped
    const int* off = nullptr;                                // per output: offset into pool
    const int* pool = nullptr;                               // one slot (or -2-row) per (output, candidate)
    const int* qrPtr = nullptr;                              // group -> candidate range
    const int* qrRows = nullptr;                             // candidate DEVICE rows ([0] = fallback)
    // PILOT-SELF gate: per (scenario, group), 1 = this group's pilot is one of its OWN controllers and is
    // NOT pinned this scenario, so it has no Q balance to relocate (its q feeds the DISTR_Q rows weighted)
    // and its raw q must NOT be emitted at the carrier row. Null = no such group, every pick as before.
    const char* psSkip = nullptr;
    int psGroups = 0;                                        // stride of psSkip (= nQr)
    // Per row: the relocated child's OWN row for a coupler dummy row, -1 elsewhere. A stock pair needs it to
    // tell "the coupler is dead AND this bus is PINNED at home" (its own q belongs on its own row) from
    // "the coupler is dead and it still SHARES" (its DISTR_Q, diverted home, carries its q).
    const int* relocOwn = nullptr;
    // Per (scenario, dq member): the member's SUBGROUP this scenario (0 = the base pilot's group). A contingency
    // that splits a merged group behind its surviving main leaves two OLF groups on ONE candidate table (tasks
    // #71): keys and q contributions pair only within a subgroup, and the base pilot's carrier is picked among
    // subgroup-0 members only. Null = no split anywhere in the batch.
    const char* dqSub = nullptr;
    const int* qrMemberStart = nullptr;                      // qr group -> its first dq member (candidate 1)
    int nDqMembers = 0;
    const char* dep = nullptr;                               // per output: dependent-pilot routing (see depPick)
};

// GROUP pilot-Q relocation (plan-A6): candidates = [fallback row, member rows in merged order];
// pick the FIRST member row whose per-scenario mode is neither identity (0: outaged/clamped/off)
// nor pinned (4) — exactly OLF's "first enabled" free-role rule — else the fallback (control off).
// psSkip is per (scenario, group); the picks read it for the current scenario, exactly as they read `mode`
// already offset. Offset a by-value copy rather than indexing with s inside every pick.
__device__ inline RelocTables relocForScenario(RelocTables rt, int s) {
    if (rt.psSkip != nullptr) {
        rt.psSkip += (size_t) s * rt.psGroups;
    }
    if (rt.dqSub != nullptr) {
        rt.dqSub += (size_t) s * rt.nDqMembers;
    }
    return rt;
}

__device__ inline int relocChoose(int g, const int* qrPtr, const int* qrRows, const int* mode) {
    int p0 = qrPtr[g];
    int cnt = qrPtr[g + 1] - p0;
    for (int c = 1; c < cnt; c++) {
        int m = mode[qrRows[p0 + c]];
        if (m != 0 && m != 4 && m != 6) {
            return c;
        }
    }
    return 0;
}

// relocChoose restricted to the base pilot's SUBGROUP (tasks #71): candidate c is member qrMemberStart[g] + c - 1,
// and a member the scenario moved to another subgroup can host neither the pilot's balance nor be its carrier.
// sub is already offset to the scenario (relocForScenario / the caller).
__device__ inline int relocChooseSub(int g, const int* qrPtr, const int* qrRows, const int* mode,
                                     const char* sub, const int* qrMemberStart) {
    if (sub == nullptr || qrMemberStart == nullptr || qrMemberStart[g] < 0) {
        return relocChoose(g, qrPtr, qrRows, mode);
    }
    int p0 = qrPtr[g];
    int cnt = qrPtr[g + 1] - p0;
    for (int c = 1; c < cnt; c++) {
        if (sub[qrMemberStart[g] + c - 1] != 0) continue;
        int m = mode[qrRows[p0 + c]];
        if (m != 0 && m != 4 && m != 6) {
            return c;
        }
    }
    return 0;
}

// The pilot of an ENROLLED pilot-self group (rotation property): its Q is re-routed through the candidate
// table, so a pin must NOT turn its own row into a Q balance — the row keeps BUS_TARGET_V while the pinned
// balance rides the carrier (updateDistrQTargetsBatch puts the pin limit there). Measured before this:
// the pilot's row went mode 4 with target -0.444 and NO q content (its q had gone to the carrier), F = -target.
// psPilotEntry marker: a PILOT-SELF group whose pilot has no reactive-limit entry (see pilotSelfGateBatch).
// Distinct from -1 ("not a pilot-self group"), and negative so every `pe >= 0` test still reads it as "no entry".
static constexpr int PS_NO_ENTRY = -2;

__device__ inline bool isEnrolledPilot(const int* psEntry, int nQr, int i) {
    if (psEntry == nullptr) return false;
    for (int g = 0; g < nQr; g++) {
        if (psEntry[g] == i) return true;
    }
    return false;
}

// DEPENDENT-PILOT routing (ExtractionColumns: a merged remote group's dependent controlled bus has a candidate
// group of its own, the same member rows as its base group's). The bus keeps its two-way reactive row - its
// parent coupler's dummy row while that coupler lives, its own row once the coupler is dead - UNLESS its own
// row has become the voltage identity (OLF re-elected it the main of the survivors): then its balance rides the
// first free member row of its group, exactly as the base pilot's does, and so does a cross identity on the
// dummy row (mode 6, the main under a live coupler). A root (no relocation) reads its own row the same way.
__device__ inline int depPick(int slot, int frozen, int altDest, int selRow, const int* mode, RelocTables rt) {
    int fr = selRow <= -2 ? -2 - selRow : selRow;
    int mf = mode[fr];
    if (selRow <= -2) {                                      // a relocated child: fr is the parent's dummy row
        if (mf == 1 || mf == 4) return frozen;               // the coupler lives (or the pin sits there)
        if (mf == 0) {                                       // the coupler is dead: home, unless home is the identity
            int mo = mode[rt.relocOwn[fr]];
            if (mo != 0 && mo != 6) return altDest;
        }
    } else if (mf != 0 && mf != 6) {                         // a root: its own row, unless that is the identity
        return frozen;
    }
    return rt.pool[rt.off[slot] + relocChooseSub(rt.group[slot], rt.qrPtr, rt.qrRows, mode, rt.dqSub, rt.qrMemberStart)];
}

// zi q-slot dynamic pick (F/J slots {0..3} per element): group -> mode-scan candidate slot,
// two-way -> alt when the select row is identity/pinned this scenario, else the frozen dest.
__device__ inline int ziDynDest(int frozen, int slot, const int* alt, const int* sel,
                                const int* mode, RelocTables rt) {
    int rg = rt.group != nullptr ? rt.group[slot] : -1;
    if (rg >= 0) {
        if (rt.dep != nullptr && rt.dep[slot]) {
            return depPick(slot, frozen, alt[slot], sel[slot], mode, rt);
        }
        if (rt.off[slot] < 0) {                              // STOCK PAIR: group slot holds the dummy ROW
            return stockPairPick(mode[rg], rt.pool, -rt.off[slot] - 1, frozen, mode, rt.relocOwn, rg);
        }
        if (rt.psSkip != nullptr && rt.psSkip[rg]) {
            return -1;                                       // pilot-self group, pilot unpinned: no raw q
        }
        return rt.pool[rt.off[slot] + relocChooseSub(rg, rt.qrPtr, rt.qrRows, mode, rt.dqSub, rt.qrMemberStart)];
    }
    int sr = sel != nullptr ? sel[slot] : -1;
    if (sr <= -2) {
        return mode[-2 - sr] == 0 ? alt[slot] : frozen;       // stock-placement: identity only (see stockAltRow)
    }
    return (sr >= 0 && (mode[sr] == 0 || mode[sr] == 4)) ? alt[slot] : frozen;
}

__device__ void scatterAddDyn(int nout, const int* dest, const int* altDest, const int* selRow,
                              const int* mode, const double* o, double* vals, double* F,
                              RelocTables rt, int eOff) {
    for (int k = 0; k < nout; k++) {
        int d;
        int rg = rt.group != nullptr ? rt.group[eOff + k] : -1;
        if (rg >= 0 && rt.dep != nullptr && rt.dep[eOff + k]) {
            d = depPick(eOff + k, dest[k], altDest[k], selRow[k], mode, rt);
        } else if (rg >= 0 && rt.off[eOff + k] < 0) {        // STOCK PAIR: group slot holds the dummy ROW
            d = stockPairPick(mode[rg], rt.pool, -rt.off[eOff + k] - 1, dest[k], mode, rt.relocOwn, rg);
        } else if (rg >= 0 && rt.psSkip != nullptr && rt.psSkip[rg]) {
            continue;                                        // pilot-self group, pilot unpinned: no raw q
        } else if (rg >= 0) {
            d = rt.pool[rt.off[eOff + k] + relocChooseSub(rg, rt.qrPtr, rt.qrRows, mode, rt.dqSub, rt.qrMemberStart)];
        } else {
            int sr = selRow[k];
            bool useAlt = sr >= 0 ? (mode[sr] == 0 || mode[sr] == 4)
                                  : (sr <= -2 && mode[-2 - sr] == 0);   // stock-placement: identity only
            d = useAlt ? altDest[k] : dest[k];
        }
        if (d >= 0) {
            atomicAdd(&vals[d], o[k]);
        } else if (d <= -2) {
            atomicAdd(&F[-2 - d], o[k]);
        }
    }
}

// Scatter one (dest, value): dest >= 0 -> J slot, dest <= -2 -> F row, -1 -> skip.
__device__ inline void addAt(int dest, double val, double* vals, double* F) {
    if (dest >= 0) {
        atomicAdd(&vals[dest], val);
    } else if (dest <= -2) {
        atomicAdd(&F[-2 - dest], val);
    }
}

__device__ inline int csrFind(const int* rowPtr, const int* colIdx, int row, int col) {
    int lo = rowPtr[row];
    int hi = rowPtr[row + 1];
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (colIdx[mid] < col) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return (lo < rowPtr[row + 1] && colIdx[lo] == col) ? lo : -1;
}

__device__ void scatterAddF(int nres, const int* dest, const int* pos, const double* o, double* F) {
    for (int k = 0; k < nres; k++) {
        int d = dest[pos[k]];
        if (d <= -2) {
            atomicAdd(&F[-2 - d], o[k]);
        }
    }
}

// Dynamic Q-relocation, residual-only variant: route each residual to its alternate F row when the
// frozen row is an identity this scenario. dest/altDest/selRow are the element's per-output slices;
// pos maps each of the nres residuals to its CB_NOUT output index.
__device__ void scatterAddFDyn(int nres, const int* dest, const int* altDest, const int* selRow,
                               const int* pos, const int* mode, const double* o, double* F,
                               RelocTables rt, int eOff) {
    for (int k = 0; k < nres; k++) {
        int idx = pos[k];
        int d;
        int rg = rt.group != nullptr ? rt.group[eOff + idx] : -1;
        if (rg >= 0 && rt.dep != nullptr && rt.dep[eOff + idx]) {
            d = depPick(eOff + idx, dest[idx], altDest[idx], selRow[idx], mode, rt);
        } else if (rg >= 0 && rt.off[eOff + idx] < 0) {      // STOCK PAIR: group slot holds the dummy ROW
            d = stockPairPick(mode[rg], rt.pool, -rt.off[eOff + idx] - 1, dest[idx], mode, rt.relocOwn, rg);
        } else if (rg >= 0 && rt.psSkip != nullptr && rt.psSkip[rg]) {
            continue;                                        // pilot-self group, pilot unpinned: no raw q
        } else if (rg >= 0) {
            d = rt.pool[rt.off[eOff + idx] + relocChooseSub(rg, rt.qrPtr, rt.qrRows, mode, rt.dqSub, rt.qrMemberStart)];
        } else {
            int sr = selRow[idx];
            bool useAlt = sr >= 0 ? (mode[sr] == 0 || mode[sr] == 4)
                                  : (sr <= -2 && mode[-2 - sr] == 0);   // stock-placement: identity only
            d = useAlt ? altDest[idx] : dest[idx];
        }
        if (d <= -2) {
            atomicAdd(&F[-2 - d], o[k]);
        }
    }
}

#endif  // OLF_GPU_EQUATIONS_SCATTER_CUH
