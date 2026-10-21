// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Resolve each generated scatter-table row to (equation row, variable column) from one element's packed index
// array, pair every equation row with its dominant-diagonal column, and collect the fixed CSR pattern.
#include <cudss.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../equations/generated/closed_branch_scatter.h"
#include "../equations/generated/open_branch_scatter.h"
#include "../equations/generated/shunt_scatter.h"
#include "../equations/generated/hvdc_ac_emulation_scatter.h"

#include "../common/check.cuh"
#include "../common/stats.cuh"
#include "../common/codes/gpu_reactive_term_kind.cuh"
#include "../common/codes/gpu_two_sides.cuh"
#include "../solver/linear_solver_settings.cuh"
#include "structure_build.cuh"

using namespace powsybl::openloadflow;

namespace olfgpu {

RC resolveClosedBranch(const ClosedBranchScatter& s, const int* ix) {            // P1 Q1 P2 Q2 ph1 v1 ph2 v2 a1col r1col
    RC rc;
    if (s.role == 2) return rc;
    rc.row = ix[s.rowEnd * 2 + s.rowEq];
    // col = 4 + endpoint*2 + var: endpoint 0/1 = bus1/bus2 (PHI/V cols), endpoint 2 = the branch's
    // own control cols (ALPHA1 at ix[8], RHO1 at ix[9]). A control col is -1 when not derived, so
    // the entry resolves to absent and buildDest skips it for that element.
    if (s.role == 1) rc.col = ix[4 + s.colEnd * 2 + s.colVar];
    return rc;
}

RC resolveOpenBranch(const OpenBranchScatter& s, const int* ix) {            // P1 Q1 P2 Q2 v1 v2
    RC rc;
    if (s.role == 2) return rc;
    rc.row = ix[s.rowEnd * 2 + s.rowEq];
    if (s.role == 1) rc.col = ix[4 + s.colEnd];              // BUS_V columns only
    return rc;
}

RC resolveShunt(const ShuntScatter& s, const int* ix) {            // P Q v bcol
    RC rc;
    if (s.role == 2) return rc;
    rc.row = ix[s.rowEq];                                    // single bus: rowEnd == 0
    // col endpoint 0 = the bus (BUS_V at ix[2]); endpoint 2 = the shunt's own control col (SHUNT_B at
    // ix[3]) — -1 when the shunt derives no voltage control, so dq/db resolves to absent and is skipped.
    if (s.role == 1) rc.col = (s.colEnd == 2) ? ix[3] : ix[2];
    return rc;
}

RC resolveHvdcAcEmulation(const HvdcAcEmulationScatter& s, const int* ix) {          // P1 P2 ph1 ph2
    RC rc;
    if (s.role == 2) return rc;
    rc.row = ix[s.rowEnd];                                   // P rows only: rowEq == 0
    if (s.role == 1) rc.col = ix[2 + s.colEnd];
    return rc;
}

// Row permutation: pair each equation row with its dominant-diagonal variable column —
// P rows with their own bus's PHI column (dP/dθ self term), Q rows with the bus's V
// column (dQ/dV), slack/PV identities with their column — and USE THE PAIRED COLUMN as
// the row index. The CSR diagonal is then structurally strong regardless of OLF's
// equation numbering: the plain creator happens to interleave P/Q/V per bus (fine), but
// the VECTORIZED creator numbers array equations in per-type blocks, whose leading
// minors defeat cuDSS's static pivoting (KLU pivots dynamically and never noticed).
// Choosing the ordering ourselves — a bus-ordered J — fixes any
// numbering. The mismatch/targets are permuted the same way; columns are untouched, so
// the solution dx needs no back-permutation.
std::vector<int> buildRowMap(int n, int nClosedBranch, const int* closedBranchIndices, int nOpenBranch, const int* openBranchIndices,
                             int nShunt, const int* shuntIndices, int nHvdcAcEmulation, const int* hvdcAcEmulationIndices,
                             int nDistrQ, const int* distrQRow, int nZeroImpedanceBranch, const int* zeroImpedanceBranchIndices) {
    std::vector<int> rowMap(n, -1);
    auto pairRc = [&](int row, int col) {
        if (row < 0 || col < 0) return;
        if (rowMap[row] == -1) {
            rowMap[row] = col;
        } else if (rowMap[row] != col) {
            throw std::runtime_error("conflicting diagonal pairing for equation row " + std::to_string(row));
        }
    };
    // A bus whose reactive balance was ROUTED ELSEWHERE (Qrow != its own V col) leaves that V col
    // carrying the v−target identity (PV / transformer / generator-pilot) or another power row that
    // no Q self-pair claims — claim it here. For an ordinary bus Qrow == V col, so this is a no-op.
    // A bus's P row pairs with its PHI column - unless the coupler-tree relocation moved that balance onto its
    // parent coupler's DUMMY_P column, where the row IS its own diagonal (the balance carries -DUMMY_P). The
    // relocated PHI column is then claimed by that coupler's ZERO_PHI, in the zero-impedance loop below. Exactly
    // the treatment the Q row already gets, where Qrow != V col means "relocated, self-pairs".
    auto pairP = [&](int prow, int phcol) {
        pairRc(prow, prow == phcol ? phcol : prow);
    };
    auto claimMovedV = [&](int qrow, int vcol) {
        if (vcol >= 0 && qrow != vcol) {
            pairRc(vcol, vcol);
        }
    };
    for (int e = 0; e < nClosedBranch; e++) {                          // P1 Q1 P2 Q2 ph1 v1 ph2 v2 a1col r1col
        const int* ix = closedBranchIndices + e * 12;
        pairP(ix[0], ix[4]);                                 // P1 -> ph1 (or itself, relocated)
        pairP(ix[2], ix[6]);                                 // P2 -> ph2 (or itself, relocated)
        // Q rows pair with their own stable index — its V col on an ordinary bus (Qrow==v col), the
        // controlling transformer's RHO col or a controller's DISTR_Q row on a voltage-controlled bus
        // (the extractor set Qrow to it). So the strong diagonal is dq/dv, dq/dr1 or the DISTR_Q diag.
        pairRc(ix[1], ix[1]);
        pairRc(ix[3], ix[3]);
        claimMovedV(ix[1], ix[5]);                           // side 1 moved-V identity
        claimMovedV(ix[3], ix[7]);                           // side 2 moved-V identity
        if (ix[9] >= 0) {
            // claim the rho row: its diagonal is the rho col either way — dq/dr1 when the ratio is a
            // free control, or the unit derivative of the r1−target identity when it is held constant.
            pairRc(ix[9], ix[9]);
        }
        if (ix[8] >= 0) {
            // Claim the a1 row (phase control). Unlike rho — whose row IS a bus Q row relocated onto the
            // rho column, so the dq/dr1 scatter lands on its diagonal — NOTHING scatters into the a1 row:
            // dp1da1/dq1da1/dp2da1/dq2da1 all go to the BUS P/Q rows against the a1 COLUMN. The a1 row is
            // therefore an identity row only (a1−target, INCREMENTAL mode holding a1 at its tap value),
            // and its 1.0 comes from the per-solve row-mode mask. The (r,r) entry is already guaranteed
            // in the pattern below; all that is missing is this claim, without which rowMap has no
            // pairing for the row and buildStructure rejects the system.
            pairRc(ix[8], ix[8]);
        }
    }
    for (int e = 0; e < nOpenBranch; e++) {                          // P1 Q1 P2 Q2 v1 v2 (no ph cols:
        const int* ix = openBranchIndices + e * 6;                       // P rows pair via closed branches)
        pairRc(ix[1], ix[1]);                                // Q rows self-pair (v col, or rho col if controlled)
        pairRc(ix[3], ix[3]);
        claimMovedV(ix[1], ix[4]);
        claimMovedV(ix[3], ix[5]);
    }
    for (int e = 0; e < nShunt; e++) {                          // P Q v bcol
        pairRc(shuntIndices[e * 5 + 1], shuntIndices[e * 5 + 1]);          // Q self-pairs (v col, or SHUNT_B col if controlled)
        claimMovedV(shuntIndices[e * 5 + 1], shuntIndices[e * 5 + 2]);
        if (shuntIndices[e * 5 + 3] >= 0) {
            // claim the SHUNT_B row: its diagonal is the b col either way — dq/db when the susceptance is a
            // free control, or the unit derivative of the b−target identity when it is held constant.
            pairRc(shuntIndices[e * 5 + 3], shuntIndices[e * 5 + 3]);
        }
    }
    for (int e = 0; e < nHvdcAcEmulation; e++) {                          // P1 P2 ph1 ph2
        const int* ix = hvdcAcEmulationIndices + e * 4;
        pairP(ix[0], ix[2]);
        pairP(ix[1], ix[3]);
    }
    // DISTR_Q reactive-distribution rows sit on a controller's BUS_V row and self-pair: their strong
    // diagonal is that controller's own weighted dq/dV (qPct_k·∂q_Gk/∂V_Gk − ∂q_Gk/∂V_Gk). Each
    // distinct distrQRow appears for several contributions; pairRc is idempotent so re-pairing is fine.
    for (int c = 0; c < nDistrQ; c++) {
        pairRc(distrQRow[c], distrQRow[c]);
    }
    // Zero-impedance branch: claim its DUMMY_P / DUMMY_Q rows (ZERO_V/ZERO_PHI or DUMMY_TARGET identities)
    // and, defensively, its bus P/Q rows (a coupler-only bus has no closed/open branch to claim them).
    for (int e = 0; e < nZeroImpedanceBranch; e++) {
        const int* ix = zeroImpedanceBranchIndices + e * 11;                      // b1P b1Q b2P b2Q dpRow distrQRow v1 v2 ...
        for (int s = 0; s < 6; s++) {
            pairRc(ix[s], ix[s]);
        }
        // ...and the MOVED-V identities, exactly as the closed-branch, open-branch and shunt loops
        // already do. Slot 1/3 is the bus's Q ROW, which for an ordinary bus IS its V col — but a
        // remote gen-VC CONTROLLER has Qrow = -1 and a PILOT has Qrow = another bus's row, so its own
        // V col (slots 6/7) is a distinct row that nothing else claims. A zero-impedance branch
        // touching such a bus therefore left its V row unpaired and buildStructure rejected the whole
        // system. That is why busbar-section contingencies never reached the GPU: retaining switches
        // creates zero-impedance branches, and past ~20 retained sections one of them lands on a
        // remote-voltage-control bus (rte6515: row 2200 of 13048, appearing ONLY in zero-impedance[slot 6]).
        claimMovedV(ix[1], ix[6]);
        claimMovedV(ix[3], ix[7]);
        if (zeroImpedanceDeadEndSideOf(ix) != GpuTwoSides::NONE) {
            // The stub's P/Q balances already sit on the dummy rows, where slots 0..3 self-pair and
            // -DUMMY_P / -DUMMY_Q is the diagonal. ZERO_V landed on the stub's v row, which
            // claimMovedV has just claimed (slot 1/3 no longer equals slot 6/7). ZERO_PHI's row is
            // the one nothing else names.
            pairRc(zeroPhiRowOf(ix), zeroPhiRowOf(ix));
        }
    }
    std::vector<char> colUsed(n, 0);
    for (int r = 0; r < n; r++) {
        if (rowMap[r] < 0) {                                 // every index in [0,n) is an active equation
            // Say WHERE the row appears. "row 2200 has no diagonal pairing" is a row index and nothing
            // else; resolving it back to an equation on the Java side is unreliable because the active
            // index shifts between solves. The builder, though, knows exactly which family arrays
            // mention the row and in which SLOT, and that is the whole diagnosis: a row no array
            // mentions was never packed, while a row some array mentions at a slot no pairing loop
            // reads is a GAP in that loop (e.g. the open-branch loop pairs only its Q slots, on the
            // assumption that "P rows pair via closed branches").
            auto where = [r](const char* name, const int* a, int count, int stride) {
                std::string out;
                for (int e = 0; e < count && a != nullptr; e++) {
                    for (int sl = 0; sl < stride; sl++) {
                        if (a[(size_t) e * stride + sl] == r) {
                            out += std::string(out.empty() ? "" : ",") + name + "[slot " + std::to_string(sl) + "]";
                            break;
                        }
                    }
                }
                return out;
            };
            std::string seen = where("cb", closedBranchIndices, nClosedBranch, 12) + where("ob", openBranchIndices, nOpenBranch, 6)
                    + where("sh", shuntIndices, nShunt, 5) + where("hv", hvdcAcEmulationIndices, nHvdcAcEmulation, 4)
                    + where("zi", zeroImpedanceBranchIndices, nZeroImpedanceBranch, 11);
            int distrQHits = 0;
            for (int c = 0; c < nDistrQ && distrQRow != nullptr; c++) {
                if (distrQRow[c] == r) {
                    distrQHits++;
                }
            }
            throw std::runtime_error("equation row " + std::to_string(r) + " of " + std::to_string(n)
                                     + " has no diagonal pairing (appears in: "
                                     + (seen.empty() ? std::string("NOTHING") : seen)
                                     + ", distrQRow hits " + std::to_string(distrQHits) + ")");
        }
        if (colUsed[rowMap[r]]) {
            throw std::runtime_error("diagonal pairing is not a bijection (column "
                                     + std::to_string(rowMap[r]) + " claimed twice)");
        }
        colUsed[rowMap[r]] = 1;
    }
    return rowMap;
}

// The DISTR_Q / DISTR_RHO / zero-impedance family contributions to the CSR pattern (the rows they add
// variable columns to). Shared by buildStructure for both solver contexts, so every consumer sees the
// SAME nnz the real context builds. closedBranchIndices/openBranchIndices/shuntIndices are the row-mapped packs; only the family ROWS are
// taken through rowMap here, the contributor columns (v/ph/dummy/r1) are variable columns left as-is.
void collectFamilyPattern(const std::vector<int>& rowMap,
                          const int* closedBranchIndices, const int* openBranchIndices, const int* shuntIndices,
                          int nDistrQ, const int* distrQElement, const int* distrQSide, const int* distrQKind, const int* distrQRow,
                          int nDistrRho, const int* distrRhoRow, const int* distrRhoColumn,
                          int nDistrShuntB, const int* distrShuntBRow, const int* distrShuntBColumn,
                          int nZeroImpedanceBranch, const int* zeroImpedanceBranchIndices, const int* relocatedChildOwnRow,
                          std::vector<std::pair<int, int>>& pat) {
    for (int c = 0; c < nDistrQ; c++) {
        // Twice for a relocated child's sharing row: where it lives while the coupler stands, and on the
        // bus's OWN row, which is where it goes in any scenario that kills that coupler (see buildDistrQDest).
        int mapped = rowMap[distrQRow[c]];
        int own = relocatedChildOwnRow != nullptr && mapped >= 0 ? relocatedChildOwnRow[mapped] : -1;
        for (int pass = 0; pass < 2; pass++) {
            int mappedRow = pass == 0 ? mapped : own;
            if (mappedRow < 0) {
                continue;
            }
            if (pass == 1) {
                pat.emplace_back(mappedRow, mappedRow);      // the stock row's own diagonal
            }
            if (distrQKind[c] == GpuReactiveTermKind::SHUNT_COMPENSATOR_REACTIVE_FLOW) {  // shunt q: bus V column
                pat.emplace_back(mappedRow, shuntIndices[static_cast<size_t>(distrQElement[c]) * 5 + 2]);
            } else if (distrQKind[c] == GpuReactiveTermKind::OPEN_BRANCH_REACTIVE_FLOW) {  // half-open: connected-side V column
                const int* openBranch = openBranchIndices + static_cast<size_t>(distrQElement[c]) * 6;
                pat.emplace_back(mappedRow, distrQSide[c] == GpuTwoSides::ONE ? openBranch[4] : openBranch[5]);
            } else if (distrQKind[c] == GpuReactiveTermKind::DUMMY_Q) {  // zero-imp DUMMY_Q column
                pat.emplace_back(mappedRow, distrQElement[c]);
            } else {                                         // closed branch: v1 v2 ph1 ph2 r1
                const int* q = closedBranchIndices + static_cast<size_t>(distrQElement[c]) * 12;
                pat.emplace_back(mappedRow, q[5]);
                pat.emplace_back(mappedRow, q[7]);
                pat.emplace_back(mappedRow, q[4]);
                pat.emplace_back(mappedRow, q[6]);
                if (q[9] >= 0) pat.emplace_back(mappedRow, q[9]);
            }
        }
    }
    for (int c = 0; c < nDistrRho; c++) {                          // DISTR_RHO: controller r1 column in a DISTR_RHO row
        pat.emplace_back(rowMap[distrRhoRow[c]], distrRhoColumn[c]);
    }
    for (int c = 0; c < nDistrShuntB; c++) {                          // DISTR_SHUNT_B: controller b column in a DISTR_SHUNT_B row
        pat.emplace_back(rowMap[distrShuntBRow[c]], distrShuntBColumn[c]);
    }
    for (int e = 0; e < nZeroImpedanceBranch; e++) {                          // zero-impedance: dummy couplings + (tree) ZERO_V/PHI
        const int* z = zeroImpedanceBranchIndices + static_cast<size_t>(e) * 11;
        int b1P = rowMap[z[0]], b2P = rowMap[z[2]];
        int b1Q = z[1] >= 0 ? rowMap[z[1]] : -1, b2Q = z[3] >= 0 ? rowMap[z[3]] : -1;
        pat.emplace_back(b1P, z[4]);
        pat.emplace_back(b2P, z[4]);
        if (b1Q >= 0) pat.emplace_back(b1Q, z[5]);
        if (b2Q >= 0) pat.emplace_back(b2Q, z[5]);
        // The ZERO_V / ZERO_PHI coupling is reserved for EVERY coupler, not only the ones that are
        // spanning-tree edges at extraction. OLF re-elects its tree during a security analysis and never
        // restores it, so a coupler out of the tree here can be IN it when OLF reaches a later
        // contingency; without the entries that scenario's row would carry no equation at all (measured:
        // the solve sat at 0% GPU and never finished a chunk). Four entries per coupler, zero while the
        // base tree holds.
        {
            int zvRow = rowMap[zeroVRowOf(z)], zpRow = rowMap[zeroPhiRowOf(z)];
            pat.emplace_back(zvRow, z[6]);
            pat.emplace_back(zvRow, z[7]);
            pat.emplace_back(zpRow, z[8]);
            pat.emplace_back(zpRow, z[9]);
        }
        // The CROSS-identity slot (row mode 6, applyRowModeBatch): a re-merged voltage control on a relocated child
        // puts its BUS_TARGET_V on the coupler's DUMMY_Q row, at the child's V column. The child's own dq/dV terms
        // reserve that slot - unless the child has none, a section joined to the rest by couplers only, whose
        // reactive balance is all dummy flows. The row then came out EMPTY and the dummy flow of another coupler
        // was left free (network.xiidm AJSL: DUMMY_Q AJUU's row held columns {25131, 25135}, the child AJSK_8's V
        // column 2816 nowhere; linear-solve ratio 2.7e12 at the first step). A duplicate pair where the slot
        // already exists changes nothing.
        if (relocatedChildOwnRow != nullptr && z[5] >= 0) {
            int distrQRow = rowMap[z[5]];
            if (distrQRow >= 0 && relocatedChildOwnRow[distrQRow] >= 0) {
                pat.emplace_back(distrQRow, relocatedChildOwnRow[distrQRow]);
            }
        }
    }
}

}  // namespace olfgpu
