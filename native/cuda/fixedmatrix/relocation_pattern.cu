// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// docs/gpu-vs-olf.md section 2.3 (equations placed on another bus's row): the union pattern and the per-scenario
// destinations of a reactive balance that moves between its own row, a pilot carrier and a coupler dummy row.
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
#include "../equations/generated/hvdc_scatter.h"

#include "../common/check.cuh"
#include "../common/stats.cuh"
#include "../solver/linear_solver_settings.cuh"
#include "../structure/structure_build.cuh"
#include "stock_placement.cuh"
#include "relocation_pattern.cuh"

using namespace powsybl::openloadflow;

namespace olfgpu {

// ---- Dynamic Q-relocation (production-design doc §11) ---------------------------------------
// A transformer/shunt voltage-controlled bus has two possible reactive-balance rows across the
// outer loop: its own BUS_V row (control OFF) and the free controller's BRANCH_RHO1 / SHUNT_B row
// (control ON). The extractor freezes ONE into closedBranchIndices; the alternate travels at closedBranchIndices[10+rowEnd]
// (packed by GpuAcFamilyPacker as alternateQRow, remapped through rowMap like every other row index).
// When enabled, the closed-branch/shunt fill routes each controlled-bus q output to whichever of the
// two candidate rows is FREE this scenario (row-mode != 0), so one shared batch structure follows a
// per-scenario control toggle instead of solving a subtly wrong system.
//
// Auto-enabled per structure: a structure needs dynamic Q-relocation exactly when some element carries
// an alternate Q row (a transformer/shunt voltage-controlled bus). alternateQRows lives at closedBranchIndices[e*12+10/11] and
// shuntIndices[e*5+4], set to a real row (>= 0) only for a controllable bus, -1 otherwise. When no element has
// one, the union pattern is empty and every fill uses the frozen dest — so plain PF keeps the fast
// path and only voltage-control structures pay the (tiny) per-output selector cost.
bool hasAlternateRows(const int* a, int len) {
    for (int i = 0; i < len; i++) {
        if (a[i] != -1) {
            return true;
        }
    }
    return false;
}

bool hasRelocatable(int nClosedBranch, const int* closedBranchIndices, int nShunt, const int* shuntIndices) {
    for (int e = 0; e < nClosedBranch; e++) {
        if (closedBranchIndices[static_cast<size_t>(e) * 12 + 10] >= 0 || closedBranchIndices[static_cast<size_t>(e) * 12 + 10] <= -2
                || closedBranchIndices[static_cast<size_t>(e) * 12 + 11] >= 0 || closedBranchIndices[static_cast<size_t>(e) * 12 + 11] <= -2) {
            return true;                                     // plain alt row, or the -2-groupId encoding (plan-A6)
        }
    }
    for (int e = 0; e < nShunt; e++) {
        if (shuntIndices[static_cast<size_t>(e) * 5 + 4] >= 0 || shuntIndices[static_cast<size_t>(e) * 5 + 4] <= -2) {
            return true;
        }
    }
    return false;
}

inline int closedBranchAlternateRow(const CbScatter& s, const int* ix) {
    return s.rowEq == 1 ? ix[10 + s.rowEnd] : -1;
}

// Union the alternate Q rows into the CSR pattern: for each closed-branch role-1 Q output emit an
// extra (alternateRow, col) entry so buildDest can resolve a slot there. Deduped by the caller's unique().
void collectClosedBranchRelocationPattern(int nClosedBranch, const int* closedBranchIndices, const int* relocatedChildOwnRow, std::vector<std::pair<int, int>>& pat,
                         int nReactiveRelocationGroups, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows, const int* reactiveRelocationDependentGroupOfRow) {
    for (int e = 0; e < nClosedBranch; e++) {
        const int* ix = closedBranchIndices + static_cast<size_t>(e) * 12;
        for (int k = 0; k < CB_NOUT; k++) {
            const CbScatter& s = CB_SCATTER[k];
            if (s.role != 1) {
                continue;
            }
            RC rc = resolveClosedBranch(s, ix);
            int packedAlternateRow = closedBranchAlternateRow(s, ix);
            int alternateRow = stockPlacementAlternateRow(rc.row, packedAlternateRow, relocatedChildOwnRow);
            int stockSelectorRow = stockPlacementSelectorRow(rc.row, packedAlternateRow, relocatedChildOwnRow);
            if (rc.col < 0) {
                continue;
            }
            int dependentGroup = dependentGroupOfHomeRow(rc.row, alternateRow, relocatedChildOwnRow, reactiveRelocationDependentGroupOfRow);
            if (dependentGroup >= 0) {                                   // DEPENDENT pilot: every candidate row of ITS group
                for (int c = reactiveRelocationGroupStart[dependentGroup]; c < reactiveRelocationGroupStart[dependentGroup + 1]; c++) {
                    pat.emplace_back(reactiveRelocationCandidateRows[c], rc.col);
                }
            }
            if (stockPairOut(rc.row, packedAlternateRow, relocatedChildOwnRow)) {   // STOCK PAIR (see stockPairPick): both rows
                pat.emplace_back(relocatedChildOwnRow[packedAlternateRow], rc.col);
                pat.emplace_back(packedAlternateRow, rc.col);
                continue;
            }
            if (alternateRow <= -2) {                              // GROUP relocation (plan-A6): union EVERY
                int g = -2 - alternateRow;                         // candidate row's slot for this column
                for (int c = reactiveRelocationGroupStart[g]; c < reactiveRelocationGroupStart[g + 1]; c++) {
                    pat.emplace_back(reactiveRelocationCandidateRows[c], rc.col);
                }
            } else if (alternateRow >= 0 && alternateRow != rc.row) {
                pat.emplace_back(alternateRow, rc.col);
            }
        }
    }
}

// Per (closed branch, output): the alternate destination (CSR slot for a role-1 J entry, -2-row for
// a role-0 F residual) and the FROZEN row to test the per-scenario row-mode against. selectorRow == -1
// marks an output that never relocates (the fill then always uses the frozen dest). Sized nClosedBranch*CB_NOUT
// like the frozen dest, so the fill indexes all three arrays identically.
void buildClosedBranchRelocationDestination(int nClosedBranch, const int* closedBranchIndices, const int* relocatedChildOwnRow, const CsrPattern& pos,
                    std::vector<int>& alternateDestination, std::vector<int>& selectorRow,
                    int nReactiveRelocationGroups, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows,
                    std::vector<int>& relocationGroup, std::vector<int>& relocationCandidateOffset, std::vector<int>& relocationCandidateDestinations,
                    const int* reactiveRelocationDependentGroupOfRow, std::vector<char>& relocationDependentPilot) {
    alternateDestination.assign(static_cast<size_t>(nClosedBranch) * CB_NOUT, -1);
    selectorRow.assign(static_cast<size_t>(nClosedBranch) * CB_NOUT, -1);
    relocationGroup.assign(static_cast<size_t>(nClosedBranch) * CB_NOUT, -1);
    relocationCandidateOffset.assign(static_cast<size_t>(nClosedBranch) * CB_NOUT, -1);
    relocationDependentPilot.assign(static_cast<size_t>(nClosedBranch) * CB_NOUT, 0);
    for (int e = 0; e < nClosedBranch; e++) {
        const int* ix = closedBranchIndices + static_cast<size_t>(e) * 12;
        for (int k = 0; k < CB_NOUT; k++) {
            const CbScatter& s = CB_SCATTER[k];
            RC rc = resolveClosedBranch(s, ix);
            int packedAlternateRow = closedBranchAlternateRow(s, ix);
            int alternateRow = stockPlacementAlternateRow(rc.row, packedAlternateRow, relocatedChildOwnRow);
            int stockSelectorRow = stockPlacementSelectorRow(rc.row, packedAlternateRow, relocatedChildOwnRow);
            size_t o = static_cast<size_t>(e) * CB_NOUT + k;
            if (stockPairOut(rc.row, packedAlternateRow, relocatedChildOwnRow)) {
                // THREE destinations (see stockPairOut): the coupler's DUMMY_Q row while an RL pin holds it
                // there (mode 4), NOTHING while the coupler is dead (that row is an identity — the bus is an
                // ordinary group controller again and its DISTR_Q row, diverted to its own row, carries its q),
                // and nothing while it shares. Writing the raw q into the dead coupler's identity row, which
                // the plain two-way rule does on mode 0, corrupts that identity.
                if ((s.role == 1 && rc.col >= 0) || s.role == 0) {
                    relocationGroup[o] = packedAlternateRow;                 // the selector is the dummy row itself
                    relocationCandidateOffset[o] = -static_cast<int>(relocationCandidateDestinations.size()) - 1;
                    int own = relocatedChildOwnRow[packedAlternateRow];
                    relocationCandidateDestinations.push_back(s.role == 1 ? pos.at(own, rc.col) : -2 - own);
                    relocationCandidateDestinations.push_back(s.role == 1 ? pos.at(packedAlternateRow, rc.col) : -2 - packedAlternateRow);
                }
                continue;
            }
            if (alternateRow <= -2) {                              // GROUP relocation (plan-A6): one slot per
                int g = -2 - alternateRow;                         // candidate, selected per scenario by the
                if (s.role == 1 && rc.col >= 0) {            // first-valid-member mode scan
                    relocationGroup[o] = g;
                    relocationCandidateOffset[o] = static_cast<int>(relocationCandidateDestinations.size());
                    for (int c = reactiveRelocationGroupStart[g]; c < reactiveRelocationGroupStart[g + 1]; c++) {
                        relocationCandidateDestinations.push_back(pos.at(reactiveRelocationCandidateRows[c], rc.col));
                    }
                } else if (s.role == 0) {
                    relocationGroup[o] = g;
                    relocationCandidateOffset[o] = static_cast<int>(relocationCandidateDestinations.size());
                    for (int c = reactiveRelocationGroupStart[g]; c < reactiveRelocationGroupStart[g + 1]; c++) {
                        relocationCandidateDestinations.push_back(-2 - reactiveRelocationCandidateRows[c]);
                    }
                }
                continue;
            }
            int dependentGroup = dependentGroupOfHomeRow(rc.row, alternateRow, relocatedChildOwnRow, reactiveRelocationDependentGroupOfRow);
            if (dependentGroup >= 0) {                                   // DEPENDENT pilot (dependentPilotDestination): the group's pool, plus the
                if ((s.role == 1 && rc.col >= 0) || s.role == 0) {   // two-way home row of a relocated child
                    relocationGroup[o] = dependentGroup;
                    relocationCandidateOffset[o] = static_cast<int>(relocationCandidateDestinations.size());
                    for (int c = reactiveRelocationGroupStart[dependentGroup]; c < reactiveRelocationGroupStart[dependentGroup + 1]; c++) {
                        relocationCandidateDestinations.push_back(s.role == 1 ? pos.at(reactiveRelocationCandidateRows[c], rc.col) : -2 - reactiveRelocationCandidateRows[c]);
                    }
                    relocationDependentPilot[o] = 1;
                    if (alternateRow >= 0 && alternateRow != rc.row) {
                        alternateDestination[o] = s.role == 1 ? pos.at(alternateRow, rc.col) : -2 - alternateRow;
                        selectorRow[o] = stockSelectorRow <= -2 ? stockSelectorRow : rc.row;
                    } else {
                        selectorRow[o] = rc.row;                  // a root: the frozen row IS its own
                    }
                }
                continue;
            }
            if (alternateRow < 0 || alternateRow == rc.row) {
                continue;                                    // not a relocatable output
            }
            if (s.role == 1 && rc.col >= 0) {
                alternateDestination[o] = pos.at(alternateRow, rc.col);
                selectorRow[o] = stockSelectorRow <= -2 ? stockSelectorRow : (rc.row >= 0 ? rc.row : alternateRow);   // suppressed frozen dest (remote member):
            } else if (s.role == 0) {                        // select on the ALT row itself (mode 4)
                alternateDestination[o] = -2 - alternateRow;
                selectorRow[o] = stockSelectorRow <= -2 ? stockSelectorRow : (rc.row >= 0 ? rc.row : alternateRow);
            }
        }
    }
}

// Shunt analogue of closedBranchAlternateRow / collectClosedBranchRelocationPattern / buildClosedBranchRelocationDestination. The alternate Q row of a
// voltage-controlling shunt (its own bus's SHUNT_B row <-> bus-V row) travels at shuntIndices[4]. A fixed
// shunt on a transformer/shunt-controlled bus also relocates (alternateQRow was set for that bus). This is
// the one-terminal case dynamic relocation exists for: SHUNT_B has no second terminal to hold its
// column, so a frozen structure freezes it at warm-start.
inline int shuntAlternateRow(const ShScatter& s, const int* ix) {
    return s.rowEq == 1 ? ix[4] : -1;
}

void collectShuntRelocationPattern(int nShunt, const int* shuntIndices, const int* relocatedChildOwnRow, std::vector<std::pair<int, int>>& pat,
                         int nReactiveRelocationGroups, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows, const int* reactiveRelocationDependentGroupOfRow) {
    for (int e = 0; e < nShunt; e++) {
        const int* ix = shuntIndices + static_cast<size_t>(e) * 5;
        for (int k = 0; k < SH_NOUT; k++) {
            const ShScatter& s = SH_SCATTER[k];
            if (s.role != 1) {
                continue;
            }
            RC rc = resolveShunt(s, ix);
            int packedAlternateRow = shuntAlternateRow(s, ix);
            int alternateRow = stockPlacementAlternateRow(rc.row, packedAlternateRow, relocatedChildOwnRow);
            int stockSelectorRow = stockPlacementSelectorRow(rc.row, packedAlternateRow, relocatedChildOwnRow);
            if (rc.col < 0) {
                continue;
            }
            int dependentGroup = dependentGroupOfHomeRow(rc.row, alternateRow, relocatedChildOwnRow, reactiveRelocationDependentGroupOfRow);
            if (dependentGroup >= 0) {                                   // DEPENDENT pilot: every candidate row of ITS group
                for (int c = reactiveRelocationGroupStart[dependentGroup]; c < reactiveRelocationGroupStart[dependentGroup + 1]; c++) {
                    pat.emplace_back(reactiveRelocationCandidateRows[c], rc.col);
                }
            }
            if (stockPairOut(rc.row, packedAlternateRow, relocatedChildOwnRow)) {   // STOCK PAIR (see stockPairPick): both rows
                pat.emplace_back(relocatedChildOwnRow[packedAlternateRow], rc.col);
                pat.emplace_back(packedAlternateRow, rc.col);
                continue;
            }
            if (alternateRow <= -2) {
                int g = -2 - alternateRow;
                for (int c = reactiveRelocationGroupStart[g]; c < reactiveRelocationGroupStart[g + 1]; c++) {
                    pat.emplace_back(reactiveRelocationCandidateRows[c], rc.col);
                }
            } else if (alternateRow >= 0 && alternateRow != rc.row) {
                pat.emplace_back(alternateRow, rc.col);
            }
        }
    }
}

void buildShuntRelocationDestination(int nShunt, const int* shuntIndices, const int* relocatedChildOwnRow, const CsrPattern& pos,
                    std::vector<int>& alternateDestination, std::vector<int>& selectorRow,
                    int nReactiveRelocationGroups, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows,
                    std::vector<int>& relocationGroup, std::vector<int>& relocationCandidateOffset, std::vector<int>& relocationCandidateDestinations,
                    const int* reactiveRelocationDependentGroupOfRow, std::vector<char>& relocationDependentPilot) {
    alternateDestination.assign(static_cast<size_t>(nShunt) * SH_NOUT, -1);
    selectorRow.assign(static_cast<size_t>(nShunt) * SH_NOUT, -1);
    relocationGroup.assign(static_cast<size_t>(nShunt) * SH_NOUT, -1);
    relocationCandidateOffset.assign(static_cast<size_t>(nShunt) * SH_NOUT, -1);
    relocationDependentPilot.assign(static_cast<size_t>(nShunt) * SH_NOUT, 0);
    for (int e = 0; e < nShunt; e++) {
        const int* ix = shuntIndices + static_cast<size_t>(e) * 5;
        for (int k = 0; k < SH_NOUT; k++) {
            const ShScatter& s = SH_SCATTER[k];
            RC rc = resolveShunt(s, ix);
            int packedAlternateRow = shuntAlternateRow(s, ix);
            int alternateRow = stockPlacementAlternateRow(rc.row, packedAlternateRow, relocatedChildOwnRow);
            int stockSelectorRow = stockPlacementSelectorRow(rc.row, packedAlternateRow, relocatedChildOwnRow);
            size_t o = static_cast<size_t>(e) * SH_NOUT + k;
            if (stockPairOut(rc.row, packedAlternateRow, relocatedChildOwnRow)) {
                // THREE destinations (see stockPairOut): the coupler's DUMMY_Q row while an RL pin holds it
                // there (mode 4), NOTHING while the coupler is dead (that row is an identity — the bus is an
                // ordinary group controller again and its DISTR_Q row, diverted to its own row, carries its q),
                // and nothing while it shares. Writing the raw q into the dead coupler's identity row, which
                // the plain two-way rule does on mode 0, corrupts that identity.
                if ((s.role == 1 && rc.col >= 0) || s.role == 0) {
                    relocationGroup[o] = packedAlternateRow;                 // the selector is the dummy row itself
                    relocationCandidateOffset[o] = -static_cast<int>(relocationCandidateDestinations.size()) - 1;
                    int own = relocatedChildOwnRow[packedAlternateRow];
                    relocationCandidateDestinations.push_back(s.role == 1 ? pos.at(own, rc.col) : -2 - own);
                    relocationCandidateDestinations.push_back(s.role == 1 ? pos.at(packedAlternateRow, rc.col) : -2 - packedAlternateRow);
                }
                continue;
            }
            if (alternateRow <= -2) {
                int g = -2 - alternateRow;
                if (s.role == 1 && rc.col >= 0) {
                    relocationGroup[o] = g;
                    relocationCandidateOffset[o] = static_cast<int>(relocationCandidateDestinations.size());
                    for (int c = reactiveRelocationGroupStart[g]; c < reactiveRelocationGroupStart[g + 1]; c++) {
                        relocationCandidateDestinations.push_back(pos.at(reactiveRelocationCandidateRows[c], rc.col));
                    }
                } else if (s.role == 0) {
                    relocationGroup[o] = g;
                    relocationCandidateOffset[o] = static_cast<int>(relocationCandidateDestinations.size());
                    for (int c = reactiveRelocationGroupStart[g]; c < reactiveRelocationGroupStart[g + 1]; c++) {
                        relocationCandidateDestinations.push_back(-2 - reactiveRelocationCandidateRows[c]);
                    }
                }
                continue;
            }
            int dependentGroup = dependentGroupOfHomeRow(rc.row, alternateRow, relocatedChildOwnRow, reactiveRelocationDependentGroupOfRow);
            if (dependentGroup >= 0) {                                   // DEPENDENT pilot (dependentPilotDestination): the group's pool, plus the
                if ((s.role == 1 && rc.col >= 0) || s.role == 0) {   // two-way home row of a relocated child
                    relocationGroup[o] = dependentGroup;
                    relocationCandidateOffset[o] = static_cast<int>(relocationCandidateDestinations.size());
                    for (int c = reactiveRelocationGroupStart[dependentGroup]; c < reactiveRelocationGroupStart[dependentGroup + 1]; c++) {
                        relocationCandidateDestinations.push_back(s.role == 1 ? pos.at(reactiveRelocationCandidateRows[c], rc.col) : -2 - reactiveRelocationCandidateRows[c]);
                    }
                    relocationDependentPilot[o] = 1;
                    if (alternateRow >= 0 && alternateRow != rc.row) {
                        alternateDestination[o] = s.role == 1 ? pos.at(alternateRow, rc.col) : -2 - alternateRow;
                        selectorRow[o] = stockSelectorRow <= -2 ? stockSelectorRow : rc.row;
                    } else {
                        selectorRow[o] = rc.row;                  // a root: the frozen row IS its own
                    }
                }
                continue;
            }
            if (alternateRow < 0 || alternateRow == rc.row) {
                continue;
            }
            if (s.role == 1 && rc.col >= 0) {
                alternateDestination[o] = pos.at(alternateRow, rc.col);
                selectorRow[o] = stockSelectorRow <= -2 ? stockSelectorRow : (rc.row >= 0 ? rc.row : alternateRow);   // suppressed frozen dest: select on ALT row
            } else if (s.role == 0) {
                alternateDestination[o] = -2 - alternateRow;
                selectorRow[o] = stockSelectorRow <= -2 ? stockSelectorRow : (rc.row >= 0 ? rc.row : alternateRow);
            }
        }
    }
}

// HVDC analogue of the stock placement. An AC-emulation converter bus that is a coupler-tree child keeps its P
// balance on the parent coupler's DUMMY_P row; a scenario that opens that coupler turns the dummy row into an
// identity and the balance goes back to the bus's own row, and the link's p1/p2 terms must follow it there like
// every branch and shunt term does. Left on the dummy row they were wiped with the identity, and once the freeze
// loop released the link the converter bus carried no HVDC flow (network.xiidm busbar ADNO).
void collectHvdcAcEmulationRelocationPattern(int nHvdcAcEmulation, const int* hvdcAcEmulationIndices, const int* relocatedChildOwnRow, std::vector<std::pair<int, int>>& pat) {
    if (relocatedChildOwnRow == nullptr) {
        return;
    }
    for (int e = 0; e < nHvdcAcEmulation; e++) {
        const int* ix = hvdcAcEmulationIndices + static_cast<size_t>(e) * 4;
        for (int k = 0; k < HVDC_NOUT; k++) {
            RC rc = resolveHvdcAcEmulation(HVDC_SCATTER[k], ix);
            if (HVDC_SCATTER[k].role == 1 && rc.row >= 0 && rc.col >= 0 && relocatedChildOwnRow[rc.row] >= 0) {
                pat.emplace_back(relocatedChildOwnRow[rc.row], rc.col);
            }
        }
    }
}

void buildHvdcAcEmulationRelocationDestination(int nHvdcAcEmulation, const int* hvdcAcEmulationIndices, const int* relocatedChildOwnRow, const CsrPattern& pos,
                    std::vector<int>& alternateDestination, std::vector<int>& selectorRow) {
    alternateDestination.assign(static_cast<size_t>(nHvdcAcEmulation) * HVDC_NOUT, -1);
    selectorRow.assign(static_cast<size_t>(nHvdcAcEmulation) * HVDC_NOUT, -1);
    if (relocatedChildOwnRow == nullptr) {
        return;
    }
    for (int e = 0; e < nHvdcAcEmulation; e++) {
        const int* ix = hvdcAcEmulationIndices + static_cast<size_t>(e) * 4;
        for (int k = 0; k < HVDC_NOUT; k++) {
            const HvdcScatter& sc = HVDC_SCATTER[k];
            RC rc = resolveHvdcAcEmulation(sc, ix);
            if (rc.row < 0 || relocatedChildOwnRow[rc.row] < 0) {
                continue;
            }
            int own = relocatedChildOwnRow[rc.row];
            size_t o = static_cast<size_t>(e) * HVDC_NOUT + k;
            if (sc.role == 1 && rc.col >= 0) {
                alternateDestination[o] = pos.at(own, rc.col);
                selectorRow[o] = stockPlacementSelectorRow(rc.row, -1, relocatedChildOwnRow);   // divert only while the dummy row is an identity
            } else if (sc.role == 0) {
                alternateDestination[o] = -2 - own;
                selectorRow[o] = stockPlacementSelectorRow(rc.row, -1, relocatedChildOwnRow);
            }
        }
    }
}

// Open-branch analogue of shuntAlternateRow / collectShuntRelocationPattern / buildShuntRelocationDestination: the ALTERNATE Q row
// of each side's bus travels in a SEPARATE openBranchAlternateReactiveDestination array (nOpenBranch*2) rather than inside openBranchIndices —
// the connected side's half-open charging q follows a pinned remote member's mode-4 row or a
// relocation group's carrier exactly like its closed-branch terms.
inline int openBranchAlternateRow(const ObScatter& s, const int* alternateQRows, int e) {
    return s.rowEq == 1 ? alternateQRows[e * 2 + s.rowEnd] : -1;
}

void collectOpenBranchRelocationPattern(int nOpenBranch, const int* openBranchIndices, const int* openBranchAlternateReactiveDestination, const int* relocatedChildOwnRow, std::vector<std::pair<int, int>>& pat,
                         int nReactiveRelocationGroups, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows, const int* reactiveRelocationDependentGroupOfRow) {
    for (int e = 0; e < nOpenBranch; e++) {
        const int* ix = openBranchIndices + static_cast<size_t>(e) * 6;
        for (int k = 0; k < OB_NOUT; k++) {
            const ObScatter& s = OB_SCATTER[k];
            if (s.role != 1) {
                continue;
            }
            RC rc = resolveOpenBranch(s, ix);
            int packedAlternateRow = openBranchAlternateRow(s, openBranchAlternateReactiveDestination, e);
            int alternateRow = stockPlacementAlternateRow(rc.row, packedAlternateRow, relocatedChildOwnRow);
            int stockSelectorRow = stockPlacementSelectorRow(rc.row, packedAlternateRow, relocatedChildOwnRow);
            if (rc.col < 0) {
                continue;
            }
            int dependentGroup = dependentGroupOfHomeRow(rc.row, alternateRow, relocatedChildOwnRow, reactiveRelocationDependentGroupOfRow);
            if (dependentGroup >= 0) {                                   // DEPENDENT pilot: every candidate row of ITS group
                for (int c = reactiveRelocationGroupStart[dependentGroup]; c < reactiveRelocationGroupStart[dependentGroup + 1]; c++) {
                    pat.emplace_back(reactiveRelocationCandidateRows[c], rc.col);
                }
            }
            if (stockPairOut(rc.row, packedAlternateRow, relocatedChildOwnRow)) {   // STOCK PAIR (see stockPairPick): both rows
                pat.emplace_back(relocatedChildOwnRow[packedAlternateRow], rc.col);
                pat.emplace_back(packedAlternateRow, rc.col);
                continue;
            }
            if (alternateRow <= -2) {
                int g = -2 - alternateRow;
                for (int c = reactiveRelocationGroupStart[g]; c < reactiveRelocationGroupStart[g + 1]; c++) {
                    pat.emplace_back(reactiveRelocationCandidateRows[c], rc.col);
                }
            } else if (alternateRow >= 0 && alternateRow != rc.row) {
                pat.emplace_back(alternateRow, rc.col);
            }
        }
    }
}

void buildOpenBranchRelocationDestination(int nOpenBranch, const int* openBranchIndices, const int* openBranchAlternateReactiveDestination, const int* relocatedChildOwnRow, const CsrPattern& pos,
                    std::vector<int>& alternateDestination, std::vector<int>& selectorRow,
                    int nReactiveRelocationGroups, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows,
                    std::vector<int>& relocationGroup, std::vector<int>& relocationCandidateOffset, std::vector<int>& relocationCandidateDestinations,
                    const int* reactiveRelocationDependentGroupOfRow, std::vector<char>& relocationDependentPilot) {
    alternateDestination.assign(static_cast<size_t>(nOpenBranch) * OB_NOUT, -1);
    selectorRow.assign(static_cast<size_t>(nOpenBranch) * OB_NOUT, -1);
    relocationGroup.assign(static_cast<size_t>(nOpenBranch) * OB_NOUT, -1);
    relocationCandidateOffset.assign(static_cast<size_t>(nOpenBranch) * OB_NOUT, -1);
    relocationDependentPilot.assign(static_cast<size_t>(nOpenBranch) * OB_NOUT, 0);
    for (int e = 0; e < nOpenBranch; e++) {
        const int* ix = openBranchIndices + static_cast<size_t>(e) * 6;
        for (int k = 0; k < OB_NOUT; k++) {
            const ObScatter& s = OB_SCATTER[k];
            RC rc = resolveOpenBranch(s, ix);
            int packedAlternateRow = openBranchAlternateRow(s, openBranchAlternateReactiveDestination, e);
            int alternateRow = stockPlacementAlternateRow(rc.row, packedAlternateRow, relocatedChildOwnRow);
            int stockSelectorRow = stockPlacementSelectorRow(rc.row, packedAlternateRow, relocatedChildOwnRow);
            size_t o = static_cast<size_t>(e) * OB_NOUT + k;
            if (stockPairOut(rc.row, packedAlternateRow, relocatedChildOwnRow)) {
                // THREE destinations (see stockPairOut): the coupler's DUMMY_Q row while an RL pin holds it
                // there (mode 4), NOTHING while the coupler is dead (that row is an identity — the bus is an
                // ordinary group controller again and its DISTR_Q row, diverted to its own row, carries its q),
                // and nothing while it shares. Writing the raw q into the dead coupler's identity row, which
                // the plain two-way rule does on mode 0, corrupts that identity.
                if ((s.role == 1 && rc.col >= 0) || s.role == 0) {
                    relocationGroup[o] = packedAlternateRow;                 // the selector is the dummy row itself
                    relocationCandidateOffset[o] = -static_cast<int>(relocationCandidateDestinations.size()) - 1;
                    int own = relocatedChildOwnRow[packedAlternateRow];
                    relocationCandidateDestinations.push_back(s.role == 1 ? pos.at(own, rc.col) : -2 - own);
                    relocationCandidateDestinations.push_back(s.role == 1 ? pos.at(packedAlternateRow, rc.col) : -2 - packedAlternateRow);
                }
                continue;
            }
            if (alternateRow <= -2) {
                int g = -2 - alternateRow;
                if (s.role == 1 && rc.col >= 0) {
                    relocationGroup[o] = g;
                    relocationCandidateOffset[o] = static_cast<int>(relocationCandidateDestinations.size());
                    for (int c = reactiveRelocationGroupStart[g]; c < reactiveRelocationGroupStart[g + 1]; c++) {
                        relocationCandidateDestinations.push_back(pos.at(reactiveRelocationCandidateRows[c], rc.col));
                    }
                } else if (s.role == 0) {
                    relocationGroup[o] = g;
                    relocationCandidateOffset[o] = static_cast<int>(relocationCandidateDestinations.size());
                    for (int c = reactiveRelocationGroupStart[g]; c < reactiveRelocationGroupStart[g + 1]; c++) {
                        relocationCandidateDestinations.push_back(-2 - reactiveRelocationCandidateRows[c]);
                    }
                }
                continue;
            }
            int dependentGroup = dependentGroupOfHomeRow(rc.row, alternateRow, relocatedChildOwnRow, reactiveRelocationDependentGroupOfRow);
            if (dependentGroup >= 0) {                                   // DEPENDENT pilot (dependentPilotDestination): the group's pool, plus the
                if ((s.role == 1 && rc.col >= 0) || s.role == 0) {   // two-way home row of a relocated child
                    relocationGroup[o] = dependentGroup;
                    relocationCandidateOffset[o] = static_cast<int>(relocationCandidateDestinations.size());
                    for (int c = reactiveRelocationGroupStart[dependentGroup]; c < reactiveRelocationGroupStart[dependentGroup + 1]; c++) {
                        relocationCandidateDestinations.push_back(s.role == 1 ? pos.at(reactiveRelocationCandidateRows[c], rc.col) : -2 - reactiveRelocationCandidateRows[c]);
                    }
                    relocationDependentPilot[o] = 1;
                    if (alternateRow >= 0 && alternateRow != rc.row) {
                        alternateDestination[o] = s.role == 1 ? pos.at(alternateRow, rc.col) : -2 - alternateRow;
                        selectorRow[o] = stockSelectorRow <= -2 ? stockSelectorRow : rc.row;
                    } else {
                        selectorRow[o] = rc.row;                  // a root: the frozen row IS its own
                    }
                }
                continue;
            }
            if (alternateRow < 0 || alternateRow == rc.row) {
                continue;
            }
            if (s.role == 1 && rc.col >= 0) {
                alternateDestination[o] = pos.at(alternateRow, rc.col);
                selectorRow[o] = stockSelectorRow <= -2 ? stockSelectorRow : (rc.row >= 0 ? rc.row : alternateRow);   // suppressed frozen dest: select on ALT row
            } else if (s.role == 0) {
                alternateDestination[o] = -2 - alternateRow;
                selectorRow[o] = stockSelectorRow <= -2 ? stockSelectorRow : (rc.row >= 0 ? rc.row : alternateRow);
            }
        }
    }
}

// plan-B4: the FROZEN role-0 dest of a GROUP-relocated output backs only the qBuf DIAGNOSTIC
// fill (dynamicRelocation = 0; every Newton fill selects a pool slot instead) — point it at the group's
// FALLBACK row (the pilot's own row, read by no outer loop) so a remote member's diagnostic q
// row carries ONLY its own injection (fillRemoteMemberQBatch). Without this the pilot's whole
// balance lands on the base carrier's row and corrupts the free member's reactive-limit check.
void patchGroupFrozenDestinations(DeviceBuf& dest, const std::vector<int>& relocationGroup,
                           const std::vector<int>& relocationCandidateOffset, int outputCount,
                           const int* scatterRole, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows) {
    for (size_t o = 0; o < relocationGroup.size(); o++) {
        int g = relocationGroup[o];
        // A STOCK PAIR (negative offset) parks the DUMMY ROW in the group slot, not a group id — reading
        // reactiveRelocationGroupStart[row] there walked off the candidate table and wrote a garbage dest, which the device then
        // used: "CUDA error: an illegal memory access" out of createBatchContext.
        if (g >= 0 && relocationCandidateOffset[o] < 0) {
            continue;
        }
        if (g >= 0 && scatterRole[o % outputCount] == 0) {
            int v = -2 - reactiveRelocationCandidateRows[reactiveRelocationGroupStart[g]];
            CUDA_CHECK(cudaMemcpy(dest.as<int>() + o, &v, sizeof(int), cudaMemcpyHostToDevice));
        }
    }
}

}  // namespace olfgpu
