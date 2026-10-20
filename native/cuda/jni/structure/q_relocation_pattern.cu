// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Dynamic Q-relocation: the alternate pattern entries and the per-scenario destinations of a reactive balance
// that moves between its own row, a pilot carrier and a coupler dummy row, plus continuous phase control.
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

#include "closed_branch_scatter.h"
#include "open_branch_scatter.h"
#include "shunt_scatter.h"
#include "hvdc_scatter.h"

#include "../common/check.cuh"
#include "../common/stats.cuh"
#include "../solver/linear_solver_settings.cuh"
#include "structure_build.cuh"

using namespace powsybl::openloadflow;

namespace olfgpu {

// ---- Dynamic Q-relocation (production-design doc §11) ---------------------------------------
// A transformer/shunt voltage-controlled bus has two possible reactive-balance rows across the
// outer loop: its own BUS_V row (control OFF) and the free controller's BRANCH_RHO1 / SHUNT_B row
// (control ON). The extractor freezes ONE into closedBranchIndices; the alternate travels at closedBranchIndices[10+rowEnd]
// (packed by GpuAcFamilyPacker as qRowAlt, remapped through rowMap like every other row index).
// When enabled, the closed-branch/shunt fill routes each controlled-bus q output to whichever of the
// two candidate rows is FREE this scenario (row-mode != 0), so one shared batch structure follows a
// per-scenario control toggle instead of solving a subtly wrong system.
//
// Auto-enabled per structure: a structure needs dynamic Q-relocation exactly when some element carries
// an alternate Q row (a transformer/shunt voltage-controlled bus). qAlt lives at closedBranchIndices[e*12+10/11] and
// shuntIndices[e*5+4], set to a real row (>= 0) only for a controllable bus, -1 otherwise. When no element has
// one, the union pattern is empty and every fill uses the frozen dest — so plain PF keeps the fast
// path and only voltage-control structures pay the (tiny) per-output selector cost.
bool hasAltEntries(const int* a, int len) {
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

inline int cbAltRow(const CbScatter& s, const int* ix) {
    return s.rowEq == 1 ? ix[10 + s.rowEnd] : -1;
}

// CONTINUOUS PHASE CONTROL (BRANCH_TARGET_P): when a phase-controller branch's control is enabled
// per scenario (row mode 3 on the CONTROLLER's a1-paired row), the controlled side's P-flow outputs
// are emitted AGAIN at that row — the p-flow equation replacing the a1 identity. Union those
// (a1Row, col) positions into the pattern; closedBranchPhaseControlSide[e] = -1 none, else (controllerA1Col << 1) | side
// — the entry sits on the CONTROLLED element, whose own a1 may be absent (REMOTE control).
void collectClosedBranchPhaseControlPattern(int nClosedBranch, const int* closedBranchIndices, const int* closedBranchPhaseControlSide,
                           const std::vector<int>& rowMap, std::vector<std::pair<int, int>>& pat) {
    for (int e = 0; e < nClosedBranch; e++) {
        if (closedBranchPhaseControlSide[e] < 0) {
            continue;
        }
        int side = closedBranchPhaseControlSide[e] & 1;
        int a1Col = closedBranchPhaseControlSide[e] >> 1;
        const int* ix = closedBranchIndices + static_cast<size_t>(e) * 12;
        int pcRow = rowMap[a1Col];
        int k0 = side == 0 ? 0 : 14;                         // p1 block / p2 block of CB_SCATTER
        for (int k = k0 + 1; k <= k0 + 6; k++) {             // the 6 derivative outputs
            RC rc = resolveClosedBranch(CB_SCATTER[k], ix);
            if (rc.col >= 0) {
                pat.emplace_back(pcRow, rc.col);
            }
        }
    }
}

// Per (closed branch, output): the phase-row destination (J slot, or -2-row for the P residual) and
// the a1 row whose per-scenario mode gates the emission (mode 3 = control enabled). -1 = no emission.
void buildClosedBranchPhaseControlDestination(int nClosedBranch, const int* closedBranchIndices, const int* closedBranchPhaseControlSide, const std::vector<int>& rowMap,
                      const CsrPattern& pos, std::vector<int>& phaseControlDestination, std::vector<int>& phaseControlSelector) {
    phaseControlDestination.assign(static_cast<size_t>(nClosedBranch) * CB_NOUT, -1);
    phaseControlSelector.assign(static_cast<size_t>(nClosedBranch) * CB_NOUT, -1);
    for (int e = 0; e < nClosedBranch; e++) {
        if (closedBranchPhaseControlSide[e] < 0) {
            continue;
        }
        int side = closedBranchPhaseControlSide[e] & 1;
        int a1Col = closedBranchPhaseControlSide[e] >> 1;
        const int* ix = closedBranchIndices + static_cast<size_t>(e) * 12;
        int pcRow = rowMap[a1Col];
        int k0 = side == 0 ? 0 : 14;
        phaseControlDestination[static_cast<size_t>(e) * CB_NOUT + k0] = -2 - pcRow;    // the P residual
        phaseControlSelector[static_cast<size_t>(e) * CB_NOUT + k0] = a1Col;          // mode-3 gate lives at the CONTROLLER's a1 row
        for (int k = k0 + 1; k <= k0 + 6; k++) {
            RC rc = resolveClosedBranch(CB_SCATTER[k], ix);
            if (rc.col >= 0) {
                phaseControlDestination[static_cast<size_t>(e) * CB_NOUT + k] = pos.at(pcRow, rc.col);
                phaseControlSelector[static_cast<size_t>(e) * CB_NOUT + k] = a1Col;
            }
        }
    }
}

// Union the alternate Q rows into the CSR pattern: for each closed-branch role-1 Q output emit an
// extra (altRow, col) entry so buildDest can resolve a slot there. Deduped by the caller's unique().
void collectCbAltPattern(int nClosedBranch, const int* closedBranchIndices, const int* relocOwn, std::vector<std::pair<int, int>>& pat,
                         int nQr, const int* qrPtr, const int* qrRows, const int* depGroupOfRow) {
    for (int e = 0; e < nClosedBranch; e++) {
        const int* ix = closedBranchIndices + static_cast<size_t>(e) * 12;
        for (int k = 0; k < CB_NOUT; k++) {
            const CbScatter& s = CB_SCATTER[k];
            if (s.role != 1) {
                continue;
            }
            RC rc = resolveClosedBranch(s, ix);
            int origAlt = cbAltRow(s, ix);
            int altRow = stockAltRow(rc.row, origAlt, relocOwn);
            int stockSel = stockSelRow(rc.row, origAlt, relocOwn);
            if (rc.col < 0) {
                continue;
            }
            int dg = depGroupOf(rc.row, altRow, relocOwn, depGroupOfRow);
            if (dg >= 0) {                                   // DEPENDENT pilot: every candidate row of ITS group
                for (int c = qrPtr[dg]; c < qrPtr[dg + 1]; c++) {
                    pat.emplace_back(qrRows[c], rc.col);
                }
            }
            if (stockPairOut(rc.row, origAlt, relocOwn)) {   // STOCK PAIR (see stockPairPick): both rows
                pat.emplace_back(relocOwn[origAlt], rc.col);
                pat.emplace_back(origAlt, rc.col);
                continue;
            }
            if (altRow <= -2) {                              // GROUP relocation (plan-A6): union EVERY
                int g = -2 - altRow;                         // candidate row's slot for this column
                for (int c = qrPtr[g]; c < qrPtr[g + 1]; c++) {
                    pat.emplace_back(qrRows[c], rc.col);
                }
            } else if (altRow >= 0 && altRow != rc.row) {
                pat.emplace_back(altRow, rc.col);
            }
        }
    }
}

// Per (closed branch, output): the alternate destination (CSR slot for a role-1 J entry, -2-row for
// a role-0 F residual) and the FROZEN row to test the per-scenario row-mode against. selRow == -1
// marks an output that never relocates (the fill then always uses the frozen dest). Sized nClosedBranch*CB_NOUT
// like the frozen dest, so the fill indexes all three arrays identically.
void buildCbDynDest(int nClosedBranch, const int* closedBranchIndices, const int* relocOwn, const CsrPattern& pos,
                    std::vector<int>& altDest, std::vector<int>& selRow,
                    int nQr, const int* qrPtr, const int* qrRows,
                    std::vector<int>& relocGroup, std::vector<int>& relocOff, std::vector<int>& relocPool,
                    const int* depGroupOfRow, std::vector<char>& relocDep) {
    altDest.assign(static_cast<size_t>(nClosedBranch) * CB_NOUT, -1);
    selRow.assign(static_cast<size_t>(nClosedBranch) * CB_NOUT, -1);
    relocGroup.assign(static_cast<size_t>(nClosedBranch) * CB_NOUT, -1);
    relocOff.assign(static_cast<size_t>(nClosedBranch) * CB_NOUT, -1);
    relocDep.assign(static_cast<size_t>(nClosedBranch) * CB_NOUT, 0);
    for (int e = 0; e < nClosedBranch; e++) {
        const int* ix = closedBranchIndices + static_cast<size_t>(e) * 12;
        for (int k = 0; k < CB_NOUT; k++) {
            const CbScatter& s = CB_SCATTER[k];
            RC rc = resolveClosedBranch(s, ix);
            int origAlt = cbAltRow(s, ix);
            int altRow = stockAltRow(rc.row, origAlt, relocOwn);
            int stockSel = stockSelRow(rc.row, origAlt, relocOwn);
            size_t o = static_cast<size_t>(e) * CB_NOUT + k;
            if (stockPairOut(rc.row, origAlt, relocOwn)) {
                // THREE destinations (see stockPairOut): the coupler's DUMMY_Q row while an RL pin holds it
                // there (mode 4), NOTHING while the coupler is dead (that row is an identity — the bus is an
                // ordinary group controller again and its DISTR_Q row, diverted to its own row, carries its q),
                // and nothing while it shares. Writing the raw q into the dead coupler's identity row, which
                // the plain two-way rule does on mode 0, corrupts that identity.
                if ((s.role == 1 && rc.col >= 0) || s.role == 0) {
                    relocGroup[o] = origAlt;                 // the selector is the dummy row itself
                    relocOff[o] = -static_cast<int>(relocPool.size()) - 1;
                    int own = relocOwn[origAlt];
                    relocPool.push_back(s.role == 1 ? pos.at(own, rc.col) : -2 - own);
                    relocPool.push_back(s.role == 1 ? pos.at(origAlt, rc.col) : -2 - origAlt);
                }
                continue;
            }
            if (altRow <= -2) {                              // GROUP relocation (plan-A6): one slot per
                int g = -2 - altRow;                         // candidate, selected per scenario by the
                if (s.role == 1 && rc.col >= 0) {            // first-valid-member mode scan
                    relocGroup[o] = g;
                    relocOff[o] = static_cast<int>(relocPool.size());
                    for (int c = qrPtr[g]; c < qrPtr[g + 1]; c++) {
                        relocPool.push_back(pos.at(qrRows[c], rc.col));
                    }
                } else if (s.role == 0) {
                    relocGroup[o] = g;
                    relocOff[o] = static_cast<int>(relocPool.size());
                    for (int c = qrPtr[g]; c < qrPtr[g + 1]; c++) {
                        relocPool.push_back(-2 - qrRows[c]);
                    }
                }
                continue;
            }
            int dg = depGroupOf(rc.row, altRow, relocOwn, depGroupOfRow);
            if (dg >= 0) {                                   // DEPENDENT pilot (depPick): the group's pool, plus the
                if ((s.role == 1 && rc.col >= 0) || s.role == 0) {   // two-way home row of a relocated child
                    relocGroup[o] = dg;
                    relocOff[o] = static_cast<int>(relocPool.size());
                    for (int c = qrPtr[dg]; c < qrPtr[dg + 1]; c++) {
                        relocPool.push_back(s.role == 1 ? pos.at(qrRows[c], rc.col) : -2 - qrRows[c]);
                    }
                    relocDep[o] = 1;
                    if (altRow >= 0 && altRow != rc.row) {
                        altDest[o] = s.role == 1 ? pos.at(altRow, rc.col) : -2 - altRow;
                        selRow[o] = stockSel <= -2 ? stockSel : rc.row;
                    } else {
                        selRow[o] = rc.row;                  // a root: the frozen row IS its own
                    }
                }
                continue;
            }
            if (altRow < 0 || altRow == rc.row) {
                continue;                                    // not a relocatable output
            }
            if (s.role == 1 && rc.col >= 0) {
                altDest[o] = pos.at(altRow, rc.col);
                selRow[o] = stockSel <= -2 ? stockSel : (rc.row >= 0 ? rc.row : altRow);   // suppressed frozen dest (remote member):
            } else if (s.role == 0) {                        // select on the ALT row itself (mode 4)
                altDest[o] = -2 - altRow;
                selRow[o] = stockSel <= -2 ? stockSel : (rc.row >= 0 ? rc.row : altRow);
            }
        }
    }
}

// Shunt analogue of cbAltRow / collectCbAltPattern / buildCbDynDest. The alternate Q row of a
// voltage-controlling shunt (its own bus's SHUNT_B row <-> bus-V row) travels at shuntIndices[4]. A fixed
// shunt on a transformer/shunt-controlled bus also relocates (qRowAlt was set for that bus). This is
// the one-terminal case dynamic relocation exists for: SHUNT_B has no second terminal to hold its
// column, so a frozen structure freezes it at warm-start.
inline int shAltRow(const ShScatter& s, const int* ix) {
    return s.rowEq == 1 ? ix[4] : -1;
}

void collectShAltPattern(int nShunt, const int* shuntIndices, const int* relocOwn, std::vector<std::pair<int, int>>& pat,
                         int nQr, const int* qrPtr, const int* qrRows, const int* depGroupOfRow) {
    for (int e = 0; e < nShunt; e++) {
        const int* ix = shuntIndices + static_cast<size_t>(e) * 5;
        for (int k = 0; k < SH_NOUT; k++) {
            const ShScatter& s = SH_SCATTER[k];
            if (s.role != 1) {
                continue;
            }
            RC rc = resolveShunt(s, ix);
            int origAlt = shAltRow(s, ix);
            int altRow = stockAltRow(rc.row, origAlt, relocOwn);
            int stockSel = stockSelRow(rc.row, origAlt, relocOwn);
            if (rc.col < 0) {
                continue;
            }
            int dg = depGroupOf(rc.row, altRow, relocOwn, depGroupOfRow);
            if (dg >= 0) {                                   // DEPENDENT pilot: every candidate row of ITS group
                for (int c = qrPtr[dg]; c < qrPtr[dg + 1]; c++) {
                    pat.emplace_back(qrRows[c], rc.col);
                }
            }
            if (stockPairOut(rc.row, origAlt, relocOwn)) {   // STOCK PAIR (see stockPairPick): both rows
                pat.emplace_back(relocOwn[origAlt], rc.col);
                pat.emplace_back(origAlt, rc.col);
                continue;
            }
            if (altRow <= -2) {
                int g = -2 - altRow;
                for (int c = qrPtr[g]; c < qrPtr[g + 1]; c++) {
                    pat.emplace_back(qrRows[c], rc.col);
                }
            } else if (altRow >= 0 && altRow != rc.row) {
                pat.emplace_back(altRow, rc.col);
            }
        }
    }
}

void buildShDynDest(int nShunt, const int* shuntIndices, const int* relocOwn, const CsrPattern& pos,
                    std::vector<int>& altDest, std::vector<int>& selRow,
                    int nQr, const int* qrPtr, const int* qrRows,
                    std::vector<int>& relocGroup, std::vector<int>& relocOff, std::vector<int>& relocPool,
                    const int* depGroupOfRow, std::vector<char>& relocDep) {
    altDest.assign(static_cast<size_t>(nShunt) * SH_NOUT, -1);
    selRow.assign(static_cast<size_t>(nShunt) * SH_NOUT, -1);
    relocGroup.assign(static_cast<size_t>(nShunt) * SH_NOUT, -1);
    relocOff.assign(static_cast<size_t>(nShunt) * SH_NOUT, -1);
    relocDep.assign(static_cast<size_t>(nShunt) * SH_NOUT, 0);
    for (int e = 0; e < nShunt; e++) {
        const int* ix = shuntIndices + static_cast<size_t>(e) * 5;
        for (int k = 0; k < SH_NOUT; k++) {
            const ShScatter& s = SH_SCATTER[k];
            RC rc = resolveShunt(s, ix);
            int origAlt = shAltRow(s, ix);
            int altRow = stockAltRow(rc.row, origAlt, relocOwn);
            int stockSel = stockSelRow(rc.row, origAlt, relocOwn);
            size_t o = static_cast<size_t>(e) * SH_NOUT + k;
            if (stockPairOut(rc.row, origAlt, relocOwn)) {
                // THREE destinations (see stockPairOut): the coupler's DUMMY_Q row while an RL pin holds it
                // there (mode 4), NOTHING while the coupler is dead (that row is an identity — the bus is an
                // ordinary group controller again and its DISTR_Q row, diverted to its own row, carries its q),
                // and nothing while it shares. Writing the raw q into the dead coupler's identity row, which
                // the plain two-way rule does on mode 0, corrupts that identity.
                if ((s.role == 1 && rc.col >= 0) || s.role == 0) {
                    relocGroup[o] = origAlt;                 // the selector is the dummy row itself
                    relocOff[o] = -static_cast<int>(relocPool.size()) - 1;
                    int own = relocOwn[origAlt];
                    relocPool.push_back(s.role == 1 ? pos.at(own, rc.col) : -2 - own);
                    relocPool.push_back(s.role == 1 ? pos.at(origAlt, rc.col) : -2 - origAlt);
                }
                continue;
            }
            if (altRow <= -2) {
                int g = -2 - altRow;
                if (s.role == 1 && rc.col >= 0) {
                    relocGroup[o] = g;
                    relocOff[o] = static_cast<int>(relocPool.size());
                    for (int c = qrPtr[g]; c < qrPtr[g + 1]; c++) {
                        relocPool.push_back(pos.at(qrRows[c], rc.col));
                    }
                } else if (s.role == 0) {
                    relocGroup[o] = g;
                    relocOff[o] = static_cast<int>(relocPool.size());
                    for (int c = qrPtr[g]; c < qrPtr[g + 1]; c++) {
                        relocPool.push_back(-2 - qrRows[c]);
                    }
                }
                continue;
            }
            int dg = depGroupOf(rc.row, altRow, relocOwn, depGroupOfRow);
            if (dg >= 0) {                                   // DEPENDENT pilot (depPick): the group's pool, plus the
                if ((s.role == 1 && rc.col >= 0) || s.role == 0) {   // two-way home row of a relocated child
                    relocGroup[o] = dg;
                    relocOff[o] = static_cast<int>(relocPool.size());
                    for (int c = qrPtr[dg]; c < qrPtr[dg + 1]; c++) {
                        relocPool.push_back(s.role == 1 ? pos.at(qrRows[c], rc.col) : -2 - qrRows[c]);
                    }
                    relocDep[o] = 1;
                    if (altRow >= 0 && altRow != rc.row) {
                        altDest[o] = s.role == 1 ? pos.at(altRow, rc.col) : -2 - altRow;
                        selRow[o] = stockSel <= -2 ? stockSel : rc.row;
                    } else {
                        selRow[o] = rc.row;                  // a root: the frozen row IS its own
                    }
                }
                continue;
            }
            if (altRow < 0 || altRow == rc.row) {
                continue;
            }
            if (s.role == 1 && rc.col >= 0) {
                altDest[o] = pos.at(altRow, rc.col);
                selRow[o] = stockSel <= -2 ? stockSel : (rc.row >= 0 ? rc.row : altRow);   // suppressed frozen dest: select on ALT row
            } else if (s.role == 0) {
                altDest[o] = -2 - altRow;
                selRow[o] = stockSel <= -2 ? stockSel : (rc.row >= 0 ? rc.row : altRow);
            }
        }
    }
}

// HVDC analogue of the stock placement. An AC-emulation converter bus that is a coupler-tree child keeps its P
// balance on the parent coupler's DUMMY_P row; a scenario that opens that coupler turns the dummy row into an
// identity and the balance goes back to the bus's own row, and the link's p1/p2 terms must follow it there like
// every branch and shunt term does. Left on the dummy row they were wiped with the identity, and once the freeze
// loop released the link the converter bus carried no HVDC flow (network.xiidm busbar ADNO).
void collectHvAltPattern(int nHvdcAcEmulation, const int* hvdcAcEmulationIndices, const int* relocOwn, std::vector<std::pair<int, int>>& pat) {
    if (relocOwn == nullptr) {
        return;
    }
    for (int e = 0; e < nHvdcAcEmulation; e++) {
        const int* ix = hvdcAcEmulationIndices + static_cast<size_t>(e) * 4;
        for (int k = 0; k < HVDC_NOUT; k++) {
            RC rc = resolveHvdcAcEmulation(HVDC_SCATTER[k], ix);
            if (HVDC_SCATTER[k].role == 1 && rc.row >= 0 && rc.col >= 0 && relocOwn[rc.row] >= 0) {
                pat.emplace_back(relocOwn[rc.row], rc.col);
            }
        }
    }
}

void buildHvDynDest(int nHvdcAcEmulation, const int* hvdcAcEmulationIndices, const int* relocOwn, const CsrPattern& pos,
                    std::vector<int>& altDest, std::vector<int>& selRow) {
    altDest.assign(static_cast<size_t>(nHvdcAcEmulation) * HVDC_NOUT, -1);
    selRow.assign(static_cast<size_t>(nHvdcAcEmulation) * HVDC_NOUT, -1);
    if (relocOwn == nullptr) {
        return;
    }
    for (int e = 0; e < nHvdcAcEmulation; e++) {
        const int* ix = hvdcAcEmulationIndices + static_cast<size_t>(e) * 4;
        for (int k = 0; k < HVDC_NOUT; k++) {
            const HvdcScatter& sc = HVDC_SCATTER[k];
            RC rc = resolveHvdcAcEmulation(sc, ix);
            if (rc.row < 0 || relocOwn[rc.row] < 0) {
                continue;
            }
            int own = relocOwn[rc.row];
            size_t o = static_cast<size_t>(e) * HVDC_NOUT + k;
            if (sc.role == 1 && rc.col >= 0) {
                altDest[o] = pos.at(own, rc.col);
                selRow[o] = stockSelRow(rc.row, -1, relocOwn);   // divert only while the dummy row is an identity
            } else if (sc.role == 0) {
                altDest[o] = -2 - own;
                selRow[o] = stockSelRow(rc.row, -1, relocOwn);
            }
        }
    }
}

// Open-branch analogue of shAltRow / collectShAltPattern / buildShDynDest: the ALTERNATE Q row
// of each side's bus travels in a SEPARATE openBranchAlternateReactiveDestination array (nOpenBranch*2) rather than inside openBranchIndices —
// the connected side's half-open charging q follows a pinned remote member's mode-4 row or a
// relocation group's carrier exactly like its closed-branch terms.
inline int obAltRow(const ObScatter& s, const int* qAlt, int e) {
    return s.rowEq == 1 ? qAlt[e * 2 + s.rowEnd] : -1;
}

void collectObAltPattern(int nOpenBranch, const int* openBranchIndices, const int* openBranchAlternateReactiveDestination, const int* relocOwn, std::vector<std::pair<int, int>>& pat,
                         int nQr, const int* qrPtr, const int* qrRows, const int* depGroupOfRow) {
    for (int e = 0; e < nOpenBranch; e++) {
        const int* ix = openBranchIndices + static_cast<size_t>(e) * 6;
        for (int k = 0; k < OB_NOUT; k++) {
            const ObScatter& s = OB_SCATTER[k];
            if (s.role != 1) {
                continue;
            }
            RC rc = resolveOpenBranch(s, ix);
            int origAlt = obAltRow(s, openBranchAlternateReactiveDestination, e);
            int altRow = stockAltRow(rc.row, origAlt, relocOwn);
            int stockSel = stockSelRow(rc.row, origAlt, relocOwn);
            if (rc.col < 0) {
                continue;
            }
            int dg = depGroupOf(rc.row, altRow, relocOwn, depGroupOfRow);
            if (dg >= 0) {                                   // DEPENDENT pilot: every candidate row of ITS group
                for (int c = qrPtr[dg]; c < qrPtr[dg + 1]; c++) {
                    pat.emplace_back(qrRows[c], rc.col);
                }
            }
            if (stockPairOut(rc.row, origAlt, relocOwn)) {   // STOCK PAIR (see stockPairPick): both rows
                pat.emplace_back(relocOwn[origAlt], rc.col);
                pat.emplace_back(origAlt, rc.col);
                continue;
            }
            if (altRow <= -2) {
                int g = -2 - altRow;
                for (int c = qrPtr[g]; c < qrPtr[g + 1]; c++) {
                    pat.emplace_back(qrRows[c], rc.col);
                }
            } else if (altRow >= 0 && altRow != rc.row) {
                pat.emplace_back(altRow, rc.col);
            }
        }
    }
}

void buildObDynDest(int nOpenBranch, const int* openBranchIndices, const int* openBranchAlternateReactiveDestination, const int* relocOwn, const CsrPattern& pos,
                    std::vector<int>& altDest, std::vector<int>& selRow,
                    int nQr, const int* qrPtr, const int* qrRows,
                    std::vector<int>& relocGroup, std::vector<int>& relocOff, std::vector<int>& relocPool,
                    const int* depGroupOfRow, std::vector<char>& relocDep) {
    altDest.assign(static_cast<size_t>(nOpenBranch) * OB_NOUT, -1);
    selRow.assign(static_cast<size_t>(nOpenBranch) * OB_NOUT, -1);
    relocGroup.assign(static_cast<size_t>(nOpenBranch) * OB_NOUT, -1);
    relocOff.assign(static_cast<size_t>(nOpenBranch) * OB_NOUT, -1);
    relocDep.assign(static_cast<size_t>(nOpenBranch) * OB_NOUT, 0);
    for (int e = 0; e < nOpenBranch; e++) {
        const int* ix = openBranchIndices + static_cast<size_t>(e) * 6;
        for (int k = 0; k < OB_NOUT; k++) {
            const ObScatter& s = OB_SCATTER[k];
            RC rc = resolveOpenBranch(s, ix);
            int origAlt = obAltRow(s, openBranchAlternateReactiveDestination, e);
            int altRow = stockAltRow(rc.row, origAlt, relocOwn);
            int stockSel = stockSelRow(rc.row, origAlt, relocOwn);
            size_t o = static_cast<size_t>(e) * OB_NOUT + k;
            if (stockPairOut(rc.row, origAlt, relocOwn)) {
                // THREE destinations (see stockPairOut): the coupler's DUMMY_Q row while an RL pin holds it
                // there (mode 4), NOTHING while the coupler is dead (that row is an identity — the bus is an
                // ordinary group controller again and its DISTR_Q row, diverted to its own row, carries its q),
                // and nothing while it shares. Writing the raw q into the dead coupler's identity row, which
                // the plain two-way rule does on mode 0, corrupts that identity.
                if ((s.role == 1 && rc.col >= 0) || s.role == 0) {
                    relocGroup[o] = origAlt;                 // the selector is the dummy row itself
                    relocOff[o] = -static_cast<int>(relocPool.size()) - 1;
                    int own = relocOwn[origAlt];
                    relocPool.push_back(s.role == 1 ? pos.at(own, rc.col) : -2 - own);
                    relocPool.push_back(s.role == 1 ? pos.at(origAlt, rc.col) : -2 - origAlt);
                }
                continue;
            }
            if (altRow <= -2) {
                int g = -2 - altRow;
                if (s.role == 1 && rc.col >= 0) {
                    relocGroup[o] = g;
                    relocOff[o] = static_cast<int>(relocPool.size());
                    for (int c = qrPtr[g]; c < qrPtr[g + 1]; c++) {
                        relocPool.push_back(pos.at(qrRows[c], rc.col));
                    }
                } else if (s.role == 0) {
                    relocGroup[o] = g;
                    relocOff[o] = static_cast<int>(relocPool.size());
                    for (int c = qrPtr[g]; c < qrPtr[g + 1]; c++) {
                        relocPool.push_back(-2 - qrRows[c]);
                    }
                }
                continue;
            }
            int dg = depGroupOf(rc.row, altRow, relocOwn, depGroupOfRow);
            if (dg >= 0) {                                   // DEPENDENT pilot (depPick): the group's pool, plus the
                if ((s.role == 1 && rc.col >= 0) || s.role == 0) {   // two-way home row of a relocated child
                    relocGroup[o] = dg;
                    relocOff[o] = static_cast<int>(relocPool.size());
                    for (int c = qrPtr[dg]; c < qrPtr[dg + 1]; c++) {
                        relocPool.push_back(s.role == 1 ? pos.at(qrRows[c], rc.col) : -2 - qrRows[c]);
                    }
                    relocDep[o] = 1;
                    if (altRow >= 0 && altRow != rc.row) {
                        altDest[o] = s.role == 1 ? pos.at(altRow, rc.col) : -2 - altRow;
                        selRow[o] = stockSel <= -2 ? stockSel : rc.row;
                    } else {
                        selRow[o] = rc.row;                  // a root: the frozen row IS its own
                    }
                }
                continue;
            }
            if (altRow < 0 || altRow == rc.row) {
                continue;
            }
            if (s.role == 1 && rc.col >= 0) {
                altDest[o] = pos.at(altRow, rc.col);
                selRow[o] = stockSel <= -2 ? stockSel : (rc.row >= 0 ? rc.row : altRow);   // suppressed frozen dest: select on ALT row
            } else if (s.role == 0) {
                altDest[o] = -2 - altRow;
                selRow[o] = stockSel <= -2 ? stockSel : (rc.row >= 0 ? rc.row : altRow);
            }
        }
    }
}

}  // namespace olfgpu
