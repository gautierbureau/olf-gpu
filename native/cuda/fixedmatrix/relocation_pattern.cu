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
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../equations/generated/closed_branch_scatter.h"
#include "../equations/generated/open_branch_scatter.h"
#include "../equations/generated/shunt_scatter.h"
#include "../equations/generated/hvdc_ac_emulation_scatter.h"

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

// ---- The three families whose reactive outputs relocate: closed branch, shunt, open branch ----------------------
// Each family's term outputs resolve to a (row, column) through its generated scatter table, and the role-1
// outputs of a Q-balance row (rowEq == 1) carry an ALTERNATE row the extractor packed: closed branches at
// closedBranchIndices[10 + rowEnd], shunts at shuntIndices[4], open branches in their own array (two per element).
// The traits below are the only family-specific part; the decision of where an output may land is shared
// (placeOutput), so the union pattern (collect) contains every destination the tables (build) look up.

namespace {

struct ClosedBranchFamily {
    using Scatter = ClosedBranchScatter;
    static constexpr int INDEX_STRIDE = 12;
    static constexpr int OUTPUT_COUNT = CLOSED_BRANCH_OUTPUT_COUNT;
    static const Scatter& scatter(int k) { return CLOSED_BRANCH_SCATTER[k]; }
    static RC resolve(const Scatter& s, const int* ix) { return resolveClosedBranch(s, ix); }
    int alternateRow(const Scatter& s, const int* ix, int) const { return s.rowEq == 1 ? ix[10 + s.rowEnd] : -1; }
    bool hasAlternate(const int* ix, int) const { return ix[10] != -1 || ix[11] != -1; }
};

struct ShuntFamily {
    using Scatter = ShuntScatter;
    static constexpr int INDEX_STRIDE = 5;
    static constexpr int OUTPUT_COUNT = SHUNT_OUTPUT_COUNT;
    static const Scatter& scatter(int k) { return SHUNT_SCATTER[k]; }
    static RC resolve(const Scatter& s, const int* ix) { return resolveShunt(s, ix); }
    int alternateRow(const Scatter& s, const int* ix, int) const { return s.rowEq == 1 ? ix[4] : -1; }
    bool hasAlternate(const int* ix, int) const { return ix[4] != -1; }
};

// The connected side's half-open charging q follows the BUS_TARGET_Q row of a remote member switched to PQ or a
// relocation group's carrier exactly like its closed-branch terms.
struct OpenBranchFamily {
    using Scatter = OpenBranchScatter;
    static constexpr int INDEX_STRIDE = 6;
    static constexpr int OUTPUT_COUNT = OPEN_BRANCH_OUTPUT_COUNT;
    static const Scatter& scatter(int k) { return OPEN_BRANCH_SCATTER[k]; }
    static RC resolve(const Scatter& s, const int* ix) { return resolveOpenBranch(s, ix); }
    const int* alternateReactiveDestination;                 // nOpenBranch * 2
    int alternateRow(const Scatter& s, const int*, int e) const {
        return s.rowEq == 1 ? alternateReactiveDestination[e * 2 + s.rowEnd] : -1;
    }
    bool hasAlternate(const int*, int e) const {
        return alternateReactiveDestination[e * 2] != -1 || alternateReactiveDestination[e * 2 + 1] != -1;
    }
};

enum class Placement {
    NONE,             // the output never relocates: the fill always uses its frozen destination
    STOCK_PAIR,       // a coupler-tree child's output: the child's own row or the coupler's DUMMY_Q row (stockPairOut)
    GROUP,            // GROUP relocation (plan-A6): one slot per candidate row, selected per scenario
    DEPENDENT_PILOT,  // a dependent pilot's output (dependentPilotDestination): its group's candidates, plus the
                      // two-way home row of a relocated child
    ALTERNATE_ROW     // plain two-way relocation: the frozen row or the alternate row, selected per scenario
};

// Where one output may land. The cases are tested in this order, the order every table has always been built in.
struct OutputPlacement {
    Placement kind;
    int alternateRow;       // after the stock placement (stockPlacementAlternateRow); <= -2 encodes a group
    int stockSelectorRow;   // stockPlacementSelectorRow
    int group;              // GROUP: the group; DEPENDENT_PILOT: the dependent group
    int dependentGroup;     // the dependent group of the output's home row, whatever the case (-1: none)
};

OutputPlacement placeOutput(int frozenRow, int packedAlternateRow, const RelocationTables& t) {
    OutputPlacement p;
    p.alternateRow = stockPlacementAlternateRow(frozenRow, packedAlternateRow, t.relocatedChildOwnRow);
    p.stockSelectorRow = stockPlacementSelectorRow(frozenRow, packedAlternateRow, t.relocatedChildOwnRow);
    p.dependentGroup = dependentGroupOfHomeRow(frozenRow, p.alternateRow, t.relocatedChildOwnRow, t.dependentGroupOfRow);
    p.group = -1;
    if (stockPairOut(frozenRow, packedAlternateRow, t.relocatedChildOwnRow)) {
        p.kind = Placement::STOCK_PAIR;
    } else if (p.alternateRow <= -2) {
        p.kind = Placement::GROUP;
        p.group = -2 - p.alternateRow;
    } else if (p.dependentGroup >= 0) {
        p.kind = Placement::DEPENDENT_PILOT;
        p.group = p.dependentGroup;
    } else if (p.alternateRow < 0 || p.alternateRow == frozenRow) {
        p.kind = Placement::NONE;
    } else {
        p.kind = Placement::ALTERNATE_ROW;
    }
    return p;
}

bool hasTwoWayAlternate(const OutputPlacement& p, int frozenRow) {
    return p.alternateRow >= 0 && p.alternateRow != frozenRow;
}

// Union the rows a role-1 output may land on into the CSR pattern, so build can resolve a slot on each. Deduped by
// the caller's sort and unique, so only the set matters. A dependent pilot's candidates are added whatever its case.
template <typename F>
void collectRelocationPattern(const F& family, int count, const int* indices, const RelocationTables& t,
                              std::vector<std::pair<int, int>>& pat) {
    auto addGroup = [&](int g, int col) {
        for (int c = t.groupStart[g]; c < t.groupStart[g + 1]; c++) {
            pat.emplace_back(t.candidateRows[c], col);
        }
    };
    for (int e = 0; e < count; e++) {
        const int* ix = indices + static_cast<size_t>(e) * F::INDEX_STRIDE;
        for (int k = 0; k < F::OUTPUT_COUNT; k++) {
            const typename F::Scatter& s = F::scatter(k);
            if (s.role != 1) {
                continue;
            }
            RC rc = F::resolve(s, ix);
            if (rc.col < 0) {
                continue;
            }
            int packedAlternateRow = family.alternateRow(s, ix, e);
            OutputPlacement p = placeOutput(rc.row, packedAlternateRow, t);
            if (p.dependentGroup >= 0) {
                addGroup(p.dependentGroup, rc.col);
            }
            if (p.kind == Placement::STOCK_PAIR) {
                pat.emplace_back(t.relocatedChildOwnRow[packedAlternateRow], rc.col);
                pat.emplace_back(packedAlternateRow, rc.col);
            } else if (p.kind == Placement::GROUP) {
                addGroup(p.group, rc.col);
            } else if (hasTwoWayAlternate(p, rc.row)) {
                pat.emplace_back(p.alternateRow, rc.col);
            }
        }
    }
}

// Per (element, output): the alternate destination and the row its per-scenario mode is tested on (selectorRow; -1:
// never relocates, the fill uses the frozen destination), and for the outputs with more than two destinations the
// relocation group, the offset of their candidate destinations in relocationCandidateDestinations (negative: a stock
// pair, -offset - 1) and the dependent-pilot flag. A destination is a CSR slot for a role-1 Jacobian entry and
// -2 - row for a role-0 residual. Sized count * OUTPUT_COUNT like the frozen destination, so the fill indexes them
// alike; candidates are appended in output order, which the offsets index into.
template <typename F>
void buildRelocationDestination(const F& family, int count, const int* indices, const RelocationTables& t, const CsrPattern& pos,
                                std::vector<int>& alternateDestination, std::vector<int>& selectorRow,
                                std::vector<int>& relocationGroup, std::vector<int>& relocationCandidateOffset,
                                std::vector<int>& relocationCandidateDestinations, std::vector<char>& relocationDependentPilot) {
    size_t outputs = static_cast<size_t>(count) * F::OUTPUT_COUNT;
    alternateDestination.assign(outputs, -1);
    selectorRow.assign(outputs, -1);
    relocationGroup.assign(outputs, -1);
    relocationCandidateOffset.assign(outputs, -1);
    relocationDependentPilot.assign(outputs, 0);
    for (int e = 0; e < count; e++) {
        const int* ix = indices + static_cast<size_t>(e) * F::INDEX_STRIDE;
        for (int k = 0; k < F::OUTPUT_COUNT; k++) {
            const typename F::Scatter& s = F::scatter(k);
            RC rc = F::resolve(s, ix);
            // A Jacobian entry needs a column to land in; a residual needs only a row.
            if (!((s.role == 1 && rc.col >= 0) || s.role == 0)) {
                continue;
            }
            auto destination = [&](int row) { return s.role == 1 ? pos.at(row, rc.col) : -2 - row; };
            auto addGroup = [&](int g) {
                for (int c = t.groupStart[g]; c < t.groupStart[g + 1]; c++) {
                    relocationCandidateDestinations.push_back(destination(t.candidateRows[c]));
                }
            };
            int packedAlternateRow = family.alternateRow(s, ix, e);
            OutputPlacement p = placeOutput(rc.row, packedAlternateRow, t);
            size_t o = static_cast<size_t>(e) * F::OUTPUT_COUNT + k;
            switch (p.kind) {
                case Placement::NONE:
                    break;
                case Placement::STOCK_PAIR:
                    // THREE destinations (see stockPairOut): the coupler's DUMMY_Q row while an RL switch to PQ holds it
                    // there (BUS_TARGET_Q), NOTHING while the coupler is dead (that row is an identity — the bus is an
                    // ordinary group controller again and its DISTR_Q row, diverted to its own row, carries its q),
                    // and nothing while it shares. Writing the raw q into the dead coupler's identity row, which
                    // the plain two-way rule does on FIXED_VARIABLE, corrupts that identity.
                    relocationGroup[o] = packedAlternateRow;                 // the selector is the dummy row itself
                    relocationCandidateOffset[o] = -static_cast<int>(relocationCandidateDestinations.size()) - 1;
                    relocationCandidateDestinations.push_back(destination(t.relocatedChildOwnRow[packedAlternateRow]));
                    relocationCandidateDestinations.push_back(destination(packedAlternateRow));
                    break;
                case Placement::GROUP:                                       // one slot per candidate, selected per
                    relocationGroup[o] = p.group;                            // scenario by the first-valid-member mode scan
                    relocationCandidateOffset[o] = static_cast<int>(relocationCandidateDestinations.size());
                    addGroup(p.group);
                    break;
                case Placement::DEPENDENT_PILOT:
                    relocationGroup[o] = p.group;
                    relocationCandidateOffset[o] = static_cast<int>(relocationCandidateDestinations.size());
                    addGroup(p.group);
                    relocationDependentPilot[o] = 1;
                    if (hasTwoWayAlternate(p, rc.row)) {
                        alternateDestination[o] = destination(p.alternateRow);
                        selectorRow[o] = p.stockSelectorRow <= -2 ? p.stockSelectorRow : rc.row;
                    } else {
                        selectorRow[o] = rc.row;                             // a root: the frozen row IS its own
                    }
                    break;
                case Placement::ALTERNATE_ROW:                               // a suppressed frozen destination (remote
                    alternateDestination[o] = destination(p.alternateRow);   // member) selects on the alternate row itself
                    selectorRow[o] = p.stockSelectorRow <= -2 ? p.stockSelectorRow : (rc.row >= 0 ? rc.row : p.alternateRow);
                    break;
            }
        }
    }
}

// plan-B4: the FROZEN role-0 dest of a GROUP-relocated output backs only the qBuf DIAGNOSTIC
// fill (dynamicRelocation = 0; every Newton fill selects a pool slot instead) — point it at the group's
// FALLBACK row (the pilot's own row, read by no outer loop) so a remote member's diagnostic q
// row carries ONLY its own injection (fillRemoteMemberQBatch). Without this the pilot's whole
// balance lands on the base carrier's row and corrupts the free member's reactive-limit check.
template <typename F>
void patchGroupFrozenDestinations(DeviceBuf& dest, const std::vector<int>& relocationGroup,
                                  const std::vector<int>& relocationCandidateOffset, const RelocationTables& t) {
    for (size_t o = 0; o < relocationGroup.size(); o++) {
        int g = relocationGroup[o];
        // A STOCK PAIR (negative offset) parks the DUMMY ROW in the group slot, not a group id — reading
        // t.groupStart[row] there walked off the candidate table and wrote a garbage dest, which the device then
        // used: "CUDA error: an illegal memory access" out of createBatchContext.
        if (g >= 0 && relocationCandidateOffset[o] < 0) {
            continue;
        }
        if (g >= 0 && F::scatter(static_cast<int>(o % F::OUTPUT_COUNT)).role == 0) {
            int v = -2 - t.candidateRows[t.groupStart[g]];
            CUDA_CHECK(cudaMemcpy(dest.as<int>() + o, &v, sizeof(int), cudaMemcpyHostToDevice));
        }
    }
}

// Calls fn with the traits of the elements' family.
template <typename Fn>
auto withFamily(const RelocatedElements& elements, Fn fn) {
    switch (elements.family) {
        case RelocatedFamily::CLOSED_BRANCH:
            return fn(ClosedBranchFamily{});
        case RelocatedFamily::SHUNT:
            return fn(ShuntFamily{});
        case RelocatedFamily::OPEN_BRANCH:
        default:
            return fn(OpenBranchFamily{elements.alternateReactiveDestination});
    }
}

}  // namespace

bool hasRelocatableOutput(const RelocatedElements& elements) {
    return withFamily(elements, [&](const auto& family) {
        for (int e = 0; e < elements.count; e++) {
            if (family.hasAlternate(elements.indices + static_cast<size_t>(e) * family.INDEX_STRIDE, e)) {
                return true;                                 // a plain alternate row, or the -2-groupId encoding (plan-A6)
            }
        }
        return false;
    });
}

void collectRelocationPattern(const RelocatedElements& elements, const RelocationTables& tables, std::vector<std::pair<int, int>>& pat) {
    withFamily(elements, [&](const auto& family) {
        collectRelocationPattern(family, elements.count, elements.indices, tables, pat);
    });
}

RelocationDestinations installRelocationDestinations(const RelocatedElements& elements, const RelocationTables& tables,
                                                     const CsrPattern& pos, Family& target) {
    RelocationDestinations d;
    withFamily(elements, [&](const auto& family) {
        using F = std::decay_t<decltype(family)>;
        buildRelocationDestination(family, elements.count, elements.indices, tables, pos, d.alternateDestination, d.selectorRow,
                                   d.relocationGroup, d.relocationCandidateOffset, d.relocationCandidateDestinations,
                                   d.relocationDependentPilot);
        target.alternateDestination.upload(d.alternateDestination.data(), d.alternateDestination.size() * sizeof(int));
        target.selectorRow.upload(d.selectorRow.data(), d.selectorRow.size() * sizeof(int));
        if (!d.relocationCandidateDestinations.empty()) {
            target.relocationGroup.upload(d.relocationGroup.data(), d.relocationGroup.size() * sizeof(int));
            target.relocationCandidateOffset.upload(d.relocationCandidateOffset.data(), d.relocationCandidateOffset.size() * sizeof(int));
            target.relocationCandidateDestinations.upload(d.relocationCandidateDestinations.data(),
                                                          d.relocationCandidateDestinations.size() * sizeof(int));
            target.relocationDependentPilot.upload(d.relocationDependentPilot.data(), d.relocationDependentPilot.size());
            patchGroupFrozenDestinations<F>(target.destination, d.relocationGroup, d.relocationCandidateOffset, tables);
        }
        target.dynamicRelocation = true;
    });
    return d;
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
        for (int k = 0; k < HVDC_AC_EMULATION_OUTPUT_COUNT; k++) {
            RC rc = resolveHvdcAcEmulation(HVDC_AC_EMULATION_SCATTER[k], ix);
            if (HVDC_AC_EMULATION_SCATTER[k].role == 1 && rc.row >= 0 && rc.col >= 0 && relocatedChildOwnRow[rc.row] >= 0) {
                pat.emplace_back(relocatedChildOwnRow[rc.row], rc.col);
            }
        }
    }
}

void buildHvdcAcEmulationRelocationDestination(int nHvdcAcEmulation, const int* hvdcAcEmulationIndices, const int* relocatedChildOwnRow, const CsrPattern& pos,
                    std::vector<int>& alternateDestination, std::vector<int>& selectorRow) {
    alternateDestination.assign(static_cast<size_t>(nHvdcAcEmulation) * HVDC_AC_EMULATION_OUTPUT_COUNT, -1);
    selectorRow.assign(static_cast<size_t>(nHvdcAcEmulation) * HVDC_AC_EMULATION_OUTPUT_COUNT, -1);
    if (relocatedChildOwnRow == nullptr) {
        return;
    }
    for (int e = 0; e < nHvdcAcEmulation; e++) {
        const int* ix = hvdcAcEmulationIndices + static_cast<size_t>(e) * 4;
        for (int k = 0; k < HVDC_AC_EMULATION_OUTPUT_COUNT; k++) {
            const HvdcAcEmulationScatter& sc = HVDC_AC_EMULATION_SCATTER[k];
            RC rc = resolveHvdcAcEmulation(sc, ix);
            if (rc.row < 0 || relocatedChildOwnRow[rc.row] < 0) {
                continue;
            }
            int own = relocatedChildOwnRow[rc.row];
            size_t o = static_cast<size_t>(e) * HVDC_AC_EMULATION_OUTPUT_COUNT + k;
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

}  // namespace olfgpu
