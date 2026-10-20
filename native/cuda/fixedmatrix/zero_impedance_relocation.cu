// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// docs/gpu-vs-olf.md section 2.3 (equations placed on another bus's row), couplers: the per-scenario destinations of
// one zero-impedance coupler's dummy terms (buildZeroImpedanceBranchDestination keeps the OLF couplings and ZERO_V /
// ZERO_PHI). See zero_impedance_relocation.cuh.
#include <cstddef>
#include <vector>

#include "../structure/structure_build.cuh"
#include "stock_placement.cuh"
#include "zero_impedance_relocation.cuh"

namespace olfgpu {

// Element e (indices z, its F/J destinations dd, its endpoint rows b1P / b2P / b1Q / b2Q): the q slots of a GROUP
// relocation (one per candidate), a STOCK PAIR, a DEPENDENT pilot or a two-way alternate, then the COUPLER-TREE
// stock-placement fallback of its P and Q terms. Slot layout: see buildZeroImpedanceBranchDestination.
void buildZeroImpedanceRelocationDestinations(const SharedStructure* structure, const CsrPattern& pos, int e, const int* z, int b1P, int b2P, int b1Q, int b2Q, int* dd,
                                              const int* zeroImpedanceBranchAlternateReactiveDestination, const int* relocatedChildOwnRow,
                                              const int* reactiveRelocationDependentGroupOfRow, std::vector<int>& zeroImpedanceAlternateDestination,
                                              std::vector<int>& zeroImpedanceSelectorRow, std::vector<int>& zeroImpedanceRelocationGroup,
                                              std::vector<int>& relocationCandidateOffset, std::vector<int>& zeroImpedanceRelocationCandidateDestinations,
                                              std::vector<char>& zeroImpedanceRelocationDependentPilot, bool& anyDynamicRelocation) {
    int frozenQ[2] = {b1Q, b2Q};
    for (int side = 0; side < 2; side++) {
        int alternateRow = zeroImpedanceBranchAlternateReactiveDestination[e * 2 + side];
        int fSlot = static_cast<int>(e) * 8 + side;
        int jSlot = static_cast<int>(e) * 8 + 2 + side;
        if (alternateRow <= -2) {                          // GROUP relocation: one slot per candidate
            int g = -2 - alternateRow;
            zeroImpedanceRelocationGroup[fSlot] = g;
            relocationCandidateOffset[fSlot] = static_cast<int>(zeroImpedanceRelocationCandidateDestinations.size());
            for (int c = structure->hReactiveRelocationGroupStart[g]; c < structure->hReactiveRelocationGroupStart[g + 1]; c++) {
                zeroImpedanceRelocationCandidateDestinations.push_back(-2 - structure->hReactiveRelocationCandidateRows[c]);
            }
            zeroImpedanceRelocationGroup[jSlot] = g;
            relocationCandidateOffset[jSlot] = static_cast<int>(zeroImpedanceRelocationCandidateDestinations.size());
            for (int c = structure->hReactiveRelocationGroupStart[g]; c < structure->hReactiveRelocationGroupStart[g + 1]; c++) {
                zeroImpedanceRelocationCandidateDestinations.push_back(pos.at(structure->hReactiveRelocationCandidateRows[c], z[5]));
            }
            dd[2 + side] = -2 - structure->hReactiveRelocationCandidateRows[structure->hReactiveRelocationGroupStart[g]];   // frozen F -> fallback (qBuf)
            anyDynamicRelocation = true;
        } else if (stockPairOut(frozenQ[side], alternateRow, relocatedChildOwnRow)) {
            // STOCK PAIR for this endpoint's +-dummyQ, exactly as the cb/sh raw-q families encode it
            // (stockPairOut / stockPairPick): a DISTR_Q controller whose own Q balance is SUPPRESSED
            // and relocated onto its PARENT coupler's dummy row has three destinations for this
            // coupler's dummy term — the parent's DUMMY_Q row while a pin holds its balance there
            // (mode 4), its OWN row while the parent coupler is dead this scenario (the dummy row is
            // an identity and the balance is back home), nothing while it shares. The plain two-way
            // alternate selected on the dummy row sent the term INTO the dead coupler's identity row
            // (wiped) and the bus's own reactive balance lost its live coupler's dummy: rte6515
            // BUS-1359_BBS_1 / BUS-1869_BBS_1 (pilot section lost, two remote controllers survive
            // on a coupler chain) — OLF's bus_target_q[VL-1359_1] carries +dummy_q(COUPLER_2), the
            // device's row did not, 0.465 / 0.338 kV and +-2 violations in every run.
            int own = relocatedChildOwnRow[alternateRow];
            zeroImpedanceRelocationGroup[fSlot] = alternateRow;                   // the selector is the dummy row itself
            relocationCandidateOffset[fSlot] = -static_cast<int>(zeroImpedanceRelocationCandidateDestinations.size()) - 1;
            zeroImpedanceRelocationCandidateDestinations.push_back(-2 - own);
            zeroImpedanceRelocationCandidateDestinations.push_back(-2 - alternateRow);
            zeroImpedanceRelocationGroup[jSlot] = alternateRow;
            relocationCandidateOffset[jSlot] = -static_cast<int>(zeroImpedanceRelocationCandidateDestinations.size()) - 1;
            zeroImpedanceRelocationCandidateDestinations.push_back(pos.at(own, z[5]));
            zeroImpedanceRelocationCandidateDestinations.push_back(pos.at(alternateRow, z[5]));
            anyDynamicRelocation = true;
        } else if (dependentGroupOfHomeRow(frozenQ[side], stockPlacementAlternateRow(frozenQ[side], alternateRow, relocatedChildOwnRow), relocatedChildOwnRow, reactiveRelocationDependentGroupOfRow) >= 0) {
            // DEPENDENT pilot (dependentPilotDestination): the group's pool for this +-dummyQ, plus the two-way home row
            int own = stockPlacementAlternateRow(frozenQ[side], alternateRow, relocatedChildOwnRow);
            int dependentGroup = dependentGroupOfHomeRow(frozenQ[side], own, relocatedChildOwnRow, reactiveRelocationDependentGroupOfRow);
            zeroImpedanceRelocationGroup[fSlot] = dependentGroup;
            relocationCandidateOffset[fSlot] = static_cast<int>(zeroImpedanceRelocationCandidateDestinations.size());
            for (int c = structure->hReactiveRelocationGroupStart[dependentGroup]; c < structure->hReactiveRelocationGroupStart[dependentGroup + 1]; c++) {
                zeroImpedanceRelocationCandidateDestinations.push_back(-2 - structure->hReactiveRelocationCandidateRows[c]);
            }
            zeroImpedanceRelocationGroup[jSlot] = dependentGroup;
            relocationCandidateOffset[jSlot] = static_cast<int>(zeroImpedanceRelocationCandidateDestinations.size());
            for (int c = structure->hReactiveRelocationGroupStart[dependentGroup]; c < structure->hReactiveRelocationGroupStart[dependentGroup + 1]; c++) {
                zeroImpedanceRelocationCandidateDestinations.push_back(pos.at(structure->hReactiveRelocationCandidateRows[c], z[5]));
            }
            zeroImpedanceRelocationDependentPilot[fSlot] = 1;
            zeroImpedanceRelocationDependentPilot[jSlot] = 1;
            if (own >= 0 && own != frozenQ[side]) {
                zeroImpedanceAlternateDestination[fSlot] = -2 - own;
                zeroImpedanceAlternateDestination[jSlot] = pos.at(own, z[5]);
                zeroImpedanceSelectorRow[fSlot] = -2 - frozenQ[side];
                zeroImpedanceSelectorRow[jSlot] = -2 - frozenQ[side];
            } else {
                zeroImpedanceSelectorRow[fSlot] = frozenQ[side];
                zeroImpedanceSelectorRow[jSlot] = frozenQ[side];
            }
            anyDynamicRelocation = true;
        } else if (alternateRow >= 0 && alternateRow != frozenQ[side]) {
            zeroImpedanceAlternateDestination[fSlot] = -2 - alternateRow;
            zeroImpedanceAlternateDestination[jSlot] = pos.at(alternateRow, z[5]);
            int sel = frozenQ[side] >= 0 ? frozenQ[side] : alternateRow;
            zeroImpedanceSelectorRow[fSlot] = sel;                      // suppressed frozen dest: select on ALT row
            zeroImpedanceSelectorRow[jSlot] = sel;
            anyDynamicRelocation = true;
        }
    }
    // STOCK-PLACEMENT fallback (see stockPlacementAlternateRow): when this element's own +-dummy coupling lands on a
    // row that is itself a RELOCATED bus equation's home (its parent coupler's dummy row), give it the
    // bus's own row as the alternate, selected by -2-row so it diverts only while that row is an
    // identity — i.e. only in the scenarios that OPEN the parent coupler.
    if (relocatedChildOwnRow != nullptr) {
        int frozenP[2] = {b1P, b2P};
        for (int side = 0; side < 2; side++) {
            int pRow = frozenP[side];
            if (pRow >= 0 && relocatedChildOwnRow[pRow] >= 0) {
                zeroImpedanceAlternateDestination[static_cast<size_t>(e) * 8 + 4 + side] = -2 - relocatedChildOwnRow[pRow];
                zeroImpedanceAlternateDestination[static_cast<size_t>(e) * 8 + 6 + side] = pos.at(relocatedChildOwnRow[pRow], z[4]);
                zeroImpedanceSelectorRow[static_cast<size_t>(e) * 8 + 4 + side] = -2 - pRow;
                zeroImpedanceSelectorRow[static_cast<size_t>(e) * 8 + 6 + side] = -2 - pRow;
                anyDynamicRelocation = true;
            }
            int qRow = frozenQ[side];
            if (qRow >= 0 && relocatedChildOwnRow[qRow] >= 0 && zeroImpedanceSelectorRow[static_cast<size_t>(e) * 8 + side] < 0) {
                zeroImpedanceAlternateDestination[static_cast<size_t>(e) * 8 + side] = -2 - relocatedChildOwnRow[qRow];
                zeroImpedanceAlternateDestination[static_cast<size_t>(e) * 8 + 2 + side] = pos.at(relocatedChildOwnRow[qRow], z[5]);
                zeroImpedanceSelectorRow[static_cast<size_t>(e) * 8 + side] = -2 - qRow;
                zeroImpedanceSelectorRow[static_cast<size_t>(e) * 8 + 2 + side] = -2 - qRow;
                anyDynamicRelocation = true;
            }
        }
    }
}

}  // namespace olfgpu
