// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Destinations of the zero-impedance couplers: the dummy flows into both buses' balances and, on a
// spanning-tree edge, ZERO_V / ZERO_PHI on the rows the stock or relocated placement picks.
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

void buildZeroImpedanceBranchDestination(SharedStructure* structure, const CsrPattern& pos, const std::vector<int>& rowMap,
                      int nZeroImpedanceBranch, const double* zeroImpedanceBranchParameters, const int* zeroImpedanceBranchIndices, const int* zeroImpedanceBranchAlternateReactiveDestination,
                      const int* relocatedChildOwnRow, const int* reactiveRelocationDependentGroupOfRow) {
    structure->nZeroImpedanceBranch = nZeroImpedanceBranch;
    if (nZeroImpedanceBranch > 0) {
        std::vector<int> zeroImpedanceBranchDestination(static_cast<size_t>(nZeroImpedanceBranch) * 14, -1);
        // zero-impedance dynamic Q-routing (ob/zero-impedance channels): 4 q slots per element {F side1, F side2, J side1,
        // J side2} with the same alt/select/group encodings as the cb/sh families. A GROUP side's
        // FROZEN F dest is re-pointed at the fallback row (the pilot's own row) so the qBuf
        // diagnostic never lands the pilot's dummy-Q on a member row.
        // 8 slots per element: Q at 0..3 {F side1, F side2, J side1, J side2} exactly as before, and P at
        // 4..7 with the same meaning. The P half exists only for the COUPLER-TREE stock-placement fallback —
        // a coupler in a CHAIN emits its +-dummyP into its parent's DUMMY_P row, so opening the parent would
        // strand it (three of the four surviving divergences open two relocated couplers).
        std::vector<int> zeroImpedanceAlternateDestination(static_cast<size_t>(nZeroImpedanceBranch) * 8, -1);
        std::vector<int> zeroImpedanceSelectorRow(static_cast<size_t>(nZeroImpedanceBranch) * 8, -1);
        std::vector<int> zeroImpedanceRelocationGroup(static_cast<size_t>(nZeroImpedanceBranch) * 8, -1);
        std::vector<int> relocationCandidateOffset(static_cast<size_t>(nZeroImpedanceBranch) * 8, -1);
        std::vector<int> zeroImpedanceRelocationCandidateDestinations;
        std::vector<char> zeroImpedanceRelocationDependentPilot(static_cast<size_t>(nZeroImpedanceBranch) * 8, 0);
        bool anyDynamicRelocation = false;
        for (int e = 0; e < nZeroImpedanceBranch; e++) {
            const int* z = zeroImpedanceBranchIndices + static_cast<size_t>(e) * 11;
            // -1 Q row = a DISTR_Q controller whose reactive balance is suppressed (its dummyQ joins the
            // DISTR_Q rows via kind 3, not its own Q balance) — skip the coupling there.
            int b1P = rowMap[z[0]], b2P = rowMap[z[2]];
            int b1Q = z[1] >= 0 ? rowMap[z[1]] : -1, b2Q = z[3] >= 0 ? rowMap[z[3]] : -1;
            int* dd = zeroImpedanceBranchDestination.data() + static_cast<size_t>(e) * 14;
            dd[0] = -2 - b1P;                                // F bus1 P (+dummyP)
            dd[1] = -2 - b2P;                                // F bus2 P (-dummyP)
            dd[2] = b1Q >= 0 ? -2 - b1Q : -1;                // F bus1 Q (+dummyQ), skip if suppressed
            dd[3] = b2Q >= 0 ? -2 - b2Q : -1;                // F bus2 Q (-dummyQ), skip if suppressed
            dd[4] = pos.at(b1P, z[4]);                       // J(bus1 P, dummyP)
            dd[5] = pos.at(b2P, z[4]);                       // J(bus2 P, dummyP)
            dd[6] = b1Q >= 0 ? pos.at(b1Q, z[5]) : -1;       // J(bus1 Q, dummyQ)
            dd[7] = b2Q >= 0 ? pos.at(b2Q, z[5]) : -1;       // J(bus2 Q, dummyQ)
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
            {                                                // every coupler: the tree flag is per scenario
                int zvRow = rowMap[zeroVRowOf(z)], zpRow = rowMap[zeroPhiRowOf(z)];
                dd[8] = -2 - zvRow;                          // F ZERO_V row
                dd[9] = pos.at(zvRow, z[6]);                 // J(ZERO_V, v1)
                dd[10] = pos.at(zvRow, z[7]);                // J(ZERO_V, v2)
                dd[11] = -2 - zpRow;                         // F ZERO_PHI row
                dd[12] = pos.at(zpRow, z[8]);                // J(ZERO_PHI, ph1)
                dd[13] = pos.at(zpRow, z[9]);                // J(ZERO_PHI, ph2)
            }
        }
        structure->dZeroImpedanceBranchParameters.upload(zeroImpedanceBranchParameters, nZeroImpedanceBranch * sizeof(double));
        structure->dZeroImpedanceBranchIndices.upload(zeroImpedanceBranchIndices, static_cast<size_t>(nZeroImpedanceBranch) * 11 * sizeof(int));
        structure->dZeroImpedanceBranchDestination.upload(zeroImpedanceBranchDestination.data(), zeroImpedanceBranchDestination.size() * sizeof(int));
        if (anyDynamicRelocation) {
            structure->dZeroImpedanceAlternateDestination.upload(zeroImpedanceAlternateDestination.data(), zeroImpedanceAlternateDestination.size() * sizeof(int));
            structure->dZeroImpedanceSelectorRow.upload(zeroImpedanceSelectorRow.data(), zeroImpedanceSelectorRow.size() * sizeof(int));
            if (!zeroImpedanceRelocationCandidateDestinations.empty()) {
                structure->dZeroImpedanceRelocationGroup.upload(zeroImpedanceRelocationGroup.data(), zeroImpedanceRelocationGroup.size() * sizeof(int));
                structure->dZeroImpedanceRelocationCandidateOffset.upload(relocationCandidateOffset.data(), relocationCandidateOffset.size() * sizeof(int));
                structure->dZeroImpedanceRelocationCandidateDestinations.upload(zeroImpedanceRelocationCandidateDestinations.data(), zeroImpedanceRelocationCandidateDestinations.size() * sizeof(int));
                structure->dZeroImpedanceRelocationDependentPilot.upload(zeroImpedanceRelocationDependentPilot.data(), zeroImpedanceRelocationDependentPilot.size());
            }
            structure->zeroImpedanceDynamicRelocation = true;
        }
    }
}

}  // namespace olfgpu
