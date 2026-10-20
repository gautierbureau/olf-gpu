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
#include "../fixedmatrix/zero_impedance_relocation.cuh"

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
            buildZeroImpedanceRelocationDestinations(structure, pos, e, z, b1P, b2P, b1Q, b2Q, dd, zeroImpedanceBranchAlternateReactiveDestination, relocatedChildOwnRow,
                                                     reactiveRelocationDependentGroupOfRow, zeroImpedanceAlternateDestination, zeroImpedanceSelectorRow,
                                                     zeroImpedanceRelocationGroup, relocationCandidateOffset, zeroImpedanceRelocationCandidateDestinations,
                                                     zeroImpedanceRelocationDependentPilot, anyDynamicRelocation);
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
