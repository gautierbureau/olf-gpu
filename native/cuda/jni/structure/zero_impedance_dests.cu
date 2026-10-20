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

void buildZeroImpDest(SharedStructure* st, const CsrPattern& pos, const std::vector<int>& rowMap,
                      int nZeroImpedanceBranch, const double* zeroImpedanceBranchParameters, const int* zeroImpedanceBranchIndices, const int* zeroImpedanceBranchAlternateReactiveDestination,
                      const int* relocOwn, const int* depGroupOfRow) {
    st->nZeroImpedanceBranch = nZeroImpedanceBranch;
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
        std::vector<int> ziAlt(static_cast<size_t>(nZeroImpedanceBranch) * 8, -1);
        std::vector<int> ziSel(static_cast<size_t>(nZeroImpedanceBranch) * 8, -1);
        std::vector<int> ziGrp(static_cast<size_t>(nZeroImpedanceBranch) * 8, -1);
        std::vector<int> relocationCandidateOffset(static_cast<size_t>(nZeroImpedanceBranch) * 8, -1);
        std::vector<int> ziPool;
        std::vector<char> ziDep(static_cast<size_t>(nZeroImpedanceBranch) * 8, 0);
        bool anyDyn = false;
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
                int altRow = zeroImpedanceBranchAlternateReactiveDestination[e * 2 + side];
                int fSlot = static_cast<int>(e) * 8 + side;
                int jSlot = static_cast<int>(e) * 8 + 2 + side;
                if (altRow <= -2) {                          // GROUP relocation: one slot per candidate
                    int g = -2 - altRow;
                    ziGrp[fSlot] = g;
                    relocationCandidateOffset[fSlot] = static_cast<int>(ziPool.size());
                    for (int c = st->hQrPtr[g]; c < st->hQrPtr[g + 1]; c++) {
                        ziPool.push_back(-2 - st->hQrRows[c]);
                    }
                    ziGrp[jSlot] = g;
                    relocationCandidateOffset[jSlot] = static_cast<int>(ziPool.size());
                    for (int c = st->hQrPtr[g]; c < st->hQrPtr[g + 1]; c++) {
                        ziPool.push_back(pos.at(st->hQrRows[c], z[5]));
                    }
                    dd[2 + side] = -2 - st->hQrRows[st->hQrPtr[g]];   // frozen F -> fallback (qBuf)
                    anyDyn = true;
                } else if (stockPairOut(frozenQ[side], altRow, relocOwn)) {
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
                    int own = relocOwn[altRow];
                    ziGrp[fSlot] = altRow;                   // the selector is the dummy row itself
                    relocationCandidateOffset[fSlot] = -static_cast<int>(ziPool.size()) - 1;
                    ziPool.push_back(-2 - own);
                    ziPool.push_back(-2 - altRow);
                    ziGrp[jSlot] = altRow;
                    relocationCandidateOffset[jSlot] = -static_cast<int>(ziPool.size()) - 1;
                    ziPool.push_back(pos.at(own, z[5]));
                    ziPool.push_back(pos.at(altRow, z[5]));
                    anyDyn = true;
                } else if (depGroupOf(frozenQ[side], stockAltRow(frozenQ[side], altRow, relocOwn), relocOwn, depGroupOfRow) >= 0) {
                    // DEPENDENT pilot (depPick): the group's pool for this +-dummyQ, plus the two-way home row
                    int own = stockAltRow(frozenQ[side], altRow, relocOwn);
                    int dg = depGroupOf(frozenQ[side], own, relocOwn, depGroupOfRow);
                    ziGrp[fSlot] = dg;
                    relocationCandidateOffset[fSlot] = static_cast<int>(ziPool.size());
                    for (int c = st->hQrPtr[dg]; c < st->hQrPtr[dg + 1]; c++) {
                        ziPool.push_back(-2 - st->hQrRows[c]);
                    }
                    ziGrp[jSlot] = dg;
                    relocationCandidateOffset[jSlot] = static_cast<int>(ziPool.size());
                    for (int c = st->hQrPtr[dg]; c < st->hQrPtr[dg + 1]; c++) {
                        ziPool.push_back(pos.at(st->hQrRows[c], z[5]));
                    }
                    ziDep[fSlot] = 1;
                    ziDep[jSlot] = 1;
                    if (own >= 0 && own != frozenQ[side]) {
                        ziAlt[fSlot] = -2 - own;
                        ziAlt[jSlot] = pos.at(own, z[5]);
                        ziSel[fSlot] = -2 - frozenQ[side];
                        ziSel[jSlot] = -2 - frozenQ[side];
                    } else {
                        ziSel[fSlot] = frozenQ[side];
                        ziSel[jSlot] = frozenQ[side];
                    }
                    anyDyn = true;
                } else if (altRow >= 0 && altRow != frozenQ[side]) {
                    ziAlt[fSlot] = -2 - altRow;
                    ziAlt[jSlot] = pos.at(altRow, z[5]);
                    int sel = frozenQ[side] >= 0 ? frozenQ[side] : altRow;
                    ziSel[fSlot] = sel;                      // suppressed frozen dest: select on ALT row
                    ziSel[jSlot] = sel;
                    anyDyn = true;
                }
            }
            // STOCK-PLACEMENT fallback (see stockAltRow): when this element's own +-dummy coupling lands on a
            // row that is itself a RELOCATED bus equation's home (its parent coupler's dummy row), give it the
            // bus's own row as the alternate, selected by -2-row so it diverts only while that row is an
            // identity — i.e. only in the scenarios that OPEN the parent coupler.
            if (relocOwn != nullptr) {
                int frozenP[2] = {b1P, b2P};
                for (int side = 0; side < 2; side++) {
                    int pRow = frozenP[side];
                    if (pRow >= 0 && relocOwn[pRow] >= 0) {
                        ziAlt[static_cast<size_t>(e) * 8 + 4 + side] = -2 - relocOwn[pRow];
                        ziAlt[static_cast<size_t>(e) * 8 + 6 + side] = pos.at(relocOwn[pRow], z[4]);
                        ziSel[static_cast<size_t>(e) * 8 + 4 + side] = -2 - pRow;
                        ziSel[static_cast<size_t>(e) * 8 + 6 + side] = -2 - pRow;
                        anyDyn = true;
                    }
                    int qRow = frozenQ[side];
                    if (qRow >= 0 && relocOwn[qRow] >= 0 && ziSel[static_cast<size_t>(e) * 8 + side] < 0) {
                        ziAlt[static_cast<size_t>(e) * 8 + side] = -2 - relocOwn[qRow];
                        ziAlt[static_cast<size_t>(e) * 8 + 2 + side] = pos.at(relocOwn[qRow], z[5]);
                        ziSel[static_cast<size_t>(e) * 8 + side] = -2 - qRow;
                        ziSel[static_cast<size_t>(e) * 8 + 2 + side] = -2 - qRow;
                        anyDyn = true;
                    }
                }
            }
            {                                                // every coupler: the tree flag is per scenario
                int zvRow = rowMap[ziZeroVRow(z)], zpRow = rowMap[ziZeroPhiRow(z)];
                dd[8] = -2 - zvRow;                          // F ZERO_V row
                dd[9] = pos.at(zvRow, z[6]);                 // J(ZERO_V, v1)
                dd[10] = pos.at(zvRow, z[7]);                // J(ZERO_V, v2)
                dd[11] = -2 - zpRow;                         // F ZERO_PHI row
                dd[12] = pos.at(zpRow, z[8]);                // J(ZERO_PHI, ph1)
                dd[13] = pos.at(zpRow, z[9]);                // J(ZERO_PHI, ph2)
            }
        }
        st->dZeroImpedanceBranchParameters.upload(zeroImpedanceBranchParameters, nZeroImpedanceBranch * sizeof(double));
        st->dZeroImpedanceBranchIndices.upload(zeroImpedanceBranchIndices, static_cast<size_t>(nZeroImpedanceBranch) * 11 * sizeof(int));
        st->dZeroImpedanceBranchDestination.upload(zeroImpedanceBranchDestination.data(), zeroImpedanceBranchDestination.size() * sizeof(int));
        if (anyDyn) {
            st->dZiAltDest.upload(ziAlt.data(), ziAlt.size() * sizeof(int));
            st->dZiSelRow.upload(ziSel.data(), ziSel.size() * sizeof(int));
            if (!ziPool.empty()) {
                st->dZiRelocGroup.upload(ziGrp.data(), ziGrp.size() * sizeof(int));
                st->dZiRelocOff.upload(relocationCandidateOffset.data(), relocationCandidateOffset.size() * sizeof(int));
                st->dZiRelocPool.upload(ziPool.data(), ziPool.size() * sizeof(int));
                st->dZiRelocDep.upload(ziDep.data(), ziDep.size());
            }
            st->ziDyn = true;
        }
    }
}

}  // namespace olfgpu
