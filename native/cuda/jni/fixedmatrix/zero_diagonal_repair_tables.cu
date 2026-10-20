// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// docs/gpu-vs-olf.md section 2.4 (a factorization without pivoting): the host side of the row repair - which rows
// may have a zero diagonal, which rows they borrow (zero_diagonal_repair.cu applies it per solve). See
// zero_diagonal_repair_tables.cuh.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <utility>
#include <vector>

#include "../common/codes/gpu_two_sides.cuh"
#include "../structure/shared_structure.cuh"
#include "zero_diagonal_repair_tables.cuh"

namespace olfgpu {

// The zero-diagonal repair hazard rows (repairZeroDiagonalRowsBatch) of the coupler rows, each with its donor
// candidates in preference order and its sign, then the pattern union so a borrowed row lands in place (pat).
void collectZeroDiagonalRepairRows(const SharedStructure* structure, int n, int nZeroImpedanceBranch, const int* zeroImpedanceBranchIndices,
                                   const int* zeroImpedanceBranchAlternateReactiveDestination, const std::vector<int>& rowMap, std::vector<std::pair<int, int>>& pat,
                                   std::vector<int>& zeroDiagonalRows, std::vector<std::vector<int>>& zeroDiagonalDonors, std::vector<char>& zeroDiagonalRepairSigns) {
    std::map<int, int> zeroDiagonalIndexOfRow;                          // physical row -> its repair entry
    auto repairEntryOf = [&](int row, char sign) {
        auto it = zeroDiagonalIndexOfRow.find(row);
        if (it != zeroDiagonalIndexOfRow.end()) return it->second;
        int i = static_cast<int>(zeroDiagonalRows.size());
        zeroDiagonalIndexOfRow[row] = i;
        zeroDiagonalRows.push_back(row);
        zeroDiagonalDonors.emplace_back();
        zeroDiagonalRepairSigns.push_back(sign);
        return i;
    };
    auto addDonor = [&](int repairEntry, int donor) {
        if (donor < 0 || donor == zeroDiagonalRows[repairEntry]) return;
        auto& v = zeroDiagonalDonors[repairEntry];
        if (std::find(v.begin(), v.end(), donor) == v.end()) v.push_back(donor);
    };
    for (int e = 0; e < nZeroImpedanceBranch; e++) {
        const int* z = zeroImpedanceBranchIndices + static_cast<size_t>(e) * 11;
        // A LOOP edge (out of the base tree) is registered too: its dummies are identities in the base, but a
        // scenario whose contingency re-elects OLF's tree RE-COUPLES it (GpuBatchChunkSolver), and its ZERO_V /
        // ZERO_PHI then sit on those dummy rows with a zero diagonal exactly like a tree edge's. Leaving them
        // out of the set left them unrepaired: network.xiidm's 1000-section busbar window, ALWQ re-coupled
        // when its loop's tree coupler ALVS opened - 189 linear-solve rejects. Only a relocated edge (its
        // ZERO rows already sit on a child's own rows, which have a diagonal) is still skipped here.
        if (zeroImpedanceDeadEndSideOf(z) != GpuTwoSides::NONE) {
            // A RELOCATED edge (stub or tree): its own rows already have a diagonal - the child's balance on the
            // dummy rows, the ZERO rows on the child's own V / PHI rows. But the PARENT end still carries this
            // edge's dummies in its balance, and a section whose every coupler is relocated has nothing BUT
            // dummies there: once a scenario changes those (an outage around it), its home P / Q rows sit with
            // no diagonal and no registered donor. Register them, borrowing this edge's ZERO rows, which carry
            // the parent's phi / v columns (network.xiidm's full busbar scope: IBAJ_0, 1375 scenario solves).
            int child = zeroImpedanceDeadEndSideOf(z);
            int parPhi = child == GpuTwoSides::ONE ? z[9] : z[8];
            int parV = child == GpuTwoSides::ONE ? z[7] : z[6];
            int zPhi = rowMap[zeroPhiRowOf(z)];
            int zV = rowMap[zeroVRowOf(z)];
            if (parPhi >= 0) addDonor(repairEntryOf(rowMap[parPhi], 1), zPhi);
            if (parV >= 0) addDonor(repairEntryOf(rowMap[parV], -1), zV);
            continue;
        }
        // The DUMMY_P row (diagonal column dummy-P, carried by the endpoint P rows) hosts ZERO_PHI and the
        // DUMMY_Q row (dummy-Q, carried by the Q rows) hosts ZERO_V - see zeroVRowOf: crosswise, so that
        // each repaired row keeps an entry of its own in its donor's diagonal column and the repaired 2x2
        // block is not singular. The donors are keyed by the DUMMY column, whatever equation the row hosts:
        // a first cut had them matched to the equations instead, so no dummy row ever found a donor -
        // invisible at 28 couplers (cuDSS still coped, with IR=2), fatal at rte6515's full 366 (the base
        // solve's linear residual was 1e9 and Newton never moved).
        int dP = rowMap[z[4]];                           // the DUMMY_P row: hosts ZERO_PHI
        int dQ = rowMap[z[5]];                           // the DUMMY_Q row: hosts ZERO_V
        int pa = z[0] >= 0 ? rowMap[z[0]] : -1;
        int pb = z[2] >= 0 ? rowMap[z[2]] : -1;
        int qaRow = z[1] >= 0 ? z[1] : z[6];             // suppressed Q balance: the V slot hosts DISTR_Q
        int qbRow = z[3] >= 0 ? z[3] : z[7];
        int qa = qaRow >= 0 ? rowMap[qaRow] : -1;
        int qb = qbRow >= 0 ? rowMap[qbRow] : -1;
        // An endpoint that is a RELOCATED child has its balance on its parent coupler's dummy rows (pa / qa
        // above), but a scenario that opens that parent moves the balance HOME, onto the child's own PHI / V
        // rows - and for a section whose only branches are couplers that home row has no diagonal. Register
        // the home rows as well (the same row as pa / qa for a bus that was never relocated): they borrow this
        // coupler's ZERO rows, and they are donors for its dummy rows (ALPJ_5 and ALWQ, the mirrored pair).
        int phA = z[8] >= 0 ? rowMap[z[8]] : -1;
        int phB = z[9] >= 0 ? rowMap[z[9]] : -1;
        int vA = z[6] >= 0 ? rowMap[z[6]] : -1;
        int vB = z[7] >= 0 ? rowMap[z[7]] : -1;
        // Signs: with the crosswise placement the borrowing closes in 2-cycles only - P(a) <-> ZERO_PHI on
        // the DUMMY_P row, Q(a) <-> ZERO_V on the DUMMY_Q row (a coupler's dummy row can borrow from and
        // lend to its two endpoint rows alone, and a tree allows no longer cycle) - and det(I + C) on a
        // 2-cycle is 1 - s_h*s_d: the two signs must differ. P rows +1 and the DUMMY_P row -1; Q rows -1
        // and the DUMMY_Q row +1. (The former placement closed 4-cycles P -> ZERO_PHI -> Q -> ZERO_V -> P,
        // whose determinant is 1 - the product of the four signs, and all-bus-rows-+1 gave det 0 there -
        // a singular, CONSISTENT system cuDSS solved to 1e-11 with the wrong answer: rte6515 500 busbars,
        // 175 contingencies above 1e-9 kV where the tree-only borrowing had 3.)
        int sP = repairEntryOf(dP, -1);
        int sQ = repairEntryOf(dQ, 1);
        addDonor(sP, pa);
        addDonor(sP, pb);
        addDonor(sQ, qa);
        addDonor(sQ, qb);
        addDonor(sP, phA);
        addDonor(sP, phB);
        addDonor(sQ, vA);
        addDonor(sQ, vB);
        for (int side = 0; side < 2; side++) {           // endpoint Q riding a relocation group
            int alt = zeroImpedanceBranchAlternateReactiveDestination[e * 2 + side];
            if (alt <= -2 && !structure->hReactiveRelocationGroupStart.empty()) {
                int g = -2 - alt;
                for (int c = structure->hReactiveRelocationGroupStart[g]; c < structure->hReactiveRelocationGroupStart[g + 1]; c++) addDonor(sQ, structure->hReactiveRelocationCandidateRows[c]);
            }
        }
        if (pa >= 0) addDonor(repairEntryOf(pa, 1), dP);       // phi_a is carried by ZERO_PHI, on the DUMMY_P row
        if (pb >= 0) addDonor(repairEntryOf(pb, 1), dP);
        if (qa >= 0) addDonor(repairEntryOf(qa, -1), dQ);      // v_a is carried by ZERO_V, on the DUMMY_Q row
        if (qb >= 0) addDonor(repairEntryOf(qb, -1), dQ);
        if (phA >= 0 && phA != pa) addDonor(repairEntryOf(phA, 1), dP);
        if (phB >= 0 && phB != pb) addDonor(repairEntryOf(phB, 1), dP);
        if (vA >= 0 && vA != qa) addDonor(repairEntryOf(vA, -1), dQ);
        if (vB >= 0 && vB != qb) addDonor(repairEntryOf(vB, -1), dQ);
    }
    if (!zeroDiagonalRows.empty()) {
        // DONOR ORDER decides LU stability without pivoting. The kernel takes the FIRST candidate
        // with a nonzero coefficient, and two rows borrowing the SAME donor come out near-identical
        // (a three-section bus: both couplers' ZERO_PHI rows plus the middle bus's Q row) -
        // nonsingular, but their elimination cancels to a pivot cuDSS's static pivoting cannot
        // survive (rte6515 full busbar structure: the repaired system solved to 9e-13 by a pivoted
        // LU, to 2.5e6 by cuDSS). So: a dummy row prefers the endpoint with the FEWEST incident
        // tree couplers (a leaf is unique to its coupler) and, among equals, the one no other
        // dummy row has taken; a bus row prefers a ZERO row no other bus row has taken.
        std::map<int, int> spanningTreeDegree;                        // bus row -> incident tree couplers
        for (int e = 0; e < nZeroImpedanceBranch; e++) {
            const int* z = zeroImpedanceBranchIndices + static_cast<size_t>(e) * 11;
            if (!isSpanningTreeEdge(z) || zeroImpedanceDeadEndSideOf(z) != GpuTwoSides::NONE) continue;
            for (int slot : {z[0], z[2], z[1] >= 0 ? z[1] : z[6], z[3] >= 0 ? z[3] : z[7]}) {
                if (slot >= 0) spanningTreeDegree[rowMap[slot]]++;
            }
        }
        std::map<int, int> used;
        for (size_t i = 0; i < zeroDiagonalRows.size(); i++) {
            auto& v = zeroDiagonalDonors[i];
            std::stable_sort(v.begin(), v.end(), [&](int a, int b) {
                int da = spanningTreeDegree.count(a) ? spanningTreeDegree[a] : 0;
                int db = spanningTreeDegree.count(b) ? spanningTreeDegree[b] : 0;
                if (da != db) return da < db;
                return used[a] < used[b];
            });
            if (!v.empty()) used[v[0]]++;
        }
        std::vector<std::pair<int, int>> sorted(pat);
        std::sort(sorted.begin(), sorted.end());
        sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
        std::vector<int> rp(n + 1, 0);
        for (auto& rc : sorted) rp[rc.first + 1]++;
        for (int r = 0; r < n; r++) rp[r + 1] += rp[r];
        for (size_t i = 0; i < zeroDiagonalRows.size(); i++) {
            for (int d : zeroDiagonalDonors[i]) {
                for (int k = rp[d]; k < rp[d + 1]; k++) {
                    pat.emplace_back(zeroDiagonalRows[i], sorted[k].second);
                }
                pat.emplace_back(d, zeroDiagonalRows[i]);         // the coefficient the numeric choice reads
            }
        }
    }
}

// The repair tables on the device, positioned in the final CSR (structure->rowPtr): hazard rows, their value
// ranges, donors (with their own repair entry when a donor is itself repaired) and signs.
void storeZeroDiagonalRepairTables(SharedStructure* structure, const std::vector<int>& zeroDiagonalRows, const std::vector<std::vector<int>>& zeroDiagonalDonors,
                                   const std::vector<char>& zeroDiagonalRepairSigns) {
    structure->nZeroDiagonalRows = static_cast<int>(zeroDiagonalRows.size());
    if (structure->nZeroDiagonalRows > 0) {
        std::vector<int> zeroDiagonalRowValueStart(structure->nZeroDiagonalRows + 1, 0);
        std::vector<int> donorPtr(structure->nZeroDiagonalRows + 1, 0);
        std::vector<int> donor;
        std::vector<int> donorOwnIndex;
        std::map<int, int> zeroDiagonalIndexOfRow;
        for (int i = 0; i < structure->nZeroDiagonalRows; i++) zeroDiagonalIndexOfRow[zeroDiagonalRows[i]] = i;
        for (int i = 0; i < structure->nZeroDiagonalRows; i++) {
            zeroDiagonalRowValueStart[i + 1] = zeroDiagonalRowValueStart[i] + (structure->rowPtr[zeroDiagonalRows[i] + 1] - structure->rowPtr[zeroDiagonalRows[i]]);
            for (int d : zeroDiagonalDonors[i]) {
                donor.push_back(d);
                auto it = zeroDiagonalIndexOfRow.find(d);
                donorOwnIndex.push_back(it == zeroDiagonalIndexOfRow.end() ? -1 : it->second);
            }
            donorPtr[i + 1] = static_cast<int>(donor.size());
        }
        structure->nZeroDiagonalRowValues = zeroDiagonalRowValueStart[structure->nZeroDiagonalRows];
        structure->hZeroDiagonalRow = zeroDiagonalRows;
        structure->dZeroDiagonalRow.upload(zeroDiagonalRows.data(), zeroDiagonalRows.size() * sizeof(int));
        structure->dZeroDiagonalRowValueStart.upload(zeroDiagonalRowValueStart.data(), zeroDiagonalRowValueStart.size() * sizeof(int));
        structure->dZeroDiagonalDonorStart.upload(donorPtr.data(), donorPtr.size() * sizeof(int));
        structure->dZeroDiagonalDonor.upload(donor.data(), donor.size() * sizeof(int));
        structure->dZeroDiagonalDonorOwnIndex.upload(donorOwnIndex.data(), donorOwnIndex.size() * sizeof(int));
        structure->dZeroDiagonalRepairSign.upload(zeroDiagonalRepairSigns.data(), zeroDiagonalRepairSigns.size() * sizeof(char));
        if (std::getenv("OLF_GPU_TRACE") != nullptr) {
            std::fprintf(stderr, "zero-diagonal repair: %d hazard rows, %zu donor candidates\n", structure->nZeroDiagonalRows, donor.size());
        }
    }
}

}  // namespace olfgpu
