// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Build the SHARED STRUCTURE once per extraction, size one scenario's LU factors, and refresh the
// per-element values in place.
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
#include "../fixedmatrix/relocation_pattern.cuh"

using namespace powsybl::openloadflow;

namespace olfgpu {

SharedStructure* buildStructure(int n,
                                int nClosedBranch, const double* closedBranchParameters, const int* closedBranchIndices,
                                int nOpenBranch, const double* openBranchParameters, const int* openBranchIndices,
                                int nShunt, const double* shuntParameters, const int* shuntIndices,
                                int nHvdcAcEmulation, const double* hvdcAcEmulationParameters, const int* hvdcAcEmulationIndices,
                                int nDistrQ, const int* distrQElement, const int* distrQSide, const double* distrQWeight,
                                const int* distrQRow, const int* distrQKind,
                                int nZeroImpedanceBranch, const double* zeroImpedanceBranchParameters, const int* zeroImpedanceBranchIndices,
                                int nDistrRho, const int* distrRhoRow, const int* distrRhoColumn, const double* distrRhoCoefficient,
                                int nDistrShuntB, const int* distrShuntBRow, const int* distrShuntBColumn, const double* distrShuntBCoefficient,
                                const int* closedBranchPhaseControlSide,
                                int nLoadModelTerms, const double* loadModelParameters, const int* loadModelIndices,
                                int nReactiveRelocationGroups, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRowsIn,
                                const int* openBranchAlternateReactiveDestinationIn, const int* zeroImpedanceBranchAlternateReactiveDestinationIn,
                                const int* reactiveRelocationDependentBaseGroupIn, const int* reactiveRelocationBaseTargetRowIn, const int* reactiveRelocationDependentGroupOfRowIn) {
    auto structure = std::make_unique<SharedStructure>();
    structure->n = n;
    structure->nElements = nClosedBranch + nOpenBranch + nShunt + nHvdcAcEmulation;
    structure->closedBranchOffset = 0;
    structure->openBranchOffset = nClosedBranch;
    structure->shuntOffset = nClosedBranch + nOpenBranch;
    structure->hvdcAcEmulationOffset = nClosedBranch + nOpenBranch + nShunt;

    // ---- diagonal-pairing row permutation + index packs rewritten through it ----
    structure->rowMap = buildRowMap(n, nClosedBranch, closedBranchIndices, nOpenBranch, openBranchIndices, nShunt, shuntIndices, nHvdcAcEmulation, hvdcAcEmulationIndices, nDistrQ, distrQRow, nZeroImpedanceBranch, zeroImpedanceBranchIndices);
    const std::vector<int>& rowMap = structure->rowMap;
    // COUPLER-TREE relocation, per-scenario placement: map each coupler DUMMY row that hosts a child bus's
    // relocated P/Q equation to that bus's OWN row, so a scenario OPENING the coupler can put the equation
    // back where the stock layout has it (see stockPlacementAlternateRow). zeroImpedanceBranchIndices is not row-mapped and its slots 4..9 are
    // variable COLUMNS, so both sides go through rowMap. Empty unless rows actually moved, which keeps every
    // existing structure bit-identical.
    std::vector<int> relocatedChildOwnRows(static_cast<size_t>(n), -1);
    bool anyRelocatedChild = false;
    for (int e = 0; e < nZeroImpedanceBranch; e++) {
        const int* z = zeroImpedanceBranchIndices + static_cast<size_t>(e) * 11;
        int childSide = (z[10] >> 1) & 3;
        if (childSide == 0 || (z[10] & 8) == 0) {
            continue;                                        // dead-end stub (bit 3 clear): long-standing case
        }
        int ownPhiCol = childSide == 1 ? z[8] : z[9];
        int ownVCol = childSide == 1 ? z[6] : z[7];
        if (z[4] >= 0 && ownPhiCol >= 0) {
            relocatedChildOwnRows[rowMap[z[4]]] = rowMap[ownPhiCol];
            anyRelocatedChild = true;
        }
        if (z[5] >= 0 && ownVCol >= 0) {
            relocatedChildOwnRows[rowMap[z[5]]] = rowMap[ownVCol];
            anyRelocatedChild = true;
        }
    }
    const int* relocatedChildOwnRow = anyRelocatedChild ? relocatedChildOwnRows.data() : nullptr;
    if (anyRelocatedChild) {                                       // the kernels that divert per scenario need it
        structure->dRelocatedChildOwnRow.upload(relocatedChildOwnRows.data(), relocatedChildOwnRows.size() * sizeof(int));
    }
    auto mapRows = [&](const int* idx, int elementCount, int stride, std::initializer_list<int> rowSlots) {
        std::vector<int> out(idx, idx + static_cast<size_t>(elementCount) * stride);
        for (int e = 0; e < elementCount; e++) {
            for (int s : rowSlots) {
                int& r = out[static_cast<size_t>(e) * stride + s];
                if (r >= 0) {
                    r = rowMap[r];
                }
            }
        }
        return out;
    };
    structure->nReactiveRelocationGroups = nReactiveRelocationGroups;
    if (nReactiveRelocationGroups > 0) {
        structure->hReactiveRelocationGroupStart.assign(reactiveRelocationGroupStart, reactiveRelocationGroupStart + nReactiveRelocationGroups + 1);
        structure->hReactiveRelocationCandidateRows.assign(reactiveRelocationCandidateRowsIn, reactiveRelocationCandidateRowsIn + reactiveRelocationGroupStart[nReactiveRelocationGroups]);
        for (int& r : structure->hReactiveRelocationCandidateRows) {
            r = rowMap[r];                                   // device row space (self-paired rows: identity)
        }
        structure->dReactiveRelocationGroupStart.upload(structure->hReactiveRelocationGroupStart.data(), structure->hReactiveRelocationGroupStart.size() * sizeof(int));
        structure->dReactiveRelocationCandidateRows.upload(structure->hReactiveRelocationCandidateRows.data(), structure->hReactiveRelocationCandidateRows.size() * sizeof(int));
        structure->hReactiveRelocationDependentBaseGroup.assign(reactiveRelocationDependentBaseGroupIn, reactiveRelocationDependentBaseGroupIn + nReactiveRelocationGroups);
        structure->hReactiveRelocationBaseTargetRow.assign(reactiveRelocationBaseTargetRowIn, reactiveRelocationBaseTargetRowIn + nReactiveRelocationGroups);
        for (int& r : structure->hReactiveRelocationBaseTargetRow) {
            if (r >= 0) {
                r = rowMap[r];
            }
        }
        structure->dReactiveRelocationDependentBaseGroup.upload(structure->hReactiveRelocationDependentBaseGroup.data(), structure->hReactiveRelocationDependentBaseGroup.size() * sizeof(int));
        structure->dReactiveRelocationBaseTargetRow.upload(structure->hReactiveRelocationBaseTargetRow.data(), structure->hReactiveRelocationBaseTargetRow.size() * sizeof(int));
    }
    structure->hReactiveRelocationDependentGroupOfRow.assign(n, -1);
    bool anyDependentGroup = false;
    for (int r = 0; r < n; r++) {
        if (reactiveRelocationDependentGroupOfRowIn[r] >= 0) {
            structure->hReactiveRelocationDependentGroupOfRow[rowMap[r]] = reactiveRelocationDependentGroupOfRowIn[r];
            anyDependentGroup = true;
        }
    }
    const int* reactiveRelocationDependentGroupOfRow = anyDependentGroup ? structure->hReactiveRelocationDependentGroupOfRow.data() : nullptr;
    std::vector<int> closedBranchIndicesMapped = mapRows(closedBranchIndices, nClosedBranch, 12, {0, 1, 2, 3, 10, 11});
    std::vector<int> openBranchIndicesMapped = mapRows(openBranchIndices, nOpenBranch, 6, {0, 1, 2, 3});
    std::vector<int> openBranchAlternateReactiveDestinationMapped = mapRows(openBranchAlternateReactiveDestinationIn, nOpenBranch, 2, {0, 1});   // negative encodings pass through
    std::vector<int> zeroImpedanceBranchAlternateReactiveDestinationMapped = mapRows(zeroImpedanceBranchAlternateReactiveDestinationIn, nZeroImpedanceBranch, 2, {0, 1});
    const int* openBranchAlternateReactiveDestination = openBranchAlternateReactiveDestinationMapped.data();
    const int* zeroImpedanceBranchAlternateReactiveDestination = zeroImpedanceBranchAlternateReactiveDestinationMapped.data();
    std::vector<int> shuntIndicesMapped = mapRows(shuntIndices, nShunt, 5, {0, 1, 4});
    std::vector<int> hvdcAcEmulationIndicesMapped = mapRows(hvdcAcEmulationIndices, nHvdcAcEmulation, 4, {0, 1});
    std::vector<int> loadModelIndicesMapped = mapRows(loadModelIndices, nLoadModelTerms, 2, {0});
    loadModelIndices = loadModelIndicesMapped.data();
    closedBranchIndices = closedBranchIndicesMapped.data();
    openBranchIndices = openBranchIndicesMapped.data();
    shuntIndices = shuntIndicesMapped.data();
    hvdcAcEmulationIndices = hvdcAcEmulationIndicesMapped.data();

    // ---- fixed CSR pattern (rows = permuted equations, cols = variables) ----
    std::vector<std::pair<int, int>> pat;
    collectPattern(nClosedBranch, closedBranchIndices, 12, CB_SCATTER, CB_NOUT, resolveClosedBranch, pat);
    bool dynamicRelocation = anyRelocatedChild || anyDependentGroup || hasRelocatable(nClosedBranch, closedBranchIndices, nShunt, shuntIndices)
            || hasAlternateRows(openBranchAlternateReactiveDestination, nOpenBranch * 2) || hasAlternateRows(zeroImpedanceBranchAlternateReactiveDestination, nZeroImpedanceBranch * 2);
    if (dynamicRelocation) {
        collectClosedBranchRelocationPattern(nClosedBranch, closedBranchIndices, relocatedChildOwnRow, pat, structure->nReactiveRelocationGroups,        // union the alternate Q rows into the pattern
                structure->hReactiveRelocationGroupStart.data(), structure->hReactiveRelocationCandidateRows.data(),
                        reactiveRelocationDependentGroupOfRow);      // (+ every GROUP candidate's slots, plan-A6)
    }
    bool hasPhaseControl = false;
    for (int e = 0; e < nClosedBranch; e++) {
        hasPhaseControl |= closedBranchPhaseControlSide[e] >= 0;
    }
    if (hasPhaseControl) {
        collectClosedBranchPhaseControlPattern(nClosedBranch, closedBranchIndices, closedBranchPhaseControlSide, rowMap, pat);   // union the a1-row P-flow positions
    }
    collectPattern(nOpenBranch, openBranchIndices, 6, OB_SCATTER, OB_NOUT, resolveOpenBranch, pat);
    if (dynamicRelocation) {
        collectOpenBranchRelocationPattern(nOpenBranch, openBranchIndices, openBranchAlternateReactiveDestination, relocatedChildOwnRow, pat, structure->nReactiveRelocationGroups, structure->hReactiveRelocationGroupStart.data(), structure->hReactiveRelocationCandidateRows.data(), reactiveRelocationDependentGroupOfRow);
    }
    collectPattern(nShunt, shuntIndices, 5, SH_SCATTER, SH_NOUT, resolveShunt, pat);
    if (dynamicRelocation) {
        collectShuntRelocationPattern(nShunt, shuntIndices, relocatedChildOwnRow, pat, structure->nReactiveRelocationGroups, structure->hReactiveRelocationGroupStart.data(), structure->hReactiveRelocationCandidateRows.data(), reactiveRelocationDependentGroupOfRow);
    }
    collectPattern(nHvdcAcEmulation, hvdcAcEmulationIndices, 4, HVDC_SCATTER, HVDC_NOUT, resolveHvdcAcEmulation, pat);
    if (dynamicRelocation) {
        collectHvdcAcEmulationRelocationPattern(nHvdcAcEmulation, hvdcAcEmulationIndices, relocatedChildOwnRow, pat);
    }
    collectFamilyPattern(rowMap, closedBranchIndices, openBranchIndices, shuntIndices, nDistrQ, distrQElement, distrQSide, distrQKind, distrQRow,
                         nDistrRho, distrRhoRow, distrRhoColumn, nDistrShuntB, distrShuntBRow, distrShuntBColumn, nZeroImpedanceBranch, zeroImpedanceBranchIndices, dynamicRelocation ? relocatedChildOwnRow : nullptr, pat);
    if (dynamicRelocation) {
        for (int e = 0; e < nZeroImpedanceBranch; e++) {                      // zero-impedance STOCK positions: a +-dummy coupling that lands on
        const int* z = zeroImpedanceBranchIndices + static_cast<size_t>(e) * 11;   // a relocated bus's home row also needs the slot
        if (relocatedChildOwnRow != nullptr) {                            // at that bus's OWN row, for the open-coupler case
            int rows[2] = {rowMap[z[0]], rowMap[z[2]]};
            for (int side = 0; side < 2; side++) {
                if (rows[side] >= 0 && relocatedChildOwnRow[rows[side]] >= 0) {
                    pat.emplace_back(relocatedChildOwnRow[rows[side]], z[4]);
                }
            }
            int qrows[2] = {z[1] >= 0 ? rowMap[z[1]] : -1, z[3] >= 0 ? rowMap[z[3]] : -1};
            for (int side = 0; side < 2; side++) {
                if (qrows[side] >= 0 && relocatedChildOwnRow[qrows[side]] >= 0) {
                    pat.emplace_back(relocatedChildOwnRow[qrows[side]], z[5]);
                }
            }
        }
    }
    if (reactiveRelocationDependentGroupOfRow != nullptr) {                       // DEPENDENT pilots: their +-dummyQ on every candidate row
        for (int e = 0; e < nZeroImpedanceBranch; e++) {
            const int* z = zeroImpedanceBranchIndices + static_cast<size_t>(e) * 11;
            int qrows[2] = {z[1] >= 0 ? rowMap[z[1]] : -1, z[3] >= 0 ? rowMap[z[3]] : -1};
            for (int side = 0; side < 2; side++) {
                int fr = qrows[side];
                int own = fr >= 0 && relocatedChildOwnRow != nullptr && relocatedChildOwnRow[fr] >= 0 ? relocatedChildOwnRow[fr] : fr;
                int dependentGroup = own >= 0 ? reactiveRelocationDependentGroupOfRow[own] : -1;
                if (dependentGroup < 0) {
                    continue;
                }
                for (int c = structure->hReactiveRelocationGroupStart[dependentGroup]; c < structure->hReactiveRelocationGroupStart[dependentGroup + 1]; c++) {
                    pat.emplace_back(structure->hReactiveRelocationCandidateRows[c], z[5]);
                }
            }
        }
    }
    for (int e = 0; e < nZeroImpedanceBranch; e++) {                      // zero-impedance alt J slots: (alt/candidate row, dummyQ col)
            int dummyQColumn = zeroImpedanceBranchIndices[static_cast<size_t>(e) * 11 + 5];
            for (int side = 0; side < 2; side++) {
                int alternateRow = zeroImpedanceBranchAlternateReactiveDestination[e * 2 + side];
                if (alternateRow <= -2) {
                    int g = -2 - alternateRow;
                    for (int c = structure->hReactiveRelocationGroupStart[g]; c < structure->hReactiveRelocationGroupStart[g + 1]; c++) {
                        pat.emplace_back(structure->hReactiveRelocationCandidateRows[c], dummyQColumn);
                    }
                } else if (alternateRow >= 0) {
                    pat.emplace_back(alternateRow, dummyQColumn);
                    if (relocatedChildOwnRow != nullptr && relocatedChildOwnRow[alternateRow] >= 0) {
                        pat.emplace_back(relocatedChildOwnRow[alternateRow], dummyQColumn);   // the stock pair's OWN-row slot
                    }
                }
            }
        }
    }
    for (int e = 0; e < nLoadModelTerms; e++) {                         // voltage-dependent load: (row, own-V) slot
        pat.emplace_back(loadModelIndices[e * 2], loadModelIndices[e * 2 + 1]);
        if (relocatedChildOwnRow != nullptr && relocatedChildOwnRow[loadModelIndices[e * 2]] >= 0) {
            pat.emplace_back(relocatedChildOwnRow[loadModelIndices[e * 2]], loadModelIndices[e * 2 + 1]);   // the re-homed child's own slot
        }
    }
    // ---- zero-diagonal repair tables (repairZeroDiagonalRowsBatch): hazard rows + donor candidates, then the
    //      pattern union so a borrowed row lands in place ----
    std::vector<int> zeroDiagonalRows;
    std::vector<std::vector<int>> zeroDiagonalDonors;
    std::vector<char> zeroDiagonalRepairSigns;
    {
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
            if (zeroImpedanceDeadEndSideOf(z) != 0) {
                // A RELOCATED edge (stub or tree): its own rows already have a diagonal - the child's balance on the
                // dummy rows, the ZERO rows on the child's own V / PHI rows. But the PARENT end still carries this
                // edge's dummies in its balance, and a section whose every coupler is relocated has nothing BUT
                // dummies there: once a scenario changes those (an outage around it), its home P / Q rows sit with
                // no diagonal and no registered donor. Register them, borrowing this edge's ZERO rows, which carry
                // the parent's phi / v columns (network.xiidm's full busbar scope: IBAJ_0, 1375 scenario solves).
                int child = zeroImpedanceDeadEndSideOf(z);
                int parPhi = child == 1 ? z[9] : z[8];
                int parV = child == 1 ? z[7] : z[6];
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
                if (!isSpanningTreeEdge(z) || zeroImpedanceDeadEndSideOf(z) != 0) continue;
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
    for (int r = 0; r < n; r++) {
        pat.emplace_back(r, r);                              // guaranteed diagonal: the per-solve
    }                                                        // row-mode mask writes 1.0 there
    std::sort(pat.begin(), pat.end());
    pat.erase(std::unique(pat.begin(), pat.end()), pat.end());
    structure->nnz = static_cast<int>(pat.size());
    if (structure->nnz == 0) throw std::runtime_error("empty Jacobian pattern");

    structure->rowPtr.assign(n + 1, 0);
    structure->colIdx.resize(structure->nnz);
    for (int i = 0; i < structure->nnz; i++) {                      // pat is row-major sorted
        structure->colIdx[i] = pat[i].second;
        structure->rowPtr[pat[i].first + 1]++;
    }
    for (int r = 0; r < n; r++) structure->rowPtr[r + 1] += structure->rowPtr[r];
    CsrPattern pos{structure->rowPtr.data(), structure->colIdx.data()};    // (row,col) -> slot by binary search
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
    structure->dRowPtr.upload(structure->rowPtr.data(), (n + 1) * sizeof(int));
    structure->dColIdx.upload(structure->colIdx.data(), structure->nnz * sizeof(int));

    // ---- family device data (inputs, indices, role-folded destinations) + static trig ----
    setupFamily(structure->closedBranch, nClosedBranch, closedBranchParameters, 8, closedBranchIndices, 12, CB_SCATTER, CB_NOUT, resolveClosedBranch, pos);
    if (dynamicRelocation && nClosedBranch) {                                   // Dynamic Q-relocation: alt-dest + frozen-row select map
        std::vector<int> alternateDestination;
        std::vector<int> selectorRow;
        std::vector<int> relocationGroup;
        std::vector<int> relocationCandidateOffset;
        std::vector<int> relocationCandidateDestinations;
        std::vector<char> relocationDependentPilot;
        buildClosedBranchRelocationDestination(nClosedBranch, closedBranchIndices, relocatedChildOwnRow, pos, alternateDestination, selectorRow, structure->nReactiveRelocationGroups, structure->hReactiveRelocationGroupStart.data(), structure->hReactiveRelocationCandidateRows.data(),
                relocationGroup, relocationCandidateOffset, relocationCandidateDestinations, reactiveRelocationDependentGroupOfRow, relocationDependentPilot);
        structure->closedBranch.alternateDestination.upload(alternateDestination.data(), alternateDestination.size() * sizeof(int));
        structure->closedBranch.selectorRow.upload(selectorRow.data(), selectorRow.size() * sizeof(int));
        if (!relocationCandidateDestinations.empty()) {
            structure->closedBranch.relocationGroup.upload(relocationGroup.data(), relocationGroup.size() * sizeof(int));
            structure->closedBranch.relocationCandidateOffset.upload(relocationCandidateOffset.data(), relocationCandidateOffset.size() * sizeof(int));
            structure->closedBranch.relocationCandidateDestinations.upload(relocationCandidateDestinations.data(), relocationCandidateDestinations.size() * sizeof(int));
            structure->closedBranch.relocationDependentPilot.upload(relocationDependentPilot.data(), relocationDependentPilot.size());
            static int cbRoles[CB_NOUT];
            for (int k = 0; k < CB_NOUT; k++) {
                cbRoles[k] = CB_SCATTER[k].role;
            }
            patchGroupFrozenDestinations(structure->closedBranch.destination, relocationGroup, relocationCandidateOffset, CB_NOUT, cbRoles,
                    structure->hReactiveRelocationGroupStart.data(), structure->hReactiveRelocationCandidateRows.data());
        }
        structure->closedBranch.dynamicRelocation = true;
    }
    if (hasPhaseControl) {                                          // continuous phase control: P-flow row at a1
        std::vector<int> phaseControlDestination;
        std::vector<int> phaseControlSelector;
        buildClosedBranchPhaseControlDestination(nClosedBranch, closedBranchIndices, closedBranchPhaseControlSide, rowMap, pos, phaseControlDestination, phaseControlSelector);
        structure->closedBranch.phaseControlDestination.upload(phaseControlDestination.data(), phaseControlDestination.size() * sizeof(int));
        structure->closedBranch.phaseControlSelector.upload(phaseControlSelector.data(), phaseControlSelector.size() * sizeof(int));
        structure->closedBranch.hasPhaseControl = true;
    }
    setupFamily(structure->openBranch, nOpenBranch, openBranchParameters, 7, openBranchIndices, 6, OB_SCATTER, OB_NOUT, resolveOpenBranch, pos);
    if (dynamicRelocation && nOpenBranch) {                                   // ob dynamic Q-routing: alt-dest + selector
        std::vector<int> openBranchAlternateDestination;
        std::vector<int> openBranchSelectorRow;
        std::vector<int> openBranchRelocationGroup;
        std::vector<int> openBranchRelocationCandidateOffset;
        std::vector<int> openBranchRelocationCandidateDestinations;
        std::vector<char> openBranchRelocationDependentPilot;
        buildOpenBranchRelocationDestination(nOpenBranch, openBranchIndices, openBranchAlternateReactiveDestination, relocatedChildOwnRow, pos, openBranchAlternateDestination, openBranchSelectorRow, structure->nReactiveRelocationGroups, structure->hReactiveRelocationGroupStart.data(), structure->hReactiveRelocationCandidateRows.data(),
                openBranchRelocationGroup, openBranchRelocationCandidateOffset, openBranchRelocationCandidateDestinations, reactiveRelocationDependentGroupOfRow, openBranchRelocationDependentPilot);
        structure->openBranch.alternateDestination.upload(openBranchAlternateDestination.data(), openBranchAlternateDestination.size() * sizeof(int));
        structure->openBranch.selectorRow.upload(openBranchSelectorRow.data(), openBranchSelectorRow.size() * sizeof(int));
        if (!openBranchRelocationCandidateDestinations.empty()) {
            structure->openBranch.relocationGroup.upload(openBranchRelocationGroup.data(), openBranchRelocationGroup.size() * sizeof(int));
            structure->openBranch.relocationCandidateOffset.upload(openBranchRelocationCandidateOffset.data(), openBranchRelocationCandidateOffset.size() * sizeof(int));
            structure->openBranch.relocationCandidateDestinations.upload(openBranchRelocationCandidateDestinations.data(), openBranchRelocationCandidateDestinations.size() * sizeof(int));
            structure->openBranch.relocationDependentPilot.upload(openBranchRelocationDependentPilot.data(), openBranchRelocationDependentPilot.size());
            static int obRoles[OB_NOUT];
            for (int k = 0; k < OB_NOUT; k++) {
                obRoles[k] = OB_SCATTER[k].role;
            }
            patchGroupFrozenDestinations(structure->openBranch.destination, openBranchRelocationGroup, openBranchRelocationCandidateOffset, OB_NOUT, obRoles,
                    structure->hReactiveRelocationGroupStart.data(), structure->hReactiveRelocationCandidateRows.data());
        }
        structure->openBranch.dynamicRelocation = true;
    }
    setupFamily(structure->shunt, nShunt, shuntParameters, 2, shuntIndices, 5, SH_SCATTER, SH_NOUT, resolveShunt, pos);
    if (dynamicRelocation && nShunt) {                                   // Dynamic Q-relocation: shunt alt-dest + selector
        std::vector<int> shuntAlternateDestination;
        std::vector<int> shuntSelectorRow;
        std::vector<int> shuntRelocationGroup;
        std::vector<int> shuntRelocationCandidateOffset;
        std::vector<int> shuntRelocationCandidateDestinations;
        std::vector<char> shuntRelocationDependentPilot;
        buildShuntRelocationDestination(nShunt, shuntIndices, relocatedChildOwnRow, pos, shuntAlternateDestination, shuntSelectorRow, structure->nReactiveRelocationGroups, structure->hReactiveRelocationGroupStart.data(), structure->hReactiveRelocationCandidateRows.data(),
                shuntRelocationGroup, shuntRelocationCandidateOffset, shuntRelocationCandidateDestinations, reactiveRelocationDependentGroupOfRow, shuntRelocationDependentPilot);
        structure->shunt.alternateDestination.upload(shuntAlternateDestination.data(), shuntAlternateDestination.size() * sizeof(int));
        structure->shunt.selectorRow.upload(shuntSelectorRow.data(), shuntSelectorRow.size() * sizeof(int));
        if (!shuntRelocationCandidateDestinations.empty()) {
            structure->shunt.relocationGroup.upload(shuntRelocationGroup.data(), shuntRelocationGroup.size() * sizeof(int));
            structure->shunt.relocationCandidateOffset.upload(shuntRelocationCandidateOffset.data(), shuntRelocationCandidateOffset.size() * sizeof(int));
            structure->shunt.relocationCandidateDestinations.upload(shuntRelocationCandidateDestinations.data(), shuntRelocationCandidateDestinations.size() * sizeof(int));
            structure->shunt.relocationDependentPilot.upload(shuntRelocationDependentPilot.data(), shuntRelocationDependentPilot.size());
            static int shRoles[SH_NOUT];
            for (int k = 0; k < SH_NOUT; k++) {
                shRoles[k] = SH_SCATTER[k].role;
            }
            patchGroupFrozenDestinations(structure->shunt.destination, shuntRelocationGroup, shuntRelocationCandidateOffset, SH_NOUT, shRoles,
                    structure->hReactiveRelocationGroupStart.data(), structure->hReactiveRelocationCandidateRows.data());
        }
        structure->shunt.dynamicRelocation = true;
    }
    setupFamily(structure->hvdcAcEmulation, nHvdcAcEmulation, hvdcAcEmulationParameters, 5, hvdcAcEmulationIndices, 4, HVDC_SCATTER, HVDC_NOUT, resolveHvdcAcEmulation, pos);
    if (dynamicRelocation && nHvdcAcEmulation) {                                   // a converter bus under a coupler the scenario opens
        std::vector<int> hvdcAcEmulationAlternateDestination;
        std::vector<int> hvdcAcEmulationSelectorRow;
        buildHvdcAcEmulationRelocationDestination(nHvdcAcEmulation, hvdcAcEmulationIndices, relocatedChildOwnRow, pos, hvdcAcEmulationAlternateDestination, hvdcAcEmulationSelectorRow);
        structure->hvdcAcEmulation.alternateDestination.upload(hvdcAcEmulationAlternateDestination.data(), hvdcAcEmulationAlternateDestination.size() * sizeof(int));
        structure->hvdcAcEmulation.selectorRow.upload(hvdcAcEmulationSelectorRow.data(), hvdcAcEmulationSelectorRow.size() * sizeof(int));
        structure->hvdcAcEmulation.dynamicRelocation = true;
    }

    // ---- DISTR_Q destinations: per contribution {F-row, dq/dv1, dq/dv2, dq/dph1, dq/dph2, dq/dr1}
    //      (-2-row encodes an F row, >=0 a J slot, -1 skip) ----
    buildDistrQDest(structure.get(), pos, rowMap, nDistrQ, distrQElement, distrQSide, distrQKind, distrQWeight, distrQRow, closedBranchIndices, openBranchIndices, shuntIndices, relocatedChildOwnRow);

    // ---- zero-impedance destinations: per branch the 8 dummy couplings + (tree) the 6 ZERO_V/ZERO_PHI ----
    buildZeroImpedanceBranchDestination(structure.get(), pos, rowMap, nZeroImpedanceBranch, zeroImpedanceBranchParameters, zeroImpedanceBranchIndices, zeroImpedanceBranchAlternateReactiveDestination, relocatedChildOwnRow, reactiveRelocationDependentGroupOfRow);

    // ---- DISTR_RHO destinations: per contribution {F row, J slot (row, r1 col)} ----
    buildDistrRhoDest(structure.get(), pos, rowMap, nDistrRho, distrRhoRow, distrRhoColumn, distrRhoCoefficient);

    // ---- DISTR_SHUNT_B destinations: per contribution {F row, J slot (row, b col)} ----
    buildDistrShuntBDest(structure.get(), pos, rowMap, nDistrShuntB, distrShuntBRow, distrShuntBColumn, distrShuntBCoefficient);

    // ---- voltage-dependent load (ZIP/exponential) destinations: one J slot (row, own-V) per term ----
    structure->nLoadModelTerms = nLoadModelTerms;
    if (nLoadModelTerms > 0) {
        std::vector<int> loadModelDestination(nLoadModelTerms);
        for (int e = 0; e < nLoadModelTerms; e++) {
            loadModelDestination[e] = pos.at(loadModelIndices[e * 2], loadModelIndices[e * 2 + 1]);
        }
        structure->dLoadModelParameters.upload(loadModelParameters, static_cast<size_t>(nLoadModelTerms) * 3 * sizeof(double));
        structure->dLoadModelIndices.upload(loadModelIndices, static_cast<size_t>(nLoadModelTerms) * 2 * sizeof(int));
        structure->dLoadModelDestination.upload(loadModelDestination.data(), static_cast<size_t>(nLoadModelTerms) * sizeof(int));
        std::vector<int> loadModelAlternateRow(nLoadModelTerms, -1);
        std::vector<int> loadModelAlternateDestination(nLoadModelTerms, -1);
        bool anyAlternateRow = false;
        for (int e = 0; e < nLoadModelTerms; e++) {
            int row = loadModelIndices[e * 2];
            if (relocatedChildOwnRow != nullptr && relocatedChildOwnRow[row] >= 0) {
                loadModelAlternateRow[e] = relocatedChildOwnRow[row];
                loadModelAlternateDestination[e] = pos.at(relocatedChildOwnRow[row], loadModelIndices[e * 2 + 1]);
                anyAlternateRow = true;
            }
        }
        structure->loadModelRelocated = anyAlternateRow;
        if (anyAlternateRow) {
            structure->dLoadModelAlternateRow.upload(loadModelAlternateRow.data(), static_cast<size_t>(nLoadModelTerms) * sizeof(int));
            structure->dLoadModelAlternateDestination.upload(loadModelAlternateDestination.data(), static_cast<size_t>(nLoadModelTerms) * sizeof(int));
        }
    }

    std::vector<int> diagPos(n);
    for (int r = 0; r < n; r++) diagPos[r] = pos.at(r, r);
    structure->dDiagPos.upload(diagPos.data(), n * sizeof(int));
    CUDA_CHECK(cudaDeviceSynchronize());                     // every structure upload done before first use
    return structure.release();
}

constexpr double LU_FILL = 12.0;     // heuristic LU-fill multiplier over nnz (fallback when LU_NNZ is unavailable)

// Per-scenario bytes the LU numeric factors occupy. The exact fill-in is only known after a
// symbolic ANALYSIS, so run a CHEAP one on a single matrix with the structure's device pattern
// and read cuDSS's reported CUDSS_DATA_LU_NNZ; fall back to the LU_FILL heuristic if that is
// unavailable (older cuDSS, query failure). The result is CACHED on the structure, so repeated
// chunk-size queries (one per SA run) probe only once per extraction.
double luBytesPerScenario(SharedStructure& structure) {
    if (structure.luBytes > 0) {
        return structure.luBytes;
    }
    int n = structure.n;
#if OLF_GPU_CUDSS_LU_NNZ
    try {
        DeviceBuf dVals;
        DeviceBuf dB;
        DeviceBuf dX;
        dVals.alloc(structure.nnz * sizeof(double));                // ANALYSIS is symbolic — values unused
        CUDA_CHECK(cudaMemset(dVals.p, 0, structure.nnz * sizeof(double)));
        dB.alloc(n * sizeof(double));
        dX.alloc(n * sizeof(double));
        CudssCtx solver;
        CUDSS_CHECK(cudssCreate(&solver.handle));
        CUDSS_CHECK(cudssConfigCreate(&solver.config));
        applyReorderAlg(solver.config, structure.nZeroImpedanceBranch > 0);
        CUDSS_CHECK(cudssDataCreate(solver.handle, &solver.data));
        CUDSS_CHECK(cudssMatrixCreateCsr(&solver.systemMatrix, n, n, structure.nnz, structure.dRowPtr.as<int>(), nullptr,
                                         structure.dColIdx.as<int>(), dVals.as<double>(), OLF_CUDSS_CSR_TYPES,
                                         CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        CUDSS_CHECK(cudssMatrixCreateDn(&solver.rightHandSideMatrix, n, 1, n, dB.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssMatrixCreateDn(&solver.solutionMatrix, n, 1, n, dX.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        gAnalysesMain++;
        { AnalysisTimer at_(__LINE__); CUDSS_CHECK(cudssExecute(solver.handle, CUDSS_PHASE_ANALYSIS, solver.config, solver.data, solver.systemMatrix, solver.solutionMatrix,
                solver.rightHandSideMatrix)); }
        // cuDSS's OWN device-memory estimate for one matrix with this pattern — [0] permanent,
        // [1] peak (16 int64 entries). This is the number to size with: the LU numeric factors are
        // only part of what a factorization allocates, and the workspace is the part our model used
        // to miss. MEASURED on rte6515 (n=13040, nnz=91002, 2026-09-08): cuDSS reports 4.97 MB/
        // matrix while the LU factors alone are ~1.3 MB, and the batch's REAL device usage sampled
        // over chunk sizes 50 and 200 was 3.7 MiB/scenario — i.e. this estimate is slightly
        // conservative, which is exactly what a chunk budget wants.
        {
            int64_t est[16] = {0};
            size_t writtenEst = 0;
            if (cudssDataGet(solver.handle, solver.data, CUDSS_DATA_MEMORY_ESTIMATES,
                             est, sizeof(est), &writtenEst) == CUDSS_STATUS_SUCCESS
                    && writtenEst >= 2 * sizeof(int64_t)) {
                int64_t deviceBytes = std::max(est[0], est[1]);
                if (deviceBytes > 0) {
                    structure.luBytes = static_cast<double>(deviceBytes);
                    return structure.luBytes;
                }
            }
        }
        int64_t luNnz = 0;
        size_t written = 0;
        CUDSS_CHECK(cudssDataGet(solver.handle, solver.data, CUDSS_DATA_LU_NNZ, &luNnz, sizeof(luNnz), &written));
        if (luNnz > 0) {
            structure.luBytes = static_cast<double>(luNnz) * sizeof(double);
            return structure.luBytes;
        }
    } catch (...) {
        // fall through to the heuristic
    }
#endif
    structure.luBytes = LU_FILL * static_cast<double>(structure.nnz) * sizeof(double);
    return structure.luBytes;
}


// VALUES-level refresh: re-upload only the per-element PARAMETER packs (closedBranchParameters/openBranchParameters/shuntParameters/hvdcAcEmulationParameters/zeroImpedanceBranchParameters)
// into the SHARED structure, in place. The CSR pattern, the diagonal pairing, the index/dest packs and
// cuDSS's symbolic ANALYSIS are UNCHANGED — they depend on the index packs and structure, not on these
// values (a transformer tap or shunt-b move changes r1/a1/b — or a coupler's rho, zeroImpedanceBranchParameters — only). The
// fill kernels read in[] every iteration, so the next batched solve refactorizes on the fresh values.
// Element COUNTS must match the cached structure (the Java side only takes this path when the index
// packs are identical), so we copy into the EXISTING device buffers (no realloc — DeviceBuf::alloc
// would leak the old pointer). The static trig is re-derived (a tap move can change ksi/a1). Mutating
// the shared structure is the correct semantic — one extraction has ONE value set; any context sharing
// it must see the new values (the Java side then re-keys the structure cache to the new extraction,
// GpuAcStructures.rebind, so content matching never aliases stale values).
void refreshStructureValues(SharedStructure& structure, int nClosedBranch, const double* closedBranchParameters, int nOpenBranch, const double* openBranchParameters,
                            int nShunt, const double* shuntParameters, int nHvdcAcEmulation, const double* hvdcAcEmulationParameters,
                            int nZeroImpedanceBranch, const double* zeroImpedanceBranchParameters) {
    auto refresh = [](Family& f, int elementCount, const double* in) {
        if (elementCount != f.elementCount) {
            throw std::runtime_error("refreshStructureValues: element count changed (not a VALUES event)");
        }
        if (elementCount) {
            CUDA_CHECK(cudaMemcpy(f.parameters.p, in, static_cast<size_t>(elementCount) * f.parameterStride * sizeof(double),
                                  cudaMemcpyHostToDevice));
        }
    };
    refresh(structure.closedBranch, nClosedBranch, closedBranchParameters);
    refresh(structure.openBranch, nOpenBranch, openBranchParameters);
    refresh(structure.shunt, nShunt, shuntParameters);
    refresh(structure.hvdcAcEmulation, nHvdcAcEmulation, hvdcAcEmulationParameters);
    if (nZeroImpedanceBranch != structure.nZeroImpedanceBranch) {
        throw std::runtime_error("refreshStructureValues: zero-impedance count changed (not a VALUES event)");
    }
    if (nZeroImpedanceBranch) {
        CUDA_CHECK(cudaMemcpy(structure.dZeroImpedanceBranchParameters.p, zeroImpedanceBranchParameters, static_cast<size_t>(nZeroImpedanceBranch) * sizeof(double), cudaMemcpyHostToDevice));
    }
}

}  // namespace olfgpu
