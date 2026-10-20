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
                                int nQr, const int* qrPtr, const int* qrRowsIn,
                                const int* openBranchAlternateReactiveDestinationIn, const int* zeroImpedanceBranchAlternateReactiveDestinationIn,
                                const int* qrDepBaseIn, const int* qrBaseTargetRowIn, const int* depGroupOfRowIn) {
    auto st = std::make_unique<SharedStructure>();
    st->n = n;
    st->nElements = nClosedBranch + nOpenBranch + nShunt + nHvdcAcEmulation;
    st->closedBranchOffset = 0;
    st->openBranchOffset = nClosedBranch;
    st->shuntOffset = nClosedBranch + nOpenBranch;
    st->hvdcAcEmulationOffset = nClosedBranch + nOpenBranch + nShunt;

    // ---- diagonal-pairing row permutation + index packs rewritten through it ----
    st->rowMap = buildRowMap(n, nClosedBranch, closedBranchIndices, nOpenBranch, openBranchIndices, nShunt, shuntIndices, nHvdcAcEmulation, hvdcAcEmulationIndices, nDistrQ, distrQRow, nZeroImpedanceBranch, zeroImpedanceBranchIndices);
    const std::vector<int>& rowMap = st->rowMap;
    // COUPLER-TREE relocation, per-scenario placement: map each coupler DUMMY row that hosts a child bus's
    // relocated P/Q equation to that bus's OWN row, so a scenario OPENING the coupler can put the equation
    // back where the stock layout has it (see stockAltRow). zeroImpedanceBranchIndices is not row-mapped and its slots 4..9 are
    // variable COLUMNS, so both sides go through rowMap. Empty unless rows actually moved, which keeps every
    // existing structure bit-identical.
    std::vector<int> relocOwnRow(static_cast<size_t>(n), -1);
    bool anyRelocOwn = false;
    for (int e = 0; e < nZeroImpedanceBranch; e++) {
        const int* z = zeroImpedanceBranchIndices + static_cast<size_t>(e) * 11;
        int childSide = (z[10] >> 1) & 3;
        if (childSide == 0 || (z[10] & 8) == 0) {
            continue;                                        // dead-end stub (bit 3 clear): long-standing case
        }
        int ownPhiCol = childSide == 1 ? z[8] : z[9];
        int ownVCol = childSide == 1 ? z[6] : z[7];
        if (z[4] >= 0 && ownPhiCol >= 0) {
            relocOwnRow[rowMap[z[4]]] = rowMap[ownPhiCol];
            anyRelocOwn = true;
        }
        if (z[5] >= 0 && ownVCol >= 0) {
            relocOwnRow[rowMap[z[5]]] = rowMap[ownVCol];
            anyRelocOwn = true;
        }
    }
    const int* relocOwn = anyRelocOwn ? relocOwnRow.data() : nullptr;
    if (anyRelocOwn) {                                       // the kernels that divert per scenario need it
        st->dRelocOwn.upload(relocOwnRow.data(), relocOwnRow.size() * sizeof(int));
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
    st->nQr = nQr;
    if (nQr > 0) {
        st->hQrPtr.assign(qrPtr, qrPtr + nQr + 1);
        st->hQrRows.assign(qrRowsIn, qrRowsIn + qrPtr[nQr]);
        for (int& r : st->hQrRows) {
            r = rowMap[r];                                   // device row space (self-paired rows: identity)
        }
        st->dQrPtr.upload(st->hQrPtr.data(), st->hQrPtr.size() * sizeof(int));
        st->dQrRows.upload(st->hQrRows.data(), st->hQrRows.size() * sizeof(int));
        st->hQrDepBase.assign(qrDepBaseIn, qrDepBaseIn + nQr);
        st->hQrBaseTargetRow.assign(qrBaseTargetRowIn, qrBaseTargetRowIn + nQr);
        for (int& r : st->hQrBaseTargetRow) {
            if (r >= 0) {
                r = rowMap[r];
            }
        }
        st->dQrDepBase.upload(st->hQrDepBase.data(), st->hQrDepBase.size() * sizeof(int));
        st->dQrBaseTargetRow.upload(st->hQrBaseTargetRow.data(), st->hQrBaseTargetRow.size() * sizeof(int));
    }
    st->hDepGroupOfRow.assign(n, -1);
    bool anyDep = false;
    for (int r = 0; r < n; r++) {
        if (depGroupOfRowIn[r] >= 0) {
            st->hDepGroupOfRow[rowMap[r]] = depGroupOfRowIn[r];
            anyDep = true;
        }
    }
    const int* depGroupOfRow = anyDep ? st->hDepGroupOfRow.data() : nullptr;
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
    bool dynReloc = anyRelocOwn || anyDep || hasRelocatable(nClosedBranch, closedBranchIndices, nShunt, shuntIndices)
            || hasAltEntries(openBranchAlternateReactiveDestination, nOpenBranch * 2) || hasAltEntries(zeroImpedanceBranchAlternateReactiveDestination, nZeroImpedanceBranch * 2);
    if (dynReloc) {
        collectCbAltPattern(nClosedBranch, closedBranchIndices, relocOwn, pat, st->nQr,        // union the alternate Q rows into the pattern
                st->hQrPtr.data(), st->hQrRows.data(), depGroupOfRow);      // (+ every GROUP candidate's slots, plan-A6)
    }
    bool hasPhaseControl = false;
    for (int e = 0; e < nClosedBranch; e++) {
        hasPhaseControl |= closedBranchPhaseControlSide[e] >= 0;
    }
    if (hasPhaseControl) {
        collectClosedBranchPhaseControlPattern(nClosedBranch, closedBranchIndices, closedBranchPhaseControlSide, rowMap, pat);   // union the a1-row P-flow positions
    }
    collectPattern(nOpenBranch, openBranchIndices, 6, OB_SCATTER, OB_NOUT, resolveOpenBranch, pat);
    if (dynReloc) {
        collectObAltPattern(nOpenBranch, openBranchIndices, openBranchAlternateReactiveDestination, relocOwn, pat, st->nQr, st->hQrPtr.data(), st->hQrRows.data(), depGroupOfRow);
    }
    collectPattern(nShunt, shuntIndices, 5, SH_SCATTER, SH_NOUT, resolveShunt, pat);
    if (dynReloc) {
        collectShAltPattern(nShunt, shuntIndices, relocOwn, pat, st->nQr, st->hQrPtr.data(), st->hQrRows.data(), depGroupOfRow);
    }
    collectPattern(nHvdcAcEmulation, hvdcAcEmulationIndices, 4, HVDC_SCATTER, HVDC_NOUT, resolveHvdcAcEmulation, pat);
    if (dynReloc) {
        collectHvAltPattern(nHvdcAcEmulation, hvdcAcEmulationIndices, relocOwn, pat);
    }
    collectFamilyPattern(rowMap, closedBranchIndices, openBranchIndices, shuntIndices, nDistrQ, distrQElement, distrQSide, distrQKind, distrQRow,
                         nDistrRho, distrRhoRow, distrRhoColumn, nDistrShuntB, distrShuntBRow, distrShuntBColumn, nZeroImpedanceBranch, zeroImpedanceBranchIndices, dynReloc ? relocOwn : nullptr, pat);
    if (dynReloc) {
        for (int e = 0; e < nZeroImpedanceBranch; e++) {                      // zero-impedance STOCK positions: a +-dummy coupling that lands on
        const int* z = zeroImpedanceBranchIndices + static_cast<size_t>(e) * 11;   // a relocated bus's home row also needs the slot
        if (relocOwn != nullptr) {                            // at that bus's OWN row, for the open-coupler case
            int rows[2] = {rowMap[z[0]], rowMap[z[2]]};
            for (int side = 0; side < 2; side++) {
                if (rows[side] >= 0 && relocOwn[rows[side]] >= 0) {
                    pat.emplace_back(relocOwn[rows[side]], z[4]);
                }
            }
            int qrows[2] = {z[1] >= 0 ? rowMap[z[1]] : -1, z[3] >= 0 ? rowMap[z[3]] : -1};
            for (int side = 0; side < 2; side++) {
                if (qrows[side] >= 0 && relocOwn[qrows[side]] >= 0) {
                    pat.emplace_back(relocOwn[qrows[side]], z[5]);
                }
            }
        }
    }
    if (depGroupOfRow != nullptr) {                       // DEPENDENT pilots: their +-dummyQ on every candidate row
        for (int e = 0; e < nZeroImpedanceBranch; e++) {
            const int* z = zeroImpedanceBranchIndices + static_cast<size_t>(e) * 11;
            int qrows[2] = {z[1] >= 0 ? rowMap[z[1]] : -1, z[3] >= 0 ? rowMap[z[3]] : -1};
            for (int side = 0; side < 2; side++) {
                int fr = qrows[side];
                int own = fr >= 0 && relocOwn != nullptr && relocOwn[fr] >= 0 ? relocOwn[fr] : fr;
                int dg = own >= 0 ? depGroupOfRow[own] : -1;
                if (dg < 0) {
                    continue;
                }
                for (int c = st->hQrPtr[dg]; c < st->hQrPtr[dg + 1]; c++) {
                    pat.emplace_back(st->hQrRows[c], z[5]);
                }
            }
        }
    }
    for (int e = 0; e < nZeroImpedanceBranch; e++) {                      // zero-impedance alt J slots: (alt/candidate row, dummyQ col)
            int dqCol = zeroImpedanceBranchIndices[static_cast<size_t>(e) * 11 + 5];
            for (int side = 0; side < 2; side++) {
                int altRow = zeroImpedanceBranchAlternateReactiveDestination[e * 2 + side];
                if (altRow <= -2) {
                    int g = -2 - altRow;
                    for (int c = st->hQrPtr[g]; c < st->hQrPtr[g + 1]; c++) {
                        pat.emplace_back(st->hQrRows[c], dqCol);
                    }
                } else if (altRow >= 0) {
                    pat.emplace_back(altRow, dqCol);
                    if (relocOwn != nullptr && relocOwn[altRow] >= 0) {
                        pat.emplace_back(relocOwn[altRow], dqCol);   // the stock pair's OWN-row slot
                    }
                }
            }
        }
    }
    for (int e = 0; e < nLoadModelTerms; e++) {                         // voltage-dependent load: (row, own-V) slot
        pat.emplace_back(loadModelIndices[e * 2], loadModelIndices[e * 2 + 1]);
        if (relocOwn != nullptr && relocOwn[loadModelIndices[e * 2]] >= 0) {
            pat.emplace_back(relocOwn[loadModelIndices[e * 2]], loadModelIndices[e * 2 + 1]);   // the re-homed child's own slot
        }
    }
    // ---- zero-diagonal repair tables (hazRepairVals): hazard rows + donor candidates, then the
    //      pattern union so a borrowed row lands in place ----
    std::vector<int> hazRows;
    std::vector<std::vector<int>> hazDonors;
    std::vector<char> hazSigns;
    {
        std::map<int, int> hazSlot;                          // physical row -> hazard slot
        auto slotOf = [&](int row, char sign) {
            auto it = hazSlot.find(row);
            if (it != hazSlot.end()) return it->second;
            int i = static_cast<int>(hazRows.size());
            hazSlot[row] = i;
            hazRows.push_back(row);
            hazDonors.emplace_back();
            hazSigns.push_back(sign);
            return i;
        };
        auto addDonor = [&](int slot, int donor) {
            if (donor < 0 || donor == hazRows[slot]) return;
            auto& v = hazDonors[slot];
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
            if (ziDeadSide(z) != 0) {
                // A RELOCATED edge (stub or tree): its own rows already have a diagonal - the child's balance on the
                // dummy rows, the ZERO rows on the child's own V / PHI rows. But the PARENT end still carries this
                // edge's dummies in its balance, and a section whose every coupler is relocated has nothing BUT
                // dummies there: once a scenario changes those (an outage around it), its home P / Q rows sit with
                // no diagonal and no registered donor. Register them, borrowing this edge's ZERO rows, which carry
                // the parent's phi / v columns (network.xiidm's full busbar scope: IBAJ_0, 1375 scenario solves).
                int child = ziDeadSide(z);
                int parPhi = child == 1 ? z[9] : z[8];
                int parV = child == 1 ? z[7] : z[6];
                int zPhi = rowMap[ziZeroPhiRow(z)];
                int zV = rowMap[ziZeroVRow(z)];
                if (parPhi >= 0) addDonor(slotOf(rowMap[parPhi], 1), zPhi);
                if (parV >= 0) addDonor(slotOf(rowMap[parV], -1), zV);
                continue;
            }
            // The DUMMY_P row (diagonal column dummy-P, carried by the endpoint P rows) hosts ZERO_PHI and the
            // DUMMY_Q row (dummy-Q, carried by the Q rows) hosts ZERO_V - see ziZeroVRow: crosswise, so that
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
            int sP = slotOf(dP, -1);
            int sQ = slotOf(dQ, 1);
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
                if (alt <= -2 && !st->hQrPtr.empty()) {
                    int g = -2 - alt;
                    for (int c = st->hQrPtr[g]; c < st->hQrPtr[g + 1]; c++) addDonor(sQ, st->hQrRows[c]);
                }
            }
            if (pa >= 0) addDonor(slotOf(pa, 1), dP);       // phi_a is carried by ZERO_PHI, on the DUMMY_P row
            if (pb >= 0) addDonor(slotOf(pb, 1), dP);
            if (qa >= 0) addDonor(slotOf(qa, -1), dQ);      // v_a is carried by ZERO_V, on the DUMMY_Q row
            if (qb >= 0) addDonor(slotOf(qb, -1), dQ);
            if (phA >= 0 && phA != pa) addDonor(slotOf(phA, 1), dP);
            if (phB >= 0 && phB != pb) addDonor(slotOf(phB, 1), dP);
            if (vA >= 0 && vA != qa) addDonor(slotOf(vA, -1), dQ);
            if (vB >= 0 && vB != qb) addDonor(slotOf(vB, -1), dQ);
        }
        if (!hazRows.empty()) {
            // DONOR ORDER decides LU stability without pivoting. The kernel takes the FIRST candidate
            // with a nonzero coefficient, and two rows borrowing the SAME donor come out near-identical
            // (a three-section bus: both couplers' ZERO_PHI rows plus the middle bus's Q row) -
            // nonsingular, but their elimination cancels to a pivot cuDSS's static pivoting cannot
            // survive (rte6515 full busbar structure: the repaired system solved to 9e-13 by a pivoted
            // LU, to 2.5e6 by cuDSS). So: a dummy row prefers the endpoint with the FEWEST incident
            // tree couplers (a leaf is unique to its coupler) and, among equals, the one no other
            // dummy row has taken; a bus row prefers a ZERO row no other bus row has taken.
            std::map<int, int> ziDeg;                        // bus row -> incident tree couplers
            for (int e = 0; e < nZeroImpedanceBranch; e++) {
                const int* z = zeroImpedanceBranchIndices + static_cast<size_t>(e) * 11;
                if (!ziSpan(z) || ziDeadSide(z) != 0) continue;
                for (int slot : {z[0], z[2], z[1] >= 0 ? z[1] : z[6], z[3] >= 0 ? z[3] : z[7]}) {
                    if (slot >= 0) ziDeg[rowMap[slot]]++;
                }
            }
            std::map<int, int> used;
            for (size_t i = 0; i < hazRows.size(); i++) {
                auto& v = hazDonors[i];
                std::stable_sort(v.begin(), v.end(), [&](int a, int b) {
                    int da = ziDeg.count(a) ? ziDeg[a] : 0;
                    int db = ziDeg.count(b) ? ziDeg[b] : 0;
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
            for (size_t i = 0; i < hazRows.size(); i++) {
                for (int d : hazDonors[i]) {
                    for (int k = rp[d]; k < rp[d + 1]; k++) {
                        pat.emplace_back(hazRows[i], sorted[k].second);
                    }
                    pat.emplace_back(d, hazRows[i]);         // the coefficient the numeric choice reads
                }
            }
        }
    }
    for (int r = 0; r < n; r++) {
        pat.emplace_back(r, r);                              // guaranteed diagonal: the per-solve
    }                                                        // row-mode mask writes 1.0 there
    std::sort(pat.begin(), pat.end());
    pat.erase(std::unique(pat.begin(), pat.end()), pat.end());
    st->nnz = static_cast<int>(pat.size());
    if (st->nnz == 0) throw std::runtime_error("empty Jacobian pattern");

    st->rowPtr.assign(n + 1, 0);
    st->colIdx.resize(st->nnz);
    for (int i = 0; i < st->nnz; i++) {                      // pat is row-major sorted
        st->colIdx[i] = pat[i].second;
        st->rowPtr[pat[i].first + 1]++;
    }
    for (int r = 0; r < n; r++) st->rowPtr[r + 1] += st->rowPtr[r];
    CsrPattern pos{st->rowPtr.data(), st->colIdx.data()};    // (row,col) -> slot by binary search
    st->nHaz = static_cast<int>(hazRows.size());
    if (st->nHaz > 0) {
        std::vector<int> hazOff(st->nHaz + 1, 0);
        std::vector<int> donorPtr(st->nHaz + 1, 0);
        std::vector<int> donor;
        std::vector<int> donorHaz;
        std::map<int, int> hazSlot;
        for (int i = 0; i < st->nHaz; i++) hazSlot[hazRows[i]] = i;
        for (int i = 0; i < st->nHaz; i++) {
            hazOff[i + 1] = hazOff[i] + (st->rowPtr[hazRows[i] + 1] - st->rowPtr[hazRows[i]]);
            for (int d : hazDonors[i]) {
                donor.push_back(d);
                auto it = hazSlot.find(d);
                donorHaz.push_back(it == hazSlot.end() ? -1 : it->second);
            }
            donorPtr[i + 1] = static_cast<int>(donor.size());
        }
        st->hazTot = hazOff[st->nHaz];
        st->hHazRow = hazRows;
        st->dHazRow.upload(hazRows.data(), hazRows.size() * sizeof(int));
        st->dHazOff.upload(hazOff.data(), hazOff.size() * sizeof(int));
        st->dHazDonorPtr.upload(donorPtr.data(), donorPtr.size() * sizeof(int));
        st->dHazDonor.upload(donor.data(), donor.size() * sizeof(int));
        st->dHazDonorHaz.upload(donorHaz.data(), donorHaz.size() * sizeof(int));
        st->dHazSign.upload(hazSigns.data(), hazSigns.size() * sizeof(char));
        if (std::getenv("OLF_GPU_TRACE") != nullptr) {
            std::fprintf(stderr, "zero-diagonal repair: %d hazard rows, %zu donor candidates\n", st->nHaz, donor.size());
        }
    }
    st->dRowPtr.upload(st->rowPtr.data(), (n + 1) * sizeof(int));
    st->dColIdx.upload(st->colIdx.data(), st->nnz * sizeof(int));

    // ---- family device data (inputs, indices, role-folded destinations) + static trig ----
    setupFamily(st->closedBranch, nClosedBranch, closedBranchParameters, 8, closedBranchIndices, 12, CB_SCATTER, CB_NOUT, resolveClosedBranch, pos);
    if (dynReloc && nClosedBranch) {                                   // Dynamic Q-relocation: alt-dest + frozen-row select map
        std::vector<int> altDest;
        std::vector<int> selRow;
        std::vector<int> relocGroup;
        std::vector<int> relocOff;
        std::vector<int> relocPool;
        std::vector<char> relocDep;
        buildCbDynDest(nClosedBranch, closedBranchIndices, relocOwn, pos, altDest, selRow, st->nQr, st->hQrPtr.data(), st->hQrRows.data(),
                relocGroup, relocOff, relocPool, depGroupOfRow, relocDep);
        st->closedBranch.alternateDestination.upload(altDest.data(), altDest.size() * sizeof(int));
        st->closedBranch.selectorRow.upload(selRow.data(), selRow.size() * sizeof(int));
        if (!relocPool.empty()) {
            st->closedBranch.relocGroup.upload(relocGroup.data(), relocGroup.size() * sizeof(int));
            st->closedBranch.relocOff.upload(relocOff.data(), relocOff.size() * sizeof(int));
            st->closedBranch.relocPool.upload(relocPool.data(), relocPool.size() * sizeof(int));
            st->closedBranch.relocDep.upload(relocDep.data(), relocDep.size());
            static int cbRoles[CB_NOUT];
            for (int k = 0; k < CB_NOUT; k++) {
                cbRoles[k] = CB_SCATTER[k].role;
            }
            patchFrozenGroupDests(st->closedBranch.destination, relocGroup, relocOff, CB_NOUT, cbRoles,
                    st->hQrPtr.data(), st->hQrRows.data());
        }
        st->closedBranch.dynReloc = true;
    }
    if (hasPhaseControl) {                                          // continuous phase control: P-flow row at a1
        std::vector<int> phaseControlDestination;
        std::vector<int> phaseControlSelector;
        buildClosedBranchPhaseControlDestination(nClosedBranch, closedBranchIndices, closedBranchPhaseControlSide, rowMap, pos, phaseControlDestination, phaseControlSelector);
        st->closedBranch.phaseControlDestination.upload(phaseControlDestination.data(), phaseControlDestination.size() * sizeof(int));
        st->closedBranch.phaseControlSelector.upload(phaseControlSelector.data(), phaseControlSelector.size() * sizeof(int));
        st->closedBranch.hasPhaseControl = true;
    }
    setupFamily(st->openBranch, nOpenBranch, openBranchParameters, 7, openBranchIndices, 6, OB_SCATTER, OB_NOUT, resolveOpenBranch, pos);
    if (dynReloc && nOpenBranch) {                                   // ob dynamic Q-routing: alt-dest + selector
        std::vector<int> obAltDest;
        std::vector<int> obSelRow;
        std::vector<int> obRelocGroup;
        std::vector<int> obRelocOff;
        std::vector<int> obRelocPool;
        std::vector<char> obRelocDep;
        buildObDynDest(nOpenBranch, openBranchIndices, openBranchAlternateReactiveDestination, relocOwn, pos, obAltDest, obSelRow, st->nQr, st->hQrPtr.data(), st->hQrRows.data(),
                obRelocGroup, obRelocOff, obRelocPool, depGroupOfRow, obRelocDep);
        st->openBranch.alternateDestination.upload(obAltDest.data(), obAltDest.size() * sizeof(int));
        st->openBranch.selectorRow.upload(obSelRow.data(), obSelRow.size() * sizeof(int));
        if (!obRelocPool.empty()) {
            st->openBranch.relocGroup.upload(obRelocGroup.data(), obRelocGroup.size() * sizeof(int));
            st->openBranch.relocOff.upload(obRelocOff.data(), obRelocOff.size() * sizeof(int));
            st->openBranch.relocPool.upload(obRelocPool.data(), obRelocPool.size() * sizeof(int));
            st->openBranch.relocDep.upload(obRelocDep.data(), obRelocDep.size());
            static int obRoles[OB_NOUT];
            for (int k = 0; k < OB_NOUT; k++) {
                obRoles[k] = OB_SCATTER[k].role;
            }
            patchFrozenGroupDests(st->openBranch.destination, obRelocGroup, obRelocOff, OB_NOUT, obRoles,
                    st->hQrPtr.data(), st->hQrRows.data());
        }
        st->openBranch.dynReloc = true;
    }
    setupFamily(st->shunt, nShunt, shuntParameters, 2, shuntIndices, 5, SH_SCATTER, SH_NOUT, resolveShunt, pos);
    if (dynReloc && nShunt) {                                   // Dynamic Q-relocation: shunt alt-dest + selector
        std::vector<int> shAltDest;
        std::vector<int> shSelRow;
        std::vector<int> shRelocGroup;
        std::vector<int> shRelocOff;
        std::vector<int> shRelocPool;
        std::vector<char> shRelocDep;
        buildShDynDest(nShunt, shuntIndices, relocOwn, pos, shAltDest, shSelRow, st->nQr, st->hQrPtr.data(), st->hQrRows.data(),
                shRelocGroup, shRelocOff, shRelocPool, depGroupOfRow, shRelocDep);
        st->shunt.alternateDestination.upload(shAltDest.data(), shAltDest.size() * sizeof(int));
        st->shunt.selectorRow.upload(shSelRow.data(), shSelRow.size() * sizeof(int));
        if (!shRelocPool.empty()) {
            st->shunt.relocGroup.upload(shRelocGroup.data(), shRelocGroup.size() * sizeof(int));
            st->shunt.relocOff.upload(shRelocOff.data(), shRelocOff.size() * sizeof(int));
            st->shunt.relocPool.upload(shRelocPool.data(), shRelocPool.size() * sizeof(int));
            st->shunt.relocDep.upload(shRelocDep.data(), shRelocDep.size());
            static int shRoles[SH_NOUT];
            for (int k = 0; k < SH_NOUT; k++) {
                shRoles[k] = SH_SCATTER[k].role;
            }
            patchFrozenGroupDests(st->shunt.destination, shRelocGroup, shRelocOff, SH_NOUT, shRoles,
                    st->hQrPtr.data(), st->hQrRows.data());
        }
        st->shunt.dynReloc = true;
    }
    setupFamily(st->hvdcAcEmulation, nHvdcAcEmulation, hvdcAcEmulationParameters, 5, hvdcAcEmulationIndices, 4, HVDC_SCATTER, HVDC_NOUT, resolveHvdcAcEmulation, pos);
    if (dynReloc && nHvdcAcEmulation) {                                   // a converter bus under a coupler the scenario opens
        std::vector<int> hvAltDest;
        std::vector<int> hvSelRow;
        buildHvDynDest(nHvdcAcEmulation, hvdcAcEmulationIndices, relocOwn, pos, hvAltDest, hvSelRow);
        st->hvdcAcEmulation.alternateDestination.upload(hvAltDest.data(), hvAltDest.size() * sizeof(int));
        st->hvdcAcEmulation.selectorRow.upload(hvSelRow.data(), hvSelRow.size() * sizeof(int));
        st->hvdcAcEmulation.dynReloc = true;
    }

    // ---- DISTR_Q destinations: per contribution {F-row, dq/dv1, dq/dv2, dq/dph1, dq/dph2, dq/dr1}
    //      (-2-row encodes an F row, >=0 a J slot, -1 skip) ----
    buildDistrQDest(st.get(), pos, rowMap, nDistrQ, distrQElement, distrQSide, distrQKind, distrQWeight, distrQRow, closedBranchIndices, openBranchIndices, shuntIndices, relocOwn);

    // ---- zero-impedance destinations: per branch the 8 dummy couplings + (tree) the 6 ZERO_V/ZERO_PHI ----
    buildZeroImpDest(st.get(), pos, rowMap, nZeroImpedanceBranch, zeroImpedanceBranchParameters, zeroImpedanceBranchIndices, zeroImpedanceBranchAlternateReactiveDestination, relocOwn, depGroupOfRow);

    // ---- DISTR_RHO destinations: per contribution {F row, J slot (row, r1 col)} ----
    buildDistrRhoDest(st.get(), pos, rowMap, nDistrRho, distrRhoRow, distrRhoColumn, distrRhoCoefficient);

    // ---- DISTR_SHUNT_B destinations: per contribution {F row, J slot (row, b col)} ----
    buildDistrShuntBDest(st.get(), pos, rowMap, nDistrShuntB, distrShuntBRow, distrShuntBColumn, distrShuntBCoefficient);

    // ---- voltage-dependent load (ZIP/exponential) destinations: one J slot (row, own-V) per term ----
    st->nLoadModelTerms = nLoadModelTerms;
    if (nLoadModelTerms > 0) {
        std::vector<int> loadModelDestination(nLoadModelTerms);
        for (int e = 0; e < nLoadModelTerms; e++) {
            loadModelDestination[e] = pos.at(loadModelIndices[e * 2], loadModelIndices[e * 2 + 1]);
        }
        st->dLoadModelParameters.upload(loadModelParameters, static_cast<size_t>(nLoadModelTerms) * 3 * sizeof(double));
        st->dLoadModelIndices.upload(loadModelIndices, static_cast<size_t>(nLoadModelTerms) * 2 * sizeof(int));
        st->dLoadModelDestination.upload(loadModelDestination.data(), static_cast<size_t>(nLoadModelTerms) * sizeof(int));
        std::vector<int> loadModelAlternateRow(nLoadModelTerms, -1);
        std::vector<int> loadModelAlternateDestination(nLoadModelTerms, -1);
        bool anyAlt = false;
        for (int e = 0; e < nLoadModelTerms; e++) {
            int row = loadModelIndices[e * 2];
            if (relocOwn != nullptr && relocOwn[row] >= 0) {
                loadModelAlternateRow[e] = relocOwn[row];
                loadModelAlternateDestination[e] = pos.at(relocOwn[row], loadModelIndices[e * 2 + 1]);
                anyAlt = true;
            }
        }
        st->loadModelRelocated = anyAlt;
        if (anyAlt) {
            st->dLoadModelAlternateRow.upload(loadModelAlternateRow.data(), static_cast<size_t>(nLoadModelTerms) * sizeof(int));
            st->dLoadModelAlternateDestination.upload(loadModelAlternateDestination.data(), static_cast<size_t>(nLoadModelTerms) * sizeof(int));
        }
    }

    std::vector<int> diagPos(n);
    for (int r = 0; r < n; r++) diagPos[r] = pos.at(r, r);
    st->dDiagPos.upload(diagPos.data(), n * sizeof(int));
    CUDA_CHECK(cudaDeviceSynchronize());                     // every structure upload done before first use
    return st.release();
}

constexpr double LU_FILL = 12.0;     // heuristic LU-fill multiplier over nnz (fallback when LU_NNZ is unavailable)

// Per-scenario bytes the LU numeric factors occupy. The exact fill-in is only known after a
// symbolic ANALYSIS, so run a CHEAP one on a single matrix with the structure's device pattern
// and read cuDSS's reported CUDSS_DATA_LU_NNZ; fall back to the LU_FILL heuristic if that is
// unavailable (older cuDSS, query failure). The result is CACHED on the structure, so repeated
// chunk-size queries (one per SA run) probe only once per extraction.
double luBytesPerScenario(SharedStructure& st) {
    if (st.luBytes > 0) {
        return st.luBytes;
    }
    int n = st.n;
#if OLF_GPU_CUDSS_LU_NNZ
    try {
        DeviceBuf dVals;
        DeviceBuf dB;
        DeviceBuf dX;
        dVals.alloc(st.nnz * sizeof(double));                // ANALYSIS is symbolic — values unused
        CUDA_CHECK(cudaMemset(dVals.p, 0, st.nnz * sizeof(double)));
        dB.alloc(n * sizeof(double));
        dX.alloc(n * sizeof(double));
        CudssCtx ds;
        CUDSS_CHECK(cudssCreate(&ds.handle));
        CUDSS_CHECK(cudssConfigCreate(&ds.config));
        applyReorderAlg(ds.config, st.nZeroImpedanceBranch > 0);
        CUDSS_CHECK(cudssDataCreate(ds.handle, &ds.data));
        CUDSS_CHECK(cudssMatrixCreateCsr(&ds.matA, n, n, st.nnz, st.dRowPtr.as<int>(), nullptr,
                                         st.dColIdx.as<int>(), dVals.as<double>(), OLF_CUDSS_CSR_TYPES,
                                         CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        CUDSS_CHECK(cudssMatrixCreateDn(&ds.matB, n, 1, n, dB.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssMatrixCreateDn(&ds.matX, n, 1, n, dX.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        gAnalysesMain++;
        { AnalysisTimer at_(__LINE__); CUDSS_CHECK(cudssExecute(ds.handle, CUDSS_PHASE_ANALYSIS, ds.config, ds.data, ds.matA, ds.matX, ds.matB)); }
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
            if (cudssDataGet(ds.handle, ds.data, CUDSS_DATA_MEMORY_ESTIMATES,
                             est, sizeof(est), &writtenEst) == CUDSS_STATUS_SUCCESS
                    && writtenEst >= 2 * sizeof(int64_t)) {
                int64_t deviceBytes = std::max(est[0], est[1]);
                if (deviceBytes > 0) {
                    st.luBytes = static_cast<double>(deviceBytes);
                    return st.luBytes;
                }
            }
        }
        int64_t luNnz = 0;
        size_t written = 0;
        CUDSS_CHECK(cudssDataGet(ds.handle, ds.data, CUDSS_DATA_LU_NNZ, &luNnz, sizeof(luNnz), &written));
        if (luNnz > 0) {
            st.luBytes = static_cast<double>(luNnz) * sizeof(double);
            return st.luBytes;
        }
    } catch (...) {
        // fall through to the heuristic
    }
#endif
    st.luBytes = LU_FILL * static_cast<double>(st.nnz) * sizeof(double);
    return st.luBytes;
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
void refreshStructureValues(SharedStructure& st, int nClosedBranch, const double* closedBranchParameters, int nOpenBranch, const double* openBranchParameters,
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
    refresh(st.closedBranch, nClosedBranch, closedBranchParameters);
    refresh(st.openBranch, nOpenBranch, openBranchParameters);
    refresh(st.shunt, nShunt, shuntParameters);
    refresh(st.hvdcAcEmulation, nHvdcAcEmulation, hvdcAcEmulationParameters);
    if (nZeroImpedanceBranch != st.nZeroImpedanceBranch) {
        throw std::runtime_error("refreshStructureValues: zero-impedance count changed (not a VALUES event)");
    }
    if (nZeroImpedanceBranch) {
        CUDA_CHECK(cudaMemcpy(st.dZeroImpedanceBranchParameters.p, zeroImpedanceBranchParameters, static_cast<size_t>(nZeroImpedanceBranch) * sizeof(double), cudaMemcpyHostToDevice));
    }
}

}  // namespace olfgpu
