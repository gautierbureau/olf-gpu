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
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../equations/generated/closed_branch_scatter.h"
#include "../equations/generated/open_branch_scatter.h"
#include "../equations/generated/shunt_scatter.h"
#include "../equations/generated/hvdc_ac_emulation_scatter.h"

#include "../common/check.cuh"
#include "../common/stats.cuh"
#include "../common/trace.cuh"
#include "../lu/cudss_order.cuh"
#include "../lu/minimum_degree.cuh"
#include "../solver/linear_solver_settings.cuh"
#include "structure_build.cuh"
#include "../fixedmatrix/relocation_pattern.cuh"
#include "../fixedmatrix/relocated_rows.cuh"
#include "../fixedmatrix/zero_diagonal_repair_tables.cuh"

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
    bool anyRelocatedChild = false;
    std::vector<int> relocatedChildOwnRows = buildRelocatedChildOwnRows(structure.get(), n, nZeroImpedanceBranch, zeroImpedanceBranchIndices, rowMap, anyRelocatedChild);
    const int* relocatedChildOwnRow = anyRelocatedChild ? relocatedChildOwnRows.data() : nullptr;
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
    bool anyDependentGroup = storeReactiveRelocationTables(structure.get(), n, nReactiveRelocationGroups, reactiveRelocationGroupStart, reactiveRelocationCandidateRowsIn,
                                                           reactiveRelocationDependentBaseGroupIn, reactiveRelocationBaseTargetRowIn, reactiveRelocationDependentGroupOfRowIn, rowMap);
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
    collectPattern(nClosedBranch, closedBranchIndices, 12, CLOSED_BRANCH_SCATTER, CLOSED_BRANCH_OUTPUT_COUNT, resolveClosedBranch, pat);
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
    collectPattern(nOpenBranch, openBranchIndices, 6, OPEN_BRANCH_SCATTER, OPEN_BRANCH_OUTPUT_COUNT, resolveOpenBranch, pat);
    if (dynamicRelocation) {
        collectOpenBranchRelocationPattern(nOpenBranch, openBranchIndices, openBranchAlternateReactiveDestination, relocatedChildOwnRow, pat, structure->nReactiveRelocationGroups, structure->hReactiveRelocationGroupStart.data(), structure->hReactiveRelocationCandidateRows.data(), reactiveRelocationDependentGroupOfRow);
    }
    collectPattern(nShunt, shuntIndices, 5, SHUNT_SCATTER, SHUNT_OUTPUT_COUNT, resolveShunt, pat);
    if (dynamicRelocation) {
        collectShuntRelocationPattern(nShunt, shuntIndices, relocatedChildOwnRow, pat, structure->nReactiveRelocationGroups, structure->hReactiveRelocationGroupStart.data(), structure->hReactiveRelocationCandidateRows.data(), reactiveRelocationDependentGroupOfRow);
    }
    collectPattern(nHvdcAcEmulation, hvdcAcEmulationIndices, 4, HVDC_AC_EMULATION_SCATTER, HVDC_AC_EMULATION_OUTPUT_COUNT, resolveHvdcAcEmulation, pat);
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
    collectZeroDiagonalRepairRows(structure.get(), n, nZeroImpedanceBranch, zeroImpedanceBranchIndices, zeroImpedanceBranchAlternateReactiveDestination, rowMap, pat,
                                  zeroDiagonalRows, zeroDiagonalDonors, zeroDiagonalRepairSigns);
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
    storeZeroDiagonalRepairTables(structure.get(), zeroDiagonalRows, zeroDiagonalDonors, zeroDiagonalRepairSigns);
    structure->dRowPtr.upload(structure->rowPtr.data(), (n + 1) * sizeof(int));
    structure->dColIdx.upload(structure->colIdx.data(), structure->nnz * sizeof(int));

    // ---- family device data (inputs, indices, role-folded destinations) + static trig ----
    setupFamily(structure->closedBranch, nClosedBranch, closedBranchParameters, 8, closedBranchIndices, 12, CLOSED_BRANCH_SCATTER, CLOSED_BRANCH_OUTPUT_COUNT, resolveClosedBranch, pos);
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
            static int cbRoles[CLOSED_BRANCH_OUTPUT_COUNT];
            for (int k = 0; k < CLOSED_BRANCH_OUTPUT_COUNT; k++) {
                cbRoles[k] = CLOSED_BRANCH_SCATTER[k].role;
            }
            patchGroupFrozenDestinations(structure->closedBranch.destination, relocationGroup, relocationCandidateOffset, CLOSED_BRANCH_OUTPUT_COUNT, cbRoles,
                    structure->hReactiveRelocationGroupStart.data(), structure->hReactiveRelocationCandidateRows.data());
        }
        // rte6515 carries a relocatable output on 41 of its 9,042 closed branches, network.xiidm on 238 of 9,959; walking
        // every output of the others through the relocation tables to land on its frozen destination anyway was 71 % of
        // the closed-branch fill (the probe of 2026-10-30: 104 of 145 ms against the static scatter)
        std::vector<char> relocatedElement(nClosedBranch, 0);
        for (int e = 0; e < nClosedBranch; e++) {
            for (int k = 0; k < CLOSED_BRANCH_OUTPUT_COUNT; k++) {
                const int t = e * CLOSED_BRANCH_OUTPUT_COUNT + k;
                if (selectorRow[t] != -1 || (!relocationCandidateDestinations.empty() && relocationGroup[t] >= 0)) {
                    relocatedElement[e] = 1;
                }
            }
        }
        structure->closedBranch.relocatedElement.upload(relocatedElement.data(), relocatedElement.size());
        std::vector<int> relocatedElements;
        for (int e = 0; e < nClosedBranch; e++) {
            if (relocatedElement[e]) {
                relocatedElements.push_back(e);
            }
        }
        structure->closedBranch.relocatedCount = static_cast<int>(relocatedElements.size());
        if (!relocatedElements.empty()) {
            structure->closedBranch.relocatedElements.upload(relocatedElements.data(), relocatedElements.size() * sizeof(int));
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
    setupFamily(structure->openBranch, nOpenBranch, openBranchParameters, 7, openBranchIndices, 6, OPEN_BRANCH_SCATTER, OPEN_BRANCH_OUTPUT_COUNT, resolveOpenBranch, pos);
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
            static int obRoles[OPEN_BRANCH_OUTPUT_COUNT];
            for (int k = 0; k < OPEN_BRANCH_OUTPUT_COUNT; k++) {
                obRoles[k] = OPEN_BRANCH_SCATTER[k].role;
            }
            patchGroupFrozenDestinations(structure->openBranch.destination, openBranchRelocationGroup, openBranchRelocationCandidateOffset, OPEN_BRANCH_OUTPUT_COUNT, obRoles,
                    structure->hReactiveRelocationGroupStart.data(), structure->hReactiveRelocationCandidateRows.data());
        }
        structure->openBranch.dynamicRelocation = true;
    }
    setupFamily(structure->shunt, nShunt, shuntParameters, 2, shuntIndices, 5, SHUNT_SCATTER, SHUNT_OUTPUT_COUNT, resolveShunt, pos);
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
            static int shRoles[SHUNT_OUTPUT_COUNT];
            for (int k = 0; k < SHUNT_OUTPUT_COUNT; k++) {
                shRoles[k] = SHUNT_SCATTER[k].role;
            }
            patchGroupFrozenDestinations(structure->shunt.destination, shuntRelocationGroup, shuntRelocationCandidateOffset, SHUNT_OUTPUT_COUNT, shRoles,
                    structure->hReactiveRelocationGroupStart.data(), structure->hReactiveRelocationCandidateRows.data());
        }
        structure->shunt.dynamicRelocation = true;
    }
    setupFamily(structure->hvdcAcEmulation, nHvdcAcEmulation, hvdcAcEmulationParameters, 5, hvdcAcEmulationIndices, 4, HVDC_AC_EMULATION_SCATTER, HVDC_AC_EMULATION_OUTPUT_COUNT, resolveHvdcAcEmulation, pos);
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

LuPlan& luPlanOf(SharedStructure& structure) {
    static std::mutex buildMutex;                            // contexts on one structure may be built concurrently
    std::lock_guard<std::mutex> lock(buildMutex);
    if (!structure.luPlan) {
        std::vector<int> rowOrder;
        std::vector<int> colOrder;
        int64_t cudssLuNnz = 0;
        cudssEliminationOrder(structure.n, structure.nnz, structure.dRowPtr.as<int>(), structure.dColIdx.as<int>(),
                              structure.nZeroImpedanceBranch > 0, rowOrder, colOrder, cudssLuNnz);
        // the elimination order the plan is built on (BatchedLuOrdering); cuDSS's row and column orders are the same
        // permutation on our symmetric-pattern Jacobians, and the minimum degree is symmetric by construction
        const int ordering = gLinearSolver.batchedLuOrdering;
        const char* chosen = "cudss";
        int cudssLevels = -1;
        int minimumDegreeLevels = -1;
        if (ordering != BATCHED_LU_ORDERING_CUDSS) {
            std::vector<int> minimumDegree = minimumDegreeOrder(structure.n, structure.rowPtr.data(), structure.colIdx.data());
            minimumDegreeLevels = eliminationLevels(structure.n, structure.rowPtr.data(), structure.colIdx.data(), minimumDegree);
            bool take = ordering == BATCHED_LU_ORDERING_MINIMUM_DEGREE;
            if (!take) {
                cudssLevels = eliminationLevels(structure.n, structure.rowPtr.data(), structure.colIdx.data(), rowOrder);
                take = minimumDegreeLevels < cudssLevels;
            }
            if (take) {
                rowOrder = minimumDegree;
                colOrder = std::move(minimumDegree);
                chosen = "minimum-degree";
            }
        }
        std::shared_ptr<LuPlan> plan = buildLuPlan(structure.n, structure.rowPtr.data(), structure.colIdx.data(),
                                                   rowOrder.data(), colOrder.data());
        if (traceOn(Trace::TRACE)) {
            std::fprintf(stderr, "OLF_GPU_LU_PLAN n=%d nnz=%d cudssLuNnz=%lld order=%s cudssLevels=%d minimumDegreeLevels=%d %s\n",
                         plan->n, plan->rowPtr[plan->n], static_cast<long long>(cudssLuNnz), chosen, cudssLevels,
                         minimumDegreeLevels, describeLuPlan(*plan).c_str());
        }
        if (const char* dump = traceValue(Trace::LU_PATTERN_DUMP)) {
            // int32 little endian: n, nnz, nSwitchRows, rowPtr[n + 1], colIdx[nnz], rowOrder[n], colOrder[n], the
            // switches' DUMMY_P / DUMMY_Q rows[nSwitchRows] - structure only, no values
            std::vector<int> z(static_cast<size_t>(structure.nZeroImpedanceBranch) * 11);
            if (!z.empty()) {
                CUDA_CHECK(cudaMemcpy(z.data(), structure.dZeroImpedanceBranchIndices.p, z.size() * sizeof(int), cudaMemcpyDeviceToHost));
            }
            std::vector<int> switchRows;
            for (int e = 0; e < structure.nZeroImpedanceBranch; e++) {
                for (int slot : {4, 5}) {
                    if (z[static_cast<size_t>(e) * 11 + slot] >= 0) {
                        switchRows.push_back(z[static_cast<size_t>(e) * 11 + slot]);
                    }
                }
            }
            const std::string path = std::string(dump) + "-n" + std::to_string(plan->n);   // one file per structure size
            if (FILE* f = std::fopen(path.c_str(), "wb")) {
                const int head[3] = {plan->n, plan->rowPtr[plan->n], static_cast<int>(switchRows.size())};
                std::fwrite(head, sizeof(int), 3, f);
                std::fwrite(plan->rowPtr.data(), sizeof(int), plan->n + 1, f);
                std::fwrite(plan->colIdx.data(), sizeof(int), plan->rowPtr[plan->n], f);
                std::fwrite(rowOrder.data(), sizeof(int), plan->n, f);
                std::fwrite(colOrder.data(), sizeof(int), plan->n, f);
                std::fwrite(switchRows.data(), sizeof(int), switchRows.size(), f);
                std::fclose(f);
            }
        }
        structure.luPlan = std::move(plan);
    }
    return *structure.luPlan;
}

}  // namespace olfgpu
