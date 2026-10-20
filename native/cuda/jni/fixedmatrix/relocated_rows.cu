// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// docs/gpu-vs-olf.md section 2.3 (equations placed on another bus's row): the relocated rows of the shared
// structure (buildStructure), in device row space - the coupler-tree relocation (a relocated child's own row) and the
// pilot Q carrier candidate tables. See relocated_rows.cuh.
#include <cstddef>
#include <vector>

#include "../structure/shared_structure.cuh"
#include "relocated_rows.cuh"

namespace olfgpu {

// COUPLER-TREE relocation, per-scenario placement: map each coupler DUMMY row that hosts a child bus's
// relocated P/Q equation to that bus's OWN row, so a scenario OPENING the coupler can put the equation
// back where the stock layout has it (see stockPlacementAlternateRow). zeroImpedanceBranchIndices is not row-mapped and its slots 4..9 are
// variable COLUMNS, so both sides go through rowMap. Empty unless rows actually moved, which keeps every
// existing structure bit-identical. Uploaded to structure->dRelocatedChildOwnRow when anyRelocatedChild.
std::vector<int> buildRelocatedChildOwnRows(SharedStructure* structure, int n, int nZeroImpedanceBranch, const int* zeroImpedanceBranchIndices,
                                            const std::vector<int>& rowMap, bool& anyRelocatedChild) {
    std::vector<int> relocatedChildOwnRows(static_cast<size_t>(n), -1);
    anyRelocatedChild = false;
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
    if (anyRelocatedChild) {                                       // the kernels that divert per scenario need it
        structure->dRelocatedChildOwnRow.upload(relocatedChildOwnRows.data(), relocatedChildOwnRows.size() * sizeof(int));
    }
    return relocatedChildOwnRows;
}

// The pilot Q carrier candidate tables (GROUP relocation, plan-A6) in device row space, on the host and the device;
// returns whether any row is a dependent pilot's (hReactiveRelocationDependentGroupOfRow then holds their groups).
bool storeReactiveRelocationTables(SharedStructure* structure, int n, int nReactiveRelocationGroups, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRowsIn,
                                   const int* reactiveRelocationDependentBaseGroupIn, const int* reactiveRelocationBaseTargetRowIn, const int* reactiveRelocationDependentGroupOfRowIn,
                                   const std::vector<int>& rowMap) {
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
    return anyDependentGroup;
}

}  // namespace olfgpu
