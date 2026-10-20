// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// docs/gpu-vs-olf.md section 2.3 (equations placed on another bus's row): the union pattern and the per-scenario
// destinations of the relocated outputs (relocation_pattern.cu).
#ifndef OLF_GPU_FIXEDMATRIX_RELOCATION_PATTERN_CUH
#define OLF_GPU_FIXEDMATRIX_RELOCATION_PATTERN_CUH

#include <utility>
#include <vector>

#include "../structure/structure_build.cuh"

namespace olfgpu {

bool hasAlternateRows(const int* a, int len);

bool hasRelocatable(int nClosedBranch, const int* closedBranchIndices, int nShunt, const int* shuntIndices);

void collectClosedBranchRelocationPattern(int nClosedBranch, const int* closedBranchIndices, const int* relocatedChildOwnRow, std::vector<std::pair<int, int>>& pat,
                         int nReactiveRelocationGroups, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows, const int* reactiveRelocationDependentGroupOfRow);

void buildClosedBranchRelocationDestination(int nClosedBranch, const int* closedBranchIndices, const int* relocatedChildOwnRow, const CsrPattern& pos,
                    std::vector<int>& alternateDestination, std::vector<int>& selectorRow,
                    int nReactiveRelocationGroups, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows,
                    std::vector<int>& relocationGroup, std::vector<int>& relocationCandidateOffset, std::vector<int>& relocationCandidateDestinations,
                    const int* reactiveRelocationDependentGroupOfRow, std::vector<char>& relocationDependentPilot);

void collectShuntRelocationPattern(int nShunt, const int* shuntIndices, const int* relocatedChildOwnRow, std::vector<std::pair<int, int>>& pat,
                         int nReactiveRelocationGroups, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows, const int* reactiveRelocationDependentGroupOfRow);

void buildShuntRelocationDestination(int nShunt, const int* shuntIndices, const int* relocatedChildOwnRow, const CsrPattern& pos,
                    std::vector<int>& alternateDestination, std::vector<int>& selectorRow,
                    int nReactiveRelocationGroups, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows,
                    std::vector<int>& relocationGroup, std::vector<int>& relocationCandidateOffset, std::vector<int>& relocationCandidateDestinations,
                    const int* reactiveRelocationDependentGroupOfRow, std::vector<char>& relocationDependentPilot);

void collectHvdcAcEmulationRelocationPattern(int nHvdcAcEmulation, const int* hvdcAcEmulationIndices, const int* relocatedChildOwnRow, std::vector<std::pair<int, int>>& pat);

void buildHvdcAcEmulationRelocationDestination(int nHvdcAcEmulation, const int* hvdcAcEmulationIndices, const int* relocatedChildOwnRow, const CsrPattern& pos,
                    std::vector<int>& alternateDestination, std::vector<int>& selectorRow);

void collectOpenBranchRelocationPattern(int nOpenBranch, const int* openBranchIndices, const int* openBranchAlternateReactiveDestination, const int* relocatedChildOwnRow, std::vector<std::pair<int, int>>& pat,
                         int nReactiveRelocationGroups, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows, const int* reactiveRelocationDependentGroupOfRow);

void buildOpenBranchRelocationDestination(int nOpenBranch, const int* openBranchIndices, const int* openBranchAlternateReactiveDestination, const int* relocatedChildOwnRow, const CsrPattern& pos,
                    std::vector<int>& alternateDestination, std::vector<int>& selectorRow,
                    int nReactiveRelocationGroups, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows,
                    std::vector<int>& relocationGroup, std::vector<int>& relocationCandidateOffset, std::vector<int>& relocationCandidateDestinations,
                    const int* reactiveRelocationDependentGroupOfRow, std::vector<char>& relocationDependentPilot);

void patchGroupFrozenDestinations(DeviceBuf& dest, const std::vector<int>& relocationGroup,
                           const std::vector<int>& relocationCandidateOffset, int outputCount,
                           const int* scatterRole, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows);

}  // namespace olfgpu

#endif  // OLF_GPU_FIXEDMATRIX_RELOCATION_PATTERN_CUH
