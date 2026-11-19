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

// The elements of one family whose reactive outputs relocate, with their row indices already mapped.
enum class RelocatedFamily { CLOSED_BRANCH, OPEN_BRANCH, SHUNT };

struct RelocatedElements {
    RelocatedFamily family;
    int count;
    const int* indices;
    const int* alternateReactiveDestination;  // open branches: their alternate rows, two per element; unused otherwise
};

// The structure-wide tables an output's placement reads.
struct RelocationTables {
    const int* relocatedChildOwnRow;   // per row: a coupler DUMMY row's relocated child's own row (nullptr: none)
    const int* groupStart;             // per relocation group: its first candidate
    const int* candidateRows;          // [fallback row, member rows...] per group
    const int* dependentGroupOfRow;    // per row: the dependent group of its owner bus (nullptr: none)
};

// Per output of the family (count * output count, like the frozen destination): the alternate destination, the row
// its per-scenario mode is tested on (-1: never relocates), the relocation group, the offset of its candidates in
// relocationCandidateDestinations (negative: a stock pair) and the dependent-pilot flag.
struct RelocationDestinations {
    std::vector<int> alternateDestination;
    std::vector<int> selectorRow;
    std::vector<int> relocationGroup;
    std::vector<int> relocationCandidateOffset;
    std::vector<int> relocationCandidateDestinations;
    std::vector<char> relocationDependentPilot;
};

// Whether some element carries an alternate reactive row (a plain row or a relocation group).
bool hasRelocatableOutput(const RelocatedElements& elements);

// Unions into the pattern every row a relocated output may land on.
void collectRelocationPattern(const RelocatedElements& elements, const RelocationTables& tables, std::vector<std::pair<int, int>>& pat);

// Builds the destinations, uploads them into the family's device data and turns its dynamic relocation on.
RelocationDestinations installRelocationDestinations(const RelocatedElements& elements, const RelocationTables& tables,
                                                     const CsrPattern& pos, Family& target);

void collectHvdcAcEmulationRelocationPattern(int nHvdcAcEmulation, const int* hvdcAcEmulationIndices, const int* relocatedChildOwnRow, std::vector<std::pair<int, int>>& pat);

void buildHvdcAcEmulationRelocationDestination(int nHvdcAcEmulation, const int* hvdcAcEmulationIndices, const int* relocatedChildOwnRow, const CsrPattern& pos,
                    std::vector<int>& alternateDestination, std::vector<int>& selectorRow);

}  // namespace olfgpu

#endif  // OLF_GPU_FIXEDMATRIX_RELOCATION_PATTERN_CUH
