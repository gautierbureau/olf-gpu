// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// What the structure-building files share: the generic pattern/destination templates and the functions one
// file calls in another. The stock-placement helpers of the Q relocation are in ../fixedmatrix/stock_placement.cuh.
#ifndef OLF_GPU_STRUCTURE_STRUCTURE_BUILD_CUH
#define OLF_GPU_STRUCTURE_STRUCTURE_BUILD_CUH

#include <utility>
#include <vector>

#include "../equations/generated/closed_branch_scatter.h"
#include "../equations/generated/open_branch_scatter.h"
#include "../equations/generated/shunt_scatter.h"
#include "../equations/generated/hvdc_ac_emulation_scatter.h"

#include "csr_pattern.cuh"
#include "shared_structure.cuh"

namespace olfgpu {

using namespace powsybl::openloadflow;

// csr_pattern.cu: one scatter-table row of one element -> (equation row, variable column), absent for role 2.
RC resolveClosedBranch(const ClosedBranchScatter& s, const int* ix);             // P1 Q1 P2 Q2 ph1 v1 ph2 v2 a1col r1col
RC resolveOpenBranch(const OpenBranchScatter& s, const int* ix);             // P1 Q1 P2 Q2 v1 v2
RC resolveShunt(const ShuntScatter& s, const int* ix);             // P Q v bcol
RC resolveHvdcAcEmulation(const HvdcAcEmulationScatter& s, const int* ix);           // P1 P2 ph1 ph2

// One pass per family: collect the J pattern entries (role 1, both indices present).
template <typename S, typename F>
void collectPattern(int elementCount, const int* idx, int stride, const S* table, int outputCount, F resolve,
                    std::vector<std::pair<int, int>>& pat) {
    for (int e = 0; e < elementCount; e++) {
        for (int k = 0; k < outputCount; k++) {
            RC rc = resolve(table[k], idx + e * stride);
            if (table[k].role == 1 && rc.row >= 0 && rc.col >= 0) {
                pat.emplace_back(rc.row, rc.col);
            }
        }
    }
}

// Lookup of a (row, col) entry's slot in the SORTED CSR pattern. The pattern is built by a
// std::sort of (row, col) pairs, so colIdx within each row is ascending — a binary search over
// colIdx[rowPtr[row] .. rowPtr[row+1]) resolves (row, col) -> slot with no extra memory (replaces
// a std::map<pair,int> tree, which was pure overhead on an already-sorted pattern).
// Second pass: the per-(element, output) destination with the role folded in:
//   role 1 -> CSR values slot (>= 0);  role 0 -> -2 - F row;  absent/skip -> -1.
template <typename S, typename F>
std::vector<int> buildDest(int elementCount, const int* idx, int stride, const S* table, int outputCount,
                           F resolve, const CsrPattern& pos) {
    std::vector<int> dest(static_cast<size_t>(elementCount) * outputCount, -1);
    for (int e = 0; e < elementCount; e++) {
        for (int k = 0; k < outputCount; k++) {
            RC rc = resolve(table[k], idx + e * stride);
            if (table[k].role == 1 && rc.row >= 0 && rc.col >= 0) {
                dest[static_cast<size_t>(e) * outputCount + k] = pos.at(rc.row, rc.col);
            } else if (table[k].role == 0 && rc.row >= 0) {
                dest[static_cast<size_t>(e) * outputCount + k] = -2 - rc.row;
            }
        }
    }
    return dest;
}

// Build one family's device-side data (inputs, indices, role-folded destinations).
template <typename S, typename R>
void setupFamily(Family& f, int elementCount, const double* in, int parameterStride, const int* idx,
                 int indexStride, const S* table, int outputCount, R resolve,
                 const CsrPattern& pos) {
    f.elementCount = elementCount; f.parameterStride = parameterStride; f.indexStride = indexStride; f.outputCount = outputCount;
    if (elementCount == 0) return;
    f.parameters.upload(in, static_cast<size_t>(elementCount) * parameterStride * sizeof(double));
    f.indices.upload(idx, static_cast<size_t>(elementCount) * indexStride * sizeof(int));
    std::vector<int> dest = buildDest(elementCount, idx, indexStride, table, outputCount, resolve, pos);
    f.destination.upload(dest.data(), dest.size() * sizeof(int));
}

// csr_pattern.cu
std::vector<int> buildRowMap(int n, int nClosedBranch, const int* closedBranchIndices, int nOpenBranch, const int* openBranchIndices,
                             int nShunt, const int* shuntIndices, int nHvdcAcEmulation, const int* hvdcAcEmulationIndices,
                             int nDistrQ, const int* distrQRow, int nZeroImpedanceBranch, const int* zeroImpedanceBranchIndices);

void collectFamilyPattern(const std::vector<int>& rowMap,
                          const int* closedBranchIndices, const int* openBranchIndices, const int* shuntIndices,
                          int nDistrQ, const int* distrQElement, const int* distrQSide, const int* distrQKind, const int* distrQRow,
                          int nDistrRho, const int* distrRhoRow, const int* distrRhoColumn,
                          int nDistrShuntB, const int* distrShuntBRow, const int* distrShuntBColumn,
                          int nZeroImpedanceBranch, const int* zeroImpedanceBranchIndices, const int* relocatedChildOwnRow,
                          std::vector<std::pair<int, int>>& pat);

// phase_control_pattern.cu
void collectClosedBranchPhaseControlPattern(int nClosedBranch, const int* closedBranchIndices, const int* closedBranchPhaseControlSide,
                           const std::vector<int>& rowMap, std::vector<std::pair<int, int>>& pat);

void buildClosedBranchPhaseControlDestination(int nClosedBranch, const int* closedBranchIndices, const int* closedBranchPhaseControlSide, const std::vector<int>& rowMap,
                      const CsrPattern& pos, std::vector<int>& phaseControlDestination, std::vector<int>& phaseControlSelector);

// distribution_dests.cu
void buildDistrQDest(SharedStructure* structure, const CsrPattern& pos, const std::vector<int>& rowMap,
                     int nDistrQ, const int* distrQElement, const int* distrQSide, const int* distrQKind,
                     const double* distrQWeight, const int* distrQRow,
                     const int* closedBranchIndices, const int* openBranchIndices, const int* shuntIndices, const int* relocatedChildOwnRow);

void buildDistrRhoDest(SharedStructure* structure, const CsrPattern& pos, const std::vector<int>& rowMap,
                       int nDistrRho, const int* distrRhoRow, const int* distrRhoColumn, const double* distrRhoCoefficient);

void buildDistrShuntBDest(SharedStructure* structure, const CsrPattern& pos, const std::vector<int>& rowMap,
                          int nDistrShuntB, const int* distrShuntBRow, const int* distrShuntBColumn, const double* distrShuntBCoefficient);

// zero_impedance_dests.cu
void buildZeroImpedanceBranchDestination(SharedStructure* structure, const CsrPattern& pos, const std::vector<int>& rowMap,
                      int nZeroImpedanceBranch, const double* zeroImpedanceBranchParameters, const int* zeroImpedanceBranchIndices, const int* zeroImpedanceBranchAlternateReactiveDestination,
                      const int* relocatedChildOwnRow, const int* reactiveRelocationDependentGroupOfRow);

}  // namespace olfgpu

#endif  // OLF_GPU_STRUCTURE_STRUCTURE_BUILD_CUH
