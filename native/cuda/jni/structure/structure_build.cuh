// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// What the structure-building files share: the generic pattern/destination templates, the stock-placement
// helpers of the Q relocation, and the functions one file calls in another.
#ifndef OLF_GPU_STRUCTURE_STRUCTURE_BUILD_CUH
#define OLF_GPU_STRUCTURE_STRUCTURE_BUILD_CUH

#include <utility>
#include <vector>

#include "closed_branch_scatter.h"
#include "open_branch_scatter.h"
#include "shunt_scatter.h"
#include "hvdc_scatter.h"

#include "csr_pattern.cuh"
#include "shared_structure.cuh"

namespace olfgpu {

using namespace powsybl::openloadflow;

// csr_pattern.cu: one scatter-table row of one element -> (equation row, variable column), absent for role 2.
RC resolveClosedBranch(const CbScatter& s, const int* ix);             // P1 Q1 P2 Q2 ph1 v1 ph2 v2 a1col r1col
RC resolveOpenBranch(const ObScatter& s, const int* ix);             // P1 Q1 P2 Q2 v1 v2
RC resolveShunt(const ShScatter& s, const int* ix);             // P Q v bcol
RC resolveHvdcAcEmulation(const HvdcScatter& s, const int* ix);           // P1 P2 ph1 ph2

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

// The alternate candidate Q row for a closed-branch output, or -1 if this output does not relocate
// (only Q rows, rowEq==1) or the endpoint bus has no second candidate (qRowAlt was -1).
// STOCK-PLACEMENT fallback for the COUPLER-TREE relocation. A child bus whose P/Q equations were relocated
// onto its parent coupler's DUMMY_P / DUMMY_Q rows must get them BACK on its own rows in any scenario that
// OPENS that coupler: the coupler's element is then skipped (no ZERO_V / ZERO_PHI, no +-dummy coupling) and
// the host pins the dummy rows with DUMMY_TARGET identities, so the relocated equations would be zeroed and
// the bus left with no balance at all. relocOwn[dummyRow] = that bus's own row.
//
// The selector is the DUMMY ROW itself, whose mode the host sets to 0 exactly when the coupler opens — but
// it must divert on mode 0 ONLY, not on mode 4: a relocated child that is RL-PINNED carries mode 4 on that
// same dummy row, where its own Q balance legitimately lives. Encoded as -2-row in selRow, which the picks
// decode as "divert only while this row is an identity".
// The DEPENDENT-PILOT group of the bus owning an output's home row (see depPick), or -1.
inline int depGroupOf(int frozenRow, int altRow, const int* relocOwn, const int* depGroupOfRow) {
    if (depGroupOfRow == nullptr) {
        return -1;
    }
    int own = altRow >= 0 ? altRow : frozenRow;
    return own >= 0 ? depGroupOfRow[own] : -1;
}

inline int stockAltRow(int frozenRow, int alt, const int* relocOwn) {
    return alt < 0 && frozenRow >= 0 && relocOwn != nullptr && relocOwn[frozenRow] >= 0
            ? relocOwn[frozenRow] : alt;
}

inline int stockSelRow(int frozenRow, int alt, const int* relocOwn) {
    return alt < 0 && frozenRow >= 0 && relocOwn != nullptr && relocOwn[frozenRow] >= 0
            ? -2 - frozenRow : frozenRow;
}

// A relocated child whose own q row is SUPPRESSED (a remote gen-VC member has no frozen dest: its q
// feeds the DISTR_Q rows) needs THREE destinations, not two:
//   the coupler's DUMMY_Q row  while a reactive limit PINS it there (that row is mode 4),
//   its OWN row                while the parent coupler is dead this scenario (the dummy row is an
//                              identity, mode 0 — the stock placement),
//   nowhere at all             while it shares (mode 1: the DISTR_Q terms carry its q).
// Two of the three cannot be expressed with one alternate and one selector, so the pair travels in the
// reloc POOL, marked by a NEGATIVE relocOff, with the DUMMY row itself in the group slot as the selector.
// Measured without it (relocation on by default): the middle-section-loss fixtures left the child's own
// row empty once the contingency killed its coupler — one zero diagonal, dx 4e+12, no convergence.
inline bool stockPairOut(int frozenRow, int alt, const int* relocOwn) {
    return frozenRow < 0 && alt >= 0 && relocOwn != nullptr && relocOwn[alt] >= 0;
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
                          int nZeroImpedanceBranch, const int* zeroImpedanceBranchIndices, const int* relocOwn,
                          std::vector<std::pair<int, int>>& pat);

// q_relocation_pattern.cu
bool hasAltEntries(const int* a, int len);

bool hasRelocatable(int nClosedBranch, const int* closedBranchIndices, int nShunt, const int* shuntIndices);

void collectClosedBranchPhaseControlPattern(int nClosedBranch, const int* closedBranchIndices, const int* closedBranchPhaseControlSide,
                           const std::vector<int>& rowMap, std::vector<std::pair<int, int>>& pat);

void buildClosedBranchPhaseControlDestination(int nClosedBranch, const int* closedBranchIndices, const int* closedBranchPhaseControlSide, const std::vector<int>& rowMap,
                      const CsrPattern& pos, std::vector<int>& phaseControlDestination, std::vector<int>& phaseControlSelector);

void collectCbAltPattern(int nClosedBranch, const int* closedBranchIndices, const int* relocOwn, std::vector<std::pair<int, int>>& pat,
                         int nQr, const int* qrPtr, const int* qrRows, const int* depGroupOfRow);

void buildCbDynDest(int nClosedBranch, const int* closedBranchIndices, const int* relocOwn, const CsrPattern& pos,
                    std::vector<int>& altDest, std::vector<int>& selRow,
                    int nQr, const int* qrPtr, const int* qrRows,
                    std::vector<int>& relocGroup, std::vector<int>& relocOff, std::vector<int>& relocPool,
                    const int* depGroupOfRow, std::vector<char>& relocDep);

void collectShAltPattern(int nShunt, const int* shuntIndices, const int* relocOwn, std::vector<std::pair<int, int>>& pat,
                         int nQr, const int* qrPtr, const int* qrRows, const int* depGroupOfRow);

void buildShDynDest(int nShunt, const int* shuntIndices, const int* relocOwn, const CsrPattern& pos,
                    std::vector<int>& altDest, std::vector<int>& selRow,
                    int nQr, const int* qrPtr, const int* qrRows,
                    std::vector<int>& relocGroup, std::vector<int>& relocOff, std::vector<int>& relocPool,
                    const int* depGroupOfRow, std::vector<char>& relocDep);

void collectHvAltPattern(int nHvdcAcEmulation, const int* hvdcAcEmulationIndices, const int* relocOwn, std::vector<std::pair<int, int>>& pat);

void buildHvDynDest(int nHvdcAcEmulation, const int* hvdcAcEmulationIndices, const int* relocOwn, const CsrPattern& pos,
                    std::vector<int>& altDest, std::vector<int>& selRow);

void collectObAltPattern(int nOpenBranch, const int* openBranchIndices, const int* openBranchAlternateReactiveDestination, const int* relocOwn, std::vector<std::pair<int, int>>& pat,
                         int nQr, const int* qrPtr, const int* qrRows, const int* depGroupOfRow);

void buildObDynDest(int nOpenBranch, const int* openBranchIndices, const int* openBranchAlternateReactiveDestination, const int* relocOwn, const CsrPattern& pos,
                    std::vector<int>& altDest, std::vector<int>& selRow,
                    int nQr, const int* qrPtr, const int* qrRows,
                    std::vector<int>& relocGroup, std::vector<int>& relocOff, std::vector<int>& relocPool,
                    const int* depGroupOfRow, std::vector<char>& relocDep);

// distribution_dests.cu
void buildDistrQDest(SharedStructure* st, const CsrPattern& pos, const std::vector<int>& rowMap,
                     int nDistrQ, const int* distrQElement, const int* distrQSide, const int* distrQKind,
                     const double* distrQWeight, const int* distrQRow,
                     const int* closedBranchIndices, const int* openBranchIndices, const int* shuntIndices, const int* relocOwn);

void buildDistrRhoDest(SharedStructure* st, const CsrPattern& pos, const std::vector<int>& rowMap,
                       int nDistrRho, const int* distrRhoRow, const int* distrRhoColumn, const double* distrRhoCoefficient);

void buildDistrShuntBDest(SharedStructure* st, const CsrPattern& pos, const std::vector<int>& rowMap,
                          int nDistrShuntB, const int* distrShuntBRow, const int* distrShuntBColumn, const double* distrShuntBCoefficient);

void patchFrozenGroupDests(DeviceBuf& dest, const std::vector<int>& relocGroup,
                           const std::vector<int>& relocOff, int outputCount,
                           const int* scatterRole, const int* qrPtr, const int* qrRows);

// zero_impedance_dests.cu
void buildZeroImpDest(SharedStructure* st, const CsrPattern& pos, const std::vector<int>& rowMap,
                      int nZeroImpedanceBranch, const double* zeroImpedanceBranchParameters, const int* zeroImpedanceBranchIndices, const int* zeroImpedanceBranchAlternateReactiveDestination,
                      const int* relocOwn, const int* depGroupOfRow);

}  // namespace olfgpu

#endif  // OLF_GPU_STRUCTURE_STRUCTURE_BUILD_CUH
