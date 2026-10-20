// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Destinations of the distribution rows: DISTR_Q (remote generator voltage control), DISTR_RHO (shared
// transformer voltage control) and DISTR_SHUNT_B (shared shunt voltage control).
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

#include "../equations/generated/closed_branch_scatter.h"
#include "../equations/generated/open_branch_scatter.h"
#include "../equations/generated/shunt_scatter.h"
#include "../equations/generated/hvdc_scatter.h"

#include "../common/check.cuh"
#include "../common/stats.cuh"
#include "../common/codes/gpu_reactive_term_kind.cuh"
#include "../common/codes/gpu_two_sides.cuh"
#include "../solver/linear_solver_settings.cuh"
#include "structure_build.cuh"

using namespace powsybl::openloadflow;

namespace olfgpu {

void buildDistrQDest(SharedStructure* structure, const CsrPattern& pos, const std::vector<int>& rowMap,
                     int nDistrQ, const int* distrQElement, const int* distrQSide, const int* distrQKind,
                     const double* distrQWeight, const int* distrQRow,
                     const int* closedBranchIndices, const int* openBranchIndices, const int* shuntIndices, const int* relocatedChildOwnRow) {
    structure->nDistrQ = nDistrQ;
    if (nDistrQ > 0) {
        std::vector<int> distrQDestination(static_cast<size_t>(nDistrQ) * 6, -1);
        // The COUPLER-TREE stock placement for a DISTR_Q row: a relocated child's sharing equation sits on
        // its parent coupler's DUMMY_Q row, and a scenario that kills that coupler pins the row to an
        // identity — the equation then has to be back on the bus's OWN row, exactly where OLF keeps a
        // controller's DISTR_Q. distrQAlternateDestination holds the same six destinations recomputed there and distrQSelectorRow the dummy
        // row whose per-scenario mode decides (0 = dead). Without it the middle-section-loss fixtures
        // converged on a system one equation short: bus vl3_b 395.43 kV against OLF's 401.69.
        std::vector<int> distrQAlternateDestination(static_cast<size_t>(nDistrQ) * 6, -1);
        std::vector<int> distrQSelectorRow(static_cast<size_t>(nDistrQ), -1);
        // A kind-3 contribution is a COUPLER's DUMMY_Q inside this controller's reactive injection; the pack
        // carries the dummy COLUMN, and the per-scenario mode that says whether that coupler is still alive
        // lives at its ROW. Keep the mapped row per contribution so the fill can ask.
        std::vector<int> distrQDummyQRow(static_cast<size_t>(nDistrQ), -1);
        for (int c = 0; c < nDistrQ; c++) {
            int mappedRow = rowMap[distrQRow[c]];
            int ownRow = relocatedChildOwnRow != nullptr && mappedRow >= 0 ? relocatedChildOwnRow[mappedRow] : -1;
            if (ownRow >= 0) {
                distrQSelectorRow[c] = mappedRow;
            }
            if (distrQKind[c] == GpuReactiveTermKind::DUMMY_Q && distrQElement[c] >= 0) {
                distrQDummyQRow[c] = rowMap[distrQElement[c]];
            }
            for (int pass = 0; pass < 2; pass++) {           // pass 0 = the real row, pass 1 = the stock row
                int row = pass == 0 ? mappedRow : ownRow;
                int* d = (pass == 0 ? distrQDestination : distrQAlternateDestination).data() + static_cast<size_t>(c) * 6;
                if (row < 0) {
                    continue;
                }
                d[0] = -2 - row;                             // q value -> F row
                if (distrQKind[c] == GpuReactiveTermKind::SHUNT_COMPENSATOR_REACTIVE_FLOW) {  // shunt: dq/dv on the bus V column
                    d[1] = pos.at(row, shuntIndices[static_cast<size_t>(distrQElement[c]) * 5 + 2]);
                } else if (distrQKind[c] == GpuReactiveTermKind::OPEN_BRANCH_REACTIVE_FLOW) {  // half-open: dq/dv on the connected-side V col
                    const int* openBranch = openBranchIndices + static_cast<size_t>(distrQElement[c]) * 6;
                    d[1] = pos.at(row, distrQSide[c] == GpuTwoSides::ONE ? openBranch[4] : openBranch[5]);
                } else if (distrQKind[c] == GpuReactiveTermKind::DUMMY_Q) {  // zero-impedance: J slot on the dummyQ column
                    d[1] = pos.at(row, distrQElement[c]);
                } else {                                     // closed branch
                    const int* q = closedBranchIndices + static_cast<size_t>(distrQElement[c]) * 12;
                    d[1] = pos.at(row, q[5]);                // dq/dv1
                    d[2] = pos.at(row, q[7]);                // dq/dv2
                    d[3] = pos.at(row, q[4]);                // dq/dph1
                    d[4] = pos.at(row, q[6]);                // dq/dph2
                    d[5] = q[9] >= 0 ? pos.at(row, q[9]) : -1;   // dq/dr1
                }
            }
        }
        structure->dDistrQElement.upload(distrQElement, nDistrQ * sizeof(int));
        structure->dDistrQSide.upload(distrQSide, nDistrQ * sizeof(int));
        structure->dDistrQKind.upload(distrQKind, nDistrQ * sizeof(int));
        structure->dDistrQWeight.upload(distrQWeight, nDistrQ * sizeof(double));
        structure->dDistrQDestination.upload(distrQDestination.data(), distrQDestination.size() * sizeof(int));
        structure->dDistrQAlternateDestination.upload(distrQAlternateDestination.data(), distrQAlternateDestination.size() * sizeof(int));
        structure->dDistrQSelectorRow.upload(distrQSelectorRow.data(), distrQSelectorRow.size() * sizeof(int));
        structure->dDistrQDummyQRow.upload(distrQDummyQRow.data(), distrQDummyQRow.size() * sizeof(int));
    }
}

void buildDistrRhoDest(SharedStructure* structure, const CsrPattern& pos, const std::vector<int>& rowMap,
                       int nDistrRho, const int* distrRhoRow, const int* distrRhoColumn, const double* distrRhoCoefficient) {
    structure->nDistrRho = nDistrRho;
    if (nDistrRho > 0) {
        std::vector<int> distrRhoDestination(static_cast<size_t>(nDistrRho) * 2);
        std::vector<int> distrRhoRelocationGroup(nDistrRho, -1);
        for (int c = 0; c < nDistrRho; c++) {
            int mappedRow = rowMap[distrRhoRow[c]];
            distrRhoDestination[c * 2 + 0] = -2 - mappedRow;              // DISTR_RHO F row
            distrRhoDestination[c * 2 + 1] = pos.at(mappedRow, distrRhoColumn[c]); // J slot
            // link the row to its pilot-Q relocation group (a member rho row is a candidate): when
            // the free role RE-PICKS to this row per scenario, it hosts the relocated pilot Q and
            // its DISTR_RHO emissions must stop — the SEQUENTIAL fill suppresses in-kernel by the
            // same mode-scan the scatter uses (the batch machine instead pushes zero coefficient overrides).
            for (int g = 0; g < structure->nReactiveRelocationGroups && distrRhoRelocationGroup[c] < 0; g++) {
                for (int k = structure->hReactiveRelocationGroupStart[g] + 1; k < structure->hReactiveRelocationGroupStart[g + 1]; k++) {
                    if (structure->hReactiveRelocationCandidateRows[k] == mappedRow) {
                        distrRhoRelocationGroup[c] = g;
                        structure->hasDistrRhoRelocationGroups = 1;
                        break;
                    }
                }
            }
        }
        structure->dDistrRhoColumn.upload(distrRhoColumn, nDistrRho * sizeof(int));
        structure->dDistrRhoCoefficient.upload(distrRhoCoefficient, nDistrRho * sizeof(double));
        structure->dDistrRhoDestination.upload(distrRhoDestination.data(), distrRhoDestination.size() * sizeof(int));
        if (structure->hasDistrRhoRelocationGroups) {
            structure->dDistrRhoRelocationGroup.upload(distrRhoRelocationGroup.data(), distrRhoRelocationGroup.size() * sizeof(int));
        }
    }
}

void buildDistrShuntBDest(SharedStructure* structure, const CsrPattern& pos, const std::vector<int>& rowMap,
                          int nDistrShuntB, const int* distrShuntBRow, const int* distrShuntBColumn, const double* distrShuntBCoefficient) {
    structure->nDistrShuntB = nDistrShuntB;
    if (nDistrShuntB > 0) {
        std::vector<int> distrShuntBDestination(static_cast<size_t>(nDistrShuntB) * 2);
        for (int c = 0; c < nDistrShuntB; c++) {
            int mappedRow = rowMap[distrShuntBRow[c]];
            distrShuntBDestination[c * 2 + 0] = -2 - mappedRow;              // DISTR_SHUNT_B F row
            distrShuntBDestination[c * 2 + 1] = pos.at(mappedRow, distrShuntBColumn[c]); // J slot
        }
        structure->dDistrShuntBColumn.upload(distrShuntBColumn, nDistrShuntB * sizeof(int));
        structure->dDistrShuntBCoefficient.upload(distrShuntBCoefficient, nDistrShuntB * sizeof(double));
        structure->dDistrShuntBDestination.upload(distrShuntBDestination.data(), distrShuntBDestination.size() * sizeof(int));
    }
}

}  // namespace olfgpu
