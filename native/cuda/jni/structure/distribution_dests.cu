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

#include "closed_branch_scatter.h"
#include "open_branch_scatter.h"
#include "shunt_scatter.h"
#include "hvdc_scatter.h"

#include "../common/check.cuh"
#include "../common/stats.cuh"
#include "../common/codes/gpu_reactive_term_kind.cuh"
#include "../solver/linear_solver_settings.cuh"
#include "structure_build.cuh"

using namespace powsybl::openloadflow;

namespace olfgpu {

void buildDistrQDest(SharedStructure* st, const CsrPattern& pos, const std::vector<int>& rowMap,
                     int nDistrQ, const int* distrQElement, const int* distrQSide, const int* distrQKind,
                     const double* distrQWeight, const int* distrQRow,
                     const int* closedBranchIndices, const int* openBranchIndices, const int* shuntIndices, const int* relocatedChildOwnRow) {
    st->nDistrQ = nDistrQ;
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
                    const int* oq = openBranchIndices + static_cast<size_t>(distrQElement[c]) * 6;
                    d[1] = pos.at(row, distrQSide[c] == 0 ? oq[4] : oq[5]);
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
        st->dDistrQElement.upload(distrQElement, nDistrQ * sizeof(int));
        st->dDistrQSide.upload(distrQSide, nDistrQ * sizeof(int));
        st->dDistrQKind.upload(distrQKind, nDistrQ * sizeof(int));
        st->dDistrQWeight.upload(distrQWeight, nDistrQ * sizeof(double));
        st->dDistrQDestination.upload(distrQDestination.data(), distrQDestination.size() * sizeof(int));
        st->dDistrQAlternateDestination.upload(distrQAlternateDestination.data(), distrQAlternateDestination.size() * sizeof(int));
        st->dDistrQSelectorRow.upload(distrQSelectorRow.data(), distrQSelectorRow.size() * sizeof(int));
        st->dDistrQDummyQRow.upload(distrQDummyQRow.data(), distrQDummyQRow.size() * sizeof(int));
    }
}

void buildDistrRhoDest(SharedStructure* st, const CsrPattern& pos, const std::vector<int>& rowMap,
                       int nDistrRho, const int* distrRhoRow, const int* distrRhoColumn, const double* distrRhoCoefficient) {
    st->nDistrRho = nDistrRho;
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
            for (int g = 0; g < st->nReactiveRelocationGroups && distrRhoRelocationGroup[c] < 0; g++) {
                for (int k = st->hReactiveRelocationGroupStart[g] + 1; k < st->hReactiveRelocationGroupStart[g + 1]; k++) {
                    if (st->hReactiveRelocationCandidateRows[k] == mappedRow) {
                        distrRhoRelocationGroup[c] = g;
                        st->hasDistrRhoRelocationGroups = 1;
                        break;
                    }
                }
            }
        }
        st->dDistrRhoColumn.upload(distrRhoColumn, nDistrRho * sizeof(int));
        st->dDistrRhoCoefficient.upload(distrRhoCoefficient, nDistrRho * sizeof(double));
        st->dDistrRhoDestination.upload(distrRhoDestination.data(), distrRhoDestination.size() * sizeof(int));
        if (st->hasDistrRhoRelocationGroups) {
            st->dDistrRhoRelocationGroup.upload(distrRhoRelocationGroup.data(), distrRhoRelocationGroup.size() * sizeof(int));
        }
    }
}

void buildDistrShuntBDest(SharedStructure* st, const CsrPattern& pos, const std::vector<int>& rowMap,
                          int nDistrShuntB, const int* distrShuntBRow, const int* distrShuntBColumn, const double* distrShuntBCoefficient) {
    st->nDistrShuntB = nDistrShuntB;
    if (nDistrShuntB > 0) {
        std::vector<int> distrShuntBDestination(static_cast<size_t>(nDistrShuntB) * 2);
        for (int c = 0; c < nDistrShuntB; c++) {
            int mappedRow = rowMap[distrShuntBRow[c]];
            distrShuntBDestination[c * 2 + 0] = -2 - mappedRow;              // DISTR_SHUNT_B F row
            distrShuntBDestination[c * 2 + 1] = pos.at(mappedRow, distrShuntBColumn[c]); // J slot
        }
        st->dDistrShuntBColumn.upload(distrShuntBColumn, nDistrShuntB * sizeof(int));
        st->dDistrShuntBCoefficient.upload(distrShuntBCoefficient, nDistrShuntB * sizeof(double));
        st->dDistrShuntBDestination.upload(distrShuntBDestination.data(), distrShuntBDestination.size() * sizeof(int));
    }
}

// plan-B4: the FROZEN role-0 dest of a GROUP-relocated output backs only the qBuf DIAGNOSTIC
// fill (dynamicRelocation = 0; every Newton fill selects a pool slot instead) — point it at the group's
// FALLBACK row (the pilot's own row, read by no outer loop) so a remote member's diagnostic q
// row carries ONLY its own injection (fillRemoteMemberQBatch). Without this the pilot's whole
// balance lands on the base carrier's row and corrupts the free member's reactive-limit check.
void patchGroupFrozenDestinations(DeviceBuf& dest, const std::vector<int>& relocationGroup,
                           const std::vector<int>& relocationCandidateOffset, int outputCount,
                           const int* scatterRole, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows) {
    for (size_t o = 0; o < relocationGroup.size(); o++) {
        int g = relocationGroup[o];
        // A STOCK PAIR (negative offset) parks the DUMMY ROW in the group slot, not a group id — reading
        // reactiveRelocationGroupStart[row] there walked off the candidate table and wrote a garbage dest, which the device then
        // used: "CUDA error: an illegal memory access" out of createBatchContext.
        if (g >= 0 && relocationCandidateOffset[o] < 0) {
            continue;
        }
        if (g >= 0 && scatterRole[o % outputCount] == 0) {
            int v = -2 - reactiveRelocationCandidateRows[reactiveRelocationGroupStart[g]];
            CUDA_CHECK(cudaMemcpy(dest.as<int>() + o, &v, sizeof(int), cudaMemcpyHostToDevice));
        }
    }
}

}  // namespace olfgpu
