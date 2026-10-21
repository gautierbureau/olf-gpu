// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// docs/gpu-vs-olf.md section 2.2 (row state): continuous phase control (OLF's PhaseControl, row mode 3) emits the
// controlled side's P flow again at the controller's a1 row; its pattern entries and destinations.
#include <utility>
#include <vector>

#include "../equations/generated/closed_branch_scatter.h"

#include "structure_build.cuh"
#include "../common/codes/gpu_two_sides.cuh"

using namespace powsybl::openloadflow;

namespace olfgpu {

// CONTINUOUS PHASE CONTROL (BRANCH_TARGET_P): when a phase-controller branch's control is enabled
// per scenario (row mode 3 on the CONTROLLER's a1-paired row), the controlled side's P-flow outputs
// are emitted AGAIN at that row — the p-flow equation replacing the a1 identity. Union those
// (a1Row, col) positions into the pattern; closedBranchPhaseControlSide[e] = -1 none, else (controllerA1Col << 1) | side (a GpuTwoSides)
// — the entry sits on the CONTROLLED element, whose own a1 may be absent (REMOTE control).
void collectClosedBranchPhaseControlPattern(int nClosedBranch, const int* closedBranchIndices, const int* closedBranchPhaseControlSide,
                           const std::vector<int>& rowMap, std::vector<std::pair<int, int>>& pat) {
    for (int e = 0; e < nClosedBranch; e++) {
        if (closedBranchPhaseControlSide[e] < 0) {
            continue;
        }
        int side = closedBranchPhaseControlSide[e] & 1;
        int a1Col = closedBranchPhaseControlSide[e] >> 1;
        const int* ix = closedBranchIndices + static_cast<size_t>(e) * 12;
        int pcRow = rowMap[a1Col];
        int k0 = side == GpuTwoSides::ONE ? 0 : 14;          // p1 block / p2 block of CLOSED_BRANCH_SCATTER
        for (int k = k0 + 1; k <= k0 + 6; k++) {             // the 6 derivative outputs
            RC rc = resolveClosedBranch(CLOSED_BRANCH_SCATTER[k], ix);
            if (rc.col >= 0) {
                pat.emplace_back(pcRow, rc.col);
            }
        }
    }
}

// Per (closed branch, output): the phase-row destination (J slot, or -2-row for the P residual) and
// the a1 row whose per-scenario mode gates the emission (mode 3 = control enabled). -1 = no emission.
void buildClosedBranchPhaseControlDestination(int nClosedBranch, const int* closedBranchIndices, const int* closedBranchPhaseControlSide, const std::vector<int>& rowMap,
                      const CsrPattern& pos, std::vector<int>& phaseControlDestination, std::vector<int>& phaseControlSelector) {
    phaseControlDestination.assign(static_cast<size_t>(nClosedBranch) * CLOSED_BRANCH_OUTPUT_COUNT, -1);
    phaseControlSelector.assign(static_cast<size_t>(nClosedBranch) * CLOSED_BRANCH_OUTPUT_COUNT, -1);
    for (int e = 0; e < nClosedBranch; e++) {
        if (closedBranchPhaseControlSide[e] < 0) {
            continue;
        }
        int side = closedBranchPhaseControlSide[e] & 1;
        int a1Col = closedBranchPhaseControlSide[e] >> 1;
        const int* ix = closedBranchIndices + static_cast<size_t>(e) * 12;
        int pcRow = rowMap[a1Col];
        int k0 = side == GpuTwoSides::ONE ? 0 : 14;
        phaseControlDestination[static_cast<size_t>(e) * CLOSED_BRANCH_OUTPUT_COUNT + k0] = -2 - pcRow;    // the P residual
        phaseControlSelector[static_cast<size_t>(e) * CLOSED_BRANCH_OUTPUT_COUNT + k0] = a1Col;          // mode-3 gate lives at the CONTROLLER's a1 row
        for (int k = k0 + 1; k <= k0 + 6; k++) {
            RC rc = resolveClosedBranch(CLOSED_BRANCH_SCATTER[k], ix);
            if (rc.col >= 0) {
                phaseControlDestination[static_cast<size_t>(e) * CLOSED_BRANCH_OUTPUT_COUNT + k] = pos.at(pcRow, rc.col);
                phaseControlSelector[static_cast<size_t>(e) * CLOSED_BRANCH_OUTPUT_COUNT + k] = a1Col;
            }
        }
    }
}

}  // namespace olfgpu
