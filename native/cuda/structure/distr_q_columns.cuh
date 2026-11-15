// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The variable columns a DISTR_Q contribution's reactive term depends on: the CSR pattern (collectFamilyPattern) and
// the destinations (buildDistrQDest) read them from here, so the pattern holds every slot the destinations look up.
#ifndef OLF_GPU_STRUCTURE_DISTR_Q_COLUMNS_CUH
#define OLF_GPU_STRUCTURE_DISTR_Q_COLUMNS_CUH

#include <cstddef>

#include "../common/codes/gpu_reactive_term_kind.cuh"
#include "../common/codes/gpu_two_sides.cuh"

namespace olfgpu {

// In the order of the contribution's derivative destinations (d[1] ..): a shunt's bus V column; a half-open branch's
// connected-side V column; a coupler's DUMMY_Q column (element is that column); a closed branch's v1 v2 ph1 ph2, and r1
// when its ratio is a variable. Returns how many: 1, 4 or 5.
inline int distrQTermColumns(int kind, int element, int side, const int* closedBranchIndices, const int* openBranchIndices,
                             const int* shuntIndices, int columns[5]) {
    if (kind == GpuReactiveTermKind::SHUNT_COMPENSATOR_REACTIVE_FLOW) {
        columns[0] = shuntIndices[static_cast<size_t>(element) * 5 + 2];
        return 1;
    }
    if (kind == GpuReactiveTermKind::OPEN_BRANCH_REACTIVE_FLOW) {
        const int* openBranch = openBranchIndices + static_cast<size_t>(element) * 6;
        columns[0] = side == GpuTwoSides::ONE ? openBranch[4] : openBranch[5];
        return 1;
    }
    if (kind == GpuReactiveTermKind::DUMMY_Q) {
        columns[0] = element;
        return 1;
    }
    const int* q = closedBranchIndices + static_cast<size_t>(element) * 12;
    columns[0] = q[5];                                       // dq/dv1
    columns[1] = q[7];                                       // dq/dv2
    columns[2] = q[4];                                       // dq/dph1
    columns[3] = q[6];                                       // dq/dph2
    columns[4] = q[9];                                       // dq/dr1
    return q[9] >= 0 ? 5 : 4;
}

}  // namespace olfgpu

#endif  // OLF_GPU_STRUCTURE_DISTR_Q_COLUMNS_CUH
