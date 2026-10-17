// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The distributed-slack participants' active-power rows, read by the slack distribution, the reactive-limit
// curves and the forced target Q.
#ifndef OLF_GPU_OUTERLOOP_SLACK_PARTICIPANTS_CUH
#define OLF_GPU_OUTERLOOP_SLACK_PARTICIPANTS_CUH

#include <cuda_runtime.h>

namespace olfgpu {

// The row this SCENARIO solves participant i's P balance on. Normally its structural dsPhiRow; under
// the coupler-tree relocation that is the parent coupler's DUMMY_P row, and a scenario opening that
// coupler moves the balance back to the bus's own BUS_PHI row (setBatchDsRowOverrides).
__host__ __device__ inline int dsRowOf(const int* dsPhiRow, const int* dsRowOv, int s, int nds, int i) {
    if (dsRowOv != nullptr) {
        int r = dsRowOv[(size_t) s * nds + i];
        if (r >= 0) {
            return r;
        }
    }
    return dsPhiRow[i];
}

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_SLACK_PARTICIPANTS_CUH
