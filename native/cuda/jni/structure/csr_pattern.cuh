// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// RC, a resolved (equation row, variable column) pair, and CsrPattern, the slot lookup in the fixed CSR
// pattern every destination array is built against.
#ifndef OLF_GPU_STRUCTURE_CSR_PATTERN_CUH
#define OLF_GPU_STRUCTURE_CSR_PATTERN_CUH

#include <stdexcept>

namespace olfgpu {

struct RC { int row = -1; int col = -1; };

struct CsrPattern {
    const int* rowPtr;
    const int* colIdx;
    int at(int row, int col) const {
        int lo = rowPtr[row];
        int hi = rowPtr[row + 1];
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (colIdx[mid] < col) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo >= rowPtr[row + 1] || colIdx[lo] != col) {
            throw std::runtime_error("CSR pattern lookup miss");
        }
        return lo;
    }
};

}  // namespace olfgpu

#endif  // OLF_GPU_STRUCTURE_CSR_PATTERN_CUH
