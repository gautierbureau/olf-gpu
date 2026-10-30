// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The elimination order of our batched LU: the one cuDSS's analysis picks for the same pattern, so the two
// factorizations have the same fill and can be compared lane by lane.
#ifndef OLF_GPU_LU_CUDSS_ORDER_CUH
#define OLF_GPU_LU_CUDSS_ORDER_CUH

#include <cstdint>
#include <vector>

namespace olfgpu {

// The row and column orders (new index -> pattern index) cuDSS's analysis picks for the device CSR pattern under the
// run's ordering setting (applyReorderAlg, as a batch context), and the LU fill it reports.
void cudssEliminationOrder(int n, int nnz, const int* dRowPtr, const int* dColIdx, bool hasZeroImpedance,
                           std::vector<int>& rowOrder, std::vector<int>& colOrder, int64_t& luNnz);

}  // namespace olfgpu

#endif  // OLF_GPU_LU_CUDSS_ORDER_CUH
