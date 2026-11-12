// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The elimination order of our batched LU computed on the host without cuDSS: SuiteSparse's approximate minimum degree
// (AMD, native/third_party/suitesparse-amd) of the symmetrized pattern A + A^T, with its default controls. It is the
// order cuDSS's analysis returns under CUDSS_REORDERING_ALG_AMD (reorderingAlgorithm AUTO), position for position, on
// every structure measured (gpu-work/ordering: network.xiidm every busbar section kept, the fixed list, rte6515 every
// busbar section, 100 contingencies x 150 strategies), so planning on it changes no number.
#ifndef OLF_GPU_LU_AMD_ORDER_CUH
#define OLF_GPU_LU_AMD_ORDER_CUH

#include <vector>

namespace olfgpu {

// The AMD elimination order of A + A^T (position -> original index). rowPtr / colIdx: A's CSR pattern, n rows; the
// pattern may hold the diagonal, unsorted or repeated entries. AMD breaks its ties in the order of its input entries:
// it is given A + A^T with the diagonal, each row sorted, as cuDSS orders it (A's own CSR gives another order on an
// unsymmetric pattern).
std::vector<int> approximateMinimumDegreeOrder(int n, const int* rowPtr, const int* colIdx);

}  // namespace olfgpu

#endif  // OLF_GPU_LU_AMD_ORDER_CUH
