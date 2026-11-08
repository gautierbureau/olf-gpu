// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// A fill-reducing elimination order of our batched LU computed on the host: the exact minimum degree of the
// symmetrized pattern A + A^T. cuDSS's analysis order (cudss_order.cuh) keeps the coupler chains of a breaker-mode
// network short, which a full busbar-section topology needs; on a network with few couplers the minimum degree gives a
// shallower elimination tree - fewer dependent levels in every factorization and triangular solve
// (gpu-work/ordering: network.xiidm, 100 contingencies x 150 phase-shifter strategies, 184 -> 160 levels; every
// busbar section kept, 354 -> 364).
#ifndef OLF_GPU_LU_MINIMUM_DEGREE_CUH
#define OLF_GPU_LU_MINIMUM_DEGREE_CUH

#include <vector>

namespace olfgpu {

// The exact minimum-degree elimination order of A + A^T (position -> original index): the vertex of least degree in
// the elimination graph is eliminated next, its neighbours becoming a clique; ties go to the lowest index, so the
// order is a function of the pattern alone. rowPtr / colIdx: A's CSR pattern, n rows.
std::vector<int> minimumDegreeOrder(int n, const int* rowPtr, const int* colIdx);

// The height of the elimination tree of A + A^T eliminated in order (position -> original index): the number of
// dependent levels of a factorization or a triangular solve on that order.
int eliminationLevels(int n, const int* rowPtr, const int* colIdx, const std::vector<int>& order);

}  // namespace olfgpu

#endif  // OLF_GPU_LU_MINIMUM_DEGREE_CUH
