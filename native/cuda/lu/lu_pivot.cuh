// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The pivot thresholds both engines of our batched LU share.
#ifndef OLF_GPU_LU_LU_PIVOT_CUH
#define OLF_GPU_LU_LU_PIVOT_CUH

namespace olfgpu {

// A pivot below this in magnitude is replaced by +-LU_PIVOT_EPSILON and counted on its lane: cuDSS's default absolute
// pivot epsilon for double precision (CUDSS_CONFIG_PIVOT_EPSILON, CUDSS_PIVOT_EPSILON_ALG_STATIC).
constexpr double LU_PIVOT_EPSILON = 1e-13;

// The fast engine's multiplier past which its unpivoted factorization of a lane is not trusted (growth from a pivot
// that is tiny for its column) and the lane goes to the pivoting engine. Far above the multipliers of a sound Jacobian
// - a voltage target's identity row pivots a whole admittance column with multipliers of the admittances.
constexpr double LU_MULTIPLIER_LIMIT = 1e8;

// A pivot below this fraction of the largest |A| of its row and column has cancelled in the fast engine's order (it
// makes multipliers past LU_MULTIPLIER_LIMIT, if it is not clamped): the order learns to delay it (pivot_delays.cuh).
constexpr double LU_PIVOT_RELATIVE_LIMIT = 1.0 / LU_MULTIPLIER_LIMIT;

}  // namespace olfgpu

#endif  // OLF_GPU_LU_LU_PIVOT_CUH
