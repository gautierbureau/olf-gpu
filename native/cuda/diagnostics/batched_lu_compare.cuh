// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// OLF_GPU_LU_COMPARE: our batched LU's Newton step against cuDSS's uniform batch on the same systems.
#ifndef OLF_GPU_DIAGNOSTICS_BATCHED_LU_COMPARE_CUH
#define OLF_GPU_DIAGNOSTICS_BATCHED_LU_COMPARE_CUH

#include "../solver/batch_context.cuh"

namespace olfgpu {

// Solves the context's systems (dVals, dB) with cuDSS's uniform batch, with the internal refinement steps a UBATCH
// context runs (the configured base steps on a zero-impedance structure, none otherwise), and prints one line comparing
// that with our LU's solution in dDx over the lanes of its last factorization: the worst relative difference, both
// solves' worst backward error, the lanes with a clamped pivot, and how the differences spread over decades.
void compareBatchedLuWithCudss(BatchSolverContext* ctx, int outer, int iteration);

}  // namespace olfgpu

#endif  // OLF_GPU_DIAGNOSTICS_BATCHED_LU_COMPARE_CUH
