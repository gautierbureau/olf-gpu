// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Diagnostics of the assembled linear system (host side, off the hot path): a full dump of one scenario's
// system as solved (OLF_GPU_DX_DUMP_ON_REJECT).
#ifndef OLF_GPU_DIAGNOSTICS_MATRIX_SCAN_CUH
#define OLF_GPU_DIAGNOSTICS_MATRIX_SCAN_CUH

#include "../solver/batch_context.cuh"

namespace olfgpu {

// One scenario's linear system as solved (repaired J, F, B, dx, row modes) written to path.
void dumpSlotSystem(BatchSolverContext* ctx, int n, int nnz, int slot, const char* path);

}  // namespace olfgpu

#endif  // OLF_GPU_DIAGNOSTICS_MATRIX_SCAN_CUH
