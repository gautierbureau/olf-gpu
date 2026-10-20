// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// docs/gpu-vs-olf.md section 2.3 (equations placed on another bus's row): the relocated rows buildStructure
// stores (relocated_rows.cu).
#ifndef OLF_GPU_FIXEDMATRIX_RELOCATED_ROWS_CUH
#define OLF_GPU_FIXEDMATRIX_RELOCATED_ROWS_CUH

#include <vector>

#include "../structure/shared_structure.cuh"

namespace olfgpu {

std::vector<int> buildRelocatedChildOwnRows(SharedStructure* structure, int n, int nZeroImpedanceBranch, const int* zeroImpedanceBranchIndices,
                                            const std::vector<int>& rowMap, bool& anyRelocatedChild);

bool storeReactiveRelocationTables(SharedStructure* structure, int n, int nReactiveRelocationGroups, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRowsIn,
                                   const int* reactiveRelocationDependentBaseGroupIn, const int* reactiveRelocationBaseTargetRowIn, const int* reactiveRelocationDependentGroupOfRowIn,
                                   const std::vector<int>& rowMap);

}  // namespace olfgpu

#endif
