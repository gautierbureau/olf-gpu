// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// docs/gpu-vs-olf.md section 2.4 (a factorization without pivoting): the zero-diagonal repair tables
// buildStructure stores (zero_diagonal_repair_tables.cu).
#ifndef OLF_GPU_FIXEDMATRIX_ZERO_DIAGONAL_REPAIR_TABLES_CUH
#define OLF_GPU_FIXEDMATRIX_ZERO_DIAGONAL_REPAIR_TABLES_CUH

#include <utility>
#include <vector>

#include "../structure/shared_structure.cuh"

namespace olfgpu {

void collectZeroDiagonalRepairRows(const SharedStructure* structure, int n, int nZeroImpedanceBranch, const int* zeroImpedanceBranchIndices,
                                   const int* zeroImpedanceBranchAlternateReactiveDestination, const std::vector<int>& rowMap, std::vector<std::pair<int, int>>& pat,
                                   std::vector<int>& zeroDiagonalRows, std::vector<std::vector<int>>& zeroDiagonalDonors, std::vector<char>& zeroDiagonalRepairSigns);

void storeZeroDiagonalRepairTables(SharedStructure* structure, const std::vector<int>& zeroDiagonalRows, const std::vector<std::vector<int>>& zeroDiagonalDonors,
                                   const std::vector<char>& zeroDiagonalRepairSigns);

}  // namespace olfgpu

#endif
