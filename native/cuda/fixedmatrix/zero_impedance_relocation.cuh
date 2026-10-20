// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// docs/gpu-vs-olf.md section 2.3 (equations placed on another bus's row), couplers: the per-scenario destinations of
// a zero-impedance coupler's dummy terms (zero_impedance_relocation.cu).
#ifndef OLF_GPU_FIXEDMATRIX_ZERO_IMPEDANCE_RELOCATION_CUH
#define OLF_GPU_FIXEDMATRIX_ZERO_IMPEDANCE_RELOCATION_CUH

#include <vector>

#include "../structure/csr_pattern.cuh"
#include "../structure/shared_structure.cuh"

namespace olfgpu {

void buildZeroImpedanceRelocationDestinations(const SharedStructure* structure, const CsrPattern& pos, int e, const int* z, int b1P, int b2P, int b1Q, int b2Q, int* dd,
                                              const int* zeroImpedanceBranchAlternateReactiveDestination, const int* relocatedChildOwnRow,
                                              const int* reactiveRelocationDependentGroupOfRow, std::vector<int>& zeroImpedanceAlternateDestination,
                                              std::vector<int>& zeroImpedanceSelectorRow, std::vector<int>& zeroImpedanceRelocationGroup,
                                              std::vector<int>& relocationCandidateOffset, std::vector<int>& zeroImpedanceRelocationCandidateDestinations,
                                              std::vector<char>& zeroImpedanceRelocationDependentPilot, bool& anyDynamicRelocation);

}  // namespace olfgpu

#endif
