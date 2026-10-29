// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The distribution rows of the batched Jacobian and mismatch: DISTR_Q (remote generator voltage control),
// DISTR_RHO (shared transformer voltage control) and DISTR_SHUNT_B (shared shunt voltage control).
#ifndef OLF_GPU_EQUATIONS_DISTRIBUTION_FILL_CUH
#define OLF_GPU_EQUATIONS_DISTRIBUTION_FILL_CUH

#include "../solver/driver_pass.cuh"
#include "scenario_overrides.cuh"

namespace olfgpu {

int gpuDistrQTraceRow();

__global__ void fillDistrQBatch(int nDistrQ, int nScenarios, int n, int nnz, int nElements,
                                int closedBranchOffset, int openBranchOffset,
                                const int* distrQElement, const int* distrQSide, const int* distrQKind,
                                const double* distrQWeight, const int* distrQDestination,
                                const int* distrQAlternateDestination, const int* distrQSelectorRow, const int* distrQDummyQRow,
                                const double* closedBranchParameters, const int* closedBranchIndices, const double* shuntParameters, const int* shuntIndices,
                                const double* openBranchParameters, const int* openBranchIndices,
                                const double* x, double* vals, double* F,
                                const char* disabled, const char* active,
                                SlotEntries distrQWeightOverrides,
                                SlotEntries distrQKeyOverrides,
                                const int* mode,
                                SlotEntries distrQInactiveMembers,
                                const int* distrQRowMember, const int* distrQInjectionMember, const double* mKey,
                                const int* distrQMemberReactiveLimitEntry, const int* mGroupStart, const int* mGroupCount,
                                const char* limitType, int nReactiveLimitEntries,
                                const char* pilotSelfCarrierSkip, int pilotSelfCarrierSkipStride,
                                const int* memberRelocationGroup, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows,
                                ClosedBranchParameterOverrides closedBranchParameterOverrides,
                                ShuntAdmittanceOverrides shuntAdmittanceOverrides,
                                int distrQTraceRow,
                                const char* distrQSubgroup, const int* reactiveRelocationDistrQMemberStart, int nDistrQMembersAll, BatchLayout layout);

__global__ void fillDistrRhoBatch(int nDistrRho, int nScenarios, int n, int nnz,
                                  const int* distrRhoColumn, const double* distrRhoCoefficient, const int* distrRhoDestination,
                                  const double* x, double* vals, double* F, const char* active,
                                  SlotEntries distrRhoCoefficientOverrides,
                                  const int* distrRhoRelocationGroup, const int* mode, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows, BatchLayout layout);

__global__ void fillDistrShuntBBatch(int nDistrShuntB, int nScenarios, int n, int nnz,
                                     const int* distrShuntBColumn, const double* distrShuntBCoefficient, const int* distrShuntBDestination,
                                     const double* x, double* vals, double* F, const char* active, BatchLayout layout);

}  // namespace olfgpu

#endif  // OLF_GPU_EQUATIONS_DISTRIBUTION_FILL_CUH
