// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Remote generator voltage control (GeneratorVoltageControl with a remote controlled bus) under reactive limits:
// the DISTR_Q row targets recomputed from the live reactive-limit states, the pilot row with no enabled controller, and the members' Q.
#ifndef OLF_GPU_OUTERLOOP_REMOTE_VOLTAGE_CONTROL_CUH
#define OLF_GPU_OUTERLOOP_REMOTE_VOLTAGE_CONTROL_CUH

#include "../solver/driver_pass.cuh"
#include "../equations/scenario_overrides.cuh"

namespace olfgpu {

// Kernels the driver launches.
__global__ void pilotRowWithoutEnabledControllerBatch(int nScenarios, int n, int nReactiveLimitEntries,
                                       SlotEntries flips, SlotEntries survivors,
                                       const char* limitType, int* mode, double* target,
                                       char* flipped, double* savedVTarget,
                                       const char* outerLoopCheckActive, char* switched);

__global__ void fillRemoteMemberQBatch(int nRemoteMemberReactiveTerms, int nScenarios, int n, int nElements,
                                       int closedBranchOffset, int openBranchOffset,
                                       const int* remoteMemberReactiveTermRow, const int* remoteMemberReactiveTermElement, const int* remoteMemberReactiveTermSide,
                                       const int* remoteMemberReactiveTermKind, const double* remoteMemberReactiveTermWeight,
                                       const double* closedBranchParameters, const int* closedBranchIndices, const double* shuntParameters, const int* shuntIndices,
                                       const double* openBranchParameters, const int* openBranchIndices,
                                       const double* x, double* qBuf,
                                       const char* disabled, const char* outerLoopCheckActive,
                                       ClosedBranchParameterOverrides closedBranchParameterOverrides,
                                       ShuntAdmittanceOverrides shuntAdmittanceOverrides);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_REMOTE_VOLTAGE_CONTROL_CUH
