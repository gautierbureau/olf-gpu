// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Remote generator voltage control (GeneratorVoltageControl with a remote controlled bus) under reactive limits:
// the DISTR_Q row targets recomputed from the live pin states, the all-pinned pilot row, and the members' Q.
#ifndef OLF_GPU_OUTERLOOP_REMOTE_VOLTAGE_CONTROL_CUH
#define OLF_GPU_OUTERLOOP_REMOTE_VOLTAGE_CONTROL_CUH

#include "../solver/driver_pass.cuh"

namespace olfgpu {

// Kernels the driver launches.
__global__ void pilotRowAllPinnedBatch(int nEntries, int nScenarios, int n, int nrl,
                                       const int* entSlot, const int* entRow, const double* entQTarget,
                                       const int* entOff, const int* entCount, const int* entRlPool,
                                       const char* limitType, int* mode, double* target,
                                       char* flipped, double* savedVTarget,
                                       const char* rlActive, char* switched);

__global__ void fillRemoteMemberQBatch(int nrmq, int nScenarios, int n, int nElements,
                                       int closedBranchOffset, int openBranchOffset,
                                       const int* rmqRow, const int* rmqElem, const int* rmqSide,
                                       const int* rmqKind, const double* rmqWeight,
                                       const double* closedBranchParameters, const int* closedBranchIndices, const double* shuntParameters, const int* shuntIndices,
                                       const double* openBranchParameters, const int* openBranchIndices,
                                       const double* x, double* qBuf,
                                       const char* disabled, const char* rlActive,
                                       int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal,
                                       int nShuntAdmittanceOverrides, const int* shuntAdmittanceOverrideScenarioElement, const double* shuntAdmittanceOverrideDelta);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_REMOTE_VOLTAGE_CONTROL_CUH
