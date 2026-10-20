// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The pilot (controlled bus) of a remote voltage control group: the pilot-self gate, the pilot Q target captured
// from the base, its per-scenario deltas, and the pilot row laid out from the current pin states.
#ifndef OLF_GPU_OUTERLOOP_PILOT_VOLTAGE_CUH
#define OLF_GPU_OUTERLOOP_PILOT_VOLTAGE_CUH

#include "../solver/driver_pass.cuh"

namespace olfgpu {

// Kernels the driver launches.
__global__ void pilotSelfGateBatch(int nQr, int nScenarios, int nReactiveLimitEntries, const int* psPilotEntry,
                                   const char* limitType, const char* reactiveLimitIslanded, char* psSkip);

__global__ void capturePilotQBatch(int nQr, int nScenarios, int n, const int* qrPtr, const int* qrRows,
                                   const int* baseMode, const double* targetBase, double* pilotQ,
                                   const int* qrBaseTargetRow);

__global__ void applyPilotQDeltasBatch(int nDelta, const int* key, const double* val, int nQr, int n,
                                       const int* qrPtr, const int* qrRows, const int* mode,
                                       const char* distrQSubgroup, int nDistrQMembers, const int* qrMemberStart,
                                       double* pilotQ, double* target);

__global__ void updatePilotVeqBatch(int nQr, int nScenarios, int n,
                                    const int* qrPtr, const int* qrRows, const char* qrGenVc,
                                    const double* pilotQ, int* mode, double* target,
                                    char* veqOff, double* savedVTarget,
                                    const char* outerLoopCheckActive, char* switched,
                                    int nFrozen, const int* frozenSlotRow,
                                    const int* psEntry,
                                    const char* limitType, int nReactiveLimitEntries, const double* reactiveLimitMinQ, const double* reactiveLimitMaxQ,
                                    const double* reactiveLimitLoadQ, const double* reactiveLimitInitialTargetQ, int perScenarioReactiveLimits,
                                    const double* reactiveLimitTargetV,
                                    const int* reactiveRelocationPilotReactiveLimitEntry,
                                    int nReactiveLimitTargetVOverrides, const int* reactiveLimitTargetVOverrideScenarioEntry, const double* reactiveLimitTargetVOverrideValue,
                                    const double* secondaryVoltageControlReactiveLimitTargetV,
                                    int nDistrQMembers, const int* distrQMemberRelocationGroup, const int* distrQMemberReactiveLimitEntry,
                                    const char* reactiveLimitIslanded,
                                const char* distrQSubgroup, const int* qrMemberStart, double tol,
                                const char* reactiveLimitGeneratorsLost, const int* qrDepBase, const int* relocOwn,
                                const double* pinTargetIn);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_PILOT_VOLTAGE_CUH
