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
__global__ void pilotSelfGateBatch(int nReactiveRelocationGroups, int nScenarios, int nReactiveLimitEntries, const int* pilotEntryOfPilotSelfGroup,
                                   const char* limitType, const char* reactiveLimitIslanded, char* pilotSelfCarrierSkip);

__global__ void capturePilotQBatch(int nReactiveRelocationGroups, int nScenarios, int n, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows,
                                   const int* baseMode, const double* targetBase, double* pilotQ,
                                   const int* reactiveRelocationBaseTargetRow);

__global__ void applyPilotQDeltasBatch(int nDelta, const int* key, const double* val, int nReactiveRelocationGroups, int n,
                                       const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows, const int* mode,
                                       const char* distrQSubgroup, int nDistrQMembers, const int* reactiveRelocationDistrQMemberStart,
                                       double* pilotQ, double* target);

__global__ void updatePilotTargetVEquationBatch(int nReactiveRelocationGroups, int nScenarios, int n,
                                    const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows, const char* reactiveRelocationIsGeneratorVoltageControl,
                                    const double* pilotQ, int* mode, double* target,
                                    char* pilotTargetVDeactivated, double* savedVTarget,
                                    const char* outerLoopCheckActive, char* switched,
                                    int nFrozen, const int* frozenSlotRow,
                                    const int* pilotSelfReactiveLimitEntry,
                                    const char* limitType, int nReactiveLimitEntries, const double* reactiveLimitMinQ, const double* reactiveLimitMaxQ,
                                    const double* reactiveLimitLoadQ, const double* reactiveLimitInitialTargetQ, int perScenarioReactiveLimits,
                                    const double* reactiveLimitTargetV,
                                    const int* reactiveRelocationPilotReactiveLimitEntry,
                                    int nReactiveLimitTargetVOverrides, const int* reactiveLimitTargetVOverrideScenarioEntry, const double* reactiveLimitTargetVOverrideValue,
                                    const double* secondaryVoltageControlReactiveLimitTargetV,
                                    int nDistrQMembers, const int* distrQMemberRelocationGroup, const int* distrQMemberReactiveLimitEntry,
                                    const char* reactiveLimitIslanded,
                                const char* distrQSubgroup, const int* reactiveRelocationDistrQMemberStart, double tol,
                                const char* reactiveLimitGeneratorsLost, const int* reactiveRelocationDependentBaseGroup, const int* relocatedChildOwnRow,
                                const double* pinTargetIn);

__global__ void updateReMergedVoltageTargetRowBatch(int nReMergedVoltageTargets, int nScenarios, int n, int nReactiveLimitEntries,
                                                    const int* reMergedVoltageTargetSlot, const int* reMergedVoltageTargetMainEntry,
                                                    const double* reMergedVoltageTargetValue, const int* reMergedVoltageTargetControllerStart,
                                                    const int* reMergedVoltageTargetControllerCount, const int* reMergedVoltageTargetControllerEntries,
                                                    const int* reactiveLimitVoltageRow, const int* reactiveLimitReactivePowerRow, const int* relocatedChildOwnRow,
                                                    const char* limitType, const char* reactiveLimitIslanded, int nFrozen, const int* frozenSlotRow,
                                                    int* reMergedVoltageTargetHolder, char* reactiveLimitHoldsReMergedVoltageTarget,
                                                    int* reactiveLimitReMergedVoltageRow, int* mode, double* target,
                                                    const char* outerLoopCheckActive, char* switched);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_PILOT_VOLTAGE_CUH
