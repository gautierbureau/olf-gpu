// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The pilot (controlled bus) of a remote voltage control group: the pilot-self gate, the pilot Q target captured
// from the base, its per-scenario deltas, and the pilot row laid out from the current reactive-limit states.
#ifndef OLF_GPU_OUTERLOOP_PILOT_VOLTAGE_CUH
#define OLF_GPU_OUTERLOOP_PILOT_VOLTAGE_CUH

#include "../solver/driver_pass.cuh"
#include "../equations/slot_entries.cuh"

namespace olfgpu {

// Kernels the driver launches.
__global__ void pilotSelfGateBatch(int nReactiveRelocationGroups, int nScenarios, int nReactiveLimitEntries, const int* pilotEntryOfPilotSelfGroup,
                                   const char* limitType, const char* reactiveLimitIslanded, char* pilotSelfCarrierSkip);

__global__ void capturePilotQBatch(int nReactiveRelocationGroups, int nScenarios, int n, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows,
                                   const int* baseMode, const double* targetBase, double* pilotQ,
                                   const int* reactiveRelocationBaseTargetRow);

__global__ void applyPilotQDeltasBatch(int nScenarios, SlotEntries deltas, int nReactiveRelocationGroups, int n,
                                       const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows, const int* mode,
                                       const char* distrQSubgroup, int nDistrQMembers, const int* reactiveRelocationDistrQMemberStart,
                                       double* pilotQ, double* target);

__global__ void updatePilotTargetVEquationBatch(int nReactiveRelocationGroups, int nScenarios, int n,
                                    const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows, const char* reactiveRelocationIsGeneratorVoltageControl,
                                    const double* pilotQ, int* mode, double* target,
                                    char* pilotTargetVDeactivated, double* savedVTarget,
                                    const char* outerLoopCheckActive, char* switched,
                                    SlotEntries frozenRows,
                                    const int* pilotSelfReactiveLimitEntry,
                                    const char* limitType, int nReactiveLimitEntries, const double* reactiveLimitMinQ, const double* reactiveLimitMaxQ,
                                    const double* reactiveLimitLoadQ, const double* reactiveLimitInitialTargetQ, int perScenarioReactiveLimits,
                                    const double* reactiveLimitTargetV,
                                    const int* reactiveRelocationPilotReactiveLimitEntry,
                                    SlotEntries reactiveLimitTargetVOverrides,
                                    const double* secondaryVoltageControlReactiveLimitTargetV,
                                    int nDistrQMembers, const int* distrQMemberRelocationGroup, const int* distrQMemberReactiveLimitEntry,
                                    const char* reactiveLimitIslanded,
                                const char* distrQSubgroup, const int* reactiveRelocationDistrQMemberStart, double tol,
                                const char* reactiveLimitGeneratorsLost, const int* reactiveRelocationDependentBaseGroup, const int* relocatedChildOwnRow,
                                const double* reactiveLimitSeedTargetQ);

__global__ void updateReMergedVoltageTargetRowBatch(int nScenarios, int n, int nReactiveLimitEntries,
                                                    SlotEntries reMergedVoltageTargets, SlotEntries reMergedVoltageTargetControllers,
                                                    const int* reactiveLimitVoltageRow, const int* reactiveLimitReactivePowerRow, const int* relocatedChildOwnRow,
                                                    const char* limitType, const char* reactiveLimitIslanded, SlotEntries frozenRows,
                                                    int* reMergedVoltageTargetHolder, char* reactiveLimitHoldsReMergedVoltageTarget,
                                                    int* reactiveLimitReMergedVoltageRow, int* mode, double* target,
                                                    const char* outerLoopCheckActive, char* switched);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_PILOT_VOLTAGE_CUH
