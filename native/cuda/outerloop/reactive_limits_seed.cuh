// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// ReactiveLimitsOuterLoop state carried across solves: the per-(controller, scenario) limit type seeded from the
// base or from a previous solve's PV -> PQ switches, and the switches and their frozen target Q exported for the next one.
#ifndef OLF_GPU_OUTERLOOP_REACTIVE_LIMITS_SEED_CUH
#define OLF_GPU_OUTERLOOP_REACTIVE_LIMITS_SEED_CUH

#include "../solver/driver_pass.cuh"

namespace olfgpu {

__global__ void initReactiveLimitQLimitTypeBatch(int nReactiveLimitEntries, int nScenarios, int n, const int* reactiveLimitVoltageRow, const int* reactiveLimitReactivePowerRow,
                                     const double* reactiveLimitTargetV, const char* reactiveLimitFlags,
                                     const int* baseMode, const double* x, char* limitType,
                                     SlotEntries reactiveLimitTargetVOverrides, const double* secondaryVoltageControlReactiveLimitTargetV,
                                     const char* reactiveLimitVoltageControlRole, const int* reactiveLimitComparedVoltageRow,
                                     const double* target, const double* reactiveLimitBaseMinQ, const double* reactiveLimitBaseMaxQ,
                                     const double* reactiveLimitBaseLoadQ, const char* reactiveLimitBaseQLimitType,
                                     const int* pilotSelfReactiveLimitEntry, int nReactiveRelocationGroups, const char* reactiveLimitGeneratorsLost, const char* installed);

__global__ void seedReactiveLimitRowModeFromState(int nReactiveLimitEntries, int nScenarios, int n, const int* reactiveLimitVoltageRow, const int* reactiveLimitReactivePowerRow,
                                    const double* reactiveLimitMinQ, const double* reactiveLimitMaxQ, const double* reactiveLimitLoadQ,
                                    const char* limitType, int* mode, double* target, int perScenarioReactiveLimits,
                                    const char* reactiveLimitVoltageControlRole, double* savedTarget, char* savedMode,
                                    const char* reactiveLimitBaseQLimitType, const int* pilotSelfReactiveLimitEntry, int nReactiveRelocationGroups,
                                    const double* reactiveLimitTargetV, SlotEntries reactiveLimitTargetVOverrides,
                                    const double* secondaryVoltageControlReactiveLimitTargetV, const char* reactiveLimitLocalOverride, const double* reactiveLimitLocalTargetV,
                                    const double* reactiveLimitSeedTargetQ, const char* reactiveLimitIslanded, const char* installed);

__global__ void exportReactiveLimitTargetQ(int nReactiveLimitEntries, int nScenarios, int n, const int* reactiveLimitVoltageRow, const int* reactiveLimitReactivePowerRow,
                                   const char* limitType, const int* mode, const double* target, const char* reactiveLimitIslanded,
                                   int nReactiveRelocationGroups, const int* pilotSelfReactiveLimitEntry, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows, const char* pilotTargetVDeactivated,
                                   const int* reactiveRelocationDependentBaseGroup, double* out);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_REACTIVE_LIMITS_SEED_CUH
