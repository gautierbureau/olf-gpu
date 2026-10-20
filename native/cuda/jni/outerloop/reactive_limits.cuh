// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// ReactiveLimitsOuterLoop as an in-kernel driver phase: a PV controller over its Q limit is pinned PQ at the limit,
// a pinned one whose voltage relieves is released, and robust mode's realistic-voltage pins.
#ifndef OLF_GPU_OUTERLOOP_REACTIVE_LIMITS_CUH
#define OLF_GPU_OUTERLOOP_REACTIVE_LIMITS_CUH

#include "../solver/driver_pass.cuh"
#include "../common/codes/gpu_outer_loop_kind.cuh"

namespace olfgpu {

__global__ void reactiveLimitsBatch(const double* secondaryVoltageControlReactiveLimitTargetV, int nReactiveLimitEntries, int nScenarios, int n, double tol, int maxSwitch,
                                    const int* reactiveLimitVoltageRow, const int* reactiveLimitReactivePowerRow, const double* reactiveLimitMinQ, const double* reactiveLimitMaxQ,
                                    const double* reactiveLimitLoadQ, const double* reactiveLimitTargetV, const double* qBuf,
                                    const double* x, const char* outerLoopCheckActive, int* mode, double* target,
                                    char* limitType, char* switchCount, char* switched,
                                    char* passSwitched, char* savedMode,
                                    int nReactiveLimitTargetVOverrides, const int* reactiveLimitTargetVOverrideScenarioEntry, const double* reactiveLimitTargetVOverrideValue,
                                    const double* rowSlope, double* savedTarget, int perScenarioReactiveLimits,
                                    const char* reactiveLimitVoltageControlRole, const int* reactiveLimitComparedVoltageRow,
                                    const double* reactiveLimitSvcMinB, const double* reactiveLimitSvcMaxB, const double* reactiveLimitBaseVSquared,
                                    const double* svcBminDyn, const double* svcBmaxDyn,
                                    const char* reactiveLimitIslanded,
                                    const double* reactiveLimitInitialTargetQ, double vLoRobust, double vHiRobust,
                                    int robust,
                                    int reactiveLimitsDecisionLogScenario, double* reactiveLimitsDecisionLog, int* reactiveLimitsDecisionLogCount, int passId,
                                    const int* pilotSelfReactiveLimitEntry, int nReactiveRelocationGroups,
                                    const char* reactiveLimitLocalOverride, const double* reactiveLimitLocalTargetV,
                                    const char* reactiveLimitPqPvSwitchForbidden, char* unpinBad, const char* reactiveLimitFlags,
                                    const int* reactiveLimitComparedVoltageRowOverride);

__global__ void reactiveLimitsRobustVoltageResetBatch(int nReactiveLimitEntries, int nScenarios, int n, const int* reactiveLimitVoltageRow, const char* reactiveLimitVoltageControlRole,
                                    const char* outerLoopCheckActive, const char* passSwitched, double* x,
                                    double vLoRobust, double vHiRobust);

__global__ void reactiveLimitsForcedPvBatch(int nReactiveLimitEntries, int nScenarios, int n,
                                     const int* reactiveLimitVoltageRow, const char* reactiveLimitVoltageControlRole, const char* reactiveLimitIslanded,
                                     const double* nomV, const int* pRow, const double* staticP, const int* idRank,
                                     char* limitType, char* switchCount, char* passSwitched, const char* savedMode,
                                     const double* savedTarget, int* mode, double* target,
                                     const char* outerLoopCheckActive, char* switched);

__global__ void pilotSelfMixedPvPqGate(int nReactiveLimitEntries, int nScenarios, const char* reactiveLimitVoltageControlRole, const int* reactiveLimitComparedVoltageRow,
                                     const char* limitType, const char* reactiveLimitIslanded, char* bad,
                                     const int* pilotSelfReactiveLimitEntry, int nReactiveRelocationGroups,
                                int nDistrQMembers, const int* distrQMemberReactiveLimitEntry, const char* distrQSubgroup);

__global__ void reactiveLimitsPqPvSwitchBatch(int nPairs, int nS, int n, int nReactiveLimitEntries, const int* pairs, const double* newTarget,
                             const int* reactiveLimitVoltageRow, char* limitType, const char* savedMode, int* mode, double* target);


// One check of ReactiveLimitsOuterLoop on every live scenario (loopKind GpuOuterLoopKind::REACTIVE_LIMITS).
void reactiveLimitsCheck(BatchSolverContext* ctx, DriverPass& drv, int loopKind);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_REACTIVE_LIMITS_CUH
