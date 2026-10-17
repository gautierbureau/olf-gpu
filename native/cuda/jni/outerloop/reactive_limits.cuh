// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// ReactiveLimitsOuterLoop as an in-kernel driver phase: a PV controller over its Q limit is pinned PQ at the limit,
// a pinned one whose voltage relieves is released, and robust mode's realistic-voltage pins.
#ifndef OLF_GPU_OUTERLOOP_REACTIVE_LIMITS_CUH
#define OLF_GPU_OUTERLOOP_REACTIVE_LIMITS_CUH

#include "../solver/driver_pass.cuh"

namespace olfgpu {

__global__ void reactiveLimitsBatch(const double* svcRlVt, int nrl, int nScenarios, int n, double tol, int maxSwitch,
                                    const int* rlVrow, const int* rlQrow, const double* rlMinQ, const double* rlMaxQ,
                                    const double* rlLoadQ, const double* rlVtarget, const double* qBuf,
                                    const double* x, const char* rlActive, int* mode, double* target,
                                    char* limitType, char* switchCount, char* switched,
                                    char* passSwitched, char* savedMode,
                                    int nRlVtOv, const int* rlVtOvSlotIdx, const double* rlVtOvVal,
                                    const double* rowSlope, double* savedTarget, int cvStride,
                                    const char* rlRemote, const int* rlCmpVrow,
                                    const double* rlSvcBmin, const double* rlSvcBmax, const double* rlV2Base,
                                    const double* svcBminDyn, const double* svcBmaxDyn,
                                    const char* rlIslanded,
                                    const double* rlInitTgtQ, double vLoRobust, double vHiRobust,
                                    int robust,
                                    int swLogSlot, double* swLog, int* swLogCnt, int passId,
                                    const int* psEntry, int nQr,
                                    const char* rlLocalOv, const double* rlLocalVt,
                                    const char* rlNoUnpin, char* unpinBad, const char* rlMonitored,
                                    const int* rlCmpOv);

__global__ void rlRobustVResetBatch(int nrl, int nScenarios, int n, const int* rlVrow, const char* rlRemote,
                                    const char* rlActive, const char* passSwitched, double* x,
                                    double vLoRobust, double vHiRobust);

__global__ void rlForcedPvElectBatch(int nrl, int nScenarios, int n,
                                     const int* rlVrow, const char* rlRemote, const char* rlIslanded,
                                     const double* nomV, const int* pRow, const double* staticP, const int* idRank,
                                     char* limitType, char* switchCount, char* passSwitched, const char* savedMode,
                                     const double* savedTarget, int* mode, double* target,
                                     const char* rlActive, char* switched);

__global__ void rlPilotSelfMixedGate(int nrl, int nScenarios, const char* rlRemote, const int* rlCmpVrow,
                                     const char* limitType, const char* rlIslanded, char* bad,
                                     const int* psEntry, int nQr,
                                int nDqMembers, const int* dqMemberRlEntry, const char* dqSub);

__global__ void rlUnpinBatch(int nPairs, int nS, int n, int nrl, const int* pairs, const double* newTarget,
                             const int* rlVrow, char* limitType, const char* savedMode, int* mode, double* target);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_REACTIVE_LIMITS_CUH
