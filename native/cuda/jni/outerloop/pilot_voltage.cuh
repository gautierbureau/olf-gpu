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
__global__ void pilotSelfGateBatch(int nQr, int nScenarios, int nrl, const int* psPilotEntry,
                                   const char* limitType, const char* rlIslanded, char* psSkip);

__global__ void capturePilotQBatch(int nQr, int nScenarios, int n, const int* qrPtr, const int* qrRows,
                                   const int* baseMode, const double* targetBase, double* pilotQ,
                                   const int* qrBaseTargetRow);

__global__ void applyPilotQDeltasBatch(int nDelta, const int* key, const double* val, int nQr, int n,
                                       const int* qrPtr, const int* qrRows, const int* mode,
                                       const char* dqSub, int nDqMembers, const int* qrMemberStart,
                                       double* pilotQ, double* target);

__global__ void updatePilotVeqBatch(int nQr, int nScenarios, int n,
                                    const int* qrPtr, const int* qrRows, const char* qrGenVc,
                                    const double* pilotQ, int* mode, double* target,
                                    char* veqOff, double* savedVTarget,
                                    const char* rlActive, char* switched,
                                    int nFrozen, const int* frozenSlotRow,
                                    const int* psEntry,
                                    const char* limitType, int nrl, const double* rlMinQ, const double* rlMaxQ,
                                    const double* rlLoadQ, const double* rlInitTgtQ, int cvStride,
                                    const double* rlVtarget,
                                    const int* qrPilotRlEntry,
                                    int nRlVtOv, const int* rlVtOvSlotIdx, const double* rlVtOvVal,
                                    const double* svcRlVt,
                                    int nDqMembers, const int* dqMemberQrGroup, const int* dqMemberRlEntry,
                                    const char* rlIslandedMask,
                                const char* dqSub, const int* qrMemberStart, double tol,
                                const char* rlGenLost, const int* qrDepBase, const int* relocOwn,
                                const double* pinTargetIn);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_PILOT_VOLTAGE_CUH
