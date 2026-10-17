// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// ReactiveLimitsOuterLoop state carried across solves: the per-(controller, scenario) limit type seeded from the
// base or from a previous solve's pins, and the pins exported for the next one.
#ifndef OLF_GPU_OUTERLOOP_REACTIVE_LIMITS_SEED_CUH
#define OLF_GPU_OUTERLOOP_REACTIVE_LIMITS_SEED_CUH

#include "../solver/driver_pass.cuh"

namespace olfgpu {

__global__ void initRlLimitTypeBatch(int nrl, int nScenarios, int n, const int* rlVrow, const int* rlQrow,
                                     const double* rlVtarget, const char* rlMonitored,
                                     const int* baseMode, const double* x, char* limitType,
                                     int nRlVtOv, const int* rlVtOvSlotIdx, const double* rlVtOvVal, const double* svcRlVt,
                                     const char* rlRemote, const int* rlCmpVrow,
                                     const double* target, const double* rlMinQb, const double* rlMaxQb,
                                     const double* rlLoadQb, const char* rlBaseLt,
                                     const int* psEntry, int nQr, const char* rlGenLost);

__global__ void seedRlModeFromState(int nrl, int nScenarios, int n, const int* rlVrow, const int* rlQrow,
                                    const double* rlMinQ, const double* rlMaxQ, const double* rlLoadQ,
                                    const char* limitType, int* mode, double* target, int cvStride,
                                    const char* rlRemote, double* savedTarget, char* savedMode,
                                    const char* rlBaseLt, const int* psEntry, int nQr,
                                    const double* rlVtarget, int nRlVtOv, const int* rlVtOvSlotIdx, const double* rlVtOvVal,
                                    const double* svcRlVt, const char* rlLocalOv, const double* rlLocalVt,
                                    const double* pinTargetIn, const char* rlIslanded);

__global__ void exportRlPinTargets(int nrl, int nScenarios, int n, const int* rlVrow, const int* rlQrow,
                                   const char* limitType, const int* mode, const double* target, const char* rlIslanded,
                                   int nQr, const int* psEntry, const int* qrPtr, const int* qrRows, const char* veqOff,
                                   const int* qrDepBase, double* out);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_REACTIVE_LIMITS_SEED_CUH
