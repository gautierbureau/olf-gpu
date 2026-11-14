// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The part of a batched solver context that The generators whose target Q is forced: their curves and the limits applied.
#ifndef OLF_GPU_SOLVER_CONTEXT_FORCED_TARGET_Q_CONTEXT_CUH
#define OLF_GPU_SOLVER_CONTEXT_FORCED_TARGET_Q_CONTEXT_CUH

#include <cublas_v2.h>

#include <vector>

#include "../../common/device_buf.cuh"
#include "../slot_rows.cuh"
#include "../slot_table.cuh"

namespace olfgpu {

struct ForcedTargetQContext {
    // forceTargetQInReactiveLimits (GpuBatchForcedTargetQ): generators whose IIDM target Q OLF clamps into their
    // curve limits at their CURRENT target P - on a bus whose Q balance carries it (qRow), or in a voltage-controlled
    // bus's reactive limits (reactiveLimitEntry). forcedTargetQIndices stride 7 {qRow, distributedSlackGeneratorIndex, ptsOff, npts, distributedSlackBusIndex, baseQRowMode, reactiveLimitEntry}; forcedTargetQParameters stride 5 {basePpu, rawTargetQPu, baseClampedPu, share,
    // baseRowTargetP}; dForcedTargetQApplied [nS * nForcedTargetQ] = the clamped value the scenario's Q target carries now; dForcedTargetQPreDistribution
    // [nS * nForcedTargetQ] = each generator's loss pre-distribution move (pu), absent when no scenario pre-distributes.
    int nForcedTargetQ = 0;
    int nForcedTargetQLimitEntries = 0;                                        // of which limit entries (voltage-controlled bus)
    int forcedTargetQExtrapolate = 0;
    SlotRows<double> forcedTargetQPreDistribution{0.0};
    DeviceBuf dForcedTargetQIndices, dForcedTargetQParameters, dForcedTargetQPoints, dForcedTargetQApplied;
};

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_CONTEXT_FORCED_TARGET_Q_CONTEXT_CUH
