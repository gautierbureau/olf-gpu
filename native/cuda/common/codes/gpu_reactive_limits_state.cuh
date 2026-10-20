// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// A reactive-limit entry's state as one int (reactiveLimitsSeedState / reactiveLimitsFinalState): its GpuQLimitType in the low bits and its
// PV -> PQ switch count above (OLF's pvPqSwitchCount). Mirrors
// com.powsybl.openloadflow.ac.solver.gpu.codes.GpuReactiveLimitsState (GpuCodesMirrorTest checks they agree).
#ifndef OLF_GPU_COMMON_CODES_GPU_REACTIVE_LIMITS_STATE_CUH
#define OLF_GPU_COMMON_CODES_GPU_REACTIVE_LIMITS_STATE_CUH

namespace GpuReactiveLimitsState {
constexpr int LIMIT_TYPE_MASK = 7;
constexpr int PV_PQ_SWITCH_COUNT_SHIFT = 3;
}

inline int reactiveLimitsStateQLimitType(int state) {
    return state & GpuReactiveLimitsState::LIMIT_TYPE_MASK;
}

inline int reactiveLimitsStatePvPqSwitchCount(int state) {
    return state >> GpuReactiveLimitsState::PV_PQ_SWITCH_COUNT_SHIFT;
}

inline int reactiveLimitsState(int limitType, int pvPqSwitchCount) {
    return (limitType & GpuReactiveLimitsState::LIMIT_TYPE_MASK)
           | (pvPqSwitchCount << GpuReactiveLimitsState::PV_PQ_SWITCH_COUNT_SHIFT);
}

#endif
