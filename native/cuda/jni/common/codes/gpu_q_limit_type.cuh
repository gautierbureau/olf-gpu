// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// A reactive-limit entry's limit type, per scenario (limitType, the low 3 bits of rlState, rlBaseQLimitType):
// OLF's LfBus.QLimitType (ordinal + 1), NONE when the entry is not pinned, and FROZEN, the device's mark for a
// voltage control OLF's GeneratorVoltageControlManager froze or whose generators are all lost. Mirrors
// com.powsybl.openloadflow.ac.solver.gpu.codes.GpuQLimitType (GpuCodesMirrorTest checks they agree).
#ifndef OLF_GPU_COMMON_CODES_GPU_Q_LIMIT_TYPE_CUH
#define OLF_GPU_COMMON_CODES_GPU_Q_LIMIT_TYPE_CUH

namespace GpuQLimitType {
constexpr int NONE = 0;              // not pinned: the bus controls its voltage
constexpr int MIN_Q = 1;             // OLF's QLimitType.MIN_Q
constexpr int MAX_Q = 2;             // OLF's QLimitType.MAX_Q
constexpr int MIN_REALISTIC_V = 3;   // OLF's QLimitType.MIN_REALISTIC_V (robust mode)
constexpr int MAX_REALISTIC_V = 4;   // OLF's QLimitType.MAX_REALISTIC_V (robust mode)
constexpr int FROZEN = 5;            // device only: voltage control frozen or all generators lost; the RL pass leaves it
}

// OLF's QLimitType.isMinLimit / isMaxLimit.
__host__ __device__ inline bool isMinLimit(int t) {
    return t == GpuQLimitType::MIN_Q || t == GpuQLimitType::MIN_REALISTIC_V;
}

__host__ __device__ inline bool isMaxLimit(int t) {
    return t == GpuQLimitType::MAX_Q || t == GpuQLimitType::MAX_REALISTIC_V;
}

#endif
