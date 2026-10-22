// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Bit flags of a reactive-limit entry (reactiveLimitFlags). Mirrors
// com.powsybl.openloadflow.ac.solver.gpu.codes.GpuReactiveLimitsEntryFlag (GpuCodesMirrorTest checks they agree).
#ifndef OLF_GPU_COMMON_CODES_GPU_REACTIVE_LIMITS_ENTRY_FLAG_CUH
#define OLF_GPU_COMMON_CODES_GPU_REACTIVE_LIMITS_ENTRY_FLAG_CUH

namespace GpuReactiveLimitsEntryFlag {
constexpr int STAND_BY_AUTOMATON = 1;       // an SVC standing by under voltage monitoring: seeded as PV
constexpr int NOT_CONTROLLER_ELEMENT = 2;  // not one of its voltage control's controller elements
}

#endif
