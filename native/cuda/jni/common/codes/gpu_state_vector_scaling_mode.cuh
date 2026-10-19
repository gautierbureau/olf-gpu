// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// OLF's StateVectorScalingMode as the device receives it (lsMode). Mirrors
// com.powsybl.openloadflow.ac.solver.gpu.codes.GpuStateVectorScalingMode (GpuCodesMirrorTest checks they agree).
#ifndef OLF_GPU_COMMON_CODES_GPU_STATE_VECTOR_SCALING_MODE_CUH
#define OLF_GPU_COMMON_CODES_GPU_STATE_VECTOR_SCALING_MODE_CUH

namespace GpuStateVectorScalingMode {
constexpr int NONE = 0;                // undamped Newton step
constexpr int LINE_SEARCH = 1;         // per-scenario backtracking (LineSearchStateVectorScaling)
constexpr int MAX_VOLTAGE_CHANGE = 2;  // per-scenario step clamp (MaxVoltageChangeStateVectorScaling)
}

#endif
