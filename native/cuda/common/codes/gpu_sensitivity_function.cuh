// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// A sensitivity factor's function as the device contracts it: a closed branch's active power, reactive power or current
// magnitude on side 1 or 2, or a bus voltage. Mirrors com.powsybl.openloadflow.ac.solver.gpu.codes.GpuSensitivityFunction
// (GpuCodesMirrorTest checks they agree).
#ifndef OLF_GPU_COMMON_CODES_GPU_SENSITIVITY_FUNCTION_CUH
#define OLF_GPU_COMMON_CODES_GPU_SENSITIVITY_FUNCTION_CUH

namespace GpuSensitivityFunction {
constexpr int ACTIVE_POWER_1 = 0;
constexpr int REACTIVE_POWER_1 = 1;
constexpr int ACTIVE_POWER_2 = 2;
constexpr int REACTIVE_POWER_2 = 3;
constexpr int CURRENT_1 = 4;
constexpr int CURRENT_2 = 5;
constexpr int BUS_VOLTAGE = 6;
}

#endif
