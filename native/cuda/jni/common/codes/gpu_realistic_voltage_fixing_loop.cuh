// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Which loop OLF lets repair an unrealistic voltage before its realistic-voltage check fails the run. Mirrors
// com.powsybl.openloadflow.ac.solver.gpu.codes.GpuRealisticVoltageFixingLoop (GpuCodesMirrorTest checks they agree).
#ifndef OLF_GPU_COMMON_CODES_GPU_REALISTIC_VOLTAGE_FIXING_LOOP_CUH
#define OLF_GPU_COMMON_CODES_GPU_REALISTIC_VOLTAGE_FIXING_LOOP_CUH

namespace GpuRealisticVoltageFixingLoop {
constexpr int NONE = -1;                        // every solve is checked
constexpr int REACTIVE_LIMITS = 0;
constexpr int TRANSFORMER_VOLTAGE_CONTROL = 1;  // the discrete TransformerVoltageControl
}

#endif
