// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Which of OLF's discrete transformer voltage control loops the device runs (transformerVoltageControlMode), named after OLF's
// TransformerVoltageControlMode constants. Mirrors
// com.powsybl.openloadflow.ac.solver.gpu.codes.GpuTransformerVoltageControlMode (GpuCodesMirrorTest checks they agree).
#ifndef OLF_GPU_COMMON_CODES_GPU_TRANSFORMER_VOLTAGE_CONTROL_MODE_CUH
#define OLF_GPU_COMMON_CODES_GPU_TRANSFORMER_VOLTAGE_CONTROL_MODE_CUH

namespace GpuTransformerVoltageControlMode {
constexpr int AFTER_GENERATOR_VOLTAGE_CONTROL = 0;  // TransformerVoltageControlOuterLoop
constexpr int WITH_GENERATOR_VOLTAGE_CONTROL = 1;   // SimpleTransformerVoltageControlOuterLoop
}

#endif
