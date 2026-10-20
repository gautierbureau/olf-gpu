// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Which of OLF's discrete transformer voltage control loops the device runs (transformerVoltageControlMode): OLF's
// OpenLoadFlowParameters.TransformerVoltageControlMode, by ordinal. Mirrors
// com.powsybl.openloadflow.ac.solver.gpu.codes.GpuTransformerVoltageControlMode, whose constants ARE OLF's ordinals
// (GpuCodesMirrorTest checks they agree).
#ifndef OLF_GPU_COMMON_CODES_GPU_TRANSFORMER_VOLTAGE_CONTROL_MODE_CUH
#define OLF_GPU_COMMON_CODES_GPU_TRANSFORMER_VOLTAGE_CONTROL_MODE_CUH

namespace GpuTransformerVoltageControlMode {
constexpr int WITH_GENERATOR_VOLTAGE_CONTROL = 0;   // SimpleTransformerVoltageControlOuterLoop
constexpr int AFTER_GENERATOR_VOLTAGE_CONTROL = 1;  // TransformerVoltageControlOuterLoop
}

#endif
