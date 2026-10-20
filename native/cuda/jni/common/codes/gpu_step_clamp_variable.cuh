// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The variable type of each state row for the MAX_VOLTAGE_CHANGE step clamp (varType). Mirrors
// com.powsybl.openloadflow.ac.solver.gpu.codes.GpuStepClampVariable (GpuCodesMirrorTest checks they agree).
#ifndef OLF_GPU_COMMON_CODES_GPU_STEP_CLAMP_VARIABLE_CUH
#define OLF_GPU_COMMON_CODES_GPU_STEP_CLAMP_VARIABLE_CUH

namespace GpuStepClampVariable {
constexpr int BUS_PHI = 0;  // clamps the angle step to maxDphi
constexpr int BUS_V = 1;    // clamps the magnitude step to maxDv
constexpr int OTHER = 2;    // scaled with the step, never a clamp source
}

#endif
