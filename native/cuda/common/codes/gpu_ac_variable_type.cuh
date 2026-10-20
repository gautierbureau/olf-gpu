// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// OLF's AcVariableType, by ordinal, for the variable types the device reads (the MAX_VOLTAGE_CHANGE step clamp's
// varType). Mirrors com.powsybl.openloadflow.ac.solver.gpu.codes.GpuAcVariableType, whose constants ARE OLF's
// ordinals (GpuCodesMirrorTest checks they agree).
#ifndef OLF_GPU_COMMON_CODES_GPU_AC_VARIABLE_TYPE_CUH
#define OLF_GPU_COMMON_CODES_GPU_AC_VARIABLE_TYPE_CUH

namespace GpuAcVariableType {
constexpr int BUS_V = 0;    // clamps the magnitude step to maxDv
constexpr int BUS_PHI = 1;  // clamps the angle step to maxDphi
}

#endif
