// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// OLF's TransformerPhaseControl.Mode of a continuous phase control, with OLF's ordinals. Mirrors
// com.powsybl.openloadflow.ac.solver.gpu.codes.GpuPhaseControlMode (GpuCodesMirrorTest checks they agree).
#ifndef OLF_GPU_COMMON_CODES_GPU_PHASE_CONTROL_MODE_CUH
#define OLF_GPU_COMMON_CODES_GPU_PHASE_CONTROL_MODE_CUH

namespace GpuPhaseControlMode {
constexpr int CONTROLLER = 0;  // controls the active power flow (a1 continuous, then discretised)
constexpr int LIMITER = 1;     // limits the current (a1 tap by tap)
}

#endif
