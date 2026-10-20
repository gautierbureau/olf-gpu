// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The role of a reactive-limit entry's bus in its generator voltage control (reactiveLimitVoltageControlRole). Mirrors
// com.powsybl.openloadflow.ac.solver.gpu.codes.GpuVoltageControlRole (GpuCodesMirrorTest checks they agree).
#ifndef OLF_GPU_COMMON_CODES_GPU_VOLTAGE_CONTROL_ROLE_CUH
#define OLF_GPU_COMMON_CODES_GPU_VOLTAGE_CONTROL_ROLE_CUH

namespace GpuVoltageControlRole {
constexpr int LOCAL = 0;                      // a local controller
constexpr int REMOTE = 1;                     // a remote member OLF's robust mode rule applies to
constexpr int PILOT_SELF = 2;                 // the controlled bus of a remote control, also one of its controllers
constexpr int REMOTE_ROBUST_MODE_EXEMPT = 3;  // a remote member the robust mode rule exempts (own control local)
}

#endif
