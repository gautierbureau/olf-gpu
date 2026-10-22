// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Where a member of a discrete (transformer or shunt) voltage control stands, per scenario. Mirrors
// com.powsybl.openloadflow.ac.solver.gpu.codes.GpuDiscreteControlMemberState (GpuCodesMirrorTest checks they agree).
#ifndef OLF_GPU_COMMON_CODES_GPU_DISCRETE_CONTROL_MEMBER_STATE_CUH
#define OLF_GPU_COMMON_CODES_GPU_DISCRETE_CONTROL_MEMBER_STATE_CUH

namespace GpuDiscreteControlMemberState {
constexpr int DISABLED = 0;    // the member's voltage control is disabled
constexpr int CONTINUOUS = 1;  // the ratio or susceptance is solved continuously
constexpr int ROUNDED = 2;     // rounded to a tap or section and held there
}

#endif
