// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Which OLF equation a stable row holds in a lane (docs/gpu-vs-olf.md, section 2.2). Mirrors
// com.powsybl.openloadflow.ac.solver.gpu.codes.GpuRowMode (GpuCodesMirrorTest checks they agree).
#ifndef OLF_GPU_COMMON_CODES_GPU_ROW_MODE_CUH
#define OLF_GPU_COMMON_CODES_GPU_ROW_MODE_CUH

namespace GpuRowMode {
constexpr int FIXED_VARIABLE = 0;           // x = target: BUS_TARGET_PHI, BUS_TARGET_V, DUMMY_TARGET_P/Q, a fixed tap
constexpr int EQUATION = 1;                 // the row's ordinary equation
constexpr int BUS_TARGET_V_WITH_SLOPE = 2;  // BUS_TARGET_V with the slope terms
constexpr int BRANCH_TARGET_P = 3;          // the controlled branch's P on a phase controller's a1 row
constexpr int BUS_TARGET_Q = 4;             // the bus's own BUS_TARGET_Q (PV -> PQ switch, remote member's balance)
constexpr int CROSS_FIXED_VARIABLE = 6;     // device only: a fixed variable written on another row
}

#endif
