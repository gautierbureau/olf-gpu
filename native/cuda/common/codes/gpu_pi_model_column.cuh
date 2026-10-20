// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// A pi model packed as COUNT doubles, in the order of OLF's PiModel getters (tap tables, closedBranchParameters). Mirrors
// com.powsybl.openloadflow.ac.solver.gpu.codes.GpuPiModelColumn (GpuCodesMirrorTest checks they agree).
#ifndef OLF_GPU_COMMON_CODES_GPU_PI_MODEL_COLUMN_CUH
#define OLF_GPU_COMMON_CODES_GPU_PI_MODEL_COLUMN_CUH

namespace GpuPiModelColumn {
constexpr int Y = 0;
constexpr int G1 = 1;
constexpr int G2 = 2;
constexpr int B1 = 3;
constexpr int B2 = 4;
constexpr int KSI = 5;
constexpr int R1 = 6;     // the ratio: the voltage control loops' tap value
constexpr int A1 = 7;     // the phase shift: the phase control loops' tap value
constexpr int COUNT = 8;
}

#endif
