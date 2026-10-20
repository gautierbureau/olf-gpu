// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// A branch side as the device reads it: powsybl's TwoSides, by ordinal (not getNum()). Mirrors
// com.powsybl.openloadflow.ac.solver.gpu.codes.GpuTwoSides, whose constants ARE TwoSides' ordinals
// (GpuCodesMirrorTest checks they agree).
#ifndef OLF_GPU_COMMON_CODES_GPU_TWO_SIDES_CUH
#define OLF_GPU_COMMON_CODES_GPU_TWO_SIDES_CUH

namespace GpuTwoSides {
constexpr int NONE = -1;  // no side: a coupler with no relocated child, a shunt term, a plain generator's curve entry
constexpr int ONE = 0;    // side 1
constexpr int TWO = 1;    // side 2
}

#endif
