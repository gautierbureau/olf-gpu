// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// How a factor group combines a basis column of a batched sensitivity analysis's contraction: a unit column by the activity
// of a bus's active power balance (P), reactive power balance (Q) or voltage target (V), a weighted or participation column
// as it is (ONE). Mirrors com.powsybl.openloadflow.ac.solver.gpu.codes.GpuSensitivityBasisKind (GpuCodesMirrorTest
// checks they agree).
#ifndef OLF_GPU_COMMON_CODES_GPU_SENSITIVITY_BASIS_KIND_CUH
#define OLF_GPU_COMMON_CODES_GPU_SENSITIVITY_BASIS_KIND_CUH

namespace GpuSensitivityBasisKind {
constexpr int P = 0;
constexpr int Q = 1;
constexpr int V = 2;
constexpr int ONE = 3;
}

#endif
