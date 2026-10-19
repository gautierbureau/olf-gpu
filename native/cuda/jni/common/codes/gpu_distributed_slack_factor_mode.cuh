// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// How the device computes each distributed-slack participant's factor (dsMode). Mirrors
// com.powsybl.openloadflow.ac.solver.gpu.codes.GpuDistributedSlackFactorMode (GpuCodesMirrorTest checks they agree).
#ifndef OLF_GPU_COMMON_CODES_GPU_DISTRIBUTED_SLACK_FACTOR_MODE_CUH
#define OLF_GPU_COMMON_CODES_GPU_DISTRIBUTED_SLACK_FACTOR_MODE_CUH

namespace GpuDistributedSlackFactorMode {
constexpr int STATIC = 0;            // the factor is fixed for the run (dsFactor)
constexpr int REMAINING_MARGIN = 1;  // OLF's REMAINING_MARGIN: the room left in the mismatch's direction
}

#endif
