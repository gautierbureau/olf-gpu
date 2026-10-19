// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The branch flow a flow-observed controller acts on, which is also its kind; the observable of flcFlowAndGrad.
// Mirrors com.powsybl.openloadflow.ac.solver.gpu.codes.GpuControlledFlow (GpuCodesMirrorTest checks they agree).
#ifndef OLF_GPU_COMMON_CODES_GPU_CONTROLLED_FLOW_CUH
#define OLF_GPU_COMMON_CODES_GPU_CONTROLLED_FLOW_CUH

namespace GpuControlledFlow {
constexpr int REACTIVE_POWER = 0;  // IncrementalTransformerReactivePowerControl (rho)
constexpr int ACTIVE_POWER = 1;    // IncrementalPhaseControl, mode CONTROLLER (a1)
constexpr int CURRENT = 2;         // IncrementalPhaseControl, mode LIMITER (a1, one-sided)
}

// The driver's kind mask: one bit per GpuControlledFlow.
__host__ __device__ inline int controlledFlowMaskBit(int kind) {
    return 1 << kind;
}

#endif
