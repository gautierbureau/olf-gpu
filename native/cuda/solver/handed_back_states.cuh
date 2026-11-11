// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The states a batched context's last solve call handed back, where they are on the device: what the violation
// screen's handback (sa/olf_gpu_violations.cu) reads between two calls when the context keeps its states on the device
// (BatchSolverContext::statesToHost off). Declared here so that the screen's translation unit needs nothing else of the
// solver's context.
#ifndef OLF_GPU_SOLVER_HANDED_BACK_STATES_CUH
#define OLF_GPU_SOLVER_HANDED_BACK_STATES_CUH

#include <jni.h>
#include <vector>

namespace olfgpu {

struct HandedBackStates {
    const double* dX = nullptr;                              // the context's states, n doubles per bank slot
    int n = 0;
    std::vector<int> slots;                                  // the handed-back bank slots, in the order of the result rows
};

// The last solve call's handed-back states of the batched context {@code handle} (defined in jni/batch_solve_jni.cu).
HandedBackStates handedBackStates(jlong handle);

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_HANDED_BACK_STATES_CUH
