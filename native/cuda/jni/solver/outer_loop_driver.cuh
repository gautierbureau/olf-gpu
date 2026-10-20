// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The outer-loop driver of a batched solve (OLF AcloadFlowEngine.run / runOuterLoop): the per-call set-up and
// resume, the loop order, the sweep that drives each in-kernel outer loop to stable, the Newton re-solves and
// the stable/unstable bookkeeping.
#ifndef OLF_GPU_SOLVER_OUTER_LOOP_DRIVER_CUH
#define OLF_GPU_SOLVER_OUTER_LOOP_DRIVER_CUH

#include "../solver/driver_pass.cuh"

namespace olfgpu {

// One batched solve of every scenario of the context (see solveBatchContext in the JNI layer for the arguments).
BatchResult solveBatchContext(BatchSolverContext* ctx, const double* x0, const double* target,
                              const int* rowMode, const int* rowModeSlots, int nRowModeSlots,
                              const int* baseRowMode, const char* disabled,
                              int nOv, const int* ovIndex, const double* ovValue,
                              const int* dsOvBus, const double* dsOvFactorDelta, const double* dsOvBaseDelta,
                              const double* dsOvMaxDelta, const double* dsOvMinDelta, const double* dsAnchorShift,
                              const double* dsSlackShift,
                              int nRlVtOv, const int* rlVtOvSlotIdx, const double* rlVtOvVal,
                              int nRlQOv, const int* rlQOvSlotEntry, const double* rlQOvMinMax,
                              int nCvOv, const int* cvOvSlotEntry, const double* cvOvStepScale,
                              int maxIter, double tol,
                              double reactiveLimitTol, double slackDistTol, double convEpsPerEq,
                              int lsMode, double maxDv, double maxDphi, const int* varType,
                              const double* x0Full, int nShuntAdmittanceOverrides, const int* shuntAdmittanceOverrideScenarioElement, const double* shuntAdmittanceOverrideDelta,
                              const int* rlStateIn, int* rlStateOut,
                              int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal,
                              int maxOuterProductive, int rlMaxSwitch, int dsRejectResidual, int skipRlPhase,
                              int nziBase, int nZiOn, const int* ziOnSlotElem,
                              int nZiOff, const int* ziOffSlotElem);

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_OUTER_LOOP_DRIVER_CUH
