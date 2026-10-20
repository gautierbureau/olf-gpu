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
                              int nTargetOverrides, const int* targetOverrideIndex, const double* targetOverrideValue,
                              const int* distributedSlackOverrideBus, const double* distributedSlackOverrideFactorDelta, const double* distributedSlackOverrideBaseTargetPDelta,
                              const double* distributedSlackOverrideMaxDelta, const double* distributedSlackOverrideMinDelta, const double* distributedSlackBusAnchorShift,
                              const double* slackBusAnchorShift,
                              int nReactiveLimitTargetVOverrides, const int* reactiveLimitTargetVOverrideScenarioEntry, const double* reactiveLimitTargetVOverrideValue,
                              int nReactiveLimitDeltaOverrides, const int* reactiveLimitDeltaOverrideScenarioEntry, const double* reactiveLimitDeltaOverrideValues,
                              int nReactiveCapabilityCurveOverrides, const int* reactiveCapabilityCurveOverrideScenarioCurve, const double* reactiveCapabilityCurveOverrideStepScale,
                              int maxIter, double tol,
                              double reactiveLimitTol, double slackDistTol, double convEpsPerEq,
                              int stateVectorScalingMode, double maxDv, double maxDphi, const int* varType,
                              const double* x0Full, int nShuntAdmittanceOverrides, const int* shuntAdmittanceOverrideScenarioElement, const double* shuntAdmittanceOverrideDelta,
                              const int* reactiveLimitsSeedState, int* reactiveLimitsFinalState,
                              int nClosedBranchParameterOverrides, const int* closedBranchParameterOverrideScenarioElement, const double* closedBranchParameterOverrideValues,
                              int maxOuterProductive, int maxPqPvSwitch, int rejectSlackDistributionFailure, int skipReactiveLimitsPhase,
                              int nBaseClosedZeroImpedanceBranches, int nZeroImpedanceBranchCloseOverrides, const int* zeroImpedanceBranchCloseScenarioElement,
                              int nZeroImpedanceBranchOpenOverrides, const int* zeroImpedanceBranchOpenScenarioElement);

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_OUTER_LOOP_DRIVER_CUH
