// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// BatchSolverContext: one batched consumer of a shared structure (its per-scenario value and state
// arrays, outer-loop tables and cuDSS objects).
#ifndef OLF_GPU_SOLVER_BATCH_CONTEXT_CUH
#define OLF_GPU_SOLVER_BATCH_CONTEXT_CUH

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudss.h>

#include <limits>
#include <map>
#include <memory>
#include <vector>

#include "../common/batch_layout.cuh"
#include "../common/device_buf.cuh"
#include "../common/loop_kinds.cuh"
#include "../common/codes/gpu_distributed_slack_factor_mode.cuh"
#include "../common/codes/gpu_ac_solver_status.cuh"
#include "../common/codes/gpu_outer_loop_kind.cuh"
#include "../common/codes/gpu_realistic_voltage_fixing_loop.cuh"
#include "../common/codes/gpu_transformer_voltage_control_mode.cuh"
#include "../lu/batched_lu.cuh"
#include "../structure/shared_structure.cuh"
#include "context/reactive_limits_context.cuh"
#include "context/area_interchange_control_context.cuh"
#include "context/distributed_slack_context.cuh"
#include "context/transformer_voltage_control_context.cuh"
#include "context/incremental_transformer_voltage_control_context.cuh"
#include "context/shunt_voltage_control_context.cuh"
#include "context/incremental_shunt_voltage_control_context.cuh"
#include "context/secondary_voltage_control_context.cuh"
#include "context/incremental_flow_control_context.cuh"
#include "context/phase_control_context.cuh"
#include "context/automation_system_context.cuh"
#include "context/monitoring_voltage_context.cuh"
#include "context/generator_voltage_control_freeze_context.cuh"
#include "context/forced_target_q_context.cuh"
#include "context/hvdc_ac_emulation_context.cuh"
#include "context/realistic_voltage_context.cuh"
#include "context/remote_voltage_control_context.cuh"
#include "context/linear_solver_context.cuh"
#include "context/newton_raphson_context.cuh"
#include "context/scenario_overrides_context.cuh"
#include "slot_rows.cuh"
#include "slot_solve_state.cuh"
#include "slot_table.cuh"

namespace olfgpu {

// The batched solver context: the SHARED structure (pattern, families, destination arrays,
// static trig — held by shared_ptr, see SharedStructure) + per-scenario value/state arrays and this consumer's own cuDSS
// objects/analysis. Structure notes that are batch-specific:
// - DISTR_Q: the controller structure is fixed across scenarios; the per-scenario disable
//   mask drops an outaged contributor's term (kinds 0/2 = closed/open).
// - DISTR_RHO: no per-scenario mask — a contingency that outages a controlling transformer
//   is rejected to the CPU (the 1/count coefficients would reform).
// - Zero-impedance: the base couplers are replicated; a scenario opens or closes couplers through
//   its per-scenario on/off overrides (zeroImpedanceBranchOpenOverrides / zeroImpedanceBranchCloseOverrides).
// The sensitivity basis tables of a context (see BatchSolverContext::sensitivityBasis).
struct SensitivityBasisTables {
    int nUnits = 0;
    int nWeighted = 0;
    int nColumns = 0;                                        // nUnits + nWeighted, + 1 with a participation column
    int nGather = 0;
    std::vector<int> hUnitDeviceRows;
    DeviceBuf dWeighted;                                     // nWeighted x n
    DeviceBuf dGatherRows;                                   // nGather
    // The contraction of the basis solutions into the factors' sensitivities (setBatchSensitivityContraction; nFactors 0:
    // none, the solutions themselves are read back). Unit rows on DEVICE rows; per unit column, whether its bus is the slack.
    int nFactors = 0;
    int nGroups = 0;
    DeviceBuf dUnitRows, dUnitBusPRows, dUnitSlack;          // nUnits each (int, int, char)
    DeviceBuf dGroupColumnStart, dGroupColumns, dGroupColumnKind;   // CSR by group: nGroups + 1, then one per entry
    DeviceBuf dFactorGroup, dFactorFunction, dFactorElement; // nFactors each
    DeviceBuf dFactorGather;                                 // 4 x nFactors: gather positions (ph1 v1 ph2 v2; a voltage: v)
};

struct BatchSolverContext {
    int loopTraceSlot = -1;                                  // OLF_GPU_LOOP_TRACE by CONTINGENCY: the slot the host resolved for this chunk
    // The bank slots the per-scenario setters install: the setters' local slot j is bank slot installSlots[j], and only
    // those slots' entries are rewritten. Empty: every slot, local slot = bank slot (a whole chunk).
    std::vector<int> installSlots;
    // Whether a solve call copies the handed-back slots' states to the host (its result rows). Off, they stay in dX until
    // the next call, which the host reads selectively in between (the violation screen's handback: screenHandedBack,
    // gatherHandedBack, readHandedBackStates), and the rows carry the iterations, norm and status alone.
    bool statesToHost = true;
    // The installed slots' rows of the per-slot arrays an install writes, in one copy (writeInstalledRows).
    mutable SlotRowUpload installRowUpload;
    DeviceBuf dInstallSlots;                                 // the install slots on the device (every slot of an install of the bank)
    DeviceBuf dInstallMask;                                  // per slot, 1: installed by the current install
    SlotSolveState slotState;                                // per slot: its scenario's solve so far (solveBatchContext)
    DeviceBuf dXBase, dBaseMode;                             // the base state and row modes an installed slot starts from
    DistributedSlackContext distributedSlack;
    SharedStructurePtr structure;                                   // built once per extraction, see SharedStructure
    int n = 0;
    int nnz = 0;
    int nScenarios = 0;
    int nElements = 0;
    int closedBranchOffset = 0, openBranchOffset = 0, shuntOffset = 0, hvdcAcEmulationOffset = 0;   // element-mask offsets per family
    int nDistrQ = 0, nDistrRho = 0, nDistrShuntB = 0, nZeroImpedanceBranch = 0;                  // copies of the structure counts
    DeviceBuf dX, dTarget, dF, dB, dDx, dVals, dMode;        // per-scenario (n or nnz strided)
    // Where the fills put a scenario's Jacobian values, mismatch and closed-branch Q buffer (common/batch_layout.cuh):
    // scenario-interleaved on our batched LU, whose lanes are interleaved too, scenario-major for cuDSS. dVals and
    // dQbufClosed are laid out so; the mismatch is filled into dFillMismatch and copied into dF, scenario-major for every
    // reader, once the fill is done (fillMismatch).
    BatchLayout layout;
    DeviceBuf dFillMismatch;
    // nS: 1 while a retired scenario's values and F hold the identity system writeIdentityForInactive wrote, so the next
    // identity write skips it. Cleared at every batched solve's start and for every scenario zeroActiveValues zeroes (a
    // check's refill writes no values: its fills get no Jacobian)
    DeviceBuf dIdentityLane;
    // per scenario: the state at its last CONVERGED inner solve (its starting state before any): what OLF's network
    // holds when a run fails (NewtonRaphson updates the buses only on convergence), and so where OLF starts the
    // strategies of a contingency that ran out of iterations (PreviousValueVoltageInitializer)
    DeviceBuf dXLastConv;
    DeviceBuf dHandBackSlots, dHandBackStates;               // a paused call's handed-back slots and their states, gathered
    DeviceBuf dTargetBase;                                   // the base targets (n), before tiling/overrides
    LinearSolverContext linearSolver;
    NewtonRaphsonContext newtonRaphson;
    ScenarioOverridesContext scenarioOverrides;
    // setBatchX0Seeds: per-scenario starting values (flat s*n+row -> value) written over the tiled BASE state by
    // the next solve that is given no full per-scenario state; consumed (cleared) by that solve.
    std::vector<int> x0SeedIdx;
    std::vector<double> x0SeedVal;
    DeviceBuf dDisabled, dActive;                            // per-scenario masks
    ReactiveLimitsContext reactiveLimits;
    RemoteVoltageControlContext remoteVoltageControl;
    ForcedTargetQContext forcedTargetQ;
    DeviceBuf dRowSlope;                                   // per row: SLOPE-SVC slope (BUS_TARGET_V_WITH_SLOPE V rows), 0 elsewhere
    // dQbuf holds the q of every row (dQbufClosed's closed branches plus the other families, refilled at every check);
    // dOuterLoopCheckActive marks the scenarios an outer loop checks this pass, dOuterLoopUnstable those it changed.
    DeviceBuf dQbuf, dOuterLoopCheckActive, dOuterLoopUnstable;               // dQbuf nS*n, dOuterLoopCheckActive/dOuterLoopUnstable nS
    // nS*n: the closed branches' part of dQbuf, written by the Newton fills (each zeroes its active scenarios' rows and
    // adds the p and q it puts in F, on the frozen rows) and copied into dQbuf by every check's refill
    DeviceBuf dQbufClosed;
    RealisticVoltageContext realisticVoltage;
    long long solveCount = 0;                                // solveBatchContext calls on this context: the DS
                                                             // read-backs above describe the LAST solve only
    int newtonLoopIterations = 0;                            // the batched Newton iterations of the last solveBatchContext call
    AreaInterchangeControlContext areaInterchangeControl;
    AutomationSystemContext automationSystem;
    std::vector<int> outerLoopFirstSolveIterations, outerLoopIterations, outerLoopSolvePasses, outerLoopChecks;              // OLF_GPU_PHASE_TIME: per-scenario Newton work by
                                                             // solve pass
    MonitoringVoltageContext monitoringVoltage;
    IncrementalTransformerVoltageControlContext incrementalTransformerVoltageControl;
    IncrementalShuntVoltageControlContext incrementalShuntVoltageControl;
    IncrementalFlowControlContext incrementalFlowControl;
    PhaseControlContext phaseControl;
    TransformerVoltageControlContext transformerVoltageControl;
    SecondaryVoltageControlContext secondaryVoltageControl;

    GeneratorVoltageControlFreezeContext generatorVoltageControlFreeze;
    ShuntVoltageControlContext shuntVoltageControl;

    SlotTable frozenRows{1, 0};                            // per slot: the rows an islanding outage froze to a flat identity state
    SlotRows<double> reactiveLimitSeedTargetQ{std::numeric_limits<double>::quiet_NaN()};   // per (slot, entry): the frozen target Q a continued
                                                                         // solve restores (setBatchReactiveLimitSeedTargetQ), NaN = the limit
    // The sensitivity basis a batched sensitivity analysis has every converged scenario solve (setBatchSensitivityBasis,
    // sensitivityBasisStep): the unit columns' device rows (rowMap applied, -1 = no row), the weighted columns dense on
    // device rows, the gather rows (state rows); per slot its participation column on device rows; and the solutions,
    // per slot nColumns x nGather (column-major by basis column).
    SensitivityBasisTables sensitivityBasis;
    SlotRows<double> sensitivityParticipation{0.0};
    DeviceBuf dSensitivityBasisOut;
    DeviceBuf dSensitivityValuesOut;                         // per slot: nFactors sensitivities, then 2 x nUnits row modes

    HvdcAcEmulationContext hvdcAcEmulation;
    // Dedicated stream for the WHOLE batch solve (every kernel/memset/async copy + cuDSS via
    // cudssSetStream): keeps the per-iteration launch sequence off the legacy default stream,
    // whose implicit device-wide serialization stitched the timeline with sync points (the
    // ~25 ms/iter host/launch gaps of the 2026-07-15 nsys attribution — a loop kept on one
    // explicit stream shows none). Created BLOCKING (default flags), so any
    // op left on the legacy stream still orders correctly against it — safety over speed for
    // whatever this conversion misses.
    cudaStream_t stream = nullptr;
    ~BatchSolverContext() {
        if (secondaryVoltageControl.secondaryVoltageControlBlasHandle != nullptr) {
            cublasDestroy(secondaryVoltageControl.secondaryVoltageControlBlasHandle);
            secondaryVoltageControl.secondaryVoltageControlBlasHandle = nullptr;
        }
        if (stream) cudaStreamDestroy(stream);
        if (linearSolver.uniformBatchResidualMatrix) cudssMatrixDestroy(linearSolver.uniformBatchResidualMatrix);
        if (linearSolver.uniformBatchDeltaMatrix) cudssMatrixDestroy(linearSolver.uniformBatchDeltaMatrix);
        if (linearSolver.compactedUniformBatchResidualMatrix) cudssMatrixDestroy(linearSolver.compactedUniformBatchResidualMatrix);
        if (linearSolver.compactedUniformBatchDeltaMatrix) cudssMatrixDestroy(linearSolver.compactedUniformBatchDeltaMatrix);
        if (linearSolver.uniformBatchSystemMatrix) cudssMatrixDestroy(linearSolver.uniformBatchSystemMatrix);
        if (linearSolver.uniformBatchRightHandSideMatrix) cudssMatrixDestroy(linearSolver.uniformBatchRightHandSideMatrix);
        if (linearSolver.uniformBatchSolutionMatrix) cudssMatrixDestroy(linearSolver.uniformBatchSolutionMatrix);
        for (auto& e : linearSolver.compactData) {
            if (e.second && linearSolver.compactedSolver.handle) cudssDataDestroy(linearSolver.compactedSolver.handle, e.second);
        }
        linearSolver.compactData.clear();
        if (linearSolver.compactedUniformBatchSystemMatrix) cudssMatrixDestroy(linearSolver.compactedUniformBatchSystemMatrix);
        if (linearSolver.compactedUniformBatchRightHandSideMatrix) cudssMatrixDestroy(linearSolver.compactedUniformBatchRightHandSideMatrix);
        if (linearSolver.compactedUniformBatchSolutionMatrix) cudssMatrixDestroy(linearSolver.compactedUniformBatchSolutionMatrix);
    }
};

// How many scenarios the setters now install: the install slots', or the whole bank's.
inline int installedSlotCount(const BatchSolverContext* ctx) {
    return ctx->installSlots.empty() ? ctx->nScenarios : static_cast<int>(ctx->installSlots.size());
}

// The bank slot of the install's local slot j.
inline int installedSlot(const BatchSolverContext* ctx, int j) {
    return ctx->installSlots.empty() ? j : ctx->installSlots[j];
}

// The slack's own target move as the distributed slack over DS buses reads it: none on a context without DS buses (a
// slack-only distribution seeds from it at reset instead).
inline double* distributedSlackBusesSlackShift(const BatchSolverContext* ctx) {
    return ctx->distributedSlack.nDistributedSlackBuses > 0 ? ctx->distributedSlack.slackBusAnchorShift.deviceOrNull() : nullptr;
}

// Build a batched context of nScenarios on a shared structure (the createBatchContext JNI entry).
BatchSolverContext* createBatchContext(const SharedStructurePtr& structure, int nScenarios, int mode,
                                       int nReactiveLimitEntries, const int* reactiveLimitVoltageRow, const double* reactiveLimitMinQ,
                                       const double* reactiveLimitMaxQ, const double* reactiveLimitLoadQ, const double* reactiveLimitTargetV,
                                       const signed char* reactiveLimitFlags, const double* rowSlope,
                                       int nDistributedSlackBuses, const int* distributedSlackBusPhiRow, const double* distributedSlackBusFactor,
                                       int slackBusPhiRow, double slackBusTargetP,
                                       const double* distributedSlackBusBaseTargetP, const double* distributedSlackBusMaxDelta, const double* distributedSlackBusMinDelta,
                                       int distributedSlackFactorMode, const double* distributedSlackBusMarginUp, const double* distributedSlackBusMarginDown,
                                       double slackBusMarginUp, double slackBusMarginDown);

// Upload the call's step-scaling parameters and per-scenario shunt, coupler and tap overrides.
void uploadBatchOverrides(BatchSolverContext* ctx, int stateVectorScalingMode, double maxDv, double maxDphi,
                          const int* varType, int nShuntAdmittanceOverrides, const int* shuntAdmittanceOverrideScenarioElement, const double* shuntAdmittanceOverrideDelta,
                          int nBaseClosedZeroImpedanceBranches, int nZeroImpedanceBranchOpenOverrides, const int* zeroImpedanceBranchOpenScenarioElement,
                          int nZeroImpedanceBranchCloseOverrides, const int* zeroImpedanceBranchCloseScenarioElement,
                          int nClosedBranchParameterOverrides, const int* closedBranchParameterOverrideScenarioElement, const double* closedBranchParameterOverrideValues);

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_BATCH_CONTEXT_CUH
