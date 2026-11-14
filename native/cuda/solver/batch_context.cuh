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
    DeviceBuf dZeroDiagonalRowSnapshot, dZeroDiagonalRightHandSideSnapshot, dZeroDiagonalDonorChoice;               // zero-diagonal repair scratch (lazy)
    int zeroDiagonalRepairCapacity = 0;
    bool clampStep = false;                                  // MAX_VOLTAGE_CHANGE per-scenario step clamp (set per solve)
    double maxDv = 0.0, maxDphi = 0.0;
    DeviceBuf dVarType, dStepFactor;                         // n ints / nS factors (lazy, first clamped solve)
    DeviceBuf dLineSearchPreviousX;                                      // n*nS pre-step iterates (lazy, LINE_SEARCH solves)
    int lineSearchStateVectorScalingMaxIteration = 10;                                      // LINE_SEARCH backtracks (setBatchLineSearch)
    double lineSearchStateVectorScalingStepFold = 4.0 / 3.0;
    DeviceBuf dNorms;                                        // nS*4 device-side convergence norms (see residualNormsBatch)
    DeviceBuf dFixedVariableRowMaxMismatch, dEquationRowMaxMismatch;                          // n each: per-row PER_EQUATION_TYPE thresholds (empty = 2-norm/infnorm criterion)
    bool perRowCriteria = false;
    DeviceBuf dAlternateRowMaxMismatch, dAlternateRowMaxMismatchSelector;                              // n each: the alternate threshold and its selector row (see residualNormsBatch)
    bool perRowAlternateCriteria = false;
    // Per-slot shunt admittance overrides (scenario_override_tables.cuh): the caller's, a CSR per slot installed per
    // solve ((slot + 1) starts, PACK shunt element, (dG, dB)), and the incremental shunt loop's moved sections per
    // (slot, controller), with the controllers acting on each PACK shunt element.
    SlotTable shuntAdmittanceOverrides{1, 2};                            // per slot: the caller's (PACK shunt element; dG, dB)
    int nShuntAdmittanceOverrideControllers = 0;
    std::vector<int> hShuntAdmittanceOverrideControllerCounts;                       // the loop sizes the controller tables were built for
    DeviceBuf dShuntAdmittanceOverrideElementControllerStart, dShuntAdmittanceOverrideElementController;
    DeviceBuf dShuntAdmittanceOverrideSectionsMoved, dShuntAdmittanceOverrideControllerDelta;
    int nBaseClosedZeroImpedanceBranches = 0;                                         // zero-impedance elements [0,nBaseClosedZeroImpedanceBranches) always enabled; [nBaseClosedZeroImpedanceBranches,nZeroImpedanceBranch) are
                                                             // base-open closable switches gated by the zeroImpedanceBranchCloseOverrides list
    // Per (slot, zero-impedance element), dense: the switches the scenario OPENS and the base-open ones it CLOSES (1 = listed),
    // and OLF's spanning-tree flag where it differs from extraction (-1 = no override, else 0/1). The fill kernel read
    // them as (slot, element) lists, scanned PER THREAD - on network.xiidm's full non-busbar scope fillZeroImpBatch was the
    // top kernel, 32 s of 126 s (45 ms a launch).
    SlotRows<char> zeroImpedanceBranchOpenMask{0};
    SlotRows<char> zeroImpedanceBranchCloseMask{0};
    SlotRows<char> spanningTreeOverrideMask{-1};
    // Per-slot closed-branch tap overrides (scenario_override_tables.cuh): the caller's tap/PST actions, a CSR per slot
    // installed per solve ((slot + 1) starts, closed-branch element, full closedBranchParameters row), and the tap
    // loops' moved taps per (slot, controller), with the controllers acting on each closed-branch element.
    SlotTable closedBranchParameterOverrides{1, 8};                      // per slot: the caller's (closed-branch element; its row)
    int nClosedBranchParameterOverrideControllers = 0;
    std::vector<int> hClosedBranchParameterOverrideControllerCounts;                 // the loop sizes the controller tables were built for
    DeviceBuf dClosedBranchParameterOverrideElementControllerStart, dClosedBranchParameterOverrideElementController;
    DeviceBuf dClosedBranchParameterOverrideTapMoved, dClosedBranchParameterOverridePiModel;
    // ---- solve-only batch compaction (UBATCH, tol > 0): lazily allocated on first engage ----
    DeviceBuf dValsCompacted, dBCompacted, dDxCompacted, dCompactedScenarioMap;                      // nnz*cap / n*cap / n*cap / cap ints
    CudssCtx compactedSolver;                                            // dedicated handle/config/data for the compact width
    cudssMatrix_t compactedUniformBatchSystemMatrix = nullptr, compactedUniformBatchRightHandSideMatrix = nullptr, compactedUniformBatchSolutionMatrix = nullptr;
    cudssMatrix_t compactedUniformBatchResidualMatrix = nullptr, compactedUniformBatchDeltaMatrix = nullptr;   // adaptive refinement on the compacted solve
    cudssMatrix_t uniformBatchResidualMatrix = nullptr, uniformBatchDeltaMatrix = nullptr;     // ...and on the full-width UBATCH solve
    DeviceBuf dRCompacted, dDeltaCompacted;
    int compactedCapacity = 0;                                      // buffer capacity (nS/2)
    int compactWidth = 0;                                    // currently ANALYZED ubatch width (0 = none)
    // One ANALYZED cuDSS data object per compact width: a data object's
    // buffers belong to the batch size it was analyzed at, but a width seen before -- the next chunk walks the
    // same tiers, and so does the next run on this context -- can reuse its analysis instead of redoing it.
    std::map<int, cudssData_t> compactData;
    // setBatchX0Seeds: per-scenario starting values (flat s*n+row -> value) written over the tiled BASE state by
    // the next solve that is given no full per-scenario state; consumed (cleared) by that solve.
    std::vector<int> x0SeedIdx;
    std::vector<double> x0SeedVal;
    DeviceBuf dDisabled, dActive;                            // per-scenario masks
    ReactiveLimitsContext reactiveLimits;
    DeviceBuf dPilotSelfCarrierSkip;                                       // nReactiveRelocationGroups * nScenarios chars
    bool hasPilotSelf = false;
    SlotTable distrQInactiveMembers{1, 0};                                     // per slot: the DISTR_Q members that stop controlling there
    SlotRows<char> distrQSubgroup{0};                                     // tasks #71: per (slot, dq member) subgroup
                                                                          // labels (0: none) and the
    DeviceBuf dReactiveRelocationDistrQMemberStart;                                    // qr group -> first dq member map
    // forceTargetQInReactiveLimits (GpuBatchForcedTargetQ): generators whose IIDM target Q OLF clamps into their
    // curve limits at their CURRENT target P - on a bus whose Q balance carries it (qRow), or in a voltage-controlled
    // bus's reactive limits (reactiveLimitEntry). forcedTargetQIndices stride 7 {qRow, distributedSlackGeneratorIndex, ptsOff, npts, distributedSlackBusIndex, baseQRowMode, reactiveLimitEntry}; forcedTargetQParameters stride 5 {basePpu, rawTargetQPu, baseClampedPu, share,
    // baseRowTargetP}; dForcedTargetQApplied [nS * nForcedTargetQ] = the clamped value the scenario's Q target carries now; dForcedTargetQPreDistribution
    // [nS * nForcedTargetQ] = each generator's loss pre-distribution move (pu), absent when no scenario pre-distributes.
    int nForcedTargetQ = 0;
    int nForcedTargetQLimitEntries = 0;                                        // of which limit entries (voltage-controlled bus)
    int forcedTargetQExtrapolate = 0;
    SlotRows<double> forcedTargetQPreDistribution{0.0};
    DeviceBuf dForcedTargetQIndices, dForcedTargetQParameters, dForcedTargetQPoints, dForcedTargetQApplied;
    // DISTR_Q weight overrides (setBatchDistrQWeightOverrides): a gen outage at a remote-VC controller
    // bus renormalizes the group's qPct keys per scenario — per slot, (contribution; new weight), the last one wins.
    SlotTable distrQWeightOverrides{1, 1};
    // Per-scenario member reactive KEYS (setBatchDistrQKeyOverrides, {slot, member} -> OLF's qPct after the scenario's
    // generator losses): the DISTR_Q fill and the target kernel renormalize these over the members a PV -> PQ switch
    // leaves enabled, so an outage and an in-flight switch to PQ in one group compose. The per-contribution weight
    // overrides stay for what keys cannot express (a re-extracted pack whose free role moved).
    SlotTable distrQKeyOverrides{1, 1};                                        // per slot: (member; key), the first one wins
    // DISTR_RHO coefficient overrides (setBatchDistrRhoCoefficientOverrides): a member outage renormalizes the
    // group's 1/count coefficients per scenario — per slot, (contribution; new coefficient), the last one wins.
    SlotTable distrRhoCoefficientOverrides{1, 1};
    // Per-scenario ACTIVE equation counts (setBatchActiveRowCounts): OLF's L2 stopping threshold is
    // sqrt(eps^2 * nActive) and the CPU REMOVES equations for outaged elements carrying variables
    // (rho/a1 identities) and islanded buses — the padded device n over-counts by those rows, a ~%
    // threshold skew that lands stops up to the criterion apart. Empty = use n for every scenario.
    std::vector<int> hNActive;
    DeviceBuf dRowSlope;                                   // per row: SLOPE-SVC slope (BUS_TARGET_V_WITH_SLOPE V rows), 0 elsewhere
    int nDistrQDynamicKeys = 0;                                        // dynamic reactive keys (plan-B3): an RL switch to PQ zeroes the
    DeviceBuf dDistrQRowMember;                                  // switched member's key and renormalizes the DISTR_Q
    DeviceBuf dDistrQInjectionMember;                                  // weights over the survivors (updateReactiveKeys) — the
    DeviceBuf dDistrQMemberKey;                                // kernel recomputes each contribution's weight from the
    DeviceBuf dDistrQMemberReactiveLimitEntry;                            // RAW keys + the per-(entry, scenario) limitType,
    DeviceBuf dDistrQMemberGroupStart;                         // zero-then-sum in member order (bit-exact vs OLF)
    DeviceBuf dDistrQMemberGroupCount;
    DeviceBuf dDistrQMemberRow;                                // per member: its own row (vCol) + whether it OWNS a
    DeviceBuf dDistrQMemberHasRow;                             // packed DISTR_Q row (non-free with existing equation)
    DeviceBuf dDistrQMemberTargetQ;                            // per member: its UNFROZEN targetQ (generationTargetQ -
                                                           // loadTargetQ), the value AcTargetVector reads for every
                                                           // member the RL loop has not frozen
    DeviceBuf dLinRp, dLinRm, dLinSc;                      // linear-solve guard scratch, reused across passes
    DeviceBuf dDistrQMemberTargetPeerStart;                             // per member (CSR): the members whose targetQ OLF sums
    DeviceBuf dDistrQMemberTargetPeer;                            // into its DISTR_Q target — its OWN control's elements,
                                                           // just itself for a control MERGED into the group
    int nDistrQMembers = 0;
    int hasDistrQRelocationGroups = 0;                                 // plan-B4: any member linked to a pilot-Q relocation group
    DeviceBuf dDistrQMemberRelocationGroup;                            // per member: the pilot's qr candidate group id, -1 = none
    DeviceBuf dReactiveRelocationPilotTargetQ;                                   // per (scenario, qr group): the pilot's Q target, captured
                                                           // at solve init from the BASE carrier's row and handed to
                                                           // the CURRENT carrier's row on every re-pick
    DeviceBuf dReactiveRelocationIsGeneratorVoltageControl;                                    // per qr group: 1 = remote gen-VC group (vs TVC — only
                                                           // those get the no-enabled-controller pilot-vEq flip)
    DeviceBuf dReactiveRelocationPilotTargetVDeactivated;                                   // per (scenario, qr group): 1 = the pilot's V equation was
    DeviceBuf dReactiveRelocationSavedPilotTargetV;                                  // flipped OFF by updatePilotTargetVEquationBatch (+ the saved V target)
    DeviceBuf dReactiveRelocationPilotReactiveLimitEntry;                             // per qr group: the pilot's own LOCAL reactive-limit entry
                                                           // (its row + the group's V target), -1 = none
    int nRemoteMemberReactiveTerms = 0;                                          // remote-member OWN-q contributions (plan-B3): a twin
    DeviceBuf dRemoteMemberReactiveTermRow;                                     // of the DISTR_Q assembly accumulating each member's own
    DeviceBuf dRemoteMemberReactiveTermElement;                                    // bus q into dQbuf at its own row (that slot is free —
    DeviceBuf dRemoteMemberReactiveTermSide;                                    // the main qBuf fill suppresses member rows)
    DeviceBuf dRemoteMemberReactiveTermKind;
    DeviceBuf dRemoteMemberReactiveTermWeight;
    // dQbuf holds the q of every row (dQbufClosed's closed branches plus the other families, refilled at every check);
    // dOuterLoopCheckActive marks the scenarios an outer loop checks this pass, dOuterLoopUnstable those it changed.
    DeviceBuf dQbuf, dOuterLoopCheckActive, dOuterLoopUnstable;               // dQbuf nS*n, dOuterLoopCheckActive/dOuterLoopUnstable nS
    // nS*n: the closed branches' part of dQbuf, written by the Newton fills (each zeroes its active scenarios' rows and
    // adds the p and q it puts in F, on the frozen rows) and copied into dQbuf by every check's refill
    DeviceBuf dQbufClosed;
    // AcloadFlowEngine's realistic-voltage check, run on the INTERMEDIATE states of the driver's
    // sweep: per row, 1 = a non-fictitious bus's BUS_V at or above minNominalVoltageRealisticVoltageCheck.
    DeviceBuf dRealisticRow, dRealisticGate, dRealisticBad;
    int realisticCheck = 0;
    double realisticMinV = 0.0, realisticMaxV = 0.0;
    // AcloadFlowEngine DEFERS the check until the last outer loop that can FIX an unrealistic state
    // has run (voltageRemoteControlRobustMode; ReactiveLimits and the discrete TransformerVoltageControl
    // are the two loops that declare canFixUnrealisticState). -1 = no such loop, check every solve.
    // 0 = ReactiveLimits is the last one, 1 = the discrete TransformerVoltageControl.
    int realisticFixingLoop = GpuRealisticVoltageFixingLoop::NONE;
    long long solveCount = 0;                                // solveBatchContext calls on this context: the DS
                                                             // read-backs above describe the LAST solve only
    int newtonLoopIterations = 0;                            // the batched Newton iterations of the last solveBatchContext call
    AreaInterchangeControlContext areaInterchangeControl;
    // AutomationSystem (setBatchAutomationSystem, GpuOuterLoopKind::AUTOMATION_SYSTEM): OLF's AutomationSystemOuterLoop, its
    // last loop. Per overload management system: its monitored closed-branch element and side (GpuTwoSides) and the slice
    // of its trippings (dOverloadManagementSystemTrippingStart, nSystems + 1). Per tripping: the disabled-mask slot of the
    // closed branch it operates (also kept host-side for the read-back), whether it opens it, its current threshold (pu).
    int nOverloadManagementSystems = 0, nOverloadManagementSystemTrippings = 0;
    DeviceBuf dOverloadManagementSystemMonitoredElement, dOverloadManagementSystemMonitoredSide, dOverloadManagementSystemTrippingStart;
    DeviceBuf dOverloadManagementSystemOperatedMaskSlot, dOverloadManagementSystemOpenAction, dOverloadManagementSystemThreshold;
    std::vector<int> hOverloadManagementSystemOperatedMaskSlot;
    DeviceBuf dAutomationSystemSwitched;                                            // per (scenario, tripping): a check's decisions
    std::vector<int> outerLoopFirstSolveIterations, outerLoopIterations, outerLoopSolvePasses, outerLoopChecks;              // OLF_GPU_PHASE_TIME: per-scenario Newton work by
                                                             // solve pass
    // MonitoringVoltage (OLF order 6): the standby-SVC automaton, as a DRIVER PHASE. Per controller:
    // the monitored V row, its band and the two setpoints; plus a CSR list of (row, mode, target) the
    // activation writes. dMonitoringVoltageActivatedTargetV holds the chosen target per (controller, scenario), NaN until
    // it fires — OLF never re-checks a controller it has already switched.
    int nMonitoringVoltageControllers = 0;
    DeviceBuf dMonitoringVoltageBusVRow, dMonitoringVoltageBand, dMonitoringVoltageDeltaStart, dMonitoringVoltageDeltaRow, dMonitoringVoltageDeltaMode, dMonitoringVoltageDeltaTarget, dMonitoringVoltageActivatedTargetV;
    IncrementalTransformerVoltageControlContext incrementalTransformerVoltageControl;
    IncrementalShuntVoltageControlContext incrementalShuntVoltageControl;
    // FLOW-OBSERVED incremental loops (GpuOuterLoopKind::INCREMENTAL_PHASE_CONTROL /
    // INCREMENTAL_TRANSFORMER_REACTIVE_POWER_CONTROL): a rho (RPC) or a1 (phase) tap driving a CONTROLLED
    // branch's flow; kind = the GpuControlledFlow: REACTIVE_POWER (rho, closest-r1), ACTIVE_POWER (phase
    // CONTROLLER, closest-a1), CURRENT (phase LIMITER, first-above-a1, one-sided).
    int nIncrementalFlowControlControllerBranches = 0, nIncrementalPhaseControlControllerBranches = 0, nIncrementalTransformerReactivePowerControlControllerBranches = 0;
    DeviceBuf dIncrementalFlowControlControlledFlow, dIncrementalFlowControlTargetRow, dIncrementalFlowControlTapColumn, dIncrementalFlowControlControlledSide, dIncrementalFlowControlTargetValue, dIncrementalFlowControlHalfTargetDeadband, dIncrementalFlowControlElement, dIncrementalFlowControlControlledClosedBranch,
              dIncrementalFlowControlTapStart, dIncrementalFlowControlTapRows, dIncrementalFlowControlBaseTap, dIncrementalFlowControlMaxTapShift;
    DeviceBuf dIncrementalFlowControlCurrentTap, dIncrementalFlowControlAppliedTap, dIncrementalFlowControlCurrentDirection, dIncrementalFlowControlDirectionChangeCount, dIncrementalFlowControlAllowedDirection, dIncrementalFlowControlSensitivity, dIncrementalFlowControlDiff, dIncrementalFlowControlOutOfDeadband;
    std::vector<int> hIncrementalFlowControlControlledFlow, hIncrementalFlowControlTargetRow, hIncrementalFlowControlElement, hIncrementalFlowControlTapStart, hIncrementalFlowControlBaseTap;
    std::vector<double> hIncrementalFlowControlTapRows;
    // CONTINUOUS phase control (OLF PhaseControlOuterLoop, CONTINUOUS_WITH_DISCRETISATION), sharing the
    // phase slot: kind = the GpuPhaseControlMode: CONTROLLER (a1 solved free under row mode BRANCH_TARGET_P for the first
    // solve, then switched off and rounded to the closest tap at the loop's first check), LIMITER (from the second
    // check on, one tap per check against the sign of dI/da1 while I exceeds the limit — local only).
    int nPhaseControlControllerBranches = 0;
    DeviceBuf dPhaseControlMode, dPhaseControlA1Row, dPhaseControlControlledSide, dPhaseControlCurrentLimit, dPhaseControlElement, dPhaseControlTapStart, dPhaseControlTapRows, dPhaseControlBaseTap;
    DeviceBuf dPhaseControlTap, dPhaseControlAppliedTap, dPhaseControlCheckCount;                // per (scenario, controller) tap; per scenario check count
    std::vector<int> hPhaseControlElement, hPhaseControlTapStart, hPhaseControlBaseTap;
    std::vector<double> hPhaseControlTapRows;
    TransformerVoltageControlContext transformerVoltageControl;
    SecondaryVoltageControlContext secondaryVoltageControl;

    // GeneratorVoltageControlManager (AFTER_GENERATOR): generator VC controller buses under the max
    // transformer-controlled nominal voltage are FROZEN at their current Q while the transformers act
    // (freezeGenerationTargetQAndDisableGeneratorVoltageControl(q.eval()) — OLF's generation target is
    // the NET injection, so the row target is eval − loadQ), re-enabled at COMPLETE; plus
    // fixTransformerVoltageControls' PV-bus lists per member (the not-controlled side's component).
    int nGeneratorVoltageControlFreezeCandidates = 0;
    DeviceBuf dGeneratorVoltageControlFreezeVRow, dGeneratorVoltageControlFreezeLoadTargetQ;                       // per candidate
    DeviceBuf dGeneratorVoltageControlFrozen, dGeneratorVoltageControlFreezeSavedMode, dGeneratorVoltageControlFreezeSavedTarget;   // per (scenario, candidate)
    DeviceBuf dGeneratorVoltageControlFreezeReactiveLimitEntry;                                 // per candidate: RL entry of a REMOTE one, else -1
    ShuntVoltageControlContext shuntVoltageControl;

    SlotTable frozenRows{1, 0};                            // per slot: the rows an islanding outage froze to a flat identity state
    SlotRows<double> reactiveLimitSeedTargetQ{std::numeric_limits<double>::quiet_NaN()};   // per (slot, entry): the frozen target Q a continued
                                                                         // solve restores (setBatchReactiveLimitSeedTargetQ), NaN = the limit

    SlotTable distrQMemberTargetQOverrides{1, 1};                                // per slot: (member; its own targetQ), the first one wins
    SlotTable reactiveRelocationPilotTargetQDeltas{1, 1};   // per slot: (relocation group; the move of its pilot's captured Q target)
    // Pilot-row flip entries (pilotRowWithoutEnabledControllerBatch), per slot: (pilot row, first and count of its surviving
    // controllers' reactive-limit entries in pilotFlipSurvivors; the pilot's Q target), and per entry whether the row is
    // flipped and the voltage target it had.
    SlotTable pilotFlips{3, 1, {{sizeof(char), 0}, {sizeof(double), 0}}};
    SlotTable pilotFlipSurvivors{1, 0};
    // Re-merged controls whose main is one of its controllers (updateReMergedVoltageTargetRowBatch), per slot: (the main's
    // reactive-limit entry, first and count of the cluster's other controllers in reMergedVoltageTargetControllers, in OLF's
    // merged order; the control's voltage target), and per entry which of those controllers' rows holds the target (-1: the
    // main's own).
    SlotTable reMergedVoltageTargets{3, 1, {{sizeof(int), 0xFF}}};
    SlotTable reMergedVoltageTargetControllers{1, 0};
    // HVDC AC-emulation outer loops (FreezingHvdcACEmulation + AcHvdcAcEmulationLimits) as DRIVER
    // PHASES rather than host orchestrators. OLF runs both BETWEEN DistributedSlack and
    // ReactiveLimits; run from the host after the whole solve they land after RL instead, which is
    // what put the device's last DS check on a state before the PV -> PQ switches.
    int nHvdcAcEmulation = 0;                                             // AC-emulation HVDCs in the loop tables
    int hvdcAcEmulationLimitsActive = 0;                                  // AcHvdcAcEmulationLimits configured
    DeviceBuf dHvdcElement, dHvdcBus1PRow, dHvdcBus2PRow;               // per hvdc: the rows the P TARGETS are written to
    DeviceBuf dHvdcBus1PhiColumn, dHvdcBus2PhiColumn;                        // per hvdc: the angle COLUMNS the emulation reads (#68)
    DeviceBuf dHvdcAcEmulationParameters;                                         // per hvdc x9: p0 k pMax12 pMax21 loss1 loss2 r baseD1 baseD2
    SlotRows<char> hvdcContingencyOutage{0};                              // per (slot, hvdc): outaged BY the contingency
    SlotRows<int> hvdcAcEmulationStatus{0};                               // per (slot, hvdc): LINEAR/SAT_1_2/SAT_2_1/FROZEN
    SlotRows<double> hvdcAppliedDelta1{0.0}, hvdcAppliedDelta2{0.0};     // the converter-row delta currently APPLIED
    CudssCtx solver;
    bool analyzed = false;
    int mode = 0;                                            // 3 = UBATCH, 6 = per-scenario loop, robust (pivoted) config,
                                                             // 7 = our batched LU
    std::unique_ptr<BatchedLu> batchedLu;                    // BATCHED_LU: the factors, one lane per slot
    // BATCHED_LU with couplers: the residual and correction of the internal refinement steps inside a refinement's own
    // correction solve (batched_lu_solve.cu solveWithInternalRefinement)
    DeviceBuf dInternalRefinementResidual, dInternalRefinementCorrection;
    // BATCHED_LU: the exact substitution of the one-variable equations after each solve (singleton_substitution.cuh)
    DeviceBuf dSingletonHazardSlot, dSingletonColumnStart, dSingletonColumnRow, dSingletonList, dSingletonPendingList,
            dSingletonFixed, dSingletonTouched, dSingletonOwner, dSingletonProposal, dSingletonProposalValue, dSingletonCounters;
    CudssCtx luComparisonSolver;                             // OLF_GPU_LU_COMPARE: cuDSS's uniform batch on the same systems
    DeviceBuf dLuComparisonDx, dLuComparisonScore, dLuComparisonValues;   // the values' scenario-major copy, when interleaved
    bool luComparisonAnalyzed = false;
    DeviceBuf dR, dDelta, dIterativeRefinementScore, dIterativeRefinementScenarioMask;                 // adaptive refinement: residual, correction, lane scores/mask
    // UBATCH single matrices (that mode only): a SINGLE per-block CSR (n, nnz, shared rowPtr/colIdx)
    // whose value/RHS pointers are the batched buffers (dVals = nS*nnz, dB/dDx = nS*n); cuDSS
    // analyzes ONE block (cheap) and factorizes/solves all nS via CUDSS_CONFIG_UBATCH_SIZE.
    cudssMatrix_t uniformBatchSystemMatrix = nullptr, uniformBatchRightHandSideMatrix = nullptr, uniformBatchSolutionMatrix = nullptr;
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
        if (uniformBatchResidualMatrix) cudssMatrixDestroy(uniformBatchResidualMatrix);
        if (uniformBatchDeltaMatrix) cudssMatrixDestroy(uniformBatchDeltaMatrix);
        if (compactedUniformBatchResidualMatrix) cudssMatrixDestroy(compactedUniformBatchResidualMatrix);
        if (compactedUniformBatchDeltaMatrix) cudssMatrixDestroy(compactedUniformBatchDeltaMatrix);
        if (uniformBatchSystemMatrix) cudssMatrixDestroy(uniformBatchSystemMatrix);
        if (uniformBatchRightHandSideMatrix) cudssMatrixDestroy(uniformBatchRightHandSideMatrix);
        if (uniformBatchSolutionMatrix) cudssMatrixDestroy(uniformBatchSolutionMatrix);
        for (auto& e : compactData) {
            if (e.second && compactedSolver.handle) cudssDataDestroy(compactedSolver.handle, e.second);
        }
        compactData.clear();
        if (compactedUniformBatchSystemMatrix) cudssMatrixDestroy(compactedUniformBatchSystemMatrix);
        if (compactedUniformBatchRightHandSideMatrix) cudssMatrixDestroy(compactedUniformBatchRightHandSideMatrix);
        if (compactedUniformBatchSolutionMatrix) cudssMatrixDestroy(compactedUniformBatchSolutionMatrix);
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
