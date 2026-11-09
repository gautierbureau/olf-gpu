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
    // Per slot, a generator loss at a DS participant bus: the bus (-1: none) and its factor, base target P, headroom and
    // footroom deltas.
    SlotRows<int> distributedSlackOverrideBus{-1};
    SlotRows<double> distributedSlackOverrideFactorDelta{0.0}, distributedSlackOverrideBaseTargetPDelta{0.0};
    SlotRows<double> distributedSlackOverrideMaxDelta{0.0}, distributedSlackOverrideMinDelta{0.0};
    // Per (slot, DS participant), a strategy pair's anchor shift: the parent's final target minus the base. Per slot, the
    // slack's own target move (a pair's parent share, or a load lost at the slack).
    SlotRows<double> distributedSlackBusAnchorShift{0.0}, slackBusAnchorShift{0.0};
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
    // Reactive limits (full-loadflow PV→PQ outer loop): per local-PV controller bus, its (rowMap-mapped)
    // BUS_V row + reactive limits + load Q. dQbuf holds Σ branch q at every row (dQbufClosed's closed branches plus the
    // other families, refilled at every check); dOuterLoopCheckActive marks which scenarios to RL-check this pass; dOuterLoopUnstable flags scenarios that switched.
    int nReactiveLimitEntries = 0;
    DeviceBuf dReactiveLimitVoltageRow, dReactiveLimitMinQ, dReactiveLimitMaxQ, dReactiveLimitLoadQ, dReactiveLimitTargetV;   // per controller (nReactiveLimitEntries)
    DeviceBuf dReactiveLimitReactivePowerRow;                                       // per controller: the row its reactive BALANCE and its
                                                             // PQ form live on — its own V row, or, when the coupler-tree
                                                             // relocation moved its power equations onto the parent
                                                             // coupler's dummy columns, that coupler's DUMMY_Q row
    // PILOT-SELF rotation: per qr group, the RL entry of a pilot that is one of its OWN controllers (-1 when
    // the group is not of that shape), and the per-(scenario, group) gate the fills read. Both are DERIVED,
    // so no new host table: a group is pilot-self exactly when its fallback row (candidate 0 = the pilot's
    // own row) also appears among its member rows, and the pilot's entry is the RL entry sitting on that row.
    DeviceBuf dPilotSelfReactiveLimitEntry;                                 // nReactiveRelocationGroups ints
    DeviceBuf dPilotSelfCarrierSkip;                                       // nReactiveRelocationGroups * nScenarios chars
    bool hasPilotSelf = false;
    DeviceBuf dReactiveLimitLoadQPerScenario;                                       // per (scenario, controller) — shed load Q
    SlotTable distrQInactiveMembers{1, 0};                                     // per slot: the DISTR_Q members that stop controlling there
    SlotRows<char> distrQSubgroup{0};                                     // tasks #71: per (slot, dq member) subgroup
                                                                          // labels (0: none) and the
    DeviceBuf dReactiveRelocationDistrQMemberStart;                                    // qr group -> first dq member map
    // Reactive capability CURVES at runtime P (setBatchReactiveCapabilityCurves): per participating curve gen —
    // reactiveCapabilityCurveIndices {reactiveLimitEntry, pRow, ptsOff, npts, distributedSlackGeneratorIndex}, reactiveCapabilityCurveParameters {basePpu, share, baseRowTargetP, baseEvalMinPu, baseEvalMaxPu},
    // reactiveCapabilityCurvePoints {p_MW, minQ_MW, maxQ_MW} sorted by p; dReactiveLimitMinQPerScenario/dReactiveLimitMaxQPerScenario = nS*nReactiveLimitEntries evaluated limits.
    int nReactiveCapabilityCurves = 0;
    int reactiveCapabilityCurveExtrapolate = 0;                                 // OLF extrapolateReactiveLimits for curve gens
    DeviceBuf dReactiveCapabilityCurveIndices, dReactiveCapabilityCurveParameters, dReactiveCapabilityCurvePoints, dReactiveLimitMinQPerScenario, dReactiveLimitMaxQPerScenario;
    DeviceBuf dReactiveCapabilityCurveHvdcStation;                                         // per curve entry {hvdc, side}, see CurveHvdcAcEmulationView
    bool reactiveCapabilityCurveHasHvdc = false;
    DeviceBuf dReactiveLimitEntryCurveStart, dReactiveLimitEntryCurves;                             // the curves of each reactive-limit entry, CSR
                                                           // (reactiveLimitEntryCurveStart[nReactiveLimitEntries + 1] into reactiveLimitEntryCurves[nReactiveCapabilityCurves], pack order kept)
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
    DeviceBuf dReactiveLimitFlags;                                // per controller: 1 = MONITORING_VOLTAGE SVC standing by
    DeviceBuf dRowSlope;                                   // per row: SLOPE-SVC slope (BUS_TARGET_V_WITH_SLOPE V rows), 0 elsewhere
    int nReactiveLimitRemoteEntries = 0;                                     // remote-member RL entries present (plan-B1/B3)
    int nReactiveLimitPilotSelfEntries = 0;                                  // reactiveLimitVoltageControlRole == 2 entries (a pilot that is its own controller)
    DeviceBuf dPilotSelfMixedPvPqRejected;                                 // nS chars: pilotSelfMixedPvPqGate flags
    DeviceBuf dReactiveLimitVoltageControlRole;                                   // per-entry remote kind (1 = remote group member)
    DeviceBuf dReactiveLimitBaseQLimitType;                            // OLF's base QLimitType per entry (0..4); see hasReactiveLimitBaseQLimitType
    int hasReactiveLimitBaseQLimitType = 0;
    DeviceBuf dReactiveLimitComparedVoltageRow;                                  // per-entry compare row (PILOT v col for remote members)
    DeviceBuf dReactiveLimitSvcMinBPerScenario, dReactiveLimitSvcMaxBPerScenario;                // per (scenario, entry) coefs when a lost SVC takes its share
    DeviceBuf dReactiveLimitSvcMinB, dReactiveLimitSvcMaxB, dReactiveLimitBaseVSquared;           // per-entry SVC quadratic limit coefs + packed base v^2
                                                           // (limit(v) = static + b*(v^2 - v2Base)); zeroed default
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
    DeviceBuf dReactiveLimitSavedTarget;                              // per (controller, scenario): the row target saved at a
                                                           // PV->PQ switch, restored on switch-back for BUS_TARGET_V_WITH_SLOPE rows
    DeviceBuf dQbuf, dOuterLoopCheckActive, dOuterLoopUnstable;               // dQbuf nS*n, dOuterLoopCheckActive/dOuterLoopUnstable nS
    // nS*n: the closed branches' part of dQbuf, written by the Newton fills (each zeroes its active scenarios' rows and
    // adds the p and q it puts in F, on the frozen rows) and copied into dQbuf by every check's refill
    DeviceBuf dQbufClosed;
    DeviceBuf dReactiveLimitQLimitType, dReactiveLimitPvPqSwitchCount;               // per (controller, scenario): 0 PV / 1 minQ / 2 maxQ; PV→PQ count
    // OLF's GLOBAL forced-PV election ("all PV buses should switch — the strongest stays PV"): what each
    // entry did THIS pass (1 = switched PV→PQ, 2 = switched back, 3 = limit re-freeze), the row mode a PV→PQ switch
    // replaced and the row it wrote (to undo the elected one), per scenario the buses the check left PV (OLF's
    // remainingPvBusCount), and the static election keys (controlled bus nominal V, the bus's P row for its LIVE
    // target P, the static target P when it has no row, the id rank).
    DeviceBuf dReactiveLimitCheckDecision, dReactiveLimitSavedRowMode, dReactiveLimitSwitchedRow, dReactiveLimitRemainingPvBusCount;
    DeviceBuf dReactiveLimitsDecisionLog, dReactiveLimitsDecisionLogCount;                           // OLF_GPU_SW_LOG: on-DEVICE decision log for one slot
    DeviceBuf dReactiveLimitNominalV, dReactiveLimitActivePowerRow, dReactiveLimitTargetP, dReactiveLimitIdRank;
    int hasReactiveLimitStrongestPvKeys = 0;
    // voltageRemoteControlRobustMode (OLF default TRUE): a REMOTE voltage controller whose OWN bus
    // voltage leaves the realistic band is reseeded at 1 pu, and — when its Q is still inside its
    // limits — frozen at its initial generation target Q with a MIN/MAX_REALISTIC_V limit type.
    DeviceBuf dReactiveLimitInitialTargetQ;                                 // per entry: Σ generator targetQ (pu)
    int reactiveLimitsRobustMode = 0;
    double reactiveLimitsRobustMinRealisticVoltage = 0.0, reactiveLimitsRobustMaxRealisticVoltage = 0.0;           // minRealisticV·1.02 and maxRealisticV/1.02
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
    // Distributed slack: the (mapped) slack BUS_PHI row + its scheduled targetP give the per-scenario
    // mismatch (read from dQbuf); dDistributedSlackBusPhiRow/dDistributedSlackBusFactor are the participating buses' P-target rows + normalized
    // participation factors. Reuses dOuterLoopCheckActive/dOuterLoopUnstable (same outer schedule as reactive limits).
    int nDistributedSlackBuses = 0, slackBusPhiRow = -1;
    double slackBusTargetP = 0.0;
    DeviceBuf dDistributedSlackBusPhiRow, dDistributedSlackBusFactor;                        // per participating bus (nDistributedSlackBuses)
    DeviceBuf dDistributedSlackBusBaseTargetP, dDistributedSlackBusMaxDelta, dDistributedSlackBusMinDelta;   // per participating bus (nDistributedSlackBuses): base P target + headroom/footroom
    double distributedSlackFactorSum = 1.0;                              // Σ emitted factors (< 1 ⇒ slack participates, keeps 1−Σ)
    int distributedSlackFactorMode = GpuDistributedSlackFactorMode::STATIC;    // REMAINING_MARGIN: direction-dependent
    DeviceBuf dDistributedSlackBusMarginUp, dDistributedSlackBusMarginDown;                  //   per-bus margins, normalized in-kernel
    double slackBusMarginUp = 0.0, slackBusMarginDown = 0.0; //   the slack's own margins (its retained share)
    DeviceBuf dSlackRetainedShare;
    DeviceBuf dDistributedSlackLastPassAccounting;                                   // per scenario {rem, fired} at the last DS gate (LOOP_TRACE reads it)
    DeviceBuf dSlackRetainedShareApplied;                              // per scenario: slackRetainedShare at the last check that FIRED,
                                                             // NaN while no distribution happened in this solve
    long long solveCount = 0;                                // solveBatchContext calls on this context: the DS
                                                             // read-backs above describe the LAST solve only
    int newtonLoopIterations = 0;                            // the batched Newton iterations of the last solveBatchContext call
    // Per-GENERATOR participants (setBatchDistributedSlackGenerators). OLF's water-fill participates and saturates
    // per generator — ParticipatingElement wraps an LfGenerator and step.run clamps each at its OWN
    // limits and REMOVES it — so a bus-level aggregate only reproduces OLF while nothing saturates.
    // The bus arrays keep owning the equation row and the anchor; these own the factors and limits.
    int nDistributedSlackGenerators = 0;
    DeviceBuf distributedSlackGeneratorBus, distributedSlackGeneratorFactor, distributedSlackGeneratorMaxDelta, distributedSlackGeneratorMinDelta, distributedSlackGeneratorMarginUp, distributedSlackGeneratorMarginDown;
    DeviceBuf dDistributedSlackBusGeneratorStart, dDistributedSlackBusGeneratorEnd;
    // Per (scenario, generator): its cumulative delta from the anchor, and whether the water-fill has
    // REMOVED it. Removal is not the same as "sitting at its bound": OLF removes an element only when
    // it strictly OVERSHOOTS, so one that lands exactly on its limit stays in the normalization for
    // another iteration — where it absorbs weight and moves nothing.
    // MonitoringVoltage (OLF order 6): the standby-SVC automaton, as a DRIVER PHASE. Per controller:
    // the monitored V row, its band and the two setpoints; plus a CSR list of (row, mode, target) the
    // activation writes. dMonitoringVoltageActivatedTargetV holds the chosen target per (controller, scenario), NaN until
    // it fires — OLF never re-checks a controller it has already switched.
    // The run's outer-loop list holds OLF's DistributedSlackOuterLoop / ReactiveLimitsOuterLoop
    // (setBatchOuterLoops). The driver installs those phases only when the run has the loop, whatever
    // the packed tables hold: the reactive-limit pack also carries the stand-by automaton SVC buses the
    // MonitoringVoltage phase activates, with no ReactiveLimitsOuterLoop in the run. Never set = on, the
    // tables alone decide.
    bool distributedSlackOuterLoop = true;
    bool reactiveLimitsOuterLoop = true;
    // AreaInterchangeControl (setBatchAreaInterchangeControl, GpuOuterLoopKind::AREA_INTERCHANGE_CONTROL): OLF's
    // AcAreaInterchangeControlOuterLoop. OLF drops the DistributedSlack loop from a run that has it and runs it after the
    // HVDC loops; it moves the distributed slack's own participants (dDistributedSlackGeneratorMove and the bus rows),
    // one area at a time. Per area: its interchange target and its share of the slack bus mismatch (pu) and the slice
    // of its boundaries (dAreaBoundaryStart, nAreas + 1). Per boundary: a GpuAreaBoundaryKind, the branch element of its
    // family, its slot in the disabled mask and its side; a tie line's half also carries {r, x, g, b, nominalV, SB} (SI),
    // OLF's SV.otherSideP inputs. Per distributed-slack bus and generator: its area, -1 for a bus without area. With no
    // area the phase is OLF's DistributedSlackOuterLoop at that place.
    bool areaInterchangeControlOuterLoop = false;
    int nAreas = 0, nAreaBoundaries = 0, slackBusArea = -1;
    double areaInterchangePMaxMismatch = 0.0;                                        // pu
    DeviceBuf dAreaInterchangeTarget, dAreaSlackDistributionFactor, dAreaBoundaryStart;
    DeviceBuf dAreaBoundaryKind, dAreaBoundaryElement, dAreaBoundaryMaskSlot, dAreaBoundarySide, dAreaBoundaryTieLineHalf;
    DeviceBuf dDistributedSlackBusArea, dDistributedSlackGeneratorArea;
    DeviceBuf dAreaInterchangeControlAreaScratch;                                   // per (scenario, area): the check's working values
    DeviceBuf dAreaInterchangeControlPreviousMove;                                  // per (scenario, generator): its move before a distribution
    DeviceBuf dAreaInterchangeControlParticipant;                                   // per (scenario, generator): still a participant
    DeviceBuf dAreaInterchangeControlDistributedActivePower;                        // per scenario: what OLF's context data accumulates (pu)
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
    int nMonitoringVoltageControllers = 0;
    DeviceBuf dMonitoringVoltageBusVRow, dMonitoringVoltageBand, dMonitoringVoltageDeltaStart, dMonitoringVoltageDeltaRow, dMonitoringVoltageDeltaMode, dMonitoringVoltageDeltaTarget, dMonitoringVoltageActivatedTargetV;
    // INCREMENTAL TRANSFORMER VOLTAGE CONTROL — in-kernel outer loop (GpuOuterLoopKind::INCREMENTAL_TRANSFORMER_VOLTAGE_CONTROL).
    // Per-controller static data uploaded once; per-(scenario, controller) mutable state allocated
    // at first use. Tap tables are CSR-packed: tapRows[tapStart[c]..tapStart[c+1]) holds nTaps rows of
    // 8 doubles [y, g1, g2, b1, b2, ksi, r1, a1] for controller c.
    int nIncrementalTransformerVoltageControlControllerBranches = 0;                                            // number of incremental TVC controllers
    DeviceBuf dIncrementalTransformerVoltageControlRhoRow, dIncrementalTransformerVoltageControlControlledBusVRow;                          // [nIncrementalTransformerVoltageControlControllerBranches] int: rho identity row, controlled bus V row
    DeviceBuf dIncrementalTransformerVoltageControlTargetV, dIncrementalTransformerVoltageControlHalfTargetDeadband;                       // [nIncrementalTransformerVoltageControlControllerBranches] double: target voltage, half deadband
    DeviceBuf dIncrementalTransformerVoltageControlElement;                                 // [nIncrementalTransformerVoltageControlControllerBranches] int: element index for disabled-mask check
    DeviceBuf dIncrementalTransformerVoltageControlTapStart;                                    // [nIncrementalTransformerVoltageControlControllerBranches+1] int: CSR pointer into flat tap table
    DeviceBuf dIncrementalTransformerVoltageControlTapRows;                                   // [totalTaps * 8] double: flat tap rows
    DeviceBuf dIncrementalTransformerVoltageControlBaseTap;                                   // [nIncrementalTransformerVoltageControlControllerBranches] int: base (extraction-time) tap index
    // Per-(scenario, controller) mutable state:
    DeviceBuf dIncrementalTransformerVoltageControlCurrentTap;                                    // [nS * nIncrementalTransformerVoltageControlControllerBranches] int: current tap position
    DeviceBuf dIncrementalTransformerVoltageControlCurrentDirection, dIncrementalTransformerVoltageControlDirectionChangeCount, dIncrementalTransformerVoltageControlAllowedDirection; // [nS * nIncrementalTransformerVoltageControlControllerBranches] int: ratchet state
    // Per-controller group structure (CSR): controllers sharing a controlled bus are grouped for
    // the one-tap-at-a-time round-robin (OLF's adjustWithSeveralControllers).
    DeviceBuf dIncrementalTransformerVoltageControlGroupControllerStart, dIncrementalTransformerVoltageControlGroupController;                   // CSR: groupControllerStart[nGroups+1], groupController[nIncrementalTransformerVoltageControlControllerBranches]
    int nIncrementalTransformerVoltageControlGroups = 0;
    int incrementalTransformerVoltageControlMaxTapShift = 1;                                  // OLF incrementalTransformerRatioTapControlOuterLoopMaxTapShift
    // Per-pass scratch and the previous tap (to apply only the pairs that moved this pass).
    DeviceBuf dIncrementalTransformerVoltageControlOutOfDeadband, dIncrementalTransformerVoltageControlDiffV, dIncrementalTransformerVoltageControlSensitivity, dIncrementalTransformerVoltageControlAppliedTap;
    // Host mirrors of the static tables, for the tap-override rebuild and the rho-row lookup.
    std::vector<int> hIncrementalTransformerVoltageControlRhoRow, hIncrementalTransformerVoltageControlElement, hIncrementalTransformerVoltageControlBaseTap, hIncrementalTransformerVoltageControlTapStart;
    std::vector<double> hIncrementalTransformerVoltageControlTapRows;
    // INCREMENTAL SHUNT VOLTAGE CONTROL — in-kernel outer loop (GpuOuterLoopKind::INCREMENTAL_SHUNT_VOLTAGE_CONTROL). Controllers are
    // grouped per controlled bus (OLF adjustB round-robin, |b|-descending); each controller shunt has
    // sub-controllers (OLF LfShunt.Controller) with their own section B/G tables and ratchet.
    int nIncrementalShuntVoltageControlControllerShunts = 0, nIncrementalShuntVoltageControlGroups = 0, nIncrementalShuntVoltageControlShuntCompensators = 0, incrementalShuntVoltageControlMaxSectionShift = 3;
    DeviceBuf dIncrementalShuntVoltageControlShuntBRow, dIncrementalShuntVoltageControlControlledBusVRow, dIncrementalShuntVoltageControlTargetV, dIncrementalShuntVoltageControlHalfTargetDeadband, dIncrementalShuntVoltageControlElement, dIncrementalShuntVoltageControlShuntAdmittanceOverrideElement;   // per controller
    DeviceBuf dIncrementalShuntVoltageControlGroupControllerStart, dIncrementalShuntVoltageControlGroupController;                   // CSR: group -> controllers (sorted)
    DeviceBuf dIncrementalShuntVoltageControlShuntCompensatorStart;                                    // CSR: controller -> subs
    DeviceBuf dIncrementalShuntVoltageControlSectionStart, dIncrementalShuntVoltageControlSectionB, dIncrementalShuntVoltageControlSectionG, dIncrementalShuntVoltageControlShuntCompensatorMinPosition, dIncrementalShuntVoltageControlShuntCompensatorBasePosition;   // per sub: section tables
    DeviceBuf dIncrementalShuntVoltageControlPosition, dIncrementalShuntVoltageControlCurrentDirection, dIncrementalShuntVoltageControlDirectionChangeCount, dIncrementalShuntVoltageControlAllowedDirection, dIncrementalShuntVoltageControlSectionShift;   // per (scenario, sub)
    DeviceBuf dIncrementalShuntVoltageControlSensitivity, dIncrementalShuntVoltageControlBaseB, dIncrementalShuntVoltageControlAppliedDeltaB;           // per (scenario, controller)
    DeviceBuf dIncrementalShuntVoltageControlOutOfDeadband, dIncrementalShuntVoltageControlDiffV;                        // per (scenario, group)
    std::vector<int> hIncrementalShuntVoltageControlShuntBRow, hIncrementalShuntVoltageControlShuntAdmittanceOverrideElement, hIncrementalShuntVoltageControlShuntCompensatorStart, hIncrementalShuntVoltageControlSectionStart, hIncrementalShuntVoltageControlShuntCompensatorBasePosition;
    std::vector<double> hIncrementalShuntVoltageControlSectionB, hIncrementalShuntVoltageControlSectionG;
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
    // DISCRETE (non-incremental) TRANSFORMER VOLTAGE CONTROL — GpuOuterLoopKind::TRANSFORMER_VOLTAGE_CONTROL (OLF order 8): OLF's
    // TransformerVoltageControlOuterLoop (AFTER_GENERATOR_VOLTAGE_CONTROL: INITIAL/CONTROL/COMPLETE) or
    // SimpleTransformerVoltageControlOuterLoop (WITH_GENERATOR_VOLTAGE_CONTROL). One group per controlled bus
    // (CSR over members); the group's ON/OFF activations are the toggle-diffed row deltas (CSR).
    int nTransformerVoltageControlMembers = 0, nTransformerVoltageControlGroups = 0, nTransformerVoltageControlDeltaRows = 0, transformerVoltageControlMode = GpuTransformerVoltageControlMode::AFTER_GENERATOR_VOLTAGE_CONTROL, transformerVoltageControlEntryOn = 0;
    DeviceBuf dTransformerVoltageControlGroupMemberStart, dTransformerVoltageControlGroupControlledBusVRow, dTransformerVoltageControlGroupTargetV, dTransformerVoltageControlGroupHalfTargetDeadband, dTransformerVoltageControlGroupUseInitialTapPosition;
    DeviceBuf dTransformerVoltageControlDeltaStart, dTransformerVoltageControlDeltaRow, dTransformerVoltageControlDeltaModeOn, dTransformerVoltageControlDeltaModeOff, dTransformerVoltageControlDeltaTargetOn, dTransformerVoltageControlDeltaTargetOff;
    DeviceBuf dTransformerVoltageControlMemberRhoRow, dTransformerVoltageControlMemberElement, dTransformerVoltageControlMemberBaseTap, dTransformerVoltageControlMemberTapStart, dTransformerVoltageControlMemberMinR1, dTransformerVoltageControlMemberMaxR1, dTransformerVoltageControlMemberInitialR1, dTransformerVoltageControlTapRows;
    DeviceBuf dTransformerVoltageControlStep, dTransformerVoltageControlGroupStep, dTransformerVoltageControlGroupSharedMinR1, dTransformerVoltageControlGroupSharedMaxR1, dTransformerVoltageControlGroupSharedInitialR1;   // per scenario / per (scenario, group)
    DeviceBuf dTransformerVoltageControlMemberState, dTransformerVoltageControlMemberTap, dTransformerVoltageControlMemberRoundingR1;                          // per (scenario, member)
    DeviceBuf dTransformerVoltageControlSavedDeltaTarget, dTransformerVoltageControlRejected;                                               // per (scenario, delta row) / per scenario
    // SECONDARY VOLTAGE CONTROL — GpuOuterLoopKind::SECONDARY_VOLTAGE_CONTROL (OLF order 4). Zones of controller buses regulating a
    // pilot point: run entirely on the device, the small dense zone system included.
    int nSecondaryVoltageControlControllerBuses = 0, nSecondaryVoltageControlZones = 0;
    DeviceBuf dSecondaryVoltageControlControlledBusVRow, dSecondaryVoltageControlControllerBusVRow, dSecondaryVoltageControlReactiveLimitEntry, dSecondaryVoltageControlBaseTargetV, dSecondaryVoltageControlMinQ, dSecondaryVoltageControlMaxQ, dSecondaryVoltageControlLoadTargetQ;
    DeviceBuf dSecondaryVoltageControlReactiveTermStart, dSecondaryVoltageControlReactiveTermKind, dSecondaryVoltageControlReactiveTermSide, dSecondaryVoltageControlReactiveTermElement, dSecondaryVoltageControlReactiveTermShuntBRow;
    DeviceBuf dSecondaryVoltageControlZoneStart, dSecondaryVoltageControlZoneControllerBus, dSecondaryVoltageControlZonePilotBusVRow, dSecondaryVoltageControlZoneTargetV;
    DeviceBuf dSecondaryVoltageControlTargetV, dSecondaryVoltageControlK0, dSecondaryVoltageControlJk, dSecondaryVoltageControlJvpp, dSecondaryVoltageControlControllerBusEnabled, dSecondaryVoltageControlNeedsAdjustment, dSecondaryVoltageControlRejected, dSecondaryVoltageControlReactiveLimitTargetV;
    // Batched dense LU workspaces: the zone systems live in DEVICE memory, one N*N column-major matrix
    // and one RHS per scenario, plus the pointer/pivot/info arrays cuBLAS's batched API takes.
    DeviceBuf dSecondaryVoltageControlZoneMatrix, dSecondaryVoltageControlZoneRightHandSide, dSecondaryVoltageControlZoneMatrixPointers, dSecondaryVoltageControlZoneRightHandSidePointers, dSecondaryVoltageControlZonePivots, dSecondaryVoltageControlZoneFactorizationInfo, dSecondaryVoltageControlControllerBusZone;
    cublasHandle_t secondaryVoltageControlBlasHandle = nullptr;
    std::vector<int> hSecondaryVoltageControlControlledBusVRow;
    std::vector<double> hSecondaryVoltageControlBaseTargetV;
    double secondaryVoltageControlMinPlausibleTargetV = 0.8, secondaryVoltageControlMaxPlausibleTargetV = 1.2;

    std::vector<int> hTransformerVoltageControlMemberElement, hTransformerVoltageControlMemberBaseTap, hTransformerVoltageControlMemberTapStart;
    std::vector<double> hTransformerVoltageControlTapRows;
    // GeneratorVoltageControlManager (AFTER_GENERATOR): generator VC controller buses under the max
    // transformer-controlled nominal voltage are FROZEN at their current Q while the transformers act
    // (freezeGenerationTargetQAndDisableGeneratorVoltageControl(q.eval()) — OLF's generation target is
    // the NET injection, so the row target is eval − loadQ), re-enabled at COMPLETE; plus
    // fixTransformerVoltageControls' PV-bus lists per member (the not-controlled side's component).
    int nGeneratorVoltageControlFreezeCandidates = 0;
    DeviceBuf dGeneratorVoltageControlFreezeVRow, dGeneratorVoltageControlFreezeLoadTargetQ;                       // per candidate
    DeviceBuf dGeneratorVoltageControlFrozen, dGeneratorVoltageControlFreezeSavedMode, dGeneratorVoltageControlFreezeSavedTarget;   // per (scenario, candidate)
    DeviceBuf dGeneratorVoltageControlFreezeReactiveLimitEntry;                                 // per candidate: RL entry of a REMOTE one, else -1
    DeviceBuf dTransformerVoltageControlFixPvRow;                                  // the fix universe: every gen-VC bus's V row
    SlotRows<int> transformerVoltageControlFixMinMask{0}, transformerVoltageControlFixMaxMask{0};   // per (slot, member): fixWords ints each
    int transformerVoltageControlFixWords = 0;
    // DISCRETE SHUNT VOLTAGE CONTROL — GpuOuterLoopKind::SHUNT_VOLTAGE_CONTROL (OLF order 10): ShuntVoltageControlOuterLoop
    // (WITH_GENERATOR): control ON for the first solve, LfShuntImpl.dispatchB at the first check.
    int nShuntVoltageControlMembers = 0, nShuntVoltageControlGroups = 0, nShuntVoltageControlDeltaRows = 0, nShuntVoltageControlShuntCompensators = 0, shuntVoltageControlEntryOn = 0;
    DeviceBuf dShuntVoltageControlGroupMemberStart, dShuntVoltageControlDeltaStart, dShuntVoltageControlDeltaRow, dShuntVoltageControlDeltaModeOn, dShuntVoltageControlDeltaModeOff, dShuntVoltageControlDeltaTargetOn, dShuntVoltageControlDeltaTargetOff;
    DeviceBuf dShuntVoltageControlMemberShuntBRow, dShuntVoltageControlMemberElement, dShuntVoltageControlMemberShuntCompensatorStart, dShuntVoltageControlSectionStart, dShuntVoltageControlSectionB, dShuntVoltageControlShuntCompensatorMinPosition, dShuntVoltageControlShuntCompensatorBasePosition;
    DeviceBuf dShuntVoltageControlGroupStep, dShuntVoltageControlMemberState, dShuntVoltageControlMemberContinuousB, dShuntVoltageControlShuntCompensatorPosition, dShuntVoltageControlSavedDeltaTarget;

    DeviceBuf dDistributedSlackGeneratorMove, dDistributedSlackGeneratorRemoved;
    DeviceBuf dDistributedSlackPreviousGeneratorMove, dDistributedSlackPreviousBusTarget;   // nS x generators / buses: before a distribution
    SlotTable frozenRows{1, 0};                            // per slot: the rows an islanding outage froze to a flat identity state
    // The reactive-limit channels of each slot's scenario (solveBatchContext installs them): (entry; dMinQ, dMaxQ, dLoadQ,
    // dBmin, dBmax) - every one on an entry adds, (curve entry; pStep, shareScale) and (entry; moved voltage target) - the
    // first one wins.
    SlotTable reactiveLimitDeltaOverrides{1, 5};
    SlotTable reactiveCapabilityCurveOverrides{1, 2};
    SlotTable reactiveLimitTargetVOverrides{1, 1};
    SlotTable distributedSlackLostGenerators{1, 0};                             // per slot: the DS generators an outage removes, by identity
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
    SlotTable distributedSlackRemovedGenerators{1, 0};                              // per slot: the DS participants a GeneratorAction took out
    DeviceBuf dDistributedSlackGeneratorMoveValid;                                // per scenario: the PER-GENERATOR water-fill ran, so
                                                           // dDistributedSlackGeneratorMove holds each generator's own movement. The
                                                           // anchored path distributes per BUS and leaves it zero,
                                                           // which a reader cannot tell from "nothing moved".

    // The WORKING factors, normalized IN PLACE each iteration exactly as OLF's
    // normalizeParticipationFactors mutates ParticipatingElement.factor: by iteration k a survivor's
    // factor has been divided k times. f/Sigma_survivors(raw) is the same number algebraically and a
    // different one in floating point, which is enough to flip a clamp decision at a knife-edge.
    DeviceBuf dDistributedSlackGeneratorWorkingFactor;
    // Per (scenario, participant) chars, 1 = this participant is in ANOTHER synchronous component in
    // this scenario. OLF distributes over getSynchronousNetworks().getFirst(), so an islanded
    // participant is simply not a participant: its share must not be computed, and — decisively — the
    // set it is renormalized against must shrink with it. Empty = every participant participates.
    SlotRows<char> distributedSlackIslanded{0};                                     // per (slot, DS participant): stranded in another component
    SlotRows<int> distributedSlackBusRowOverride{-1};                                // per (slot, DS participant): its P-balance row diversion,
                                                                                    // -1 = use distributedSlackBusPhiRow
    // Same idea for the reactive-limits pass: per (scenario, controller), 1 = its bus is in another
    // synchronous component in this scenario, so the loop must not look at it.
    SlotRows<char> reactiveLimitGeneratorsLost{0};                             // (slot, entry): the bus lost every generator this
                                                                               // scenario — not a controller, seeded FROZEN
    SlotRows<char> reactiveLimitLocalOverride{0};                                         // re-merged PV buses: (slot, entry) mask + own V target
    SlotRows<double> reactiveLimitLocalTargetV{0.0};
    SlotRows<int> reactiveLimitComparedVoltageRowOverride{-1};                               // remote controllers re-merged onto a NEW pilot: (slot,
                                                                                                // entry) -> the new pilot's V row, -1 = the base pilot's
    SlotRows<char> reactiveLimitIslanded{0};                                         // (slot, entry): its bus islanded this scenario
    SlotTable distrQMemberTargetQOverrides{1, 1};                                // per slot: (member; its own targetQ), the first one wins
    SlotRows<int> slackBusRowOverride{-1};                        // per-slot slack row (a relocated slack), device rows
                                                                  // -1 = the base slack row
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
    DeviceBuf dReactiveLimitHoldsReMergedVoltageTarget;    // per (scenario, entry): its sharing row holds its control's voltage target
    DeviceBuf dReactiveLimitReMergedVoltageRow;            // per (scenario, entry) of a main: the voltage row that target is held on, -1 = its own
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
    DeviceBuf dSlackDistributionFailure;                                   // nS chars: waterfill left an out-of-tolerance residual                                   // per scenario (nS): frozen slack-retained mismatch share
    // The slack bus's own participating generators' total rooms from their anchor targets (setBatchSlackBusRooms):
    // what a slack-only distribution can place before OLF's water-fill leaves a remaining mismatch. Unbounded until
    // installed, and for a load-based slack.
    double slackBusRoomUp = INFINITY, slackBusRoomDown = -INFINITY;
    bool slackOnlyDistribution = false;                                // DS enabled with the SLACK generator as the only participant:
    DeviceBuf dSlackOnlyAbsorbed;                              // nS doubles — mismatch absorbed into the slack's own target so far
    SlotRows<double> slackOnlyAnchor{0.0};                     // per slot — slack-only strategy pairs: the parent's total
                                                               // (setBatchSlackOnlyAnchor; zero anchor when unset)
    SlotRows<double> distributedSlackGeneratorAnchor{0.0};                          // per (slot, DS generator) — strategy pairs: each participating
                                                                                    // generator's target at the parent's end (setBatchDistributedSlackGeneratorAnchors)
    SlotRows<double> distributedSlackGeneratorInitialMove{0.0};                           // per (slot, DS generator) — each participating generator's move
                                                                                          // from its anchor at the solve's start (setBatchDistributedSlackGeneratorAnchors)
    SlotRows<double> distributedSlackGeneratorSeedAnchor{0.0};                      // per (slot, DS generator) — strategy pairs: the parent's own end,
                                                                                    // without the actions (the stamps of the continued PQ switches)
                                                             // (OLF distributes into the slack gen and re-solves; no flow change,
                                                             // but the extra polish run must happen to match the CPU trajectory)
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
        if (secondaryVoltageControlBlasHandle != nullptr) {
            cublasDestroy(secondaryVoltageControlBlasHandle);
            secondaryVoltageControlBlasHandle = nullptr;
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
    return ctx->nDistributedSlackBuses > 0 ? ctx->slackBusAnchorShift.deviceOrNull() : nullptr;
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
