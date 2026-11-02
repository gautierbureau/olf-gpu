// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// BatchSolverContext: one batched consumer of a shared structure (its per-scenario value and state
// arrays, outer-loop tables and cuDSS objects), and BatchResult, what a batched solve hands back.
#ifndef OLF_GPU_SOLVER_BATCH_CONTEXT_CUH
#define OLF_GPU_SOLVER_BATCH_CONTEXT_CUH

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudss.h>

#include <map>
#include <memory>
#include <vector>

#include "../common/device_buf.cuh"
#include "../common/loop_kinds.cuh"
#include "../common/codes/gpu_distributed_slack_factor_mode.cuh"
#include "../common/codes/gpu_ac_solver_status.cuh"
#include "../common/codes/gpu_outer_loop_kind.cuh"
#include "../common/codes/gpu_realistic_voltage_fixing_loop.cuh"
#include "../common/codes/gpu_transformer_voltage_control_mode.cuh"
#include "../lu/batched_lu.cuh"
#include "../structure/shared_structure.cuh"

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
    SharedStructurePtr structure;                                   // built once per extraction, see SharedStructure
    int n = 0;
    int nnz = 0;
    int nScenarios = 0;
    int nElements = 0;
    int closedBranchOffset = 0, openBranchOffset = 0, shuntOffset = 0, hvdcAcEmulationOffset = 0;   // element-mask offsets per family
    int nDistrQ = 0, nDistrRho = 0, nDistrShuntB = 0, nZeroImpedanceBranch = 0;                  // copies of the structure counts
    DeviceBuf dX, dTarget, dF, dB, dDx, dVals, dMode;        // per-scenario (n or nnz strided)
    // nS: 1 while a retired scenario's values and F hold the identity system writeIdentityForInactive wrote, so the next
    // identity write skips it. Cleared at every batched solve's start and for every scenario zeroActiveValues zeroes (a
    // check's refill writes no values: its fills get no Jacobian)
    DeviceBuf dIdentityLane;
    // per scenario: the state at its last CONVERGED inner solve (its starting state before any): what OLF's network
    // holds when a run fails (NewtonRaphson updates the buses only on convergence), and so where OLF starts the
    // strategies of a contingency that ran out of iterations (PreviousValueVoltageInitializer)
    DeviceBuf dXLastConv;
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
    int nShuntAdmittanceOverrides = 0;                                           // per-scenario shunt admittance overrides (set per solve)
    DeviceBuf dShuntAdmittanceOverrideScenarioElement, dShuntAdmittanceOverrideDelta;                        // nShuntAdmittanceOverrides*2 ints (slot, PACK elem) / nShuntAdmittanceOverrides*2 doubles (dG, dB)
    int nBaseClosedZeroImpedanceBranches = 0;                                         // zero-impedance elements [0,nBaseClosedZeroImpedanceBranches) always enabled; [nBaseClosedZeroImpedanceBranches,nZeroImpedanceBranch) are
                                                             // base-open closable switches gated by the zeroImpedanceBranchCloseOverrides list
    int nZeroImpedanceBranchCloseOverrides = 0;
    int nZeroImpedanceBranchOpenOverrides = 0;
    DeviceBuf dSpanningTreeOverrideScenarioElementValue;                          // per (scenario, coupler) OLF tree flag where it differs from extraction
    int nSpanningTreeOverrides = 0;
    DeviceBuf dZeroImpedanceBranchOpenScenarioElement;                                // nZeroImpedanceBranchOpenOverrides*2 ints (slot, zero-impedance elem) — scenarios that OPEN a switch
    DeviceBuf dZeroImpedanceBranchCloseScenarioElement;                                 // nZeroImpedanceBranchCloseOverrides*2 ints (slot, zero-impedance elem) — scenarios that CLOSE a switch
    // DENSE (scenario, zero-impedance element) forms of the three lists above and of the tree overrides, nScenarios*nZeroImpedanceBranch chars
    // each: the fill kernel read them as linear scans PER THREAD, O(nZeroImpedanceBranch * nS * (nZeroImpedanceBranchOpenOverrides + nZeroImpedanceBranchCloseOverrides + nSpanningTreeOverrides)) - on
    // network.xiidm's full non-busbar scope fillZeroImpBatch was the top kernel, 32 s of 126 s (45 ms a launch).
    DeviceBuf dZeroImpedanceBranchOpenMask, dZeroImpedanceBranchCloseMask, dSpanningTreeOverrideMask;            // off/on: 1 = listed; tree: -1 = no override, else 0/1
    bool hasZeroImpedanceBranchMasks = false;
    int nClosedBranchParameterOverrides = 0;                                           // per-scenario closed-branch tap overrides (tap/PST actions)
    DeviceBuf dClosedBranchParameterOverrideScenarioElement, dClosedBranchParameterOverrideValues;                       // nClosedBranchParameterOverrides*2 ints / nClosedBranchParameterOverrides*8 doubles (full closedBranchParameters row per entry)
    // ---- solve-only batch compaction (mode 3, tol > 0): lazily allocated on first engage ----
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
    int nDistrQInactiveMembers = 0;                                            // {slot, distrQMember} pairs: a controller
    DeviceBuf dDistrQInactiveMembers;                                          // that stops controlling in ONE scenario
    bool hasDistrQSubgroups = false;                                       // tasks #71: per (scenario, dq member) subgroup
    DeviceBuf dDistrQSubgroup;                                            // labels (char, dense nS x nDistrQMembers) and the
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
    bool hasForcedTargetQPreDistribution = false;
    DeviceBuf dForcedTargetQIndices, dForcedTargetQParameters, dForcedTargetQPoints, dForcedTargetQApplied, dForcedTargetQPreDistribution;
    // DISTR_Q weight overrides (setBatchDistrQWeightOverrides): a gen outage at a remote-VC controller
    // bus renormalizes the group's qPct keys per scenario — {s, contrib} pairs + the new weight.
    int nDistrQWeightOverrides = 0;
    DeviceBuf dDistrQWeightOverrideScenarioTerm, dDistrQWeightOverrideWeight;
    // Per-scenario member reactive KEYS (setBatchDistrQKeyOverrides, {slot, member} -> OLF's qPct after the scenario's
    // generator losses): the DISTR_Q fill and the target kernel renormalize these over the members a PV -> PQ switch
    // leaves enabled, so an outage and an in-flight switch to PQ in one group compose. The per-contribution weight
    // overrides stay for what keys cannot express (a re-extracted pack whose free role moved).
    int nDistrQKeyOverrides = 0;
    DeviceBuf dDistrQKeyOverrideScenarioMember, dDistrQKeyOverrideKey;
    // DISTR_RHO coefficient overrides (setBatchDistrRhoCoefficientOverrides): a member outage renormalizes the
    // group's 1/count coefficients per scenario — {s, contrib} pairs + the new coefficient.
    int nDistrRhoCoefficientOverrides = 0;
    DeviceBuf dDistrRhoOverrideScenarioTerm, dDistrRhoOverrideCoefficient;
    // Per-scenario ACTIVE equation counts (setBatchActiveRowCounts): OLF's L2 stopping threshold is
    // sqrt(eps^2 * nActive) and the CPU REMOVES equations for outaged elements carrying variables
    // (rho/a1 identities) and islanded buses — the padded device n over-counts by those rows, a ~%
    // threshold skew that lands stops up to the criterion apart. Empty = use n for every scenario.
    std::vector<int> hNActive;
    DeviceBuf dReactiveLimitFlags;                                // per controller: 1 = MONITORING_VOLTAGE SVC standing by
    DeviceBuf dRowSlope;                                   // per row: SLOPE-SVC slope (mode-2 V rows), 0 elsewhere
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
                                                           // PV->PQ switch, restored on switch-back for mode-2 rows
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
    std::vector<int> outerLoopFirstSolveIterations, outerLoopIterations, outerLoopSolvePasses;              // OLF_GPU_PHASE_TIME: per-scenario Newton work by
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
    // Host copy of the CALLER's tap overrides (contingency tap actions) as uploaded at solve entry:
    // the TVC loop's moves are merged IN FRONT of them (findTapOverride takes the first match).
    std::vector<int> hClosedBranchParameterOverrideBaseScenarioElement;
    std::vector<double> hClosedBranchParameterOverrideBaseValues;
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
    // phase slot: kind = the GpuPhaseControlMode: CONTROLLER (a1 solved free under row mode 3 for the first
    // solve, then switched off and rounded to the closest tap at the loop's first check), LIMITER (from the second
    // check on, one tap per check against the sign of dI/da1 while I exceeds the limit — local only).
    int nPhaseControlControllerBranches = 0;
    DeviceBuf dPhaseControlMode, dPhaseControlA1Row, dPhaseControlControlledSide, dPhaseControlCurrentLimit, dPhaseControlElement, dPhaseControlTapStart, dPhaseControlTapRows, dPhaseControlBaseTap;
    DeviceBuf dPhaseControlTap, dPhaseControlAppliedTap, dPhaseControlCheckCount;                // per (scenario, controller) tap; per scenario check count
    std::vector<int> hPhaseControlElement, hPhaseControlTapStart, hPhaseControlBaseTap;
    std::vector<double> hPhaseControlTapRows;
    std::vector<int> hShuntAdmittanceOverrideBaseScenarioElement;                      // the caller's shunt overrides at solve entry
    std::vector<double> hShuntAdmittanceOverrideBaseDelta;
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
    DeviceBuf dTransformerVoltageControlFixMinMask, dTransformerVoltageControlFixMaxMask;                // per (scenario, member): fixWords ints each
    int transformerVoltageControlFixWords = 0;
    bool hasTransformerVoltageControlFixMasks = false;
    // DISCRETE SHUNT VOLTAGE CONTROL — GpuOuterLoopKind::SHUNT_VOLTAGE_CONTROL (OLF order 10): ShuntVoltageControlOuterLoop
    // (WITH_GENERATOR): control ON for the first solve, LfShuntImpl.dispatchB at the first check.
    int nShuntVoltageControlMembers = 0, nShuntVoltageControlGroups = 0, nShuntVoltageControlDeltaRows = 0, nShuntVoltageControlShuntCompensators = 0, shuntVoltageControlEntryOn = 0;
    DeviceBuf dShuntVoltageControlGroupMemberStart, dShuntVoltageControlDeltaStart, dShuntVoltageControlDeltaRow, dShuntVoltageControlDeltaModeOn, dShuntVoltageControlDeltaModeOff, dShuntVoltageControlDeltaTargetOn, dShuntVoltageControlDeltaTargetOff;
    DeviceBuf dShuntVoltageControlMemberShuntBRow, dShuntVoltageControlMemberElement, dShuntVoltageControlMemberShuntCompensatorStart, dShuntVoltageControlSectionStart, dShuntVoltageControlSectionB, dShuntVoltageControlShuntCompensatorMinPosition, dShuntVoltageControlShuntCompensatorBasePosition;
    DeviceBuf dShuntVoltageControlGroupStep, dShuntVoltageControlMemberState, dShuntVoltageControlMemberContinuousB, dShuntVoltageControlShuntCompensatorPosition, dShuntVoltageControlSavedDeltaTarget;

    DeviceBuf dDistributedSlackGeneratorMove, dDistributedSlackGeneratorRemoved;
    DeviceBuf dFrozenRow;                                  // (scenario, row) pairs: rows an islanding
    int nFrozenRow = 0;                                    // outage froze to a flat identity state
    DeviceBuf dDistributedSlackLostGenerators;                                  // (scenario, DS generator slot) pairs: the
    int nDistributedSlackLostGenerators = 0;                                    // generators an outage removes, by identity
    DeviceBuf dReactiveLimitSeedTargetQ;                                 // per (scenario, entry): the frozen target Q a continued
    bool hasReactiveLimitSeedTargetQ = false;                            // solve restores (setBatchReactiveLimitSeedTargetQ), NaN = the limit
    DeviceBuf dDistributedSlackRemovedGenerators;                                   // (scenario, DS generator slot) pairs: participants
    int nDistributedSlackRemovedGenerators = 0;                                     // a GeneratorAction took out of the distribution
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
    int nDistributedSlackIslanded = 0;
    DeviceBuf dDistributedSlackIslanded;
    int nDistributedSlackBusRowOverrides = 0;                                        // per-scenario DS participant row diversions
    DeviceBuf dDistributedSlackBusRowOverride;                                      // nScenarios * nDistributedSlackBuses ints, -1 = use distributedSlackBusPhiRow
    // Same idea for the reactive-limits pass: per (scenario, controller), 1 = its bus is in another
    // synchronous component in this scenario, so the loop must not look at it.
    int nReactiveLimitIslanded = 0;
    DeviceBuf dReactiveLimitGeneratorsLost;                                    // (scenario, entry): the bus lost every generator this
    int nReactiveLimitGeneratorsLost = 0;                                      // scenario — not a controller, seeded FROZEN
    DeviceBuf dReactiveLimitLocalOverride, dReactiveLimitLocalTargetV;                        // re-merged PV buses: (scenario, entry) mask + own V target
    int nReactiveLimitLocalOverrides = 0;
    DeviceBuf dReactiveLimitComparedVoltageRowOverride;                                      // remote controllers re-merged onto a NEW pilot: (scenario,
    int nReactiveLimitComparedVoltageRowOverrides = 0;                                          // entry) -> the new pilot's V row, -1 = the base pilot's
    DeviceBuf dReactiveLimitIslanded;
    int nDistrQMemberTargetQOverrides = 0;                                       // per-scenario member targetQ overrides
    DeviceBuf dDistrQMemberTargetQOverrideKey;
    DeviceBuf dDistrQMemberTargetQOverrideValue;
    int hasSlackBusRowOverride = 0;                               // per-slot slack row (a relocated slack), device rows
    DeviceBuf dSlackBusRowOverride;                               // -1 = the base slack row
    int nReactiveRelocationPilotTargetQDeltas = 0;                                    // per-scenario moves of a pilot's captured Q target
    DeviceBuf dReactiveRelocationPilotTargetQDeltaKey;                               // slot * nReactiveRelocationGroups + group
    DeviceBuf dReactiveRelocationPilotTargetQDeltaValue;
    int nPilotFlip = 0;                                    // pilot-row flip entries (no enabled controller left)
    DeviceBuf dPfSlot;
    DeviceBuf dPfRow;
    DeviceBuf dPfQTarget;
    DeviceBuf dPfOff;
    DeviceBuf dPfCount;
    DeviceBuf dPilotFlipSurvivorReactiveLimitEntries;
    DeviceBuf dPfFlipped;
    DeviceBuf dPfSavedVT;
    int nReMergedVoltageTargets = 0;                       // re-merged controls whose main is one of its controllers
    DeviceBuf dReMergedVoltageTargetSlot;                  // per control: its scenario, the main's reactive-limit entry,
    DeviceBuf dReMergedVoltageTargetMainEntry;             // its voltage target, and the cluster's other controllers'
    DeviceBuf dReMergedVoltageTargetValue;                 // entries in OLF's merged order (start/count into the list)
    DeviceBuf dReMergedVoltageTargetControllerStart;
    DeviceBuf dReMergedVoltageTargetControllerCount;
    DeviceBuf dReMergedVoltageTargetControllerEntries;
    DeviceBuf dReMergedVoltageTargetHolder;                // per control: which of those controllers' rows holds the target, -1 = the main's own
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
    DeviceBuf dHvdcContingencyOutage;                                     // per (scenario, hvdc): outaged BY the contingency
    DeviceBuf dHvdcAcEmulationStatus;                                     // per (scenario, hvdc): LINEAR/SAT_1_2/SAT_2_1/FROZEN
    DeviceBuf dHvdcAppliedDelta1, dHvdcAppliedDelta2;                    // the converter-row delta currently APPLIED
    DeviceBuf dSlackDistributionFailure;                                   // nS chars: waterfill left an out-of-tolerance residual                                   // per scenario (nS): frozen slack-retained mismatch share
    // The slack bus's own participating generators' total rooms from their anchor targets (setBatchSlackBusRooms):
    // what a slack-only distribution can place before OLF's water-fill leaves a remaining mismatch. Unbounded until
    // installed, and for a load-based slack.
    double slackBusRoomUp = INFINITY, slackBusRoomDown = -INFINITY;
    bool slackOnlyDistribution = false;                                // DS enabled with the SLACK generator as the only participant:
    DeviceBuf dSlackOnlyAbsorbed;                              // nS doubles — mismatch absorbed into the slack's own target so far
    DeviceBuf dSlackOnlyAnchor;                                // nS doubles — slack-only strategy pairs: the parent's total
    bool hasSlackOnlyAnchor = false;                           // (setBatchSlackOnlyAnchor; zero anchor when unset)
    DeviceBuf dDistributedSlackGeneratorAnchor;                                     // nS * nDistributedSlackGenerators doubles — strategy pairs: each participating
    bool hasDistributedSlackGeneratorAnchor = false;                                // generator's target at the parent's end (setBatchDistributedSlackGeneratorAnchors)
    DeviceBuf dDistributedSlackGeneratorInitialMove;                                      // nS * nDistributedSlackGenerators doubles — each participating generator's move
    bool hasDistributedSlackGeneratorInitialMove = false;                                 // from its anchor at the solve's start (setBatchDistributedSlackGeneratorAnchors)
    DeviceBuf dDistributedSlackGeneratorSeedAnchor;                                 // nS * nDistributedSlackGenerators doubles — strategy pairs: the parent's own end,
    bool hasDistributedSlackGeneratorSeedAnchor = false;                            // without the actions (the stamps of the continued PQ switches)
                                                             // (OLF distributes into the slack gen and re-solves; no flow change,
                                                             // but the extra polish run must happen to match the CPU trajectory)
    CudssCtx solver;
    bool analyzed = false;
    int mode = 0;                                            // 3 = UBATCH, 6 = per-scenario loop, robust (pivoted) config,
                                                             // 7 = our batched LU
    std::unique_ptr<BatchedLu> batchedLu;                    // mode 7: the factors, one lane per slot
    // mode 7 with couplers: the residual and correction of the internal refinement steps inside a refinement's own
    // correction solve (batched_lu_solve.cu solveWithInternalRefinement)
    DeviceBuf dInternalRefinementResidual, dInternalRefinementCorrection;
    // mode 7: the exact substitution of the one-variable equations after each solve (singleton_substitution.cuh)
    DeviceBuf dSingletonHazardSlot, dSingletonShortRow, dSingletonFixed, dSingletonOwner, dSingletonProposal,
            dSingletonProposalValue, dSingletonFixedAny;
    CudssCtx luComparisonSolver;                             // OLF_GPU_LU_COMPARE: cuDSS's uniform batch on the same systems
    DeviceBuf dLuComparisonDx, dLuComparisonScore;
    bool luComparisonAnalyzed = false;
    DeviceBuf dR, dDelta, dIterativeRefinementScore, dIterativeRefinementScenarioMask;                 // adaptive refinement: residual, correction, lane scores/mask
    // UBATCH single matrices (mode 3 only): a SINGLE per-block CSR (n, nnz, shared rowPtr/colIdx)
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

struct BatchResult {
    std::vector<double> x;                                   // nScenarios * n
    std::vector<int> iters;                                  // nScenarios
    std::vector<double> norm;                                // nScenarios
    std::vector<int> status;                                 // nScenarios: a GpuAcSolverStatus
};


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
