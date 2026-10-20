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
    SharedStructurePtr st;                                   // built once per extraction, see SharedStructure
    int n = 0;
    int nnz = 0;
    int nScenarios = 0;
    int nElements = 0;
    int closedBranchOffset = 0, openBranchOffset = 0, shuntOffset = 0, hvdcAcEmulationOffset = 0;   // element-mask offsets per family
    int nDistrQ = 0, nDistrRho = 0, nDistrShuntB = 0, nZeroImpedanceBranch = 0;                  // copies of the structure counts
    DeviceBuf dX, dTarget, dF, dB, dDx, dVals, dMode;        // per-scenario (n or nnz strided)
    // per scenario: the state at its last CONVERGED inner solve (its starting state before any): what OLF's network
    // holds when a run fails (NewtonRaphson updates the buses only on convergence), and so where OLF starts the
    // strategies of a contingency that ran out of iterations (PreviousValueVoltageInitializer)
    DeviceBuf dXLastConv;
    DeviceBuf dTargetBase;                                   // the base targets (n), before tiling/overrides
    DeviceBuf dHazSnap, dHazBSnap, dHazChoice;               // zero-diagonal repair scratch (lazy)
    int hazCap = 0;
    bool clampStep = false;                                  // MAX_VOLTAGE_CHANGE per-scenario step clamp (set per solve)
    double maxDv = 0.0, maxDphi = 0.0;
    DeviceBuf dVarType, dStepFactor;                         // n ints / nS factors (lazy, first clamped solve)
    DeviceBuf dXprevLs;                                      // n*nS pre-step iterates (lazy, LINE_SEARCH solves)
    int lsMaxIter = 10;                                      // LINE_SEARCH backtracks (setBatchLineSearch)
    double lsStepFold = 4.0 / 3.0;
    DeviceBuf dNorms;                                        // nS*4 device-side convergence norms (see residualNormsBatch)
    DeviceBuf dThrIdent, dThrPower;                          // n each: per-row PER_EQUATION_TYPE thresholds (empty = 2-norm/infnorm criterion)
    bool perRowCriteria = false;
    DeviceBuf dThrAlt, dThrSel;                              // n each: the alternate threshold and its selector row (see residualNormsBatch)
    bool perRowAlt = false;
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
    int nR1Ov = 0;                                           // per-scenario closed-branch tap overrides (tap/PST actions)
    DeviceBuf dR1OvSlotElem, dR1OvVal;                       // nR1Ov*2 ints / nR1Ov*8 doubles (full closedBranchParameters row per entry)
    // ---- solve-only batch compaction (mode 3, tol > 0): lazily allocated on first engage ----
    DeviceBuf dValsC, dBC, dDxC, dMapC;                      // nnz*cap / n*cap / n*cap / cap ints
    CudssCtx dsC;                                            // dedicated handle/config/data for the compact width
    cudssMatrix_t matUbC = nullptr, matUbCB = nullptr, matUbCX = nullptr;
    cudssMatrix_t matUbCR = nullptr, matUbCDelta = nullptr;   // adaptive refinement on the compacted solve
    cudssMatrix_t matUbR = nullptr, matUbDelta = nullptr;     // ...and on the full-width UBATCH solve
    DeviceBuf dRC, dDeltaC;
    int compactCap = 0;                                      // buffer capacity (nS/2)
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
    // BUS_V row + reactive limits + load Q. dQbuf holds Σ branch q at every row (a fill into a zeroed
    // buffer); dOuterLoopCheckActive marks which scenarios to RL-check this pass; dOuterLoopUnstable flags scenarios that switched.
    int nReactiveLimitEntries = 0;
    DeviceBuf dReactiveLimitVoltageRow, dReactiveLimitMinQ, dReactiveLimitMaxQ, dReactiveLimitLoadQ, dReactiveLimitTargetV;   // per controller (nReactiveLimitEntries)
    DeviceBuf dReactiveLimitReactivePowerRow;                                       // per controller: the row its reactive BALANCE and its
                                                             // pin live on — its own V row, or, when the coupler-tree
                                                             // relocation moved its power equations onto the parent
                                                             // coupler's dummy columns, that coupler's DUMMY_Q row
    // PILOT-SELF rotation: per qr group, the RL entry of a pilot that is one of its OWN controllers (-1 when
    // the group is not of that shape), and the per-(scenario, group) gate the fills read. Both are DERIVED,
    // so no new host table: a group is pilot-self exactly when its fallback row (candidate 0 = the pilot's
    // own row) also appears among its member rows, and the pilot's entry is the RL entry sitting on that row.
    DeviceBuf dPsPilotEntry;                                 // nQr ints
    DeviceBuf dPsSkip;                                       // nQr * nScenarios chars
    bool hasPilotSelf = false;
    DeviceBuf dReactiveLimitLoadQPerScenario;                                       // per (scenario, controller) — shed load Q
    int nDistrQInactiveMembers = 0;                                            // {slot, distrQMember} pairs: a controller
    DeviceBuf dDistrQInactiveMembers;                                          // that stops controlling in ONE scenario
    bool hasDistrQSubgroups = false;                                       // tasks #71: per (scenario, dq member) subgroup
    DeviceBuf dDistrQSubgroup;                                            // labels (char, dense nS x nDistrQMembers) and the
    DeviceBuf dQrMemberStart;                                    // qr group -> first dq member map
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
    // generator losses): the DISTR_Q fill and the target kernel renormalize these over the members a pin leaves
    // enabled, so an outage and an in-flight pin in one group compose. The per-contribution weight overrides stay
    // for what keys cannot express (a re-extracted pack whose free role moved).
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
    int allPvPinnable = 0;                                 // every non-monitored RL entry has finite limits — the
                                                           // ONLY networks where OLF's global "strongest stays PV"
                                                           // rule (remainingPvBusCount 0) is reachable
    DeviceBuf dRowSlope;                                   // per row: SLOPE-SVC slope (mode-2 V rows), 0 elsewhere
    int nReactiveLimitRemoteEntries = 0;                                     // remote-member RL entries present (plan-B1/B3)
    int nReactiveLimitPilotSelfEntries = 0;                                  // reactiveLimitVoltageControlRole == 2 entries (a pilot that is its own controller)
    DeviceBuf dPilotSelfMixedPvPqRejected;                                 // nS chars: pilotSelfMixedPvPqGate flags
    DeviceBuf dReactiveLimitVoltageControlRole;                                   // per-entry remote kind (1 = remote group member)
    DeviceBuf dReactiveLimitBaseQLimitType;                            // OLF's base pin kind per entry (0..4); see hasReactiveLimitBaseQLimitType
    int hasReactiveLimitBaseQLimitType = 0;
    DeviceBuf dReactiveLimitComparedVoltageRow;                                  // per-entry compare row (PILOT v col for remote members)
    DeviceBuf dReactiveLimitSvcMinBPerScenario, dReactiveLimitSvcMaxBPerScenario;                // per (scenario, entry) coefs when a lost SVC takes its share
    DeviceBuf dReactiveLimitSvcMinB, dReactiveLimitSvcMaxB, dReactiveLimitBaseVSquared;           // per-entry SVC quadratic limit coefs + packed base v^2
                                                           // (limit(v) = static + b*(v^2 - v2Base)); zeroed default
    int nDistrQDynamicKeys = 0;                                        // dynamic reactive keys (plan-B3): an RL pin zeroes the
    DeviceBuf dDistrQRowMember;                                  // pinned member's key and renormalizes the DISTR_Q
    DeviceBuf dDistrQInjectionMember;                                  // weights over the survivors (updateReactiveKeys) — the
    DeviceBuf dDistrQMemberKey;                                // kernel recomputes each contribution's weight from the
    DeviceBuf dDistrQMemberReactiveLimitEntry;                            // RAW keys + the per-(entry, scenario) limitType,
    DeviceBuf dDistrQMemberGroupStart;                         // zero-then-sum in member order (bit-exact vs OLF)
    DeviceBuf dDistrQMemberGroupCount;
    DeviceBuf dDistrQMemberRow;                                // per member: its own row (vCol) + whether it OWNS a
    DeviceBuf dDistrQMemberHasRow;                             // packed DISTR_Q row (non-free with existing equation)
    DeviceBuf dDistrQMemberTargetQ;                            // per member: its UNPINNED targetQ (generationTargetQ -
                                                           // loadTargetQ), the value AcTargetVector reads for every
                                                           // member the RL loop has not frozen
    DeviceBuf dLinRp, dLinRm, dLinSc;                      // linear-solve guard scratch, reused across passes
    DeviceBuf dDistrQMemberTargetPeerStart;                             // per member (CSR): the members whose targetQ OLF sums
    DeviceBuf dDistrQMemberTargetPeer;                            // into its DISTR_Q target — its OWN control's elements,
                                                           // just itself for a control MERGED into the group
    int nDistrQMembers = 0;
    int hasDistrQRelocationGroups = 0;                                 // plan-B4: any member linked to a pilot-Q relocation group
    DeviceBuf dDistrQMemberRelocationGroup;                            // per member: the pilot's qr candidate group id, -1 = none
    DeviceBuf dQrPilotQ;                                   // per (scenario, qr group): the pilot's Q target, captured
                                                           // at solve init from the BASE carrier's row and handed to
                                                           // the CURRENT carrier's row on every re-pick
    DeviceBuf dQrGenVc;                                    // per qr group: 1 = remote gen-VC group (vs TVC — only
                                                           // those get the all-pinned pilot-vEq flip)
    DeviceBuf dQrVeqOff;                                   // per (scenario, qr group): 1 = the pilot's V equation was
    DeviceBuf dQrSavedVT;                                  // flipped OFF by updatePilotVeqBatch (+ the saved V target)
    DeviceBuf dReactiveRelocationPilotReactiveLimitEntry;                             // per qr group: the pilot's own LOCAL reactive-limit entry
                                                           // (its row + the group's V target), -1 = none
    int nRemoteMemberReactiveTerms = 0;                                          // remote-member OWN-q contributions (plan-B3): a twin
    DeviceBuf dRemoteMemberReactiveTermRow;                                     // of the DISTR_Q assembly accumulating each member's own
    DeviceBuf dRemoteMemberReactiveTermElement;                                    // bus q into dQbuf at its own row (that slot is free —
    DeviceBuf dRemoteMemberReactiveTermSide;                                    // the main qBuf fill suppresses member rows)
    DeviceBuf dRemoteMemberReactiveTermKind;
    DeviceBuf dRemoteMemberReactiveTermWeight;
    DeviceBuf dReactiveLimitSavedTarget;                              // per (controller, scenario): the row target saved at a
                                                           // PV->PQ pin, restored on switch-back for mode-2 rows
    DeviceBuf dQbuf, dOuterLoopCheckActive, dOuterLoopUnstable;               // dQbuf nS*n, dOuterLoopCheckActive/dOuterLoopUnstable nS
    DeviceBuf dReactiveLimitQLimitType, dReactiveLimitPvPqSwitchCount;               // per (controller, scenario): 0 PV / 1 minQ / 2 maxQ; PV→PQ count
    // OLF's GLOBAL forced-PV election ("all PV buses should switch — the strongest stays PV"): what each
    // entry did THIS pass (1 = pinned PV→PQ, 2 = switched back, 3 = limit re-freeze), the row mode a pin
    // replaced (to undo the elected one), and the static election keys (controlled bus nominal V, the
    // bus's P row for its LIVE target P, the static target P when it has no row, the id rank).
    DeviceBuf dReactiveLimitCheckDecision, dReactiveLimitSavedRowMode;
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
    // activation writes. dMonActivated holds the chosen target per (controller, scenario), NaN until
    // it fires — OLF never re-checks a controller it has already switched.
    // DRIVER PAUSE/RESUME. Some of OLF's outer loops cannot run on the device without RE-DERIVING
    // what OLF already computes in Java (SecondaryVoltageControl's sensitivity observables; the
    // discrete tap/section tables of the transformer and shunt loops). Rather than port that maths —
    // a wrong target converges cleanly and nothing flags it — the driver PAUSES at their slot in
    // OLF's order, hands control back, and RESUMES where it stopped. The per-solve device state
    // (RL pins, monitoring activations, DS absorbed totals, HVDC statuses, x/targets/modes) simply
    // stays in the context across the pause; only the host-side driver locals need saving.
    int hostLoopMask = 0;                                    // bitmask of enabled host-owned loops
                                                             // one hostLoopMaskBit per GpuOuterLoopKind::HOST_* kind
    int driverResume = 0;                                    // this call continues a paused driver
    int driverPaused = 0;                                    // the last call stopped at a GpuOuterLoopKind::HOST_*
    int pausedLoopKind = GpuOuterLoopKind::NOT_PAUSED;       // WHICH host loop the driver stopped at
    std::vector<int> pausedStatus;                           // per-scenario solve status at the pause: the resumed
                                                             // call's checks (outerLoopCheckActive = converged) need it back
    int startLoopKind = GpuOuterLoopKind::NONE;              // a RESTARTED sequence resumes its sweep AT this
                                                             // loop (OLF re-checks the loop that went UNSTABLE
                                                             // after the re-solve, before restarting the sweep)
    int pausedLoopIdx = 0;
    int pausedSeqChanged = 0;
    std::vector<int> pausedProdIters;
    std::vector<int> outerLoopFirstSolveIterations, outerLoopIterations, outerLoopSolvePasses;              // OLF_GPU_PHASE_TIME: per-scenario Newton work by
                                                             // solve pass (kept across a host-loop pause)
    std::vector<char> pausedLive, pausedActive;
    std::vector<char> hostChanged;                           // per scenario: the host loop was UNSTABLE
    int nmon = 0;
    DeviceBuf dMonVRow, dMonBand, dMonDeltaPtr, dMonDeltaRow, dMonDeltaMode, dMonDeltaTarget, dMonActivated;
    // INCREMENTAL TRANSFORMER VOLTAGE CONTROL — in-kernel outer loop (GpuOuterLoopKind::INCREMENTAL_TRANSFORMER_VOLTAGE_CONTROL).
    // Per-controller static data uploaded once; per-(scenario, controller) mutable state allocated
    // at first use. Tap tables are CSR-packed: tapData[tapPtr[c]..tapPtr[c+1]) holds nTaps rows of
    // 8 doubles [y, g1, g2, b1, b2, ksi, r1, a1] for controller c.
    int nTvc = 0;                                            // number of incremental TVC controllers
    DeviceBuf dTvcRhoRow, dTvcVRow;                          // [nTvc] int: rho identity row, controlled bus V row
    DeviceBuf dTvcTargetV, dTvcHalfDb;                       // [nTvc] double: target voltage, half deadband
    DeviceBuf dTvcElemIndex;                                 // [nTvc] int: element index for disabled-mask check
    DeviceBuf dTvcTapPtr;                                    // [nTvc+1] int: CSR pointer into flat tap table
    DeviceBuf dTvcTapData;                                   // [totalTaps * 8] double: flat tap rows
    DeviceBuf dTvcBaseTap;                                   // [nTvc] int: base (extraction-time) tap index
    // Per-(scenario, controller) mutable state:
    DeviceBuf dTvcCurTap;                                    // [nS * nTvc] int: current tap position
    DeviceBuf dTvcRatchetDir, dTvcRatchetCount, dTvcRatchetAllowed; // [nS * nTvc] int: ratchet state
    // Per-controller group structure (CSR): controllers sharing a controlled bus are grouped for
    // the one-tap-at-a-time round-robin (OLF's adjustWithSeveralControllers).
    DeviceBuf dTvcGroupPtr, dTvcGroupCtrl;                   // CSR: groupPtr[nGroups+1], groupCtrl[nTvc]
    int nTvcGroups = 0;
    int tvcMaxTapShift = 1;                                  // OLF incrementalTransformerRatioTapControlOuterLoopMaxTapShift
    // Per-pass scratch and the previous tap (to apply only the pairs that moved this pass).
    DeviceBuf dTvcOutOfDb, dTvcDiffV, dTvcSensi, dTvcPrevTap;
    // Host mirrors of the static tables, for the tap-override rebuild and the rho-row lookup.
    std::vector<int> hTvcRhoRow, hTvcElemIndex, hTvcBaseTap, hTvcTapPtr;
    std::vector<double> hTvcTapData;
    // Host copy of the CALLER's tap overrides (contingency tap actions) as uploaded at solve entry:
    // the TVC loop's moves are merged IN FRONT of them (findTapOverride takes the first match).
    std::vector<int> hR1OvBaseSlotElem;
    std::vector<double> hR1OvBaseVal;
    // INCREMENTAL SHUNT VOLTAGE CONTROL — in-kernel outer loop (GpuOuterLoopKind::INCREMENTAL_SHUNT_VOLTAGE_CONTROL). Controllers are
    // grouped per controlled bus (OLF adjustB round-robin, |b|-descending); each controller shunt has
    // sub-controllers (OLF LfShunt.Controller) with their own section B/G tables and ratchet.
    int nShc = 0, nShcGroups = 0, nShcSubs = 0, shcMaxSectionShift = 3;
    DeviceBuf dShcBRow, dShcVRow, dShcTargetV, dShcHalfDb, dShcElemAbs, dShcShOvElem;   // per controller
    DeviceBuf dShcGroupPtr, dShcGroupCtrl;                   // CSR: group -> controllers (sorted)
    DeviceBuf dShcSubPtr;                                    // CSR: controller -> subs
    DeviceBuf dShcSecPtr, dShcSecB, dShcSecG, dShcSubMin, dShcSubBase;   // per sub: section tables
    DeviceBuf dShcPos, dShcRatDir, dShcRatCount, dShcRatAllowed, dShcShift;   // per (scenario, sub)
    DeviceBuf dShcSensi, dShcBaseB, dShcAppliedDb;           // per (scenario, controller)
    DeviceBuf dShcOutOfDb, dShcDiffV;                        // per (scenario, group)
    std::vector<int> hShcBRow, hShcShOvElem, hShcSubPtr, hShcSecPtr, hShcSubBase;
    std::vector<double> hShcSecB, hShcSecG;
    // FLOW-OBSERVED incremental loops (GpuOuterLoopKind::INCREMENTAL_PHASE_CONTROL /
    // INCREMENTAL_TRANSFORMER_REACTIVE_POWER_CONTROL): a rho (RPC) or a1 (phase) tap driving a CONTROLLED
    // branch's flow; kind = the GpuControlledFlow: REACTIVE_POWER (rho, closest-r1), ACTIVE_POWER (phase
    // CONTROLLER, closest-a1), CURRENT (phase LIMITER, first-above-a1, one-sided).
    int nFlc = 0, nFlcPhase = 0, nFlcRpc = 0;
    DeviceBuf dFlcKind, dFlcPinRow, dFlcPinCol, dFlcSide, dFlcTargetValue, dFlcHalfDb, dFlcElem, dFlcCtlPack,
              dFlcTapPtr, dFlcTapData, dFlcBaseTap, dFlcMaxShift;
    DeviceBuf dFlcCurTap, dFlcPrevTap, dFlcRatDir, dFlcRatCount, dFlcRatAllowed, dFlcSensi, dFlcDiff, dFlcOutOfDb;
    std::vector<int> hFlcKind, hFlcPinRow, hFlcElem, hFlcTapPtr, hFlcBaseTap;
    std::vector<double> hFlcTapData;
    // CONTINUOUS phase control (OLF PhaseControlOuterLoop, CONTINUOUS_WITH_DISCRETISATION), sharing the
    // phase slot: kind = the GpuPhaseControlMode: CONTROLLER (a1 solved free under row mode 3 for the first
    // solve, then switched off and rounded to the closest tap at the loop's first check), LIMITER (from the second
    // check on, one tap per check against the sign of dI/da1 while I exceeds the limit — local only).
    int nCph = 0;
    DeviceBuf dCphKind, dCphA1Row, dCphSide, dCphLimit, dCphElem, dCphTapPtr, dCphTapData, dCphBaseTap;
    DeviceBuf dCphTap, dCphPrevTap, dCphIter;                // per (scenario, controller) tap; per scenario check count
    std::vector<int> hCphElem, hCphTapPtr, hCphBaseTap;
    std::vector<double> hCphTapData;
    std::vector<int> hShuntAdmittanceOverrideBaseScenarioElement;                      // the caller's shunt overrides at solve entry
    std::vector<double> hShuntAdmittanceOverrideBaseDelta;
    // DISCRETE (non-incremental) TRANSFORMER VOLTAGE CONTROL — GpuOuterLoopKind::TRANSFORMER_VOLTAGE_CONTROL (OLF order 8): OLF's
    // TransformerVoltageControlOuterLoop (AFTER_GENERATOR_VOLTAGE_CONTROL: INITIAL/CONTROL/COMPLETE) or
    // SimpleTransformerVoltageControlOuterLoop (WITH_GENERATOR_VOLTAGE_CONTROL). One group per controlled bus
    // (CSR over members); the group's ON/OFF activations are the toggle-diffed row deltas (CSR).
    int nTvd = 0, nTvdGroups = 0, nTvdDelta = 0, tvdMode = GpuTransformerVoltageControlMode::AFTER_GENERATOR_VOLTAGE_CONTROL, tvdEntryOn = 0;
    DeviceBuf dTvdGrpPtr, dTvdGrpVRow, dTvdGrpTargetV, dTvdGrpHalfDb, dTvdGrpUseInit;
    DeviceBuf dTvdDeltaPtr, dTvdDeltaRow, dTvdDeltaModeOn, dTvdDeltaModeOff, dTvdDeltaTargetOn, dTvdDeltaTargetOff;
    DeviceBuf dTvdMemRhoRow, dTvdMemElem, dTvdMemBaseTap, dTvdMemTapPtr, dTvdMemMin, dTvdMemMax, dTvdMemInit, dTvdTapData;
    DeviceBuf dTvdScnStep, dTvdGrpStep, dTvdGrpShMin, dTvdGrpShMax, dTvdGrpShInit;   // per scenario / per (scenario, group)
    DeviceBuf dTvdMemState, dTvdMemTap, dTvdMemRoundFrom;                          // per (scenario, member)
    DeviceBuf dTvdSaved, dTvdReject;                                               // per (scenario, delta row) / per scenario
    // SECONDARY VOLTAGE CONTROL — GpuOuterLoopKind::SECONDARY_VOLTAGE_CONTROL (OLF order 4). Zones of controller buses regulating a
    // pilot point: run entirely on the device, the small dense zone system included.
    int nSvc = 0, nSvcZones = 0;
    DeviceBuf dSvcCtldVRow, dSvcBusVRow, dSecondaryVoltageControlReactiveLimitEntry, dSvcBaseTargetV, dSvcMinQ, dSvcMaxQ, dSvcLoadQ;
    DeviceBuf dSvcQPtr, dSvcQKind, dSvcQElem, dSvcQBRow;
    DeviceBuf dSvcZonePtr, dSvcZoneCtl, dSvcZonePilotRow, dSvcZoneTargetV;
    DeviceBuf dSvcTargetV, dSvcK0, dSvcJk, dSvcJvpp, dSvcEnabled, dSvcNeed, dSvcReject, dSecondaryVoltageControlReactiveLimitTargetV;
    // Batched dense LU workspaces: the zone systems live in DEVICE memory, one N*N column-major matrix
    // and one RHS per scenario, plus the pointer/pivot/info arrays cuBLAS's batched API takes.
    DeviceBuf dSvcMat, dSvcRhs, dSvcMatPtr, dSvcRhsPtr, dSvcPiv, dSvcInfo, dSvcCtlZone;
    cublasHandle_t svcBlas = nullptr;
    std::vector<int> hSvcCtldVRow;
    std::vector<double> hSvcBaseTargetV;
    double svcMinPlausibleV = 0.8, svcMaxPlausibleV = 1.2;

    std::vector<int> hTvdMemElem, hTvdMemBaseTap, hTvdMemTapPtr;
    std::vector<double> hTvdTapData;
    // GeneratorVoltageControlManager (AFTER_GENERATOR): generator VC controller buses under the max
    // transformer-controlled nominal voltage are FROZEN at their current Q while the transformers act
    // (freezeGenerationTargetQAndDisableGeneratorVoltageControl(q.eval()) — OLF's generation target is
    // the NET injection, so the row target is eval − loadQ), re-enabled at COMPLETE; plus
    // fixTransformerVoltageControls' PV-bus lists per member (the not-controlled side's component).
    int nTvdFz = 0;
    DeviceBuf dTvdFzVRow, dTvdFzLoadQ;                       // per candidate
    DeviceBuf dTvdFzFrozen, dTvdFzSavedMode, dTvdFzSavedTarget;   // per (scenario, candidate)
    DeviceBuf dTvdFzRlEntry;                                 // per candidate: RL entry of a REMOTE one, else -1
    DeviceBuf dTvdFixPvRow;                                  // the fix universe: every gen-VC bus's V row
    DeviceBuf dTvdFixMinMask, dTvdFixMaxMask;                // per (scenario, member): fixWords ints each
    int tvdFixWords = 0;
    bool tvdFixTables = false;
    // DISCRETE SHUNT VOLTAGE CONTROL — GpuOuterLoopKind::SHUNT_VOLTAGE_CONTROL (OLF order 10): ShuntVoltageControlOuterLoop
    // (WITH_GENERATOR): control ON for the first solve, LfShuntImpl.dispatchB at the first check.
    int nSvd = 0, nSvdGroups = 0, nSvdDelta = 0, nSvdSubs = 0, svdEntryOn = 0;
    DeviceBuf dSvdGrpPtr, dSvdDeltaPtr, dSvdDeltaRow, dSvdDeltaModeOn, dSvdDeltaModeOff, dSvdDeltaTargetOn, dSvdDeltaTargetOff;
    DeviceBuf dSvdMemBRow, dSvdMemElem, dSvdMemSubPtr, dSvdSecPtr, dSvdSecB, dSvdSubMin, dSvdSubBase;
    DeviceBuf dSvdGrpStep, dSvdMemState, dSvdMemContB, dSvdSubPos, dSvdSaved;

    DeviceBuf dDistributedSlackGeneratorMove, dDistributedSlackGeneratorRemoved;
    DeviceBuf dFrozenRow;                                  // (scenario, row) pairs: rows an islanding
    int nFrozenRow = 0;                                    // outage froze to a flat identity state
    DeviceBuf dDistributedSlackLostGenerators;                                  // (scenario, DS generator slot) pairs: the
    int nDistributedSlackLostGenerators = 0;                                    // generators an outage removes, by identity
    DeviceBuf dReactiveLimitSeedTargetQ;                                 // per (scenario, entry): the pinned target a continued
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
    DeviceBuf dReactiveLimitPqPvSwitchForbidden, dReactiveLimitPqPvSwitchRejected;                       // (scenario, entry) unpin forbidden; per-scenario flag
    int nReactiveLimitPqPvSwitchForbidden = 0;
    DeviceBuf dReactiveLimitGeneratorsLost;                                    // (scenario, entry): the bus lost every generator this
    int nReactiveLimitGeneratorsLost = 0;                                      // scenario — not a controller, seeded FROZEN
    DeviceBuf dReactiveLimitLocalOverride, dReactiveLimitLocalTargetV;                        // re-homed PV buses: (scenario, entry) mask + own V target
    int nReactiveLimitLocalOverrides = 0;
    DeviceBuf dReactiveLimitComparedVoltageRowOverride;                                      // remote controllers re-homed onto a NEW pilot: (scenario,
    int nReactiveLimitComparedVoltageRowOverrides = 0;                                          // entry) -> the new pilot's V row, -1 = the base pilot's
    DeviceBuf dReactiveLimitIslanded;
    int nDistrQMemberTargetQOverrides = 0;                                       // per-scenario member targetQ overrides
    DeviceBuf dDistrQMemberTargetQOverrideKey;
    DeviceBuf dDistrQMemberTargetQOverrideValue;
    int hasSlackBusRowOverride = 0;                               // per-slot slack row (a relocated slack), device rows
    DeviceBuf dSlackBusRowOverride;                               // -1 = the base slack row
    int nQrPqDelta = 0;                                    // per-scenario moves of a pilot's captured Q target
    DeviceBuf dQrPqDeltaKey;                               // slot * nQr + group
    DeviceBuf dQrPqDeltaVal;
    int nPilotFlip = 0;                                    // pilot-row flip entries (all-survivors-pinned)
    DeviceBuf dPfSlot;
    DeviceBuf dPfRow;
    DeviceBuf dPfQTarget;
    DeviceBuf dPfOff;
    DeviceBuf dPfCount;
    DeviceBuf dPilotFlipSurvivorReactiveLimitEntries;
    DeviceBuf dPfFlipped;
    DeviceBuf dPfSavedVT;
    // HVDC AC-emulation outer loops (FreezingHvdcACEmulation + AcHvdcAcEmulationLimits) as DRIVER
    // PHASES rather than host orchestrators. OLF runs both BETWEEN DistributedSlack and
    // ReactiveLimits; run from the host after the whole solve they land after RL instead, which is
    // what put the device's last DS check on a pre-pin state.
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
    bool hasDistributedSlackGeneratorSeedAnchor = false;                            // without the actions (the continued pins' stamps)
                                                             // (OLF distributes into the slack gen and re-solves; no flow change,
                                                             // but the extra polish run must happen to match the CPU trajectory)
    CudssCtx ds;
    bool analyzed = false;
    int mode = 0;                                            // 3 = UBATCH, 6 = per-scenario loop, robust (pivoted) config
    DeviceBuf dR, dDelta, dIrScore, dIrLane;                 // adaptive refinement: residual, correction, lane scores/mask
    // UBATCH single matrices (mode 3 only): a SINGLE per-block CSR (n, nnz, shared rowPtr/colIdx)
    // whose value/RHS pointers are the batched buffers (dVals = nS*nnz, dB/dDx = nS*n); cuDSS
    // analyzes ONE block (cheap) and factorizes/solves all nS via CUDSS_CONFIG_UBATCH_SIZE.
    cudssMatrix_t matUb = nullptr, matUbB = nullptr, matUbX = nullptr;
    // Dedicated stream for the WHOLE batch solve (every kernel/memset/async copy + cuDSS via
    // cudssSetStream): keeps the per-iteration launch sequence off the legacy default stream,
    // whose implicit device-wide serialization stitched the timeline with sync points (the
    // ~25 ms/iter host/launch gaps of the 2026-07-15 nsys attribution — a loop kept on one
    // explicit stream shows none). Created BLOCKING (default flags), so any
    // op left on the legacy stream still orders correctly against it — safety over speed for
    // whatever this conversion misses.
    cudaStream_t stream = nullptr;
    ~BatchSolverContext() {
        if (svcBlas != nullptr) {
            cublasDestroy(svcBlas);
            svcBlas = nullptr;
        }
        if (stream) cudaStreamDestroy(stream);
        if (matUbR) cudssMatrixDestroy(matUbR);
        if (matUbDelta) cudssMatrixDestroy(matUbDelta);
        if (matUbCR) cudssMatrixDestroy(matUbCR);
        if (matUbCDelta) cudssMatrixDestroy(matUbCDelta);
        if (matUb) cudssMatrixDestroy(matUb);
        if (matUbB) cudssMatrixDestroy(matUbB);
        if (matUbX) cudssMatrixDestroy(matUbX);
        for (auto& e : compactData) {
            if (e.second && dsC.handle) cudssDataDestroy(dsC.handle, e.second);
        }
        compactData.clear();
        if (matUbC) cudssMatrixDestroy(matUbC);
        if (matUbCB) cudssMatrixDestroy(matUbCB);
        if (matUbCX) cudssMatrixDestroy(matUbCX);
    }
};

struct BatchResult {
    std::vector<double> x;                                   // nScenarios * n
    std::vector<int> iters;                                  // nScenarios
    std::vector<double> norm;                                // nScenarios
    std::vector<int> status;                                 // nScenarios: a GpuAcSolverStatus
};


// Build a batched context of nScenarios on a shared structure (the createBatchContext JNI entry).
BatchSolverContext* createBatchContext(const SharedStructurePtr& st, int nScenarios, int mode,
                                       int nReactiveLimitEntries, const int* reactiveLimitVoltageRow, const double* reactiveLimitMinQ,
                                       const double* reactiveLimitMaxQ, const double* reactiveLimitLoadQ, const double* reactiveLimitTargetV,
                                       const signed char* reactiveLimitFlags, const double* rowSlope,
                                       int nDistributedSlackBuses, const int* distributedSlackBusPhiRow, const double* distributedSlackBusFactor,
                                       int slackBusPhiRow, double slackBusTargetP,
                                       const double* distributedSlackBusBaseTargetP, const double* dsMaxDelta, const double* dsMinDelta,
                                       int distributedSlackFactorMode, const double* distributedSlackBusMarginUp, const double* distributedSlackBusMarginDown,
                                       double slackBusMarginUp, double slackBusMarginDown);

// Upload the call's step-scaling parameters and per-scenario shunt, coupler and tap overrides.
void uploadBatchOverrides(BatchSolverContext* ctx, int lsMode, double maxDv, double maxDphi,
                          const int* varType, int nShuntAdmittanceOverrides, const int* shuntAdmittanceOverrideScenarioElement, const double* shuntAdmittanceOverrideDelta,
                          int nBaseClosedZeroImpedanceBranches, int nZeroImpedanceBranchOpenOverrides, const int* zeroImpedanceBranchOpenScenarioElement,
                          int nZeroImpedanceBranchCloseOverrides, const int* zeroImpedanceBranchCloseScenarioElement,
                          int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal);

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_BATCH_CONTEXT_CUH
