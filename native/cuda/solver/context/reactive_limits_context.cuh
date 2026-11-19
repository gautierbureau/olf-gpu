// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The part of a batched solver context that ReactiveLimitsOuterLoop reads and writes: per local-PV controller (reactive
// limit entry) its rows, limits and capability curves, the per-scenario switch state of the loop, and the per-slot
// overrides a contingency or a strategy installs.
#ifndef OLF_GPU_SOLVER_CONTEXT_REACTIVE_LIMITS_CONTEXT_CUH
#define OLF_GPU_SOLVER_CONTEXT_REACTIVE_LIMITS_CONTEXT_CUH

#include "../../common/device_buf.cuh"
#include "../slot_rows.cuh"
#include "../slot_table.cuh"

namespace olfgpu {

struct ReactiveLimitsContext {
    // Per local-PV controller bus (reactive limit entry): its (rowMap-mapped) BUS_V row, reactive limits and load Q. The
    // check reads the q of every row from the context's dQbuf and the scenarios to check from dOuterLoopCheckActive.
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
    DeviceBuf dReactiveLimitLoadQPerScenario;                                       // per (scenario, controller) — shed load Q
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
    DeviceBuf dReactiveLimitFlags;                                // per controller: 1 = MONITORING_VOLTAGE SVC standing by
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
    DeviceBuf dReactiveLimitSavedTarget;                              // per (controller, scenario): the row target saved at a
                                                           // PV->PQ switch, restored on switch-back for BUS_TARGET_V_WITH_SLOPE rows
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
    bool reactiveLimitsOuterLoop = true;
    // The reactive-limit channels of each slot's scenario (solveBatchContext installs them): (entry; dMinQ, dMaxQ, dLoadQ,
    // dBmin, dBmax) - every one on an entry adds, (curve entry; pStep, shareScale) and (entry; moved voltage target) - the
    // first one wins.
    SlotTable reactiveLimitDeltaOverrides{1, 5};
    SlotTable reactiveCapabilityCurveOverrides{1, 2};
    SlotTable reactiveLimitTargetVOverrides{1, 1};
    // Same idea for the reactive-limits pass: per (scenario, controller), 1 = its bus is in another
    // synchronous component in this scenario, so the loop must not look at it.
    SlotRows<char> reactiveLimitGeneratorsLost{0};                             // (slot, entry): the bus lost every generator this
                                                                               // scenario — not a controller, seeded FROZEN
    SlotRows<char> reactiveLimitLocalOverride{0};                                         // re-merged PV buses: (slot, entry) mask + own V target
    SlotRows<double> reactiveLimitLocalTargetV{0.0};
    SlotRows<int> reactiveLimitComparedVoltageRowOverride{-1};                               // remote controllers re-merged onto a NEW pilot: (slot,
                                                                                                // entry) -> the new pilot's V row, -1 = the base pilot's
    SlotRows<char> reactiveLimitIslanded{0};                                         // (slot, entry): its bus islanded this scenario
    DeviceBuf dReactiveLimitHoldsReMergedVoltageTarget;    // per (scenario, entry): its sharing row holds its control's voltage target
    DeviceBuf dReactiveLimitReMergedVoltageRow;            // per (scenario, entry) of a main: the voltage row that target is held on, -1 = its own
};

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_CONTEXT_REACTIVE_LIMITS_CONTEXT_CUH
