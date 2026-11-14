// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The part of a batched solver context that a voltage control shared by several controllers or remote from its bus reads and writes: the
// DISTR_Q members and their keys, the reactive relocation of a group's balance between its candidate rows, the
// remote members' reactive terms, the pilot that is its own controller, the pilot rows flipped when a group loses
// its last controller, and the controls re-merged by a contingency.
#ifndef OLF_GPU_SOLVER_CONTEXT_REMOTE_VOLTAGE_CONTROL_CONTEXT_CUH
#define OLF_GPU_SOLVER_CONTEXT_REMOTE_VOLTAGE_CONTROL_CONTEXT_CUH

#include <cublas_v2.h>

#include <vector>

#include "../../common/device_buf.cuh"
#include "../slot_rows.cuh"
#include "../slot_table.cuh"

namespace olfgpu {

struct RemoteVoltageControlContext {
    DeviceBuf dPilotSelfCarrierSkip;                                       // nReactiveRelocationGroups * nScenarios chars
    bool hasPilotSelf = false;
    SlotTable distrQInactiveMembers{1, 0};                                     // per slot: the DISTR_Q members that stop controlling there
    SlotRows<char> distrQSubgroup{0};                                     // tasks #71: per (slot, dq member) subgroup
                                                                          // labels (0: none) and the
    DeviceBuf dReactiveRelocationDistrQMemberStart;                                    // qr group -> first dq member map
    // DISTR_Q weight overrides (setBatchDistrQWeightOverrides): a gen outage at a remote-VC controller
    // bus renormalizes the group's qPct keys per scenario — per slot, (contribution; new weight), the last one wins.
    SlotTable distrQWeightOverrides{1, 1};
    // Per-scenario member reactive KEYS (setBatchDistrQKeyOverrides, {slot, member} -> OLF's qPct after the scenario's
    // generator losses): the DISTR_Q fill and the target kernel renormalize these over the members a PV -> PQ switch
    // leaves enabled, so an outage and an in-flight switch to PQ in one group compose. The per-contribution weight
    // overrides stay for what keys cannot express (a re-extracted pack whose free role moved).
    SlotTable distrQKeyOverrides{1, 1};                                        // per slot: (member; key), the first one wins
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
};

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_CONTEXT_REMOTE_VOLTAGE_CONTROL_CONTEXT_CUH
