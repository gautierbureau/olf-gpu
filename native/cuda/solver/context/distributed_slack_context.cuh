// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The part of a batched solver context that The slack distribution (DistributedSlackOuterLoop and the slack-only distribution): the slack bus, the participating
// buses and generators with their factors and margins, what a scenario moved or removed, and the per-slot overrides.
#ifndef OLF_GPU_SOLVER_CONTEXT_DISTRIBUTED_SLACK_CONTEXT_CUH
#define OLF_GPU_SOLVER_CONTEXT_DISTRIBUTED_SLACK_CONTEXT_CUH

#include <cublas_v2.h>

#include <vector>

#include "../../common/device_buf.cuh"
#include "../slot_rows.cuh"
#include "../slot_table.cuh"

namespace olfgpu {

struct DistributedSlackContext {
    // Per slot, a generator loss at a DS participant bus: the bus (-1: none) and its factor, base target P, headroom and
    // footroom deltas.
    SlotRows<int> distributedSlackOverrideBus{-1};
    SlotRows<double> distributedSlackOverrideFactorDelta{0.0}, distributedSlackOverrideBaseTargetPDelta{0.0};
    SlotRows<double> distributedSlackOverrideMaxDelta{0.0}, distributedSlackOverrideMinDelta{0.0};
    // Per (slot, DS participant), a strategy pair's anchor shift: the parent's final target minus the base. Per slot, the
    // slack's own target move (a pair's parent share, or a load lost at the slack).
    SlotRows<double> distributedSlackBusAnchorShift{0.0}, slackBusAnchorShift{0.0};
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
    // Per-GENERATOR participants (setBatchDistributedSlackGenerators). OLF's water-fill participates and saturates
    // per generator — ParticipatingElement wraps an LfGenerator and step.run clamps each at its OWN
    // limits and REMOVES it — so a bus-level aggregate only reproduces OLF while nothing saturates.
    // The bus arrays keep owning the equation row and the anchor; these own the factors and limits.
    int nDistributedSlackGenerators = 0;
    DeviceBuf distributedSlackGeneratorBus, distributedSlackGeneratorFactor, distributedSlackGeneratorMaxDelta, distributedSlackGeneratorMinDelta, distributedSlackGeneratorMarginUp, distributedSlackGeneratorMarginDown;
    DeviceBuf dDistributedSlackBusGeneratorStart, dDistributedSlackBusGeneratorEnd;
    // The run's outer-loop list holds OLF's DistributedSlackOuterLoop / ReactiveLimitsOuterLoop
    // (setBatchOuterLoops). The driver installs those phases only when the run has the loop, whatever
    // the packed tables hold: the reactive-limit pack also carries the stand-by automaton SVC buses the
    // MonitoringVoltage phase activates, with no ReactiveLimitsOuterLoop in the run. Never set = on, the
    // tables alone decide.
    bool distributedSlackOuterLoop = true;
    // Per (scenario, generator): its cumulative delta from the anchor, and whether the water-fill has
    // REMOVED it. Removal is not the same as "sitting at its bound": OLF removes an element only when
    // it strictly OVERSHOOTS, so one that lands exactly on its limit stays in the normalization for
    // another iteration — where it absorbs weight and moves nothing.
    DeviceBuf dDistributedSlackGeneratorMove, dDistributedSlackGeneratorRemoved;
    DeviceBuf dDistributedSlackPreviousGeneratorMove, dDistributedSlackPreviousBusTarget;   // nS x generators / buses: before a distribution
    SlotTable distributedSlackLostGenerators{1, 0};                             // per slot: the DS generators an outage removes, by identity
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
    SlotRows<int> slackBusRowOverride{-1};                        // per-slot slack row (a relocated slack), device rows
                                                                  // -1 = the base slack row
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
};

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_CONTEXT_DISTRIBUTED_SLACK_CONTEXT_CUH
