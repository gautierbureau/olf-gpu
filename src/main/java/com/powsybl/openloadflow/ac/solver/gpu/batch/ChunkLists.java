/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.RemoteVoltageControlNative;
import com.powsybl.openloadflow.ac.outerloop.gpu.DistributedSlackNative;
import com.powsybl.openloadflow.ac.outerloop.gpu.ForcedTargetQ;
import com.powsybl.openloadflow.ac.outerloop.gpu.ReactiveLimitsNative;
import com.powsybl.openloadflow.ac.solver.gpu.NewtonSettingsNative;

import java.util.ArrayList;
import java.util.List;

/**
 * The per-chunk seeding lists of the chunk solver: what the per-slot seeding writes and the device reads, built
 * FRESH for every chunk. The scenario index in each entry is a SLOT of this chunk, so a list that outlived its
 * chunk would apply chunk N's entries to chunk N+1's slots, which no validation run in windows of one chunk sees.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class ChunkLists {

    // {slot, participant} pairs: the distributed-slack participants this scenario
    // strands in another synchronous component (see distributeSlackBatch).
    final List<Integer> dsIslandedList = new ArrayList<>();
    // {slot, participant, row} triples: a DS participant whose P-balance row this
    // scenario MOVES. Under the coupler-tree relocation a child bus's P balance lives on
    // its parent coupler's DUMMY_P row, and dsPhiRow — a STRUCTURAL array — points there.
    // A contingency that opens that coupler pins the dummy row to a DUMMY_TARGET identity
    // and the stock-placement fallback puts the balance back on the child's OWN phi row,
    // so for THAT scenario the participant's row is the child's. Without this the whole DS
    // machinery kept writing to the pinned row: measured on the middle-section fixture,
    // g2's 0.173869 pu share landed on a dead row while both DS kernels read its
    // (0 - anchor) as -1.043213 of "absorbed", so the slack loop fired four times where
    // OLF fires once and over-placed 0.0687 pu on the rows that still worked (0.34 kV on
    // vl3_b). Per chunk, like every other list here — the index is a chunk SLOT.
    final List<Integer> dsRowOvList = new ArrayList<>();
    // ...and the same diversion host-side, for the loss pre-distribution below (it runs
    // after this loop and composes its shares onto the participants' rows).
    final int[][] dsRowOvBySlot;
    final List<Integer> rlIslandedList = new ArrayList<>();
    final List<Integer> rlNoUnpinList = new ArrayList<>();   // (scenario, entry): an unpin rejects the scenario
    final List<Integer> rlGenLostList = new ArrayList<>();   // (scenario, entry): the bus lost every generator
    // tasks #71: per (scenario, dq member) subgroup labels of a split group, dense, null when none
    final int nDqMembersAll;
    final byte[] dqSubDense;
    boolean anyDqSub = false;
    final List<Integer> rlLocalList = new ArrayList<>();     // (slot, entry): re-homed PV buses checked as LOCAL
    final List<Integer> rlCmpList = new ArrayList<>();       // (slot, entry): remote controllers re-homed onto a NEW pilot
    final List<Integer> rlCmpRows = new ArrayList<>();       // ...the new pilot's V row their limit check reads
    final List<Integer> rlVtOvIdx = new ArrayList<>();       // (slot, entry): per-scenario reactive-limit V target
    final List<Double> rlVtOvVal = new ArrayList<>();
    final List<Double> rlLocalVt = new ArrayList<>();        // their own voltage target (pu)
    // {slot, rlEntry} pairs + {dMinQ, dMaxQ}: the reactive capability a lost generator
    // takes with it, at a bus that still controls voltage.
    final List<Integer> rlQOvSlot = new ArrayList<>();
    final List<Double> rlQOvVal = new ArrayList<>();
    // {slot, curveEntry} pairs + {pStep, shareScale}: how a generator outage repairs
    // the reactive-capability CURVE channel at its own bus (shareScale < 0 = the entry
    // is the lost generator's own and stops evaluating).
    final List<Integer> cvOvSlot = new ArrayList<>();
    final List<Double> cvOvVal = new ArrayList<>();
    // {slot, dqMember} pairs: a DISTR_Q controller that stops controlling in this
    // scenario. Its key has to leave the renormalization denominator on the TARGET
    // side, which the qPct override channel does not reach.
    final List<Integer> dqInactList = new ArrayList<>();
    // PER CHUNK, like dqInactList above: the scenario index in every one of these is a
    // SLOT in this chunk, so a list that outlives the chunk applies chunk N's entries to
    // chunk N+1's slots. That is invisible to any validation run in windows of one chunk.
    final List<Integer> dsLostGenList = new ArrayList<>();
    final List<Integer> dsOutList = new ArrayList<>();   // (slot, DS generator) a GeneratorAction took out
    final List<Integer> frozenSeedSlot = new ArrayList<>();
    final List<Integer> frozenSeedRow = new ArrayList<>();
    final List<Double> frozenSeedValue = new ArrayList<>();
    // slot -> the RL entries of the controllers still holding a deactivated pilot's
    // voltage. All pinned = its Q balance lost its row; the CPU replays that scenario.
    // Pilot-row flip entries: the scenarios that used to be REJECTED by that check are
    // FIXED in the solve instead. Per chunk, like every other list here — the scenario
    // index is a chunk SLOT.
    final List<Integer> dqTqOvKey = new ArrayList<>();     // slot*nDqMembers + member
    final List<Double> dqTqOvVal = new ArrayList<>();
    final java.util.Map<Integer, Double> pilotQDeltas = new java.util.LinkedHashMap<>();   // slot*nQr + group
    double[] outageSlackShift = null;    // per slot: the move of the slack's scheduled P by a load lost there
    int[] dsSlackRows = null;            // per slot: the row the distributed slack reads the slack on (-1 = base)
    final List<Integer> pfSlot = new ArrayList<>();
    final List<Integer> pfRow = new ArrayList<>();
    final List<Double> pfQTarget = new ArrayList<>();
    final List<Integer> pfRlCount = new ArrayList<>();
    final List<Integer> pfRlPool = new ArrayList<>();
    final List<Integer> dqOvIdxList = new ArrayList<>();   // {s, contrib} pairs (DISTR_Q reweight)
    final List<Double> dqOvWList = new ArrayList<>();
    final List<Integer> dqKeyOvIdxList = new ArrayList<>();
    final List<Double> dqKeyOvList = new ArrayList<>();
    final List<Integer> drOvIdxList = new ArrayList<>();   // {s, contrib} pairs (DISTR_RHO renorm, plan A5)
    final List<Double> drOvCList = new ArrayList<>();
    // per-scenario ACTIVE equation counts (L2 threshold parity — the CPU removes an
    // equation per frozen islanded row and per outaged element carrying a variable)
    final int[] nActive;
    final double[] preLossBySlot;   // scheduled active power loss per slot (plain phase)
    // each slot's per-generator loss pre-distribution moves: where the solve starts each generator (the
    // curve and forced-target-Q channels evaluate it there until a slack pass runs), and the anchors
    // of the contingency's strategy pairs when none does
    final double[][] genPreMovesBySlot;
    // ...and what the pre-distribution put on the PARTICIPANTS alone (the slack keeps the
    // rest). PRELOSS_AFTER has to compare the device's final participant total against
    // THIS, not against the whole mismatch: with a participating slack the two differ by
    // the slack's retained share, which is not a device defect.
    final double[] preAbsorbedBySlot;

    ChunkLists(int chunkSize, int n, GpuAcData gpuData, ForcedTargetQ.Channel forcedTargetQ, int ngen) {
        this.dsRowOvBySlot = new int[chunkSize][];
        this.nDqMembersAll = gpuData.dqMemberBusNum() != null ? gpuData.dqMemberBusNum().length : 0;
        this.dqSubDense = new byte[chunkSize * nDqMembersAll];
        this.nActive = new int[chunkSize];
        java.util.Arrays.fill(nActive, n);
        this.preLossBySlot = new double[chunkSize];
        this.genPreMovesBySlot = forcedTargetQ != null || ngen > 0 ? new double[chunkSize][] : null;
        this.preAbsorbedBySlot = new double[chunkSize];
    }

    /** Upload the lists the device reads as per-chunk context state; an empty list clears its channel. */
    void upload(long handle, GpuAcData gpuData) {
        // DISTR_Q weight overrides: per-chunk context state (orchestrator re-solves keep
        // the slot assignment); cleared when the chunk has none.
        // Context state, like the qPct overrides beside it: a monitoring or secondary-VC
        // re-solve on this context must not drop it (as a solve PARAMETER it did — the
        // re-solve passed null and the DISTR_Q target was recomputed with the dead
        // controller's key back in the denominator, clobbering the right target).
        RemoteVoltageControlNative.setBatchDqInactiveMembers(handle,
                ChunkArrays.intsOrNull(dqInactList));
        NewtonSettingsNative.setBatchFrozenRows(handle,
                frozenSeedSlot.isEmpty() ? null : java.util.stream.IntStream
                        .range(0, frozenSeedSlot.size())
                        .flatMap(k -> java.util.stream.IntStream.of(
                                frozenSeedSlot.get(k), frozenSeedRow.get(k)))
                        .toArray());
        DistributedSlackNative.setBatchDsLostGenerators(handle,
                ChunkArrays.intsOrNull(dsLostGenList));
        DistributedSlackNative.setBatchDsOutGenerators(handle,
                ChunkArrays.intsOrNull(dsOutList));
        RemoteVoltageControlNative.setBatchDqWeightOverrides(handle,
                ChunkArrays.intsOrNull(dqOvIdxList),
                ChunkArrays.doublesOrNull(dqOvWList));
        RemoteVoltageControlNative.setBatchDqKeyOverrides(handle,
                ChunkArrays.intsOrNull(dqKeyOvIdxList),
                ChunkArrays.doublesOrNull(dqKeyOvList));
        RemoteVoltageControlNative.setBatchDrCoefOverrides(handle,
                ChunkArrays.intsOrNull(drOvIdxList),
                ChunkArrays.doublesOrNull(drOvCList));
        NewtonSettingsNative.setBatchActiveRowCounts(handle, nActive);
        // Cleared when the chunk strands nobody, so a chunk never inherits the previous
        // chunk's participant mask through the reused context.
        DistributedSlackNative.setBatchDsIslanded(handle,
                ChunkArrays.intsOrNull(dsIslandedList));
        DistributedSlackNative.setBatchDsRowOverrides(handle,
                ChunkArrays.intsOrNull(dsRowOvList));
        ReactiveLimitsNative.setBatchRlIslanded(handle,
                ChunkArrays.intsOrNull(rlIslandedList));
        ReactiveLimitsNative.setBatchRlNoUnpin(handle,
                ChunkArrays.intsOrNull(rlNoUnpinList));
        ReactiveLimitsNative.setBatchRlGenLost(handle,
                ChunkArrays.intsOrNull(rlGenLostList));
        RemoteVoltageControlNative.setBatchDqSubgroups(handle, anyDqSub ? dqSubDense : null,
                anyDqSub ? RelocatedRows.qrMemberStart(gpuData) : null);
        ReactiveLimitsNative.setBatchRlLocal(handle,
                ChunkArrays.intsOrNull(rlLocalList),
                rlLocalList.isEmpty() ? null : rlLocalVt.stream().mapToDouble(Double::doubleValue).toArray());
        ReactiveLimitsNative.setBatchRlCmpOverrides(handle,
                ChunkArrays.intsOrNull(rlCmpList),
                rlCmpList.isEmpty() ? null : rlCmpRows.stream().mapToInt(Integer::intValue).toArray());
        RemoteVoltageControlNative.setBatchDqMemberTargetQOverrides(handle,
                ChunkArrays.intsOrNull(dqTqOvKey),
                dqTqOvKey.isEmpty() ? null : dqTqOvVal.stream().mapToDouble(Double::doubleValue).toArray());
        DistributedSlackNative.setBatchDsSlackRows(handle, dsSlackRows);
        RemoteVoltageControlNative.setBatchPilotQDeltas(handle,
                pilotQDeltas.isEmpty() ? null : pilotQDeltas.keySet().stream().mapToInt(Integer::intValue).toArray(),
                pilotQDeltas.isEmpty() ? null : pilotQDeltas.values().stream().mapToDouble(Double::doubleValue).toArray());
        RemoteVoltageControlNative.setBatchPilotAllPinnedFlip(handle,
                ChunkArrays.intsOrNull(pfSlot),
                pfSlot.isEmpty() ? null : pfRow.stream().mapToInt(Integer::intValue).toArray(),
                pfSlot.isEmpty() ? null : pfQTarget.stream().mapToDouble(Double::doubleValue).toArray(),
                pfSlot.isEmpty() ? null : pfRlCount.stream().mapToInt(Integer::intValue).toArray(),
                pfSlot.isEmpty() ? null : pfRlPool.stream().mapToInt(Integer::intValue).toArray());
    }
}
