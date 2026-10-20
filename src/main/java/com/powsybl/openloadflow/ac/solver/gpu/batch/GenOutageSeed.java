/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.outerloop.gpu.ReactiveCapabilityCurves;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.GenOutage;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuRowMode;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuVoltageControlRole;

/**
 * The seeding of one slot's generator and injection outages ({@link GenOutage} plan entries): the per-bus target
 * overrides of the lost injections, the voltage controls they leave (flips to a reactive balance, DISTR_Q seeds,
 * killed pilot identities), the reactive capability and curve entries the lost generators take with them, and
 * their distributed-slack participation.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GenOutageSeed {

    private GenOutageSeed() {
    }

    /** Seed slot {@code s} from its plan entries {@code slotGos}; the rows in
     *  {@code pilotBalanceRows} host a pilot's relocated reactive balance in this scenario. */
    static void seed(SlotSeeding ctx, int s, GenOutage[] slotGos, gnu.trove.set.hash.TIntHashSet pilotBalanceRows) {
        final int chunkSize = ctx.chunkSize();
        final int n = ctx.n();
        final GpuAcData gpuData = ctx.gpuData();
        final int[] baseRowMode = ctx.baseRowMode();
        final ReactiveCapabilityCurves.Channel rlCurves = ctx.rlCurves();
        final java.util.Map<Integer, int[]> curveEntriesByPRow = ctx.curveEntriesByPRow();
        final RunScratch rs = ctx.rs();
        final ChunkLists cl = ctx.cl();
        GenOutage[] gos = slotGos;
        if (gos != null) {               // generator loss(es): per-BUS target overrides (merged by bus;
            for (GenOutage go : gos) {   // at most one DS-participant bus per contingency by construction)
                if (go.pRow() >= 0) {    // pRow < 0: a row-mode-only entry (a voltage
                    rs.overrides.add(s * n + go.pRow(), go.pTarget());   // control the island killed)
                    // A bus whose target does not move by the lost injection leaves that power in
                    // the balance, and the distributed-slack loop then reads it as a mismatch and
                    // undoes the loss pre-distribution (network.xiidm loads: 4183 of 7494 totals wrong).
                }
                if (go.qOverride()) {    // Q target sheds the lost gens' Q: on the PV->PQ flip AND
                    // on an ALREADY-PQ bus losing fixed-Q gens (#21)
                    // ...at the Q BALANCE row, which under a coupler is NOT the BUS_V row the
                    // PV->PQ mode flip below rewrites: the relocation moved the balance onto the
                    // parent's DUMMY_Q, so an absolute Q target written at BUS_V is a no-op and
                    // the lost load's REACTIVE power stays in the system. Measured on
                    // network.xiidm's load scope as AAHR_11 keeping F = +2.3268 pu, exactly minus
                    // its own q0, on the eight AAP* contingencies (worst 0.565 MW).
                    rs.overrides.add(s * n + go.qTargetRowOrV(), go.qTarget());
                }
                if (go.killsPilotVoltageIdentity()) {
                    // The pilot's voltage identity goes NOW, with the member's
                    // mode-4 row, so the two halves of the dead control land in
                    // the same fill.
                    int[] row = rs.rowModePerScenario[s] == baseRowMode
                            ? baseRowMode.clone() : rs.rowModePerScenario[s];
                    row[go.deadPilotVRow()] = GpuRowMode.EQUATION;
                    rs.rowModePerScenario[s] = row;
                    rs.overrides.add(s * n + go.deadPilotVRow(), go.deadPilotQTarget());
                }
                if (go.toPq()) {
                    int[] row = rs.rowModePerScenario[s] == baseRowMode ? baseRowMode.clone() : rs.rowModePerScenario[s];
                    row[go.vRow()] = go.pqRowModeOrDefault();   // BUS_V row becomes the reactive-power equation (PV→PQ)
                    rs.rowModePerScenario[s] = row;
                }
                if (go.reHomesRlEntry()) {
                    cl.rlLocalList.add(s);
                    cl.rlLocalList.add(go.rlEntry());
                    cl.rlLocalVt.add(go.reHomeVTarget());
                }
                if (go.reHomesRemoteController()) {
                    // OLF's re-merged control: this remote member's reactive-limit check reads the
                    // NEW pilot's voltage against the new pilot's target; its row stays the base
                    // table's sharing row, which the dependent pilot's pool uses as the carrier.
                    if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                        System.err.printf("QUALIFY_TRACE reHomeRemoteCtl slot=%d entry=%d of %d cmpRow=%d vt=%.6f%n",
                                s, go.rlEntry(), gpuData.reactiveLimits().voltageRow().length, go.reHomePilotVRow(), go.qTarget());
                    }
                    cl.rlCmpList.add(s);
                    cl.rlCmpList.add(go.rlEntry());
                    cl.rlCmpRows.add(go.reHomePilotVRow());
                    cl.rlVtOvIdx.add(s);
                    cl.rlVtOvIdx.add(go.rlEntry());
                    cl.rlVtOvVal.add(go.qTarget());
                }
                if (go.forbidsUnpin()) {
                    cl.rlNoUnpinList.add(s);
                    cl.rlNoUnpinList.add(go.rlEntry());
                }
                if (go.silencesReactiveLimitEntry()) {
                    // Nothing left to control voltage at that bus: its reactive-limit
                    // entry is seeded FROZEN for this scenario (setBatchRlGenLost) - no
                    // pin, no switch back to PV with its only generator gone, and not a
                    // live controller to the pilot-self layout. NOT the islanded channel
                    // a contingency-stranded controller uses: "islanded" means the bus
                    // itself is gone and closes the pilot-self gate (no balance to
                    // emit), while this bus is alive and its reactive balance must be
                    // emitted - on the carrier, then on its own row once every survivor
                    // pins. Measured on rte6515 GEN-1522 (the pilot VL-1522_0 loses its
                    // only generator, the survivor VL-1522_1 pins): islanded, the gate
                    // suppressed the pilot's q, its row flipped to host it assembled
                    // EMPTY and the bus sat at its 1.122 target against OLF's 1.0811,
                    // 3.2 kV at VL-1525.
                    // Only the PILOT of a pilot-self group (the row flip armed below) needs
                    // the frozen channel: a dead MEMBER's balance is already expressed by the
                    // inactive-member weights on the survivors' DISTR_Q rows, and frozen it
                    // would also be emitted raw on the carrier (deviceGenOutageAtDistrQBus,
                    // deviceTwoMemberPilotControllerLosingAController: 0.37 kV).
                    // EXCEPT a LOCAL controller the base had PINNED: OLF's processLostGenerators disables its
                    // voltage control but leaves its qLimitType and its place among the controller elements,
                    // so the reactive-limits loop still runs checkPqBus on it - the bus switches back to PV
                    // when its voltage asks (MIN_Q pin, V under target) and pins again at its now-empty
                    // range ([0, 0]: no generator left) on the next pass. Measured on network.xiidm/FBQV:
                    // FBQF_0 back to PV at OLF's first reactive-limit pass and pinned MAX_Q at the second;
                    // silenced, the device kept it pinned and CQEM_0 pinned one pass late - the only
                    // contingency of 24,376 whose outer-loop sequence differed from the 1-thread CPU's.
                    // Its limits already collapse with the outage (minQDelta/maxQDelta, the curve channel),
                    // so the entry is simply left live (fixture:
                    // deviceSwitchedToPqInBaseBusLosingItsOnlyGeneratorIsStillCheckedMatchesCpu).
                    boolean liveDeadLocal = (gpuData.reactiveLimits().voltageControlRole() == null || gpuData.reactiveLimits().voltageControlRole().length <= go.rlEntry()
                                    || gpuData.reactiveLimits().voltageControlRole()[go.rlEntry()] == GpuVoltageControlRole.LOCAL)
                            && gpuData.switchedPvToPqInBase(go.rlEntry());
                    if (liveDeadLocal) {
                        // left live: no channel
                    } else if (go.flipsPilotRowWhenAllPinned() || gpuData.switchedPvToPqInBase(go.rlEntry())) {
                        // ...and a MEMBER the base PINNED: its reactive balance already sits on
                        // its own mode-4 row (its pin row), and OLF recomputes that balance's
                        // target over the generators left - none - the moment the last one goes
                        // (setGeneratorVoltageControlEnabledAndRecomputeTargetQ). Frozen gen-lost,
                        // the seed stamps 0 - loadQ there and the check never touches it again;
                        // islanded, the row kept the pinned LIMIT as its target: measured on the
                        // merged-control fixture as 0.36 kV at b1 (row 7 at 0.05 against OLF's 0),
                        // and the shape of rte6515's VL-5997 / VL-6003 / VL-6194 / VL-6229
                        // relocated pins.
                        cl.rlGenLostList.add(s);
                        cl.rlGenLostList.add(go.rlEntry());
                    } else {
                        cl.rlIslandedList.add(s);
                        cl.rlIslandedList.add(go.rlEntry());
                    }
                }
                if (go.deactivatesDqMember()) {
                    cl.dqInactList.add(s);
                    cl.dqInactList.add(go.inactiveDqMember());
                }
                if (go.onSlack()) {
                    if (cl.outageSlackShift == null) {
                        cl.outageSlackShift = new double[chunkSize];
                    }
                    cl.outageSlackShift[s] += go.pStep();
                }
                if (go.pilotQGroup() >= 0) {
                    cl.pilotQDeltas.merge(s * (gpuData.reactiveRelocation().groupStart().length - 1) + go.pilotQGroup(),
                            go.pilotQDelta(), Double::sum);
                }
                if (go.overridesDqMemberTargetQ() && gpuData.distrQMembers().busNum() != null) {
                    cl.dqTqOvKey.add(s * gpuData.distrQMembers().busNum().length
                            + go.dqTargetQOvMember());
                    cl.dqTqOvVal.add(go.dqTargetQOvValue());
                }
                if (go.seedsDqTargets()) {
                    for (int t = 0; t < go.dqTargetRows().length; t++) {
                        int dqRow2 = go.dqTargetRows()[t];
                        int dqMode = rs.rowModePerScenario[s] != null
                                ? rs.rowModePerScenario[s][dqRow2] : baseRowMode[dqRow2];
                        // Seed only where the row still HOSTS a DISTR_Q equation this
                        // scenario. Mode 4 means it hosts its own reactive balance (an RL
                        // pin, or a control the contingency killed and whose BUS_TARGET_Q
                        // the injection override owns); mode 0 means it is an IDENTITY —
                        // a re-homed control's voltage target (reHomedPvEntry, the remote
                        // main of a split group) or a frozen/outaged row. updateDistrQ-
                        // TargetsBatch refuses to maintain a DISTR_Q target on either ("A
                        // DISTR_Q target has no business here"), so uploading one is a
                        // write this code does not own: it lands as a SECOND override on a
                        // row that already has one. Measured over the full busbar scope:
                        // 26 mode-4 rows across 24 contingencies
                        // disagreed, all from this site — BUS-1359_BBS_1 and BUS-1869_BBS_1
                        // (the +-2 violation pair), BUS-1599_BBS_1. Then, once the split
                        // re-homes batched (tasks #70), the mode-0 case: BUS-2640_BBS_1's
                        // elected main VL-2640_1 got its 1.034 identity on row 5956 AND
                        // this seed's 0.0 (its renormalized DISTR_Q target); the device
                        // picked 0.0 in the full 6667-contingency batch (slot 1844: V
                        // driven to 0, 193.85 MW of distributed power, +197 violations)
                        // and 1.034 in a 3-contingency batch (slot 0, bit-identical to
                        // OLF). The earlier "mode 0 is NOT skipped: skipping it moved
                        // deviceMultiChunkMatchesCpu by 1.06e-12 kV" was the race's own
                        // noise being guarded.
                        if (dqMode == GpuRowMode.BUS_TARGET_Q || dqMode == GpuRowMode.FIXED_VARIABLE || dqMode == GpuRowMode.CROSS_FIXED_VARIABLE) {
                            continue;
                        }
                        // ...nor on the row that HOSTS A PILOT'S BALANCE this scenario: a frozen base
                        // carrier hands the pilot's reactive balance to the next live member's row
                        // (the block above stamps the pilot's target there), and that member's own
                        // renormalized DISTR_Q target is not that row's equation any more. The two
                        // writes raced on the device and the pilot's won by luck — measured once the
                        // race was closed: deviceBusLossIslandingBaseCarrierMemberMatchesCpu, row 2
                        // −2.0 (b4's load) shadowed by this seed's 0.0, b2 22.49 → 22.04 kV.
                        if (pilotBalanceRows.contains(dqRow2)) {
                            continue;
                        }
                        // Any OTHER earlier override on a seed row is a producer this code does not
                        // know about: keep the more specific one, and count it so a composition gap
                        // (two seeds for one row from two paths) is measurable, never a race.
                        if (rs.overrides.has(s * n + dqRow2)) {
                            com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("HOST:DQSEED_SHADOWED");
                            continue;
                        }
                        rs.overrides.add(s * n + dqRow2, go.dqTargetValues()[t]);
                    }
                }
                if (go.flipsPilotRowWhenAllPinned()) {
                    cl.pfSlot.add(s);                     // handled IN the solve
                    cl.pfRow.add(go.pinnedPilotVRow());
                    cl.pfQTarget.add(go.pinnedPilotQTarget());
                    cl.pfRlCount.add(go.pilotGroupSurvivorRlEntries().length);
                    for (int rl : go.pilotGroupSurvivorRlEntries()) {
                        cl.pfRlPool.add(rl);
                    }
                }
                if (go.shrinksReactiveLimits()) {
                    // The bus KEEPS voltage control, so its reactive-limit entry is
                    // still read — and the device's aggregate still holds the lost
                    // generator's capability, which OLF's bus.getMinQ() does not.
                    cl.rlQOvSlot.add(s);
                    cl.rlQOvSlot.add(go.rlEntry());
                    cl.rlQOvVal.add(go.minQDelta());
                    cl.rlQOvVal.add(go.maxQDelta());
                    cl.rlQOvVal.add(go.loadQDelta());
                    cl.rlQOvVal.add(go.svcBminDelta());
                    cl.rlQOvVal.add(go.svcBmaxDelta());
                }
                if (go.pRow() >= 0 && curveEntriesByPRow.containsKey(go.pRow())) {
                    // The CURVE channel reconstructs each generator's OWN active
                    // power from the bus row target, to re-evaluate its capability
                    // curve there. A generator loss skews that reconstruction three
                    // ways, and all three are repaired here, per scenario:
                    //  - the lost generator's entry must stop evaluating (its base
                    //    capability is already taken off by minQDelta/maxQDelta);
                    //    a pure LOAD loss loses nobody and only needs the second;
                    //  - the step the outage itself puts in the row target is not
                    //    distributed-slack power and must move nobody's curve;
                    //  - the survivors now share the bus's slack over their own
                    //    factors alone, so each one's share renormalizes.
                    for (int e : curveEntriesByPRow.get(go.pRow())) {
                        double busFactor = rlCurves.cvBusFactor()[e];
                        double left = busFactor - go.lostFactorSum();
                        boolean dead = go.lostGenIds().contains(rlCurves.cvGenId()[e]);
                        cl.cvOvSlot.add(s);
                        cl.cvOvSlot.add(e);
                        cl.cvOvVal.add(go.pStep());
                        // an AC-emulation VSC's P is its link's: only its own loss matters
                        cl.cvOvVal.add(dead || !rlCurves.isHvdc(e) && left <= 0 ? -1
                                : rlCurves.isHvdc(e) ? 1 : busFactor / left);
                    }
                }
                if (rs.hasDs && go.dsBus() >= 0) {   // lost gen(s) on a DS-participant bus: per-scenario reweight
                    rs.dsOvBus[s] = go.dsBus();
                    rs.dsOvFactorDelta[s] = go.dsFactorDelta();
                    rs.dsOvBaseDelta[s] = go.dsBaseDelta();
                    rs.dsOvMaxDelta[s] = go.dsMaxDelta();
                    rs.dsOvMinDelta[s] = go.dsMinDelta();
                    // ...and WHICH generators left, as slots in the DS generator pack. With the
                    // identity the outaged bus's SURVIVORS can take part in the per-generator
                    // water-fill like any others, instead of the bus riding on one aggregate
                    // entry — which is what leaves their own P unknown (genCum stays zero
                    // there) and sends the curve channel back to the share model that
                    // per-generator clamping breaks.
                    String[] dgIds = gpuData.distributedSlack().generatorId();
                    for (int g = 0; dgIds != null && g < dgIds.length; g++) {
                        if (go.lostGenIds().contains(dgIds[g])) {
                            cl.dsLostGenList.add(s);
                            cl.dsLostGenList.add(g);
                        }
                    }
                }
            }
        }
    }
}
