/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.vector.gpu.DistrQPacker;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfShunt;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Qualification helpers for the batched N-1 path: the frozen-island bookkeeping (which islanded-bus
 * variable rows a fragmenting outage freezes to a clean identity state, so the islanded sub-block stays
 * non-singular in the batched factorization) and the fixed-shunt-outage batchability check (a packed
 * fixed shunt that is not a DISTR_Q contributor).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuBatchQualification {

    private GpuBatchQualification() {
    }

    /**
     * Whether a fixed-shunt disconnection can be batched as a plain element mask. Requires every outaged shunt
     * to be a packed fixed shunt ({@code getElementIndex >= 0}) and NOT a reactive-distribution (DISTR_Q)
     * contributor — a shunt whose reactive feeds a controlled bus's DISTR_Q sum is read unconditionally by the
     * batched DISTR_Q fill (it ignores the disabled mask), so masking it would leave the sum stale. Such a shunt
     * outage is routed to the CPU instead.
     */
    static boolean canBatchShuntOutage(List<LfShunt> shunts, GpuBatchElementMapper elementMapper,
                                               GpuAcData gpuData) {
        // A shunt that feeds a DISTR_Q sum used to be refused here ("a mask would not update the
        // sum"): fillDistrQBatch and the reactive-limits family evaluation both read the shunt's
        // admittance through applyShuntOverride, the same per-scenario channel the bus fill uses, so
        // a shifted shunt's share of every DISTR_Q row follows the shift. Measured on the
        // middle-section fixture: dropping the shift instead leaks the islanded shunt's base
        // reactive power into the survivors' rows (0.115 kV); keeping it lands at 1e-12.
        for (LfShunt shunt : shunts) {
            if (elementMapper.getElementIndex(shunt) < 0) {
                return false;                                // not a packed fixed shunt (controller/SVC shunt)
            }
        }
        return true;
    }


    /** A plan refusal, counted in the census AND named by contingency id: the census counts are per attempt and
     *  carry no id, which made 19 refusals read as device failures (tasks #70). */
    static void refuse(BatchContingency c, String tag) {
        com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record(tag);
        com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.planRefused(c.contingencyId(), tag);
    }



    /** The batched-scenario plan: which contingencies qualified ({@code batchedIndices}) and, per
     *  contingency, the frozen islanded-bus rows/targets, the islanded LfBus nums (excluded from result
     *  building), the merged per-bus generator outages (else null), and the count of on-GPU frozen islands. */
    record ScenarioPlan(List<Integer> batchedIndices, int[][] frozenRowsByIndex,
                        int[][] islandedNumsByIndex, double[][] frozenTargetsByIndex,
                        GenOutage[][] genOutagesByIndex, int islandingCount,
                        double[] activePowerLossByIndex,
                        int[][] dqOvContribByIndex, double[][] dqOvWByIndex,
                        /** Per scenario, the members whose reactive KEY the contingency changes (a generator lost at a
                         *  bus that keeps a controller) and OLF's post-loss key; the kernels renormalize them live. */
                        int[][] dqKeyOvMemberByIndex, double[][] dqKeyOvByIndex,
                        int[][] drOvContribByIndex, double[][] drOvCByIndex,
                        /** Per scenario, the distributed-slack participants this contingency strands in
                         *  another synchronous component — the set the device must renormalize over. */
                        int[][] islandedDsByIndex,
                        /** Per scenario, the reactive-limits controllers stranded in another component. */
                        int[][] islandedRlByIndex,
                        /** Per scenario, AC-emulation HVDCs with an ISLANDED endpoint. OLF's
                         *  processHvdcsWithoutPower zeroes both converter stations' targetP; in AC
                         *  emulation those targets are already 0 (the injection is an equation term,
                         *  not a target), so masking the element is the whole of it. */
                        List<com.powsybl.openloadflow.network.LfHvdc>[] islandedHvdcsByIndex,
                        /** Per scenario, {dq member, subgroup label} pairs of a split group (tasks #71), else null. */
                        int[][] dqSubByIndex,
                        /** Per scenario, the slack OLF re-picks when the outage isolates the base one. */
                        SlackRelocations slackRelocations) {
    }

    /** Per contingency index: the angle row of the bus OLF makes the slack when the outage isolates the base one
     *  (-1 = the base slack stays), which becomes the reference identity, and how far that bus's anchor target sits
     *  from the base slack's (the distributed-slack check reads the new slack's balance against it). */
    record SlackRelocations(int[] rowByIndex, double[] shiftByIndex) {

        int row(int index) {
            return rowByIndex[index];
        }
    }

    /**
     * Qualify each contingency for the batched GPU path and build the per-scenario plan. An outage that
     * keeps the main component intact batches as a value-only mask; a fragmenting one batches with its
     * islanded buses frozen to identity; injection/shift/tap/reclose/switch shapes are checked against the
     * packed pattern and the on-device retain math, and anything unrepresentable is left out (CPU fallback).
     */
    static ScenarioPlan planScenarios(List<BatchContingency> contingencies,
            List<GpuBatchContingencyValidator.Qualification> qualifications,
            GpuBatchElementMapper elementMapper, GpuAcData gpuData,
            EquationSystem<AcVariableType, AcEquationType> es,
            double dsSum, boolean slackParticipates,
            com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDsParams dsParams, Set<LfBranch> distrRhoControllers,
            Set<LfBranch> relocatedCouplers, boolean reactiveLimits, boolean incrementalTvc) {
        ScenarioPlanBuilder pb = new ScenarioPlanBuilder(contingencies.size(), gpuData);
        PlanContext ctx = new PlanContext(contingencies, qualifications, elementMapper, gpuData, es, dsSum,
                slackParticipates, dsParams, distrRhoControllers, relocatedCouplers, reactiveLimits, incrementalTvc);
        for (int i = 0; i < qualifications.size(); i++) {
            GpuBatchContingencyValidator.Qualification q = qualifications.get(i);
            if (!q.qualified()) {
                // The validator's OWN reason, not a fixed tag: this site is reached by every disqualified
                // verdict, so recording "SIDE_OPEN_UNPACKED" for all of them misreports the census — the one
                // entry at rte6515's full scope is an ISLANDS_SLACK outage, which reads as a side-open shape.
                String tag = "PLAN:" + (q.reason() != null ? q.reason() : "UNQUALIFIED");
                com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record(tag);
                // ...and by id, so a scope report names the reason of every refusal (it printed "?" for these)
                com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.planRefused(contingencies.get(i).contingencyId(), tag);
                continue;
            }
            BatchContingency c = contingencies.get(i);
            // Reclosing into a multi-bus island. OLF's reclose re-adds the reclosed branch to the post-contingency
            // connectivity, from which the contingency removed the island's own branches too (both their ends were
            // lost), so only the reclosed branch's end comes back and the rest of the island stays out. The device
            // reconnects the whole island (rte6515 LINE-3273-1812, a 10-bus island reclosed: the device redistributed
            // 154.7 MW of the island's load, OLF nothing, 3 kV at VL-1812). A pair that brings back any other bus: CPU.
            Set<Integer> islandedHere = new java.util.HashSet<>();
            q.islandedBuses().forEach(b -> islandedHere.add(b.getNum()));
            int strategySep = c.contingencyId().indexOf(GpuBatchStatus.STRATEGY_KEY_SEP);
            if (strategySep < 0) {
                pb.islandedByContingency.put(c.contingencyId(), islandedHere);
                pb.openedByContingency.put(c.contingencyId(), c.outagedBranches());
            } else {
                String parentId = c.contingencyId().substring(0, strategySep);
                Set<Integer> parentIslanded = pb.islandedByContingency.getOrDefault(parentId, Set.of());
                Set<Integer> reclosedEnds = new java.util.HashSet<>();
                for (LfBranch b : pb.openedByContingency.getOrDefault(parentId, List.of())) {
                    if (!c.outagedBranches().contains(b)) {
                        if (b.getBus1() != null) {
                            reclosedEnds.add(b.getBus1().getNum());
                        }
                        if (b.getBus2() != null) {
                            reclosedEnds.add(b.getBus2().getNum());
                        }
                    }
                }
                if (parentIslanded.stream().anyMatch(num -> !islandedHere.contains(num) && !reclosedEnds.contains(num))) {
                    refuse(c, "PAIR:RECLOSE_INTO_ISLAND");
                    continue;
                }
            }
            // A GeneratorAction moves its generator's target P, and with reactive limits its capability curve moves
            // with it. The curve and forced-target-Q channels track a generator's P only when it is in the
            // distributed-slack generator pack (its anchor and its own move); any other curve generator keeps its
            // limits at the base target P - CPU.
            if (reactiveLimits && c.generatorActions().stream()
                    .anyMatch(m -> ControlReHoming.hasCurve(m.generator()) && !ControlReHoming.inGeneratorPack(gpuData, m.generator().getId()))) {
                refuse(c, "PLAN:PAIR_ACTION_UNTRACKED_CURVE");
                continue;
            }
            // ...and re-checks its participation at the new target, which the device applies to the generator pack.
            // A participant outside a non-empty pack (its bus has no active P balance row) has nowhere to sit out. (In
            // a slack-only distribution there is no pack, and any action there moves the slack bus: SLACK_BUS.)
            if (gpuData.dgGenId() != null && gpuData.dgGenId().length > 0 && c.generatorActions().stream()
                    .anyMatch(m -> dsParams.factor(m.generator()) != 0 && !ControlReHoming.inGeneratorPack(gpuData, m.generator().getId()))) {
                refuse(c, "PLAN:PAIR_ACTION_UNTRACKED_PARTICIPANT");
                continue;
            }
            LfBus newSlack = q.relocatedSlack();
            if (newSlack != null) {
                // OLF moved the slack to newSlack (the base one is in the island): the new slack's P balance leaves
                // the system and its angle becomes the reference. The device follows when that balance sits on the
                // bus's own angle row (not relocated onto a coupler) and neither slack takes part in the
                // distribution (the retained share would move with it): the angle row turns into the reference
                // identity, and the distributed-slack check reads that row's injection against the new slack's
                // target.
                if (pb.busRows == null) {
                    pb.busRows = com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor
                            .busPowerRows(newSlack.getNetwork(), es);
                }
                var phiVar = es.getVariableSet().getVariable(newSlack.getNum(), AcVariableType.BUS_PHI);
                int phiRow = phiVar == null ? -1 : phiVar.getRow();
                LfBus baseSlack = newSlack.getNetwork().getSynchronousNetworks().getFirst().getSlackBuses().getFirst();
                boolean ds = gpuData.dsSlackPhiRow() >= 0;
                boolean participates = ds && (dsParams.busFactor(baseSlack) != 0 || dsParams.busFactor(newSlack) != 0);
                boolean slackOnly = ds && gpuData.dsPhiRow().length == 0;
                // the slack's balance is read from the injection buffer, which carries no AC-emulation HVDC term
                boolean emulatedHvdc = newSlack.getHvdcs().stream().anyMatch(com.powsybl.openloadflow.network.LfHvdc::isAcEmulation);
                boolean balanceActive = es.getEquation(newSlack.getNum(), AcEquationType.BUS_TARGET_P)
                        .map(com.powsybl.openloadflow.equations.Equation::isActive).orElse(false);
                if (phiRow < 0 || !balanceActive || pb.busRows[0][newSlack.getNum()] != phiRow
                        || participates || slackOnly || emulatedHvdc) {
                    refuse(c, "PLAN:SLACK_RELOCATION_SHAPE");
                    continue;
                }
                pb.relocatedSlackRow[i] = phiRow;
                pb.relocatedSlackShift[i] = ds ? newSlack.getTargetP() - gpuData.dsSlackTargetP() : 0;
                com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:SLACK_RELOCATED");
            }
            java.util.Set<Integer> outagedZiNums = c.outagedZiBranches().stream()
                    .map(LfBranch::getNum).collect(java.util.stream.Collectors.toSet());
            // SIDE-OPEN scope (conservative): a branch that carries a VARIABLE (BRANCH_RHO1 /
            // BRANCH_ALPHA1) or belongs to a DISTR_RHO group changes the live es shape when
            // half-opened — CPU for those; plain impedance branches batch.
            if (!c.sideOpenBranches().isEmpty() && c.sideOpenBranches().stream().anyMatch(so ->
                    distrRhoControllers.contains(so.branch())
                    || es.getVariableSet().getVariable(so.branch().getNum(), AcVariableType.BRANCH_RHO1).getRow() >= 0
                    || es.getVariableSet().getVariable(so.branch().getNum(), AcVariableType.BRANCH_ALPHA1).getRow() >= 0)) {
                refuse(c, "PLAN:VAR_CARRYING_SIDE_OPEN");
                continue;                                    // var-carrying / DISTR_RHO side-open — CPU
            }
            if (!c.lostGenerators().isEmpty() || !c.lostLoads().isEmpty() || !c.shuntShifts().isEmpty()
                    || !c.tapOverrides().isEmpty() || !c.reclosedBranches().isEmpty()
                    || !c.outagedZiBranches().isEmpty()
                    || !c.outagedHvdcs().isEmpty()) {        // injection/shift/tap/reclose/switch/hvdc-outage (or branch+)
                InjectionOutagePlanner.plan(ctx, pb, i, c, q, outagedZiNums);
            } else if (!c.outagedShunts().isEmpty()) {       // fixed-shunt disconnection: mask the shunt element
                if (canBatchShuntOutage(c.outagedShunts(), elementMapper, gpuData)) {
                    pb.batchedIndices.add(i);
                    pb.frozenRowsByIndex[i] = IslandedBuses.NO_FROZEN_ROWS;
                    pb.frozenTargetsByIndex[i] = IslandedBuses.NO_FROZEN_TARGETS;
                }                                            // else: not a packed fixed shunt / DISTR_Q contributor → CPU
            } else {
                List<LfBus> lostVcControllers = new ArrayList<>();
                List<LfBus> deadControlBuses = new ArrayList<>();
                List<com.powsybl.openloadflow.network.LfHvdc> islandedHvdcs = new ArrayList<>();
                VoltageControlOverrides.VcSplit vcSplit = VoltageControlOverrides.classifyVcSplit(es, q.islandedBuses(), lostVcControllers, deadControlBuses,
                        islandedHvdcs);
                ControlReHoming.ReHomed reHomed = ControlReHoming.reHomedControls(q.islandedBuses(), c.outagedZiBranches(), c.contingencyId(), es);
                if (reHomed.reject() != null) {
                    refuse(c, "PLAN:" + reHomed.reject());
                    continue;
                }
                String[] subReject = new String[1];
                pb.dqSubByIndex[i] = ReHomedEntries.subgroupPairs(reHomed, gpuData, subReject);
                if (subReject[0] != null) {
                    refuse(c, subReject[0]);
                    continue;
                }
                deadControlBuses.removeAll(reHomed.pvTargets().keySet());
                deadControlBuses.removeAll(reHomed.remoteClusterBuses());   // a surviving remote cluster, not dead
                deadControlBuses.removeAll(reHomed.reHomedRemoteControllers().keySet());   // re-pointed at the new pilot
                for (LfBus lb : reHomed.leftControllers()) {
                    if (!lostVcControllers.contains(lb)) {
                        lostVcControllers.add(lb);
                    }
                }
                if (!reHomed.leftControllers().isEmpty() && vcSplit == VoltageControlOverrides.VcSplit.CONTAINED) {
                    vcSplit = VoltageControlOverrides.VcSplit.RENORMALIZE;           // the group goes on without them
                }
                if (!islandedHvdcs.isEmpty()) {
                    pb.islandedHvdcsByIndex[i] = islandedHvdcs;
                }
                if (vcSplit == VoltageControlOverrides.VcSplit.UNMODELED) {
                    if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                        System.err.printf("QUALIFY_TRACE reject %s tag=ISLANDS_REMOTE_VC_MEMBER site=tail-unmodeled"
                                + " lostControllers=%s deadControl=%s%n", c.contingencyId(),
                                lostVcControllers.stream().map(LfBus::getId).toList(),
                                deadControlBuses.stream().map(LfBus::getId).toList());
                    }
                    refuse(c, "PLAN:ISLANDS_REMOTE_VC_MEMBER");
                    continue;                                // a DISTR_Q row with no reachable control — CPU
                }
                if (!deadControlBuses.isEmpty()) {
                    // Row-mode-only GenOutage entries (pRow < 0, no P change): each surviving member
                    // bus of the killed control solves its own reactive balance again, at its own
                    // target Q (LfBus.getTargetQ() = generation - load, which is precisely the target
                    // OLF's re-activated BUS_TARGET_Q carries).
                    List<GenOutage> vcOff = new ArrayList<>();
                    for (LfBus sb : deadControlBuses) {
                        var vVar = es.getVariableSet().getVariable(sb.getNum(), AcVariableType.BUS_V);
                        if (vVar == null || vVar.getRow() < 0) {
                            vcOff = null;
                            break;
                        }
                        vcOff.add(new GenOutage(-1, 0, RelocatedRows.relocQRow(vVar.getRow(), pb.relocRows, outagedZiNums),
                                sb.getTargetQ(), true, true,
                                -1, 0, 0, 0, 0, -1, 0, 0, false, java.util.Set.of(), 0, 0, 0, -1, new int[0],
                                new int[0], new double[0], RelocatedRows.deadControlRowMode(gpuData, sb), -1, 0, -1, 0, -1, 0, -1, -1, 0, false, -1, 0, 0));
                    }
                    if (vcOff == null) {
                        if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                            System.err.printf("QUALIFY_TRACE reject %s tag=ISLANDS_REMOTE_VC_MEMBER site=tail-deadControl-noVrow"
                                    + " deadControl=%s%n", c.contingencyId(), deadControlBuses.stream().map(LfBus::getId).toList());
                        }
                        refuse(c, "PLAN:ISLANDS_REMOTE_VC_MEMBER");
                        continue;                            // no V row to flip — CPU
                    }
                    pb.genOutagesByIndex[i] = vcOff.toArray(new GenOutage[0]);
                }
                List<LoadLoss> stationResets = ScheduledLoss.hvdcStationResets(q.islandedBuses(), c);
                if (!stationResets.isEmpty()) {              // a setpoint link losing its other end: the survivor's P
                    if (pb.busRows == null) {
                        pb.busRows = com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor
                                .busPowerRows(stationResets.get(0).bus().getNetwork(), es);
                    }
                    GenOutage[] st = GpuBatchInjectionOutages.computeInjectionOutages(es, gpuData, dsSum, slackParticipates,
                            dsParams, List.of(), stationResets, pb.busRows);
                    if (st == null) {
                        refuse(c, "PLAN:INJECTION_OUTAGE_SHAPE");
                        continue;
                    }
                    List<GenOutage> merged = new ArrayList<>();
                    if (pb.genOutagesByIndex[i] != null) {
                        merged.addAll(java.util.Arrays.asList(pb.genOutagesByIndex[i]));
                    }
                    merged.addAll(java.util.Arrays.asList(st));
                    pb.genOutagesByIndex[i] = merged.toArray(new GenOutage[0]);
                }
                if (!reHomed.pvTargets().isEmpty() || !reHomed.remoteMainTargets().isEmpty()) {
                    List<GenOutage> pvOn = new ArrayList<>();
                    if (pb.genOutagesByIndex[i] != null) {
                        pvOn.addAll(java.util.Arrays.asList(pb.genOutagesByIndex[i]));
                    }
                    boolean ok = true;
                    boolean relocIdentity = false;
                    boolean pinnedMain = false;
                    for (var e : reHomed.pvTargets().entrySet()) {
                        var vVar = es.getVariableSet().getVariable(e.getKey().getNum(), AcVariableType.BUS_V);
                        if (vVar == null || vVar.getRow() < 0) {
                            ok = false;
                            break;
                        }
                        int reHomedRl = RelocatedRows.rlEntryOf(gpuData, e.getKey());
                        int crossRow = RelocatedRows.relocQRow(vVar.getRow(), pb.relocRows, outagedZiNums);
                        if (crossRow != vVar.getRow()) {
                            pvOn.add(ReHomedEntries.reHomedCrossEntry(crossRow, e.getValue(), reHomedRl,
                                    RelocatedRows.basePinned(gpuData, reHomedRl), e.getKey().getTargetQ()));
                            continue;
                        }
                        // Same rule as the island site above: a bus PINNED at its reactive limit in the base
                        // cannot hold a voltage identity (OLF has it PQ). This site lacked the check, so a
                        // middle-section loss whose surviving section is base-pinned batched with the
                        // identity: rte6515 BUS-6157_BBS_1 (VL-6157_1, baseLimitType=1), BUS-6194_BBS_2
                        // (VL-6194_2, 2) and BUS-6229_BBS_1 (VL-6229_1, 2) — three device failures on
                        // systems OLF converges, replayed on the CPU.
                        if (RelocatedRows.basePinned(gpuData, reHomedRl)) {
                            pvOn.add(ReHomedEntries.reHomedPinnedPvEntry(vVar.getRow(), e.getKey().getTargetQ(), e.getValue(), reHomedRl));
                            continue;
                        }
                        pvOn.add(ReHomedEntries.reHomedPvEntry(vVar.getRow(), e.getValue(), reHomedRl));
                    }
                    if (!ok) {
                        refuse(c, 
                                pinnedMain ? "PLAN:VC_REMERGE_PINNED_MAIN"
                                        : relocIdentity ? "PLAN:VC_REMERGE_RELOC_VROW" : "PLAN:VC_REMERGE_NO_VROW");
                        continue;
                    }
                    if (!reHomed.remoteMainTargets().isEmpty()) {
                        String[] remoteReject = new String[1];
                        List<GenOutage> remoteIds = ReHomedEntries.remoteMainIdentityEntries(reHomed, es, gpuData, pb.relocRows,
                                outagedZiNums, java.util.Set.of(), remoteReject);
                        List<GenOutage> reHomedCtl = remoteIds == null ? null
                                : ReHomedEntries.reHomedRemoteControllerEntries(reHomed, es, gpuData, remoteReject);
                        if (remoteIds == null || reHomedCtl == null) {
                            refuse(c, remoteReject[0]);
                            continue;
                        }
                        pvOn.addAll(remoteIds);
                        pvOn.addAll(reHomedCtl);
                    }
                    pb.genOutagesByIndex[i] = pvOn.toArray(new GenOutage[0]);
                }
                if (vcSplit == VoltageControlOverrides.VcSplit.RENORMALIZE) {
                    // The group survives without its islanded controller(s). Renormalize its reactive
                    // keys over the survivors through the SAME channel a generator outage uses: the
                    // islanded bus's generators are gone from this synchronous component, which is
                    // what distrQWeightOverrides replays.
                    List<com.powsybl.openloadflow.network.LfGenerator> goneControllers = lostVcControllers.stream()
                            .flatMap(b -> b.getGenerators().stream())
                            .toList();
                    double[][] vcOv = ReHomedEntries.withoutSplitGroups(VoltageControlOverrides.distrQWeightOverrides(es, gpuData, goneControllers), pb.dqSubByIndex[i], gpuData);
                    if (vcOv == null) {
                        refuse(c, "PLAN:DISTR_Q_WEIGHT_REDIFF");
                        continue;
                    }
                    pb.dqOvContribByIndex[i] = java.util.Arrays.stream(vcOv[0]).mapToInt(d -> (int) d).toArray();
                    pb.dqOvWByIndex[i] = vcOv[1];
                    pb.dqKeyOvMemberByIndex[i] = java.util.Arrays.stream(vcOv[2]).mapToInt(d -> (int) d).toArray();
                    pb.dqKeyOvByIndex[i] = vcOv[3];
                    // and the survivors' DISTR_Q row TARGETS + the islanded members' inactive entries
                    List<GenOutage> dq = GpuBatchInjectionOutages.islandedDistrQEntries(gpuData, lostVcControllers, pb.relocDummyQRows, outagedZiNums,
                            GpuBatchInjectionOutages.labelsOf(pb.dqSubByIndex[i]));
                    if (!dq.isEmpty()) {
                        List<GenOutage> merged = new ArrayList<>();
                        if (pb.genOutagesByIndex[i] != null) {
                            merged.addAll(java.util.Arrays.asList(pb.genOutagesByIndex[i]));
                        }
                        merged.addAll(dq);
                        pb.genOutagesByIndex[i] = merged.toArray(new GenOutage[0]);
                    }
                }
                double[][] drOv = VoltageControlOverrides.distrRhoOverrides(es, gpuData, distrRhoControllers, c.outagedBranches(), incrementalTvc);
                if (drOv == null) {
                    if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
                        System.err.println("QUAL_REJECT drOv2 " + c.contingencyId());
                    }
                    refuse(c, "PLAN:DISTR_RHO_GROUP_OUTAGE2");
                    continue;                                // multi-member / no-survivor group outage — CPU
                }
                if (drOv[0].length > 0) {
                    pb.drOvContribByIndex[i] = java.util.Arrays.stream(drOv[0]).mapToInt(d -> (int) d).toArray();
                    pb.drOvCByIndex[i] = drOv[1];
                }
                IslandedBuses.FrozenIsland fi = IslandedBuses.islandedFrozen(es, q.islandedBuses(), relocatedCouplers);
                // A contingency that BOTH splits the network AND strands a distributed-slack participant.
                // OLF distributes over getSynchronousNetworks().getFirst(), so a stranded participant is
                // not a participant at all: its share must not be computed, and the factors of the ones
                // that remain must be renormalized WITHOUT it. Masking it out of the distribution loops
                // alone is not enough and was reverted once for exactly that reason — the slack keeps the
                // retained share it had while the stranded bus still counted, and the pass retires with
                // most of the mismatch undistributed. The device now takes the per-scenario participant
                // set and derives BOTH the shares and the stopping gate from it.
                pb.islandedDsByIndex[i] = IslandedBuses.islandedDsParticipants(fi.rows(), gpuData, q.islandedBuses());
                pb.islandedRlByIndex[i] = IslandedBuses.islandedRlControllers(fi.rows(), gpuData);
                if ("1".equals(System.getenv("OLF_GPU_DBG_BUSDUMP")) && fi.rows().length > 0) {
                    // Which reactive-limits ENTRIES the island masks, and which frozen rows are an
                    // rlVrow at all: an unmasked frozen controller row is a row the RL pass can pin,
                    // and a pin on a frozen row survives every later target override.
                    System.err.println("ISLANDED-RL '" + c.contingencyId() + "' frozenRows="
                            + java.util.Arrays.toString(fi.rows()) + " maskedEntries="
                            + java.util.Arrays.toString(pb.islandedRlByIndex[i]) + " rlVrowOfMasked="
                            + java.util.Arrays.stream(pb.islandedRlByIndex[i]).map(e -> gpuData.rlVrow()[e]).boxed().toList());
                }
                if ("1".equals(System.getenv("OLF_GPU_DBG_BUSDUMP")) && q.islandedBuses().size() > 0) {
                    // The count that must match OLF's participant drop: OLF re-runs
                    // filterParticipatingBuses on the surviving component, so if it loses N buses and
                    // this masks fewer, the device still distributes to a bus OLF has dropped.
                    System.err.println("ISLANDED-DS '" + c.contingencyId() + "' islandedBuses="
                            + q.islandedBuses().size() + " maskedParticipants=" + pb.islandedDsByIndex[i].length
                            + " participatingIslandedBuses=" + q.islandedBuses().stream()
                                    .filter(b -> b.isParticipating()).count());
                    // Per bus, so an unmatched participant is EXPLAINED rather than assumed harmless:
                    // the pack emits no row for a zero-factor bus, and masking one would be a no-op —
                    // but a nonzero-factor bus missing from the pack would be a real hole.
                    for (LfBus b : q.islandedBuses()) {
                        double maxP = b.getGenerators().stream()
                                .filter(com.powsybl.openloadflow.network.LfGenerator::isParticipating)
                                .mapToDouble(com.powsybl.openloadflow.network.LfGenerator::getMaxTargetP).sum();
                        // Everything the island still COUPLES to the main component through: an HVDC
                        // whose other station is live, a controller in a remote group, a controlled
                        // shunt. Each is a place a frozen 1<0 row can leak a spurious injection.
                        var vc = b.getGeneratorVoltageControl().orElse(null);
                        System.err.println("  ISLANDED-BUS " + b.getId() + " num=" + b.getNum()
                                + " participating=" + b.isParticipating()
                                + " participatingGens=" + b.getGenerators().stream()
                                        .filter(com.powsybl.openloadflow.network.LfGenerator::isParticipating).count()
                                + " ΣmaxTargetP=" + maxP
                                + " hvdcs=" + b.getHvdcs().size()
                                + " shunts=" + (b.getShunt().isPresent() ? 1 : 0)
                                + " controllerShunt=" + (b.getControllerShunt().isPresent() ? 1 : 0)
                                + " svcShunt=" + (b.getSvcShunt().isPresent() ? 1 : 0)
                                + " genVc=" + (vc != null)
                                + " genVcControlledBus=" + (vc != null ? vc.getControlledBus().getId() : "-")
                                + " genVcControllers=" + (vc != null ? vc.getMergedControllerElements().size() : 0)
                                + " loads=" + b.getLoads().size());
                    }
                }
                pb.batchedIndices.add(i);
                pb.islandedNumsByIndex[i] = q.islandedBuses().stream().mapToInt(LfBus::getNum).toArray();
                if ("1".equals(System.getenv("OLF_GPU_DBG_BUSDUMP"))) {
                    System.err.println("FROZEN-ISLAND '" + c.contingencyId().replace(GpuBatchStatus.STRATEGY_KEY_SEP, '|') + "' buses="
                            + q.islandedBuses().stream().map(b -> b.getId() + "#" + b.getNum()).toList()
                            + " rows=" + java.util.Arrays.toString(fi.rows()));
                }
                pb.frozenRowsByIndex[i] = fi.rows();
                pb.frozenTargetsByIndex[i] = fi.targets();
                pb.activePowerLossByIndex[i] = ScheduledLoss.scheduledActivePowerLoss(c, q.islandedBuses());
                ScheduledLoss.traceScheduledLoss(c, q.islandedBuses(), pb.activePowerLossByIndex[i]);
                if (pb.frozenRowsByIndex[i].length > 0) {
                    pb.islandingCount++;
                }
            }
        }
        return pb.build();
    }
}
