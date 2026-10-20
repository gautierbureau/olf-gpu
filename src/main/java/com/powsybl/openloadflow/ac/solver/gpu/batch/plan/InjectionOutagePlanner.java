/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch.plan;

import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcElementMapper;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDsParams;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * The plan of a contingency that loses injections or changes the topology without islanding by branch outage alone:
 * lost generators and loads, shunt shifts, tap overrides, recloses, coupler and HVDC outages (with or without branch
 * outages beside them). It builds the per-bus generator-outage entries, the voltage-control overrides the losses
 * imply (DISTR_RHO, re-homed controls, DISTR_Q weights and keys) and the islanded rows, or refuses the contingency to
 * the CPU when a shape is outside what the device models.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class InjectionOutagePlanner {

    private InjectionOutagePlanner() {
    }

    /** Plan contingency {@code i}: on success its entries are in {@code pb} and it is batched; a refusal leaves it to
     *  the CPU. {@code outagedZiNums} are the numbers of the couplers it opens. */
    static void plan(PlanContext ctx, ScenarioPlanBuilder pb, int i, BatchContingency c,
                     GpuBatchContingencyValidator.Qualification q, java.util.Set<Integer> outagedZiNums) {
        final Set<LfBranch> distrRhoControllers = ctx.distrRhoControllers();
        final GpuDsParams dsParams = ctx.dsParams();
        final double dsSum = ctx.dsSum();
        final GpuAcElementMapper elementMapper = ctx.elementMapper();
        final EquationSystem<AcVariableType, AcEquationType> es = ctx.es();
        final GpuAcData gpuData = ctx.gpuData();
        final boolean incrementalTvc = ctx.incrementalTvc();
        final Set<LfBranch> relocatedCouplers = ctx.relocatedCouplers();
        final boolean slackParticipates = ctx.slackParticipates();
        java.util.Set<Integer> baseOpenSet = java.util.Arrays.stream(gpuData.baseOpenBranches().closedBranchElements())
                .boxed().collect(java.util.stream.Collectors.toSet());
        java.util.Map<Integer, Integer> ziElemByBranchNum = gpuData.baseOpenBranches().zeroImpedanceElementByBranchNum();
        boolean recloseOk = true;
        int ziCloses = 0;
        for (LfBranch rb : c.reclosedBranches()) {
            if (rb.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC)) {
                // switch close: packed base-open zi element, exactly one per scenario
                // (multi-switch closes can interact through the spanning tree — CPU)
                if (!ziElemByBranchNum.containsKey(rb.getNum()) || ++ziCloses > 1) {
                    recloseOk = false;
                }
            } else if (!baseOpenSet.contains(elementMapper.getElementIndex(rb))) {
                recloseOk = false;
            }
        }
        if (!recloseOk) {
            GpuBatchQualification.refuse(c, "PLAN:RECLOSE_UNPACKED_BASE_OPEN");
            return;                                // reclose of an unpacked base-open branch — CPU
        }
        boolean tapOk = true;
        for (TapR1Override tap : c.tapOverrides()) { // packed CLOSED branch with FIXED ratio AND
            int ei = elementMapper.getElementIndex(tap.branch());   // phase (neither is a state variable)
            if (ei < 0 || ei >= elementMapper.getClosedBranchCount()
                    || gpuData.closedBranch().indices()[ei * 12 + 9] >= 0 || gpuData.closedBranch().indices()[ei * 12 + 8] >= 0) {
                tapOk = false;
            }
        }
        if (!tapOk) {
            GpuBatchQualification.refuse(c, "PLAN:CONTROLLED_OR_UNPACKED_TAP");
            return;                                // controlled/unpacked transformer tap — CPU
        }
        java.util.Set<Integer> islandedNumSet = new java.util.HashSet<>();
        for (LfBus ib : q.islandedBuses()) {
            islandedNumSet.add(ib.getNum());
        }
        if (!c.shuntShifts().isEmpty()
                && !GpuBatchQualification.canBatchShuntOutage(c.shuntShifts().stream().map(ShuntShift::shunt).toList(),
                                                              elementMapper, gpuData)) {
            GpuBatchQualification.refuse(c, "PLAN:SHUNT_UNPACKED");
            return;                                // unpacked shunt (controller / SVC) — CPU
        }
        // A lost generator whose bus the contingency ISLANDS is not an injection outage: the
        // island's rows are frozen whole and its group effect is classifyVcSplit's verdict
        // (CONTROL_DIES / RENORMALIZE below). Handing it to the composer made a DISTR_Q member
        // "lose its last controller" and rejected the scenario (rte6515: 17 of the 500 busbar
        // sections, VL-1014_0, VL-1113_0, ... - every one a section OLF simply disables).
        List<com.powsybl.openloadflow.network.LfGenerator> liveLostGens = c.lostGenerators().stream()
                .filter(g -> g.getBus() == null || !islandedNumSet.contains(g.getBus().getNum()))
                .toList();
        List<LoadLoss> liveLostLoads = new ArrayList<>(c.lostLoads().stream()
                .filter(ll -> ll.bus() == null || !islandedNumSet.contains(ll.bus().getNum()))
                .toList());
        liveLostLoads.addAll(ScheduledLoss.hvdcStationResets(q.islandedBuses(), c));   // a setpoint link losing its other end
        if (pb.busRows == null) {                       // once per plan: the coupler-tree relocation
            LfBus anyBus = !liveLostGens.isEmpty() ? liveLostGens.get(0).getBus()
                    : liveLostLoads.isEmpty() ? null : liveLostLoads.get(0).bus();
            if (anyBus != null) {                    // stays null when nothing is lost here — and then
                pb.busRows = com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor
                        .busPowerRows(anyBus.getNetwork(), es);   // no bus is looked up either
            }
        }
        GenOutage[] gos = GpuBatchInjectionOutages.computeInjectionOutages(es, gpuData, dsSum, slackParticipates,
                dsParams, liveLostGens, liveLostLoads, pb.busRows);
        if (gos == null) {                           // slack / >1 DS bus / remote gen — route to CPU
            // (was tagged SHUNT_SHAPE, a copy-paste from the block above: this is the INJECTION
            // outage shape, and the census is how we pick what to fix — a tag that lies about
            // the reason sends the next person at the wrong gate.)
            GpuBatchQualification.refuse(c, "PLAN:INJECTION_OUTAGE_SHAPE");
            if (System.getenv("OLF_GPU_GENOUT_REJECT") != null) {
                // The exit named itself and its bus a moment ago; name the CONTINGENCY too, so
                // the reason, the bus and the id that reproduces it are on adjacent lines.
                System.err.println("GENOUT_REJECT   ctg=" + c.contingencyId());
            }
            return;
        }
        List<LfBus> islandLostVcControllers = new ArrayList<>();   // filled by the island-injection path
        if (!c.outagedBranches().isEmpty()) {        // COMBO: the branch part must also qualify
            double[][] drOv = VoltageControlOverrides.distrRhoOverrides(es, gpuData, distrRhoControllers,
                    c.outagedBranches(), incrementalTvc);
            if (drOv == null) {
                if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
                    System.err.println("QUAL_REJECT drOv " + c.contingencyId());
                }
                GpuBatchQualification.refuse(c, "PLAN:DISTR_RHO_GROUP_OUTAGE");
                return;                            // multi-member / no-survivor group outage — CPU
            }
            if (drOv[0].length > 0) {
                pb.drOvContribByIndex[i] = java.util.Arrays.stream(drOv[0]).mapToInt(d -> (int) d).toArray();
                pb.drOvCByIndex[i] = drOv[1];
            }
        }
        // A coupler opened WITHOUT islanding anything still splits a merged voltage control when it separates
        // the control's sections: OLF re-merges per surviving zero-impedance cluster, and a separated section
        // with a local control becomes a PV bus at its own target. The island tail below plans that re-homing
        // (with an empty island nothing is frozen or masked). Without it the entries stayed the base group's:
        // a member pinned in the base and released after the opening went back to the GROUP's sharing row and
        // put the group's voltage identity back on the pinned main's section, an equation system OLF does not
        // have (the split-group fixture: the device diverged where OLF converges in one release).
        boolean splitsMergedControl = q.islandedBuses().isEmpty() && !c.outagedZiBranches().isEmpty()
                && ControlReHoming.reHomedControls(List.of(), c.outagedZiBranches(), c.contingencyId(),
                                                   es).changesControls();
        gos = planIslandTail(q, splitsMergedControl, es, c, pb, i, gpuData, islandLostVcControllers,
                             relocatedCouplers, gos, outagedZiNums);
        if (gos == null) {
            return;
        }
        // Reactive-key rediff over EVERY controller this scenario removes from its group: the
        // lost generators the composer saw AND the controllers the island took (RENORMALIZE,
        // the tail's channel). One call, so the two compose instead of one excluding the other.
        List<com.powsybl.openloadflow.network.LfGenerator> goneForKeys = new ArrayList<>(c.lostGenerators());
        for (LfBus lb : islandLostVcControllers) {
            goneForKeys.addAll(lb.getGenerators());
        }
        if (goneForKeys.stream().anyMatch(g -> g.getBus() != null
                && es.getEquation(g.getBus().getNum(), AcEquationType.DISTR_Q).isPresent())) {
            double[][] ov = ReHomedEntries.withoutSplitGroups(VoltageControlOverrides.distrQWeightOverrides(es, gpuData, goneForKeys),
                                                              pb.dqSubByIndex[i], gpuData);
            if (ov == null) {
                GpuBatchQualification.refuse(c, "PLAN:DISTR_Q_WEIGHT_REDIFF");
                return;                            // weight rediff unavailable — CPU
            }
            pb.dqOvContribByIndex[i] = java.util.Arrays.stream(ov[0]).mapToInt(d -> (int) d).toArray();
            pb.dqOvWByIndex[i] = ov[1];
            pb.dqKeyOvMemberByIndex[i] = java.util.Arrays.stream(ov[2]).mapToInt(d -> (int) d).toArray();
            pb.dqKeyOvByIndex[i] = ov[3];
            String dqOvTrace = System.getenv("OLF_GPU_DQOV_TRACE");   // <contingency id>: every reweighted entry
            if (dqOvTrace != null && dqOvTrace.equals(c.contingencyId())) {
                int[] mBus = gpuData.distrQMembers().busNum();
                for (int k = 0; k < ov[0].length; k++) {
                    int cc = (int) ov[0][k];
                    int kM = gpuData.distrQ().rowMember() != null ? gpuData.distrQ().rowMember()[cc] : -1;
                    int jM = gpuData.distrQ().injectionMember() != null ? gpuData.distrQ().injectionMember()[cc] : -1;
                    System.err.printf("DQOV_TRACE ctg=%s c=%d row=%d kind=%d elem=%d oldW=%.6f newW=%.6f kMember=%s jMember=%s%n",
                            c.contingencyId(), cc, gpuData.distrQ().row()[cc], gpuData.distrQ().kind()[cc], gpuData.distrQ().element()[cc],
                            gpuData.distrQ().weight()[cc], ov[1][k],
                            kM >= 0 && mBus != null ? goneForKeys.get(0).getBus().getNetwork().getBus(mBus[kM]).getId() : "?",
                            jM >= 0 && mBus != null ? goneForKeys.get(0).getBus().getNetwork().getBus(mBus[jM]).getId() : "?");
                }
                System.err.println("DQOV_TRACE ctg=" + c.contingencyId() + " total=" + ov[0].length
                        + " lostGens=" + goneForKeys.stream().map(g -> g.getId()).toList());
            }
        }
        if (pb.relocatedSlackRow[i] >= 0 && gos != null) {
            int reference = pb.relocatedSlackRow[i];
            if (java.util.Arrays.stream(gos).anyMatch(go -> go.pRow() == reference)) {
                GpuBatchQualification.refuse(c, "PLAN:SLACK_RELOCATION_SHAPE");   // an injection lost AT the new slack: unmeasured
                return;
            }
        }
        pb.genOutagesByIndex[i] = gos;
        pb.activePowerLossByIndex[i] = ScheduledLoss.scheduledActivePowerLoss(c, q.islandedBuses());
        ScheduledLoss.traceScheduledLoss(c, q.islandedBuses(), pb.activePowerLossByIndex[i]);
        pb.batchedIndices.add(i);
    }

    /** The island and split-control tail of an injection or topology-change outage: the islanded rows frozen, the
     *  voltage controls the island cuts or re-homes, and the participants it strands; false when a shape refuses
     *  the contingency. */
    private static GenOutage[] planIslandTail(final GpuBatchContingencyValidator.Qualification q,
                                              final boolean splitsMergedControl,
                                              final EquationSystem<AcVariableType, AcEquationType> es,
                                              final BatchContingency c, final ScenarioPlanBuilder pb, final int i,
                                              final GpuAcData gpuData, final List<LfBus> islandLostVcControllers,
                                              final Set<LfBranch> relocatedCouplers, final GenOutage[] gosIn,
                                              final java.util.Set<Integer> outagedZiNums) {
        GenOutage[] gos = gosIn;
        if (!q.islandedBuses().isEmpty() || splitsMergedControl) {
            // An outage that ISLANDS buses while it also removes injections: a busbar section
            // whose loss is expressed as a switch outage plus a side-open (busIdsToLose is
            // EMPTY - the bus is lost by connectivity alone), and - since 2026-09-22 - a
            // BRANCH outage combined with a generator on the bus it islands. This tail used to
            // drop the island on the floor: no frozen rows, no control-death check, the
            // islanded pilot's voltage identity left live and its group renormalised as if
            // alive. OLF disables the islanded bus and, when it was a group's pilot, kills the
            // control and hands every surviving member its own reactive balance. Do the same
            // here, with the machinery the branch-outage tail already has.
            //
            // The combo used to demand a FULLY CONTAINED island ("a SECOND rediff from an
            // islanded controller would have to compose with the generator outage's own"):
            // they DO compose - the key rediff below takes one list holding both - and the
            // island's own generators never reach the composer (liveLostGens drops them), so
            // there is no second rediff to compose with. 65 of network.xiidm's 11249 branch
            // contingencies were CPU-replayed on that gate, every one of them a line whose
            // loss islands one generator bus.
            List<LfBus> lostVcControllers = new ArrayList<>();
            List<LfBus> deadControlBuses = new ArrayList<>();
            List<com.powsybl.openloadflow.network.LfHvdc> islandedHvdcs = new ArrayList<>();
            VoltageControlOverrides.VcSplit vcSplit = VoltageControlOverrides.classifyVcSplit(es, q.islandedBuses(), lostVcControllers, deadControlBuses,
                    islandedHvdcs);
            ControlReHoming.ReHomed reHomed = ControlReHoming.reHomedControls(q.islandedBuses(),
                    c.outagedZiBranches(), c.contingencyId(), es);
            if (reHomed.reject() != null) {
                GpuBatchQualification.refuse(c, "PLAN:" + reHomed.reject());
                return null;
            }
            String[] subReject = new String[1];
            pb.dqSubByIndex[i] = ReHomedEntries.subgroupPairs(reHomed, gpuData, subReject);
            if (subReject[0] != null) {
                GpuBatchQualification.refuse(c, subReject[0]);
                return null;
            }
            deadControlBuses.removeAll(reHomed.pvTargets().keySet());
            deadControlBuses.removeAll(reHomed.remoteClusterBuses());   // a surviving remote cluster, not dead   // re-homed, not dead
            deadControlBuses.removeAll(reHomed.reHomedRemoteControllers().keySet());   // re-pointed at the new pilot
            for (LfBus lb : reHomed.leftControllers()) {
                if (!lostVcControllers.contains(lb)) {
                    lostVcControllers.add(lb);
                }
            }
            islandLostVcControllers.addAll(lostVcControllers);
            if (!islandedHvdcs.isEmpty()) {
                pb.islandedHvdcsByIndex[i] = islandedHvdcs;   // masked in the chunk solver, as on the tail
            }
            if (vcSplit == VoltageControlOverrides.VcSplit.UNMODELED) {
                if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                    System.err.printf("QUALIFY_TRACE reject %s tag=ISLANDS_REMOTE_VC_MEMBER site=island split=%s"
                            + " islandedHvdcs=%d lostControllers=%s%n", c.contingencyId(), vcSplit, islandedHvdcs.size(),
                            lostVcControllers.stream().map(LfBus::getId).toList());
                }
                GpuBatchQualification.refuse(c, "PLAN:ISLANDS_REMOTE_VC_MEMBER");
                return null;
            }
            IslandedBuses.FrozenIsland fi = IslandedBuses.islandedFrozen(es, q.islandedBuses(), relocatedCouplers);
            pb.islandedNumsByIndex[i] = q.islandedBuses().stream().mapToInt(LfBus::getNum).toArray();
            java.util.Set<Integer> frozenSet = java.util.Arrays.stream(fi.rows()).boxed()
                    .collect(java.util.stream.Collectors.toSet());
            // A generator outage ON the island is moot: the bus's rows are frozen, its scheduled
            // active power is already excluded from the loss (scheduledActivePowerLoss skips
            // islanded buses), and its group effect is the control death handled below. Keep
            // only the entries that act on live rows.
            List<GenOutage> live = new ArrayList<>();
            for (GenOutage go : gos) {
                if (!frozenSet.contains(go.pRow()) && !frozenSet.contains(go.vRow())) {
                    live.add(go);
                }
            }
            if (!deadControlBuses.isEmpty()) {
                for (LfBus sb : deadControlBuses) {
                    var vVar = es.getVariableSet().getVariable(sb.getNum(), AcVariableType.BUS_V);
                    if (vVar == null || vVar.getRow() < 0 || frozenSet.contains(vVar.getRow())) {
                        live = null;
                        break;
                    }
                    int qFlipRow = RelocatedRows.relocQRow(vVar.getRow(), pb.relocRows, outagedZiNums);
                    if (frozenSet.contains(qFlipRow)) {
                        live = null;
                        break;
                    }
                    live.add(new GenOutage(-1, 0, qFlipRow, sb.getTargetQ(), true, true,
                            -1, 0, 0, 0, 0, -1, 0, 0, false, java.util.Set.of(), 0, 0, 0, -1, new int[0],
                            new int[0], new double[0], RelocatedRows.deadControlRowMode(gpuData,
                                    sb), -1, 0, -1, 0, -1, 0, -1, -1, 0, false, -1, 0, 0));
                }
                if (live == null) {
                    if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                        System.err.printf("QUALIFY_TRACE reject %s tag=ISLANDS_REMOTE_VC_MEMBER site=island-deadControl-frozenV"
                                + " deadControl=%s%n", c.contingencyId(), deadControlBuses.stream().map(LfBus::getId).toList());
                    }
                    GpuBatchQualification.refuse(c, "PLAN:ISLANDS_REMOTE_VC_MEMBER");
                    return null;
                }
            }
            boolean relocIdentity = false;
            boolean pinnedMain = false;
            if (live != null) {
                for (var e : reHomed.pvTargets().entrySet()) {
                    var vVar = es.getVariableSet().getVariable(e.getKey().getNum(), AcVariableType.BUS_V);
                    if (vVar == null || vVar.getRow() < 0 || frozenSet.contains(vVar.getRow())) {
                        live = null;
                        break;
                    }
                    int reHomedRl = RelocatedRows.rlEntryOf(gpuData, e.getKey());
                    int crossRow = RelocatedRows.relocQRow(vVar.getRow(), pb.relocRows, outagedZiNums);
                    if (crossRow != vVar.getRow()) {
                        // its own row carries a LIVE coupler's ZERO_V: cross identity on the parent's
                        // DUMMY_Q row (rte6515 BUS-6014_BBS_1/2 VL-6014_3, BUS-6029_BBS_4 VL-6029_2)
                        live.add(ReHomedEntries.reHomedCrossEntry(crossRow, e.getValue(), reHomedRl,
                                gpuData.switchedPvToPqInBase(reHomedRl), e.getKey().getTargetQ()));
                        continue;
                    }
                    if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                        // Is the bus we are about to give a VOLTAGE IDENTITY already PINNED at its
                        // reactive limit? A pinned bus cannot hold its voltage — OLF has it as PQ —
                        // and the RL kernel's "limit drift while pinned" branch writes only the
                        // TARGET, leaving the host's mode 0 in place, so the identity then holds v
                        // at a reactive value.
                        byte baseLt = gpuData.reactiveLimits().baseQLimitType() != null && reHomedRl >= 0
                                ? gpuData.reactiveLimits().baseQLimitType()[reHomedRl] : (byte) -1;
                        System.err.printf("QUALIFY_TRACE reHomePv bus=%s vRow=%d target=%.6f"
                                        + " rlEntry=%d baseLimitType=%d%n",
                                e.getKey().getId(), vVar.getRow(), e.getValue(), reHomedRl, baseLt);
                    }
                    // A bus PINNED at its reactive limit cannot also hold its voltage: OLF has it
                    // as PQ, and giving it a V identity here is a state the device cannot represent.
                    // The RL kernel then never takes the switch path that would set mode 4 — the
                    // entry is already pinned, so it takes the "limit drift while pinned" branch,
                    // which writes only the TARGET. The host's mode 0 stays, and a mode-0 identity
                    // means x = target, so the bus's VOLTAGE becomes its reactive limit. Measured on
                    // rte6515 BUS-6194_BBS_2: VL-6194_2 re-homed to a 1.123 pu identity with
                    // baseLimitType=2, row 12938 uploaded at 1.123 and ended at 0.6251 (= maxQ −
                    // loadQ), outside the realistic band [0.8, 1.2] — the run reported FAILED where
                    // OLF reports CONVERGED, losing all 71 of its violations, the WHOLE violation
                    // gap of the full busbar scope.
                    if (gpuData.switchedPvToPqInBase(reHomedRl)) {
                        // Base-pinned: start as the reactive balance at the pinned value, checked as a
                        // LOCAL PV bus at the re-homed voltage target (ROW_PINNED_V). rte6515
                        // BUS-6157_BBS_1 (unpins, re-pins MAX_Q), BUS-6194_BBS_2 (unpins), BUS-6229_BBS_1.
                        live.add(ReHomedEntries.reHomedPinnedPvEntry(vVar.getRow(), e.getKey().getTargetQ(), e.getValue(), reHomedRl));
                        continue;
                    }
                    live.add(ReHomedEntries.reHomedPvEntry(vVar.getRow(), e.getValue(), reHomedRl));
                }
            }
            String[] remoteReject = new String[1];
            if (live != null && !reHomed.remoteMainTargets().isEmpty()) {
                List<GenOutage> remoteIds = ReHomedEntries.remoteMainIdentityEntries(reHomed, es, gpuData, pb.relocRows,
                        outagedZiNums, frozenSet, remoteReject);
                if (remoteIds == null) {
                    live = null;
                } else {
                    live.addAll(remoteIds);
                    List<GenOutage> reHomedCtl = ReHomedEntries.reHomedRemoteControllerEntries(reHomed, es, gpuData,
                            remoteReject);
                    if (reHomedCtl == null) {
                        live = null;
                    } else {
                        live.addAll(reHomedCtl);
                    }
                }
            }
            if (live == null && remoteReject[0] != null) {
                GpuBatchQualification.refuse(c, remoteReject[0]);
                return null;
            }
            if (live == null) {
                GpuBatchQualification.refuse(c,
                        pinnedMain ? "PLAN:VC_REMERGE_PINNED_MAIN"
                                : relocIdentity ? "PLAN:VC_REMERGE_RELOC_VROW" : "PLAN:VC_REMERGE_NO_VROW");
                return null;
            }
            if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                System.err.printf("QUALIFY_TRACE island-outage %s: split=%s frozenRows=%d dropped=%d"
                        + " deadControl=%s%n", c.contingencyId(), vcSplit, fi.rows().length,
                        gos.length - (live.size() - deadControlBuses.size()),
                        deadControlBuses.stream().map(LfBus::getId).toList());
            }
            // RENORMALIZE: the islanded controllers' keys leave the group (weights below,
            // through the rediff); their DISTR_Q row targets and inactive-member entries go
            // the way a generator outage's do.
            // The ISLANDED packed members count as inactive too, not only the controllers the split or
            // the re-homing listed: a survivor's renormalized target was otherwise written onto an islanded
            // member's own DISTR_Q row — a FROZEN identity row — as a second override, and the identity
            // then drove that bus's V from 1.0 to 0 in MAX_VOLTAGE_CHANGE-clamped steps (BUS-612_BBS_2:
            // three scaled iterations on the device against OLF's one, the distributed-slack decision
            // taken at another residual level, 0.141 MW of distributed active power). No entry may target
            // a frozen row at all: those rows belong to buses OLF removes from the system.
            List<LfBus> inactiveForTargets = new ArrayList<>(islandLostVcControllers);
            for (LfBus ib : q.islandedBuses()) {
                if (!inactiveForTargets.contains(ib) && GpuBatchInjectionOutages.dqMemberSlot(gpuData, ib) >= 0) {
                    inactiveForTargets.add(ib);
                }
            }
            for (GenOutage go : GpuBatchInjectionOutages.islandedDistrQEntries(gpuData, inactiveForTargets, pb.relocDummyQRows, outagedZiNums,
                    GpuBatchInjectionOutages.labelsOf(pb.dqSubByIndex[i]))) {
                live.add(go.withoutFrozenTargetRows(frozenSet));
            }
            gos = live.toArray(new GenOutage[0]);
            pb.frozenRowsByIndex[i] = fi.rows();
            pb.frozenTargetsByIndex[i] = fi.targets();
            // The island's distributed-slack participants and reactive-limit controllers are
            // NOT participants for this scenario (OLF distributes over the main synchronous
            // component and disables the island's buses). The branch-outage tail masks them;
            // this path froze the rows but left the masks unset, so the water-fill read each
            // frozen P row's identity target (an ANGLE) as absorbed power, reset it to the
            // base P and handed the island shares of the mismatch that never reached the grid.
            // Measured on rte6515 BUS-1238_BBS: absorbed -0.305 pu where OLF pre-distributed
            // +0.155, a second DS round OLF never runs (+0.471 pu reappearing at the slack)
            // and 1.18e-03 kV at the end.
            pb.islandedDsByIndex[i] = IslandedBuses.islandedDsParticipants(fi.rows(), gpuData, q.islandedBuses());
            pb.islandedRlByIndex[i] = IslandedBuses.islandedRlControllers(fi.rows(), gpuData);
            if (fi.rows().length > 0) {
                pb.islandingCount++;
            }
        } else {
            pb.frozenRowsByIndex[i] = IslandedBuses.NO_FROZEN_ROWS;
            pb.frozenTargetsByIndex[i] = IslandedBuses.NO_FROZEN_TARGETS;
        }
        return gos;
    }
}
