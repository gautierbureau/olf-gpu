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
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDistributedSlackParameters;
import com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix.CouplerTreeReHoming;
import com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix.CrossIdentity;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * The plan of a contingency that only takes branches out, islanding buses or not: the voltage controls the islands
 * cut or re-merge (DISTR_Q weights and keys, re-merged and pinned controllers, split groups), the distributed-slack
 * participants and reactive-limit controllers they strand, the islanded HVDCs, and the islanded rows frozen at a flat
 * state; or a refusal to the CPU when a shape is outside what the device models.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class BranchOutagePlanner {

    private BranchOutagePlanner() {
    }

    /** Plan contingency {@code i}: on success its entries are in {@code pb} and it is batched; a refusal leaves it to
     *  the CPU. {@code outagedZeroImpedanceBranchNums} are the numbers of the couplers it opens. */
    static void plan(PlanContext ctx, ScenarioPlanBuilder pb, int i, BatchContingency c,
                     GpuBatchContingencyValidator.Qualification q, java.util.Set<Integer> outagedZeroImpedanceBranchNums) {
        final Set<LfBranch> distrRhoControllers = ctx.distrRhoControllers();
        final GpuDistributedSlackParameters distributedSlackParameters = ctx.distributedSlackParameters();
        final double distributedSlackRawFactorSum = ctx.distributedSlackRawFactorSum();
        final EquationSystem<AcVariableType, AcEquationType> es = ctx.es();
        final GpuAcData gpuData = ctx.gpuData();
        final boolean incrementalTransformerVoltageControllers = ctx.incrementalTransformerVoltageControllers();
        final Set<LfBranch> relocatedCouplers = ctx.relocatedCouplers();
        final boolean slackParticipates = ctx.slackParticipates();
        List<LfBus> lostVcControllers = new ArrayList<>();
        List<LfBus> deadControlBuses = new ArrayList<>();
        List<com.powsybl.openloadflow.network.LfHvdc> islandedHvdcs = new ArrayList<>();
        VoltageControlOverrides.VcSplit vcSplit = VoltageControlOverrides.classifyVcSplit(es, q.islandedBuses(), lostVcControllers, deadControlBuses,
                islandedHvdcs);
        VoltageControlReMerge.ReMerged reMerged = VoltageControlReMerge.reMergedControls(q.islandedBuses(), c.outagedZeroImpedanceBranches(),
                c.contingencyId(), es);
        if (reMerged.reject() != null) {
            GpuBatchQualification.refuse(c, "PLAN:" + reMerged.reject());
            return;
        }
        String[] subReject = new String[1];
        pb.distrQSubgroupByIndex[i] = VoltageControlReMergeEntries.subgroupPairs(reMerged, gpuData, subReject);
        if (subReject[0] != null) {
            GpuBatchQualification.refuse(c, subReject[0]);
            return;
        }
        deadControlBuses.removeAll(reMerged.pvTargets().keySet());
        deadControlBuses.removeAll(reMerged.remoteClusterBuses());   // a surviving remote cluster, not dead
        deadControlBuses.removeAll(reMerged.reMergedRemoteControllers().keySet());   // re-pointed at the new pilot
        for (LfBus lb : reMerged.leftControllers()) {
            if (!lostVcControllers.contains(lb)) {
                lostVcControllers.add(lb);
            }
        }
        if (!reMerged.leftControllers().isEmpty() && vcSplit == VoltageControlOverrides.VcSplit.CONTAINED) {
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
            GpuBatchQualification.refuse(c, "PLAN:ISLANDS_REMOTE_VC_MEMBER");
            return;                                // a DISTR_Q row with no reachable control — CPU
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
                vcOff.add(new GenOutage(-1, 0, CouplerTreeReHoming.scenarioReactiveRowOfVRow(vVar.getRow(), pb.relocatedChildRows, outagedZeroImpedanceBranchNums),
                        sb.getTargetQ(), true, true,
                        -1, 0, 0, 0, 0, -1, 0, 0, false, java.util.Set.of(), 0, 0, 0, -1, new int[0],
                        new int[0], new double[0], GpuBatchInjectionOutages.deadControlRowMode(gpuData,
                                sb), -1, 0, -1, 0, -1, 0, -1, -1, 0, false, -1, 0, 0));
            }
            if (vcOff == null) {
                if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                    System.err.printf("QUALIFY_TRACE reject %s tag=ISLANDS_REMOTE_VC_MEMBER site=tail-deadControl-noVrow"
                            + " deadControl=%s%n", c.contingencyId(), deadControlBuses.stream().map(LfBus::getId).toList());
                }
                GpuBatchQualification.refuse(c, "PLAN:ISLANDS_REMOTE_VC_MEMBER");
                return;                            // no V row to flip — CPU
            }
            pb.genOutagesByIndex[i] = vcOff.toArray(new GenOutage[0]);
        }
        List<LoadLoss> stationResets = ScheduledLoss.hvdcStationResets(q.islandedBuses(), c);
        if (!stationResets.isEmpty()) {              // a setpoint link losing its other end: the survivor's P
            if (pb.busRows == null) {
                pb.busRows = com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor
                        .busPowerRows(stationResets.get(0).bus().getNetwork(), es);
            }
            GenOutage[] injectionOutages = GpuBatchInjectionOutages.computeInjectionOutages(es, gpuData, distributedSlackRawFactorSum, slackParticipates,
                    distributedSlackParameters, List.of(), stationResets, pb.busRows);
            if (injectionOutages == null) {
                GpuBatchQualification.refuse(c, "PLAN:INJECTION_OUTAGE_SHAPE");
                return;
            }
            List<GenOutage> merged = new ArrayList<>();
            if (pb.genOutagesByIndex[i] != null) {
                merged.addAll(java.util.Arrays.asList(pb.genOutagesByIndex[i]));
            }
            merged.addAll(java.util.Arrays.asList(injectionOutages));
            pb.genOutagesByIndex[i] = merged.toArray(new GenOutage[0]);
        }
        if (!planReMergedControls(reMerged, pb, i, es, gpuData, outagedZeroImpedanceBranchNums, c)) {
            return;
        }
        if (vcSplit == VoltageControlOverrides.VcSplit.RENORMALIZE) {
            // The group survives without its islanded controller(s). Renormalize its reactive
            // keys over the survivors through the SAME channel a generator outage uses: the
            // islanded bus's generators are gone from this synchronous component, which is
            // what distrQWeightOverrides replays.
            List<com.powsybl.openloadflow.network.LfGenerator> goneControllers = lostVcControllers.stream()
                    .flatMap(b -> b.getGenerators().stream())
                    .toList();
            double[][] vcOv = VoltageControlReMergeEntries.withoutSplitGroups(VoltageControlOverrides.distrQWeightOverrides(es, gpuData, goneControllers),
                                                                pb.distrQSubgroupByIndex[i], gpuData);
            if (vcOv == null) {
                GpuBatchQualification.refuse(c, "PLAN:DISTR_Q_WEIGHT_REDIFF");
                return;
            }
            pb.distrQWeightOverrideTermByIndex[i] = java.util.Arrays.stream(vcOv[0]).mapToInt(d -> (int) d).toArray();
            pb.distrQWeightOverrideWeightByIndex[i] = vcOv[1];
            pb.distrQKeyOverrideMemberByIndex[i] = java.util.Arrays.stream(vcOv[2]).mapToInt(d -> (int) d).toArray();
            pb.distrQKeyOverrideKeyByIndex[i] = vcOv[3];
            // and the survivors' DISTR_Q row TARGETS + the islanded members' inactive entries
            List<GenOutage> islandedDistrQEntries = GpuBatchInjectionOutages.islandedDistrQEntries(gpuData, lostVcControllers, pb.relocatedDummyQRows, outagedZeroImpedanceBranchNums,
                    GpuBatchInjectionOutages.labelsOf(pb.distrQSubgroupByIndex[i]));
            if (!islandedDistrQEntries.isEmpty()) {
                List<GenOutage> merged = new ArrayList<>();
                if (pb.genOutagesByIndex[i] != null) {
                    merged.addAll(java.util.Arrays.asList(pb.genOutagesByIndex[i]));
                }
                merged.addAll(islandedDistrQEntries);
                pb.genOutagesByIndex[i] = merged.toArray(new GenOutage[0]);
            }
        }
        double[][] distrRhoOverrideTable = VoltageControlOverrides.distrRhoOverrides(es, gpuData, distrRhoControllers,
                                                                    c.outagedBranches(), incrementalTransformerVoltageControllers);
        if (distrRhoOverrideTable == null) {
            if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
                System.err.println("QUAL_REJECT distrRhoOverrides (branch) " + c.contingencyId());
            }
            GpuBatchQualification.refuse(c, "PLAN:DISTR_RHO_GROUP_OUTAGE2");
            return;                                // multi-member / no-survivor group outage — CPU
        }
        if (distrRhoOverrideTable[0].length > 0) {
            pb.distrRhoOverrideTermByIndex[i] = java.util.Arrays.stream(distrRhoOverrideTable[0]).mapToInt(d -> (int) d).toArray();
            pb.distrRhoOverrideCoefficientByIndex[i] = distrRhoOverrideTable[1];
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
        pb.islandedDistributedSlackByIndex[i] = IslandedBuses.islandedDistributedSlackParticipants(fi.rows(), gpuData, q.islandedBuses());
        pb.islandedReactiveLimitEntriesByIndex[i] = IslandedBuses.islandedReactiveLimitControllers(fi.rows(), gpuData);
        if ("1".equals(System.getenv("OLF_GPU_DBG_BUSDUMP")) && fi.rows().length > 0) {
            // Which reactive-limits ENTRIES the island masks, and which frozen rows are an
            // reactiveLimitVoltageRow at all: an unmasked frozen controller row is a row the RL pass can pin,
            // and a pin on a frozen row survives every later target override.
            System.err.println("ISLANDED-RL '" + c.contingencyId() + "' frozenRows="
                    + java.util.Arrays.toString(fi.rows()) + " maskedEntries="
                    + java.util.Arrays.toString(pb.islandedReactiveLimitEntriesByIndex[i]) + " rlVrowOfMasked="
                    + java.util.Arrays.stream(pb.islandedReactiveLimitEntriesByIndex[i]).map(e -> gpuData.reactiveLimits().voltageRow()[e]).boxed().toList());
        }
        if ("1".equals(System.getenv("OLF_GPU_DBG_BUSDUMP")) && q.islandedBuses().size() > 0) {
            // The count that must match OLF's participant drop: OLF re-runs
            // filterParticipatingBuses on the surviving component, so if it loses N buses and
            // this masks fewer, the device still distributes to a bus OLF has dropped.
            System.err.println("ISLANDED-DS '" + c.contingencyId() + "' islandedBuses="
                    + q.islandedBuses().size() + " maskedParticipants=" + pb.islandedDistributedSlackByIndex[i].length
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
                var voltageControl = b.getGeneratorVoltageControl().orElse(null);
                System.err.println("  ISLANDED-BUS " + b.getId() + " num=" + b.getNum()
                        + " participating=" + b.isParticipating()
                        + " participatingGens=" + b.getGenerators().stream()
                                .filter(com.powsybl.openloadflow.network.LfGenerator::isParticipating).count()
                        + " ΣmaxTargetP=" + maxP
                        + " hvdcs=" + b.getHvdcs().size()
                        + " shunts=" + (b.getShunt().isPresent() ? 1 : 0)
                        + " controllerShunt=" + (b.getControllerShunt().isPresent() ? 1 : 0)
                        + " svcShunt=" + (b.getSvcShunt().isPresent() ? 1 : 0)
                        + " genVc=" + (voltageControl != null)
                        + " genVcControlledBus=" + (voltageControl != null ? voltageControl.getControlledBus().getId() : "-")
                        + " genVcControllers=" + (voltageControl != null ? voltageControl.getMergedControllerElements().size() : 0)
                        + " loads=" + b.getLoads().size());
            }
        }
        pb.batchedIndices.add(i);
        pb.islandedNumsByIndex[i] = q.islandedBuses().stream().mapToInt(LfBus::getNum).toArray();
        if ("1".equals(System.getenv("OLF_GPU_DBG_BUSDUMP"))) {
            System.err.println("FROZEN-ISLAND '" + c.contingencyId().replace(BatchContingency.STRATEGY_KEY_SEP, '|') + "' buses="
                    + q.islandedBuses().stream().map(b -> b.getId() + "#" + b.getNum()).toList()
                    + " rows=" + java.util.Arrays.toString(fi.rows()));
        }
        pb.frozenRowsByIndex[i] = fi.rows();
        pb.frozenTargetsByIndex[i] = fi.targets();
        pb.activePowerLossByIndex[i] = ScheduledLoss.scheduledActivePowerLoss(c, q.islandedBuses());
        if (pb.frozenRowsByIndex[i].length > 0) {
            pb.islandingCount++;
        }
    }

    /** The re-merged voltage controls of an islanding branch outage: the PV entries of the controllers re-merged by
     *  the island, the remote mains turned identities, and their reactive-limit entries; false when a shape refuses
     *  the contingency. */
    private static boolean planReMergedControls(final VoltageControlReMerge.ReMerged reMerged, final ScenarioPlanBuilder pb,
                                               final int i, final EquationSystem<AcVariableType, AcEquationType> es,
                                               final GpuAcData gpuData, final java.util.Set<Integer> outagedZeroImpedanceBranchNums,
                                               final BatchContingency c) {
        if (!reMerged.pvTargets().isEmpty() || !reMerged.remoteMainTargets().isEmpty()) {
            List<GenOutage> pvOn = new ArrayList<>();
            if (pb.genOutagesByIndex[i] != null) {
                pvOn.addAll(java.util.Arrays.asList(pb.genOutagesByIndex[i]));
            }
            boolean ok = true;
            boolean crossIdentity = false;
            boolean pinnedMain = false;
            for (var e : reMerged.pvTargets().entrySet()) {
                var vVar = es.getVariableSet().getVariable(e.getKey().getNum(), AcVariableType.BUS_V);
                if (vVar == null || vVar.getRow() < 0) {
                    ok = false;
                    break;
                }
                int reMergedReactiveLimitEntry = gpuData.reactiveLimits().entryOf(e.getKey().getNum());
                int crossRow = CrossIdentity.row(vVar.getRow(), pb.relocatedChildRows, outagedZeroImpedanceBranchNums);
                // Same rule as the island site above: a bus PINNED at its reactive limit in the base
                // cannot hold a voltage identity (OLF has it PQ). This site lacked the check, so a
                // middle-section loss whose surviving section is base-pinned batched with the
                // identity: rte6515 BUS-6157_BBS_1 (VL-6157_1, baseLimitType=1), BUS-6194_BBS_2
                // (VL-6194_2, 2) and BUS-6229_BBS_1 (VL-6229_1, 2) — three device failures on
                // systems OLF converges, replayed on the CPU.
                pvOn.add(VoltageControlReMergeEntries.pvEntry(vVar.getRow(), crossRow, e.getValue(), reMergedReactiveLimitEntry,
                        gpuData.switchedPvToPqInBase(reMergedReactiveLimitEntry), e.getKey().getTargetQ()));
            }
            if (!ok) {
                GpuBatchQualification.refuse(c,
                        pinnedMain ? "PLAN:VC_REMERGE_PINNED_MAIN"
                                : crossIdentity ? "PLAN:VC_REMERGE_RELOC_VROW" : "PLAN:VC_REMERGE_NO_VROW");
                return false;
            }
            if (!reMerged.remoteMainTargets().isEmpty()) {
                String[] remoteReject = new String[1];
                List<GenOutage> remoteIds = VoltageControlReMergeEntries.remoteMainIdentityEntries(reMerged, es, gpuData, pb.relocatedChildRows,
                        outagedZeroImpedanceBranchNums, java.util.Set.of(), remoteReject);
                List<GenOutage> reMergedController = remoteIds == null ? null
                        : VoltageControlReMergeEntries.reMergedRemoteControllerEntries(reMerged, es, gpuData, remoteReject);
                if (remoteIds == null || reMergedController == null) {
                    GpuBatchQualification.refuse(c, remoteReject[0]);
                    return false;
                }
                pvOn.addAll(remoteIds);
                pvOn.addAll(reMergedController);
            }
            pb.genOutagesByIndex[i] = pvOn.toArray(new GenOutage[0]);
        }
        return true;
    }
}
