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
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfShunt;

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
public final class GpuBatchQualification {

    private GpuBatchQualification() {
    }

    /**
     * Whether a fixed-shunt disconnection can be batched as a plain element mask. Requires every outaged shunt
     * to be a packed fixed shunt ({@code getElementIndex >= 0}) and NOT a reactive-distribution (DISTR_Q)
     * contributor — a shunt whose reactive feeds a controlled bus's DISTR_Q sum is read unconditionally by the
     * batched DISTR_Q fill (it ignores the disabled mask), so masking it would leave the sum stale. Such a shunt
     * outage is routed to the CPU instead.
     */
    static boolean canBatchShuntOutage(List<LfShunt> shunts, GpuAcElementMapper elementMapper,
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
        GpuBatchScopeCensus.record(tag);
        GpuBatchScopeCensus.planRefused(c.contingencyId(), tag);
    }

    /** The batched-scenario plan: which contingencies qualified ({@code batchedIndices}) and, per
     *  contingency, the frozen islanded-bus rows/targets, the islanded LfBus nums (excluded from result
     *  building), the merged per-bus generator outages (else null), and the count of on-GPU frozen islands. */
    public record ScenarioPlan(List<Integer> batchedIndices, int[][] frozenRowsByIndex,
                        int[][] islandedNumsByIndex, double[][] frozenTargetsByIndex,
                        GenOutage[][] genOutagesByIndex, int islandingCount,
                        double[] activePowerLossByIndex,
                        int[][] distrQWeightOverrideTermByIndex, double[][] distrQWeightOverrideWeightByIndex,
                        /** Per scenario, the members whose reactive KEY the contingency changes (a generator lost at a
                         *  bus that keeps a controller) and OLF's post-loss key; the kernels renormalize them live. */
                        int[][] distrQKeyOverrideMemberByIndex, double[][] distrQKeyOverrideKeyByIndex,
                        int[][] distrRhoOverrideTermByIndex, double[][] distrRhoOverrideCoefficientByIndex,
                        /** Per scenario, the distributed-slack participants this contingency strands in
                         *  another synchronous component — the set the device must renormalize over. */
                        int[][] islandedDistributedSlackByIndex,
                        /** Per scenario, the reactive-limits controllers stranded in another component. */
                        int[][] islandedReactiveLimitEntriesByIndex,
                        /** Per scenario, AC-emulation HVDCs with an ISLANDED endpoint. OLF's
                         *  processHvdcsWithoutPower zeroes both converter stations' targetP; in AC
                         *  emulation those targets are already 0 (the injection is an equation term,
                         *  not a target), so masking the element is the whole of it. */
                        List<com.powsybl.openloadflow.network.LfHvdc>[] islandedHvdcsByIndex,
                        /** Per scenario, {dq member, subgroup label} pairs of a split group (tasks #71), else null. */
                        int[][] distrQSubgroupByIndex,
                        /** Per scenario, the slack OLF re-picks when the outage isolates the base one. */
                        SlackRelocations slackRelocations) {

        /** This plan with only {@code indices} batched. */
        public ScenarioPlan withBatchedIndices(List<Integer> indices) {
            return new ScenarioPlan(indices, frozenRowsByIndex, islandedNumsByIndex, frozenTargetsByIndex, genOutagesByIndex,
                    islandingCount, activePowerLossByIndex, distrQWeightOverrideTermByIndex, distrQWeightOverrideWeightByIndex,
                    distrQKeyOverrideMemberByIndex, distrQKeyOverrideKeyByIndex, distrRhoOverrideTermByIndex,
                    distrRhoOverrideCoefficientByIndex, islandedDistributedSlackByIndex, islandedReactiveLimitEntriesByIndex,
                    islandedHvdcsByIndex, distrQSubgroupByIndex, slackRelocations);
        }
    }

    /**
     * The plan with the scenarios AreaInterchangeControl cannot balance on the device left out (each one recorded as a
     * plan refusal, so it runs on the CPU). OLF's security analysis distributes a contingency's active power loss over the
     * whole synchronous network before the post-contingency load flow, resetting every participant to its initial target,
     * which the area phase does not replay; a contingency that loses an injection or an HVDC, leaves an island (its buses
     * leave their areas and its boundaries leave the interchanges) or moves the slack, and a strategy, keep that to OLF.
     * {@code tablesBuilt} false (a boundary the device does not model): every scenario.
     */
    public static ScenarioPlan refuseAreaInterchangeControlShapes(ScenarioPlan plan, List<BatchContingency> contingencies, boolean tablesBuilt) {
        List<Integer> kept = new java.util.ArrayList<>();
        for (int i : plan.batchedIndices()) {
            BatchContingency c = contingencies.get(i);
            String tag = null;
            if (!tablesBuilt) {
                tag = "PLAN:AREA_INTERCHANGE_CONTROL_BOUNDARY_UNMODELED";
            } else if (c.contingencyId().indexOf(BatchContingency.STRATEGY_KEY_SEP) >= 0) {
                tag = "PLAN:AREA_INTERCHANGE_CONTROL_STRATEGY";
            } else if (!c.lostGenerators().isEmpty() || !c.lostLoads().isEmpty() || !c.outagedHvdcs().isEmpty()
                    || plan.activePowerLossByIndex()[i] != 0) {
                tag = "PLAN:AREA_INTERCHANGE_CONTROL_ACTIVE_POWER_LOSS";
            } else if (plan.islandedNumsByIndex()[i] != null && plan.islandedNumsByIndex()[i].length > 0) {
                tag = "PLAN:AREA_INTERCHANGE_CONTROL_ISLAND";
            } else if (plan.slackRelocations() != null && plan.slackRelocations().row(i) >= 0) {
                tag = "PLAN:AREA_INTERCHANGE_CONTROL_SLACK_RELOCATION";
            }
            if (tag != null) {
                refuse(c, tag);
            } else {
                kept.add(i);
            }
        }
        return plan.withBatchedIndices(kept);
    }

    /** Per contingency index: the angle row of the bus OLF makes the slack when the outage isolates the base one
     *  (-1 = the base slack stays), which becomes the reference identity, and how far that bus's anchor target sits
     *  from the base slack's (the distributed-slack check reads the new slack's balance against it). */
    public record SlackRelocations(int[] rowByIndex, double[] shiftByIndex) {

        public int row(int index) {
            return rowByIndex[index];
        }
    }

    /**
     * Qualify each contingency for the batched GPU path and build the per-scenario plan. An outage that
     * keeps the main component intact batches as a value-only mask; a fragmenting one batches with its
     * islanded buses frozen to identity; injection/shift/tap/reclose/switch shapes are checked against the
     * packed pattern and the on-device retain math, and anything unrepresentable is left out (CPU fallback).
     */
    public static ScenarioPlan planScenarios(List<BatchContingency> contingencies,
            List<GpuBatchContingencyValidator.Qualification> qualifications,
            GpuAcElementMapper elementMapper, GpuAcData gpuData,
            EquationSystem<AcVariableType, AcEquationType> es,
            double distributedSlackRawFactorSum, boolean slackParticipates,
            com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDistributedSlackParameters distributedSlackParameters, Set<LfBranch> distrRhoControllers,
            Set<LfBranch> relocatedCouplers, boolean distributedSlack, boolean reactiveLimits, boolean incrementalTransformerVoltageControllers) {
        ScenarioPlanBuilder pb = new ScenarioPlanBuilder(contingencies.size(), gpuData);
        PlanContext ctx = new PlanContext(contingencies, qualifications, elementMapper, gpuData, es, distributedSlackRawFactorSum,
                slackParticipates, distributedSlackParameters, distrRhoControllers, relocatedCouplers, distributedSlack, reactiveLimits,
                incrementalTransformerVoltageControllers);
        for (int i = 0; i < qualifications.size(); i++) {
            GpuBatchContingencyValidator.Qualification q = qualifications.get(i);
            if (!q.qualified()) {
                // The validator's OWN reason, not a fixed tag: this site is reached by every disqualified
                // verdict, so recording "SIDE_OPEN_UNPACKED" for all of them misreports the census — the one
                // entry at rte6515's full scope is an ISLANDS_SLACK outage, which reads as a side-open shape.
                String tag = "PLAN:" + (q.reason() != null ? q.reason() : "UNQUALIFIED");
                GpuBatchScopeCensus.record(tag);
                // ...and by id, so a scope report names the reason of every refusal (it printed "?" for these)
                GpuBatchScopeCensus.planRefused(contingencies.get(i).contingencyId(), tag);
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
            int strategySep = c.contingencyId().indexOf(BatchContingency.STRATEGY_KEY_SEP);
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
            // with it. The curve and forced-target-Q channels track a generator's P only when the run distributes the
            // slack and it is in the distributed-slack generator pack (its anchor and its own move); any other curve
            // generator keeps its limits at the base target P - CPU.
            if (reactiveLimits && c.generatorActions().stream()
                    .anyMatch(m -> VoltageControlReMerge.hasCurve(m.generator())
                            && !(distributedSlack && VoltageControlReMerge.inGeneratorPack(gpuData, m.generator().getId())))) {
                refuse(c, "PLAN:PAIR_ACTION_UNTRACKED_CURVE");
                continue;
            }
            // ...and re-checks its participation at the new target, which the device applies to the generator pack.
            // A participant outside a non-empty pack (its bus has no active P balance row) has nowhere to sit out. (In
            // a slack-only distribution there is no pack, and any action there moves the slack bus: SLACK_BUS.) Without the
            // run's DistributedSlackOuterLoop nobody participates.
            if (distributedSlack && gpuData.distributedSlack().generatorId() != null && gpuData.distributedSlack().generatorId().length > 0 && c.generatorActions().stream()
                    .anyMatch(m -> distributedSlackParameters.factor(m.generator()) != 0 && !VoltageControlReMerge.inGeneratorPack(gpuData, m.generator().getId()))) {
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
                // the run's DistributedSlackOuterLoop, not the pack: the pack is there either way
                boolean participates = distributedSlack && (distributedSlackParameters.busFactor(baseSlack) != 0 || distributedSlackParameters.busFactor(newSlack) != 0);
                boolean slackOnly = distributedSlack && gpuData.distributedSlack().busPhiRow().length == 0;
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
                pb.relocatedSlackShift[i] = distributedSlack ? newSlack.getTargetP() - gpuData.distributedSlack().slackTargetP() : 0;
                GpuBatchScopeCensus.record("PLAN:SLACK_RELOCATED");
            }
            java.util.Set<Integer> outagedZeroImpedanceBranchNums = c.outagedZeroImpedanceBranches().stream()
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
                    || !c.outagedZeroImpedanceBranches().isEmpty()
                    || !c.outagedHvdcs().isEmpty()) {        // injection/shift/tap/reclose/switch/hvdc-outage (or branch+)
                InjectionOutagePlanner.plan(ctx, pb, i, c, q, outagedZeroImpedanceBranchNums);
            } else if (!c.outagedShunts().isEmpty()) {       // fixed-shunt disconnection: mask the shunt element
                if (canBatchShuntOutage(c.outagedShunts(), elementMapper, gpuData)) {
                    pb.batchedIndices.add(i);
                    pb.frozenRowsByIndex[i] = IslandedBuses.NO_FROZEN_ROWS;
                    pb.frozenTargetsByIndex[i] = IslandedBuses.NO_FROZEN_TARGETS;
                }                                            // else: not a packed fixed shunt / DISTR_Q contributor → CPU
            } else {
                BranchOutagePlanner.plan(ctx, pb, i, c, q, outagedZeroImpedanceBranchNums);
            }
        }
        return pb.build();
    }
}
