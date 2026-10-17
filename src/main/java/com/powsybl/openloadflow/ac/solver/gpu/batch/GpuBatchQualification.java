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
    private static void refuse(BatchContingency c, String tag) {
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
        List<Integer> batchedIndices = new ArrayList<>();
        java.util.Map<Integer, int[]> relocRows = RelocatedRows.relocatedChildRows(gpuData);
        java.util.Map<Integer, int[]> relocDummyQRows = RelocatedRows.relocatedDummyQRows(gpuData);
        int[][] frozenRowsByIndex = new int[contingencies.size()][];     // islanded buses' variable rows, per contingency
        int[][] islandedNumsByIndex = new int[contingencies.size()][];   // islanded LfBus nums (result-building excludes them)
        double[][] frozenTargetsByIndex = new double[contingencies.size()][];   // the clean value each frozen row holds (V=1, phi=0)
        GenOutage[][] genOutagesByIndex = new GenOutage[contingencies.size()][];   // per-bus merged outages, else null
        // Per-contingency SCHEDULED active power loss (pu, CPU sign: lost generation - lost load,
        // LfContingency.getActivePowerLoss incl. islanded buses) — drives the host replay of OLF's
        // DefaultContingencyActivePowerLossDistribution (GpuBatchLossPreDistribution).
        double[] activePowerLossByIndex = new double[contingencies.size()];
        // Per-contingency DISTR_Q weight overrides: a gen outage at a remote-VC controller bus makes
        // the CPU renormalize the group's qPct keys (LfContingency.processLostGenerators ->
        // updateReactiveKeys); replay the toggle host-side and diff the extracted weights.
        @SuppressWarnings("unchecked")
        List<com.powsybl.openloadflow.network.LfHvdc>[] islandedHvdcsByIndex = new List[contingencies.size()];
        int[][] dqOvContribByIndex = new int[contingencies.size()][];
        double[][] dqOvWByIndex = new double[contingencies.size()][];
        int[][] dqKeyOvMemberByIndex = new int[contingencies.size()][];
        double[][] dqKeyOvByIndex = new double[contingencies.size()][];
        // Per-contingency DISTR_RHO coefficient overrides (plan A5): outaging a NON-FREE member of a
        // shared-TVC group renormalizes the survivors' 1/count coefficients (the CPU's coefficients
        // are live suppliers over !isDisabled) and zeroes every contribution touching the outaged
        // member's rho; the FREE member's outage shifts the pilot-Q role — CPU (A6).
        int[][] drOvContribByIndex = new int[contingencies.size()][];
        double[][] drOvCByIndex = new double[contingencies.size()][];
        int[][] islandedDsByIndex = new int[contingencies.size()][];
        int[][] islandedRlByIndex = new int[contingencies.size()][];
        // {pRow, qRow} per LfBus.getNum(), resolved lazily on the first injection outage: the P/Q
        // balance rows the extraction writes targets at, which the coupler-tree relocation moves off
        // the bus's own BUS_PHI/BUS_V columns. One O(buses+branches) pass for the whole plan.
        int[][] busRows = null;
        int[][] dqSubByIndex = new int[contingencies.size()][];
        int[] relocatedSlackRow = new int[contingencies.size()];
        java.util.Arrays.fill(relocatedSlackRow, -1);
        double[] relocatedSlackShift = new double[contingencies.size()];
        int islandingCount = 0;
        // per contingency id: the buses it islands, and the branches it opens itself
        java.util.Map<String, Set<Integer>> islandedByContingency = new java.util.HashMap<>();
        java.util.Map<String, List<LfBranch>> openedByContingency = new java.util.HashMap<>();
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
                islandedByContingency.put(c.contingencyId(), islandedHere);
                openedByContingency.put(c.contingencyId(), c.outagedBranches());
            } else {
                String parentId = c.contingencyId().substring(0, strategySep);
                Set<Integer> parentIslanded = islandedByContingency.getOrDefault(parentId, Set.of());
                Set<Integer> reclosedEnds = new java.util.HashSet<>();
                for (LfBranch b : openedByContingency.getOrDefault(parentId, List.of())) {
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
                if (busRows == null) {
                    busRows = com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor
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
                if (phiRow < 0 || !balanceActive || busRows[0][newSlack.getNum()] != phiRow
                        || participates || slackOnly || emulatedHvdc) {
                    refuse(c, "PLAN:SLACK_RELOCATION_SHAPE");
                    continue;
                }
                relocatedSlackRow[i] = phiRow;
                relocatedSlackShift[i] = ds ? newSlack.getTargetP() - gpuData.dsSlackTargetP() : 0;
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
                java.util.Set<Integer> baseOpenSet = java.util.Arrays.stream(gpuData.baseOpenCbElements())
                        .boxed().collect(java.util.stream.Collectors.toSet());
                java.util.Map<Integer, Integer> ziElemByBranchNum = new java.util.HashMap<>();
                for (int z = 0; z < gpuData.baseOpenZiElements().length; z++) {
                    ziElemByBranchNum.put(gpuData.baseOpenZiBranchNums()[z], gpuData.baseOpenZiElements()[z]);
                }
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
                    refuse(c, "PLAN:RECLOSE_UNPACKED_BASE_OPEN");
                    continue;                                // reclose of an unpacked base-open branch — CPU
                }
                boolean tapOk = true;
                for (TapR1Override tap : c.tapOverrides()) { // packed CLOSED branch with FIXED ratio AND
                    int ei = elementMapper.getElementIndex(tap.branch());   // phase (neither is a state variable)
                    if (ei < 0 || ei >= elementMapper.getClosedBranchCount()
                            || gpuData.cbIdx()[ei * 12 + 9] >= 0 || gpuData.cbIdx()[ei * 12 + 8] >= 0) {
                        tapOk = false;
                    }
                }
                if (!tapOk) {
                    refuse(c, "PLAN:CONTROLLED_OR_UNPACKED_TAP");
                    continue;                                // controlled/unpacked transformer tap — CPU
                }
                java.util.Set<Integer> islandedNumSet = new java.util.HashSet<>();
                for (LfBus ib : q.islandedBuses()) {
                    islandedNumSet.add(ib.getNum());
                }
                if (!c.shuntShifts().isEmpty()
                        && !canBatchShuntOutage(c.shuntShifts().stream().map(ShuntShift::shunt).toList(), elementMapper, gpuData)) {
                    refuse(c, "PLAN:SHUNT_UNPACKED");
                    continue;                                // unpacked shunt (controller / SVC) — CPU
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
                if (busRows == null) {                       // once per plan: the coupler-tree relocation
                    LfBus anyBus = !liveLostGens.isEmpty() ? liveLostGens.get(0).getBus()
                            : liveLostLoads.isEmpty() ? null : liveLostLoads.get(0).bus();
                    if (anyBus != null) {                    // stays null when nothing is lost here — and then
                        busRows = com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor
                                .busPowerRows(anyBus.getNetwork(), es);   // no bus is looked up either
                    }
                }
                GenOutage[] gos = GpuBatchInjectionOutages.computeInjectionOutages(es, gpuData, dsSum, slackParticipates,
                        dsParams, liveLostGens, liveLostLoads, busRows);
                if (gos == null) {                           // slack / >1 DS bus / remote gen — route to CPU
                    // (was tagged SHUNT_SHAPE, a copy-paste from the block above: this is the INJECTION
                    // outage shape, and the census is how we pick what to fix — a tag that lies about
                    // the reason sends the next person at the wrong gate.)
                    refuse(c, "PLAN:INJECTION_OUTAGE_SHAPE");
                    if (System.getenv("OLF_GPU_GENOUT_REJECT") != null) {
                        // The exit named itself and its bus a moment ago; name the CONTINGENCY too, so
                        // the reason, the bus and the id that reproduces it are on adjacent lines.
                        System.err.println("GENOUT_REJECT   ctg=" + c.contingencyId());
                    }
                    continue;
                }
                List<LfBus> islandLostVcControllers = new ArrayList<>();   // filled by the island-injection path
                if (!c.outagedBranches().isEmpty()) {        // COMBO: the branch part must also qualify
                    double[][] drOv = VoltageControlOverrides.distrRhoOverrides(es, gpuData, distrRhoControllers, c.outagedBranches(), incrementalTvc);
                    if (drOv == null) {
                        if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
                            System.err.println("QUAL_REJECT drOv " + c.contingencyId());
                        }
                        refuse(c, "PLAN:DISTR_RHO_GROUP_OUTAGE");
                        continue;                            // multi-member / no-survivor group outage — CPU
                    }
                    if (drOv[0].length > 0) {
                        drOvContribByIndex[i] = java.util.Arrays.stream(drOv[0]).mapToInt(d -> (int) d).toArray();
                        drOvCByIndex[i] = drOv[1];
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
                        && ControlReHoming.reHomedControls(List.of(), c.outagedZiBranches(), c.contingencyId(), es).changesControls();
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
                    ControlReHoming.ReHomed reHomed = ControlReHoming.reHomedControls(q.islandedBuses(), c.outagedZiBranches(), c.contingencyId(), es);
                    if (reHomed.reject() != null) {
                        refuse(c, "PLAN:" + reHomed.reject());
                        continue;
                    }
                    String[] subReject = new String[1];
                    dqSubByIndex[i] = ReHomedEntries.subgroupPairs(reHomed, gpuData, subReject);
                    if (subReject[0] != null) {
                        refuse(c, subReject[0]);
                        continue;
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
                        islandedHvdcsByIndex[i] = islandedHvdcs;   // masked in the chunk solver, as on the tail
                    }
                    if (vcSplit == VoltageControlOverrides.VcSplit.UNMODELED) {
                        if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                            System.err.printf("QUALIFY_TRACE reject %s tag=ISLANDS_REMOTE_VC_MEMBER site=island split=%s"
                                    + " islandedHvdcs=%d lostControllers=%s%n", c.contingencyId(), vcSplit, islandedHvdcs.size(),
                                    lostVcControllers.stream().map(LfBus::getId).toList());
                        }
                        refuse(c, "PLAN:ISLANDS_REMOTE_VC_MEMBER");
                        continue;
                    }
                    IslandedBuses.FrozenIsland fi = IslandedBuses.islandedFrozen(es, q.islandedBuses(), relocatedCouplers);
                    islandedNumsByIndex[i] = q.islandedBuses().stream().mapToInt(LfBus::getNum).toArray();
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
                            int qFlipRow = RelocatedRows.relocQRow(vVar.getRow(), relocRows, outagedZiNums);
                            if (frozenSet.contains(qFlipRow)) {
                                live = null;
                                break;
                            }
                            live.add(new GenOutage(-1, 0, qFlipRow, sb.getTargetQ(), true, true,
                                    -1, 0, 0, 0, 0, -1, 0, 0, false, java.util.Set.of(), 0, 0, 0, -1, new int[0],
                                    new int[0], new double[0], RelocatedRows.deadControlRowMode(gpuData, sb), -1, 0, -1, 0, -1, 0, -1, -1, 0, false, -1, 0, 0));
                        }
                        if (live == null) {
                            if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                                System.err.printf("QUALIFY_TRACE reject %s tag=ISLANDS_REMOTE_VC_MEMBER site=island-deadControl-frozenV"
                                        + " deadControl=%s%n", c.contingencyId(), deadControlBuses.stream().map(LfBus::getId).toList());
                            }
                            refuse(c, "PLAN:ISLANDS_REMOTE_VC_MEMBER");
                            continue;
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
                            int crossRow = RelocatedRows.relocQRow(vVar.getRow(), relocRows, outagedZiNums);
                            if (crossRow != vVar.getRow()) {
                                // its own row carries a LIVE coupler's ZERO_V: cross identity on the parent's
                                // DUMMY_Q row (rte6515 BUS-6014_BBS_1/2 VL-6014_3, BUS-6029_BBS_4 VL-6029_2)
                                live.add(ReHomedEntries.reHomedCrossEntry(crossRow, e.getValue(), reHomedRl,
                                        RelocatedRows.basePinned(gpuData, reHomedRl), e.getKey().getTargetQ()));
                                continue;
                            }
                            if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                                // Is the bus we are about to give a VOLTAGE IDENTITY already PINNED at its
                                // reactive limit? A pinned bus cannot hold its voltage — OLF has it as PQ —
                                // and the RL kernel's "limit drift while pinned" branch writes only the
                                // TARGET, leaving the host's mode 0 in place, so the identity then holds v
                                // at a reactive value.
                                byte baseLt = gpuData.rlBaseQLimitType() != null && reHomedRl >= 0
                                        ? gpuData.rlBaseQLimitType()[reHomedRl] : (byte) -1;
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
                            if (RelocatedRows.basePinned(gpuData, reHomedRl)) {
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
                        List<GenOutage> remoteIds = ReHomedEntries.remoteMainIdentityEntries(reHomed, es, gpuData, relocRows,
                                outagedZiNums, frozenSet, remoteReject);
                        if (remoteIds == null) {
                            live = null;
                        } else {
                            live.addAll(remoteIds);
                            List<GenOutage> reHomedCtl = ReHomedEntries.reHomedRemoteControllerEntries(reHomed, es, gpuData, remoteReject);
                            if (reHomedCtl == null) {
                                live = null;
                            } else {
                                live.addAll(reHomedCtl);
                            }
                        }
                    }
                    if (live == null && remoteReject[0] != null) {
                        refuse(c, remoteReject[0]);
                        continue;
                    }
                    if (live == null) {
                        refuse(c, 
                                pinnedMain ? "PLAN:VC_REMERGE_PINNED_MAIN"
                                        : relocIdentity ? "PLAN:VC_REMERGE_RELOC_VROW" : "PLAN:VC_REMERGE_NO_VROW");
                        continue;
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
                    for (GenOutage go : GpuBatchInjectionOutages.islandedDistrQEntries(gpuData, inactiveForTargets, relocDummyQRows, outagedZiNums,
                            GpuBatchInjectionOutages.labelsOf(dqSubByIndex[i]))) {
                        live.add(go.withoutFrozenTargetRows(frozenSet));
                    }
                    gos = live.toArray(new GenOutage[0]);
                    frozenRowsByIndex[i] = fi.rows();
                    frozenTargetsByIndex[i] = fi.targets();
                    // The island's distributed-slack participants and reactive-limit controllers are
                    // NOT participants for this scenario (OLF distributes over the main synchronous
                    // component and disables the island's buses). The branch-outage tail masks them;
                    // this path froze the rows but left the masks unset, so the water-fill read each
                    // frozen P row's identity target (an ANGLE) as absorbed power, reset it to the
                    // base P and handed the island shares of the mismatch that never reached the grid.
                    // Measured on rte6515 BUS-1238_BBS: absorbed -0.305 pu where OLF pre-distributed
                    // +0.155, a second DS round OLF never runs (+0.471 pu reappearing at the slack)
                    // and 1.18e-03 kV at the end.
                    islandedDsByIndex[i] = IslandedBuses.islandedDsParticipants(fi.rows(), gpuData, q.islandedBuses());
                    islandedRlByIndex[i] = IslandedBuses.islandedRlControllers(fi.rows(), gpuData);
                    if (fi.rows().length > 0) {
                        islandingCount++;
                    }
                } else {
                    frozenRowsByIndex[i] = IslandedBuses.NO_FROZEN_ROWS;
                    frozenTargetsByIndex[i] = IslandedBuses.NO_FROZEN_TARGETS;
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
                    double[][] ov = ReHomedEntries.withoutSplitGroups(VoltageControlOverrides.distrQWeightOverrides(es, gpuData, goneForKeys), dqSubByIndex[i], gpuData);
                    if (ov == null) {
                        refuse(c, "PLAN:DISTR_Q_WEIGHT_REDIFF");
                        continue;                            // weight rediff unavailable — CPU
                    }
                    dqOvContribByIndex[i] = java.util.Arrays.stream(ov[0]).mapToInt(d -> (int) d).toArray();
                    dqOvWByIndex[i] = ov[1];
                    dqKeyOvMemberByIndex[i] = java.util.Arrays.stream(ov[2]).mapToInt(d -> (int) d).toArray();
                    dqKeyOvByIndex[i] = ov[3];
                    String dqOvTrace = System.getenv("OLF_GPU_DQOV_TRACE");   // <contingency id>: every reweighted entry
                    if (dqOvTrace != null && dqOvTrace.equals(c.contingencyId())) {
                        int[] mBus = gpuData.dqMemberBusNum();
                        for (int k = 0; k < ov[0].length; k++) {
                            int cc = (int) ov[0][k];
                            int kM = gpuData.dqKMember() != null ? gpuData.dqKMember()[cc] : -1;
                            int jM = gpuData.dqJMember() != null ? gpuData.dqJMember()[cc] : -1;
                            System.err.printf("DQOV_TRACE ctg=%s c=%d row=%d kind=%d elem=%d oldW=%.6f newW=%.6f kMember=%s jMember=%s%n",
                                    c.contingencyId(), cc, gpuData.dqRow()[cc], gpuData.dqKind()[cc], gpuData.dqElem()[cc],
                                    gpuData.dqWeight()[cc], ov[1][k],
                                    kM >= 0 && mBus != null ? goneForKeys.get(0).getBus().getNetwork().getBus(mBus[kM]).getId() : "?",
                                    jM >= 0 && mBus != null ? goneForKeys.get(0).getBus().getNetwork().getBus(mBus[jM]).getId() : "?");
                        }
                        System.err.println("DQOV_TRACE ctg=" + c.contingencyId() + " total=" + ov[0].length
                                + " lostGens=" + goneForKeys.stream().map(g -> g.getId()).toList());
                    }
                }
                if (relocatedSlackRow[i] >= 0 && gos != null) {
                    int reference = relocatedSlackRow[i];
                    if (java.util.Arrays.stream(gos).anyMatch(go -> go.pRow() == reference)) {
                        refuse(c, "PLAN:SLACK_RELOCATION_SHAPE");   // an injection lost AT the new slack: unmeasured
                        continue;
                    }
                }
                genOutagesByIndex[i] = gos;
                activePowerLossByIndex[i] = ScheduledLoss.scheduledActivePowerLoss(c, q.islandedBuses());
                ScheduledLoss.traceScheduledLoss(c, q.islandedBuses(), activePowerLossByIndex[i]);
                batchedIndices.add(i);
            } else if (!c.outagedShunts().isEmpty()) {       // fixed-shunt disconnection: mask the shunt element
                if (canBatchShuntOutage(c.outagedShunts(), elementMapper, gpuData)) {
                    batchedIndices.add(i);
                    frozenRowsByIndex[i] = IslandedBuses.NO_FROZEN_ROWS;
                    frozenTargetsByIndex[i] = IslandedBuses.NO_FROZEN_TARGETS;
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
                dqSubByIndex[i] = ReHomedEntries.subgroupPairs(reHomed, gpuData, subReject);
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
                    islandedHvdcsByIndex[i] = islandedHvdcs;
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
                        vcOff.add(new GenOutage(-1, 0, RelocatedRows.relocQRow(vVar.getRow(), relocRows, outagedZiNums),
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
                    genOutagesByIndex[i] = vcOff.toArray(new GenOutage[0]);
                }
                List<LoadLoss> stationResets = ScheduledLoss.hvdcStationResets(q.islandedBuses(), c);
                if (!stationResets.isEmpty()) {              // a setpoint link losing its other end: the survivor's P
                    if (busRows == null) {
                        busRows = com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor
                                .busPowerRows(stationResets.get(0).bus().getNetwork(), es);
                    }
                    GenOutage[] st = GpuBatchInjectionOutages.computeInjectionOutages(es, gpuData, dsSum, slackParticipates,
                            dsParams, List.of(), stationResets, busRows);
                    if (st == null) {
                        refuse(c, "PLAN:INJECTION_OUTAGE_SHAPE");
                        continue;
                    }
                    List<GenOutage> merged = new ArrayList<>();
                    if (genOutagesByIndex[i] != null) {
                        merged.addAll(java.util.Arrays.asList(genOutagesByIndex[i]));
                    }
                    merged.addAll(java.util.Arrays.asList(st));
                    genOutagesByIndex[i] = merged.toArray(new GenOutage[0]);
                }
                if (!reHomed.pvTargets().isEmpty() || !reHomed.remoteMainTargets().isEmpty()) {
                    List<GenOutage> pvOn = new ArrayList<>();
                    if (genOutagesByIndex[i] != null) {
                        pvOn.addAll(java.util.Arrays.asList(genOutagesByIndex[i]));
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
                        int crossRow = RelocatedRows.relocQRow(vVar.getRow(), relocRows, outagedZiNums);
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
                        List<GenOutage> remoteIds = ReHomedEntries.remoteMainIdentityEntries(reHomed, es, gpuData, relocRows,
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
                    genOutagesByIndex[i] = pvOn.toArray(new GenOutage[0]);
                }
                if (vcSplit == VoltageControlOverrides.VcSplit.RENORMALIZE) {
                    // The group survives without its islanded controller(s). Renormalize its reactive
                    // keys over the survivors through the SAME channel a generator outage uses: the
                    // islanded bus's generators are gone from this synchronous component, which is
                    // what distrQWeightOverrides replays.
                    List<com.powsybl.openloadflow.network.LfGenerator> goneControllers = lostVcControllers.stream()
                            .flatMap(b -> b.getGenerators().stream())
                            .toList();
                    double[][] vcOv = ReHomedEntries.withoutSplitGroups(VoltageControlOverrides.distrQWeightOverrides(es, gpuData, goneControllers), dqSubByIndex[i], gpuData);
                    if (vcOv == null) {
                        refuse(c, "PLAN:DISTR_Q_WEIGHT_REDIFF");
                        continue;
                    }
                    dqOvContribByIndex[i] = java.util.Arrays.stream(vcOv[0]).mapToInt(d -> (int) d).toArray();
                    dqOvWByIndex[i] = vcOv[1];
                    dqKeyOvMemberByIndex[i] = java.util.Arrays.stream(vcOv[2]).mapToInt(d -> (int) d).toArray();
                    dqKeyOvByIndex[i] = vcOv[3];
                    // and the survivors' DISTR_Q row TARGETS + the islanded members' inactive entries
                    List<GenOutage> dq = GpuBatchInjectionOutages.islandedDistrQEntries(gpuData, lostVcControllers, relocDummyQRows, outagedZiNums,
                            GpuBatchInjectionOutages.labelsOf(dqSubByIndex[i]));
                    if (!dq.isEmpty()) {
                        List<GenOutage> merged = new ArrayList<>();
                        if (genOutagesByIndex[i] != null) {
                            merged.addAll(java.util.Arrays.asList(genOutagesByIndex[i]));
                        }
                        merged.addAll(dq);
                        genOutagesByIndex[i] = merged.toArray(new GenOutage[0]);
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
                    drOvContribByIndex[i] = java.util.Arrays.stream(drOv[0]).mapToInt(d -> (int) d).toArray();
                    drOvCByIndex[i] = drOv[1];
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
                islandedDsByIndex[i] = IslandedBuses.islandedDsParticipants(fi.rows(), gpuData, q.islandedBuses());
                islandedRlByIndex[i] = IslandedBuses.islandedRlControllers(fi.rows(), gpuData);
                if ("1".equals(System.getenv("OLF_GPU_DBG_BUSDUMP")) && fi.rows().length > 0) {
                    // Which reactive-limits ENTRIES the island masks, and which frozen rows are an
                    // rlVrow at all: an unmasked frozen controller row is a row the RL pass can pin,
                    // and a pin on a frozen row survives every later target override.
                    System.err.println("ISLANDED-RL '" + c.contingencyId() + "' frozenRows="
                            + java.util.Arrays.toString(fi.rows()) + " maskedEntries="
                            + java.util.Arrays.toString(islandedRlByIndex[i]) + " rlVrowOfMasked="
                            + java.util.Arrays.stream(islandedRlByIndex[i]).map(e -> gpuData.rlVrow()[e]).boxed().toList());
                }
                if ("1".equals(System.getenv("OLF_GPU_DBG_BUSDUMP")) && q.islandedBuses().size() > 0) {
                    // The count that must match OLF's participant drop: OLF re-runs
                    // filterParticipatingBuses on the surviving component, so if it loses N buses and
                    // this masks fewer, the device still distributes to a bus OLF has dropped.
                    System.err.println("ISLANDED-DS '" + c.contingencyId() + "' islandedBuses="
                            + q.islandedBuses().size() + " maskedParticipants=" + islandedDsByIndex[i].length
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
                batchedIndices.add(i);
                islandedNumsByIndex[i] = q.islandedBuses().stream().mapToInt(LfBus::getNum).toArray();
                if ("1".equals(System.getenv("OLF_GPU_DBG_BUSDUMP"))) {
                    System.err.println("FROZEN-ISLAND '" + c.contingencyId().replace(GpuBatchStatus.STRATEGY_KEY_SEP, '|') + "' buses="
                            + q.islandedBuses().stream().map(b -> b.getId() + "#" + b.getNum()).toList()
                            + " rows=" + java.util.Arrays.toString(fi.rows()));
                }
                frozenRowsByIndex[i] = fi.rows();
                frozenTargetsByIndex[i] = fi.targets();
                activePowerLossByIndex[i] = ScheduledLoss.scheduledActivePowerLoss(c, q.islandedBuses());
                ScheduledLoss.traceScheduledLoss(c, q.islandedBuses(), activePowerLossByIndex[i]);
                if (frozenRowsByIndex[i].length > 0) {
                    islandingCount++;
                }
            }
        }
        int maskedDs = 0;
        for (int[] m : islandedDsByIndex) {
            maskedDs += m == null ? 0 : m.length;
        }
        com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.LAST_ISLANDED_DS_MASKED.set(maskedDs);
        return new ScenarioPlan(batchedIndices, frozenRowsByIndex, islandedNumsByIndex,
                frozenTargetsByIndex, genOutagesByIndex, islandingCount, activePowerLossByIndex,
                dqOvContribByIndex, dqOvWByIndex, dqKeyOvMemberByIndex, dqKeyOvByIndex, drOvContribByIndex, drOvCByIndex, islandedDsByIndex,
                islandedRlByIndex, islandedHvdcsByIndex, dqSubByIndex,
                new SlackRelocations(relocatedSlackRow, relocatedSlackShift));
    }
}
