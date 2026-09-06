/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

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

    static final int[] NO_FROZEN_ROWS = new int[0];
    static final double[] NO_FROZEN_TARGETS = new double[0];

    /** The islanded buses' variable rows to freeze + the clean flat value each holds (phi=0, V=1 pu). */
    record FrozenIsland(int[] rows, double[] targets) { }

    /**
     * The variable rows (BUS_PHI + BUS_V) of every islanded bus — frozen to identity in a fragmenting
     * scenario's row mode — paired with the clean flat value each is HELD at: {@code phi=0}, {@code V=1} pu.
     * With the outaged branch (the only island↔main link) already masked, freezing these rows makes the
     * islanded sub-block a benign identity block that cannot make the batched factorization singular, while
     * the main component solves correctly. Holding the island at a flat 1∠0 (rather than the base equation
     * target, a P/Q value that produces a nonsensical islanded voltage/angle) gives the disconnected buses a
     * well-defined reported state — no garbage voltage, no spurious limit violations — without touching the main.
     */
    static FrozenIsland islandedFrozen(EquationSystem<AcVariableType, AcEquationType> es, List<LfBus> islandedBuses) {
        if (islandedBuses.isEmpty()) {
            return new FrozenIsland(NO_FROZEN_ROWS, NO_FROZEN_TARGETS);
        }
        List<Integer> rows = new ArrayList<>(islandedBuses.size() * 2);
        List<Double> targets = new ArrayList<>(islandedBuses.size() * 2);
        for (LfBus bus : islandedBuses) {
            addFrozenVarRow(es, bus.getNum(), AcVariableType.BUS_PHI, 0.0, rows, targets);   // angle 0
            addFrozenVarRow(es, bus.getNum(), AcVariableType.BUS_V, 1.0, rows, targets);     // voltage 1 pu
        }
        int[] outRows = new int[rows.size()];
        double[] outTargets = new double[targets.size()];
        for (int i = 0; i < outRows.length; i++) {
            outRows[i] = rows.get(i);
            outTargets[i] = targets.get(i);
        }
        return new FrozenIsland(outRows, outTargets);
    }

    private static void addFrozenVarRow(EquationSystem<AcVariableType, AcEquationType> es, int busNum,
                                        AcVariableType type, double target, List<Integer> rows, List<Double> targets) {
        Variable<AcVariableType> v = es.getVariableSet().getVariable(busNum, type);
        if (v != null && v.getRow() >= 0) {
            rows.add(v.getRow());
            targets.add(target);
        }
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
        int shOffset = elementMapper.getClosedBranchCount() + elementMapper.getOpenBranchCount();
        int[] dqElem = gpuData.dqElem();
        int[] dqKind = gpuData.dqKind();
        for (LfShunt shunt : shunts) {
            int elementIndex = elementMapper.getElementIndex(shunt);
            if (elementIndex < 0) {
                return false;                                // not a packed fixed shunt (controller/SVC shunt)
            }
            int packIndex = elementIndex - shOffset;
            for (int i = 0; i < dqKind.length; i++) {
                if (dqKind[i] == 1 && dqElem[i] == packIndex) {
                    return false;                            // shunt feeds a DISTR_Q sum — mask wouldn't update it
                }
            }
        }
        return true;
    }

    /** True when the islanded set touches a REMOTE / DISTRIBUTED (DISTR_Q) generator-VC group —
     *  pilot bus or any controller (incl. the free controller carrying the pilot's Q row, which has
     *  no DISTR_Q equation of its own). Islanding a member reforms the group's qPct weights / row
     *  assignment, which the fixed batch pattern cannot represent (measured 0.47 kV wrong on the
     *  tr1 outage of createWithGeneratorRemoteControl) — route to CPU, exactly like the DISTR_RHO
     *  controller-outage rejection. */
    private static boolean islandsRemoteVcMember(EquationSystem<AcVariableType, AcEquationType> es,
                                                 List<LfBus> islandedBuses) {
        for (LfBus bus : islandedBuses) {
            if (!bus.getHvdcs().isEmpty()) {
                return true;                                 // islanded HVDC endpoint: the CPU zeroes the pair's
            }                                                // station targets (hvdcsWithoutPower) — unmodeled, CPU
            if (es.getEquation(bus.getNum(), AcEquationType.DISTR_Q).isPresent()
                    || bus.getGeneratorVoltageControl()
                            .map(vc -> vc.getControlledBus() != bus
                                    || vc.getControllerElements().stream().anyMatch(cb -> cb != bus))
                            .orElse(false)) {
                return true;
            }
        }
        return false;
    }

    /** The renormalized DISTR_Q weights after losing {@code lost}: replay the CPU's
     *  LfContingency.processLostGenerators reactive-key update (control type -> OFF +
     *  updateReactiveKeys), re-extract the dq pack, and diff against the base weights.
     *  Returns {contribIdx[], newWeight[]} (as double[][]), or null when not replayable. */
    private static double[][] distrQWeightOverrides(EquationSystem<AcVariableType, AcEquationType> es,
                                                    GpuAcData gpuData,
                                                    List<com.powsybl.openloadflow.network.LfGenerator> lost) {
        var network = lost.get(0).getBus().getNetwork();
        List<com.powsybl.openloadflow.network.LfGenerator> toggled = new ArrayList<>();
        java.util.Set<LfBus> touched = new java.util.LinkedHashSet<>();
        try {
            for (var g : lost) {
                if (g.getGeneratorControlType() == com.powsybl.openloadflow.network.LfGenerator.GeneratorControlType.VOLTAGE
                        && g.getBus() != null
                        && es.getEquation(g.getBus().getNum(), AcEquationType.DISTR_Q).isPresent()) {
                    g.setGeneratorControlType(com.powsybl.openloadflow.network.LfGenerator.GeneratorControlType.OFF);
                    toggled.add(g);
                    touched.add(g.getBus());
                }
            }
            if (toggled.isEmpty()) {
                return new double[][] {new double[0], new double[0]};
            }
            for (LfBus bus : touched) {
                bus.getGeneratorVoltageControl().ifPresent(
                        com.powsybl.openloadflow.network.GeneratorVoltageControl::updateReactiveKeys);
            }
            double[] newW = com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor
                    .extract(network, es, false, false).dqWeight();
            double[] baseW = gpuData.dqWeight();
            if (newW.length != baseW.length) {
                return null;                                 // structural drift — not a pure reweight
            }
            List<Double> idx = new ArrayList<>();
            List<Double> w = new ArrayList<>();
            for (int k = 0; k < newW.length; k++) {
                if (newW[k] != baseW[k]) {
                    idx.add((double) k);
                    w.add(newW[k]);
                }
            }
            return new double[][] {idx.stream().mapToDouble(Double::doubleValue).toArray(),
                w.stream().mapToDouble(Double::doubleValue).toArray()};
        } finally {
            for (var g : toggled) {
                g.setGeneratorControlType(com.powsybl.openloadflow.network.LfGenerator.GeneratorControlType.VOLTAGE);
            }
            for (LfBus bus : touched) {
                bus.getGeneratorVoltageControl().ifPresent(
                        com.powsybl.openloadflow.network.GeneratorVoltageControl::updateReactiveKeys);
            }
        }
    }

    /** The contingency's SCHEDULED active power loss (pu): mirror of LfContingency's
     *  disconnectedGenerationActivePower - disconnectedLoadActivePower, incl. islanded buses'
     *  injections (with the same non-AC-emulation HVDC generation exclusion). */
    private static double scheduledActivePowerLoss(BatchContingency c, List<LfBus> islandedBuses) {
        double loss = 0;
        for (com.powsybl.openloadflow.network.LfGenerator g : c.lostGenerators()) {
            loss += g.getTargetP();
        }
        for (LoadLoss ll : c.lostLoads()) {
            loss -= ll.p();
        }
        for (LfBus bus : islandedBuses) {
            double gen = bus.getGenerationTargetP();
            gen -= bus.getHvdcs().stream()
                    .filter(h -> !h.isAcEmulation())
                    .map(h -> h.getBus1() == bus ? h.getConverterStation1() : h.getConverterStation2())
                    .mapToDouble(com.powsybl.openloadflow.network.LfGenerator::getTargetP)
                    .sum();
            loss += gen - bus.getLoadTargetP();
        }
        return loss;
    }

    /** The batched-scenario plan: which contingencies qualified ({@code batchedIndices}) and, per
     *  contingency, the frozen islanded-bus rows/targets, the islanded LfBus nums (excluded from result
     *  building), the merged per-bus generator outages (else null), and the count of on-GPU frozen islands. */
    record ScenarioPlan(List<Integer> batchedIndices, int[][] frozenRowsByIndex,
                        int[][] islandedNumsByIndex, double[][] frozenTargetsByIndex,
                        GenOutage[][] genOutagesByIndex, int islandingCount,
                        double[] activePowerLossByIndex,
                        int[][] dqOvContribByIndex, double[][] dqOvWByIndex) {
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
            int[] curvePRows) {
        List<Integer> batchedIndices = new ArrayList<>();
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
        int[][] dqOvContribByIndex = new int[contingencies.size()][];
        double[][] dqOvWByIndex = new double[contingencies.size()][];
        int islandingCount = 0;
        for (int i = 0; i < qualifications.size(); i++) {
            GpuBatchContingencyValidator.Qualification q = qualifications.get(i);
            if (!q.qualified()) {
                continue;
            }
            BatchContingency c = contingencies.get(i);
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
                    continue;                                // controlled/unpacked transformer tap — CPU
                }
                if (!c.shuntShifts().isEmpty()
                        && !canBatchShuntOutage(c.shuntShifts().stream().map(ShuntShift::shunt).toList(), elementMapper, gpuData)) {
                    continue;                                // unpacked / DISTR_Q-contributing shunt — CPU
                }
                GenOutage[] gos = GpuBatchInjectionOutages.computeInjectionOutages(es, gpuData, dsSum, slackParticipates,
                        dsParams, c.lostGenerators(), c.lostLoads());
                if (gos == null) {                           // slack / >1 DS bus / remote gen — route to CPU
                    continue;
                }
                // Curve-limits channel active: its p_now linkage models the DS delta ONLY. An injection
                // override at a curve bus's P row, or a strategy PAIR (anchor shifts move every DS row),
                // would skew the evaluated limits — CPU.
                if (curvePRows != null) {
                    boolean skewsCurve = c.contingencyId().indexOf(GpuBatchStatus.STRATEGY_KEY_SEP) >= 0;
                    for (GenOutage go : gos) {
                        for (int pr : curvePRows) {
                            if (go.pRow() == pr) {
                                skewsCurve = true;
                            }
                        }
                    }
                    if (skewsCurve) {
                        continue;
                    }
                }
                if (!c.outagedBranches().isEmpty()) {        // COMBO: the branch part must also qualify
                    if (c.outagedBranches().stream().anyMatch(distrRhoControllers::contains)
                            || islandsRemoteVcMember(es, q.islandedBuses())) {
                        continue;
                    }
                    // conservative: a lost generator on an ISLANDED bus is out of combo scope
                    FrozenIsland fi = islandedFrozen(es, q.islandedBuses());
                    islandedNumsByIndex[i] = q.islandedBuses().stream().mapToInt(LfBus::getNum).toArray();
                    if ("1".equals(System.getenv("OLF_GPU_DBG_BUSDUMP"))) {
                        System.err.println("FROZEN-ISLAND(combo) '" + c.contingencyId().replace(GpuBatchStatus.STRATEGY_KEY_SEP, '|') + "' buses="
                                + q.islandedBuses().stream().map(b -> b.getId() + "#" + b.getNum()).toList()
                                + " rows=" + java.util.Arrays.toString(fi.rows()));
                    }
                    boolean genOnIsland = false;
                    for (GenOutage go : gos) {
                        for (int fr : fi.rows()) {
                            if (fr == go.pRow() || fr == go.vRow()) {
                                genOnIsland = true;
                            }
                        }
                    }
                    if (genOnIsland) {
                        continue;
                    }
                    frozenRowsByIndex[i] = fi.rows();
                    frozenTargetsByIndex[i] = fi.targets();
                    if (fi.rows().length > 0) {
                        islandingCount++;
                    }
                } else {
                    frozenRowsByIndex[i] = NO_FROZEN_ROWS;
                    frozenTargetsByIndex[i] = NO_FROZEN_TARGETS;
                }
                if (c.lostGenerators().stream().anyMatch(g -> g.getBus() != null
                        && es.getEquation(g.getBus().getNum(), AcEquationType.DISTR_Q).isPresent())) {
                    double[][] ov = distrQWeightOverrides(es, gpuData, c.lostGenerators());
                    if (ov == null) {
                        continue;                            // weight rediff unavailable — CPU
                    }
                    dqOvContribByIndex[i] = java.util.Arrays.stream(ov[0]).mapToInt(d -> (int) d).toArray();
                    dqOvWByIndex[i] = ov[1];
                }
                genOutagesByIndex[i] = gos;
                activePowerLossByIndex[i] = scheduledActivePowerLoss(c, q.islandedBuses());
                batchedIndices.add(i);
            } else if (!c.outagedShunts().isEmpty()) {       // fixed-shunt disconnection: mask the shunt element
                if (canBatchShuntOutage(c.outagedShunts(), elementMapper, gpuData)) {
                    batchedIndices.add(i);
                    frozenRowsByIndex[i] = NO_FROZEN_ROWS;
                    frozenTargetsByIndex[i] = NO_FROZEN_TARGETS;
                }                                            // else: not a packed fixed shunt / DISTR_Q contributor → CPU
            } else if (c.outagedBranches().stream().noneMatch(distrRhoControllers::contains)) {
                if (islandsRemoteVcMember(es, q.islandedBuses())) {
                    continue;                                // DISTR_Q group member islanded — CPU
                }
                if (curvePRows != null && c.contingencyId().indexOf(GpuBatchStatus.STRATEGY_KEY_SEP) >= 0) {
                    continue;                                // pair anchor shifts skew the curve linkage — CPU
                }
                batchedIndices.add(i);
                FrozenIsland fi = islandedFrozen(es, q.islandedBuses());
                islandedNumsByIndex[i] = q.islandedBuses().stream().mapToInt(LfBus::getNum).toArray();
                if ("1".equals(System.getenv("OLF_GPU_DBG_BUSDUMP"))) {
                    System.err.println("FROZEN-ISLAND '" + c.contingencyId().replace(GpuBatchStatus.STRATEGY_KEY_SEP, '|') + "' buses="
                            + q.islandedBuses().stream().map(b -> b.getId() + "#" + b.getNum()).toList()
                            + " rows=" + java.util.Arrays.toString(fi.rows()));
                }
                frozenRowsByIndex[i] = fi.rows();
                frozenTargetsByIndex[i] = fi.targets();
                activePowerLossByIndex[i] = scheduledActivePowerLoss(c, q.islandedBuses());
                if (frozenRowsByIndex[i].length > 0) {
                    islandingCount++;
                }
            }
        }
        return new ScenarioPlan(batchedIndices, frozenRowsByIndex, islandedNumsByIndex,
                frozenTargetsByIndex, genOutagesByIndex, islandingCount, activePowerLossByIndex,
                dqOvContribByIndex, dqOvWByIndex);
    }
}
