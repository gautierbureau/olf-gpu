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

    /** System property: BATCH the contingencies that strand a distributed-slack participant in
     *  another synchronous component (default false = route them to the CPU).
     *
     *  <p>The device side is complete and correct: the per-scenario participant mask reaches the
     *  kernel, and both the slack's retained share and the stopping gate are renormalized over the
     *  survivors (leaving them global is what made the first attempt at this fail). What is NOT yet
     *  right is a difference these scenarios merely EXPOSE — OLF water-fills per generator, the pack
     *  aggregates per bus, and only a mismatch this large saturates anything. Flip this on to measure
     *  that gap; leave it off to keep violation parity exact. */
    static final String BATCH_STRANDED_DS_PROPERTY = "olf.gpu.sa.batchStrandedDs";

    private static boolean batchStrandedDsEnabled() {
        return Boolean.parseBoolean(System.getProperty(BATCH_STRANDED_DS_PROPERTY, "false"));
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

    /** The renormalized DISTR_RHO coefficients after outaging shared-TVC group members among
     *  {@code outaged} (plan A5 / A6). Returns {contribIdx[], newCoef[]} as double[][] — EMPTY
     *  arrays when no group member is outaged; null (CPU route) when more than one member of a
     *  group is outaged or no survivor would remain. Coefficients are ANALYTIC: the CPU's are live
     *  1/count suppliers over !isDisabled(), so survivors get 1/c' − (self ? 1 : 0) and anything
     *  touching the outaged member's rho goes to 0 (its cross terms deactivate with the element;
     *  its own row flips back to the rho identity). FREE-member outage (plan A6): the free role
     *  RE-PICKS to the first survivor — that carrier's DISTR_RHO row deactivates (all its
     *  contributions go to 0; the pilot's relocated Q lands there via the group mode-scan scatter)
     *  and the remaining survivors renormalize over c'. */
    private static double[][] distrRhoOverrides(EquationSystem<AcVariableType, AcEquationType> es,
                                                GpuAcData gpuData, Set<LfBranch> distrRhoControllers,
                                                List<LfBranch> outaged) {
        List<LfBranch> members = outaged.stream().filter(distrRhoControllers::contains).toList();
        if (members.isEmpty()) {
            return new double[][] {new double[0], new double[0]};
        }
        if (members.size() > 1) {
            return null;                                     // multi-member outage — CPU
        }
        LfBranch d = members.get(0);
        var tvc = d.getVoltageControl().orElse(null);
        if (tvc == null) {
            return null;
        }
        List<LfBranch> group = tvc.getMergedControllerElements().stream()
                .filter(b -> !b.isDisabled()).toList();
        if (group.size() < 2) {
            return null;                                     // no survivor — CPU
        }
        // plan-A6: outaging the FREE member (first enabled) re-picks the free role to the FIRST
        // SURVIVOR, whose DISTR_RHO row deactivates (it becomes the pilot-Q carrier)
        LfBranch newCarrier = group.get(0) == d ? group.stream().filter(b -> b != d).findFirst().orElse(null) : null;
        if (group.get(0) == d && group.size() == 2) {
            return null;                                     // free outaged, single survivor: control degenerates — CPU
        }
        java.util.Map<Integer, LfBranch> byRhoCol = new java.util.HashMap<>();
        for (LfBranch b : group) {
            var v = es.getVariableSet().getVariable(b.getNum(), AcVariableType.BRANCH_RHO1);
            if (v == null) {
                return null;
            }
            byRhoCol.put(v.getRow(), b);
        }
        int dCol = group.stream().filter(b -> b == d).findFirst()
                .map(b -> es.getVariableSet().getVariable(b.getNum(), AcVariableType.BRANCH_RHO1).getRow())
                .orElseThrow();
        double cPrime = group.size() - 1.0;
        List<Double> idx = new ArrayList<>();
        List<Double> coefs = new ArrayList<>();
        for (int k = 0; k < gpuData.drRow().length; k++) {
            LfBranch rowOwner = byRhoCol.get(gpuData.drRow()[k]);
            LfBranch colOwner = byRhoCol.get(gpuData.drCol()[k]);
            if (rowOwner == null && colOwner == null) {
                continue;                                    // another group's contribution
            }
            double coef;
            if (rowOwner == d || colOwner == d || gpuData.drRow()[k] == dCol || gpuData.drCol()[k] == dCol) {
                coef = 0;                                    // cross terms to the outaged member die with it
            } else if (newCarrier != null && rowOwner == newCarrier) {
                coef = 0;                                    // the re-picked carrier's DISTR_RHO row deactivates
            } else {
                coef = 1.0 / cPrime - (gpuData.drRow()[k] == gpuData.drCol()[k] ? 1.0 : 0.0);
            }
            idx.add((double) k);
            coefs.add(coef);
        }
        return new double[][] {idx.stream().mapToDouble(Double::doubleValue).toArray(),
            coefs.stream().mapToDouble(Double::doubleValue).toArray()};
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
    /** How many distributed-slack participants this contingency puts in another synchronous component. */
    /** The distributed-slack participants whose P-target row this contingency froze — i.e. those it
     *  put in another synchronous component. OLF distributes over
     *  {@code getSynchronousNetworks().getFirst()}, so these are simply not participants for this
     *  scenario, and the device renormalizes the remaining factors over the survivors. */
    private static int[] islandedDsParticipants(int[] frozenRows, GpuAcData gpuData) {
        int[] dsPhiRow = gpuData.dsPhiRow();
        if (dsPhiRow == null || dsPhiRow.length == 0 || frozenRows.length == 0) {
            return NO_ISLANDED_DS;
        }
        int[] found = new int[dsPhiRow.length];
        int count = 0;
        for (int i = 0; i < dsPhiRow.length; i++) {
            for (int frozen : frozenRows) {
                if (frozen == dsPhiRow[i]) {
                    found[count++] = i;
                    break;
                }
            }
        }
        return count == 0 ? NO_ISLANDED_DS : java.util.Arrays.copyOf(found, count);
    }

    private static final int[] NO_ISLANDED_DS = new int[0];

    /** The reactive-limits controllers this contingency puts in another synchronous component: their
     *  BUS_V row is one of the frozen island rows. OLF disables those buses, so its ReactiveLimits
     *  loop never sees them; the device must not either, or it pins a PV bus OLF leaves alone. */
    private static int[] islandedRlControllers(int[] frozenRows, GpuAcData gpuData) {
        int[] rlVrow = gpuData.rlVrow();
        if (rlVrow == null || rlVrow.length == 0 || frozenRows.length == 0) {
            return NO_ISLANDED_DS;
        }
        int[] found = new int[rlVrow.length];
        int count = 0;
        for (int i = 0; i < rlVrow.length; i++) {
            for (int frozen : frozenRows) {
                if (frozen == rlVrow[i]) {
                    found[count++] = i;
                    break;
                }
            }
        }
        return count == 0 ? NO_ISLANDED_DS : java.util.Arrays.copyOf(found, count);
    }

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
                        int[][] dqOvContribByIndex, double[][] dqOvWByIndex,
                        int[][] drOvContribByIndex, double[][] drOvCByIndex,
                        /** Per scenario, the distributed-slack participants this contingency strands in
                         *  another synchronous component — the set the device must renormalize over. */
                        int[][] islandedDsByIndex,
                        /** Per scenario, the reactive-limits controllers stranded in another component. */
                        int[][] islandedRlByIndex) {
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
        // Per-contingency DISTR_RHO coefficient overrides (plan A5): outaging a NON-FREE member of a
        // shared-TVC group renormalizes the survivors' 1/count coefficients (the CPU's coefficients
        // are live suppliers over !isDisabled) and zeroes every contribution touching the outaged
        // member's rho; the FREE member's outage shifts the pilot-Q role — CPU (A6).
        int[][] drOvContribByIndex = new int[contingencies.size()][];
        double[][] drOvCByIndex = new double[contingencies.size()][];
        int[][] islandedDsByIndex = new int[contingencies.size()][];
        int[][] islandedRlByIndex = new int[contingencies.size()][];
        int islandingCount = 0;
        for (int i = 0; i < qualifications.size(); i++) {
            GpuBatchContingencyValidator.Qualification q = qualifications.get(i);
            if (!q.qualified()) {
                com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:SIDE_OPEN_UNPACKED");
                continue;
            }
            BatchContingency c = contingencies.get(i);
            // SIDE-OPEN scope (conservative): a branch that carries a VARIABLE (BRANCH_RHO1 /
            // BRANCH_ALPHA1) or belongs to a DISTR_RHO group changes the live es shape when
            // half-opened — CPU for those; plain impedance branches batch.
            if (!c.sideOpenBranches().isEmpty() && c.sideOpenBranches().stream().anyMatch(so ->
                    distrRhoControllers.contains(so.branch())
                    || es.getVariableSet().getVariable(so.branch().getNum(), AcVariableType.BRANCH_RHO1).getRow() >= 0
                    || es.getVariableSet().getVariable(so.branch().getNum(), AcVariableType.BRANCH_ALPHA1).getRow() >= 0)) {
                com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:VAR_CARRYING_SIDE_OPEN");
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
                    com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:RECLOSE_UNPACKED_BASE_OPEN");
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
                    com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:CONTROLLED_OR_UNPACKED_TAP");
                    continue;                                // controlled/unpacked transformer tap — CPU
                }
                if (!c.shuntShifts().isEmpty()
                        && !canBatchShuntOutage(c.shuntShifts().stream().map(ShuntShift::shunt).toList(), elementMapper, gpuData)) {
                    com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:SHUNT_UNPACKED_OR_DISTRQ");
                    continue;                                // unpacked / DISTR_Q-contributing shunt — CPU
                }
                GenOutage[] gos = GpuBatchInjectionOutages.computeInjectionOutages(es, gpuData, dsSum, slackParticipates,
                        dsParams, c.lostGenerators(), c.lostLoads());
                if (gos == null) {                           // slack / >1 DS bus / remote gen — route to CPU
                    com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:SHUNT_SHAPE");
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
                        com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:GEN_OUTAGE_SHAPE");
                        continue;
                    }
                }
                if (!c.outagedBranches().isEmpty()) {        // COMBO: the branch part must also qualify
                    if (islandsRemoteVcMember(es, q.islandedBuses())) {
                        com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:GEN_ON_ISLAND");
                        continue;
                    }
                    double[][] drOv = distrRhoOverrides(es, gpuData, distrRhoControllers, c.outagedBranches());
                    if (drOv == null) {
                        if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
                            System.err.println("QUAL_REJECT drOv " + c.contingencyId());
                        }
                        com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:DISTR_RHO_GROUP_OUTAGE");
                        continue;                            // multi-member / no-survivor group outage — CPU
                    }
                    if (drOv[0].length > 0) {
                        drOvContribByIndex[i] = java.util.Arrays.stream(drOv[0]).mapToInt(d -> (int) d).toArray();
                        drOvCByIndex[i] = drOv[1];
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
                        com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:ZI_OUTAGE_SHAPE");
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
                        com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:DISTR_Q_WEIGHT_REDIFF");
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
            } else {
                if (islandsRemoteVcMember(es, q.islandedBuses())) {
                    com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:ISLANDS_REMOTE_VC_MEMBER");
                    continue;                                // DISTR_Q group member islanded — CPU
                }
                double[][] drOv = distrRhoOverrides(es, gpuData, distrRhoControllers, c.outagedBranches());
                if (drOv == null) {
                    if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
                        System.err.println("QUAL_REJECT drOv2 " + c.contingencyId());
                    }
                    com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:DISTR_RHO_GROUP_OUTAGE2");
                    continue;                                // multi-member / no-survivor group outage — CPU
                }
                if (drOv[0].length > 0) {
                    drOvContribByIndex[i] = java.util.Arrays.stream(drOv[0]).mapToInt(d -> (int) d).toArray();
                    drOvCByIndex[i] = drOv[1];
                }
                if (curvePRows != null && c.contingencyId().indexOf(GpuBatchStatus.STRATEGY_KEY_SEP) >= 0) {
                    com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:PAIR_CURVE_LINKAGE");
                    continue;                                // pair anchor shifts skew the curve linkage — CPU
                }
                FrozenIsland fi = islandedFrozen(es, q.islandedBuses());
                // A contingency that BOTH splits the network AND strands a distributed-slack participant.
                // OLF distributes over getSynchronousNetworks().getFirst(), so a stranded participant is
                // not a participant at all: its share must not be computed, and the factors of the ones
                // that remain must be renormalized WITHOUT it. Masking it out of the distribution loops
                // alone is not enough and was reverted once for exactly that reason — the slack keeps the
                // retained share it had while the stranded bus still counted, and the pass retires with
                // most of the mismatch undistributed. The device now takes the per-scenario participant
                // set and derives BOTH the shares and the stopping gate from it.
                islandedDsByIndex[i] = islandedDsParticipants(fi.rows(), gpuData);
                islandedRlByIndex[i] = islandedRlControllers(fi.rows(), gpuData);
                if (islandedDsByIndex[i].length > 0 && !batchStrandedDsEnabled()) {
                    // The renormalization below is correct and measured (see the field's javadoc), but
                    // a SECOND difference remains on these scenarios and it is not this one: OLF's
                    // water-fill saturates per GENERATOR while the device pack aggregates per BUS, and a
                    // fragmenting outage is the only thing on this network with a mismatch big enough to
                    // saturate anything (12.86 pu, ~29% of the participation). Measured on the 9
                    // reproducers: 10 differing violation lines with active limits ON, 1 with them OFF,
                    // 0 with distributed slack off entirely. So these stay on the CPU until the pack is
                    // per-generator — correctness over coverage, 91 of 2000 on rte6515.
                    com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("VALIDATOR:STRANDS_DS_PARTICIPANT");
                    com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:OTHER");
                    continue;
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
                activePowerLossByIndex[i] = scheduledActivePowerLoss(c, q.islandedBuses());
                if (frozenRowsByIndex[i].length > 0) {
                    islandingCount++;
                }
            }
        }
        return new ScenarioPlan(batchedIndices, frozenRowsByIndex, islandedNumsByIndex,
                frozenTargetsByIndex, genOutagesByIndex, islandingCount, activePowerLossByIndex,
                dqOvContribByIndex, dqOvWByIndex, drOvContribByIndex, drOvCByIndex, islandedDsByIndex,
                islandedRlByIndex);
    }
}
