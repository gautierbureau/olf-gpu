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
    static FrozenIsland islandedFrozen(EquationSystem<AcVariableType, AcEquationType> es, List<LfBus> islandedBuses,
                                       Set<LfBranch> relocatedCouplers) {
        if (islandedBuses.isEmpty()) {
            return new FrozenIsland(NO_FROZEN_ROWS, NO_FROZEN_TARGETS);
        }
        List<Integer> rows = new ArrayList<>(islandedBuses.size() * 2);
        List<Double> targets = new ArrayList<>(islandedBuses.size() * 2);
        for (LfBus bus : islandedBuses) {
            addFrozenVarRow(es, bus.getNum(), AcVariableType.BUS_PHI, 0.0, rows, targets);   // angle 0
            addFrozenVarRow(es, bus.getNum(), AcVariableType.BUS_V, 1.0, rows, targets);     // voltage 1 pu
        }
        freezeRelocatedCouplerRows(es, islandedBuses, relocatedCouplers, rows, targets);
        int[] outRows = new int[rows.size()];
        double[] outTargets = new double[targets.size()];
        for (int i = 0; i < outRows.length; i++) {
            outRows[i] = rows.get(i);
            outTargets[i] = targets.get(i);
        }
        return new FrozenIsland(outRows, outTargets);
    }

    /**
     * The extra identity rows an island needs under the COUPLER-TREE RELOCATION. Relocated, a coupler inside the
     * island hosts the child bus's P/Q balance on its own {@code DUMMY_P} / {@code DUMMY_Q} rows, while the
     * coupler's {@code ZERO_V} / {@code ZERO_PHI} moved onto that bus's V / PHI rows — which the freeze above turns
     * into identities, erasing them. Freezing the coupler's dummy rows as well makes EVERY row of the island an
     * identity again, which is what an island wants (it has no equation left to satisfy): without it the child
     * bus's power balance is stranded on a dummy row nobody pins, and the island cannot converge.
     *
     * <p>Only couplers with BOTH endpoints islanded are frozen here; the boundary coupler the contingency opened
     * already has its dummy rows pinned by the chunk solver's zi-outage path. The set is empty when the relocation
     * is off, so the default path keeps its exact row list (and with it {@code nActive}).
     */
    private static void freezeRelocatedCouplerRows(EquationSystem<AcVariableType, AcEquationType> es,
                                                   List<LfBus> islandedBuses, Set<LfBranch> relocatedCouplers,
                                                   List<Integer> rows, List<Double> targets) {
        if (relocatedCouplers.isEmpty()) {
            return;
        }
        Set<Integer> islandedNums = islandedBuses.stream().map(LfBus::getNum)
                .collect(java.util.stream.Collectors.toSet());
        Set<LfBranch> done = new java.util.HashSet<>();
        for (LfBus bus : islandedBuses) {
            for (LfBranch br : bus.getBranches()) {
                if (relocatedCouplers.contains(br) && br.getBus1() != null && br.getBus2() != null
                        && islandedNums.contains(br.getBus1().getNum())
                        && islandedNums.contains(br.getBus2().getNum())
                        && done.add(br)) {
                    addFrozenVarRow(es, br.getNum(), AcVariableType.DUMMY_P, 0.0, rows, targets);
                    addFrozenVarRow(es, br.getNum(), AcVariableType.DUMMY_Q, 0.0, rows, targets);
                }
            }
        }
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
                                                List<LfBranch> outaged, boolean incrementalTvc) {
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
        // The INCREMENTAL transformer mode never activates a DISTR_RHO equation (each tap is a fixed ratio the outer
        // loop moves), so an outage has no sharing coefficient to renormalize and no free role to re-pick: the
        // survivors' tap moves are the in-kernel loop's business (tvcGroupDecisionBatch decides on the live members).
        // Not "inactive at plan time": the other modes start with the sharing off and switch it on during the run.
        if (incrementalTvc) {
            return new double[][] {new double[0], new double[0]};
        }
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
    /** What an island does to the remote generator-voltage-control groups it touches. */
    private enum VcSplit {
        /** No remote group affected, or every group the island touches goes into it WHOLE. */
        CONTAINED,
        /** The island KILLS a control — every surviving member bus solves its own reactive balance. */
        CONTROL_DIES,
        /** A group survives with FEWER controllers — batchable once its reactive keys are renormalized. */
        RENORMALIZE,
        /** The island kills a control outright (pilot gone with controllers left, or the last
         *  controller gone with the pilot left), or touches an HVDC endpoint. Not modelled — CPU. */
        UNMODELED
    }

    /**
     * Classify the island, and collect into {@code lostControllers} the islanded controller buses
     * whose group survives without them.
     *
     * <p>A group that goes into the island WHOLE is not a split: {@link #islandedFrozen} pins the
     * BUS_V and BUS_PHI row of every islanded bus, so the pilot's target-V row and every
     * controller's DISTR_Q row leave the solved system together.
     *
     * <p>A group that merely LOSES a controller is still solvable: OLF renormalizes the reactive
     * keys over the enabled controllers, which is exactly what {@link #distrQWeightOverrides}
     * replays for a lost generator — so the same channel carries it.
     *
     * <p>What is NOT modelled is a control that DIES: the pilot islanded while a controller
     * survives, or the last controller islanded while the pilot survives. OLF then switches the
     * survivors to PQ, which is a row-mode change this path does not express yet.
     */
    private static VcSplit classifyVcSplit(EquationSystem<AcVariableType, AcEquationType> es,
                                           List<LfBus> islandedBuses, List<LfBus> lostControllers,
                                           List<LfBus> deadControlBuses,
                                           List<com.powsybl.openloadflow.network.LfHvdc> islandedHvdcs) {
        java.util.Set<LfBus> islanded = new java.util.HashSet<>(islandedBuses);
        VcSplit result = VcSplit.CONTAINED;
        for (LfBus bus : islandedBuses) {
            if (!bus.getHvdcs().isEmpty()) {
                // An islanded HVDC endpoint. OLF (LfContingency.processHvdcsWithoutPower) sets BOTH
                // converter stations' targetP to 0 — and in AC emulation those targets ARE already 0,
                // because the injection is an equation term rather than a bus target. So the whole of
                // it is: stop the AC-emulation equation injecting, i.e. mask the hv element. A
                // NON-AC-emulation station does carry its targetP in the bus target, and zeroing that
                // is not expressed here, so those still go to the CPU.
                for (var h : bus.getHvdcs()) {
                    if (!h.isAcEmulation()) {
                        continue;                            // SETPOINT link: its stations' targets are the caller's
                    }                                        // injection channel (hvdcStationResets), no element to mask
                    if (!islandedHvdcs.contains(h)) {
                        islandedHvdcs.add(h);
                    }
                }
                continue;
            }
            // The MAIN control, and its MERGED element list. A bus whose control was merged into
            // another's is DEPENDENT: its own object describes only itself, and a MAIN control's
            // own controllerElements omits the members merged into it - so on either object a
            // merged survivor is invisible, and a pilot that loses its only local generator looks
            // like a whole group leaving with the island. rte6515 BUS-1115_BBS_1: pilot VL-1115_0
            // islanded, merged survivor VL-1115_1 left with a DISTR_Q row and no Q balance of its
            // own, V frozen at the base value, 0.373 kV.
            var vc = bus.getGeneratorVoltageControl()
                    .<com.powsybl.openloadflow.network.GeneratorVoltageControl>map(
                            com.powsybl.openloadflow.network.VoltageControl::getMainVoltageControl)
                    .orElse(null);
            if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                System.err.printf("QUALIFY_TRACE vcSplit-enter bus=%s hasVc=%b distrQ=%b merge=%s%n", bus.getId(),
                        vc != null, es.getEquation(bus.getNum(), AcEquationType.DISTR_Q).isPresent(),
                        vc == null ? "-" : vc.getMergeStatus());
            }
            boolean remoteGroupMember = es.getEquation(bus.getNum(), AcEquationType.DISTR_Q).isPresent()
                    || vc != null && (vc.getControlledBus() != bus
                            || vc.getMergedControllerElements().stream().anyMatch(cb -> cb != bus));
            if (!remoteGroupMember) {
                continue;
            }
            if (vc == null) {
                return VcSplit.UNMODELED;                    // DISTR_Q row with no reachable control handle
            }
            boolean pilotIslanded = islanded.contains(vc.getControlledBus());
            List<LfBus> surviving = vc.getMergedControllerElements().stream()
                    .filter(cb -> !islanded.contains(cb))
                    .toList();
            if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                // The group decision, with BOTH element lists: a MAIN control's own controllerElements
                // omits the members that were MERGED into it, so a survivor that lives only in the
                // merged list is invisible to this decision.
                System.err.printf("QUALIFY_TRACE vcSplit bus=%s merge=%s pilot=%s pilotIslanded=%b"
                                + " controllers=%s merged=%s surviving=%s%n",
                        bus.getId(), vc.getMergeStatus(), vc.getControlledBus().getId(), pilotIslanded,
                        vc.getControllerElements().stream().map(LfBus::getId).toList(),
                        vc.getMergedControllerElements().stream().map(LfBus::getId).toList(),
                        surviving.stream().map(LfBus::getId).toList());
            }
            if (pilotIslanded && surviving.isEmpty()) {
                continue;                                    // the whole group left with the island
            }
            if (pilotIslanded || surviving.isEmpty()) {
                // The control DIES (OLF: VoltageControl.isDisabled — every controlled bus disabled,
                // or every controller element disabled). updateRemoteVoltageControlEquations then
                // deactivates the pilot's BUS_TARGET_V and every controller's DISTR_Q, so each
                // SURVIVING member bus goes back to solving its own reactive balance. On the device
                // that is exactly a PV->PQ row-mode flip with the bus's own Q target.
                List<LfBus> survivors = new ArrayList<>(surviving);
                if (!pilotIslanded) {
                    survivors.add(vc.getControlledBus());
                }
                for (LfBus sb : survivors) {
                    if (!deadControlBuses.contains(sb)) {
                        deadControlBuses.add(sb);
                    }
                }
                result = result == VcSplit.CONTAINED ? VcSplit.CONTROL_DIES : result;
                continue;
            }
            lostControllers.add(bus);
            result = VcSplit.RENORMALIZE;
        }
        return result;
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
                    g.setDisabled(true);                     // OLF's key rules skip a lost (disabled) generator
                    toggled.add(g);
                    touched.add(g.getBus());
                }
            }
            if (toggled.isEmpty()) {
                return new double[][] {new double[0], new double[0], new double[0], new double[0]};
            }
            // On the MAIN control (the merged group), as OLF does: a member's OWN control is the DEPENDENT one
            // whose element list is just itself, and updateReactiveKeys on it set that member's reactive
            // percent to 1.0 — permanently, since the restore below repeated the same call. Measured on
            // rte6515's [5000,6000) busbar window: the number of re-weighted entries grew with every
            // contingency (0, 10, 15, 27, 35, 45, 51, 56 ...), each late contingency carrying bogus overrides
            // for every group an earlier one had touched; BUS-612_BBS_2's VL-5967 DISTR_Q row became the
            // pilot's raw q alone (weights 1.0 / 0), the split (1.694, -1.574) against OLF's (0.912, -0.782)
            // pinned both members spuriously, and the distributed power ended 0.141 MW off.
            for (LfBus bus : touched) {
                updateMainReactiveKeys(bus);
            }
            double[] baseW = gpuData.dqWeight();
            // OLF_GPU_DQ_REWEIGHT=extract: the whole-network re-extraction instead of the analytic reweight (A/B)
            String reweightMode = System.getenv("OLF_GPU_DQ_REWEIGHT");
            double[] newW = "extract".equals(reweightMode) ? null : reweightedDqWeights(es, gpuData, touched);
            // The member KEYS after the loss (OLF's qPct on the touched groups' MAIN controls, just updated): the
            // kernels renormalize them over the members a pin leaves enabled, so the loss and an in-flight pin in the
            // same group compose - which fixed per-contribution weights cannot do (the fill applied them over its
            // pin-aware recompute and the rows kept the outage-only split while the targets followed the pin).
            boolean keyed = newW != null;                    // the analytic path: the pack's structure is the base's
            List<Double> keyMember = new ArrayList<>();
            List<Double> keyValue = new ArrayList<>();
            int[] memberBusNum = gpuData.dqMemberBusNum();
            if (memberBusNum != null) {
                java.util.Map<Integer, Integer> memberOfBusNum = new java.util.HashMap<>();
                for (int m = 0; m < memberBusNum.length; m++) {
                    memberOfBusNum.put(memberBusNum[m], m);
                }
                java.util.Set<Integer> done = new java.util.HashSet<>();
                for (LfBus bus : touched) {
                    var main = bus.getGeneratorVoltageControl()
                            .<com.powsybl.openloadflow.network.GeneratorVoltageControl>map(
                                    com.powsybl.openloadflow.network.VoltageControl::getMainVoltageControl).orElse(null);
                    if (main == null) {
                        continue;
                    }
                    for (LfBus mb : main.getMergedControllerElements()) {
                        Integer m = memberOfBusNum.get(mb.getNum());
                        if (m != null && done.add(m)) {
                            keyMember.add((double) m);
                            keyValue.add(mb.getRemoteControlReactivePercent());
                        }
                    }
                }
            }
            GpuAcData nd = null;
            if (newW == null || "check".equals(reweightMode)) {
                // the free role of a table-less group may move with the loss: re-extract, as before
                nd = com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor
                        .extract(network, es, false, false);
                if ("check".equals(reweightMode) && newW != null) {
                    // OLF_GPU_DQ_REWEIGHT=check: both, the extraction wins, every difference named
                    double[] ex = nd.dqWeight();
                    int nd0 = 0;
                    StringBuilder sb = new StringBuilder();
                    if (ex.length != newW.length) {
                        sb.append(" LENGTH ").append(newW.length).append(" vs ").append(ex.length);
                    } else {
                        for (int k = 0; k < ex.length; k++) {
                            if (ex[k] != newW[k]) {
                                if (nd0++ < 8) {
                                    sb.append(String.format(" k=%d row=%d kind=%d kM=%d jM=%d analytic=%.6f extract=%.6f base=%.6f",
                                            k, gpuData.dqRow()[k], gpuData.dqKind()[k], gpuData.dqKMember()[k], gpuData.dqJMember()[k],
                                            newW[k], ex[k], baseW[k]));
                                }
                            }
                        }
                    }
                    System.err.println("DQ_REWEIGHT_CHECK lost=" + lost.stream().map(g -> g.getId()).toList()
                            + " touched=" + touched.stream().map(LfBus::getId).toList() + " differing=" + nd0 + sb);
                }
                newW = nd.dqWeight();
            }
            if (newW.length != baseW.length) {
                return null;                                 // structural drift — not a pure reweight
            }
            List<Double> idx = new ArrayList<>();
            List<Double> w = new ArrayList<>();
            boolean dqOvTrace = System.getenv("OLF_GPU_DQOV_TRACE") != null;
            int traced = 0;
            int[] kMemberOf = gpuData.dqKMember();
            for (int k = 0; k < newW.length; k++) {
                if (newW[k] != baseW[k]) {
                    if (keyed && kMemberOf != null && kMemberOf[k] >= 0) {
                        continue;                            // the keys carry it (the kernels' live recompute)
                    }
                    idx.add((double) k);
                    w.add(newW[k]);
                    if (dqOvTrace && traced++ < 12 && nd != null) {   // the two packs' metadata at a differing index
                        System.err.printf("DQOV_PACK k=%d old(row=%d kind=%d elem=%d side=%d w=%.6f) new(row=%d kind=%d elem=%d side=%d w=%.6f)%n",
                                k, gpuData.dqRow()[k], gpuData.dqKind()[k], gpuData.dqElem()[k], gpuData.dqSide()[k], baseW[k],
                                nd.dqRow()[k], nd.dqKind()[k], nd.dqElem()[k], nd.dqSide()[k], newW[k]);
                    }
                }
            }
            if (dqOvTrace) {
                System.err.println("DQOV_PACK lost=" + lost.stream().map(g -> g.getId()).toList() + " differing=" + idx.size()
                        + " of " + newW.length + " oldFlags(ds,rl)=? newFlags=(false,false)");
            }
            return new double[][] {idx.stream().mapToDouble(Double::doubleValue).toArray(),
                w.stream().mapToDouble(Double::doubleValue).toArray(),
                keyMember.stream().mapToDouble(Double::doubleValue).toArray(),
                keyValue.stream().mapToDouble(Double::doubleValue).toArray()};
        } finally {
            for (var g : toggled) {
                g.setGeneratorControlType(com.powsybl.openloadflow.network.LfGenerator.GeneratorControlType.VOLTAGE);
                g.setDisabled(false);
            }
            for (LfBus bus : touched) {
                updateMainReactiveKeys(bus);
            }
        }
    }

    /** OLF's own key update: on the group's MAIN control, never on a member's dependent one. */
    /**
     * The base DISTR_Q pack's weights after {@link #updateMainReactiveKeys} ran on the touched buses, WITHOUT
     * re-extracting the network: a contribution's weight is {@code qPct_k - (j == k ? 1 : 0)} for the row's
     * controller k and the contributing controller j (GpuAcOuterLoopPacks.buildDistrQContributions), and the
     * toggle changes only the qPct of the touched groups' controllers. Measured on network.xiidm: the
     * whole-network extraction this replaces cost 18.8 s of a 207 s cold run (one per generator contingency at
     * a DISTR_Q member). Returns null when a touched bus is the free controller of a group WITHOUT a candidate
     * table — there the loss can move the free role, which changes the pack's row set, not only its weights.
     */
    private static double[] reweightedDqWeights(EquationSystem<AcVariableType, AcEquationType> es, GpuAcData gpuData,
                                                java.util.Set<LfBus> touched) {
        int[] kMember = gpuData.dqKMember();
        int[] jMember = gpuData.dqJMember();
        int[] memberBus = gpuData.dqMemberBusNum();
        double[] baseW = gpuData.dqWeight();
        if (kMember == null || jMember == null || memberBus == null || baseW == null) {
            return null;
        }
        int[] memberQrGroup = gpuData.dqMemberQrGroup();
        java.util.Map<Integer, Integer> memberOfBus = new java.util.HashMap<>();
        for (int m = 0; m < memberBus.length; m++) {
            memberOfBus.put(memberBus[m], m);
        }
        java.util.Map<Integer, Double> qPctByBus = new java.util.HashMap<>();
        for (LfBus bus : touched) {
            var main = bus.getGeneratorVoltageControl()
                    .<com.powsybl.openloadflow.network.GeneratorVoltageControl>map(
                            com.powsybl.openloadflow.network.VoltageControl::getMainVoltageControl).orElse(null);
            if (main == null) {
                continue;
            }
            // a group WITH a candidate table has a fixed free member (the first); without one the free role is
            // "first enabled" and may move with the loss - the pack's own qr-group index says which it is
            Integer m0 = memberOfBus.get(bus.getNum());
            if (m0 == null || memberQrGroup == null || m0 >= memberQrGroup.length || memberQrGroup[m0] < 0) {
                return null;
            }
            for (LfBus m : main.getMergedControllerElements()) {
                qPctByBus.put(m.getNum(), m.getRemoteControlReactivePercent());
            }
        }
        int[] kind = gpuData.dqKind();
        int[] side = gpuData.dqSide();
        double[] w = baseW.clone();
        for (int i = 0; i < w.length; i++) {
            int k = kMember[i];
            if (k < 0 || k >= memberBus.length) {
                continue;
            }
            Double qpk = qPctByBus.get(memberBus[k]);
            if (qpk == null) {
                continue;                                    // a group the loss did not touch keeps its weights
            }
            double weight = qpk - (jMember[i] == k ? 1.0 : 0.0);
            // a zero-impedance spanning-tree edge's DUMMY_Q term (kind 3) enters with the coupler side's sign,
            // +1 at bus1 and -1 at bus2 (buildDistrQContributions) - measured without it: every such weight
            // came out sign-flipped and seven busbar-section fixtures failed
            if (kind != null && side != null && kind[i] == 3 && side[i] != 0) {
                weight = -weight;
            }
            w[i] = weight;
        }
        return w;
    }

    private static void updateMainReactiveKeys(LfBus bus) {
        bus.getGeneratorVoltageControl()
                .<com.powsybl.openloadflow.network.GeneratorVoltageControl>map(
                        com.powsybl.openloadflow.network.VoltageControl::getMainVoltageControl)
                .ifPresent(com.powsybl.openloadflow.network.GeneratorVoltageControl::updateReactiveKeys);
    }

    /** The contingency's SCHEDULED active power loss (pu): mirror of LfContingency's
     *  disconnectedGenerationActivePower - disconnectedLoadActivePower, incl. islanded buses'
     *  injections (with the same non-AC-emulation HVDC generation exclusion). */
    /** How many distributed-slack participants this contingency puts in another synchronous component. */
    /** The distributed-slack participants whose P-target row this contingency froze — i.e. those it
     *  put in another synchronous component. OLF distributes over
     *  {@code getSynchronousNetworks().getFirst()}, so these are simply not participants for this
     *  scenario, and the device renormalizes the remaining factors over the survivors. */
    private static int[] islandedDsParticipants(int[] frozenRows, GpuAcData gpuData, List<LfBus> islandedBuses) {
        int[] dsPhiRow = gpuData.dsPhiRow();
        if (dsPhiRow == null || dsPhiRow.length == 0 || frozenRows.length == 0) {
            return NO_ISLANDED_DS;
        }
        // Match on the BUS as well as on the row. The row test alone misses a participant whose P row was
        // RELOCATED by the coupler-tree relocation: dsPhiRow carries pRow, which for a relocated bus is its
        // parent coupler's DUMMY_P, while the frozen rows are the bus's own BUS_PHI / BUS_V. The device then
        // kept distributing slack to a generator on a DEAD bus — measured on BUS-6014_BBS_2 as 7.386 MW of
        // distributed active power against the CPU, with every bus in the network shifted 2-4e-2 kV in one
        // direction, while the scheduled loss itself agreed exactly (both 0.0 pu).
        java.util.Set<Integer> islandedNums = new java.util.HashSet<>();
        for (LfBus b : islandedBuses) {
            islandedNums.add(b.getNum());
        }
        int[] dsBusNum = gpuData.dsBusNum();
        int[] found = new int[dsPhiRow.length];
        int count = 0;
        for (int i = 0; i < dsPhiRow.length; i++) {
            boolean islanded = dsBusNum != null && i < dsBusNum.length && islandedNums.contains(dsBusNum[i]);
            for (int frozen : frozenRows) {
                if (frozen == dsPhiRow[i]) {
                    islanded = true;
                    break;
                }
            }
            if (islanded) {
                found[count++] = i;
            }
            if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                System.err.printf("QUALIFY_TRACE dsPart i=%d row=%d busNum=%s islanded=%b%n", i, dsPhiRow[i],
                        dsBusNum != null && i < dsBusNum.length ? Integer.toString(dsBusNum[i]) : "-", islanded);
            }
        }
        return count == 0 ? NO_ISLANDED_DS : java.util.Arrays.copyOf(found, count);
    }

    /** The row mode a surviving member of a DEAD voltage control gets on its BUS_V row. A DISTR_Q
     *  member's row must be mode 4 ("hosts its own reactive balance"): fillDistrQBatch skips every
     *  contribution whose F row is mode 4 and the member's own q rides its alternate dest, which is
     *  keyed on mode 4. Mode 1 left the DISTR_Q family writing into the row on top of nothing that
     *  balanced the bus: rte6515 BUS-1055_BBS_1 (pilot VL-1055_0 islanded, merged section VL-1055_1
     *  surviving) - OLF's BUS_TARGET_Q at the survivor off by 0.039 pu, 1.3 kV. A survivor that is
     *  not a packed DISTR_Q member keeps the plain reactive row. */

    /** The outcome of OLF's voltage-control RE-MERGE after a contingency: buses that become plain PV buses at their
     *  own target, the controllers that thereby leave their base group's DISTR_Q, or a reject tag. */
    record ReHomed(java.util.LinkedHashMap<LfBus, Double> pvTargets, List<LfBus> leftControllers, String reject,
                   java.util.LinkedHashMap<LfBus, Double> remoteMainTargets, java.util.Set<LfBus> remoteClusterBuses,
                   /** Per controlled bus of a surviving cluster that is NOT the base main's: its SUBGROUP label
                    *  (1, 2, ...) this scenario — the base main's own cluster stays label 0 (tasks #71). */
                   java.util.LinkedHashMap<LfBus, Integer> subgroupOf,
                   /** REMOTE controllers (generators behind their own branch) of a re-merged cluster, each mapped to the
                    *  controlled bus OLF elects as the cluster's new main: their rows stay the base table's sharing rows
                    *  (the new pilot's balance rides the first free one through its dependent-pilot pool) and their
                    *  reactive-limit check reads the new pilot's voltage against the new pilot's target. */
                   java.util.LinkedHashMap<LfBus, LfBus> reHomedRemoteControllers) {

        /** Whether OLF's re-merge changes any control at all (or the device refuses the shape). */
        boolean changesControls() {
            return reject != null || !pvTargets.isEmpty() || !leftControllers.isEmpty() || !remoteMainTargets.isEmpty();
        }
    }

    /**
     * OLF merges the voltage controls of one zero-impedance network into one MAIN (LfZeroImpedanceNetwork
     * .updateVoltageControlMergeStatus: highest target, then controlled-bus id) and re-runs that merge on the
     * post-contingency networks. So a merged group whose controlled section is lost, or whose sections a coupler
     * outage separates, does NOT die: every surviving cluster of its controlled buses re-elects a main, and a
     * cluster of ONE bus with a LOCAL control is a plain PV bus at its own target — its controllers leave the
     * base group's DISTR_Q. Measured on rte6515 [5000,6000): BUS-6029_BBS_1 (section VL-6029_0 islanded, OLF
     * keeps bus_target_v(VL-6029_1) active — the device applied "control dies", 5.3 kV and 1.632 MW of slack) and
     * BUS-6002_BBS_2 (coupler open, VL-6002_2 8.7 kV off, 29 phantom violations). A cluster with several controls
     * or a remote controller is a NEW remote group the tables cannot host — CPU, tagged.
     */
    static ReHomed reHomedControls(List<LfBus> islandedBuses, List<LfBranch> outagedZiBranches, String contingencyId,
                                   EquationSystem<AcVariableType, AcEquationType> es) {
        java.util.Set<LfBus> islanded = new java.util.HashSet<>(islandedBuses);
        java.util.Set<LfBranch> outaged = new java.util.HashSet<>(outagedZiBranches);
        java.util.LinkedHashSet<com.powsybl.openloadflow.network.GeneratorVoltageControl> mains = new java.util.LinkedHashSet<>();
        java.util.function.Consumer<LfBus> addMain = b -> {
            if (b != null) {
                b.getGeneratorVoltageControl()
                        .<com.powsybl.openloadflow.network.GeneratorVoltageControl>map(
                                com.powsybl.openloadflow.network.VoltageControl::getMainVoltageControl)
                        .filter(m -> !m.getMergedDependentVoltageControls().isEmpty())
                        .ifPresent(mains::add);
            }
        };
        islandedBuses.forEach(addMain);
        for (LfBranch br : outagedZiBranches) {
            addMain.accept(br.getBus1());
            addMain.accept(br.getBus2());
        }
        java.util.LinkedHashMap<LfBus, Double> pv = new java.util.LinkedHashMap<>();
        List<LfBus> left = new ArrayList<>();
        java.util.LinkedHashMap<LfBus, Double> remoteMains = new java.util.LinkedHashMap<>();
        java.util.Set<LfBus> remoteCluster = new java.util.LinkedHashSet<>();
        java.util.LinkedHashMap<LfBus, Integer> subgroupOf = new java.util.LinkedHashMap<>();
        java.util.LinkedHashMap<LfBus, LfBus> reHomedRemoteControllers = new java.util.LinkedHashMap<>();
        for (var main : mains) {
            java.util.Map<LfBus, com.powsybl.openloadflow.network.VoltageControl<LfBus>> controlOf = new java.util.LinkedHashMap<>();
            controlOf.put(main.getControlledBus(), main);
            for (var d : main.getMergedDependentVoltageControls()) {
                controlOf.put(d.getControlledBus(), d);
            }
            java.util.Set<LfBus> alive = new java.util.LinkedHashSet<>();
            for (LfBus b : controlOf.keySet()) {
                if (!islanded.contains(b) && !b.isDisabled()) {
                    alive.add(b);
                }
            }
            java.util.Set<LfBus> seen = new java.util.HashSet<>();
            for (LfBus start : alive) {
                if (!seen.add(start)) {
                    continue;
                }
                // BFS over the surviving zero-impedance edges through EVERY live bus — an uncontrolled section
                // between two controlled ones still joins them (OLF's zero-impedance network is the bus set, not
                // the controlled set): walking controlled buses only split a group behind a plain section and
                // re-homed a controller OLF keeps merged (the 4-section fixture with bbs1 uncontrolled, bus-3:
                // vl12_2 "re-homed" 1.4 kV off; rte6515 BUS-6029_BBS_4's VL-6029_2 behind the uncontrolled VL-6029_1).
                List<LfBus> cluster = new ArrayList<>(List.of(start));   // the CONTROLLED buses reached
                List<LfBus> walk = new ArrayList<>(List.of(start));      // every bus reached
                java.util.Set<LfBus> walked = new java.util.HashSet<>(List.of(start));
                for (int k = 0; k < walk.size(); k++) {
                    for (LfBranch br : walk.get(k).getBranches()) {
                        if (!br.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC) || outaged.contains(br)
                                || br.isDisabled()) {
                            continue;
                        }
                        LfBus other = br.getBus1() == walk.get(k) ? br.getBus2() : br.getBus1();
                        if (other == null || islanded.contains(other) || other.isDisabled() || !walked.add(other)) {
                            continue;
                        }
                        walk.add(other);
                        if (alive.contains(other) && seen.add(other)) {
                            cluster.add(other);
                        }
                    }
                }
                if (cluster.contains(main.getControlledBus())) {
                    continue;                                // the base main keeps this cluster
                }
                var newMain = cluster.stream().map(controlOf::get)
                        .sorted(java.util.Comparator.<com.powsybl.openloadflow.network.VoltageControl<LfBus>>comparingDouble(
                                com.powsybl.openloadflow.network.VoltageControl::getTargetValue).reversed()
                                .thenComparing(vc -> vc.getControlledBus().getId()))
                        .findFirst().orElseThrow();
                boolean local = cluster.size() == 1
                        && newMain.getControllerElements().stream().allMatch(ce -> ce == newMain.getControlledBus());
                if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                    System.err.printf("QUALIFY_TRACE reHome ctg=%s main=%s cluster=%s newMain=%s target=%.6f local=%b%n",
                            contingencyId, main.getControlledBus().getId(), cluster.stream().map(LfBus::getId).toList(),
                            newMain.getControlledBus().getId(), newMain.getTargetValue(), local);
                }
                boolean remoteControllers = !local && cluster.stream().map(controlOf::get)
                        .flatMap(vc -> vc.getControllerElements().stream()).anyMatch(ce -> !cluster.contains(ce));
                if (remoteControllers) {
                    // The re-merged group is REMOTE: generators behind their own line or transformer regulate the
                    // cluster. OLF puts BUS_TARGET_V on the elected main, keeps that bus's reactive balance and frees
                    // the controllers' Q. The device hosts it on the base group's own table: the elected main's
                    // identity goes on its own row (the killed coupler's child, remoteMainIdentityEntries) and its
                    // reactive balance rides the first free member row through its DEPENDENT-PILOT pool
                    // (ExtractionColumns: the same member rows, that bus's reactive terms), which coincides with the
                    // base group's carrier pick on the same modes; each controller's reactive-limit check is re-pointed
                    // at the new pilot (ROW_REMOTE_REHOME). The base pilot's own balance has no carrier to write while
                    // its row is frozen (updateDistrQTargetsBatch), so nothing of the dead group is left in the system.
                    // network.xiidm busbar GOOD (GOPN_0 islanded, GOMV_1 re-homed onto GOPN_1), CKTZ, CXUJ, FUDD.
                    //
                    // Hosted when the base pilot is LOST and this cluster is every survivor of the group. A cluster
                    // splitting off BEHIND a surviving main (the base group keeps its pilot) is a second OLF group on
                    // the one table whose dependent pilot's pool would have to pick among that cluster's controllers
                    // only (a subgroup label the pools do not read yet): CPU, tagged.
                    boolean baseMainAlive = alive.contains(main.getControlledBus());
                    if (baseMainAlive || cluster.size() != alive.size() || !remoteMains.isEmpty()) {
                        if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                            System.err.printf("QUALIFY_TRACE reHome ctg=%s main=%s cluster=%s REMOTE controllers behind a surviving main -> CPU%n",
                                    contingencyId, main.getControlledBus().getId(), cluster.stream().map(LfBus::getId).toList());
                        }
                        return new ReHomed(pv, left, "VC_REMERGE_REMOTE_SUBGROUP", remoteMains, remoteCluster, subgroupOf,
                                reHomedRemoteControllers);
                    }
                    remoteMains.put(newMain.getControlledBus(), newMain.getTargetValue());
                    remoteCluster.addAll(cluster);
                    for (LfBus cb : cluster) {
                        for (LfBus ce : controlOf.get(cb).getControllerElements()) {
                            reHomedRemoteControllers.put(ce, newMain.getControlledBus());
                        }
                    }
                    for (LfBus cb : controlOf.keySet()) {   // the controllers the group LOST with the islanded sections
                        if (!alive.contains(cb)) {
                            for (LfBus ce : controlOf.get(cb).getControllerElements()) {
                                if (!left.contains(ce)) {
                                    left.add(ce);
                                }
                            }
                        }
                    }
                    continue;
                }
                if (!local) {
                    // A cluster of SEVERAL controls (or one with a remote controller) is a NEW remote group:
                    // OLF re-elects its main, rebuilds DISTR_Q over the cluster's controllers with keys
                    // renormalized over them (updateReactiveKeys: islanded -> 0) and puts BUS_TARGET_V on
                    // the new main. The device hosts that WITHOUT a new table when the cluster is EVERY
                    // survivor of the base group (one surviving cluster, the lost sections gone with the
                    // island): the lost controllers are INACTIVE members (their keys leave the denominator,
                    // islandedDistrQEntries / distrQWeightOverrides), the survivors' DISTR_Q rows stay plain
                    // sharing rows, and the new main's voltage identity is either the one the stock
                    // placement already installs on a killed coupler's child or an explicit ROW_V_IDENTITY
                    // (the caller decides, it holds the relocation map). Measured on rte6515's full busbar
                    // scope: 46 distinct refused re-homes, 45 with the elected main coupler-adjacent to the
                    // island, 31 of them exactly "pilot section lost, two sections survive" (tasks #69).
                    // Two or more surviving non-local clusters of ONE base group would be two new groups
                    // sharing one candidate table — still refused.
                    boolean wholeSurvivors = cluster.size() == alive.size();
                    if (!wholeSurvivors || !remoteMains.isEmpty()) {
                        // tasks #71: the base main keeps its own cluster (or another cluster already re-homed) and
                        // THIS cluster is a further OLF group on the same candidate table — a SUBGROUP: its members
                        // pair keys and q terms among themselves only, its elected main gets the identity, the base
                        // pilot's carrier is picked among label-0 members. Needs the group's candidate table
                        // (subgroupPairs checks every labelled member is enrolled in one).
                        int label = subgroupOf.values().stream().mapToInt(Integer::intValue).max().orElse(0) + 1;
                        for (LfBus b : cluster) {
                            subgroupOf.put(b, label);
                        }
                    }
                    remoteMains.put(newMain.getControlledBus(), newMain.getTargetValue());
                    remoteCluster.addAll(cluster);
                    for (LfBus cb : controlOf.keySet()) {   // the controllers the group LOST: the islanded/disabled
                        if (!alive.contains(cb)) {          // sections' own controllers leave the group's keys
                            for (LfBus ce : controlOf.get(cb).getControllerElements()) {
                                if (!left.contains(ce)) {
                                    left.add(ce);
                                }
                            }
                        }
                    }
                    continue;
                }
                pv.put(newMain.getControlledBus(), newMain.getTargetValue());
                for (LfBus ce : newMain.getControllerElements()) {
                    if (!left.contains(ce)) {
                        left.add(ce);
                    }
                }
                // A locally re-homed bus is its OWN OLF group too: label it so the base pilot's "enabled
                // controller" scan and the DISTR_Q rows of the main's side stop counting it (rte6515
                // BUS-6194_BBS_2: the re-homed VL-6194_2 unpinned, the kernel read it as a live controller of the
                // pinned main VL-6194_0 and restored the main's identity OLF has off — 4.775 kV).
                if (!alive.isEmpty() && alive.stream().anyMatch(b -> b == main.getControlledBus())) {
                    int label = subgroupOf.values().stream().mapToInt(Integer::intValue).max().orElse(0) + 1;
                    subgroupOf.put(newMain.getControlledBus(), label);
                }
            }
        }
        return new ReHomed(pv, left, null, remoteMains, remoteCluster, subgroupOf, reHomedRemoteControllers);
    }

    /** A plan refusal, counted in the census AND named by contingency id: the census counts are per attempt and
     *  carry no id, which made 19 refusals read as device failures (tasks #70). */
    /** Whether the generator's reactive limits are a capability CURVE (they then depend on its target P). */
    private static boolean hasCurve(com.powsybl.openloadflow.network.LfGenerator g) {
        return g instanceof com.powsybl.openloadflow.network.impl.AbstractLfGenerator alg
                && alg.getReactiveLimits().map(rl -> rl.getKind() == com.powsybl.iidm.network.ReactiveLimitsKind.CURVE)
                        .orElse(false);
    }

    /** Whether the generator is in the distributed-slack generator pack (the device then knows its own P). */
    private static boolean inGeneratorPack(GpuAcData gpuData, String genId) {
        return gpuData.dgGenId() != null && java.util.Arrays.asList(gpuData.dgGenId()).contains(genId);
    }

    private static void refuse(BatchContingency c, String tag) {
        com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record(tag);
        com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.planRefused(c.contingencyId(), tag);
    }


    /** tasks #71: the {dq member, label} pairs of a re-home with subgroups, or null (and a reject tag) when a
     *  labelled bus's controller is not a packed member or the candidate table's member order cannot be trusted. */
    private static int[] subgroupPairs(ReHomed reHomed, GpuAcData gpuData, String[] reject) {
        if (reHomed.subgroupOf().isEmpty()) {
            return null;
        }
        if (!qrMemberOrderConsistent(gpuData)) {
            reject[0] = "PLAN:VC_REMERGE_SUBGROUP_TABLE";
            return null;
        }
        List<Integer> out = new ArrayList<>();
        for (var e : reHomed.subgroupOf().entrySet()) {
            var vc = e.getKey().getGeneratorVoltageControl().orElse(null);
            List<LfBus> controllers = vc != null ? vc.getControllerElements() : List.of(e.getKey());
            for (LfBus ce : controllers) {
                int slot = GpuBatchInjectionOutages.dqMemberSlot(gpuData, ce);
                if (slot < 0) {
                    reject[0] = "PLAN:VC_REMERGE_SUBGROUP_UNPACKED";
                    return null;
                }
                if (gpuData.dqMemberQrGroup() == null || gpuData.dqMemberQrGroup()[slot] < 0) {
                    reject[0] = "PLAN:VC_REMERGE_SUBGROUP_TABLE";    // no candidate table: no per-scenario carrier pick
                    return null;
                }
                out.add(slot);
                out.add(e.getValue());
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    /** The kernels map candidate c of qr group g to dq member qrMemberStart[g] + c - 1: true when every packed
     *  member with a row sits at that candidate (merged controller order on both sides). */
    static boolean qrMemberOrderConsistent(GpuAcData gpuData) {
        int[] start = qrMemberStart(gpuData);
        int[] qrPtr = gpuData.qrPtr();
        int[] qrRows = gpuData.qrRows();
        int[] mRow = gpuData.dqMemberRow();
        byte[] mHasRow = gpuData.dqMemberHasRow();
        int[] mQrGroup = gpuData.dqMemberQrGroup();
        if (start == null || qrPtr == null || mRow == null || mQrGroup == null) {
            return false;
        }
        for (int m = 0; m < mQrGroup.length; m++) {
            int g = mQrGroup[m];
            if (g < 0 || mHasRow[m] == 0 || mRow[m] < 0) {
                continue;
            }
            int c = m - start[g] + 1;
            if (c < 1 || qrPtr[g] + c >= qrPtr[g + 1] || qrRows[qrPtr[g] + c] != mRow[m]) {
                return false;
            }
        }
        return true;
    }

    /** qr group -> its first dq member, -1 for a group without packed members. */
    static int[] qrMemberStart(GpuAcData gpuData) {
        int[] mQrGroup = gpuData.dqMemberQrGroup();
        int[] qrPtr = gpuData.qrPtr();
        if (mQrGroup == null || qrPtr == null || qrPtr.length < 2) {
            return null;
        }
        int[] start = new int[qrPtr.length - 1];
        java.util.Arrays.fill(start, -1);
        for (int m = 0; m < mQrGroup.length; m++) {
            int g = mQrGroup[m];
            if (g >= 0 && g < start.length && start[g] < 0) {
                start[g] = m;
            }
        }
        return start;
    }

    /** Drop the host-replayed weight overrides of a group the scenario SPLIT: the kernel's dynamic keys, now
     *  subgroup-aware, own those rows (a host weight computed over the whole base group would overwrite them). */
    private static double[][] withoutSplitGroups(double[][] ov, int[] subPairs, GpuAcData gpuData) {
        if (ov == null || subPairs == null) {
            return ov;
        }
        java.util.Set<Integer> labelled = new java.util.HashSet<>();
        for (int p = 0; p < subPairs.length; p += 2) {
            labelled.add(subPairs[p]);
        }
        java.util.Set<Integer> groups = new java.util.HashSet<>();
        for (int m : labelled) {
            groups.add(gpuData.dqMemberGroupStart()[m]);
        }
        List<Integer> keep = new ArrayList<>();
        for (int k = 0; k < ov[0].length; k++) {
            int c = (int) ov[0][k];
            int kM = gpuData.dqKMember() != null ? gpuData.dqKMember()[c] : -1;
            if (kM >= 0 && groups.contains(gpuData.dqMemberGroupStart()[kM])) {
                continue;
            }
            keep.add(k);
        }
        List<Integer> keepKey = new ArrayList<>();
        for (int k = 0; ov.length > 3 && k < ov[2].length; k++) {
            int m = (int) ov[2][k];
            if (!groups.contains(gpuData.dqMemberGroupStart()[m])) {
                keepKey.add(k);
            }
        }
        double[][] out = new double[4][];
        out[0] = new double[keep.size()];
        out[1] = new double[keep.size()];
        for (int i = 0; i < keep.size(); i++) {
            out[0][i] = ov[0][keep.get(i)];
            out[1][i] = ov[1][keep.get(i)];
        }
        out[2] = new double[keepKey.size()];
        out[3] = new double[keepKey.size()];
        for (int i = 0; i < keepKey.size(); i++) {
            out[2][i] = ov[2][keepKey.get(i)];
            out[3][i] = ov[3][keepKey.get(i)];
        }
        return out;
    }

    /** Row-mode-only GenOutage: the bus's V row becomes a VOLTAGE IDENTITY at {@code vTarget} (a re-homed control). */

    /**
     * The voltage identities a surviving REMOTE cluster's elected main still needs EXPLICITLY. The stock
     * placement (GpuBatchChunkSolver's killed-coupler block) already installs one on the CHILD of every
     * coupler the contingency opens — on rte6515 the elected main is that child in most accepted re-homes
     * ({@code blockCovers}); a main whose own row is NOT relocated (the tree's root, BUS-2640_BBS_1's
     * VL-2640_1) gets an explicit ROW_V_IDENTITY here. Emitting it in BOTH places would be a duplicate
     * override on one row (#62's race), so a main whose own row is the relocated child row of an OUTAGED
     * coupler is skipped. A main whose own row carries a LIVE coupler's ZERO_V (BUS-6029_BBS_1: the lost
     * pilot's neighbour has no control, the elected main is two couplers away) is refused: its identity
     * would displace that equation. Returns null when an identity cannot be hosted (no V row, frozen, that
     * relocation, or the bus is PINNED at a reactive limit — a pinned bus cannot also hold its voltage, see
     * the pvTargets loop above), with the reject tag in {@code reject[0]}.
     */
    private static List<GenOutage> remoteMainIdentityEntries(ReHomed reHomed, EquationSystem<AcVariableType, AcEquationType> es,
                                                             GpuAcData gpuData, java.util.Map<Integer, int[]> relocRows,
                                                             java.util.Set<Integer> outagedZiNums, java.util.Set<Integer> frozenSet,
                                                             String[] reject) {
        List<GenOutage> out = new ArrayList<>();
        for (var e : reHomed.remoteMainTargets().entrySet()) {
            var vVar = es.getVariableSet().getVariable(e.getKey().getNum(), AcVariableType.BUS_V);
            if (vVar == null || vVar.getRow() < 0 || frozenSet.contains(vVar.getRow())) {
                reject[0] = "PLAN:VC_REMERGE_NO_VROW";
                return null;
            }
            int[] reloc = relocRows.get(vVar.getRow());
            boolean blockCovers = reloc != null && outagedZiNums.contains(reloc[0]);
            int rl = rlEntryOf(gpuData, e.getKey());
            boolean basePinned = gpuData.rlBaseQLimitType() != null && rl >= 0 && gpuData.rlBaseQLimitType()[rl] != 0;
            if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                System.err.printf("QUALIFY_TRACE remoteMain bus=%s vRow=%d target=%.6f blockCovers=%b basePinned=%b%n",
                        e.getKey().getId(), vVar.getRow(), e.getValue(), blockCovers, basePinned);
            }
            // A main PINNED at its reactive limit in the base cannot hold a voltage identity in EITHER placement.
            // This test used to sit below the blockCovers exit, so the killed-coupler block installed the identity on
            // a pinned child and the RL kernel's "limit drift while pinned" branch then wrote the reactive limit as
            // that row's target: rte6515 BUS-6194_BBS_1, VL-6194_1 (base MAX_Q) held V = 0.6251 = maxQ - loadQ,
            // FAILED where OLF converges with 60 violations; BUS-6003_BBS_1 (VL-6003_1, base MAX_Q) 0.27 kV and
            // 0.457 MW. OLF has the pinned main PQ with BUS_TARGET_V off until it unpins — a per-scenario pilot the
            // device does not host yet (tasks #70), so the scenario is the CPU's.
            if (basePinned) {
                // The elected main is PQ in OLF. If EVERY controller of its cluster is pinned too, OLF's re-merged
                // group has no enabled controller: BUS_TARGET_V stays off and the cluster is plain PQ at the
                // limits — the stock placement's layout already (mode 4 rows at the pinned targets), so no
                // identity is installed; an UNPIN would re-activate BUS_TARGET_V on the main, which the device
                // cannot host for a lost pilot's table, so each member's entry forbids it (kernel reject, CPU
                // replay). rte6515 BUS-6194_BBS_1 (VL-6194_1/_2 both MAX_Q) and BUS-6003_BBS_1: OLF never
                // unpins them. A pinned main beside an ENABLED sibling (BUS_TARGET_V kept through the sibling)
                // is still refused.
                int mainLabel = reHomed.subgroupOf().getOrDefault(e.getKey(), 0);
                List<GenOutage> guards = new ArrayList<>();
                for (LfBus cb : reHomed.remoteClusterBuses()) {
                    if (reHomed.subgroupOf().getOrDefault(cb, 0) != mainLabel) {
                        continue;
                    }
                    var cvc = cb.getGeneratorVoltageControl().orElse(null);
                    List<LfBus> ctl = cvc != null ? cvc.getControllerElements() : List.of(cb);
                    for (LfBus ce : ctl) {
                        int ceRl = rlEntryOf(gpuData, ce);
                        if (!basePinned(gpuData, ceRl)) {
                            reject[0] = "PLAN:VC_REMERGE_PINNED_MAIN";
                            return null;
                        }
                        guards.add(noUnpinEntry(ceRl));
                    }
                }
                out.addAll(guards);
                continue;
            }
            if (blockCovers) {
                continue;
            }
            if (reloc != null) {
                // its own row carries a LIVE coupler's ZERO_V: the identity goes on the parent's DUMMY_Q row as
                // a cross identity (rte6515 BUS-6029_BBS_1/2: the elected main two couplers from the island)
                out.add(reHomedCrossEntry(reloc[1], e.getValue(), rl, false, 0));
                continue;
            }
            out.add(reHomedPvEntry(vVar.getRow(), e.getValue(), rl));
        }
        return out;
    }

    /** The reactive-limit entries of the REMOTE controllers OLF re-merges onto a new pilot ({@code reHomedControls}):
     *  each is checked against the new pilot's voltage and target this scenario. No row of theirs changes. Null (and a
     *  reject tag) when the new pilot has no V row to read. A controller without a reactive-limit entry (limits off)
     *  needs nothing: its row carries the new pilot's balance through the dependent-pilot pool regardless. */
    private static List<GenOutage> reHomedRemoteControllerEntries(ReHomed reHomed, EquationSystem<AcVariableType, AcEquationType> es,
                                                                  GpuAcData gpuData, String[] reject) {
        List<GenOutage> out = new ArrayList<>();
        for (var e : reHomed.reHomedRemoteControllers().entrySet()) {
            int rl = rlEntryOf(gpuData, e.getKey());
            if (rl < 0) {
                continue;
            }
            var pilotV = es.getVariableSet().getVariable(e.getValue().getNum(), AcVariableType.BUS_V);
            Double target = reHomed.remoteMainTargets().get(e.getValue());
            if (pilotV == null || pilotV.getRow() < 0 || target == null) {
                reject[0] = "PLAN:VC_REMERGE_NO_VROW";
                return null;
            }
            if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                System.err.printf("QUALIFY_TRACE reHomeRemoteCtl controller=%s rlEntry=%d pilot=%s pilotVRow=%d target=%.6f%n",
                        e.getKey().getId(), rl, e.getValue().getId(), pilotV.getRow(), target);
            }
            out.add(new GenOutage(-1, 0, -1, target, false, false,
                    -1, 0, 0, 0, 0, rl, 0, 0, false, java.util.Set.of(), 0, 0, 0, -1, new int[0],
                    new int[0], new double[0], GenOutage.ROW_REMOTE_REHOME, -1, 0, -1, 0, -1, 0, -1, -1, 0, false,
                    pilotV.getRow(), 0, 0));
        }
        return out;
    }

    private static GenOutage reHomedPvEntry(int vRow, double vTarget, int rlEntry) {
        return new GenOutage(-1, 0, vRow, vTarget, true, true,
                -1, 0, 0, 0, 0, rlEntry, 0, 0, false, java.util.Set.of(), 0, 0, 0, -1, new int[0],
                new int[0], new double[0], GenOutage.ROW_V_IDENTITY, -1, 0, -1, 0, -1, 0, -1, -1, 0, false, -1, 0, 0);
    }

    /** A re-homed control whose bus the BASE pinned: its own row starts as its reactive balance at the pinned
     *  value (mode 4, {@code qTarget} = LfBus.getTargetQ(), the frozen generation minus the load) and its entry is a
     *  LOCAL PV bus at {@code vTarget}, so an unpin restores the identity as OLF re-activates BUS_TARGET_V. */
    private static GenOutage reHomedPinnedPvEntry(int vRow, double qTarget, double vTarget, int rlEntry) {
        return new GenOutage(-1, 0, vRow, qTarget, true, true,
                -1, 0, 0, 0, 0, rlEntry, 0, 0, false, java.util.Set.of(), 0, 0, 0, -1, new int[0],
                new int[0], new double[0], GenOutage.ROW_PINNED_V, -1, vTarget, -1, 0, -1, 0, -1, -1, 0, false, -1, 0, 0);
    }

    /** A re-homed control whose own V row hosts a LIVE coupler's ZERO_V (tasks #70, VC_REMERGE_RELOC_VROW): its
     *  identity is a CROSS identity on the parent coupler's DUMMY_Q row (mode 6, the bus's V column), or — base
     *  pinned — that row starts as the balance at the pinned value and the entry unpins back to the cross form. */
    private static GenOutage reHomedCrossEntry(int dummyQRow, double vTarget, int rlEntry, boolean pinned, double qTarget) {
        return pinned ? reHomedPinnedPvEntry(dummyQRow, qTarget, vTarget, rlEntry)
                : new GenOutage(-1, 0, dummyQRow, vTarget, true, true,
                -1, 0, 0, 0, 0, rlEntry, 0, 0, false, java.util.Set.of(), 0, 0, 0, -1, new int[0],
                new int[0], new double[0], GenOutage.ROW_CROSS_V_IDENTITY, -1, 0, -1, 0, -1, 0, -1, -1, 0, false, -1, 0, 0);
    }

    /** A base-pinned member of a re-homed cluster with no enabled controller: no row change, unpin forbidden. */
    private static GenOutage noUnpinEntry(int rlEntry) {
        return new GenOutage(-1, 0, -1, 0, false, false,
                -1, 0, 0, 0, 0, rlEntry, 0, 0, false, java.util.Set.of(), 0, 0, 0, -1, new int[0],
                new int[0], new double[0], GenOutage.ROW_PINNED_NO_UNPIN, -1, 0, -1, 0, -1, 0, -1, -1, 0, false, -1, 0, 0);
    }

    static boolean basePinned(GpuAcData gpuData, int rlEntry) {
        return gpuData.rlBaseQLimitType() != null && rlEntry >= 0 && gpuData.rlBaseQLimitType()[rlEntry] != 0;
    }

    /** The bus's reactive-limit entry, or -1: a re-homed bus is checked as a LOCAL controller by the device
     *  (OLF checks it as the PV bus it now is; BUS-6113_BBS_1 pinned VL-6113_1 at RL#1, the device never
     *  looked — 7.1 kV). */
    static int rlEntryOf(GpuAcData gpuData, LfBus bus) {
        int[] rlBusNum = gpuData.rlBusNum();
        if (rlBusNum != null) {
            for (int e = 0; e < rlBusNum.length; e++) {
                if (rlBusNum[e] == bus.getNum()) {
                    return e;
                }
            }
        }
        return -1;
    }

    static int deadControlRowMode(GpuAcData gpuData, LfBus survivor) {
        int slot = GpuBatchInjectionOutages.dqMemberSlot(gpuData, survivor);
        if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
            System.err.println("QUALIFY_TRACE deadControlRowMode survivor=" + survivor.getId() + " dqMemberSlot=" + slot
                    + " -> mode " + (slot >= 0 ? 4 : 1));
        }
        return slot >= 0 ? 4 : 1;
    }

    /**
     * The relocated coupler children, keyed by the child's own {@code BUS_V} variable row:
     * {@code {coupler branch num, the coupler's DUMMY_Q variable row}}. The coupler-tree relocation moved that
     * bus's reactive balance (and its {@code BUS_TARGET_Q}) onto the coupler's dummy row and put the coupler's
     * {@code ZERO_V} on the bus's own row, so a per-scenario flip of "solve your own reactive balance at this
     * target" has to land on the dummy row — the native reactive-limit pass makes exactly the same diversion
     * (see {@code rlQrow}) — and a flip to a VOLTAGE identity has nowhere to land at all.
     */
    static java.util.Map<Integer, int[]> relocatedChildRows(GpuAcData gpuData) {
        java.util.Map<Integer, int[]> out = new java.util.HashMap<>();
        int[] zi = gpuData.ziIdx();
        int[] nums = gpuData.ziBranchNums();
        for (int e = 0; e < zi.length / 11 && e < nums.length; e++) {
            int o = e * 11;
            int childSide = (zi[o + 10] >> 1) & 3;
            int ownV = childSide == 1 ? zi[o + 6] : (childSide == 2 ? zi[o + 7] : -1);
            if (ownV >= 0 && zi[o + 5] >= 0) {
                out.put(ownV, new int[] {nums[e], zi[o + 5]});
            }
        }
        return out;
    }

    /** Where a per-scenario reactive flip lands: the bus's own {@code BUS_V} row, or its parent coupler's
     *  {@code DUMMY_Q} row while that coupler stays closed. An OUTAGED coupler puts the equations back on the
     *  bus's own rows (the stock placement the chunk solver switches to), so the own row is right there. */
    private static int relocQRow(int vRow, java.util.Map<Integer, int[]> relocRows, java.util.Set<Integer> outagedZiNums) {
        int[] reloc = relocRows.get(vRow);
        return reloc == null || outagedZiNums.contains(reloc[0]) ? vRow : reloc[1];
    }

    /** The INVERSE of {@link #relocatedChildRows}, keyed by the coupler's {@code DUMMY_Q} row:
     *  {@code {branchNum, the child bus's own BUS_V row}}. A consumer holding a STRUCTURAL member row
     *  (which the coupler-tree relocation already moved onto the dummy) needs this to map back, because
     *  a scenario that OUTAGES the coupler puts the child's reactive equation on the bus's own row. */
    static java.util.Map<Integer, int[]> relocatedDummyQRows(GpuAcData gpuData) {
        java.util.Map<Integer, int[]> out = new java.util.HashMap<>();
        int[] zi = gpuData.ziIdx();
        int[] nums = gpuData.ziBranchNums();
        for (int e = 0; e < zi.length / 11 && e < nums.length; e++) {
            int o = e * 11;
            int childSide = (zi[o + 10] >> 1) & 3;
            int ownV = childSide == 1 ? zi[o + 6] : (childSide == 2 ? zi[o + 7] : -1);
            if (ownV >= 0 && zi[o + 5] >= 0) {
                out.put(zi[o + 5], new int[] {nums[e], ownV});
            }
        }
        return out;
    }

    /** The row a consumer holding the structural (already relocated) row must write in THIS scenario:
     *  the child bus's own row once the coupler carrying its equations is outaged, else unchanged. */
    static int unrelocQRow(int row, java.util.Map<Integer, int[]> dummyQRows, java.util.Set<Integer> outagedZiNums) {
        int[] reloc = dummyQRows.get(row);
        return reloc != null && outagedZiNums.contains(reloc[0]) ? reloc[1] : row;
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

    /**
     * {@code OLF_GPU_LOSS_TRACE=<contingency id>}: the device's SCHEDULED active power loss and the islanded
     * bus set it was computed from. The formula mirrors {@code LfContingency}'s
     * {@code disconnectedGenerationActivePower - disconnectedLoadActivePower} term for term, but only if this
     * islanded set equals OLF's {@code disabledNetwork.getBuses()} — an islanded bus contributes its WHOLE
     * injection and is skipped in the named lost load/generator sums, so a set that differs by one bus moves
     * the loss by that bus's (generation - load). Compare against the {@code disabled buses:} line that
     * {@code OLF_GPU_EQ_DUMP_CTG} prints.
     */
    private static void traceScheduledLoss(BatchContingency c, List<LfBus> islandedBuses, double loss) {
        String want = System.getenv("OLF_GPU_LOSS_TRACE");
        if (want == null || !want.equals(c.contingencyId())) {
            return;
        }
        System.err.printf("LOSS_TRACE ctg=%s deviceScheduledLoss=%.9f pu islanded=%s lostGens=%s lostLoads=%s%n",
                c.contingencyId(), loss,
                islandedBuses.stream().map(LfBus::getId).sorted().toList(),
                c.lostGenerators().stream().map(com.powsybl.openloadflow.network.LfGenerator::getId).sorted()
                        .toList(),
                c.lostLoads().stream().map(ll -> ll.bus() == null ? "?" : ll.bus().getId()).sorted().toList());
    }

    /** The surviving converter station's injection when this contingency islands the other endpoint of its link.
     *  AC emulation: OLF's station reports targetP 0 once a bus of its link is disabled, so a saturated or frozen
     *  link's survivor always loses its injection (a linear one has none). SETPOINT links follow
     *  {@code LfContingency.processHvdcsWithoutPower}: when the islanded converter bus is ISOLATED in OLF's sense
     *  ({@code Networks.isIsolatedBusForHvdc}: a one-bus component after the cut, no non-fictitious load, no
     *  generator other than converter stations) and the other endpoint survives, OLF zeroes BOTH stations' targetP.
     *  The islanded station leaves with its frozen bus (its scheduled P is already out of the loss,
     *  {@link #scheduledActivePowerLoss} mirrors {@code getBusLostGenerationTargetP}); the SURVIVING station's
     *  injection is what changes on the live side, and it rides the plain injection channel as generation
     *  removed - a negative load loss on its bus, composed with whatever else that bus loses. A link whose islanded
     *  bus is NOT isolated (the island also holds a load or a generator, or more than one bus) keeps its power in
     *  OLF: the survivor goes on injecting with its counterpart gone, and the distributed slack absorbs it on both
     *  arms - nothing to do here. network.xiidm EWVZ/EWWB/ZJI/ZJJ: a section loss that opens the station's own
     *  breaker (a converter station is not an end node for the tripping), leaving EWVX_5 / EWVX_4 / ZJH_8 / ZJH_9
     *  alone with their station - refused as ISLANDS_REMOTE_VC_MEMBER until this. */
    static List<LoadLoss> hvdcStationResets(List<LfBus> islandedBuses, BatchContingency c) {
        List<LoadLoss> out = new ArrayList<>();
        java.util.Set<LfBus> islanded = new java.util.HashSet<>(islandedBuses);
        for (LfBus bus : islandedBuses) {
            for (com.powsybl.openloadflow.network.LfHvdc h : bus.getHvdcs()) {
                // AC emulation: a SATURATED or FROZEN link carries its power as station targetP (a LINEAR one as an
                // equation term, masked with the link), and LfVscConverterStationImpl.getTargetP returns 0 for an
                // AC-emulation station as soon as either endpoint bus is disabled - isolated or not. A setpoint link
                // loses its power only through processHvdcsWithoutPower, i.e. when the islanded bus is isolated.
                LfBus other = h.getBus1() == bus ? h.getBus2() : h.getBus1();
                if (other == null || islanded.contains(other) || other.isDisabled()
                        || !h.isAcEmulation() && !isolatedForHvdc(bus, c)) {
                    continue;
                }
                com.powsybl.openloadflow.network.LfGenerator survivor = h.getBus1() == other
                        ? h.getConverterStation1() : h.getConverterStation2();
                double p = survivor.getTargetP();
                if (p != 0) {
                    out.add(new LoadLoss(other, -p, 0));   // generation removed: the row target drops by p
                }
                if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                    System.err.printf("QUALIFY_TRACE hvdc %s without power in %s: station %s on %s targetP %.6f -> 0%n",
                            h.getId(), c.contingencyId(), survivor.getId(), other.getId(), p);
                }
            }
        }
        return out;
    }

    /** {@code Networks.isIsolatedBusForHvdc} on the post-contingency connectivity: OLF removes every branch of the
     *  contingency from the graph (closed outages, opened couplers, side-opened branches alike) and asks whether the
     *  bus is alone in its component. */
    private static boolean isolatedForHvdc(LfBus bus, BatchContingency c) {
        if (bus.getNonFictitiousLoadTargetP() != 0.0
                || bus.getGenerators().stream().anyMatch(com.powsybl.openloadflow.network.impl.LfGeneratorImpl.class::isInstance)) {
            return false;
        }
        java.util.Set<LfBranch> cut = new java.util.HashSet<>(c.outagedBranches());
        cut.addAll(c.outagedZiBranches());
        for (BatchContingency.SideOpen so : c.sideOpenBranches()) {
            cut.add(so.branch());
        }
        for (LfBranch br : bus.getBranches()) {
            if (!br.isDisabled() && br.isConnectedAtBothSides() && !cut.contains(br)) {
                return false;                                // a live branch: the component is bigger than the bus
            }
        }
        return true;
    }

    private static double scheduledActivePowerLoss(BatchContingency c, List<LfBus> islandedBuses) {
        double loss = 0;
        // An ISLANDED bus contributes its WHOLE injection below (generation minus load), so anything
        // the contingency also names individually on that bus must not be added again. A busbar
        // outage does exactly that: the propagation names the bus's load as a lost load AND the
        // connectivity pass finds the bus islanded, and charging both doubled the scheduled loss —
        // rte6515 BUS-1004_BBS pre-distributed -1.110 pu for a 0.555 pu load. The device's DS loop
        // then claws most of it back during the solve and lands at a different point inside the
        // slack band, which is why the error was a fixed ~0.13% of the lost load rather than the
        // whole of it. A plain load contingency islands nothing, so loads alone were never wrong.
        java.util.Set<Integer> islandedNums = new java.util.HashSet<>();
        for (LfBus bus : islandedBuses) {
            islandedNums.add(bus.getNum());
        }
        for (com.powsybl.openloadflow.network.LfGenerator g : c.lostGenerators()) {
            if (g.getBus() != null && islandedNums.contains(g.getBus().getNum())) {
                continue;
            }
            loss += g.getTargetP();
        }
        for (LoadLoss ll : c.lostLoads()) {
            if (ll.bus() != null && islandedNums.contains(ll.bus().getNum())) {
                continue;
            }
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
        java.util.Map<Integer, int[]> relocRows = relocatedChildRows(gpuData);
        java.util.Map<Integer, int[]> relocDummyQRows = relocatedDummyQRows(gpuData);
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
                    .anyMatch(m -> hasCurve(m.generator()) && !inGeneratorPack(gpuData, m.generator().getId()))) {
                refuse(c, "PLAN:PAIR_ACTION_UNTRACKED_CURVE");
                continue;
            }
            // ...and re-checks its participation at the new target, which the device applies to the generator pack.
            // A participant outside a non-empty pack (its bus has no active P balance row) has nowhere to sit out. (In
            // a slack-only distribution there is no pack, and any action there moves the slack bus: SLACK_BUS.)
            if (gpuData.dgGenId() != null && gpuData.dgGenId().length > 0 && c.generatorActions().stream()
                    .anyMatch(m -> dsParams.factor(m.generator()) != 0 && !inGeneratorPack(gpuData, m.generator().getId()))) {
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
                liveLostLoads.addAll(hvdcStationResets(q.islandedBuses(), c));   // a setpoint link losing its other end
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
                    double[][] drOv = distrRhoOverrides(es, gpuData, distrRhoControllers, c.outagedBranches(), incrementalTvc);
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
                        && reHomedControls(List.of(), c.outagedZiBranches(), c.contingencyId(), es).changesControls();
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
                    VcSplit vcSplit = classifyVcSplit(es, q.islandedBuses(), lostVcControllers, deadControlBuses,
                            islandedHvdcs);
                    ReHomed reHomed = reHomedControls(q.islandedBuses(), c.outagedZiBranches(), c.contingencyId(), es);
                    if (reHomed.reject() != null) {
                        refuse(c, "PLAN:" + reHomed.reject());
                        continue;
                    }
                    String[] subReject = new String[1];
                    dqSubByIndex[i] = subgroupPairs(reHomed, gpuData, subReject);
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
                    if (vcSplit == VcSplit.UNMODELED) {
                        if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                            System.err.printf("QUALIFY_TRACE reject %s tag=ISLANDS_REMOTE_VC_MEMBER site=island split=%s"
                                    + " islandedHvdcs=%d lostControllers=%s%n", c.contingencyId(), vcSplit, islandedHvdcs.size(),
                                    lostVcControllers.stream().map(LfBus::getId).toList());
                        }
                        refuse(c, "PLAN:ISLANDS_REMOTE_VC_MEMBER");
                        continue;
                    }
                    FrozenIsland fi = islandedFrozen(es, q.islandedBuses(), relocatedCouplers);
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
                            int qFlipRow = relocQRow(vVar.getRow(), relocRows, outagedZiNums);
                            if (frozenSet.contains(qFlipRow)) {
                                live = null;
                                break;
                            }
                            live.add(new GenOutage(-1, 0, qFlipRow, sb.getTargetQ(), true, true,
                                    -1, 0, 0, 0, 0, -1, 0, 0, false, java.util.Set.of(), 0, 0, 0, -1, new int[0],
                                    new int[0], new double[0], deadControlRowMode(gpuData, sb), -1, 0, -1, 0, -1, 0, -1, -1, 0, false, -1, 0, 0));
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
                            int reHomedRl = rlEntryOf(gpuData, e.getKey());
                            int crossRow = relocQRow(vVar.getRow(), relocRows, outagedZiNums);
                            if (crossRow != vVar.getRow()) {
                                // its own row carries a LIVE coupler's ZERO_V: cross identity on the parent's
                                // DUMMY_Q row (rte6515 BUS-6014_BBS_1/2 VL-6014_3, BUS-6029_BBS_4 VL-6029_2)
                                live.add(reHomedCrossEntry(crossRow, e.getValue(), reHomedRl,
                                        basePinned(gpuData, reHomedRl), e.getKey().getTargetQ()));
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
                            if (basePinned(gpuData, reHomedRl)) {
                                // Base-pinned: start as the reactive balance at the pinned value, checked as a
                                // LOCAL PV bus at the re-homed voltage target (ROW_PINNED_V). rte6515
                                // BUS-6157_BBS_1 (unpins, re-pins MAX_Q), BUS-6194_BBS_2 (unpins), BUS-6229_BBS_1.
                                live.add(reHomedPinnedPvEntry(vVar.getRow(), e.getKey().getTargetQ(), e.getValue(), reHomedRl));
                                continue;
                            }
                            live.add(reHomedPvEntry(vVar.getRow(), e.getValue(), reHomedRl));
                        }
                    }
                    String[] remoteReject = new String[1];
                    if (live != null && !reHomed.remoteMainTargets().isEmpty()) {
                        List<GenOutage> remoteIds = remoteMainIdentityEntries(reHomed, es, gpuData, relocRows,
                                outagedZiNums, frozenSet, remoteReject);
                        if (remoteIds == null) {
                            live = null;
                        } else {
                            live.addAll(remoteIds);
                            List<GenOutage> reHomedCtl = reHomedRemoteControllerEntries(reHomed, es, gpuData, remoteReject);
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
                    islandedDsByIndex[i] = islandedDsParticipants(fi.rows(), gpuData, q.islandedBuses());
                    islandedRlByIndex[i] = islandedRlControllers(fi.rows(), gpuData);
                    if (fi.rows().length > 0) {
                        islandingCount++;
                    }
                } else {
                    frozenRowsByIndex[i] = NO_FROZEN_ROWS;
                    frozenTargetsByIndex[i] = NO_FROZEN_TARGETS;
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
                    double[][] ov = withoutSplitGroups(distrQWeightOverrides(es, gpuData, goneForKeys), dqSubByIndex[i], gpuData);
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
                activePowerLossByIndex[i] = scheduledActivePowerLoss(c, q.islandedBuses());
                traceScheduledLoss(c, q.islandedBuses(), activePowerLossByIndex[i]);
                batchedIndices.add(i);
            } else if (!c.outagedShunts().isEmpty()) {       // fixed-shunt disconnection: mask the shunt element
                if (canBatchShuntOutage(c.outagedShunts(), elementMapper, gpuData)) {
                    batchedIndices.add(i);
                    frozenRowsByIndex[i] = NO_FROZEN_ROWS;
                    frozenTargetsByIndex[i] = NO_FROZEN_TARGETS;
                }                                            // else: not a packed fixed shunt / DISTR_Q contributor → CPU
            } else {
                List<LfBus> lostVcControllers = new ArrayList<>();
                List<LfBus> deadControlBuses = new ArrayList<>();
                List<com.powsybl.openloadflow.network.LfHvdc> islandedHvdcs = new ArrayList<>();
                VcSplit vcSplit = classifyVcSplit(es, q.islandedBuses(), lostVcControllers, deadControlBuses,
                        islandedHvdcs);
                ReHomed reHomed = reHomedControls(q.islandedBuses(), c.outagedZiBranches(), c.contingencyId(), es);
                if (reHomed.reject() != null) {
                    refuse(c, "PLAN:" + reHomed.reject());
                    continue;
                }
                String[] subReject = new String[1];
                dqSubByIndex[i] = subgroupPairs(reHomed, gpuData, subReject);
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
                if (!reHomed.leftControllers().isEmpty() && vcSplit == VcSplit.CONTAINED) {
                    vcSplit = VcSplit.RENORMALIZE;           // the group goes on without them
                }
                if (!islandedHvdcs.isEmpty()) {
                    islandedHvdcsByIndex[i] = islandedHvdcs;
                }
                if (vcSplit == VcSplit.UNMODELED) {
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
                        vcOff.add(new GenOutage(-1, 0, relocQRow(vVar.getRow(), relocRows, outagedZiNums),
                                sb.getTargetQ(), true, true,
                                -1, 0, 0, 0, 0, -1, 0, 0, false, java.util.Set.of(), 0, 0, 0, -1, new int[0],
                                new int[0], new double[0], deadControlRowMode(gpuData, sb), -1, 0, -1, 0, -1, 0, -1, -1, 0, false, -1, 0, 0));
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
                List<LoadLoss> stationResets = hvdcStationResets(q.islandedBuses(), c);
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
                        int reHomedRl = rlEntryOf(gpuData, e.getKey());
                        int crossRow = relocQRow(vVar.getRow(), relocRows, outagedZiNums);
                        if (crossRow != vVar.getRow()) {
                            pvOn.add(reHomedCrossEntry(crossRow, e.getValue(), reHomedRl,
                                    basePinned(gpuData, reHomedRl), e.getKey().getTargetQ()));
                            continue;
                        }
                        // Same rule as the island site above: a bus PINNED at its reactive limit in the base
                        // cannot hold a voltage identity (OLF has it PQ). This site lacked the check, so a
                        // middle-section loss whose surviving section is base-pinned batched with the
                        // identity: rte6515 BUS-6157_BBS_1 (VL-6157_1, baseLimitType=1), BUS-6194_BBS_2
                        // (VL-6194_2, 2) and BUS-6229_BBS_1 (VL-6229_1, 2) — three device failures on
                        // systems OLF converges, replayed on the CPU.
                        if (basePinned(gpuData, reHomedRl)) {
                            pvOn.add(reHomedPinnedPvEntry(vVar.getRow(), e.getKey().getTargetQ(), e.getValue(), reHomedRl));
                            continue;
                        }
                        pvOn.add(reHomedPvEntry(vVar.getRow(), e.getValue(), reHomedRl));
                    }
                    if (!ok) {
                        refuse(c, 
                                pinnedMain ? "PLAN:VC_REMERGE_PINNED_MAIN"
                                        : relocIdentity ? "PLAN:VC_REMERGE_RELOC_VROW" : "PLAN:VC_REMERGE_NO_VROW");
                        continue;
                    }
                    if (!reHomed.remoteMainTargets().isEmpty()) {
                        String[] remoteReject = new String[1];
                        List<GenOutage> remoteIds = remoteMainIdentityEntries(reHomed, es, gpuData, relocRows,
                                outagedZiNums, java.util.Set.of(), remoteReject);
                        List<GenOutage> reHomedCtl = remoteIds == null ? null
                                : reHomedRemoteControllerEntries(reHomed, es, gpuData, remoteReject);
                        if (remoteIds == null || reHomedCtl == null) {
                            refuse(c, remoteReject[0]);
                            continue;
                        }
                        pvOn.addAll(remoteIds);
                        pvOn.addAll(reHomedCtl);
                    }
                    genOutagesByIndex[i] = pvOn.toArray(new GenOutage[0]);
                }
                if (vcSplit == VcSplit.RENORMALIZE) {
                    // The group survives without its islanded controller(s). Renormalize its reactive
                    // keys over the survivors through the SAME channel a generator outage uses: the
                    // islanded bus's generators are gone from this synchronous component, which is
                    // what distrQWeightOverrides replays.
                    List<com.powsybl.openloadflow.network.LfGenerator> goneControllers = lostVcControllers.stream()
                            .flatMap(b -> b.getGenerators().stream())
                            .toList();
                    double[][] vcOv = withoutSplitGroups(distrQWeightOverrides(es, gpuData, goneControllers), dqSubByIndex[i], gpuData);
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
                double[][] drOv = distrRhoOverrides(es, gpuData, distrRhoControllers, c.outagedBranches(), incrementalTvc);
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
                FrozenIsland fi = islandedFrozen(es, q.islandedBuses(), relocatedCouplers);
                // A contingency that BOTH splits the network AND strands a distributed-slack participant.
                // OLF distributes over getSynchronousNetworks().getFirst(), so a stranded participant is
                // not a participant at all: its share must not be computed, and the factors of the ones
                // that remain must be renormalized WITHOUT it. Masking it out of the distribution loops
                // alone is not enough and was reverted once for exactly that reason — the slack keeps the
                // retained share it had while the stranded bus still counted, and the pass retires with
                // most of the mismatch undistributed. The device now takes the per-scenario participant
                // set and derives BOTH the shares and the stopping gate from it.
                islandedDsByIndex[i] = islandedDsParticipants(fi.rows(), gpuData, q.islandedBuses());
                islandedRlByIndex[i] = islandedRlControllers(fi.rows(), gpuData);
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
                activePowerLossByIndex[i] = scheduledActivePowerLoss(c, q.islandedBuses());
                traceScheduledLoss(c, q.islandedBuses(), activePowerLossByIndex[i]);
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
