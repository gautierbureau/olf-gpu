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
                        return VcSplit.UNMODELED;
                    }
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
                    toggled.add(g);
                    touched.add(g.getBus());
                }
            }
            if (toggled.isEmpty()) {
                return new double[][] {new double[0], new double[0]};
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
            GpuAcData nd = com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor
                    .extract(network, es, false, false);
            double[] newW = nd.dqWeight();
            double[] baseW = gpuData.dqWeight();
            if (newW.length != baseW.length) {
                return null;                                 // structural drift — not a pure reweight
            }
            List<Double> idx = new ArrayList<>();
            List<Double> w = new ArrayList<>();
            boolean dqOvTrace = System.getenv("OLF_GPU_DQOV_TRACE") != null;
            int traced = 0;
            for (int k = 0; k < newW.length; k++) {
                if (newW[k] != baseW[k]) {
                    idx.add((double) k);
                    w.add(newW[k]);
                    if (dqOvTrace && traced++ < 12) {        // the two packs' metadata at a differing index
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
                w.stream().mapToDouble(Double::doubleValue).toArray()};
        } finally {
            for (var g : toggled) {
                g.setGeneratorControlType(com.powsybl.openloadflow.network.LfGenerator.GeneratorControlType.VOLTAGE);
            }
            for (LfBus bus : touched) {
                updateMainReactiveKeys(bus);
            }
        }
    }

    /** OLF's own key update: on the group's MAIN control, never on a member's dependent one. */
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
    record ReHomed(java.util.LinkedHashMap<LfBus, Double> pvTargets, List<LfBus> leftControllers, String reject) {
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
    static ReHomed reHomedControls(List<LfBus> islandedBuses, List<LfBranch> outagedZiBranches) {
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
                List<LfBus> cluster = new ArrayList<>(List.of(start));   // BFS over the surviving zero-impedance edges
                for (int k = 0; k < cluster.size(); k++) {
                    for (LfBranch br : cluster.get(k).getBranches()) {
                        if (!br.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC) || outaged.contains(br)
                                || br.isDisabled()) {
                            continue;
                        }
                        LfBus other = br.getBus1() == cluster.get(k) ? br.getBus2() : br.getBus1();
                        if (other != null && alive.contains(other) && seen.add(other)) {
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
                    System.err.printf("QUALIFY_TRACE reHome main=%s cluster=%s newMain=%s target=%.6f local=%b%n",
                            main.getControlledBus().getId(), cluster.stream().map(LfBus::getId).toList(),
                            newMain.getControlledBus().getId(), newMain.getTargetValue(), local);
                }
                if (!local) {
                    return new ReHomed(pv, left, "VC_REMERGE_MULTI");
                }
                pv.put(newMain.getControlledBus(), newMain.getTargetValue());
                for (LfBus ce : newMain.getControllerElements()) {
                    if (!left.contains(ce)) {
                        left.add(ce);
                    }
                }
            }
        }
        return new ReHomed(pv, left, null);
    }

    /** Row-mode-only GenOutage: the bus's V row becomes a VOLTAGE IDENTITY at {@code vTarget} (a re-homed control). */
    private static GenOutage reHomedPvEntry(int vRow, double vTarget, int rlEntry) {
        return new GenOutage(-1, 0, vRow, vTarget, true, true,
                -1, 0, 0, 0, 0, rlEntry, 0, 0, false, java.util.Set.of(), 0, 0, 0, -1, new int[0],
                new int[0], new double[0], GenOutage.ROW_V_IDENTITY, -1, 0, -1, 0, -1, 0);
    }

    /** The bus's reactive-limit entry, or -1: a re-homed bus is checked as a LOCAL controller by the device
     *  (OLF checks it as the PV bus it now is; BUS-6113_BBS_1 pinned VL-6113_1 at RL#1, the device never
     *  looked — 7.1 kV). */
    private static int rlEntryOf(GpuAcData gpuData, LfBus bus) {
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
                        List<com.powsybl.openloadflow.network.LfHvdc>[] islandedHvdcsByIndex) {
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
            Set<LfBranch> relocatedCouplers, int[] curvePRows) {
        List<Integer> batchedIndices = new ArrayList<>();
        java.util.Map<Integer, int[]> relocRows = relocatedChildRows(gpuData);
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
                // The validator's OWN reason, not a fixed tag: this site is reached by every disqualified
                // verdict, so recording "SIDE_OPEN_UNPACKED" for all of them misreports the census — the one
                // entry at rte6515's full scope is an ISLANDS_SLACK outage, which reads as a side-open shape.
                com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:"
                        + (q.reason() != null ? q.reason() : "UNQUALIFIED"));
                continue;
            }
            BatchContingency c = contingencies.get(i);
            java.util.Set<Integer> outagedZiNums = c.outagedZiBranches().stream()
                    .map(LfBranch::getNum).collect(java.util.stream.Collectors.toSet());
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
                java.util.Set<Integer> islandedNumSet = new java.util.HashSet<>();
                for (LfBus ib : q.islandedBuses()) {
                    islandedNumSet.add(ib.getNum());
                }
                if (!c.shuntShifts().isEmpty()
                        && !canBatchShuntOutage(c.shuntShifts().stream().map(ShuntShift::shunt).toList(), elementMapper, gpuData)) {
                    com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:SHUNT_UNPACKED");
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
                List<LoadLoss> liveLostLoads = c.lostLoads().stream()
                        .filter(ll -> ll.bus() == null || !islandedNumSet.contains(ll.bus().getNum()))
                        .toList();
                GenOutage[] gos = GpuBatchInjectionOutages.computeInjectionOutages(es, gpuData, dsSum, slackParticipates,
                        dsParams, liveLostGens, liveLostLoads);
                if (gos == null) {                           // slack / >1 DS bus / remote gen — route to CPU
                    // (was tagged SHUNT_SHAPE, a copy-paste from the block above: this is the INJECTION
                    // outage shape, and the census is how we pick what to fix — a tag that lies about
                    // the reason sends the next person at the wrong gate.)
                    com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:INJECTION_OUTAGE_SHAPE");
                    if (System.getenv("OLF_GPU_GENOUT_REJECT") != null) {
                        // The exit named itself and its bus a moment ago; name the CONTINGENCY too, so
                        // the reason, the bus and the id that reproduces it are on adjacent lines.
                        System.err.println("GENOUT_REJECT   ctg=" + c.contingencyId());
                    }
                    continue;
                }
                // Curve-limits channel active: its p_now linkage models the DS delta ONLY. An injection
                // outage at a curve bus's P row used to skew the evaluated limits, and the chunk solver
                // now repairs that per scenario (the lost entry, the outage's own row step, and the
                // survivors' renormalized shares). A strategy PAIR still cannot: its anchor shift moves
                // EVERY distributed-slack row, so there is no single bus step to take back out — CPU.
                if (curvePRows != null && c.contingencyId().indexOf(GpuBatchStatus.STRATEGY_KEY_SEP) >= 0) {
                    com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:PAIR_CURVE_LINKAGE");
                    continue;                                // same reason as the other arm: one tag
                }
                List<LfBus> islandLostVcControllers = new ArrayList<>();   // filled by the island-injection path
                if (!c.outagedBranches().isEmpty()) {        // COMBO: the branch part must also qualify
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
                }
                if (!q.islandedBuses().isEmpty()) {
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
                    ReHomed reHomed = reHomedControls(q.islandedBuses(), c.outagedZiBranches());
                    if (reHomed.reject() != null) {
                        com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:" + reHomed.reject());
                        continue;
                    }
                    deadControlBuses.removeAll(reHomed.pvTargets().keySet());   // re-homed, not dead
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
                        com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:ISLANDS_REMOTE_VC_MEMBER");
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
                                    new int[0], new double[0], deadControlRowMode(gpuData, sb), -1, 0, -1, 0, -1, 0));
                        }
                        if (live == null) {
                            if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                                System.err.printf("QUALIFY_TRACE reject %s tag=ISLANDS_REMOTE_VC_MEMBER site=island-deadControl-frozenV"
                                        + " deadControl=%s%n", c.contingencyId(), deadControlBuses.stream().map(LfBus::getId).toList());
                            }
                            com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:ISLANDS_REMOTE_VC_MEMBER");
                            continue;
                        }
                    }
                    boolean relocIdentity = false;
                    if (live != null) {
                        for (var e : reHomed.pvTargets().entrySet()) {
                            var vVar = es.getVariableSet().getVariable(e.getKey().getNum(), AcVariableType.BUS_V);
                            if (vVar == null || vVar.getRow() < 0 || frozenSet.contains(vVar.getRow())) {
                                live = null;
                                break;
                            }
                            if (relocQRow(vVar.getRow(), relocRows, outagedZiNums) != vVar.getRow()) {
                                relocIdentity = true;        // its own row carries the coupler's ZERO_V
                                live = null;
                                break;
                            }
                            live.add(reHomedPvEntry(vVar.getRow(), e.getValue(), rlEntryOf(gpuData, e.getKey())));
                        }
                    }
                    if (live == null) {
                        com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record(
                                relocIdentity ? "PLAN:VC_REMERGE_RELOC_VROW" : "PLAN:VC_REMERGE_NO_VROW");
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
                    for (GenOutage go : GpuBatchInjectionOutages.islandedDistrQEntries(gpuData, inactiveForTargets)) {
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
                    double[][] ov = distrQWeightOverrides(es, gpuData, goneForKeys);
                    if (ov == null) {
                        com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:DISTR_Q_WEIGHT_REDIFF");
                        continue;                            // weight rediff unavailable — CPU
                    }
                    dqOvContribByIndex[i] = java.util.Arrays.stream(ov[0]).mapToInt(d -> (int) d).toArray();
                    dqOvWByIndex[i] = ov[1];
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
                ReHomed reHomed = reHomedControls(q.islandedBuses(), c.outagedZiBranches());
                if (reHomed.reject() != null) {
                    com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:" + reHomed.reject());
                    continue;
                }
                deadControlBuses.removeAll(reHomed.pvTargets().keySet());
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
                    com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:ISLANDS_REMOTE_VC_MEMBER");
                    continue;                                // an islanded HVDC endpoint — CPU
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
                                new int[0], new double[0], deadControlRowMode(gpuData, sb), -1, 0, -1, 0, -1, 0));
                    }
                    if (vcOff == null) {
                        if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                            System.err.printf("QUALIFY_TRACE reject %s tag=ISLANDS_REMOTE_VC_MEMBER site=tail-deadControl-noVrow"
                                    + " deadControl=%s%n", c.contingencyId(), deadControlBuses.stream().map(LfBus::getId).toList());
                        }
                        com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:ISLANDS_REMOTE_VC_MEMBER");
                        continue;                            // no V row to flip — CPU
                    }
                    genOutagesByIndex[i] = vcOff.toArray(new GenOutage[0]);
                }
                if (!reHomed.pvTargets().isEmpty()) {
                    List<GenOutage> pvOn = new ArrayList<>();
                    if (genOutagesByIndex[i] != null) {
                        pvOn.addAll(java.util.Arrays.asList(genOutagesByIndex[i]));
                    }
                    boolean ok = true;
                    boolean relocIdentity = false;
                    for (var e : reHomed.pvTargets().entrySet()) {
                        var vVar = es.getVariableSet().getVariable(e.getKey().getNum(), AcVariableType.BUS_V);
                        if (vVar == null || vVar.getRow() < 0) {
                            ok = false;
                            break;
                        }
                        if (relocQRow(vVar.getRow(), relocRows, outagedZiNums) != vVar.getRow()) {
                            ok = false;
                            relocIdentity = true;            // its own row carries the coupler's ZERO_V
                            break;
                        }
                        pvOn.add(reHomedPvEntry(vVar.getRow(), e.getValue(), rlEntryOf(gpuData, e.getKey())));
                    }
                    if (!ok) {
                        com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record(
                                relocIdentity ? "PLAN:VC_REMERGE_RELOC_VROW" : "PLAN:VC_REMERGE_NO_VROW");
                        continue;
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
                    double[][] vcOv = distrQWeightOverrides(es, gpuData, goneControllers);
                    if (vcOv == null) {
                        com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PLAN:DISTR_Q_WEIGHT_REDIFF");
                        continue;
                    }
                    dqOvContribByIndex[i] = java.util.Arrays.stream(vcOv[0]).mapToInt(d -> (int) d).toArray();
                    dqOvWByIndex[i] = vcOv[1];
                    // and the survivors' DISTR_Q row TARGETS + the islanded members' inactive entries
                    List<GenOutage> dq = GpuBatchInjectionOutages.islandedDistrQEntries(gpuData, lostVcControllers);
                    if (!dq.isEmpty()) {
                        List<GenOutage> merged = new ArrayList<>();
                        if (genOutagesByIndex[i] != null) {
                            merged.addAll(java.util.Arrays.asList(genOutagesByIndex[i]));
                        }
                        merged.addAll(dq);
                        genOutagesByIndex[i] = merged.toArray(new GenOutage[0]);
                    }
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
                dqOvContribByIndex, dqOvWByIndex, drOvContribByIndex, drOvCByIndex, islandedDsByIndex,
                islandedRlByIndex, islandedHvdcsByIndex);
    }
}
