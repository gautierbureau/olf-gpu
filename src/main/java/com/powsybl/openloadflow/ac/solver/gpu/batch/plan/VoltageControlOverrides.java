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
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuReactiveTermKind;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Per-contingency overrides of the voltage-control distribution rows: the DISTR_RHO coefficients of a shared
 * transformer group, the split of a voltage-control group, and the DISTR_Q weights re-keyed by OLF's
 * updateReactiveKeys.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class VoltageControlOverrides {

    private VoltageControlOverrides() {
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
    static double[][] distrRhoOverrides(EquationSystem<AcVariableType, AcEquationType> es,
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
        for (int k = 0; k < gpuData.distrRho().row().length; k++) {
            LfBranch rowOwner = byRhoCol.get(gpuData.distrRho().row()[k]);
            LfBranch colOwner = byRhoCol.get(gpuData.distrRho().column()[k]);
            if (rowOwner == null && colOwner == null) {
                continue;                                    // another group's contribution
            }
            double coef;
            if (rowOwner == d || colOwner == d || gpuData.distrRho().row()[k] == dCol || gpuData.distrRho().column()[k] == dCol) {
                coef = 0;                                    // cross terms to the outaged member die with it
            } else if (newCarrier != null && rowOwner == newCarrier) {
                coef = 0;                                    // the re-picked carrier's DISTR_RHO row deactivates
            } else {
                coef = 1.0 / cPrime - (gpuData.distrRho().row()[k] == gpuData.distrRho().column()[k] ? 1.0 : 0.0);
            }
            idx.add((double) k);
            coefs.add(coef);
        }
        return new double[][] {idx.stream().mapToDouble(Double::doubleValue).toArray(),
            coefs.stream().mapToDouble(Double::doubleValue).toArray()};
    }

    /** What an island does to the remote generator-voltage-control groups it touches. */
    enum VcSplit {
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
    static VcSplit classifyVcSplit(EquationSystem<AcVariableType, AcEquationType> es,
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
    static double[][] distrQWeightOverrides(EquationSystem<AcVariableType, AcEquationType> es,
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
            double[] baseW = gpuData.distrQ().weight();
            double[] newW = reweightedDqWeights(es, gpuData, touched);
            // The member KEYS after the loss (OLF's qPct on the touched groups' MAIN controls, just updated): the
            // kernels renormalize them over the members a pin leaves enabled, so the loss and an in-flight pin in the
            // same group compose - which fixed per-contribution weights cannot do (the fill applied them over its
            // pin-aware recompute and the rows kept the outage-only split while the targets followed the pin).
            boolean keyed = newW != null;                    // the analytic path: the pack's structure is the base's
            List<Double> keyMember = new ArrayList<>();
            List<Double> keyValue = new ArrayList<>();
            int[] memberBusNum = gpuData.distrQMembers().busNum();
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
            if (newW == null) {
                // the free role of a table-less group may move with the loss: re-extract, as before
                nd = com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor
                        .extract(network, es, false, false);
                newW = nd.distrQ().weight();
            }
            if (newW.length != baseW.length) {
                return null;                                 // structural drift — not a pure reweight
            }
            List<Double> idx = new ArrayList<>();
            List<Double> w = new ArrayList<>();
            boolean dqOvTrace = System.getenv("OLF_GPU_DQOV_TRACE") != null;
            int traced = 0;
            int[] kMemberOf = gpuData.distrQ().rowMember();
            for (int k = 0; k < newW.length; k++) {
                if (newW[k] != baseW[k]) {
                    if (keyed && kMemberOf != null && kMemberOf[k] >= 0) {
                        continue;                            // the keys carry it (the kernels' live recompute)
                    }
                    idx.add((double) k);
                    w.add(newW[k]);
                    if (dqOvTrace && traced++ < 12 && nd != null) {   // the two packs' metadata at a differing index
                        System.err.printf("DQOV_PACK k=%d old(row=%d kind=%d elem=%d side=%d w=%.6f) new(row=%d kind=%d elem=%d side=%d w=%.6f)%n",
                                k, gpuData.distrQ().row()[k], gpuData.distrQ().kind()[k], gpuData.distrQ().element()[k], gpuData.distrQ().side()[k], baseW[k],
                                nd.distrQ().row()[k], nd.distrQ().kind()[k], nd.distrQ().element()[k], nd.distrQ().side()[k], newW[k]);
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

    /**
     * The base DISTR_Q pack's weights after {@link #updateMainReactiveKeys} ran on the touched buses, WITHOUT
     * re-extracting the network: a contribution's weight is {@code qPct_k - (j == k ? 1 : 0)} for the row's
     * controller k and the contributing controller j (DistrQPacker.buildDistrQContributions), and the
     * toggle changes only the qPct of the touched groups' controllers. Measured on network.xiidm: the
     * whole-network extraction this replaces cost 18.8 s of a 207 s cold run (one per generator contingency at
     * a DISTR_Q member). Returns null when a touched bus is the free controller of a group WITHOUT a candidate
     * table — there the loss can move the free role, which changes the pack's row set, not only its weights.
     */
    static double[] reweightedDqWeights(EquationSystem<AcVariableType, AcEquationType> es, GpuAcData gpuData,
                                                java.util.Set<LfBus> touched) {
        int[] kMember = gpuData.distrQ().rowMember();
        int[] jMember = gpuData.distrQ().injectionMember();
        int[] memberBus = gpuData.distrQMembers().busNum();
        double[] baseW = gpuData.distrQ().weight();
        if (kMember == null || jMember == null || memberBus == null || baseW == null) {
            return null;
        }
        int[] memberQrGroup = gpuData.distrQMembers().relocationGroup();
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
        int[] kind = gpuData.distrQ().kind();
        int[] side = gpuData.distrQ().side();
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
            if (kind != null && side != null && kind[i] == GpuReactiveTermKind.DUMMY_Q && side[i] != 0) {
                weight = -weight;
            }
            w[i] = weight;
        }
        return w;
    }

    /** OLF's own key update: on the group's MAIN control, never on a member's dependent one. */
    static void updateMainReactiveKeys(LfBus bus) {
        bus.getGeneratorVoltageControl()
                .<com.powsybl.openloadflow.network.GeneratorVoltageControl>map(
                        com.powsybl.openloadflow.network.VoltageControl::getMainVoltageControl)
                .ifPresent(com.powsybl.openloadflow.network.GeneratorVoltageControl::updateReactiveKeys);
    }
}
