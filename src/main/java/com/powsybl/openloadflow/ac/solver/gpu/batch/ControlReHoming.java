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
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;

import java.util.ArrayList;
import java.util.List;

/**
 * Voltage controls re-homed by an islanding contingency: which controllers survive, which pilot or main generator
 * takes over, and which shapes the device refuses (OLF's voltage-control re-merge after a topology change).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class ControlReHoming {

    private ControlReHoming() {
    }

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

    /** Whether the generator's reactive limits are a capability CURVE (they then depend on its target P). */
    static boolean hasCurve(com.powsybl.openloadflow.network.LfGenerator g) {
        return g instanceof com.powsybl.openloadflow.network.impl.AbstractLfGenerator alg
                && alg.getReactiveLimits().map(rl -> rl.getKind() == com.powsybl.iidm.network.ReactiveLimitsKind.CURVE)
                        .orElse(false);
    }

    /** Whether the generator is in the distributed-slack generator pack (the device then knows its own P). */
    static boolean inGeneratorPack(GpuAcData gpuData, String genId) {
        return gpuData.dgGenId() != null && java.util.Arrays.asList(gpuData.dgGenId()).contains(genId);
    }
}
