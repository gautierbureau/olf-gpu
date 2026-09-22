/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.iidm.network.Network;
import com.powsybl.openloadflow.ac.solver.gpu.batch.LoadLoss;
import com.powsybl.openloadflow.ac.solver.gpu.batch.ShuntShift;
import com.powsybl.openloadflow.network.DisabledBranchStatus;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfShunt;
import com.powsybl.openloadflow.network.impl.PropagatedContingency;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Maps an OLF {@link PropagatedContingency} to the batchable device shapes the GPU N-1 path
 * understands — lost generators, lost loads (as per-bus injection deltas), shunt admittance shifts,
 * a single fixed-shunt disconnection, and 0–N closed-branch outages — returning {@code null} for
 * anything out of batch scope (bus losses, unresolvable elements, partial opens),
 * which routes the contingency to the CPU path. Nearly-pure functions over the network + contingency.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuContingencyMapper {

    private GpuContingencyMapper() {
    }

    /** The contingency's lost generators (any count, possibly alongside branches and loads), or null
     *  when any of them is unresolvable. Bus losses stay out of scope. */
    static List<LfGenerator> generatorOutages(LfNetwork net, PropagatedContingency pc, java.util.Set<String> lostLfBusIds) {
        if (pc.getGeneratorIdsToLose().isEmpty()) {
            return null;
        }
        List<LfGenerator> gens = new java.util.ArrayList<>();
        for (String id : pc.getGeneratorIdsToLose()) {
            LfGenerator gen = net.getGeneratorById(id);
            if (gen == null || gen.getBus() == null || gen.getBus().isDisabled()) {
                return null;
            }
            if (lostLfBusIds.contains(gen.getBus().getId())) {
                continue;                                    // bus loss: the injection freezes WITH its bus
            }                                                // (islanded frozen rows + scheduled-loss accounting)
            gens.add(gen);
        }
        return gens;
    }

    /** The LF-bus ids the contingency LOSES outright (bus/busbar loss): each busIdsToLose entry is a
     *  BUS-BREAKER-view id — resolve to the containing BUS-VIEW bus (any connected terminal), which is
     *  the LfBus id space. Their rows freeze via the islanding machinery, so injections there must
     *  NOT also become explicit target overrides (double counting). */
    static java.util.Set<String> lostLfBusIds(LfNetwork net, Network network, PropagatedContingency pc) {
        if (pc.getBusIdsToLose().isEmpty()) {
            return java.util.Set.of();
        }
        java.util.Set<String> out = new java.util.HashSet<>();
        for (String busId : pc.getBusIdsToLose()) {
            if (net.getBusById(busId) != null) {             // breaker-mode LfNetwork: same id space
                out.add(busId);
                continue;
            }
            com.powsybl.iidm.network.Bus bb = network.getBusBreakerView().getBus(busId);
            if (bb == null) {
                continue;
            }
            bb.getConnectedTerminalStream()                  // bus-view LfNetwork: translate
                    .map(t -> t.getBusView().getBus())
                    .filter(java.util.Objects::nonNull)
                    .findFirst()
                    .ifPresent(b -> out.add(b.getId()));
        }
        return out;
    }

    /** The contingency's outaged AC-EMULATION hvdcs (per-scenario hv element disable), or null when
     *  any is unresolvable or not device-packed (setpoint mode / missing bus end / LCC — the coupling
     *  is then not an hv element and the outage is not expressible as a mask). The VSC stations'
     *  injection losses ride the normal generator-outage channel (PropagatedContingency puts them in
     *  generatorIdsToLose). */
    static List<com.powsybl.openloadflow.network.LfHvdc> hvdcOutages(LfNetwork net, PropagatedContingency pc) {
        if (pc.getHvdcIdsToOpen().isEmpty()) {
            return List.of();
        }
        List<com.powsybl.openloadflow.network.LfHvdc> out = new java.util.ArrayList<>();
        for (String id : pc.getHvdcIdsToOpen()) {
            com.powsybl.openloadflow.network.LfHvdc hvdc = net.getHvdcById(id);
            if (hvdc == null || hvdc.getBus1() == null || hvdc.getBus2() == null || !hvdc.isAcEmulation()) {
                return null;                                 // not a packed hv element — CPU
            }
            out.add(hvdc);
        }
        return out;
    }

    /** The contingency's lost loads as per-bus injection deltas (pu, from the propagation's PowerShift),
     *  or null when unresolvable. Guards: bus losses out of scope; the ZIP/voltage-dependent load model
     *  makes per-scenario load removal unrepresentable as a target override (checked by the caller). */
    static List<LoadLoss> loadOutages(Network network, LfNetwork net, PropagatedContingency pc, java.util.Set<String> lostLfBusIds) {
        if (pc.getLoadIdsToLose().isEmpty()) {
            return null;
        }
        List<LoadLoss> out = new java.util.ArrayList<>();
        for (Map.Entry<String, com.powsybl.openloadflow.network.PowerShift> e : pc.getLoadIdsToLose().entrySet()) {
            com.powsybl.iidm.network.Load load = network.getLoad(e.getKey());
            if (load == null) {
                return null;                                 // an LCC station's load share: CPU
            }
            // The load's LfBus through OLF's OWN map (loadsById), never through the bus view: a
            // node-breaker level's bus-view buses and its breaker-mode LfBuses share the "<VL>_<n>"
            // id pattern but partition the level differently (the bus view merges across every
            // closed breaker, the LfNetwork keeps retained ones), so "GIU_0" named a different
            // physical bus in each. Measured on network.xiidm / LJBS: the island's two loads (26 MW,
            // 106 MVar) were removed from the neighbour GIU_0 instead of the frozen GIU_1 — 3.1 kV.
            com.powsybl.openloadflow.network.LfLoad lfLoad = net.getLoadById(e.getKey());
            LfBus bus = lfLoad != null ? lfLoad.getBus() : null;
            if (bus == null || bus.isDisabled()) {
                return null;
            }
            if (lostLfBusIds.contains(bus.getId())) {
                continue;                                    // bus loss: freezes with its bus (no override)
            }
            out.add(new LoadLoss(bus, e.getValue().getActive(), e.getValue().getReactive()));
        }
        return out;
    }

    /** The contingency's shunt admittance shifts (pu deltas, negated: the propagation's shift is the
     *  admittance REMOVED), or null when a shunt is unresolvable. */
    static List<ShuntShift> shuntShifts(LfNetwork net, PropagatedContingency pc) {
        List<ShuntShift> out = new java.util.ArrayList<>();
        for (Map.Entry<String, com.powsybl.openloadflow.network.AdmittanceShift> e : pc.getShuntIdsToShift().entrySet()) {
            LfShunt shunt = net.getShuntById(e.getKey());
            if (shunt == null || shunt.isDisabled()) {
                return null;
            }
            out.add(new ShuntShift(shunt, -e.getValue().getG(), -e.getValue().getB()));
        }
        return out;
    }

    /** The single fixed shunt a contingency disconnects (which the GPU batch can mask off like a branch), or
     *  null if it is not a clean full disconnection of a packed fixed shunt — a multi-element contingency, a
     *  voltage-controlling/SVC shunt the GPU does not pack, or a multi-compensator aggregate the mask cannot
     *  partially remove (those fall through to the CPU). */
    static LfShunt singleShuntOutage(LfNetwork net, PropagatedContingency pc) {
        Set<String> shuntIds = pc.getShuntIdsToShift().keySet();
        if (shuntIds.size() != 1 || !pc.getBranchIdsToOpen().isEmpty()
                || !pc.getGeneratorIdsToLose().isEmpty() || !pc.getBusIdsToLose().isEmpty()
                || !pc.getLoadIdsToLose().isEmpty()) {
            return null;
        }
        LfShunt shunt = net.getShuntById(shuntIds.iterator().next());
        // A sole-compensator LfShunt: masking its element removes exactly the contingency's full admittance.
        return shunt != null && !shunt.isDisabled() && shunt.getOriginalIds().size() == 1 ? shunt : null;
    }

    /** The contingency's closed-branch outages (0-N, BOTH_SIDES only — the <=2 cap was lifted
     *  2026-09-08; the per-element mask carries any count and the validator rejects fragmenting
     *  sets), or null when the branch part is out of batch scope. Generators are allowed alongside
     *  (branch+generator combos batch since 2026-07-17); buses and loads are not. */
    /** The contingency's SINGLE-SIDE branch openings (node-breaker propagation opening one breaker):
     *  each becomes a masked closed-branch element plus the pre-packed side-open OB view for the
     *  surviving terminal, or null when a one-sided shape is out of scope (zero-impedance). */
    static List<com.powsybl.openloadflow.ac.solver.gpu.batch.BatchContingency.SideOpen> sideOpens(
            LfNetwork net, PropagatedContingency pc) {
        List<com.powsybl.openloadflow.ac.solver.gpu.batch.BatchContingency.SideOpen> out = new ArrayList<>();
        for (Map.Entry<String, DisabledBranchStatus> e : pc.getBranchIdsToOpen().entrySet()) {
            if (e.getValue() == DisabledBranchStatus.BOTH_SIDES) {
                continue;
            }
            LfBranch branch = net.getBranchById(e.getKey());
            if (branch == null || branch.getBus1() == null || branch.getBus2() == null) {
                return null;                                 // unresolvable (closedBranchOutages nulls too)
            }
            if (branch.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC)) {
                trace(pc, "one-side open zero-impedance " + e.getKey());
                return null;                                 // zi one-side: spanning-tree re-pick — CPU
            }
            out.add(new com.powsybl.openloadflow.ac.solver.gpu.batch.BatchContingency.SideOpen(branch,
                    e.getValue() == DisabledBranchStatus.SIDE_1 ? 1 : 2));
        }
        // BUS/BUSBAR LOSS: OLF's findBranchToOpenDirectlyImpactedByContingency opens every branch
        // INCIDENT to a lost bus on the LOST side (SIDE_1 when bus1 is the lost bus) — replicate.
        for (String busId : pc.getBusIdsToLose()) {
            com.powsybl.openloadflow.network.LfBus bus = net.getBusById(busId);
            if (bus == null) {
                trace(pc, "unresolvable lost bus " + busId);
                return null;
            }
            for (LfBranch branch : bus.getBranches()) {
                if (pc.getBranchIdsToOpen().get(branch.getId()) == DisabledBranchStatus.BOTH_SIDES) {
                    continue;                                // already a full outage
                }
                if (branch.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC)) {
                    trace(pc, "lost bus " + busId + " has zero-impedance incident " + branch.getId());
                    return null;                             // zi incident: spanning-tree re-pick — CPU
                }
                if (branch.getBus1() == null || branch.getBus2() == null) {
                    continue;                                // already open-ended — nothing carries flow
                }
                com.powsybl.openloadflow.network.LfBus other = branch.getBus1() == bus ? branch.getBus2() : branch.getBus1();
                if (pc.getBusIdsToLose().contains(other.getId())) {
                    trace(pc, "branch between two lost buses " + branch.getId());
                    return null;                             // fully dead branch between lost buses — CPU (rare)
                }
                out.add(new com.powsybl.openloadflow.ac.solver.gpu.batch.BatchContingency.SideOpen(branch,
                        branch.getBus1() == bus ? 1 : 2));
            }
        }
        // Does the side-open set actually COVER the disconnection? Freezing a lost bus's rows makes
        // its own equations trivial; it does NOT stop a branch from injecting into the LIVE bus at the
        // far end. So every branch incident to a lost bus needs a side-open, and any shortfall leaves
        // a phantom flow into the surviving network.
        if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
            int incident = 0;
            for (String busId : pc.getBusIdsToLose()) {
                var b = net.getBusById(busId);
                if (b != null) {
                    incident += b.getBranches().size();
                }
            }
            System.err.printf("QUALIFY_TRACE sideOpen-cover %s: busIdsToLose=%d resolved=%d"
                            + " incidentBranches=%d sideOpens=%d branchIdsToOpen=%d%n",
                    pc.getContingency().getId(), pc.getBusIdsToLose().size(),
                    (int) pc.getBusIdsToLose().stream().filter(id -> net.getBusById(id) != null).count(),
                    incident, out.size(), pc.getBranchIdsToOpen().size());
        }
        return out;
    }

    private static void trace(PropagatedContingency pc, String why) {
        if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {   // test logback is broken — stderr lever
            System.err.println("QUALIFY_TRACE mapper drops " + pc.getContingency().getId() + ": " + why);
        }
    }

    static List<LfBranch> closedBranchOutages(LfNetwork net, PropagatedContingency pc, List<LfBranch> ziOutages) {
        java.util.Set<String> ziIds = ziOutages.stream().map(LfBranch::getId)
                .collect(java.util.stream.Collectors.toSet());
        List<LfBranch> branches = new ArrayList<>(pc.getBranchIdsToOpen().size());
        for (Map.Entry<String, DisabledBranchStatus> e : pc.getBranchIdsToOpen().entrySet()) {
            if (ziIds.contains(e.getKey())) {
                continue;
            }
            LfBranch branch = net.getBranchById(e.getKey());
            if (branch == null || (branch.getBus1() == null && branch.getBus2() == null)) {
                // name the ELEMENT KIND too: an id alone does not say which family
                trace(pc, "unresolvable " + e.getKey() + " kind=" + pc.getContingency().getElements().stream()
                        .map(el -> el.getType().name()).distinct().toList()
                        + (branch == null ? " lfBranch=absent" : " bus1=false bus2=false"));
                return null;
            }
            if (branch.getBus1() == null || branch.getBus2() == null) {
                // A branch already open at ONE end in the base: the device packs it as an open-branch
                // element (the connected end's view of it), so its outage is that element's disabled
                // bit, like a closed branch's. It used to be "unresolvable": 2049 of network.xiidm's
                // 11249 branch outages (380 of the first 2000), every one replayed on the CPU - most
                // of the GPU arm's wall time there. It cuts nothing, so the validator's connectivity
                // check (which skips half-open branches) is unaffected.
                branches.add(branch);
                continue;
            }
            if (e.getValue() != DisabledBranchStatus.BOTH_SIDES) {
                continue;                                    // ONE-SIDE opening: handled by sideOpens()
            }
            branches.add(branch);
        }
        return branches;
    }
}
