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
 * a single fixed-shunt disconnection, and 0–2 closed-branch outages — returning {@code null} for
 * anything out of batch scope (bus losses, unresolvable elements, &gt;2 branches, partial opens),
 * which routes the contingency to the CPU path. Nearly-pure functions over the network + contingency.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuContingencyMapper {

    private GpuContingencyMapper() {
    }

    /** The contingency's lost generators (any count, possibly alongside branches and loads), or null
     *  when any of them is unresolvable. Bus losses stay out of scope. */
    static List<LfGenerator> generatorOutages(LfNetwork net, PropagatedContingency pc) {
        if (pc.getGeneratorIdsToLose().isEmpty() || !pc.getBusIdsToLose().isEmpty()) {
            return null;
        }
        List<LfGenerator> gens = new java.util.ArrayList<>();
        for (String id : pc.getGeneratorIdsToLose()) {
            LfGenerator gen = net.getGeneratorById(id);
            if (gen == null || gen.getBus() == null || gen.getBus().isDisabled()) {
                return null;
            }
            gens.add(gen);
        }
        return gens;
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
    static List<LoadLoss> loadOutages(Network network, LfNetwork net, PropagatedContingency pc) {
        if (pc.getLoadIdsToLose().isEmpty() || !pc.getBusIdsToLose().isEmpty()) {
            return null;
        }
        List<LoadLoss> out = new java.util.ArrayList<>();
        for (Map.Entry<String, com.powsybl.openloadflow.network.PowerShift> e : pc.getLoadIdsToLose().entrySet()) {
            com.powsybl.iidm.network.Load load = network.getLoad(e.getKey());
            if (load == null || load.getTerminal().getBusView().getBus() == null) {
                return null;
            }
            LfBus bus = net.getBusById(load.getTerminal().getBusView().getBus().getId());
            if (bus == null || bus.isDisabled()) {
                return null;
            }
            out.add(new LoadLoss(bus, e.getValue().getActive(), e.getValue().getReactive()));
        }
        return out;
    }

    /** The contingency's shunt admittance shifts (pu deltas, negated: the propagation's shift is the
     *  admittance REMOVED), or null when a shunt is unresolvable. */
    static List<ShuntShift> shuntShifts(LfNetwork net, PropagatedContingency pc) {
        if (!pc.getBusIdsToLose().isEmpty()) {
            return null;
        }
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

    /** The contingency's closed-branch outages (0-2, BOTH_SIDES only), or null when the branch part is
     *  out of batch scope. Generators are allowed alongside (branch+generator combos batch since 2026-07-17);
     *  buses and loads are not. */
    static List<LfBranch> closedBranchOutages(LfNetwork net, PropagatedContingency pc, List<LfBranch> ziOutages) {
        java.util.Set<String> ziIds = ziOutages.stream().map(LfBranch::getId)
                .collect(java.util.stream.Collectors.toSet());
        int n = pc.getBranchIdsToOpen().size() - ziIds.size();   // the switch part is masked separately
        if (n > 2 || !pc.getBusIdsToLose().isEmpty()) {
            return null;
        }
        List<LfBranch> branches = new ArrayList<>(Math.max(n, 0));
        for (Map.Entry<String, DisabledBranchStatus> e : pc.getBranchIdsToOpen().entrySet()) {
            if (ziIds.contains(e.getKey())) {
                continue;
            }
            if (e.getValue() != DisabledBranchStatus.BOTH_SIDES) {
                return null;
            }
            LfBranch branch = net.getBranchById(e.getKey());
            if (branch == null || branch.getBus1() == null || branch.getBus2() == null) {
                return null;
            }
            branches.add(branch);
        }
        return branches;
    }
}
