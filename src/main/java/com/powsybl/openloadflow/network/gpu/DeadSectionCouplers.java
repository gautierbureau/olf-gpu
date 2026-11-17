/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.network.gpu;

import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LoadFlowModel;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The couplers through a dead section. A section out of the network at base (its buses disabled), linked by
 * zero-impedance branches only and reached from the network by exactly two switches open at base, each to a
 * different bus of the network, couples those two buses once both switches close: it is then one node with them.
 * The device holds it as ONE base-open coupler between the two buses, packed on the dummy rows of the switch with
 * the lower number (its carrier); the other switch is not packed. The section's own buses have no rows: their
 * injections go to the coupled node and their voltage is the node's.
 *
 * <p>The packer and the strategy composer both read it, so the switch the composer closes is the one packed.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class DeadSectionCouplers {

    /** The coupler a dead section makes: {@code carrier} packed between {@code bus1} (its own end in the network)
     *  and {@code bus2} (the other switch's end in the network). */
    public record Coupler(LfBranch carrier, LfBranch other, LfBus bus1, LfBus bus2) {
    }

    private final Map<LfBranch, Coupler> byCarrier;
    private final Map<LfBranch, Coupler> bySwitch;

    private DeadSectionCouplers(Map<LfBranch, Coupler> byCarrier, Map<LfBranch, Coupler> bySwitch) {
        this.byCarrier = byCarrier;
        this.bySwitch = bySwitch;
    }

    /** The coupler {@code carrier} is packed as, or null when it is not the carrier of one. */
    public Coupler ofCarrier(LfBranch carrier) {
        return byCarrier.get(carrier);
    }

    /** The coupler either of its two switches belongs to, or null. */
    public Coupler ofSwitch(LfBranch sw) {
        return bySwitch.get(sw);
    }

    public static DeadSectionCouplers of(LfNetwork network) {
        Map<LfBus, List<LfBranch>> gatesBySectionRoot = new LinkedHashMap<>();
        Map<LfBus, LfBus> rootOf = new HashMap<>();
        for (LfBus bus : network.getBuses()) {
            if (bus.isDisabled() && !rootOf.containsKey(bus)) {
                walkSection(bus, rootOf, gatesBySectionRoot);
            }
        }
        Map<LfBranch, Coupler> byCarrier = new HashMap<>();
        Map<LfBranch, Coupler> bySwitch = new HashMap<>();
        for (List<LfBranch> gates : gatesBySectionRoot.values()) {
            if (gates.size() != 2) {
                continue;
            }
            gates.sort(Comparator.comparingInt(LfBranch::getNum));
            LfBranch carrier = gates.get(0);
            LfBranch other = gates.get(1);
            LfBus bus1 = liveEnd(carrier);
            LfBus bus2 = liveEnd(other);
            if (bus1 == bus2) {
                continue;                                    // both switches to one bus: a section hanging on it
            }
            Coupler coupler = new Coupler(carrier, other, bus1, bus2);
            byCarrier.put(carrier, coupler);
            bySwitch.put(carrier, coupler);
            bySwitch.put(other, coupler);
        }
        return new DeadSectionCouplers(Collections.unmodifiableMap(byCarrier), Collections.unmodifiableMap(bySwitch));
    }

    /** The section of {@code start} (disabled buses linked by zero-impedance branches) and the switches open at base
     *  that reach the network from it. */
    private static void walkSection(LfBus start, Map<LfBus, LfBus> rootOf, Map<LfBus, List<LfBranch>> gatesBySectionRoot) {
        List<LfBranch> gates = new ArrayList<>();
        Set<LfBranch> seen = new HashSet<>();
        Deque<LfBus> queue = new ArrayDeque<>();
        rootOf.put(start, start);
        queue.add(start);
        while (!queue.isEmpty()) {
            LfBus bus = queue.poll();
            for (LfBranch b : bus.getBranches()) {
                if (b.getBus1() == null || b.getBus2() == null || !b.isZeroImpedance(LoadFlowModel.AC) || !seen.add(b)) {
                    continue;
                }
                LfBus other = b.getBus1() == bus ? b.getBus2() : b.getBus1();
                if (other.isDisabled()) {
                    if (!rootOf.containsKey(other)) {
                        rootOf.put(other, start);
                        queue.add(other);
                    }
                } else if (b.isDisabled()) {
                    gates.add(b);                            // a switch open at base, to a bus of the network
                }
            }
        }
        gatesBySectionRoot.put(start, gates);
    }

    private static LfBus liveEnd(LfBranch gate) {
        return gate.getBus1().isDisabled() ? gate.getBus2() : gate.getBus1();
    }
}
