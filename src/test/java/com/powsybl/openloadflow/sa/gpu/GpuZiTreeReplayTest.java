/*
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.contingency.BranchContingency;
import com.powsybl.contingency.Contingency;
import com.powsybl.iidm.network.Bus;
import com.powsybl.iidm.network.Network;
import com.powsybl.openloadflow.graph.GraphConnectivity;
import com.powsybl.openloadflow.network.AbstractLoadFlowNetworkFactory;
import com.powsybl.openloadflow.network.FirstSlackBusSelector;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfNetworkParameters;
import com.powsybl.openloadflow.network.LfTopoConfig;
import com.powsybl.openloadflow.network.impl.Networks;
import com.powsybl.openloadflow.network.impl.PropagatedContingency;
import com.powsybl.openloadflow.network.impl.PropagatedContingencyCreationParameters;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The tree replay must leave OLF's connectivity exactly as it found it.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuZiTreeReplayTest extends AbstractLoadFlowNetworkFactory {

    /** The slack s hangs on the rest of the network by one line l0; the rest is the hub m, its coupler c1 to n, and
     *  the leaves a (on m) and c (on n). Losing l0 isolates the slack: the largest component is the other side. */
    private static Network slackOnAStub() {
        Network network = Network.create("slack-on-a-stub", "code");
        Bus s = createBus(network, "x", "s");
        createGenerator(s, "gs", 3, 1.0);
        Bus m = createBus(network, "y", "m");
        createLoad(m, "lm", 0.5, 0.2);
        Bus n = createBus(network, "y", "n");
        createLoad(n, "ln", 0.5, 0.2);
        Bus a = createBus(network, "y", "a");
        createLoad(a, "la", 0.5, 0.2);
        Bus c = createBus(network, "y", "c");
        createLoad(c, "lc", 0.5, 0.2);
        createLine(network, s, m, "l0", 0.1);
        createLine(network, m, n, "c1", 0.0, 0.0);
        createLine(network, m, a, "lA", 0.1);
        createLine(network, n, c, "lC", 0.1);
        return network;
    }

    private static Set<String> lostWhenCut(LfNetwork network, String branchId) {
        GraphConnectivity<LfBus, LfBranch> connectivity = network.getConnectivity();
        connectivity.startTemporaryChanges();
        try {
            connectivity.removeEdge(network.getBranchById(branchId));
            return connectivity.getVerticesRemovedFromMainComponent().stream().map(LfBus::getId).collect(Collectors.toSet());
        } finally {
            connectivity.undoTemporaryChanges();
        }
    }

    /** A contingency that isolates the slack anchors the replay's connectivity query on the largest component, as
     *  OLF's own slack relocation does. OLF's anchor is undone with its network state; the replay's was not: its
     *  anchor stayed the DEFAULT main-component vertex of the network's connectivity, chosen as the first bus of a
     *  hash-ordered set, so a different bus each run. Every later contingency that lost THAT bus then saw the whole
     *  network as lost - every coupler disabled, the replay's tree empty - and the device solved it with all 366
     *  of rte6515's couplers decoupled (one random busbar section per run failing its linear solve at a ratio of
     *  3.06e+06: BUS-5243, BUS-506, BUS-747, BUS-528, BUS-608, BUS-5498 on successive runs). */
    @Test
    void replayLeavesTheConnectivityAnchorWhereItFoundIt() {
        Network iidm = slackOnAStub();
        LfNetwork network = Networks.load(iidm, new LfNetworkParameters().setSlackBusSelector(new FirstSlackBusSelector())).get(0);
        assertEquals("s_vl_0", network.getSynchronousNetworks().getFirst().getSlackBuses().getFirst().getId(), "the slack is the stub");
        Map<String, PropagatedContingency> pcs = PropagatedContingency.createList(iidm, List.of(
                        new Contingency("l0", new BranchContingency("l0")),
                        new Contingency("lA", new BranchContingency("lA")),
                        new Contingency("lC", new BranchContingency("lC")),
                        new Contingency("c1lC", new BranchContingency("c1"), new BranchContingency("lC"))),
                new LfTopoConfig(), new PropagatedContingencyCreationParameters()).stream()
                .collect(Collectors.toMap(pc -> pc.getContingency().getId(), Function.identity()));
        Set<String> otherSide = Set.of("m_vl_0", "n_vl_0", "a_vl_0", "c_vl_0");
        assertEquals(otherSide, lostWhenCut(network, "l0"), "before: the main component is the slack's");

        GpuZiTreeReplay replay = new GpuZiTreeReplay(network);
        List<String> base = replay.nonTree().stream().map(LfBranch::getId).toList();
        assertEquals(List.of(), base, "c1 is the only coupler, in the tree");
        assertEquals(base, ids(replay.applyAndRevert(pcs.get("l0"))), "losing the slack's line disables no coupler");

        assertEquals(otherSide, lostWhenCut(network, "l0"), "after: the main component is still the slack's");
        // ...and so every later loss is read from the slack's side: a leaf, the other leaf, n with its leaf
        assertEquals(base, ids(replay.applyAndRevert(pcs.get("lA"))));
        assertEquals(base, ids(replay.applyAndRevert(pcs.get("lC"))));
        assertEquals(List.of("c1"), ids(replay.applyAndRevert(pcs.get("c1lC"))), "its own opened coupler is out");
        assertEquals(base, ids(replay.applyAndRevert(pcs.get("lA"))), "and back in the tree once reverted");
    }

    /** A loop of three couplers z12, z13, z23 between b1, b2 and b3 (numbered by id: z23 closes the loop), and
     *  two lines to b4. Opening z12 brings z23 into the tree for that contingency; OLF re-elects the tree when z12
     *  comes back, so the next contingency starts from the base tree again (olf-local-patches 7dd48d92b). Before
     *  that fix z12 stayed out of OLF's tree for every later contingency, and the replay copied that. */
    @Test
    void replayStartsEveryContingencyFromOlfsBaseTree() {
        Network iidm = Network.create("coupler-loop", "code");
        Bus b1 = createBus(iidm, "y", "b1");
        createGenerator(b1, "g1", 2, 1.0);
        Bus b2 = createBus(iidm, "y", "b2");
        Bus b3 = createBus(iidm, "y", "b3");
        Bus b4 = createBus(iidm, "x", "b4");
        createLoad(b4, "l4", 2, 0.5);
        createLine(iidm, b1, b2, "z12", 0.0, 0.0);
        createLine(iidm, b2, b3, "z23", 0.0, 0.0);
        createLine(iidm, b1, b3, "z13", 0.0, 0.0);
        createLine(iidm, b3, b4, "l34", 0.1);
        createLine(iidm, b1, b4, "l14", 0.1);
        LfNetwork network = Networks.load(iidm, new LfNetworkParameters()).get(0);
        Map<String, PropagatedContingency> pcs = PropagatedContingency.createList(iidm, List.of(
                        new Contingency("z12", new BranchContingency("z12")),
                        new Contingency("l34", new BranchContingency("l34"))),
                new LfTopoConfig(), new PropagatedContingencyCreationParameters()).stream()
                .collect(Collectors.toMap(pc -> pc.getContingency().getId(), Function.identity()));

        GpuZiTreeReplay replay = new GpuZiTreeReplay(network);
        assertEquals(List.of("z23"), ids(replay.nonTree()), "z23 closes the loop");
        assertEquals(List.of("z12"), ids(replay.applyAndRevert(pcs.get("z12"))), "z12 opened, z23 in the tree");
        assertEquals(List.of("z23"), ids(replay.applyAndRevert(pcs.get("l34"))), "the next contingency starts from the base tree");

        // and that is OLF's own tree after the same open and close
        LfBranch z12 = network.getBranchById("z12");
        z12.setDisabled(true);
        z12.setDisabled(false);
        assertEquals(List.of("z23"), network.getBranches().stream()
                .filter(b -> b.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC)
                        && !b.isSpanningTreeEdge(com.powsybl.openloadflow.network.LoadFlowModel.AC))
                .map(LfBranch::getId).toList());
    }

    private static List<String> ids(List<LfBranch> branches) {
        return branches.stream().map(LfBranch::getId).toList();
    }
}
