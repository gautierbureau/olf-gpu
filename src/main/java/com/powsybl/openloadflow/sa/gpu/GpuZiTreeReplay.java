/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.openloadflow.graph.GraphConnectivity;
import com.powsybl.openloadflow.network.DisabledBranchStatus;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfContingency;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfZeroImpedanceNetwork;
import com.powsybl.openloadflow.network.LoadFlowModel;
import com.powsybl.openloadflow.network.impl.PropagatedContingency;
import org.jgrapht.Graph;
import org.jgrapht.Graphs;
import org.jgrapht.alg.connectivity.ConnectivityInspector;
import org.jgrapht.alg.spanning.KruskalMinimumSpanningTree;
import org.jgrapht.graph.AsSubgraph;
import org.jgrapht.graph.Pseudograph;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * OLF's AC zero-impedance spanning trees, replayed contingency by contingency on PRIVATE copies of its
 * graphs. OLF's tree is history-dependent: a contingency that disables a tree coupler re-elects the tree
 * of its network, and the revert only re-adds the edge ({@code LfZeroImpedanceNetwork#addBranch}), so the
 * tree OLF solves the k-th contingency on depends on every contingency before it. The device extracted
 * its structure on the BASE tree, so it must be told, per contingency, which couplers are out of OLF's
 * tree at that point.
 *
 * <p>The replay runs the SAME jgrapht operations OLF runs, in the same order, so Kruskal sees the same
 * edge iteration order and breaks its (all-equal-weight) ties identically:
 * {@code removeBranchAndTryToSplit} on every BOTH_SIDES-disabled zero-impedance branch of the
 * contingency's {@link LfContingency} in its own order, then the {@code NetworkState} restore in branch
 * number order ({@code addBranch} within a network, {@code addBranchAndMerge} across two). Nothing on the
 * {@link LfNetwork} is mutated: the tree flags live in {@link #inTree}, the graphs are copies, and the
 * connectivity analysis is wrapped in OLF's own temporary changes.
 *
 * <p>It replaces a map that kept ONE non-tree coupler per network: a network with several independent
 * cycles has several, and every forgotten one was re-coupled by the device, putting a duplicate zero
 * equation on a zero-diagonal dummy row (network.xiidm busbar scope: 0 of 150 on device).
 */
final class GpuZiTreeReplay {

    private static final LoadFlowModel AC = LoadFlowModel.AC;

    private final LfNetwork network;
    private final Map<LfBus, Graph<LfBus, LfBranch>> netOf = new HashMap<>();
    private final Set<LfBranch> inTree = new HashSet<>();
    private final List<LfBranch> ziBranches;

    GpuZiTreeReplay(LfNetwork network) {
        this.network = network;
        for (LfZeroImpedanceNetwork zn : network.getZeroImpedanceNetworks(AC)) {
            Graph<LfBus, LfBranch> copy = new Pseudograph<>(LfBranch.class);
            Graphs.addGraph(copy, zn.getGraph());           // same vertex and edge iteration order
            register(copy);
        }
        this.ziBranches = network.getBranches().stream().filter(b -> b.isZeroImpedance(AC)).toList();
        for (LfBranch b : ziBranches) {
            if (b.isSpanningTreeEdge(AC)) {
                inTree.add(b);
            }
        }
    }

    /** The zero-impedance branches out of OLF's tree right now (branch order), the base's included. */
    List<LfBranch> nonTree() {
        return ziBranches.stream().filter(b -> !inTree.contains(b)).toList();
    }

    /**
     * Applies the contingency to the shadow trees, reads the tree OLF SOLVES it on, and reverts it, exactly
     * as OLF's SA loop does. The answer is the POST-apply tree, not the one OLF reached the contingency
     * with: disabling a tree coupler re-elects inside the apply. Its own disabled couplers come back as
     * non-tree too, which keeps them out of the device's re-coupling. Measured on network.xiidm's busbar
     * window: the pre-apply tree left AATX/AATY 4.8/2.5 MW and ADJN/ADJO 32 MW off the CPU - exactly
     * the four contingencies whose apply re-elects.
     */
    List<LfBranch> applyAndRevert(PropagatedContingency pc) {
        Optional<LfContingency> lfc = pc.toLfContingency(network, false, GpuZiTreeReplay::connectivityLoss);
        if (lfc.isEmpty()) {
            return nonTree();                                // no impact: OLF never applies it
        }
        List<LfBranch> disabled = new ArrayList<>();
        for (Map.Entry<LfBranch, DisabledBranchStatus> e : lfc.get().getDisabledNetwork().getBranchesStatus().entrySet()) {
            LfBranch b = e.getKey();
            if (e.getValue() == DisabledBranchStatus.BOTH_SIDES && !b.isDisabled() && b.isZeroImpedance(AC)
                    && b.getBus1() != null && b.getBus2() != null) {
                disabled.add(b);
            }
        }
        for (LfBranch b : disabled) {
            removeBranchAndTryToSplit(b);
        }
        List<LfBranch> solvedOn = nonTree();
        disabled.sort(Comparator.comparingInt(LfBranch::getNum));
        for (LfBranch b : disabled) {
            Graph<LfBus, LfBranch> g1 = netOf.get(b.getBus1());
            Graph<LfBus, LfBranch> g2 = netOf.get(b.getBus2());
            if (g1 == g2) {
                g1.addEdge(b.getBus1(), b.getBus2(), b);     // addBranch: the tree is NOT re-elected
            } else {
                Graph<LfBus, LfBranch> merged = new Pseudograph<>(LfBranch.class);
                Graphs.addGraph(merged, g1);
                Graphs.addGraph(merged, g2);
                merged.addEdge(b.getBus1(), b.getBus2(), b);
                register(merged);
            }
        }
        return solvedOn;
    }

    private void removeBranchAndTryToSplit(LfBranch b) {
        Graph<LfBus, LfBranch> graph = netOf.get(b.getBus1());
        graph.removeEdge(b);
        List<Set<LfBus>> sets = new ConnectivityInspector<>(graph).connectedSets();
        if (sets.size() > 1) {
            inTree.remove(b);
            for (Set<LfBus> set : sets) {
                Set<LfBus> sorted = new LinkedHashSet<>();
                set.stream().sorted(Comparator.comparing(LfBus::getId)).forEach(sorted::add);
                Graph<LfBus, LfBranch> sub = new Pseudograph<>(LfBranch.class);
                Graphs.addGraph(sub, new AsSubgraph<>(graph, sorted));
                register(sub);
            }
        } else if (inTree.remove(b)) {
            elect(graph);
        }
    }

    /** The LfZeroImpedanceNetwork constructor: own the buses, elect the tree. */
    private void register(Graph<LfBus, LfBranch> graph) {
        for (LfBus bus : graph.vertexSet()) {
            netOf.put(bus, graph);
        }
        elect(graph);
    }

    private void elect(Graph<LfBus, LfBranch> graph) {
        Set<LfBranch> tree = new KruskalMinimumSpanningTree<>(graph).getSpanningTree().getEdges();
        for (LfBranch e : graph.edgeSet()) {
            if (tree.contains(e)) {
                inTree.add(e);
            } else {
                inTree.remove(e);
            }
        }
    }

    /**
     * OLF's findBusesAndBranchesImpactedBecauseOfConnectivityLoss without its one side effect: where the
     * slack is isolated, OLF writes the excluded slack buses onto the network (restored by its
     * NetworkState, which this replay has no use for) and re-anchors the main component on the relocated
     * slack. The relocated slack lies in the largest component by construction, so anchoring on any bus
     * of it yields the same lost buses.
     */
    private static PropagatedContingency.ContingencyConnectivityLossImpact connectivityLoss(
            LfNetwork network, String contingencyId, Map<LfBranch, DisabledBranchStatus> branchesToOpen,
            boolean relocateSlackBus) {
        GraphConnectivity<LfBus, LfBranch> connectivity = network.getConnectivity();
        connectivity.startTemporaryChanges();
        try {
            branchesToOpen.keySet().stream()
                    .filter(LfBranch::isConnectedAtBothSides)
                    .forEach(connectivity::removeEdge);
            LfBus slack = network.getSynchronousNetworks().getFirst().getSlackBuses().getFirst();
            if (connectivity.getComponentNumber(slack) != 0
                    && connectivity.getLargestConnectedComponent().size() != connectivity.getConnectedComponent(slack).size()) {
                connectivity.setMainComponentVertex(connectivity.getLargestConnectedComponent().iterator().next());
            }
            int created = connectivity.getNbConnectedComponents() - 1;
            Set<LfBus> busesToLost = connectivity.getVerticesRemovedFromMainComponent();
            return new PropagatedContingency.ContingencyConnectivityLossImpact(created, busesToLost,
                    PropagatedContingency.getHvdcsWithoutPower(network, busesToLost, connectivity));
        } finally {
            connectivity.undoTemporaryChanges();
        }
    }
}
