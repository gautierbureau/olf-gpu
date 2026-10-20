/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.network.gpu;

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
import org.jgrapht.graph.AsWeightedGraph;
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
 * graphs. A contingency that disables a tree coupler re-elects the tree of its network inside the apply, so
 * the device, which extracted its structure on the BASE tree, must be told which couplers are out of OLF's
 * tree for that contingency. OLF elects by Kruskal weighted by branch number and re-elects when a coupler
 * comes back (olf-local-patches 7dd48d92b): the tree a contingency starts from is the base one, whatever ran
 * before. Before that fix the tree depended on the contingencies run before it, and this replay followed it.
 *
 * <p>The replay runs the SAME jgrapht operations OLF runs, in the same order:
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
public final class ZeroImpedanceTreeReplay {

    private static final LoadFlowModel AC = LoadFlowModel.AC;

    private final LfNetwork network;
    private final Map<LfBus, Graph<LfBus, LfBranch>> netOf = new HashMap<>();
    private final Set<LfBranch> inTree = new HashSet<>();
    private final List<LfBranch> zeroImpedanceBranches;

    public ZeroImpedanceTreeReplay(LfNetwork network) {
        this.network = network;
        for (LfZeroImpedanceNetwork zn : network.getZeroImpedanceNetworks(AC)) {
            Graph<LfBus, LfBranch> copy = new Pseudograph<>(LfBranch.class);
            Graphs.addGraph(copy, zn.getGraph());           // same vertex and edge iteration order
            register(copy);
        }
        this.zeroImpedanceBranches = network.getBranches().stream().filter(b -> b.isZeroImpedance(AC)).toList();
        for (LfBranch b : zeroImpedanceBranches) {
            if (b.isSpanningTreeEdge(AC)) {
                inTree.add(b);
            }
        }
    }

    /** The zero-impedance branches out of OLF's tree right now (branch order), the base's included. */
    List<LfBranch> nonTree() {
        return zeroImpedanceBranches.stream().filter(b -> !inTree.contains(b)).toList();
    }

    /**
     * Applies the contingency to the shadow trees, reads the tree OLF SOLVES it on, and reverts it, exactly
     * as OLF's SA loop does. The answer is the POST-apply tree, not the one OLF reached the contingency
     * with: disabling a tree coupler re-elects inside the apply. Its own disabled couplers come back as
     * non-tree too, which keeps them out of the device's re-coupling. Measured on network.xiidm's busbar
     * window: the pre-apply tree left AATX/AATY 4.8/2.5 MW and ADJN/ADJO 32 MW off the CPU - exactly
     * the four contingencies whose apply re-elects.
     */
    public List<LfBranch> applyAndRevert(PropagatedContingency propagatedContingency) {
        return applyAndRevert(propagatedContingency, List.of(), List.of());
    }

    /**
     * The same for a strategy pair: the contingency, then the strategy's coupler openings ({@code opened}) and
     * closings ({@code closed}: a coupler the contingency opened, or one open at base), the tree read, everything
     * reverted. A pair otherwise inherits its contingency's tree, which still holds a reclosed coupler out of it.
     */
    public List<LfBranch> applyAndRevert(PropagatedContingency propagatedContingency, List<LfBranch> opened, List<LfBranch> closed) {
        Optional<LfContingency> lfContingency = propagatedContingency.toLfContingency(network, false, ZeroImpedanceTreeReplay::connectivityLoss);
        if (lfContingency.isEmpty() && opened.isEmpty() && closed.isEmpty()) {
            return nonTree();                                // no impact: OLF never applies it
        }
        List<LfBranch> disabled = new ArrayList<>();
        if (lfContingency.isPresent()) {
            for (Map.Entry<LfBranch, DisabledBranchStatus> e : lfContingency.get().getDisabledNetwork().getBranchesStatus().entrySet()) {
                LfBranch b = e.getKey();
                if (e.getValue() == DisabledBranchStatus.BOTH_SIDES && !b.isDisabled() && b.isZeroImpedance(AC)
                        && b.getBus1() != null && b.getBus2() != null && !closed.contains(b)) {
                    disabled.add(b);
                }
            }
        }
        for (LfBranch b : opened) {
            if (!b.isDisabled() && b.isZeroImpedance(AC) && b.getBus1() != null && b.getBus2() != null && !disabled.contains(b)) {
                disabled.add(b);
            }
        }
        for (LfBranch b : disabled) {
            removeBranchAndTryToSplit(b);
        }
        List<LfBranch> added = new ArrayList<>();         // couplers open at base that the strategy closes
        for (LfBranch b : closed) {
            if (b.isDisabled() && b.isZeroImpedance(AC) && b.getBus1() != null && b.getBus2() != null) {
                addBranch(b);
                added.add(b);
            }
        }
        List<LfBranch> solvedOn = nonTree();
        for (LfBranch b : added) {
            removeBranchAndTryToSplit(b);
        }
        disabled.sort(Comparator.comparingInt(LfBranch::getNum));
        for (LfBranch b : disabled) {
            Graph<LfBus, LfBranch> graph1 = netOf.get(b.getBus1());
            Graph<LfBus, LfBranch> graph2 = netOf.get(b.getBus2());
            if (graph1 == graph2) {
                graph1.addEdge(b.getBus1(), b.getBus2(), b);     // addBranch: the tree is re-elected
                elect(graph1);
            } else {
                Graph<LfBus, LfBranch> merged = new Pseudograph<>(LfBranch.class);
                Graphs.addGraph(merged, graph1);
                Graphs.addGraph(merged, graph2);
                merged.addEdge(b.getBus1(), b.getBus2(), b);
                register(merged);
            }
        }
        return solvedOn;
    }

    /** OLF's addBranch: merge the two networks the coupler joins (a lone bus is its own), or re-elect within one. */
    private void addBranch(LfBranch b) {
        Graph<LfBus, LfBranch> graph1 = graphOf(b.getBus1());
        Graph<LfBus, LfBranch> graph2 = graphOf(b.getBus2());
        if (graph1 == graph2) {
            graph1.addEdge(b.getBus1(), b.getBus2(), b);
            elect(graph1);
        } else {
            Graph<LfBus, LfBranch> merged = new Pseudograph<>(LfBranch.class);
            Graphs.addGraph(merged, graph1);
            Graphs.addGraph(merged, graph2);
            merged.addEdge(b.getBus1(), b.getBus2(), b);
            register(merged);
        }
    }

    private Graph<LfBus, LfBranch> graphOf(LfBus bus) {
        return netOf.computeIfAbsent(bus, k -> {
            Graph<LfBus, LfBranch> single = new Pseudograph<>(LfBranch.class);
            single.addVertex(k);
            return single;
        });
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
                Graph<LfBus, LfBranch> subGraph = new Pseudograph<>(LfBranch.class);
                Graphs.addGraph(subGraph, new AsSubgraph<>(graph, sorted));
                register(subGraph);
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
        // weighted by branch number like LfZeroImpedanceNetwork#updateSpanningTree: the tree is a function of
        // the enabled couplers alone, whatever order the graph received them in
        Set<LfBranch> tree = new KruskalMinimumSpanningTree<>(new AsWeightedGraph<>(graph, b -> (double) b.getNum(), true, false))
                .getSpanningTree().getEdges();
        for (LfBranch e : graph.edgeSet()) {
            if (tree.contains(e)) {
                inTree.add(e);
            } else {
                inTree.remove(e);
            }
        }
    }

    /**
     * OLF's findBusesAndBranchesImpactedBecauseOfConnectivityLoss without its side effects: where the
     * slack is isolated, OLF writes the excluded slack buses onto the network (restored by its
     * NetworkState, which this replay has no use for) and re-anchors the main component on the relocated
     * slack. The relocated slack lies in the largest component by construction, so anchoring on any bus
     * of it yields the same lost buses.
     *
     * <p>The anchor is PUT BACK afterwards. {@code setMainComponentVertex} also sets the connectivity's
     * DEFAULT main-component vertex, the one every later query without an anchor of its own starts from
     * (this replay's, and OLF's own toLfContingency for the contingencies the CPU replays), and OLF only
     * restores it with the slack, through its NetworkState. Left on the first bus of a hash-ordered set,
     * a different bus each run, the next contingency in the list that lost THAT bus read the whole
     * network as lost: every coupler disabled, the replay's tree empty, and the device solved it with all
     * 366 of rte6515's couplers decoupled - one random busbar section per full run failing its linear solve
     * at a ratio of 3.06e+06 (BUS-5243, BUS-506, BUS-747, BUS-528, BUS-608 on successive runs).
     */
    private static PropagatedContingency.ContingencyConnectivityLossImpact connectivityLoss(
            LfNetwork network, String contingencyId, Map<LfBranch, DisabledBranchStatus> branchesToOpen,
            boolean relocateSlackBus) {
        GraphConnectivity<LfBus, LfBranch> connectivity = network.getConnectivity();
        LfBus slack = network.getSynchronousNetworks().getFirst().getSlackBuses().getFirst();
        boolean reAnchored = false;
        connectivity.startTemporaryChanges();
        try {
            branchesToOpen.keySet().stream()
                    .filter(LfBranch::isConnectedAtBothSides)
                    .forEach(connectivity::removeEdge);
            if (connectivity.getComponentNumber(slack) != 0
                    && connectivity.getLargestConnectedComponent().size() != connectivity.getConnectedComponent(slack).size()) {
                connectivity.setMainComponentVertex(connectivity.getLargestConnectedComponent().iterator().next());
                reAnchored = true;
            }
            int created = connectivity.getNbConnectedComponents() - 1;
            Set<LfBus> busesToLost = connectivity.getVerticesRemovedFromMainComponent();
            return new PropagatedContingency.ContingencyConnectivityLossImpact(created, busesToLost,
                    PropagatedContingency.getHvdcsWithoutPower(network, busesToLost, connectivity));
        } finally {
            connectivity.undoTemporaryChanges();
            if (reAnchored) {
                connectivity.setMainComponentVertex(slack);  // the default OLF keeps between contingencies
            }
        }
    }
}
