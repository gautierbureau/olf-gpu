/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu.dc;

import com.powsybl.action.Action;
import com.powsybl.action.GeneratorActionBuilder;
import com.powsybl.action.SwitchAction;
import com.powsybl.contingency.Contingency;
import com.powsybl.contingency.ContingencyContext;
import com.powsybl.contingency.strategy.OperatorStrategy;
import com.powsybl.contingency.strategy.condition.TrueCondition;
import com.powsybl.iidm.network.*;
import com.powsybl.math.graph.TraverseResult;

import java.util.*;

/**
 * A synthetic topology-optimization study on a node-breaker network: for each contingency, operator strategies that
 * reconfigure buses near it (open or close a bus coupler, open a breaker in the same substation) or redispatch
 * (one generator up, another down, by the same amount). There is no real strategy list for these networks, so the
 * study is generated, deterministically from a seed.
 *
 * <p>Couplers are the breakers the file marks retained (the case's own coupler marking), read when the study is
 * built: OLF resets every node-breaker switch's retained flag when it loads a network, then retains only the switches
 * an action or a contingency operates. Hence {@code retainAllBreakers}: one switch action declared on every breaker
 * of the main synchronous component (a strategy needs not use it), which is how a study keeps every breaker, and so
 * every bus bar section a breaker separates, in OLF's model.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class TopologyStudy {

    record Study(List<Action> actions, List<OperatorStrategy> strategies, String description) {
    }

    /** A bay moved from one bus bar to another: its disconnector to the first opened, the one to the other closed. */
    private record Move(Switch from, Switch to) {
    }

    /** The topological strategy kinds, cycled through: coupler (open or close it), move (a bay to the other bus bar),
     *  breaker (open or close one). */
    static final List<String> DEFAULT_KINDS = List.of("coupler", "move", "breaker", "move");

    private TopologyStudy() {
    }

    static Study generate(Network network, List<Contingency> contingencies, int strategiesPerContingency, double redispatchShare,
                          double redispatchMw, boolean retainAllBreakers, long seed) {
        return generate(network, contingencies, strategiesPerContingency, redispatchShare, redispatchMw, retainAllBreakers, seed, DEFAULT_KINDS);
    }

    static Study generate(Network network, List<Contingency> contingencies, int strategiesPerContingency, double redispatchShare,
                          double redispatchMw, boolean retainAllBreakers, long seed, List<String> kinds) {
        Random random = new Random(seed);
        Set<String> mainVoltageLevels = new HashSet<>();
        for (VoltageLevel vl : network.getVoltageLevels()) {
            for (Bus bus : vl.getBusView().getBuses()) {
                if (bus.getSynchronousComponent() != null && bus.getSynchronousComponent().getNum() == 0) {
                    mainVoltageLevels.add(vl.getId());
                    break;
                }
            }
        }
        List<Switch> couplers = new ArrayList<>();
        List<Switch> breakers = new ArrayList<>();
        Map<String, List<Switch>> couplersBySubstation = new HashMap<>();
        Map<String, List<Switch>> breakersBySubstation = new HashMap<>();
        for (Switch sw : network.getSwitches()) {
            VoltageLevel vl = sw.getVoltageLevel();
            if (sw.getKind() != SwitchKind.BREAKER || vl.getTopologyKind() != TopologyKind.NODE_BREAKER || !mainVoltageLevels.contains(vl.getId())) {
                continue;
            }
            String substation = vl.getSubstation().map(Identifiable::getId).orElse(vl.getId());
            breakers.add(sw);
            breakersBySubstation.computeIfAbsent(substation, k -> new ArrayList<>()).add(sw);
            if (sw.isRetained()) {
                couplers.add(sw);
                couplersBySubstation.computeIfAbsent(substation, k -> new ArrayList<>()).add(sw);
            }
        }
        List<Move> moves = new ArrayList<>();
        Map<String, List<Move>> movesBySubstation = new HashMap<>();
        for (VoltageLevel vl : network.getVoltageLevels()) {
            if (vl.getTopologyKind() == TopologyKind.NODE_BREAKER && mainVoltageLevels.contains(vl.getId())) {
                String substation = vl.getSubstation().map(Identifiable::getId).orElse(vl.getId());
                for (Move move : busBarMoves(vl)) {
                    moves.add(move);
                    movesBySubstation.computeIfAbsent(substation, k -> new ArrayList<>()).add(move);
                }
            }
        }
        List<Generator> up = new ArrayList<>();
        List<Generator> down = new ArrayList<>();
        for (Generator g : network.getGenerators()) {
            Bus bus = g.getTerminal().getBusView().getBus();
            if (bus == null || bus.getSynchronousComponent() == null || bus.getSynchronousComponent().getNum() != 0) {
                continue;
            }
            if (g.getMaxP() - g.getTargetP() >= redispatchMw) {
                up.add(g);
            }
            if (g.getTargetP() - Math.max(0, g.getMinP()) >= redispatchMw) {
                down.add(g);
            }
        }

        Map<String, Action> actions = new LinkedHashMap<>();
        List<OperatorStrategy> strategies = new ArrayList<>();
        int topologyPerContingency = (int) Math.round(strategiesPerContingency * (1 - redispatchShare));
        int[] counts = new int[4];
        for (Contingency contingency : contingencies) {
            List<String> substations = substationsOf(network, contingency);
            for (int k = 0; k < strategiesPerContingency; k++) {
                List<String> actionIds = new ArrayList<>();
                String kind = kinds.get(k % kinds.size());
                if (k < topologyPerContingency && "move".equals(kind)) {
                    Move move = near(random, substations, movesBySubstation, moves);
                    if (move == null) {
                        continue;
                    }
                    actionIds.add(toggle(actions, move.from()).getId());
                    actionIds.add(toggle(actions, move.to()).getId());
                    counts[3]++;
                } else if (k < topologyPerContingency) {
                    boolean coupler = "coupler".equals(kind);
                    Switch sw = near(random, substations, coupler ? couplersBySubstation : breakersBySubstation, coupler ? couplers : breakers);
                    if (sw == null || sw.isOpen() && !bothSidesEnergized(sw)) {
                        continue; // closing towards a bus dead in the base case: out of the fast DC model (no row for it)
                    }
                    actionIds.add(toggle(actions, sw).getId());
                    counts[coupler ? 0 : 1]++;
                } else {
                    if (up.isEmpty() || down.size() < 2) {
                        continue;
                    }
                    Generator g1 = up.get(random.nextInt(up.size()));
                    Generator g2;
                    do {
                        g2 = down.get(random.nextInt(down.size()));
                    } while (g2 == g1);
                    actionIds.add(redispatch(actions, g1, redispatchMw).getId());
                    actionIds.add(redispatch(actions, g2, -redispatchMw).getId());
                    counts[2]++;
                }
                strategies.add(new OperatorStrategy(contingency.getId() + "#" + k, ContingencyContext.specificContingency(contingency.getId()),
                        new TrueCondition(), actionIds));
            }
        }
        if (retainAllBreakers) {
            breakers.forEach(sw -> toggle(actions, sw));
        }
        String description = String.format("%d strategies (%d coupler, %d bus bar move, %d breaker, %d redispatch of %.0f MW) over %d contingencies, "
                        + "%d actions declared%s; %d couplers, %d bay moves, %d breakers in the main component",
                strategies.size(), counts[0], counts[3], counts[1], counts[2], redispatchMw, contingencies.size(), actions.size(),
                retainAllBreakers ? " (one per breaker: every breaker retained)" : "", couplers.size(), moves.size(), breakers.size());
        return new Study(new ArrayList<>(actions.values()), strategies, description);
    }

    /**
     * The bay moves of a node-breaker voltage level: a node with disconnectors to two bus bar sections or more, one closed
     * (the bay's bus bar now) and the others open, each open one onto an energized bus bar.
     */
    private static List<Move> busBarMoves(VoltageLevel vl) {
        VoltageLevel.NodeBreakerView view = vl.getNodeBreakerView();
        Set<Integer> busBarNodes = new HashSet<>();
        view.getBusbarSections().forEach(b -> busBarNodes.add(b.getTerminal().getNodeBreakerView().getNode()));
        Map<Integer, List<Switch>> toBusBar = new TreeMap<>();
        for (Switch sw : view.getSwitches()) {
            if (sw.getKind() != SwitchKind.DISCONNECTOR) {
                continue;
            }
            int n1 = view.getNode1(sw.getId());
            int n2 = view.getNode2(sw.getId());
            if (busBarNodes.contains(n1) != busBarNodes.contains(n2)) {
                toBusBar.computeIfAbsent(busBarNodes.contains(n1) ? n2 : n1, k -> new ArrayList<>()).add(sw);
            }
        }
        List<Move> moves = new ArrayList<>();
        for (List<Switch> disconnectors : toBusBar.values()) {
            List<Switch> closed = disconnectors.stream().filter(sw -> !sw.isOpen()).toList();
            if (closed.size() != 1) {
                continue;
            }
            for (Switch open : disconnectors) {
                if (open.isOpen() && bothSidesEnergized(open)) {
                    moves.add(new Move(closed.getFirst(), open));
                }
            }
        }
        return moves;
    }

    /** An element of one of these substations, else any. */
    private static <T> T near(Random random, List<String> substations, Map<String, List<T>> bySubstation, List<T> all) {
        List<T> candidates = new ArrayList<>();
        for (String s : substations) {
            candidates.addAll(bySubstation.getOrDefault(s, List.of()));
        }
        if (candidates.isEmpty()) {
            candidates = all;
        }
        return candidates.isEmpty() ? null : candidates.get(random.nextInt(candidates.size()));
    }

    private static List<String> substationsOf(Network network, Contingency contingency) {
        List<String> substations = new ArrayList<>();
        contingency.getElements().forEach(e -> {
            Identifiable<?> identifiable = network.getIdentifiable(e.getId());
            if (identifiable instanceof Branch<?> branch) {
                for (Terminal t : List.of(branch.getTerminal1(), branch.getTerminal2())) {
                    substations.add(t.getVoltageLevel().getSubstation().map(Identifiable::getId).orElse(t.getVoltageLevel().getId()));
                }
            }
        });
        return substations;
    }

    /** Whether each side of this (node-breaker) switch reaches an energized terminal through closed switches. */
    private static boolean bothSidesEnergized(Switch sw) {
        VoltageLevel.NodeBreakerView view = sw.getVoltageLevel().getNodeBreakerView();
        return energized(view, view.getNode1(sw.getId()), sw) && energized(view, view.getNode2(sw.getId()), sw);
    }

    private static boolean energized(VoltageLevel.NodeBreakerView view, int node, Switch excluded) {
        boolean[] found = {isEnergized(view, node)};
        if (!found[0]) {
            view.traverse(node, (n1, s, n2) -> {
                if (s == excluded || s != null && s.isOpen()) {
                    return TraverseResult.TERMINATE_PATH;
                }
                if (isEnergized(view, n2)) {
                    found[0] = true;
                    return TraverseResult.TERMINATE_TRAVERSER;
                }
                return TraverseResult.CONTINUE;
            });
        }
        return found[0];
    }

    private static boolean isEnergized(VoltageLevel.NodeBreakerView view, int node) {
        return view.getOptionalTerminal(node).map(t -> t.getBusView().getBus())
                .map(b -> b.getSynchronousComponent() != null && b.getSynchronousComponent().getNum() == 0).orElse(false);
    }

    /** The action switching this switch to the other position. */
    private static Action toggle(Map<String, Action> actions, Switch sw) {
        boolean open = !sw.isOpen();
        String id = (open ? "OPEN_" : "CLOSE_") + sw.getId();
        return actions.computeIfAbsent(id, i -> new SwitchAction(i, sw.getId(), open));
    }

    private static Action redispatch(Map<String, Action> actions, Generator g, double mw) {
        String id = (mw > 0 ? "UP_" : "DOWN_") + g.getId();
        return actions.computeIfAbsent(id, i -> new GeneratorActionBuilder().withId(i).withGeneratorId(g.getId())
                .withActivePowerRelativeValue(true).withActivePowerValue(mw).build());
    }
}
