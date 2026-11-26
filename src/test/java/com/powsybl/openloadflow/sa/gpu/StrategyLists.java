/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.action.Action;
import com.powsybl.action.ActionList;
import com.powsybl.action.GeneratorAction;
import com.powsybl.action.LoadAction;
import com.powsybl.action.PhaseTapChangerTapPositionAction;
import com.powsybl.action.SwitchAction;
import com.powsybl.contingency.Contingency;
import com.powsybl.contingency.ContingencyContextType;
import com.powsybl.contingency.strategy.OperatorStrategy;
import com.powsybl.contingency.strategy.OperatorStrategyList;
import com.powsybl.iidm.network.Bus;
import com.powsybl.iidm.network.Network;
import com.powsybl.iidm.network.Switch;
import com.powsybl.iidm.network.Terminal;
import com.powsybl.iidm.network.TopologyKind;
import com.powsybl.iidm.network.VoltageLevel;
import com.powsybl.math.graph.TraverseResult;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Actions and operator strategies read with powsybl's own readers (ActionList, OperatorStrategyList) for a run on a
 * contingency list: the AC comparison and the DC probe read the operator's lists the same way. Strategies on a
 * contingency outside the run's list are dropped, as are actions no kept strategy uses. With action types given
 * (powsybl action type names, e.g. SWITCH, LOAD), only the strategies whose actions are all of these types are kept; a
 * strategy without any action is dropped then. A strategy with an action on an element the network does not have is
 * dropped too: OLF refuses the whole analysis for one such action (Actions.checkValidity). The summary holds counts
 * only (network data rule).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class StrategyLists {

    /** The kept actions and strategies, and the counts of what was kept and dropped. */
    public record Lists(List<Action> actions, List<OperatorStrategy> strategies, String summary) {
    }

    private StrategyLists() {
    }

    /**
     * @param types       the action types a kept strategy is made of, or null: every type
     * @param missingOut  a CSV of every action whose element the network lacks (type, action, element, number of
     *                    strategies using it, the network's element type under that id if any), or null; written to the
     *                    file only, never printed
     */
    public static Lists read(Path actionsFile, Path operatorStrategiesFile, Network network, List<Contingency> contingencies,
                             Set<String> types, Path missingOut) {
        List<Action> allActions = ActionList.readJsonFile(actionsFile).getActions();
        List<OperatorStrategy> all = OperatorStrategyList.read(operatorStrategiesFile).getOperatorStrategies();
        Map<String, Action> actionById = new HashMap<>();
        allActions.forEach(a -> actionById.put(a.getId(), a));
        Set<String> contingencyIds = new HashSet<>();
        contingencies.forEach(c -> contingencyIds.add(c.getId()));
        Map<String, Long> missingByType = new TreeMap<>();
        Set<String> missing = new HashSet<>();
        for (Action a : allActions) {
            if (!elementExists(network, a)) {
                missing.add(a.getId());
                missingByType.merge(a.getType(), 1L, Long::sum);
            }
        }
        if (missingOut != null) {
            writeMissing(missingOut, allActions, all, missing, network);
        }
        List<OperatorStrategy> kept = new ArrayList<>();
        int outsideList = 0;
        int otherTypes = 0;
        int missingElement = 0;
        for (OperatorStrategy s : all) {
            var context = s.getContingencyContext();
            if (context.getContextType() == ContingencyContextType.SPECIFIC && !contingencyIds.contains(context.getContingencyId())) {
                outsideList++;
                continue;
            }
            List<String> ids = s.getConditionalActions().stream().flatMap(ca -> ca.getActionIds().stream()).toList();
            if (types != null && (ids.isEmpty() || !ids.stream().allMatch(id -> types.contains(actionById.get(id).getType())))) {
                otherTypes++;
                continue;
            }
            if (ids.stream().anyMatch(missing::contains)) {
                missingElement++;
                continue;
            }
            kept.add(s);
        }
        Set<String> used = new HashSet<>();
        kept.forEach(s -> s.getConditionalActions().forEach(ca -> used.addAll(ca.getActionIds())));
        Map<String, Long> usedTypes = new TreeMap<>();
        List<Action> acts = new ArrayList<>();
        for (Action a : allActions) {
            if (used.contains(a.getId())) {
                acts.add(a);
                usedTypes.merge(a.getType(), 1L, Long::sum);
            }
        }
        String summary = String.format("%d of %d kept (%d on a contingency outside the list, %d with other action types,"
                        + " %d with an action on an element the network lacks) on %d contingencies, %d actions %s"
                        + " | actions on a missing element: %s", kept.size(), all.size(), outsideList, otherTypes, missingElement,
                kept.stream().map(x -> x.getContingencyContext().getContingencyId()).distinct().count(), acts.size(), usedTypes,
                missingByType);
        return new Lists(List.copyOf(acts), List.copyOf(kept), summary);
    }

    /**
     * The lists without the strategies having a switch action that closes a switch towards a part of the network that is
     * dead in the base case (no terminal of the main synchronous component on one side of the open switch). OLF's fast DC
     * security analysis rejects the whole analysis for one such action: its Woodbury update reuses the base case's
     * matrix, which has no row for such a part. Counts only are added to the summary.
     */
    public static Lists withoutClosesTowardsDeadParts(Lists lists, Network network) {
        Set<String> dead = new HashSet<>();
        Map<String, Long> deadByType = new TreeMap<>();
        for (Action a : lists.actions()) {
            if (a instanceof SwitchAction s && closesTowardsDeadPart(network, s)) {
                dead.add(a.getId());
                deadByType.merge(a.getType(), 1L, Long::sum);
            }
        }
        List<OperatorStrategy> kept = lists.strategies().stream()
                .filter(st -> st.getConditionalActions().stream().flatMap(ca -> ca.getActionIds().stream()).noneMatch(dead::contains))
                .toList();
        Set<String> used = new HashSet<>();
        kept.forEach(st -> st.getConditionalActions().forEach(ca -> used.addAll(ca.getActionIds())));
        List<Action> acts = lists.actions().stream().filter(a -> used.contains(a.getId())).toList();
        String summary = String.format("%s | closing towards a part dead in the base case: %d actions, %d strategies left out -> %d strategies"
                        + " on %d contingencies, %d actions", lists.summary(), dead.size(), lists.strategies().size() - kept.size(), kept.size(),
                kept.stream().map(x -> x.getContingencyContext().getContingencyId()).distinct().count(), acts.size());
        return new Lists(acts, kept, summary);
    }

    /** Whether a switch action closes an open switch with one side reaching no terminal of the main synchronous component. */
    static boolean closesTowardsDeadPart(Network network, SwitchAction action) {
        Switch sw = network.getSwitch(action.getSwitchId());
        if (sw == null || action.isOpen() || !sw.isOpen()) {
            return false;
        }
        VoltageLevel vl = sw.getVoltageLevel();
        if (vl.getTopologyKind() == TopologyKind.NODE_BREAKER) {
            VoltageLevel.NodeBreakerView view = vl.getNodeBreakerView();
            return !sideIsLive(view, view.getNode1(sw.getId())) || !sideIsLive(view, view.getNode2(sw.getId()));
        }
        return !isLive(vl.getBusBreakerView().getBus1(sw.getId()).getConnectedTerminals())
                || !isLive(vl.getBusBreakerView().getBus2(sw.getId()).getConnectedTerminals());
    }

    /** Whether the nodes reached from a node without crossing an open switch hold a terminal of the main synchronous component. */
    private static boolean sideIsLive(VoltageLevel.NodeBreakerView view, int node) {
        List<Terminal> terminals = new ArrayList<>();
        view.getOptionalTerminal(node).ifPresent(terminals::add);
        view.traverse(node, (nodeBefore, sw, nodeAfter) -> {
            if (sw != null && sw.isOpen()) {
                return TraverseResult.TERMINATE_PATH;
            }
            view.getOptionalTerminal(nodeAfter).ifPresent(terminals::add);
            return TraverseResult.CONTINUE;
        });
        return isLive(terminals);
    }

    private static boolean isLive(Iterable<? extends Terminal> terminals) {
        for (Terminal t : terminals) {
            Bus bus = t.getBusView().getBus();
            if (bus != null && bus.isInMainSynchronousComponent()) {
                return true;
            }
        }
        return false;
    }

    /** The action types of an environment variable (comma separated), or null when it is unset or empty: every type. */
    public static Set<String> types(String env) {
        String value = System.getenv(env);
        return value == null || value.isEmpty() ? null : Set.of(value.split(","));
    }

    private static void writeMissing(Path missingOut, List<Action> allActions, List<OperatorStrategy> all, Set<String> missing,
                                     Network network) {
        Map<String, Long> strategiesByAction = new HashMap<>();
        all.forEach(s -> s.getConditionalActions().forEach(ca -> ca.getActionIds().forEach(id -> strategiesByAction.merge(id, 1L, Long::sum))));
        // foundAs: what the network has under that id instead (another equipment type), empty when nothing
        List<String> lines = new ArrayList<>(List.of("type;actionId;elementId;strategies;foundAs"));
        for (Action a : allActions) {
            if (missing.contains(a.getId())) {
                var found = network.getIdentifiable(actionElementId(a));
                lines.add(a.getType() + ";" + a.getId() + ";" + actionElementId(a) + ";" + strategiesByAction.getOrDefault(a.getId(), 0L)
                        + ";" + (found == null ? "" : found.getType()));
            }
        }
        try {
            Files.write(missingOut, lines);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** The element an action of the loaded lists acts on. */
    private static String actionElementId(Action a) {
        return switch (a) {
            case SwitchAction s -> s.getSwitchId();
            case LoadAction l -> l.getLoadId();
            case GeneratorAction g -> g.getGeneratorId();
            case PhaseTapChangerTapPositionAction p -> p.getTransformerId();
            default -> "";
        };
    }

    /** Whether the network has the element an action of the loaded lists acts on (the types OLF checks). */
    private static boolean elementExists(Network network, Action a) {
        return switch (a) {
            case SwitchAction s -> network.getSwitch(s.getSwitchId()) != null;
            case LoadAction l -> network.getLoad(l.getLoadId()) != null;
            case GeneratorAction g -> network.getGenerator(g.getGeneratorId()) != null;
            case PhaseTapChangerTapPositionAction p -> network.getTwoWindingsTransformer(p.getTransformerId()) != null
                    || network.getThreeWindingsTransformer(p.getTransformerId()) != null;
            default -> true;
        };
    }
}
