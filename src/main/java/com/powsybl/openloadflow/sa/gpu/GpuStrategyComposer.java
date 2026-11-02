/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.action.Action;
import com.powsybl.contingency.strategy.OperatorStrategy;
import com.powsybl.iidm.network.Network;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuTransformerVoltageControlOuterLoop;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.BatchContingency;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.LoadLoss;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.ShuntShift;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.TapR1Override;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfShunt;
import com.powsybl.openloadflow.util.PerUnit;

import java.util.List;
import java.util.Map;

/**
 * Compiles a stage-1 operator strategy into ONE extra batch scenario: a single-stage TrueCondition
 * strategy whose actions (branch open/reclose, switch open/close, load/generator active-power
 * redispatch, RTC/PST tap positions, shunt section changes) are COMPOSED onto the parent
 * contingency's device deltas as another {@link BatchContingency}. Returns {@code null} for anything
 * out of stage-1 scope, leaving the (contingency, strategy) pair to the CPU per-strategy path.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuStrategyComposer {

    private GpuStrategyComposer() {
    }

    /** Stage-1 applicability: single-stage strategy with a TrueCondition whose contingency context
     *  covers this contingency. Anything else stays on the CPU per-strategy path. */
    static boolean strategyApplies(OperatorStrategy strategy, String contingencyId) {
        if (candidateActionLists(strategy).isEmpty()) {
            return false;                                    // too many conditional stages to speculate on
        }
        // ANY single-stage condition qualifies — including the violation/threshold conditions: the
        // ENGINE evaluates the condition (AbstractSecurityAnalysis.checkCondition, on the CPU-detected
        // post-contingency violations) BEFORE our runActionSimulation override is invoked, so the pair
        // state is precomputed SPECULATIVELY and simply goes unused when the condition does not pass.
        // Condition semantics therefore stay 100% stock — the batch only accelerates the solve.
        var contingencyContext = strategy.getContingencyContext();
        return switch (contingencyContext.getContextType()) {
            case ALL, ONLY_CONTINGENCIES -> true;
            case SPECIFIC -> contingencyId.equals(contingencyContext.getContingencyId());
            case NONE -> false;
        };
    }

    /** Conditional stages a strategy may carry and still be precomputed: every combination is one pair. */
    static final int MAX_CONDITIONAL_STAGES = 3;

    /**
     * The action lists OLF may apply for this strategy. OLF evaluates EVERY stage's condition on the same post-contingency
     * violations and applies the actions of the stages that hold, concatenated in stage order
     * (AbstractSecurityAnalysis.checkCondition); none holding means no strategy result. The always-true stages are in
     * every list; each combination of the conditional ones gives one list, precomputed speculatively. Empty when the
     * strategy has more than {@link #MAX_CONDITIONAL_STAGES} conditional stages.
     */
    static List<List<String>> candidateActionLists(OperatorStrategy strategy) {
        List<com.powsybl.contingency.strategy.ConditionalActions> stages = strategy.getConditionalActions();
        List<Integer> conditional = new java.util.ArrayList<>();
        for (int i = 0; i < stages.size(); i++) {
            if (!(stages.get(i).getCondition() instanceof com.powsybl.contingency.strategy.condition.TrueCondition)) {
                conditional.add(i);
            }
        }
        if (conditional.size() > MAX_CONDITIONAL_STAGES) {
            return List.of();
        }
        java.util.LinkedHashSet<List<String>> lists = new java.util.LinkedHashSet<>();
        for (int mask = 0; mask < 1 << conditional.size(); mask++) {
            List<String> ids = new java.util.ArrayList<>();
            for (int i = 0; i < stages.size(); i++) {
                int c = conditional.indexOf(i);
                if (c < 0 || (mask & 1 << c) != 0) {
                    ids.addAll(stages.get(i).getActionIds());
                }
            }
            if (!ids.isEmpty()) {
                lists.add(List.copyOf(ids));
            }
        }
        return List.copyOf(lists);
    }

    /** The batch id of a (contingency, strategy) pair applying {@code actionIds}: a single-stage strategy has one
     *  candidate list and keeps the plain id; a multi-stage one adds the list, which is what OLF hands
     *  runActionSimulation once its conditions are evaluated. */
    static String pairId(String contingencyId, OperatorStrategy strategy, List<String> actionIds) {
        String id = contingencyId + BatchContingency.STRATEGY_KEY_SEP + strategy.getId();
        return strategy.getConditionalActions().size() == 1 ? id
                : id + BatchContingency.STRATEGY_KEY_SEP + String.join("\u0002", actionIds);
    }

    /** Compile the strategy's actions onto the contingency's deltas into ONE extra batch scenario,
     *  or null when any action is out of stage-1 scope (branch open/RECLOSE, LoadAction,
     *  GeneratorAction — absolute or relative active power). {@code ctgLostLoadIds} are the loads the
     *  contingency itself loses, on which a load action is a no-op. */
    static BatchContingency composeStrategyScenario(Network network, LfNetwork net, String contingencyId,
                                                    OperatorStrategy strategy, List<String> actionIds, List<Action> currentActions,
                                                    List<LfBranch> contingencyZeroImpedanceBranchOutages,
                                                    List<LfBranch> ctgBranches,
                                                    List<LfGenerator> ctgGens,
                                                    List<LoadLoss> ctgLoads,
                                                    java.util.Set<String> ctgLostLoadIds,
                                                    List<ShuntShift> ctgShifts,
                                                    List<com.powsybl.openloadflow.network.LfHvdc> ctgHvdcs,
                                                    List<BatchContingency.SideOpen> ctgSideOpens,
                                                    List<LfBranch> contingencyStickyNonSpanningTreeBranches,
                                                    com.powsybl.openloadflow.network.LfNetworkParameters networkParameters,
                                                    com.powsybl.loadflow.LoadFlowParameters.BalanceType balanceType) {
        List<LfBranch> branches = new java.util.ArrayList<>(ctgBranches);
        List<LfBranch> zeroImpedanceOutages = new java.util.ArrayList<>(contingencyZeroImpedanceBranchOutages);
        List<LoadLoss> loads = new java.util.ArrayList<>(ctgLoads);
        List<ShuntShift> shifts = new java.util.ArrayList<>(ctgShifts);
        List<TapR1Override> taps = new java.util.ArrayList<>();
        List<LfBranch> reclosed = new java.util.ArrayList<>();
        List<LfBranch> reenabled = new java.util.ArrayList<>();      // contingency outages a close puts back
        List<LfBranch> opened = new java.util.ArrayList<>();         // branches an open takes out
        List<BatchContingency.GeneratorMove> generatorActions = new java.util.ArrayList<>();
        Map<LfGenerator, Double> generatorNetMove = new java.util.LinkedHashMap<>();
        List<com.powsybl.openloadflow.network.LfHvdc> acEmulationOff = new java.util.ArrayList<>();
        Map<String, Action> actionById = new java.util.HashMap<>();
        for (Action a : currentActions) {
            actionById.put(a.getId(), a);
        }
        for (String actionId : actionIds) {
            Action a = actionById.get(actionId);
            if (a instanceof com.powsybl.action.TerminalsConnectionAction terminalsConnectionAction) {
                // OLF operates every LfBranch of the equipment (getBranchesByOriginalId): the three legs of a
                // three-winding transformer, the branch itself otherwise
                LfBranch single = net.getBranchById(terminalsConnectionAction.getElementId());
                List<LfBranch> legs = single != null ? List.of(single) : net.getBranchesByOriginalId(terminalsConnectionAction.getElementId());
                if (legs == null || legs.isEmpty()) {
                    // Not a branch: a SHUNT compensator's terminals (OLF supports (re)connection since
                    // the olf-local-patches shuntIdsToClose retention) — composed as a per-scenario
                    // admittance shift: the retained base-open compensator contributes 0 at base, so
                    // close = +its full section admittance; open of a connected one = -it. LINEAR only.
                    com.powsybl.iidm.network.ShuntCompensator shuntCompensator = network.getShuntCompensator(terminalsConnectionAction.getElementId());
                    LfShunt lfShunt = shuntCompensator != null ? net.getShuntById(terminalsConnectionAction.getElementId()) : null;
                    if (shuntCompensator == null || lfShunt == null
                            || !(shuntCompensator.getModel() instanceof com.powsybl.iidm.network.ShuntCompensatorLinearModel linearModel)) {
                        return null;
                    }
                    // NEVER from the iidm terminal: during the SA the working variant is the TEMPORARY
                    // one where the retention machinery already connected the closable shunt — ask the
                    // LF model, which tracks the compensator's true base state.
                    boolean connectedAtBase = !lfShunt.isCompensatorInitiallyDisconnected(terminalsConnectionAction.getElementId());
                    if (terminalsConnectionAction.isOpen() == !connectedAtBase) {
                        continue;                            // open-of-disconnected / close-of-connected: no-op
                    }
                    double gPerSection = Double.isNaN(linearModel.getGPerSection()) ? 0 : linearModel.getGPerSection();
                    double zb = shuntCompensator.getTerminal().getVoltageLevel().getNominalV()
                            * shuntCompensator.getTerminal().getVoltageLevel().getNominalV() / PerUnit.SB;
                    double sign = terminalsConnectionAction.isOpen() ? -1 : 1;
                    double n = shuntCompensator.getSectionCount();
                    shifts.add(new ShuntShift(lfShunt,
                            sign * gPerSection * n * zb, sign * linearModel.getBPerSection() * n * zb));
                    continue;
                }
                for (LfBranch b : legs) {
                    if (b.getBus1() == null || b.getBus2() == null) {
                        continue;                            // OLF operates a branch with both ends only: as it is
                    }
                    if (ctgSideOpens.stream().anyMatch(sideOpen -> sideOpen.branch() == b)) {
                        return null;                         // the contingency opened ONE side of it: CPU
                    }
                    if (terminalsConnectionAction.isOpen()) {
                        if (!branches.contains(b)) {
                            branches.add(b);
                            opened.add(b);
                        }
                    } else {
                        if (branches.remove(b)) {
                            reenabled.add(b);
                        } else {
                            // close of a BASE-OPEN branch (retained disabled by LfTopoConfig): batched as a
                            // per-scenario mask CLEAR on the base-open packed element — dead-island
                            // reconnection (a disabled endpoint bus) is rejected at qualification.
                            if (!b.isDisabled()) {
                                return null;                 // closing an already-closed branch — CPU decides
                            }
                            reclosed.add(b);
                        }
                    }
                }
            } else if (a instanceof com.powsybl.action.SwitchAction switchAction) {
                if (!composeSwitchAction(net, switchAction, ctgSideOpens, branches, zeroImpedanceOutages, opened, reclosed)) {
                    return null;
                }
            } else if (a instanceof com.powsybl.action.LoadAction loadAction) {
                com.powsybl.iidm.network.Load load = network.getLoad(loadAction.getLoadId());
                if (load == null) {
                    return null;
                }
                // OLF's own load -> LfBus map, never the bus view (see GpuContingencyMapper.loadOutages)
                com.powsybl.openloadflow.network.LfLoad lfLoad = net.getLoadById(loadAction.getLoadId());
                LfBus bus = lfLoad != null ? lfLoad.getBus() : null;
                if (bus == null) {
                    return null;
                }
                if (ctgLostLoadIds.contains(loadAction.getLoadId())) {
                    continue;                                // OLF's LfLoadAction does nothing on a load the contingency lost
                }
                // LfLoadAction's power shift: the value itself when relative, value - P0 (Q0) when absolute, 0 for
                // the part the action leaves out. LoadLoss carries the load REMOVED, hence the sign (pu).
                double dP = loadAction.getActivePowerValue().isPresent()
                        ? -(loadAction.isRelativeValue() ? loadAction.getActivePowerValue().getAsDouble() : loadAction.getActivePowerValue().getAsDouble() - load.getP0())
                        : 0.0;
                double dQ = loadAction.getReactivePowerValue().isPresent()
                        ? -(loadAction.isRelativeValue() ? loadAction.getReactivePowerValue().getAsDouble() : loadAction.getReactivePowerValue().getAsDouble() - load.getQ0())
                        : 0.0;
                loads.add(new LoadLoss(bus, dP / PerUnit.SB, dQ / PerUnit.SB));
            } else if (a instanceof com.powsybl.action.RatioTapChangerTapPositionAction ratioTapChangerAction) {
                var tapChanger = network.getTwoWindingsTransformer(ratioTapChangerAction.getTransformerId()) != null
                        ? network.getTwoWindingsTransformer(ratioTapChangerAction.getTransformerId()).getRatioTapChanger() : null;
                /** Compiles a tap-position action (RTC or PST) into a per-scenario full-row override — the
                 *  target tap's [y, g1, g2, b1, b2, ksi, r1, a1] replaces the branch's static inputs, so
                 *  per-tap impedance changes are covered. Null when out of batch scope: unknown transformer /
                 *  tap changer, no tap table, or out-of-range position. */
                TapR1Override override = tapOverride(net, ratioTapChangerAction.getTransformerId(),
                        tapChanger, ratioTapChangerAction.isRelativeValue(), ratioTapChangerAction.getTapPosition());
                if (override == null) {
                    return null;
                }
                taps.add(override);
            } else if (a instanceof com.powsybl.action.PhaseTapChangerTapPositionAction phaseTapChangerAction) {
                var tapChanger = network.getTwoWindingsTransformer(phaseTapChangerAction.getTransformerId()) != null
                        ? network.getTwoWindingsTransformer(phaseTapChangerAction.getTransformerId()).getPhaseTapChanger() : null;
                TapR1Override override = tapOverride(net, phaseTapChangerAction.getTransformerId(),
                        tapChanger, phaseTapChangerAction.isRelativeValue(), phaseTapChangerAction.getTapPosition());
                if (override == null) {
                    return null;
                }
                taps.add(override);
            } else if (a instanceof com.powsybl.action.ShuntCompensatorPositionAction shuntCompensatorPositionAction) {
                com.powsybl.iidm.network.ShuntCompensator shuntCompensator = network.getShuntCompensator(shuntCompensatorPositionAction.getShuntCompensatorId());
                LfShunt shunt = net.getShuntById(shuntCompensatorPositionAction.getShuntCompensatorId());
                if (shuntCompensator == null || shunt == null
                        || !(shuntCompensator.getModel() instanceof com.powsybl.iidm.network.ShuntCompensatorLinearModel linearModel)) {
                    return null;                                 // unpacked / non-linear model — CPU
                }
                double dSections = shuntCompensatorPositionAction.getSectionCount() - shuntCompensator.getSectionCount();
                double zb = shuntCompensator.getTerminal().getVoltageLevel().getNominalV()
                        * shuntCompensator.getTerminal().getVoltageLevel().getNominalV() / PerUnit.SB;
                double gPerSection = Double.isNaN(linearModel.getGPerSection()) ? 0 : linearModel.getGPerSection();   // optional in iidm
                shifts.add(new ShuntShift(shunt,
                        gPerSection * dSections * zb, linearModel.getBPerSection() * dSections * zb));
            } else if (a instanceof com.powsybl.action.GeneratorAction generatorAction) {
                // OLF's LfGeneratorAction reads the active power value and its relative flag ONLY (it refuses an action
                // without them): a target V, a target Q or a regulator flag on the action changes nothing in OLF's
                // equations, so it changes nothing here. The P delta is composed like a load delta with q=0.
                if (generatorAction.getActivePowerValue().isEmpty() || generatorAction.isActivePowerRelativeValue().isEmpty()) {
                    return null;                             // OLF does not build such an action
                }
                com.powsybl.iidm.network.Generator gen = network.getGenerator(generatorAction.getGeneratorId());
                if (gen == null) {
                    return null;
                }
                com.powsybl.openloadflow.network.LfGenerator lfGen = net.getGeneratorById(generatorAction.getGeneratorId());
                LfBus bus = lfGen != null ? lfGen.getBus() : null;    // OLF's own map, never the bus view
                if (bus == null) {
                    return null;
                }
                if (ctgGens.contains(lfGen)) {
                    continue;                                // OLF's LfGeneratorAction does nothing on a generator the contingency lost
                }
                boolean relative = generatorAction.isActivePowerRelativeValue().orElse(false);
                // An absolute value is measured from the generator's target in the BASE solution the device anchors
                // at, which the base load flow's distribution moved for a participant - not from the network's own
                // target (measured: B2-G set to 70 MW landed at 66.97, its base share of a 27 MW distribution short).
                double deltaP = relative ? generatorAction.getActivePowerValue().getAsDouble() / PerUnit.SB
                                         : generatorAction.getActivePowerValue().getAsDouble() / PerUnit.SB - lfGen.getTargetP();
                // the bus injection moves by the generator's NET change: an absolute action replaces what the
                // actions before it did (measured: 70 then 50 MW on B2-G injected both, 34 MW too much)
                generatorNetMove.merge(lfGen, deltaP, (before, d) -> relative ? before + d : d);
                // absolute: the value replaces the parent's distributed target
                generatorActions.add(new BatchContingency.GeneratorMove(lfGen, deltaP, !relative,
                        generatorAction.getActivePowerValue().getAsDouble() / PerUnit.SB,
                        com.powsybl.openloadflow.ac.solver.gpu.batch.GpuGeneratorParticipation.atTargetP(lfGen, gen,
                                networkParameters, balanceType)));
            } else if (a instanceof com.powsybl.action.HvdcAction hvdcAction) {
                // OLF's LfHvdcAction: only "AC emulation off" does anything (any other content is a logged no-op, and
                // "on" is refused when the action is built); the link then holds each station at the flow of the state
                // the strategy starts from, unclamped, for the whole strategy load flow.
                if (hvdcAction.isAcEmulationEnabled().isEmpty()) {
                    continue;
                }
                com.powsybl.openloadflow.network.LfHvdc hvdc = net.getHvdcById(hvdcAction.getHvdcId());
                if (hvdcAction.isAcEmulationEnabled().get() || hvdc == null || !hvdc.isAcEmulation()) {
                    return null;                             // refused by OLF, unknown or not emulated: CPU
                }
                if (hvdc.isDisabled() || ctgHvdcs.contains(hvdc)) {
                    continue;                                // a link without power: OLF's action changes nothing
                }
                if (!acEmulationOff.contains(hvdc)) {
                    acEmulationOff.add(hvdc);
                }
            } else {
                return null;                                 // out of stage-1 scope — CPU per-strategy path
            }
        }
        generatorNetMove.forEach((g, d) -> loads.add(new LoadLoss(g.getBus(), d, 0)));
        if (!reenabled.isEmpty()) {
            // OLF re-enables every branch at a bus its actions bring back into the main component that is not connected
            // at both ends (AbstractLfBranchAction.updateBusesAndBranchStatus), even one the contingency outaged
            List<LfBranch> ctgCut = new java.util.ArrayList<>(contingencyZeroImpedanceBranchOutages);
            ctgBranches.stream().filter(b -> b.getBus1() != null && b.getBus2() != null).forEach(ctgCut::add);
            ctgSideOpens.forEach(sideOpen -> ctgCut.add(sideOpen.branch()));
            for (LfBus bus : busesBroughtBack(net, ctgCut, reenabled, opened)) {
                for (LfBranch b : bus.getBranches()) {
                    if (b.getBus1() == null || b.getBus2() == null) {
                        branches.remove(b);
                    }
                }
            }
        }
        return new BatchContingency(pairId(contingencyId, strategy, actionIds),
                // the parent's one-side openings and the zero-impedance tree OLF reaches the strategy with: the pair is
                // the contingency's topology plus the actions (dropping them solved another network: 0.52 kV at B4 on a
                // three-winding transformer outage)
                branches, ctgGens, loads, shifts, taps, List.of(), reclosed, zeroImpedanceOutages, ctgHvdcs, ctgSideOpens,
                contingencyStickyNonSpanningTreeBranches, generatorActions, acEmulationOff);
    }

    /** Composes a SwitchAction onto the pair's outages; false when it is out of batch scope (the CPU per-strategy path). */
    private static boolean composeSwitchAction(LfNetwork net, com.powsybl.action.SwitchAction switchAction,
                                               List<BatchContingency.SideOpen> ctgSideOpens, List<LfBranch> branches,
                                               List<LfBranch> zeroImpedanceOutages, List<LfBranch> opened, List<LfBranch> reclosed) {
        LfBranch b = net.getBranchById(switchAction.getSwitchId());
        if (b != null && ctgSideOpens.stream().anyMatch(sideOpen -> sideOpen.branch() == b)) {
            return false;                                    // the contingency opened ONE side of it: CPU
        }
        if (b != null && switchAction.isOpen()) {
            // OPEN: OLF's LfSwitchAction takes the switch out of the connectivity and disables what that islands. A switch
            // open at base or opened by the contingency stays as it is; otherwise it joins the pair's zero-impedance
            // outages, as the switches a contingency's propagation opens do
            if (!b.isDisabled() && !zeroImpedanceOutages.contains(b) && !branches.contains(b)) {
                if (b.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC)) {
                    zeroImpedanceOutages.add(b);
                } else {
                    branches.add(b);
                }
                opened.add(b);
            }
            return true;
        }
        if (b != null && branches.remove(b)) {
            return true;
        }
        if (b == null || !b.isDisabled() || !b.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC)) {
            return false;
        }
        // CLOSE of a retained-open switch (zero-impedance branch). Only the spanning-tree MERGE shape is batched:
        // distinct zero-impedance groups (or none) on the two sides. A loop-closing switch is an electrical no-op on
        // the CPU too (non-tree edge: dummies stay fixed) — compose nothing.
        LfBus bus1 = b.getBus1();
        LfBus bus2 = b.getBus2();
        if (bus1 == null || bus2 == null || bus1.isDisabled() || bus2.isDisabled()) {
            return false;                                    // dead-island re-energization — CPU
        }
        var zn1 = bus1.getZeroImpedanceNetwork(com.powsybl.openloadflow.network.LoadFlowModel.AC);
        var zn2 = bus2.getZeroImpedanceNetwork(com.powsybl.openloadflow.network.LoadFlowModel.AC);
        if (zn1 == null || zn2 == null || zn1 != zn2) {
            reclosed.add(b);                                 // tree merge — batched
        }
        return true;                                         // same zero-impedance network: loop close, no-op
    }

    /**
     * The buses OLF adds back to the main component when a strategy re-enables {@code reenabled} and takes out
     * {@code opened}, as LfActionUtils.updateConnectivity finds them: the contingency's whole disabled network removed
     * from the connectivity (its cut branches and every branch of the buses it lost), then the actions applied.
     */
    private static java.util.Set<LfBus> busesBroughtBack(LfNetwork net, List<LfBranch> ctgCut, List<LfBranch> reenabled,
                                                         List<LfBranch> opened) {
        com.powsybl.openloadflow.graph.GraphConnectivity<LfBus, LfBranch> connectivity = net.getConnectivity();
        LfBus slack = net.getSynchronousNetworks().getFirst().getSlackBuses().getFirst();
        connectivity.startTemporaryChanges();
        try {
            ctgCut.forEach(connectivity::removeEdge);
            connectivity.setMainComponentVertex(slack);
            java.util.Set<LfBranch> cut = new java.util.HashSet<>(ctgCut);
            for (LfBus lost : connectivity.getVerticesRemovedFromMainComponent()) {
                for (LfBranch b : lost.getBranches()) {
                    if (b.getBus1() != null && b.getBus2() != null && !b.isDisabled() && cut.add(b)) {
                        connectivity.removeEdge(b);
                    }
                }
            }
            connectivity.startTemporaryChanges();
            try {
                opened.stream().filter(b -> b.getBus1() != null && b.getBus2() != null && !cut.contains(b))
                        .forEach(connectivity::removeEdge);
                reenabled.forEach(b -> connectivity.addEdge(b.getBus1(), b.getBus2(), b));
                return new java.util.HashSet<>(connectivity.getVerticesAddedToMainComponent());
            } finally {
                connectivity.undoTemporaryChanges();
            }
        } finally {
            connectivity.undoTemporaryChanges();
            connectivity.setMainComponentVertex(slack);
        }
    }

    private static TapR1Override tapOverride(LfNetwork net, String transformerId,
                                             com.powsybl.iidm.network.TapChanger<?, ?, ?, ?> tapChanger,
                                             boolean relative, int tapPosition) {
        LfBranch b = net.getBranchById(transformerId);
        if (b == null || tapChanger == null) {
            return null;
        }
        double[][] table = GpuTransformerVoltageControlOuterLoop.tapModels(b.getPiModel());
        int low = tapChanger.getLowTapPosition();
        int idx = (relative ? tapChanger.getTapPosition() + tapPosition : tapPosition) - low;
        if (table.length == 0 || idx < 0 || idx >= table.length) {
            return null;                                     // no tap table / out-of-range position — CPU
        }
        return new TapR1Override(b, table[idx]);
    }
}
