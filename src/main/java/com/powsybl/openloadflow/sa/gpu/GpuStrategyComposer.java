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
import com.powsybl.openloadflow.ac.solver.gpu.batch.BatchContingency;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchStatus;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchTransformerControl;
import com.powsybl.openloadflow.ac.solver.gpu.batch.LoadLoss;
import com.powsybl.openloadflow.ac.solver.gpu.batch.ShuntShift;
import com.powsybl.openloadflow.ac.solver.gpu.batch.TapR1Override;
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
        List<com.powsybl.contingency.strategy.ConditionalActions> stages = strategy.getConditionalActions();
        if (stages.size() != 1) {
            return false;                                    // multi-stage stays on the CPU path (stage-on-stage
        }                                                    // semantics need intermediate-state evaluation)
        // ANY single-stage condition qualifies — including the violation/threshold conditions: the
        // ENGINE evaluates the condition (AbstractSecurityAnalysis.checkCondition, on the CPU-detected
        // post-contingency violations) BEFORE our runActionSimulation override is invoked, so the pair
        // state is precomputed SPECULATIVELY and simply goes unused when the condition does not pass.
        // Condition semantics therefore stay 100% stock — the batch only accelerates the solve.
        var ctx = strategy.getContingencyContext();
        return switch (ctx.getContextType()) {
            case ALL, ONLY_CONTINGENCIES -> true;
            case SPECIFIC -> contingencyId.equals(ctx.getContingencyId());
            case NONE -> false;
        };
    }

    /** Compile the strategy's actions onto the contingency's deltas into ONE extra batch scenario,
     *  or null when any action is out of stage-1 scope (branch open/RECLOSE, LoadAction,
     *  GeneratorAction — absolute or relative active power). {@code ctgLostLoadIds} are the loads the
     *  contingency itself loses, on which a load action is a no-op. */
    static BatchContingency composeStrategyScenario(Network network, LfNetwork net, String contingencyId,
                                                    OperatorStrategy strategy, List<Action> currentActions,
                                                    List<LfBranch> ctgZiOutages,
                                                    List<LfBranch> ctgBranches,
                                                    List<LfGenerator> ctgGens,
                                                    List<LoadLoss> ctgLoads,
                                                    java.util.Set<String> ctgLostLoadIds,
                                                    List<ShuntShift> ctgShifts,
                                                    List<com.powsybl.openloadflow.network.LfHvdc> ctgHvdcs,
                                                    com.powsybl.openloadflow.network.LfNetworkParameters networkParameters,
                                                    com.powsybl.loadflow.LoadFlowParameters.BalanceType balanceType) {
        List<LfBranch> branches = new java.util.ArrayList<>(ctgBranches);
        List<LoadLoss> loads = new java.util.ArrayList<>(ctgLoads);
        List<ShuntShift> shifts = new java.util.ArrayList<>(ctgShifts);
        List<TapR1Override> taps = new java.util.ArrayList<>();
        List<LfBranch> reclosed = new java.util.ArrayList<>();
        List<BatchContingency.GeneratorMove> generatorActions = new java.util.ArrayList<>();
        Map<LfGenerator, Double> generatorNetMove = new java.util.LinkedHashMap<>();
        Map<String, Action> actionById = new java.util.HashMap<>();
        for (Action a : currentActions) {
            actionById.put(a.getId(), a);
        }
        for (String actionId : strategy.getConditionalActions().get(0).getActionIds()) {
            Action a = actionById.get(actionId);
            if (a instanceof com.powsybl.action.TerminalsConnectionAction tca) {
                LfBranch b = net.getBranchById(tca.getElementId());
                if (b == null) {
                    // Not a branch: a SHUNT compensator's terminals (OLF supports (re)connection since
                    // the olf-local-patches shuntIdsToClose retention) — composed as a per-scenario
                    // admittance shift: the retained base-open compensator contributes 0 at base, so
                    // close = +its full section admittance; open of a connected one = -it. LINEAR only.
                    com.powsybl.iidm.network.ShuntCompensator sc = network.getShuntCompensator(tca.getElementId());
                    LfShunt lfShunt = sc != null ? net.getShuntById(tca.getElementId()) : null;
                    if (sc == null || lfShunt == null
                            || !(sc.getModel() instanceof com.powsybl.iidm.network.ShuntCompensatorLinearModel lin)) {
                        return null;
                    }
                    // NEVER from the iidm terminal: during the SA the working variant is the TEMPORARY
                    // one where the retention machinery already connected the closable shunt — ask the
                    // LF model, which tracks the compensator's true base state.
                    boolean connectedAtBase = !lfShunt.isCompensatorInitiallyDisconnected(tca.getElementId());
                    if (tca.isOpen() == !connectedAtBase) {
                        continue;                            // open-of-disconnected / close-of-connected: no-op
                    }
                    double gPerSection = Double.isNaN(lin.getGPerSection()) ? 0 : lin.getGPerSection();
                    double zb = sc.getTerminal().getVoltageLevel().getNominalV()
                            * sc.getTerminal().getVoltageLevel().getNominalV() / PerUnit.SB;
                    double sign = tca.isOpen() ? -1 : 1;
                    double n = sc.getSectionCount();
                    shifts.add(new ShuntShift(lfShunt,
                            sign * gPerSection * n * zb, sign * lin.getBPerSection() * n * zb));
                    continue;
                }
                if (tca.isOpen()) {
                    if (!branches.contains(b)) {
                        branches.add(b);
                    }
                } else {
                    if (!branches.remove(b)) {
                        // close of a BASE-OPEN branch (retained disabled by LfTopoConfig): batched as a
                        // per-scenario mask CLEAR on the base-open packed element — dead-island
                        // reconnection (a disabled endpoint bus) is rejected at qualification.
                        if (!b.isDisabled()) {
                            return null;                     // closing an already-closed branch — CPU decides
                        }
                        reclosed.add(b);
                    }
                }
            } else if (a instanceof com.powsybl.action.SwitchAction sw) {
                LfBranch b = net.getBranchById(sw.getSwitchId());
                if (b == null || (sw.isOpen() ? branches.contains(b) : !branches.remove(b))) {
                    if (b != null && sw.isOpen()) {
                        branches.add(b);
                    } else if (b != null && b.isDisabled()
                            && b.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC)) {
                        // CLOSE of a retained-open switch (zero-impedance branch). Only the
                        // spanning-tree MERGE shape is batched: distinct zero-impedance groups (or
                        // none) on the two sides. A loop-closing switch is an electrical no-op on
                        // the CPU too (non-tree edge: dummies stay pinned) — compose nothing.
                        LfBus zb1 = b.getBus1();
                        LfBus zb2 = b.getBus2();
                        if (zb1 == null || zb2 == null || zb1.isDisabled() || zb2.isDisabled()) {
                            return null;                     // dead-island re-energization — CPU
                        }
                        var zn1 = zb1.getZeroImpedanceNetwork(com.powsybl.openloadflow.network.LoadFlowModel.AC);
                        var zn2 = zb2.getZeroImpedanceNetwork(com.powsybl.openloadflow.network.LoadFlowModel.AC);
                        if (zn1 == null || zn2 == null || zn1 != zn2) {
                            reclosed.add(b);                 // tree merge — batched
                        }
                        // same zero-impedance network: loop close, no-op — skip
                    } else {
                        return null;
                    }
                }
            } else if (a instanceof com.powsybl.action.LoadAction la) {
                com.powsybl.iidm.network.Load load = network.getLoad(la.getLoadId());
                if (load == null) {
                    return null;
                }
                // OLF's own load -> LfBus map, never the bus view (see GpuContingencyMapper.loadOutages)
                com.powsybl.openloadflow.network.LfLoad lfLoad = net.getLoadById(la.getLoadId());
                LfBus bus = lfLoad != null ? lfLoad.getBus() : null;
                if (bus == null) {
                    return null;
                }
                if (ctgLostLoadIds.contains(la.getLoadId())) {
                    continue;                                // OLF's LfLoadAction does nothing on a load the contingency lost
                }
                // LfLoadAction's power shift: the value itself when relative, value - P0 (Q0) when absolute, 0 for
                // the part the action leaves out. LoadLoss carries the load REMOVED, hence the sign (pu).
                double dP = la.getActivePowerValue().isPresent()
                        ? -(la.isRelativeValue() ? la.getActivePowerValue().getAsDouble() : la.getActivePowerValue().getAsDouble() - load.getP0())
                        : 0.0;
                double dQ = la.getReactivePowerValue().isPresent()
                        ? -(la.isRelativeValue() ? la.getReactivePowerValue().getAsDouble() : la.getReactivePowerValue().getAsDouble() - load.getQ0())
                        : 0.0;
                loads.add(new LoadLoss(bus, dP / PerUnit.SB, dQ / PerUnit.SB));
            } else if (a instanceof com.powsybl.action.RatioTapChangerTapPositionAction rtca) {
                var tapChanger = network.getTwoWindingsTransformer(rtca.getTransformerId()) != null
                        ? network.getTwoWindingsTransformer(rtca.getTransformerId()).getRatioTapChanger() : null;
                TapR1Override ov = tapOverride(net, rtca.getTransformerId(),
                        tapChanger, rtca.isRelativeValue(), rtca.getTapPosition());
                if (ov == null) {
                    return null;
                }
                taps.add(ov);
            } else if (a instanceof com.powsybl.action.PhaseTapChangerTapPositionAction ptca) {
                var tapChanger = network.getTwoWindingsTransformer(ptca.getTransformerId()) != null
                        ? network.getTwoWindingsTransformer(ptca.getTransformerId()).getPhaseTapChanger() : null;
                TapR1Override ov = tapOverride(net, ptca.getTransformerId(),
                        tapChanger, ptca.isRelativeValue(), ptca.getTapPosition());
                if (ov == null) {
                    return null;
                }
                taps.add(ov);
            } else if (a instanceof com.powsybl.action.ShuntCompensatorPositionAction spa) {
                com.powsybl.iidm.network.ShuntCompensator sc = network.getShuntCompensator(spa.getShuntCompensatorId());
                LfShunt shunt = net.getShuntById(spa.getShuntCompensatorId());
                if (sc == null || shunt == null
                        || !(sc.getModel() instanceof com.powsybl.iidm.network.ShuntCompensatorLinearModel lin)) {
                    return null;                                 // unpacked / non-linear model — CPU
                }
                double dSections = spa.getSectionCount() - sc.getSectionCount();
                double zb = sc.getTerminal().getVoltageLevel().getNominalV()
                        * sc.getTerminal().getVoltageLevel().getNominalV() / PerUnit.SB;
                double gPerSection = Double.isNaN(lin.getGPerSection()) ? 0 : lin.getGPerSection();   // optional in iidm
                shifts.add(new ShuntShift(shunt,
                        gPerSection * dSections * zb, lin.getBPerSection() * dSections * zb));
            } else if (a instanceof com.powsybl.action.GeneratorAction ga) {
                // Pure ACTIVE-POWER redispatch only: a targetV/targetQ/regulator change alters the
                // voltage-control problem itself — CPU path. The P delta is composed exactly like a
                // load delta with q=0 (net injection at the bus; the device DS reweight machinery is
                // NOT touched: participation factors are unchanged by a redispatch, and the headroom
                // shift only matters if this generator saturates during redistribution — CPU handles
                // any such contingency via the DS-residual rejection).
                if (ga.getTargetV().isPresent() || ga.getTargetQ().isPresent()
                        || ga.isVoltageRegulatorOn().isPresent() || ga.getActivePowerValue().isEmpty()) {
                    return null;
                }
                com.powsybl.iidm.network.Generator gen = network.getGenerator(ga.getGeneratorId());
                if (gen == null) {
                    return null;
                }
                com.powsybl.openloadflow.network.LfGenerator lfGen = net.getGeneratorById(ga.getGeneratorId());
                LfBus bus = lfGen != null ? lfGen.getBus() : null;    // OLF's own map, never the bus view
                if (bus == null) {
                    return null;
                }
                if (ctgGens.contains(lfGen)) {
                    continue;                                // OLF's LfGeneratorAction does nothing on a generator the contingency lost
                }
                boolean relative = ga.isActivePowerRelativeValue().orElse(false);
                // An absolute value is measured from the generator's target in the BASE solution the device anchors
                // at, which the base load flow's distribution moved for a participant - not from the network's own
                // target (measured: B2-G set to 70 MW landed at 66.97, its base share of a 27 MW distribution short).
                double deltaP = relative ? ga.getActivePowerValue().getAsDouble() / PerUnit.SB
                                         : ga.getActivePowerValue().getAsDouble() / PerUnit.SB - lfGen.getTargetP();
                // the bus injection moves by the generator's NET change: an absolute action replaces what the
                // actions before it did (measured: 70 then 50 MW on B2-G injected both, 34 MW too much)
                generatorNetMove.merge(lfGen, deltaP, (before, d) -> relative ? before + d : d);
                // absolute: the value replaces the parent's distributed target
                generatorActions.add(new BatchContingency.GeneratorMove(lfGen, deltaP, !relative,
                        ga.getActivePowerValue().getAsDouble() / PerUnit.SB,
                        com.powsybl.openloadflow.ac.solver.gpu.batch.GpuGeneratorParticipation.atTargetP(lfGen, gen,
                                networkParameters, balanceType)));
            } else {
                return null;                                 // out of stage-1 scope — CPU per-strategy path
            }
        }
        generatorNetMove.forEach((g, d) -> loads.add(new LoadLoss(g.getBus(), d, 0)));
        return new BatchContingency(contingencyId + GpuBatchStatus.STRATEGY_KEY_SEP + strategy.getId(),
                branches, ctgGens, loads, shifts, taps, List.of(), reclosed, ctgZiOutages, ctgHvdcs, List.of(), List.of(),
                generatorActions);
    }

    /** Compiles a tap-position action (RTC or PST) into a per-scenario full-row override — the
     *  target tap's [y, g1, g2, b1, b2, ksi, r1, a1] replaces the branch's static inputs, so
     *  per-tap impedance changes are covered. Null when out of batch scope: unknown transformer /
     *  tap changer, no tap table, or out-of-range position. */
    private static TapR1Override tapOverride(LfNetwork net, String transformerId,
                                             com.powsybl.iidm.network.TapChanger<?, ?, ?, ?> tapChanger,
                                             boolean relative, int tapPosition) {
        LfBranch b = net.getBranchById(transformerId);
        if (b == null || tapChanger == null) {
            return null;
        }
        double[][] table = GpuBatchTransformerControl.tapModels(b.getPiModel());
        int low = tapChanger.getLowTapPosition();
        int idx = (relative ? tapChanger.getTapPosition() + tapPosition : tapPosition) - low;
        if (table.length == 0 || idx < 0 || idx >= table.length) {
            return null;                                     // no tap table / out-of-range position — CPU
        }
        return new TapR1Override(b, table[idx]);
    }
}
