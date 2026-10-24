/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDistributedSlackParameters;
import com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix.CouplerTreeReHoming;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfLoad;
import com.powsybl.openloadflow.network.LfNetwork;

import java.util.Map;

/**
 * DEVICE port of OLF's {@code DistributedSlackOuterLoop}, Java side: the loop runs in the kernel
 * (distributed_slack.cu); this class applies its decisions to the network, as OLF's ActivePowerDistribution leaves
 * them: generator targets from their anchor plus their share, or the loads moved for a load-based balance type.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuDistributedSlackOuterLoop {

    private GpuDistributedSlackOuterLoop() {
    }

    /** DistributedSlack, exactly OLF's ActivePowerDistribution: every participating generator ends at
     *  its ANCHOR target ({@code initialTargetP}) plus the cumulative share the water-fill gave it
     *  ({@code distributedSlackGeneratorMove}, per generator — each saturates at its own limits); the slack bus's own
     *  generators, which have no device row, split the slack's retained share by factor — the value
     *  captured at the last check that actually DISTRIBUTED, not the sub-tolerance one a STABLE check
     *  measures. With slack-only participation the slack takes the whole absorbed total ({@code slackAbsorbed}, null
     *  when the context is not slack-only). A solve whose DistributedSlack phase never fired leaves the generators alone.
     *  Returns what it distributed (pu, see GpuNewtonRaphson#lastDistributedActivePower), NaN when nothing. */
    public static double applyToNetwork(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> equationSystem,
                            GpuDistributedSlackParameters distributedSlackParameters,
                            Map<LfLoad, Double> loadAnchorTargetP,
                            double[] deviceTargets, double[] distributedSlackGeneratorMove, double[] retain, double[] slackAbsorbed,
                            GpuAcData data) {
        double lastDistributedActivePower = Double.NaN;
        if (deviceTargets == null || data == null) {
            return lastDistributedActivePower;
        }
        LfBus slackBus = slackBusOf(network, equationSystem, data);
        if (data.distributedSlack().busPhiRow().length == 0) {
            // slack-only participation: the whole distribution is the slack's absorbed total
            if (slackBus != null && slackAbsorbed != null && slackAbsorbed.length > 0 && slackAbsorbed[0] != 0) {
                setAnchoredShare(slackBus, slackAbsorbed[0], distributedSlackParameters);
                lastDistributedActivePower = slackAbsorbed[0];
            }
            return lastDistributedActivePower;
        }
        return applyParticipantsToNetwork(network, distributedSlackParameters, loadAnchorTargetP, deviceTargets, distributedSlackGeneratorMove,
                retain, data, slackBus);
    }

    /** The bus the distribution pack's slack row belongs to, null without one. */
    public static LfBus slackBusOf(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> equationSystem, GpuAcData data) {
        LfBus slackBus = null;
        if (data.distributedSlack().slackPhiRow() >= 0) {
            for (LfBus bus : network.getBuses()) {
                var v = equationSystem.getVariableSet().getVariable(bus.getNum(), AcVariableType.BUS_PHI);
                if (v != null && v.getRow() == data.distributedSlack().slackPhiRow()) {
                    slackBus = bus;
                    break;
                }
            }
            if (slackBus == null) {
                // The COUPLER-TREE RELOCATION moved that bus's P balance onto its parent coupler's DUMMY_P row,
                // which no BUS_PHI variable owns — the pack ships the relocated row (slackBusPhiRow = pRow). Left
                // unresolved, the slack's retained share was never written back and OLF measured the device's
                // converged state against stale targets: on rte6515's full busbar scope the device's own norm was
                // 9.45e-06 (OLF's own final 9.50e-06) while OLF re-evaluated 1.67 and called the base
                // MAX_ITERATION_REACHED, which took the whole security analysis with it.
                slackBus = CouplerTreeReHoming.relocatedChildOfPRow(network, data.zeroImpedanceBranch().indices(), data.zeroImpedanceBranch().branchNum(),
                        data.distributedSlack().slackPhiRow());
            }
        }
        return slackBus;
    }

    /** Each participating generator's move from its initial target (the distribution's anchor) when a solve starts: zero in a
     *  run's first solve, the earlier solves' distribution in a later one (a loop on the host made the run solve again).
     *  OLF's distribution resets them there and adds the move back to the mismatch, so a solve starting without them reads
     *  the earlier solves' distribution as a new mismatch and distributes again. */
    public static double[] initialMoves(LfNetwork network, GpuAcData data) {
        Map<String, LfGenerator> genById = new java.util.HashMap<>();
        for (LfBus bus : network.getBuses()) {
            for (LfGenerator g : bus.getGenerators()) {
                genById.put(g.getId(), g);
            }
        }
        String[] generatorId = data.distributedSlack().generatorId();
        double[] moves = new double[generatorId.length];
        for (int g = 0; g < moves.length; g++) {
            LfGenerator generator = genById.get(generatorId[g]);
            moves[g] = generator != null ? generator.getTargetP() - generator.getInitialTargetP() : 0.0;
        }
        return moves;
    }

    /** The slack-only distribution's counterpart of {@link #initialMoves}: the slack bus's participating generators' total
     *  move from their initial targets, which seeds what the slack has absorbed. */
    public static double slackInitialMove(LfBus slackBus, GpuDistributedSlackParameters distributedSlackParameters) {
        double move = 0.0;
        for (LfGenerator g : slackBus.getGenerators()) {
            if (distributedSlackParameters.factor(g) != 0) {
                move += g.getTargetP() - g.getInitialTargetP();
            }
        }
        return move;
    }

    private static double applyParticipantsToNetwork(LfNetwork network, GpuDistributedSlackParameters distributedSlackParameters,
                                                     Map<LfLoad, Double> loadAnchorTargetP, double[] deviceTargets,
                                                     double[] distributedSlackGeneratorMove, double[] retain, GpuAcData data, LfBus slackBus) {
        double lastDistributedActivePower = Double.NaN;
        boolean fired = retain != null && retain.length > 0 && !Double.isNaN(retain[0]);
        if (!fired) {
            return lastDistributedActivePower;
        }
        Map<String, LfGenerator> genById = new java.util.HashMap<>();
        for (LfBus bus : network.getBuses()) {
            for (LfGenerator g : bus.getGenerators()) {
                genById.put(g.getId(), g);
            }
        }
        double distributed = retain[0];                     // the slack's retained share + the participants' below
        boolean slackInPack = false;                        // the slack's own generators listed (DS row -1)
        if (distributedSlackGeneratorMove != null && data.distributedSlack().generatorId() != null && distributedSlackGeneratorMove.length >= data.distributedSlack().generatorId().length) {
            for (int g = 0; g < data.distributedSlack().generatorId().length; g++) {
                if (data.distributedSlack().generatorBus()[g] < 0) {
                    slackInPack = true;                     // its move is the retained share already counted
                } else {
                    distributed += distributedSlackGeneratorMove[g];
                }
                LfGenerator gen = genById.get(data.distributedSlack().generatorId()[g]);
                if (gen != null) {
                    gen.setTargetP(distributedSlackParameters.anchorTargetP(gen) + distributedSlackGeneratorMove[g]);
                }
            }
        } else {
            for (int i = 0; i < data.distributedSlack().busPhiRow().length; i++) {        // bus-level fallback: split by factor
                int row = data.distributedSlack().busPhiRow()[i];
                if (row < 0 || row >= deviceTargets.length) {
                    continue;
                }
                LfBus bus = data.distributedSlack().busNum() != null && i < data.distributedSlack().busNum().length ? network.getBus(data.distributedSlack().busNum()[i]) : null;
                distributed += deviceTargets[row] - data.distributedSlack().busBaseTargetP()[i];
                if (bus != null) {
                    if (distributedSlackParameters.isLoadBased()) {
                        moveLoads(bus, deviceTargets[row] - data.distributedSlack().busBaseTargetP()[i], loadAnchorTargetP);
                    } else {
                        setAnchoredShare(bus, deviceTargets[row] - data.distributedSlack().busBaseTargetP()[i], distributedSlackParameters);
                    }
                }
            }
        }
        lastDistributedActivePower = distributed;
        if (slackBus != null && retain[0] != 0 && !slackInPack) {
            if (distributedSlackParameters.isLoadBased()) {
                moveLoads(slackBus, retain[0], loadAnchorTargetP);
            } else {
                setAnchoredShare(slackBus, retain[0], distributedSlackParameters);
            }
        }

        return lastDistributedActivePower;
    }

    /** Each load's target P now: the anchor a load-based distribution moves the loads from. */
    public static Map<LfLoad, Double> captureLoadAnchors(LfNetwork network) {
        Map<LfLoad, Double> anchors = new java.util.IdentityHashMap<>();
        for (LfBus bus : network.getBuses()) {
            for (LfLoad load : bus.getLoads()) {
                anchors.put(load, load.getTargetP());
            }
        }
        return anchors;
    }

    /** A load-based distribution's bus-level amount on the bus's loads, as OLF's LoadActivePowerDistributionStep
     *  moves them: from where they stood when the distribution started (the step is incremental, it has no initial
     *  target to go back to), each in proportion to its variable active power. A bus injection rising by
     *  {@code delta} is its loads consuming {@code delta} less. Moving only generators, as for the generation-based
     *  types, left OLF's load targets where they were, and OLF measured the converged state against them: an ieee14
     *  base whose slack bus carries a participating load came back MAX_ITERATION_REACHED. */
    private static void moveLoads(LfBus bus, double delta,
                                  Map<LfLoad, Double> loadAnchorTargetP) {
        double factorSum = 0;
        for (LfLoad load : bus.getLoads()) {
            factorSum += load.getAbsVariableTargetP();
        }
        if (factorSum <= 0) {
            return;
        }
        for (LfLoad load : bus.getLoads()) {
            double f = load.getAbsVariableTargetP();
            if (f != 0) {
                load.setTargetP(loadAnchorTargetP.getOrDefault(load, load.getTargetP()) - delta * f / factorSum);
            }
        }
    }

    /** Give a bus's participating generators {@code anchor + delta × f / Σf} — OLF's split of a
     *  bus-level amount from the anchor, with the run's balance-type factors. */
    private static void setAnchoredShare(LfBus bus, double delta, GpuDistributedSlackParameters distributedSlackParameters) {
        double factorSum = 0;
        for (LfGenerator g : bus.getGenerators()) {
            factorSum += distributedSlackParameters.factor(g);
        }
        if (factorSum <= 0) {
            return;
        }
        for (LfGenerator g : bus.getGenerators()) {
            double f = distributedSlackParameters.factor(g);
            if (f != 0) {
                g.setTargetP(distributedSlackParameters.anchorTargetP(g) + delta * f / factorSum);
            }
        }
    }
}
