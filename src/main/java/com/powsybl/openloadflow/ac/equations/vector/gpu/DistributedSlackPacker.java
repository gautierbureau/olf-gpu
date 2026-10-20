/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix.StableRowIndexing;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuDistributedSlackFactorMode;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfNetwork;

import java.util.ArrayList;
import java.util.List;

/**
 * The distributed-slack pack (DistributedSlackOuterLoop): the participating buses and generators with their
 * factors, headroom and anchors, and the slack bus's own target.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class DistributedSlackPacker {

    private DistributedSlackPacker() {
    }

    /** Σ over the bus's participating generators of (current − anchor target P): what OLF's
     *  resetToInitialState undoes and adds back as previousMismatch. 0 when anchoring at the current
     *  targets. The anchor row target = the row's current target minus this. */
    static double anchorOffset(LfBus bus, GpuDistributedSlackParameters distributedSlackParameters) {
        double off = 0;
        for (LfGenerator g : bus.getGenerators()) {
            if (distributedSlackParameters.factor(g) != 0) {
                off += g.getTargetP() - distributedSlackParameters.anchorTargetP(g);
            }
        }
        return off;
    }

    static DistributedSlackParticipants distributedSlackPack(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                       ExtractionColumns columns, boolean distributedSlack, GpuDistributedSlackParameters distributedSlackParameters) {
        int[] phiCol = columns.phiCol();
        int[] pRow = columns.pRow();       // a DS target is written at the bus's P ROW, relocated under a coupler tree
        // Distributed-slack pack: the single slack/reference bus (BUS_TARGET_PHI active, BUS_TARGET_P
        // inactive — its P is free) provides the mismatch; every OTHER bus with participating generators
        // takes a share of it into its BUS_TARGET_P target. Default balance type PROPORTIONAL_TO_GENERATION_
        // P_MAX: per-bus factor = Σ (maxP / droop) over its participating generators, normalized to sum 1.
        //
        // PARTICIPATING SLACK: if the slack/reference bus ITSELF has participating generators, OLF lists them with
        // the other participants (it resets every one of them and water-fills the full mismatch over all). The
        // slack's factor is in the NORMALIZATION DENOMINATOR, and its generators join the per-generator pack with
        // no row (distributedSlackGeneratorBus -1): they take their share, clamp at their own limits and leave the list like any other, so
        // a saturated peer's remainder is renormalized over them too. The slack's placement is the sum of their
        // moves (slackRetainedShare / retainApplied).
        //
        // SATURATION: per-generator rooms (maxTargetP−targetP / minTargetP−targetP with OLF's no-sign-change rule)
        // bound each generator's move; the water-fill removes a generator when it strictly overshoots and
        // renormalizes over the rest, as GenerationActivePowerDistributionStep does. The per-bus headroom/footroom
        // below remain for the bus-level aggregate.
        int slackBusPhiRow = -1;
        double slackBusTargetP = 0;
        double slackBusRawFactor = 0;                                       // slack's own participation (denominator only)
        double slackBusMarginUp = 0;                                     // REMAINING_MARGIN: the slack's own margins
        double slackBusMarginDown = 0;
        double slackBusRoomUp = Double.POSITIVE_INFINITY;                // the slack's participating generators' total
        double slackBusRoomDown = Double.NEGATIVE_INFINITY;              // rooms: what a slack-only distribution can place
        List<Integer> distributedSlackBusPhiRow = new ArrayList<>();
        List<Integer> distributedSlackGeneratorBus = new ArrayList<>();
        List<Double> distributedSlackGeneratorRawFactor = new ArrayList<>();
        List<Double> distributedSlackGeneratorMaxDelta = new ArrayList<>();
        List<Double> distributedSlackGeneratorMinDelta = new ArrayList<>();
        List<Double> distributedSlackGeneratorMarginUp = new ArrayList<>();
        List<Double> distributedSlackGeneratorMarginDown = new ArrayList<>();
        List<Integer> distributedSlackBusGeneratorStart = new ArrayList<>();
        List<Integer> distributedSlackBusGeneratorEnd = new ArrayList<>();
        List<Integer> distributedSlackBusNum = new ArrayList<>();                     // WHICH bus each row is (applyToNetwork)
        List<String> distributedSlackGeneratorId = new ArrayList<>();                       // WHICH generator each slot is
        List<Double> distributedSlackBusRawFactor = new ArrayList<>();
        List<Double> distributedSlackBusBaseTargetP = new ArrayList<>();                 // each bus's initial BUS_TARGET_P
        List<Double> distributedSlackBusMaxDelta = new ArrayList<>();                    // headroom ≥ 0
        List<Double> distributedSlackBusMinDelta = new ArrayList<>();                    // footroom ≤ 0
        List<Double> distributedSlackBusMarginUp = new ArrayList<>();                    // REMAINING_MARGIN per-bus margins
        List<Double> distributedSlackBusMarginDown = new ArrayList<>();
        for (LfBus bus : network.getBuses()) {
            if (!distributedSlack) {                                    // empty pack when the DS outer loop is disabled
                break;
            }
            int bn = bus.getNum();
            if (StableRowIndexing.row(es, bn, AcEquationType.BUS_TARGET_PHI) >= 0) {       // the slack/reference bus
                slackBusPhiRow = pRow[bn];
                slackBusTargetP = bus.getTargetP() - anchorOffset(bus, distributedSlackParameters);   // the slack's ANCHOR target
                slackBusRawFactor = distributedSlackParameters.busFactor(bus);
                if (distributedSlackParameters.isRemainingMargin()) {
                    double[] m = distributedSlackParameters.busMargins(bus);
                    slackBusMarginUp = m[0];
                    slackBusMarginDown = m[1];
                }
                // OLF's water-fill lists the slack bus's participating generators with every other participant,
                // in bus order: they take a share, are clamped at their own limits and leave the list like any
                // other, so a peer's saturated remainder is renormalized over them too. They join the generator
                // pack with no row (distributedSlackGeneratorBus -1); the slack's placement is the sum of their moves. Loads have no
                // limits, so a load-based slack keeps its fixed share.
                if (!distributedSlackParameters.isLoadBased()) {
                    slackBusRoomUp = 0;
                    slackBusRoomDown = 0;
                    for (LfGenerator g : bus.getGenerators()) {
                        if (distributedSlackParameters.factor(g) != 0) {
                            double[] r = distributedSlackParameters.rooms(g, distributedSlackParameters.anchorTargetP(g));
                            slackBusRoomUp += r[0];
                            slackBusRoomDown += r[1];
                            addPackGenerator(g, -1, distributedSlackParameters, distributedSlackGeneratorBus,
                                    distributedSlackGeneratorId, distributedSlackGeneratorRawFactor,
                                    distributedSlackGeneratorMaxDelta, distributedSlackGeneratorMinDelta,
                                    distributedSlackGeneratorMarginUp, distributedSlackGeneratorMarginDown);
                        }
                    }
                }
                continue;
            }
            if (StableRowIndexing.row(es, bn, AcEquationType.BUS_TARGET_P) < 0) {
                continue;                                               // no active P balance to retarget
            }
            double factor = distributedSlackParameters.busFactor(bus);
            double headroom = 0;
            double footroom = 0;
            for (LfGenerator g : bus.getGenerators()) {
                if (distributedSlackParameters.factor(g) != 0) {
                    double[] r = distributedSlackParameters.rooms(g, distributedSlackParameters.anchorTargetP(g));   // OLF's bounds + sign rule
                    headroom += r[0];
                    footroom += r[1];
                }
            }
            if (distributedSlackParameters.isLoadBased()) {
                headroom = Double.POSITIVE_INFINITY;                    // loads: OLF rescales them unbounded
                footroom = Double.NEGATIVE_INFINITY;
            }
            if (factor > 0) {
                distributedSlackBusPhiRow.add(pRow[bn]);
                distributedSlackBusNum.add(bn);
                distributedSlackBusRawFactor.add(factor);
                distributedSlackBusBaseTargetP.add(bus.getTargetP() - anchorOffset(bus, distributedSlackParameters));   // the row's ANCHOR target
                distributedSlackBusMaxDelta.add(headroom);
                distributedSlackBusMinDelta.add(footroom);
                if (distributedSlackParameters.isRemainingMargin()) {
                    double[] m = distributedSlackParameters.busMargins(bus);
                    distributedSlackBusMarginUp.add(m[0]);
                    distributedSlackBusMarginDown.add(m[1]);
                }
                // Per GENERATOR as well, contiguous within the bus. OLF's water-fill participates and
                // SATURATES per generator (ParticipatingElement wraps an LfGenerator, and step.run
                // clamps each one at its OWN minTargetP/maxTargetP and removes it), so a bus-level
                // aggregate only reproduces it while nothing saturates. Both groupings are kept: the
                // bus one owns the equation row and the anchor, the generator one owns the factors
                // and the limits.
                distributedSlackBusGeneratorStart.add(distributedSlackGeneratorRawFactor.size());
                for (LfGenerator g : bus.getGenerators()) {
                    if (distributedSlackParameters.factor(g) == 0) {
                        continue;                            // OLF: isParticipating(gen) && factor != 0
                    }
                    addPackGenerator(g, distributedSlackBusPhiRow.size() - 1, distributedSlackParameters,
                            distributedSlackGeneratorBus, distributedSlackGeneratorId,
                            distributedSlackGeneratorRawFactor, distributedSlackGeneratorMaxDelta,
                            distributedSlackGeneratorMinDelta,
                            distributedSlackGeneratorMarginUp, distributedSlackGeneratorMarginDown);
                }
                distributedSlackBusGeneratorEnd.add(distributedSlackGeneratorRawFactor.size());
            }
        }
        double distributedSlackRawFactorSum = distributedSlackBusRawFactor.stream().mapToDouble(Double::doubleValue).sum() + slackBusRawFactor;
        double[] distributedSlackBusFactor = new double[distributedSlackBusPhiRow.size()];
        for (int i = 0; i < distributedSlackBusFactor.length; i++) {
            distributedSlackBusFactor[i] = distributedSlackBusRawFactor.get(i) / distributedSlackRawFactorSum;
        }
        // A pack with participants but no slack row is unusable; one with NEITHER participants nor a
        // participating slack means DS never fires. But a PARTICIPATING SLACK ALONE is kept (nDistributedSlackBuses == 0,
        // slackBusPhiRow >= 0): OLF then distributes the whole mismatch into the slack generator's own
        // target — no flow change, but the outer loop reports UNSTABLE and re-solves, and the device
        // must take that same extra polish run (slackOnlyDistributedSlackCheck) to match the CPU trajectory.
        if (slackBusPhiRow < 0 || distributedSlackBusPhiRow.isEmpty()) {
            // a participating slack alone keeps its own path (slackOnlyDistributedSlackCheck): no generator pack
            distributedSlackGeneratorBus.clear();
            distributedSlackGeneratorRawFactor.clear();
            distributedSlackGeneratorMaxDelta.clear();
            distributedSlackGeneratorMinDelta.clear();
            distributedSlackGeneratorMarginUp.clear();
            distributedSlackGeneratorMarginDown.clear();
            distributedSlackGeneratorId.clear();
        }
        double[] distributedSlackGeneratorFactor = new double[distributedSlackGeneratorRawFactor.size()];     // SAME denominator, so Σ over a bus's
        for (int i = 0; i < distributedSlackGeneratorFactor.length; i++) {             // generators is exactly its distributedSlackBusFactor
            distributedSlackGeneratorFactor[i] = distributedSlackGeneratorRawFactor.get(i) / distributedSlackRawFactorSum;
        }
        if (slackBusPhiRow < 0 || distributedSlackBusPhiRow.isEmpty() && slackBusRawFactor <= 0) {
            distributedSlackBusPhiRow.clear();
            distributedSlackBusFactor = new double[0];
            distributedSlackBusBaseTargetP.clear();
            distributedSlackBusMaxDelta.clear();
            distributedSlackBusMinDelta.clear();
            distributedSlackBusMarginUp.clear();
            distributedSlackBusMarginDown.clear();
            slackBusPhiRow = -1;
            distributedSlackGeneratorBus.clear();
            distributedSlackGeneratorFactor = new double[0];
            distributedSlackGeneratorMaxDelta.clear();
            distributedSlackGeneratorMinDelta.clear();
            distributedSlackGeneratorMarginUp.clear();
            distributedSlackGeneratorMarginDown.clear();
            distributedSlackBusGeneratorStart.clear();
            distributedSlackBusGeneratorEnd.clear();
            distributedSlackBusNum.clear();
            distributedSlackGeneratorId.clear();
        }
        return new DistributedSlackParticipants(GpuAcArrays.toIntArray(distributedSlackBusPhiRow), distributedSlackBusFactor, slackBusPhiRow, slackBusTargetP,
                GpuAcArrays.toDoubleArray(distributedSlackBusBaseTargetP), GpuAcArrays.toDoubleArray(distributedSlackBusMaxDelta),
                GpuAcArrays.toDoubleArray(distributedSlackBusMinDelta),
                GpuDistributedSlackFactorMode.of(distributedSlackParameters.isRemainingMargin()),
                GpuAcArrays.toDoubleArray(distributedSlackBusMarginUp), GpuAcArrays.toDoubleArray(distributedSlackBusMarginDown),
                slackBusMarginUp, slackBusMarginDown, slackBusRoomUp, slackBusRoomDown,
                GpuAcArrays.toIntArray(distributedSlackGeneratorBus), distributedSlackGeneratorFactor,
                GpuAcArrays.toDoubleArray(distributedSlackGeneratorMaxDelta), GpuAcArrays.toDoubleArray(distributedSlackGeneratorMinDelta),
                GpuAcArrays.toDoubleArray(distributedSlackGeneratorMarginUp), GpuAcArrays.toDoubleArray(distributedSlackGeneratorMarginDown),
                GpuAcArrays.toIntArray(distributedSlackBusGeneratorStart), GpuAcArrays.toIntArray(distributedSlackBusGeneratorEnd),
                GpuAcArrays.toIntArray(distributedSlackBusNum), distributedSlackGeneratorId.toArray(new String[0]));
    }

    /** One participating generator of the slack distribution, on DS row {@code distributedSlackBusRow} (-1: the slack bus's own,
     *  which has no row): its factor, its rooms from the anchor and, for REMAINING_MARGIN, its margins. */
    private static void addPackGenerator(LfGenerator g, int distributedSlackBusRow, GpuDistributedSlackParameters distributedSlackParameters, List<Integer> distributedSlackGeneratorBus,
                                         List<String> distributedSlackGeneratorId, List<Double> distributedSlackGeneratorRawFactor, List<Double> distributedSlackGeneratorMaxDelta,
                                         List<Double> distributedSlackGeneratorMinDelta, List<Double> distributedSlackGeneratorMarginUp, List<Double> distributedSlackGeneratorMarginDown) {
        double gf = distributedSlackParameters.factor(g);
        double tp = distributedSlackParameters.anchorTargetP(g);
        distributedSlackGeneratorBus.add(distributedSlackBusRow);
        distributedSlackGeneratorId.add(g.getId());
        distributedSlackGeneratorRawFactor.add(gf);
        double[] rooms = distributedSlackParameters.rooms(g, tp);              // with active limits off the sign rule still holds
        distributedSlackGeneratorMaxDelta.add(rooms[0]);
        distributedSlackGeneratorMinDelta.add(rooms[1]);
        if (distributedSlackParameters.isRemainingMargin()) {
            double minP = g.getMinP();
            double maxP = g.getMaxP();
            if (tp < 0) {
                maxP = Math.min(maxP, 0);
            } else {
                minP = Math.max(minP, 0);
            }
            distributedSlackGeneratorMarginUp.add(Math.max(0.0, maxP - tp));
            distributedSlackGeneratorMarginDown.add(Math.max(0.0, tp - minP));
        }
    }
}
