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
    static double anchorOffset(LfBus bus, GpuDsParams dsParams) {
        double off = 0;
        for (LfGenerator g : bus.getGenerators()) {
            if (dsParams.factor(g) != 0) {
                off += g.getTargetP() - dsParams.anchorTargetP(g);
            }
        }
        return off;
    }

    static DistributedSlackParticipants distributedSlackPack(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                       ExtractionColumns columns, boolean distributedSlack, GpuDsParams dsParams) {
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
        // no row (dgBus -1): they take their share, clamp at their own limits and leave the list like any other, so
        // a saturated peer's remainder is renormalized over them too. The slack's placement is the sum of their
        // moves (dsRetain / retainApplied).
        //
        // SATURATION: per-generator rooms (maxTargetP−targetP / minTargetP−targetP with OLF's no-sign-change rule)
        // bound each generator's move; the water-fill removes a generator when it strictly overshoots and
        // renormalizes over the rest, as GenerationActivePowerDistributionStep does. The per-bus headroom/footroom
        // below remain for the bus-level aggregate.
        int dsSlackPhiRow = -1;
        double dsSlackTargetP = 0;
        double dsSlackFactor = 0;                                       // slack's own participation (denominator only)
        double dsSlackMarginUp = 0;                                     // REMAINING_MARGIN: the slack's own margins
        double dsSlackMarginDown = 0;
        double dsSlackRoomUp = Double.POSITIVE_INFINITY;                // the slack's participating generators' total
        double dsSlackRoomDown = Double.NEGATIVE_INFINITY;              // rooms: what a slack-only distribution can place
        List<Integer> dsPhiRow = new ArrayList<>();
        List<Integer> dgBus = new ArrayList<>();
        List<Double> dgFactorRaw = new ArrayList<>();
        List<Double> dgMaxDelta = new ArrayList<>();
        List<Double> dgMinDelta = new ArrayList<>();
        List<Double> dgMarginUp = new ArrayList<>();
        List<Double> dgMarginDown = new ArrayList<>();
        List<Integer> busGenStart = new ArrayList<>();
        List<Integer> busGenEnd = new ArrayList<>();
        List<Integer> dsBusNum = new ArrayList<>();                     // WHICH bus each row is (applyToNetwork)
        List<String> dgGenId = new ArrayList<>();                       // WHICH generator each slot is
        List<Double> dsFactorRaw = new ArrayList<>();
        List<Double> dsBaseTargetP = new ArrayList<>();                 // each bus's initial BUS_TARGET_P
        List<Double> dsMaxDelta = new ArrayList<>();                    // headroom ≥ 0
        List<Double> dsMinDelta = new ArrayList<>();                    // footroom ≤ 0
        List<Double> dsMarginUp = new ArrayList<>();                    // REMAINING_MARGIN per-bus margins
        List<Double> dsMarginDown = new ArrayList<>();
        for (LfBus bus : network.getBuses()) {
            if (!distributedSlack) {                                    // empty pack when the DS outer loop is disabled
                break;
            }
            int bn = bus.getNum();
            if (GpuAcRowIndexing.row(es, bn, AcEquationType.BUS_TARGET_PHI) >= 0) {       // the slack/reference bus
                dsSlackPhiRow = pRow[bn];
                dsSlackTargetP = bus.getTargetP() - anchorOffset(bus, dsParams);   // the slack's ANCHOR target
                dsSlackFactor = dsParams.busFactor(bus);
                if (dsParams.isRemainingMargin()) {
                    double[] m = dsParams.busMargins(bus);
                    dsSlackMarginUp = m[0];
                    dsSlackMarginDown = m[1];
                }
                // OLF's water-fill lists the slack bus's participating generators with every other participant,
                // in bus order: they take a share, are clamped at their own limits and leave the list like any
                // other, so a peer's saturated remainder is renormalized over them too. They join the generator
                // pack with no row (dgBus -1); the slack's placement is the sum of their moves. Loads have no
                // limits, so a load-based slack keeps its fixed share.
                if (!dsParams.isLoadBased()) {
                    dsSlackRoomUp = 0;
                    dsSlackRoomDown = 0;
                    for (LfGenerator g : bus.getGenerators()) {
                        if (dsParams.factor(g) != 0) {
                            double[] r = dsParams.rooms(g, dsParams.anchorTargetP(g));
                            dsSlackRoomUp += r[0];
                            dsSlackRoomDown += r[1];
                            addPackGenerator(g, -1, dsParams, dgBus, dgGenId, dgFactorRaw, dgMaxDelta, dgMinDelta,
                                    dgMarginUp, dgMarginDown);
                        }
                    }
                }
                continue;
            }
            if (GpuAcRowIndexing.row(es, bn, AcEquationType.BUS_TARGET_P) < 0) {
                continue;                                               // no active P balance to retarget
            }
            double factor = dsParams.busFactor(bus);
            double headroom = 0;
            double footroom = 0;
            for (LfGenerator g : bus.getGenerators()) {
                if (dsParams.factor(g) != 0) {
                    double[] r = dsParams.rooms(g, dsParams.anchorTargetP(g));   // OLF's bounds + sign rule
                    headroom += r[0];
                    footroom += r[1];
                }
            }
            if (dsParams.isLoadBased()) {
                headroom = Double.POSITIVE_INFINITY;                    // loads: OLF rescales them unbounded
                footroom = Double.NEGATIVE_INFINITY;
            }
            if (factor > 0) {
                dsPhiRow.add(pRow[bn]);
                dsBusNum.add(bn);
                dsFactorRaw.add(factor);
                dsBaseTargetP.add(bus.getTargetP() - anchorOffset(bus, dsParams));   // the row's ANCHOR target
                dsMaxDelta.add(headroom);
                dsMinDelta.add(footroom);
                if (dsParams.isRemainingMargin()) {
                    double[] m = dsParams.busMargins(bus);
                    dsMarginUp.add(m[0]);
                    dsMarginDown.add(m[1]);
                }
                // Per GENERATOR as well, contiguous within the bus. OLF's water-fill participates and
                // SATURATES per generator (ParticipatingElement wraps an LfGenerator, and step.run
                // clamps each one at its OWN minTargetP/maxTargetP and removes it), so a bus-level
                // aggregate only reproduces it while nothing saturates. Both groupings are kept: the
                // bus one owns the equation row and the anchor, the generator one owns the factors
                // and the limits.
                busGenStart.add(dgFactorRaw.size());
                for (LfGenerator g : bus.getGenerators()) {
                    if (dsParams.factor(g) == 0) {
                        continue;                            // OLF: isParticipating(gen) && factor != 0
                    }
                    addPackGenerator(g, dsPhiRow.size() - 1, dsParams, dgBus, dgGenId, dgFactorRaw, dgMaxDelta, dgMinDelta,
                            dgMarginUp, dgMarginDown);
                }
                busGenEnd.add(dgFactorRaw.size());
            }
        }
        double dsSum = dsFactorRaw.stream().mapToDouble(Double::doubleValue).sum() + dsSlackFactor;
        double[] dsFactor = new double[dsPhiRow.size()];
        for (int i = 0; i < dsFactor.length; i++) {
            dsFactor[i] = dsFactorRaw.get(i) / dsSum;
        }
        // A pack with participants but no slack row is unusable; one with NEITHER participants nor a
        // participating slack means DS never fires. But a PARTICIPATING SLACK ALONE is kept (nds == 0,
        // dsSlackPhiRow >= 0): OLF then distributes the whole mismatch into the slack generator's own
        // target — no flow change, but the outer loop reports UNSTABLE and re-solves, and the device
        // must take that same extra polish run (slackOnlyDsCheck) to match the CPU trajectory.
        if (dsSlackPhiRow < 0 || dsPhiRow.isEmpty()) {
            // a participating slack alone keeps its own path (slackOnlyDsCheck): no generator pack
            dgBus.clear();
            dgFactorRaw.clear();
            dgMaxDelta.clear();
            dgMinDelta.clear();
            dgMarginUp.clear();
            dgMarginDown.clear();
            dgGenId.clear();
        }
        double[] dgFactor = new double[dgFactorRaw.size()];     // SAME denominator, so Σ over a bus's
        for (int i = 0; i < dgFactor.length; i++) {             // generators is exactly its dsFactor
            dgFactor[i] = dgFactorRaw.get(i) / dsSum;
        }
        if (dsSlackPhiRow < 0 || dsPhiRow.isEmpty() && dsSlackFactor <= 0) {
            dsPhiRow.clear();
            dsFactor = new double[0];
            dsBaseTargetP.clear();
            dsMaxDelta.clear();
            dsMinDelta.clear();
            dsMarginUp.clear();
            dsMarginDown.clear();
            dsSlackPhiRow = -1;
            dgBus.clear();
            dgFactor = new double[0];
            dgMaxDelta.clear();
            dgMinDelta.clear();
            dgMarginUp.clear();
            dgMarginDown.clear();
            busGenStart.clear();
            busGenEnd.clear();
            dsBusNum.clear();
            dgGenId.clear();
        }
        return new DistributedSlackParticipants(GpuAcArrays.toIntArray(dsPhiRow), dsFactor, dsSlackPhiRow, dsSlackTargetP,
                GpuAcArrays.toDoubleArray(dsBaseTargetP), GpuAcArrays.toDoubleArray(dsMaxDelta),
                GpuAcArrays.toDoubleArray(dsMinDelta),
                GpuDistributedSlackFactorMode.of(dsParams.isRemainingMargin()),
                GpuAcArrays.toDoubleArray(dsMarginUp), GpuAcArrays.toDoubleArray(dsMarginDown),
                dsSlackMarginUp, dsSlackMarginDown, dsSlackRoomUp, dsSlackRoomDown,
                GpuAcArrays.toIntArray(dgBus), dgFactor,
                GpuAcArrays.toDoubleArray(dgMaxDelta), GpuAcArrays.toDoubleArray(dgMinDelta),
                GpuAcArrays.toDoubleArray(dgMarginUp), GpuAcArrays.toDoubleArray(dgMarginDown),
                GpuAcArrays.toIntArray(busGenStart), GpuAcArrays.toIntArray(busGenEnd),
                GpuAcArrays.toIntArray(dsBusNum), dgGenId.toArray(new String[0]));
    }

    /** One participating generator of the slack distribution, on DS row {@code dsRow} (-1: the slack bus's own,
     *  which has no row): its factor, its rooms from the anchor and, for REMAINING_MARGIN, its margins. */
    private static void addPackGenerator(LfGenerator g, int dsRow, GpuDsParams dsParams, List<Integer> dgBus,
                                         List<String> dgGenId, List<Double> dgFactorRaw, List<Double> dgMaxDelta,
                                         List<Double> dgMinDelta, List<Double> dgMarginUp, List<Double> dgMarginDown) {
        double gf = dsParams.factor(g);
        double tp = dsParams.anchorTargetP(g);
        dgBus.add(dsRow);
        dgGenId.add(g.getId());
        dgFactorRaw.add(gf);
        double[] rooms = dsParams.rooms(g, tp);              // with active limits off the sign rule still holds
        dgMaxDelta.add(rooms[0]);
        dgMinDelta.add(rooms[1]);
        if (dsParams.isRemainingMargin()) {
            double minP = g.getMinP();
            double maxP = g.getMaxP();
            if (tp < 0) {
                maxP = Math.min(maxP, 0);
            } else {
                minP = Math.max(minP, 0);
            }
            dgMarginUp.add(Math.max(0.0, maxP - tp));
            dgMarginDown.add(Math.max(0.0, tp - minP));
        }
    }
}
