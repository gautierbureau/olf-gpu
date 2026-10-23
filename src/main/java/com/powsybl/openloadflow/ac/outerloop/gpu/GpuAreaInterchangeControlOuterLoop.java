/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import com.powsybl.iidm.network.BoundaryLine;
import com.powsybl.iidm.network.TwoSides;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcElementMapper;
import com.powsybl.openloadflow.ac.outerloop.AcAreaInterchangeControlOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.AcOuterLoop;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuAreaBoundaryKind;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuTwoSides;
import com.powsybl.openloadflow.network.LfArea;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.impl.LfAreaImpl;
import com.powsybl.openloadflow.network.impl.LfTieLineBranch;
import com.powsybl.openloadflow.util.PerUnit;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * AreaInterchangeControl as a device driver phase (native/cuda/outerloop/area_interchange_control.cu): OLF's
 * {@link AcAreaInterchangeControlOuterLoop}, which takes the DistributedSlack loop's place in a run and moves the same
 * participating generators, one area at a time. Builds the phase's tables from the network and the distributed-slack pack,
 * and gives the generators' targets back to the network after a solve.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuAreaInterchangeControlOuterLoop {

    /** The area of a generator on a fictitious bus without area: OLF's buses without area leave fictitious buses out. */
    private static final int NO_DISTRIBUTION = -2;

    /** Device solves whose AreaInterchangeControl phase moved generators, in either of its forms (how a test tells the
     *  device drove the loop: OLF's own loop run on the host around the device reaches the same targets). */
    public static final java.util.concurrent.atomic.LongAdder DEVICE_DISTRIBUTIONS = new java.util.concurrent.atomic.LongAdder();

    private GpuAreaInterchangeControlOuterLoop() {
    }

    /** The phase's tables (see AreaInterchangeControlNative.setBatchAreaInterchangeControl); no area at all makes the
     *  phase OLF's DistributedSlackOuterLoop. */
    public record Tables(double areaInterchangePMaxMismatch, double[] interchangeTarget, double[] slackDistributionFactor,
                         int[] boundaryStart, int[] boundaryKind, int[] boundaryElement, int[] boundaryMaskSlot,
                         int[] boundarySide, double[] boundaryTieLineHalf, int[] busArea, int[] generatorArea) {

        public int areaCount() {
            return interchangeTarget.length;
        }
    }

    /** The run's AreaInterchangeControl loop, null when it has none. */
    public static AcAreaInterchangeControlOuterLoop find(List<AcOuterLoop> outerLoops) {
        for (AcOuterLoop outerLoop : outerLoops) {
            if (outerLoop instanceof AcAreaInterchangeControlOuterLoop areaInterchangeControlOuterLoop) {
                return areaInterchangeControlOuterLoop;
            }
        }
        return null;
    }

    /**
     * The tables of {@code loop} on {@code network}, or null when the device cannot represent its areas: a distribution
     * without per-generator participants (the load-based balance types) or without the slack bus's row, a boundary on a
     * branch the device does not model (a zero-impedance branch) or a tie line connected at one side only.
     */
    public static Tables build(LfNetwork network, AcAreaInterchangeControlOuterLoop loop, GpuAcData data, GpuAcElementMapper mapper) {
        double areaInterchangePMaxMismatch = loop.getAreaInterchangePMaxMismatch() / PerUnit.SB;
        if (!network.hasArea()) {
            return new Tables(areaInterchangePMaxMismatch, new double[0], new double[0], new int[] {0}, new int[0], new int[0],
                    new int[0], new int[0], new double[0], new int[0], new int[0]);
        }
        var participants = data.distributedSlack();
        if (participants.generatorBus().length == 0 || participants.slackPhiRow() < 0
                || network.getSynchronousNetworks().getFirst().getSlackBuses().size() != 1) {
            return null;
        }
        List<LfArea> areas = network.getAreas();
        Map<String, Double> slackDistributionFactorByAreaId = loop.allocateSlackDistributionParticipationFactors(network);
        double[] interchangeTarget = new double[areas.size()];
        double[] slackDistributionFactor = new double[areas.size()];
        int[] boundaryStart = new int[areas.size() + 1];
        List<Integer> boundaryKind = new ArrayList<>();
        List<Integer> boundaryElement = new ArrayList<>();
        List<Integer> boundaryMaskSlot = new ArrayList<>();
        List<Integer> boundarySide = new ArrayList<>();
        List<Double> boundaryTieLineHalf = new ArrayList<>();
        int closedBranchCount = mapper.getClosedBranchCount();
        for (LfArea area : areas) {
            int a = area.getNum();
            interchangeTarget[a] = area.getInterchangeTarget();
            slackDistributionFactor[a] = slackDistributionFactorByAreaId.getOrDefault(area.getId(), 0.0);
            boundaryStart[a] = boundaryKind.size();
            for (LfArea.Boundary boundary : area.getBoundaries()) {
                LfBranch branch = boundary.getBranch();
                if (!(boundary instanceof LfAreaImpl.BoundaryImpl boundaryImpl)) {
                    return null;
                }
                int element = mapper.getElementIndex(branch);
                if (element < 0) {
                    if (branch.isDisabled()) {
                        continue;                                // OLF reads 0 for a disabled branch
                    }
                    return null;                                 // not in the device model
                }
                boolean closed = element < closedBranchCount;
                int kind;
                if (branch instanceof LfTieLineBranch tieLine) {
                    if (!closed) {
                        return null;
                    }
                    kind = GpuAreaBoundaryKind.TIE_LINE;
                    BoundaryLine half = boundaryImpl.getSide() == TwoSides.ONE ? tieLine.getHalf1() : tieLine.getHalf2();
                    boundaryTieLineHalf.add(half.getR());
                    boundaryTieLineHalf.add(half.getX());
                    boundaryTieLineHalf.add(half.getG());
                    boundaryTieLineHalf.add(half.getB());
                    boundaryTieLineHalf.add(half.getTerminal().getVoltageLevel().getNominalV());
                    boundaryTieLineHalf.add(PerUnit.SB);
                } else {
                    kind = closed ? GpuAreaBoundaryKind.CLOSED_BRANCH : GpuAreaBoundaryKind.OPEN_BRANCH;
                    for (int k = 0; k < 6; k++) {
                        boundaryTieLineHalf.add(0.0);
                    }
                }
                boundaryKind.add(kind);
                boundaryElement.add(closed ? element : element - closedBranchCount);
                boundaryMaskSlot.add(element);
                boundarySide.add(boundaryImpl.getSide() == TwoSides.ONE ? GpuTwoSides.ONE : GpuTwoSides.TWO);
            }
        }
        boundaryStart[areas.size()] = boundaryKind.size();
        int[] busArea = new int[participants.busNum().length];
        for (int i = 0; i < busArea.length; i++) {
            busArea[i] = areaOf(network.getBus(participants.busNum()[i]));
        }
        Map<String, LfGenerator> generatorById = generatorsById(network);
        int[] generatorArea = new int[participants.generatorId().length];
        for (int g = 0; g < generatorArea.length; g++) {
            LfGenerator generator = generatorById.get(participants.generatorId()[g]);
            generatorArea[g] = generator != null ? areaOf(generator.getBus()) : NO_DISTRIBUTION;
        }
        return new Tables(areaInterchangePMaxMismatch, interchangeTarget, slackDistributionFactor, boundaryStart,
                toIntArray(boundaryKind), toIntArray(boundaryElement), toIntArray(boundaryMaskSlot), toIntArray(boundarySide),
                boundaryTieLineHalf.stream().mapToDouble(Double::doubleValue).toArray(), busArea, generatorArea);
    }

    /** Install the run's tables, or clear the phase with null. */
    public static void uploadToDevice(long handle, Tables tables) {
        if (tables == null) {
            AreaInterchangeControlNative.setBatchAreaInterchangeControl(handle, false, 0, null, null, null, null, null, null, null,
                    null, null, null);
            return;
        }
        AreaInterchangeControlNative.setBatchAreaInterchangeControl(handle, true, tables.areaInterchangePMaxMismatch(),
                tables.interchangeTarget(), tables.slackDistributionFactor(), tables.boundaryStart(), tables.boundaryKind(),
                tables.boundaryElement(), tables.boundaryMaskSlot(), tables.boundarySide(), tables.boundaryTieLineHalf(),
                tables.busArea(), tables.generatorArea());
    }

    /** Each participating generator's move from its initial target when the solve starts: OLF's distribution resets them
     *  there and adds the move back to the mismatch. */
    public static double[] initialMoves(LfNetwork network, GpuAcData data) {
        Map<String, LfGenerator> generatorById = generatorsById(network);
        String[] generatorId = data.distributedSlack().generatorId();
        double[] moves = new double[generatorId.length];
        for (int g = 0; g < moves.length; g++) {
            LfGenerator generator = generatorById.get(generatorId[g]);
            moves[g] = generator != null ? generator.getTargetP() - generator.getInitialTargetP() : 0.0;
        }
        return moves;
    }

    /** Give every participating generator its initial target plus the move the device left it at. */
    public static void applyToNetwork(LfNetwork network, GpuAcData data, double[] moves) {
        if (moves == null) {
            return;
        }
        Map<String, LfGenerator> generatorById = generatorsById(network);
        String[] generatorId = data.distributedSlack().generatorId();
        boolean moved = false;
        for (int g = 0; g < generatorId.length && g < moves.length; g++) {
            LfGenerator generator = generatorById.get(generatorId[g]);
            if (generator != null) {
                double targetP = generator.getInitialTargetP() + moves[g];
                moved |= targetP != generator.getTargetP();
                generator.setTargetP(targetP);
            }
        }
        if (moved) {
            DEVICE_DISTRIBUTIONS.increment();
        }
    }

    private static int areaOf(LfBus bus) {
        if (bus == null) {
            return NO_DISTRIBUTION;
        }
        return bus.getArea().map(LfArea::getNum).orElse(bus.isFictitious() ? NO_DISTRIBUTION : -1);
    }

    private static Map<String, LfGenerator> generatorsById(LfNetwork network) {
        Map<String, LfGenerator> generatorById = new HashMap<>();
        for (LfBus bus : network.getBuses()) {
            for (LfGenerator generator : bus.getGenerators()) {
                generatorById.put(generator.getId(), generator);
            }
        }
        return generatorById;
    }

    private static int[] toIntArray(List<Integer> values) {
        return values.stream().mapToInt(Integer::intValue).toArray();
    }
}
