/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.commons.PowsyblException;
import com.powsybl.iidm.network.ReactiveCapabilityCurve;
import com.powsybl.iidm.network.ReactiveLimitsKind;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDsParams;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.impl.AbstractLfGenerator;
import com.powsybl.openloadflow.network.impl.LfGeneratorImpl;

import java.util.ArrayList;
import java.util.List;

/**
 * forceTargetQInReactiveLimits (ON in the RTE production parameters): {@code LfGeneratorImpl.getTargetQ} clamps a
 * generator's IIDM target Q into its reactive limits at its CURRENT target P, and a bus's generation target Q (the
 * sum over its generators not in voltage control) is re-read whenever one of them changes P
 * ({@code AbstractLfBus.invalidateGenerationTargetP}). The device moves P itself - the security analysis' loss
 * pre-distribution and every distributed-slack pass - so it must clamp again: this packs, per generator whose clamp
 * can move, its curve, its raw target and the bus Q row its target sits on, and the kernels
 * ({@code forcedTargetQInitBatch}, {@code forcedTargetQBatch}) move that row by the clamp's change.
 *
 * <p>Only generators that can move: a curve (fixed min/max limits clamp to the same value at every P) and a
 * distributed-slack participant (no one else moves its P). Only on a bus WITHOUT generator voltage control, whose
 * Q balance row carries the generation target: on a voltage-controlled bus the target enters the bus's reactive
 * LIMITS instead (AbstractLfBus.getLimitQ), which this channel does not move - that shape throws, and the caller
 * keeps the run on the CPU.
 *
 * <p>Layout (setBatchForcedTargetQ): fqIdx stride 6 {qRow, dsGenSlot, ptsOff, npts, dsBusIdx, baseQRowMode};
 * fqIn stride 5 {basePpu, rawTargetQPu, baseClampedPu, share, baseRowTargetP}; fqPts stride 3 {p_MW, minQ_MW,
 * maxQ_MW} sorted by p.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuBatchForcedTargetQ {

    private GpuBatchForcedTargetQ() {
    }

    /** The packed channel; {@code genSlot} is each entry's generator slot in the distributed-slack generator pack
     *  (host side, to pick the entry's pre-distribution move). */
    public record Channel(int[] fqIdx, double[] fqIn, double[] fqPts, boolean extrapolate, int[] genSlot) {

        public int size() {
            return genSlot.length;
        }

        /** {@code [nS * nfq]}: each entry's share of the scenarios' loss pre-distribution, from the
         *  per-generator water-fill moves ({@code genMoves[s]} indexed by generator slot, null = no move). */
        public double[] preMoves(double[][] genMoves) {
            int nfq = size();
            double[] pre = new double[genMoves.length * nfq];
            boolean any = false;
            for (int s = 0; s < genMoves.length; s++) {
                if (genMoves[s] == null) {
                    continue;
                }
                for (int e = 0; e < nfq; e++) {
                    pre[s * nfq + e] = genMoves[s][genSlot[e]];
                    any |= pre[s * nfq + e] != 0;
                }
            }
            return any ? pre : null;
        }
    }

    /** Null when nothing can move: the flag off, no distributed slack, or no clamping curve generator. */
    public static Channel build(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                GpuAcData gpuData, GpuDsParams dsParams, boolean forceTargetQ,
                                boolean distributedSlack, double[] baseTargets, int[] baseRowMode,
                                boolean extrapolate) {
        if (!forceTargetQ || !distributedSlack || gpuData.dgGenId() == null) {
            return null;
        }
        int[][] busRows = com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor.busPowerRows(network, es);
        List<int[]> idx = new ArrayList<>();
        List<double[]> in = new ArrayList<>();
        List<Double> pts = new ArrayList<>();
        List<Integer> slots = new ArrayList<>();
        for (LfBus bus : network.getBuses()) {
            if (bus.isDisabled()) {
                continue;
            }
            double busFactor = bus.getGenerators().stream().mapToDouble(dsParams::factor).sum();
            for (LfGenerator gen : bus.getGenerators()) {
                if (!(gen instanceof LfGeneratorImpl) || gen.isDisabled()
                        || gen.getGeneratorControlType() == LfGenerator.GeneratorControlType.VOLTAGE) {
                    continue;                                // not in the bus's generation target Q
                }
                var curve = ((AbstractLfGenerator) gen).getReactiveLimits()
                        .filter(rl -> rl.getKind() == ReactiveLimitsKind.CURVE).orElse(null);
                int slot = dsSlotOf(gpuData.dgGenId(), gen.getId());
                if (curve == null || slot < 0 || dsParams.factor(gen) <= 0) {
                    continue;                                // its clamp cannot move: the base target stays exact
                }
                if (bus.isGeneratorVoltageControlEnabled()
                        || gen.getGeneratorControlType() == LfGenerator.GeneratorControlType.REMOTE_REACTIVE_POWER) {
                    throw new PowsyblException("GPU AC solver: forced target Q of generator " + gen.getId()
                            + " on a voltage- or reactive-power-controlled bus is not supported");
                }
                boolean qActive = es.getEquation(bus.getNum(), AcEquationType.BUS_TARGET_Q)
                        .map(eq -> eq.isActive()).orElse(false);
                int qRow = busRows[1][bus.getNum()];
                if (!qActive || qRow < 0) {
                    continue;                                // no Q balance row carries its target
                }
                int dsIdx = dsBusIndexOf(gpuData, bus.getNum());
                if (dsIdx < 0) {
                    throw new PowsyblException("GPU AC solver: participating generator " + gen.getId()
                            + " on a bus outside the distributed-slack pack");
                }
                var points = ((ReactiveCapabilityCurve) curve).getPoints().stream()
                        .sorted(java.util.Comparator.comparingDouble(ReactiveCapabilityCurve.Point::getP)).toList();
                int ptsOff = pts.size() / 3;
                for (var pt : points) {
                    pts.add(pt.getP());
                    pts.add(pt.getMinQ());
                    pts.add(pt.getMaxQ());
                }
                int pRow = gpuData.dsPhiRow()[dsIdx];
                idx.add(new int[] {qRow, slot, ptsOff, points.size(), dsIdx, baseRowMode[qRow]});
                in.add(new double[] {gen.getTargetP(), gen.getRawTargetQ(), gen.getTargetQ(),
                    dsParams.factor(gen) / busFactor, baseTargets[pRow]});
                slots.add(slot);
            }
        }
        if (idx.isEmpty()) {
            return null;
        }
        return new Channel(idx.stream().flatMapToInt(java.util.stream.IntStream::of).toArray(),
                in.stream().flatMapToDouble(java.util.stream.DoubleStream::of).toArray(),
                pts.stream().mapToDouble(Double::doubleValue).toArray(), extrapolate,
                slots.stream().mapToInt(Integer::intValue).toArray());
    }

    /** Install the channel, or clear it (a cached context reused by a run without it). */
    public static void install(long handle, Channel channel) {
        com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.setBatchForcedTargetQ(handle,
                channel != null ? channel.fqIdx() : null,
                channel != null ? channel.fqIn() : null,
                channel != null ? channel.fqPts() : null,
                channel != null && channel.extrapolate() ? 1 : 0);
    }

    private static int dsBusIndexOf(GpuAcData gpuData, int busNum) {
        int[] nums = gpuData.dsBusNum();
        if (nums == null) {
            return -1;
        }
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == busNum) {
                return i;
            }
        }
        return -1;
    }

    private static int dsSlotOf(String[] dgGenId, String genId) {
        for (int i = 0; i < dgGenId.length; i++) {
            if (dgGenId[i].equals(genId)) {
                return i;
            }
        }
        return -1;
    }
}
