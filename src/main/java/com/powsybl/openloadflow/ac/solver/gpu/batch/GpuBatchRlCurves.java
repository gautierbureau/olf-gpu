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

import java.util.ArrayList;
import java.util.List;

/**
 * Reactive capability CURVES at runtime P: the CPU RL loop re-reads {@code bus.getMinQ()/getMaxQ()}
 * each check, so a DS redistribution that moves a curve-limited generator's targetP moves its Q box
 * — while the device RL pack snapshots the limits at BASE targetP. This builder packs, per
 * PARTICIPATING curve generator, the curve points plus the linkage {@code p_now = baseP +
 * share·(rowTarget − baseRowTarget)} (share = the gen's DS factor over its bus's — exact while the
 * intra-bus split stays proportional, which is the pack's DS model), and the kernel
 * ({@code rlCurveLimitsBatch}) re-evaluates the bus limits at the CURRENT targets before every RL
 * decision. Non-participating curve gens never move — their base snapshot stays exact, no entry.
 *
 * <p>Channel layout (setBatchRlCurves): cvIdx stride 4 {rlIdx, pRow, ptsOff, npts}; cvIn stride 5
 * {basePpu, share, baseRowTargetP, baseEvalMinPu, baseEvalMaxPu}; cvPts stride 3
 * {p_MW, minQ_MW, maxQ_MW} sorted by p.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuBatchRlCurves {

    private GpuBatchRlCurves() {
    }

    /** The packed channel, null when the network needs none (no participating curve gen). */
    public record Channel(int[] cvIdx, double[] cvIn, double[] cvPts) {
    }

    /** True when any generator carries a reactive capability CURVE (the guard input for the
     *  extrapolation and injection-outage scope decisions). */
    public static boolean hasCurveGenerators(LfNetwork network) {
        return network.getBuses().stream().flatMap(b -> b.getGenerators().stream())
                .anyMatch(g -> g instanceof AbstractLfGenerator alg
                        && alg.getReactiveLimits().map(rl -> rl.getKind() == ReactiveLimitsKind.CURVE).orElse(false));
    }

    /** Build the channel by replaying the RL pack's bus enumeration (verified against
     *  {@code gpuData.rlVrow()} — a drift throws, falling back to CPU). Returns null when no
     *  PARTICIPATING curve generator exists (static snapshots stay exact). */
    public static Channel build(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                GpuAcData gpuData, GpuDsParams dsParams, boolean distributedSlack,
                                double[] baseTargets) {
        if (!distributedSlack) {
            return null;                                     // P never moves — the base snapshot is exact
        }
        List<int[]> cvIdx = new ArrayList<>();
        List<double[]> cvIn = new ArrayList<>();
        List<Double> cvPts = new ArrayList<>();
        int rlIdx = 0;
        for (LfBus bus : network.getBuses()) {
            int bn = bus.getNum();
            var vVar = es.getVariableSet().getVariable(bn, AcVariableType.BUS_V);
            boolean vActive = es.getEquation(bn, AcEquationType.BUS_TARGET_V).map(e -> e.isActive()).orElse(false);
            boolean qActive = es.getEquation(bn, AcEquationType.BUS_TARGET_Q).map(e -> e.isActive()).orElse(false);
            boolean basePv = vActive && !qActive;
            boolean basePqCapable = qActive && !vActive && !bus.isGeneratorVoltageControlEnabled();
            if (vVar == null || bus.getGeneratorVoltageControl().isEmpty() || !(basePv || basePqCapable)) {
                continue;
            }
            if (rlIdx >= gpuData.rlVrow().length || gpuData.rlVrow()[rlIdx] != vVar.getRow()) {
                throw new PowsyblException("GPU AC solver: RL-pack enumeration drift while packing curve limits (bus "
                        + bus.getId() + ")");
            }
            double busFactor = bus.getGenerators().stream().mapToDouble(dsParams::factor).sum();
            for (LfGenerator gen : bus.getGenerators()) {
                var curve = gen instanceof AbstractLfGenerator alg
                        ? alg.getReactiveLimits().filter(rl -> rl.getKind() == ReactiveLimitsKind.CURVE).orElse(null)
                        : null;
                if (curve == null || busFactor <= 0 || dsParams.factor(gen) <= 0) {
                    continue;                                // no curve, or its P never moves — static is exact
                }
                var points = ((ReactiveCapabilityCurve) curve).getPoints().stream()
                        .sorted(java.util.Comparator.comparingDouble(ReactiveCapabilityCurve.Point::getP)).toList();
                int ptsOff = cvPts.size() / 3;
                for (var pt : points) {
                    cvPts.add(pt.getP());
                    cvPts.add(pt.getMinQ());
                    cvPts.add(pt.getMaxQ());
                }
                int pRow = es.getVariableSet().getVariable(bn, AcVariableType.BUS_PHI).getRow();
                cvIdx.add(new int[] {rlIdx, pRow, ptsOff, points.size()});
                cvIn.add(new double[] {gen.getTargetP(), dsParams.factor(gen) / busFactor, baseTargets[pRow],
                    gen.getMinQ(), gen.getMaxQ()});          // base evals AT base targetP (bit-zero delta at base)
            }
            rlIdx++;
        }
        if (cvIdx.isEmpty()) {
            return null;
        }
        return new Channel(
                cvIdx.stream().flatMapToInt(java.util.stream.IntStream::of).toArray(),
                cvIn.stream().flatMapToDouble(java.util.stream.DoubleStream::of).toArray(),
                cvPts.stream().mapToDouble(Double::doubleValue).toArray());
    }
}
