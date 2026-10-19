/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

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
 * <p>Channel layout (setBatchRlCurves): cvIdx stride 5 {rlIdx, pRow, ptsOff, npts, dsGenSlot}; cvIn stride 5
 * {basePpu, share, baseRowTargetP, baseEvalMinPu, baseEvalMaxPu}; cvPts stride 3
 * {p_MW, minQ_MW, maxQ_MW} sorted by p; cvHv stride 2 {hvdc, side} (null when no entry is one).
 *
 * <p>An AC-emulation VSC station is an entry too, whatever the slack distribution: OLF evaluates its curve at the
 * live flow of its link in the linear mode, and at its target P otherwise
 * (LfVscConverterStationImpl.getReactiveLimitsActivePower). Its {hvdc, side} points the kernel at the link's state
 * in the HVDC loop tables (the same enumeration as HvdcAcEmulation.buildControllers).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class ReactiveCapabilityCurves {

    private ReactiveCapabilityCurves() {
    }

    /** The packed channel, null when the network needs none (no participating curve gen).
     *  {@code extrapolate} = OLF's extrapolateReactiveLimits: the kernel continues the boundary
     *  two-point slope past [minP, maxP] instead of clamping (the base evals in cvIn come from
     *  OLF's own gen.getMinQ()/getMaxQ(), which already honor the flag). */
    public record Channel(int[] cvIdx, double[] cvIn, double[] cvPts, boolean extrapolate,
                         String[] cvGenId, double[] cvBusFactor, int[] cvHv) {

        /** Whether entry {@code e} is an AC-emulation VSC: its P is its link's, not the slack's. */
        public boolean isHvdc(int e) {
            return cvHv != null && cvHv[2 * e] >= 0;
        }
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
                                double[] baseTargets, boolean extrapolate) {
        // the AC-emulation stations, in the HVDC loop tables' order: station -> {hvdc, side}
        java.util.Map<LfGenerator, int[]> hvdcSides = new java.util.IdentityHashMap<>();
        List<com.powsybl.openloadflow.network.LfHvdc> packedHvdcs = network.getHvdcs().stream()
                .filter(h -> h.getBus1() != null && h.getBus2() != null && h.isAcEmulation())
                .toList();
        for (int h = 0; h < packedHvdcs.size(); h++) {
            hvdcSides.put(packedHvdcs.get(h).getConverterStation1(), new int[] {h, 1});
            hvdcSides.put(packedHvdcs.get(h).getConverterStation2(), new int[] {h, 2});
        }
        List<Integer> cvHv = new ArrayList<>();
        List<int[]> cvIdx = new ArrayList<>();
        List<double[]> cvIn = new ArrayList<>();
        List<Double> cvPts = new ArrayList<>();
        List<String> cvGenId = new ArrayList<>();
        List<Double> cvBusFactor = new ArrayList<>();
        // Each bus's P-BALANCE row, from the one source of truth. Resolving it from the variable set
        // instead gives the bus's own BUS_PHI column, which is NOT where its P target lives once the
        // coupler-tree relocation has moved it onto the parent coupler's DUMMY_P — and this channel
        // uses the row twice, as a kernel row index and as a baseTargets[] index, so a stale row
        // evaluates every curve on this bus at the wrong active power.
        int[] busPRow = com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor
                .busPowerRows(network, es)[0];
        int rlIdx = 0;
        for (LfBus bus : network.getBuses()) {
            int bn = bus.getNum();
            var vVar = es.getVariableSet().getVariable(bn, AcVariableType.BUS_V);
            // REPLAY the RL pack's enumeration exactly (ReactiveLimitsPacker.reactiveLimitsPack):
            // a remote group member with finite limits is an rlRemote entry regardless of its base
            // pin state; a local bus enters as basePv or basePqCapable.
            // the SAME predicates as the reactive-limit pack (a local copy drifted on rte6515's full
            // busbar scope at VL-5_0: a pilot whose own control is merged is reached through the
            // main control there)
            boolean remoteMember = com.powsybl.openloadflow.ac.equations.vector.gpu.RemoteMemberQPacker.isRemoteMemberEntry(bus);
            boolean pilotSelf = com.powsybl.openloadflow.ac.equations.vector.gpu.PilotSelfPacker.isPilotSelfEntry(bus);
            if (remoteMember || pilotSelf) {
                var vc = com.powsybl.openloadflow.ac.equations.vector.gpu.RemoteMemberQPacker.mainGenVc(bus);
                var pilotVar = es.getVariableSet().getVariable(vc.getControlledBus().getNum(), AcVariableType.BUS_V);
                // a row < 0 is a bus a contingency disabled: the pack skips it (vCol < 0) and so must the replay
                if (vVar == null || vVar.getRow() < 0 || pilotVar == null || pilotVar.getRow() < 0
                        || !(bus.getMinQ() < bus.getMaxQ())) {
                    continue;                                // not packed
                }
                if (rlIdx >= gpuData.rlVrow().length || gpuData.rlVrow()[rlIdx] != vVar.getRow()) {
                    throw new PowsyblException("GPU AC solver: RL-pack enumeration drift while packing curve limits (bus "
                            + bus.getId() + ")");
                }
                // remote members keep STATIC limits in the channel (a curve gen at a remote member is
                // possible but its own P participation is the same as a local bus's) — fall through to
                // the same per-generator curve packing below.
                packBusCurves(bus, rlIdx, es, distributedSlack ? dsParams : null, baseTargets, busPRow, cvIdx, cvIn, cvPts,
                        cvGenId, cvBusFactor, gpuData, hvdcSides, cvHv);
                rlIdx++;
                continue;
            }
            boolean vActive = es.getEquation(bn, AcEquationType.BUS_TARGET_V).map(e -> e.isActive()).orElse(false);
            boolean qActive = es.getEquation(bn, AcEquationType.BUS_TARGET_Q).map(e -> e.isActive()).orElse(false);
            boolean basePv = vActive && !qActive;
            boolean basePqCapable = qActive && !vActive && !bus.isGeneratorVoltageControlEnabled();
            if (vVar == null || vVar.getRow() < 0 || bus.getGeneratorVoltageControl().isEmpty() || !(basePv || basePqCapable)) {
                continue;
            }
            if (rlIdx >= gpuData.rlVrow().length || gpuData.rlVrow()[rlIdx] != vVar.getRow()) {
                throw new PowsyblException("GPU AC solver: RL-pack enumeration drift while packing curve limits (bus "
                        + bus.getId() + ")");
            }
            packBusCurves(bus, rlIdx, es, distributedSlack ? dsParams : null, baseTargets, busPRow, cvIdx, cvIn, cvPts,
                    cvGenId, cvBusFactor, gpuData, hvdcSides, cvHv);
            rlIdx++;
        }
        if (cvIdx.isEmpty()) {
            return null;
        }
        // OLF_GPU_CURVE_DUMP=<file>: what the channel packs per curve generator — the generator, its
        // base P and the SHARE the kernel uses to reconstruct its own P from its bus's row target.
        // Joined against OLF's per-generator movement (OLF_DS_SPLIT's G lines) this says directly
        // whether the proportional-share model holds for a given scenario.
        String dump = System.getenv("OLF_GPU_CURVE_DUMP");
        if (dump != null) {
            try (java.io.PrintWriter w = new java.io.PrintWriter(new java.io.FileWriter(dump))) {
                for (int g = 0; g < cvGenId.size(); g++) {
                    double[] in = cvIn.get(g);
                    int[] ix = cvIdx.get(g);
                    StringBuilder sb = new StringBuilder(String.format(
                            "%s rlIdx=%d baseP=%.12f share=%.12f baseRowTarget=%.12f busFactor=%.12f "
                            + "baseEvalMin=%.12f baseEvalMax=%.12f dsSlot=%d npts=%d pts=",
                            cvGenId.get(g), ix[0], in[0], in[1], in[2], cvBusFactor.get(g), in[3], in[4],
                            ix[4], ix[3]));
                    for (int k = 0; k < ix[3]; k++) {        // cvPts stride 3 {p_MW, minQ_MW, maxQ_MW}
                        int o = (ix[2] + k) * 3;
                        sb.append(String.format("%s%.9f:%.9f:%.9f", k == 0 ? "" : ",",
                                cvPts.get(o), cvPts.get(o + 1), cvPts.get(o + 2)));
                    }
                    w.println(sb);
                }
            } catch (java.io.IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
        }
        return new Channel(
                cvIdx.stream().flatMapToInt(java.util.stream.IntStream::of).toArray(),
                cvIn.stream().flatMapToDouble(java.util.stream.DoubleStream::of).toArray(),
                cvPts.stream().mapToDouble(Double::doubleValue).toArray(), extrapolate,
                cvGenId.toArray(new String[0]),
                cvBusFactor.stream().mapToDouble(Double::doubleValue).toArray(),
                cvHv.stream().anyMatch(h -> h >= 0) ? cvHv.stream().mapToInt(Integer::intValue).toArray() : null);
    }

    /** The bus's index in the distributed-slack BUS pack, or -1 when it does not participate. */
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

    @SuppressWarnings("java:S107")
    /** The generator's index in the distributed-slack GENERATOR pack, or -1 when it is not in it. */
    private static int dsSlotOf(String[] dgGenId, String genId) {
        if (dgGenId == null) {
            return -1;
        }
        for (int i = 0; i < dgGenId.length; i++) {
            if (dgGenId[i].equals(genId)) {
                return i;
            }
        }
        return -1;
    }

    private static void packBusCurves(LfBus bus, int rlIdx, EquationSystem<AcVariableType, AcEquationType> es,
                                      GpuDsParams dsParams, double[] baseTargets, int[] busPRow,
                                      List<int[]> cvIdx, List<double[]> cvIn, List<Double> cvPts,
                                      List<String> cvGenId, List<Double> cvBusFactor, GpuAcData gpuData,
                                      java.util.Map<LfGenerator, int[]> hvdcSides, List<Integer> cvHv) {
        String[] dgGenId = gpuData.dgGenId();
        // dsParams == null: no slack distribution, only the AC-emulation stations' P moves
        double busFactor = dsParams != null ? bus.getGenerators().stream().mapToDouble(dsParams::factor).sum() : 0;
        for (LfGenerator gen : bus.getGenerators()) {
            var curve = gen instanceof AbstractLfGenerator alg
                    ? alg.getReactiveLimits().filter(rl -> rl.getKind() == ReactiveLimitsKind.CURVE).orElse(null)
                    : null;
            int[] hvdcSide = hvdcSides.get(gen);
            if (curve == null || hvdcSide == null && (busFactor <= 0 || dsParams.factor(gen) <= 0)) {
                continue;                                    // no curve, or its P never moves — static is exact
            }
            // A bus limit is NOT the sum of its generators' curve limits. AbstractLfBus.getLimitQ
            // takes a generator's own limit only when it is in VOLTAGE or REMOTE_REACTIVE_POWER
            // control, and its fixed targetQ otherwise - so a machine that is not controlling
            // contributes a CONSTANT to the bus limit and its curve must not move that limit when
            // the distributed slack moves its P. Packing it anyway made the device re-freeze a
            // pinned bus at a limit OLF never computed. Measured on network.xiidm/DJXY: a BATTERY
            // (FQHV, curve points p=-9.45/+9.45) shares bus FQHL_0 with a voltage-controlling
            // generator; the slack moved the battery's P by 0.06 MW, the device moved the bus's
            // MAX_Q by the battery's curve slope (-7.9087e-06 pu, which IIDM's own curve confirms
            // to 10 digits) and OLF did not - 5.166e-05 kV across the network, invariant to convEps.
            if (gen.getGeneratorControlType() != LfGenerator.GeneratorControlType.VOLTAGE
                    && gen.getGeneratorControlType() != LfGenerator.GeneratorControlType.REMOTE_REACTIVE_POWER) {
                // a constant in the bus limit: its P may move, the limit does not
                continue;
            }
            var points = ((ReactiveCapabilityCurve) curve).getPoints().stream()
                    .sorted(java.util.Comparator.comparingDouble(ReactiveCapabilityCurve.Point::getP)).toList();
            int ptsOff = cvPts.size() / 3;
            for (var pt : points) {
                cvPts.add(pt.getP());
                cvPts.add(pt.getMinQ());
                cvPts.add(pt.getMaxQ());
            }
            int pRow = busPRow[bus.getNum()];             // the RELOCATED P-balance row (see build)
            // A slack participant's row is the slack pack's own (dsPhiRow is built from the same busPRow, so
            // this only asserts the two packs agree); the kernel resolves the per-scenario row (dsRowOv) itself.
            int dsIdx = dsBusIndexOf(gpuData, bus.getNum());
            if (dsIdx >= 0) {
                pRow = gpuData.dsPhiRow()[dsIdx];
            }
            // ...and WHICH generator this is in the distributed-slack generator pack. The kernel reads
            // that generator's own cumulative P movement (genCum) instead of splitting its bus's
            // movement by a fixed share: the share is exact only while the intra-bus split stays
            // proportional, and OLF's water-fill clamps PER GENERATOR, which breaks it the moment one
            // generator on the bus saturates. -1 when the generator is not in the pack (then the share
            // reconstruction is the only thing available).
            if (hvdcSide != null) {
                // its P is its link's (the kernel reads it from the HVDC loop state): no slack slot, no share
                cvIdx.add(new int[] {rlIdx, pRow, ptsOff, points.size(), -1});
                cvIn.add(new double[] {gen.getTargetP(), 0, baseTargets[pRow], gen.getMinQ(), gen.getMaxQ()});
                cvHv.add(hvdcSide[0]);
                cvHv.add(hvdcSide[1]);
                cvGenId.add(gen.getId());
                cvBusFactor.add(busFactor);
                continue;
            }
            cvIdx.add(new int[] {rlIdx, pRow, ptsOff, points.size(), dsSlotOf(dgGenId, gen.getId())});
            cvIn.add(new double[] {gen.getTargetP(), dsParams.factor(gen) / busFactor, baseTargets[pRow],
                gen.getMinQ(), gen.getMaxQ()});              // base evals AT base targetP (bit-zero delta at base)
            cvHv.add(-1);
            cvHv.add(-1);
            // host-side only: which generator this entry is, and the denominator its share was built on, so a
            // scenario losing a sibling can renormalize it
            cvGenId.add(gen.getId());
            cvBusFactor.add(busFactor);
        }
    }
}
