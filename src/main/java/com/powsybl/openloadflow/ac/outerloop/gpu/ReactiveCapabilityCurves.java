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
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDistributedSlackParameters;
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
 * ({@code reactiveCapabilityCurveLimitsBatch}) re-evaluates the bus limits at the CURRENT targets before every RL
 * decision. Non-participating curve gens never move — their base snapshot stays exact, no entry.
 *
 * <p>Channel layout (setBatchReactiveCapabilityCurves): reactiveCapabilityCurveIndices stride 5 {reactiveLimitEntry,
         pRow, ptsOff, npts, distributedSlackGeneratorIndex}; reactiveCapabilityCurveParameters stride 5
 * {basePpu, share, baseRowTargetP, baseEvalMinPu, baseEvalMaxPu}; reactiveCapabilityCurvePoints stride 3
 * {p_MW, minQ_MW, maxQ_MW} sorted by p; reactiveCapabilityCurveHvdcStation stride 2 {hvdc, side} (null when no entry is one).
 *
 * <p>An AC-emulation VSC station is an entry too, whatever the slack distribution: OLF evaluates its curve at the
 * live flow of its link in the linear mode, and at its target P otherwise
 * (LfVscConverterStationImpl.getReactiveLimitsActivePower). Its {hvdc, side} points the kernel at the link's state
 * in the HVDC loop tables (the same enumeration as GpuHvdcAcEmulationLimitsOuterLoop.buildControllers).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class ReactiveCapabilityCurves {

    private ReactiveCapabilityCurves() {
    }

    /** The packed channel, null when the network needs none (no participating curve gen).
     *  {@code extrapolate} = OLF's extrapolateReactiveLimits: the kernel continues the boundary
     *  two-point slope past [minP, maxP] instead of clamping (the base evals in reactiveCapabilityCurveParameters come from
     *  OLF's own gen.getMinQ()/getMaxQ(), which already honor the flag). */
    public record Channel(int[] reactiveCapabilityCurveIndices, double[] reactiveCapabilityCurveParameters, double[] reactiveCapabilityCurvePoints, boolean extrapolate,
                         String[] reactiveCapabilityCurveGeneratorId, double[] reactiveCapabilityCurveBusFactor, int[] reactiveCapabilityCurveHvdcStation) {

        /** Whether entry {@code e} is an AC-emulation VSC: its P is its link's, not the slack's. */
        public boolean isHvdc(int e) {
            return reactiveCapabilityCurveHvdcStation != null && reactiveCapabilityCurveHvdcStation[2 * e] >= 0;
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
     *  {@code gpuData.reactiveLimits().voltageRow()} — a drift throws, falling back to CPU). Returns null when no
     *  PARTICIPATING curve generator exists (static snapshots stay exact). */
    public static Channel build(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                GpuAcData gpuData, GpuDistributedSlackParameters distributedSlackParameters, boolean distributedSlack,
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
        List<Integer> reactiveCapabilityCurveHvdcStation = new ArrayList<>();
        List<int[]> reactiveCapabilityCurveIndices = new ArrayList<>();
        List<double[]> reactiveCapabilityCurveParameters = new ArrayList<>();
        List<Double> reactiveCapabilityCurvePoints = new ArrayList<>();
        List<String> reactiveCapabilityCurveGeneratorId = new ArrayList<>();
        List<Double> reactiveCapabilityCurveBusFactor = new ArrayList<>();
        // Each bus's P-BALANCE row, from the one source of truth. Resolving it from the variable set
        // instead gives the bus's own BUS_PHI column, which is NOT where its P target lives once the
        // coupler-tree relocation has moved it onto the parent coupler's DUMMY_P — and this channel
        // uses the row twice, as a kernel row index and as a baseTargets[] index, so a stale row
        // evaluates every curve on this bus at the wrong active power.
        int[] busPRow = com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor
                .busPowerRows(network, es)[0];
        int reactiveLimitEntry = 0;
        for (LfBus bus : network.getBuses()) {
            int bn = bus.getNum();
            var vVar = es.getVariableSet().getVariable(bn, AcVariableType.BUS_V);
            // REPLAY the RL pack's enumeration exactly (ReactiveLimitsPacker.reactiveLimitsPack):
            // a remote group member with finite limits is an reactiveLimitVoltageControlRole entry regardless of its base
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
                if (reactiveLimitEntry >= gpuData.reactiveLimits().voltageRow().length || gpuData.reactiveLimits().voltageRow()[reactiveLimitEntry] != vVar.getRow()) {
                    throw new PowsyblException("GPU AC solver: RL-pack enumeration drift while packing curve limits (bus "
                            + bus.getId() + ")");
                }
                // remote members keep STATIC limits in the channel (a curve gen at a remote member is
                // possible but its own P participation is the same as a local bus's) — fall through to
                // the same per-generator curve packing below.
                packBusCurves(bus, reactiveLimitEntry, es, distributedSlack ? distributedSlackParameters : null,
                        baseTargets, busPRow, reactiveCapabilityCurveIndices, reactiveCapabilityCurveParameters,
                        reactiveCapabilityCurvePoints,
                        reactiveCapabilityCurveGeneratorId, reactiveCapabilityCurveBusFactor, gpuData, hvdcSides, reactiveCapabilityCurveHvdcStation);
                reactiveLimitEntry++;
                continue;
            }
            boolean vActive = es.getEquation(bn, AcEquationType.BUS_TARGET_V).map(e -> e.isActive()).orElse(false);
            boolean qActive = es.getEquation(bn, AcEquationType.BUS_TARGET_Q).map(e -> e.isActive()).orElse(false);
            boolean basePv = vActive && !qActive;
            boolean basePqCapable = qActive && !vActive && !bus.isGeneratorVoltageControlEnabled();
            if (vVar == null || vVar.getRow() < 0 || bus.getGeneratorVoltageControl().isEmpty() || !(basePv || basePqCapable)) {
                continue;
            }
            if (reactiveLimitEntry >= gpuData.reactiveLimits().voltageRow().length || gpuData.reactiveLimits().voltageRow()[reactiveLimitEntry] != vVar.getRow()) {
                throw new PowsyblException("GPU AC solver: RL-pack enumeration drift while packing curve limits (bus "
                        + bus.getId() + ")");
            }
            packBusCurves(bus, reactiveLimitEntry, es, distributedSlack ? distributedSlackParameters : null,
                    baseTargets, busPRow, reactiveCapabilityCurveIndices, reactiveCapabilityCurveParameters,
                    reactiveCapabilityCurvePoints,
                    reactiveCapabilityCurveGeneratorId, reactiveCapabilityCurveBusFactor, gpuData, hvdcSides, reactiveCapabilityCurveHvdcStation);
            reactiveLimitEntry++;
        }
        if (reactiveCapabilityCurveIndices.isEmpty()) {
            return null;
        }
        return new Channel(
                reactiveCapabilityCurveIndices.stream().flatMapToInt(java.util.stream.IntStream::of).toArray(),
                reactiveCapabilityCurveParameters.stream().flatMapToDouble(java.util.stream.DoubleStream::of).toArray(),
                reactiveCapabilityCurvePoints.stream().mapToDouble(Double::doubleValue).toArray(), extrapolate,
                reactiveCapabilityCurveGeneratorId.toArray(new String[0]),
                reactiveCapabilityCurveBusFactor.stream().mapToDouble(Double::doubleValue).toArray(),
                reactiveCapabilityCurveHvdcStation.stream().anyMatch(h -> h >= 0) ? reactiveCapabilityCurveHvdcStation.stream().mapToInt(Integer::intValue).toArray() : null);
    }

    /** The bus's index in the distributed-slack BUS pack, or -1 when it does not participate. */
    private static int distributedSlackBusIndexOf(GpuAcData gpuData, int busNum) {
        int[] nums = gpuData.distributedSlack().busNum();
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
    private static int distributedSlackGeneratorIndexOf(String[] distributedSlackGeneratorId, String genId) {
        if (distributedSlackGeneratorId == null) {
            return -1;
        }
        for (int i = 0; i < distributedSlackGeneratorId.length; i++) {
            if (distributedSlackGeneratorId[i].equals(genId)) {
                return i;
            }
        }
        return -1;
    }

    private static void packBusCurves(LfBus bus, int reactiveLimitEntry, EquationSystem<AcVariableType, AcEquationType> es,
                                      GpuDistributedSlackParameters distributedSlackParameters, double[] baseTargets, int[] busPRow,
                                      List<int[]> reactiveCapabilityCurveIndices, List<double[]> reactiveCapabilityCurveParameters, List<Double> reactiveCapabilityCurvePoints,
                                      List<String> reactiveCapabilityCurveGeneratorId, List<Double> reactiveCapabilityCurveBusFactor, GpuAcData gpuData,
                                      java.util.Map<LfGenerator, int[]> hvdcSides, List<Integer> reactiveCapabilityCurveHvdcStation) {
        String[] distributedSlackGeneratorId = gpuData.distributedSlack().generatorId();
        // distributedSlackParameters == null: no slack distribution, only the AC-emulation stations' P moves
        double busFactor = distributedSlackParameters != null ? bus.getGenerators().stream().mapToDouble(distributedSlackParameters::factor).sum() : 0;
        for (LfGenerator gen : bus.getGenerators()) {
            var curve = gen instanceof AbstractLfGenerator alg
                    ? alg.getReactiveLimits().filter(rl -> rl.getKind() == ReactiveLimitsKind.CURVE).orElse(null)
                    : null;
            int[] hvdcSide = hvdcSides.get(gen);
            if (curve == null || hvdcSide == null && (busFactor <= 0 || distributedSlackParameters.factor(gen) <= 0)) {
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
            int ptsOff = reactiveCapabilityCurvePoints.size() / 3;
            for (var pt : points) {
                reactiveCapabilityCurvePoints.add(pt.getP());
                reactiveCapabilityCurvePoints.add(pt.getMinQ());
                reactiveCapabilityCurvePoints.add(pt.getMaxQ());
            }
            int pRow = busPRow[bus.getNum()];             // the RELOCATED P-balance row (see build)
            // A slack participant's row is the slack pack's own (distributedSlackBusPhiRow is built from the same busPRow, so
            // this only asserts the two packs agree); the kernel resolves the per-scenario row (distributedSlackBusRowOverride) itself.
            int distributedSlackBusIndex = distributedSlackBusIndexOf(gpuData, bus.getNum());
            if (distributedSlackBusIndex >= 0) {
                pRow = gpuData.distributedSlack().busPhiRow()[distributedSlackBusIndex];
            }
            // ...and WHICH generator this is in the distributed-slack generator pack. The kernel reads
            // that generator's own cumulative P movement (distributedSlackGeneratorMove) instead of splitting its bus's
            // movement by a fixed share: the share is exact only while the intra-bus split stays
            // proportional, and OLF's water-fill clamps PER GENERATOR, which breaks it the moment one
            // generator on the bus saturates. -1 when the generator is not in the pack (then the share
            // reconstruction is the only thing available).
            if (hvdcSide != null) {
                // its P is its link's (the kernel reads it from the HVDC loop state): no slack slot, no share
                reactiveCapabilityCurveIndices.add(new int[] {reactiveLimitEntry, pRow, ptsOff, points.size(), -1});
                reactiveCapabilityCurveParameters.add(new double[] {gen.getTargetP(), 0, baseTargets[pRow], gen.getMinQ(), gen.getMaxQ()});
                reactiveCapabilityCurveHvdcStation.add(hvdcSide[0]);
                reactiveCapabilityCurveHvdcStation.add(hvdcSide[1]);
                reactiveCapabilityCurveGeneratorId.add(gen.getId());
                reactiveCapabilityCurveBusFactor.add(busFactor);
                continue;
            }
            reactiveCapabilityCurveIndices.add(new int[] {reactiveLimitEntry, pRow, ptsOff, points.size(), distributedSlackGeneratorIndexOf(distributedSlackGeneratorId, gen.getId())});
            reactiveCapabilityCurveParameters.add(new double[] {gen.getTargetP(), distributedSlackParameters.factor(gen) / busFactor, baseTargets[pRow],
                gen.getMinQ(), gen.getMaxQ()});              // base evals AT base targetP (bit-zero delta at base)
            reactiveCapabilityCurveHvdcStation.add(-1);
            reactiveCapabilityCurveHvdcStation.add(-1);
            // host-side only: which generator this entry is, and the denominator its share was built on, so a
            // scenario losing a sibling can renormalize it
            reactiveCapabilityCurveGeneratorId.add(gen.getId());
            reactiveCapabilityCurveBusFactor.add(busFactor);
        }
    }
}
