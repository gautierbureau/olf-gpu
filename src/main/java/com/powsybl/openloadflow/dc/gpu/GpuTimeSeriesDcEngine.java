/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.dc.gpu;

import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.LoadFlowResult;
import com.powsybl.math.matrix.MatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.dc.DcLoadFlowContext;
import com.powsybl.openloadflow.dc.DcLoadFlowEngine;
import com.powsybl.openloadflow.dc.DcLoadFlowParameters;
import com.powsybl.openloadflow.graph.GraphConnectivityFactory;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfHvdc;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfNetworkParameters;
import com.powsybl.openloadflow.network.LoadFlowModel;
import com.powsybl.openloadflow.network.NetworkState;
import com.powsybl.openloadflow.network.util.ActivePowerDistribution;
import com.powsybl.openloadflow.ts.TimeSeriesLoadFlow;
import com.powsybl.openloadflow.ts.TimeSeriesLoadFlowParameters;
import com.powsybl.iidm.network.Network;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The DC time-series step engine on the device: a chunk of steps is one factorization against many
 * right-hand sides.
 *
 * <p>This is the case the DC port was aimed at. A step is an independent DC load flow over an
 * unchanged topology, so every step of a chunk shares the same matrix: the device fills one target
 * column per step, distributes the slack for every step in parallel (one thread per step, each
 * replaying {@code ActivePowerDistribution} on its own copy of the generator targets), solves the
 * whole chunk in one multi-right-hand-side solve, and keeps the states for the host to read back
 * step by step.
 *
 * <p>Measured on rte6515 before any of this was written, a CPU step costs 10.6 ms: 3.9 of restore
 * and set points, 2.3 of solve, 4.4 of emitting the result rows. The device can take the middle
 * one, and — through {@link #restoreState} — most of the first: a DC engine with no outer loop
 * moves only the generator targets it distributes over and the phase shift of branches carrying an
 * angle variable, so putting those two back is enough, instead of restoring the whole network. The
 * last one, the row emission, is the host's and stays: it is the writer's own cost.
 *
 * <p>Falls back to OLF's engine, step by step, when the run configures DC outer loops (phase control
 * or HVDC AC-emulation limits): those mutate taps and statuses, which the cheap restore does not
 * undo, and they make the steps of a chunk stop sharing a matrix.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuTimeSeriesDcEngine implements TimeSeriesLoadFlow.StepEngine {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuTimeSeriesDcEngine.class);

    /** Steps solved together; the device holds one state column per step of the chunk. */
    public static final int DEFAULT_CHUNK = Integer.getInteger("olf.gpu.dc.ts.chunk", 256);

    /** The last run's steps solved on the device (-1 when the device never engaged). */
    public static final AtomicInteger LAST_DEVICE_STEP_COUNT = new AtomicInteger(-1);

    /** The last run's AC-emulation configurations applied and re-solved (0 = no link ever switched). */
    public static final AtomicInteger LAST_DEVICE_LOOP_REGROUPINGS = new AtomicInteger(0);

    private final DcLoadFlowContext context;
    private final LfNetwork network;
    private GpuDcAssembly assembly;
    private final GpuDcSolver solver;
    private final int chunk;
    private final int participationType;
    private final boolean distributedSlack;
    private final boolean useActiveLimits;

    // the network as loaded: what a step starts from
    private final List<LfGenerator> generators;
    private final double[] baseGeneratorTargetP;
    private final List<LfBranch> alphaBranches;
    private final double[] baseAlpha;

    // per-chunk host buffers
    private final double[] chunkGeneratorTargets;
    private final double[] chunkLoadTargets;
    // the distribution reads all of these, and a plan series changes them per step
    private final double[] chunkGeneratorInitial;
    private final int[] chunkGeneratorParticipating;
    private final double[] chunkGeneratorMinTargetP;
    private final double[] chunkGeneratorMaxTargetP;
    private final double[] chunkGeneratorMinP;
    private final double[] chunkGeneratorMaxP;
    private final double[] chunkGeneratorDroop;
    private final double[] chunkGeneratorParticipationFactor;
    private double[] chunkStates;
    private double[] chunkDistributed;
    private double[] chunkMismatch;
    private int chunkCount;

    // the AC-emulation limits outer loop, driven over the whole chunk
    private final List<LfHvdc> hvdcs;
    private final int[] hvdcStation1Generator;
    private final int[] hvdcStation2Generator;
    private final int[] baseHvdcStatus;
    private final int[] chunkHvdcStatus;
    private final double[] chunkHvdcP0;
    private final int maxOuterLoopIterations;
    private int[] currentConfiguration;

    private GpuTimeSeriesDcEngine(DcLoadFlowContext context, GpuDcAssembly assembly, GpuDcSolver solver, int chunk) {
        this.context = context;
        this.network = context.getNetwork();
        this.assembly = assembly;
        this.solver = solver;
        this.chunk = chunk;
        DcLoadFlowParameters parameters = context.getParameters();
        this.distributedSlack = parameters.isDistributedSlack();
        this.useActiveLimits = parameters.getNetworkParameters().isUseActiveLimits();
        this.participationType = GpuDcLoadFlowEngine.participationType(parameters.getBalanceType());
        this.generators = assembly.generators();
        this.baseGeneratorTargetP = new double[generators.size()];
        for (int i = 0; i < baseGeneratorTargetP.length; i++) {
            baseGeneratorTargetP[i] = generators.get(i).getTargetP();
        }
        this.alphaBranches = network.getBranches().stream()
                .filter(branch -> assembly.a1Row[branch.getNum()] >= 0)
                .toList();
        this.baseAlpha = new double[alphaBranches.size()];
        for (int i = 0; i < baseAlpha.length; i++) {
            baseAlpha[i] = alphaBranches.get(i).getPiModel().getA1();
        }
        int genSlots = chunk * Math.max(generators.size(), 1);
        this.chunkGeneratorTargets = new double[genSlots];
        this.chunkLoadTargets = new double[chunk * Math.max(network.getBuses().size(), 1)];
        this.chunkGeneratorInitial = new double[genSlots];
        this.chunkGeneratorParticipating = new int[genSlots];
        this.chunkGeneratorMinTargetP = new double[genSlots];
        this.chunkGeneratorMaxTargetP = new double[genSlots];
        this.chunkGeneratorMinP = new double[genSlots];
        this.chunkGeneratorMaxP = new double[genSlots];
        this.chunkGeneratorDroop = new double[genSlots];
        this.chunkGeneratorParticipationFactor = new double[genSlots];
        this.maxOuterLoopIterations = parameters.getMaxOuterLoopIterations();

        // the AC-emulation links the loop looks at, and where their converter stations sit in the
        // generator tables: a saturation retargets them, which moves the injections of that step
        this.hvdcs = network.getHvdcs().stream()
                .filter(hvdc -> hvdc.isAcEmulation() && !hvdc.isDisabled()
                        && hvdc.getBus1() != null && hvdc.getBus2() != null
                        && !hvdc.getBus1().isDisabled() && !hvdc.getBus2().isDisabled())
                .toList();
        this.hvdcStation1Generator = new int[hvdcs.size()];
        this.hvdcStation2Generator = new int[hvdcs.size()];
        this.baseHvdcStatus = new int[hvdcs.size()];
        for (int h = 0; h < hvdcs.size(); h++) {
            LfHvdc hvdc = hvdcs.get(h);
            hvdcStation1Generator[h] = generators.indexOf(hvdc.getConverterStation1());
            hvdcStation2Generator[h] = generators.indexOf(hvdc.getConverterStation2());
            baseHvdcStatus[h] = statusCode(hvdc.getAcEmulationControl().getAcEmulationStatus());
        }
        this.chunkHvdcStatus = new int[chunk * Math.max(hvdcs.size(), 1)];
        this.chunkHvdcP0 = new double[chunk * Math.max(hvdcs.size(), 1)];
        this.currentConfiguration = baseHvdcStatus.clone();
        if (!hvdcs.isEmpty()) {
            uploadHvdcTables();
        }
    }

    private static int statusCode(LfHvdc.AcEmulationControl.AcEmulationStatus status) {
        return switch (status) {
            case LINEAR_MODE -> 0;
            case SATURATION_MODE_FROM_CS1_TO_CS2 -> 1;
            case SATURATION_MODE_FROM_CS2_TO_CS1 -> 2;
            default -> 3;
        };
    }

    /** The constant part of the link tables; the status and the offset are per step. */
    private void uploadHvdcTables() {
        int m = hvdcs.size();
        int[] ph1 = new int[m];
        int[] ph2 = new int[m];
        double[] k = new double[m];
        double[] p0 = new double[m];
        double[] pMax12 = new double[m];
        double[] pMax21 = new double[m];
        int[] status = new int[m];
        int[] active = new int[m];
        for (int h = 0; h < m; h++) {
            LfHvdc hvdc = hvdcs.get(h);
            var control = hvdc.getAcEmulationControl();
            ph1[h] = assembly.busPhRow[hvdc.getBus1().getNum()];
            ph2[h] = assembly.busPhRow[hvdc.getBus2().getNum()];
            k[h] = control.getDroop() * 180 / Math.PI;
            p0[h] = control.getP0();
            pMax12[h] = control.getPMaxFromCS1toCS2();
            pMax21[h] = control.getPMaxFromCS2toCS1();
            status[h] = baseHvdcStatus[h];
            active[h] = ph1[h] >= 0 && ph2[h] >= 0 ? 1 : 0;
        }
        solver.setHvdcs(ph1, ph2, k, p0, pMax12, pMax21, status, active);
    }

    /**
     * The engine plan provider that installs this engine for DC runs, falling back to OLF's own
     * engine without a device, for AC, and when DC outer loops are configured.
     */
    public static TimeSeriesLoadFlow.EnginePlanProvider planProvider() {
        return (network, parameters, matrixFactory, connectivityFactory) -> {
            LoadFlowParameters lfParameters = parameters.getLoadFlowParameters();
            TimeSeriesLoadFlow.EnginePlan fallback = TimeSeriesLoadFlow.EnginePlanProvider.DEFAULT
                    .create(network, parameters, matrixFactory, connectivityFactory);
            if (!lfParameters.isDc() || !GpuDcSolver.isAvailable()) {
                LAST_DEVICE_STEP_COUNT.set(-1);
                return fallback;
            }
            return devicePlan(network, parameters, matrixFactory, connectivityFactory, fallback);
        };
    }

    private static TimeSeriesLoadFlow.EnginePlan devicePlan(Network network, TimeSeriesLoadFlowParameters parameters,
                                                            MatrixFactory matrixFactory,
                                                            GraphConnectivityFactory<LfBus, LfBranch> connectivityFactory,
                                                            TimeSeriesLoadFlow.EnginePlan fallback) {
        LoadFlowParameters lfParameters = parameters.getLoadFlowParameters();
        OpenLoadFlowParameters parametersExt = OpenLoadFlowParameters.get(lfParameters);
        DcLoadFlowParameters dcParameters = OpenLoadFlowParameters.createDcParameters(network, lfParameters,
                parametersExt, matrixFactory, connectivityFactory, false);
        if (GpuDcLoadFlowEngine.participationType(dcParameters.getBalanceType()) < 0 && dcParameters.isDistributedSlack()) {
            LOGGER.info("GPU time-series DC: balance type {} is not modelled on the device, stock engine", dcParameters.getBalanceType());
            LAST_DEVICE_STEP_COUNT.set(-1);
            return fallback;
        }
        int chunk = Math.max(1, DEFAULT_CHUNK);
        return new TimeSeriesLoadFlow.EnginePlan() {
            @Override
            public LfNetworkParameters networkParameters() {
                return dcParameters.getNetworkParameters();
            }

            @Override
            public LoadFlowModel loadFlowModel() {
                return LoadFlowModel.DC;
            }

            @Override
            public TimeSeriesLoadFlow.StepEngine createEngine(LfNetwork lfNetwork) {
                DcLoadFlowContext context = new DcLoadFlowContext(lfNetwork, dcParameters);
                try {
                    // the outer loops OLF would actually run on this network: the configured list is
                    // filtered by isNeeded, and the default list carries the HVDC AC-emulation loop
                    // whether or not the network has an AC-emulation link
                    // the AC-emulation limits loop is driven over the chunk (below); any other loop
                    // still sends the run to the stock engine
                    List<?> needed = dcParameters.getOuterLoops().stream()
                            .filter(o -> o.isNeeded(context))
                            .filter(o -> !(o instanceof com.powsybl.openloadflow.dc.DcHvdcAcEmulationLimitsOuterLoop))
                            .toList();
                    if (!needed.isEmpty()) {
                        LOGGER.info("GPU time-series DC: {} outer loop(s) with no device model on this network, stock engine", needed.size());
                        context.close();
                        LAST_DEVICE_STEP_COUNT.set(-1);
                        return fallback.createEngine(lfNetwork);
                    }
                    GpuDcAssembly assembly = GpuDcAssembly.capture(lfNetwork, context.getEquationSystem(),
                            dcParameters.getEquationSystemCreationParameters());
                    GpuDcSolver solver = GpuDcSolver.createAssembled(assembly);
                    LAST_DEVICE_STEP_COUNT.set(0);
                    return new GpuTimeSeriesDcEngine(context, assembly, solver, chunk);
                } catch (UnsupportedOperationException e) {
                    LOGGER.warn("GPU time-series DC: {} — stock engine", e.getMessage());
                    context.close();
                    LAST_DEVICE_STEP_COUNT.set(-1);
                    return fallback.createEngine(lfNetwork);
                }
            }
        };
    }

    @Override
    public int batchSize() {
        return chunk;
    }

    /**
     * What a DC step without outer loops dirties, and only that: the generator targets the slack
     * distribution moved, and the phase shift {@code updateNetwork} wrote back into the branches
     * carrying an angle variable. Restoring the whole network here costs 3.9 ms a step on rte6515,
     * against the 0.04 ms this takes.
     */
    @Override
    public void restoreState(NetworkState baseState) {
        for (int i = 0; i < baseGeneratorTargetP.length; i++) {
            generators.get(i).setTargetP(baseGeneratorTargetP[i]);
        }
        for (int i = 0; i < baseAlpha.length; i++) {
            alphaBranches.get(i).getPiModel().setA1(baseAlpha[i]);
        }
        applyConfiguration(baseHvdcStatus);
    }

    /** Capture the step's injections: the generator targets and the buses' load target P. */
    @Override
    public void prepare(int slot) {
        int generatorCount = generators.size();
        for (int i = 0; i < generatorCount; i++) {
            LfGenerator generator = generators.get(i);
            int at = slot * generatorCount + i;
            chunkGeneratorTargets[at] = generator.getTargetP();
            chunkGeneratorInitial[at] = generator.getInitialTargetP();
            chunkGeneratorParticipating[at] = generator.isParticipating() ? 1 : 0;
            chunkGeneratorMinTargetP[at] = generator.getMinTargetP();
            chunkGeneratorMaxTargetP[at] = generator.getMaxTargetP();
            chunkGeneratorMinP[at] = generator.getMinP();
            chunkGeneratorMaxP[at] = generator.getMaxP();
            chunkGeneratorDroop[at] = generator.getDroop();
            chunkGeneratorParticipationFactor[at] = generator.getParticipationFactor();
        }
        List<LfBus> buses = network.getBuses();
        for (int b = 0; b < buses.size(); b++) {
            chunkLoadTargets[slot * buses.size() + b] = buses.get(b).getLoadTargetP();
        }
        int hvdcCount = hvdcs.size();
        for (int h = 0; h < hvdcCount; h++) {
            // a plan series can schedule the AC-emulation offset, so it is captured per step; the
            // status every step starts from is the one the network was loaded with
            chunkHvdcP0[slot * hvdcCount + h] = hvdcs.get(h).getAcEmulationControl().getP0();
            chunkHvdcStatus[slot * hvdcCount + h] = baseHvdcStatus[h];
        }
    }

    /**
     * The chunk on the device: per step the slack distribution and the target vector, then one
     * multi-right-hand-side solve for every step at once.
     */
    @Override
    public void solveBatch(int count) {
        chunkCount = count;
        int generatorCount = generators.size();
        int busCount = network.getBuses().size();
        if (chunkStates == null || chunkStates.length < (long) assembly.size() * count) {
            chunkStates = new double[assembly.size() * count];
            chunkDistributed = new double[count];
            chunkMismatch = new double[count];
        }
        solver.timeSeriesChunk(count, chunkGeneratorTargets, generatorCount, chunkLoadTargets, busCount,
                chunkGeneratorInitial, chunkGeneratorParticipating, chunkGeneratorMinTargetP, chunkGeneratorMaxTargetP,
                chunkGeneratorMinP, chunkGeneratorMaxP, chunkGeneratorDroop, chunkGeneratorParticipationFactor, chunkHvdcP0,
                distributedSlack ? participationType : -1, useActiveLimits, assembly.referenceGenerator,
                chunkStates, chunkGeneratorTargets, chunkDistributed, chunkMismatch);
        runHvdcLoop(count);
        LAST_DEVICE_STEP_COUNT.addAndGet(count);
    }

    /**
     * {@code DcHvdcAcEmulationLimitsOuterLoop} over the whole chunk: every step's decision is taken
     * on its own state by a kernel, and the steps that changed are regrouped by the configuration
     * they now ask for. A configuration is applied to the network with OLF's own mutators, the
     * device matrix is refilled and refactorized for it — 12 ms on rte6515, against 580 for the
     * analysis, which is why grouping beats a per-step Woodbury correction here — and that group's
     * steps are re-solved together. Repeats until no step changes, within OLF's outer-loop budget.
     */
    private void runHvdcLoop(int count) {
        int hvdcCount = hvdcs.size();
        if (hvdcCount == 0) {
            return;
        }
        int[] status = java.util.Arrays.copyOf(chunkHvdcStatus, count * hvdcCount);
        double[] offsets = java.util.Arrays.copyOf(chunkHvdcP0, count * hvdcCount);
        for (int round = 0; round < maxOuterLoopIterations; round++) {
            int[] decisions = solver.timeSeriesHvdcCheck(count, status, offsets);
            java.util.Map<String, List<Integer>> groups = new java.util.LinkedHashMap<>();
            for (int step = 0; step < count; step++) {
                boolean changed = false;
                for (int h = 0; h < hvdcCount; h++) {
                    int decision = decisions[step * hvdcCount + h];
                    if (decision >= 0) {
                        status[step * hvdcCount + h] = decision;
                        changed = true;
                    }
                }
                if (changed) {
                    StringBuilder key = new StringBuilder();
                    for (int h = 0; h < hvdcCount; h++) {
                        key.append(status[step * hvdcCount + h]).append(',');
                    }
                    groups.computeIfAbsent(key.toString(), k -> new java.util.ArrayList<>()).add(step);
                }
            }
            if (groups.isEmpty()) {
                break;
            }
            LAST_DEVICE_LOOP_REGROUPINGS.addAndGet(groups.size());
            for (List<Integer> group : groups.values()) {
                int[] configuration = new int[hvdcCount];
                int first = group.get(0);
                System.arraycopy(status, first * hvdcCount, configuration, 0, hvdcCount);
                applyConfiguration(configuration);
                refreshDeviceMatrix();
                // the converter stations of a saturating link are retargeted: that is an injection
                // change, and it belongs to the steps of this group only
                int generatorCount = generators.size();
                for (int step : group) {
                    for (int h = 0; h < hvdcCount; h++) {
                        if (hvdcStation1Generator[h] >= 0) {
                            chunkGeneratorTargets[step * generatorCount + hvdcStation1Generator[h]] =
                                    hvdcs.get(h).getConverterStation1().getTargetP();
                        }
                        if (hvdcStation2Generator[h] >= 0) {
                            chunkGeneratorTargets[step * generatorCount + hvdcStation2Generator[h]] =
                                    hvdcs.get(h).getConverterStation2().getTargetP();
                        }
                    }
                }
                int[] steps = group.stream().mapToInt(Integer::intValue).toArray();
                solver.timeSeriesResolve(steps, chunkGeneratorTargets, generatorCount, chunkLoadTargets,
                        network.getBuses().size(), chunkStates);
            }
        }
        System.arraycopy(status, 0, chunkHvdcStatus, 0, count * hvdcCount);
        // the reported mismatch is the one the network ends on, and the loop moved injections
        solver.timeSeriesMismatch(count, chunkMismatch);
    }

    /** Put the links in the given AC-emulation configuration, with OLF's own mutators. */
    private void applyConfiguration(int[] configuration) {
        if (hvdcs.isEmpty() || java.util.Arrays.equals(configuration, currentConfiguration)) {
            return;
        }
        for (int h = 0; h < hvdcs.size(); h++) {
            if (configuration[h] == currentConfiguration[h]) {
                continue;
            }
            var control = hvdcs.get(h).getAcEmulationControl();
            switch (configuration[h]) {
                case 0 -> control.switchToLinearMode();
                case 1 -> control.switchToSaturationFromCS1toCS2(false);
                case 2 -> control.switchToSaturationFromCS2toCS1(false);
                default -> throw new IllegalStateException("unexpected AC emulation status " + configuration[h]);
            }
        }
        currentConfiguration = configuration.clone();
    }

    /** Refill the device matrix for the configuration the network now carries, and refactorize. */
    private void refreshDeviceMatrix() {
        GpuDcAssembly refreshed = GpuDcAssembly.capture(network, context.getEquationSystem(),
                context.getParameters().getEquationSystemCreationParameters());
        if (!assembly.samePattern(refreshed)) {
            throw new IllegalStateException("the AC emulation switch changed the matrix pattern");
        }
        if (!assembly.sameStructure(refreshed)) {
            solver.setContributions(refreshed);
        }
        solver.refreshBranches(refreshed);
        assembly = refreshed;
    }

    /**
     * Make the step's solution current on the network, so that the results are emitted from the
     * same places a CPU step emits them from: the state vector (branch flows and bus angles) and the
     * generator targets after distribution.
     */
    @Override
    public TimeSeriesLoadFlow.SolveResult apply(int slot) {
        int n = assembly.size();
        double[] state = new double[n];
        System.arraycopy(chunkStates, slot * n, state, 0, n);
        int generatorCount = generators.size();
        for (int i = 0; i < generatorCount; i++) {
            LfGenerator generator = generators.get(i);
            int at = slot * generatorCount + i;
            // both of the values a generator row carries: the target the plan asked for (which the
            // set point writes into the initial target) and the one the distribution landed on
            generator.setTargetP(chunkGeneratorTargets[at]);
            generator.setInitialTargetP(chunkGeneratorInitial[at]);
        }
        context.getEquationSystem().getStateVector().set(state);
        DcLoadFlowEngine.updateNetwork(network, context.getEquationSystem(), state);
        if (context.getParameters().isSetVToNan()) {
            for (LfBus bus : network.getBuses()) {
                bus.setV(Double.NaN);
            }
        }
        return new TimeSeriesLoadFlow.SolveResult(LoadFlowResult.ComponentResult.Status.CONVERGED,
                chunkMismatch[slot], chunkDistributed[slot]);
    }

    /** Not used: this engine always batches. */
    @Override
    public TimeSeriesLoadFlow.SolveResult solve() {
        prepare(0);
        solveBatch(1);
        return apply(0);
    }

    @Override
    public void close() {
        solver.close();
        context.close();
    }

    /** For the golden test: the distribution epsilon OLF applies before distributing at all. */
    public static double residueEps() {
        return ActivePowerDistribution.P_RESIDUE_EPS;
    }
}
