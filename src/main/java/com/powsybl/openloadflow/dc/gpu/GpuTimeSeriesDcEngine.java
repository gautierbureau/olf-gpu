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

    private final DcLoadFlowContext context;
    private final LfNetwork network;
    private final GpuDcAssembly assembly;
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
                    List<?> needed = dcParameters.getOuterLoops().stream().filter(o -> o.isNeeded(context)).toList();
                    if (!needed.isEmpty()) {
                        LOGGER.info("GPU time-series DC: {} outer loop(s) needed on this network, stock engine", needed.size());
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
                chunkGeneratorMinP, chunkGeneratorMaxP, chunkGeneratorDroop, chunkGeneratorParticipationFactor,
                distributedSlack ? participationType : -1, useActiveLimits, assembly.referenceGenerator,
                chunkStates, chunkGeneratorTargets, chunkDistributed, chunkMismatch);
        LAST_DEVICE_STEP_COUNT.addAndGet(count);
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
