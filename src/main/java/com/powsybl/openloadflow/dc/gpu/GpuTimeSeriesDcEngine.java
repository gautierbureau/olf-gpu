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
import com.powsybl.openloadflow.network.PiModelArray;
import com.powsybl.openloadflow.network.TransformerPhaseControl;
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
    public static final int DEFAULT_CHUNK = Integer.getInteger("olf.gpu.dc.ts.chunk", 1024);

    /** The last run's steps solved on the device (-1 when the device never engaged). */
    public static final AtomicInteger LAST_DEVICE_STEP_COUNT = new AtomicInteger(-1);

    /** The last run's AC-emulation configurations applied and re-solved (0 = no link ever switched). */
    public static final AtomicInteger LAST_DEVICE_LOOP_REGROUPINGS = new AtomicInteger(0);

    /**
     * Traces the outer-loop sweep to stderr: the rounds, the configuration groups of each round and
     * every matrix refresh. The sweep is the subtlest part of the batched engine - a group's
     * decisions are taken on the device but its termination is per step - so the trace stays, behind
     * {@code OLF_TS_TRACE}.
     */
    private static final boolean TRACE = System.getenv("OLF_TS_TRACE") != null;

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
    private int[] currentHvdcConfiguration;
    private int[] currentTapConfiguration;

    // the incremental phase control loop, per step
    private final List<TransformerPhaseControl> phaseControls;
    private final List<PiModelArray> phaseModels;
    private final int[] phaseLowTapPosition;
    private final int[] baseTapIndex;
    private int[] chunkTapIndex;
    private int[] chunkAllowedDirection;
    private int[] chunkDirectionChangeCount;
    private int[] chunkCurrentDirection;

    /** The loops OLF would run on this network, in its order; 0 = AC-emulation limits, 1 = phase control. */
    private final int[] loopKinds;

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
        this.currentHvdcConfiguration = baseHvdcStatus.clone();
        if (!hvdcs.isEmpty()) {
            uploadHvdcTables();
        }

        // AbstractPhaseControlOuterLoop.getControllerBranches, then the CONTROLLER-mode controls
        List<TransformerPhaseControl> controls = new java.util.ArrayList<>();
        List<PiModelArray> models = new java.util.ArrayList<>();
        List<Integer> lows = new java.util.ArrayList<>();
        for (LfBranch branch : network.getBranches()) {
            if (branch.isDisabled() || !branch.isPhaseController() || !branch.isConnectedAtBothSides()) {
                continue;
            }
            var control = branch.getPhaseControl().orElse(null);
            if (control == null || control.getMode() != TransformerPhaseControl.Mode.CONTROLLER) {
                continue;
            }
            if (!(branch.getPiModel() instanceof PiModelArray piModelArray)) {
                throw new UnsupportedOperationException("phase controller '" + branch.getId() + "' has no tap table");
            }
            controls.add(control);
            models.add(piModelArray);
            lows.add(piModelArray.getTapPositionRange().getMinimum());
        }
        this.phaseControls = List.copyOf(controls);
        this.phaseModels = List.copyOf(models);
        this.phaseLowTapPosition = new int[lows.size()];
        this.baseTapIndex = new int[lows.size()];
        for (int i = 0; i < lows.size(); i++) {
            phaseLowTapPosition[i] = lows.get(i);
            baseTapIndex[i] = phaseModels.get(i).getTapPosition() - phaseLowTapPosition[i];
        }
        this.currentTapConfiguration = baseTapIndex.clone();
        if (!phaseControls.isEmpty()) {
            uploadPhaseTables();
        }
        this.loopKinds = context.getParameters().getOuterLoops().stream()
                .filter(o -> o.isNeeded(context))
                .mapToInt(o -> o instanceof com.powsybl.openloadflow.dc.DcHvdcAcEmulationLimitsOuterLoop ? 0 : 1)
                .toArray();
    }

    /** The controller tables; the tap position and the ratchet are per step. */
    private void uploadPhaseTables() {
        int m = phaseControls.size();
        int[] branch = new int[m];
        int[] alphaColumn = new int[m];
        int[] controlled = new int[m];
        int[] side = new int[m];
        double[] target = new double[m];
        double[] halfDeadband = new double[m];
        int[] tapIndex = new int[m];
        int[] tapPtr = new int[m + 1];
        List<Double> tapA1 = new java.util.ArrayList<>();
        for (int i = 0; i < m; i++) {
            TransformerPhaseControl control = phaseControls.get(i);
            PiModelArray piModel = phaseModels.get(i);
            branch[i] = control.getControllerBranch().getNum();
            alphaColumn[i] = context.getEquationSystem()
                    .getEquation(branch[i], com.powsybl.openloadflow.dc.equations.DcEquationType.BRANCH_TARGET_ALPHA1)
                    .map(com.powsybl.openloadflow.equations.Equation::getColumn)
                    .orElse(-1);
            controlled[i] = control.getControlledBranch().getNum();
            side[i] = control.getControlledSide() == com.powsybl.iidm.network.TwoSides.ONE ? 1 : 2;
            target[i] = control.getTargetValue();
            halfDeadband[i] = Math.max(control.getTargetDeadband(), 1 / com.powsybl.openloadflow.util.PerUnit.SB) / 2;
            tapPtr[i] = tapA1.size();
            var range = piModel.getTapPositionRange();
            for (int tap = range.getMinimum(); tap <= range.getMaximum(); tap++) {
                tapA1.add(piModel.getModel(tap).getA1());
            }
            tapIndex[i] = piModel.getTapPosition() - phaseLowTapPosition[i];
        }
        tapPtr[m] = tapA1.size();
        double[] a1 = new double[tapA1.size()];
        for (int i = 0; i < a1.length; i++) {
            a1[i] = tapA1.get(i);
        }
        solver.setPhaseControllerTables(branch, alphaColumn, controlled, side, target, halfDeadband, tapIndex, tapPtr, a1);
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
        // once per RUN: the partitions each create an engine and then accumulate into this, so a
        // reset inside createEngine would race with another partition's steps and lose them
        LAST_DEVICE_STEP_COUNT.set(0);
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
                    // both DC loops that decide from the state are driven over the chunk; anything
                    // else (area interchange control) still sends the run to the stock engine
                    List<?> needed = dcParameters.getOuterLoops().stream()
                            .filter(o -> o.isNeeded(context))
                            .filter(o -> !(o instanceof com.powsybl.openloadflow.dc.DcHvdcAcEmulationLimitsOuterLoop)
                                    && !(o instanceof com.powsybl.openloadflow.dc.DcIncrementalPhaseControlOuterLoop))
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
        applyConfiguration(baseHvdcStatus, baseTapIndex);
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

    private void resetPhaseState(int count) {
        int m = phaseControls.size();
        if (m == 0) {
            return;
        }
        int size = count * m;
        if (chunkTapIndex == null || chunkTapIndex.length < size) {
            chunkTapIndex = new int[size];
            chunkAllowedDirection = new int[size];
            chunkDirectionChangeCount = new int[size];
            chunkCurrentDirection = new int[size];
        }
        // every step is an independent run, so every step starts from the network's tap positions
        // and from a fresh IncrementalContextData
        for (int step = 0; step < count; step++) {
            for (int c = 0; c < m; c++) {
                chunkTapIndex[step * m + c] = baseTapIndex[c];
                chunkAllowedDirection[step * m + c] = 2;          // AllowedDirection.BOTH
                chunkDirectionChangeCount[step * m + c] = 0;
                chunkCurrentDirection[step * m + c] = -1;
            }
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
        // No lock here. Every partition has its own context, its own cuDSS handle and its own CUDA
        // stream, and those run concurrently; only the analysis phase and the handle lifecycle share
        // library state between handles, and the native layer excludes those itself. Serializing the
        // whole chunk instead - which is what this used to do - made the device section a global
        // critical section and capped the run: 4 to 8 threads moved the GPU arm 1.44 to 1.39 ms/step
        // while the CPU arm moved 2.39 to 1.97.
        applyConfiguration(baseHvdcStatus, baseTapIndex);
        refreshDeviceMatrix();
        solver.timeSeriesChunk(count, chunkGeneratorTargets, generatorCount, chunkLoadTargets, busCount,
                chunkGeneratorInitial, chunkGeneratorParticipating, chunkGeneratorMinTargetP, chunkGeneratorMaxTargetP,
                chunkGeneratorMinP, chunkGeneratorMaxP, chunkGeneratorDroop, chunkGeneratorParticipationFactor, chunkHvdcP0,
                distributedSlack ? participationType : -1, useActiveLimits, assembly.referenceGenerator,
                chunkStates, chunkGeneratorTargets, chunkDistributed, chunkMismatch);
        runOuterLoops(count);
        LAST_DEVICE_STEP_COUNT.addAndGet(count);
    }

    /**
     * OLF's DC outer loops, over the whole chunk.
     *
     * <p>A step whose loop fires needs a different matrix, and the steps of a chunk only batch
     * because they share one factorization. So the steps are grouped by the CONFIGURATION they ask
     * for — the AC-emulation status of every link and the tap position of every phase controller —
     * and a group is solved against its own. A refill and a cuDSS refactorization is 12 ms on
     * rte6515 against 580 for the analysis, which is what makes switching between configurations
     * affordable.
     *
     * <p>Each step carries its own loop state machine, because a step is an independent run:
     * {@code DcLoadFlowEngine.run}'s sweep runs one loop to stability before moving to the next and
     * stops when a full pass solves nothing, and the phase controllers' direction ratchet
     * ({@code IncrementalContextData}) is per run, hence here per step.
     */
    private void runOuterLoops(int count) {
        int hvdcCount = hvdcs.size();
        int controllerCount = phaseControls.size();
        if (loopKinds.length == 0 || hvdcCount + controllerCount == 0) {
            return;
        }
        resetPhaseState(count);
        int[] status = java.util.Arrays.copyOf(chunkHvdcStatus, Math.max(count * hvdcCount, 1));
        double[] offsets = java.util.Arrays.copyOf(chunkHvdcP0, Math.max(count * hvdcCount, 1));

        // per step: where it is in OLF's sweep, what fired last, how many solves it has had, and
        // whether the pass it is in has solved anything
        int[] loopIndex = new int[count];
        int[] lastUnstable = new int[count];
        int[] totalIterations = new int[count];
        int[] passSolves = new int[count];
        boolean[] done = new boolean[count];
        boolean[] needsResolve = new boolean[count];
        java.util.Arrays.fill(lastUnstable, -1);

        for (int round = 0; round < MAX_SWEEP_ROUNDS; round++) {
            java.util.Map<String, List<Integer>> groups = new java.util.LinkedHashMap<>();
            for (int step = 0; step < count; step++) {
                if (!done[step]) {
                    groups.computeIfAbsent(configurationKey(step, status, hvdcCount, controllerCount), k -> new java.util.ArrayList<>())
                            .add(step);
                }
            }
            if (groups.isEmpty()) {
                break;
            }
            if (TRACE) {
                System.err.printf("TS_SWEEP round=%d groups=%d%n", round, groups.size());
            }
            for (List<Integer> group : groups.values()) {
                int first = group.get(0);
                applyConfiguration(configurationOf(first, status, hvdcCount, baseHvdcStatus.length),
                        tapConfigurationOf(first, controllerCount));
                refreshDeviceMatrix();
                int[] steps = group.stream().mapToInt(Integer::intValue).toArray();
                boolean resolve = false;
                for (int step : steps) {
                    resolve |= needsResolve[step];
                }
                if (TRACE) {
                    System.err.printf("TS_SWEEP   group steps=%s config=%s resolve=%s loopIdx=%d lastUnstable=%d%n",
                            group, configurationKey(first, status, hvdcCount, controllerCount), resolve, loopIndex[first], lastUnstable[first]);
                }
                if (resolve) {
                    LAST_DEVICE_LOOP_REGROUPINGS.incrementAndGet();
                    updateHvdcInjections(steps, hvdcCount);
                    solver.timeSeriesResolve(steps, chunkGeneratorTargets, generators.size(), chunkLoadTargets,
                            network.getBuses().size(), chunkStates);
                    for (int step : steps) {
                        needsResolve[step] = false;
                    }
                }
                // the decisions of both loops on this group's steps, each on its own state
                int[] hvdcDecisions = hvdcCount > 0 ? solver.timeSeriesHvdcCheck(count, status, offsets) : new int[0];
                int[] phaseDecisions = new int[0];
                int[] tapSlice = null;
                int[] allowedSlice = null;
                int[] countSlice = null;
                int[] directionSlice = null;
                if (controllerCount > 0) {
                    tapSlice = gather(chunkTapIndex, steps, controllerCount);
                    allowedSlice = gather(chunkAllowedDirection, steps, controllerCount);
                    countSlice = gather(chunkDirectionChangeCount, steps, controllerCount);
                    directionSlice = gather(chunkCurrentDirection, steps, controllerCount);
                    phaseDecisions = solver.timeSeriesPhaseCheck(steps, tapSlice, allowedSlice, countSlice, directionSlice);
                }
                for (int i = 0; i < steps.length; i++) {
                    int step = steps[i];
                    int kind = loopKinds[loopIndex[step]];
                    boolean changed = false;
                    if (kind == 0) {
                        for (int h = 0; h < hvdcCount; h++) {
                            int decision = hvdcDecisions[step * hvdcCount + h];
                            if (decision >= 0) {
                                status[step * hvdcCount + h] = decision;
                                changed = true;
                            }
                        }
                    } else {
                        for (int c = 0; c < controllerCount; c++) {
                            if (phaseDecisions[i * controllerCount + c] >= 0) {
                                changed = true;
                            }
                        }
                    }
                    if (controllerCount > 0 && kind == 1) {
                        // the decisions are evaluated for the whole group, but only a step whose
                        // sweep is ON the phase control loop has actually run it: keeping the
                        // ratchet of the others would advance a loop OLF has not reached for them
                        scatter(chunkTapIndex, tapSlice, step, i, controllerCount);
                        scatter(chunkAllowedDirection, allowedSlice, step, i, controllerCount);
                        scatter(chunkDirectionChangeCount, countSlice, step, i, controllerCount);
                        scatter(chunkCurrentDirection, directionSlice, step, i, controllerCount);
                    }
                    if (changed) {
                        lastUnstable[step] = loopIndex[step];
                        totalIterations[step]++;
                        passSolves[step]++;
                        needsResolve[step] = true;
                        if (totalIterations[step] >= maxOuterLoopIterations) {
                            done[step] = true;
                        }
                    } else {
                        advance(step, loopIndex, lastUnstable, passSolves, done);
                    }
                }
            }
        }
        if (hvdcCount > 0) {
            System.arraycopy(status, 0, chunkHvdcStatus, 0, count * hvdcCount);
        }
        // the reported mismatch is the one the network ends on, and a loop moves injections
        solver.timeSeriesMismatch(count, chunkMismatch);
    }

    /** A bound on the sweep; each round advances every step's own state machine at least one slot. */
    private static final int MAX_SWEEP_ROUNDS = 1000;

    /**
     * {@code DcLoadFlowEngine.run}'s sweep for one step: a loop that reports stable hands over to
     * the next, the pass ends when the list is exhausted or the next loop is the one that fired
     * last, and the run ends when a whole pass has solved nothing.
     */
    private void advance(int step, int[] loopIndex, int[] lastUnstable, int[] passSolves, boolean[] done) {
        loopIndex[step]++;
        while (loopIndex[step] >= loopKinds.length || loopIndex[step] == lastUnstable[step]) {
            if (passSolves[step] == 0) {
                done[step] = true;
                return;
            }
            passSolves[step] = 0;
            loopIndex[step] = 0;
            if (loopKinds.length > 0 && loopIndex[step] != lastUnstable[step]) {
                return;
            }
            if (loopKinds.length == 0) {
                done[step] = true;
                return;
            }
            // the first loop of the new pass is the one that fired last: the pass ends at once
            done[step] = true;
            return;
        }
    }

    private static int[] gather(int[] source, int[] steps, int width) {
        int[] slice = new int[steps.length * width];
        for (int i = 0; i < steps.length; i++) {
            System.arraycopy(source, steps[i] * width, slice, i * width, width);
        }
        return slice;
    }

    private static void scatter(int[] target, int[] slice, int step, int i, int width) {
        System.arraycopy(slice, i * width, target, step * width, width);
    }

    private String configurationKey(int step, int[] status, int hvdcCount, int controllerCount) {
        StringBuilder key = new StringBuilder();
        for (int h = 0; h < hvdcCount; h++) {
            key.append(status[step * hvdcCount + h]).append(',');
        }
        key.append('|');
        for (int c = 0; c < controllerCount; c++) {
            key.append(chunkTapIndex[step * controllerCount + c]).append(',');
        }
        return key.toString();
    }

    private static int[] configurationOf(int step, int[] status, int hvdcCount, int size) {
        int[] configuration = new int[size];
        if (hvdcCount > 0) {
            System.arraycopy(status, step * hvdcCount, configuration, 0, hvdcCount);
        }
        return configuration;
    }

    private int[] tapConfigurationOf(int step, int controllerCount) {
        int[] taps = new int[controllerCount];
        if (controllerCount > 0) {
            System.arraycopy(chunkTapIndex, step * controllerCount, taps, 0, controllerCount);
        }
        return taps;
    }

    /** A saturating link retargets its converter stations: that is an injection change of those steps. */
    private void updateHvdcInjections(int[] steps, int hvdcCount) {
        int generatorCount = generators.size();
        for (int step : steps) {
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
    }

    /** Put the network in the given configuration, with OLF's own mutators. */
    private boolean applyConfiguration(int[] hvdcConfiguration, int[] tapConfiguration) {
        boolean changed = false;
        for (int h = 0; h < hvdcs.size(); h++) {
            if (hvdcConfiguration[h] == currentHvdcConfiguration[h]) {
                continue;
            }
            changed = true;
            var control = hvdcs.get(h).getAcEmulationControl();
            switch (hvdcConfiguration[h]) {
                case 0 -> control.switchToLinearMode();
                case 1 -> control.switchToSaturationFromCS1toCS2(false);
                case 2 -> control.switchToSaturationFromCS2toCS1(false);
                default -> throw new IllegalStateException("unexpected AC emulation status " + hvdcConfiguration[h]);
            }
        }
        currentHvdcConfiguration = hvdcConfiguration.clone();
        for (int c = 0; c < phaseControls.size(); c++) {
            if (tapConfiguration[c] != currentTapConfiguration[c]) {
                phaseModels.get(c).setTapPosition(phaseLowTapPosition[c] + tapConfiguration[c]);
                changed = true;
            }
        }
        currentTapConfiguration = tapConfiguration.clone();
        deviceMatrixDirty |= changed;
        return changed;
    }

    /** True when the network's configuration has moved since the device matrix was last filled. */
    private boolean deviceMatrixDirty;

    /** Refill the device matrix for the configuration the network now carries, and refactorize. */
    private void refreshDeviceMatrix() {
        if (!deviceMatrixDirty) {
            return;
        }
        GpuDcAssembly refreshed = GpuDcAssembly.capture(network, context.getEquationSystem(),
                context.getParameters().getEquationSystemCreationParameters());
        if (!assembly.samePattern(refreshed)) {
            throw new IllegalStateException("an outer loop changed the matrix pattern");
        }
        boolean structureChanged = !assembly.sameStructure(refreshed);
        if (structureChanged) {
            solver.setContributions(refreshed);
        }
        if (TRACE) {
            System.err.printf("TS_SWEEP   refresh structureChanged=%s nnz=%d%n", structureChanged, refreshed.nonZeroCount());
        }
        solver.refreshBranches(refreshed);
        assembly = refreshed;
        deviceMatrixDirty = false;
        if (!phaseControls.isEmpty()) {
            uploadPhaseTables();
        }
    }

    /**
     * Make the step's solution current on the network, so that the results are emitted from the
     * same places a CPU step emits them from: the state vector (branch flows and bus angles) and the
     * generator targets after distribution.
     */
    @Override
    public TimeSeriesLoadFlow.SolveResult apply(int slot) {
        // the step's own configuration first: a branch flow is read from the network's pi model, so
        // a controller sitting on another step's tap would be emitted with the wrong power and shift
        applyConfiguration(configurationOf(slot, chunkHvdcStatus, hvdcs.size(), baseHvdcStatus.length),
                tapConfigurationOf(slot, phaseControls.size()));
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
