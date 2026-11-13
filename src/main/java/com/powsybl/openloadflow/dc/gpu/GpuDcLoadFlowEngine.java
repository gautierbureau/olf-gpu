/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.dc.gpu;

import com.google.common.collect.Lists;
import com.powsybl.commons.PowsyblException;
import com.powsybl.commons.report.ReportNode;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.dc.DcAreaInterchangeControlOuterLoop;
import com.powsybl.openloadflow.dc.DcHvdcAcEmulationLimitsOuterLoop;
import com.powsybl.openloadflow.dc.DcIncrementalPhaseControlOuterLoop;
import com.powsybl.openloadflow.dc.DcLoadFlowContext;
import com.powsybl.openloadflow.dc.DcLoadFlowEngine;
import com.powsybl.openloadflow.dc.DcLoadFlowParameters;
import com.powsybl.openloadflow.dc.DcLoadFlowResult;
import com.powsybl.openloadflow.dc.DcOuterLoop;
import com.powsybl.openloadflow.dc.DcOuterLoopContext;
import com.powsybl.openloadflow.dc.equations.DcEquationType;
import com.powsybl.openloadflow.dc.equations.DcVariableType;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.lf.outerloop.OuterLoopResult;
import com.powsybl.openloadflow.lf.outerloop.OuterLoopStatus;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfSynchronousNetwork;
import com.powsybl.openloadflow.network.util.ActivePowerDistribution;
import com.powsybl.openloadflow.network.util.UniformValueVoltageInitializer;
import com.powsybl.openloadflow.util.PerUnit;
import com.powsybl.openloadflow.util.Reports;
import com.powsybl.openloadflow.util.gpu.GpuTrace;
import org.apache.commons.lang3.tuple.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * {@code DcLoadFlowEngine.run} with the DC system assembled and solved on the device: the matrix
 * values, the target vector, the slack distribution and the linear solve never touch a host matrix.
 * The host keeps what OLF keeps on the host anyway — the equation system (the structure), the
 * outer loops' decisions, the network update — and the reporting is the stock engine's, line for
 * line.
 *
 * <p>Protocol: the {@link LfNetwork} is the source of truth between phases. {@link #prepare()}
 * captures it ({@link GpuDcAssembly}) and synchronizes the device; the device distribution's
 * generator targets are written back to the network at once, so every later host step
 * ({@code handleDistributionFailureBehavior}, the outer loops, {@code updateState}) sees what the
 * stock path would. An outer-loop iteration re-synchronizes (a tap change refills the values on
 * the same pattern; a structural change rebuilds the context).
 *
 * <p>Load-based balance types distribute on the host (OLF's own {@code ActivePowerDistribution})
 * and refresh the device tables — the generation types run the device kernel.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuDcLoadFlowEngine implements AutoCloseable {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuDcLoadFlowEngine.class);

    /** The last run's device outer-loop decisions (test/validation probes): {@code -1} when the loops
     *  ran on the host, otherwise the number of taps moved and of AC-emulation switches applied. */
    public static final java.util.concurrent.atomic.AtomicInteger LAST_DEVICE_TAP_CHANGES = new java.util.concurrent.atomic.AtomicInteger(-1);

    public static final java.util.concurrent.atomic.AtomicInteger LAST_DEVICE_HVDC_SWITCHES = new java.util.concurrent.atomic.AtomicInteger(-1);

    private final DcLoadFlowContext context;
    private GpuDcAssembly assembly;
    private GpuDcSolver solver;
    private GpuDcOuterLoops loops;
    private boolean deviceLoops;

    public GpuDcLoadFlowEngine(DcLoadFlowContext context) {
        this.context = Objects.requireNonNull(context);
    }

    public DcLoadFlowContext getContext() {
        return context;
    }

    /** The device context (S columns, Woodbury chunks): valid after {@link #prepare()}. */
    public GpuDcSolver getSolver() {
        if (solver == null) {
            throw new IllegalStateException("not prepared");
        }
        return solver;
    }

    public GpuDcAssembly getAssembly() {
        if (assembly == null) {
            throw new IllegalStateException("not prepared");
        }
        return assembly;
    }

    /**
     * The device participation type of a balance type, or {@code -1} for the load-based types the
     * kernel does not model (0 = P_MAX, 1 = P, 2 = PARTICIPATION_FACTOR, 3 = REMAINING_MARGIN).
     */
    public static int participationType(LoadFlowParameters.BalanceType balanceType) {
        return switch (balanceType) {
            case PROPORTIONAL_TO_GENERATION_P_MAX -> 0;
            case PROPORTIONAL_TO_GENERATION_P -> 1;
            case PROPORTIONAL_TO_GENERATION_PARTICIPATION_FACTOR -> 2;
            case PROPORTIONAL_TO_GENERATION_REMAINING_MARGIN -> 3;
            default -> -1;
        };
    }

    /**
     * Capture the network's DC system and synchronize the device: the first time, a new context
     * (analysis + factorization); afterwards a refill on the same pattern, or a rebuild when the
     * structure changed.
     *
     * @throws UnsupportedOperationException when an equation term has no device model (the caller keeps the host path)
     */
    public void prepare() {
        LfNetwork network = context.getNetwork();
        long t0 = System.nanoTime();
        GpuDcAssembly a = GpuDcAssembly.capture(network, context.getEquationSystem(), context.getParameters().getEquationSystemCreationParameters());
        long t1 = System.nanoTime();
        if (solver == null) {
            solver = GpuDcSolver.createAssembled(a);
            if (GpuTrace.PHASE_TIME.isOn()) {
                System.err.printf("DC_ASSEMBLY n=%d nnz=%d capture=%.1f ms upload+fill+analysis+factorization=%.1f ms%n",
                        a.size(), a.nonZeroCount(), (t1 - t0) / 1e6, (System.nanoTime() - t1) / 1e6);
            }
        } else if (assembly.samePattern(a)) {
            // same non-zero pattern: refill the values and the target on the SAME factorization structure.
            // The contribution lists move when a term is switched on or off (an HVDC entering or leaving
            // saturation), so they are re-uploaded too — no new analysis either way.
            if (!assembly.sameStructure(a)) {
                solver.setContributions(a);
            }
            solver.refreshBranches(a);
            solver.setBusLoads(a.busLoadTargetP);
            solver.setGeneratorTargets(a.genTargetP);
        } else {
            LOGGER.debug("DC structure changed, device context rebuilt");
            solver.close();
            solver = null;
            solver = GpuDcSolver.createAssembled(a);
            loops = null;
        }
        assembly = a;
        if (loops != null) {
            loops.upload(context.getEquationSystem());
        }
    }

    /**
     * Build the device outer-loop tables ({@link GpuDcOuterLoops}) for the loops that decide from the
     * solved state. Call after the outer loops' {@code initialize}. Returns false when a loop has no
     * device model — the engine then runs every loop through OLF.
     */
    private boolean prepareLoops(List<DcOuterLoop> outerLoops) {
        boolean phase = outerLoops.stream().anyMatch(DcIncrementalPhaseControlOuterLoop.class::isInstance);
        boolean hvdc = outerLoops.stream().anyMatch(DcHvdcAcEmulationLimitsOuterLoop.class::isInstance);
        if (!phase && !hvdc || HOST_OUTER_LOOPS.get()) {
            loops = null;
            return false;
        }
        try {
            loops = GpuDcOuterLoops.capture(context.getNetwork(), context.getEquationSystem(), solver, assembly);
            loops.resetPhaseControlDirections();
            LAST_DEVICE_TAP_CHANGES.set(0);
            LAST_DEVICE_HVDC_SWITCHES.set(0);
            return true;
        } catch (UnsupportedOperationException e) {
            LOGGER.warn("GPU DC outer loops: {} — the outer loops run on the host", e.getMessage());
            loops = null;
            return false;
        }
    }

    /** Test hook, the A/B of the kernels: true runs OLF's own loop checks instead of the device ones. Not a
     *  configuration: the device loops are the product, this only proves they decide as OLF's do. */
    public static final java.util.concurrent.atomic.AtomicBoolean HOST_OUTER_LOOPS = new java.util.concurrent.atomic.AtomicBoolean();

    /**
     * The device check of a loop that decides from the state, or {@code null} when this loop has no
     * device model and OLF's own {@code check} must run ({@code DcAreaInterchangeControl}, and every
     * loop when the device tables could not be built).
     */
    private OuterLoopResult deviceCheck(DcOuterLoop outerLoop, ReportNode reportNode) {
        if (!deviceLoops || loops == null) {
            return null;
        }
        if (outerLoop instanceof DcIncrementalPhaseControlOuterLoop) {
            return loops.hasPhaseControllers()
                    ? new OuterLoopResult(outerLoop, loops.checkPhaseControl())
                    : new OuterLoopResult(outerLoop, OuterLoopStatus.STABLE);
        }
        if (outerLoop instanceof DcHvdcAcEmulationLimitsOuterLoop) {
            return loops.hasHvdcs()
                    ? new OuterLoopResult(outerLoop, loops.checkHvdc(reportNode))
                    : new OuterLoopResult(outerLoop, OuterLoopStatus.STABLE);
        }
        return null;
    }

    /** The device generator targets back into the network (setTargetP fires the listeners only on a change). */
    private void writeBackGenerators() {
        double[] targets = solver.readGeneratorTargets();
        List<LfGenerator> generators = assembly.generators();
        for (int i = 0; i < targets.length; i++) {
            generators.get(i).setTargetP(targets[i]);
        }
    }

    /** The network's generator targets onto the device (after a host-side mutation). */
    private void pushGenerators() {
        List<LfGenerator> generators = assembly.generators();
        double[] targets = new double[generators.size()];
        for (int i = 0; i < targets.length; i++) {
            targets[i] = generators.get(i).getTargetP();
        }
        solver.setGeneratorTargets(targets);
    }

    private void pushLoads() {
        List<LfBus> buses = context.getNetwork().getBuses();
        double[] loads = new double[buses.size()];
        for (int i = 0; i < loads.length; i++) {
            loads[i] = buses.get(i).getLoadTargetP();
        }
        solver.setBusLoads(loads);
    }

    /**
     * {@code ActivePowerDistribution.create(balanceType, false, useActiveLimits).run(referenceGenerator, buses, mismatch)}:
     * on the device for the generation balance types (generators written back), on the host otherwise
     * (device refreshed). {@code excludedBuses} are the buses left out (islanded), by number.
     */
    public ActivePowerDistribution.Result distribute(LoadFlowParameters.BalanceType balanceType, boolean useActiveLimits,
                                                     Set<Integer> excludedBuses, double mismatch) {
        int type = participationType(balanceType);
        if (type >= 0) {
            boolean withExcluded = excludedBuses != null && !excludedBuses.isEmpty();
            if (withExcluded) {
                solver.setExcludedBuses(excludedBuses.stream().mapToInt(Integer::intValue).toArray());
            }
            double[] before = solver.readGeneratorTargets();
            GpuDcSolver.Distribution d = solver.distributeSlack(type, useActiveLimits, assembly.referenceGenerator, withExcluded, mismatch);
            writeBackGenerators();
            boolean moved = false;
            double[] after = solver.readGeneratorTargets();
            double diff = 0;
            for (int i = 0; i < after.length; i++) {
                diff += Math.abs(after[i] - before[i]);
            }
            moved = diff > 0.9 * ActivePowerDistribution.P_RESIDUE_EPS;   // PreviousStateInfo.moved's criterion
            return new ActivePowerDistribution.Result(d.iterations(), d.remainingMismatch(), moved);
        }
        LfNetwork network = context.getNetwork();
        Collection<LfBus> buses = network.getBuses();
        if (excludedBuses != null && !excludedBuses.isEmpty()) {
            buses = buses.stream().filter(b -> !excludedBuses.contains(b.getNum())).toList();
        }
        LfGenerator reference = network.getSynchronousNetworks().getFirst().getReferenceGenerator();
        ActivePowerDistribution.Result result = ActivePowerDistribution.create(balanceType, false, useActiveLimits).run(reference, buses, mismatch);
        pushGenerators();
        pushLoads();
        return result;
    }

    /** {@code DcLoadFlowEngine.getActivePowerMismatch} over the network's buses, on the device. */
    public double activePowerMismatch() {
        return solver.activePowerMismatch(false);
    }

    /** {@code DcLoadFlowEngine.getActivePowerMismatch} over the buses not excluded, on the device. */
    public double activePowerMismatch(Set<Integer> excludedBuses) {
        if (excludedBuses == null || excludedBuses.isEmpty()) {
            return solver.activePowerMismatch(false);
        }
        solver.setExcludedBuses(excludedBuses.stream().mapToInt(Integer::intValue).toArray());
        return solver.activePowerMismatch(true);
    }

    /** The target vector from the device tables, then the solve: the state, or {@code null} when the solver failed. */
    public double[] solve(ReportNode reportNode) {
        try {
            solver.fillTarget();
            return solver.solveTarget();
        } catch (RuntimeException e) {
            Reports.reportDcLfSolverFailure(reportNode, e.getMessage());
            LOGGER.error("Failed to solve linear system for DC load flow", e);
            return null;
        }
    }

    /**
     * The pre-contingency solve of the Woodbury security analysis
     * ({@code WoodburyEngine.runDcLoadFlowWithModifiedTargetVector} with nothing disabled): the
     * slack distributed over every bus when asked — no epsilon, no failure behavior — then the
     * target and the solve on the device. The state is loaded into the equation system and the
     * network updated, like the stock path.
     */
    public double[] solveBase() {
        DcLoadFlowParameters parameters = context.getParameters();
        if (parameters.isDistributedSlack()) {
            distribute(parameters.getBalanceType(), parameters.getNetworkParameters().isUseActiveLimits(), null, activePowerMismatch());
        }
        solver.fillTarget();
        double[] x = solver.solveTarget();
        EquationSystem<DcVariableType, DcEquationType> equationSystem = context.getEquationSystem();
        equationSystem.getStateVector().set(x);
        DcLoadFlowEngine.updateNetwork(context.getNetwork(), equationSystem, x);
        return x;
    }

    private static final class RunningContext {
        private boolean lastSolverSuccess = false;
        private int solverTotalExecutions = 0;
        private int outerLoopTotalIterations = 0;
        private OuterLoopResult lastOuterLoopResult = OuterLoopResult.stable();
        private DcOuterLoop lastUnstableOuterLoop;
    }

    private void runOuterLoop(DcOuterLoop outerLoop, DcOuterLoopContext outerLoopContext, RunningContext runningContext) {
        ReportNode olReportNode = Reports.createOuterLoopReporter(outerLoopContext.getNetwork().getReportNode(), outerLoop.getName());
        OuterLoopResult outerLoopResult;
        int outerLoopIteration = 0;
        do {
            outerLoopContext.setIteration(outerLoopIteration);
            outerLoopContext.setLoadFlowContext(context);
            outerLoopContext.setOuterLoopTotalIterations(runningContext.outerLoopTotalIterations);
            OuterLoopResult deviceResult = deviceCheck(outerLoop, olReportNode);
            outerLoopResult = deviceResult != null ? deviceResult : outerLoop.check(outerLoopContext, olReportNode);
            runningContext.lastOuterLoopResult = outerLoopResult;

            if (outerLoopResult.status() == OuterLoopStatus.UNSTABLE) {
                LOGGER.debug("Start outer loop '{}' iteration {}", outerLoop.getName(), outerLoopIteration);
                // the outer loop mutated the network (taps, HVDC, injections): the device follows the network
                prepare();
                double[] x = solve(olReportNode);
                runningContext.lastSolverSuccess = x != null;
                runningContext.solverTotalExecutions++;
                runningContext.lastUnstableOuterLoop = outerLoop;
                if (runningContext.lastSolverSuccess) {
                    context.getEquationSystem().getStateVector().set(x);
                    DcLoadFlowEngine.updateNetwork(outerLoopContext.getNetwork(), context.getEquationSystem(), x);
                }
                outerLoopIteration++;
                runningContext.outerLoopTotalIterations++;
            }
        } while (outerLoopResult.status() == OuterLoopStatus.UNSTABLE
                && runningContext.lastSolverSuccess
                && runningContext.outerLoopTotalIterations < context.getParameters().getMaxOuterLoopIterations());

        if (outerLoopResult.status() != OuterLoopStatus.STABLE) {
            Reports.reportUnsuccessfulOuterLoop(olReportNode, outerLoopResult.status().name());
        }
    }

    /** {@code DcLoadFlowEngine.run}, the device doing the assembly, the distribution and the solves. */
    public DcLoadFlowResult run() {
        if (context.getParameters().getNetworkParameters().isAcDcNetwork()) {
            throw new PowsyblException("DC load flow does not support detailed AC-DC networks");
        }
        LfNetwork network = context.getNetwork();
        ReportNode reportNode = network.getReportNode();
        EquationSystem<DcVariableType, DcEquationType> equationSystem = context.getEquationSystem();
        DcLoadFlowParameters parameters = context.getParameters();
        RunningContext runningContext = new RunningContext();
        List<DcOuterLoop> outerLoops = parameters.getOuterLoops().stream().filter(o -> o.isNeeded(context)).toList();

        List<Pair<DcOuterLoop, DcOuterLoopContext>> outerLoopsAndContexts = outerLoops.stream()
                .map(outerLoop -> Pair.of(outerLoop, new DcOuterLoopContext(network)))
                .toList();
        for (var outerLoopAndContext : outerLoopsAndContexts) {
            outerLoopAndContext.getLeft().initialize(outerLoopAndContext.getRight());
        }

        DcLoadFlowEngine.initStateVector(network, equationSystem, new UniformValueVoltageInitializer());
        prepare();
        deviceLoops = prepareLoops(outerLoops);

        double initialSlackBusActivePowerMismatch = activePowerMismatch();
        double distributedActivePower = 0.0;

        boolean isAreaInterchangeControl = outerLoops.stream().anyMatch(DcAreaInterchangeControlOuterLoop.class::isInstance);
        if ((parameters.isDistributedSlack() || isAreaInterchangeControl)
                && Math.abs(initialSlackBusActivePowerMismatch) > ActivePowerDistribution.P_RESIDUE_EPS) {
            LoadFlowParameters.BalanceType balanceType = parameters.getBalanceType();
            boolean useActiveLimits = parameters.getNetworkParameters().isUseActiveLimits();
            // ActivePowerDistribution.run(firstSynchronousNetwork, mismatch): its reference generator, its buses
            LfSynchronousNetwork first = network.getSynchronousNetworks().getFirst();
            ActivePowerDistribution.Result result = distribute(balanceType, useActiveLimits, excludedOutside(network, first), initialSlackBusActivePowerMismatch);
            final LfGenerator referenceGenerator;
            final OpenLoadFlowParameters.SlackDistributionFailureBehavior behavior;
            if (isAreaInterchangeControl && network.hasArea()) {
                behavior = OpenLoadFlowParameters.SlackDistributionFailureBehavior.LEAVE_ON_SLACK_BUS;
                referenceGenerator = null;
            } else {
                behavior = parameters.getSlackDistributionFailureBehavior();
                referenceGenerator = first.getReferenceGenerator();
            }
            ActivePowerDistribution.ResultWithFailureBehaviorHandling resultWbh = ActivePowerDistribution.handleDistributionFailureBehavior(
                    behavior, referenceGenerator, initialSlackBusActivePowerMismatch, result,
                    "Failed to distribute slack bus active power mismatch, %.2f MW remains");
            pushGenerators();                                // DISTRIBUTE_ON_REFERENCE_GENERATOR moved one on the host
            double remainingMismatch = resultWbh.remainingMismatch();
            distributedActivePower = initialSlackBusActivePowerMismatch - remainingMismatch;
            if (Math.abs(remainingMismatch) > parameters.getSlackBusPMaxMismatch() / PerUnit.SB) {
                Reports.reportMismatchDistributionFailure(reportNode, remainingMismatch * PerUnit.SB);
            } else {
                if (Math.abs(remainingMismatch) > ActivePowerDistribution.P_RESIDUE_EPS) {
                    Reports.reportResidualDistributionMismatch(reportNode, remainingMismatch * PerUnit.SB);
                }
                ActivePowerDistribution.reportAndLogSuccess(reportNode, initialSlackBusActivePowerMismatch, resultWbh);
            }
            if (resultWbh.failed()) {
                distributedActivePower -= resultWbh.failedDistributedActivePower();
                runningContext.lastSolverSuccess = false;
                runningContext.lastOuterLoopResult = new OuterLoopResult("DistributedSlack", OuterLoopStatus.FAILED, resultWbh.failedMessage());
                Reports.reportDcLfComplete(reportNode, runningContext.lastSolverSuccess, runningContext.lastOuterLoopResult.status().name());
                return buildDcLoadFlowResult(network, runningContext, initialSlackBusActivePowerMismatch, distributedActivePower);
            }
        }

        double[] x = solve(reportNode);
        runningContext.lastSolverSuccess = x != null;
        if (x != null) {
            equationSystem.getStateVector().set(x);
            DcLoadFlowEngine.updateNetwork(network, equationSystem, x);
        }

        if (runningContext.lastSolverSuccess) {
            int oldSolverTotalExecutions;
            do {
                oldSolverTotalExecutions = runningContext.solverTotalExecutions;
                for (var outerLoopAndContext : outerLoopsAndContexts) {
                    DcOuterLoop outerLoop = outerLoopAndContext.getLeft();
                    if (outerLoop == runningContext.lastUnstableOuterLoop
                            || !runningContext.lastSolverSuccess
                            || runningContext.lastOuterLoopResult.status() == OuterLoopStatus.FAILED
                            || runningContext.outerLoopTotalIterations >= parameters.getMaxOuterLoopIterations()) {
                        break;
                    }
                    runOuterLoop(outerLoop, outerLoopAndContext.getRight(), runningContext);
                }
            } while (runningContext.solverTotalExecutions > oldSolverTotalExecutions
                    && runningContext.lastSolverSuccess
                    && runningContext.lastOuterLoopResult.status() != OuterLoopStatus.FAILED
                    && runningContext.outerLoopTotalIterations < parameters.getMaxOuterLoopIterations());
        }

        if (runningContext.outerLoopTotalIterations >= parameters.getMaxOuterLoopIterations()) {
            Reports.reportMaxOuterLoopIterations(reportNode, runningContext.outerLoopTotalIterations, true, LOGGER);
        }

        for (var outerLoopAndContext : Lists.reverse(outerLoopsAndContexts)) {
            var outerLoop = outerLoopAndContext.getLeft();
            var outerLoopContext = outerLoopAndContext.getRight();
            if (outerLoop instanceof DcAreaInterchangeControlOuterLoop activePowerDistributionOuterLoop) {
                distributedActivePower += activePowerDistributionOuterLoop.getDistributedActivePower(outerLoopContext);
            }
            outerLoop.cleanup(outerLoopContext);
        }

        if (parameters.isSetVToNan()) {
            for (LfBus bus : network.getBuses()) {
                bus.setV(Double.NaN);
            }
        }

        Reports.reportDcLfComplete(reportNode, runningContext.lastSolverSuccess, runningContext.lastOuterLoopResult.status().name());
        return buildDcLoadFlowResult(network, runningContext, initialSlackBusActivePowerMismatch, distributedActivePower);
    }

    /** The buses outside the first synchronous network, excluded from its distribution (none on a plain DC network). */
    private static Set<Integer> excludedOutside(LfNetwork network, LfSynchronousNetwork first) {
        if (network.getSynchronousNetworks().size() == 1) {
            return Set.of();
        }
        return network.getBuses().stream().filter(b -> b.getNumSC() != first.getNumSC()).map(LfBus::getNum)
                .collect(java.util.stream.Collectors.toSet());
    }

    private DcLoadFlowResult buildDcLoadFlowResult(LfNetwork network, RunningContext runningContext, double initialSlackBusActivePowerMismatch,
                                                   double finalDistributedActivePower) {
        double slackBusActivePowerMismatch;
        double distributedActivePower;
        if (runningContext.lastSolverSuccess && runningContext.lastOuterLoopResult.status() == OuterLoopStatus.STABLE) {
            slackBusActivePowerMismatch = DcLoadFlowEngine.getActivePowerMismatch(network.getBuses());
            distributedActivePower = finalDistributedActivePower;
        } else {
            slackBusActivePowerMismatch = initialSlackBusActivePowerMismatch;
            distributedActivePower = 0.0;
        }
        DcLoadFlowResult result = new DcLoadFlowResult(network, runningContext.outerLoopTotalIterations,
                runningContext.lastSolverSuccess, runningContext.lastOuterLoopResult, slackBusActivePowerMismatch, distributedActivePower);
        LOGGER.info("DC loadflow complete on network {} (result={})", network, result);
        return result;
    }

    @Override
    public void close() {
        if (solver != null) {
            solver.close();
            solver = null;
        }
    }
}
