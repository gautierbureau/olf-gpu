/*
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.google.common.base.Stopwatch;
import com.powsybl.action.Action;
import com.powsybl.commons.report.ReportNode;
import com.powsybl.contingency.Contingency;
import com.powsybl.contingency.strategy.OperatorStrategy;
import com.powsybl.iidm.network.Network;
import com.powsybl.math.matrix.MatrixFactory;
import com.powsybl.openloadflow.ac.AcLoadFlowContext;
import com.powsybl.openloadflow.ac.AcLoadFlowParameters;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.ac.solver.AcSolverUtil;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonFactory;
import com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphsonFactory;
import com.powsybl.openloadflow.ac.solver.gpu.batch.BatchConfig;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchStatus;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchedSecurityAnalysisSolver;
import com.powsybl.openloadflow.graph.GraphConnectivityFactory;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfContingency;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LoadFlowModel;
import com.powsybl.openloadflow.network.impl.PropagatedContingency;
import com.powsybl.openloadflow.sa.AbstractNetworkResult;
import com.powsybl.openloadflow.sa.AcSecurityAnalysis;
import com.powsybl.openloadflow.sa.ContingencyActivePowerLossDistribution;
import com.powsybl.openloadflow.sa.LimitViolationManager;
import com.powsybl.openloadflow.sa.OpenSecurityAnalysisParameters;
import com.powsybl.openloadflow.sa.PostContingencyNetworkResult;
import com.powsybl.openloadflow.sa.PreContingencyNetworkResult;
import com.powsybl.openloadflow.util.PerUnit;
import com.powsybl.security.LimitViolationsResult;
import com.powsybl.security.PostContingencyComputationStatus;
import com.powsybl.security.SecurityAnalysisParameters;
import com.powsybl.security.SecurityAnalysisResult;
import com.powsybl.security.limitreduction.LimitReduction;
import com.powsybl.security.monitor.StateMonitor;
import com.powsybl.security.results.ConnectivityResult;
import com.powsybl.security.results.NetworkResult;
import com.powsybl.security.results.OperatorStrategyResult;
import com.powsybl.security.results.PostContingencyResult;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Prototype integration of the batched GPU N-1 ({@link GpuBatchedSecurityAnalysisSolver})
 * into OLF's AC security-analysis engine. Single-branch contingencies whose post-contingency
 * states the GPU batch can solve (qualified by {@link com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchContingencyValidator})
 * are solved ALL AT ONCE, once, right after the pre-contingency load flow; the engine's
 * per-contingency loop then INJECTS each precomputed state instead of running its own
 * Newton-Raphson, and reuses the existing violation-detection and result-building machinery
 * unchanged.
 *
 * <p>This subclass intercepts at three seams: {@link #runSimulations} (capture the network +
 * contingency list), {@link #afterPreContingencySimulation} (precompute the batch from the
 * converged base state), and {@link #runPostContingencySimulation} (inject a precomputed
 * state, or delegate to the standard CPU path). Islanding outages ARE batched: the
 * islanded buses' rows are frozen to identity at 1&#8736;0 over the fixed superset pattern
 * (per-scenario row mode), so the main component solves on the device and the islanded
 * buses report a clean placeholder. The batch covers MULTI-BRANCH outages (the disabled
 * mask is per element), single-generator outages (with slack-distribution overrides) and
 * shunt outages. Everything it cannot batch — contingency shapes beyond those mapped
 * kinds (e.g. mixed branch+generator), elements outside the GPU model, slack-islanding
 * outages, or scenarios that diverged / were straggler-offloaded on the device — falls
 * through to {@code super}, so results are never wrong, only un-accelerated.
 *
 * <p>SCOPE (prototype): plain post-contingency power flow (no device-side outer loops —
 * distributed slack / reactive limits are not re-run on the GPU, so distributed active
 * power is reported from the pre-distribution only). Operator strategies still run on the
 * CPU from the injected post-contingency state. Single-threaded (the device context is not
 * shared across threads). When the GPU is unavailable the class behaves exactly like
 * {@link AcSecurityAnalysis}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public class GpuBatchedAcSecurityAnalysis extends AcSecurityAnalysis {

    /** System property selecting how the batched N-1 runs its OUTER loops: {@code DEVICE} (CUDA-kernel outer
     *  loops folded into the device solve, the default) or {@code JAVA} (device inner solve only, OLF's Java
     *  outer loops make the per-scenario decisions via {@code GpuHybridBatchOuterLoopEngine}). Matches the
     *  module's other GPU toggles (there is no GPU parameters extension). Read in {@link GpuSaRunScope}. */
    public static final String GPU_OUTER_LOOP_MODE_PROPERTY = "olf.gpu.sa.outerLoopMode";

    /** How many contingencies the run solved on the GPU device: {@code -1} when no batch was attempted
     *  at all (no device, nothing to batch), {@code 0+} otherwise. Test-visibility hook to assert the
     *  GPU actually engaged on a device-equipped machine (vs a silent CPU fallback), and the source of
     *  the GPU/CPU split the validation harness reports.
     *
     *  <p>ACCUMULATED across the run's precomputes, not overwritten by each: a threadCount &gt; 1 run
     *  partitions its contingencies and precomputes one batch per partition (and a multi-component
     *  network precomputes one per component), so the last precompute's count is a fraction of the
     *  run's. The provider resets it to the sentinel before each run, which is what makes summing
     *  well-defined. */
    /** Wall-clock spent in the phases of the GPU SA path OUTSIDE the batched solve, accumulated over
     *  the run's contingencies and printed under {@code OLF_GPU_PHASE_TIME} beside the batch's own
     *  split. Most of a batched run's wall time is not the batch: the engine still applies each
     *  contingency, injects the precomputed state, updates the network and detects violations one
     *  contingency at a time, and that tail is invisible in the batch timings. */
    private static final ThreadLocal<Long> RUN_START_NS = new ThreadLocal<>();

    static final java.util.Map<String, java.util.concurrent.atomic.LongAdder> PHASE_NS =
            new java.util.concurrent.ConcurrentSkipListMap<>();

    static final java.util.concurrent.atomic.AtomicInteger LAST_GPU_PRECOMPUTED_COUNT =
            new java.util.concurrent.atomic.AtomicInteger(-1);

    /** Separator inside the precomputed-state key of a (contingency, strategy) pair — the SOLVER's
     *  constant, which also drives its two-phase warm-continuation ordering. */
    private static final char STRATEGY_KEY_SEP = GpuBatchStatus.STRATEGY_KEY_SEP;

    /** The per-RUN mutable state, one holder PER PARTITION THREAD. Under threadCount > 1 OLF invokes
     *  {@code runSimulations} CONCURRENTLY on this same analysis instance (one call per contingency
     *  partition, each with its own LfNetwork), and every hook this class overrides runs on the
     *  partition's own thread inside that call stack — so the state is ThreadLocal, never instance
     *  fields. As INSTANCE fields these raced: partition A's precompute could read partition B's
     *  lfNetwork/context and LOAD STATES INTO B's equation system while B was mid-solve — one
     *  corrupted scenario (the 19/1 MAX_ITERATION flake) or an NPE inside B's JacobianMatrix on
     *  pure CPU stacks (both observed, threadCountAboveOneRunsSafelyOnGpuProvider). */
    private static final class RunState {
        private List<PropagatedContingency> contingencies = List.of();
        private List<OperatorStrategy> operatorStrategies = List.of();
        private List<Action> actions = List.of();
        private LfNetwork lfNetwork;
        private AcLoadFlowContext context;
        /** The batch precomputed on the device once the base load flow converged (see
         *  GpuBatchPrecomputer); PrecomputedBatch.empty() until then / on CPU fallback. */
        private PrecomputedBatch precomputed = PrecomputedBatch.empty();
        /** The state injector's row permutation, kept across this partition's contingencies. */
        private final GpuStateInjector.Remap remap = new GpuStateInjector.Remap();
        // The run's actual outer loops, captured in runSimulations: the device must run the distributed-slack /
        // reactive-limits passes ONLY when OLF would (i.e. when the outer loop is in the configured list), never
        // unconditionally from network capability. Default to OLF's defaults until runSimulations sets them.
        private boolean batchDistributedSlack = true;
        private boolean batchReactiveLimits = true;
        // The run's outer-loop convergence tolerances (per-unit), captured from the loadflow parameters so the
        // device aligns to the CPU loadflow (GPU follows CPU, not the reverse): slackBusPMaxMismatch /
        // maxReactivePowerMismatch divided by the 100 MVA base. Default to OLF's defaults until runSimulations sets.
        private double batchSlackDistTol = GpuBatchedSecurityAnalysisSolver.DEFAULT_SLACK_DIST_TOL;
        private double batchReactiveLimitTol = GpuBatchedSecurityAnalysisSolver.DEFAULT_REACTIVE_LIMIT_TOL;
        // The device inner-Newton stopping criterion, matched to OLF's: for UNIFORM_CRITERIA (default) the device
        // stops on OLF's exact ‖F‖₂ < sqrt(convEpsPerEq²·n) so the reactive-limit switch DECISIONS (whose tolerance
        // is convEpsPerEq for UNIFORM) are made on a state converged to the SAME depth as the CPU — else the device
        // switched on a 1e-8-converged state while the RL threshold was convEpsPerEq (1e-12), flipping borderline
        // buses vs the CPU. 0 = the device infinity-norm floor (kept for PER_EQUATION_TYPE, whose per-eq criteria
        // are not ported to the device).
        private double batchConvEpsPerEq;
        // The run's batched outer-loop parameterization, captured per run and assembled into a BatchConfig at
        // the solveBatchN1 call. This replaced the solver's former JVM-global static config, whose leak between
        // a provider run and a direct-solver run caused the pegase full-suite failure — see BatchConfig's javadoc.
        private BatchConfig batchConfig = BatchConfig.DEVICE_DEFAULTS;
        private com.powsybl.loadflow.LoadFlowParameters loadFlowParameters;
    }

    private final ThreadLocal<RunState> runState = ThreadLocal.withInitial(RunState::new);

    protected GpuBatchedAcSecurityAnalysis(Network network, MatrixFactory matrixFactory,
                                           GraphConnectivityFactory<LfBus, LfBranch> connectivityFactory,
                                           List<StateMonitor> stateMonitors, ReportNode reportNode) {
        super(network, matrixFactory, connectivityFactory, stateMonitors, reportNode);
    }

    @Override
    protected SecurityAnalysisResult runSimulations(LfNetwork lfNetwork, List<PropagatedContingency> propagatedContingencies,
                                                    AcLoadFlowParameters acParameters, SecurityAnalysisParameters securityAnalysisParameters,
                                                    List<OperatorStrategy> operatorStrategies, List<Action> actions,
                                                    List<LimitReduction> limitReductions,
                                                    ContingencyActivePowerLossDistribution contingencyActivePowerLossDistribution) {
        // The GPU path is MONO-THREADED BY DESIGN (one device; its parallelism IS the batch), but
        // threadCount > 1 stays useful for the CPU side: partition workers run their sequential
        // per-contingency (fallback) solves in parallel, while each partition's batch precompute
        // queues on the global device lock (see precomputeBatch). Under multithreading the
        // base/fall-through GPU solver swap is ALSO skipped — concurrent partition load flows on
        // the single-device sequential solver would race in native code (SIGSEGV in cuLaunchKernel,
        // measured 2026-07-17).
        RUN_START_NS.set(System.nanoTime());
        boolean multiThread = OpenSecurityAnalysisParameters.getOrDefault(securityAnalysisParameters).getThreadCount() > 1;
        RunState rs = new RunState();
        runState.set(rs);
        rs.loadFlowParameters = securityAnalysisParameters.getLoadFlowParameters();
        // STRUCTURAL: run the base (pre-contingency) load flow on the GPU solver when the device
        // is available and the user left the default CPU Newton-Raphson. This absorbs the one-time
        // CUDA init into the base phase (so the batched contingency pass that follows is warm) and
        // solves the base faster by keeping it on the device. An
        // explicit non-default solver choice (e.g. FAST_DECOUPLED) is respected; a network the GPU
        // extractor rejects degrades to the CPU solver inside GpuNewtonRaphson (cpuFallback).
        // DETECT-AND-FALLBACK (the provider's safety contract: enabling it must never change results).
        // The batched device path implements ONLY distributed slack + reactive limits; any OTHER loop
        // in the resolved list (e.g. transformer voltage control under the RTE production parameters)
        // would be silently dropped per scenario — measured 2026-07-17 as thousands of non-border
        // violation diffs vs the CPU. The HVDC-emulation loops are tolerated when the network has no
        // HVDC (structural no-ops). On fallback the provider behaves EXACTLY like the stock CPU
        // AcSecurityAnalysis — no batch AND no solver swap (the sequential GPU-inner-solver path has
        // an open CME bug under outer-loop switching).
        GpuSaRunScope.Scope scope = GpuSaRunScope.evaluate(lfNetwork, acParameters, securityAnalysisParameters);
        if (scope.batchDisabled()) {
            LOGGER.info("GPU batched SA: outer loops {} are not implemented in the batched device path — "
                    + "falling back to the sequential CPU path for all contingencies", scope.unsupportedOuterLoops());
            if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
                System.err.println("SCOPE_FALLBACK " + scope.unsupportedOuterLoops());   // test logback is broken
            }
        }
        // STRUCTURAL: run the base (pre-contingency) load flow on the GPU solver when the device is available
        // and the user left the default CPU Newton-Raphson — absorbs the one-time CUDA init into the base
        // phase (so the batched contingency pass is warm). Skipped under multithreading (concurrent partition
        // load flows on the single-device sequential solver race in native code) and when the batch is
        // disabled by the outer-loop guard (then the provider behaves exactly like the stock CPU SA).
        // startWithFrozenACEmulation (frozen HVDCs in the base solve) runs on the sequential path
        // since 2026-09-08: the batch-of-1 route derives the hv element disabled bits from the live
        // AC-emulation statuses per solve (a non-LINEAR status = coupling off, injections in the
        // re-extracted targets — exactly OLF's own term deactivation). The LAST cpu-base-only
        // condition is gone: base-on-GPU runs whenever the batch itself is enabled.
        // MONITORING-SVC mid-base activation solves on the sequential path since the batch-of-1
        // route (2026-09-08): the VoltageMonitoring loop's activation flips es equations, the
        // activation re-extract carries the new row modes — no CPU-base condition anymore
        // (GpuLoadFlowTest.monitoringSvcActivatesInBase).
        // SLOPE-SVC rows (row mode 2) solve on the sequential path since the batch-of-1 route
        // (2026-09-08) — no CPU-base condition anymore.
        // CONTROLLER-mode phase control (row mode 3) solves on the sequential path since the
        // batch-of-1 route (2026-09-08) — no CPU-base condition anymore.
        // PER_EQUATION_TYPE stopping criteria run on the sequential path since the batch-of-1
        // route (2026-09-08): the batch-of-1 solve installs the per-row thresholds
        // (setBatchRowCriteria) and stops on the SAME decision as the CPU — no CPU-base condition.
        if (!scope.batchDisabled() && !multiThread && gpuBaseLoadFlowEnabled()
                && GpuAcNewtonSolver.isAvailable()
                && acParameters.getSolverFactory().getClass() == NewtonRaphsonFactory.class) {
            LOGGER.info("GPU-batched SA: running the base load flow on the GPU Newton-Raphson "
                    + "(absorbs CUDA init; CPU fallback inside the solver)");
            acParameters.setSolverFactory(new GpuNewtonRaphsonFactory(),
                    securityAnalysisParameters.getLoadFlowParameters());
        }
        // Capture this component's network + contingencies; the batch is precomputed once the
        // pre-contingency load flow has converged (afterPreContingencySimulation hook).
        rs.lfNetwork = lfNetwork;
        rs.contingencies = propagatedContingencies;
        rs.operatorStrategies = operatorStrategies != null ? operatorStrategies : List.of();
        rs.actions = actions != null ? actions : List.of();
        rs.precomputed = PrecomputedBatch.empty();
        rs.batchDistributedSlack = scope.distributedSlack();
        rs.batchReactiveLimits = scope.reactiveLimits();
        if (scope.batchDisabled()) {                         // outer-loop guard: also skip the solver swap below
            rs.lfNetwork = null;
            rs.contingencies = null;
        }
        rs.batchSlackDistTol = scope.slackDistTol();
        rs.batchReactiveLimitTol = scope.reactiveLimitTol();
        rs.batchConvEpsPerEq = scope.convEpsPerEq();
        rs.batchConfig = scope.batchConfig();
        // Run the engine's load flows — the pre-contingency BASE and any contingency that falls through to
        // the sequential path — on the FULL GPU Newton-Raphson when the device + network support it. The base
        // then solves on-device and PRE-WARMS CUDA for the batched contingency pass (the one-time device init
        // is absorbed into the base phase, so the cold batch becomes warm — the base-case warm-up
        // strategy). Guarded by a GPU-support probe so an unsupported network degrades gracefully to the CPU
        // solver. Toggle with -Dolf.gpu.sa.gpuBaseLoadFlow=false.
        // When the batch is disabled by the outer-loop guard, ALSO skip the solver swap: the provider then
        // behaves exactly like the stock CPU AcSecurityAnalysis (the safety contract), and the sequential
        // per-contingency runs avoid the GPU-inner-solver + outer-loop-switching path (open CME bug there).
        if (!scope.batchDisabled() && gpuBaseLoadFlowEnabled()
                && GpuAcNewtonSolver.isAvailable() && GpuSaRunScope.gpuSupports(lfNetwork, acParameters)) {
            acParameters.setSolverFactory(new GpuNewtonRaphsonFactory(), securityAnalysisParameters.getLoadFlowParameters());
            LOGGER.info("GPU batched SA: base + fall-through load flows run on the GPU Newton-Raphson (CUDA pre-warm)");
        }
        try {
            return super.runSimulations(lfNetwork, propagatedContingencies, acParameters, securityAnalysisParameters,
                    operatorStrategies, actions, limitReductions, contingencyActivePowerLossDistribution);
        } finally {
            runState.remove();                               // pooled partition threads must not leak a run's state
            if (System.getenv("OLF_GPU_PHASE_TIME") != null) {
                // CUMULATIVE over the JVM, never cleared: threadCount > 1 runs one partition per
                // thread through this same static map, and clearing here would hand whichever
                // partition finished first the whole run's numbers and the rest almost nothing.
                // Read the LAST block printed — like JNI_PHASE_TIME, it is the running total.
                PHASE_NS.forEach((k, v) ->
                        System.err.printf("SA_PHASE_TIME  %-40s %6d ms (cumulative)%n", k, v.sum() / 1_000_000));
            }
        }
    }

    /** System property toggle for the GPU base/fall-through load flow (default on). */
    public static final String GPU_BASE_LOADFLOW_PROPERTY = "olf.gpu.sa.gpuBaseLoadFlow";

    /** System property: keep the GPU solver for SEQUENTIAL fall-through contingency solves
     *  (default off = revert to the CPU Newton-Raphson after the base LF). */
    public static final String GPU_SEQUENTIAL_LOADFLOW_PROPERTY = "olf.gpu.sa.gpuSequentialLoadFlow";

    private static boolean gpuSequentialLoadFlowEnabled() {
        return Boolean.parseBoolean(System.getProperty(GPU_SEQUENTIAL_LOADFLOW_PROPERTY, "false"));
    }

    private static boolean gpuBaseLoadFlowEnabled() {
        return Boolean.parseBoolean(System.getProperty(GPU_BASE_LOADFLOW_PROPERTY, "true"));
    }

    @Override
    protected AcLoadFlowContext createLoadFlowContext(LfNetwork lfNetwork, AcLoadFlowParameters parameters) {
        // Capture the engine's load-flow context so precomputeBatch can REUSE its equation system instead of
        // creating a second one. Creating a second AcEquationSystem re-points every bus/branch evaluable
        // (bus.setP/setQ, branch.setP1/P2, …) off the engine's system, so the CPU solver of a fall-through
        // contingency would then evaluate the batch system's stale-indexed equations (an index -1 crash, or a
        // mis-converged generator/N-2 outage). The context is created per component, just before the base
        // load flow and afterPreContingencySimulation, and lives through the whole contingency loop.
        AcLoadFlowContext context = super.createLoadFlowContext(lfNetwork, parameters);
        runState.get().context = context;
        return context;
    }

    @Override
    protected void afterPreContingencySimulation(AcLoadFlowParameters parameters) {
        super.afterPreContingencySimulation(parameters);
        Long t0 = RUN_START_NS.get();
        if (t0 != null) {
            PHASE_NS.computeIfAbsent("base load flow + setup", k -> new java.util.concurrent.atomic.LongAdder())
                    .add(System.nanoTime() - t0);
        }
        RunState rs = runState.get();
        if (System.getenv("OLF_GPU_Q_PROBE") != null && rs.lfNetwork != null) {
            String qp = System.getenv("OLF_GPU_Q_PROBE");
            for (LfBus b : rs.lfNetwork.getBuses()) {
                if (b.getId().startsWith(qp)) {
                    System.err.printf("Q_PROBE_BASE bus=%s q=%.9f maxQ=%.9f%n",
                            b.getId(), b.getQ().eval() + b.getLoadTargetQ(), b.getMaxQ());
                }
            }
        }
        long tPre = System.nanoTime();
        rs.precomputed = GpuBatchPrecomputer.precompute(network, rs.lfNetwork, rs.context,
                rs.contingencies, rs.operatorStrategies, rs.actions, rs.batchDistributedSlack,
                rs.batchReactiveLimits, rs.batchReactiveLimitTol, rs.batchSlackDistTol, rs.batchConvEpsPerEq,
                rs.batchConfig, reportNode);
        PHASE_NS.computeIfAbsent("precompute (extraction + batched solves)", k -> new java.util.concurrent.atomic.LongAdder())
                .add(System.nanoTime() - tPre);
        if (rs.precomputed.isReady()) {
            int solved = rs.precomputed.states().size();
            LAST_GPU_PRECOMPUTED_COUNT.updateAndGet(prev -> Math.max(prev, 0) + solved);
        }
        // Sequential fall-through routing is an OPTION (2026-07-17): by DEFAULT the engine's
        // solver reverts to the stock CPU Newton-Raphson after the base LF — single-system GPU
        // solves are not where the device wins (its parallelism is the batch), and the CPU path
        // parallelizes under threadCount. Set -Dolf.gpu.sa.gpuSequentialLoadFlow=true to keep
        // the GPU solver for the sequential fall-through solves instead. The base LF has already
        // run at this point (on the GPU when enabled, absorbing CUDA init).
        if (!gpuSequentialLoadFlowEnabled()
                && parameters.getSolverFactory() instanceof GpuNewtonRaphsonFactory && rs.loadFlowParameters != null) {
            parameters.setSolverFactory(new NewtonRaphsonFactory(), rs.loadFlowParameters);
        }
    }

    /**
     * Solve the qualified single-branch contingencies in one GPU batch, from the converged
     * base state. Failures (no GPU, an out-of-scope network the extractor rejects, a device
     * error) leave the cache empty so every contingency falls back to the CPU path.
     */
    /** ONE device: concurrent partitions (threadCount > 1) queue their batch precompute here. */

    @Override
    protected OperatorStrategyResult runActionSimulation(LfNetwork network, AcLoadFlowContext context, OperatorStrategy operatorStrategy,
                                                         java.util.List<String> actionsIds,
                                                         LimitViolationManager preContingencyLimitViolationManager,
                                                         SecurityAnalysisParameters securityAnalysisParameters,
                                                         Map<String, com.powsybl.openloadflow.network.action.LfAction> lfActionById,
                                                         boolean createResultExtension, LfContingency lfContingency,
                                                         Contingency contingency, PreContingencyNetworkResult preContingencyNetworkResult,
                                                         com.powsybl.openloadflow.network.LfNetworkParameters networkParameters,
                                                         List<LimitReduction> limitReductions) {
        String scId = contingency.getId() + STRATEGY_KEY_SEP + operatorStrategy.getId();
        PrecomputedBatch precomputed = runState.get().precomputed;
        double[] state = precomputed.states().get(scId);
        if (state == null || !precomputed.isReady()) {
            // pair not precomputed (condition not TrueCondition / out-of-scope action / diverged) — CPU path
            return super.runActionSimulation(network, context, operatorStrategy, actionsIds, preContingencyLimitViolationManager,
                    securityAnalysisParameters, lfActionById, createResultExtension, lfContingency, contingency,
                    preContingencyNetworkResult, networkParameters, limitReductions);
        }
        // Apply the actions' NETWORK effects (topology/limits bookkeeping — no solve), inject the
        // precomputed post-action state, and reuse the engine's result building — the same pattern
        // as the post-contingency injection.
        List<com.powsybl.openloadflow.network.action.LfAction> lfActions = actionsIds.stream()
                .map(lfActionById::get)
                .filter(Objects::nonNull)
                .toList();
        com.powsybl.openloadflow.network.action.LfActionUtils.applyListOfActions(lfActions, network, lfContingency, networkParameters);
        GpuStateInjector.injectState(precomputed.equationSystem(), precomputed.baseVariableRows(), state,
                runState.get().remap);
        AcSolverUtil.updateNetwork(network, precomputed.equationSystem());
        var postActionsViolationManager = new LimitViolationManager(preContingencyLimitViolationManager, limitReductions,
                securityAnalysisParameters.getIncreasedViolationsParameters());
        LoadFlowModel loadFlowModel = securityAnalysisParameters.getLoadFlowParameters().isDc() ? LoadFlowModel.DC : LoadFlowModel.AC;
        var postActionsNetworkResult = new PostContingencyNetworkResult(network,
                new AbstractNetworkResult.StateMonitorIndexes(monitorIndex, zeroImpedanceMonitoredIndex), createResultExtension,
                preContingencyNetworkResult, contingency, loadFlowModel, securityAnalysisParameters.getLoadFlowParameters().getDcPowerFactor(),
                securityAnalysisParameters.getModifiedMonitoredElementsParameters());
        postActionsNetworkResult.update();
        postActionsViolationManager.detectViolations(network);
        // Stage 1 reports no ADDITIONAL distributed power for the action solve (the batch already ran
        // its device-side DS within the scenario) — mirrors the post-contingency injection convention.
        return new OperatorStrategyResult(operatorStrategy,
                List.of(new OperatorStrategyResult.ConditionalActionsResult(operatorStrategy.getId(),
                        PostContingencyComputationStatus.CONVERGED,
                        new LimitViolationsResult(postActionsViolationManager.getLimitViolations()),
                        GpuStateInjector.filterIslanded(network, precomputed.islandedBuses().get(scId), scId,
                                new NetworkResult(postActionsNetworkResult.getBranchResults(), postActionsNetworkResult.getBusResults(),
                                        postActionsNetworkResult.getThreeWindingsTransformerResults())),
                        0.0)));
    }

    @Override
    protected PostContingencyResult runPostContingencySimulation(LfNetwork network, AcLoadFlowContext context, Contingency contingency,
                                                                 LfContingency lfContingency, LimitViolationManager preContingencyLimitViolationManager,
                                                                 SecurityAnalysisParameters securityAnalysisParameters,
                                                                 PreContingencyNetworkResult preContingencyNetworkResult, boolean createResultExtension,
                                                                 List<LimitReduction> limitReductions, double preDistributedActivePower) {
        PrecomputedBatch precomputed = runState.get().precomputed;
        double[] state = precomputed.states().get(contingency.getId());
        if (state == null || !precomputed.isReady()) {
            // not batched (multi-element / islanding / out of scope / diverged) — standard CPU path
            return super.runPostContingencySimulation(network, context, contingency, lfContingency, preContingencyLimitViolationManager,
                    securityAnalysisParameters, preContingencyNetworkResult, createResultExtension, limitReductions, preDistributedActivePower);
        }

        // Inject the precomputed post-contingency state (the branch is already disabled by
        // lfContingency.apply()), then reuse the engine's result-building unchanged.
        long tPost = System.nanoTime();
        logPostContingencyStart(network, lfContingency);
        Stopwatch stopwatch = Stopwatch.createStarted();

        long tInj = System.nanoTime();
        GpuStateInjector.injectState(precomputed.equationSystem(), precomputed.baseVariableRows(), state,
                runState.get().remap);
        long tUpd = System.nanoTime();
        PHASE_NS.computeIfAbsent("   injectState()", k -> new java.util.concurrent.atomic.LongAdder())
                .add(tUpd - tInj);
        AcSolverUtil.updateNetwork(network, precomputed.equationSystem());
        PHASE_NS.computeIfAbsent("   AcSolverUtil.updateNetwork()", k -> new java.util.concurrent.atomic.LongAdder())
                .add(System.nanoTime() - tUpd);
        PHASE_NS.computeIfAbsent("  state injection + network update", k -> new java.util.concurrent.atomic.LongAdder())
                .add(System.nanoTime() - tInj);

        // OLF_GPU_Q_PROBE=<busId>: the device's post-contingency q at a controller bus, computed with
        // OLF's OWN expression (ReactiveLimitsOuterLoop.checkControllerBus:
        // getQ().eval() + getLoadTargetQ()) against the same limits, on the state just injected. This
        // is what decides a PV -> PQ switch, so it is the number to compare when the two arms pin
        // different controllers.
        String qProbe = System.getenv("OLF_GPU_Q_PROBE");
        if (qProbe != null) {
            for (LfBus b : network.getBuses()) {
                if (java.util.Arrays.stream(qProbe.split(",")).anyMatch(b.getId()::startsWith)) {
                    double q = b.getQ().eval() + b.getLoadTargetQ();
                    // DISTR_Q shares the DEVIATION from targetQ, not q itself: its target is
                    // (qPct-1)*targetQ_i + qPct*sum_j targetQ_j (AcTargetVector), so two members with
                    // different targetQ legitimately settle at different q. Print the parts, and the
                    // equation's OWN residual at the injected state — the only number that says
                    // whether the device honoured the constraint or solved a different one.
                    String resid = precomputed.equationSystem()
                            .getEquation(b.getNum(), AcEquationType.DISTR_Q)
                            .map(e -> String.format("%.9f", e.eval() - distrQTarget(network, b)))
                            .orElse("n/a");
                    System.err.printf("Q_PROBE ctg=%s bus=%s q=%.9f genTgtQ=%.9f loadTgtQ=%.9f qPct=%.9f "
                            + "distrQResid=%s minQ=%.9f maxQ=%.9f overMaxBy=%.9f%n",
                            contingency.getId(), b.getId(), q, b.getGenerationTargetQ(), b.getLoadTargetQ(),
                            b.getRemoteControlReactivePercent(), resid, b.getMinQ(), b.getMaxQ(), q - b.getMaxQ());
                    // A CURVE reactive limit is a function of the GENERATOR's targetP, so a limit that
                    // disagrees between arms is a P disagreement, not a limit-model disagreement.
                    for (var g : b.getGenerators()) {
                        System.err.printf("Q_PROBE_GEN ctg=%s bus=%s gen=%s targetP=%.9f minQ=%.9f maxQ=%.9f%n",
                                contingency.getId(), b.getId(), g.getId(), g.getTargetP(), g.getMinQ(), g.getMaxQ());
                    }
                }
            }
        }
        // A converged device solve is not automatically a CONVERGED post-contingency RESULT: OLF's
        // AcloadFlowEngine rejects a solution whose voltages are outside the realistic band, and a
        // rejected result reports NO violations at all. Skipping that check does not make the GPU
        // arm optimistic by a violation or two — it makes it report a full violation set where the
        // CPU reports none.
        PostContingencyComputationStatus status = isStateUnrealistic(precomputed.equationSystem(), network,
                context.getParameters())
                ? PostContingencyComputationStatus.FAILED
                : PostContingencyComputationStatus.CONVERGED;
        var postContingencyLimitViolationManager = new LimitViolationManager(preContingencyLimitViolationManager, limitReductions,
                securityAnalysisParameters.getIncreasedViolationsParameters());
        LoadFlowModel loadFlowModel = securityAnalysisParameters.getLoadFlowParameters().isDc() ? LoadFlowModel.DC : LoadFlowModel.AC;
        var postContingencyNetworkResult = new PostContingencyNetworkResult(network,
                new AbstractNetworkResult.StateMonitorIndexes(monitorIndex, zeroImpedanceMonitoredIndex), createResultExtension,
                preContingencyNetworkResult, contingency, loadFlowModel, securityAnalysisParameters.getLoadFlowParameters().getDcPowerFactor(),
                securityAnalysisParameters.getModifiedMonitoredElementsParameters());
        if (status != PostContingencyComputationStatus.CONVERGED) {
            // OLF only builds the network result and detects violations for a CONVERGED result
            // (AbstractSecurityAnalysis.runPostContingencySimulation) — an unrealistic state reports
            // an empty violation list, not the violations of a state it rejected.
            PHASE_NS.computeIfAbsent(" post-contingency call (all of it)", k -> new java.util.concurrent.atomic.LongAdder())
                    .add(System.nanoTime() - tPost);
            return new PostContingencyResult(contingency, status, new LimitViolationsResult(List.of()),
                    new NetworkResult(List.of(), List.of(), List.of()),
                    new ConnectivityResult(lfContingency.getCreatedSynchronousComponentsCount(), 0,
                            lfContingency.getDisconnectedLoadActivePower() * PerUnit.SB,
                            lfContingency.getDisconnectedGenerationActivePower() * PerUnit.SB,
                            lfContingency.getDisconnectedElementIds()),
                    preDistributedActivePower * PerUnit.SB);
        }
        long tRes = System.nanoTime();
        postContingencyNetworkResult.update();
        long tDet = System.nanoTime();
        PHASE_NS.computeIfAbsent("   networkResult.update()", k -> new java.util.concurrent.atomic.LongAdder())
                .add(tDet - tRes);
        postContingencyLimitViolationManager.detectViolations(network);
        PHASE_NS.computeIfAbsent("   detectViolations()", k -> new java.util.concurrent.atomic.LongAdder())
                .add(System.nanoTime() - tDet);
        PHASE_NS.computeIfAbsent("  result build + violation detection", k -> new java.util.concurrent.atomic.LongAdder())
                .add(System.nanoTime() - tRes);

        stopwatch.stop();
        logPostContingencyEnd(network, lfContingency, stopwatch);

        var connectivityResult = new ConnectivityResult(lfContingency.getCreatedSynchronousComponentsCount(), 0,
                lfContingency.getDisconnectedLoadActivePower() * PerUnit.SB,
                lfContingency.getDisconnectedGenerationActivePower() * PerUnit.SB,
                lfContingency.getDisconnectedElementIds());

        // No device-side slack distribution in the prototype: only the pre-distribution counts.
        PHASE_NS.computeIfAbsent(" post-contingency call (all of it)", k -> new java.util.concurrent.atomic.LongAdder())
                .add(System.nanoTime() - tPost);
        return new PostContingencyResult(contingency, status,
                new LimitViolationsResult(postContingencyLimitViolationManager.getLimitViolations()),
                GpuStateInjector.filterIslanded(network, precomputed.islandedBuses().get(contingency.getId()), contingency.getId(),
                        new NetworkResult(postContingencyNetworkResult.getBranchResults(), postContingencyNetworkResult.getBusResults(),
                                postContingencyNetworkResult.getThreeWindingsTransformerResults())),
                connectivityResult, preDistributedActivePower * PerUnit.SB);
    }

    /**
     * OLF's realistic-voltage check, ported from {@code AcloadFlowEngine.isStateUnrealistic}: every
     * non-fictitious {@code BUS_V} variable whose value leaves
     * {@code [minRealisticVoltage, maxRealisticVoltage]} counts, but only on buses whose nominal
     * voltage reaches {@code minNominalVoltageRealisticVoltageCheck}. One such bus makes the whole
     * run {@code UNREALISTIC_STATE}, which the security analysis maps to {@code FAILED}.
     *
     * <p>Walks the variables and reads the STATE VECTOR, exactly as OLF does, rather than reading
     * {@code bus.getV()} — same numbers here, but the point of a port is not to paraphrase.
     *
     * <p>{@code OLF_GPU_UNREALISTIC_TRACE} names the offending buses.
     */
    private static boolean isStateUnrealistic(EquationSystem<AcVariableType, AcEquationType> es,
                                              LfNetwork network, AcLoadFlowParameters parameters) {
        double minNominalV = parameters.getMinNominalVoltageRealisticVoltageCheck();
        boolean trace = System.getenv("OLF_GPU_UNREALISTIC_TRACE") != null;
        boolean unrealistic = false;
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            if (v.getType() != AcVariableType.BUS_V || network.getBus(v.getElementNum()).isFictitious()) {
                continue;
            }
            double value = es.getStateVector().get(v.getRow());
            if ((value < parameters.getMinRealisticVoltage() || value > parameters.getMaxRealisticVoltage())
                    && network.getBus(v.getElementNum()).getNominalV() >= minNominalV) {
                unrealistic = true;
                if (!trace) {
                    return true;
                }
                System.err.printf("UNREALISTIC bus=%s v=%.9f nominalV=%.3f band=[%.3f, %.3f]%n",
                        network.getBus(v.getElementNum()).getId(), value,
                        network.getBus(v.getElementNum()).getNominalV(),
                        parameters.getMinRealisticVoltage(), parameters.getMaxRealisticVoltage());
            }
        }
        return unrealistic;
    }

    /** OLF's DISTR_Q target, reproduced from {@code AcTargetVector} (which keeps it private):
     *  {@code (qPct_i - 1) * targetQ_i + qPct_i * sum_{j != i} targetQ_j}. Diagnostic only. */
    private static double distrQTarget(LfNetwork network, LfBus controllerBus) {
        double qPct = controllerBus.getRemoteControlReactivePercent();
        double target = (qPct - 1) * controllerBus.getTargetQ();
        var members = controllerBus.getGeneratorVoltageControl().isPresent()
                ? controllerBus.getGeneratorVoltageControl().orElseThrow().getMergedControllerElements()
                : controllerBus.getGeneratorReactivePowerControl().orElseThrow().getControllerBuses();
        for (LfBus other : members) {
            if (other != controllerBus) {
                target += qPct * other.getTargetQ();
            }
        }
        return target;
    }
}
