/*
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa;

import com.google.common.base.Stopwatch;
import com.powsybl.action.Action;
import com.powsybl.commons.report.ReportNode;
import com.powsybl.contingency.Contingency;
import com.powsybl.contingency.strategy.OperatorStrategy;
import com.powsybl.iidm.network.Network;
import com.powsybl.math.matrix.MatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcLoadFlowContext;
import com.powsybl.openloadflow.ac.AcLoadFlowParameters;
import com.powsybl.openloadflow.ac.equations.AcEquationSystemCreator;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.outerloop.DistributedSlackOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop;
import com.powsybl.openloadflow.ac.solver.AcSolverUtil;
import com.powsybl.openloadflow.ac.solver.GpuBatchedSecurityAnalysisSolver;
import com.powsybl.openloadflow.ac.solver.GpuBatchedSecurityAnalysisSolver.BatchContingency;
import com.powsybl.openloadflow.ac.solver.GpuBatchedSecurityAnalysisSolver.BatchScenarioResult;
import com.powsybl.openloadflow.ac.solver.GpuNewtonRaphsonFactory;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonFactory;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.graph.GraphConnectivityFactory;
import com.powsybl.openloadflow.network.DisabledBranchStatus;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfContingency;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfShunt;
import com.powsybl.openloadflow.network.LoadFlowModel;
import com.powsybl.openloadflow.network.impl.PropagatedContingency;
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
import java.util.Objects;
import com.powsybl.security.results.PostContingencyResult;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Prototype integration of the batched GPU N-1 ({@link GpuBatchedSecurityAnalysisSolver})
 * into OLF's AC security-analysis engine. Single-branch contingencies whose post-contingency
 * states the GPU batch can solve (qualified by {@link com.powsybl.openloadflow.ac.solver.GpuBatchContingencyValidator})
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

    /** How many contingencies the LAST {@link #precomputeBatch()} solved on the GPU device: {@code -1} when
     *  the batch was not attempted (no device, nothing to batch), {@code 0+} otherwise. Test-visibility hook
     *  to assert the GPU actually engaged on a device-equipped machine (vs a silent CPU fallback). */
    static final java.util.concurrent.atomic.AtomicInteger LAST_GPU_PRECOMPUTED_COUNT =
            new java.util.concurrent.atomic.AtomicInteger(-1);

    private List<PropagatedContingency> currentContingencies = List.of();
    private List<OperatorStrategy> currentOperatorStrategies = List.of();
    private List<Action> currentActions = List.of();
    /** Separator inside the precomputed-state key of a (contingency, strategy) pair — the SOLVER's
     *  constant, which also drives its two-phase warm-continuation ordering. */
    private static final char STRATEGY_KEY_SEP = GpuBatchedSecurityAnalysisSolver.STRATEGY_KEY_SEP;
    private LfNetwork currentLfNetwork;
    private AcLoadFlowContext currentContext;
    private EquationSystem<AcVariableType, AcEquationType> batchEquationSystem;
    private final Map<String, double[]> precomputedStates = new HashMap<>();
    /** Islanded LfBus nums per precomputed scenario id: the CPU path disables these buses through the
     *  SOLVE's connectivity processing, which the injection path skips — result building must exclude
     *  them explicitly or bus results misalign (an islanded bus reported at its frozen flat state
     *  shifts every following bus in the list — measured on ieee14 L7-8-1). The engine's per-
     *  contingency NetworkState restore reverts the disabled flags afterwards. */
    private final Map<String, int[]> precomputedIslandedBuses = new HashMap<>();
    /** Variable -> row snapshot of the BASE equation-system layout (taken at precompute). The engine's
     *  lfContingency.apply() on an ISLANDING contingency disables the islanded bus, deactivating its
     *  variables and RENUMBERING the rows — a device state (base layout) injected by raw index would
     *  shift every variable after the disabled bus (measured: the bus after the island read the
     *  island's frozen 1-pu slot). Injection remaps through this snapshot instead. */
    private Map<com.powsybl.openloadflow.equations.Variable<com.powsybl.openloadflow.ac.equations.AcVariableType>, Integer> baseVariableRows;
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
    private com.powsybl.loadflow.LoadFlowParameters currentLoadFlowParameters;

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
        boolean multiThread = OpenSecurityAnalysisParameters.getOrDefault(securityAnalysisParameters).getThreadCount() > 1;
        this.currentLoadFlowParameters = securityAnalysisParameters.getLoadFlowParameters();
        // STRUCTURAL: run the base (pre-contingency) load flow on the GPU solver when the device
        // is available and the user left the default CPU Newton-Raphson. This absorbs the one-time
        // CUDA init into the base phase (so the batched contingency pass that follows is warm) and
        // solves the base faster — the same base-on-GPU placement gpusim2grid has natively. An
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
        boolean noHvdc = lfNetwork.getHvdcs().isEmpty();
        // TransformerVoltageControl is BATCHABLE since the per-scenario orchestrator port
        // (AFTER_GENERATOR mode only; stage-1 scope = no generator voltage controls under the
        // transformer-controlled level — the solver checks). Anything else falls back.
        OpenLoadFlowParameters olfGuardParams = OpenLoadFlowParameters.get(securityAnalysisParameters.getLoadFlowParameters());
        boolean transformerControlBatchable =
                olfGuardParams.getTransformerVoltageControlMode() == OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL
                && GpuBatchedSecurityAnalysisSolver.transformerControlSupported(lfNetwork);
        boolean noSvc = lfNetwork.getBuses().stream()
                .flatMap(b -> b.getGenerators().stream())
                .noneMatch(g -> g instanceof com.powsybl.openloadflow.network.LfStaticVarCompensator);
        List<String> unsupportedOuterLoops = new java.util.ArrayList<>(acParameters.getOuterLoops().stream()
                .map(ol -> ol.getName())
                .filter(name -> !DistributedSlackOuterLoop.NAME.equals(name)
                        && !ReactiveLimitsOuterLoop.NAME.equals(name)
                        && !(transformerControlBatchable && "TransformerVoltageControl".equals(name))
                        && !(noHvdc && ("AcHvdcAcEmulationLimits".equals(name) || "FreezingHvdcACEmulation".equals(name)))
                        && !(noSvc && "VoltageMonitoring".equals(name)))
                .toList());
        // Parameterization corner cases the device DS loop does not model (see
        // docs/gpu-vs-cpu-capabilities.md Part II §A): the device weights are always maxP/droop with
        // active limits — any other balance type or unlimited distribution must fall back whole-run.
        boolean dsLoopActive = acParameters.getOuterLoops().stream()
                .anyMatch(ol -> DistributedSlackOuterLoop.NAME.equals(ol.getName()));
        com.powsybl.loadflow.LoadFlowParameters lfGuardParams = securityAnalysisParameters.getLoadFlowParameters();
        if (dsLoopActive && lfGuardParams.getBalanceType()
                != com.powsybl.loadflow.LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX) {
            unsupportedOuterLoops.add("DistributedSlack[balanceType=" + lfGuardParams.getBalanceType() + "]");
        }
        if (dsLoopActive && !olfGuardParams.isUseActiveLimits()) {
            unsupportedOuterLoops.add("DistributedSlack[useActiveLimits=false]");
        }
        // Parameterization the device loops DO model — wire the run's values in place of the
        // device defaults (maxOuterLoopIterations, reactiveLimitsMaxPqPvSwitch), and make a
        // saturated slack distribution reject its scenario to the CPU fallback whenever the
        // failure behavior is not LEAVE_ON_SLACK_BUS (e.g. RTE production uses FAIL).
        GpuBatchedSecurityAnalysisSolver.setOuterLoopParams(
                olfGuardParams.getMaxOuterLoopIterations(),
                olfGuardParams.getReactiveLimitsMaxPqPvSwitch(),
                olfGuardParams.getSlackDistributionFailureBehavior()
                        != com.powsybl.openloadflow.OpenLoadFlowParameters.SlackDistributionFailureBehavior.LEAVE_ON_SLACK_BUS);
        boolean batchDisabled = !unsupportedOuterLoops.isEmpty();
        if (batchDisabled) {
            LOGGER.info("GPU batched SA: outer loops {} are not implemented in the batched device path — "
                    + "falling back to the sequential CPU path for all contingencies", unsupportedOuterLoops);
        }
        if (!batchDisabled && !multiThread && gpuBaseLoadFlowEnabled() && GpuAcNewtonSolver.isAvailable()
                && acParameters.getSolverFactory().getClass() == NewtonRaphsonFactory.class) {
            LOGGER.info("GPU-batched SA: running the base load flow on the GPU Newton-Raphson "
                    + "(absorbs CUDA init; CPU fallback inside the solver)");
            acParameters.setSolverFactory(new GpuNewtonRaphsonFactory(),
                    securityAnalysisParameters.getLoadFlowParameters());
        }
        // Capture this component's network + contingencies; the batch is precomputed once the
        // pre-contingency load flow has converged (afterPreContingencySimulation hook).
        this.currentLfNetwork = lfNetwork;
        this.currentContingencies = propagatedContingencies;
        this.currentOperatorStrategies = operatorStrategies != null ? operatorStrategies : List.of();
        this.currentActions = actions != null ? actions : List.of();
        this.precomputedStates.clear();
        this.precomputedIslandedBuses.clear();
        this.batchEquationSystem = null;
        // Capture which outer loops this run actually applies, so the batched device solver runs the
        // distributed-slack / reactive-limits passes only when OLF does (presence in the resolved outer-loop
        // list is the definitive signal — gens may report participation / buses may be PV regardless).
        this.batchDistributedSlack = acParameters.getOuterLoops().stream()
                .anyMatch(ol -> DistributedSlackOuterLoop.NAME.equals(ol.getName()));
        this.batchReactiveLimits = acParameters.getOuterLoops().stream()
                .anyMatch(ol -> ReactiveLimitsOuterLoop.NAME.equals(ol.getName()));
        GpuBatchedSecurityAnalysisSolver.setTransformerControl(transformerControlBatchable
                && acParameters.getOuterLoops().stream().anyMatch(ol -> "TransformerVoltageControl".equals(ol.getName())));
        // DETECT-AND-FALLBACK (the provider's safety contract: enabling it must never change results).
        // The batched device path implements ONLY the two outer loops above; any OTHER loop in the
        // resolved list (e.g. transformer voltage control under the RTE production parameters) would be
        // silently dropped per scenario — measured 2026-07-17 as thousands of non-border violation
        // diffs vs the CPU (a controlled bus held at 1.154 pu across unrelated outages). Until those
        // loops are ported, such a run stays entirely on the sequential CPU path (the base +
        // fall-through load flows still run on the GPU inner Newton below, where the Java engine
        // applies every outer loop unchanged).
        if (batchDisabled) {                                 // outer-loop guard (computed above, before the solver swap)
            this.currentLfNetwork = null;
            this.currentContingencies = null;
        }
        // Align the device outer-loop tolerances to the CPU loadflow's (per-unit on the 100 MVA base). The
        // reactive-limits PV→PQ switch threshold is NOT always maxReactivePowerMismatch: OLF derives it from the
        // stopping-criteria type exactly as AbstractAcOuterLoopConfig.createReactiveLimitsOuterLoop does —
        // UNIFORM_CRITERIA (the default) ties it to convEpsPerEq, PER_EQUATION_TYPE_CRITERIA to
        // maxReactivePowerMismatch/SB. Deriving it the same way keeps the device switch DECISIONS identical to
        // OLF when a user changes convEpsPerEq (with defaults both are 1e-4, which masked the divergence).
        OpenLoadFlowParameters olfParams = OpenLoadFlowParameters.get(securityAnalysisParameters.getLoadFlowParameters());
        this.batchSlackDistTol = olfParams.getSlackBusPMaxMismatch() / PerUnit.SB;
        this.batchReactiveLimitTol = switch (olfParams.getNewtonRaphsonStoppingCriteriaType()) {
            case UNIFORM_CRITERIA -> olfParams.getNewtonRaphsonConvEpsPerEq();
            case PER_EQUATION_TYPE_CRITERIA -> olfParams.getMaxReactivePowerMismatch() / PerUnit.SB;
        };
        // Device inner Newton uses OLF's exact criterion for UNIFORM_CRITERIA (so switch decisions are made at
        // the same convergence depth as the CPU); the device infinity-norm floor otherwise.
        this.batchConvEpsPerEq = switch (olfParams.getNewtonRaphsonStoppingCriteriaType()) {
            case UNIFORM_CRITERIA -> olfParams.getNewtonRaphsonConvEpsPerEq();
            case PER_EQUATION_TYPE_CRITERIA -> 0.0;
        };
        // MAX_VOLTAGE_CHANGE state-vector scaling: apply the SAME per-step clamp in the batched device
        // solve as the CPU per-contingency Newton (per scenario). Without this the batch takes raw full
        // steps, PV/PQ switch decisions read different intermediate states, and converged states drift
        // (measured 2026-07-17 under the RTE parameters: 6,137 one-sided violation diffs).
        if (olfParams.getStateVectorScalingMode() == com.powsybl.openloadflow.ac.solver.StateVectorScalingMode.MAX_VOLTAGE_CHANGE) {
            GpuBatchedSecurityAnalysisSolver.setStateVectorScaling(2,
                    olfParams.getMaxVoltageChangeStateVectorScalingMaxDv(),
                    olfParams.getMaxVoltageChangeStateVectorScalingMaxDphi());
        } else {
            GpuBatchedSecurityAnalysisSolver.setStateVectorScaling(0, 0.0, 0.0);
        }
        // Run the engine's load flows — the pre-contingency BASE and any contingency that falls through to
        // the sequential path — on the FULL GPU Newton-Raphson when the device + network support it. The base
        // then solves on-device and PRE-WARMS CUDA for the batched contingency pass (the one-time device init
        // is absorbed into the base phase, so the cold batch becomes warm — the base-case warm-up
        // strategy). Guarded by a GPU-support probe so an unsupported network degrades gracefully to the CPU
        // solver. Toggle with -Dolf.gpu.sa.gpuBaseLoadFlow=false.
        // When the batch is disabled by the outer-loop guard, ALSO skip the solver swap: the provider then
        // behaves exactly like the stock CPU AcSecurityAnalysis (the safety contract), and the sequential
        // per-contingency runs avoid the GPU-inner-solver + outer-loop-switching path (open CME bug there).
        if (!batchDisabled && gpuBaseLoadFlowEnabled() && GpuAcNewtonSolver.isAvailable() && gpuSupports(lfNetwork, acParameters)) {
            acParameters.setSolverFactory(new GpuNewtonRaphsonFactory(), securityAnalysisParameters.getLoadFlowParameters());
            LOGGER.info("GPU batched SA: base + fall-through load flows run on the GPU Newton-Raphson (CUDA pre-warm)");
        }
        return super.runSimulations(lfNetwork, propagatedContingencies, acParameters, securityAnalysisParameters,
                operatorStrategies, actions, limitReductions, contingencyActivePowerLossDistribution);
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

    /** True if the GPU full Newton-Raphson can solve this network — a throwaway extract (which VALIDATES and
     *  throws on anything uncovered) is the same gate the batch uses, so the base never runs on the GPU for a
     *  network whose contingencies would fall back to the CPU anyway. */
    private static boolean gpuSupports(LfNetwork lfNetwork, AcLoadFlowParameters acParameters) {
        try {
            EquationSystem<AcVariableType, AcEquationType> es =
                    new AcEquationSystemCreator(lfNetwork, acParameters.getEquationSystemCreationParameters()).create();
            GpuAcDataExtractor.extract(lfNetwork, es);
            return true;
        } catch (RuntimeException e) {
            LOGGER.debug("GPU batched SA: base load flow stays on CPU (network not GPU-supported: {})", e.toString());
            return false;
        }
    }

    @Override
    protected AcLoadFlowContext createLoadFlowContext(LfNetwork lfNetwork, AcLoadFlowParameters parameters) {
        // Capture the engine's load-flow context so precomputeBatch can REUSE its equation system instead of
        // creating a second one. Creating a second AcEquationSystem re-points every bus/branch evaluable
        // (bus.setP/setQ, branch.setP1/P2, …) off the engine's system, so the CPU solver of a fall-through
        // contingency would then evaluate the batch system's stale-indexed equations (an index -1 crash, or a
        // mis-converged generator/N-2 outage). The context is created per component, just before the base
        // load flow and afterPreContingencySimulation, and lives through the whole contingency loop.
        currentContext = super.createLoadFlowContext(lfNetwork, parameters);
        return currentContext;
    }

    @Override
    protected void afterPreContingencySimulation(AcLoadFlowParameters parameters) {
        super.afterPreContingencySimulation(parameters);
        precomputeBatch();
        // Sequential fall-through routing is an OPTION (2026-07-17): by DEFAULT the engine's
        // solver reverts to the stock CPU Newton-Raphson after the base LF — single-system GPU
        // solves are not where the device wins (its parallelism is the batch), and the CPU path
        // parallelizes under threadCount. Set -Dolf.gpu.sa.gpuSequentialLoadFlow=true to keep
        // the GPU solver for the sequential fall-through solves instead. The base LF has already
        // run at this point (on the GPU when enabled, absorbing CUDA init).
        if (!gpuSequentialLoadFlowEnabled()
                && parameters.getSolverFactory() instanceof GpuNewtonRaphsonFactory && currentLoadFlowParameters != null) {
            parameters.setSolverFactory(new NewtonRaphsonFactory(), currentLoadFlowParameters);
        }
    }

    /**
     * Solve the qualified single-branch contingencies in one GPU batch, from the converged
     * base state. Failures (no GPU, an out-of-scope network the extractor rejects, a device
     * error) leave the cache empty so every contingency falls back to the CPU path.
     */
    /** ONE device: concurrent partitions (threadCount > 1) queue their batch precompute here. */
    private static final Object GPU_DEVICE_LOCK = new Object();

    private void precomputeBatch() {
        synchronized (GPU_DEVICE_LOCK) {
            precomputeBatchLocked();
        }
    }

    private void precomputeBatchLocked() {
        precomputedStates.clear();
        batchEquationSystem = null;
        LAST_GPU_PRECOMPUTED_COUNT.set(-1);                          // not attempted unless we reach the solve
        if (!GpuAcNewtonSolver.isAvailable() || currentLfNetwork == null || currentContext == null
                || currentContingencies.isEmpty()) {
            return;
        }
        try {
            LfNetwork net = currentLfNetwork;
            // REUSE the engine's equation system + target vector (do NOT create a second system — that would
            // re-point the network's evaluables off the engine's system, breaking fall-through CPU solves).
            // After the converged base load flow the engine's state vector holds the base operating point.
            EquationSystem<AcVariableType, AcEquationType> es = currentContext.getEquationSystem();
            double[] baseState = es.getStateVector().get().clone();
            var target = currentContext.getTargetVector();

            List<BatchContingency> batch = new ArrayList<>();
            for (PropagatedContingency pc : currentContingencies) {
                List<LfGenerator> gens = pc.getGeneratorIdsToLose().isEmpty() ? List.of() : generatorOutages(net, pc);
                // ZIP/voltage-dependent load model: per-scenario load removal is not a target override — CPU.
                boolean loadsBatchable = !currentContext.getParameters().getNetworkParameters().isUseLoadModel();
                List<GpuBatchedSecurityAnalysisSolver.LoadLoss> loads =
                        pc.getLoadIdsToLose().isEmpty() ? List.of() : (loadsBatchable ? loadOutages(net, pc) : null);
                List<LfBranch> branches = pc.getBranchIdsToOpen().isEmpty() ? List.of() : closedBranchOutages(net, pc);
                // Shunt section shifts (both directions; full disconnection of any aggregate = minus the
                // section admittance): per-scenario admittance overrides, combinable with everything.
                List<GpuBatchedSecurityAnalysisSolver.ShuntShift> shifts =
                        pc.getShuntIdsToShift().isEmpty() ? List.of() : shuntShifts(net, pc);
                if (branches != null && gens != null && loads != null && shifts != null
                        && (!branches.isEmpty() || !gens.isEmpty() || !loads.isEmpty() || !shifts.isEmpty())) {
                    batch.add(BatchContingency.branchesAndInjections(pc.getContingency().getId(), branches, gens, loads, shifts, List.of()));
                    // Batched OPERATOR STRATEGIES stage 1: TrueCondition (always-applied) strategies
                    // matching this contingency become EXTRA batch scenarios — the (ctg, strategy)
                    // pair is just another BatchContingency with the actions COMPOSED onto the
                    // contingency's deltas (reclose = drop the branch from the outage list;
                    // extra open = extend it; load/gen actions = injection deltas). Unsupported
                    // action types or shapes leave the pair to the CPU path (super).
                    for (OperatorStrategy strategy : currentOperatorStrategies) {
                        if (!strategyApplies(strategy, pc.getContingency().getId())) {
                            continue;
                        }
                        BatchContingency pair = composeStrategyScenario(net, pc.getContingency().getId(), strategy,
                                branches, gens, loads, shifts);
                        if (pair != null) {
                            batch.add(pair);
                        }
                    }
                    continue;
                }
                LfShunt shunt = singleShuntOutage(net, pc);
                if (shunt != null) {
                    batch.add(BatchContingency.shunt(pc.getContingency().getId(), shunt));
                }
            }
            if (batch.isEmpty()) {
                return;
            }

            List<BatchScenarioResult> results = GpuBatchedSecurityAnalysisSolver.solveBatchN1(
                    net, baseState, batch, es, target, new NewtonRaphsonParameters(),
                    batchDistributedSlack, batchReactiveLimits,
                    batchReactiveLimitTol, batchSlackDistTol, batchConvEpsPerEq, reportNode);
            for (BatchScenarioResult r : results) {
                if (r.hasConverged()) {
                    precomputedStates.put(r.contingencyId(), r.state());
                    if (r.islandedBusNums().length > 0) {
                        precomputedIslandedBuses.put(r.contingencyId(), r.islandedBusNums());
                    }
                }
            }
            batchEquationSystem = es;
            baseVariableRows = new HashMap<>();
            for (var v : es.getIndex().getSortedVariablesToFind()) {
                baseVariableRows.put(v, v.getRow());
            }
            LAST_GPU_PRECOMPUTED_COUNT.set(precomputedStates.size());
            LOGGER.info("GPU batched SA: precomputed {} of {} branch (N-1/N-2) contingencies on the device",
                    precomputedStates.size(), batch.size());
        } catch (Exception e) {
            LOGGER.warn("GPU batched SA precompute failed ({}); falling back to per-contingency CPU solves", e.toString());
            precomputedStates.clear();
            batchEquationSystem = null;
        }
    }

    /**
     * The 1–2 closed branches a contingency outages (N-1 or N-2), or null if it is not a clean
     * branch-only (both sides) outage the GPU mask can represent (≥3 branches, partial open, or a
     * bus/generator loss). Each branch is disabled in the per-scenario element mask.
     */
    /** The single generator a contingency loses (and that the GPU batch can represent via a per-scenario
     *  target override), or null if it is not a clean single-generator outage (multi-element, bus loss, or a
     *  generator absent from the LF network). */
    /** The contingency's lost generators (any count, possibly alongside branches and loads), or null
     *  when any of them is unresolvable. Bus losses stay out of scope. */
    private static List<LfGenerator> generatorOutages(LfNetwork net, PropagatedContingency pc) {
        if (pc.getGeneratorIdsToLose().isEmpty() || !pc.getBusIdsToLose().isEmpty()) {
            return null;
        }
        List<LfGenerator> gens = new java.util.ArrayList<>();
        for (String id : pc.getGeneratorIdsToLose()) {
            LfGenerator gen = net.getGeneratorById(id);
            if (gen == null || gen.getBus() == null || gen.getBus().isDisabled()) {
                return null;
            }
            gens.add(gen);
        }
        return gens;
    }

    /** The contingency's lost loads as per-bus injection deltas (pu, from the propagation's PowerShift),
     *  or null when unresolvable. Guards: bus losses out of scope; the ZIP/voltage-dependent load model
     *  makes per-scenario load removal unrepresentable as a target override (checked by the caller). */
    private List<GpuBatchedSecurityAnalysisSolver.LoadLoss> loadOutages(LfNetwork net, PropagatedContingency pc) {
        if (pc.getLoadIdsToLose().isEmpty() || !pc.getBusIdsToLose().isEmpty()) {
            return null;
        }
        List<GpuBatchedSecurityAnalysisSolver.LoadLoss> out = new java.util.ArrayList<>();
        for (Map.Entry<String, com.powsybl.openloadflow.network.PowerShift> e : pc.getLoadIdsToLose().entrySet()) {
            com.powsybl.iidm.network.Load load = network.getLoad(e.getKey());
            if (load == null || load.getTerminal().getBusView().getBus() == null) {
                return null;
            }
            LfBus bus = net.getBusById(load.getTerminal().getBusView().getBus().getId());
            if (bus == null || bus.isDisabled()) {
                return null;
            }
            out.add(new GpuBatchedSecurityAnalysisSolver.LoadLoss(bus, e.getValue().getActive(), e.getValue().getReactive()));
        }
        return out;
    }

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
        double[] state = precomputedStates.get(contingency.getId() + STRATEGY_KEY_SEP + operatorStrategy.getId());
        if (state == null || batchEquationSystem == null) {
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
        injectState(state);
        AcSolverUtil.updateNetwork(network, batchEquationSystem);
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
                        filterIslanded(network, contingency.getId() + STRATEGY_KEY_SEP + operatorStrategy.getId(),
                                new NetworkResult(postActionsNetworkResult.getBranchResults(), postActionsNetworkResult.getBusResults(),
                                        postActionsNetworkResult.getThreeWindingsTransformerResults())),
                        0.0)));
    }

    /** Injects a device state (BASE row layout) into the possibly contingency-mutated equation
     *  system: rows are remapped variable-by-variable through the base snapshot, so a disabled
     *  islanded bus's deactivated variables cannot shift its neighbors. */
    private void injectState(double[] deviceState) {
        var vars = batchEquationSystem.getIndex().getSortedVariablesToFind();
        double[] mapped = new double[vars.size()];
        for (var v : vars) {
            Integer baseRow = baseVariableRows.get(v);
            mapped[v.getRow()] = baseRow != null ? deviceState[baseRow] : 0;
        }
        batchEquationSystem.getStateVector().set(mapped);
    }

    /** Mirrors the CPU solve's connectivity processing for an injected state: islanded buses (frozen
     *  to a clean flat state on the device) must be EXCLUDED from the results the way the CPU's
     *  connectivity-driven disabling excludes them. NEVER by LfBus.setDisabled on the live network —
     *  that fires the equation-system listeners, deactivates rows and shifts the state mapping for
     *  every subsequent injection (measured: the NEXT scenario's bus read its neighbor's slot).
     *  Pure result-space filtering instead. */
    private NetworkResult filterIslanded(LfNetwork network, String scenarioId, NetworkResult result) {
        int[] nums = precomputedIslandedBuses.get(scenarioId);
        if (nums == null) {
            return result;
        }
        java.util.Set<String> islandedBusIds = new java.util.HashSet<>();
        boolean dbg = "1".equals(System.getenv("OLF_GPU_DBG_BUSDUMP"));
        for (int num : nums) {
            LfBus b = network.getBuses().get(num);
            b.createBusResults().forEach(br -> islandedBusIds.add(br.getBusId()));
        }
        var filtered = result.getBusResults().stream().filter(br -> !islandedBusIds.contains(br.getBusId())).toList();
        if (dbg) {
            System.err.println("FILTER-ISLANDED '" + scenarioId.replace(GpuBatchedSecurityAnalysisSolver.STRATEGY_KEY_SEP, '|')
                    + "' ids=" + islandedBusIds + " " + result.getBusResults().size() + "->" + filtered.size());
        }
        return new NetworkResult(result.getBranchResults(), filtered,
                result.getThreeWindingsTransformerResults());
    }

    /** Stage-1 applicability: single-stage strategy with a TrueCondition whose contingency context
     *  covers this contingency. Anything else stays on the CPU per-strategy path. */
    private static boolean strategyApplies(OperatorStrategy strategy, String contingencyId) {
        List<com.powsybl.contingency.strategy.ConditionalActions> stages = strategy.getConditionalActions();
        if (stages.size() != 1) {
            return false;                                    // multi-stage stays on the CPU path (stage-on-stage
        }                                                    // semantics need intermediate-state evaluation)
        // ANY single-stage condition qualifies — including the violation/threshold conditions: the
        // ENGINE evaluates the condition (AbstractSecurityAnalysis.checkCondition, on the CPU-detected
        // post-contingency violations) BEFORE our runActionSimulation override is invoked, so the pair
        // state is precomputed SPECULATIVELY and simply goes unused when the condition does not pass.
        // Condition semantics therefore stay 100% stock — the batch only accelerates the solve.
        var ctx = strategy.getContingencyContext();
        return switch (ctx.getContextType()) {
            case ALL, ONLY_CONTINGENCIES -> true;
            case SPECIFIC -> contingencyId.equals(ctx.getContingencyId());
            case NONE -> false;
        };
    }

    /** Compile the strategy's actions onto the contingency's deltas into ONE extra batch scenario,
     *  or null when any action is out of stage-1 scope (branch open/RECLOSE, LoadAction,
     *  GeneratorAction — absolute or relative active power). */
    private BatchContingency composeStrategyScenario(LfNetwork net, String contingencyId, OperatorStrategy strategy,
                                                     List<LfBranch> ctgBranches,
                                                     List<LfGenerator> ctgGens,
                                                     List<GpuBatchedSecurityAnalysisSolver.LoadLoss> ctgLoads,
                                                     List<GpuBatchedSecurityAnalysisSolver.ShuntShift> ctgShifts) {
        List<LfBranch> branches = new java.util.ArrayList<>(ctgBranches);
        List<GpuBatchedSecurityAnalysisSolver.LoadLoss> loads = new java.util.ArrayList<>(ctgLoads);
        List<GpuBatchedSecurityAnalysisSolver.ShuntShift> shifts = new java.util.ArrayList<>(ctgShifts);
        List<GpuBatchedSecurityAnalysisSolver.TapR1Override> taps = new java.util.ArrayList<>();
        List<LfBranch> reclosed = new java.util.ArrayList<>();
        Map<String, Action> actionById = new java.util.HashMap<>();
        for (Action a : currentActions) {
            actionById.put(a.getId(), a);
        }
        for (String actionId : strategy.getConditionalActions().get(0).getActionIds()) {
            Action a = actionById.get(actionId);
            if (a instanceof com.powsybl.action.TerminalsConnectionAction tca) {
                LfBranch b = net.getBranchById(tca.getElementId());
                if (b == null) {
                    return null;
                }
                if (tca.isOpen()) {
                    if (!branches.contains(b)) {
                        branches.add(b);
                    }
                } else {
                    if (!branches.remove(b)) {
                        // close of a BASE-OPEN branch (retained disabled by LfTopoConfig): batched as a
                        // per-scenario mask CLEAR on the base-open packed element — dead-island
                        // reconnection (a disabled endpoint bus) is rejected at qualification.
                        if (!b.isDisabled()) {
                            return null;                     // closing an already-closed branch — CPU decides
                        }
                        reclosed.add(b);
                    }
                }
            } else if (a instanceof com.powsybl.action.SwitchAction sw) {
                LfBranch b = net.getBranchById(sw.getSwitchId());
                if (b == null || (sw.isOpen() ? branches.contains(b) : !branches.remove(b))) {
                    if (b != null && sw.isOpen()) {
                        branches.add(b);
                    } else {
                        return null;
                    }
                }
            } else if (a instanceof com.powsybl.action.LoadAction la) {
                com.powsybl.iidm.network.Load load = network.getLoad(la.getLoadId());
                if (load == null || load.getTerminal().getBusView().getBus() == null) {
                    return null;
                }
                LfBus bus = net.getBusById(load.getTerminal().getBusView().getBus().getId());
                if (bus == null) {
                    return null;
                }
                double newP = la.getActivePowerValue().orElse(Double.NaN);
                if (Double.isNaN(newP)) {
                    return null;
                }
                double dP = la.isRelativeValue() ? -newP : load.getP0() - newP;   // LoadLoss.p = load REMOVED (pu)
                loads.add(new GpuBatchedSecurityAnalysisSolver.LoadLoss(bus, dP / PerUnit.SB, 0));
            } else if (a instanceof com.powsybl.action.RatioTapChangerTapPositionAction rtca) {
                var tapChanger = network.getTwoWindingsTransformer(rtca.getTransformerId()) != null
                        ? network.getTwoWindingsTransformer(rtca.getTransformerId()).getRatioTapChanger() : null;
                GpuBatchedSecurityAnalysisSolver.TapR1Override ov = tapOverride(net, rtca.getTransformerId(),
                        tapChanger, rtca.isRelativeValue(), rtca.getTapPosition());
                if (ov == null) {
                    return null;
                }
                taps.add(ov);
            } else if (a instanceof com.powsybl.action.PhaseTapChangerTapPositionAction ptca) {
                var tapChanger = network.getTwoWindingsTransformer(ptca.getTransformerId()) != null
                        ? network.getTwoWindingsTransformer(ptca.getTransformerId()).getPhaseTapChanger() : null;
                GpuBatchedSecurityAnalysisSolver.TapR1Override ov = tapOverride(net, ptca.getTransformerId(),
                        tapChanger, ptca.isRelativeValue(), ptca.getTapPosition());
                if (ov == null) {
                    return null;
                }
                taps.add(ov);
            } else if (a instanceof com.powsybl.action.ShuntCompensatorPositionAction spa) {
                com.powsybl.iidm.network.ShuntCompensator sc = network.getShuntCompensator(spa.getShuntCompensatorId());
                LfShunt shunt = net.getShuntById(spa.getShuntCompensatorId());
                if (sc == null || shunt == null
                        || !(sc.getModel() instanceof com.powsybl.iidm.network.ShuntCompensatorLinearModel lin)) {
                        return null;                             // unpacked / non-linear model — CPU
                }
                double dSections = spa.getSectionCount() - sc.getSectionCount();
                double zb = sc.getTerminal().getVoltageLevel().getNominalV()
                        * sc.getTerminal().getVoltageLevel().getNominalV() / PerUnit.SB;
                double gPerSection = Double.isNaN(lin.getGPerSection()) ? 0 : lin.getGPerSection();   // optional in iidm
                shifts.add(new GpuBatchedSecurityAnalysisSolver.ShuntShift(shunt,
                        gPerSection * dSections * zb, lin.getBPerSection() * dSections * zb));
            } else if (a instanceof com.powsybl.action.GeneratorAction ga) {
                // Pure ACTIVE-POWER redispatch only: a targetV/targetQ/regulator change alters the
                // voltage-control problem itself — CPU path. The P delta is composed exactly like a
                // load delta with q=0 (net injection at the bus; the device DS reweight machinery is
                // NOT touched: participation factors are unchanged by a redispatch, and the headroom
                // shift only matters if this generator saturates during redistribution — CPU handles
                // any such contingency via the DS-residual rejection).
                if (ga.getTargetV().isPresent() || ga.getTargetQ().isPresent()
                        || ga.isVoltageRegulatorOn().isPresent() || ga.getActivePowerValue().isEmpty()) {
                    return null;
                }
                com.powsybl.iidm.network.Generator gen = network.getGenerator(ga.getGeneratorId());
                if (gen == null) {
                    return null;
                }
                LfBus bus = net.getBusById(gen.getTerminal().getBusView().getBus() != null
                        ? gen.getTerminal().getBusView().getBus().getId() : "");
                if (bus == null) {
                    return null;
                }
                boolean relative = ga.isActivePowerRelativeValue().orElse(false);
                double deltaP = relative ? ga.getActivePowerValue().getAsDouble()
                                         : ga.getActivePowerValue().getAsDouble() - gen.getTargetP();
                loads.add(new GpuBatchedSecurityAnalysisSolver.LoadLoss(bus, deltaP / PerUnit.SB, 0));
            } else {
                return null;                                 // out of stage-1 scope — CPU per-strategy path
            }
        }
        if (branches.size() > 2) {
            return null;
        }
        return new BatchContingency(contingencyId + STRATEGY_KEY_SEP + strategy.getId(),
                branches, ctgGens, loads, shifts, taps, List.of(), reclosed);
    }


    /** Compiles a tap-position action (RTC or PST) into a per-scenario full-row override — the
     *  target tap's [y, g1, g2, b1, b2, ksi, r1, a1] replaces the branch's static inputs, so
     *  per-tap impedance changes are covered. Null when out of batch scope: unknown transformer /
     *  tap changer, no tap table, or out-of-range position. */
    private static GpuBatchedSecurityAnalysisSolver.TapR1Override tapOverride(LfNetwork net, String transformerId,
                                                                              com.powsybl.iidm.network.TapChanger<?, ?, ?, ?> tapChanger,
                                                                              boolean relative, int tapPosition) {
        LfBranch b = net.getBranchById(transformerId);
        if (b == null || tapChanger == null) {
            return null;
        }
        double[][] table = GpuBatchedSecurityAnalysisSolver.tapModels(b.getPiModel());
        int low = tapChanger.getLowTapPosition();
        int idx = (relative ? tapChanger.getTapPosition() + tapPosition : tapPosition) - low;
        if (table.length == 0 || idx < 0 || idx >= table.length) {
            return null;                                     // no tap table / out-of-range position — CPU
        }
        return new GpuBatchedSecurityAnalysisSolver.TapR1Override(b, table[idx]);
    }

    /** The contingency's shunt admittance shifts (pu deltas, negated: the propagation's shift is the
     *  admittance REMOVED), or null when a shunt is unresolvable. */
    private static List<GpuBatchedSecurityAnalysisSolver.ShuntShift> shuntShifts(LfNetwork net, PropagatedContingency pc) {
        if (!pc.getBusIdsToLose().isEmpty()) {
            return null;
        }
        List<GpuBatchedSecurityAnalysisSolver.ShuntShift> out = new java.util.ArrayList<>();
        for (Map.Entry<String, com.powsybl.openloadflow.network.AdmittanceShift> e : pc.getShuntIdsToShift().entrySet()) {
            LfShunt shunt = net.getShuntById(e.getKey());
            if (shunt == null || shunt.isDisabled()) {
                return null;
            }
            out.add(new GpuBatchedSecurityAnalysisSolver.ShuntShift(shunt, -e.getValue().getG(), -e.getValue().getB()));
        }
        return out;
    }

    /** The single fixed shunt a contingency disconnects (which the GPU batch can mask off like a branch), or
     *  null if it is not a clean full disconnection of a packed fixed shunt — a multi-element contingency, a
     *  voltage-controlling/SVC shunt the GPU does not pack, or a multi-compensator aggregate the mask cannot
     *  partially remove (those fall through to the CPU). */
    private static LfShunt singleShuntOutage(LfNetwork net, PropagatedContingency pc) {
        Set<String> shuntIds = pc.getShuntIdsToShift().keySet();
        if (shuntIds.size() != 1 || !pc.getBranchIdsToOpen().isEmpty()
                || !pc.getGeneratorIdsToLose().isEmpty() || !pc.getBusIdsToLose().isEmpty()
                || !pc.getLoadIdsToLose().isEmpty()) {
            return null;
        }
        LfShunt shunt = net.getShuntById(shuntIds.iterator().next());
        // A sole-compensator LfShunt: masking its element removes exactly the contingency's full admittance.
        return shunt != null && !shunt.isDisabled() && shunt.getOriginalIds().size() == 1 ? shunt : null;
    }

    /** The contingency's closed-branch outages (0-2, BOTH_SIDES only), or null when the branch part is
     *  out of batch scope. Generators are allowed alongside (branch+generator combos batch since 2026-07-17);
     *  buses and loads are not. */
    private static List<LfBranch> closedBranchOutages(LfNetwork net, PropagatedContingency pc) {
        int n = pc.getBranchIdsToOpen().size();
        if (n > 2 || !pc.getBusIdsToLose().isEmpty()) {
            return null;
        }
        List<LfBranch> branches = new ArrayList<>(n);
        for (Map.Entry<String, DisabledBranchStatus> e : pc.getBranchIdsToOpen().entrySet()) {
            if (e.getValue() != DisabledBranchStatus.BOTH_SIDES) {
                return null;
            }
            LfBranch branch = net.getBranchById(e.getKey());
            if (branch == null || branch.getBus1() == null || branch.getBus2() == null) {
                return null;
            }
            branches.add(branch);
        }
        return branches;
    }

    @Override
    protected PostContingencyResult runPostContingencySimulation(LfNetwork network, AcLoadFlowContext context, Contingency contingency,
                                                                 LfContingency lfContingency, LimitViolationManager preContingencyLimitViolationManager,
                                                                 SecurityAnalysisParameters securityAnalysisParameters,
                                                                 PreContingencyNetworkResult preContingencyNetworkResult, boolean createResultExtension,
                                                                 List<LimitReduction> limitReductions, double preDistributedActivePower) {
        double[] state = precomputedStates.get(contingency.getId());
        if (state == null || batchEquationSystem == null) {
            // not batched (multi-element / islanding / out of scope / diverged) — standard CPU path
            return super.runPostContingencySimulation(network, context, contingency, lfContingency, preContingencyLimitViolationManager,
                    securityAnalysisParameters, preContingencyNetworkResult, createResultExtension, limitReductions, preDistributedActivePower);
        }

        // Inject the precomputed post-contingency state (the branch is already disabled by
        // lfContingency.apply()), then reuse the engine's result-building unchanged.
        logPostContingencyStart(network, lfContingency);
        Stopwatch stopwatch = Stopwatch.createStarted();

        injectState(state);
        AcSolverUtil.updateNetwork(network, batchEquationSystem);

        PostContingencyComputationStatus status = PostContingencyComputationStatus.CONVERGED;
        var postContingencyLimitViolationManager = new LimitViolationManager(preContingencyLimitViolationManager, limitReductions,
                securityAnalysisParameters.getIncreasedViolationsParameters());
        LoadFlowModel loadFlowModel = securityAnalysisParameters.getLoadFlowParameters().isDc() ? LoadFlowModel.DC : LoadFlowModel.AC;
        var postContingencyNetworkResult = new PostContingencyNetworkResult(network,
                new AbstractNetworkResult.StateMonitorIndexes(monitorIndex, zeroImpedanceMonitoredIndex), createResultExtension,
                preContingencyNetworkResult, contingency, loadFlowModel, securityAnalysisParameters.getLoadFlowParameters().getDcPowerFactor(),
                securityAnalysisParameters.getModifiedMonitoredElementsParameters());
        postContingencyNetworkResult.update();
        postContingencyLimitViolationManager.detectViolations(network);

        stopwatch.stop();
        logPostContingencyEnd(network, lfContingency, stopwatch);

        var connectivityResult = new ConnectivityResult(lfContingency.getCreatedSynchronousComponentsCount(), 0,
                lfContingency.getDisconnectedLoadActivePower() * PerUnit.SB,
                lfContingency.getDisconnectedGenerationActivePower() * PerUnit.SB,
                lfContingency.getDisconnectedElementIds());

        // No device-side slack distribution in the prototype: only the pre-distribution counts.
        return new PostContingencyResult(contingency, status,
                new LimitViolationsResult(postContingencyLimitViolationManager.getLimitViolations()),
                filterIslanded(network, contingency.getId(),
                        new NetworkResult(postContingencyNetworkResult.getBranchResults(), postContingencyNetworkResult.getBusResults(),
                                postContingencyNetworkResult.getThreeWindingsTransformerResults())),
                connectivityResult, preDistributedActivePower * PerUnit.SB);
    }
}
