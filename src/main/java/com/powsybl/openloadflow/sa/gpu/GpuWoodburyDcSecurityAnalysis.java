/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.action.Action;
import com.powsybl.commons.report.ReportNode;
import com.powsybl.contingency.Contingency;
import com.powsybl.contingency.strategy.OperatorStrategy;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlowResult;
import com.powsybl.math.matrix.MatrixFactory;
import com.powsybl.openloadflow.ac.solver.gpu.GpuLoadFlowParameters;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuTwoSides;
import com.powsybl.openloadflow.dc.DcLoadFlowContext;
import com.powsybl.openloadflow.dc.DcLoadFlowParameters;
import com.powsybl.openloadflow.dc.equations.AbstractClosedBranchDcFlowEquationTerm;
import com.powsybl.openloadflow.dc.equations.DcEquationSystemCreationParameters;
import com.powsybl.openloadflow.dc.equations.DcEquationType;
import com.powsybl.openloadflow.dc.equations.DcVariableType;
import com.powsybl.openloadflow.dc.fastdc.ComputedContingencyElement;
import com.powsybl.openloadflow.dc.fastdc.ComputedElement;
import com.powsybl.openloadflow.dc.fastdc.ComputedSwitchBranchElement;
import com.powsybl.openloadflow.dc.fastdc.ComputedTapPositionChangeElement;
import com.powsybl.openloadflow.dc.fastdc.ConnectivityBreakAnalysis;
import com.powsybl.openloadflow.dc.gpu.GpuDcAssembly;
import com.powsybl.openloadflow.dc.gpu.GpuDcLoadFlowEngine;
import com.powsybl.openloadflow.dc.gpu.GpuDcSolver;
import com.powsybl.openloadflow.equations.Equation;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.graph.GraphConnectivityFactory;
import com.powsybl.openloadflow.network.*;
import com.powsybl.openloadflow.network.action.*;
import com.powsybl.openloadflow.network.impl.PropagatedContingency;
import com.powsybl.openloadflow.sa.*;
import com.powsybl.openloadflow.util.Indexed;
import com.powsybl.openloadflow.util.PerUnit;
import com.powsybl.openloadflow.util.Reports;
import com.powsybl.openloadflow.util.gpu.GpuTrace;
import com.powsybl.security.LimitViolationsResult;
import com.powsybl.security.PostContingencyComputationStatus;
import com.powsybl.security.SecurityAnalysisParameters;
import com.powsybl.security.SecurityAnalysisResult;
import com.powsybl.security.limitreduction.LimitReduction;
import com.powsybl.security.monitor.StateMonitor;
import com.powsybl.security.monitor.StateMonitorIndex;
import com.powsybl.security.results.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Predicate;

import static com.powsybl.openloadflow.dc.DcLoadFlowEngine.updateNetwork;
import static com.powsybl.openloadflow.network.impl.PropagatedContingency.cleanContingencies;

/**
 * The Woodbury (fast DC) security analysis with the per-contingency work on the device.
 *
 * <p>Stock {@link WoodburyDcSecurityAnalysis} spends its wall on the host, per contingency: the
 * n-vector Woodbury combination, {@code updateNetwork}, the walk of every branch's limits and a
 * {@code NetworkState.restore} (measured on rte6515 x 9042: detection ~45%, restore ~35%, the
 * linear algebra under 5%). Here the device holds the element states S once, and per chunk of
 * contingencies computes the flow-transfer factors, the post-contingency states, every branch flow
 * and a RELAXED limit screen. The host then does only what must stay exact and OLF's:
 * <ul>
 *   <li>loads the state into the equation system (the vectorized DC network recomputes the branch
 *       flows) and confirms each screen candidate with OLF's own {@code >} on OLF's own value
 *       ({@link GpuLimitScanner#detectFromCandidates});</li>
 *   <li>builds the network result with {@link PostContingencyNetworkResult} and the connectivity
 *       result from OLF's {@link LfContingency}, so monitored flows and reported violations are
 *       the stock classes' output;</li>
 *   <li>never applies the contingency to the network and never restores it — for a pure branch
 *       outage the stock path's {@code apply} only flags the branch and everything downstream reads
 *       the disabled-branch PREDICATE, which is what the device path evaluates too.</li>
 * </ul>
 *
 * <p>What is routed to the stock path per contingency, exactly and unaccelerated: outages that
 * break connectivity, lose generation/load/HVDC/shunts/buses or a phase-shifting transformer
 * (the modified-target classes), and contingencies with more than {@link GpuDcSolver#MAX_ELEMENTS}
 * branches. Operator strategies and actions send the whole run to the stock path.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public class GpuWoodburyDcSecurityAnalysis extends WoodburyDcSecurityAnalysis {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuWoodburyDcSecurityAnalysis.class);

    /** Contingencies solved on the device in the last run, ACCUMULATED over the run's connected
     *  components (runSimulations is called once per component); -1 = the whole run went to the
     *  stock path. The provider resets it before a run. */
    public static final AtomicInteger LAST_GPU_PRECOMPUTED_COUNT = new AtomicInteger(-1);

    private static void addToDeviceCount(int onDevice) {
        LAST_GPU_PRECOMPUTED_COUNT.updateAndGet(previous -> Math.max(previous, 0) + onDevice);
    }

    /** Contingencies per device chunk: bounds the state download ({@code chunk x n} doubles). */
    static final int DEVICE_CHUNK = 256;

    /** Candidate records per contingency the device buffer holds; overflow = host screen. */
    static final int CAPACITY = 1024;

    /** Test hook: print how accurate the device base state is against OLF's own system (GpuDcBaseAccuracy). */
    public static final AtomicBoolean BASE_ACCURACY_DIAGNOSTIC = new AtomicBoolean();

    /** The last base accuracy measured under that hook (the main component's), or null. */
    public static final java.util.concurrent.atomic.AtomicReference<GpuDcBaseAccuracy.Result> LAST_BASE_ACCURACY =
            new java.util.concurrent.atomic.AtomicReference<>();

    /** Test hook: also load each contingency's state into OLF's model and compare every branch value the state reader
     *  gives (P, I, S of both sides) with OLF's evaluation, bit for bit. */
    public static final AtomicBoolean CHECK_STATE_READER = new AtomicBoolean();

    /** Branch values differing between the state reader and OLF's model under that hook, and the contingencies answered
     *  from their state without loading it, accumulated (a test resets them). */
    public static final AtomicInteger STATE_READER_MISMATCH_COUNT = new AtomicInteger();
    public static final AtomicInteger STATE_FREE_CONTINGENCY_COUNT = new AtomicInteger();

    /** The branch values of a state, read without loading it (built with the device's branch arrays). */
    private GpuDcStateReader stateReader;

    /** Test hook: build the monitored branch results on BOTH paths (GpuMonitoredBranchResults and OLF's update), compare. */
    public static final AtomicBoolean CHECK_NETWORK_RESULTS = new AtomicBoolean();

    /** Network results differing between the two paths under that hook, accumulated (a test resets it). */
    public static final AtomicInteger NETWORK_RESULT_MISMATCH_COUNT = new AtomicInteger();

    /** Network results built by GpuMonitoredBranchResults, accumulated (a test resets it). */
    public static final AtomicInteger FAST_NETWORK_RESULT_COUNT = new AtomicInteger();

    /** Test hook: build the modified target columns on BOTH sides and compare them bit for bit. */
    public static final AtomicBoolean CHECK_DEVICE_TARGETS = new AtomicBoolean();

    /** Modified target columns built on the device, and (check hook) target values that differed from the host's,
     *  accumulated over the runs (a test resets them): a run solves each connected component separately. */
    public static final AtomicInteger DEVICE_TARGET_COUNT = new AtomicInteger();
    public static final AtomicInteger TARGET_MISMATCH_COUNT = new AtomicInteger();
    public static final AtomicLong LAST_TARGET_MAX_DIFF_BITS = new AtomicLong();

    private record DeviceCandidate(ConnectivityBreakAnalysis.ConnectivityAnalysisResult connectivity,
                                   List<ComputedContingencyElement> elements, LfContingency lfContingency) {
    }

    public GpuWoodburyDcSecurityAnalysis(Network network, MatrixFactory matrixFactory,
                                         GraphConnectivityFactory<LfBus, LfBranch> connectivityFactory,
                                         List<StateMonitor> stateMonitors, ReportNode reportNode) {
        super(network, matrixFactory, connectivityFactory, stateMonitors, reportNode);
    }

    @Override
    protected SecurityAnalysisResult runSimulations(LfNetwork lfNetwork, List<PropagatedContingency> propagatedContingencies,
                                                    DcLoadFlowParameters dcParameters, SecurityAnalysisParameters securityAnalysisParameters,
                                                    List<OperatorStrategy> operatorStrategies, List<Action> actions,
                                                    List<LimitReduction> limitReductions,
                                                    ContingencyActivePowerLossDistribution contingencyActivePowerLossDistribution) {
        if (!GpuDcSolver.isAvailable()) {
            LOGGER.info("GPU Woodbury: whole run on the stock path (no device)");
            LAST_GPU_PRECOMPUTED_COUNT.set(-1);
            return super.runSimulations(lfNetwork, propagatedContingencies, dcParameters, securityAnalysisParameters,
                    operatorStrategies, actions, limitReductions, contingencyActivePowerLossDistribution);
        }

        OpenSecurityAnalysisParameters openSecurityAnalysisParameters = OpenSecurityAnalysisParameters.getOrDefault(securityAnalysisParameters);
        boolean createResultExtension = openSecurityAnalysisParameters.isCreateResultExtension();
        DcEquationSystemCreationParameters creationParameters = dcParameters.getEquationSystemCreationParameters();
        double dcPowerFactor = creationParameters.getDcPowerFactor();

        try (DcLoadFlowContext context = new DcLoadFlowContext(lfNetwork, dcParameters, false)) {
            ReportNode networkReportNode = lfNetwork.getReportNode();
            ReportNode preContSimReportNode = Reports.createPreContingencySimulation(networkReportNode);
            lfNetwork.setReportNode(preContSimReportNode);

            // same preparation as the stock path: contingencies on one-side-connected branches are dropped
            cleanContingencies(lfNetwork, propagatedContingencies);

            long tPre0 = System.nanoTime();
            EquationSystem<DcVariableType, DcEquationType> equationSystem = context.getEquationSystem();
            // the device assembles the DC system itself (GpuDcAssembly) and factorizes it: the base solve runs on
            // it too — the slack distribution over every bus, the target vector, the solve — so no host matrix,
            // no KLU, and the generators written back so the host sees the distributed dispatch
            GpuDcLoadFlowEngine engine = new GpuDcLoadFlowEngine(context,
                    GpuLoadFlowParameters.get(securityAnalysisParameters.getLoadFlowParameters()).getLinearSolver());
            try {
                engine.prepare();
            } catch (UnsupportedOperationException e) {
                LOGGER.warn("GPU Woodbury: {} — whole run on the stock path", e.getMessage());
                LAST_GPU_PRECOMPUTED_COUNT.set(-1);
                return super.runSimulations(lfNetwork, propagatedContingencies, dcParameters, securityAnalysisParameters,
                        operatorStrategies, actions, limitReductions, contingencyActivePowerLossDistribution);
            }
            GpuDcSolver solver = engine.getSolver();
            double[] preContingencyStates;
            try {
                preContingencyStates = engine.solveBase();
            } catch (RuntimeException e) {
                solver.close();
                throw e;
            }
            if (BASE_ACCURACY_DIAGNOSTIC.get()) {
                LAST_BASE_ACCURACY.set(GpuDcBaseAccuracy.report(context, solver, preContingencyStates));
            }
            if (context.getParameters().isSetVToNan()) {
                for (LfBus bus : lfNetwork.getBuses()) {
                    bus.setV(Double.NaN);
                }
            }
            List<StateMonitor> zeroImpedanceStateMonitors = extractZeroImpedanceStateMonitors(lfNetwork);
            this.zeroImpedanceMonitoredIndex = new StateMonitorIndex(zeroImpedanceStateMonitors);
            AbstractNetworkResult.StateMonitorIndexes monitorIndexes = new AbstractNetworkResult.StateMonitorIndexes(monitorIndex, zeroImpedanceMonitoredIndex);
            var preContingencyNetworkResult = new PreContingencyNetworkResult(lfNetwork, monitorIndexes, createResultExtension,
                    LoadFlowModel.DC, securityAnalysisParameters.getLoadFlowParameters().getDcPowerFactor());
            preContingencyNetworkResult.update();
            var preContingencyLimitViolationManager = new LimitViolationManager(limitReductions);
            preContingencyLimitViolationManager.detectViolations(lfNetwork);

            long tPre1 = System.nanoTime();
            // contingency elements: OLF's own index (one S column per distinct branch, its computedElementIndex)
            Map<String, ComputedContingencyElement> elementByBranch =
                    ConnectivityBreakAnalysis.createContingencyElementsIndexByBranchId(propagatedContingencies, lfNetwork, equationSystem, creationParameters);

            // operator strategies: the stock's indexes, the needed actions, and one action element per branch an
            // action switches or shifts (its S column is appended after the contingency elements')
            Map<String, Action> actionsById = Actions.indexById(actions);
            Map<String, List<Indexed<OperatorStrategy>>> operatorStrategiesByContingencyId =
                    OperatorStrategies.indexByContingencyId(propagatedContingencies, operatorStrategies, actionsById, true);
            Set<Action> neededActions = OperatorStrategies.getNeededActions(operatorStrategiesByContingencyId, actionsById);
            Map<String, LfAction> lfActionById = LfActionUtils.createLfActions(lfNetwork, neededActions, network);
            Map<LfAction, List<ComputedElement>> actionElementsByLfAction =
                    ComputedElement.createActionElementsIndexByLfAction(lfActionById, equationSystem, creationParameters);
            List<ComputedElement> actionElements = actionElementsByLfAction.values().stream().flatMap(Collection::stream).toList();
            StrategySetup strategySetup = new StrategySetup(operatorStrategiesByContingencyId, lfActionById, actionElementsByLfAction);

            // the device: branch model, S — before anything else needs S
            long tFactor = System.nanoTime();
            int elementCount;
            try {
                stateReader = prepareBranches(solver, lfNetwork, equationSystem, creationParameters);
                elementCount = prepareElements(solver, equationSystem, creationParameters, elementByBranch.values(), actionElements, strategySetup);
            } catch (RuntimeException e) {
                solver.close();
                throw e;
            }
            if (elementCount < 0) {
                solver.close();
                LOGGER.warn("GPU Woodbury: the device cannot hold the element states, whole run on the stock path");
                LAST_GPU_PRECOMPUTED_COUNT.set(-1);
                return super.runSimulations(lfNetwork, propagatedContingencies, dcParameters, securityAnalysisParameters,
                        operatorStrategies, actions, limitReductions, contingencyActivePowerLossDistribution);
            }
            long tS = System.nanoTime();

            // connectivity analysis: OLF's sensitivity criterion evaluated on the device S, then OLF's own graph
            // confirmation for the potentially breaking ones — the same two phases, the same lists, the same order
            ConnectivityBreakAnalysis.ConnectivityBreakAnalysisResults connectivity =
                    connectivityAnalysis(solver, lfNetwork, propagatedContingencies, elementByBranch);
            long tConn = System.nanoTime();
            if (GpuTrace.PHASE_TIME.isOn()) {
                System.err.printf("DC_WOODBURY_PRE factor+preContingency(device)=%.0f ms strategies+elements=%.0f ms S=%.0f ms connectivity=%.0f ms (%d potentially breaking -> graph)%n",
                        (tPre1 - tPre0) / 1e6, (tFactor - tPre1) / 1e6, (tS - tFactor) / 1e6, (tConn - tS) / 1e6, lastPotentiallyBreaking);
            }

            // qualification: a contingency that keeps the base target vector, or one whose modified target
            // (islanded buses, lost PST) the host can build without touching the network
            List<DeviceCandidate> device = new ArrayList<>();
            List<PropagatedContingency> cpu = new ArrayList<>();
            Map<String, Integer> cpuReasons = new TreeMap<>();
            for (var list : List.of(connectivity.nonBreakingConnectivityAnalysisResults(), connectivity.connectivityBreakingAnalysisResults())) {
                for (ConnectivityBreakAnalysis.ConnectivityAnalysisResult r : list) {
                    String reason = deviceRejectReason(r, elementByBranch, lfNetwork);
                    if (reason == null) {
                        // the stock builds this LfContingency first (before the HVDC stations are added to the lost
                        // generators for the modified target): it is the one the connectivity result is reported from
                        Optional<LfContingency> lfContingency = r.toLfContingency();
                        if (lfContingency.isEmpty()) {
                            continue;                        // no impact: the stock path reports nothing either
                        }
                        List<ComputedContingencyElement> elements = r.getPropagatedContingency().getBranchIdsToOpen().keySet().stream()
                                .filter(id -> !r.getElementsToReconnect().contains(id))
                                .map(elementByBranch::get).toList();
                        device.add(new DeviceCandidate(r, elements, lfContingency.get()));
                    } else {
                        cpu.add(r.getPropagatedContingency());
                        cpuReasons.merge(reason, 1, Integer::sum);
                    }
                }
            }
            LOGGER.info("GPU Woodbury: {} contingencies on the device, {} on the stock path {}", device.size(), cpu.size(), cpuReasons);
            if (GpuTrace.PHASE_TIME.isOn()) {
                System.err.printf("DC_WOODBURY_ROUTING device=%d stock=%d %s%n", device.size(), cpu.size(), cpuReasons);
            }

            Map<String, PostContingencyResult> resultById = new HashMap<>();
            Map<String, List<OperatorStrategyResult>> strategyResultsByContingency = new HashMap<>();
            int onDevice = 0;

            try (solver) {
                if (!device.isEmpty()) {
                    onDevice = runOnDevice(solver, engine.getAssembly(), networkReportNode, elementCount, lfNetwork, context, device, strategySetup, elementByBranch,
                            preContingencyStates, preContingencyNetworkResult,
                            preContingencyLimitViolationManager, monitorIndexes, createResultExtension, securityAnalysisParameters,
                            limitReductions, dcPowerFactor, resultById, strategyResultsByContingency);
                }
            }
            lfNetwork.setReportNode(networkReportNode);

            if (!cpu.isEmpty()) {
                // the stock path, on the same network, for what the device did not take: it restores the
                // network to base after each contingency, and its pre-contingency result is ours by construction
                List<PropagatedContingency> cpuInOrder = propagatedContingencies.stream().filter(cpu::contains).toList();
                SecurityAnalysisResult stock = super.runSimulations(lfNetwork, new ArrayList<>(cpuInOrder), dcParameters, securityAnalysisParameters,
                        operatorStrategies, actions, limitReductions, contingencyActivePowerLossDistribution);
                for (PostContingencyResult r : stock.getPostContingencyResults()) {
                    resultById.put(r.getContingency().getId(), r);
                }
                for (OperatorStrategyResult r : stock.getOperatorStrategyResults()) {
                    strategyResultsByContingency.computeIfAbsent(r.getOperatorStrategy().getContingencyContext().getContingencyId(), k -> new ArrayList<>()).add(r);
                }
            }
            addToDeviceCount(onDevice);

            // the stock order: the non-breaking contingencies (as the connectivity analysis listed them),
            // then the connectivity-breaking ones
            List<PostContingencyResult> postContingencyResults = new ArrayList<>();
            List<OperatorStrategyResult> operatorStrategyResults = new ArrayList<>();
            for (var list : List.of(connectivity.nonBreakingConnectivityAnalysisResults(), connectivity.connectivityBreakingAnalysisResults())) {
                for (ConnectivityBreakAnalysis.ConnectivityAnalysisResult r : list) {
                    String id = r.getPropagatedContingency().getContingency().getId();
                    PostContingencyResult result = resultById.get(id);
                    if (result != null) {
                        postContingencyResults.add(result);
                        operatorStrategyResults.addAll(strategyResultsByContingency.getOrDefault(id, List.of()));
                    }
                }
            }
            return new SecurityAnalysisResult(
                    new PreContingencyResult(LoadFlowResult.ComponentResult.Status.CONVERGED,
                            new LimitViolationsResult(preContingencyLimitViolationManager.getLimitViolations()),
                            new NetworkResult(preContingencyNetworkResult.getBranchResults(), preContingencyNetworkResult.getBusResults(),
                                    preContingencyNetworkResult.getThreeWindingsTransformerResults()),
                            Double.NaN),
                    postContingencyResults, operatorStrategyResults);
        }
    }

    /** The stock's operator-strategy indexes and the action elements' S columns. */
    private record StrategySetup(Map<String, List<Indexed<OperatorStrategy>>> operatorStrategiesByContingencyId,
                                 Map<String, LfAction> lfActionById,
                                 Map<LfAction, List<ComputedElement>> actionElementsByLfAction) {
        boolean any() {
            return !operatorStrategiesByContingencyId.isEmpty();
        }
    }

    /** One (contingency, operator strategy) scenario of the strategy pass. */
    private record StrategyScenario(DeviceCandidate candidate, Indexed<OperatorStrategy> operatorStrategy, LfOperatorStrategy lfOperatorStrategy,
                                    ConnectivityBreakAnalysis.ConnectivityAnalysisResult connectivity, List<ComputedElement> elements,
                                    ReportNode contingencyReportNode) {
    }

    /**
     * Why a contingency cannot go to the device, or null when it can. Lost generation/load/HVDC,
     * HVDCs left without power, shunt shifts, bus losses, load-based balance types and multi-slack
     * networks all go through the modified-target column (built on the host exactly like the
     * stock's, solved on the device); what stays on the stock path is what the device machinery
     * cannot express: a lost slack bus, a branch without a DC flow equation or open on one side,
     * more Woodbury elements than the kernel solves in registers.
     */
    static String deviceRejectReason(ConnectivityBreakAnalysis.ConnectivityAnalysisResult r,
                                     Map<String, ComputedContingencyElement> elementByBranch, LfNetwork lfNetwork) {
        PropagatedContingency c = r.getPropagatedContingency();
        for (String busId : c.getBusIdsToLose()) {
            LfBus bus = lfNetwork.getBusById(busId);
            if (bus != null && bus.isSlack()) {
                return "slack bus loss";                     // not supported in DC by the stock path either
            }
        }
        // a contingency with no branch element (a lost generator or load) is a modified target alone;
        // one with no impact at all is skipped by toLfContingency, as in the stock path
        boolean lostPst = false;
        int woodbury = 0;
        for (String branchId : c.getBranchIdsToOpen().keySet()) {
            ComputedContingencyElement element = elementByBranch.get(branchId);
            if (element == null || element.getLfBranchEquation() == null) {
                return "branch without a DC flow equation";
            }
            LfBranch branch = element.getLfBranch();
            if (branch.getBus1() == null || branch.getBus2() == null) {
                return "branch open on one side";
            }
            lostPst |= branch.hasPhaseControllerCapability();
            if (!r.getElementsToReconnect().contains(branchId)) {
                woodbury++;
            }
        }
        if (woodbury > GpuDcSolver.MAX_ELEMENTS) {
            return "more than " + GpuDcSolver.MAX_ELEMENTS + " Woodbury elements";
        }
        return null;
    }

    /** The device pass over the qualified contingencies; returns the number solved on the device. */
    private int runOnDevice(GpuDcSolver solver, GpuDcAssembly assembly, ReportNode networkReportNode, int elementCount, LfNetwork lfNetwork, DcLoadFlowContext context, List<DeviceCandidate> device,
                            StrategySetup strategySetup, Map<String, ComputedContingencyElement> elementByBranch,
                            double[] preContingencyStates, PreContingencyNetworkResult preContingencyNetworkResult,
                            LimitViolationManager preContingencyLimitViolationManager, AbstractNetworkResult.StateMonitorIndexes monitorIndexes,
                            boolean createResultExtension, SecurityAnalysisParameters securityAnalysisParameters,
                            List<LimitReduction> limitReductions, double dcPowerFactor, Map<String, PostContingencyResult> resultById,
                            Map<String, List<OperatorStrategyResult>> strategyResultsByContingency) {
        List<StrategyScenario> strategyScenarios = new ArrayList<>();
        EquationSystem<DcVariableType, DcEquationType> equationSystem = context.getEquationSystem();
        int n = equationSystem.getIndex().getColumnCount();
        long t0 = System.nanoTime();
        solver.setBase(preContingencyStates);
        long tFactor = t0;
        long tBranches = t0;
        long tS = t0;

        // --- limit table
        GpuLimitTable table = uploadLimitTable(solver, lfNetwork, limitReductions, dcPowerFactor);
        GpuLimitScanner scanner = new GpuLimitScanner(table);
        long tTable = System.nanoTime();
        boolean needAngles = !monitorIndex.getAllStateMonitor().getVoltageLevelIds().isEmpty()
                || monitorIndex.getSpecificStateMonitors().values().stream().anyMatch(m -> !m.getVoltageLevelIds().isEmpty())
                || !lfNetwork.getVoltageAngleLimits().isEmpty();
        // --- the modified-target machinery: the base target vector and the slack distribution replayed on arrays.
        // The stock path snapshots the network here (NetworkState.save), and that snapshot re-anchors every
        // generator's INITIAL target P to its base value: the per-contingency redistribution then starts from
        // the base dispatch, not from the original network's. Do the same before modelling it.
        NetworkState.save(lfNetwork);
        double[] baseTargets = context.getTargetVector().getArray().clone();
        boolean multiSlack = lfNetwork.getSynchronousNetworks().getFirst().getSlackBuses().size() > 1;
        boolean arrayModelApplies = !multiSlack && (!context.getParameters().isDistributedSlack()
                || DcSlackDistributionModel.supports(context.getParameters().getBalanceType()));
        // per-bus net injection at base, in bus order: getActivePowerMismatch over the remaining buses is
        // -sum over them, and summing in bus order while skipping the islanded ones is the same sum in the
        // same order as the stock's LinkedHashSet walk (bit-identical, no per-contingency getter walk)
        double[] busNetInjection = new double[lfNetwork.getBuses().size()];
        for (LfBus bus : lfNetwork.getBuses()) {
            busNetInjection[bus.getNum()] = bus.isDisabled() ? Double.NaN : bus.getGenerationTargetP() - bus.getLoadTargetP();
        }
        boolean distributedSlack = context.getParameters().isDistributedSlack();
        DcSlackDistributionModel distribution = distributedSlack && arrayModelApplies
                ? DcSlackDistributionModel.build(lfNetwork, context.getParameters().getBalanceType(),
                        context.getParameters().getNetworkParameters().isUseActiveLimits())
                : null;
        // the device build of the modified target columns: generation balance types only (a lost load's
        // getUpdatedLoadP0 has no factor there), on the generator and load tables as NetworkState.save left them
        int participationType = GpuDcLoadFlowEngine.participationType(context.getParameters().getBalanceType());
        boolean deviceTargets = participationType >= 0;
        boolean checkTargets = deviceTargets && CHECK_DEVICE_TARGETS.get();
        Map<LfGenerator, Integer> generatorIndex = deviceTargets ? syncInjectionTables(solver, assembly, lfNetwork) : Map.of();
        int deviceTargetCount = 0;
        int targetMismatches = 0;
        GpuMonitoredBranchResults monitoredBranchResults = GpuMonitoredBranchResults.build(lfNetwork, monitorIndex, zeroImpedanceMonitoredIndex);
        boolean needBusResults = !monitorIndex.getAllStateMonitor().getVoltageLevelIds().isEmpty()
                || monitorIndex.getSpecificStateMonitors().values().stream().anyMatch(m -> !m.getVoltageLevelIds().isEmpty());
        // contingencies answered from their state, as the AC path does: no voltage-angle limit or monitored bus (they read
        // the buses' angles), no operator strategy (it re-runs on the loaded state), monitored branches whose results the
        // state gives
        boolean stateFree = stateReader != null && !needAngles && !needBusResults
                && strategySetup.operatorStrategiesByContingencyId().isEmpty()
                && monitoredBranchResults.readableFromState(createResultExtension);
        boolean checkStateReader = stateFree && CHECK_STATE_READER.get();
        long tSetup = System.nanoTime();

        // --- chunks
        SecurityAnalysisParameters.IncreasedViolationsParameters violationsParameters = securityAnalysisParameters.getIncreasedViolationsParameters();
        double[] states = new double[n * DEVICE_CHUNK];
        double[] targets = new double[n * DEVICE_CHUNK];
        long tDevice = 0;
        long tHost = 0;
        long tColumns = 0;
        int hostScreened = 0;
        int modifiedTargets = 0;
        int replayedTargets = 0;
        for (int from = 0; from < device.size(); from += DEVICE_CHUNK) {
            int to = Math.min(from + DEVICE_CHUNK, device.size());
            int count = to - from;
            int[] ctgPtr = new int[count + 1];
            int[] ctgBase = new int[count];
            List<Integer> elemList = new ArrayList<>();
            int nBases = 0;
            long tb0 = System.nanoTime();
            ChunkChanges changes = deviceTargets ? new ChunkChanges() : null;
            for (int i = 0; i < count; i++) {
                DeviceCandidate candidate = device.get(from + i);
                for (ComputedContingencyElement element : candidate.elements()) {
                    elemList.add(deviceColumn(element));
                }
                ctgPtr[i + 1] = elemList.size();
                ctgBase[i] = -1;
                // the stock adds the stations of an HVDC left without power to the lost generators FIRST,
                // then decides on lost power
                PropagatedContingency propagated = candidate.connectivity().getPropagatedContingency();
                candidate.connectivity().getHvdcsWithoutPower().forEach(hvdc -> {
                    propagated.getGeneratorIdsToLose().add(hvdc.getConverterStation1().getId());
                    propagated.getGeneratorIdsToLose().add(hvdc.getConverterStation2().getId());
                });
                boolean lostPower = !propagated.getGeneratorIdsToLose().isEmpty() || !propagated.getLoadIdsToLose().isEmpty();
                if (lostPower || needsModifiedTarget(candidate)) {
                    if (changes != null && !changes.add(candidate, lostPower, lfNetwork, equationSystem, generatorIndex)) {
                        changes = null;                      // something the device cannot place: this chunk on the host
                        modifiedTargets -= nBases;
                        nBases = 0;
                        for (int k = 0; k < i; k++) {
                            ctgBase[k] = -1;
                        }
                        i = -1;                              // rebuild the chunk's columns on the host from the start
                        elemList.clear();
                        continue;
                    }
                    if (changes != null && !checkTargets) {
                        ctgBase[i] = nBases++;
                        modifiedTargets++;
                        continue;
                    }
                    if (!lostPower && arrayModelApplies && !checkTargets) {   // the check compares with the exact replay
                        buildModifiedTarget(candidate, lfNetwork, equationSystem, baseTargets, busNetInjection, distribution, targets, nBases * n);
                    } else {
                        buildModifiedTargetByReplay(candidate, lostPower, lfNetwork, context, targets, nBases * n);
                        replayedTargets++;
                    }
                    ctgBase[i] = nBases++;
                    modifiedTargets++;
                }
            }
            int[] ctgElem = elemList.stream().mapToInt(Integer::intValue).toArray();
            if (changes != null && nBases > 0) {
                double[] built = checkTargets ? new double[n * nBases] : null;
                solver.buildChunkTargets(changes.build(), context.getParameters().isDistributedSlack(), participationType,
                        context.getParameters().getNetworkParameters().isUseActiveLimits(), assembly.referenceGeneratorIndex(), built);
                deviceTargetCount += nBases;
                if (built != null) {
                    targetMismatches += compareTargets(built, targets, n * nBases, n, targetMismatches);
                }
                tColumns += System.nanoTime() - tb0;
            } else {
                tColumns += System.nanoTime() - tb0;
                solver.setChunkTargets(targets, nBases);
            }
            long tc0 = System.nanoTime();
            int[] records = solver.woodburyChunk(ctgPtr, ctgElem, ctgBase, GpuViolationScreen.BAND, CAPACITY, states);
            tDevice += System.nanoTime() - tc0;

            long th0 = System.nanoTime();
            for (int i = 0; i < count; i++) {
                DeviceCandidate candidate = device.get(from + i);
                PropagatedContingency propagated = candidate.connectivity().getPropagatedContingency();
                LfContingency lfContingency = candidate.lfContingency();
                Contingency contingency = propagated.getContingency();
                // a child of the network's report node, as the AC security analysis does: not of the previous
                // contingency's, which nested every contingency in the one before (and re-checked every ancestor's values)
                ReportNode postContSimReportNode = Reports.createPostContingencySimulation(networkReportNode, contingency.getId());
                lfNetwork.setReportNode(postContSimReportNode);
                logPostContingencyStart(lfNetwork, lfContingency);

                // the post-contingency state: answered from the array where the run allows it (stateFree), else
                // loaded into the equation system, where the vectorized DC network recomputes every branch flow
                double[] state = Arrays.copyOfRange(states, i * n, (i + 1) * n);
                int emitted = records[i];
                boolean fromState = stateFree && emitted <= CAPACITY && monitoredBranchResults.appliesTo(contingency.getId());
                if (!fromState || checkStateReader) {
                    equationSystem.getStateVector().set(state);
                    if (needAngles) {
                        updateNetwork(lfNetwork, equationSystem, state);
                    }
                }
                Map<LfBranch, DisabledBranchStatus> disabledBranches = lfContingency.getDisabledNetwork().getBranchesStatus();
                Predicate<LfBranch> isBranchDisabled = disabledBranches::containsKey;

                // violations: the screen's candidates confirmed on OLF's own values (read from the state, or from the model)
                var postContingencyLimitViolationManager = new LimitViolationManager(preContingencyLimitViolationManager, limitReductions, violationsParameters);
                if (emitted > CAPACITY) {
                    hostScreened++;
                    postContingencyLimitViolationManager.detectViolations(lfNetwork, isBranchDisabled);
                } else {
                    int[] candidates = Arrays.copyOfRange(records, count + i * CAPACITY, count + i * CAPACITY + emitted);
                    Arrays.sort(candidates);             // table order: a group's limits contiguous, most severe first
                    scanner.detectFromCandidates(lfNetwork, postContingencyLimitViolationManager, candidates, isBranchDisabled, new int[0],
                            fromState ? stateReader : null, state, dcPowerFactor);
                }

                NetworkResult networkResult = fromState
                        ? networkResultFromState(monitoredBranchResults, lfNetwork, candidate, contingency, isBranchDisabled, state, checkStateReader,
                                monitorIndexes, preContingencyNetworkResult, dcPowerFactor, securityAnalysisParameters.getModifiedMonitoredElementsParameters())
                        : networkResult(monitoredBranchResults, lfNetwork, candidate, contingency, isBranchDisabled,
                                needBusResults, monitorIndexes, createResultExtension, preContingencyNetworkResult, dcPowerFactor,
                                securityAnalysisParameters.getModifiedMonitoredElementsParameters());

                var connectivityResult = new ConnectivityResult(
                        lfContingency.getCreatedSynchronousComponentsCount(), 0,
                        lfContingency.getDisconnectedLoadActivePower() * PerUnit.SB,
                        lfContingency.getDisconnectedGenerationActivePower() * PerUnit.SB,
                        lfContingency.getDisconnectedElementIds());
                LimitViolationsResult postContingencyViolations = new LimitViolationsResult(postContingencyLimitViolationManager.getLimitViolations());
                resultById.put(contingency.getId(), new PostContingencyResult(contingency, PostContingencyComputationStatus.CONVERGED,
                        postContingencyViolations, networkResult, connectivityResult, Double.NaN));

                // the operator strategies of this contingency: the condition is read on the post-contingency
                // violations (and state), exactly where the stock reads it
                List<Indexed<OperatorStrategy>> strategies = strategySetup.operatorStrategiesByContingencyId().get(contingency.getId());
                if (strategies != null) {
                    for (Indexed<OperatorStrategy> operatorStrategy : strategies) {
                        List<String> actionIds = checkCondition(operatorStrategy.value(), postContingencyViolations, lfNetwork);
                        List<LfAction> lfActions = actionIds.stream().map(strategySetup.lfActionById()::get).filter(Objects::nonNull).toList();
                        LfOperatorStrategy lfOperatorStrategy = new LfOperatorStrategy(operatorStrategy, lfActions);
                        strategyScenarios.add(new StrategyScenario(candidate, operatorStrategy, lfOperatorStrategy, null, null, postContSimReportNode));
                    }
                }
            }
            tHost += System.nanoTime() - th0;
        }
        if (!strategyScenarios.isEmpty()) {
            runStrategies(solver, lfNetwork, context, strategyScenarios, strategySetup, elementByBranch, baseTargets, busNetInjection, distribution,
                    arrayModelApplies, scanner, table, needAngles, preContingencyNetworkResult, preContingencyLimitViolationManager,
                    monitorIndexes, createResultExtension, securityAnalysisParameters, limitReductions, dcPowerFactor, violationsParameters,
                    strategyResultsByContingency);
        }
        // leave the base state loaded, as the stock path's restore would
        equationSystem.getStateVector().set(preContingencyStates);
        updateNetwork(lfNetwork, equationSystem, preContingencyStates);
        if (GpuTrace.PHASE_TIME.isOn()) {
            System.err.printf("DC_WOODBURY_PHASE n=%d elements=%d device=%d modifiedTargets=%d (replayed %d, device-built %d) setup=%.0f ms [factor=%.0f branches=%.0f S=%.0f table=%.0f model=%.0f]"
                        + " columns=%.0f ms device=%.0f ms host=%.0f ms hostScreened=%d%n",
                    n, elementCount, device.size(), modifiedTargets, replayedTargets, deviceTargetCount, (tSetup - t0) / 1e6, (tFactor - t0) / 1e6, (tBranches - tFactor) / 1e6,
                    (tS - tBranches) / 1e6, (tTable - tS) / 1e6, (tSetup - tTable) / 1e6, tColumns / 1e6, tDevice / 1e6, tHost / 1e6, hostScreened);
        }
        DEVICE_TARGET_COUNT.addAndGet(deviceTargetCount);
        TARGET_MISMATCH_COUNT.addAndGet(targetMismatches);
        return device.size();
    }

    /**
     * The operator strategies of the device contingencies: per (contingency, strategy) scenario the
     * stock's connectivity re-check (criterion on the device S over the contingency elements and the
     * disabling switch actions, OLF's graph confirmation when it fires), the modified target with the
     * actions (the exact replay), Woodbury over contingency AND action elements, then the stock result
     * built with the contingency and the actions APPLIED on the network (the vectorized flows of a
     * re-tapped transformer need its new model), and restored.
     */
    private void runStrategies(GpuDcSolver solver, LfNetwork lfNetwork, DcLoadFlowContext context, List<StrategyScenario> scenarios,
                               StrategySetup strategySetup, Map<String, ComputedContingencyElement> elementByBranch,
                               double[] baseTargets, double[] busNetInjection, DcSlackDistributionModel distribution, boolean arrayModelApplies,
                               GpuLimitScanner scanner, GpuLimitTable table, boolean needAngles,
                               PreContingencyNetworkResult preContingencyNetworkResult, LimitViolationManager preContingencyLimitViolationManager,
                               AbstractNetworkResult.StateMonitorIndexes monitorIndexes, boolean createResultExtension,
                               SecurityAnalysisParameters securityAnalysisParameters, List<LimitReduction> limitReductions, double dcPowerFactor,
                               SecurityAnalysisParameters.IncreasedViolationsParameters violationsParameters,
                               Map<String, List<OperatorStrategyResult>> strategyResultsByContingency) {
        EquationSystem<DcVariableType, DcEquationType> equationSystem = context.getEquationSystem();
        int n = equationSystem.getIndex().getColumnCount();
        DcLoadFlowParameters parameters = context.getParameters();

        // --- connectivity re-check per scenario (ConnectivityBreakAnalysis.processPostContingencyAndPostOperatorStrategyConnectivityAnalysisResult)
        int[] ctgPtr = new int[scenarios.size() + 1];
        List<Integer> criterionElems = new ArrayList<>();
        for (int i = 0; i < scenarios.size(); i++) {
            StrategyScenario sc = scenarios.get(i);
            PropagatedContingency c = sc.candidate().connectivity().getPropagatedContingency();
            for (String id : c.getBranchIdsToOpen().keySet()) {
                ComputedContingencyElement element = elementByBranch.get(id);
                if (element != null) {
                    criterionElems.add(deviceColumn(element));
                }
            }
            for (LfAction action : sc.lfOperatorStrategy().getActions()) {
                List<ComputedElement> elements = strategySetup.actionElementsByLfAction().get(action);
                if (elements != null) {
                    for (ComputedElement element : elements) {
                        if (element instanceof ComputedSwitchBranchElement sw && !sw.isEnabled()) {
                            criterionElems.add(deviceColumn(element));
                        }
                    }
                }
            }
            ctgPtr[i + 1] = criterionElems.size();
        }
        double[] maxSum = solver.connectivityCriterion(ctgPtr, criterionElems.stream().mapToInt(Integer::intValue).toArray());
        List<StrategyScenario> resolved = new ArrayList<>();
        for (int i = 0; i < scenarios.size(); i++) {
            StrategyScenario sc = scenarios.get(i);
            ConnectivityBreakAnalysis.ConnectivityAnalysisResult postContingency = sc.candidate().connectivity();
            LfOperatorStrategy los = sc.lfOperatorStrategy();
            boolean topological = los.getActions().stream().anyMatch(a -> a instanceof AbstractLfBranchAction<?>);
            ConnectivityBreakAnalysis.ConnectivityAnalysisResult connectivity;
            if (!topological || maxSum[i] <= 1d - CONNECTIVITY_LOSS_THRESHOLD) {
                connectivity = postContingency.withOperatorStrategy(los);
            } else {
                connectivity = ConnectivityBreakAnalysis.computeConnectivityAnalysisResult(lfNetwork, postContingency.getPropagatedContingency(),
                                elementByBranch, los, strategySetup.actionElementsByLfAction(), false)
                        .orElseGet(() -> ConnectivityBreakAnalysis.ConnectivityAnalysisResult.createNonBreakingConnectivityAnalysisResult(
                                postContingency.getPropagatedContingency(), los, lfNetwork));
            }
            Set<String> toReconnect = connectivity.getElementsToReconnect();
            List<ComputedElement> elements = new ArrayList<>();
            for (String id : postContingency.getPropagatedContingency().getBranchIdsToOpen().keySet()) {
                if (!toReconnect.contains(id)) {
                    elements.add(elementByBranch.get(id));
                }
            }
            for (LfAction action : los.getActions()) {
                List<ComputedElement> actionElements = strategySetup.actionElementsByLfAction().get(action);
                if (actionElements != null) {
                    for (ComputedElement element : actionElements) {
                        if (!toReconnect.contains(element.getLfBranch().getId())) {
                            elements.add(element);
                        }
                    }
                }
            }
            if (elements.size() > GpuDcSolver.MAX_ELEMENTS) {
                throw new IllegalStateException("operator strategy " + los.getIndexedOperatorStrategy().value().getId()
                        + " has more than " + GpuDcSolver.MAX_ELEMENTS + " Woodbury elements");
            }
            resolved.add(new StrategyScenario(sc.candidate(), sc.operatorStrategy(), los, connectivity, elements, sc.contingencyReportNode()));
        }

        // --- chunks: every strategy scenario has a modified target (the stock re-solves whenever an action is present)
        double[] states = new double[n * DEVICE_CHUNK];
        double[] targets = new double[n * DEVICE_CHUNK];
        NetworkState networkState = NetworkState.save(lfNetwork);
        for (int from = 0; from < resolved.size(); from += DEVICE_CHUNK) {
            int to = Math.min(from + DEVICE_CHUNK, resolved.size());
            int count = to - from;
            int[] ptr = new int[count + 1];
            int[] base = new int[count];
            List<Integer> elemList = new ArrayList<>();
            for (int i = 0; i < count; i++) {
                StrategyScenario sc = resolved.get(from + i);
                for (ComputedElement element : sc.elements()) {
                    elemList.add(deviceColumn(element));
                }
                ptr[i + 1] = elemList.size();
                PropagatedContingency propagated = sc.connectivity().getPropagatedContingency();
                boolean lostPower = !propagated.getGeneratorIdsToLose().isEmpty() || !propagated.getLoadIdsToLose().isEmpty();
                buildModifiedTargetByReplay(new DeviceCandidate(sc.connectivity(), List.of(), sc.candidate().lfContingency()), lostPower,
                        sc.lfOperatorStrategy().getActions(), lfNetwork, context, targets, i * n);
                base[i] = i;
            }
            solver.setChunkTargets(targets, count);
            int[] records = solver.woodburyChunk(ptr, elemList.stream().mapToInt(Integer::intValue).toArray(), base, GpuViolationScreen.BAND, CAPACITY, states);

            for (int i = 0; i < count; i++) {
                StrategyScenario sc = resolved.get(from + i);
                LfContingency lfContingency = sc.candidate().lfContingency();
                Contingency contingency = sc.connectivity().getPropagatedContingency().getContingency();
                OperatorStrategy operatorStrategy = sc.operatorStrategy().value();
                List<LfAction> lfActions = sc.lfOperatorStrategy().getActions();
                ReportNode osSimReportNode = Reports.createOperatorStrategySimulation(sc.contingencyReportNode(), operatorStrategy.getId());
                lfNetwork.setReportNode(osSimReportNode);

                // apply first (the stock, on this OLF: flows are evaluated on the final topology and tap positions)
                lfContingency.apply(parameters.getBalanceType());
                LfActionUtils.applyListOfActions(lfActions, lfNetwork, lfContingency, parameters.getNetworkParameters());
                double[] state = Arrays.copyOfRange(states, i * n, (i + 1) * n);
                equationSystem.getStateVector().set(state);
                if (needAngles) {
                    updateNetwork(lfNetwork, equationSystem, state);
                }
                // the stock predicate: the contingency's branches and the partially disabled ones of THIS connectivity result
                Set<LfBranch> disabledBranches = new HashSet<>();
                sc.connectivity().getPropagatedContingency().getBranchIdsToOpen().keySet().forEach(id -> disabledBranches.add(lfNetwork.getBranchById(id)));
                disabledBranches.addAll(sc.connectivity().getPartialDisabledBranches());
                Predicate<LfBranch> isBranchDisabled = disabledBranches::contains;

                var violationManager = new LimitViolationManager(preContingencyLimitViolationManager, limitReductions, violationsParameters);
                int emitted = records[i];
                // the action branches' models moved on the host (tap, switch): the screen's flows for them are not
                // the host's, so they are scanned in full
                int[] uncovered = lfActions.stream()
                        .map(strategySetup.actionElementsByLfAction()::get).filter(Objects::nonNull).flatMap(Collection::stream)
                        .mapToInt(e -> e.getLfBranch().getNum()).distinct().toArray();
                if (emitted > CAPACITY) {
                    violationManager.detectViolations(lfNetwork, isBranchDisabled);
                } else {
                    int[] candidates = Arrays.copyOfRange(records, count + i * CAPACITY, count + i * CAPACITY + emitted);
                    Arrays.sort(candidates);
                    scanner.detectFromCandidates(lfNetwork, violationManager, candidates, isBranchDisabled, uncovered);
                }
                var postActionsNetworkResult = new PostContingencyNetworkResult(lfNetwork, monitorIndexes, createResultExtension,
                        preContingencyNetworkResult, contingency, LoadFlowModel.DC, dcPowerFactor,
                        securityAnalysisParameters.getModifiedMonitoredElementsParameters());
                postActionsNetworkResult.update(isBranchDisabled);
                strategyResultsByContingency.computeIfAbsent(contingency.getId(), k -> new ArrayList<>()).add(new OperatorStrategyResult(operatorStrategy,
                        List.of(new OperatorStrategyResult.ConditionalActionsResult(operatorStrategy.getId(), PostContingencyComputationStatus.CONVERGED,
                                new LimitViolationsResult(violationManager.getLimitViolations()),
                                new NetworkResult(postActionsNetworkResult.getBranchResults(), postActionsNetworkResult.getBusResults(),
                                        postActionsNetworkResult.getThreeWindingsTransformerResults()),
                                Double.NaN))));
                networkState.restore();
            }
        }
    }

    /** The branch model, in LfNetwork order (== GpuLimitTable order == branch num). */
    private static GpuDcStateReader prepareBranches(GpuDcSolver solver, LfNetwork lfNetwork, EquationSystem<DcVariableType, DcEquationType> equationSystem,
                                        DcEquationSystemCreationParameters creationParameters) {
        int busCount = lfNetwork.getBuses().size();
        int branchCount = lfNetwork.getBranches().size();
        int[] busPhRow = new int[busCount];
        int[] branchA1Row = new int[branchCount];
        Arrays.fill(busPhRow, -1);
        Arrays.fill(branchA1Row, -1);
        for (Variable<DcVariableType> v : equationSystem.getIndex().getSortedVariablesToFind()) {
            switch (v.getType()) {
                case BUS_PHI -> busPhRow[v.getElementNum()] = v.getRow();
                case BRANCH_ALPHA1 -> branchA1Row[v.getElementNum()] = v.getRow();
                default -> { }
            }
        }
        int[] ph1Row = new int[branchCount];
        int[] ph2Row = new int[branchCount];
        int[] a1Row = new int[branchCount];
        double[] a1Const = new double[branchCount];
        double[] power = new double[branchCount];
        for (int b = 0; b < branchCount; b++) {
            LfBranch branch = lfNetwork.getBranches().get(b);
            if (branch.getNum() != b) {
                throw new IllegalStateException("branch numbering is not positional");
            }
            LfBus bus1 = branch.getBus1();
            LfBus bus2 = branch.getBus2();
            boolean flowing = bus1 != null && bus2 != null && !branch.isZeroImpedance(LoadFlowModel.DC);
            ph1Row[b] = flowing ? busPhRow[bus1.getNum()] : -1;
            ph2Row[b] = flowing ? busPhRow[bus2.getNum()] : -1;
            a1Row[b] = flowing ? branchA1Row[b] : -1;
            a1Const[b] = branch.getPiModel().getA1();
            power[b] = flowing ? AbstractClosedBranchDcFlowEquationTerm.computePower(creationParameters.isUseTransformerRatio(),
                    creationParameters.getDcApproximationType(), branch.getPiModel()) : 0;
        }
        solver.setBranches(ph1Row, ph2Row, a1Row, a1Const, power);
        return new GpuDcStateReader(ph1Row, ph2Row, a1Row, a1Const, power);
    }

    private int actionElementOffset;

    /** A device column index for any computed element: contingency elements at OLF's index, action elements after them. */
    private int deviceColumn(ComputedElement element) {
        return element instanceof ComputedContingencyElement ? element.getComputedElementIndex() : actionElementOffset + element.getComputedElementIndex();
    }

    /**
     * The elements' +1/-1 rows (ComputedElement.fillRhs) at OLF's column indices — the contingency
     * elements first, the action elements after them — with WoodburyEngine's per-element terms;
     * -1 when the device refuses S.
     */
    private int prepareElements(GpuDcSolver solver, EquationSystem<DcVariableType, DcEquationType> equationSystem,
                                DcEquationSystemCreationParameters creationParameters,
                                Collection<ComputedContingencyElement> contingencyElements, List<ComputedElement> actionElements,
                                StrategySetup strategySetup) {
        actionElementOffset = contingencyElements.stream().mapToInt(ComputedContingencyElement::getComputedElementIndex).max().orElse(-1) + 1;
        int actionCount = actionElements.stream().mapToInt(ComputedElement::getComputedElementIndex).max().orElse(-1) + 1;
        int elementCount = actionElementOffset + actionCount;
        int[] rowPlus = new int[elementCount];
        int[] rowMinus = new int[elementCount];
        int[] elemBranch = new int[elementCount];
        double[] diag = new double[elementCount];
        double[] newAlpha = new double[elementCount];
        Arrays.fill(rowPlus, -1);
        Arrays.fill(rowMinus, -1);
        Arrays.fill(elemBranch, -1);
        List<ComputedElement> all = new ArrayList<>(contingencyElements);
        all.addAll(actionElements);
        for (ComputedElement element : all) {
            LfBranch branch = element.getLfBranch();
            int e = deviceColumn(element);
            elemBranch[e] = branch.getNum();
            // WoodburyEngine.getAlphaMatrixValue's diagonal and getAlphaRhsValue's new alpha
            double power = AbstractClosedBranchDcFlowEquationTerm.computePower(creationParameters.isUseTransformerRatio(),
                    creationParameters.getDcApproximationType(), branch.getPiModel());
            if (element instanceof ComputedContingencyElement) {
                diag[e] = 1d / power;
            } else if (element instanceof ComputedTapPositionChangeElement tap) {
                TapPositionChange change = tap.getTapPositionChange();
                double newPower = AbstractClosedBranchDcFlowEquationTerm.computePower(creationParameters.isUseTransformerRatio(),
                        creationParameters.getDcApproximationType(), change.getNewPiModel());
                diag[e] = 1d / (power - newPower);
                newAlpha[e] = change.getNewPiModel().getA1();
            } else if (element instanceof ComputedSwitchBranchElement sw) {
                diag[e] = sw.isEnabled() ? 1d / (0 - power) : 1d / power;
            } else {
                throw new IllegalStateException("Unexpected computed element type " + element.getClass().getSimpleName());
            }
            if (branch.getBus1() == null || branch.getBus2() == null) {
                continue;
            }
            LfBus bus1 = branch.getBus1();
            LfBus bus2 = branch.getBus2();
            if (bus1.isSlack()) {
                rowMinus[e] = pColumn(equationSystem, bus2);
            } else if (bus2.isSlack()) {
                rowPlus[e] = pColumn(equationSystem, bus1);
            } else {
                rowPlus[e] = pColumn(equationSystem, bus1);
                rowMinus[e] = pColumn(equationSystem, bus2);
            }
        }
        if (!solver.setElements(rowPlus, rowMinus, elemBranch)) {
            return -1;
        }
        solver.setElementAttributes(diag, newAlpha);
        return elementCount;
    }

    private int lastPotentiallyBreaking;

    /**
     * ConnectivityBreakAnalysis.run with the sensitivity criterion on the device S: the non-breaking
     * list is the criterion's non-breaking contingencies in order, then the graph-confirmed ones in
     * order; the breaking list is the graph's, in order — the stock lists, the stock order.
     */
    private ConnectivityBreakAnalysis.ConnectivityBreakAnalysisResults connectivityAnalysis(GpuDcSolver solver, LfNetwork lfNetwork,
                                                                                            List<PropagatedContingency> contingencies,
                                                                                            Map<String, ComputedContingencyElement> elementByBranch) {
        int count = contingencies.size();
        int[] ctgPtr = new int[count + 1];
        List<Integer> elemList = new ArrayList<>();
        boolean[] incomplete = new boolean[count];
        for (int c = 0; c < count; c++) {
            for (String id : contingencies.get(c).getBranchIdsToOpen().keySet()) {
                ComputedContingencyElement element = elementByBranch.get(id);
                if (element == null) {
                    incomplete[c] = true;                    // no DC flow equation: the stock path handles it
                } else {
                    elemList.add(element.getComputedElementIndex());
                }
            }
            ctgPtr[c + 1] = elemList.size();
        }
        double[] maxSum = solver.connectivityCriterion(ctgPtr, elemList.stream().mapToInt(Integer::intValue).toArray());

        List<ConnectivityBreakAnalysis.ConnectivityAnalysisResult> nonBreaking = new ArrayList<>();
        List<ConnectivityBreakAnalysis.ConnectivityAnalysisResult> breaking = new ArrayList<>();
        List<PropagatedContingency> potentiallyBreaking = new ArrayList<>();
        for (int c = 0; c < count; c++) {
            PropagatedContingency contingency = contingencies.get(c);
            if (incomplete[c] || maxSum[c] > 1d - CONNECTIVITY_LOSS_THRESHOLD) {
                potentiallyBreaking.add(contingency);
            } else {
                nonBreaking.add(ConnectivityBreakAnalysis.ConnectivityAnalysisResult.createNonBreakingConnectivityAnalysisResult(contingency, null, lfNetwork));
            }
        }
        lastPotentiallyBreaking = potentiallyBreaking.size();
        for (PropagatedContingency contingency : potentiallyBreaking) {
            // the security analysis never reads the slack's connected component: not materialized (a copy of every bus of
            // the main component per breaking contingency, most of this loop's time on network.xiidm)
            ConnectivityBreakAnalysis.computeConnectivityAnalysisResult(lfNetwork, contingency, elementByBranch, null, Collections.emptyMap(), false)
                    .ifPresentOrElse(breaking::add,
                        () -> nonBreaking.add(ConnectivityBreakAnalysis.ConnectivityAnalysisResult.createNonBreakingConnectivityAnalysisResult(contingency, null, lfNetwork)));
        }
        return new ConnectivityBreakAnalysis.ConnectivityBreakAnalysisResults(nonBreaking, breaking, com.powsybl.math.matrix.DenseMatrix.EMPTY, elementByBranch);
    }

    /** ConnectivityBreakAnalysis.CONNECTIVITY_LOSS_THRESHOLD (private there): 10e-7. */
    private static final double CONNECTIVITY_LOSS_THRESHOLD = 10e-7;

    /** The stock rule: a modified target when buses are islanded or a NON-reconnecting phase shifter is lost. */
    private static boolean needsModifiedTarget(DeviceCandidate candidate) {
        if (!candidate.connectivity().getDisabledBuses().isEmpty()) {
            return true;
        }
        for (ComputedContingencyElement element : candidate.elements()) {
            if (element.getLfBranch().hasPhaseControllerCapability()) {
                return true;
            }
        }
        return false;
    }

    /**
     * The target column of {@code WoodburyEngine.runDcLoadFlowWithModifiedTargetVector} for this
     * contingency, built from the base target vector without touching the network: the slack
     * redistributed over the buses that remain (the array replay of {@code distributeSlack}), the
     * islanded buses' P rows and the outaged phase shifters' alpha rows zeroed.
     */
    private static void buildModifiedTarget(DeviceCandidate candidate, LfNetwork lfNetwork,
                                            EquationSystem<DcVariableType, DcEquationType> equationSystem, double[] baseTargets,
                                            double[] busNetInjection, DcSlackDistributionModel distribution, double[] out, int offset) {
        int n = baseTargets.length;
        System.arraycopy(baseTargets, 0, out, offset, n);
        Set<LfBus> disabledBuses = candidate.connectivity().getDisabledBuses();
        if (distribution != null) {
            Set<Integer> excluded = new HashSet<>();
            disabledBuses.forEach(b -> excluded.add(b.getNum()));
            // DcLoadFlowEngine.getActivePowerMismatch(remainingBuses): -sum of (generation - load) over the
            // non-disabled remaining buses, in bus order
            double sum = 0;
            for (int b = 0; b < busNetInjection.length; b++) {
                if (!excluded.contains(b) && !Double.isNaN(busNetInjection[b])) {
                    sum += busNetInjection[b];
                }
            }
            double mismatch = -sum;
            for (Map.Entry<Integer, Double> e : distribution.distribute(excluded, mismatch).entrySet()) {
                int column = pColumnOrMinusOne(equationSystem, lfNetwork.getBus(e.getKey()));
                if (column >= 0) {
                    out[offset + column] += e.getValue();
                }
            }
        }
        for (LfBus bus : disabledBuses) {
            int column = pColumnOrMinusOne(equationSystem, bus);
            if (column >= 0) {
                out[offset + column] = 0;
            }
        }
        // disabled branches: the contingency's own (all of them, reconnecting or not) and the partially disabled ones
        PropagatedContingency c = candidate.connectivity().getPropagatedContingency();
        List<LfBranch> disabledBranches = new ArrayList<>();
        c.getBranchIdsToOpen().keySet().forEach(id -> disabledBranches.add(lfNetwork.getBranchById(id)));
        disabledBranches.addAll(candidate.connectivity().getPartialDisabledBranches());
        for (LfBranch branch : disabledBranches) {
            equationSystem.getEquation(branch.getNum(), DcEquationType.BRANCH_TARGET_ALPHA1)
                    .filter(Equation::isActive)
                    .ifPresent(eq -> out[offset + eq.getColumn()] = 0);
        }
    }

    /**
     * The stock's own column, mutation for mutation, restored afterwards: the lost-power branch of
     * {@code calculatePostContingencyAndOperatorStrategyStates} (bus DC states saved,
     * {@code processLostPowerChanges}), then {@code runDcLoadFlowWithModifiedTargetVector} minus its
     * solve ({@code distributeSlack} on the remaining buses, the target vector re-read from the
     * network, islanded P rows and disabled alpha rows zeroed). Exact for every balance type and for
     * multi-slack networks, at the price of an O(buses) save/restore — the array model is the fast
     * path where it applies.
     */
    private static void buildModifiedTargetByReplay(DeviceCandidate candidate, boolean lostPower, LfNetwork lfNetwork,
                                                    DcLoadFlowContext context, double[] out, int offset) {
        buildModifiedTargetByReplay(candidate, lostPower, List.of(), lfNetwork, context, out, offset);
    }

    private static void buildModifiedTargetByReplay(DeviceCandidate candidate, boolean lostPower, List<LfAction> lfActions, LfNetwork lfNetwork,
                                                    DcLoadFlowContext context, double[] out, int offset) {
        DcLoadFlowParameters parameters = context.getParameters();
        Set<LfBus> disabledBuses = candidate.connectivity().getDisabledBuses();
        List<BusDcState> busStates = ElementState.save(lfNetwork.getBuses(), BusDcState::save);
        try {
            if (lostPower) {
                candidate.connectivity().toLfContingency()
                        .ifPresent(lfContingency -> lfContingency.processLostPowerChanges(parameters.getBalanceType(), false));
            }
            if (parameters.isDistributedSlack()) {
                Collection<LfBus> remaining;
                if (disabledBuses.isEmpty()) {
                    remaining = lfNetwork.getBuses();
                } else {
                    remaining = new LinkedHashSet<>(lfNetwork.getBuses());
                    remaining.removeAll(disabledBuses);
                }
                com.powsybl.openloadflow.dc.DcLoadFlowEngine.distributeSlack(lfNetwork, remaining, parameters.getBalanceType(),
                        parameters.getNetworkParameters().isUseActiveLimits());
            }
            double[] column = context.getTargetVector().getArray();      // recomputed from the mutated network
            System.arraycopy(column, 0, out, offset, column.length);
        } finally {
            ElementState.restore(busStates);
        }
        EquationSystem<DcVariableType, DcEquationType> equationSystem = context.getEquationSystem();
        for (LfBus bus : disabledBuses) {
            int column = pColumnOrMinusOne(equationSystem, bus);
            if (column >= 0) {
                out[offset + column] = 0;
            }
        }
        PropagatedContingency c = candidate.connectivity().getPropagatedContingency();
        List<LfBranch> disabledBranches = new ArrayList<>();
        c.getBranchIdsToOpen().keySet().forEach(id -> disabledBranches.add(lfNetwork.getBranchById(id)));
        disabledBranches.addAll(candidate.connectivity().getPartialDisabledBranches());
        for (LfBranch branch : disabledBranches) {
            equationSystem.getEquation(branch.getNum(), DcEquationType.BRANCH_TARGET_ALPHA1)
                    .filter(Equation::isActive)
                    .ifPresent(eq -> out[offset + eq.getColumn()] = 0);
        }
        if (!lfActions.isEmpty()) {
            // WoodburyEngine.runDcLoadFlowWithModifiedTargetVector, the actions part, in its order
            for (LfAction action : lfActions) {
                if (action instanceof AbstractLfTapChangerAction<?> tapAction && tapAction.getChange() != null) {
                    LfBranch branch = tapAction.getChange().getBranch();
                    equationSystem.getEquation(branch.getNum(), DcEquationType.BRANCH_TARGET_ALPHA1)
                            .ifPresent(eq -> out[offset + eq.getColumn()] = tapAction.getChange().getNewPiModel().getA1());
                }
            }
            for (LfAction action : lfActions) {
                if (action instanceof LfGeneratorAction genAction && !genAction.getGenerator().isDisabled()) {
                    LfBus bus = genAction.getGenerator().getBus();
                    double deltaTargetP = genAction.getNewTargetP() - genAction.getGenerator().getTargetP();
                    equationSystem.getEquation(bus.getNum(), DcEquationType.BUS_TARGET_P)
                            .filter(Equation::isActive)
                            .ifPresent(eq -> out[offset + eq.getColumn()] += deltaTargetP);
                }
            }
            for (LfAction action : lfActions) {
                if (action instanceof LfLoadAction loadAction && loadAction.isValid() && !loadAction.getLfLoad().isOriginalLoadDisabled(loadAction.getLoadId())) {
                    LfBus bus = loadAction.getLfLoad().getBus();
                    double deltaTargetP = -loadAction.getPowerShift().getActive();
                    equationSystem.getEquation(bus.getNum(), DcEquationType.BUS_TARGET_P)
                            .filter(Equation::isActive)
                            .ifPresent(eq -> out[offset + eq.getColumn()] += deltaTargetP);
                }
            }
            for (LfAction action : lfActions) {
                if (action instanceof AbstractLfBranchAction<?> branchAction && branchAction.getDisabledBranches() != null) {
                    for (LfBranch branch : branchAction.getDisabledBranches()) {
                        equationSystem.getEquation(branch.getNum(), DcEquationType.BRANCH_TARGET_ALPHA1)
                                .ifPresent(eq -> out[offset + eq.getColumn()] = 0);
                    }
                }
            }
        }
    }

    /** The limit table, built as the host scanner reads it, and its copy on the device for the screen. */
    private static GpuLimitTable uploadLimitTable(GpuDcSolver solver, LfNetwork lfNetwork, List<LimitReduction> limitReductions, double dcPowerFactor) {
        int branchCount = lfNetwork.getBranches().size();
        LimitReductionManager limitReductionManager = LimitReductionManager.create(limitReductions);
        GpuLimitTable table = GpuLimitTable.build(lfNetwork, limitReductionManager);
        int groupCount = table.groupCount();
        int[] branchGroupStart = new int[branchCount];
        int[] branchGroupEnd = new int[branchCount];
        for (int b = 0; b < branchCount; b++) {
            branchGroupStart[b] = table.groupStart(b);
            branchGroupEnd[b] = table.groupEnd(b);
        }
        int[] groupQuantity = new int[groupCount];
        int[] groupLimitStart = new int[groupCount];
        int[] groupLimitEnd = new int[groupCount];
        for (int g = 0; g < groupCount; g++) {
            groupQuantity[g] = table.quantity(g);
            groupLimitStart[g] = table.limitStart(g);
            groupLimitEnd[g] = table.limitEnd(g);
        }
        double[] limitThreshold = new double[table.limitCount()];
        for (int l = 0; l < limitThreshold.length; l++) {
            limitThreshold[l] = table.threshold(l);
        }
        solver.setLimitTable(branchGroupStart, branchGroupEnd, groupQuantity, groupLimitStart, groupLimitEnd, limitThreshold, dcPowerFactor);
        return table;
    }

    /**
     * The contingency's network result: the monitored branches found once per run (GpuMonitoredBranchResults) where
     * that covers it, OLF's own {@code PostContingencyNetworkResult} over the loaded state otherwise. The stock path's
     * apply() flags the islanded buses disabled and the bus results skip them: flagged for the duration of OLF's build
     * when bus results are wanted at all.
     */
    private static NetworkResult networkResult(GpuMonitoredBranchResults monitoredBranchResults, LfNetwork lfNetwork, DeviceCandidate candidate,
                                               Contingency contingency, Predicate<LfBranch> isBranchDisabled, boolean needBusResults,
                                               AbstractNetworkResult.StateMonitorIndexes monitorIndexes, boolean createResultExtension,
                                               PreContingencyNetworkResult preContingencyNetworkResult, double dcPowerFactor,
                                               SecurityAnalysisParameters.ModifiedMonitoredElementsParameters modified) {
        boolean fast = monitoredBranchResults.appliesTo(contingency.getId());
        List<BranchResult> fastBranchResults = fast
                ? monitoredBranchResults.branchResults(contingency, isBranchDisabled, preContingencyNetworkResult, createResultExtension, modified)
                : null;
        if (fast) {
            FAST_NETWORK_RESULT_COUNT.incrementAndGet();
        }
        if (fast && !CHECK_NETWORK_RESULTS.get()) {
            return new NetworkResult(fastBranchResults, List.of(), List.of());
        }
        Set<LfBus> islanded = candidate.connectivity().getDisabledBuses();
        boolean flagBuses = needBusResults && !islanded.isEmpty();
        if (flagBuses) {
            islanded.forEach(bus -> bus.setDisabled(true));
        }
        var postContingencyNetworkResult = new PostContingencyNetworkResult(lfNetwork, monitorIndexes, createResultExtension,
                preContingencyNetworkResult, contingency, LoadFlowModel.DC, dcPowerFactor, modified);
        postContingencyNetworkResult.update(isBranchDisabled);
        if (flagBuses) {
            islanded.forEach(bus -> bus.setDisabled(false));
        }
        if (fast) {
            NETWORK_RESULT_MISMATCH_COUNT.addAndGet(GpuMonitoredBranchResults.compare(fastBranchResults, postContingencyNetworkResult.getBranchResults())
                    + postContingencyNetworkResult.getBusResults().size() + postContingencyNetworkResult.getThreeWindingsTransformerResults().size());
        }
        return new NetworkResult(postContingencyNetworkResult.getBranchResults(), postContingencyNetworkResult.getBusResults(),
                postContingencyNetworkResult.getThreeWindingsTransformerResults());
    }

    /** The contingency's network result read from its state (stateFree runs); the check hook compares it with OLF's. */
    private NetworkResult networkResultFromState(GpuMonitoredBranchResults monitoredBranchResults, LfNetwork lfNetwork, DeviceCandidate candidate,
                                                 Contingency contingency, Predicate<LfBranch> isBranchDisabled, double[] state, boolean check,
                                                 AbstractNetworkResult.StateMonitorIndexes monitorIndexes,
                                                 PreContingencyNetworkResult preContingencyNetworkResult, double dcPowerFactor,
                                                 SecurityAnalysisParameters.ModifiedMonitoredElementsParameters modified) {
        List<BranchResult> branchResults = monitoredBranchResults.branchResultsFromState(contingency, isBranchDisabled, preContingencyNetworkResult,
                modified, stateReader, state, dcPowerFactor);
        STATE_FREE_CONTINGENCY_COUNT.incrementAndGet();
        if (check) {
            List<BranchResult> olf = networkResult(monitoredBranchResults, lfNetwork, candidate, contingency, isBranchDisabled, false, monitorIndexes,
                    false, preContingencyNetworkResult, dcPowerFactor, modified).getBranchResults();
            STATE_READER_MISMATCH_COUNT.addAndGet(compareStateReader(lfNetwork, state, dcPowerFactor) + GpuMonitoredBranchResults.compare(branchResults, olf));
        }
        return new NetworkResult(branchResults, List.of(), List.of());
    }

    /** The check hook: every branch value of the state reader against OLF's model with the state loaded; the differences. */
    private int compareStateReader(LfNetwork lfNetwork, double[] state, double dcPowerFactor) {
        int differences = 0;
        for (LfBranch branch : lfNetwork.getBranches()) {
            double[] reader = {stateReader.p(state, branch, GpuTwoSides.ONE), stateReader.p(state, branch, GpuTwoSides.TWO),
                stateReader.i(state, branch, GpuTwoSides.ONE, dcPowerFactor), stateReader.i(state, branch, GpuTwoSides.TWO, dcPowerFactor),
                stateReader.s(state, branch, GpuTwoSides.ONE), stateReader.s(state, branch, GpuTwoSides.TWO)};
            double[] model = {branch.getP1().eval(), branch.getP2().eval(), branch.getI1().eval(), branch.getI2().eval(),
                branch.computeApparentPower1(), branch.computeApparentPower2()};
            for (int k = 0; k < reader.length; k++) {
                if (Double.doubleToLongBits(reader[k]) != Double.doubleToLongBits(model[k])) {
                    if (differences < 5) {
                        System.err.printf("DC_STATE_READER_CHECK branch %s value %d: reader %.17g model %.17g%n", branch.getId(), k, reader[k], model[k]);
                    }
                    differences++;
                }
            }
        }
        return differences;
    }

    /**
     * The device generator and load tables as {@code NetworkState.save} left the network (every initial
     * target P re-anchored to the base dispatch), and each generator's index in them.
     */
    private static Map<LfGenerator, Integer> syncInjectionTables(GpuDcSolver solver, GpuDcAssembly assembly, LfNetwork lfNetwork) {
        Map<LfGenerator, Integer> generatorIndex = new IdentityHashMap<>();
        List<LfGenerator> generators = assembly.generators();
        double[] initialTargets = new double[generators.size()];
        double[] generatorTargets = new double[generators.size()];
        for (int g = 0; g < generators.size(); g++) {
            generatorIndex.put(generators.get(g), g);
            initialTargets[g] = generators.get(g).getInitialTargetP();
            generatorTargets[g] = generators.get(g).getTargetP();
        }
        solver.setGeneratorInitialTargets(initialTargets);
        solver.setGeneratorTargets(generatorTargets);
        double[] busLoads = new double[lfNetwork.getBuses().size()];
        for (LfBus bus : lfNetwork.getBuses()) {
            busLoads[bus.getNum()] = bus.getLoadTargetP();
        }
        solver.setBusLoads(busLoads);
        return generatorIndex;
    }

    /** The check hook: the device-built columns against the host-built ones, bit for bit; returns the differing values. */
    private static int compareTargets(double[] built, double[] targets, int length, int n, int alreadyReported) {
        int mismatches = 0;
        double maxDiff = Double.longBitsToDouble(LAST_TARGET_MAX_DIFF_BITS.get());
        for (int k = 0; k < length; k++) {
            if (Double.doubleToRawLongBits(built[k]) != Double.doubleToRawLongBits(targets[k])) {
                if (alreadyReported + mismatches < 10) {
                    System.err.printf("DC_TARGET_CHECK column %d row %d: device %.17g host %.17g%n", k / n, k % n, built[k], targets[k]);
                }
                mismatches++;
                maxDiff = Math.max(maxDiff, Math.abs(built[k] - targets[k]));
            }
        }
        LAST_TARGET_MAX_DIFF_BITS.set(Double.doubleToRawLongBits(maxDiff));
        return mismatches;
    }

    /**
     * What each modified target column of a chunk changes from the base, for the device build: the
     * same inputs the stock column reads, gathered without touching the network. A lost load's new
     * bus total is summed exactly as {@code AbstractLfBus.getLoadTargetP} sums it, with the load's
     * target as {@code processLostLoads} leaves it (generation balance types: {@code getUpdatedLoadP0}
     * is the shift's active power).
     */
    private static final class ChunkChanges {
        private int count;
        private final List<Integer> excludedPtr = new ArrayList<>(List.of(0));
        private final List<Integer> excludedBus = new ArrayList<>();
        private final List<Integer> loadPtr = new ArrayList<>(List.of(0));
        private final List<Integer> loadBus = new ArrayList<>();
        private final List<Double> loadP = new ArrayList<>();
        private final List<Integer> lostGeneratorPtr = new ArrayList<>(List.of(0));
        private final List<Integer> lostGenerator = new ArrayList<>();
        private final List<Integer> stationPtr = new ArrayList<>(List.of(0));
        private final List<Integer> station = new ArrayList<>();
        private final List<Integer> zeroRowPtr = new ArrayList<>(List.of(0));
        private final List<Integer> zeroRow = new ArrayList<>();

        /** One column; false (nothing added) when an element is not in the device tables. */
        boolean add(DeviceCandidate candidate, boolean lostPower, LfNetwork lfNetwork,
                    EquationSystem<DcVariableType, DcEquationType> equationSystem, Map<LfGenerator, Integer> generatorIndex) {
            List<Integer> gens = new ArrayList<>();
            List<Integer> stations = new ArrayList<>();
            Map<LfBus, Map<LfLoad, Double>> changedLoads = new LinkedHashMap<>();
            if (lostPower) {
                Optional<LfContingency> lfContingency = candidate.connectivity().toLfContingency();
                if (lfContingency.isPresent()) {
                    for (Map.Entry<LfLoad, LfLostLoad> e : lfContingency.get().getLostLoads().entrySet()) {
                        LfLoad load = e.getKey();
                        if (load.getBus() == null) {
                            return false;
                        }
                        double p0 = e.getValue().getPowerShift().getActive() + (load.getTargetP() - load.getInitialTargetP()) * 0.0;
                        changedLoads.computeIfAbsent(load.getBus(), b -> new IdentityHashMap<>()).put(load, load.getTargetP() - p0);
                    }
                    for (LfGenerator generator : lfContingency.get().getLostGenerators()) {
                        Integer g = generatorIndex.get(generator);
                        if (g == null) {
                            return false;
                        }
                        gens.add(g);
                    }
                    for (LfHvdc hvdc : candidate.connectivity().getHvdcsWithoutPower()) {
                        for (LfGenerator converterStation : List.<LfGenerator>of(hvdc.getConverterStation1(), hvdc.getConverterStation2())) {
                            Integer g = generatorIndex.get(converterStation);
                            if (g == null) {
                                return false;
                            }
                            stations.add(g);
                        }
                    }
                }
            }
            for (LfBus bus : candidate.connectivity().getDisabledBuses()) {
                excludedBus.add(bus.getNum());
            }
            for (Map.Entry<LfBus, Map<LfLoad, Double>> e : changedLoads.entrySet()) {
                LfBus bus = e.getKey();
                double total = 0.0;
                for (LfLoad load : bus.getLoads()) {
                    double targetP = e.getValue().getOrDefault(load, load.getTargetP());
                    total += targetP * load.getLoadModel().flatMap(lm -> lm.getExpTermP(0).map(LfLoadModel.ExpTerm::c)).orElse(1d);
                }
                loadBus.add(bus.getNum());
                loadP.add(total + bus.getFictitiousInjectionTargetP());
            }
            lostGenerator.addAll(gens);
            station.addAll(stations);
            for (LfBus bus : candidate.connectivity().getDisabledBuses()) {
                int column = pColumnOrMinusOne(equationSystem, bus);
                if (column >= 0) {
                    zeroRow.add(column);
                }
            }
            PropagatedContingency c = candidate.connectivity().getPropagatedContingency();
            List<LfBranch> disabledBranches = new ArrayList<>();
            c.getBranchIdsToOpen().keySet().forEach(id -> disabledBranches.add(lfNetwork.getBranchById(id)));
            disabledBranches.addAll(candidate.connectivity().getPartialDisabledBranches());
            for (LfBranch branch : disabledBranches) {
                equationSystem.getEquation(branch.getNum(), DcEquationType.BRANCH_TARGET_ALPHA1)
                        .filter(Equation::isActive)
                        .ifPresent(eq -> zeroRow.add(eq.getColumn()));
            }
            count++;
            excludedPtr.add(excludedBus.size());
            loadPtr.add(loadBus.size());
            lostGeneratorPtr.add(lostGenerator.size());
            stationPtr.add(station.size());
            zeroRowPtr.add(zeroRow.size());
            return true;
        }

        GpuDcSolver.ChunkTargetChanges build() {
            return new GpuDcSolver.ChunkTargetChanges(count, ints(excludedPtr), ints(excludedBus), ints(loadPtr), ints(loadBus),
                    loadP.stream().mapToDouble(Double::doubleValue).toArray(), ints(lostGeneratorPtr), ints(lostGenerator),
                    ints(stationPtr), ints(station), ints(zeroRowPtr), ints(zeroRow));
        }

        private static int[] ints(List<Integer> list) {
            return list.stream().mapToInt(Integer::intValue).toArray();
        }
    }

    private static int pColumnOrMinusOne(EquationSystem<DcVariableType, DcEquationType> equationSystem, LfBus bus) {
        return equationSystem.getEquation(bus.getNum(), DcEquationType.BUS_TARGET_P)
                .filter(Equation::isActive)
                .map(Equation::getColumn)
                .orElse(-1);
    }

    private static int pColumn(EquationSystem<DcVariableType, DcEquationType> equationSystem, LfBus bus) {
        return equationSystem.getEquation(bus.getNum(), DcEquationType.BUS_TARGET_P)
                .map(Equation::getColumn)
                .orElseThrow(() -> new IllegalStateException("no P equation for bus " + bus.getId()));
    }
}
