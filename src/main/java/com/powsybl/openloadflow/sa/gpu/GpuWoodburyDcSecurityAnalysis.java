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
import com.powsybl.math.matrix.SparseMatrix;
import com.powsybl.openloadflow.dc.DcLoadFlowContext;
import com.powsybl.openloadflow.dc.DcLoadFlowParameters;
import com.powsybl.openloadflow.dc.equations.AbstractClosedBranchDcFlowEquationTerm;
import com.powsybl.openloadflow.dc.equations.DcEquationSystemCreationParameters;
import com.powsybl.openloadflow.dc.equations.DcEquationType;
import com.powsybl.openloadflow.dc.equations.DcVariableType;
import com.powsybl.openloadflow.dc.fastdc.ComputedContingencyElement;
import com.powsybl.openloadflow.dc.fastdc.ConnectivityBreakAnalysis;
import com.powsybl.openloadflow.dc.fastdc.WoodburyEngine;
import com.powsybl.openloadflow.dc.gpu.GpuDcSolver;
import com.powsybl.openloadflow.equations.Equation;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.graph.GraphConnectivityFactory;
import com.powsybl.openloadflow.network.*;
import com.powsybl.openloadflow.network.impl.PropagatedContingency;
import com.powsybl.openloadflow.sa.*;
import com.powsybl.openloadflow.util.PerUnit;
import com.powsybl.openloadflow.util.Reports;
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
import java.util.concurrent.atomic.AtomicInteger;
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

    /** Contingencies solved on the device in the last run (-1 = the whole run went to the stock path). */
    public static final AtomicInteger LAST_GPU_PRECOMPUTED_COUNT = new AtomicInteger(-1);

    /** Contingencies per device chunk: bounds the state download ({@code chunk x n} doubles). */
    static final int DEVICE_CHUNK = Integer.getInteger("olf.gpu.dc.chunk", 256);

    /** Candidate records per contingency the device buffer holds; overflow = host screen. */
    static final int CAPACITY = 1024;

    private record DeviceCandidate(ConnectivityBreakAnalysis.ConnectivityAnalysisResult connectivity,
                                   List<ComputedContingencyElement> elements) {
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
        if (!GpuDcSolver.isAvailable() || !operatorStrategies.isEmpty() || !actions.isEmpty()) {
            LOGGER.info("GPU Woodbury: whole run on the stock path ({})",
                    GpuDcSolver.isAvailable() ? "operator strategies / actions" : "no device");
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
            // pre-contingency state: OLF's own solve (the context's KLU), exactly the stock result
            double[] preContingencyStates = WoodburyEngine.runDcLoadFlowWithModifiedTargetVector(context, new DisabledNetwork(), reportNode);
            EquationSystem<DcVariableType, DcEquationType> equationSystem = context.getEquationSystem();
            equationSystem.getStateVector().set(preContingencyStates);
            updateNetwork(lfNetwork, equationSystem, preContingencyStates);
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

            // the device: factorization, branch model, S — before anything else needs S
            SparseMatrix matrix = (SparseMatrix) context.getJacobianMatrix().getMatrix();
            GpuDcSolver solver = GpuDcSolver.create(matrix);
            long tFactor = System.nanoTime();
            int elementCount;
            try {
                prepareBranches(solver, lfNetwork, equationSystem, creationParameters);
                elementCount = prepareElements(solver, equationSystem, elementByBranch.values());
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
            if (System.getenv("OLF_GPU_PHASE_TIME") != null) {
                System.err.printf("DC_WOODBURY_PRE preContingency=%.0f ms factor=%.0f ms S=%.0f ms connectivity=%.0f ms (%d potentially breaking -> graph)%n",
                        (tPre1 - tPre0) / 1e6, (tFactor - tPre1) / 1e6, (tS - tFactor) / 1e6, (tConn - tS) / 1e6, lastPotentiallyBreaking);
            }

            // qualification: a contingency that keeps the base target vector, or one whose modified target
            // (islanded buses, lost PST) the host can build without touching the network
            boolean distributedSlack = dcParameters.isDistributedSlack();
            boolean multiSlack = lfNetwork.getSynchronousNetworks().getFirst().getSlackBuses().size() > 1;
            boolean modifiedTargetsSupported = !multiSlack && (!distributedSlack || DcSlackDistributionModel.supports(dcParameters.getBalanceType()));
            List<DeviceCandidate> device = new ArrayList<>();
            List<PropagatedContingency> cpu = new ArrayList<>();
            Map<String, Integer> cpuReasons = new TreeMap<>();
            for (var list : List.of(connectivity.nonBreakingConnectivityAnalysisResults(), connectivity.connectivityBreakingAnalysisResults())) {
                for (ConnectivityBreakAnalysis.ConnectivityAnalysisResult r : list) {
                    String reason = deviceRejectReason(r, elementByBranch, modifiedTargetsSupported);
                    if (reason == null) {
                        List<ComputedContingencyElement> elements = r.getPropagatedContingency().getBranchIdsToOpen().keySet().stream()
                                .filter(id -> !r.getElementsToReconnect().contains(id))
                                .map(elementByBranch::get).toList();
                        device.add(new DeviceCandidate(r, elements));
                    } else {
                        cpu.add(r.getPropagatedContingency());
                        cpuReasons.merge(reason, 1, Integer::sum);
                    }
                }
            }
            LOGGER.info("GPU Woodbury: {} contingencies on the device, {} on the stock path {}", device.size(), cpu.size(), cpuReasons);
            if (System.getenv("OLF_GPU_PHASE_TIME") != null) {
                System.err.printf("DC_WOODBURY_ROUTING device=%d stock=%d %s%n", device.size(), cpu.size(), cpuReasons);
            }

            Map<String, PostContingencyResult> resultById = new HashMap<>();
            int onDevice = 0;

            try (solver) {
                if (!device.isEmpty()) {
                    onDevice = runOnDevice(solver, elementCount, lfNetwork, context, device, preContingencyStates, preContingencyNetworkResult,
                            preContingencyLimitViolationManager, monitorIndexes, createResultExtension, securityAnalysisParameters,
                            limitReductions, dcPowerFactor, resultById);
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
            }
            LAST_GPU_PRECOMPUTED_COUNT.set(onDevice);

            // the stock order: the non-breaking contingencies (as the connectivity analysis listed them),
            // then the connectivity-breaking ones
            List<PostContingencyResult> postContingencyResults = new ArrayList<>();
            for (var list : List.of(connectivity.nonBreakingConnectivityAnalysisResults(), connectivity.connectivityBreakingAnalysisResults())) {
                for (ConnectivityBreakAnalysis.ConnectivityAnalysisResult r : list) {
                    PostContingencyResult result = resultById.get(r.getPropagatedContingency().getContingency().getId());
                    if (result != null) {
                        postContingencyResults.add(result);
                    }
                }
            }
            return new SecurityAnalysisResult(
                    new PreContingencyResult(LoadFlowResult.ComponentResult.Status.CONVERGED,
                            new LimitViolationsResult(preContingencyLimitViolationManager.getLimitViolations()),
                            new NetworkResult(preContingencyNetworkResult.getBranchResults(), preContingencyNetworkResult.getBusResults(),
                                    preContingencyNetworkResult.getThreeWindingsTransformerResults()),
                            Double.NaN),
                    postContingencyResults, List.of());
        }
    }

    /** Why a contingency cannot go to the device, or null when it can. */
    static String deviceRejectReason(ConnectivityBreakAnalysis.ConnectivityAnalysisResult r,
                                     Map<String, ComputedContingencyElement> elementByBranch, boolean modifiedTargetsSupported) {
        PropagatedContingency c = r.getPropagatedContingency();
        if (!c.getGeneratorIdsToLose().isEmpty() || !c.getLoadIdsToLose().isEmpty() || !c.getHvdcIdsToOpen().isEmpty()) {
            return "lost power (generator/load/HVDC)";
        }
        if (!r.getHvdcsWithoutPower().isEmpty()) {
            return "HVDC without power";
        }
        if (!c.getShuntIdsToShift().isEmpty()) {
            return "shunt shift";
        }
        if (!c.getBusIdsToLose().isEmpty()) {
            return "bus loss";
        }
        int k = c.getBranchIdsToOpen().size();
        if (k == 0) {
            return "no branch element";
        }
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
        boolean needsModifiedTarget = !r.getDisabledBuses().isEmpty() || lostPst;
        if (needsModifiedTarget && !modifiedTargetsSupported) {
            return "modified target (load-based balance type or multi-slack)";
        }
        return null;
    }

    /** The device pass over the qualified contingencies; returns the number solved on the device. */
    private int runOnDevice(GpuDcSolver solver, int elementCount, LfNetwork lfNetwork, DcLoadFlowContext context, List<DeviceCandidate> device,
                            double[] preContingencyStates, PreContingencyNetworkResult preContingencyNetworkResult,
                            LimitViolationManager preContingencyLimitViolationManager, AbstractNetworkResult.StateMonitorIndexes monitorIndexes,
                            boolean createResultExtension, SecurityAnalysisParameters securityAnalysisParameters,
                            List<LimitReduction> limitReductions, double dcPowerFactor, Map<String, PostContingencyResult> resultById) {
        EquationSystem<DcVariableType, DcEquationType> equationSystem = context.getEquationSystem();
        int n = equationSystem.getIndex().getColumnCount();
        int branchCount = lfNetwork.getBranches().size();
        long t0 = System.nanoTime();
        {
            solver.setBase(preContingencyStates);
            long tFactor = t0;
            long tBranches = t0;
            long tS = t0;

            // --- limit table
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
            // per-bus net injection at base, in bus order: getActivePowerMismatch over the remaining buses is
            // -sum over them, and summing in bus order while skipping the islanded ones is the same sum in the
            // same order as the stock's LinkedHashSet walk (bit-identical, no per-contingency getter walk)
            double[] busNetInjection = new double[lfNetwork.getBuses().size()];
            for (LfBus bus : lfNetwork.getBuses()) {
                busNetInjection[bus.getNum()] = bus.isDisabled() ? Double.NaN : bus.getGenerationTargetP() - bus.getLoadTargetP();
            }
            boolean distributedSlack = context.getParameters().isDistributedSlack();
            DcSlackDistributionModel distribution = distributedSlack
                    ? DcSlackDistributionModel.build(lfNetwork, context.getParameters().getBalanceType(),
                            context.getParameters().getNetworkParameters().isUseActiveLimits())
                    : null;
            boolean needBusResults = !monitorIndex.getAllStateMonitor().getVoltageLevelIds().isEmpty()
                    || monitorIndex.getSpecificStateMonitors().values().stream().anyMatch(m -> !m.getVoltageLevelIds().isEmpty());
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
            for (int from = 0; from < device.size(); from += DEVICE_CHUNK) {
                int to = Math.min(from + DEVICE_CHUNK, device.size());
                int count = to - from;
                int[] ctgPtr = new int[count + 1];
                int[] ctgBase = new int[count];
                List<Integer> elemList = new ArrayList<>();
                int nBases = 0;
                long tb0 = System.nanoTime();
                for (int i = 0; i < count; i++) {
                    DeviceCandidate candidate = device.get(from + i);
                    for (ComputedContingencyElement element : candidate.elements()) {
                        elemList.add(element.getComputedElementIndex());
                    }
                    ctgPtr[i + 1] = elemList.size();
                    ctgBase[i] = -1;
                    if (needsModifiedTarget(candidate)) {
                        buildModifiedTarget(candidate, lfNetwork, equationSystem, baseTargets, busNetInjection, distribution, targets, nBases * n);
                        ctgBase[i] = nBases++;
                        modifiedTargets++;
                    }
                }
                tColumns += System.nanoTime() - tb0;
                int[] ctgElem = elemList.stream().mapToInt(Integer::intValue).toArray();
                long tc0 = System.nanoTime();
                solver.setChunkTargets(targets, nBases);
                int[] records = solver.woodburyChunk(ctgPtr, ctgElem, ctgBase, GpuViolationScreen.BAND, CAPACITY, states);
                tDevice += System.nanoTime() - tc0;

                long th0 = System.nanoTime();
                for (int i = 0; i < count; i++) {
                    DeviceCandidate candidate = device.get(from + i);
                    PropagatedContingency propagated = candidate.connectivity().getPropagatedContingency();
                    Optional<LfContingency> lfContingencyOpt = candidate.connectivity().toLfContingency();
                    if (lfContingencyOpt.isEmpty()) {
                        continue;                            // no impact: the stock path reports nothing either
                    }
                    LfContingency lfContingency = lfContingencyOpt.get();
                    Contingency contingency = propagated.getContingency();
                    ReportNode postContSimReportNode = Reports.createPostContingencySimulation(lfNetwork.getReportNode(), contingency.getId());
                    lfNetwork.setReportNode(postContSimReportNode);
                    logPostContingencyStart(lfNetwork, lfContingency);

                    // the post-contingency state into the equation system: the vectorized DC network recomputes the flows
                    double[] state = Arrays.copyOfRange(states, i * n, (i + 1) * n);
                    equationSystem.getStateVector().set(state);
                    if (needAngles) {
                        updateNetwork(lfNetwork, equationSystem, state);
                    }
                    Map<LfBranch, DisabledBranchStatus> disabledBranches = lfContingency.getDisabledNetwork().getBranchesStatus();
                    Predicate<LfBranch> isBranchDisabled = disabledBranches::containsKey;

                    // violations: the screen's candidates confirmed on OLF's own values
                    var postContingencyLimitViolationManager = new LimitViolationManager(preContingencyLimitViolationManager, limitReductions, violationsParameters);
                    int emitted = records[i];
                    if (emitted > CAPACITY) {
                        hostScreened++;
                        postContingencyLimitViolationManager.detectViolations(lfNetwork, isBranchDisabled);
                    } else {
                        int[] candidates = Arrays.copyOfRange(records, count + i * CAPACITY, count + i * CAPACITY + emitted);
                        Arrays.sort(candidates);             // table order: a group's limits contiguous, most severe first
                        scanner.detectFromCandidates(lfNetwork, postContingencyLimitViolationManager, candidates, isBranchDisabled);
                    }

                    // network result: the stock class over the loaded state. The stock path's apply() flags the
                    // islanded buses disabled and the bus results skip them; flag them for the duration of the
                    // result build when bus results are wanted at all.
                    Set<LfBus> islanded = candidate.connectivity().getDisabledBuses();
                    boolean flagBuses = needBusResults && !islanded.isEmpty();
                    if (flagBuses) {
                        islanded.forEach(bus -> bus.setDisabled(true));
                    }
                    var postContingencyNetworkResult = new PostContingencyNetworkResult(lfNetwork, monitorIndexes, createResultExtension,
                            preContingencyNetworkResult, contingency, LoadFlowModel.DC, dcPowerFactor,
                            securityAnalysisParameters.getModifiedMonitoredElementsParameters());
                    postContingencyNetworkResult.update(isBranchDisabled);
                    if (flagBuses) {
                        islanded.forEach(bus -> bus.setDisabled(false));
                    }

                    var connectivityResult = new ConnectivityResult(
                            lfContingency.getCreatedSynchronousComponentsCount(), 0,
                            lfContingency.getDisconnectedLoadActivePower() * PerUnit.SB,
                            lfContingency.getDisconnectedGenerationActivePower() * PerUnit.SB,
                            lfContingency.getDisconnectedElementIds());
                    resultById.put(contingency.getId(), new PostContingencyResult(contingency, PostContingencyComputationStatus.CONVERGED,
                            new LimitViolationsResult(postContingencyLimitViolationManager.getLimitViolations()),
                            new NetworkResult(postContingencyNetworkResult.getBranchResults(), postContingencyNetworkResult.getBusResults(),
                                    postContingencyNetworkResult.getThreeWindingsTransformerResults()),
                            connectivityResult, Double.NaN));
                }
                tHost += System.nanoTime() - th0;
            }
            // leave the base state loaded, as the stock path's restore would
            equationSystem.getStateVector().set(preContingencyStates);
            updateNetwork(lfNetwork, equationSystem, preContingencyStates);
            if (System.getenv("OLF_GPU_PHASE_TIME") != null) {
                System.err.printf("DC_WOODBURY_PHASE n=%d elements=%d device=%d modifiedTargets=%d setup=%.0f ms [factor=%.0f branches=%.0f S=%.0f table=%.0f model=%.0f] columns=%.0f ms device=%.0f ms host=%.0f ms hostScreened=%d%n",
                        n, elementCount, device.size(), modifiedTargets, (tSetup - t0) / 1e6, (tFactor - t0) / 1e6, (tBranches - tFactor) / 1e6,
                        (tS - tBranches) / 1e6, (tTable - tS) / 1e6, (tSetup - tTable) / 1e6, tColumns / 1e6, tDevice / 1e6, tHost / 1e6, hostScreened);
            }
            return device.size();
        }
    }

    /** The branch model, in LfNetwork order (== GpuLimitTable order == branch num). */
    private static void prepareBranches(GpuDcSolver solver, LfNetwork lfNetwork, EquationSystem<DcVariableType, DcEquationType> equationSystem,
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
    }

    /** The elements' +1/-1 rows (ComputedElement.fillRhs) at OLF's column indices; -1 when the device refuses S. */
    private static int prepareElements(GpuDcSolver solver, EquationSystem<DcVariableType, DcEquationType> equationSystem,
                                       Collection<ComputedContingencyElement> elements) {
        int elementCount = elements.stream().mapToInt(ComputedContingencyElement::getComputedElementIndex).max().orElse(-1) + 1;
        int[] rowPlus = new int[elementCount];
        int[] rowMinus = new int[elementCount];
        int[] elemBranch = new int[elementCount];
        Arrays.fill(rowPlus, -1);
        Arrays.fill(rowMinus, -1);
        Arrays.fill(elemBranch, -1);
        for (ComputedContingencyElement element : elements) {
            LfBranch branch = element.getLfBranch();
            int e = element.getComputedElementIndex();
            elemBranch[e] = branch.getNum();
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
        return solver.setElements(rowPlus, rowMinus, elemBranch) ? elementCount : -1;
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
            ConnectivityBreakAnalysis.computeConnectivityAnalysisResult(lfNetwork, contingency, elementByBranch, null, Collections.emptyMap())
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
