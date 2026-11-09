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
import com.powsybl.contingency.strategy.OperatorStrategy;
import com.powsybl.iidm.network.Network;
import com.powsybl.openloadflow.ac.AcLoadFlowContext;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataCache;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcElementMapper;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.solver.AbstractNewtonParameters;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.ac.solver.gpu.batch.BatchConfig;
import com.powsybl.openloadflow.ac.solver.gpu.batch.BatchScenarioResult;
import com.powsybl.openloadflow.ac.solver.gpu.batch.BatchSolveRequest;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchedSecurityAnalysisSolver;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.BatchContingency;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.GpuBatchScopeCensus;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.LoadLoss;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.ShuntShift;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuAcSolverStatus;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfShunt;
import com.powsybl.openloadflow.network.gpu.ZeroImpedanceTreeReplay;
import com.powsybl.openloadflow.network.impl.PropagatedContingency;
import com.powsybl.openloadflow.util.gpu.GpuTrace;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Precomputes the batched N-1 (and (contingency, strategy) pair) states on the GPU device once the base
 * load flow has converged: maps each contingency to its batchable device shape (via
 * {@link GpuContingencyMapper} / {@link GpuStrategyComposer}), runs them all through
 * {@link GpuBatchedSecurityAnalysisSolver#solveBatchN1} under the global device lock, and returns the
 * converged states as a {@link PrecomputedBatch}. Anything unbatchable is simply absent, so the SA's
 * per-contingency overrides fall back to the sequential CPU path for it.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuBatchPrecomputer {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuBatchPrecomputer.class);

    /** Serializes device access across partition-worker threads (the batch precompute queues here). */
    private static final Object GPU_DEVICE_LOCK = new Object();

    private GpuBatchPrecomputer() {
    }

    /** One batched solve's settings: the run's, or those of the contingencies sharing their own load flow parameters. */
    record SolveSettings(boolean distributedSlack, boolean reactiveLimits, double reactiveLimitTol, double slackDistTol,
                         double convEpsPerEq, BatchConfig config) {

        static SolveSettings of(GpuSaRunScope.Scope scope) {
            return new SolveSettings(scope.distributedSlack(), scope.reactiveLimits(), scope.reactiveLimitTol(),
                    scope.slackDistTol(), scope.convEpsPerEq(), scope.batchConfig());
        }
    }

    /**
     * The contingencies one batched solve runs with one set of settings: the run's ({@code contingencyIds} null: every
     * contingency in no other group), or those carrying the same load flow parameters (ContingencyLoadFlowParameters).
     * The strategies of those in {@code withoutStrategies} run with the run's outer loops (scope CONTINGENCY_ONLY), and
     * none is composed here.
     */
    record SolveGroup(SolveSettings settings, java.util.Set<String> contingencyIds, java.util.Set<String> withoutStrategies) {

        static SolveGroup run(SolveSettings settings) {
            return new SolveGroup(settings, null, java.util.Set.of());
        }
    }

    /**
     * Solve the qualified single-branch contingencies in one GPU batch, from the converged
     * base state. Failures (no GPU, an out-of-scope network the extractor rejects, a device
     * error) leave the cache empty so every contingency falls back to the CPU path.
     */
    static PrecomputedBatch precompute(Network network, LfNetwork net, AcLoadFlowContext context,
                                       List<PropagatedContingency> contingencies, List<OperatorStrategy> strategies,
                                       List<Action> actions, boolean distributedSlack, boolean reactiveLimits,
                                       double reactiveLimitTol, double slackDistTol, double convEpsPerEq,
                                       BatchConfig config, ReportNode reportNode) {
        return precompute(network, net, context, contingencies, strategies, actions,
                List.of(SolveGroup.run(new SolveSettings(distributedSlack, reactiveLimits, reactiveLimitTol, slackDistTol,
                        convEpsPerEq, config))), java.util.Set.of(), null, false, reportNode);
    }

    /**
     * As above; with {@code reactiveLimitDecisions} every converged contingency also keeps the reactive-limit decisions
     * its solve ended with ({@link PrecomputedBatch#reactiveLimitDecisions()}), for a caller that brings the contingency's
     * state into OLF's network itself and needs its equation system to be the one the device converged on.
     */
    @SuppressWarnings("java:S107")
    static PrecomputedBatch precompute(Network network, LfNetwork net, AcLoadFlowContext context,
                                       List<PropagatedContingency> contingencies, List<OperatorStrategy> strategies,
                                       List<Action> actions, boolean distributedSlack, boolean reactiveLimits,
                                       double reactiveLimitTol, double slackDistTol, double convEpsPerEq,
                                       BatchConfig config, boolean reactiveLimitDecisions, ReportNode reportNode) {
        return precompute(network, net, context, contingencies, strategies, actions,
                List.of(SolveGroup.run(new SolveSettings(distributedSlack, reactiveLimits, reactiveLimitTol, slackDistTol,
                        convEpsPerEq, config))), java.util.Set.of(), null, reactiveLimitDecisions, reportNode);
    }

    /**
     * What the precompute needs to screen each handed-back scenario on the device and keep only what the host answers it
     * with (P4 of docs/device-final-results-plan.md, {@link GpuStateHandback}): the run's limit reductions (the screen's
     * table), which scenarios the host may answer from their state rows alone ({@code sparse}), whether the comparison
     * also reads every full state, and the realistic-voltage band the screen reports buses outside of.
     */
    record HandbackSettings(List<com.powsybl.security.limitreduction.LimitReduction> limitReductions,
                            java.util.function.Predicate<String> sparse, boolean compare, double minRealisticV,
                            double maxRealisticV) {
    }

    /**
     * The same with the contingencies split in {@code groups}, one solve each, the run's group among them; the
     * contingencies in {@code leftToCpu} are not batched (their own outer loops are not run on the device).
     */
    static PrecomputedBatch precompute(Network network, LfNetwork net, AcLoadFlowContext context,
                                       List<PropagatedContingency> contingencies, List<OperatorStrategy> strategies,
                                       List<Action> actions, List<SolveGroup> groups, java.util.Set<String> leftToCpu,
                                       HandbackSettings handbackSettings, ReportNode reportNode) {
        return precompute(network, net, context, contingencies, strategies, actions, groups, leftToCpu, handbackSettings,
                false, reportNode);
    }

    /** The same, every group's converged contingencies keeping their reactive-limit decisions with
     *  {@code reactiveLimitDecisions} (see the flat overload above). */
    @SuppressWarnings("java:S107")
    static PrecomputedBatch precompute(Network network, LfNetwork net, AcLoadFlowContext context,
                                       List<PropagatedContingency> contingencies, List<OperatorStrategy> strategies,
                                       List<Action> actions, List<SolveGroup> groups, java.util.Set<String> leftToCpu,
                                       HandbackSettings handbackSettings, boolean reactiveLimitDecisions,
                                       ReportNode reportNode) {
        if (GpuTrace.BATCH_TRACE.isOn()) {   // the network the batch starts from
            for (var b : net.getBuses()) {
                for (var g : b.getGenerators()) {
                    System.err.println("BASE_GEN " + g.getId() + " targetP=" + g.getTargetP() + " targetQ=" + g.getTargetQ()
                            + " vc=" + b.isGeneratorVoltageControlEnabled());
                }
            }
        }
        synchronized (GPU_DEVICE_LOCK) {
            return precomputeLocked(network, net, context, contingencies, strategies, actions, groups, leftToCpu,
                    handbackSettings, reactiveLimitDecisions, reportNode);
        }
    }

    private static PrecomputedBatch precomputeLocked(Network network, LfNetwork net, AcLoadFlowContext context,
                                                     List<PropagatedContingency> contingencies, List<OperatorStrategy> strategies,
                                                     List<Action> actions, List<SolveGroup> groups,
                                                     java.util.Set<String> leftToCpu, HandbackSettings handbackSettings,
                                                     boolean reactiveLimitDecisions,
                                                     ReportNode reportNode) {
        SolveGroup runGroup = groups.stream().filter(g -> g.contingencyIds() == null).findFirst().orElseThrow();
        if (!GpuAcNewtonSolver.isAvailable() || net == null || context == null || contingencies.isEmpty()) {
            if (GpuTrace.BATCH_TRACE.isOn()) {
                System.err.println("PRECOMP_BAIL avail=" + GpuAcNewtonSolver.isAvailable() + " net=" + (net != null)
                        + " ctx=" + (context != null) + " nCtg=" + contingencies.size());
            }
            return PrecomputedBatch.empty();
        }
        if (GpuTrace.BATCH_TRACE.isOn()) {
            System.err.println("PRECOMP_ENTER nCtg=" + contingencies.size());
        }
        try {
            // REUSE the engine's equation system + target vector (do NOT create a second system — that would
            // re-point the network's evaluables off the engine's system, breaking fall-through CPU solves).
            // After the converged base load flow the engine's state vector holds the base operating point.
            EquationSystem<AcVariableType, AcEquationType> es = context.getEquationSystem();
            double[] baseState = es.getStateVector().get().clone();
            var target = context.getTargetVector();

            // one batch per group, the contingencies taken in OLF's order
            List<List<BatchContingency>> batches = new ArrayList<>();
            Map<String, Integer> groupIndex = new HashMap<>();
            for (int g = 0; g < groups.size(); g++) {
                batches.add(new ArrayList<>());
                if (groups.get(g).contingencyIds() != null) {
                    for (String id : groups.get(g).contingencyIds()) {
                        groupIndex.put(id, g);
                    }
                }
            }
            int runGroupIndex = groups.indexOf(runGroup);
            // OLF's zero-impedance spanning trees, replayed contingency by contingency (ZeroImpedanceTreeReplay):
            // the device extracted its structure on the base tree, and every coupler OLF has out of its
            // tree when it reaches a contingency must be held out of it there, ALL of them, not one per network.
            ZeroImpedanceTreeReplay ziTree = new ZeroImpedanceTreeReplay(net);
            for (PropagatedContingency propagatedContingency : contingencies) {
                String contingencyId = propagatedContingency.getContingency().getId();
                if (leftToCpu.contains(contingencyId)) {
                    GpuBatchScopeCensus.record("PLAN:CONTINGENCY_PARAMETERS");
                    GpuBatchScopeCensus.planRefused(contingencyId, "PLAN:CONTINGENCY_PARAMETERS");
                    continue;
                }
                int g = groupIndex.getOrDefault(contingencyId, runGroupIndex);
                SolveGroup group = groups.get(g);
                addBatchScenarios(net, network, propagatedContingency, context, ziTree, batches.get(g),
                        group.withoutStrategies().contains(contingencyId) ? List.of() : strategies,
                        group.settings().config(), actions);
            }
            int batchSize = batches.stream().mapToInt(List::size).sum();
            if (GpuTrace.BATCH_TRACE.isOn()) {
                System.err.println("PRECOMP_BATCH size=" + batchSize);
            }
            if (batchSize == 0) {
                return PrecomputedBatch.empty();
            }

            // The packed element data and the branch->element map, for the violation screen (see
            // GpuViolationScreen). Extracted HERE rather than lifted out of the solver: without a base-open branch it
            // is the cache hit the solver is about to take. A branch a scenario closes (a strategy's reclosing, an
            // automation system) is packed open as the solver packs it, closed in the scenarios that close it
            // (GpuScreenScenario.IN), so the screen evaluates it as the device solved it.
            java.util.Set<String> baseOpenIds = new java.util.HashSet<>();
            for (List<BatchContingency> batch : batches) {
                for (BatchContingency c : batch) {
                    for (LfBranch reclosed : c.reclosedBranches()) {
                        if (!reclosed.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC)) {
                            baseOpenIds.add(reclosed.getId());
                        }
                    }
                }
            }
            if (groups.stream().anyMatch(g -> g.settings().config().automationSystem())) {
                baseOpenIds.addAll(com.powsybl.openloadflow.ac.outerloop.gpu.GpuAutomationSystemOuterLoop.baseOpenOperatedBranchIds(net));
            }
            GpuAcData screenData = baseOpenIds.isEmpty()
                    ? GpuAcDataCache.extract(net, es, runGroup.settings().config().distributedSlackParameters())
                    : com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor.extract(net, es, baseOpenIds,
                            java.util.Set.of(), runGroup.settings().config().distributedSlackParameters());
            GpuAcElementMapper screenMapper = new GpuAcElementMapper(net, baseOpenIds);
            // what each scenario is solved as: the screen evaluates its branches in that topology, with those parameters
            Map<String, BatchContingency> scenarios = new HashMap<>();
            for (List<BatchContingency> batch : batches) {
                for (BatchContingency c : batch) {
                    scenarios.put(c.contingencyId(), c);
                }
            }
            Map<Variable<AcVariableType>, Integer> baseVariableRows = new HashMap<>();
            for (var v : es.getIndex().getSortedVariablesToFind()) {
                baseVariableRows.put(v, v.getRow());
            }
            // The screen on the device as each scenario is handed back, keeping only what the host answers it with.
            GpuStateHandback handback = null;
            GpuLimitTable limitTable = null;
            GpuViolationScreen violationScreen = null;
            int fullStates = batchSize;
            if (handbackSettings != null && com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable()) {
                limitTable = GpuLimitTable.build(net, com.powsybl.openloadflow.sa.LimitReductionManager.create(
                        handbackSettings.limitReductions()));
                violationScreen = GpuViolationScreen.build(net, limitTable, screenData, screenMapper, es, baseVariableRows);
                GpuStateView view = new GpuStateView(net, baseVariableRows);
                java.util.function.Predicate<String> sparse = view.usable() ? handbackSettings.sparse() : id -> false;
                handback = new GpuStateHandback(violationScreen, limitTable, view, net, scenarios, sparse,
                        handbackSettings.compare(), true, handbackSettings.minRealisticV(), handbackSettings.maxRealisticV());
                if (!handbackSettings.compare()) {
                    fullStates = (int) scenarios.keySet().stream().filter(id -> !sparse.test(id)).count();
                }
            }

            // Every state the host needs whole is held until the analysis ends (the results are injected later in OLF's
            // loop). Past GpuStateStore's threshold they go to a temporary file as each chunk completes instead of the
            // heap: network.xiidm's all-kinds + busbar list is 37k states of 280 KB, 10.4 GB, more than this machine's heap
            // can hold beside the rest of the run. With the handback, only the scenarios it does not answer from their
            // rows come back whole.
            Map<String, double[]> stateSink = runGroup.settings().config().stateSink();
            if (stateSink == null && com.powsybl.openloadflow.sa.gpu.GpuStateStore.wanted(fullStates, baseState.length)) {
                stateSink = com.powsybl.openloadflow.sa.gpu.GpuStateStore.create(baseState.length);
                LOGGER.info("Batched N-1: {} states of {} entries go to a temporary file (over {} bytes)", batchSize,
                        baseState.length, com.powsybl.openloadflow.sa.gpu.GpuStateStore.thresholdBytes());
            }
            // one solve per group, with its own outer loops, into the same results
            List<BatchScenarioResult> results = new ArrayList<>();
            for (int g = 0; g < groups.size(); g++) {
                List<BatchContingency> batch = batches.get(g);
                if (batch.isEmpty()) {
                    continue;
                }
                SolveSettings settings = groups.get(g).settings();
                BatchConfig config = stateSink != null ? settings.config().withStateSink(stateSink) : settings.config();
                if (handback != null) {
                    config = config.withStateHandback(handback);
                }
                List<BatchScenarioResult> groupResults = GpuBatchedSecurityAnalysisSolver.solveBatchN1(
                        BatchSolveRequest.of(net, baseState, batch, es, target, deviceSolverParameters(context),
                                reportNode)
                                .withOuterLoops(settings.distributedSlack(), settings.reactiveLimits())
                                .withOuterLoopTolerances(settings.reactiveLimitTol(), settings.slackDistTol())
                                .withConvEpsPerEq(settings.convEpsPerEq())
                                .withConfig(config)
                                .withReactiveLimitDecisions(reactiveLimitDecisions));
                results.addAll(retryRejectedOnDevice(groupResults, batch, net, baseState, es, target, context,
                        settings.distributedSlack(), settings.reactiveLimits(), settings.reactiveLimitTol(),
                        settings.slackDistTol(), settings.convEpsPerEq(), reportNode, config, reactiveLimitDecisions));
            }
            // the converged states: the sink when the run has one (every chunk put its states there as it
            // completed), a heap map otherwise
            Map<String, double[]> states = stateSink != null ? stateSink : new HashMap<>();
            Map<String, int[]> islandedBuses = new HashMap<>();
            java.util.Set<String> exhausted = new java.util.HashSet<>();
            Map<String, Double> preDistributed = new HashMap<>();
            Map<String, Double> distributed = new HashMap<>();
            Map<String, int[]> automationSystemSwitched = new HashMap<>();
            Map<String, double[]> deviceTapMoves = new HashMap<>();
            Map<String, BatchScenarioResult.GeneratorTargets> generatorTargets = new HashMap<>();
            Map<String, BatchScenarioResult.ReactiveLimitDecisions> decisions = new HashMap<>();
            for (BatchScenarioResult r : results) {
                if (r.generatorTargets() != null && r.contingencyId().indexOf(BatchContingency.STRATEGY_KEY_SEP) < 0
                        && (r.hasConverged() || r.convergenceStatus() == GpuAcSolverStatus.MAX_ITERATION_REACHED)) {
                    generatorTargets.put(r.contingencyId(), r.generatorTargets());
                }
                GpuBatchedAcSecurityAnalysis.LAST_DEVICE_FINAL_MISMATCH.put(r.contingencyId(), r.finalMismatch());
                if (GpuTrace.QUALIFY_TRACE.isOn()) {
                    System.err.println("QUALIFY_TRACE result " + r.contingencyId() + " status=" + r.convergenceStatus()
                            + " iters=" + r.iterations() + " norm=" + r.finalMismatch()
                            + (GpuBatchScopeCensus.planRefusedIds().contains(r.contingencyId())
                                    ? " (never batched: refused at plan time)" : ""));
                }
                if (r.hasConverged()) {
                    if (r.state() != null) {                 // null: already in the sink
                        states.put(r.contingencyId(), r.state());
                    }
                    recordDistributed(r, preDistributed, distributed);
                    if (r.islandedBusNums().length > 0) {
                        islandedBuses.put(r.contingencyId(), r.islandedBusNums());
                    }
                    if (r.automationSystemSwitchedBranchNums().length > 0) {
                        automationSystemSwitched.put(r.contingencyId(), r.automationSystemSwitchedBranchNums());
                    }
                    if (r.deviceTapMoves().length > 0) {
                        deviceTapMoves.put(r.contingencyId(), r.deviceTapMoves());
                    }
                    if (r.reactiveLimitDecisions() != null) {
                        decisions.put(r.contingencyId(), r.reactiveLimitDecisions());
                    }
                } else if (r.convergenceStatus() == GpuAcSolverStatus.MAX_ITERATION_REACHED
                        && r.contingencyId().indexOf(BatchContingency.STRATEGY_KEY_SEP) < 0) {
                    // The device ran OLF's full Newton budget (deviceSolverParameters carries the
                    // run's maxIterations) and the scenario did not converge: that IS the answer OLF
                    // gives - MAX_ITERATION_REACHED, no network result - so it is reported from here
                    // rather than replayed. Measured on rte6515's 500 busbar sections with the
                    // straggler exit off: the device's exhausted set is exactly the CPU's five
                    // MAX_ITERATION_REACHED sections. Strategy pairs keep the replay.
                    exhausted.add(r.contingencyId());
                    // OLF reports on a MAX_ITERATION_REACHED result what its slack distribution placed before
                    // giving up (AbstractSecurityAnalysis adds the load flow's own distributed power to the loss
                    // pre-distribution whatever the status), so the device's total is carried for these too.
                    recordDistributed(r, preDistributed, distributed);
                    GpuBatchScopeCensus.record("DEVICE:MAX_ITERATION_REACHED");
                } else {
                    GpuBatchScopeCensus.record("RUNTIME:status=" + r.convergenceStatus());
                    if (GpuTrace.REJECT_TRACE.isOn()) {
                        // pairs with the kernel's GPU_REJECT slot= reason= lines (same switch)
                        System.err.println("GPU_REJECT_CTG ctg=" + r.contingencyId() + " status=" + r.convergenceStatus());
                    }
                }
            }
            if (handback != null) {
                handback.close();                            // the device table: every scenario is screened
                if (GpuTrace.PHASE_TIME.isOn()) {
                    System.err.println(handback.summary());
                }
            }
            LOGGER.info("GPU batched SA: precomputed {} of {} branch (N-1..N-k) contingencies on the device",
                    states.size(), batchSize);
            // ...and on stderr, unconditionally: the device's share is decided HERE, before the CPU replays the
            // contingencies it did not solve. On a large scope that replay is the long tail (network.xiidm's full
            // busbar scope: about two hours on one thread), and the engagement number should not wait for it.
            System.err.println("GPU_PRECOMPUTED " + states.size() + " / " + batchSize + " batched scenarios on device"
                    + " census=" + GpuBatchScopeCensus.snapshot());

            return new PrecomputedBatch(states, islandedBuses, exhausted, baseVariableRows, es, screenData, screenMapper,
                    preDistributed, distributed, automationSystemSwitched, deviceTapMoves, generatorTargets, scenarios,
                    handback != null ? handback.candidates : Map.of(), handback != null ? handback.gathered : Map.of(),
                    limitTable, violationScreen, decisions);
        } catch (Exception e) {
            // full stack at WARN: a swallowed exception here silently un-accelerates the whole run —
            // the 2026-09-06 bridge AIOOBE hid behind the message-only log for days
            LOGGER.warn("GPU batched SA precompute failed; falling back to per-contingency CPU solves", e);
            // ...and ONE LINE on stderr unconditionally. This failure silently un-accelerates the
            // WHOLE run - every contingency reverts to a CPU solve and the only outward sign is a
            // precomputed count of -1 - and it has now hidden twice behind a log nobody saw: the
            // 2026-09-06 bridge AIOOBE, and again at 6876 busbar sections where the run reported
            // matching statuses and a zero violation diff that were vacuous because NOTHING ran on
            // the device. The test harness's logback appenders are broken (core/classic version
            // mismatch), so the WARN above does not reach anyone here.
            System.err.println("GPU_PRECOMPUTE_FAILED " + e);
            StackTraceElement[] frames = e.getStackTrace();      // the first frames always: the line that says where
            for (int i = 0; i < Math.min(6, frames.length); i++) {
                System.err.println("    at " + frames[i]);
            }
            if (GpuTrace.BATCH_TRACE.isOn()) {
                e.printStackTrace();                         // the test harness's logback appenders are broken
            }                                                // (core/classic version mismatch) — surface it here
            return PrecomputedBatch.empty();
        }
    }

    /** Translate one propagated contingency into the batch: its BatchContingency (branches, injections, shunt
     *  shifts, HVDCs, side-opens, coupler outages, spanning tree) and one entry per operator strategy the device
     *  can solve after it, or the census entry that names why it stays on the CPU. */
    private static void addBatchScenarios(final LfNetwork net, final Network network, final PropagatedContingency propagatedContingency,
                                          final AcLoadFlowContext context, final ZeroImpedanceTreeReplay ziTree,
                                          final List<BatchContingency> batch,
                                          final List<OperatorStrategy> strategies, final BatchConfig config,
                                          final List<Action> actions) {
        java.util.Set<String> lostLfBusIds = GpuContingencyMapper.lostLfBusIds(net, network, propagatedContingency);
        List<LfGenerator> gens = propagatedContingency.getGeneratorIdsToLose().isEmpty() ? List.of() : GpuContingencyMapper.generatorOutages(net,
                                                          propagatedContingency, lostLfBusIds);
        // ZIP/voltage-dependent load model: per-scenario load removal is not a target override — CPU.
        boolean loadsBatchable = !context.getParameters().getNetworkParameters().isUseLoadModel();
        List<LoadLoss> loads =
                propagatedContingency.getLoadIdsToLose().isEmpty() ? List.of() : (loadsBatchable ? GpuContingencyMapper.loadOutages(network, net, propagatedContingency, lostLfBusIds) : null);
        // Node-breaker contingency PROPAGATION opens the target's breakers alongside it, and
        // switches are zero-impedance LfBranches — split them out so the impedant part still
        // meets the <=2-branch mask rule; the switch part becomes a per-scenario zero-impedance-element
        // outage (validator: dead-stub shape only).
        List<LfBranch> zeroImpedanceBranchOutages = propagatedContingency.getBranchIdsToOpen().keySet().stream()
                .map(net::getBranchById).filter(Objects::nonNull)
                .filter(b -> b.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC))
                .toList();
        // The tree OLF SOLVES this contingency on: the one it reached it with, re-elected by this
        // contingency's own disabled couplers (opened, or sitting on a lost bus).
        List<LfBranch> nonSpanningTreeBranches = ziTree.applyAndRevert(propagatedContingency);
        if (GpuTrace.ZITREE_TRACE.isOn() && nonSpanningTreeBranches.size() > 40) {
            System.err.println("ZITREE_REPLAY_WIDE ctg=" + propagatedContingency.getContingency().getId() + " nonTree=" + nonSpanningTreeBranches.size());
        }
        List<LfBranch> branches = propagatedContingency.getBranchIdsToOpen().isEmpty() ? List.of()
                : GpuContingencyMapper.closedBranchOutages(net, propagatedContingency, zeroImpedanceBranchOutages);
        List<BatchContingency.SideOpen> sideOpens =
                propagatedContingency.getBranchIdsToOpen().isEmpty() && propagatedContingency.getBusIdsToLose().isEmpty() ? List.of()
                : GpuContingencyMapper.sideOpens(net, propagatedContingency);
        // Shunt section shifts (both directions; full disconnection of any aggregate = minus the
        // section admittance): per-scenario admittance overrides, combinable with everything.
        List<ShuntShift> shifts =
                propagatedContingency.getShuntIdsToShift().isEmpty() ? List.of() : GpuContingencyMapper.shuntShifts(net, propagatedContingency);
        // A lost static var compensator takes its stand-by automaton shunt with it (OLF's
        // processLostGenerators), and that shunt is NOT in the propagation's shift list.
        if (shifts != null && gens != null && !gens.isEmpty()) {
            List<ShuntShift> svcShifts = GpuContingencyMapper.svcStandbyShuntShifts(gens);
            if (!svcShifts.isEmpty()) {
                shifts = java.util.stream.Stream.concat(shifts.stream(), svcShifts.stream()).toList();
            }
        }
        // AC-emulation HVDC outage: the hv element's per-scenario disabled bit (the stations'
        // injection losses ride the generator channel above); unresolvable/setpoint -> CPU.
        List<com.powsybl.openloadflow.network.LfHvdc> hvdcs = GpuContingencyMapper.hvdcOutages(net, propagatedContingency);
        if (GpuTrace.QUALIFY_TRACE.isOn()) {
            System.err.println("QUALIFY_TRACE compose " + propagatedContingency.getContingency().getId()
                    + " branches=" + (branches == null ? "null" : branches.size())
                    + " gens=" + (gens == null ? "null" : gens.size())
                    + " loads=" + (loads == null ? "null" : loads.size())
                    + " shifts=" + (shifts == null ? "null" : shifts.size())
                    + " hvdcs=" + (hvdcs == null ? "null" : hvdcs.size())
                    + " zi=" + zeroImpedanceBranchOutages.size()
                    + " sideOpens=" + (sideOpens == null ? "null" : sideOpens.size()));
            // Name the zero-impedance branches and their two ends. A busbar outage that opens
            // a COUPLER is the shape that separates the batched busbars from the CPU-routed
            // ones, and "zero-impedance=1" alone does not say which coupler or which buses it merged.
            // Each lost load as the DEVICE will apply it, next to the RAW PowerShift OLF
            // propagated. A busbar outage loses its loads through the propagation rather than
            // by naming them, and "loads=1" does not say whether the bus, the active part or
            // the reactive part is the one that differs.
            for (var loadLoss : loads == null ? List.<LoadLoss>of() : loads) {
                System.err.printf("QUALIFY_TRACE   load bus=%s p=%.9f q=%.9f%n",
                        loadLoss.bus() == null ? "?" : loadLoss.bus().getId(), loadLoss.p(), loadLoss.q());
            }
            // Each shunt shift as the DEVICE will apply it (deltaG / deltaB added to the packed g / b) next to
            // the RAW AdmittanceShift the propagation carries and the shunt's CURRENT admittance:
            // "shifts=1" does not say whether the two arms remove the same susceptance.
            for (var shift : shifts == null ? List.<ShuntShift>of() : shifts) {
                System.err.printf("QUALIFY_TRACE   shunt=%s dg=%.9f db=%.9f baseG=%.9f baseB=%.9f bus=%s%n",
                        shift.shunt().getId(), shift.deltaG(), shift.deltaB(), shift.shunt().getG(), shift.shunt().getB(),
                        "n/a");
            }
            for (var e : propagatedContingency.getShuntIdsToShift().entrySet()) {
                System.err.printf("QUALIFY_TRACE   olfShift shunt=%s g=%.9f b=%.9f%n",
                        e.getKey(), e.getValue().getG(), e.getValue().getB());
            }
            for (var e : propagatedContingency.getLoadIdsToLose().entrySet()) {
                System.err.printf("QUALIFY_TRACE   olfShift load=%s active=%.9f variableActive=%.9f"
                        + " reactive=%.9f%n", e.getKey(), e.getValue().getActive(),
                        e.getValue().getVariableActive(), e.getValue().getReactive());
            }
            // Each SIDE-OPEN with the side and the pi-model the dangling reduction will use.
            // r1 != 1 or a1 != 0 means danglingClosedBranchParameterRow takes its TRANSFORMER path, where r1^2
            // multiplies OLF's whole open-side-2 bracket and side-1-open uses the constant R2 —
            // analytically verified but, until something prints this, never known to have run.
            for (var sideOpen : sideOpens == null ? List.<BatchContingency.SideOpen>of() : sideOpens) {
                var pi = sideOpen.branch().getPiModel();
                var transformerVoltageControl = sideOpen.branch().getVoltageControl();
                System.err.printf("QUALIFY_TRACE   sideOpen %s side=%d r1=%.9f a1=%.9f y=%.6e ksi=%.6e"
                                + " vcCtl=%b vc=%s members=%d%n",
                        sideOpen.branch().getId(), sideOpen.side(), pi.getR1(), pi.getA1(), pi.getY(), pi.getKsi(),
                        sideOpen.branch().isVoltageController(),
                        transformerVoltageControl.map(voltageControl -> voltageControl.getControlledBus().getId()).orElse("-"),
                        transformerVoltageControl.map(voltageControl -> voltageControl.getMergedControllerElements().size()).orElse(0));
            }
            for (var zeroImpedanceBranch : zeroImpedanceBranchOutages) {
                LfBranch branch = net.getBranchById(zeroImpedanceBranch.toString());
                if (branch == null) {
                    System.err.println("QUALIFY_TRACE   zi " + zeroImpedanceBranch);
                } else {
                    System.err.printf("QUALIFY_TRACE   zi %s: b1=%s b2=%s spanningTree=%b%n", branch.getId(),
                            branch.getBus1() == null ? "?" : branch.getBus1().getId(),
                            branch.getBus2() == null ? "?" : branch.getBus2().getId(),
                            branch.isSpanningTreeEdge(com.powsybl.openloadflow.network.LoadFlowModel.AC));
                }
            }
        }
        if (branches != null && gens != null && loads != null && shifts != null && hvdcs != null && sideOpens != null
                && (!branches.isEmpty() || !gens.isEmpty() || !loads.isEmpty() || !shifts.isEmpty()
                    || !zeroImpedanceBranchOutages.isEmpty() || !hvdcs.isEmpty() || !sideOpens.isEmpty())) {
            batch.add(new BatchContingency(propagatedContingency.getContingency().getId(), branches, gens, loads,
                    shifts, List.of(), List.of(), List.of(), zeroImpedanceBranchOutages, hvdcs, sideOpens,
                    nonSpanningTreeBranches));
            // Batched OPERATOR STRATEGIES stage 1: TrueCondition (always-applied) strategies
            // matching this contingency become EXTRA batch scenarios — the (ctg, strategy)
            // pair is just another BatchContingency with the actions COMPOSED onto the
            // contingency's deltas (reclose = drop the branch from the outage list;
            // extra open = extend it; load/gen actions = injection deltas). Unsupported
            // action types or shapes leave the pair to the CPU path (super).
            for (OperatorStrategy strategy : strategies) {
                if (!GpuStrategyComposer.strategyApplies(strategy, propagatedContingency.getContingency().getId())) {
                    continue;
                }
                // Strategy pairs batch only under ANCHOR-INDEPENDENT factor types (P_MAX,
                // PARTICIPATION_FACTOR): the pair phase re-anchors the DS water-fill at the parent's
                // distributed targets, and TARGET (|targetP|) and the load-based factors are evaluated
                // AT THAT MOVED ANCHOR — the device's static factors would drift. Needs a per-scenario
                // factor channel (like distributedSlackBusAnchorShift) first.
                var pairBt = config.distributedSlackParameters().balanceType();
                if (pairBt != com.powsybl.loadflow.LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX
                        && pairBt != com.powsybl.loadflow.LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_PARTICIPATION_FACTOR) {
                    continue;                        // pair handled by the CPU per-strategy path
                }
                // one pair per action list OLF may apply (a multi-stage strategy's stage combinations)
                for (List<String> actionIds : GpuStrategyComposer.candidateActionLists(strategy)) {
                    BatchContingency pair = GpuStrategyComposer.composeStrategyScenario(network, net, propagatedContingency.getContingency().getId(), strategy, actionIds, actions,
                            zeroImpedanceBranchOutages, branches, gens, loads, propagatedContingency.getLoadIdsToLose().keySet(), shifts, hvdcs,
                            sideOpens, nonSpanningTreeBranches, context.getParameters().getNetworkParameters(), pairBt);
                    if (pair == null) {
                        GpuBatchScopeCensus.record("PAIR:COMPOSE_REFUSED");
                    }
                    // An absolute GeneratorAction on a participant restarts that generator from the value, over
                    // its share of the parent's distribution. The pair leg drops the parent's anchor shift of the
                    // generator's BUS, which is that share only when no other participant sits on the bus.
                    var pairDistributedSlackParameters = config.distributedSlackParameters();
                    if (pair != null && pair.absoluteTargetGenerators().stream()
                            .filter(g -> pairDistributedSlackParameters.factor(g) != 0)
                            .anyMatch(g -> g.getBus().getGenerators().stream()
                                    .anyMatch(o -> o != g && pairDistributedSlackParameters.factor(o) != 0))) {
                        pair = null;                     // the bus's split of the parent share is not known: CPU
                    }
                    // A GeneratorAction re-checks its generator's participation at the new target. A generator that
                    // did not participate and now does would have to join the device's generator pack: CPU.
                    if (pair != null && pair.generatorActions().stream().anyMatch(m -> pairDistributedSlackParameters.factor(m.generator()) == 0
                            && m.participatesAt() != null
                            && m.participatesAt().test(m.newTargetP(m.generator().getTargetP())))) {
                        GpuBatchScopeCensus.record("PAIR:ACTION_STARTS_PARTICIPATION");
                        pair = null;
                    }
                    if (pair != null) {
                        pair = withPairTree(pair, propagatedContingency, branches, zeroImpedanceBranchOutages, ziTree);
                        batch.add(pair);
                    }
                }
            }
            return;
        }
        LfShunt shunt = GpuContingencyMapper.singleShuntOutage(net, propagatedContingency);
        if (shunt != null) {
            batch.add(BatchContingency.shunt(propagatedContingency.getContingency().getId(), shunt));
        } else {
            String rejectReason = composeRejectReason(branches, gens, loads, shifts, hvdcs, sideOpens,
                    loadsBatchable);
            GpuBatchScopeCensus.record(rejectReason);
            // ...and by id: a mapper reject never reaches the device, so the harness's verdict must count it
            // with the plan refusals (a scope gap), not as a scenario the device tried and failed.
            GpuBatchScopeCensus.planRefused(propagatedContingency.getContingency().getId(), rejectReason);
            if (GpuTrace.REJECT_TRACE.isOn()) {
                // the census counts these by KIND; the id is what lets a scope report say
                // whether a non-batched contingency is one OLF itself does not answer
                System.err.println("GPU_REJECT_CTG ctg=" + propagatedContingency.getContingency().getId()
                        + " reason=" + rejectReason);
            }
        }
    }

    /** Which unmodeled element kind kept a contingency out of the batch (the compose funnel's
     *  null-guard, named): the FIRST null wins, and an all-empty-but-non-null compose means the
     *  contingency touches nothing the device models. */
    @SuppressWarnings("java:S107")
    /**
     * SECOND DEVICE PASS over the scenarios the uniform batch could not solve, in the ROBUST
     * single-matrix mode, before any CPU fallback.
     *
     * <p>Why this exists: the dominant non-batching reason on rte6515's full busbar scope is the
     * kernel's {@code max-iter-bad-solve} — a scenario spends its whole iteration budget on Newton
     * steps the UNPIVOTED batch factorization got wrong, so the device refuses to call the result
     * OLF's answer and rejects it. Measured 2026-10-01: 108 of the 117 non-batched contingencies,
     * and two thirds of those rejects sit within ONE DECADE of the acceptance gate
     * ({@code lastSolveRatio} 1e-2..1e-1 against a 1e-2 threshold), with no NaN/Inf — inaccurate
     * solves, not divergences. The remedy already exists and is proven on the BASE load flow, which
     * falls into ROBUST automatically after a linear-solve reject ({@code GpuNewtonRaphson}); a chunk
     * scenario simply had no route to it. ROBUST is a per-scenario loop with matching and global
     * column pivoting, which is what rte6515's 366 retained couplers need to factor at all.
     *
     * <p>Only scenarios that would otherwise LEAVE the device are retried, so the pass cannot change
     * an answer the device already gave: a CONVERGED result is kept, and so is a
     * {@code GpuAcSolverStatus.MAX_ITERATION_REACHED} one, which IS OLF's answer (MAX_ITERATION_REACHED, no network
     * result) rather than a reject. A retried scenario that still fails falls back to the CPU exactly
     * as before.
     *
     * <p><b>DEFAULT ON since 2026-10-02, and what it actually buys:</b> ROBUST turns 39 of the 113
     * from a bad solve into {@code GpuAcSolverStatus.MAX_ITERATION_REACHED} with a SOUND last solve
     * ({@code GPU_ROBUST_RETRY attempted=113 recovered=39 statuses={-2=74, 1=39}}), taking CPU
     * replays from 113 to 74. None of them CONVERGE — but that is the right answer: OLF does not
     * converge them either. On the full busbar scope OLF's own statuses are
     * {@code {CONVERGED=6602, FAILED=5, MAX_ITERATION_REACHED=60}}, and with this pass on, BOTH arms
     * report exactly that, with violations still diff 0 and the worst DAP unchanged.
     *
     * <p>It was first dismissed as recovering nothing, which was a measurement error worth recording:
     * it was judged by COVERAGE (6550 / 6667, unmoved), and coverage counts precomputed device
     * STATES — a {@code MAX_ITERATION_REACHED} result carries no state, so coverage CANNOT move for
     * these. The metric is how many contingencies leave the device, and the guard against claiming
     * {@code MAX_ITERATION} where OLF converges is the harness's STATUS-COUNT EQUALITY, which a
     * GPU-only run never checks.
     *
     * <p>Cost, NOT yet measured: switching mode evicts the single-entry context cache, so the pass
     * pays one cuDSS analysis plus a solve over the rejected scenarios on every run.
     */
    private static List<BatchScenarioResult> retryRejectedOnDevice(
            List<BatchScenarioResult> results, List<BatchContingency> batch, LfNetwork net,
            double[] baseState, EquationSystem<AcVariableType, AcEquationType> es,
            com.powsybl.openloadflow.equations.TargetVector<AcVariableType, AcEquationType> target,
            AcLoadFlowContext context,
            boolean distributedSlack, boolean reactiveLimits, double reactiveLimitTol,
            double slackDistTol, double convEpsPerEq, ReportNode reportNode, BatchConfig config, boolean reactiveLimitDecisions) {
        if (config.robustSolveMode()) {
            return results;
        }
        java.util.Set<String> rejected = new java.util.HashSet<>();
        for (BatchScenarioResult r : results) {
            if (r.needsCpuFallback()) {
                rejected.add(r.contingencyId());
            }
        }
        if (rejected.isEmpty()) {
            return results;
        }
        List<BatchContingency> retry = batch.stream().filter(c -> rejected.contains(c.contingencyId())).toList();
        if (retry.isEmpty()) {
            return results;
        }
        LOGGER.info("Batched N-1: retrying {} rejected scenarios on the device in the robust single-matrix mode",
                retry.size());
        List<BatchScenarioResult> retried;
        try {
            retried = GpuBatchedSecurityAnalysisSolver.solveBatchN1(
                    BatchSolveRequest.of(net, baseState, retry, es, target, deviceSolverParameters(context),
                            reportNode)
                            .withOuterLoops(distributedSlack, reactiveLimits)
                            .withOuterLoopTolerances(reactiveLimitTol, slackDistTol)
                            .withConvEpsPerEq(convEpsPerEq)
                            .withConfig(config.withRobustSolveMode())
                            .withReactiveLimitDecisions(reactiveLimitDecisions));
        } catch (RuntimeException e) {
            // The retry is an OPTIONAL extra pass: a failure in it must not lose the first pass's
            // results, which are already correct for every scenario that converged.
            LOGGER.warn("Batched N-1: robust retry pass failed ({}); keeping the first pass's results",
                    e.toString());
            return results;
        }
        // CONVERGED ONLY. Accepting !needsCpuFallback() also accepts GpuAcSolverStatus.MAX_ITERATION_REACHED, and that is a
        // REGRESSION, not a recovery: the precomputer reports such a result as OLF's own
        // MAX_ITERATION_REACHED with no network result, replacing a CPU replay that may well have
        // converged. Measured 2026-10-01 on the full busbar scope: 43 of 113 retried came back
        // MAX_ITERATION (DEVICE:MAX_ITERATION_REACHED 4 -> 47) while device coverage stayed at
        // 6550/6667 — nothing gained and 43 CPU answers thrown away. A retried scenario that does not
        // CONVERGE keeps its reject and goes to the CPU exactly as before.
        // The retry ALSO accepts GpuAcSolverStatus.MAX_ITERATION_REACHED.
        //
        // That is not the contradiction of the comment above that it looks like. OLF ITSELF does not
        // converge a large part of this tail: on the full busbar scope its own statuses are
        // {CONVERGED=6602, FAILED=5, MAX_ITERATION_REACHED=60}, so for those contingencies
        // MAX_ITERATION_REACHED IS the right answer and reporting it from the device costs nothing and
        // saves the CPU replay. What must NOT happen is claiming it for a scenario OLF converges — and
        // the guard for that is the harness's status-count equality, not coverage. COVERAGE IS THE
        // WRONG METRIC here and reading it as one is what made this look like a regression first time
        // round: it counts precomputed device STATES, and a MAX_ITERATION result carries no state, so
        // it cannot rise for these by construction. The metric that moves is how many contingencies
        // LEAVE the device: measured 113 -> 70 (tasks #67).
        Map<String, BatchScenarioResult> better = new HashMap<>();
        java.util.Map<Integer, Integer> retriedStatus = new java.util.TreeMap<>();
        int recovered = 0;
        for (BatchScenarioResult r : retried) {
            retriedStatus.merge(r.convergenceStatus(), 1, Integer::sum);
            boolean ok = r.convergenceStatus() == GpuAcSolverStatus.CONVERGED
                    || r.convergenceStatus() == GpuAcSolverStatus.MAX_ITERATION_REACHED;
            if (ok) {
                better.put(r.contingencyId(), r);
                recovered++;
            }
        }
        if (GpuTrace.REJECT_TRACE.isOn()) {
            System.err.println("GPU_ROBUST_RETRY attempted=" + retry.size() + " recovered=" + recovered
                    + " statuses=" + retriedStatus);
        }
        GpuBatchScopeCensus.record("RETRY:robust-attempted=" + retry.size());
        GpuBatchScopeCensus.record("RETRY:robust-recovered=" + recovered);
        if (better.isEmpty()) {
            return results;
        }
        return results.stream().map(r -> better.getOrDefault(r.contingencyId(), r)).toList();
    }

    /** A pair whose actions open or close couplers is solved on the tree OLF elects after those actions, not on its
     *  contingency's: a coupler the strategy recloses (open at base, or opened by the contingency) goes back in it. */
    private static BatchContingency withPairTree(BatchContingency pair, PropagatedContingency propagatedContingency, List<LfBranch> ctgBranches,
                                                 List<LfBranch> contingencyZeroImpedanceBranchOutages, ZeroImpedanceTreeReplay ziTree) {
        java.util.Set<LfBranch> ctgOut = new java.util.HashSet<>(ctgBranches);
        ctgOut.addAll(contingencyZeroImpedanceBranchOutages);
        java.util.Set<LfBranch> pairOut = new java.util.HashSet<>(pair.outagedBranches());
        pairOut.addAll(pair.outagedZeroImpedanceBranches());
        List<LfBranch> opened = pairOut.stream()
                .filter(b -> !ctgOut.contains(b) && b.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC)).toList();
        List<LfBranch> closed = new ArrayList<>(pair.reclosedBranches().stream()
                .filter(b -> b.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC)).toList());
        ctgOut.stream().filter(b -> !pairOut.contains(b) && b.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC))
                .forEach(closed::add);
        if (opened.isEmpty() && closed.isEmpty()) {
            return pair;
        }
        return new BatchContingency(pair.contingencyId(), pair.outagedBranches(), pair.lostGenerators(), pair.lostLoads(),
                pair.shuntShifts(), pair.tapOverrides(), pair.outagedShunts(), pair.reclosedBranches(), pair.outagedZeroImpedanceBranches(),
                pair.outagedHvdcs(), pair.sideOpenBranches(), ziTree.applyAndRevert(propagatedContingency, opened, closed), pair.generatorActions(),
                pair.acEmulationOffHvdcs());
    }

    private static String composeRejectReason(List<LfBranch> branches, List<LfGenerator> gens, List<LoadLoss> loads,
                                              List<ShuntShift> shifts,
                                              List<com.powsybl.openloadflow.network.LfHvdc> hvdcs,
                                              List<BatchContingency.SideOpen> sideOpens, boolean loadsBatchable) {
        if (branches == null) {
            return "MAPPER:branch-outage-unresolvable";
        }
        if (gens == null) {
            return "MAPPER:generator-outage-unresolvable";
        }
        if (loads == null) {
            return loadsBatchable ? "MAPPER:load-outage-unresolvable" : "MAPPER:load-outage-with-load-model";
        }
        if (shifts == null) {
            return "MAPPER:shunt-shift-unresolvable";
        }
        if (hvdcs == null) {
            return "MAPPER:hvdc-outage-unsupported";
        }
        if (sideOpens == null) {
            return "MAPPER:side-open-unsupported";
        }
        return "MAPPER:no-batchable-element";
    }
    /**
     * The Newton parameters the DEVICE arm solves with — the RUN's, not fresh defaults.
     *
     * <p>This used to be {@code new NewtonRaphsonParameters()}, whose
     * {@code DEFAULT_MAX_ITERATIONS} is 15. The RTE production parameters configure
     * {@code maxNewtonRaphsonIterations = 30}, so the device gave every scenario HALF the Newton
     * budget OLF gives it: one needing 16-30 iterations was declared non-converged on the device and
     * offloaded to the per-contingency CPU fallback, which then converged it. Correct results — the
     * fallback is the safety net — but wasted coverage, and it made "the device did not converge" a
     * strictly different predicate from "OLF will not converge". That difference is exactly why a
     * rejected scenario must be REPLAYED on the CPU rather than reported as diverged.
     *
     * <p>{@code getMaxIterations()} is the only field the batched path reads, so any Newton-family
     * solver's budget carries over; there is deliberately no silent default left.
     */

    private static NewtonRaphsonParameters deviceSolverParameters(AcLoadFlowContext context) {
        NewtonRaphsonParameters newtonRaphsonParameters = new NewtonRaphsonParameters();
        if (context.getParameters().getAcSolverParameters() instanceof AbstractNewtonParameters<?> newtonParameters) {
            newtonRaphsonParameters.setMaxIterations(newtonParameters.getMaxIterations());
        }
        return newtonRaphsonParameters;
    }

    /** The device's own loss pre-distribution and distributed total for a scenario, where it reported them. */
    private static void recordDistributed(BatchScenarioResult r, Map<String, Double> preDistributed,
                                          Map<String, Double> distributed) {
        if (!Double.isNaN(r.preDistributedActivePower())) {
            preDistributed.put(r.contingencyId(), r.preDistributedActivePower());
        }
        if (!Double.isNaN(r.distributedActivePower())) {
            distributed.put(r.contingencyId(), r.distributedActivePower());
        }
        // OLF_GPU_ITER_TRACE_CTG=<id>[,<id>]: the two parts of a named scenario's distributed active
        // power, in MW - the replayed loss pre-distribution and the device's own slack passes on top.
        // OLF reports their sum, so a total that differs cannot say which half moved; OLF's own
        // LF-level part is on OLF_LF_STEPS_TRACE and its pre-distribution is the reported total minus
        // that.
        String want = GpuTrace.ITER_TRACE_CTG.value();
        if (want != null) {
            for (String w : want.split(",")) {
                if (w.trim().equals(r.contingencyId())) {
                    double preDistributedMw = r.preDistributedActivePower() * com.powsybl.openloadflow.util.PerUnit.SB;
                    double totalMw = r.distributedActivePower() * com.powsybl.openloadflow.util.PerUnit.SB;
                    System.err.printf("DAP_CTG ctg=%s preDistributedMW=%.9f totalMW=%.9f ownDsMW=%.9f iterations=%d status=%d%n",
                            r.contingencyId(), preDistributedMw, totalMw, totalMw - preDistributedMw, r.iterations(), r.convergenceStatus());
                }
            }
        }
    }
}
