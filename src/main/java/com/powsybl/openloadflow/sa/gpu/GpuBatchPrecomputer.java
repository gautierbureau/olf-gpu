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
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.ac.solver.gpu.batch.BatchConfig;
import com.powsybl.openloadflow.ac.solver.gpu.batch.BatchContingency;
import com.powsybl.openloadflow.ac.solver.gpu.batch.BatchScenarioResult;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchedSecurityAnalysisSolver;
import com.powsybl.openloadflow.ac.solver.gpu.batch.LoadLoss;
import com.powsybl.openloadflow.ac.solver.gpu.batch.OuterLoopMode;
import com.powsybl.openloadflow.ac.solver.gpu.batch.ShuntShift;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfShunt;
import com.powsybl.openloadflow.network.impl.PropagatedContingency;
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

    static PrecomputedBatch precompute(Network network, LfNetwork net, AcLoadFlowContext context,
                                       List<PropagatedContingency> contingencies, List<OperatorStrategy> strategies,
                                       List<Action> actions, boolean distributedSlack, boolean reactiveLimits,
                                       double reactiveLimitTol, double slackDistTol, double convEpsPerEq,
                                       BatchConfig config, ReportNode reportNode) {
        synchronized (GPU_DEVICE_LOCK) {
            return precomputeLocked(network, net, context, contingencies, strategies, actions, distributedSlack,
                    reactiveLimits, reactiveLimitTol, slackDistTol, convEpsPerEq, config, reportNode);
        }
    }

    private static PrecomputedBatch precomputeLocked(Network network, LfNetwork net, AcLoadFlowContext context,
                                                     List<PropagatedContingency> contingencies, List<OperatorStrategy> strategies,
                                                     List<Action> actions, boolean distributedSlack, boolean reactiveLimits,
                                                     double reactiveLimitTol, double slackDistTol, double convEpsPerEq,
                                                     BatchConfig config, ReportNode reportNode) {
        if (!GpuAcNewtonSolver.isAvailable() || net == null || context == null || contingencies.isEmpty()) {
            return PrecomputedBatch.empty();
        }
        try {
            // REUSE the engine's equation system + target vector (do NOT create a second system — that would
            // re-point the network's evaluables off the engine's system, breaking fall-through CPU solves).
            // After the converged base load flow the engine's state vector holds the base operating point.
            EquationSystem<AcVariableType, AcEquationType> es = context.getEquationSystem();
            double[] baseState = es.getStateVector().get().clone();
            var target = context.getTargetVector();

            List<BatchContingency> batch = new ArrayList<>();
            for (PropagatedContingency pc : contingencies) {
                List<LfGenerator> gens = pc.getGeneratorIdsToLose().isEmpty() ? List.of() : GpuContingencyMapper.generatorOutages(net, pc);
                // ZIP/voltage-dependent load model: per-scenario load removal is not a target override — CPU.
                boolean loadsBatchable = !context.getParameters().getNetworkParameters().isUseLoadModel();
                List<LoadLoss> loads =
                        pc.getLoadIdsToLose().isEmpty() ? List.of() : (loadsBatchable ? GpuContingencyMapper.loadOutages(network, net, pc) : null);
                // Node-breaker contingency PROPAGATION opens the target's breakers alongside it, and
                // switches are zero-impedance LfBranches — split them out so the impedant part still
                // meets the <=2-branch mask rule; the switch part becomes a per-scenario zi-element
                // outage (validator: dead-stub shape only).
                List<LfBranch> ziOutages = pc.getBranchIdsToOpen().keySet().stream()
                        .map(net::getBranchById).filter(Objects::nonNull)
                        .filter(b -> b.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC))
                        .toList();
                List<LfBranch> branches = pc.getBranchIdsToOpen().isEmpty() ? List.of()
                        : GpuContingencyMapper.closedBranchOutages(net, pc, ziOutages);
                // Shunt section shifts (both directions; full disconnection of any aggregate = minus the
                // section admittance): per-scenario admittance overrides, combinable with everything.
                List<ShuntShift> shifts =
                        pc.getShuntIdsToShift().isEmpty() ? List.of() : GpuContingencyMapper.shuntShifts(net, pc);
                if (branches != null && gens != null && loads != null && shifts != null
                        && (!branches.isEmpty() || !gens.isEmpty() || !loads.isEmpty() || !shifts.isEmpty()
                            || !ziOutages.isEmpty())) {
                    batch.add(BatchContingency.branchesAndInjections(pc.getContingency().getId(), branches, gens, loads,
                            shifts, List.of(), ziOutages));
                    // Batched OPERATOR STRATEGIES stage 1: TrueCondition (always-applied) strategies
                    // matching this contingency become EXTRA batch scenarios — the (ctg, strategy)
                    // pair is just another BatchContingency with the actions COMPOSED onto the
                    // contingency's deltas (reclose = drop the branch from the outage list;
                    // extra open = extend it; load/gen actions = injection deltas). Unsupported
                    // action types or shapes leave the pair to the CPU path (super).
                    for (OperatorStrategy strategy : strategies) {
                        if (!GpuStrategyComposer.strategyApplies(strategy, pc.getContingency().getId())) {
                            continue;
                        }
                        BatchContingency pair = GpuStrategyComposer.composeStrategyScenario(network, net, pc.getContingency().getId(), strategy, actions,
                                ziOutages, branches, gens, loads, shifts);
                        if (pair != null) {
                            batch.add(pair);
                        }
                    }
                    continue;
                }
                LfShunt shunt = GpuContingencyMapper.singleShuntOutage(net, pc);
                if (shunt != null) {
                    batch.add(BatchContingency.shunt(pc.getContingency().getId(), shunt));
                }
            }
            if (batch.isEmpty()) {
                return PrecomputedBatch.empty();
            }

            List<BatchScenarioResult> results = config.outerLoopMode() == OuterLoopMode.JAVA
                    ? GpuBatchedSecurityAnalysisSolver.solveBatchN1JavaOuterLoops(
                            net, context, baseState, batch, es, target, config, reportNode)
                    : GpuBatchedSecurityAnalysisSolver.solveBatchN1(
                            net, baseState, batch, es, target, new NewtonRaphsonParameters(),
                            distributedSlack, reactiveLimits,
                            reactiveLimitTol, slackDistTol, convEpsPerEq, reportNode, config);
            Map<String, double[]> states = new HashMap<>();
            Map<String, int[]> islandedBuses = new HashMap<>();
            for (BatchScenarioResult r : results) {
                if (r.hasConverged()) {
                    states.put(r.contingencyId(), r.state());
                    if (r.islandedBusNums().length > 0) {
                        islandedBuses.put(r.contingencyId(), r.islandedBusNums());
                    }
                }
            }
            Map<Variable<AcVariableType>, Integer> baseVariableRows = new HashMap<>();
            for (var v : es.getIndex().getSortedVariablesToFind()) {
                baseVariableRows.put(v, v.getRow());
            }
            LOGGER.info("GPU batched SA: precomputed {} of {} branch (N-1/N-2) contingencies on the device",
                    states.size(), batch.size());
            return new PrecomputedBatch(states, islandedBuses, baseVariableRows, es);
        } catch (Exception e) {
            LOGGER.warn("GPU batched SA precompute failed ({}); falling back to per-contingency CPU solves", e.toString());
            return PrecomputedBatch.empty();
        }
    }
}
