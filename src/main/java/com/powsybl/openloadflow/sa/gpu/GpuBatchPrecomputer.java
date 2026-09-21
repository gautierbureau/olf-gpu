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
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.solver.AbstractNewtonParameters;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.ac.solver.gpu.batch.BatchConfig;
import com.powsybl.openloadflow.ac.solver.gpu.batch.BatchContingency;
import com.powsybl.openloadflow.ac.solver.gpu.batch.BatchScenarioResult;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchElementMapper;
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
        if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {   // the network the batch starts from
            for (var b : net.getBuses()) {
                for (var g : b.getGenerators()) {
                    System.err.println("BASE_GEN " + g.getId() + " targetP=" + g.getTargetP() + " targetQ=" + g.getTargetQ()
                            + " vc=" + b.isGeneratorVoltageControlEnabled());
                }
            }
        }
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
            if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
                System.err.println("PRECOMP_BAIL avail=" + GpuAcNewtonSolver.isAvailable() + " net=" + (net != null)
                        + " ctx=" + (context != null) + " nCtg=" + contingencies.size());
            }
            return PrecomputedBatch.empty();
        }
        if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
            System.err.println("PRECOMP_ENTER nCtg=" + contingencies.size());
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
                java.util.Set<String> lostLfBusIds = GpuContingencyMapper.lostLfBusIds(net, network, pc);
                List<LfGenerator> gens = pc.getGeneratorIdsToLose().isEmpty() ? List.of() : GpuContingencyMapper.generatorOutages(net, pc, lostLfBusIds);
                // ZIP/voltage-dependent load model: per-scenario load removal is not a target override — CPU.
                boolean loadsBatchable = !context.getParameters().getNetworkParameters().isUseLoadModel();
                List<LoadLoss> loads =
                        pc.getLoadIdsToLose().isEmpty() ? List.of() : (loadsBatchable ? GpuContingencyMapper.loadOutages(network, net, pc, lostLfBusIds) : null);
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
                List<BatchContingency.SideOpen> sideOpens =
                        pc.getBranchIdsToOpen().isEmpty() && pc.getBusIdsToLose().isEmpty() ? List.of()
                        : GpuContingencyMapper.sideOpens(net, pc);
                // Shunt section shifts (both directions; full disconnection of any aggregate = minus the
                // section admittance): per-scenario admittance overrides, combinable with everything.
                List<ShuntShift> shifts =
                        pc.getShuntIdsToShift().isEmpty() ? List.of() : GpuContingencyMapper.shuntShifts(net, pc);
                // AC-emulation HVDC outage: the hv element's per-scenario disabled bit (the stations'
                // injection losses ride the generator channel above); unresolvable/setpoint -> CPU.
                List<com.powsybl.openloadflow.network.LfHvdc> hvdcs = GpuContingencyMapper.hvdcOutages(net, pc);
                if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                    System.err.println("QUALIFY_TRACE compose " + pc.getContingency().getId()
                            + " branches=" + (branches == null ? "null" : branches.size())
                            + " gens=" + (gens == null ? "null" : gens.size())
                            + " loads=" + (loads == null ? "null" : loads.size())
                            + " shifts=" + (shifts == null ? "null" : shifts.size())
                            + " hvdcs=" + (hvdcs == null ? "null" : hvdcs.size())
                            + " zi=" + ziOutages.size()
                            + " sideOpens=" + (sideOpens == null ? "null" : sideOpens.size()));
                    // Name the zero-impedance branches and their two ends. A busbar outage that opens
                    // a COUPLER is the shape that separates the batched busbars from the CPU-routed
                    // ones, and "zi=1" alone does not say which coupler or which buses it merged.
                    // Each lost load as the DEVICE will apply it, next to the RAW PowerShift OLF
                    // propagated. A busbar outage loses its loads through the propagation rather than
                    // by naming them, and "loads=1" does not say whether the bus, the active part or
                    // the reactive part is the one that differs.
                    for (var ll : loads == null ? List.<LoadLoss>of() : loads) {
                        System.err.printf("QUALIFY_TRACE   load bus=%s p=%.9f q=%.9f%n",
                                ll.bus() == null ? "?" : ll.bus().getId(), ll.p(), ll.q());
                    }
                    for (var e : pc.getLoadIdsToLose().entrySet()) {
                        System.err.printf("QUALIFY_TRACE   olfShift load=%s active=%.9f variableActive=%.9f"
                                + " reactive=%.9f%n", e.getKey(), e.getValue().getActive(),
                                e.getValue().getVariableActive(), e.getValue().getReactive());
                    }
                    // Each SIDE-OPEN with the side and the pi-model the dangling reduction will use.
                    // r1 != 1 or a1 != 0 means danglingCbRow takes its TRANSFORMER path, where r1^2
                    // multiplies OLF's whole open-side-2 bracket and side-1-open uses the constant R2 —
                    // analytically verified but, until something prints this, never known to have run.
                    for (var so : sideOpens == null ? List.<BatchContingency.SideOpen>of() : sideOpens) {
                        var pi = so.branch().getPiModel();
                        var vcO = so.branch().getVoltageControl();
                        System.err.printf("QUALIFY_TRACE   sideOpen %s side=%d r1=%.9f a1=%.9f y=%.6e ksi=%.6e"
                                        + " vcCtl=%b vc=%s members=%d%n",
                                so.branch().getId(), so.side(), pi.getR1(), pi.getA1(), pi.getY(), pi.getKsi(),
                                so.branch().isVoltageController(),
                                vcO.map(vc -> vc.getControlledBus().getId()).orElse("-"),
                                vcO.map(vc -> vc.getMergedControllerElements().size()).orElse(0));
                    }
                    for (var zi : ziOutages) {
                        LfBranch br = net.getBranchById(zi.toString());
                        if (br == null) {
                            System.err.println("QUALIFY_TRACE   zi " + zi);
                        } else {
                            System.err.printf("QUALIFY_TRACE   zi %s: b1=%s b2=%s spanningTree=%b%n", br.getId(),
                                    br.getBus1() == null ? "?" : br.getBus1().getId(),
                                    br.getBus2() == null ? "?" : br.getBus2().getId(),
                                    br.isSpanningTreeEdge(com.powsybl.openloadflow.network.LoadFlowModel.AC));
                        }
                    }
                }
                if (branches != null && gens != null && loads != null && shifts != null && hvdcs != null && sideOpens != null
                        && (!branches.isEmpty() || !gens.isEmpty() || !loads.isEmpty() || !shifts.isEmpty()
                            || !ziOutages.isEmpty() || !hvdcs.isEmpty() || !sideOpens.isEmpty())) {
                    batch.add(new BatchContingency(pc.getContingency().getId(), branches, gens, loads,
                            shifts, List.of(), List.of(), List.of(), ziOutages, hvdcs, sideOpens));
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
                        // Strategy pairs batch only under ANCHOR-INDEPENDENT factor types (P_MAX,
                        // PARTICIPATION_FACTOR): the pair phase re-anchors the DS water-fill at the parent's
                        // distributed targets, and TARGET (|targetP|) and the load-based factors are evaluated
                        // AT THAT MOVED ANCHOR — the device's static factors would drift. Needs a per-scenario
                        // factor channel (like dsAnchorShift) first.
                        var pairBt = config.dsParams().balanceType();
                        if (pairBt != com.powsybl.loadflow.LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX
                                && pairBt != com.powsybl.loadflow.LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_PARTICIPATION_FACTOR) {
                            continue;                        // pair handled by the CPU per-strategy path
                        }
                        BatchContingency pair = GpuStrategyComposer.composeStrategyScenario(network, net, pc.getContingency().getId(), strategy, actions,
                                ziOutages, branches, gens, loads, shifts, hvdcs);
                        if (pair != null) {
                            batch.add(pair);
                        }
                    }
                    continue;
                }
                LfShunt shunt = GpuContingencyMapper.singleShuntOutage(net, pc);
                if (shunt != null) {
                    batch.add(BatchContingency.shunt(pc.getContingency().getId(), shunt));
                } else {
                    GpuBatchScopeCensus.record(composeRejectReason(branches, gens, loads, shifts, hvdcs, sideOpens,
                            loadsBatchable));
                }
            }
            if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
                System.err.println("PRECOMP_BATCH size=" + batch.size());
            }
            if (batch.isEmpty()) {
                return PrecomputedBatch.empty();
            }

            // The packed element data and the branch->element map, for the violation screen (see
            // GpuViolationScreen). Extracted HERE rather than lifted out of the solver: at this point
            // it is the cache hit the solver is about to take, and the plain (no base-open) flavor is
            // self-consistent with the plain mapper below — a base-open branch simply maps to no
            // element and falls back to the host scan.
            GpuAcData screenData = GpuAcDataCache.extract(net, es, distributedSlack, reactiveLimits, config.dsParams());
            // OLF_GPU_DS_BUSMAP=<file>: DS-pack index -> bus id, written at BASE state. The chunk
            // solver dumps the distributed power per DS index (GPU_DS_TARGET) but has no network
            // handle to name the bus; OLF's own side (OLF_DS_SPLIT) is keyed by bus id. This map is
            // what joins the two, which is the only way to compare the SPLIT rather than the total.
            String dsBusMap = System.getenv("OLF_GPU_DS_BUSMAP");
            if (dsBusMap != null && screenData != null && screenData.dsPhiRow() != null) {
                java.util.Map<Integer, String> busByPhiRow = new java.util.HashMap<>();
                for (var b : net.getBuses()) {
                    var v = es.getVariableSet().getVariable(b.getNum(), AcVariableType.BUS_PHI);
                    if (v != null && v.getRow() >= 0) {
                        busByPhiRow.put(v.getRow(), b.getId());
                    }
                }
                try (java.io.PrintWriter w = new java.io.PrintWriter(new java.io.FileWriter(dsBusMap))) {
                    int[] rows = screenData.dsPhiRow();
                    for (int i = 0; i < rows.length; i++) {
                        w.printf("%d %s%n", i, busByPhiRow.getOrDefault(rows[i], "?row" + rows[i]));
                    }
                }
            }
            GpuBatchElementMapper screenMapper = new GpuBatchElementMapper(net);

            List<BatchScenarioResult> results = config.outerLoopMode() == OuterLoopMode.JAVA
                    ? GpuBatchedSecurityAnalysisSolver.solveBatchN1JavaOuterLoops(
                            net, context, baseState, batch, es, target,
                            reactiveLimitTol, slackDistTol, convEpsPerEq, config, reportNode)
                    : GpuBatchedSecurityAnalysisSolver.solveBatchN1(
                            net, baseState, batch, es, target, deviceSolverParameters(context),
                            distributedSlack, reactiveLimits,
                            reactiveLimitTol, slackDistTol, convEpsPerEq, reportNode, config);
            Map<String, double[]> states = new HashMap<>();
            Map<String, int[]> islandedBuses = new HashMap<>();
            for (BatchScenarioResult r : results) {
                if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                    System.err.println("QUALIFY_TRACE result " + r.contingencyId() + " status=" + r.convergenceStatus()
                            + " iters=" + r.iterations() + " norm=" + r.finalMismatch());
                }
                if (r.hasConverged()) {
                    states.put(r.contingencyId(), r.state());
                    if (r.islandedBusNums().length > 0) {
                        islandedBuses.put(r.contingencyId(), r.islandedBusNums());
                    }
                } else {
                    GpuBatchScopeCensus.record("RUNTIME:status=" + r.convergenceStatus());
                }
            }
            Map<Variable<AcVariableType>, Integer> baseVariableRows = new HashMap<>();
            // OLF_GPU_NAME_ROW=<n>[,<n>]: which element does a DEVICE row belong to? The device
            // numbers its rows by OLF's own variable row (this map is the identity), so a row index
            // printed by OLF_GPU_WORST_ROW or OLF_GPU_PROBE_ROWS can be turned back into an element
            // here and nowhere else — `getEquationAtColumn` resolves against an activation-dependent
            // index and has named the wrong element three times.
            java.util.Set<Integer> nameRows = new java.util.HashSet<>();
            boolean nameAll = "all".equals(System.getenv("OLF_GPU_NAME_ROW"));
            if (System.getenv("OLF_GPU_NAME_ROW") != null && !nameAll) {
                for (String t : System.getenv("OLF_GPU_NAME_ROW").split(",")) {
                    nameRows.add(Integer.parseInt(t.trim()));
                }
            }
            for (var v : es.getIndex().getSortedVariablesToFind()) {
                baseVariableRows.put(v, v.getRow());
                if (nameAll || nameRows.contains(v.getRow())) {
                    var el = net.getElement(v.getType().getElementType(), v.getElementNum());
                    System.err.printf("NAME_ROW %d: var=%s elementNum=%d id=%s%n",
                            v.getRow(), v.getType(), v.getElementNum(), el == null ? "?" : el.getId());
                }
            }
            LOGGER.info("GPU batched SA: precomputed {} of {} branch (N-1..N-k) contingencies on the device",
                    states.size(), batch.size());

            return new PrecomputedBatch(states, islandedBuses, baseVariableRows, es, screenData, screenMapper);
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
            if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
                e.printStackTrace();                         // the test harness's logback appenders are broken
            }                                                // (core/classic version mismatch) — surface it here
            return PrecomputedBatch.empty();
        }
    }

    /** Which unmodeled element kind kept a contingency out of the batch (the compose funnel's
     *  null-guard, named): the FIRST null wins, and an all-empty-but-non-null compose means the
     *  contingency touches nothing the device models. */
    @SuppressWarnings("java:S107")
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
        NewtonRaphsonParameters nrp = new NewtonRaphsonParameters();
        if (context.getParameters().getAcSolverParameters() instanceof AbstractNewtonParameters<?> run) {
            nrp.setMaxIterations(run.getMaxIterations());
        }
        return nrp;
    }

}
