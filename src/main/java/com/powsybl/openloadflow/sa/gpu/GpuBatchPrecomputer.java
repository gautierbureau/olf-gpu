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
            // OLF's zero-impedance spanning tree, replayed. The non-tree edge of each ZI network is the
            // LAST one a contingency disabled (removeBranchAndTryToSplit marks it non-tree and re-elects;
            // addBranch never restores it), so walking the list in OLF's own order reproduces the tree it
            // will have at every contingency. Union-find partitions the couplers into networks; the seed is
            // the base tree, which is what the device's structure was extracted on.
            java.util.List<LfBranch> ziAll = net.getBranches().stream()
                    .filter(b -> b.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC))
                    .toList();
            java.util.Map<Integer, Integer> uf = new java.util.HashMap<>();   // bus num -> root
            java.util.function.IntUnaryOperator find = new java.util.function.IntUnaryOperator() {
                @Override
                public int applyAsInt(int x) {
                    int r = x;
                    while (uf.getOrDefault(r, r) != r) {
                        r = uf.get(r);
                    }
                    return r;
                }
            };
            for (LfBranch b : ziAll) {
                if (b.getBus1() != null && b.getBus2() != null) {
                    uf.put(find.applyAsInt(b.getBus1().getNum()), find.applyAsInt(b.getBus2().getNum()));
                }
            }
            java.util.Map<String, Integer> ziNetOf = new java.util.HashMap<>();
            for (LfBranch b : ziAll) {
                if (b.getBus1() != null) {
                    ziNetOf.put(b.getId(), find.applyAsInt(b.getBus1().getNum()));
                }
            }
            java.util.Map<Integer, LfBranch> sticky = new java.util.TreeMap<>();
            for (LfBranch b : ziAll) {
                if (!b.isSpanningTreeEdge(com.powsybl.openloadflow.network.LoadFlowModel.AC)
                        && ziNetOf.containsKey(b.getId())) {
                    sticky.put(ziNetOf.get(b.getId()), b);
                }
            }
            java.util.List<LfBranch> baseNonTreeZi = List.copyOf(sticky.values());
            // A coupler only STAYS out of the tree if removing it leaves its network connected. A BRIDGE
            // splits the network instead: removeBranchAndTryToSplit builds two new LfZeroImpedanceNetworks,
            // and on revert addBranchAndMerge builds a merged one whose constructor re-elects over the full
            // edge set and puts the branch back in the tree. So only non-bridges are sticky - which is why
            // OLF shows AAWA out at LITJ and back in at LITK, while LMO, once disabled, never returns.
            java.util.Map<Integer, java.util.List<LfBranch>> ziByNet = new java.util.HashMap<>();
            for (LfBranch b : ziAll) {
                Integer k = ziNetOf.get(b.getId());
                if (k != null) {
                    ziByNet.computeIfAbsent(k, x -> new ArrayList<>()).add(b);
                }
            }
            java.util.Set<String> ziNonBridge = new java.util.HashSet<>();
            for (var e : ziByNet.entrySet()) {
                java.util.List<LfBranch> edges = e.getValue();
                for (LfBranch skip : edges) {
                    java.util.Map<Integer, Integer> uf2 = new java.util.HashMap<>();
                    java.util.function.IntUnaryOperator f2 = new java.util.function.IntUnaryOperator() {
                        @Override
                        public int applyAsInt(int x) {
                            int r = x;
                            while (uf2.getOrDefault(r, r) != r) {
                                r = uf2.get(r);
                            }
                            return r;
                        }
                    };
                    java.util.Set<Integer> buses = new java.util.HashSet<>();
                    for (LfBranch b : edges) {
                        if (b.getBus1() == null || b.getBus2() == null) {
                            continue;
                        }
                        buses.add(b.getBus1().getNum());
                        buses.add(b.getBus2().getNum());
                        if (b != skip) {
                            uf2.put(f2.applyAsInt(b.getBus1().getNum()), f2.applyAsInt(b.getBus2().getNum()));
                        }
                    }
                    long roots = buses.stream().map(f2::applyAsInt).distinct().count();
                    if (roots == 1) {
                        ziNonBridge.add(skip.getId());       // still connected without it: a cycle edge
                    }
                }
            }
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
                // The tree OLF has when it REACHES this contingency (its own couplers come off on top of
                // it, which the per-scenario zi-outage rows already carry), then the evolution for the
                // contingencies after it: a coupler this one disables becomes its network's non-tree edge
                // and stays there. Couplers go off both by being opened and by sitting on a lost bus.
                List<LfBranch> stickyBefore = List.copyOf(sticky.values());
                if (System.getenv("OLF_GPU_ZI_TREE_REPLAY") != null) {
                    System.err.println("ZI_TREE_REPLAY ctg=" + pc.getContingency().getId() + " nonTree="
                            + java.util.stream.Stream.concat(stickyBefore.stream(), ziDisabledPreview(net, pc).stream())
                                    .map(LfBranch::getId).distinct().sorted().toList());
                }
                java.util.List<LfBranch> ziDisabled = new ArrayList<>(ziOutages);
                for (String busId : pc.getBusIdsToLose()) {
                    com.powsybl.openloadflow.network.LfBus lost = net.getBusById(busId);
                    if (lost != null) {
                        for (LfBranch inc : lost.getBranches()) {
                            if (inc.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC)) {
                                ziDisabled.add(inc);
                            }
                        }
                    }
                }
                // Per network: ONE cycle edge off leaves that edge out for good (the reduced graph is
                // re-elected and the revert only re-adds the edge). TWO or more off splits the network,
                // and the revert MERGES it - and a merged LfZeroImpedanceNetwork re-elects in its
                // constructor over the whole edge set, which lands back on the base choice. Anything
                // else (bridges) is restored by that same merge.
                java.util.Map<Integer, java.util.List<LfBranch>> disabledByNet = new java.util.HashMap<>();
                for (LfBranch d : ziDisabled) {
                    Integer k = ziNetOf.get(d.getId());
                    if (k != null) {
                        disabledByNet.computeIfAbsent(k, x -> new ArrayList<>()).add(d);
                    }
                }
                for (var en : disabledByNet.entrySet()) {
                    java.util.List<LfBranch> off = en.getValue().stream().distinct().toList();
                    int netSize = ziByNet.getOrDefault(en.getKey(), List.of()).size();
                    java.util.List<LfBranch> offCycle = off.stream()
                            .filter(b -> ziNonBridge.contains(b.getId())).toList();
                    if (off.size() < netSize && !offCycle.isEmpty()) {
                        sticky.put(en.getKey(), offCycle.get(offCycle.size() - 1));
                    } else if (off.size() >= netSize && netSize > 0) {
                        baseNonTreeZi.stream()
                                .filter(b -> en.getKey().equals(ziNetOf.get(b.getId())))
                                .findFirst().ifPresent(b -> sticky.put(en.getKey(), b));
                    }
                }
                List<LfBranch> branches = pc.getBranchIdsToOpen().isEmpty() ? List.of()
                        : GpuContingencyMapper.closedBranchOutages(net, pc, ziOutages);
                List<BatchContingency.SideOpen> sideOpens =
                        pc.getBranchIdsToOpen().isEmpty() && pc.getBusIdsToLose().isEmpty() ? List.of()
                        : GpuContingencyMapper.sideOpens(net, pc);
                // Shunt section shifts (both directions; full disconnection of any aggregate = minus the
                // section admittance): per-scenario admittance overrides, combinable with everything.
                List<ShuntShift> shifts =
                        pc.getShuntIdsToShift().isEmpty() ? List.of() : GpuContingencyMapper.shuntShifts(net, pc);
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
                    // Each shunt shift as the DEVICE will apply it (dg/db added to the packed g/b) next to
                    // the RAW AdmittanceShift the propagation carries and the shunt's CURRENT admittance:
                    // "shifts=1" does not say whether the two arms remove the same susceptance.
                    for (var sh : shifts == null ? List.<ShuntShift>of() : shifts) {
                        System.err.printf("QUALIFY_TRACE   shunt=%s dg=%.9f db=%.9f baseG=%.9f baseB=%.9f bus=%s%n",
                                sh.shunt().getId(), sh.dg(), sh.db(), sh.shunt().getG(), sh.shunt().getB(),
                                "n/a");
                    }
                    for (var e : pc.getShuntIdsToShift().entrySet()) {
                        System.err.printf("QUALIFY_TRACE   olfShift shunt=%s g=%.9f b=%.9f%n",
                                e.getKey(), e.getValue().getG(), e.getValue().getB());
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
                            shifts, List.of(), List.of(), List.of(), ziOutages, hvdcs, sideOpens,
                            stickyBefore));
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
                    String rejectReason = composeRejectReason(branches, gens, loads, shifts, hvdcs, sideOpens,
                            loadsBatchable);
                    GpuBatchScopeCensus.record(rejectReason);
                    if (System.getenv("OLF_GPU_REJECT_TRACE") != null) {
                        // the census counts these by KIND; the id is what lets a scope report say
                        // whether a non-batched contingency is one OLF itself does not answer
                        System.err.println("GPU_REJECT_CTG ctg=" + pc.getContingency().getId()
                                + " reason=" + rejectReason);
                    }
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
            results = retryRejectedOnDevice(results, batch, net, baseState, es, target, context,
                    distributedSlack, reactiveLimits, reactiveLimitTol, slackDistTol, convEpsPerEq,
                    reportNode, config);
            Map<String, double[]> states = new HashMap<>();
            Map<String, int[]> islandedBuses = new HashMap<>();
            java.util.Set<String> exhausted = new java.util.HashSet<>();
            Map<String, Double> preDistributed = new HashMap<>();
            Map<String, Double> distributed = new HashMap<>();
            for (BatchScenarioResult r : results) {
                if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                    System.err.println("QUALIFY_TRACE result " + r.contingencyId() + " status=" + r.convergenceStatus()
                            + " iters=" + r.iterations() + " norm=" + r.finalMismatch());
                }
                if (r.hasConverged()) {
                    states.put(r.contingencyId(), r.state());
                    recordDistributed(r, preDistributed, distributed);
                    if (r.islandedBusNums().length > 0) {
                        islandedBuses.put(r.contingencyId(), r.islandedBusNums());
                    }
                } else if (r.convergenceStatus() == com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchStatus.STATUS_MAX_ITERATION
                        && r.contingencyId().indexOf(com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchStatus.STRATEGY_KEY_SEP) < 0) {
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
                    if (System.getenv("OLF_GPU_REJECT_TRACE") != null) {
                        // pairs with the kernel's GPU_REJECT slot= reason= lines (same switch)
                        System.err.println("GPU_REJECT_CTG ctg=" + r.contingencyId() + " status=" + r.convergenceStatus());
                    }
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

            return new PrecomputedBatch(states, islandedBuses, exhausted, baseVariableRows, es, screenData, screenMapper,
                    preDistributed, distributed);
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
     * falls into mode 6 automatically after a linear-solve reject ({@code GpuNewtonRaphson}); a chunk
     * scenario simply had no route to it. Mode 6 is a per-scenario loop with matching and global
     * column pivoting, which is what rte6515's 366 retained couplers need to factor at all.
     *
     * <p>Only scenarios that would otherwise LEAVE the device are retried, so the pass cannot change
     * an answer the device already gave: a CONVERGED result is kept, and so is a
     * {@code STATUS_MAX_ITERATION} one, which IS OLF's answer (MAX_ITERATION_REACHED, no network
     * result) rather than a reject. A retried scenario that still fails falls back to the CPU exactly
     * as before.
     *
     * <p><b>DEFAULT ON since 2026-10-02, and what it actually buys:</b> mode 6 turns 39 of the 113
     * from a bad solve into {@code STATUS_MAX_ITERATION} with a SOUND last solve
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
     * {@code -Dolf.gpu.sa.robustRetry=false} disables it; {@code
     * -Dolf.gpu.sa.robustRetryAcceptMaxIter=false} keeps the pass but accepts only CONVERGED.
     */
    private static List<BatchScenarioResult> retryRejectedOnDevice(
            List<BatchScenarioResult> results, List<BatchContingency> batch, LfNetwork net,
            double[] baseState, EquationSystem<AcVariableType, AcEquationType> es,
            com.powsybl.openloadflow.equations.TargetVector<AcVariableType, AcEquationType> target,
            AcLoadFlowContext context,
            boolean distributedSlack, boolean reactiveLimits, double reactiveLimitTol,
            double slackDistTol, double convEpsPerEq, ReportNode reportNode, BatchConfig config) {
        if (!Boolean.parseBoolean(System.getProperty("olf.gpu.sa.robustRetry", "true"))
                || config.robustSolveMode()) {
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
                    net, baseState, retry, es, target, deviceSolverParameters(context),
                    distributedSlack, reactiveLimits,
                    reactiveLimitTol, slackDistTol, convEpsPerEq, reportNode, config.withRobustSolveMode());
        } catch (RuntimeException e) {
            // The retry is an OPTIONAL extra pass: a failure in it must not lose the first pass's
            // results, which are already correct for every scenario that converged.
            LOGGER.warn("Batched N-1: robust retry pass failed ({}); keeping the first pass's results",
                    e.toString());
            return results;
        }
        // CONVERGED ONLY. Accepting !needsCpuFallback() also accepts STATUS_MAX_ITERATION, and that is a
        // REGRESSION, not a recovery: the precomputer reports such a result as OLF's own
        // MAX_ITERATION_REACHED with no network result, replacing a CPU replay that may well have
        // converged. Measured 2026-10-01 on the full busbar scope: 43 of 113 retried came back
        // MAX_ITERATION (DEVICE:MAX_ITERATION_REACHED 4 -> 47) while device coverage stayed at
        // 6550/6667 — nothing gained and 43 CPU answers thrown away. A retried scenario that does not
        // CONVERGE keeps its reject and goes to the CPU exactly as before.
        // -Dolf.gpu.sa.robustRetryAcceptMaxIter=true ALSO accepts STATUS_MAX_ITERATION from the retry.
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
        boolean acceptMaxIter = Boolean.parseBoolean(
                System.getProperty("olf.gpu.sa.robustRetryAcceptMaxIter", "true"));
        Map<String, BatchScenarioResult> better = new HashMap<>();
        java.util.Map<Integer, Integer> retriedStatus = new java.util.TreeMap<>();
        int recovered = 0;
        for (BatchScenarioResult r : retried) {
            retriedStatus.merge(r.convergenceStatus(), 1, Integer::sum);
            boolean ok = r.convergenceStatus() == com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchStatus.STATUS_CONVERGED
                    || (acceptMaxIter
                        && r.convergenceStatus() == com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchStatus.STATUS_MAX_ITERATION);
            if (ok) {
                better.put(r.contingencyId(), r);
                recovered++;
            }
        }
        if (System.getenv("OLF_GPU_REJECT_TRACE") != null) {
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

    /** The device's own loss pre-distribution and distributed total for a scenario, where it reported them. */
    /** The zero-impedance branches a contingency takes off: opened couplers, and couplers on a lost bus. */
    private static java.util.List<LfBranch> ziDisabledPreview(LfNetwork net, PropagatedContingency pc) {
        java.util.List<LfBranch> out = new ArrayList<>(pc.getBranchIdsToOpen().keySet().stream()
                .map(net::getBranchById).filter(Objects::nonNull)
                .filter(b -> b.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC)).toList());
        for (String busId : pc.getBusIdsToLose()) {
            com.powsybl.openloadflow.network.LfBus lost = net.getBusById(busId);
            if (lost != null) {
                for (LfBranch inc : lost.getBranches()) {
                    if (inc.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC)) {
                        out.add(inc);
                    }
                }
            }
        }
        return out;
    }

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
        String want = System.getenv("OLF_GPU_ITER_TRACE_CTG");
        if (want != null) {
            for (String w : want.split(",")) {
                if (w.trim().equals(r.contingencyId())) {
                    double pre = r.preDistributedActivePower() * com.powsybl.openloadflow.util.PerUnit.SB;
                    double tot = r.distributedActivePower() * com.powsybl.openloadflow.util.PerUnit.SB;
                    System.err.printf("DAP_CTG ctg=%s preDistributedMW=%.9f totalMW=%.9f ownDsMW=%.9f iterations=%d status=%d%n",
                            r.contingencyId(), pre, tot, tot - pre, r.iterations(), r.convergenceStatus());
                }
            }
        }
    }
}
