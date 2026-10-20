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
import com.powsybl.openloadflow.ac.solver.AcSolverUtil;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonFactory;
import com.powsybl.openloadflow.ac.solver.gpu.GpuDriverOuterLoopEngine;
import com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphsonFactory;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchedSecurityAnalysisSolver;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.BatchContingency;
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
import com.powsybl.openloadflow.sa.LimitReductionManager;
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
 * states the GPU batch can solve (qualified by {@link com.powsybl.openloadflow.ac.solver.gpu.batch.plan.GpuBatchContingencyValidator})
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

    /** When the current run started, for the OLF_GPU_PHASE_TIME split. */
    private static final ThreadLocal<Long> RUN_START_NS = new ThreadLocal<>();

    /** Wall-clock spent in the phases of the GPU SA path OUTSIDE the batched solve, accumulated over
     *  the run's contingencies and printed under {@code OLF_GPU_PHASE_TIME} beside the batch's own
     *  split. Most of a batched run's wall time is not the batch: the engine still applies each
     *  contingency, injects the precomputed state, updates the network and detects violations one
     *  contingency at a time, and that tail is invisible in the batch timings. */
    static final java.util.Map<String, java.util.concurrent.atomic.LongAdder> PHASE_NS =
            new java.util.concurrent.ConcurrentSkipListMap<>();

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
    static final java.util.concurrent.atomic.AtomicInteger LAST_GPU_PRECOMPUTED_COUNT =
            new java.util.concurrent.atomic.AtomicInteger(-1);

    /**
     * The contingency ids the DEVICE solved in the last run. A comparison of the two arms' STATUSES is
     * meaningless for a contingency the device declined: the GPU arm's status for it comes from the CPU
     * fallback, so it is OLF's own verdict compared with OLF's own verdict. Only the ids in here carry a
     * device verdict; for the rest the honest question is the other one — did OLF converge what the device
     * declined? Reset with the counter above.
     */
    public static final java.util.Set<String> LAST_GPU_PRECOMPUTED_IDS =
            java.util.concurrent.ConcurrentHashMap.newKeySet();

    /** How many contingencies of the last run had their loss distribution answered from the batch's replay
     *  instead of OLF's host call ({@link ReplayedLossDistribution}); reset with the counter above. */
    static final java.util.concurrent.atomic.AtomicInteger LAST_REPLAYED_LOSS_COUNT =
            new java.util.concurrent.atomic.AtomicInteger();

    /** How many base-state restores of the last run were targeted (see restoreBaseState). */
    static final java.util.concurrent.atomic.AtomicInteger LAST_TARGETED_RESTORE_COUNT =
            new java.util.concurrent.atomic.AtomicInteger();

    /** Separator inside the precomputed-state key of a (contingency, strategy) pair — the SOLVER's
     *  constant, which also drives its two-phase warm-continuation ordering. */
    private static final char STRATEGY_KEY_SEP = BatchContingency.STRATEGY_KEY_SEP;

    /**
     * OLF's contingency loss distribution, answered from the batch where that is exact. For a device-solved
     * contingency the device state already embodies the distribution (the batch replays it to seed the
     * solve), and all the host call still contributes is its RETURN value — the amount every
     * PostContingencyResult reports — which the replay carries too (6e-14 pu worst on rte6515). Running it
     * anyway rescanned every bus per contingency and changed participating generators' target P, which
     * the next base-state restore then had to set back through the network listeners.
     *
     * <p>OLF's call still runs for a contingency: not solved on the device (CPU route); without a replayed
     * value; with contingency-specific load flow parameters (the replay used the run's); or targeted by any
     * operator strategy, because a strategy condition on an injection's ACTIVE_POWER reads the target P it sets
     * ({@code ThresholdConditionEvaluator}; the batch's own strategy composition reads IIDM base targets and does
     * not depend on it). The report OLF adds is reproduced.
     */
    private final class ReplayedLossDistribution implements ContingencyActivePowerLossDistribution {

        private final ContingencyActivePowerLossDistribution delegate;

        private ReplayedLossDistribution(ContingencyActivePowerLossDistribution delegate) {
            this.delegate = delegate;
        }

        @Override
        public String getName() {
            return delegate.getName();
        }

        @Override
        public double run(LfNetwork network, com.powsybl.openloadflow.network.LfContingency lfContingency,
                          com.powsybl.contingency.Contingency contingency, SecurityAnalysisParameters securityAnalysisParameters,
                          com.powsybl.openloadflow.LoadFlowParametersOverride loadFlowParametersOverride,
                          com.powsybl.commons.report.ReportNode reportNode) {
            GpuSaRunState rs = runState.get();
            Double replayed = rs.precomputed.states().containsKey(contingency.getId())
                    ? rs.precomputed.preDistributedActivePower().get(contingency.getId())
                    : null;
            if (replayed == null || loadFlowParametersOverride != null
                    || targetedByOperatorStrategy(rs.operatorStrategies, contingency.getId())) {
                return delegate.run(network, lfContingency, contingency, securityAnalysisParameters,
                        loadFlowParametersOverride, reportNode);
            }
            double mismatch = lfContingency.getActivePowerLoss();
            if (Math.abs(mismatch) > 0) {                    // a replayed value exists only with distributed slack on
                com.powsybl.openloadflow.util.Reports.reportContingencyActivePowerLossDistribution(reportNode,
                        mismatch * PerUnit.SB, (mismatch - replayed) * PerUnit.SB);
            }
            LAST_REPLAYED_LOSS_COUNT.incrementAndGet();
            rs.lossReplayedFor = contingency.getId();
            return replayed;
        }
    }

    @Override
    protected void restoreBaseState(com.powsybl.openloadflow.network.NetworkState networkState, LfNetwork network,
                                    com.powsybl.openloadflow.network.LfContingency lfContingency) {
        GpuSaRunState rs = runState.get();
        boolean targeted = lfContingency.getId().equals(rs.lossReplayedFor)
                && lfContingency.getId().equals(rs.deviceStateInjectedFor)
                && !(GpuRestoreCheck.ENABLED && !rs.restoreCheck.hasReference(network));   // the check's reference is a full restore
        rs.lossReplayedFor = null;
        rs.deviceStateInjectedFor = null;
        if (targeted) {
            java.util.Set<LfBus> buses = touchedBuses(network, lfContingency, rs);
            // the voltage loop over every bus is redundant when the next contingency is device-answered: its injection
            // writes every live bus (restoreStaleVoltages covers the rest); not under the restore check, whose
            // reference is a full restore
            boolean lazy = !GpuRestoreCheck.ENABLED;
            networkState.restore(buses, lfContingency.getDisabledNetwork().getBranchesStatus().keySet(), !lazy);
            rs.staleVoltages = lazy ? networkState : null;
            LAST_TARGETED_RESTORE_COUNT.incrementAndGet();
        } else {
            rs.staleVoltages = null;                         // the full restore below puts every voltage back
            super.restoreBaseState(networkState, network, lfContingency);
        }
        if (GpuRestoreCheck.ENABLED) {
            GpuRestoreCheck check = runState.get().restoreCheck;
            if (check.hasReference(network)) {
                check.check(network, lfContingency.getId());
            } else {
                check.setReference(network);                // the first restore is a full one
            }
        }
    }

    /** Drops the run's file-backed state map, if its states went to one (GpuStateStore). */
    private static void closeStateStore(GpuSaRunState rs) {
        if (rs != null && rs.precomputed != null && rs.precomputed.states() instanceof GpuStateStore store) {
            store.close();
        }
    }

    /** Ends a lazy voltage restore: {@code onlyDisabled} null restores every bus voltage, otherwise only those buses
     *  (the disabled ones of a device-answered contingency, whose voltages its injection does not write). */
    private static void restoreStaleVoltages(GpuSaRunState rs, java.util.Collection<LfBus> onlyDisabled) {
        if (rs.staleVoltages == null) {
            return;
        }
        if (onlyDisabled == null) {
            rs.staleVoltages.restoreVoltages();
        } else {
            rs.staleVoltages.restoreVoltages(onlyDisabled);
        }
        rs.staleVoltages = null;
    }

    /**
     * The buses a device-solved contingency changed beyond their voltage: what LfContingency.apply touched — the
     * buses it disabled and those of the loads, generators and shunts it lost or shifted — closed over their
     * voltage and reactive-power control groups, because enabling or disabling a controller recomputes its
     * whole group's reactive targets.
     */
    private static java.util.Set<LfBus> touchedBuses(LfNetwork network, com.powsybl.openloadflow.network.LfContingency c,
                                                     GpuSaRunState rs) {
        java.util.Set<LfBus> touched = new java.util.LinkedHashSet<>(c.getDisabledNetwork().getBuses());
        c.getLostLoads().keySet().forEach(l -> touched.add(l.getBus()));
        c.getLostGenerators().forEach(g -> touched.add(g.getBus()));
        if (!c.getShuntsShift().isEmpty()) {
            if (rs.shuntBus == null) {
                rs.shuntBus = new java.util.IdentityHashMap<>();
                for (LfBus b : network.getBuses()) {
                    b.getShunt().ifPresent(sh -> rs.shuntBus.put(sh, b));
                    b.getControllerShunt().ifPresent(sh -> rs.shuntBus.put(sh, b));
                    b.getSvcShunt().ifPresent(sh -> rs.shuntBus.put(sh, b));
                }
            }
            c.getShuntsShift().keySet().forEach(sh -> {
                LfBus b = rs.shuntBus.get(sh);
                if (b != null) {
                    touched.add(b);
                }
            });
        }
        java.util.Set<LfBus> closed = new java.util.LinkedHashSet<>(touched);
        for (LfBus b : touched) {
            b.getGeneratorVoltageControl().ifPresent(vc -> {
                closed.add(vc.getControlledBus());
                closed.addAll(vc.getMergedControllerElements());
            });
            b.getGeneratorReactivePowerControl().ifPresent(rpc -> closed.addAll(rpc.getControllerBuses()));
        }
        return closed;
    }

    /** OLF_GPU_PARTIAL_FLOW_CHECK=1: after each partial-flow detection, detect again with every flow refreshed from
     *  the same state and require the same violations, bit for bit. */
    private static final boolean PARTIAL_FLOW_CHECK = System.getenv("OLF_GPU_PARTIAL_FLOW_CHECK") != null;

    /**
     * The branches whose flows anything reads after a device state is injected for {@code contingencyId}, or null
     * for "refresh them all". With the device violation screen on, the detection reads only the branches of the
     * limits the screen flagged for this contingency and the branches it does not cover (GpuLimitScanner). Every
     * other reader forces the full refresh: a branch or three-windings transformer state monitor (network results),
     * an operator strategy on the contingency (threshold conditions on branch flows), the first contingency (the
     * screen is built during its detection), and the diagnostics that read flows — OLF_GPU_VIOL_CROSSCHECK among them,
     * which runs the stock detection over EVERY branch.
     */
    private int[] flowReadSet(GpuSaRunState rs, String contingencyId) {
        if (rs.violationScreen == null || rs.candidates == null || flowsMonitored()
                || targetedByOperatorStrategy(rs.operatorStrategies, contingencyId)
                || System.getenv("OLF_GPU_Q_PROBE") != null || System.getenv("OLF_GPU_SA_RESID") != null
                || System.getenv("OLF_GPU_VIOL_CROSSCHECK") != null) {
            return null;
        }
        GpuLimitTable table = rs.limitTable;
        java.util.BitSet nums = new java.util.BitSet();
        int[] candidates = rs.candidates.get(contingencyId);
        if (candidates == null) {
            return null;                                     // the detection would screen it on the host: read all
        }
        for (int c : candidates) {
            if (c < table.limitCount()) {                    // bus records come after the limits: they read bus V only
                nums.set(table.branches()[table.groupBranch(table.limitGroup(c))].getNum());
            }
        }
        for (int b : rs.uncoveredBranches) {
            nums.set(table.branches()[b].getNum());
        }
        rs.sideOpened.forEach(branch -> nums.set(branch.getNum()));
        return nums.stream().toArray();
    }

    /** The branches a contingency opened at one side only: still in the network, but in a topology the screen — built
     *  on the base — does not evaluate (it would read them closed). */
    private static List<LfBranch> sideOpenedBranches(LfContingency lfContingency) {
        List<LfBranch> sideOpened = null;
        for (var e : lfContingency.getDisabledNetwork().getBranchesStatus().entrySet()) {
            if (e.getValue() != com.powsybl.openloadflow.network.DisabledBranchStatus.BOTH_SIDES) {
                if (sideOpened == null) {
                    sideOpened = new java.util.ArrayList<>();
                }
                sideOpened.add(e.getKey());
            }
        }
        return sideOpened != null ? sideOpened : List.of();
    }

    /** The current contingency's side-opened branches as limit-table indices (null when none has limits): their
     *  screen records are dropped and the host scans them in full, in table order. */
    private static java.util.BitSet sideOpenedTableIndices(GpuSaRunState rs, LfNetwork network) {
        if (rs.sideOpened.isEmpty()) {
            return null;
        }
        GpuLimitTable table = rs.limitTable;
        if (rs.tableIndexOfBranchNum == null) {
            rs.tableIndexOfBranchNum = new int[network.getBranches().size()];
            java.util.Arrays.fill(rs.tableIndexOfBranchNum, -1);
            for (int b = 0; b < table.branches().length; b++) {
                rs.tableIndexOfBranchNum[table.branches()[b].getNum()] = b;
            }
        }
        java.util.BitSet sideOpened = new java.util.BitSet();
        for (LfBranch branch : rs.sideOpened) {
            int b = rs.tableIndexOfBranchNum[branch.getNum()];
            if (b >= 0 && table.groupStart(b) != table.groupEnd(b)) {
                sideOpened.set(b);
            }
        }
        return sideOpened.isEmpty() ? null : sideOpened;
    }

    /** True when a state monitor asks for branch or three-windings transformer results, which read flows. */
    private boolean flowsMonitored() {
        java.util.List<com.powsybl.security.monitor.StateMonitor> monitors = new java.util.ArrayList<>();
        monitors.add(monitorIndex.getAllStateMonitor());
        monitors.add(monitorIndex.getNoneStateMonitor());
        monitors.addAll(monitorIndex.getSpecificStateMonitors().values());
        return monitors.stream().anyMatch(m -> m != null
                && (!m.getBranchIds().isEmpty() || !m.getThreeWindingsTransformerIds().isEmpty()));
    }

    private static boolean targetedByOperatorStrategy(List<OperatorStrategy> strategies, String contingencyId) {
        for (OperatorStrategy strategy : strategies) {
            if (targets(strategy, contingencyId)) {
                return true;
            }
        }
        return false;
    }

    private static boolean targets(OperatorStrategy strategy, String contingencyId) {
        var ctx = strategy.getContingencyContext();
        return switch (ctx.getContextType()) {
            case ALL, ONLY_CONTINGENCIES -> true;
            case SPECIFIC -> contingencyId.equals(ctx.getContingencyId());
            case NONE -> false;
        };
    }

    /**
     * What a strategy pair's device answer reports as distributed active power, or null when the pair is not answered
     * on the device: no state, the batch not ready, or its distribution not recorded under distributed slack. OLF
     * reports the distribution of the strategy load flow alone (no loss pre-distribution, counted from the parent's
     * distributed targets); the device records it when one solve covered the pair's chunk.
     */
    private static Double pairDistributedOnDevice(PrecomputedBatch precomputed, String pairId,
                                                  SecurityAnalysisParameters securityAnalysisParameters) {
        if (!precomputed.isReady() || precomputed.states().get(pairId) == null) {
            return null;
        }
        return securityAnalysisParameters.getLoadFlowParameters().isDistributedSlack()
                ? precomputed.distributedActivePower().get(pairId) : Double.valueOf(0.0);
    }

    /**
     * A strategy pair the device does not answer runs OLF's strategy load flow from the network its parent left. The
     * parent's injected device state holds the bus voltages only, not what OLF's post-contingency run leaves behind
     * (the generators' distributed targets, the PV/PQ switches, tap moves...), so such a parent is solved on the CPU.
     * Measured: an absolute GeneratorAction pair routed to the CPU after a device parent reported -43.62 MW of
     * distributed power where OLF reports -42.67.
     */
    private static boolean everyStrategyAnsweredOnDevice(GpuSaRunState rs, String contingencyId,
                                                         SecurityAnalysisParameters securityAnalysisParameters) {
        for (OperatorStrategy strategy : rs.operatorStrategies) {
            if (!targets(strategy, contingencyId)) {
                continue;
            }
            // whichever of its action lists OLF ends up applying (the stages' conditions are evaluated later, on the
            // post-contingency violations), the device must hold it
            List<List<String>> candidates = GpuStrategyComposer.candidateActionLists(strategy);
            if (candidates.isEmpty()) {
                return false;
            }
            for (List<String> actionIds : candidates) {
                if (pairDistributedOnDevice(rs.precomputed, GpuStrategyComposer.pairId(contingencyId, strategy, actionIds),
                        securityAnalysisParameters) == null) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * {@code OLF_GPU_PRELOSS_CHECK=1}: runs OLF's own contingency loss distribution UNCHANGED and compares,
     * per device-solved contingency, the amount it returns with the one the batch replayed
     * ({@link PrecomputedBatch#preDistributedActivePower()}). Measures whether the replay could stand in
     * for the host call (assessment lead L2) before anything relies on it.
     */
    private final class LossDistributionCheck implements ContingencyActivePowerLossDistribution {

        private final ContingencyActivePowerLossDistribution delegate;
        private long compared;
        private long differing;
        private long missing;
        private double worst;
        private String worstId = "-";
        private final java.util.List<String> samples = new java.util.ArrayList<>();

        private LossDistributionCheck(ContingencyActivePowerLossDistribution delegate) {
            this.delegate = delegate;
        }

        @Override
        public String getName() {
            return delegate.getName();
        }

        @Override
        public double run(LfNetwork network, com.powsybl.openloadflow.network.LfContingency lfContingency,
                          com.powsybl.contingency.Contingency contingency, SecurityAnalysisParameters securityAnalysisParameters,
                          com.powsybl.openloadflow.LoadFlowParametersOverride loadFlowParametersOverride,
                          com.powsybl.commons.report.ReportNode reportNode) {
            double olf = delegate.run(network, lfContingency, contingency, securityAnalysisParameters,
                    loadFlowParametersOverride, reportNode);
            PrecomputedBatch precomputed = runState.get().precomputed;
            if (precomputed.states().containsKey(contingency.getId())) {
                Double replayed = precomputed.preDistributedActivePower().get(contingency.getId());
                if (replayed == null) {
                    missing++;
                } else {
                    compared++;
                    double d = Math.abs(olf - replayed);
                    if (d != 0) {
                        differing++;
                        if (samples.size() < 5) {
                            samples.add(String.format("%s olf=%s replayed=%s pu", contingency.getId(),
                                    Double.toString(olf), Double.toString(replayed)));
                        }
                    }
                    if (d > worst) {
                        worst = d;
                        worstId = contingency.getId();
                    }
                }
            }
            return olf;
        }

        private void print() {
            System.err.printf("PRELOSS_CHECK device-solved contingencies compared=%d bit-identical=%d differing=%d "
                    + "no-replayed-value=%d | worst |olf - replayed| = %.3e pu (%.3e MW) on %s%n",
                    compared, compared - differing, differing, missing, worst, worst * PerUnit.SB, worstId);
            samples.forEach(x -> System.err.println("PRELOSS_CHECK   " + x));
        }
    }

    /** The per-RUN mutable state, one holder PER PARTITION THREAD. Under threadCount > 1 OLF invokes
     *  {@code runSimulations} CONCURRENTLY on this same analysis instance (one call per contingency
     *  partition, each with its own LfNetwork), and every hook this class overrides runs on the
     *  partition's own thread inside that call stack — so the state is ThreadLocal, never instance
     *  fields. As INSTANCE fields these raced: partition A's precompute could read partition B's
     *  lfNetwork/context and LOAD STATES INTO B's equation system while B was mid-solve — one
     *  corrupted scenario (the 19/1 MAX_ITERATION flake) or an NPE inside B's JacobianMatrix on
     *  pure CPU stacks (both observed, threadCountAboveOneRunsSafelyOnGpuProvider). */
    private final ThreadLocal<GpuSaRunState> runState = ThreadLocal.withInitial(GpuSaRunState::new);

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
        GpuSaRunState rs = new GpuSaRunState();
        runState.set(rs);
        rs.loadFlowParameters = securityAnalysisParameters.getLoadFlowParameters();
        rs.securityAnalysisParameters = securityAnalysisParameters;
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
        if (!scope.batchDisabled() && !multiThread
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
        // solver.
        // When the batch is disabled by the outer-loop guard, ALSO skip the solver swap: the provider then
        // behaves exactly like the stock CPU AcSecurityAnalysis (the safety contract), and the sequential
        // per-contingency runs avoid the GPU-inner-solver + outer-loop-switching path (open CME bug there).
        if (!scope.batchDisabled()
                && GpuAcNewtonSolver.isAvailable() && GpuSaRunScope.gpuSupports(lfNetwork, acParameters)) {
            acParameters.setSolverFactory(new GpuNewtonRaphsonFactory(), securityAnalysisParameters.getLoadFlowParameters());
            LOGGER.info("GPU batched SA: base + fall-through load flows run on the GPU Newton-Raphson (CUDA pre-warm)");
        }
        List<PropagatedContingency> toSimulate = propagatedContingencies;
        ContingencyActivePowerLossDistribution lossDistribution;
        if (System.getenv("OLF_GPU_PRELOSS_CHECK") != null) {
            lossDistribution = new LossDistributionCheck(contingencyActivePowerLossDistribution);
        } else {
            lossDistribution = new ReplayedLossDistribution(contingencyActivePowerLossDistribution);
        }
        try {
            return super.runSimulations(lfNetwork, toSimulate, acParameters, securityAnalysisParameters,
                    operatorStrategies, actions, limitReductions, lossDistribution);
        } finally {
            restoreStaleVoltages(runState.get(), null);      // the network leaves the loop in its base state
            if (lossDistribution instanceof LossDistributionCheck check) {
                check.print();
            }
            if (PARTIAL_FLOW_CHECK) {
                runState.get().partialFlowCheck.print();
            }
            if (GpuPostContingencyState.BUS_ARRAY_CHECK) {
                GpuSaRunState check = runState.get();
                System.err.printf("BUS_ARRAY_CHECK contingencies=%d identical=%d differing=%d | array path usable=%b | first difference: %s%n",
                        check.busArrayChecked, check.busArrayChecked - check.busArrayDiffering, check.busArrayDiffering,
                        check.busRowsUsable, check.busArrayFirstDiff == null ? "-" : check.busArrayFirstDiff);
            }
            if (GpuRestoreCheck.ENABLED) {
                runState.get().restoreCheck.print();
                System.err.println("RESTORE_CHECK targeted restores (cumulative this run): " + LAST_TARGETED_RESTORE_COUNT.get());
            }
            closeStateStore(runState.get());                 // a file-backed state map is dropped with the run
            runState.remove();                               // pooled partition threads must not leak a run's state
            if (System.getenv("OLF_GPU_PHASE_TIME") != null) {
                // CUMULATIVE over the JVM, never cleared: threadCount > 1 runs one partition per
                // thread through this same static map, and clearing here would hand whichever
                // partition finished first the whole run's numbers and the rest almost nothing.
                // Read the LAST block printed — like JNI_PHASE_TIME, it is the running total.
                PHASE_NS.forEach((k, v) ->
                        System.err.printf("SA_PHASE_TIME  %-40s %6d ms (cumulative)%n", k, v.sum() / 1_000_000));
            }
            if (System.getenv("OLF_GPU_VIOL_CROSSCHECK") != null) {
                System.err.println("VIOL_CROSSCHECK contingencies compared against the stock detection: "
                        + GpuSaDiagnostics.CROSSCHECKED.sum());
            }
        }
    }

    /** Every load flow of the analysis runs on the device driver's engine — the base as a batch of one,
     *  the same driver as the chunks, which keeps the two on one trajectory. A solve with OLF's CPU
     *  Newton-Raphson underneath (the fall-through contingencies) leaves the loops to OLF's own code. */
    @Override
    protected com.powsybl.openloadflow.ac.AcloadFlowEngine createLoadFlowEngine(
            com.powsybl.openloadflow.ac.AcLoadFlowContext context) {
        return new GpuDriverOuterLoopEngine(context);
    }

    /** The branches the screen cannot evaluate (zero-impedance, or absent from the packed model):
     *  they keep the full host scan, so coverage is a performance property and never a correctness one. */
    private static int[] uncoveredBranches(GpuLimitTable table, GpuViolationScreen screen) {
        java.util.List<Integer> uncovered = new java.util.ArrayList<>();
        for (int b = 0; b < table.branches().length; b++) {
            if (table.groupStart(b) != table.groupEnd(b) && !screen.covers(b)) {
                uncovered.add(b);
            }
        }
        if (System.getenv("OLF_GPU_PHASE_TIME") != null) {
            System.err.println("VIOL_SCREEN branches with limits not covered by the screen: " + uncovered.size());
        }
        return uncovered.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * Detect the post-contingency violations of a device-solved contingency.
     *
     * <p>The scan is the largest block of GPU security analysis wall time that is not a device solve
     * (rte6515 at 2000 contingencies, 1 thread: 10.6 s of 39.3 s), and 98.4% of it is the walk
     * rather than the reporting — so it is worth doing over flat arrays even before it moves to the
     * device. The table is built once per run and cached on the partition's GpuSaRunState.
     */
    private void detectViolations(LfNetwork network, LimitViolationManager manager, List<LimitReduction> limitReductions,
                                  LimitViolationManager reference, SecurityAnalysisParameters saParameters,
                                  String contingencyId) {
        GpuSaRunState rs = runState.get();
        if (rs.limitScanner == null) {
            GpuLimitTable table = GpuLimitTable.build(network, LimitReductionManager.create(limitReductions));
            rs.limitScanner = new GpuLimitScanner(table);
            rs.limitTable = table;
            if (rs.precomputed.gpuData() != null) {
                rs.violationScreen = GpuViolationScreen.build(network, table, rs.precomputed.gpuData(),
                        rs.precomputed.elementMapper(), rs.precomputed.equationSystem(),
                        rs.precomputed.baseVariableRows());
                rs.uncoveredBranches = uncoveredBranches(table, rs.violationScreen);
                // Screen the WHOLE run at once: the states are all precomputed already, and one
                // device call per 512 scenarios is what makes the kernel worth its upload.
                long tScreen = System.nanoTime();
                rs.candidates = rs.violationScreen.screenAll(rs.precomputed.states(),
                        rs.precomputed.baseVariableRows().size());
                PHASE_NS.computeIfAbsent("  violation screen (all scenarios)",
                        k -> new java.util.concurrent.atomic.LongAdder()).add(System.nanoTime() - tScreen);
                if (System.getenv("OLF_GPU_PHASE_TIME") != null) {
                    System.err.println("VIOL_SCREEN scenarios=" + rs.candidates.size()
                            + " screened on the device=" + rs.violationScreen.onDevice()
                            + " overflowed to the host scan=" + rs.violationScreen.overflows()
                            + " in " + (System.nanoTime() - tScreen) / 1_000_000 + " ms");
                }
            }
        }
        java.util.BitSet sideOpened = rs.violationScreen != null ? sideOpenedTableIndices(rs, network) : null;
        if (rs.violationScreen != null) {
            int[] candidates = rs.candidates.get(contingencyId);
            rs.limitScanner.detectFromCandidates(network, manager,
                    candidates != null ? candidates : rs.violationScreen.screen(rs.precomputed.states().get(contingencyId)),
                    rs.violationScreen, rs.uncoveredBranches, sideOpened);
        } else {
            rs.limitScanner.detectViolations(network, manager);
        }
        if (System.getenv("OLF_GPU_VIOL_CROSSCHECK") != null) {
            GpuSaDiagnostics.crossCheck(network, manager, limitReductions, reference, saParameters, contingencyId);
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
        AcLoadFlowContext context = super.createLoadFlowContext(lfNetwork, parameters);
        runState.get().context = context;
        return context;
    }

    @Override
    protected void afterPreContingencySimulation(AcLoadFlowParameters parameters) {
        super.afterPreContingencySimulation(parameters);
        applyContingencyOuterLoops(parameters);
        Long t0 = RUN_START_NS.get();
        if (t0 != null) {
            PHASE_NS.computeIfAbsent("base load flow + setup", k -> new java.util.concurrent.atomic.LongAdder())
                    .add(System.nanoTime() - t0);
        }
        GpuSaRunState rs = runState.get();
        if (System.getenv("OLF_GPU_Q_PROBE") != null && rs.lfNetwork != null) {
            String qp = System.getenv("OLF_GPU_Q_PROBE");
            for (LfBus b : rs.lfNetwork.getBuses()) {
                if (java.util.Arrays.stream(qp.split(",")).anyMatch(b.getId()::startsWith)) {
                    System.err.printf("Q_PROBE_BASE bus=%s q=%.9f maxQ=%.9f busTargetP=%.9f%n",
                            b.getId(), b.getQ().eval() + b.getLoadTargetQ(), b.getMaxQ(), b.getTargetP());
                    // The generator-voltage-control state AT THE INSTANT THE PRECOMPUTE CAPTURES, which is
                    // what decides whether re-extracting here could fix anything (tasks #66): the device's
                    // last base extraction read these buses FROZEN (qLimit=MIN_REALISTIC_V) while OLF's
                    // converged base has their DISTR_Q active. If they read RELEASED here, the base LF
                    // simply ended after its last extraction and a fresh extraction is the whole fix; if
                    // they read FROZEN, the base LF itself diverged and the fix is not capture timing.
                    // The TWO SIDES of ReactiveLimitsOuterLoop's switch-back test, at FULL precision and
                    // as that method computes them, because a claim about which way a `<` resolves is
                    // worthless without both numbers: `getBusV(bus)` is the CONTROLLED bus's v (not this
                    // bus's) and `getBusTargetV(bus)` the control's target. ctlV < tgtV (strict) with a
                    // MIN-kind qLimit is what switches a frozen bus back to PV.
                    var vcp = b.getGeneratorVoltageControl().orElse(null);
                    System.err.printf("Q_PROBE_BASE_VC bus=%s vcEnabled=%b frozenQ=%b qLimit=%s"
                            + " ownV=%.17g ctlBus=%s ctlV=%.17g tgtV=%.17g ctlV<tgtV=%b d=%.3e%n",
                            b.getId(), b.isGeneratorVoltageControlEnabled(), b.isGenerationTargetQFrozen(),
                            b.getQLimitType().map(Object::toString).orElse("-"), b.getV(),
                            vcp == null ? "-" : vcp.getControlledBus().getId(),
                            vcp == null ? Double.NaN : vcp.getControlledBus().getV(),
                            vcp == null ? Double.NaN : vcp.getTargetValue(),
                            vcp != null && vcp.getControlledBus().getV() < vcp.getTargetValue(),
                            vcp == null ? Double.NaN : vcp.getControlledBus().getV() - vcp.getTargetValue());
                    // the BASE generator P: the reference every arm's slack distribution moves from
                    for (var g : b.getGenerators()) {
                        System.err.printf("Q_PROBE_BASE_GEN bus=%s gen=%s targetP=%.9f minP=%.9f maxP=%.9f participating=%b%n",
                                b.getId(), g.getId(), g.getTargetP(), g.getMinP(), g.getMaxP(), g.isParticipating());
                    }
                }
            }
        }
        long tPre = System.nanoTime();
        closeStateStore(rs);                                 // a previous batch's file-backed states
        rs.precomputed = GpuBatchPrecomputer.precompute(network, rs.lfNetwork, rs.context,
                rs.contingencies, rs.operatorStrategies, rs.actions, rs.batchDistributedSlack,
                rs.batchReactiveLimits, rs.batchReactiveLimitTol, rs.batchSlackDistTol, rs.batchConvEpsPerEq,
                rs.batchConfig, reportNode);
        PHASE_NS.computeIfAbsent("precompute (extraction + batched solves)", k -> new java.util.concurrent.atomic.LongAdder())
                .add(System.nanoTime() - tPre);
        if (rs.precomputed.isReady()) {
            int solved = rs.precomputed.states().size();
            LAST_GPU_PRECOMPUTED_COUNT.updateAndGet(prev -> Math.max(prev, 0) + solved);
            LAST_GPU_PRECOMPUTED_IDS.addAll(rs.precomputed.states().keySet());
        }
        // The engine's solver reverts to the stock CPU Newton-Raphson after the base LF: single-system GPU
        // solves are not where the device wins (its parallelism is the batch), and the CPU path parallelizes
        // under threadCount. The base LF has already run at this point (on the GPU, absorbing CUDA init).
        if (parameters.getSolverFactory() instanceof GpuNewtonRaphsonFactory && rs.loadFlowParameters != null) {
            parameters.setSolverFactory(new NewtonRaphsonFactory(), rs.loadFlowParameters);
        }
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
        restoreStaleVoltages(runState.get(), null);         // an action simulation reads the network as it stands
        // the pair holding the actions OLF's condition check selected
        String scId = GpuStrategyComposer.pairId(contingency.getId(), operatorStrategy, actionsIds);
        PrecomputedBatch precomputed = runState.get().precomputed;
        double[] state = precomputed.states().get(scId);
        Double distributed = pairDistributedOnDevice(precomputed, scId, securityAnalysisParameters);
        if (distributed == null) {
            // pair not precomputed (condition not TrueCondition / out-of-scope action / diverged) or its
            // distributed active power not recorded — CPU path
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
        return new OperatorStrategyResult(operatorStrategy,
                List.of(new OperatorStrategyResult.ConditionalActionsResult(operatorStrategy.getId(),
                        PostContingencyComputationStatus.CONVERGED,
                        new LimitViolationsResult(postActionsViolationManager.getLimitViolations()),
                        GpuStateInjector.filterIslanded(network, precomputed.islandedBuses().get(scId), scId,
                                new NetworkResult(postActionsNetworkResult.getBranchResults(), postActionsNetworkResult.getBusResults(),
                                        postActionsNetworkResult.getThreeWindingsTransformerResults())),
                        distributed * PerUnit.SB)));
    }

    @Override
    protected PostContingencyResult runPostContingencySimulation(LfNetwork network, AcLoadFlowContext context, Contingency contingency,
                                                                 LfContingency lfContingency, LimitViolationManager preContingencyLimitViolationManager,
                                                                 SecurityAnalysisParameters securityAnalysisParameters,
                                                                 PreContingencyNetworkResult preContingencyNetworkResult, boolean createResultExtension,
                                                                 List<LimitReduction> limitReductions, double preDistributedActivePower) {
        GpuSaRunState rsv = runState.get();
        if (rsv.staleVoltages != null) {
            PrecomputedBatch pc = rsv.precomputed;
            // the same test as the injection below: a contingency with a device state still runs on OLF's own path
            // when one of its strategies is not answered on the device, and OLF then starts from every bus's voltage
            // (rte6515 LINE-2349-694 after LINE-2348-693: started from the previous injection's voltages, 1.44 MW off)
            boolean devicePath = pc.isReady() && !pc.exhausted().contains(contingency.getId())
                    && pc.states().get(contingency.getId()) != null
                    && everyStrategyAnsweredOnDevice(rsv, contingency.getId(), securityAnalysisParameters);
            // device-answered: the injection writes every live bus, so only the disabled ones need their base back
            restoreStaleVoltages(rsv, devicePath ? lfContingency.getDisabledNetwork().getBuses() : null);
        }
        GpuSaDiagnostics.dumpPostContingencyEquations(context, contingency, lfContingency, network);
        // OLF_GPU_ZI_TREE_CTG=1: the zero-impedance branches OLF has OUT of its spanning tree for THIS
        // contingency, right after lfContingency.apply(). OLF's tree is history-dependent within a
        // security analysis - removeBranchAndTryToSplit marks a disabled tree edge non-tree and
        // re-elects, addBranch never restores it - so this is the ground truth a device-side replay of
        // that history has to reproduce, contingency by contingency.
        if (System.getenv("OLF_GPU_ZI_TREE_CTG") != null) {
            System.err.println("ZI_TREE_CTG ctg=" + contingency.getId() + " nonTree="
                    + network.getBranches().stream()
                            .filter(b -> b.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC))
                            .filter(b -> !b.isSpanningTreeEdge(com.powsybl.openloadflow.network.LoadFlowModel.AC))
                            .map(com.powsybl.openloadflow.network.LfBranch::getId).sorted().toList());
        }
        PrecomputedBatch precomputed = runState.get().precomputed;
        if (precomputed.isReady() && precomputed.exhausted().contains(contingency.getId())
                && everyStrategyAnsweredOnDevice(runState.get(), contingency.getId(), securityAnalysisParameters)) {
            // The device carried this scenario through OLF's whole Newton budget without
            // convergence. OLF's own path (AbstractSecurityAnalysis.runPostContingencySimulation)
            // reports MAX_ITERATION_REACHED with an empty network result and no violations for
            // that outcome; the same result is built here, from the contingency already applied.
            // A strategy of it the device did not answer starts from what OLF's failed run leaves on the network
            // (its targets, its bus types), which only that run produces: the contingency then runs on the CPU.
            logPostContingencyStart(network, lfContingency);
            return new PostContingencyResult(contingency, PostContingencyComputationStatus.MAX_ITERATION_REACHED,
                    new LimitViolationsResult(List.of()),
                    new NetworkResult(List.of(), List.of(), List.of()),
                    new ConnectivityResult(lfContingency.getCreatedSynchronousComponentsCount(), 0,
                            lfContingency.getDisconnectedLoadActivePower() * PerUnit.SB,
                            lfContingency.getDisconnectedGenerationActivePower() * PerUnit.SB,
                            lfContingency.getDisconnectedElementIds()),
                    // what the device's own slack passes placed before the budget ran out, as OLF reports for its
                    // own failed run; the pre-distribution alone where the device did not report a total
                    precomputed.distributedActivePower()
                            .getOrDefault(contingency.getId(), preDistributedActivePower) * PerUnit.SB);
        }
        double[] state = precomputed.states().get(contingency.getId());
        if (state == null || !precomputed.isReady()
                || !everyStrategyAnsweredOnDevice(runState.get(), contingency.getId(), securityAnalysisParameters)) {
            // not batched (multi-element / islanding / out of scope / rejected), or a strategy pair of it is left to
            // the CPU (everyStrategyAnsweredOnDevice) — standard CPU path
            long tCpu = System.nanoTime();
            PostContingencyResult cpuResult = super.runPostContingencySimulation(network, context, contingency, lfContingency,
                    preContingencyLimitViolationManager, securityAnalysisParameters, preContingencyNetworkResult,
                    createResultExtension, limitReductions, preDistributedActivePower);
            // The CPU replay's own cost (OLF's whole outer-loop run for this contingency), apart from the
            // device-answered ones; the count is printed in the "ms" column like the remap counter.
            PHASE_NS.computeIfAbsent("   CPU replay: OLF solve + results", k -> new java.util.concurrent.atomic.LongAdder())
                    .add(System.nanoTime() - tCpu);
            PHASE_NS.computeIfAbsent("   CPU replay: contingencies (count)", k -> new java.util.concurrent.atomic.LongAdder())
                    .add(1_000_000);
            return cpuResult;
        }

        GpuSaDiagnostics.dumpDeviceStateVariables(state, precomputed, network, contingency);

        // Inject the precomputed post-contingency state (the branch is already disabled by
        // lfContingency.apply()), then reuse the engine's result-building unchanged.
        long tPost = System.nanoTime();
        logPostContingencyStart(network, lfContingency);
        Stopwatch stopwatch = Stopwatch.createStarted();

        // What OLF reports on the result: the pre-distribution PLUS the load flow's own distributed-slack passes
        // (AbstractSecurityAnalysis: preDistributedActivePower + result.getDistributedActivePower()). The device
        // total covers both; where it is unknown the pre-distribution alone is reported, as before.
        double distributedActivePower = precomputed.distributedActivePower()
                .getOrDefault(contingency.getId(), preDistributedActivePower);
        long tInj = System.nanoTime();
        runState.get().deviceStateInjectedFor = contingency.getId();
        runState.get().sideOpened = sideOpenedBranches(lfContingency);
        int[] flowReadSet = flowReadSet(runState.get(), contingency.getId());
        GpuStateInjector.injectState(precomputed.equationSystem(), precomputed.baseVariableRows(), state,
                runState.get().remap, flowReadSet);
        long tUpd = System.nanoTime();
        PHASE_NS.computeIfAbsent("   injectState()", k -> new java.util.concurrent.atomic.LongAdder())
                .add(tUpd - tInj);
        // OLF_GPU_SA_RESID=<n>: OLF's OWN residual at the state the device returned, on OLF's
        // CONTINGENCY-updated equation system (lfContingency.apply() ran above). The batched path's
        // analogue of OLF_GPU_RESIDUAL_TRACE: a large value on a row means the device solved a
        // DIFFERENT equation there — a row-assembly bug — not that it failed to converge. Prints the
        // n worst rows with the equation OLF has on each, which names the family to look at.
        if (System.getenv("OLF_GPU_SA_RESID") != null) {
            GpuSaDiagnostics.reportOlfResidual(context, precomputed.equationSystem(), contingency.getId(),
                    Integer.parseInt(System.getenv("OLF_GPU_SA_RESID")), precomputed.baseVariableRows(),
                    precomputed.islandedBuses().get(contingency.getId()));
        }
        boolean unrealistic = GpuPostContingencyState.updateBusesFromState(state, precomputed, network, context.getParameters(), runState.get());
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
                            .map(e -> String.format("%.9f", e.eval() - GpuPostContingencyState.distrQTarget(network, b)))
                            .orElse("n/a");
                    System.err.printf("Q_PROBE ctg=%s bus=%s q=%.9f genTgtQ=%.9f loadTgtQ=%.9f qPct=%.9f "
                            + "distrQResid=%s minQ=%.9f maxQ=%.9f overMaxBy=%.9f%n",
                            contingency.getId(), b.getId(), q, b.getGenerationTargetQ(), b.getLoadTargetQ(),
                            b.getRemoteControlReactivePercent(), resid, b.getMinQ(), b.getMaxQ(), q - b.getMaxQ());
                    // Every term of OLF's BUS_TARGET_Q equation at this state (value + variables): the
                    // device's row must carry the same terms, or a pin lands at a different q (a
                    // zero-impedance dummy, a load-model part, a shunt the pack routes elsewhere).
                    precomputed.equationSystem().getEquation(b.getNum(), AcEquationType.BUS_TARGET_Q).ifPresent(eq -> {
                        for (var t : eq.getTerms()) {
                            System.err.printf("Q_PROBE_TERM ctg=%s bus=%s active=%b term=%s value=%.9f vars=%s%n",
                                    contingency.getId(), b.getId(), t.isActive(), t.getClass().getSimpleName(),
                                    t.isActive() ? t.eval() : Double.NaN,
                                    t.getVariables().stream().map(v -> v.getType() + "#" + v.getElementNum()).toList());
                        }
                    });
                    // Every incident branch: OLF's pi model next to the row the device pack carries for
                    // it (matched by the two phi columns) — a q that lands 0.2 MVar off its pin with the
                    // same terms on both arms is a pi-model disagreement on one of these.
                    var gd = precomputed.gpuData();
                    if (gd != null) {
                        int cbCount = gd.closedBranch().parameters().length / 8;
                        int idxStride = cbCount > 0 ? gd.closedBranch().indices().length / cbCount : 0;
                        for (var br : b.getBranches()) {
                            var pm = br.getPiModel();
                            Integer ph1 = br.getBus1() == null ? null : precomputed.baseVariableRows().get(precomputed.equationSystem().getVariableSet()
                                    .getVariable(br.getBus1().getNum(), AcVariableType.BUS_PHI));
                            Integer ph2 = br.getBus2() == null ? null : precomputed.baseVariableRows().get(precomputed.equationSystem().getVariableSet()
                                    .getVariable(br.getBus2().getNum(), AcVariableType.BUS_PHI));
                            String pack = "absent";
                            for (int e = 0; e < cbCount && idxStride > 0; e++) {
                                if (ph1 != null && ph2 != null && gd.closedBranch().indices()[e * idxStride] == ph1 && gd.closedBranch().indices()[e * idxStride + 2] == ph2) {
                                    double[] r = java.util.Arrays.copyOfRange(gd.closedBranch().parameters(), e * 8, e * 8 + 8);
                                    pack = String.format("cb=%d y=%.9f g1=%.9f g2=%.9f b1=%.9f b2=%.9f ksi=%.9f r1=%.9f a1=%.9f",
                                            e, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
                                }
                            }
                            System.err.printf("Q_PROBE_BRANCH ctg=%s bus=%s branch=%s num=%d disabled=%b sides=%b/%b olf: y=%.9f g1=%.9f g2=%.9f b1=%.9f b2=%.9f ksi=%.9f r1=%.9f a1=%.9f | pack: %s%n",
                                    contingency.getId(), b.getId(), br.getId(), br.getNum(), br.isDisabled(), br.isConnectedSide1(), br.isConnectedSide2(),
                                    pm.getY(), pm.getG1(), pm.getG2(), pm.getB1(), pm.getB2(), pm.getKsi(), pm.getR1(), pm.getA1(), pack);
                        }
                    }
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
        PostContingencyComputationStatus status = unrealistic
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
                    distributedActivePower * PerUnit.SB);
        }
        long tRes = System.nanoTime();
        postContingencyNetworkResult.update();
        long tDet = System.nanoTime();
        PHASE_NS.computeIfAbsent("   networkResult.update()", k -> new java.util.concurrent.atomic.LongAdder())
                .add(tDet - tRes);
        detectViolations(network, postContingencyLimitViolationManager, limitReductions,
                preContingencyLimitViolationManager, securityAnalysisParameters, contingency.getId());
        PHASE_NS.computeIfAbsent("   detectViolations()", k -> new java.util.concurrent.atomic.LongAdder())
                .add(System.nanoTime() - tDet);
        if (flowReadSet != null && PARTIAL_FLOW_CHECK) {
            // Same state, every flow refreshed, the detection again: any branch the partial refresh left stale
            // and the detection read shows up as a differing violation.
            var es = precomputed.equationSystem();
            es.getStateVector().setValuesOnly(es.getStateVector().get().clone());
            var fullManager = new LimitViolationManager(preContingencyLimitViolationManager, limitReductions,
                    securityAnalysisParameters.getIncreasedViolationsParameters());
            detectViolations(network, fullManager, limitReductions, preContingencyLimitViolationManager,
                    securityAnalysisParameters, contingency.getId());
            runState.get().partialFlowCheck.compare(contingency.getId(), postContingencyLimitViolationManager.getLimitViolations(),
                    fullManager.getLimitViolations(), flowReadSet.length);
        }
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
                connectivityResult, distributedActivePower * PerUnit.SB);
    }

    /**
     * Give the precompute the CONTINGENCY outer-loop list, not the pre-contingency one.
     *
     * <p>{@code AbstractSecurityAnalysis} calls {@code afterPreContingencySimulation} — where the batch is
     * precomputed — BEFORE {@code applyGenericContingencyParameters}, and it is the latter that rebuilds
     * the outer-loop list with {@code FreezingHvdcACEmulation} in it when the security analysis asks to
     * start contingencies with the AC emulation frozen (its default). So at precompute time the
     * parameters still carry the BASE loop list.
     *
     * <p>Recomputes the same list the SA is about to install, by the same call. The device does not
     * depend on it — it implements the freezing itself in {@code GpuFreezingHvdcACEmulationOuterloop} — but the list
     * is what the run's own gates read, so aligning it early can only make them agree with the SA.
     */
    private void applyContingencyOuterLoops(AcLoadFlowParameters parameters) {
        SecurityAnalysisParameters saParameters = runState.get().securityAnalysisParameters;
        if (saParameters == null) {
            return;
        }
        var openSaParameters = OpenSecurityAnalysisParameters.getOrDefault(saParameters);
        if (!openSaParameters.isStartWithFrozenACEmulation()) {
            return;                                          // the SA will not override the list
        }
        var loadFlowParameters = saParameters.getLoadFlowParameters();
        var openLoadFlowParameters = com.powsybl.openloadflow.OpenLoadFlowParameters.clone(
                com.powsybl.openloadflow.OpenLoadFlowParameters.get(loadFlowParameters));
        openLoadFlowParameters.setStartWithFrozenACEmulation(true);
        parameters.setOuterLoops(com.powsybl.openloadflow.OpenLoadFlowParameters.createAcOuterLoops(
                loadFlowParameters, openLoadFlowParameters));
    }

}
