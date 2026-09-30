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
import com.powsybl.commons.PowsyblException;
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
import com.powsybl.openloadflow.network.LfGenerator;
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

    /** How many base-state restores of the last run were targeted ({@link #TARGETED_RESTORE_PROPERTY}). */
    static final java.util.concurrent.atomic.AtomicInteger LAST_TARGETED_RESTORE_COUNT =
            new java.util.concurrent.atomic.AtomicInteger();

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
    /** System property: answer OLF's per-contingency loss distribution from the batch's replay for
     *  device-solved contingencies instead of running it on the host (default on). Set to false for the
     *  A/B: the reported distributed active power must agree to rounding (see OLF_GPU_PRELOSS_CHECK). */
    public static final String REPLAY_LOSS_DISTRIBUTION_PROPERTY = "olf.gpu.sa.replayLossDistribution";

    private static boolean replayLossDistributionEnabled() {
        return Boolean.parseBoolean(System.getProperty(REPLAY_LOSS_DISTRIBUTION_PROPERTY, "true"));
    }

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
            RunState rs = runState.get();
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
        RunState rs = runState.get();
        boolean targeted = targetedRestoreEnabled()
                && lfContingency.getId().equals(rs.lossReplayedFor)
                && lfContingency.getId().equals(rs.deviceStateInjectedFor)
                && !(GpuRestoreCheck.ENABLED && !rs.restoreCheck.hasReference(network));   // the check's reference is a full restore
        rs.lossReplayedFor = null;
        rs.deviceStateInjectedFor = null;
        if (targeted) {
            java.util.Set<LfBus> buses = touchedBuses(network, lfContingency, rs);
            networkState.restore(buses, lfContingency.getDisabledNetwork().getBranchesStatus().keySet());
            LAST_TARGETED_RESTORE_COUNT.incrementAndGet();
        } else {
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

    /** System property: after a contingency the device solved and whose loss distribution the replay answered,
     *  restore only that contingency's own elements and the bus voltages instead of the whole network (default
     *  on). Set to false for the A/B; OLF_GPU_RESTORE_CHECK proves the two leave the same network. */
    public static final String TARGETED_RESTORE_PROPERTY = "olf.gpu.sa.targetedRestore";

    private static boolean targetedRestoreEnabled() {
        return Boolean.parseBoolean(System.getProperty(TARGETED_RESTORE_PROPERTY, "true"));
    }

    /**
     * The buses a device-solved contingency changed beyond their voltage: what LfContingency.apply touched — the
     * buses it disabled and those of the loads, generators and shunts it lost or shifted — closed over their
     * voltage and reactive-power control groups, because enabling or disabling a controller recomputes its
     * whole group's reactive targets.
     */
    private static java.util.Set<LfBus> touchedBuses(LfNetwork network, com.powsybl.openloadflow.network.LfContingency c,
                                                     RunState rs) {
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

    /** System property: after injecting a device state, refresh only the flows of the branches the violation
     *  detection will read (default on). Set to false for the A/B. */
    public static final String PARTIAL_FLOWS_PROPERTY = "olf.gpu.sa.partialFlows";

    /** OLF_GPU_PARTIAL_FLOW_CHECK=1: after each partial-flow detection, detect again with every flow refreshed from
     *  the same state and require the same violations, bit for bit. */
    private static final boolean PARTIAL_FLOW_CHECK = System.getenv("OLF_GPU_PARTIAL_FLOW_CHECK") != null;

    /** Tally of OLF_GPU_PARTIAL_FLOW_CHECK. */
    static final class PartialFlowCheck {
        private long checked;
        private long differing;
        private long branchesRead;
        private final List<String> samples = new java.util.ArrayList<>();

        void compare(String contingencyId, List<com.powsybl.contingency.violations.LimitViolation> partial,
                     List<com.powsybl.contingency.violations.LimitViolation> full, int branches) {
            checked++;
            branchesRead += branches;
            List<String> a = partial.stream().map(PartialFlowCheck::key).toList();
            List<String> b = full.stream().map(PartialFlowCheck::key).toList();
            if (!a.equals(b)) {
                differing++;
                if (samples.size() < 10) {
                    samples.add(contingencyId + ": partial " + a.size() + " violations, full " + b.size()
                            + " | first partial-only " + a.stream().filter(k -> !b.contains(k)).findFirst().orElse("-")
                            + " | first full-only " + b.stream().filter(k -> !a.contains(k)).findFirst().orElse("-"));
                }
            }
        }

        private static String key(com.powsybl.contingency.violations.LimitViolation v) {
            return v.getSubjectId() + "|" + v.getLimitType() + "|" + v.getSide() + "|" + v.getLimitName()
                    + "|" + Double.doubleToLongBits(v.getValue()) + "|" + Double.doubleToLongBits(v.getLimit());
        }

        void print() {
            System.err.printf("PARTIAL_FLOW_CHECK contingencies compared=%d identical=%d differing=%d | mean branches read %.1f%n",
                    checked, checked - differing, differing, checked > 0 ? (double) branchesRead / checked : 0.0);
            samples.forEach(x -> System.err.println("PARTIAL_FLOW_CHECK   " + x));
        }
    }

    /** EXPERIMENTAL system property: the device DECIDES the post-contingency violations with OLF's strict
     *  comparison on its own values, and the host only builds them — no candidate confirm, no branch flow on the
     *  host (default off). Reported values then differ from stock OLF's in the last bits. */
    public static final String DEVICE_VIOLATIONS_PROPERTY = "olf.gpu.sa.deviceViolations";

    private static boolean deviceViolationsEnabled() {
        return Boolean.parseBoolean(System.getProperty(DEVICE_VIOLATIONS_PROPERTY, "false"));
    }

    /** OLF_GPU_DEVICE_VIOLATIONS_CHECK=1: after each device decision, run the host's exact detection on the same
     *  state and compare — the violation SET (subject, type, side, limit) and how far the values are apart. */
    private static final boolean DEVICE_VIOLATIONS_CHECK = System.getenv("OLF_GPU_DEVICE_VIOLATIONS_CHECK") != null;

    /** Tally of OLF_GPU_DEVICE_VIOLATIONS_CHECK. */
    static final class DeviceViolationCheck {
        private long checked;
        private long setDiffers;
        private long valuesBitIdentical;
        private long violations;
        private double worstRelative;
        private String worstWhere = "-";
        private final List<String> samples = new java.util.ArrayList<>();

        void compare(String contingencyId, List<com.powsybl.contingency.violations.LimitViolation> device,
                     List<com.powsybl.contingency.violations.LimitViolation> host) {
            checked++;
            java.util.Map<String, Double> d = new java.util.LinkedHashMap<>();
            device.forEach(v -> d.put(key(v), v.getValue()));
            java.util.Map<String, Double> h = new java.util.LinkedHashMap<>();
            host.forEach(v -> h.put(key(v), v.getValue()));
            if (!d.keySet().equals(h.keySet())) {
                setDiffers++;
                if (samples.size() < 10) {
                    samples.add(contingencyId + ": device " + d.size() + ", host " + h.size() + " | device-only "
                            + d.keySet().stream().filter(k -> !h.containsKey(k)).findFirst().orElse("-")
                            + " | host-only " + h.keySet().stream().filter(k -> !d.containsKey(k)).findFirst().orElse("-"));
                }
            }
            boolean identical = true;
            for (var e : d.entrySet()) {
                Double hv = h.get(e.getKey());
                if (hv == null) {
                    continue;
                }
                violations++;
                if (Double.doubleToLongBits(hv) != Double.doubleToLongBits(e.getValue())) {
                    identical = false;
                    double rel = Math.abs(hv - e.getValue()) / Math.max(1e-300, Math.abs(hv));
                    if (rel > worstRelative) {
                        worstRelative = rel;
                        worstWhere = contingencyId + " " + e.getKey();
                    }
                }
            }
            if (identical) {
                valuesBitIdentical++;
            }
        }

        private static String key(com.powsybl.contingency.violations.LimitViolation v) {
            return v.getSubjectId() + "|" + v.getLimitType() + "|" + v.getSide() + "|" + v.getLimitName();
        }

        void print() {
            System.err.printf("DEVICE_VIOLATIONS_CHECK contingencies=%d same-set=%d different-set=%d | values bit-identical in %d"
                            + " | %d violations compared, worst relative value difference %.3e (%s)%n",
                    checked, checked - setDiffers, setDiffers, valuesBitIdentical, violations, worstRelative, worstWhere);
            samples.forEach(x -> System.err.println("DEVICE_VIOLATIONS_CHECK   " + x));
        }
    }

    private static boolean partialFlowsEnabled() {
        return Boolean.parseBoolean(System.getProperty(PARTIAL_FLOWS_PROPERTY, "true"));
    }

    /**
     * The branches whose flows anything reads after a device state is injected for {@code contingencyId}, or null
     * for "refresh them all". With the device violation screen on, the detection reads only the branches of the
     * limits the screen flagged for this contingency and the branches it does not cover (GpuLimitScanner). Every
     * other reader forces the full refresh: a branch or three-windings transformer state monitor (network results),
     * an operator strategy on the contingency (threshold conditions on branch flows), the first contingency (the
     * screen is built during its detection), and the diagnostics that read flows — OLF_GPU_VIOL_CROSSCHECK among them,
     * which runs the stock detection over EVERY branch.
     */
    private int[] flowReadSet(RunState rs, String contingencyId) {
        if (!partialFlowsEnabled() || rs.violationScreen == null || rs.candidates == null || flowsMonitored()
                || targetedByOperatorStrategy(rs.operatorStrategies, contingencyId)
                || System.getenv("OLF_GPU_Q_PROBE") != null || System.getenv("OLF_GPU_SA_RESID") != null
                || System.getenv("OLF_GPU_VIOL_CROSSCHECK") != null) {
            return null;
        }
        GpuLimitTable table = rs.limitTable;
        java.util.BitSet nums = new java.util.BitSet();
        if (rs.decided != null) {                            // the device decided: only uncovered branches are read
            if (!rs.decided.containsKey(contingencyId) || DEVICE_VIOLATIONS_CHECK) {
                return null;
            }
            for (int b : rs.uncoveredBranches) {
                nums.set(table.branches()[b].getNum());
            }
            rs.sideOpened.forEach(branch -> nums.set(branch.getNum()));
            return nums.stream().toArray();
        }
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
    private static java.util.BitSet sideOpenedTableIndices(RunState rs, LfNetwork network) {
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
            var ctx = strategy.getContingencyContext();
            boolean targets = switch (ctx.getContextType()) {
                case ALL, ONLY_CONTINGENCIES -> true;
                case SPECIFIC -> contingencyId.equals(ctx.getContingencyId());
                case NONE -> false;
            };
            if (targets) {
                return true;
            }
        }
        return false;
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
        private final PartialFlowCheck partialFlowCheck = new PartialFlowCheck();
        /** OLF_GPU_RESTORE_CHECK: this partition's restores against the state a full restore produces. */
        private final GpuRestoreCheck restoreCheck = new GpuRestoreCheck();
        /** The contingency whose loss distribution the replay answered, and the one whose state the device supplied:
         *  when both name the contingency just simulated, only its own elements and the bus voltages changed. */
        private String lossReplayedFor;
        private String deviceStateInjectedFor;
        /** Shunt -> its bus, for the shunts a contingency shifts (LfShunt does not know its bus). Built once. */
        private java.util.Map<com.powsybl.openloadflow.network.LfShunt, LfBus> shuntBus;
        /** Per bus num: does the realistic-voltage band apply (not fictitious, nominal V at or above the
         *  check's minimum)? Built once for {@link #realisticBandNetwork}. */
        private boolean[] realisticBandApplies;
        private LfNetwork realisticBandNetwork;
        /** Per bus num: its BUS_V / BUS_PHI row in the device state (base layout), for the extraction they were
         *  built from; null when the base system is not exactly one V and one angle per bus (the walk is used). */
        private int[] vRowOfBus;
        private int[] phiRowOfBus;
        private Map<Variable<AcVariableType>, Integer> busRowsFor;
        private boolean busRowsUsable;
        private long busArrayChecked;
        private long busArrayDiffering;
        private String busArrayFirstDiff;
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
        /** The run's security-analysis parameters — needed at precompute time because the SA's own
         *  CONTINGENCY parameter override has not been applied yet (see afterPreContingencySimulation). */
        private SecurityAnalysisParameters securityAnalysisParameters;
        /** The run's flattened limit thresholds, built on first use and valid for the whole run
         *  (see GpuLimitTable); null when the accelerated detection is off. */
        private GpuLimitScanner limitScanner;
        private GpuLimitTable limitTable;
        /** Non-null when the state is screened against the table instead of fully scanned. */
        private GpuViolationScreen violationScreen;
        private int[] uncoveredBranches = new int[0];
        /** The current contingency's branches opened at ONE side (a bus or busbar loss): the screen evaluates every
         *  branch in the base topology, so their records are not used and they are scanned on the host. */
        private List<LfBranch> sideOpened = List.of();
        /** Limit-table index per branch num, -1 = no limits (built with the table). */
        private int[] tableIndexOfBranchNum;
        /** Every scenario's screened candidates, computed in one pass over the precomputed states. */
        private Map<String, int[]> candidates = Map.of();
        /** EXPERIMENTAL olf.gpu.sa.deviceViolations: the device's decision per scenario (absent = host path). */
        private Map<String, GpuViolationScreen.Decided> decided;
        private final DeviceViolationCheck deviceViolationCheck = new DeviceViolationCheck();
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
        // OLF_GPU_SA_BASE_ONLY=1: solve the BASE and stop - no contingency simulated. The network
        // was built for the whole contingency list (the retained switches, the structure the base
        // has to factor), which is the point: the full busbar scope's base on the device without the
        // 20 minutes of replays behind it. Debug lever for the base work; the result carries the
        // pre-contingency state only.
        List<PropagatedContingency> toSimulate = "1".equals(System.getenv("OLF_GPU_SA_BASE_ONLY"))
                ? List.of() : propagatedContingencies;
        ContingencyActivePowerLossDistribution lossDistribution;
        if (System.getenv("OLF_GPU_PRELOSS_CHECK") != null) {
            lossDistribution = new LossDistributionCheck(contingencyActivePowerLossDistribution);
        } else if (replayLossDistributionEnabled()) {
            lossDistribution = new ReplayedLossDistribution(contingencyActivePowerLossDistribution);
        } else {
            lossDistribution = contingencyActivePowerLossDistribution;
        }
        try {
            return super.runSimulations(lfNetwork, toSimulate, acParameters, securityAnalysisParameters,
                    operatorStrategies, actions, limitReductions, lossDistribution);
        } finally {
            if (lossDistribution instanceof LossDistributionCheck check) {
                check.print();
            }
            if (PARTIAL_FLOW_CHECK) {
                runState.get().partialFlowCheck.print();
            }
            if (BUS_ARRAY_CHECK) {
                RunState check = runState.get();
                System.err.printf("BUS_ARRAY_CHECK contingencies=%d identical=%d differing=%d | array path usable=%b | first difference: %s%n",
                        check.busArrayChecked, check.busArrayChecked - check.busArrayDiffering, check.busArrayDiffering,
                        check.busRowsUsable, check.busArrayFirstDiff == null ? "-" : check.busArrayFirstDiff);
            }
            if (DEVICE_VIOLATIONS_CHECK) {
                runState.get().deviceViolationCheck.print();
            }
            if (GpuRestoreCheck.ENABLED) {
                runState.get().restoreCheck.print();
                System.err.println("RESTORE_CHECK targeted restores (cumulative this run): " + LAST_TARGETED_RESTORE_COUNT.get());
            }
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
                        + CROSSCHECKED.sum());
            }
        }
    }

    /** True when the base load flow's batch-of-one drives OLF's outer loops on the DEVICE, so this
     *  analysis must not also drive them in Java. DEFAULT since 2026-09-16: a batch of one is still a
     *  batch, and running the base through the same driver as the chunks is what keeps the two on one
     *  trajectory — rte6515 parity is identical either way (9040/9042, the same 2 knife-edge lines) and
     *  the driver is never slower. {@code OLF_GPU_DRIVER_OUTER_LOOPS=0} or
     *  {@code -Dolf.gpu.driverOuterLoops=false} puts OLF's Java engine back in charge of the base. */
    static boolean driverOuterLoopsEnabled() {
        return !"0".equals(System.getenv("OLF_GPU_DRIVER_OUTER_LOOPS"))
                && !"false".equalsIgnoreCase(System.getProperty("olf.gpu.driverOuterLoops", "true"));
    }

    @Override
    protected com.powsybl.openloadflow.ac.AcloadFlowEngine createLoadFlowEngine(
            com.powsybl.openloadflow.ac.AcLoadFlowContext context) {
        return driverOuterLoopsEnabled() ? new GpuDriverOuterLoopEngine(context) : super.createLoadFlowEngine(context);
    }

    /** System property toggle for the GPU base/fall-through load flow (default on). */
    public static final String GPU_BASE_LOADFLOW_PROPERTY = "olf.gpu.sa.gpuBaseLoadFlow";

    /** System property: skip the batch entirely and let OLF's engine solve every contingency
     *  sequentially — the A/B arm for "does batching pay for the multiplexing?". */
    public static final String GPU_DISABLE_BATCH_PROPERTY = "olf.gpu.sa.disableBatch";

    /** System property: keep the GPU solver for SEQUENTIAL fall-through contingency solves
     *  (default off = revert to the CPU Newton-Raphson after the base LF). */
    public static final String GPU_SEQUENTIAL_LOADFLOW_PROPERTY = "olf.gpu.sa.gpuSequentialLoadFlow";

    private static boolean gpuSequentialLoadFlowEnabled() {
        return Boolean.parseBoolean(System.getProperty(GPU_SEQUENTIAL_LOADFLOW_PROPERTY, "false"));
    }

    private static boolean gpuBaseLoadFlowEnabled() {
        return Boolean.parseBoolean(System.getProperty(GPU_BASE_LOADFLOW_PROPERTY, "true"));
    }

    /** System property: detect post-contingency violations by scanning the run's limits as flat
     *  arrays ({@link GpuLimitTable}) instead of walking OLF's per-call suppliers and lists
     *  (default on). Set to false for the A/B — the two must report the SAME violations, since the
     *  scan makes the same decisions on the same values. */
    public static final String GPU_TABLE_VIOLATIONS_PROPERTY = "olf.gpu.sa.tableViolations";

    /** How many contingencies the cross-check actually compared — printed at the end of the run, so
     *  "no failures" cannot be confused with "never ran". */
    private static final java.util.concurrent.atomic.LongAdder CROSSCHECKED = new java.util.concurrent.atomic.LongAdder();

    private static boolean tableViolationsEnabled() {
        return Boolean.parseBoolean(System.getProperty(GPU_TABLE_VIOLATIONS_PROPERTY, "true"));
    }

    /** System property: SCREEN the post-contingency states on the DEVICE against the limit table
     *  and confirm only the candidates on the host, instead of scanning every limit (default on).
     *  Set to false for the A/B — the two must report the same violations, because the screen only
     *  ever proposes and the host still decides with OLF's own value. */
    public static final String GPU_SCREEN_VIOLATIONS_PROPERTY = "olf.gpu.sa.screenViolations";

    private static boolean screenViolationsEnabled() {
        return Boolean.parseBoolean(System.getProperty(GPU_SCREEN_VIOLATIONS_PROPERTY, "true"));
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
     * device. The table is built once per run and cached on the partition's RunState.
     */
    private void detectViolations(LfNetwork network, LimitViolationManager manager, List<LimitReduction> limitReductions,
                                  LimitViolationManager reference, SecurityAnalysisParameters saParameters,
                                  String contingencyId) {
        if (!tableViolationsEnabled()) {
            manager.detectViolations(network);
            return;
        }
        RunState rs = runState.get();
        if (rs.limitScanner == null) {
            GpuLimitTable table = GpuLimitTable.build(network, LimitReductionManager.create(limitReductions));
            rs.limitScanner = new GpuLimitScanner(table);
            rs.limitTable = table;
            if (screenViolationsEnabled() && rs.precomputed.gpuData() != null) {
                rs.violationScreen = GpuViolationScreen.build(network, table, rs.precomputed.gpuData(),
                        rs.precomputed.elementMapper(), rs.precomputed.equationSystem(),
                        rs.precomputed.baseVariableRows());
                rs.uncoveredBranches = uncoveredBranches(table, rs.violationScreen);
                // Screen the WHOLE run at once: the states are all precomputed already, and one
                // device call per 512 scenarios is what makes the kernel worth its upload.
                long tScreen = System.nanoTime();
                if (deviceViolationsEnabled()) {
                    rs.decided = rs.violationScreen.decideAll(rs.precomputed.states(),
                            rs.precomputed.baseVariableRows().size());
                    rs.candidates = new java.util.HashMap<>();   // undecided scenarios screen on the host
                } else {
                    rs.candidates = rs.violationScreen.screenAll(rs.precomputed.states(),
                            rs.precomputed.baseVariableRows().size());
                }
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
        GpuViolationScreen.Decided decision = rs.decided != null ? rs.decided.get(contingencyId) : null;
        java.util.BitSet sideOpened = rs.violationScreen != null ? sideOpenedTableIndices(rs, network) : null;
        if (decision != null) {
            rs.limitScanner.reportDecided(network, manager, decision, rs.uncoveredBranches, sideOpened);
            if (DEVICE_VIOLATIONS_CHECK) {
                var es = rs.precomputed.equationSystem();
                es.getStateVector().setValuesOnly(es.getStateVector().get().clone());   // every flow, same state
                var host = new LimitViolationManager(reference, limitReductions, saParameters.getIncreasedViolationsParameters());
                rs.limitScanner.detectFromCandidates(network, host,
                        rs.violationScreen.screen(rs.precomputed.states().get(contingencyId)),
                        rs.violationScreen, rs.uncoveredBranches, sideOpened);
                rs.deviceViolationCheck.compare(contingencyId, manager.getLimitViolations(), host.getLimitViolations());
            }
        } else if (rs.violationScreen != null) {
            int[] candidates = rs.candidates.get(contingencyId);
            rs.limitScanner.detectFromCandidates(network, manager,
                    candidates != null ? candidates : rs.violationScreen.screen(rs.precomputed.states().get(contingencyId)),
                    rs.violationScreen, rs.uncoveredBranches, sideOpened);
        } else {
            rs.limitScanner.detectViolations(network, manager);
        }
        if (System.getenv("OLF_GPU_VIOL_CROSSCHECK") != null) {
            crossCheck(network, manager, limitReductions, reference, saParameters, contingencyId);
        }
    }

    /** {@code OLF_GPU_VIOL_CROSSCHECK=1}: run the stock detection on the SAME state and demand the
     *  same violations, field for field and in the same order. A toy network cannot exercise the
     *  cases that make the flattening delicate — three-winding legs, tie lines, several operational
     *  limits groups, temporary limits — so the check runs on whatever network the caller is running,
     *  and fails loudly on the first contingency that disagrees. */
    private static void crossCheck(LfNetwork network, LimitViolationManager manager, List<LimitReduction> limitReductions,
                                   LimitViolationManager reference, SecurityAnalysisParameters saParameters, String contingencyId) {
        CROSSCHECKED.increment();
        var stock = new LimitViolationManager(reference, limitReductions, saParameters.getIncreasedViolationsParameters());
        stock.detectViolations(network);
        List<com.powsybl.contingency.violations.LimitViolation> fromTable = manager.getLimitViolations();
        List<com.powsybl.contingency.violations.LimitViolation> fromStock = stock.getLimitViolations();
        if (fromTable.size() != fromStock.size()) {
            throw new PowsyblException("GPU violation table: " + contingencyId + " reported " + fromTable.size()
                    + " violations, the stock detection " + fromStock.size());
        }
        for (int i = 0; i < fromTable.size(); i++) {
            String a = describe(fromTable.get(i));
            String b = describe(fromStock.get(i));
            if (!a.equals(b)) {
                throw new PowsyblException("GPU violation table: " + contingencyId + " violation " + i
                        + " is " + a + ", the stock detection says " + b);
            }
        }
    }

    /** Every reported field of a violation, so the cross-check catches a wrong scale or a wrong
     *  temporary limit — not only a wrong verdict. */
    private static String describe(com.powsybl.contingency.violations.LimitViolation v) {
        return v.getSubjectId() + "|" + v.getSide() + "|" + v.getLimitType() + "|" + v.getOperationalLimitsGroupId()
                + "|" + v.getLimitName() + "|" + v.getAcceptableDuration() + "|" + v.getLimit()
                + "|" + v.getLimitReduction() + "|" + v.getValue();
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
        RunState rs = runState.get();
        if (System.getenv("OLF_GPU_Q_PROBE") != null && rs.lfNetwork != null) {
            String qp = System.getenv("OLF_GPU_Q_PROBE");
            for (LfBus b : rs.lfNetwork.getBuses()) {
                if (java.util.Arrays.stream(qp.split(",")).anyMatch(b.getId()::startsWith)) {
                    System.err.printf("Q_PROBE_BASE bus=%s q=%.9f maxQ=%.9f busTargetP=%.9f%n",
                            b.getId(), b.getQ().eval() + b.getLoadTargetQ(), b.getMaxQ(), b.getTargetP());
                    // the BASE generator P: the reference every arm's slack distribution moves from
                    for (var g : b.getGenerators()) {
                        System.err.printf("Q_PROBE_BASE_GEN bus=%s gen=%s targetP=%.9f minP=%.9f maxP=%.9f participating=%b%n",
                                b.getId(), g.getId(), g.getTargetP(), g.getMinP(), g.getMaxP(), g.isParticipating());
                    }
                }
            }
        }
        if (Boolean.getBoolean(GPU_DISABLE_BATCH_PROPERTY)) {
            // A/B arm: OLF's own engine drives every contingency sequentially (its outer loops, its
            // one-restore-per-contingency), with the GPU solver underneath if
            // olf.gpu.sa.gpuSequentialLoadFlow is on. This is the comparison that says whether BATCHING
            // pays for the multiplexing it forces on the hybrid.
            return;
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
            LAST_GPU_PRECOMPUTED_IDS.addAll(rs.precomputed.states().keySet());
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

    /** OLF's residual at the injected device state, worst rows first — see {@code OLF_GPU_SA_RESID}. */
    private static void reportOlfResidual(AcLoadFlowContext context,
                                          com.powsybl.openloadflow.equations.EquationSystem<com.powsybl.openloadflow.ac.equations.AcVariableType,
                                                  com.powsybl.openloadflow.ac.equations.AcEquationType> es,
                                          String contingencyId, int topN,
                                          Map<com.powsybl.openloadflow.equations.Variable<com.powsybl.openloadflow.ac.equations.AcVariableType>,
                                                  Integer> baseVariableRows,
                                          int[] islandedBusNums) {
        var ev = context.getEquationVector();
        var tv = context.getTargetVector();
        ev.minus(tv);
        double[] fx = ev.getArray();
        System.err.printf("SA_RESID ctg=%s olfActiveEquations=%d%n", contingencyId, fx.length);
        Integer[] order = new Integer[fx.length];
        for (int i = 0; i < fx.length; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, (a, b) -> Double.compare(Math.abs(fx[b]), Math.abs(fx[a])));
        for (int i = 0; i < Math.min(topN, order.length); i++) {
            int row = order[i];
            var eq = es.getIndex().getEquationAtColumn(row);
            var elem = context.getNetwork().getElement(eq.getType().getElementType(), eq.getElementNum());
            // OLF's own TARGET for that row, and the DEVICE row the same quantity lives on. A residual
            // is (state - target): printing OLF's target next to the device row lets the two be
            // compared directly (point OLF_GPU_FINAL_TARGET_ROW at devRow), which separates "the
            // device solved a different TARGET" from "the device assembled a different EQUATION" —
            // indistinguishable from the residual alone.
            int devRow = -1;
            if (baseVariableRows != null && eq != null) {
                var vr = es.getVariableSet().getVariable(eq.getElementNum(),
                        eq.getType() == com.powsybl.openloadflow.ac.equations.AcEquationType.BUS_TARGET_P
                                ? com.powsybl.openloadflow.ac.equations.AcVariableType.BUS_PHI
                                : com.powsybl.openloadflow.ac.equations.AcVariableType.BUS_V);
                Integer r = vr == null ? null : baseVariableRows.get(vr);
                devRow = r == null ? -1 : r;
            }
            // SIGNED, not |F|. Two buses whose residuals are equal and OPPOSITE are the two ends of a
            // branch whose flow term is missing from both balances; two buses off the same way are not.
            // The magnitude alone cannot tell those apart.
            // Is this bus a NEIGHBOUR OF THE ISLAND? The island is frozen at a flat 1<0 while the live
            // network sits at its real voltages, so any branch still bridging the two would push power
            // into the live side. If the erring buses turn out NOT to touch the island, that whole
            // explanation is dead and the injection comes from somewhere else entirely.
            int toIsland = 0;
            int degree = 0;
            double islandShunt = Double.NaN;
            if (islandedBusNums != null && elem instanceof com.powsybl.openloadflow.network.LfBus b) {
                java.util.Set<Integer> isl = new java.util.HashSet<>();
                for (int bn : islandedBusNums) {
                    isl.add(bn);
                }
                for (var br : b.getBranches()) {
                    degree++;
                    var other = br.getBus1() == b ? br.getBus2() : br.getBus1();
                    if (other != null && isl.contains(other.getNum())) {
                        toIsland++;
                        // The QUANTITATIVE test of "the device dropped the dangling half": if that is
                        // what happened, the residual should be this branch's connected-side shunt
                        // term, b_side * v^2, which OLF keeps and the device deletes with the element.
                        var pi = br.getPiModel();
                        double bSide = br.getBus1() == b ? pi.getB1() : pi.getB2();
                        double v = b.getV();
                        islandShunt = bSide * v * v;
                    }
                }
            }
            // WHAT IS ATTACHED to the bus whose equation is violated. A residual that is the same
            // round number in every contingency is a fixed missing TERM, not a solve difference,
            // and only the bus's own inventory says which term: a shunt's b, the generators' and
            // loads' reactive, and whether OLF considers the bus disabled at all.
            if (elem instanceof LfBus rb) {
                double shB = rb.getShunt().map(sh -> sh.getB()).orElse(0.0)
                        + rb.getControllerShunt().map(sh -> sh.getB()).orElse(0.0)
                        + rb.getSvcShunt().map(sh -> sh.getB()).orElse(0.0);
                double genQ = 0;
                for (LfGenerator g : rb.getGenerators()) {
                    genQ += g.getTargetQ();
                }
                System.err.printf("SA_RESID_BUS ctg=%s id=%s disabled=%s v=%.9f shuntB=%.9f"
                        + " nGen=%d genTargetQ=%.9f loadTargetQ=%.9f vcEnabled=%s%n",
                        contingencyId, rb.getId(), rb.isDisabled(), rb.getV(), shB,
                        rb.getGenerators().size(), genQ, rb.getLoadTargetQ(),
                        rb.isGeneratorVoltageControlEnabled());
            }
            // EVERY branch of the network incident to this bus, disabled ones included. getBranches()
            // is the bus's LIVE list and came back empty while the equation still evaluated non-zero,
            // so one of the two is lying; this says which, and names the branch either way.
            if (elem instanceof LfBus ib) {
                for (LfBranch nb : context.getNetwork().getBranches()) {
                    if (nb.getBus1() == ib || nb.getBus2() == ib) {
                        System.err.printf("SA_RESID_INC ctg=%s bus=%s branch=%s num=%d disabled=%s"
                                + " zeroImp=%s side=%d%n", contingencyId, ib.getId(), nb.getId(),
                                nb.getNum(), nb.isDisabled(),
                                nb.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC),
                                nb.getBus1() == ib ? 1 : 2);
                    }
                }
            }
            // The TERMS of the violated equation, each with its element and its value at the
            // device's state. A bus with no branches and no shunt should have a Q balance with no
            // terms at all; when it evaluates to a round non-zero number something IS in there, and
            // only the term list says what.
            {
                int shown = 0;
                for (var t : eq.<com.powsybl.openloadflow.equations.EquationTerm<AcVariableType,
                        AcEquationType>>getTerms()) {
                    if (!t.isActive() || shown++ >= 12) {
                        continue;
                    }
                    double termVal;
                    try {
                        termVal = t.eval();
                    } catch (RuntimeException ex) {
                        termVal = Double.NaN;
                    }
                    // Name the term by its VARIABLES. The element number of an
                    // EquationTermArrayElementImpl is the array wrapper's, not the branch's, so it
                    // identifies nothing; the variable list distinguishes a branch flow (v/phi of
                    // two buses) from a zero-impedance DUMMY_Q, which is the whole question when a
                    // bus has no branches left.
                    StringBuilder vars = new StringBuilder();
                    try {
                        for (var vv : t.getVariables()) {
                            vars.append(' ').append(vv.getType()).append('#').append(vv.getElementNum());
                            if (vv.getType() == AcVariableType.BUS_V || vv.getType() == AcVariableType.BUS_PHI) {
                                LfBus vb = context.getNetwork().getBus(vv.getElementNum());
                                vars.append('(').append(vb == null ? "?" : vb.getId())
                                        .append(vb != null && vb.isDisabled() ? ",DISABLED" : "").append(')');
                            }
                        }
                    } catch (RuntimeException ex) {
                        vars.append(" <unavailable>");
                    }
                    System.err.printf("SA_RESID_TERM ctg=%s row=%d term=%s elemType=%s value=%.9f vars:%s%n",
                            contingencyId, row, t.getClass().getSimpleName(),
                            t.getElementType(), termVal, vars);
                }
            }
            // For a violated DISTR_Q, OLF's own q for EVERY member of the group, evaluated at the
            // DEVICE's state. The device believes its sharing equation is satisfied there; if OLF
            // disagrees, these are the numbers that say which member's q the two compute
            // differently - the equation residual alone cannot.
            if (eq.getType() == AcEquationType.DISTR_Q && elem instanceof LfBus db) {
                db.getGeneratorVoltageControl().ifPresent(vc -> {
                    for (LfBus m : vc.getMergedControllerElements()) {
                        System.err.printf("SA_RESID_Q ctg=%s member=%s qPercent=%.9f qAtDeviceState=%.9f"
                                + " loadQ=%.9f v=%.9f%n", contingencyId, m.getId(),
                                m.getRemoteControlReactivePercent(), m.getQ().eval(),
                                m.getLoadTargetQ(), m.getV());
                    }
                });
            }
            System.err.printf("SA_RESID ctg=%s row=%d F=%+.9e eq=%s elem=%d id=%s olfTarget=%.9f devRow=%d"
                            + " degree=%d toIsland=%d islandBranchShunt=%+.9f%n",
                    contingencyId, row, fx[row], eq.getType(), eq.getElementNum(),
                    elem == null ? "?" : elem.getId(), tv.getArray()[row], devRow, degree, toIsland,
                    islandShunt);
        }
    }

    /**
     * {@code OLF_GPU_EQ_DUMP_CTG=<contingency id>}: writes OLF's OWN equation set for the POST-CONTINGENCY
     * state — {@code lfContingency.apply()} has already run when this is called, so the system here is the
     * one the CPU actually solves for that scenario, with inactive equations in {@code [ ]}.
     *
     * <p>This is the reference the device side has been missing. Reasoning about where a relocated row COULD
     * go produced two wrong fixes in a row (a DISTR_Q routed back to its bus's V row made the worst bus
     * difference 150x larger), because a coupler opening does not merely free a row: it changes the GROUP, and
     * OLF re-evaluates {@code updateRemoteVoltageControlEquations} over the live controllers, which can
     * DEACTIVATE an equation rather than move it. The only way to know is to read what OLF has.
     *
     * <p>Written to {@code OLF_GPU_EQ_DUMP_CTG_FILE}, or {@code <id>.olf.post} in the working directory.
     * Also lists the contingency's own elements, so the dump identifies which couplers opened.
     */
    private void dumpPostContingencyEquations(AcLoadFlowContext context, Contingency contingency,
                                              LfContingency lfContingency, LfNetwork network) {
        String want = System.getenv("OLF_GPU_EQ_DUMP_CTG");
        if (want == null || !want.equals(contingency.getId())) {
            return;
        }
        String path = System.getenv().getOrDefault("OLF_GPU_EQ_DUMP_CTG_FILE",
                contingency.getId().replaceAll("[^A-Za-z0-9_.-]", "_") + ".olf.post");
        try (java.io.Writer w = java.nio.file.Files.newBufferedWriter(java.nio.file.Path.of(path))) {
            w.write("# OLF's POST-CONTINGENCY equation system for " + contingency.getId()
                    + " (inactive equations in [ ])\n");
            w.write("# disabled buses: " + lfContingency.getDisabledNetwork().getBuses().stream()
                    .map(LfBus::getId).sorted().toList() + "\n");
            w.write("# disabled branches: " + lfContingency.getDisabledNetwork().getBranches().stream()
                    .map(LfBranch::getId).sorted().toList() + "\n");
            // OLF's own mismatch to distribute, the reference for the device's scheduledActivePowerLoss
            // (OLF_GPU_LOSS_TRACE prints that one): gen - load over the DISABLED buses plus the named lost
            // loads/generators that are not on them.
            w.write(String.format("# OLF activePowerLoss=%.9f pu (disconnectedGeneration=%.9f"
                            + " disconnectedLoad=%.9f)%n", lfContingency.getActivePowerLoss(),
                    lfContingency.getDisconnectedGenerationActivePower(),
                    lfContingency.getDisconnectedLoadActivePower()));
            for (LfBus b : lfContingency.getDisabledNetwork().getBuses()) {
                w.write(String.format("#   disabled bus %s: generationTargetP=%.9f loadTargetP=%.9f%n",
                        b.getId(), b.getGenerationTargetP(), b.getLoadTargetP()));
            }
            // OLF writes equations as <symbol><elementNum>, so a legend is what makes the file readable
            // next to the device's row dump, which names elements by id.
            w.write("# bus legend: num = id\n");
            for (LfBus b : network.getBuses()) {
                w.write("#   bus " + b.getNum() + " = " + b.getId() + (b.isDisabled() ? "  [DISABLED]" : "") + "\n");
            }
            w.write("# branch legend: num = id\n");
            for (LfBranch br : network.getBranches()) {
                w.write("#   branch " + br.getNum() + " = " + br.getId()
                        + (br.isDisabled() ? "  [DISABLED]" : "") + "\n");
            }
            context.getEquationSystem().write(w, true);
            // ...and OLF's OWN f(x0) - target per ACTIVE equation at the post-contingency start (the residual
            // its first |f(x0)| line is the norm of), the contingency counterpart of OLF_GPU_F0_TRACE: lined
            // up against the device's first fill of that slot (OLF_GPU_VALS_SCAN_SLOT + OLF_GPU_VALS_DUMP), it
            // names the equation the two arms evaluate differently before any step blurs it.
            // What this section IS, since a previous version of this comment got it wrong: the
            // POST-contingency equation system evaluated at the PRE-contingency state. The dump runs on
            // the first line of runPostContingencySimulation, before the solver touches x, but
            // lfContingency.apply() has already deactivated equations, which invalidates the vector, so
            // AbstractVector.getArray() RECREATES it against the current index -- verified on rte6515
            // BUS-4467_BBS: fx0.length is 14494, exactly the active-equation count of the
            // post-contingency system section, not the base's. So it is NOT stale.
            //
            // It is still NOT directly comparable with a device fill's norm. Both arms start from the
            // base state, but the device's first fill is taken after its own seeding (frozen islanded
            // rows held at a flat V=1/phi=0, the pre-loss target seeding) and after applyRowMode has
            // turned identity rows into x - target. Measured: OLF norm2=774.03 against the device's
            // 35.95, where OLF's is dominated almost entirely by TWO equations -- distr_q1952/1953 at
            // 519.00 and 518.98 -- so the norms are a statement about those two, not about the network.
            //
            // Hence f and target are written SEPARATELY below. With only the difference printed, a value
            // of 519 cannot be attributed: a large residual, a large target, or a variable the state
            // does not constrain yet all look identical. Those two equations are DISTR_Q rows built on
            // coupler DUMMY_Q variables (dummy_q9161/9162), whose base values are exactly what a
            // pre-contingency state leaves unconstrained -- which the split makes readable.
            try {
                double[] fx0 = context.getEquationVector().getArray();
                double[] tg0 = context.getTargetVector().getArray();
                double n2 = 0;
                w.write("# F0: column type elementNum id residual(f-target) f target\n");
                for (int c = 0; c < fx0.length; c++) {
                    double v = fx0[c] - (c < tg0.length ? tg0[c] : 0.0);
                    n2 += v * v;
                    var eq = context.getEquationSystem().getIndex().getEquationAtColumn(c);
                    String id;
                    try {
                        id = switch (eq.getType().getElementType()) {
                            case BUS -> network.getBus(eq.getElementNum()).getId();
                            case BRANCH -> network.getBranch(eq.getElementNum()).getId();
                            default -> "#" + eq.getElementNum();
                        };
                    } catch (RuntimeException e) {
                        id = "#" + eq.getElementNum();
                    }
                    w.write("F0 " + c + " " + eq.getType() + " " + eq.getElementNum() + " " + id + " " + v
                            + " " + fx0[c] + " " + (c < tg0.length ? tg0[c] : 0.0) + "\n");
                }
                w.write("# F0 norm2=" + Math.sqrt(n2) + "\n");
            } catch (RuntimeException e) {
                w.write("# F0 unavailable: " + e + "\n");
            }
        } catch (java.io.IOException | RuntimeException e) {
            LOGGER.warn("OLF_GPU_EQ_DUMP_CTG failed: {}", e.toString());
        }
        System.err.println("EQ_DUMP_CTG wrote " + path + " for " + contingency.getId());
    }

    @Override
    protected PostContingencyResult runPostContingencySimulation(LfNetwork network, AcLoadFlowContext context, Contingency contingency,
                                                                 LfContingency lfContingency, LimitViolationManager preContingencyLimitViolationManager,
                                                                 SecurityAnalysisParameters securityAnalysisParameters,
                                                                 PreContingencyNetworkResult preContingencyNetworkResult, boolean createResultExtension,
                                                                 List<LimitReduction> limitReductions, double preDistributedActivePower) {
        dumpPostContingencyEquations(context, contingency, lfContingency, network);
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
        if (precomputed.isReady() && precomputed.exhausted().contains(contingency.getId())) {
            // The device carried this scenario through OLF's whole Newton budget without
            // convergence. OLF's own path (AbstractSecurityAnalysis.runPostContingencySimulation)
            // reports MAX_ITERATION_REACHED with an empty network result and no violations for
            // that outcome; the same result is built here, from the contingency already applied.
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
        if (state == null || !precomputed.isReady()) {
            // not batched (multi-element / islanding / out of scope / rejected) — standard CPU path
            return super.runPostContingencySimulation(network, context, contingency, lfContingency, preContingencyLimitViolationManager,
                    securityAnalysisParameters, preContingencyNetworkResult, createResultExtension, limitReductions, preDistributedActivePower);
        }

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
            reportOlfResidual(context, precomputed.equationSystem(), contingency.getId(),
                    Integer.parseInt(System.getenv("OLF_GPU_SA_RESID")), precomputed.baseVariableRows(),
                    precomputed.islandedBuses().get(contingency.getId()));
        }
        boolean unrealistic = updateBusesFromState(state, precomputed, network, context.getParameters(), runState.get());
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
                        int cbCount = gd.cbIn().length / 8;
                        int idxStride = cbCount > 0 ? gd.cbIdx().length / cbCount : 0;
                        for (var br : b.getBranches()) {
                            var pm = br.getPiModel();
                            Integer ph1 = br.getBus1() == null ? null : precomputed.baseVariableRows().get(precomputed.equationSystem().getVariableSet()
                                    .getVariable(br.getBus1().getNum(), AcVariableType.BUS_PHI));
                            Integer ph2 = br.getBus2() == null ? null : precomputed.baseVariableRows().get(precomputed.equationSystem().getVariableSet()
                                    .getVariable(br.getBus2().getNum(), AcVariableType.BUS_PHI));
                            String pack = "absent";
                            for (int e = 0; e < cbCount && idxStride > 0; e++) {
                                if (ph1 != null && ph2 != null && gd.cbIdx()[e * idxStride] == ph1 && gd.cbIdx()[e * idxStride + 2] == ph2) {
                                    double[] r = java.util.Arrays.copyOfRange(gd.cbIn(), e * 8, e * 8 + 8);
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
     * <p>That matters for the JAVA outer-loop mode, which drives OLF's own loop objects built from this
     * list: without the freezing loop it runs a strictly shorter sequence than the CPU. Measured on
     * rte6515 / LINE-2634-61, the CPU checks FreezingHvdcACEmulation, gets UNSTABLE, re-solves and then
     * re-checks every loop; the hybrid never checked it at all, and its state came out 2.405e-04 off the
     * CPU network-wide (23029 of 29832 buses over 1e-9) where the DEVICE mode was at 3.45e-13.
     *
     * <p>Recomputes the same list the SA is about to install, by the same call. The DEVICE mode does not
     * depend on it — it implements the freezing itself in {@code GpuBatchHvdcAcEmulation} — but the list
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
    /** OLF_GPU_BUS_ARRAY_CHECK=1: after each array update, run the variable walk too and require every bus's V,
     *  angle and the realistic-voltage verdict to be bit-identical. */
    private static final boolean BUS_ARRAY_CHECK = System.getenv("OLF_GPU_BUS_ARRAY_CHECK") != null;

    /**
     * Bus voltages and angles straight from the device state, and the realistic-voltage check, in one loop over the
     * buses. The device state is in the BASE row layout and every enabled bus keeps its base V and angle variables,
     * so {@code state[vRowOfBus[b]]} is exactly the value the injection put in the state vector for that bus; a
     * disabled bus has no variable in the contingency's index, and {@code AcSolverUtil.updateNetwork} never touched
     * it either. Replaces a walk over OLF's ~13k variable objects per contingency (1.9 s cold on rte6515). Falls
     * back to {@link #updateNetworkAndCheckRealistic} unless the base system is exactly one BUS_V and one BUS_PHI
     * per bus.
     */
    private static boolean updateBusesFromState(double[] state, PrecomputedBatch precomputed, LfNetwork network,
                                                AcLoadFlowParameters parameters, RunState rs) {
        Map<Variable<AcVariableType>, Integer> baseRows = precomputed.baseVariableRows();
        if (rs.busRowsFor != baseRows) {
            rs.busRowsFor = baseRows;
            int nb = network.getBuses().size();
            int[] vRow = new int[nb];
            int[] phiRow = new int[nb];
            java.util.Arrays.fill(vRow, -1);
            java.util.Arrays.fill(phiRow, -1);
            boolean usable = true;
            for (Map.Entry<Variable<AcVariableType>, Integer> e : baseRows.entrySet()) {
                Variable<AcVariableType> v = e.getKey();
                if (v.getType() == AcVariableType.BUS_V && v.getElementNum() < nb) {
                    vRow[v.getElementNum()] = e.getValue();
                } else if (v.getType() == AcVariableType.BUS_PHI && v.getElementNum() < nb) {
                    phiRow[v.getElementNum()] = e.getValue();
                } else {
                    usable = false;                          // another kind of variable: keep the exact walk
                }
            }
            for (int b = 0; b < nb && usable; b++) {
                usable = vRow[b] >= 0 && phiRow[b] >= 0;
            }
            rs.vRowOfBus = vRow;
            rs.phiRowOfBus = phiRow;
            rs.busRowsUsable = usable;
        }
        if (!rs.busRowsUsable || state == null) {
            return updateNetworkAndCheckRealistic(precomputed.equationSystem(), network, parameters, rs);
        }
        if (rs.realisticBandApplies == null || rs.realisticBandNetwork != network) {
            double minNominalV = parameters.getMinNominalVoltageRealisticVoltageCheck();
            boolean[] applies = new boolean[network.getBuses().size()];
            for (LfBus b : network.getBuses()) {
                applies[b.getNum()] = !b.isFictitious() && b.getNominalV() >= minNominalV;
            }
            rs.realisticBandApplies = applies;
            rs.realisticBandNetwork = network;
        }
        boolean[] applies = rs.realisticBandApplies;
        double minV = parameters.getMinRealisticVoltage();
        double maxV = parameters.getMaxRealisticVoltage();
        boolean trace = System.getenv("OLF_GPU_UNREALISTIC_TRACE") != null;
        int[] vRow = rs.vRowOfBus;
        int[] phiRow = rs.phiRowOfBus;
        List<LfBus> buses = network.getBuses();
        boolean unrealistic = false;
        for (int num = 0; num < vRow.length; num++) {
            LfBus bus = buses.get(num);
            if (bus.isDisabled()) {
                continue;
            }
            double value = state[vRow[num]];
            bus.setV(value);
            bus.setAngle(state[phiRow[num]]);
            if ((!unrealistic || trace) && (value < minV || value > maxV) && applies[num]) {
                unrealistic = true;
                if (trace) {
                    System.err.printf("UNREALISTIC bus=%s v=%.9f nominalV=%.3f band=[%.3f, %.3f]%n",
                            bus.getId(), value, bus.getNominalV(), minV, maxV);
                }
            }
        }
        if (BUS_ARRAY_CHECK) {
            double[] v = new double[vRow.length];
            double[] a = new double[vRow.length];
            for (int num = 0; num < vRow.length; num++) {
                v[num] = buses.get(num).getV();
                a[num] = buses.get(num).getAngle();
            }
            boolean walked = updateNetworkAndCheckRealistic(precomputed.equationSystem(), network, parameters, rs);
            rs.busArrayChecked++;
            String diff = walked != unrealistic ? "realistic verdict " + unrealistic + " vs " + walked : null;
            for (int num = 0; num < vRow.length && diff == null; num++) {
                LfBus bus = buses.get(num);
                if (Double.doubleToLongBits(v[num]) != Double.doubleToLongBits(bus.getV())
                        || Double.doubleToLongBits(a[num]) != Double.doubleToLongBits(bus.getAngle())) {
                    diff = "bus " + bus.getId() + " v " + v[num] + " vs " + bus.getV() + " angle " + a[num] + " vs " + bus.getAngle();
                }
            }
            if (diff != null) {
                rs.busArrayDiffering++;
                if (rs.busArrayFirstDiff == null) {
                    rs.busArrayFirstDiff = diff;
                }
            }
        }
        return unrealistic;
    }

    /**
     * {@code AcSolverUtil.updateNetwork} and {@link #isStateUnrealistic} in ONE walk over the variables. Both
     * walked the ~13k variables of rte6515 once per contingency (2.9 s of the GPU arm, nearly all of it memory
     * traffic), the second one with two bus lookups per variable. Only BUS_V / BUS_PHI are handled here —
     * the only kinds the batch solves — so any other kind in the system falls back to the two original
     * calls. Which buses the realistic-voltage band applies to is a per-network constant, cached per run.
     */
    private static boolean updateNetworkAndCheckRealistic(EquationSystem<AcVariableType, AcEquationType> es,
                                                          LfNetwork network, AcLoadFlowParameters parameters,
                                                          RunState rs) {
        List<Variable<AcVariableType>> vars = es.getIndex().getSortedVariablesToFind();
        for (Variable<AcVariableType> v : vars) {
            if (v.getType() != AcVariableType.BUS_V && v.getType() != AcVariableType.BUS_PHI) {
                AcSolverUtil.updateNetwork(network, es);
                return isStateUnrealistic(es, network, parameters);
            }
        }
        if (rs.realisticBandApplies == null || rs.realisticBandNetwork != network) {
            double minNominalV = parameters.getMinNominalVoltageRealisticVoltageCheck();
            boolean[] applies = new boolean[network.getBuses().size()];
            for (LfBus b : network.getBuses()) {
                applies[b.getNum()] = !b.isFictitious() && b.getNominalV() >= minNominalV;
            }
            rs.realisticBandApplies = applies;
            rs.realisticBandNetwork = network;
        }
        boolean[] applies = rs.realisticBandApplies;
        double minV = parameters.getMinRealisticVoltage();
        double maxV = parameters.getMaxRealisticVoltage();
        boolean trace = System.getenv("OLF_GPU_UNREALISTIC_TRACE") != null;
        com.powsybl.openloadflow.equations.StateVector sv = es.getStateVector();
        boolean unrealistic = false;
        for (Variable<AcVariableType> v : vars) {
            double value = sv.get(v.getRow());
            LfBus bus = network.getBus(v.getElementNum());
            if (v.getType() == AcVariableType.BUS_PHI) {
                bus.setAngle(value);
                continue;
            }
            bus.setV(value);
            if (!unrealistic || trace) {
                if ((value < minV || value > maxV) && applies[v.getElementNum()]) {
                    unrealistic = true;
                    if (trace) {
                        System.err.printf("UNREALISTIC bus=%s v=%.9f nominalV=%.3f band=[%.3f, %.3f]%n",
                                bus.getId(), value, bus.getNominalV(), minV, maxV);
                    }
                }
            }
        }
        return unrealistic;
    }

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
