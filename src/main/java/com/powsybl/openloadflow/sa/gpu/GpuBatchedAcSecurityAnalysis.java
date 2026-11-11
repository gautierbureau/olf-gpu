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
import com.powsybl.contingency.strategy.ConditionalActions;
import com.powsybl.contingency.strategy.OperatorStrategy;
import com.powsybl.contingency.strategy.condition.AbstractThresholdCondition;
import com.powsybl.contingency.strategy.condition.BranchThresholdCondition;
import com.powsybl.contingency.strategy.condition.InjectionThresholdCondition;
import com.powsybl.contingency.strategy.condition.ThreeWindingsTransformerThresholdCondition;
import com.powsybl.iidm.network.Network;
import com.powsybl.math.matrix.MatrixFactory;
import com.powsybl.openloadflow.ac.AcLoadFlowContext;
import com.powsybl.openloadflow.ac.AcLoadFlowParameters;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.solver.AcSolverUtil;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonFactory;
import com.powsybl.openloadflow.ac.solver.gpu.GpuDriverOuterLoopEngine;
import com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphsonFactory;
import com.powsybl.openloadflow.ac.solver.gpu.batch.BatchScenarioResult;
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
import com.powsybl.openloadflow.util.gpu.GpuTrace;
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

    /** The device's final mismatch (infinity norm) of every contingency it solved in the last run, converged or not:
     *  for one that ran its Newton budget out, the only trace of the pass's last fill (OLF reports no network result
     *  for it). Reset with the counters above. */
    static final java.util.Map<String, Double> LAST_DEVICE_FINAL_MISMATCH = new java.util.concurrent.ConcurrentHashMap<>();

    /** How many contingencies of the last run had their loss distribution answered from the batch's replay
     *  instead of OLF's host call ({@link ReplayedLossDistribution}); reset with the counter above. */
    static final java.util.concurrent.atomic.AtomicInteger LAST_REPLAYED_LOSS_COUNT =
            new java.util.concurrent.atomic.AtomicInteger();

    /** How many base-state restores of the last run were targeted (see restoreBaseState). */
    static final java.util.concurrent.atomic.AtomicInteger LAST_TARGETED_RESTORE_COUNT =
            new java.util.concurrent.atomic.AtomicInteger();

    /** How many contingencies of the last run were answered from their device state without writing it back
     *  (answeredFromState); reset with the counters above. */
    static final java.util.concurrent.atomic.AtomicInteger LAST_ANSWERED_FROM_STATE_COUNT =
            new java.util.concurrent.atomic.AtomicInteger();

    /** How many contingencies of the last run were answered without applying them to the network (resultWithoutNetwork);
     *  reset with the counters above. */
    static final java.util.concurrent.atomic.AtomicInteger LAST_ANSWERED_WITHOUT_NETWORK_COUNT =
            new java.util.concurrent.atomic.AtomicInteger();

    /** How many post-strategy restores of the last run were targeted (see restorePostContingencyState). */
    static final java.util.concurrent.atomic.AtomicInteger LAST_TARGETED_STRATEGY_RESTORE_COUNT =
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
            GpuSaRunState run = runState.get();
            Double replayed = run.precomputed.states().containsKey(contingency.getId())
                    ? run.precomputed.preDistributedActivePower().get(contingency.getId())
                    : null;
            if (replayed == null || loadFlowParametersOverride != null
                    || targetedByOperatorStrategy(run.operatorStrategies, contingency.getId())) {
                return delegate.run(network, lfContingency, contingency, securityAnalysisParameters,
                        loadFlowParametersOverride, reportNode);
            }
            double mismatch = lfContingency.getActivePowerLoss();
            if (Math.abs(mismatch) > 0) {                    // a replayed value exists only with distributed slack on
                com.powsybl.openloadflow.util.Reports.reportContingencyActivePowerLossDistribution(reportNode,
                        mismatch * PerUnit.SB, (mismatch - replayed) * PerUnit.SB);
            }
            LAST_REPLAYED_LOSS_COUNT.incrementAndGet();
            run.lossReplayedFor = contingency.getId();
            return replayed;
        }
    }

    @Override
    protected void restoreBaseState(com.powsybl.openloadflow.network.NetworkState networkState, LfNetwork network,
                                    com.powsybl.openloadflow.network.LfContingency lfContingency) {
        GpuSaRunState run = runState.get();
        boolean targeted = lfContingency.getId().equals(run.lossReplayedFor)
                && lfContingency.getId().equals(run.deviceAnsweredFor);
        run.lossReplayedFor = null;
        run.deviceAnsweredFor = null;
        run.deviceStateInjectedFor = null;
        List<LfBranch> automationSystemSwitched = run.automationSystemSwitched;
        run.automationSystemSwitched = List.of();
        List<LfBranch> deviceRetapped = new java.util.ArrayList<>(run.deviceTapBase.keySet());
        run.deviceTapBase.clear();
        run.deviceRetapped = List.of();
        if (targeted) {
            java.util.Set<LfBus> buses = touchedBuses(network, lfContingency, run);
            java.util.Collection<LfBranch> branches = lfContingency.getDisabledNetwork().getBranchesStatus().keySet();
            // the susceptance, ratio and phase state variables the network update wrote: a later contingency left to
            // the CPU starts its Newton from them, and holds them as targets where its control stays off
            controlStateElements(network, run);
            buses.addAll(run.controlStateBuses);
            if (!automationSystemSwitched.isEmpty() || !deviceRetapped.isEmpty() || !run.controlStateBranches.isEmpty()) {
                branches = new java.util.ArrayList<>(branches);
                branches.addAll(automationSystemSwitched);
                branches.addAll(deviceRetapped);              // back on their base tap
                branches.addAll(run.controlStateBranches);
            }
            // the voltage loop over every bus is redundant when the next contingency is device-answered: its injection
            // writes every live bus (restoreStaleVoltages covers the rest)
            networkState.restore(buses, branches, false);
            run.staleVoltages = networkState;
            LAST_TARGETED_RESTORE_COUNT.incrementAndGet();
        } else {
            run.staleVoltages = null;                         // the full restore below puts every voltage back
            super.restoreBaseState(networkState, network, lfContingency);
        }
    }

    /**
     * After a strategy answered on the device with tap actions only, the network differs from its post-contingency
     * state in the retapped branches and in what the injection wrote: the bus voltages, restored lazily like the base
     * state's (restoreStaleVoltages), and the shunt B and transformer alpha/rho state variables, which force OLF's full
     * restore when the equation system has any. Every other strategy (CPU path, topology or injection actions) gets
     * the full restore.
     */
    @Override
    protected void restorePostContingencyState(com.powsybl.openloadflow.network.NetworkState postContingencyNetworkState,
                                               LfNetwork network, com.powsybl.openloadflow.network.LfContingency lfContingency) {
        GpuSaRunState run = runState.get();
        List<LfBranch> retapped = run.strategyRetappedOnly;
        run.strategyRetappedOnly = null;
        if (retapped == null || run.precomputed == null || run.precomputed.equationSystem() == null
                || hasControlStateVariables(run.precomputed.equationSystem())) {
            run.staleVoltages = null;                         // the full restore puts every voltage back
            super.restorePostContingencyState(postContingencyNetworkState, network, lfContingency);
            return;
        }
        postContingencyNetworkState.restore(List.of(), retapped, false);
        run.staleVoltages = postContingencyNetworkState;
        LAST_TARGETED_STRATEGY_RESTORE_COUNT.incrementAndGet();
    }

    /** True when updateNetwork writes more than bus voltages: shunt B or transformer alpha/rho are state variables. */
    private static boolean hasControlStateVariables(
            com.powsybl.openloadflow.equations.EquationSystem<com.powsybl.openloadflow.ac.equations.AcVariableType,
                    com.powsybl.openloadflow.ac.equations.AcEquationType> equationSystem) {
        for (var v : equationSystem.getIndex().getSortedVariablesToFind()) {
            switch (v.getType()) {
                case SHUNT_B, BRANCH_ALPHA1, BRANCH_RHO1 -> {
                    return true;
                }
                default -> {
                }
            }
        }
        return false;
    }

    /** Drops the run's file-backed state map, if its states went to one (GpuStateStore). */
    private static void closeStateStore(GpuSaRunState run) {
        if (run != null && run.precomputed != null && run.precomputed.states() instanceof GpuStateStore store) {
            store.close();
        }
    }

    /** Ends a lazy voltage restore: {@code onlyDisabled} null restores every bus voltage, otherwise only those buses
     *  (the disabled ones of a device-answered contingency, whose voltages its injection does not write). */
    private static void restoreStaleVoltages(GpuSaRunState run, java.util.Collection<LfBus> onlyDisabled) {
        if (run.staleVoltages == null) {
            return;
        }
        if (onlyDisabled == null) {
            run.staleVoltages.restoreVoltages();
        } else {
            run.staleVoltages.restoreVoltages(onlyDisabled);
        }
        run.staleVoltages = null;
    }

    /**
     * The buses a device-solved contingency changed beyond their voltage: what LfContingency.apply touched — the
     * buses it disabled and those of the loads, generators and shunts it lost or shifted — closed over their
     * voltage and reactive-power control groups, because enabling or disabling a controller recomputes its
     * whole group's reactive targets.
     */
    private static java.util.Set<LfBus> touchedBuses(LfNetwork network, com.powsybl.openloadflow.network.LfContingency c,
                                                     GpuSaRunState run) {
        java.util.Set<LfBus> touched = new java.util.LinkedHashSet<>(c.getDisabledNetwork().getBuses());
        c.getLostLoads().keySet().forEach(l -> touched.add(l.getBus()));
        c.getLostGenerators().forEach(g -> touched.add(g.getBus()));
        if (!c.getShuntsShift().isEmpty()) {
            java.util.Map<com.powsybl.openloadflow.network.LfShunt, LfBus> shuntBus = shuntBus(network, run);
            c.getShuntsShift().keySet().forEach(shunt -> {
                LfBus b = shuntBus.get(shunt);
                if (b != null) {
                    touched.add(b);
                }
            });
        }
        java.util.Set<LfBus> closed = new java.util.LinkedHashSet<>(touched);
        for (LfBus b : touched) {
            b.getGeneratorVoltageControl().ifPresent(voltageControl -> {
                closed.add(voltageControl.getControlledBus());
                closed.addAll(voltageControl.getMergedControllerElements());
            });
            b.getGeneratorReactivePowerControl().ifPresent(reactivePowerControl -> closed.addAll(reactivePowerControl.getControllerBuses()));
        }
        return closed;
    }

    /** Shunt -> its bus (LfShunt does not know its bus), built once per run. */
    private static java.util.Map<com.powsybl.openloadflow.network.LfShunt, LfBus> shuntBus(LfNetwork network, GpuSaRunState run) {
        if (run.shuntBus == null) {
            run.shuntBus = new java.util.IdentityHashMap<>();
            for (LfBus b : network.getBuses()) {
                b.getShunt().ifPresent(shunt -> run.shuntBus.put(shunt, b));
                b.getControllerShunt().ifPresent(shunt -> run.shuntBus.put(shunt, b));
                b.getSvcShunt().ifPresent(shunt -> run.shuntBus.put(shunt, b));
            }
        }
        return run.shuntBus;
    }

    /**
     * Sets the run's control state elements: what AcSolverUtil.updateNetwork writes beyond the bus voltages when the
     * base system holds those variables (GpuPostContingencyState.updateNetworkAndCheckRealistic) — a SHUNT_B into its
     * controller shunt's susceptance, restored with the shunt's bus, and a BRANCH_ALPHA1 or BRANCH_RHO1 into its
     * branch's phase or ratio override. From the base system's variables, which no contingency adds to; once per
     * extraction.
     */
    private static void controlStateElements(LfNetwork network, GpuSaRunState run) {
        Map<com.powsybl.openloadflow.equations.Variable<com.powsybl.openloadflow.ac.equations.AcVariableType>, Integer> baseRows =
                run.precomputed.baseVariableRows();
        if (run.controlStateFor == baseRows) {
            return;
        }
        java.util.Set<LfBus> buses = new java.util.LinkedHashSet<>();
        java.util.Set<LfBranch> branches = new java.util.LinkedHashSet<>();
        for (var v : baseRows.keySet()) {
            switch (v.getType()) {
                case SHUNT_B -> buses.add(Objects.requireNonNull(shuntBus(network, run).get(network.getShunt(v.getElementNum()))));
                case BRANCH_ALPHA1, BRANCH_RHO1 -> branches.add(network.getBranch(v.getElementNum()));
                default -> {
                }
            }
        }
        run.controlStateBuses = List.copyOf(buses);
        run.controlStateBranches = List.copyOf(branches);
        run.controlStateFor = baseRows;
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
    private int[] flowReadSet(GpuSaRunState run, String contingencyId) {
        if (run.violationScreen == null || run.candidates == null || flowsMonitored()
                || targetedByOperatorStrategy(run.operatorStrategies, contingencyId)
                || GpuTrace.SA_RESID.isOn()
                || GpuTrace.VIOL_CROSSCHECK.isOn()
                || GpuTrace.STATE_VIEW_CHECK.isOn()) {
            return null;
        }
        GpuLimitTable table = run.limitTable;
        java.util.BitSet nums = new java.util.BitSet();
        int[] candidates = run.candidates.get(contingencyId);
        if (candidates == null) {
            return null;                                     // the detection would screen it on the host: read all
        }
        for (int c : candidates) {
            if (c < table.limitCount()) {                    // bus records come after the limits: they read bus V only
                nums.set(table.branches()[table.groupBranch(table.limitGroup(c))].getNum());
            }
        }
        for (int b : run.uncoveredBranches) {
            nums.set(table.branches()[b].getNum());
        }
        run.sideOpened.forEach(branch -> nums.set(branch.getNum()));
        run.automationSystemSwitched.forEach(branch -> nums.set(branch.getNum()));
        run.deviceRetapped.forEach(branch -> nums.set(branch.getNum()));
        return nums.stream().toArray();
    }

    /**
     * The branches whose flows the detection of a device-answered operator strategy reads (see flowReadSet): those of
     * the limits the screen flagged for the pair, those it does not cover, and {@code scanInFull}; null for "refresh
     * them all" when the screen is not available for the pair or another reader of flows is on (a branch or
     * three-windings transformer state monitor, the diagnostics, another strategy's flow condition).
     */
    private int[] strategyFlowReadSet(GpuSaRunState run, String pairId, String contingencyId, OperatorStrategy strategy,
                                      java.util.Collection<LfBranch> scanInFull) {
        if (run.violationScreen == null || run.candidates == null || flowsMonitored()
                || anotherStrategyReadsFlows(run, contingencyId, strategy)
                || GpuTrace.SA_RESID.isOn() || GpuTrace.VIOL_CROSSCHECK.isOn()) {
            return null;
        }
        int[] candidates = run.candidates.get(pairId);
        if (candidates == null) {
            return null;                                     // the detection would screen it on the host: read all
        }
        GpuLimitTable table = run.limitTable;
        java.util.BitSet nums = new java.util.BitSet();
        for (int c : candidates) {
            if (c < table.limitCount()) {                    // bus records come after the limits: they read bus V only
                nums.set(table.branches()[table.groupBranch(table.limitGroup(c))].getNum());
            }
        }
        for (int b : run.uncoveredBranches) {
            nums.set(table.branches()[b].getNum());
        }
        scanInFull.forEach(branch -> nums.set(branch.getNum()));
        return nums.stream().toArray();
    }

    /**
     * Gives the network the branches the device's AutomationSystem phase switched for a contingency, as OLF's loop leaves
     * them: an opened branch disabled, a closed one enabled, and nothing else, the device switching only where every bus
     * stays connected. The contingency leaves an operated branch as it is in the base (GpuBatchQualification
     * .refuseAutomationSystemShapes), so each one changes state; restoreBaseState puts them back.
     */
    private static List<LfBranch> applyAutomationSystemSwitches(LfNetwork network, int[] switchedBranchNums) {
        if (switchedBranchNums == null) {
            return List.of();
        }
        List<LfBranch> switched = new java.util.ArrayList<>(switchedBranchNums.length);
        for (int num : switchedBranchNums) {
            LfBranch branch = network.getBranch(num);
            branch.setDisabled(!branch.isDisabled());
            switched.add(branch);
        }
        com.powsybl.openloadflow.ac.outerloop.gpu.GpuAutomationSystemOuterLoop.DEVICE_SWITCHED_BRANCHES.add(switched.size());
        return switched;
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

    /** The current contingency's side-opened and device-retapped branches as limit-table indices (null when none has
     *  limits): the screen read them closed and at their base tap, so their records are dropped and the host scans them
     *  in full, in table order. */
    private static java.util.BitSet sideOpenedTableIndices(GpuSaRunState run, LfNetwork network) {
        if (run.deviceRetapped.isEmpty()) {
            return tableIndices(run, network, run.sideOpened);
        }
        java.util.Set<LfBranch> scanInFull = new java.util.LinkedHashSet<>(run.sideOpened);
        scanInFull.addAll(run.deviceRetapped);
        return tableIndices(run, network, scanInFull);
    }

    /** These branches as limit-table indices (null when none has limits): the host scans them in full, in table order,
     *  instead of trusting the screen, which evaluates every branch in the base topology with its base parameters. */
    private static java.util.BitSet tableIndices(GpuSaRunState run, LfNetwork network, java.util.Collection<LfBranch> branches) {
        if (branches.isEmpty()) {
            return null;
        }
        GpuLimitTable table = run.limitTable;
        if (run.tableIndexOfBranchNum == null) {
            run.tableIndexOfBranchNum = new int[network.getBranches().size()];
            java.util.Arrays.fill(run.tableIndexOfBranchNum, -1);
            for (int b = 0; b < table.branches().length; b++) {
                run.tableIndexOfBranchNum[table.branches()[b].getNum()] = b;
            }
        }
        java.util.BitSet sideOpened = new java.util.BitSet();
        for (LfBranch branch : branches) {
            int b = run.tableIndexOfBranchNum[branch.getNum()];
            if (b >= 0 && table.groupStart(b) != table.groupEnd(b)) {
                sideOpened.set(b);
            }
        }
        return sideOpened.isEmpty() ? null : sideOpened;
    }

    /**
     * Whether a device-answered contingency's result is built from its state array without writing the state into OLF's
     * objects (P1 of {@code docs/device-final-results-plan.md}). The detection then reads every value through
     * {@link GpuStateView}, the same doubles OLF's accessors return once the state is written back (checked bit for bit
     * by OLF_GPU_STATE_VIEW_CHECK), and the injection, its flow refresh and the bus writes are skipped. Not when
     * something reads OLF's objects for this contingency: a diagnostic of the written-back state (OLF_GPU_SA_RESID,
     * OLF_GPU_VIOL_CROSSCHECK), an operator strategy on it, a state monitor asking for flows, OLF's voltage-angle check
     * (it reads the buses' angles), a contingency with its own load flow parameters; nor when the view cannot stand for
     * the objects on this run ({@link GpuStateView#usable()}). A bus-results monitor is served by writing the buses'
     * voltages and angles alone.
     */
    private boolean answeredFromState(GpuSaRunState run, Contingency contingency, LfNetwork network,
                                      PrecomputedBatch precomputed) {
        if (GpuTrace.SA_RESID.isOn() || GpuTrace.VIOL_CROSSCHECK.isOn()
                || targetedByOperatorStrategy(run.operatorStrategies, contingency.getId())
                || flowsMonitored()
                || !network.getVoltageAngleLimits().isEmpty()
                || contingency.getExtension(com.powsybl.openloadflow.sa.extensions.ContingencyLoadFlowParameters.class) != null) {
            return false;
        }
        return stateView(run, network, precomputed).usable();
    }

    /** The run's view of the device states, for the precomputed batch's base rows. */
    private static GpuStateView stateView(GpuSaRunState run, LfNetwork network, PrecomputedBatch precomputed) {
        if (run.stateView == null || !run.stateView.isFor(precomputed.baseVariableRows())) {
            run.stateView = new GpuStateView(network, precomputed.baseVariableRows());
        }
        return run.stateView;
    }

    /** True when a state monitor asks for bus results, which read the buses' voltages and angles. */
    private boolean busesMonitored() {
        java.util.List<com.powsybl.security.monitor.StateMonitor> monitors = new java.util.ArrayList<>();
        monitors.add(monitorIndex.getAllStateMonitor());
        monitors.add(monitorIndex.getNoneStateMonitor());
        monitors.addAll(monitorIndex.getSpecificStateMonitors().values());
        return monitors.stream().anyMatch(m -> m != null && !m.getVoltageLevelIds().isEmpty());
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

    /**
     * Whether another strategy on the contingency has a stage conditioned on a flow (a branch or three-windings
     * transformer threshold). OLF evaluates a strategy's condition on the network the previous strategy's load flow
     * left: the restore between them puts the network's elements back, not the state that load flow solved, so the
     * condition reads every flow of that state, and a device-answered strategy before it must have refreshed them all.
     * Which strategy runs before which is left out: refreshing for one that runs after the conditioned one costs only
     * the refresh.
     */
    private static boolean anotherStrategyReadsFlows(GpuSaRunState run, String contingencyId, OperatorStrategy strategy) {
        for (OperatorStrategy other : run.operatorStrategies) {
            if (other.getId().equals(strategy.getId()) || !targets(other, contingencyId)) {
                continue;
            }
            for (ConditionalActions stage : other.getConditionalActions()) {
                if (stage.getCondition() instanceof BranchThresholdCondition
                        || stage.getCondition() instanceof ThreeWindingsTransformerThresholdCondition) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Whether a strategy on the contingency has a stage conditioned on a threshold: a branch's or a three-windings
     * transformer's flow, or a generator's target. ThresholdConditionEvaluator reads them on the network the
     * contingency's load flow left; after a load flow that ran out of Newton iterations the flows are those of its last
     * iterate (NewtonRaphson leaves the state vector there), which only OLF's own run reproduces.
     */
    private static boolean strategyConditionReadsNetwork(GpuSaRunState run, String contingencyId) {
        return strategyConditionOfType(run, contingencyId, AbstractThresholdCondition.class);
    }

    /** Whether a strategy on the contingency has a stage conditioned on a generator's active power or target P. */
    private static boolean strategyConditionReadsGeneratorTargets(GpuSaRunState run, String contingencyId) {
        return strategyConditionOfType(run, contingencyId, InjectionThresholdCondition.class);
    }

    private static boolean strategyConditionOfType(GpuSaRunState run, String contingencyId,
                                                   Class<? extends com.powsybl.contingency.strategy.condition.Condition> type) {
        for (OperatorStrategy strategy : run.operatorStrategies) {
            if (targets(strategy, contingencyId)) {
                for (ConditionalActions stage : strategy.getConditionalActions()) {
                    if (type.isInstance(stage.getCondition())) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    /**
     * Gives the participating generators the target P the device's distribution left them for this contingency, which
     * OLF's own post-contingency load flow leaves on them: a strategy condition on a generator's active power reads it
     * (ThresholdConditionEvaluator), and so does one on its target P once OLF copies it to the initial target before a
     * single strategy. The injection writes the state's voltages only, and the network holds the loss pre-distribution
     * alone. A generator the contingency lost keeps what OLF's contingency left it. OLF's full restore (a contingency
     * a strategy targets does not take the targeted one) puts the targets back.
     */
    private static void applyDeviceGeneratorTargets(LfNetwork network, BatchScenarioResult.GeneratorTargets targets) {
        if (targets == null) {
            return;
        }
        for (int g = 0; g < targets.generatorIds().length; g++) {
            com.powsybl.openloadflow.network.LfGenerator generator = network.getGeneratorById(targets.generatorIds()[g]);
            if (generator != null && !generator.isDisabled() && !Double.isNaN(targets.targetP()[g])) {
                generator.setTargetP(targets.targetP()[g]);
            }
        }
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
        var contingencyContext = strategy.getContingencyContext();
        return switch (contingencyContext.getContextType()) {
            case ALL, ONLY_CONTINGENCIES -> true;
            case SPECIFIC -> contingencyId.equals(contingencyContext.getContingencyId());
            case NONE -> false;
        };
    }

    /**
     * What a strategy pair's device answer reports as distributed active power, or null when the pair is not answered
     * on the device: no state, the batch not ready, or its distribution not recorded under distributed slack. OLF
     * reports the distribution of the strategy load flow alone (no loss pre-distribution, counted from the parent's
     * distributed targets); the device records it when one solve covered the pair's chunk.
     */
    private static Double pairDistributedOnDevice(GpuSaRunState run, String contingencyId, String pairId,
                                                  SecurityAnalysisParameters securityAnalysisParameters) {
        PrecomputedBatch precomputed = run.precomputed;
        // an existence test: with the state store a get reads the whole state from its file
        if (!precomputed.isReady() || !precomputed.states().containsKey(pairId)) {
            return null;
        }
        // a contingency's own load flow parameters, when they hold for its strategies too, decide the distribution
        boolean distributes = run.ownDistributedSlack.getOrDefault(contingencyId,
                securityAnalysisParameters.getLoadFlowParameters().isDistributedSlack());
        return distributes ? precomputed.distributedActivePower().get(pairId) : Double.valueOf(0.0);
    }

    /**
     * A strategy pair the device does not answer runs OLF's strategy load flow from the network its parent left. The
     * parent's injected device state holds the bus voltages only, not what OLF's post-contingency run leaves behind
     * (the generators' distributed targets, the PV/PQ switches, tap moves...), so such a parent is solved on the CPU.
     * Measured: an absolute GeneratorAction pair routed to the CPU after a device parent reported -43.62 MW of
     * distributed power where OLF reports -42.67.
     */
    private static boolean everyStrategyAnsweredOnDevice(GpuSaRunState run, String contingencyId,
                                                         SecurityAnalysisParameters securityAnalysisParameters) {
        for (OperatorStrategy strategy : run.operatorStrategies) {
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
                if (pairDistributedOnDevice(run, contingencyId, GpuStrategyComposer.pairId(contingencyId, strategy, actionIds),
                        securityAnalysisParameters) == null) {
                    return false;
                }
            }
        }
        return true;
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
        GpuSaRunState run = new GpuSaRunState();
        runState.set(run);
        run.loadFlowParameters = securityAnalysisParameters.getLoadFlowParameters();
        run.securityAnalysisParameters = securityAnalysisParameters;
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
            if (GpuTrace.BATCH_TRACE.isOn()) {
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
        // SLOPE-SVC rows (row mode BUS_TARGET_V_WITH_SLOPE) solve on the sequential path since the batch-of-1 route
        // (2026-09-08) — no CPU-base condition anymore.
        // CONTROLLER-mode phase control (row mode BRANCH_TARGET_P) solves on the sequential path since the
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
        run.lfNetwork = lfNetwork;
        run.contingencies = propagatedContingencies;
        run.operatorStrategies = operatorStrategies != null ? operatorStrategies : List.of();
        run.actions = actions != null ? actions : List.of();
        run.precomputed = PrecomputedBatch.empty();
        run.batchDistributedSlack = scope.distributedSlack();
        run.batchReactiveLimits = scope.reactiveLimits();
        if (scope.batchDisabled()) {                         // outer-loop guard: also skip the solver swap below
            run.lfNetwork = null;
            run.contingencies = null;
        }
        run.batchSlackDistTol = scope.slackDistTol();
        run.batchReactiveLimitTol = scope.reactiveLimitTol();
        run.batchConvEpsPerEq = scope.convEpsPerEq();
        run.batchConfig = scope.batchConfig();
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
        ContingencyActivePowerLossDistribution lossDistribution = new ReplayedLossDistribution(contingencyActivePowerLossDistribution);
        try {
            return super.runSimulations(lfNetwork, toSimulate, acParameters, securityAnalysisParameters,
                    operatorStrategies, actions, limitReductions, lossDistribution);
        } finally {
            restoreStaleVoltages(runState.get(), null);      // the network leaves the loop in its base state
            closeStateStore(runState.get());                 // a file-backed state map is dropped with the run
            runState.remove();                               // pooled partition threads must not leak a run's state
            if (GpuTrace.PHASE_TIME.isOn()) {
                // CUMULATIVE over the JVM, never cleared: threadCount > 1 runs one partition per
                // thread through this same static map, and clearing here would hand whichever
                // partition finished first the whole run's numbers and the rest almost nothing.
                // Read the LAST block printed — like JNI_PHASE_TIME, it is the running total.
                PHASE_NS.forEach((k, v) ->
                        System.err.printf("SA_PHASE_TIME  %-40s %6d ms (cumulative)%n", k, v.sum() / 1_000_000));
            }
            if (GpuTrace.VIOL_CROSSCHECK.isOn()) {
                System.err.println("VIOL_CROSSCHECK contingencies compared against the stock detection: "
                        + GpuSaDiagnostics.CROSSCHECKED.sum());
            }
            if (GpuTrace.STATE_VIEW_CHECK.isOn()) {
                System.err.println(GpuStateViewCheck.summary());   // cumulative over the JVM, like the phase times
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
        if (GpuTrace.PHASE_TIME.isOn()) {
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
        detectViolations(network, manager, limitReductions, reference, saParameters, contingencyId, null);
    }

    /** {@code scanInFull}: the branches the screen cannot be trusted on for this scenario, scanned in full on the host
     *  (null: the current contingency's side-opened branches). */
    private void detectViolations(LfNetwork network, LimitViolationManager manager, List<LimitReduction> limitReductions,
                                  LimitViolationManager reference, SecurityAnalysisParameters saParameters,
                                  String contingencyId, java.util.Collection<LfBranch> scanInFull) {
        detectViolations(network, manager, limitReductions, reference, saParameters, contingencyId, scanInFull, null);
    }

    /** As above, the values read from {@code fromState} through the run's {@link GpuStateView} when it is non-null
     *  (a contingency answered without writing its state back), else from OLF's objects. */
    private void detectViolations(LfNetwork network, LimitViolationManager manager, List<LimitReduction> limitReductions,
                                  LimitViolationManager reference, SecurityAnalysisParameters saParameters,
                                  String contingencyId, java.util.Collection<LfBranch> scanInFull, double[] fromState) {
        detectViolations(network, manager, limitReductions, reference, saParameters, contingencyId, scanInFull, fromState,
                GpuContingencyTopology.NETWORK);
    }

    /** As above, the branches, sides and buses out of service as {@code topology} says (a contingency answered without
     *  being applied). */
    private void detectViolations(LfNetwork network, LimitViolationManager manager, List<LimitReduction> limitReductions,
                                  LimitViolationManager reference, SecurityAnalysisParameters saParameters,
                                  String contingencyId, java.util.Collection<LfBranch> scanInFull, double[] fromState,
                                  GpuContingencyTopology topology) {
        GpuSaRunState run = runState.get();
        GpuStateView view = fromState != null ? run.stateView : null;
        if (run.limitScanner == null) {
            GpuLimitTable table = GpuLimitTable.build(network, LimitReductionManager.create(limitReductions));
            run.limitScanner = new GpuLimitScanner(table);
            run.limitTable = table;
            if (run.precomputed.gpuData() != null) {
                run.violationScreen = GpuViolationScreen.build(network, table, run.precomputed.gpuData(),
                        run.precomputed.elementMapper(), run.precomputed.equationSystem(),
                        run.precomputed.baseVariableRows());
                run.uncoveredBranches = uncoveredBranches(table, run.violationScreen);
                // Screen the WHOLE run at once: the states are all precomputed already, and one
                // device call per 512 scenarios is what makes the kernel worth its upload.
                long tScreen = System.nanoTime();
                run.candidates = run.violationScreen.screenAll(run.precomputed.states(),
                        run.precomputed.baseVariableRows().size());
                PHASE_NS.computeIfAbsent("  violation screen (all scenarios)",
                        k -> new java.util.concurrent.atomic.LongAdder()).add(System.nanoTime() - tScreen);
                if (GpuTrace.PHASE_TIME.isOn()) {
                    System.err.println("VIOL_SCREEN scenarios=" + run.candidates.size()
                            + " screened on the device=" + run.violationScreen.onDevice()
                            + " overflowed to the host scan=" + run.violationScreen.overflows()
                            + " in " + (System.nanoTime() - tScreen) / 1_000_000 + " ms");
                }
            }
        }
        java.util.BitSet sideOpened = run.violationScreen == null ? null
                : scanInFull != null ? tableIndices(run, network, scanInFull) : sideOpenedTableIndices(run, network);
        if (run.violationScreen != null) {
            int[] candidates = run.candidates.get(contingencyId);
            run.limitScanner.detectFromCandidates(network, manager,
                    candidates != null ? candidates : run.violationScreen.screen(run.precomputed.states().get(contingencyId)),
                    run.violationScreen, run.uncoveredBranches, sideOpened, view, fromState, topology);
        } else {
            run.limitScanner.detectViolations(network, manager, view, fromState, topology);
        }
        if (GpuTrace.VIOL_CROSSCHECK.isOn()) {
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
        GpuSaRunState run = runState.get();
        if (run.lfNetwork != null) {                          // what a contingency answered without the network compares with
            run.baseExcludedSlackBuses = excludedSlackBuses(run.lfNetwork);
        }
        long tPre = System.nanoTime();
        closeStateStore(run);                                 // a previous batch's file-backed states
        GpuBatchPrecomputer.SolveGroup runGroup = GpuBatchPrecomputer.SolveGroup.run(new GpuBatchPrecomputer.SolveSettings(
                run.batchDistributedSlack, run.batchReactiveLimits, run.batchReactiveLimitTol, run.batchSlackDistTol,
                run.batchConvEpsPerEq, run.batchConfig));
        java.util.Set<String> leftToCpu = new java.util.HashSet<>();
        List<GpuBatchPrecomputer.SolveGroup> groups = run.lfNetwork != null && run.contingencies != null
                ? solveGroups(run, parameters, runGroup, leftToCpu) : List.of(runGroup);
        run.precomputed = GpuBatchPrecomputer.precompute(network, run.lfNetwork, run.context,
                run.contingencies, run.operatorStrategies, run.actions, groups, leftToCpu, reportNode);
        PHASE_NS.computeIfAbsent("precompute (extraction + batched solves)", k -> new java.util.concurrent.atomic.LongAdder())
                .add(System.nanoTime() - tPre);
        if (run.precomputed.isReady()) {
            int solved = run.precomputed.states().size();
            LAST_GPU_PRECOMPUTED_COUNT.updateAndGet(prev -> Math.max(prev, 0) + solved);
            LAST_GPU_PRECOMPUTED_IDS.addAll(run.precomputed.states().keySet());
        }
        // The engine's solver reverts to the stock CPU Newton-Raphson after the base LF: single-system GPU
        // solves are not where the device wins (its parallelism is the batch), and the CPU path parallelizes
        // under threadCount. The base LF has already run at this point (on the GPU, absorbing CUDA init).
        if (parameters.getSolverFactory() instanceof GpuNewtonRaphsonFactory && run.loadFlowParameters != null) {
            parameters.setSolverFactory(new NewtonRaphsonFactory(), run.loadFlowParameters);
        }
    }

    /**
     * The precompute's solve groups: the run's, and one per distinct set of a contingency's own load flow parameters
     * (ContingencyLoadFlowParameters), run with the outer loops OLF runs that contingency with. They are OLF's own:
     * applySpecificContingencyParameters on the context's parameters, put back after as OLF's resetter does, from the
     * generic contingency parameters OLF passes it (applyGenericContingencyParameters: the frozen AC emulation). A group
     * whose loops the device does not run leaves its contingencies to the CPU ({@code leftToCpu}). The strategies of a
     * contingency whose parameters hold for its strategies too run in its group; with scope CONTINGENCY_ONLY they run
     * with the run's loops from the contingency's state, which another solve holds, so none is composed.
     */
    private List<GpuBatchPrecomputer.SolveGroup> solveGroups(GpuSaRunState run, AcLoadFlowParameters parameters,
                                                             GpuBatchPrecomputer.SolveGroup runGroup,
                                                             java.util.Set<String> leftToCpu) {
        Map<List<Object>, List<Contingency>> byParameters = new java.util.LinkedHashMap<>();
        for (PropagatedContingency propagatedContingency : run.contingencies) {
            Contingency contingency = propagatedContingency.getContingency();
            var own = contingency.getExtension(com.powsybl.openloadflow.sa.extensions.ContingencyLoadFlowParameters.class);
            if (own != null) {
                byParameters.computeIfAbsent(java.util.Arrays.asList(own.isDistributedSlack().orElse(null),
                        own.isAreaInterchangeControl().orElse(null), own.getBalanceType().orElse(null),
                        own.getOuterLoopNames().orElse(null)), k -> new java.util.ArrayList<>()).add(contingency);
            }
        }
        List<GpuBatchPrecomputer.SolveGroup> groups = new java.util.ArrayList<>(List.of(runGroup));
        if (byParameters.isEmpty()) {
            return groups;
        }
        var loadFlowParameters = run.securityAnalysisParameters.getLoadFlowParameters();
        var contingencyOpenLoadFlowParameters = com.powsybl.openloadflow.OpenLoadFlowParameters.get(loadFlowParameters);
        if (OpenSecurityAnalysisParameters.getOrDefault(run.securityAnalysisParameters).isStartWithFrozenACEmulation()) {
            contingencyOpenLoadFlowParameters = com.powsybl.openloadflow.OpenLoadFlowParameters.clone(contingencyOpenLoadFlowParameters);
            contingencyOpenLoadFlowParameters.setStartWithFrozenACEmulation(true);
        }
        List<com.powsybl.openloadflow.ac.outerloop.AcOuterLoop> runLoops = List.copyOf(parameters.getOuterLoops());
        for (List<Contingency> sameParameters : byParameters.values()) {
            var own = sameParameters.getFirst().getExtension(com.powsybl.openloadflow.sa.extensions.ContingencyLoadFlowParameters.class);
            GpuSaRunScope.Scope scope;
            try {
                applySpecificContingencyParameters(parameters, own, loadFlowParameters, contingencyOpenLoadFlowParameters);
                scope = GpuSaRunScope.evaluate(run.lfNetwork, parameters, run.securityAnalysisParameters,
                        own.getBalanceType(loadFlowParameters), true);
            } finally {
                parameters.setOuterLoops(runLoops);
            }
            java.util.Set<String> ids = new java.util.LinkedHashSet<>();
            java.util.Set<String> withoutStrategies = new java.util.HashSet<>();
            for (Contingency contingency : sameParameters) {
                ids.add(contingency.getId());
                if (contingency.getExtension(com.powsybl.openloadflow.sa.extensions.ContingencyLoadFlowParameters.class).getScope()
                        == com.powsybl.openloadflow.sa.extensions.ContingencyLoadFlowParameters.Scope.CONTINGENCY_ONLY) {
                    withoutStrategies.add(contingency.getId());
                }
            }
            if (scope.batchDisabled()) {
                LOGGER.info("GPU batched SA: outer loops {} of {} contingencies' own load flow parameters are not run on the "
                        + "device, they run on the CPU", scope.unsupportedOuterLoops(), ids.size());
                leftToCpu.addAll(ids);
                continue;
            }
            groups.add(new GpuBatchPrecomputer.SolveGroup(GpuBatchPrecomputer.SolveSettings.of(scope), ids, withoutStrategies));
            for (String id : ids) {
                if (!withoutStrategies.contains(id)) {
                    run.ownDistributedSlack.put(id, scope.distributedSlack());
                }
            }
        }
        return groups;
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
        Double distributed = pairDistributedOnDevice(runState.get(), contingency.getId(), scId, securityAnalysisParameters);
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
        // the base tap of each branch a tap action retaps, before the action: the device's tap moves are from the base
        java.util.Map<LfBranch, Integer> actionBaseTaps = new java.util.HashMap<>();
        for (com.powsybl.openloadflow.network.action.LfAction lfAction : lfActions) {
            if (lfAction instanceof com.powsybl.openloadflow.network.action.AbstractLfTapChangerAction<?> tapAction
                    && tapAction.getChange() != null) {
                LfBranch branch = tapAction.getChange().getBranch();
                actionBaseTaps.put(branch, runState.get().deviceTapBase.getOrDefault(branch, branch.getPiModel().getTapPosition()));
            }
        }
        if (lfActions.stream().anyMatch(a -> a instanceof com.powsybl.openloadflow.network.action.AbstractLfBranchAction<?>)) {
            com.powsybl.openloadflow.network.action.LfActionUtils.applyListOfActions(lfActions, network, lfContingency, networkParameters);
        } else {
            // applyListOfActions first redoes the post-contingency connectivity and re-derives every bus and branch
            // status from it (a full connected-components pass per strategy), which without a branch action changes
            // nothing: the contingency already applied that topology. Then it applies the actions, as here.
            for (com.powsybl.openloadflow.network.action.LfAction lfAction : lfActions) {
                if (!lfAction.apply(network, lfContingency, networkParameters)) {
                    com.powsybl.openloadflow.util.Reports.reportActionApplicationFailure(lfAction.getId(), lfContingency.getId(),
                            network.getReportNode());
                }
            }
        }
        // The device screen covers the pair's state like a contingency's (screenAll ran over every precomputed state). It
        // reads each branch in the base topology with its base parameters, so the branches the contingency opened at one
        // side and those the actions open, close or retap are scanned in full on the host.
        java.util.Set<LfBranch> scanInFull = new java.util.LinkedHashSet<>(sideOpenedBranches(lfContingency));
        for (com.powsybl.openloadflow.network.action.LfAction lfAction : lfActions) {
            if (lfAction instanceof com.powsybl.openloadflow.network.action.AbstractLfBranchAction<?> branchAction) {
                scanInFull.addAll(branchAction.getDisabledBranches());
                scanInFull.addAll(branchAction.getEnabledBranches());
            } else if (lfAction instanceof com.powsybl.openloadflow.network.action.AbstractLfTapChangerAction<?> tapAction
                    && tapAction.getChange() != null) {
                scanInFull.add(tapAction.getChange().getBranch());
            }
        }
        if (contingency.getId().equals(runState.get().deviceStateInjectedFor)) {
            scanInFull.addAll(runState.get().automationSystemSwitched);
        }
        // the taps the pair's outer loops left (from the base, like its parent's): a branch the actions retapped and the
        // loops did not move keeps the action's tap
        List<LfBranch> pairRetapped = GpuDeviceTaps.apply(network, precomputed.deviceTapMoves().get(scId),
                runState.get().deviceTapBase, actionBaseTaps, actionBaseTaps.keySet());
        scanInFull.addAll(pairRetapped);
        // only the flows the screened detection reads are refreshed, as for a contingency (flowReadSet)
        GpuStateInjector.injectState(precomputed.equationSystem(), precomputed.baseVariableRows(), state,
                runState.get().remap, strategyFlowReadSet(runState.get(), scId, contingency.getId(), operatorStrategy, scanInFull));
        AcSolverUtil.updateNetwork(network, precomputed.equationSystem());
        var postActionsViolationManager = new LimitViolationManager(preContingencyLimitViolationManager, limitReductions,
                securityAnalysisParameters.getIncreasedViolationsParameters());
        LoadFlowModel loadFlowModel = securityAnalysisParameters.getLoadFlowParameters().isDc() ? LoadFlowModel.DC : LoadFlowModel.AC;
        var postActionsNetworkResult = new PostContingencyNetworkResult(network,
                new AbstractNetworkResult.StateMonitorIndexes(monitorIndex, zeroImpedanceMonitoredIndex), createResultExtension,
                preContingencyNetworkResult, contingency, loadFlowModel, securityAnalysisParameters.getLoadFlowParameters().getDcPowerFactor(),
                securityAnalysisParameters.getModifiedMonitoredElementsParameters());
        postActionsNetworkResult.update();
        detectViolations(network, postActionsViolationManager, limitReductions, preContingencyLimitViolationManager,
                securityAnalysisParameters, scId, scanInFull);
        // what restorePostContingencyState must undo: with tap actions only, the retapped branches and the voltages
        List<LfBranch> retapped = new java.util.ArrayList<>();
        for (com.powsybl.openloadflow.network.action.LfAction lfAction : lfActions) {
            if (!(lfAction instanceof com.powsybl.openloadflow.network.action.AbstractLfTapChangerAction<?> tapAction)) {
                retapped = null;
                break;
            }
            if (tapAction.getChange() != null) {
                retapped.add(tapAction.getChange().getBranch());
            }
        }
        if (retapped != null) {
            retapped.addAll(pairRetapped);                    // back to the post-contingency taps
        }
        runState.get().strategyRetappedOnly = retapped;
        return new OperatorStrategyResult(operatorStrategy,
                List.of(new OperatorStrategyResult.ConditionalActionsResult(operatorStrategy.getId(),
                        PostContingencyComputationStatus.CONVERGED,
                        new LimitViolationsResult(postActionsViolationManager.getLimitViolations()),
                        GpuStateInjector.filterIslanded(network, precomputed.islandedBuses().get(scId), scId,
                                new NetworkResult(postActionsNetworkResult.getBranchResults(), postActionsNetworkResult.getBusResults(),
                                        postActionsNetworkResult.getThreeWindingsTransformerResults())),
                        distributed * PerUnit.SB)));
    }

    /**
     * A device-answered contingency answered without applying it to the network (P2 of
     * {@code docs/device-final-results-plan.md}): what {@link #runPostContingencySimulation} answers from the state, with
     * the contingency's topology read from its disabled network ({@link GpuContingencyTopology}) instead of from the
     * network's flags after {@code apply}, and the loss distribution's report written from its replayed value, as
     * {@code ReplayedLossDistribution} writes it. Nothing is applied, distributed, simulated or restored, but the
     * excluded slack buses {@code toLfContingency} would have changed. Only when nothing needs the applied network
     * ({@link #answeredWithoutNetwork}). While the comparison is on, the contingency is answered the usual way too and the
     * two results compared ({@link GpuStateViewCheck#compareResults}); the usual one is returned.
     */
    @Override
    protected java.util.Optional<PostContingencyResult> resultWithoutNetwork(LfNetwork network, AcLoadFlowContext context,
                                                                            PropagatedContingency propagatedContingency,
                                                                            LfContingency lfContingency,
                                                                            com.powsybl.openloadflow.network.NetworkState networkState,
                                                                            LimitViolationManager preContingencyLimitViolationManager,
                                                                            SecurityAnalysisParameters securityAnalysisParameters,
                                                                            PreContingencyNetworkResult preContingencyNetworkResult,
                                                                            boolean createResultExtension, List<LimitReduction> limitReductions,
                                                                            ReportNode reportNode) {
        GpuSaRunState run = runState.get();
        Contingency contingency = propagatedContingency.getContingency();
        if (!answeredWithoutNetwork(run, contingency, network)) {
            return java.util.Optional.empty();
        }
        boolean compare = GpuStateViewCheck.comparesDecisions();
        PostContingencyResult result = answerWithoutNetwork(run, network, context, contingency, lfContingency,
                preContingencyLimitViolationManager, securityAnalysisParameters, limitReductions, compare ? null : reportNode);
        if (compare) {                                       // the usual path answers it too, and compares the two
            run.withoutNetwork = result;
            run.withoutNetworkFor = contingency.getId();
            return java.util.Optional.empty();
        }
        // what toLfContingency may have changed: the excluded slack buses of a relocated slack bus (the device leaves
        // those contingencies to the CPU, so this is for safety; the restore also puts back the HVDCs and areas, which
        // nothing changed)
        if (!excludedSlackBuses(network).equals(run.baseExcludedSlackBuses)) {
            networkState.restore(List.of(), List.of(), false);
        }
        LAST_ANSWERED_WITHOUT_NETWORK_COUNT.incrementAndGet();
        return java.util.Optional.of(result);
    }

    private static java.util.Set<LfBus> excludedSlackBuses(LfNetwork network) {
        java.util.Set<LfBus> excluded = new java.util.HashSet<>();
        network.getSynchronousNetworks().forEach(sc -> excluded.addAll(sc.getExcludedSlackBuses()));
        return excluded;
    }

    /**
     * Whether a contingency can be answered without applying it: it is answered from its state
     * ({@link #answeredFromState}), and nothing needs the applied network: the device moved no tap and switched no branch
     * for it (both are set on the network today), its loss distribution has a replayed value (OLF's own distribution
     * writes the network), no state monitor asks for results (they read the network's objects), it did not run out of
     * Newton iterations on the device (that result comes from the usual path), and no dump of the applied network is on.
     */
    private boolean answeredWithoutNetwork(GpuSaRunState run, Contingency contingency, LfNetwork network) {
        PrecomputedBatch precomputed = run.precomputed;
        String id = contingency.getId();
        return precomputed.isReady()
                && !precomputed.exhausted().contains(id)
                && precomputed.states().containsKey(id)
                && precomputed.preDistributedActivePower().get(id) != null
                && precomputed.deviceTapMoves().get(id) == null
                && precomputed.automationSystemSwitched().get(id) == null
                && !flowsMonitored() && !busesMonitored()
                && !GpuTrace.SA_VAR_DUMP.isOn() && !GpuTrace.EQ_DUMP_CTG.isOn()
                && answeredFromState(run, contingency, network, precomputed);
    }

    /** The result {@link #runPostContingencySimulation} would answer from the state, the contingency not applied. A null
     *  {@code reportNode} writes no report (the usual path then writes it). */
    private PostContingencyResult answerWithoutNetwork(GpuSaRunState run, LfNetwork network, AcLoadFlowContext context,
                                                       Contingency contingency, LfContingency lfContingency,
                                                       LimitViolationManager preContingencyLimitViolationManager,
                                                       SecurityAnalysisParameters securityAnalysisParameters,
                                                       List<LimitReduction> limitReductions, ReportNode reportNode) {
        long t0 = System.nanoTime();
        PrecomputedBatch precomputed = run.precomputed;
        String id = contingency.getId();
        double[] state = precomputed.states().get(id);
        logPostContingencyStart(network, lfContingency);
        Stopwatch stopwatch = Stopwatch.createStarted();
        double replayed = precomputed.preDistributedActivePower().get(id);
        if (reportNode != null) {                            // what ReplayedLossDistribution reports on the usual path
            double mismatch = lfContingency.getActivePowerLoss();
            if (Math.abs(mismatch) > 0) {
                com.powsybl.openloadflow.util.Reports.reportContingencyActivePowerLossDistribution(reportNode,
                        mismatch * PerUnit.SB, (mismatch - replayed) * PerUnit.SB);
            }
            LAST_REPLAYED_LOSS_COUNT.incrementAndGet();
        }
        double distributedActivePower = precomputed.distributedActivePower().getOrDefault(id, replayed);
        GpuContingencyTopology topology = GpuContingencyTopology.of(lfContingency);
        run.sideOpened = sideOpenedBranches(lfContingency);
        run.deviceRetapped = List.of();
        run.automationSystemSwitched = List.of();
        boolean unrealistic = GpuPostContingencyState.checkBusesFromState(state, run.stateView, network, context.getParameters(),
                run, false, topology);
        var connectivityResult = new ConnectivityResult(lfContingency.getCreatedSynchronousComponentsCount(), 0,
                lfContingency.getDisconnectedLoadActivePower() * PerUnit.SB,
                lfContingency.getDisconnectedGenerationActivePower() * PerUnit.SB,
                lfContingency.getDisconnectedElementIds());
        PostContingencyResult result;
        if (unrealistic) {                                   // OLF reports no violations for an unrealistic state
            result = new PostContingencyResult(contingency, PostContingencyComputationStatus.FAILED,
                    new LimitViolationsResult(List.of()), new NetworkResult(List.of(), List.of(), List.of()), connectivityResult,
                    distributedActivePower * PerUnit.SB);
        } else {
            var manager = new LimitViolationManager(preContingencyLimitViolationManager, limitReductions,
                    securityAnalysisParameters.getIncreasedViolationsParameters());
            detectViolations(network, manager, limitReductions, preContingencyLimitViolationManager, securityAnalysisParameters,
                    id, null, state, topology);
            result = new PostContingencyResult(contingency, PostContingencyComputationStatus.CONVERGED,
                    new LimitViolationsResult(manager.getLimitViolations()),
                    GpuStateInjector.filterIslanded(network, precomputed.islandedBuses().get(id), id,
                            new NetworkResult(List.of(), List.of(), List.of())),
                    connectivityResult, distributedActivePower * PerUnit.SB);
        }
        stopwatch.stop();
        logPostContingencyEnd(network, lfContingency, stopwatch);
        PHASE_NS.computeIfAbsent(" answered without the network (no apply, no restore)",
                k -> new java.util.concurrent.atomic.LongAdder()).add(System.nanoTime() - t0);
        return result;
    }

    @Override
    protected PostContingencyResult runPostContingencySimulation(LfNetwork network, AcLoadFlowContext context, Contingency contingency,
                                                                 LfContingency lfContingency, LimitViolationManager preContingencyLimitViolationManager,
                                                                 SecurityAnalysisParameters securityAnalysisParameters,
                                                                 PreContingencyNetworkResult preContingencyNetworkResult, boolean createResultExtension,
                                                                 List<LimitReduction> limitReductions, double preDistributedActivePower) {
        PostContingencyResult result = postContingencyResult(network, context, contingency, lfContingency,
                preContingencyLimitViolationManager, securityAnalysisParameters, preContingencyNetworkResult, createResultExtension,
                limitReductions, preDistributedActivePower);
        GpuSaRunState run = runState.get();
        if (run.withoutNetwork != null) {                    // also answered without the network: compare the two
            if (contingency.getId().equals(run.withoutNetworkFor)) {
                GpuStateViewCheck.compareResults(run.withoutNetwork, result);
            }
            run.withoutNetwork = null;
            run.withoutNetworkFor = null;
        }
        return result;
    }

    private PostContingencyResult postContingencyResult(LfNetwork network, AcLoadFlowContext context, Contingency contingency,
                                                        LfContingency lfContingency, LimitViolationManager preContingencyLimitViolationManager,
                                                        SecurityAnalysisParameters securityAnalysisParameters,
                                                        PreContingencyNetworkResult preContingencyNetworkResult, boolean createResultExtension,
                                                        List<LimitReduction> limitReductions, double preDistributedActivePower) {
        GpuSaRunState run = runState.get();
        if (run.staleVoltages != null) {
            PrecomputedBatch precomputed = run.precomputed;
            // the same test as the injection below: a contingency with a device state still runs on OLF's own path
            // when one of its strategies is not answered on the device, and OLF then starts from every bus's voltage
            // (rte6515 LINE-2349-694 after LINE-2348-693: started from the previous injection's voltages, 1.44 MW off)
            boolean devicePath = precomputed.isReady() && !precomputed.exhausted().contains(contingency.getId())
                    && precomputed.states().containsKey(contingency.getId())
                    && everyStrategyAnsweredOnDevice(run, contingency.getId(), securityAnalysisParameters);
            // device-answered: the injection writes every live bus, so only the disabled ones need their base back
            restoreStaleVoltages(run, devicePath ? lfContingency.getDisabledNetwork().getBuses() : null);
        }
        GpuSaDiagnostics.dumpPostContingencyEquations(context, contingency, lfContingency, network);
        PrecomputedBatch precomputed = runState.get().precomputed;
        if (precomputed.isReady() && precomputed.exhausted().contains(contingency.getId())
                && everyStrategyAnsweredOnDevice(runState.get(), contingency.getId(), securityAnalysisParameters)
                && !strategyConditionReadsNetwork(runState.get(), contingency.getId())) {
            // The device carried this scenario through OLF's whole Newton budget without
            // convergence. OLF's own path (AbstractSecurityAnalysis.runPostContingencySimulation)
            // reports MAX_ITERATION_REACHED with an empty network result and no violations for
            // that outcome; the same result is built here, from the contingency already applied.
            // A strategy of it the device did not answer starts from what OLF's failed run leaves on the network
            // (its targets, its bus types), which only that run produces: the contingency then runs on the CPU. So does
            // one whose strategy is conditioned on a threshold: the condition reads the flows of the failed run's last
            // Newton iterate (a plateau's last sample is a lottery, which the device does not hand back) or its targets.
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

        // The contingency is applied (lfContingency.apply()). Its state is either read from the device state array
        // without being written into OLF's objects (answeredFromState: the detection reads it through GpuStateView, the
        // same doubles), or injected into the equation system and the buses and read back through OLF's objects.
        long tPost = System.nanoTime();
        logPostContingencyStart(network, lfContingency);
        Stopwatch stopwatch = Stopwatch.createStarted();

        // What OLF reports on the result: the pre-distribution PLUS the load flow's own distributed-slack passes
        // (AbstractSecurityAnalysis: preDistributedActivePower + result.getDistributedActivePower()). The device
        // total covers both; where it is unknown the pre-distribution alone is reported, as before.
        double distributedActivePower = precomputed.distributedActivePower()
                .getOrDefault(contingency.getId(), preDistributedActivePower);
        long tInj = System.nanoTime();
        run.deviceAnsweredFor = contingency.getId();         // injected or not, the targeted restore applies to it
        run.sideOpened = sideOpenedBranches(lfContingency);
        // the taps the device's outer loops left: before the state, whose flows are evaluated with them
        run.deviceRetapped = GpuDeviceTaps.apply(network, precomputed.deviceTapMoves().get(contingency.getId()),
                run.deviceTapBase, Map.of(), java.util.Set.of());
        // before the state: a switched branch is enabled or disabled, which the detection reads (and the injection
        // refreshes the flows of the enabled branches only)
        run.automationSystemSwitched = applyAutomationSystemSwitches(network,
                precomputed.automationSystemSwitched().get(contingency.getId()));
        boolean fromState = answeredFromState(run, contingency, network, precomputed);
        boolean compareDecisions = fromState && GpuStateViewCheck.comparesDecisions();
        boolean unrealistic = false;
        if (fromState) {
            // the realistic-voltage check on the array; the buses get their voltages only for a bus-results monitor
            unrealistic = GpuPostContingencyState.checkBusesFromState(state, run.stateView, network, context.getParameters(),
                    run, busesMonitored());
            LAST_ANSWERED_FROM_STATE_COUNT.incrementAndGet();
            PHASE_NS.computeIfAbsent("  state read from the device array (no write-back)", k -> new java.util.concurrent.atomic.LongAdder())
                    .add(System.nanoTime() - tInj);
        }
        if (!fromState || compareDecisions) {
            long tWrite = System.nanoTime();
            run.deviceStateInjectedFor = contingency.getId();
            int[] flowReadSet = flowReadSet(run, contingency.getId());
            GpuStateInjector.injectState(precomputed.equationSystem(), precomputed.baseVariableRows(), state,
                    run.remap, flowReadSet);
            long tUpd = System.nanoTime();
            PHASE_NS.computeIfAbsent("   injectState()", k -> new java.util.concurrent.atomic.LongAdder())
                    .add(tUpd - tWrite);
            // OLF_GPU_SA_RESID=<n>: OLF's OWN residual at the state the device returned, on OLF's
            // CONTINGENCY-updated equation system (lfContingency.apply() ran above). A large value on a row
            // means the device solved a DIFFERENT equation there — a row-assembly bug — not that it failed
            // to converge. Prints the n worst rows with the equation OLF has on each, which names the family
            // to look at.
            if (GpuTrace.SA_RESID.isOn()) {
                GpuSaDiagnostics.reportOlfResidual(context, precomputed.equationSystem(), contingency.getId(),
                        Integer.parseInt(GpuTrace.SA_RESID.value()), precomputed.baseVariableRows(),
                        precomputed.islandedBuses().get(contingency.getId()));
            }
            boolean unrealisticFromObjects = GpuPostContingencyState.updateBusesFromState(state, precomputed, network,
                    context.getParameters(), run);
            if (compareDecisions) {
                GpuStateViewCheck.compareStatus(unrealistic, unrealisticFromObjects);
            } else {
                unrealistic = unrealisticFromObjects;
            }
            PHASE_NS.computeIfAbsent("   AcSolverUtil.updateNetwork()", k -> new java.util.concurrent.atomic.LongAdder())
                    .add(System.nanoTime() - tUpd);
            PHASE_NS.computeIfAbsent("  state injection + network update", k -> new java.util.concurrent.atomic.LongAdder())
                    .add(System.nanoTime() - (fromState ? tWrite : tInj));
        }
        if (strategyConditionReadsGeneratorTargets(run, contingency.getId())) {
            applyDeviceGeneratorTargets(network, precomputed.generatorTargets().get(contingency.getId()));
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
                preContingencyLimitViolationManager, securityAnalysisParameters, contingency.getId(), null,
                fromState ? state : null);
        PHASE_NS.computeIfAbsent("   detectViolations()", k -> new java.util.concurrent.atomic.LongAdder())
                .add(System.nanoTime() - tDet);
        if (compareDecisions) {                              // the same detection on the written-back objects
            var fromObjects = new LimitViolationManager(preContingencyLimitViolationManager, limitReductions,
                    securityAnalysisParameters.getIncreasedViolationsParameters());
            detectViolations(network, fromObjects, limitReductions, preContingencyLimitViolationManager,
                    securityAnalysisParameters, contingency.getId());
            GpuStateViewCheck.compareViolations(postContingencyLimitViolationManager.getLimitViolations(),
                    fromObjects.getLimitViolations());
        }
        if (GpuTrace.STATE_VIEW_CHECK.isOn()) {              // written back: compareDecisions holds for every answer
            GpuStateViewCheck.check(stateView(run, network, precomputed), state, run.limitTable, network);
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
