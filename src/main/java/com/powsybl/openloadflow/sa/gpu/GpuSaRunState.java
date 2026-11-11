/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.action.Action;
import com.powsybl.contingency.strategy.OperatorStrategy;
import com.powsybl.openloadflow.ac.AcLoadFlowContext;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.solver.gpu.batch.BatchConfig;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchedSecurityAnalysisSolver;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.impl.PropagatedContingency;
import com.powsybl.security.SecurityAnalysisParameters;

import java.util.List;
import java.util.Map;

/**
 * One security-analysis partition's run state (the batch precomputed on the device, the restore bookkeeping,
 * the read sets): one per thread, held by GpuBatchedAcSecurityAnalysis in a ThreadLocal and passed explicitly
 * to every helper (the SA runs partitions in parallel).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuSaRunState {
    List<PropagatedContingency> contingencies = List.of();
    List<OperatorStrategy> operatorStrategies = List.of();
    List<Action> actions = List.of();
    LfNetwork lfNetwork;
    AcLoadFlowContext context;
    /** The batch precomputed on the device once the base load flow converged (see
     *  GpuBatchPrecomputer); PrecomputedBatch.empty() until then / on CPU fallback. */
    PrecomputedBatch precomputed = PrecomputedBatch.empty();
    /** The state injector's row permutation, kept across this partition's contingencies. */
    final GpuStateInjector.Remap remap = new GpuStateInjector.Remap();
    /** The contingency whose loss distribution the replay answered, and the one whose state the device supplied:
     *  when both name the contingency just simulated, only its own elements and the bus voltages changed. */
    String lossReplayedFor;
    String deviceStateInjectedFor;
    /** LAZY VOLTAGE RESTORE: the last targeted restore left the bus voltages at the previous contingency's injected
     *  state. The next device-answered contingency overwrites every live bus and puts the base back on its disabled
     *  ones; anything else restores them all first (restoreStaleVoltages). */
    com.powsybl.openloadflow.network.NetworkState staleVoltages;
    /** Shunt -> its bus, for the shunts a contingency shifts (LfShunt does not know its bus). Built once. */
    java.util.Map<com.powsybl.openloadflow.network.LfShunt, LfBus> shuntBus;
    /** What the network update writes beyond the bus voltages when the base system holds control state variables, for
     *  the extraction {@link #controlStateFor} was built from: the buses of the controller shunts with a SHUNT_B
     *  variable and the branches with a BRANCH_ALPHA1 or BRANCH_RHO1 one. A targeted restore restores them too. */
    Map<Variable<AcVariableType>, Integer> controlStateFor;
    List<LfBus> controlStateBuses = List.of();
    List<LfBranch> controlStateBranches = List.of();
    /** Per bus num: does the realistic-voltage band apply (not fictitious, nominal V at or above the
     *  check's minimum)? Built once for {@link #realisticBandNetwork}. */
    boolean[] realisticBandApplies;
    LfNetwork realisticBandNetwork;
    /** Per bus num: its BUS_V / BUS_PHI row in the device state (base layout), for the extraction they were
     *  built from; null when the base system is not exactly one V and one angle per bus (the walk is used). */
    int[] vRowOfBus;
    int[] phiRowOfBus;
    Map<Variable<AcVariableType>, Integer> busRowsFor;
    boolean busRowsUsable;
    // The run's actual outer loops, captured in runSimulations: the device must run the distributed-slack /
    // reactive-limits passes ONLY when OLF would (i.e. when the outer loop is in the configured list), never
    // unconditionally from network capability. Default to OLF's defaults until runSimulations sets them.
    boolean batchDistributedSlack = true;
    boolean batchReactiveLimits = true;
    // The run's outer-loop convergence tolerances (per-unit), captured from the loadflow parameters so the
    // device aligns to the CPU loadflow (GPU follows CPU, not the reverse): slackBusPMaxMismatch /
    // maxReactivePowerMismatch divided by the 100 MVA base. Default to OLF's defaults until runSimulations sets.
    double batchSlackDistTol = GpuBatchedSecurityAnalysisSolver.DEFAULT_SLACK_DIST_TOL;
    double batchReactiveLimitTol = GpuBatchedSecurityAnalysisSolver.DEFAULT_REACTIVE_LIMIT_TOL;
    // The device inner-Newton stopping criterion, matched to OLF's: for UNIFORM_CRITERIA (default) the device
    // stops on OLF's exact ‖F‖₂ < sqrt(convEpsPerEq²·n) so the reactive-limit switch DECISIONS (whose tolerance
    // is convEpsPerEq for UNIFORM) are made on a state converged to the SAME depth as the CPU — else the device
    // switched on a 1e-8-converged state while the RL threshold was convEpsPerEq (1e-12), flipping borderline
    // buses vs the CPU. 0 = the device infinity-norm floor (kept for PER_EQUATION_TYPE, whose per-eq criteria
    // are not ported to the device).
    double batchConvEpsPerEq;
    // The run's batched outer-loop parameterization, captured per run and assembled into a BatchConfig at
    // the solveBatchN1 call. This replaced the solver's former JVM-global static config, whose leak between
    // a provider run and a direct-solver run caused the pegase full-suite failure — see BatchConfig's javadoc.
    BatchConfig batchConfig = BatchConfig.DEVICE_DEFAULTS;
    com.powsybl.loadflow.LoadFlowParameters loadFlowParameters;
    /** The run's security-analysis parameters — needed at precompute time because the SA's own
     *  CONTINGENCY parameter override has not been applied yet (see afterPreContingencySimulation). */
    SecurityAnalysisParameters securityAnalysisParameters;
    /** The run's flattened limit thresholds, built on first use and valid for the whole run
     *  (see GpuLimitTable); null when the accelerated detection is off. */
    GpuLimitScanner limitScanner;
    GpuLimitTable limitTable;
    /** Non-null when the state is screened against the table instead of fully scanned. */
    GpuViolationScreen violationScreen;
    int[] uncoveredBranches = new int[0];
    /** The current contingency's branches opened at ONE side (a bus or busbar loss): the screen evaluates every
     *  branch in the base topology, so their records are not used and they are scanned on the host. */
    List<LfBranch> sideOpened = List.of();
    /** The branches the current contingency's automation systems switched on the device, given to the network with its
     *  state and restored with its own elements. */
    List<LfBranch> automationSystemSwitched = List.of();
    /** The base tap of every branch a device tap move changed during the current contingency, its own and its strategy
     *  pairs' (GpuDeviceTaps): each scenario's moves are from the base, so a branch moved earlier and not listed goes back
     *  to its base tap, and the restores put every one of them back. */
    final Map<LfBranch, Integer> deviceTapBase = new java.util.LinkedHashMap<>();
    /** Whether the strategy load flows of a contingency carrying its own load flow parameters distribute active power
     *  (its group's outer loops, GpuBatchPrecomputer.SolveGroup); the run's flag for every other contingency. */
    final Map<String, Boolean> ownDistributedSlack = new java.util.HashMap<>();
    /** The branches the current contingency's device tap moves changed: the screen read them at their base tap, so they
     *  are scanned in full on the host and their flows refreshed. */
    List<LfBranch> deviceRetapped = List.of();
    /** The branches the last operator strategy retapped, when it was answered on the device with tap actions only (a
     *  targeted post-contingency restore suffices); null otherwise (OLF's full restore). */
    List<LfBranch> strategyRetappedOnly;
    /** Limit-table index per branch num, -1 = no limits (built with the table). */
    int[] tableIndexOfBranchNum;
    /** The detection's inputs read from the device state array (OLF_GPU_STATE_VIEW_CHECK), for these base rows. */
    GpuStateView stateView;
    /** Every scenario's screened candidates, computed in one pass over the precomputed states. */
    Map<String, int[]> candidates = Map.of();
}
