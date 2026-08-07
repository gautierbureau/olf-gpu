/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.commons.report.ReportNode;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivation;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivationExtractor;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataCache;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcOuterLoopPacks;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.TargetVector;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfShunt;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Solver for batched N-1 security analysis on the GPU: runs hundreds to thousands of
 * post-contingency AC power flows in one batched Newton-Raphson, with per-scenario
 * convergence retirement (converged scenarios stop filling/solving early). Expected
 * throughput gain: one-to-two orders of magnitude over sequential per-contingency solves,
 * amortizing the device context creation and cuDSS symbolic analysis over the entire
 * contingency set.
 *
 * <p>Scope: single-branch outages that keep the main synchronous component intact
 * (checked per scenario; contingencies that would island or fragment the network are
 * rejected). Distributed slack, reactive limits, and secondary controls are not yet
 * device-resident (scope: plain N-1 with base-case participation factors).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuBatchedSecurityAnalysisSolver {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuBatchedSecurityAnalysisSolver.class);

    /**
     * Solver for batched contingencies: given a base-case state and a list of
     * single-branch contingencies, extract one batched GPU context and run all
     * scenarios in a single Newton-Raphson loop.
     *
     * @param network         the AC network (base case)
     * @param baseState       the converged base-case state vector
     * @param contingencies   list of single-branch outages to analyze
     * @param equationSystem  the base-case equation system
     * @param targetVector    the base-case targets
     * @param parameters      Newton-Raphson stopping criteria
     * @param reportNode      for logging and diagnostics
     * @return per-scenario results (state, iterations, convergence status)
     */
    public static List<BatchScenarioResult> solveBatchN1(LfNetwork network, double[] baseState,
                                                         List<BatchContingency> contingencies,
                                                         EquationSystem<AcVariableType, AcEquationType> equationSystem,
                                                         TargetVector<AcVariableType, AcEquationType> targetVector,
                                                         NewtonRaphsonParameters parameters,
                                                         ReportNode reportNode) {
        return solveBatchN1(network, baseState, contingencies, equationSystem, targetVector, parameters,
                GpuNewtonRaphson.GPU_TOLERANCE, true, true, reportNode);
    }

    /**
     * As {@link #solveBatchN1(LfNetwork, double[], List, EquationSystem, TargetVector, NewtonRaphsonParameters, ReportNode)}
     * but with the run's actual outer loops declared. {@code distributedSlack} / {@code reactiveLimits} must
     * match the outer loops OLF applied to the base case: the device runs the distributed-slack pass only when
     * {@code distributedSlack} and the PV→PQ reactive-limits pass only when {@code reactiveLimits}, so a
     * disabled outer loop is never run on the device. Passing the wrong flags diverges from the CPU under a
     * non-trivial slack mismatch or a reactive-limit violation.
     */
    public static List<BatchScenarioResult> solveBatchN1(LfNetwork network, double[] baseState,
                                                         List<BatchContingency> contingencies,
                                                         EquationSystem<AcVariableType, AcEquationType> equationSystem,
                                                         TargetVector<AcVariableType, AcEquationType> targetVector,
                                                         NewtonRaphsonParameters parameters,
                                                         boolean distributedSlack, boolean reactiveLimits,
                                                         ReportNode reportNode) {
        return solveBatchN1(network, baseState, contingencies, equationSystem, targetVector, parameters,
                GpuNewtonRaphson.GPU_TOLERANCE, distributedSlack, reactiveLimits, reportNode);
    }

    /**
     * As the {@code (distributedSlack, reactiveLimits)} overload but with the device outer-loop tolerances
     * declared explicitly (per-unit) so the device aligns to the CPU loadflow's outer-loop criteria. Used by
     * the production SA path, which reads them from the run's loadflow parameters.
     */
    public static List<BatchScenarioResult> solveBatchN1(LfNetwork network, double[] baseState,
                                                         List<BatchContingency> contingencies,
                                                         EquationSystem<AcVariableType, AcEquationType> equationSystem,
                                                         TargetVector<AcVariableType, AcEquationType> targetVector,
                                                         NewtonRaphsonParameters parameters,
                                                         boolean distributedSlack, boolean reactiveLimits,
                                                         double reactiveLimitTol, double slackDistTol,
                                                         ReportNode reportNode) {
        return solveBatchN1(network, baseState, contingencies, equationSystem, targetVector, parameters,
                GpuNewtonRaphson.GPU_TOLERANCE, distributedSlack, reactiveLimits, reactiveLimitTol, slackDistTol,
                reportNode);
    }

    /**
     * As the {@code (reactiveLimitTol, slackDistTol)} overload but with an explicit {@code convEpsPerEq} for the
     * device inner Newton (device tol defaults to {@link GpuNewtonRaphson#GPU_TOLERANCE} as the fixed-iter
     * sentinel). {@code convEpsPerEq > 0} makes the device stop on OLF's exact {@code ‖F‖₂ < sqrt(convEpsPerEq²·n)}
     * so its reactive-limit switch decisions are taken at the SAME convergence depth as the CPU — used by the
     * production SA path so GPU and CPU agree beyond the outer-loop tolerance band.
     */
    public static List<BatchScenarioResult> solveBatchN1(LfNetwork network, double[] baseState,
                                                         List<BatchContingency> contingencies,
                                                         EquationSystem<AcVariableType, AcEquationType> equationSystem,
                                                         TargetVector<AcVariableType, AcEquationType> targetVector,
                                                         NewtonRaphsonParameters parameters,
                                                         boolean distributedSlack, boolean reactiveLimits,
                                                         double reactiveLimitTol, double slackDistTol,
                                                         double convEpsPerEq, ReportNode reportNode) {
        return solveBatchN1(network, baseState, contingencies, equationSystem, targetVector, parameters,
                GpuNewtonRaphson.GPU_TOLERANCE, distributedSlack, reactiveLimits, reactiveLimitTol, slackDistTol,
                convEpsPerEq, reportNode);
    }

    /**
     * As {@link #solveBatchN1(LfNetwork, double[], List, EquationSystem, TargetVector, NewtonRaphsonParameters, ReportNode)}
     * but with an explicit mismatch tolerance. Passing {@code tol <= 0} selects the native
     * DEVICE-SIDE fixed-iteration mode: every scenario runs exactly {@code parameters.getMaxIterations()}
     * Newton steps entirely on the device, with NO per-iteration mismatch copy back to the host and
     * NO convergence retirement (the mismatch is read back once, at the end, only to report it).
     * This is a fixed-iteration NR loop — it drops the per-iteration PCIe sync, so it is
     * the fastest way to benchmark raw batched throughput. Tradeoff: a diverging scenario is not
     * isolated and can poison a uniform-batched factorization, so use {@code tol > 0} when scenarios
     * may diverge.
     */
    public static List<BatchScenarioResult> solveBatchN1(LfNetwork network, double[] baseState,
                                                         List<BatchContingency> contingencies,
                                                         EquationSystem<AcVariableType, AcEquationType> equationSystem,
                                                         TargetVector<AcVariableType, AcEquationType> targetVector,
                                                         NewtonRaphsonParameters parameters,
                                                         double tol,
                                                         ReportNode reportNode) {
        return solveBatchN1(network, baseState, contingencies, equationSystem, targetVector, parameters,
                tol, true, true, reportNode);
    }

    /** OLF default outer-loop convergence tolerances (per-unit on the 100 MVA base): slackBusPMaxMismatch =
     *  1 MW → 1e-2 pu, maxReactivePowerMismatch = 1 MVAr → 1e-2/… → 1e-4 pu. The device uses these so its
     *  outer-loop fixed point matches the CPU loadflow's; a test wanting machine-precision agreement passes
     *  tight values on BOTH sides (a loose tolerance leaves a redistribution band ~its size). */
    public static final double DEFAULT_REACTIVE_LIMIT_TOL = 1e-4;
    public static final double DEFAULT_SLACK_DIST_TOL = 1e-2;

    /**
     * As {@link #solveBatchN1(LfNetwork, double[], List, EquationSystem, TargetVector, NewtonRaphsonParameters, double, boolean, boolean, ReportNode)}
     * with the device outer-loop tolerances defaulting to OLF's ({@link #DEFAULT_REACTIVE_LIMIT_TOL} /
     * {@link #DEFAULT_SLACK_DIST_TOL}).
     */
    public static List<BatchScenarioResult> solveBatchN1(LfNetwork network, double[] baseState,
                                                         List<BatchContingency> contingencies,
                                                         EquationSystem<AcVariableType, AcEquationType> equationSystem,
                                                         TargetVector<AcVariableType, AcEquationType> targetVector,
                                                         NewtonRaphsonParameters parameters,
                                                         double tol,
                                                         boolean distributedSlack, boolean reactiveLimits,
                                                         ReportNode reportNode) {
        return solveBatchN1(network, baseState, contingencies, equationSystem, targetVector, parameters, tol,
                distributedSlack, reactiveLimits, DEFAULT_REACTIVE_LIMIT_TOL, DEFAULT_SLACK_DIST_TOL, reportNode);
    }

    /**
     * Lowest-level entry point. {@code reactiveLimitTol} / {@code slackDistTol} are the device outer-loop
     * convergence tolerances (pu) — set them to the run's loadflow criteria (maxReactivePowerMismatch /
     * slackBusPMaxMismatch, per-unitized) so the device runs the DS/RL outer loops to the SAME tolerance as
     * the CPU. With matching parameters and the same equations the GPU and CPU outer-loop fixed points agree
     * to the inner-Newton precision.
     */
    public static List<BatchScenarioResult> solveBatchN1(LfNetwork network, double[] baseState,
                                                         List<BatchContingency> contingencies,
                                                         EquationSystem<AcVariableType, AcEquationType> equationSystem,
                                                         TargetVector<AcVariableType, AcEquationType> targetVector,
                                                         NewtonRaphsonParameters parameters,
                                                         double tol,
                                                         boolean distributedSlack, boolean reactiveLimits,
                                                         double reactiveLimitTol, double slackDistTol,
                                                         ReportNode reportNode) {
        return solveBatchN1(network, baseState, contingencies, equationSystem, targetVector, parameters, tol,
                distributedSlack, reactiveLimits, reactiveLimitTol, slackDistTol, 0.0, reportNode);
    }

    /**
     * As the lowest {@code (reactiveLimitTol, slackDistTol)} overload but with an explicit {@code convEpsPerEq}:
     * when {@code > 0} the device inner-Newton stops on OLF's EXACT criterion
     * ({@code DefaultNewtonRaphsonStoppingCriteria}: {@code ‖F‖₂ < sqrt(convEpsPerEq² · n)}) instead of the
     * {@code tol} infinity-norm floor — so a batched SA can be run with the SAME stopping decision as the CPU
     * loadflow for an apples-to-apples GPU-vs-CPU timing comparison. {@code convEpsPerEq <= 0} keeps the
     * infinity-norm floor (the default the correctness/parity tests rely on).
     */
    public static List<BatchScenarioResult> solveBatchN1(LfNetwork network, double[] baseState,
                                                         List<BatchContingency> contingencies,
                                                         EquationSystem<AcVariableType, AcEquationType> equationSystem,
                                                         TargetVector<AcVariableType, AcEquationType> targetVector,
                                                         NewtonRaphsonParameters parameters,
                                                         double tol,
                                                         boolean distributedSlack, boolean reactiveLimits,
                                                         double reactiveLimitTol, double slackDistTol,
                                                         double convEpsPerEq,
                                                         ReportNode reportNode) {
        // Non-provider callers (tests, direct use) get the device defaults — the per-call BatchConfig
        // replaces the former JVM-global statics, so no run can inherit another's config.
        return solveBatchN1(network, baseState, contingencies, equationSystem, targetVector, parameters, tol,
                distributedSlack, reactiveLimits, reactiveLimitTol, slackDistTol, convEpsPerEq, reportNode,
                BatchConfig.DEVICE_DEFAULTS);
    }

    /** As the no-{@code tol} / {@code convEpsPerEq} overload but carrying the run's {@link BatchConfig}
     *  (outer-loop parameterization). This is the entry point {@link
     *  com.powsybl.openloadflow.sa.gpu.GpuBatchedAcSecurityAnalysis} uses; it defaults {@code tol} to the
     *  GPU tolerance floor, exactly like the matching provider-facing overload without a config. */
    public static List<BatchScenarioResult> solveBatchN1(LfNetwork network, double[] baseState,
                                                         List<BatchContingency> contingencies,
                                                         EquationSystem<AcVariableType, AcEquationType> equationSystem,
                                                         TargetVector<AcVariableType, AcEquationType> targetVector,
                                                         NewtonRaphsonParameters parameters,
                                                         boolean distributedSlack, boolean reactiveLimits,
                                                         double reactiveLimitTol, double slackDistTol,
                                                         double convEpsPerEq, ReportNode reportNode,
                                                         BatchConfig config) {
        return solveBatchN1(network, baseState, contingencies, equationSystem, targetVector, parameters,
                GpuNewtonRaphson.GPU_TOLERANCE, distributedSlack, reactiveLimits, reactiveLimitTol, slackDistTol,
                convEpsPerEq, reportNode, config);
    }

    /** Terminal implementation carrying the per-call {@link BatchConfig}. */
    public static List<BatchScenarioResult> solveBatchN1(LfNetwork network, double[] baseState,
                                                         List<BatchContingency> contingencies,
                                                         EquationSystem<AcVariableType, AcEquationType> equationSystem,
                                                         TargetVector<AcVariableType, AcEquationType> targetVector,
                                                         NewtonRaphsonParameters parameters,
                                                         double tol,
                                                         boolean distributedSlack, boolean reactiveLimits,
                                                         double reactiveLimitTol, double slackDistTol,
                                                         double convEpsPerEq,
                                                         ReportNode reportNode,
                                                         BatchConfig config) {
        Objects.requireNonNull(network);
        Objects.requireNonNull(config);
        Objects.requireNonNull(baseState);
        Objects.requireNonNull(contingencies);
        Objects.requireNonNull(equationSystem);
        Objects.requireNonNull(targetVector);
        Objects.requireNonNull(parameters);

        if (contingencies.isEmpty()) {
            LOGGER.debug("Batched N-1 with empty contingency list");
            return List.of();
        }

        // Extract base-case GPU data once (shared structure, all scenarios use same pattern). The batch path
        // now carries all three family passes: DISTR_Q (remote/distributed generator VC), DISTR_RHO
        // (transformer ratio distribution) and zero-impedance branches (bus couplers). A contingency that
        // outages a DISTR_RHO controller or a zero-impedance branch is routed to the CPU (below / via the
        // element mapper) because the fixed batch pattern cannot represent the resulting structural change.
        java.util.Set<String> baseOpenIds = contingencies.stream()
                .flatMap(c -> c.reclosedBranches().stream())
                .filter(b -> !b.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC))
                .map(LfBranch::getId)
                .collect(java.util.stream.Collectors.toSet());
        java.util.Set<String> baseOpenZiIds = contingencies.stream()
                .flatMap(c -> c.reclosedBranches().stream())
                .filter(b -> b.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC))
                .map(LfBranch::getId)
                .collect(java.util.stream.Collectors.toSet());
        GpuAcData gpuData = baseOpenIds.isEmpty() && baseOpenZiIds.isEmpty()
                ? GpuAcDataCache.extract(network, equationSystem, distributedSlack, reactiveLimits)
                : GpuAcDataExtractor.extract(network, equationSystem, distributedSlack, reactiveLimits, baseOpenIds, baseOpenZiIds);
        GpuAcActivation baseActivation = GpuAcActivation.fromEquationSystem(network, equationSystem);
        GpuBatchElementMapper elementMapper = new GpuBatchElementMapper(network, baseOpenIds);

        int n = baseState.length;
        int nElements = gpuData.getTotalElementCount();
        double[] baseTargets = baseActivation.applyTargets(targetVector.getArray());

        // Qualify each contingency. Outages that keep the main component intact batch as a value-only
        // mask over the fixed pattern. A FRAGMENTING outage batches too in the full-loadflow mode
        // (tol > 0): its islanded buses are frozen to identity (held at base) in that scenario's row
        // mode, so the islanded sub-block stays non-singular and the main component solves correctly.
        // In the fixed-iter benchmark mode (tol <= 0) islanding routes to the CPU path.
        boolean handleIslanding = tol > 0;
        List<GpuBatchContingencyValidator.Qualification> qualifications =
                GpuBatchContingencyValidator.qualify(network, contingencies, elementMapper, handleIslanding);
        // Outaging a DISTR_RHO controller transformer reforms the ratio-distribution 1/count coefficients,
        // which the fixed batch pattern cannot represent — route those to the CPU (the DISTR_Q pass IS
        // batched, so DISTR_RHO is the only family that needs a contributor-outage rejection).
        Set<LfBranch> distrRhoControllers = gpuData.getDistrRhoContributionCount() > 0
                ? GpuAcOuterLoopPacks.distrRhoControllerBranches(network, equationSystem) : Set.of();
        // Distributed-slack factor denominator (Σ maxP/droop over ALL participating gens, incl. the slack's)
        // and whether the slack itself participates: a generator at a DS-participant bus is batched with a
        // per-scenario reweight (computeGenOutage), but only when the slack does not participate.
        double dsSum = 0;
        if (distributedSlack) {                              // no slack distribution ⇒ no per-scenario DS reweight
            for (LfBus b : network.getBuses()) {
                for (LfGenerator g : b.getGenerators()) {
                    if (g.isParticipating() && g.getDroop() != 0) {
                        dsSum += g.getMaxP() / g.getDroop();
                    }
                }
            }
        }
        boolean slackParticipates = java.util.Arrays.stream(gpuData.dsFactor()).sum() < 1 - 1e-9;
        List<Integer> batchedIndices = new ArrayList<>();
        int[][] frozenRowsByIndex = new int[contingencies.size()][];     // islanded buses' variable rows, per contingency
        int[][] islandedNumsByIndex = new int[contingencies.size()][];   // islanded LfBus nums (result-building excludes them)
        double[][] frozenTargetsByIndex = new double[contingencies.size()][];   // the clean value each frozen row holds (V=1, phi=0)
        GenOutage[][] genOutagesByIndex = new GenOutage[contingencies.size()][];   // per-bus merged outages, else null
        int islandingCount = 0;
        for (int i = 0; i < qualifications.size(); i++) {
            GpuBatchContingencyValidator.Qualification q = qualifications.get(i);
            if (!q.qualified()) {
                continue;
            }
            BatchContingency c = contingencies.get(i);
            if (!c.lostGenerators().isEmpty() || !c.lostLoads().isEmpty() || !c.shuntShifts().isEmpty()
                    || !c.tapOverrides().isEmpty() || !c.reclosedBranches().isEmpty()
                    || !c.outagedZiBranches().isEmpty()) {   // injection/shift/tap/reclose/switch-outage (or branch+)
                java.util.Set<Integer> baseOpenSet = java.util.Arrays.stream(gpuData.baseOpenCbElements())
                        .boxed().collect(java.util.stream.Collectors.toSet());
                java.util.Map<Integer, Integer> ziElemByBranchNum = new java.util.HashMap<>();
                for (int z = 0; z < gpuData.baseOpenZiElements().length; z++) {
                    ziElemByBranchNum.put(gpuData.baseOpenZiBranchNums()[z], gpuData.baseOpenZiElements()[z]);
                }
                boolean recloseOk = true;
                int ziCloses = 0;
                for (LfBranch rb : c.reclosedBranches()) {
                    if (rb.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC)) {
                        // switch close: packed base-open zi element, exactly one per scenario
                        // (multi-switch closes can interact through the spanning tree — CPU)
                        if (!ziElemByBranchNum.containsKey(rb.getNum()) || ++ziCloses > 1) {
                            recloseOk = false;
                        }
                    } else if (!baseOpenSet.contains(elementMapper.getElementIndex(rb))) {
                        recloseOk = false;
                    }
                }
                if (!recloseOk) {
                    continue;                                // reclose of an unpacked base-open branch — CPU
                }
                boolean tapOk = true;
                for (TapR1Override tap : c.tapOverrides()) { // packed CLOSED branch with FIXED ratio AND
                    int ei = elementMapper.getElementIndex(tap.branch());   // phase (neither is a state variable)
                    if (ei < 0 || ei >= elementMapper.getClosedBranchCount()
                            || gpuData.cbIdx()[ei * 10 + 9] >= 0 || gpuData.cbIdx()[ei * 10 + 8] >= 0) {
                        tapOk = false;
                    }
                }
                if (!tapOk) {
                    continue;                                // controlled/unpacked transformer tap — CPU
                }
                if (!c.shuntShifts().isEmpty()
                        && !canBatchShuntOutage(c.shuntShifts().stream().map(ShuntShift::shunt).toList(), elementMapper, gpuData)) {
                    continue;                                // unpacked / DISTR_Q-contributing shunt — CPU
                }
                GenOutage[] gos = GpuBatchInjectionOutages.computeInjectionOutages(equationSystem, gpuData, dsSum, slackParticipates,
                        c.lostGenerators(), c.lostLoads());
                if (gos == null) {                           // slack / >1 DS bus / remote gen — route to CPU
                    continue;
                }
                if (!c.outagedBranches().isEmpty()) {        // COMBO: the branch part must also qualify
                    if (c.outagedBranches().stream().anyMatch(distrRhoControllers::contains)) {
                        continue;
                    }
                    // conservative: a lost generator on an ISLANDED bus is out of combo scope
                    FrozenIsland fi = islandedFrozen(equationSystem, q.islandedBuses());
                    islandedNumsByIndex[i] = q.islandedBuses().stream().mapToInt(LfBus::getNum).toArray();
                    if ("1".equals(System.getenv("OLF_GPU_DBG_BUSDUMP"))) {
                        System.err.println("FROZEN-ISLAND(combo) '" + c.contingencyId().replace(GpuBatchStatus.STRATEGY_KEY_SEP, '|') + "' buses="
                                + q.islandedBuses().stream().map(b -> b.getId() + "#" + b.getNum()).toList()
                                + " rows=" + java.util.Arrays.toString(fi.rows()));
                    }
                    boolean genOnIsland = false;
                    for (GenOutage go : gos) {
                        for (int fr : fi.rows()) {
                            if (fr == go.pRow() || fr == go.vRow()) {
                                genOnIsland = true;
                            }
                        }
                    }
                    if (genOnIsland) {
                        continue;
                    }
                    frozenRowsByIndex[i] = fi.rows();
                    frozenTargetsByIndex[i] = fi.targets();
                    if (fi.rows().length > 0) {
                        islandingCount++;
                    }
                } else {
                    frozenRowsByIndex[i] = NO_FROZEN_ROWS;
                    frozenTargetsByIndex[i] = NO_FROZEN_TARGETS;
                }
                genOutagesByIndex[i] = gos;
                batchedIndices.add(i);
            } else if (!c.outagedShunts().isEmpty()) {       // fixed-shunt disconnection: mask the shunt element
                if (canBatchShuntOutage(c.outagedShunts(), elementMapper, gpuData)) {
                    batchedIndices.add(i);
                    frozenRowsByIndex[i] = NO_FROZEN_ROWS;
                    frozenTargetsByIndex[i] = NO_FROZEN_TARGETS;
                }                                            // else: not a packed fixed shunt / DISTR_Q contributor → CPU
            } else if (c.outagedBranches().stream().noneMatch(distrRhoControllers::contains)) {
                batchedIndices.add(i);
                FrozenIsland fi = islandedFrozen(equationSystem, q.islandedBuses());
                islandedNumsByIndex[i] = q.islandedBuses().stream().mapToInt(LfBus::getNum).toArray();
                if ("1".equals(System.getenv("OLF_GPU_DBG_BUSDUMP"))) {
                    System.err.println("FROZEN-ISLAND '" + c.contingencyId().replace(GpuBatchStatus.STRATEGY_KEY_SEP, '|') + "' buses="
                            + q.islandedBuses().stream().map(b -> b.getId() + "#" + b.getNum()).toList()
                            + " rows=" + java.util.Arrays.toString(fi.rows()));
                }
                frozenRowsByIndex[i] = fi.rows();
                frozenTargetsByIndex[i] = fi.targets();
                if (frozenRowsByIndex[i].length > 0) {
                    islandingCount++;
                }
            }
        }
        int rejected = contingencies.size() - batchedIndices.size();
        if (rejected > 0 || islandingCount > 0) {
            LOGGER.info("Batched N-1: {} of {} contingencies routed to the CPU path (absent from the GPU "
                    + "model or islanding the slack); {} fragmenting outages solved on-GPU with frozen islands",
                    rejected, contingencies.size(), islandingCount);
        }

        // Solve the qualified scenarios on the GPU, in chunks bounded by the device-memory budget (the
        // uniform batched factorization holds an LU per scenario, so the footprint scales with the chunk).
        double[][] resultsByIndex = new double[contingencies.size()][];
        if (!batchedIndices.isEmpty()) {
            solveQualifiedInChunks(batchedIndices, contingencies, elementMapper, gpuData, baseState,
                    baseTargets, baseActivation.rowMode(), frozenRowsByIndex, frozenTargetsByIndex, genOutagesByIndex,
                    n, nElements, parameters, tol, reactiveLimitTol, slackDistTol, convEpsPerEq, resultsByIndex,
                    config.lsMode() == 2 ? GpuAcActivationExtractor.extractVarType(equationSystem, n) : null,
                    config.transformerControl()
                            ? GpuBatchTransformerControl.buildTransformerControllers(network, equationSystem, targetVector, baseActivation,
                                    baseTargets, elementMapper)
                            : null,
                    config);
        }

        // Assemble results in the original contingency order: batched scenarios carry their
        // GPU outcome; rejected ones carry the base state with GpuBatchStatus.STATUS_REJECTED.
        List<BatchScenarioResult> results = new ArrayList<>(contingencies.size());
        for (int i = 0; i < contingencies.size(); i++) {
            String id = contingencies.get(i).contingencyId();
            double[] result = resultsByIndex[i];
            if (result != null) {
                double[] state = new double[n];
                System.arraycopy(result, 0, state, 0, n);
                results.add(new BatchScenarioResult(id, state, (int) result[n], result[n + 1], (int) result[n + 2],
                        islandedNumsByIndex[i] != null ? islandedNumsByIndex[i] : new int[0]));
            } else {
                results.add(new BatchScenarioResult(id, baseState.clone(), 0, Double.NaN, GpuBatchStatus.STATUS_REJECTED, new int[0]));
            }
        }
        return results;
    }

    private static final int[] NO_FROZEN_ROWS = new int[0];
    private static final double[] NO_FROZEN_TARGETS = new double[0];

    /** The islanded buses' variable rows to freeze + the clean flat value each holds (phi=0, V=1 pu). */
    private record FrozenIsland(int[] rows, double[] targets) { }

    /**
     * The variable rows (BUS_PHI + BUS_V) of every islanded bus — frozen to identity in a fragmenting
     * scenario's row mode — paired with the clean flat value each is HELD at: {@code phi=0}, {@code V=1} pu.
     * With the outaged branch (the only island↔main link) already masked, freezing these rows makes the
     * islanded sub-block a benign identity block that cannot make the batched factorization singular, while
     * the main component solves correctly. Holding the island at a flat 1∠0 (rather than the base equation
     * target, a P/Q value that produces a nonsensical islanded voltage/angle) gives the disconnected buses a
     * well-defined reported state — no garbage voltage, no spurious limit violations — without touching the main.
     */
    private static FrozenIsland islandedFrozen(EquationSystem<AcVariableType, AcEquationType> es, List<LfBus> islandedBuses) {
        if (islandedBuses.isEmpty()) {
            return new FrozenIsland(NO_FROZEN_ROWS, NO_FROZEN_TARGETS);
        }
        List<Integer> rows = new ArrayList<>(islandedBuses.size() * 2);
        List<Double> targets = new ArrayList<>(islandedBuses.size() * 2);
        for (LfBus bus : islandedBuses) {
            addFrozenVarRow(es, bus.getNum(), AcVariableType.BUS_PHI, 0.0, rows, targets);   // angle 0
            addFrozenVarRow(es, bus.getNum(), AcVariableType.BUS_V, 1.0, rows, targets);     // voltage 1 pu
        }
        int[] outRows = new int[rows.size()];
        double[] outTargets = new double[targets.size()];
        for (int i = 0; i < outRows.length; i++) {
            outRows[i] = rows.get(i);
            outTargets[i] = targets.get(i);
        }
        return new FrozenIsland(outRows, outTargets);
    }

    private static void addFrozenVarRow(EquationSystem<AcVariableType, AcEquationType> es, int busNum,
                                        AcVariableType type, double target, List<Integer> rows, List<Double> targets) {
        Variable<AcVariableType> v = es.getVariableSet().getVariable(busNum, type);
        if (v != null && v.getRow() >= 0) {
            rows.add(v.getRow());
            targets.add(target);
        }
    }

    /**
     * Whether a fixed-shunt disconnection can be batched as a plain element mask. Requires every outaged shunt
     * to be a packed fixed shunt ({@code getElementIndex >= 0}) and NOT a reactive-distribution (DISTR_Q)
     * contributor — a shunt whose reactive feeds a controlled bus's DISTR_Q sum is read unconditionally by the
     * batched DISTR_Q fill (it ignores the disabled mask), so masking it would leave the sum stale. Such a shunt
     * outage is routed to the CPU instead.
     */
    private static boolean canBatchShuntOutage(List<LfShunt> shunts, GpuBatchElementMapper elementMapper,
                                               GpuAcData gpuData) {
        int shOffset = elementMapper.getClosedBranchCount() + elementMapper.getOpenBranchCount();
        int[] dqElem = gpuData.dqElem();
        int[] dqKind = gpuData.dqKind();
        for (LfShunt shunt : shunts) {
            int elementIndex = elementMapper.getElementIndex(shunt);
            if (elementIndex < 0) {
                return false;                                // not a packed fixed shunt (controller/SVC shunt)
            }
            int packIndex = elementIndex - shOffset;
            for (int i = 0; i < dqKind.length; i++) {
                if (dqKind[i] == 1 && dqElem[i] == packIndex) {
                    return false;                            // shunt feeds a DISTR_Q sum — mask wouldn't update it
                }
            }
        }
        return true;
    }

    /**
     * Solve the qualified scenarios on the GPU in chunks sized by {@link #resolveChunkSize}
     * (auto from device memory, or a manual cap). ONE native context, sized to the chunk, is created and reused across all
     * chunks: cuDSS's symbolic ANALYSIS (on the shared pattern, identical for every chunk) is
     * done once on the first chunk and reused, so chunking costs nothing in amortization — it
     * only bounds the resident per-scenario LU factors. The last (partial) chunk is padded
     * with base-case scenarios (no element disabled): they converge at iteration 0 and their
     * results are discarded.
     */

    private static void solveQualifiedInChunks(List<Integer> batchedIndices, List<BatchContingency> contingencies,
                                               GpuBatchElementMapper elementMapper, GpuAcData gpuData,
                                               double[] baseState, double[] baseTargets, int[] baseRowMode,
                                               int[][] frozenRowsByIndex, double[][] frozenTargetsByIndex,
                                               GenOutage[][] genOutagesByIndex,
                                               int n, int nElements, NewtonRaphsonParameters parameters,
                                               double tol, double reactiveLimitTol, double slackDistTol,
                                               double convEpsPerEq, double[][] resultsByIndex, int[] varType,
                                               List<GpuBatchTransformerControl.TransformerController> transformerControllers,
                                               BatchConfig config) {
        int nb = batchedIndices.size();
        int chunkSize = GpuBatchContextCache.resolveChunkSize(nb, n, gpuData);
        int nChunks = (nb + chunkSize - 1) / chunkSize;

        // Reuse a cached analyzed context when the pattern/admittances/chunk/mode match (across
        // SA runs on the same network) — the context is NOT destroyed here, it stays cached.
        long handle = GpuBatchContextCache.acquireBatchContext(gpuData, chunkSize, GpuBatchContextCache.resolveBatchMode(), n);

        // The base state / targets are IDENTICAL for every scenario (a single-branch N-1 starts from the
        // same converged base operating point): passed as single length-n rows and tiled on-device. The
        // disabled mask is per-scenario (each chunk clears the one flag each slot set, sets the new one).
        // The ROW MODE is per-scenario too: a non-fragmenting outage shares the base row (same PV/slack
        // layout); a fragmenting one gets a cloned base row with its islanded buses frozen to identity.
        boolean[][] disabledMask = new boolean[chunkSize][nElements];
        int[][] disabledIdx = new int[chunkSize][];      // element indices each slot currently disables (1-2), null = none
        int[][] rowModePerScenario = new int[chunkSize][];   // per-scenario row mode (re-assigned every chunk)
        // Per-scenario target overrides (generator contingencies): flat (dTargetIndex = slot*n + row, value)
        // lists rebuilt per chunk. Empty when no generator outage is in the chunk.
        List<Integer> ovIndexList = new ArrayList<>();
        List<Double> ovValueList = new ArrayList<>();
        java.util.Map<Integer, Integer> ziElemByBranchNumChunk = new java.util.HashMap<>();
        for (int z = 0; z < gpuData.baseOpenZiElements().length; z++) {
            ziElemByBranchNumChunk.put(gpuData.baseOpenZiBranchNums()[z], gpuData.baseOpenZiElements()[z]);
        }
        int nziBase = gpuData.ziIdx().length / 11 - gpuData.baseOpenZiElements().length;
        java.util.Map<Integer, Integer> ziElemByNumAll = new java.util.HashMap<>();
        for (int z = 0; z < gpuData.ziIdx().length / 11; z++) {
            ziElemByNumAll.put(gpuData.ziBranchNums()[z], z);
        }
        List<Integer> ziOffSlotElemList = new ArrayList<>(); // per-scenario zi-element outages (switch opens)
        List<Integer> ziOnSlotElemList = new ArrayList<>(); // per-scenario zi-element enables (switch closes)
        List<Integer> shOvSlotElemList = new ArrayList<>();  // per-scenario shunt admittance overrides
        List<Double> shOvGbList = new ArrayList<>();
        List<Integer> r1OvSlotElemList = new ArrayList<>();  // per-scenario fixed-ratio tap overrides
        List<Double> r1OvValList = new ArrayList<>();
        int shPackOffset = elementMapper.getClosedBranchCount() + elementMapper.getOpenBranchCount();
        // Per-scenario distributed-slack reweight (a generator loss at a DS-participant bus): one affected DS
        // bus per slot (dsOvBus[s] = its DS-pack index, -1 = none) whose factor/base/headroom/footroom and the
        // factor sum shrink by the lost gen's share. Only allocated when the network has distributed slack.
        boolean hasDs = gpuData.dsPhiRow().length > 0;
        int[] dsOvBus = hasDs ? new int[chunkSize] : new int[0];
        double[] dsOvFactorDelta = hasDs ? new double[chunkSize] : new double[0];
        double[] dsOvBaseDelta = hasDs ? new double[chunkSize] : new double[0];
        double[] dsOvMaxDelta = hasDs ? new double[chunkSize] : new double[0];
        double[] dsOvMinDelta = hasDs ? new double[chunkSize] : new double[0];
        // TWO-PHASE ordering (operator strategies, warm continuation): plain contingencies first,
        // then (ctg, strategy) PAIRS — each pair chunk is seeded with its PARENT contingency's
        // solved state (x0PerScenario) and reactive-limits pinning (rlStateIn), so the strategy
        // solve continues the contingency's outer-loop trajectory exactly like the CPU's
        // per-contingency action solve. Pair whose parent did not converge is dropped (CPU path).
        List<Integer> ctgIdxList = new ArrayList<>();
        List<Integer> pairIdxList = new ArrayList<>();
        for (int idx : batchedIndices) {
            (contingencies.get(idx).contingencyId().indexOf(GpuBatchStatus.STRATEGY_KEY_SEP) >= 0 ? pairIdxList : ctgIdxList).add(idx);
        }
        java.util.Map<String, Integer> indexById = new java.util.HashMap<>();
        for (int i = 0; i < contingencies.size(); i++) {
            indexById.put(contingencies.get(i).contingencyId(), i);
        }
        int nrl = gpuData.rlVrow().length;
        java.util.Map<Integer, int[]> rlByIndex = new java.util.HashMap<>();    // ctg index -> its nrl RL-state slice
        // Host-side base RL seed (mirrors initRlLimitTypeBatch) for PADDING slots of pair chunks.
        int[] baseRlSeed = new int[nrl];
        for (int i = 0; i < nrl; i++) {
            int vrow = gpuData.rlVrow()[i];
            baseRlSeed[i] = baseRowMode[vrow] == 1 ? (baseState[vrow] < gpuData.rlVtarget()[i] ? 2 : 1) : 0;
        }
        try {
            long startTime = System.currentTimeMillis();
            for (List<Integer> group : List.of(ctgIdxList, pairIdxList)) {
                boolean pairPhase = group == pairIdxList;
                int nbg = group.size();
                for (int chunkStart = 0; chunkStart < nbg; chunkStart += chunkSize) {
                    int realCount = Math.min(chunkSize, nbg - chunkStart);
                    ovIndexList.clear();
                    ovValueList.clear();
                    ziOffSlotElemList.clear();
                    ziOnSlotElemList.clear();
                    shOvSlotElemList.clear();
                    shOvGbList.clear();
                    r1OvSlotElemList.clear();
                    r1OvValList.clear();
                    if (hasDs) {
                        java.util.Arrays.fill(dsOvBus, -1);
                        java.util.Arrays.fill(dsOvFactorDelta, 0);
                        java.util.Arrays.fill(dsOvBaseDelta, 0);
                        java.util.Arrays.fill(dsOvMaxDelta, 0);
                        java.util.Arrays.fill(dsOvMinDelta, 0);
                    }
                    for (int s = 0; s < chunkSize; s++) {
                        if (disabledIdx[s] != null) {        // clear this slot's outage(s) from the previous chunk
                            for (int ei : disabledIdx[s]) {
                                disabledMask[s][ei] = false;
                            }
                            disabledIdx[s] = null;
                        }
                        for (int e : gpuData.baseOpenCbElements()) {
                            disabledMask[s][e] = true;           // base-open elements stay OUT unless this scenario recloses them
                        }
                        rowModePerScenario[s] = baseRowMode;     // default: base layout (padding + non-fragmenting)
                        if (s < realCount) {                 // real scenario: disable its outaged branch(es)
                            int ci = group.get(chunkStart + s);
                            List<LfBranch> branches = contingencies.get(ci).outagedBranches();
                            List<LfShunt> shunts = contingencies.get(ci).outagedShunts();
                            int[] eis = new int[branches.size() + shunts.size()];
                            int ei = 0;
                            for (LfBranch branch : branches) {
                                int idx = elementMapper.getElementIndex(branch);          // qualification guarantees >= 0
                                disabledMask[s][idx] = true;
                                eis[ei++] = idx;
                            }
                            for (LfShunt shunt : shunts) {                                // fixed-shunt disconnection
                                int idx = elementMapper.getElementIndex(shunt);          // qualification guarantees >= 0
                                disabledMask[s][idx] = true;
                                eis[ei++] = idx;
                            }
                            disabledIdx[s] = eis;
                            int[] frozen = frozenRowsByIndex[ci];
                            if (frozen.length > 0) {         // fragmenting outage: freeze the islanded buses' rows
                                int[] row = baseRowMode.clone();
                                double[] frozenTargets = frozenTargetsByIndex[ci];
                                for (int k = 0; k < frozen.length; k++) {
                                    row[frozen[k]] = 0;      // identity row — bus decoupled from the main component
                                    // ...held at a CLEAN flat state (V=1, phi=0) via the target override, NOT the base
                                    // equation target (a P/Q value that gives a nonsensical islanded voltage/angle).
                                    ovIndexList.add(s * n + frozen[k]);
                                    ovValueList.add(frozenTargets[k]);
                                }
                                rowModePerScenario[s] = row;
                            }
                            for (LfBranch ziOut : contingencies.get(ci).outagedZiBranches()) {
                                Integer ze = ziElemByNumAll.get(ziOut.getNum());
                                ziOffSlotElemList.add(s);
                                ziOffSlotElemList.add(ze);
                                // the decoupled branch's ZERO_V/ZERO_PHI rows become identities again
                                // (dummies pinned to 0) — the inverse of a close
                                int dpRow = gpuData.ziIdx()[ze * 11 + 4];
                                int dqRow = gpuData.ziIdx()[ze * 11 + 5];
                                int[] rowZi = rowModePerScenario[s] == baseRowMode ? baseRowMode.clone() : rowModePerScenario[s];
                                rowZi[dpRow] = 0;
                                rowZi[dqRow] = 0;
                                ovIndexList.add(s * n + dpRow);
                                ovValueList.add(0.0);
                                ovIndexList.add(s * n + dqRow);
                                ovValueList.add(0.0);
                                rowModePerScenario[s] = rowZi;
                            }
                            for (LfBranch reclosed : contingencies.get(ci).reclosedBranches()) {
                                if (reclosed.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC)) {
                                    Integer ze = ziElemByBranchNumChunk.get(reclosed.getNum());
                                    ziOnSlotElemList.add(s);
                                    ziOnSlotElemList.add(ze);
                                    // flip the two dummy rows from identity (pinned 0) to the ZERO_V/ZERO_PHI
                                    // coupling; targets stay 0 (switch: rho=1, phase shift 0)
                                    int dpRow = gpuData.ziIdx()[ze * 11 + 4];
                                    int dqRow = gpuData.ziIdx()[ze * 11 + 5];
                                    int[] row = rowModePerScenario[s] == baseRowMode ? baseRowMode.clone() : rowModePerScenario[s];
                                    row[dpRow] = 1;
                                    row[dqRow] = 1;
                                    rowModePerScenario[s] = row;
                                } else {
                                    disabledMask[s][elementMapper.getElementIndex(reclosed)] = false;
                                }
                            }
                            for (TapR1Override tap : contingencies.get(ci).tapOverrides()) {
                                r1OvSlotElemList.add(s);
                                r1OvSlotElemList.add(elementMapper.getElementIndex(tap.branch()));   // cb pack index == element index
                                for (double v : tap.cbRow()) {
                                    r1OvValList.add(v);
                                }
                            }
                            for (ShuntShift sh : contingencies.get(ci).shuntShifts()) {
                                shOvSlotElemList.add(s);
                                shOvSlotElemList.add(elementMapper.getElementIndex(sh.shunt()) - shPackOffset);
                                shOvGbList.add(sh.dg());
                                shOvGbList.add(sh.db());
                            }
                            GenOutage[] gos = genOutagesByIndex[ci];
                            if (gos != null) {               // generator loss(es): per-BUS target overrides (merged by bus;
                                for (GenOutage go : gos) {   // at most one DS-participant bus per contingency by construction)
                                    ovIndexList.add(s * n + go.pRow());
                                    ovValueList.add(go.pTarget());
                                    if (go.qOverride()) {    // Q target sheds the lost gens' Q: on the PV->PQ flip AND
                                        ovIndexList.add(s * n + go.vRow());   // on an ALREADY-PQ bus losing fixed-Q gens (#21)
                                        ovValueList.add(go.qTarget());
                                    }
                                    if (go.toPq()) {
                                        int[] row = rowModePerScenario[s] == baseRowMode ? baseRowMode.clone() : rowModePerScenario[s];
                                        row[go.vRow()] = 1;  // BUS_V row becomes the reactive-power equation (PV→PQ)
                                        rowModePerScenario[s] = row;
                                    }
                                    if (hasDs && go.dsBus() >= 0) {   // lost gen(s) on a DS-participant bus: per-scenario reweight
                                        dsOvBus[s] = go.dsBus();
                                        dsOvFactorDelta[s] = go.dsFactorDelta();
                                        dsOvBaseDelta[s] = go.dsBaseDelta();
                                        dsOvMaxDelta[s] = go.dsMaxDelta();
                                        dsOvMinDelta[s] = go.dsMinDelta();
                                    }
                                }
                            }
                        }
                        // else: padding scenario = base case (no element disabled), discarded below
                    }
                    int[] targetOverrideIndex = ovIndexList.stream().mapToInt(Integer::intValue).toArray();
                    double[] targetOverrideValue = ovValueList.stream().mapToDouble(Double::doubleValue).toArray();
                    if (pairPhase && "1".equals(System.getenv("OLF_GPU_DBG_PAIR_OV"))) {
                        for (int s2 = 0; s2 < realCount; s2++) {
                            StringBuilder sb = new StringBuilder("GPU-PAIR-OV "
                                    + contingencies.get(group.get(chunkStart + s2)).contingencyId().replace(GpuBatchStatus.STRATEGY_KEY_SEP, '|'));
                            for (int k2 = 0; k2 < targetOverrideIndex.length; k2++) {
                                if (targetOverrideIndex[k2] / n == s2) {
                                    int row = targetOverrideIndex[k2] % n;
                                    sb.append(String.format(" row%d=%.4f(base %.4f)", row, targetOverrideValue[k2], baseTargets[row]));
                                }
                            }
                            System.err.println(sb);
                        }
                        System.err.println("GPU-PAIR-SHOV slotElem=" + shOvSlotElemList + " gb=" + shOvGbList
                                + " shPackOffset=" + shPackOffset);
                    }
                    // Pair phase: warm continuation from each parent contingency's state + RL pinning.
                    double[] x0Full = null;
                    int[] rlStateIn = null;
                    if (pairPhase) {
                        x0Full = new double[chunkSize * n];
                        rlStateIn = nrl > 0 ? new int[chunkSize * nrl] : null;
                        for (int s = 0; s < chunkSize; s++) {
                            double[] parent = null;
                            int[] parentRl = null;
                            if (s < realCount) {
                                String pairId = contingencies.get(group.get(chunkStart + s)).contingencyId();
                                Integer pIdx = indexById.get(pairId.substring(0, pairId.indexOf(GpuBatchStatus.STRATEGY_KEY_SEP)));
                                if (pIdx != null && resultsByIndex[pIdx] != null
                                        && (int) resultsByIndex[pIdx][n + 2] == GpuBatchStatus.STATUS_CONVERGED) {
                                    parent = resultsByIndex[pIdx];
                                    parentRl = rlByIndex.get(pIdx);
                                }
                            }
                            for (int i = 0; i < n; i++) {
                                x0Full[s * n + i] = parent != null ? parent[i] : baseState[i];
                            }
                            if (rlStateIn != null) {
                                System.arraycopy(parentRl != null ? parentRl : baseRlSeed, 0, rlStateIn, s * nrl, nrl);
                            }
                        }
                    }
                    boolean dbgPairRl = pairPhase && nrl > 0 && "1".equals(System.getenv("OLF_GPU_DBG_PAIR_RL"));
                    int[] rlStateOut = (!pairPhase && !pairIdxList.isEmpty() || dbgPairRl) && nrl > 0 ? new int[chunkSize * nrl] : null;
                    double[][] chunkResults = GpuAcNewtonSolver.solveBatchContext(handle, baseState, baseTargets,
                            rowModePerScenario, baseRowMode, disabledMask, targetOverrideIndex, targetOverrideValue,
                            dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta,
                            parameters.getMaxIterations(), tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                            config.lsMode(), config.maxDv(), config.maxDphi(),
                            config.lsMode() == 2 ? varType : null, x0Full,
                            shOvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                            shOvGbList.stream().mapToDouble(Double::doubleValue).toArray(),
                            rlStateIn, rlStateOut,
                            r1OvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                            r1OvValList.stream().mapToDouble(Double::doubleValue).toArray(),
                            config.maxOuterIterations(), config.maxPqPvSwitch(), config.dsRejectResidualFlag(),
                            nziBase, ziOnSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                            ziOffSlotElemList.stream().mapToInt(Integer::intValue).toArray());
                    if (dbgPairRl) {
                        for (int s2 = 0; s2 < realCount; s2++) {
                            StringBuilder sb = new StringBuilder("GPU-PAIR-RL " + contingencies.get(group.get(chunkStart + s2)).contingencyId().replace(GpuBatchStatus.STRATEGY_KEY_SEP, '|'));
                            for (int i2 = 0; i2 < nrl; i2++) {
                                int v = rlStateOut[s2 * nrl + i2];
                                if ((v & 3) != 0) {
                                    sb.append(" ctl").append(i2).append("=lt").append(v & 3).append("/sc").append(v >> 2);
                                }
                            }
                            System.err.println(sb);
                        }
                    }
                    if (rlStateOut != null && !pairPhase) {      // capture each ctg's final RL pinning for its pairs
                        for (int s = 0; s < realCount; s++) {
                            int[] slice = new int[nrl];
                            System.arraycopy(rlStateOut, s * nrl, slice, 0, nrl);
                            rlByIndex.put(group.get(chunkStart + s), slice);
                        }
                    }
                    if (transformerControllers != null && !transformerControllers.isEmpty()) {
                        chunkResults = GpuBatchTransformerControl.orchestrateTransformerControl(handle, chunkSize, realCount, n,
                                chunkResults, transformerControllers, rowModePerScenario, baseRowMode, disabledMask,
                                targetOverrideIndex, targetOverrideValue,
                                dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta,
                                baseState, baseTargets, parameters, tol, reactiveLimitTol, slackDistTol, convEpsPerEq, varType,
                                config);
                    }
                    for (int s = 0; s < realCount; s++) {
                        resultsByIndex[group.get(chunkStart + s)] = chunkResults[s];
                    }
                }
            }
            long elapsed = System.currentTimeMillis() - startTime;
            LOGGER.info("Batched N-1: {} GPU scenarios in {} chunk(s) of <= {} solved in {} ms ({} ms/scenario)",
                    nb, nChunks, chunkSize, elapsed, (double) elapsed / nb);
        } catch (RuntimeException e) {
            GpuBatchContextCache.clearContextCache();   // a failed solve may leave the context inconsistent — rebuild next time
            throw e;
        }
    }

    /** Absolute upper bound on an auto-sized chunk, so a huge free-memory reading never yields a pathological chunk. */
    private GpuBatchedSecurityAnalysisSolver() {
    }
}
