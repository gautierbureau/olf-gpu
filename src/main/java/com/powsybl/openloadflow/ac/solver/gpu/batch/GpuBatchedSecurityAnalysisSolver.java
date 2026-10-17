/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.vector.gpu.DistrRhoPacker;
import com.powsybl.commons.report.ReportNode;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivation;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivationExtractor;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataCache;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.TargetVector;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfNetwork;
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
        com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDsParams dsParams = config.dsParams();
        GpuAcData gpuData = baseOpenIds.isEmpty() && baseOpenZiIds.isEmpty()
                ? GpuAcDataCache.extract(network, equationSystem, distributedSlack, reactiveLimits, dsParams)
                : GpuAcDataExtractor.extract(network, equationSystem, distributedSlack, reactiveLimits, baseOpenIds, baseOpenZiIds, dsParams);
        GpuAcActivation baseActivation = GpuAcActivation.fromEquationSystem(network, equationSystem);
        String saEqDump = System.getenv("OLF_GPU_SA_EQ_DUMP");   // the SA extraction's row map (GpuEquationDump)
        if (saEqDump != null && !saEqDump.isEmpty()) {
            try (java.io.Writer w = java.nio.file.Files.newBufferedWriter(java.nio.file.Path.of(saEqDump))) {
                // The REAL base row modes, not a zero-filled placeholder. This dump is half of the
                // row-by-row comparison against OLF_GPU_EQ_DUMP_CTG, and with all-zero modes every row
                // read "mode=0" (identity) — including the coupler DUMMY_Q rows that carry the DISTR_Q
                // terms — which silently invents a system difference wherever OLF has a power row.
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuEquationDump.write(w, network, equationSystem, gpuData,
                        baseActivation.rowMode());
            } catch (java.io.IOException | RuntimeException e) {
                LOGGER.warn("OLF_GPU_SA_EQ_DUMP failed: {}", e.toString());
            }
        }
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
                GpuBatchContingencyValidator.qualify(network, contingencies, elementMapper, handleIslanding,
                        config.slackRelocation());
        // Outaging a DISTR_RHO controller transformer reforms the ratio-distribution 1/count coefficients,
        // which the fixed batch pattern cannot represent — route those to the CPU (the DISTR_Q pass IS
        // batched, so DISTR_RHO is the only family that needs a contributor-outage rejection).
        Set<LfBranch> distrRhoControllers = gpuData.getDistrRhoContributionCount() > 0
                ? DistrRhoPacker.distrRhoControllerBranches(network, equationSystem) : Set.of();
        // Distributed-slack factor denominator (Σ maxP/droop over ALL participating gens, incl. the slack's)
        // and whether the slack itself participates: a generator at a DS-participant bus is batched with a
        // per-scenario reweight (computeGenOutage), but only when the slack does not participate.
        double dsSum = 0;
        if (distributedSlack) {                              // no slack distribution ⇒ no per-scenario DS reweight
            for (LfBus b : network.getBuses()) {
                for (LfGenerator g : b.getGenerators()) {
                    dsSum += dsParams.factor(g);
                }
            }
        }
        boolean slackParticipates = java.util.Arrays.stream(gpuData.dsFactor()).sum() < 1 - 1e-9;
        // Reactive capability CURVES at runtime P (GpuBatchRlCurves): each tracked generator is evaluated at its own
        // P (its anchor plus its own slack move); a strategy pair moving an untracked curve generator -> CPU.
        GpuBatchRlCurves.Channel rlCurves = reactiveLimits
                ? GpuBatchRlCurves.build(network, equationSystem, gpuData, config.dsParams(), distributedSlack, baseTargets,
                        config.rlCurveExtrapolate())
                : null;
        // forceTargetQInReactiveLimits: the curve generators whose clamped Q target follows their P (null = none).
        GpuBatchForcedTargetQ.Channel forcedTargetQ = reactiveLimits
                ? GpuBatchForcedTargetQ.build(network, equationSystem, gpuData, config.dsParams(), config.forceTargetQ(),
                        distributedSlack, baseTargets, baseActivation.rowMode(), config.rlCurveExtrapolate())
                : null;
        // Couplers whose DUMMY rows host a bus's power equations under the tree relocation: an island has to
        // freeze those rows too (see GpuBatchQualification.islandedFrozen). Empty when the relocation is off.
        Set<LfBranch> relocatedCouplers = GpuAcDataExtractor.relocatedCouplerBranches(network, equationSystem);
        GpuBatchQualification.ScenarioPlan plan = GpuBatchQualification.planScenarios(contingencies, qualifications,
                elementMapper, gpuData, equationSystem, dsSum, slackParticipates, dsParams, distrRhoControllers,
                relocatedCouplers, reactiveLimits, config.incrementalTransformerControl());
        List<Integer> batchedIndices = plan.batchedIndices();
        if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
            System.err.println("PLAN batched=" + batchedIndices.size() + " of " + contingencies.size());
        }
        int islandingCount = plan.islandingCount();
        int rejected = contingencies.size() - batchedIndices.size();
        if (rejected > 0 || islandingCount > 0) {
            LOGGER.info("Batched N-1: {} of {} contingencies routed to the CPU path (absent from the GPU "
                    + "model or islanding the slack); {} fragmenting outages solved on-GPU with frozen islands",
                    rejected, contingencies.size(), islandingCount);
        }

        // Solve the qualified scenarios on the GPU, in chunks bounded by the device-memory budget (the
        // uniform batched factorization holds an LU per scenario, so the footprint scales with the chunk).
        double[][] resultsByIndex = new double[contingencies.size()][];
        // The replayed contingency loss pre-distribution per contingency (pu); NaN where none was replayed.
        double[] preDistributedByIndex = new double[contingencies.size()];
        java.util.Arrays.fill(preDistributedByIndex, Double.NaN);
        // What the device distributed in all (pre-distribution + its own distributed-slack passes, pu); NaN = unknown.
        double[] distributedByIndex = new double[contingencies.size()];
        java.util.Arrays.fill(distributedByIndex, Double.NaN);
        if (!batchedIndices.isEmpty()) {
            // PER_EQUATION_TYPE stopping criterion: per-row thresholds from the equation system's
            // variable pairing (null under UNIFORM_CRITERIA — the chunk solver then CLEARS the cached
            // context's thresholds, since contexts survive across runs).
            double[][] rowCriteria = config.perEqCriteria() != null
                    ? GpuAcActivationExtractor.extractRowCriteria(network, equationSystem, baseActivation, n,
                            config.perEqCriteria().p(), config.perEqCriteria().q(), config.perEqCriteria().v(),
                            config.perEqCriteria().angle(), config.perEqCriteria().ratio(), config.perEqCriteria().b(),
                            config.perEqCriteria().eps())
                    : null;
            // Every phase's tables, built by the SAME function the base load flow calls (#58): a phase
            // added there reaches both routes or neither.
            GpuBatchPhases.Built phases = GpuBatchPhases.build(new GpuBatchPhases.Request(
                    config.voltageMonitoring(),
                    config.incrementalTransformerControl(),
                    config.incrementalShuntControl(),
                    config.simpleShuntControl(),
                    config.incrementalRpcControl(), config.incrementalPhaseControl(), config.incTvcMaxTapShift(),
                    config.continuousPhaseControl(),
                    config.transformerControl() || config.simpleTransformerControl(),
                    config.tvcUseInitialTapPosition(), config.genVcMinNominalV(),
                    config.secondaryVoltageControl(), config.minPlausibleTargetV(), config.maxPlausibleTargetV()),
                    network, equationSystem, targetVector, baseActivation, baseTargets,
                    gpuData.rlVrow(), gpuData.rlBusNum(), n);
            // All-or-nothing, the same rule the base load flow applies: a loop the builders represented
            // only PARTIALLY would have the device drive a SUBSET of it, and this path has no host-pause
            // slot to hand it back to, so the whole run goes to the CPU (resultsByIndex stays null).
            if (GpuBatchPhaseOwnership.batchable()) {
                GpuBatchChunkSolver.solveQualifiedInChunks(batchedIndices, contingencies, elementMapper, gpuData, baseState,
                        baseTargets, baseActivation.rowMode(), plan.frozenRowsByIndex(), plan.frozenTargetsByIndex(), plan.genOutagesByIndex(),
                        plan.activePowerLossByIndex(),
                        GpuBatchHvdcAcEmulation.elementIndices(network, nElements), rowCriteria, rlCurves, forcedTargetQ,
                        plan.dqOvContribByIndex(), plan.dqOvWByIndex(), plan.dqKeyOvMemberByIndex(), plan.dqKeyOvByIndex(),
                        plan.drOvContribByIndex(), plan.drOvCByIndex(), plan.islandedDsByIndex(),
                        plan.islandedRlByIndex(), plan.islandedHvdcsByIndex(), plan.dqSubByIndex(), plan.slackRelocations(),
                        n, nElements, parameters, tol, reactiveLimitTol, slackDistTol, convEpsPerEq, resultsByIndex,
                        preDistributedByIndex, distributedByIndex, config.lsMode() == 2 ? GpuAcActivationExtractor.extractVarType(equationSystem, n) : null,
                        phases,
                        config.hvdcFreezing() || config.hvdcLimits()
                                ? GpuBatchHvdcAcEmulation.buildControllers(network, equationSystem, nElements, gpuData)
                                : null,
                        config);
            }
        }

        // Assemble results in the original contingency order: batched scenarios carry their
        // GPU outcome; rejected ones carry the base state with GpuBatchStatus.STATUS_REJECTED.
        List<BatchScenarioResult> results = new ArrayList<>(contingencies.size());
        for (int i = 0; i < contingencies.size(); i++) {
            String id = contingencies.get(i).contingencyId();
            double[] result = resultsByIndex[i];
            if (result != null && result.length == 3) {    // the state went to the sink: the result carries no state
                results.add(new BatchScenarioResult(id, null, (int) result[0], result[1], (int) result[2],
                        plan.islandedNumsByIndex()[i] != null ? plan.islandedNumsByIndex()[i] : new int[0],
                        preDistributedByIndex[i], distributedByIndex[i]));
            } else if (result != null) {                   // the state is the result row's first n entries
                results.add(new BatchScenarioResult(id, result, (int) result[n], result[n + 1], (int) result[n + 2],
                        plan.islandedNumsByIndex()[i] != null ? plan.islandedNumsByIndex()[i] : new int[0],
                        preDistributedByIndex[i], distributedByIndex[i]));
            } else {
                // Never batched: the plan refused it. Named by id for the harness, because the same
                // STATUS_REJECTED also comes back for a scenario the device TRIED and threw away, and the
                // census counts cannot tell the two apart per contingency.
                com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.planRefused(id);
                results.add(new BatchScenarioResult(id, baseState.clone(), 0, Double.NaN, GpuBatchStatus.STATUS_REJECTED, new int[0],
                        Double.NaN, Double.NaN));
            }
        }
        return results;
    }

    private GpuBatchedSecurityAnalysisSolver() {
    }
}
