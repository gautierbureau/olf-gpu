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
import com.powsybl.openloadflow.ac.equations.vector.gpu.DistrRhoPacker;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivation;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivationExtractor;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataCache;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcElementMapper;
import com.powsybl.openloadflow.ac.outerloop.gpu.DevicePhaseOwnership;
import com.powsybl.openloadflow.ac.outerloop.gpu.DevicePhases;
import com.powsybl.openloadflow.ac.outerloop.gpu.ForceTargetQInReactiveLimits;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuAreaInterchangeControlOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuAutomationSystemOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuHvdcAcEmulationLimitsOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.gpu.ReactiveCapabilityCurves;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.BatchContingency;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.GpuBatchContingencyValidator;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.GpuBatchQualification;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.GpuBatchScopeCensus;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuAcSolverStatus;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuStateVectorScalingMode;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.TargetVector;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.util.gpu.GpuTrace;
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

    /** OLF default outer-loop convergence tolerances (per-unit on the 100 MVA base): slackBusPMaxMismatch =
     *  1 MW → 1e-2 pu, maxReactivePowerMismatch = 1 MVAr → 1e-2/… → 1e-4 pu. The device uses these so its
     *  outer-loop fixed point matches the CPU loadflow's; a test wanting machine-precision agreement passes
     *  tight values on BOTH sides (a loose tolerance leaves a redistribution band ~its size). */
    public static final double DEFAULT_REACTIVE_LIMIT_TOL = 1e-4;
    public static final double DEFAULT_SLACK_DIST_TOL = 1e-2;

    /** TESTS ONLY: record every batched contingency's outer-loop statistics into {@code sink}, by contingency id: {solve
     *  passes, first solve's Newton iterations, total Newton iterations, outer-loop checks, its chunk's batched Newton
     *  iterations}. Null stops recording. */
    public static void recordOuterLoopStatsForTests(java.util.Map<String, int[]> sink) {
        GpuBatchChunkSolver.outerLoopStatsForTests = sink;
    }

    /** TESTS ONLY: refill the bank as soon as {@code ended} scenarios have ended (1: each slot as soon as its scenario
     *  ends), instead of when a share of it is free. Null goes back to the share. */
    public static void refillAtForTests(Integer ended) {
        GpuBatchChunkSolver.refillAtForTests = ended;
        GpuBatchChunkSolver.REFILLS_FOR_TESTS.set(0);
    }

    /** TESTS ONLY: how many installs went into a bank whose other slots still held scenarios solving, since the last
     *  {@link #refillAtForTests} call. */
    public static int refillsForTests() {
        return GpuBatchChunkSolver.REFILLS_FOR_TESTS.get();
    }

    /**
     * Solve the request's contingencies on the GPU: extract one batched context from the base case and run every
     * scenario in one Newton-Raphson loop with the device outer loops the request declares.
     *
     * @return per-scenario results (state, iterations, convergence status)
     */
    public static List<BatchScenarioResult> solveBatchN1(BatchSolveRequest request) {
        LfNetwork network = request.network();
        double[] baseState = request.baseState();
        List<BatchContingency> contingencies = request.contingencies();
        EquationSystem<AcVariableType, AcEquationType> equationSystem = request.equationSystem();
        TargetVector<AcVariableType, AcEquationType> targetVector = request.targetVector();
        NewtonRaphsonParameters parameters = request.parameters();
        double tol = request.tol();
        boolean distributedSlack = request.distributedSlack();
        boolean reactiveLimits = request.reactiveLimits();
        double reactiveLimitTol = request.reactiveLimitTol();
        double slackDistTol = request.slackDistTol();
        double convEpsPerEq = request.convEpsPerEq();
        ReportNode reportNode = request.reportNode();
        BatchConfig config = request.config();
        config.resolvedGpuParameters().install();            // this solve's linear solver, never a previous run's
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
                .collect(java.util.stream.Collectors.toCollection(java.util.HashSet::new));
        if (config.automationSystem()) {
            // a branch an automation system closes is packed open, closed by its bit in the scenario's mask
            baseOpenIds.addAll(GpuAutomationSystemOuterLoop.baseOpenOperatedBranchIds(network));
        }
        java.util.Set<String> baseOpenZeroImpedanceBranchIds = contingencies.stream()
                .flatMap(c -> c.reclosedBranches().stream())
                .filter(b -> b.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC))
                .map(LfBranch::getId)
                .collect(java.util.stream.Collectors.toSet());
        com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDistributedSlackParameters distributedSlackParameters = config.distributedSlackParameters();
        GpuAcData gpuData = baseOpenIds.isEmpty() && baseOpenZeroImpedanceBranchIds.isEmpty()
                ? GpuAcDataCache.extract(network, equationSystem, distributedSlackParameters)
                : GpuAcDataExtractor.extract(network, equationSystem, baseOpenIds, baseOpenZeroImpedanceBranchIds, distributedSlackParameters);
        GpuAcActivation baseActivation = GpuAcActivation.fromEquationSystem(network, equationSystem);
        String saEqDump = GpuTrace.SA_EQ_DUMP.value();   // the SA extraction's row map (GpuEquationDump)
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
        GpuAcElementMapper elementMapper = new GpuAcElementMapper(network, baseOpenIds);

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
        double distributedSlackRawFactorSum = 0;
        if (distributedSlack) {                              // no slack distribution ⇒ no per-scenario DS reweight
            for (LfBus b : network.getBuses()) {
                for (LfGenerator g : b.getGenerators()) {
                    distributedSlackRawFactorSum += distributedSlackParameters.factor(g);
                }
            }
        }
        boolean slackParticipates = java.util.Arrays.stream(gpuData.distributedSlack().busFactor()).sum() < 1 - 1e-9;
        // Reactive capability CURVES at runtime P (ReactiveCapabilityCurves): each tracked generator is evaluated at its own
        // P (its anchor plus its own slack move); a strategy pair moving an untracked curve generator -> CPU.
        ReactiveCapabilityCurves.Channel reactiveCapabilityCurves = reactiveLimits
                ? ReactiveCapabilityCurves.build(network, equationSystem, gpuData, config.distributedSlackParameters(), distributedSlack, baseTargets,
                        config.extrapolateReactiveLimits())
                : null;
        // forceTargetQInReactiveLimits: the curve generators whose clamped Q target follows their P (null = none).
        ForceTargetQInReactiveLimits.Channel forcedTargetQ = reactiveLimits
                ? ForceTargetQInReactiveLimits.build(network, equationSystem, gpuData, config.distributedSlackParameters(), config.forceTargetQ(),
                        distributedSlack, baseTargets, baseActivation.rowMode(), config.extrapolateReactiveLimits())
                : null;
        // Couplers whose DUMMY rows host a bus's power equations under the tree relocation: an island has to
        // freeze those rows too (see IslandedBuses.islandedFrozen). Empty when the relocation is off.
        Set<LfBranch> relocatedCouplers = GpuAcDataExtractor.relocatedCouplerBranches(network, equationSystem);
        GpuBatchQualification.ScenarioPlan plan = GpuBatchQualification.planScenarios(contingencies, qualifications,
                elementMapper, gpuData, equationSystem, distributedSlackRawFactorSum, slackParticipates, distributedSlackParameters, distrRhoControllers,
                relocatedCouplers, distributedSlack, reactiveLimits, config.incrementalTransformerControl());
        // AreaInterchangeControl with areas: the device balances them in its own phase, on the scenarios whose areas keep
        // the base's participants (GpuBatchQualification.refuseAreaInterchangeControlShapes); without the tables (a
        // boundary the device does not model) every scenario goes to the CPU
        GpuAreaInterchangeControlOuterLoop.Tables areaInterchangeControl = config.areaInterchangeControl() == null ? null
                : GpuAreaInterchangeControlOuterLoop.build(network, config.areaInterchangeControl(), gpuData, elementMapper);
        double[] areaInterchangeControlBaseMoves = null;
        if (config.areaInterchangeControl() != null && network.hasArea()) {
            plan = GpuBatchQualification.refuseAreaInterchangeControlShapes(plan, contingencies, areaInterchangeControl != null);
            areaInterchangeControlBaseMoves = com.powsybl.openloadflow.ac.outerloop.gpu.GpuDistributedSlackOuterLoop.initialMoves(network, gpuData);
        }
        // AutomationSystem: the device switches the operated branches in its own phase, on the scenarios that keep every bus
        // connected whatever their systems switch (GpuBatchQualification.refuseAutomationSystemShapes); without the tables (a
        // system branch outside the closed-branch pack) every scenario goes to the CPU
        GpuAutomationSystemOuterLoop.Tables automationSystem = config.automationSystem()
                ? GpuAutomationSystemOuterLoop.build(network, elementMapper) : null;
        if (config.automationSystem()) {
            plan = GpuBatchQualification.refuseAutomationSystemShapes(plan, contingencies, network,
                    automationSystem != null ? GpuAutomationSystemOuterLoop.systemBranches(network) : null,
                    GpuAutomationSystemOuterLoop.openableBranches(network));
        }
        List<Integer> batchedIndices = plan.batchedIndices();
        if (GpuTrace.BATCH_TRACE.isOn()) {
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
        // The nums of the branches the AutomationSystem phase switched per contingency (null = none).
        int[][] automationSystemSwitchedByIndex = new int[contingencies.size()][];
        // The taps the outer loops moved per contingency (DeviceTapMoves; null = none).
        double[][] deviceTapMovesByIndex = new double[contingencies.size()][];
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
            DevicePhases.Built phases = DevicePhases.build(new DevicePhases.Request(
                    config.voltageMonitoring(),
                    config.incrementalTransformerControl(),
                    config.incrementalShuntControl(),
                    config.simpleShuntControl(),
                    config.incrementalTransformerReactivePowerControl(), config.incrementalPhaseControl(), config.incrementalTransformerVoltageControlMaxTapShift(),
                    config.continuousPhaseControl(),
                    config.transformerControl() || config.simpleTransformerControl(),
                    config.transformerVoltageControlUseInitialTapPosition(), config.genVcMinNominalV(),
                    config.secondaryVoltageControl(), config.minPlausibleTargetV(), config.maxPlausibleTargetV(), reactiveLimits),
                    network, equationSystem, targetVector, baseActivation, baseTargets,
                    gpuData.reactiveLimits().voltageRow(), gpuData.reactiveLimits().busNum(), n);
            // All-or-nothing, the same rule the base load flow applies: a loop the builders represented
            // only PARTIALLY would have the device drive a SUBSET of it, so the whole run goes to the CPU
            // (resultsByIndex stays null).
            if (DevicePhaseOwnership.batchable()) {
                // built in the order the call evaluated them when they were its arguments
                var hvdcElemIndex = GpuHvdcAcEmulationLimitsOuterLoop.elementIndices(network, nElements);
                int[] varType = config.stateVectorScalingMode() == GpuStateVectorScalingMode.MAX_VOLTAGE_CHANGE ? GpuAcActivationExtractor.extractVarType(equationSystem, n) : null;
                // every AC emulation link of the pack, whichever of its loops the run names: the slack check counts a link's
                // flow at the slack bus from these tables (hvdcSlackBalanceBatch), with or without a loop to freeze or limit
                // it, which an explicit loop list may leave out (measured: 2.6 MW distributed on a 2.4 MW link where OLF
                // distributes none); without the freeze loop nothing is frozen, and the freeze phase has nothing to release
                List<GpuHvdcAcEmulationLimitsOuterLoop.Controller> hvdcControllers =
                        GpuHvdcAcEmulationLimitsOuterLoop.buildControllers(network, equationSystem, nElements, gpuData);
                GpuBatchChunkSolver.solveQualifiedInChunks(new ChunkInputs(contingencies,
                        new ChunkInputs.BaseCase(elementMapper, gpuData, baseState, baseTargets, baseActivation.rowMode(),
                                n, nElements, varType, hvdcElemIndex),
                        plan,
                        new ChunkInputs.SolveSettings(parameters, distributedSlack, reactiveLimits, tol, reactiveLimitTol, slackDistTol,
                                convEpsPerEq, rowCriteria),
                        new ChunkInputs.DevicePhaseInputs(phases, hvdcControllers, reactiveCapabilityCurves, forcedTargetQ,
                                areaInterchangeControl, areaInterchangeControlBaseMoves, automationSystem),
                        new ChunkInputs.ResultSinks(resultsByIndex, preDistributedByIndex, distributedByIndex,
                                automationSystemSwitchedByIndex, deviceTapMovesByIndex),
                        config));
            }
        }

        // Assemble results in the original contingency order: batched scenarios carry their
        // GPU outcome; rejected ones carry the base state with GpuAcSolverStatus.REJECTED.
        List<BatchScenarioResult> results = new ArrayList<>(contingencies.size());
        for (int i = 0; i < contingencies.size(); i++) {
            String id = contingencies.get(i).contingencyId();
            double[] result = resultsByIndex[i];
            if (result != null && result.length == 3) {    // the state went to the sink: the result carries no state
                results.add(new BatchScenarioResult(id, null, (int) result[0], result[1], (int) result[2],
                        plan.islandedNumsByIndex()[i] != null ? plan.islandedNumsByIndex()[i] : new int[0],
                        preDistributedByIndex[i], distributedByIndex[i], switchedOrNone(automationSystemSwitchedByIndex[i]),
                        movesOrNone(deviceTapMovesByIndex[i])));
            } else if (result != null) {                   // the state is the result row's first n entries
                results.add(new BatchScenarioResult(id, result, (int) result[n], result[n + 1], (int) result[n + 2],
                        plan.islandedNumsByIndex()[i] != null ? plan.islandedNumsByIndex()[i] : new int[0],
                        preDistributedByIndex[i], distributedByIndex[i], switchedOrNone(automationSystemSwitchedByIndex[i]),
                        movesOrNone(deviceTapMovesByIndex[i])));
            } else {
                // Never batched: the plan refused it. Named by id for the harness, because the same
                // GpuAcSolverStatus.REJECTED also comes back for a scenario the device TRIED and threw away, and the
                // census counts cannot tell the two apart per contingency.
                GpuBatchScopeCensus.planRefused(id);
                results.add(new BatchScenarioResult(id, baseState.clone(), 0, Double.NaN, GpuAcSolverStatus.REJECTED, new int[0],
                        Double.NaN, Double.NaN, new int[0], new double[0]));
            }
        }
        return results;
    }

    private static int[] switchedOrNone(int[] switched) {
        return switched != null ? switched : new int[0];
    }

    private static double[] movesOrNone(double[] moves) {
        return moves != null ? moves : new double[0];
    }

    private GpuBatchedSecurityAnalysisSolver() {
    }
}
