/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcElementMapper;
import com.powsybl.openloadflow.ac.equations.vector.gpu.ZeroImpedanceNative;
import com.powsybl.openloadflow.ac.outerloop.gpu.AreaInterchangeControlNative;
import com.powsybl.openloadflow.ac.outerloop.gpu.AutomationSystemNative;
import com.powsybl.openloadflow.ac.outerloop.gpu.DeviceLoopTables;
import com.powsybl.openloadflow.ac.outerloop.gpu.DevicePhaseInstaller;
import com.powsybl.openloadflow.ac.outerloop.gpu.DevicePhases;
import com.powsybl.openloadflow.ac.outerloop.gpu.DistributedSlackNative;
import com.powsybl.openloadflow.ac.outerloop.gpu.ForceTargetQInReactiveLimits;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuAreaInterchangeControlOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuAutomationSystemOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuFreezingHvdcACEmulationOuterloop;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuHvdcAcEmulationLimitsOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuMonitoringVoltageOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuPhaseControlOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuSecondaryVoltageControlOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuTransformerVoltageControlOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.gpu.HvdcNative;
import com.powsybl.openloadflow.ac.outerloop.gpu.ReactiveCapabilityCurves;
import com.powsybl.openloadflow.ac.outerloop.gpu.ReactiveLimitsNative;
import com.powsybl.openloadflow.ac.outerloop.gpu.VoltageMonitoringNative;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.ac.solver.gpu.BatchReadbackNative;
import com.powsybl.openloadflow.ac.solver.gpu.BatchSolveNative;
import com.powsybl.openloadflow.ac.solver.gpu.DriverNative;
import com.powsybl.openloadflow.ac.solver.gpu.GpuBatchContextCache;
import com.powsybl.openloadflow.ac.solver.gpu.NewtonSettingsNative;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.BatchContingency;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.GenOutage;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.GpuBatchLossPreDistribution;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.GpuBatchQualification;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.GpuBatchScopeCensus;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuAcSolverStatus;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuLinearSolveMode;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuQLimitType;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuReactiveLimitsState;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuRowMode;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuStateVectorScalingMode;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuVoltageControlRole;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfShunt;
import com.powsybl.openloadflow.util.gpu.GpuTrace;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Drives the batched GPU solve of the qualified N-1 scenarios in device-memory-bounded chunks:
 * one cached native context (sized to the chunk, cuDSS analyzed once) reused across all chunks and
 * two phases (plain contingencies, then warm-continued (ctg, strategy) pairs). Assembles each
 * scenario's per-slot element mask, row-mode overrides (islanding freeze / PV->PQ flip / zero-impedance switch),
 * target/shunt/tap/distributed-slack overrides, runs the batched Newton, then the per-scenario
 * transformer-control orchestration, and writes results back by original contingency index.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuBatchChunkSolver {

    private static final boolean OUTER_LOOP_STATS = GpuTrace.OL_STATS.isOn();

    /** Test hook: where each batched contingency's outer-loop statistics go, by contingency id, or null. */
    static volatile java.util.Map<String, int[]> outerLoopStatsForTests;

    /** Test hook: how many scenarios must have ended before the solve hands their slots back for a refill, or null. */
    static volatile Integer refillAtForTests;

    /** Test hook: the installs into a bank whose other slots still held scenarios solving. */
    static final java.util.concurrent.atomic.AtomicInteger REFILLS_FOR_TESTS = new java.util.concurrent.atomic.AtomicInteger();

    /** A refill waits for this share of the bank (1 / REFILL_SHARE) to be free, or for the rest of the queue to fit. */
    private static final int REFILL_SHARE = 8;

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuBatchChunkSolver.class);

    private GpuBatchChunkSolver() {
    }

    /**
     * Solve the qualified scenarios on the GPU in a bank of slots sized by {@link GpuBatchContextCache#resolveChunkSize}
     * (auto from device memory, or a manual cap) and balanced over the run, refilled from each phase's queue as its
     * scenarios end. ONE native context, sized to the bank, is created and reused for every install: cuDSS's symbolic
     * ANALYSIS (on the shared pattern, identical for every scenario) is done once and reused, so the bank's size costs
     * nothing in amortization - it only bounds the resident per-scenario LU factors.
     */
    static void solveQualifiedInChunks(ChunkInputs inputs) {
        GpuAcData gpuData = inputs.base().gpuData();
        ChunkInputs in = inputs.withBaseState(ZeroImpedanceClusters.zeroZeroImpedanceBranchFlows(gpuData, inputs.base().baseState()));
        List<Integer> batchedIndices = in.plan().batchedIndices();
        int n = in.base().n();
        double[][] resultsByIndex = in.sinks().resultsByIndex();
        int nb = batchedIndices.size();
        // Chunk sizing is a device-memory ESTIMATE (what a slot of the run's mode takes, priced on the shared structure).
        // When it is too optimistic the factorization fails to allocate mid-solve, and a thrown batch means
        // the WHOLE run un-accelerates — so an allocation failure HALVES the chunk and retries
        // instead (measured on a 6.5k-bus RTE-like network: the probe recommended a chunk whose LU
        // factors did not fit, and 9042 contingencies all fell back to the CPU). Priced for the run's mode, the ROBUST
        // retry pass included: it reuses the run's recommendation, its slots taking less.
        int chunkSize = GpuBatchContextCache.resolveChunkSize(nb, n, gpuData,
                GpuBatchContextCache.resolveBatchMode(in.config().resolvedGpuParameters()));
        List<Integer> remaining = batchedIndices;
        while (true) {
            try {
                solveQualifiedInChunksAt(chunkSize, remaining, in);
                return;
            } catch (RuntimeException e) {
                if (chunkSize <= 1 || !isDeviceAllocFailure(e)) {
                    throw e;
                }
                // RESUME rather than redo: chunks that already produced results keep them (the retry
                // only re-solves what is still missing), so a shrink costs one chunk, not the batch.
                List<Integer> stillMissing = new ArrayList<>();
                for (Integer idx : remaining) {
                    if (resultsByIndex[idx] == null) {
                        stillMissing.add(idx);
                    }
                }
                remaining = stillMissing;
                if (remaining.isEmpty()) {
                    return;
                }
                GpuBatchContextCache.noteChunkAllocFailure(chunkSize);   // remember it for later runs
                int retryChunk = Math.max(1, Math.min(chunkSize / 2, remaining.size()));
                LOGGER.warn("GPU batched SA: device allocation failed at chunk {} ({}); retrying at chunk {}",
                        chunkSize, e.getMessage(), retryChunk);
                // the test harness's logback appenders are broken (core/classic version mismatch) —
                // a silent shrink would look like an unexplained slowdown, so say it on stderr too
                System.err.println("GPU_CHUNK_SHRINK device allocation failed at chunk " + chunkSize
                        + "; retrying at " + retryChunk);
                chunkSize = retryChunk;
            }
        }
    }

    /** cuDSS reports an out-of-device-memory as status 2 (CUDSS_STATUS_ALLOC_FAILED); a CUDA OOM
     *  surfaces as its own message. Both mean "this chunk does not fit", not "this network cannot
     *  be batched". */
    private static boolean isDeviceAllocFailure(RuntimeException e) {
        String m = e.getMessage();
        return m != null && (m.contains("cuDSS error, status 2") || m.contains("out of memory")
                || m.contains("cudaErrorMemoryAllocation"));
    }

    @SuppressWarnings({"java:S107", "java:S3776"})
    private static void solveQualifiedInChunksAt(int chunkCap, List<Integer> batchedIndices, ChunkInputs in) {
        List<BatchContingency> contingencies = in.contingencies();
        GpuAcElementMapper elementMapper = in.base().elementMapper();
        GpuAcData gpuData = in.base().gpuData();
        double[] baseState = in.base().baseState();
        double[] baseTargets = in.base().baseTargets();
        int[] baseRowMode = in.base().baseRowMode();
        int n = in.base().n();
        int nElements = in.base().nElements();
        int[] varType = in.base().varType();
        java.util.Map<com.powsybl.openloadflow.network.LfHvdc, Integer> hvdcElemIndex = in.base().hvdcElemIndex();
        GpuBatchQualification.ScenarioPlan plan = in.plan();
        int[][] frozenRowsByIndex = plan.frozenRowsByIndex();
        double[][] frozenTargetsByIndex = plan.frozenTargetsByIndex();
        GenOutage[][] genOutagesByIndex = plan.genOutagesByIndex();
        double[] activePowerLossByIndex = plan.activePowerLossByIndex();
        int[][] distrQWeightOverrideTermByIndex = plan.distrQWeightOverrideTermByIndex();
        double[][] distrQWeightOverrideWeightByIndex = plan.distrQWeightOverrideWeightByIndex();
        int[][] distrQKeyOverrideMemberByIndex = plan.distrQKeyOverrideMemberByIndex();
        double[][] distrQKeyOverrideKeyByIndex = plan.distrQKeyOverrideKeyByIndex();
        int[][] distrRhoOverrideTermByIndex = plan.distrRhoOverrideTermByIndex();
        double[][] distrRhoOverrideCoefficientByIndex = plan.distrRhoOverrideCoefficientByIndex();
        int[][] islandedDistributedSlackByIndex = plan.islandedDistributedSlackByIndex();
        int[][] islandedReactiveLimitEntriesByIndex = plan.islandedReactiveLimitEntriesByIndex();
        List<com.powsybl.openloadflow.network.LfHvdc>[] islandedHvdcsByIndex = plan.islandedHvdcsByIndex();
        int[][] distrQSubgroupByIndex = plan.distrQSubgroupByIndex();
        GpuBatchQualification.SlackRelocations slackRelocations = plan.slackRelocations();
        NewtonRaphsonParameters parameters = in.settings().parameters();
        double tol = in.settings().tol();
        double reactiveLimitTol = in.settings().reactiveLimitTol();
        double slackDistTol = in.settings().slackDistTol();
        double convEpsPerEq = in.settings().convEpsPerEq();
        double[][] rowCriteria = in.settings().rowCriteria();
        DevicePhases.Built phases = in.phases().phases();
        List<GpuHvdcAcEmulationLimitsOuterLoop.Controller> hvdcControllers = in.phases().hvdcControllers();
        ReactiveCapabilityCurves.Channel reactiveCapabilityCurves = in.phases().reactiveCapabilityCurves();
        ForceTargetQInReactiveLimits.Channel forcedTargetQ = in.phases().forcedTargetQ();
        double[][] resultsByIndex = in.sinks().resultsByIndex();
        double[] preDistributedByIndex = in.sinks().preDistributedByIndex();
        double[] distributedByIndex = in.sinks().distributedByIndex();
        int[][] automationSystemSwitchedByIndex = in.sinks().automationSystemSwitchedByIndex();
        double[][] deviceTapMovesByIndex = in.sinks().deviceTapMovesByIndex();
        double[][] distributedSlackGeneratorMovesByIndex = in.sinks().distributedSlackGeneratorMovesByIndex();
        BatchConfig config = in.config();
        int nb = batchedIndices.size();
        int nChunks = (nb + chunkCap - 1) / chunkCap;
        // BALANCE the chunks over that count instead of filling each to the cap and leaving a ragged
        // tail. The device context is allocated at the chunk size and every solve is that wide, so a
        // tail chunk holding 72 of 642 slots still solves a 642-wide batch — 570 of them padding.
        // Splitting 1998 scenarios as 4 x 500 rather than 3 x 642 + 72 measured 13.7 s vs 15.6 s of
        // batched time on rte6515, at LOWER device memory (the size only ever goes down from the cap).
        // The bank has that many slots, refilled from each phase's queue as its scenarios end (below). Balanced still: at the
        // cap (641 slots on rte6515) or narrower (320 on rte6515, 320, 240 or 160 on network.xiidm) it was no faster, a
        // narrower bank paying more per lane in the batched factorization than the slots its tail leaves idle (2026-10-29).
        int bankSize = (nb + nChunks - 1) / nChunks;
        // Reuse a cached analyzed context when the pattern/admittances/chunk/mode match (across
        // SA runs on the same network) — the context is NOT destroyed here, it stays cached.
        // MODE: the run's configured batch mode, or the ROBUST single-matrix solve on the retry pass
        // (config.robustSolveMode(), set only by BatchConfig.withRobustSolveMode) — mode 6 is mode 0's
        // per-scenario loop with matching and global column pivoting, which is what rte6515's full
        // busbar structure needs to factor at all. Switching mode EVICTS the single-entry context cache,
        // so the retry pays one cuDSS analysis; that is the price of keeping the scenarios on device.
        long handle = GpuBatchContextCache.acquireBatchContext(gpuData, bankSize,
                config.robustSolveMode() ? GpuLinearSolveMode.ROBUST
                        : GpuBatchContextCache.resolveBatchMode(config.resolvedGpuParameters()), n);
        // PER_EQUATION_TYPE criterion: install the per-row thresholds on the (possibly cached)
        // context — or CLEAR a previous run's thresholds under UNIFORM_CRITERIA.
        NewtonSettingsNative.installRowCriteria(handle, rowCriteria);
        // LINE_SEARCH: OLF's backtracking budget and step fold, on the (possibly cached) context; read only in stateVectorScalingMode 1.
        NewtonSettingsNative.setBatchLineSearch(handle, parameters.getLineSearchStateVectorScalingMaxIteration(),
                parameters.getLineSearchStateVectorScalingStepFold());
        // The run-constant outer-loop tables, shared with the base load flow's batch-of-one.
        DeviceLoopTables.install(handle, gpuData, in.settings().reactiveLimitsOuterLoop(), reactiveCapabilityCurves, forcedTargetQ,
                config.reactiveLimitsRobustMode(), config.minRealisticVoltage(), config.maxRealisticVoltage(),
                config.minNominalVoltageRealisticVoltageCheck(), config.realisticFixingLoop());

        // Which reactive-capability CURVE entries hang off each bus P row. Run-constant, and the
        // index a generator outage needs to find the entries it disturbs at its own bus.
        java.util.Map<Integer, int[]> curveEntriesByPRow = new java.util.HashMap<>();
        if (reactiveCapabilityCurves != null) {
            java.util.Map<Integer, List<Integer>> acc = new java.util.HashMap<>();
            for (int e = 0; e < reactiveCapabilityCurves.reactiveCapabilityCurveGeneratorId().length; e++) {
                acc.computeIfAbsent(reactiveCapabilityCurves.reactiveCapabilityCurveIndices()[e * 5 + 1], k -> new ArrayList<>()).add(e);
            }
            acc.forEach((row, es) -> curveEntriesByPRow.put(row, es.stream().mapToInt(Integer::intValue).toArray()));
        }

        // A control loop the kernel REFUSES must never fall back to an after-the-sweep host orchestrator.
        // Such an orchestrator runs out of OLF's order — after the loops OLF puts at orders 7 to 10 — and
        // two of them in a chain discard each other's target moves (measured 2.4e-1 kV). Un-accelerated is
        // the only honest alternative to wrong, so the whole group goes to the CPU: leaving resultsByIndex
        // null is exactly how the caller routes a contingency to the stock per-contingency path.
        DevicePhaseInstaller.InstalledPhases installed = DevicePhaseInstaller.install(handle, new DevicePhaseInstaller.Request(
                phases.incrementalTransformerVoltageControllers(), config.incrementalTransformerVoltageControlMaxTapShift(), phases.flow(), phases.continuousPhase(),
                phases.incrementalShunt(), config.incrementalShuntVoltageControlMaxSectionShift(),
                config.transformerControl() || config.simpleTransformerControl() ? phases.discreteTransformerVoltageControl() : null,
                config.simpleTransformerControl(), config.transformerControl(), phases.freeze(),
                config.simpleShuntControl() ? phases.simpleShunt() : null, false));
        if (!installed.refused().isEmpty()) {
            LOGGER.warn("GPU batched N-1 disabled for this run: the device cannot own {} — every contingency "
                    + "runs on the CPU rather than through a host orchestrator out of OLF's loop order", installed.refused());
            return;
        }
        // SecondaryVoltageControl (OLF order 4), the whole check on the device, by the same rule: a zone set the kernel
        // refuses (wider than its dense system, or an observable on an unpacked element) sends the run to the CPU. An
        // empty table clears a cached context's.
        GpuSecondaryVoltageControlOuterLoop.Controllers secondaryVoltageControlControllers = phases.secondaryVoltageControlControllers();
        boolean secondaryVoltageControlActive = secondaryVoltageControlControllers != null && !secondaryVoltageControlControllers.controllerBuses().isEmpty();
        if (!GpuSecondaryVoltageControlOuterLoop.uploadToDevice(handle, secondaryVoltageControlActive ? secondaryVoltageControlControllers : null)
                && secondaryVoltageControlActive) {
            LOGGER.warn("GPU batched N-1 disabled for this run: the device cannot own SecondaryVoltageControl — every contingency "
                    + "runs on the CPU");
            return;
        }

        RunScratch rs = new RunScratch(bankSize, nElements, gpuData, elementMapper, baseRowMode, in.settings().distributedSlackOuterLoop());
        // TWO-PHASE ordering (operator strategies, warm continuation): plain contingencies first,
        // then (ctg, strategy) PAIRS — each pair chunk is seeded with its PARENT contingency's
        // solved state (x0PerScenario) and PV/PQ states (reactiveLimitsSeedState), so the strategy
        // solve continues the contingency's outer-loop trajectory exactly like the CPU's
        // per-contingency action solve. Pair whose parent did not converge is dropped (CPU path).
        List<Integer> ctgIdxList = new ArrayList<>();
        List<Integer> pairIdxList = new ArrayList<>();
        for (int idx : batchedIndices) {
            (contingencies.get(idx).contingencyId().indexOf(BatchContingency.STRATEGY_KEY_SEP) >= 0 ? pairIdxList : ctgIdxList).add(idx);
        }
        java.util.Map<String, Integer> indexById = new java.util.HashMap<>();
        for (int i = 0; i < contingencies.size(); i++) {
            indexById.put(contingencies.get(i).contingencyId(), i);
        }
        // the contingencies whose strategy pairs this call solves: the host reads their generator targets
        java.util.Set<Integer> parentsOfPairs = new java.util.HashSet<>();
        for (int idx : pairIdxList) {
            String pairId = contingencies.get(idx).contingencyId();
            Integer parent = indexById.get(pairId.substring(0, pairId.indexOf(BatchContingency.STRATEGY_KEY_SEP)));
            if (parent != null) {
                parentsOfPairs.add(parent);
            }
        }
        int nReactiveLimitEntries = gpuData.reactiveLimits().voltageRow().length;
        ParentCaptures captures = new ParentCaptures();
        java.util.Map<Integer, int[]> reactiveLimitsStateByIndex = captures.reactiveLimitsStateByIndex;    // ctg index -> its nReactiveLimitEntries RL-state slice
        java.util.Map<Integer, double[]> reactiveLimitTargetQByIndex = captures.reactiveLimitTargetQByIndex;   // ctg index -> its final frozen target Q
        // ctg index -> its FINAL distributed-slack targets at the DS participant rows (the device DS mutates
        // targets in place; read back after each ctg chunk). The pair phase re-anchors its distribution here,
        // mirroring OLF's setGeneratorsInitialTargetPToTargetP before the strategy re-solve.
        // The run's DistributedSlackOuterLoop decides whether anything distributes, not the pack (packed either way):
        // without the loop no participant, no loss pre-distribution, the slack bus absorbs, as OLF.
        boolean distributedSlackOuterLoop = in.settings().distributedSlackOuterLoop();
        boolean reactiveLimitsOuterLoop = in.settings().reactiveLimitsOuterLoop();
        int nDistributedSlackBuses = distributedSlackOuterLoop ? gpuData.distributedSlack().busPhiRow().length : 0;
        java.util.Map<Integer, double[]> distributedSlackTargetsByIndex = captures.distributedSlackTargetsByIndex;
        // ctg index -> each participating generator's final move from its base target (the device's distributedSlackGeneratorMove, or the
        // loss pre-distribution alone when no slack pass fired): its pairs' per-generator anchors
        int packedDistributedSlackGenerators = gpuData.distributedSlack().generatorBus() == null ? 0 : gpuData.distributedSlack().generatorBus().length;
        int nDistributedSlackGenerators = distributedSlackOuterLoop ? packedDistributedSlackGenerators : 0;
        java.util.Map<Integer, double[]> genMovesByIndex = captures.genMovesByIndex;
        // ctg index -> the state its pairs start from when it ran out of Newton iterations (see the capture)
        java.util.Map<Integer, double[]> exhaustedStartByIndex = captures.exhaustedStartByIndex;
        java.util.Map<Integer, Double> slackAbsorbedByIndex = captures.slackAbsorbedByIndex;   // slack-only DS parents
        boolean slackOnlyDistribution = distributedSlackOuterLoop && gpuData.distributedSlack().busPhiRow().length == 0
                && gpuData.distributedSlack().slackPhiRow() >= 0;
        // Host-side base RL seed (mirrors initReactiveLimitQLimitTypeBatch) for PADDING slots of pair chunks: all NONE
        // (all PV) without the run's ReactiveLimitsOuterLoop, as the device seeds it.
        int[] baseReactiveLimitsState = baseReactiveLimitsState(gpuData, baseTargets, baseRowMode, baseState, reactiveLimitsOuterLoop);
        RunView run = new RunView(in, n, gpuData, baseState, baseTargets, contingencies, config, nDistributedSlackBuses,
                nDistributedSlackGenerators, nReactiveLimitEntries, slackOnlyDistribution, reactiveLimitsOuterLoop, baseReactiveLimitsState,
                indexById, captures, frozenRowsByIndex, islandedDistributedSlackByIndex, islandedHvdcsByIndex, hvdcControllers, hvdcElemIndex,
                resultsByIndex, preDistributedByIndex);
        try {
            long startTime = System.currentTimeMillis();
            // PHASE TIMING (OLF_GPU_BATCH_TRACE): the batched main solve vs the per-scenario host-side
            // outer-loop orchestrators — on a control-heavy network the orchestrators, not the solve,
            // are where the wall time goes, and that is invisible without splitting the two.
            long tMainSolveNs = 0;
            long tOrchestratorNs = 0;
            java.util.Map<String, Long> orchNs = new java.util.LinkedHashMap<>();
            // The zero-impedance elements OUT of the base tree, once: a scenario with sticky non-tree couplers puts
            // every one of them back except its still-out set, and scanning all zeroImpedanceBranchNumbers per scenario with a
            // boxed set was 1.4 s of host time on network.xiidm (2653 elements x every such scenario).
            int[] baseNonSpanningTreeBranches = ZeroImpedanceClusters.baseNonTreeElements(gpuData);
            Integer refillAtTest = refillAtForTests;
            int refillAt = refillAtTest != null ? refillAtTest : Math.max(1, bankSize / REFILL_SHARE);
            for (List<Integer> group : List.of(ctgIdxList, pairIdxList)) {
                boolean pairPhase = group == pairIdxList;
                int nbg = group.size();
                // The phase's scenarios queue for the bank's slots: the first ones fill it, and as scenarios end the solve
                // hands their slots back, refilled with the next ones while the others go on solving - one tail per phase
                // instead of one per chunk. Each install is a chunk: the scenarios that go into the free slots.
                SlotOccupant[] bank = new SlotOccupant[bankSize];
                java.util.TreeSet<Integer> freeSlots = new java.util.TreeSet<>();
                for (int b = 0; b < bankSize; b++) {
                    freeSlots.add(b);
                }
                int lastInstall = 0;
                for (int chunkStart = 0; chunkStart < nbg; chunkStart += lastInstall) {
                    int chunkSize = Math.min(freeSlots.size(), nbg - chunkStart);
                    int realCount = chunkSize;
                    lastInstall = chunkSize;
                    if (freeSlots.size() < bankSize) {
                        REFILLS_FOR_TESTS.incrementAndGet();
                    }
                    int[] installSlots = new int[chunkSize];
                    for (int j = 0; j < chunkSize; j++) {
                        installSlots[j] = freeSlots.pollFirst();
                    }
                    // before anything per scenario: the bank slots the chunk's scenarios go into (the whole bank, in order, at a
                    // phase's start)
                    BatchSolveNative.beginBatchInstall(handle, chunkSize == bankSize ? null : installSlots);
                    rs.overrides.clear();
                    // fixTransformerVoltageControls is a POST-CONTINGENCY check on the CPU, so its
                    // connectivity masks are per scenario and belong to the chunk, not to the run.
                    if (installed.discreteTransformerVoltageControl() && config.transformerControl() && phases.freeze() != null && phases.freeze().hasFix()) {
                        List<List<LfBranch>> slotOutaged = new ArrayList<>(realCount);
                        for (int s = 0; s < realCount; s++) {
                            BatchContingency ctg = contingencies.get(group.get(chunkStart + s));
                            List<LfBranch> outaged = new ArrayList<>(ctg.outagedBranches());
                            outaged.addAll(ctg.outagedZeroImpedanceBranches());
                            slotOutaged.add(outaged);
                        }
                        GpuTransformerVoltageControlOuterLoop.uploadFixMasks(handle, phases.freeze(), slotOutaged, chunkSize);
                    }
                    ChunkLists cl = new ChunkLists(chunkSize, n, gpuData, forcedTargetQ, nDistributedSlackGenerators);
                    SlotSeeding seeding = new SlotSeeding(chunkSize, pairPhase, n, gpuData, elementMapper, contingencies,
                            plan, baseState, baseTargets, baseRowMode, hvdcElemIndex, baseNonSpanningTreeBranches, nDistributedSlackBuses, reactiveCapabilityCurves,
                            curveEntriesByPRow, rs, cl);
                    rs.resetChunk();
                    for (int s = 0; s < chunkSize; s++) {
                        rs.resetSlot(s);
                        if (s < realCount) {                 // real scenario: disable its outaged branch(es)
                            int ci = group.get(chunkStart + s);
                            PlanOverrideSeed.seed(seeding, s, ci);
                            List<LfBranch> branches = contingencies.get(ci).outagedBranches();
                            List<LfShunt> shunts = contingencies.get(ci).outagedShunts();
                            List<com.powsybl.openloadflow.network.LfHvdc> hvdcs = contingencies.get(ci).outagedHvdcs();
                            if (islandedHvdcsByIndex != null && islandedHvdcsByIndex[ci] != null) {
                                // An AC-emulation HVDC whose endpoint this contingency islanded: OLF
                                // zeroes both stations' targetP, which in AC emulation they already
                                // are, so masking the element is the whole of it.
                                hvdcs = new ArrayList<>(hvdcs);
                                hvdcs.addAll(islandedHvdcsByIndex[ci]);
                            }
                            List<BatchContingency.SideOpen> sideOpens = contingencies.get(ci).sideOpenBranches();
                            OutageMaskSeed.seed(seeding, s, ci, branches, shunts, hvdcs, sideOpens);
                            // Rows that host a PILOT'S relocated reactive balance this scenario (the carrier a frozen base
                            // carrier handed it to): a DISTR_Q target seed must not land on them, see GenOutageSeed.
                            gnu.trove.set.hash.TIntHashSet pilotBalanceRows = new gnu.trove.set.hash.TIntHashSet();
                            IslandSeed.seed(seeding, s, ci, branches, sideOpens, pilotBalanceRows);
                            SlackRelocationSeed.seed(seeding, s, ci);
                            // This scenario's plan entries, with the rows of every bus whose balances the structure
                            // moves HOME here (a relocated child whose coupler opens or leaves the tree) rewritten:
                            // the plan wrote them at the base's relocated rows, which turn into identities below.
                            GenOutage[] slotGos = genOutagesByIndex[ci];
                            slotGos = CouplerSeed.seed(seeding, s, ci, slotGos);
                            ActionSeed.seed(seeding, s, ci);
                            GenOutageSeed.seed(seeding, s, slotGos, pilotBalanceRows);
                        }
                        // else: padding scenario = base case (no element disabled), discarded below
                    }
                    ChunkView chunk = new ChunkView(group, chunkStart, chunkSize, realCount, pairPhase);
                    ChunkSeeds seeds = new ChunkSeeds(chunkSize);
                    if (!pairPhase && (rs.hasDistributedSlack || slackOnlyDistribution) && config.contingencyLossPreDistribution()) {
                        preDistributeContingencyLosses(run, chunk, cl, rs, seeds);
                    }
                    // CONTINUOUS phase control: OLF's initialize() enables the CONTROLLER-mode controls
                    // BEFORE the first solve. This is the loop's initialize(), not an orchestrator — the
                    // in-kernel phase expects the activation to already carry it.
                    if (phases.continuousPhase() != null && !phases.continuousPhase().isEmpty()) {
                        GpuPhaseControlOuterLoop.applyPhaseOn(realCount, n, phases.continuousPhase(),
                                rs.rowModePerScenario, baseRowMode, rs.disabledMask, rs.overrides);
                    }
                    int[] targetOverrideIndex = rs.overrides.indexArray();
                    double[] targetOverrideValue = rs.overrides.valueArray();
                    if (rs.overrides.conflictingAddCount() > 0) {
                        // Two producers disagreed on one row's target. GpuBatchOverrides resolved it to the
                        // later one (no row reaches the device twice), but a sound plan has NO such row:
                        // every one so far was a write that should not exist (a DISTR_Q seed on a mode-0/4
                        // row). Name it.
                        LOGGER.warn("GPU batched SA: {} conflicting target overrides in chunk at {} (resolved to the later"
                                + " writer)", rs.overrides.conflictingAddCount(), chunkStart);
                        GpuBatchScopeCensus.record("HOST:OVERRIDE_CONFLICT");
                    }
                    // the warm starts: frozen rows, and each operator strategy's parent contingency state
                    seedWarmStarts(run, chunk, cl, rs, seeds);
                    if (pairPhase) {
                        targetOverrideIndex = rs.overrides.indexArray();
                        targetOverrideValue = rs.overrides.valueArray();
                    }
                    double[] distributedSlackBusAnchorShift = seeds.distributedSlackBusAnchorShift;
                    double[] distributedSlackGeneratorAnchor = seeds.distributedSlackGeneratorAnchor;
                    double[] distributedSlackGeneratorSeedAnchor = seeds.distributedSlackGeneratorSeedAnchor;
                    double[] slackBusAnchorShift = seeds.slackBusAnchorShift;
                    double[] pairSlackAbsorbedSeed = seeds.pairSlackAbsorbedSeed;
                    boolean[] rejectAfterSolve = seeds.rejectAfterSolve;
                    boolean[] pairParentConverged = seeds.pairParentConverged;
                    double[] reactiveLimitSeedTargetQ = seeds.reactiveLimitSeedTargetQ;
                    double[] preSlackSeed = seeds.preSlackSeed;
                    double[] x0Full = seeds.x0Full;
                    int[] reactiveLimitsSeedState = seeds.reactiveLimitsSeedState;
                    int[] x0SeedIdx = seeds.x0SeedIdx;
                    double[] x0SeedVal = seeds.x0SeedVal;
                    // HVDC AC-emulation: snapshot the pre-freeze override lists (the round rebuild base),
                    // then OLF's FreezingHvdcACEmulation freezes every live hvdc at its scenario's
                    // WARM-START flows BEFORE the first solve.
                    GpuBatchOverrides.Snapshot preHvdcOverrides = null;
                    int[][] hvdcStatus = null;
                    double[] hvdcAppliedDelta1 = null;
                    double[] hvdcAppliedDelta2 = null;
                    // HvdcAction (AC emulation off): OLF sets each station's target to the link's flow at the state the
                    // strategy starts from - the parent's converged solution, which is this pair's warm start - unclamped,
                    // and takes the link out of its equations. On the device: the element masked, the two set points as
                    // converter-bus target overrides, the status left LINEAR so the HVDC loops treat the link as the
                    // scenario's own (never frozen, unfrozen or saturated by them). A station on the slack bus has no row:
                    // its set point reaches the slack mismatch through the applied delta (hvdcSlackBalanceBatch), which
                    // the VSC curve channel reads too - set below, once the HVDC arrays exist.
                    List<double[]> hvdcActionApplied = new ArrayList<>();   // {slot, ctl, d1 - baseD1, d2 - baseD2}
                    rejectAfterSolve = applyHvdcActions(run, chunk, rs, x0Full, pairParentConverged, rejectAfterSolve, hvdcActionApplied);
                    if (hvdcControllers != null && !hvdcControllers.isEmpty()) {
                        hvdcAppliedDelta1 = new double[chunkSize * hvdcControllers.size()];
                        hvdcAppliedDelta2 = new double[chunkSize * hvdcControllers.size()];
                        preHvdcOverrides = rs.overrides.snapshot();
                        if (GpuTrace.HVDC_TRACE.isOn()) {
                            for (int s = 0; s < realCount; s++) {
                                System.err.println("HVDC_SLOT s=" + s + " ctg="
                                        + contingencies.get(group.get(chunkStart + s)).contingencyId());
                                // the PRE-freeze override entries on this slot's converter-bus P rows: the freeze
                                // composes on top of them, so a stale one here is a double count downstream
                                for (GpuHvdcAcEmulationLimitsOuterLoop.Controller c : hvdcControllers) {
                                    for (int k = 0; k < preHvdcOverrides.index().length; k++) {
                                        int r = preHvdcOverrides.index()[k] - s * n;
                                        if (r == c.bus1PRow() || r == c.bus2PRow()) {
                                            System.err.printf("HVDC_PREOV s=%d row=%d value=%.9f base=%.9f%n", s, r,
                                                    preHvdcOverrides.value()[k], baseTargets[r]);
                                        }
                                    }
                                }
                            }
                        }
                        if (config.hvdcFreezing()) {
                            hvdcStatus = GpuFreezingHvdcACEmulationOuterloop.applyFreeze(chunkSize, realCount, n, hvdcControllers,
                                    baseState, x0Full, x0Full == null ? x0SeedIdx : null, x0Full == null ? x0SeedVal : null,
                                    rs.disabledMask, baseTargets, rs.overrides, hvdcAppliedDelta1, hvdcAppliedDelta2, rs.rowModePerScenario);
                            targetOverrideIndex = rs.overrides.indexArray();
                            targetOverrideValue = rs.overrides.valueArray();
                        } else {
                            hvdcStatus = new int[chunkSize][hvdcControllers.size()];
                        }
                        // A NON-LINEAR-at-base hvdc (saturated/frozen base — no longer a whole-run
                        // gate): entries the freeze did not claim keep the base status + hv bit
                        // (no override, baseTargets already carry the base injections); padding
                        // slots get the base bits so they converge at iteration 0.
                        GpuHvdcAcEmulationLimitsOuterLoop.seedBaseStatuses(chunkSize, realCount, hvdcControllers,
                                hvdcStatus, rs.disabledMask);
                    }
                    boolean monitoringVoltageActive = phases.monitoring() != null && !phases.monitoring().isEmpty();
                    boolean readReactiveLimitsFinalState = (!pairPhase && !pairIdxList.isEmpty() || monitoringVoltageActive
                            || GpuTrace.RL_TRACE.isOn()) && nReactiveLimitEntries > 0;
                    cl.upload(handle, gpuData);
                    // The two HVDC AC-emulation loops run as DRIVER PHASES, in OLF's own position
                    // between DistributedSlack and ReactiveLimits. Running them instead as host
                    // orchestrators after the whole solve put them out of OLF's order and left one
                    // differing violation line over 2000 rte6515 branch contingencies.
                    for (double[] e : hvdcActionApplied) {           // the HvdcAction set points, as applied deltas
                        int k = (int) e[0] * hvdcControllers.size() + (int) e[1];
                        hvdcAppliedDelta1[k] = e[2];
                        hvdcAppliedDelta2[k] = e[3];
                    }
                    boolean hasHvdcDriverLoops = hvdcControllers != null && !hvdcControllers.isEmpty();
                    GpuHvdcAcEmulationLimitsOuterLoop.installDriverLoops(handle,
                            hasHvdcDriverLoops ? hvdcControllers : null, chunkSize, hvdcStatus, rs.disabledMask,
                            hvdcAppliedDelta1, hvdcAppliedDelta2, config.hvdcLimits());
                    // MonitoringVoltage as a DRIVER PHASE, in OLF's own slot (order 6, between the
                    // HVDC loops and ReactiveLimits) instead of as a host orchestrator after the whole
                    // solve. Run-constant table, but the context is per chunk.
                    boolean monitoringVoltageInKernel = monitoringVoltageActive;
                    GpuMonitoringVoltageOuterLoop.installDriverLoop(handle, monitoringVoltageInKernel ? phases.monitoring() : null);
                    // the run's own loops
                    DriverNative.setBatchOuterLoops(handle, in.settings().distributedSlackOuterLoop(), in.settings().reactiveLimitsOuterLoop());
                    // AreaInterchangeControl (in DistributedSlack's place), or cleared on a cached context
                    GpuAreaInterchangeControlOuterLoop.uploadToDevice(handle, in.phases().areaInterchangeControl());
                    // AutomationSystem (the last loop), or cleared on a cached context
                    GpuAutomationSystemOuterLoop.uploadToDevice(handle, in.phases().automationSystem());
                    long solveCountBefore = BatchReadbackNative.readBatchSolveCount(handle);
                    // Seeds only where no full per-scenario state goes (the pair phase builds its own from the parents).
                    NewtonSettingsNative.setBatchX0Seeds(handle, x0Full == null ? x0SeedIdx : null, x0Full == null ? x0SeedVal : null);
                    long tSolve0 = System.nanoTime();
                    // OLF_GPU_LOOP_TRACE_CTG=<id>: trace that contingency's slot in THIS chunk (a fixed slot number
                    // follows the chunk composition, which moves with the resolved chunk size).
                    // OLF's tree for the scenarios of this chunk, where it has moved from the one the
                    // structure was extracted on (empty clears the previous chunk's list).
                    ZeroImpedanceNative.setBatchSpanningTreeOverrides(handle,
                            rs.spanningTreeOverrideScenarioElementValueList.stream().mapToInt(Integer::intValue).toArray());
                    String traceCtg = GpuTrace.LOOP_TRACE_CTG.value();
                    int traceSlot = -1;
                    if (traceCtg != null) {
                        for (int s = 0; s < realCount; s++) {
                            if (traceCtg.equals(contingencies.get(group.get(chunkStart + s)).contingencyId())) {
                                traceSlot = s;
                            }
                        }
                        if (traceSlot >= 0) {
                            System.err.println("LOOP_TRACE_CTG ctg=" + traceCtg + " slot=" + traceSlot);
                        }
                    }
                    if (traceSlot >= 0 || chunkStart == 0) {
                        NewtonSettingsNative.setBatchLoopTraceSlot(handle, traceSlot);
                    }
                    // OLF_GPU_CHUNK_DUMP=<file>: every chunk's members in slot order, appended one line per chunk
                    // ("<pairPhase> <chunkStart> <chunkSize> <id> <id> ..."). A scenario whose answer moves with
                    // its chunk mates can only be reproduced in the chunk it was solved in.
                    String chunkDump = GpuTrace.CHUNK_DUMP.value();
                    if (chunkDump != null) {
                        StringBuilder line = new StringBuilder().append(pairPhase).append(' ').append(chunkStart)
                                .append(' ').append(chunkSize);
                        for (int s = 0; s < realCount; s++) {
                            line.append(' ').append(contingencies.get(group.get(chunkStart + s)).contingencyId());
                        }
                        try {
                            java.nio.file.Files.writeString(java.nio.file.Path.of(chunkDump), line.append('\n'),
                                    java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND);
                        } catch (java.io.IOException e) {
                            throw new java.io.UncheckedIOException(e);
                        }
                    }
                    if (forcedTargetQ != null) {                 // where each slot's pre-distribution put the clamping gens
                        // a pair starts each generator at its anchor, a contingency at its pre-distributed target
                        ReactiveLimitsNative.setBatchForcedTargetQPre(handle, forcedTargetQ.preMoves(
                                pairPhase && distributedSlackGeneratorAnchor != null ? ChunkArrays.perSlot(distributedSlackGeneratorAnchor,
                                        chunkSize, nDistributedSlackGenerators) : cl.genPreMovesBySlot));
                    }
                    if (cl.outageSlackShift != null && rs.hasDistributedSlack) {     // on top of whatever the pre-distribution or the pair's
                        if (slackBusAnchorShift == null) {               // parent put there: both are slack-target moves
                            slackBusAnchorShift = new double[chunkSize];
                        }
                        for (int s = 0; s < chunkSize; s++) {
                            slackBusAnchorShift[s] += cl.outageSlackShift[s];
                        }
                    }
                    if (slackOnlyDistribution) {
                        DistributedSlackNative.setBatchSlackOnlyAnchor(handle, pairPhase ? pairSlackAbsorbedSeed : null);
                    }
                    if (nReactiveLimitEntries > 0) {
                        ReactiveLimitsNative.setBatchReactiveLimitSeedTargetQ(handle, pairPhase ? reactiveLimitSeedTargetQ : null);
                    }
                    if (packedDistributedSlackGenerators > 0) {         // set, or cleared on a cached context, whatever the run's loops
                        double[] start = pairPhase || nDistributedSlackGenerators == 0 ? null : ChunkArrays.flat(cl.genPreMovesBySlot, chunkSize, nDistributedSlackGenerators);
                        if (start != null && in.phases().areaInterchangeControlWithAreas()) {
                            // the area phase keeps the moves from the INITIAL targets: every scenario starts at the base's
                            for (int s = 0; s < chunkSize; s++) {
                                for (int g = 0; g < nDistributedSlackGenerators; g++) {
                                    start[s * nDistributedSlackGenerators + g] += in.phases().areaInterchangeControlBaseMoves()[g];
                                }
                            }
                        }
                        DistributedSlackNative.setBatchDistributedSlackGeneratorAnchors(handle, pairPhase ? distributedSlackGeneratorAnchor : null,
                                start, pairPhase ? distributedSlackGeneratorSeedAnchor : null);
                    }
                    // What the harvest of each slot needs from the scenario installed into it (null: a padding slot)
                    SlotOccupant[] occupants = new SlotOccupant[chunkSize];
                    for (int s = 0; s < realCount; s++) {
                        occupants[s] = new SlotOccupant(group.get(chunkStart + s), cl.preLossBySlot[s], cl.preAbsorbedBySlot[s],
                                cl.distributedSlackBusRowOverrideByScenario[s],
                                rs.distributedSlackOverrideBus.length > 0 ? rs.distributedSlackOverrideBus[s] : -1,
                                rs.distributedSlackOverrideBaseTargetPDelta.length > 0 ? rs.distributedSlackOverrideBaseTargetPDelta[s] : 0.0,
                                rs.distributedSlackOverrideFactorDelta.length > 0 ? rs.distributedSlackOverrideFactorDelta[s] : 0.0,
                                cl.genPreMovesBySlot != null ? cl.genPreMovesBySlot[s] : null,
                                pairPhase && distributedSlackBusAnchorShift != null
                                        ? java.util.Arrays.copyOfRange(distributedSlackBusAnchorShift, s * nDistributedSlackBuses, (s + 1) * nDistributedSlackBuses) : null,
                                pairPhase && pairSlackAbsorbedSeed != null ? pairSlackAbsorbedSeed[s] : 0.0,
                                !pairPhase && preSlackSeed != null ? preSlackSeed[s] : 0.0,
                                pairPhase && (distributedSlackBusAnchorShift != null || pairSlackAbsorbedSeed != null),
                                pairPhase && distributedSlackGeneratorAnchor != null,
                                rejectAfterSolve != null && rejectAfterSolve[s]);
                        bank[installSlots[s]] = occupants[s];
                    }
                    // more scenarios queue: the solve returns once refillAt of them can go in, the others still solving
                    int queued = nbg - chunkStart - chunkSize;
                    int pauseAtEnded = queued > 0 ? Math.min(queued, refillAt) : 0;
                    double[][] chunkResults = BatchSolveNative.solveBatchContext(handle, baseState, baseTargets,
                            java.util.Arrays.copyOf(rs.rowModePerScenario, chunkSize), baseRowMode,
                            java.util.Arrays.copyOf(rs.disabledMask, chunkSize), targetOverrideIndex, targetOverrideValue,
                            rs.distributedSlackOverrideBus, rs.distributedSlackOverrideFactorDelta,
                            rs.distributedSlackOverrideBaseTargetPDelta, rs.distributedSlackOverrideMaxDelta,
                            rs.distributedSlackOverrideMinDelta, distributedSlackBusAnchorShift, slackBusAnchorShift,
                            ChunkArrays.intsOrNull(cl.reactiveLimitTargetVOverrideScenarioEntryList),
                            cl.reactiveLimitTargetVOverrideScenarioEntryList.isEmpty() ? null : cl.reactiveLimitTargetVOverrideValue.stream().mapToDouble(Double::doubleValue).toArray(),
                            cl.reactiveLimitDeltaOverrideScenarioEntryList.stream().mapToInt(Integer::intValue).toArray(),
                            cl.reactiveLimitDeltaOverrideValueList.stream().mapToDouble(Double::doubleValue).toArray(),
                            cl.reactiveCapabilityCurveOverrideScenarioCurve.stream().mapToInt(Integer::intValue).toArray(),
                            cl.reactiveCapabilityCurveOverrideStepScale.stream().mapToDouble(Double::doubleValue).toArray(),
                                                        parameters.getMaxIterations(), tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                            config.stateVectorScalingMode(), config.maxDv(), config.maxDphi(),
                            config.stateVectorScalingMode() == GpuStateVectorScalingMode.MAX_VOLTAGE_CHANGE ? varType : null, x0Full,
                            rs.shuntAdmittanceOverrideScenarioElementList.stream().mapToInt(Integer::intValue).toArray(),
                            rs.shuntAdmittanceOverrideDeltaList.stream().mapToDouble(Double::doubleValue).toArray(),
                            reactiveLimitsSeedState,
                            rs.closedBranchParameterOverrideScenarioElementList.stream().mapToInt(Integer::intValue).toArray(),
                            rs.closedBranchParameterOverrideValueList.stream().mapToDouble(Double::doubleValue).toArray(),
                            config.maxOuterIterations(), config.maxPqPvSwitch(), config.rejectSlackDistributionFailureFlag(),
                            // monitoring active: OLF checks VoltageMonitoring BEFORE ReactiveLimits — the main
                            // solve surfaces the DS-stable / pre-RL state; the orchestrator then interleaves
                            // and finishes with the full RL-enabled solves.
                            0,
                            rs.nBaseClosedZeroImpedanceBranches, rs.zeroImpedanceBranchCloseScenarioElementList.stream().mapToInt(Integer::intValue).toArray(),
                            rs.zeroImpedanceBranchOpenScenarioElementList.stream().mapToInt(Integer::intValue).toArray(),
                            pauseAtEnded);
                    // the scenarios the solve handed back, in the order of its rows: their occupants, whose slots go back to the bank
                    int[] handedBack = BatchSolveNative.readBatchHandedBackSlots(handle);
                    SlotOccupant[] harvested = new SlotOccupant[handedBack.length];
                    for (int j = 0; j < handedBack.length; j++) {
                        harvested[j] = bank[handedBack[j]];
                        bank[handedBack[j]] = null;
                        freeSlots.add(handedBack[j]);
                    }
                    int harvestedCount = handedBack.length;
                    // each scenario's PV/PQ states as its solve ended: what its pairs continue from, and the traces
                    int[] reactiveLimitsFinalState = readReactiveLimitsFinalState ? ReactiveLimitsNative.readBatchReactiveLimitsFinalState(handle) : null;
                    if (GpuTrace.REJECT_TRACE.isOn()) {
                        // slot -> contingency for every scenario the device rejected in this chunk: the native
                        // GPU_REJECT lines name slots, and a slot only means something with the chunk's composition.
                        for (int s = 0; s < harvestedCount; s++) {
                            int status = chunkResults[s].length > n + 2 ? (int) chunkResults[s][n + 2] : 0;
                            if (status <= GpuAcSolverStatus.REJECTED) {
                                System.err.println("CHUNK_REJECT slot=" + s + " ctg=" + contingencies.get(harvested[s].index()).contingencyId()
                                        + " status=" + status);
                            }
                        }
                    }
                    tMainSolveNs += System.nanoTime() - tSolve0;
                    recordAfterSolve(run, chunk, handle, chunkResults, harvested, harvestedCount, reactiveLimitsFinalState,
                            hasHvdcDriverLoops, monitoringVoltageInKernel);
                    long tOrch0 = System.nanoTime();
                    captureForStrategies(run, chunk, handle, chunkResults, harvested, harvestedCount, reactiveLimitsFinalState, pairIdxList);
                    if (!chunk.pairPhase()) {
                        // ...and the generator moves of a contingency with strategy pairs for the host too: a strategy
                        // condition on a generator's active power reads the target OLF's post-contingency load flow leaves
                        for (int s = 0; s < harvestedCount; s++) {
                            double[] moves = run.captures().genMovesByIndex.get(harvested[s].index());
                            if (moves != null && parentsOfPairs.contains(harvested[s].index())) {
                                distributedSlackGeneratorMovesByIndex[harvested[s].index()] = moves;
                            }
                        }
                    }
                    // HOST ORCHESTRATORS — in OLF's DefaultAcOuterLoopConfig order.
                    // In-kernel: DS(0), HVDC(1,2), Monitoring(5), RL(6) already ran in the main solve.
                    // Post-RL host orchestrators below: SVC(4), Monitoring(5), Phase(7), TVC(8), TRPC(9), Shunt(10).

                    tOrchestratorNs += System.nanoTime() - tOrch0;
                    if (GpuTrace.RL_TRACE.isOn() && reactiveLimitsFinalState != null) {
                        // The FINAL reactive-limits state, after every orchestrator re-solve. The
                        // per-solve trace above only sees the main solve, and OLF switches PV -> PQ late in its
                        // outer-loop sequence (DS, HVDC freeze, monitoring, THEN ReactiveLimits), so
                        // comparing the main solve's count against the CPU's compares two different
                        // points in the trajectory.
                        for (int s = 0; s < harvestedCount; s++) {
                            int switchedToPq = 0;
                            for (int i = 0; i < nReactiveLimitEntries; i++) {
                                if (GpuReactiveLimitsState.limitType(reactiveLimitsFinalState[s * nReactiveLimitEntries + i]) != GpuQLimitType.NONE) {
                                    switchedToPq++;
                                }
                            }
                            StringBuilder ids = new StringBuilder();
                            for (int i = 0; i < nReactiveLimitEntries; i++) {
                                if (GpuReactiveLimitsState.limitType(reactiveLimitsFinalState[s * nReactiveLimitEntries + i]) != GpuQLimitType.NONE) {
                                    ids.append(' ').append(i).append("(vRow=").append(gpuData.reactiveLimits().voltageRow()[i])
                                       .append(",lt=").append(GpuReactiveLimitsState.limitType(reactiveLimitsFinalState[s * nReactiveLimitEntries + i])).append(')');
                                }
                            }
                            System.err.printf("GPU_RL_FINAL ctg=%s switchedToPqTotal=%d%s%n",
                                    contingencies.get(harvested[s].index()).contingencyId(), switchedToPq, ids);
                        }
                    }
                    boolean[] switchesUnread = new boolean[harvestedCount];
                    if (in.phases().automationSystem() != null) {
                        // the branches each scenario's AutomationSystem phase left switched: in the mask the solve ended with,
                        // which a second solve of this chunk would have re-uploaded from the host's (none runs: then reject)
                        byte[] operatedDisabled = BatchReadbackNative.readBatchSolveCount(handle) - solveCountBefore == 1
                                ? AutomationSystemNative.readBatchAutomationSystemOperatedDisabled(handle) : null;
                        for (int s = 0; s < harvestedCount; s++) {
                            if (operatedDisabled == null) {
                                switchesUnread[s] = true;
                                GpuBatchScopeCensus.record("RUNTIME:AUTOMATION_SYSTEM_SWITCHES_UNREAD");
                            } else {
                                int[] switched = GpuAutomationSystemOuterLoop.switchedBranchNums(in.phases().automationSystem(), operatedDisabled, s);
                                automationSystemSwitchedByIndex[harvested[s].index()] = switched.length > 0 ? switched : null;
                            }
                        }
                    }
                    // the taps the outer loops left on each scenario's transformers: the host evaluates its flows with them
                    double[][] tapMoves = DeviceTapMoves.read(in.phases().phases(), handle, harvestedCount);
                    if (tapMoves != null) {
                        for (int s = 0; s < harvestedCount; s++) {
                            deviceTapMovesByIndex[harvested[s].index()] = tapMoves[s];
                        }
                    }
                    if (in.phases().areaInterchangeControlWithAreas() && BatchReadbackNative.readBatchSolveCount(handle) - solveCountBefore == 1) {
                        // what the area phase's checks distributed (its scenarios lose no active power: nothing pre-distributed)
                        double[] distributed = AreaInterchangeControlNative.readBatchAreaInterchangeControlDistributedActivePower(handle);
                        for (int s = 0; s < harvestedCount && distributed != null; s++) {
                            distributedByIndex[harvested[s].index()] = distributed[s];
                        }
                    } else if ((rs.hasDistributedSlack || slackOnlyDistribution)
                            && BatchReadbackNative.readBatchSolveCount(handle) - solveCountBefore == 1) {
                        recordDistributedActivePower(handle, gpuData, harvested, harvestedCount, n, slackOnlyDistribution,
                                preDistributedByIndex, distributedByIndex);
                    }
                    for (int s = 0; s < harvestedCount; s++) {
                        double[] row = chunkResults[s];
                        if (harvested[s].rejectAfterSolve() || switchesUnread[s]) {
                            row[n + 2] = GpuAcSolverStatus.REJECTED;   // the census names the reason where it is decided
                        } else if (pairPhase && (int) row[n + 2] != GpuAcSolverStatus.CONVERGED) {
                            // a pair the device solved and did not converge: a device failure wherever OLF converges it
                            GpuBatchScopeCensus.pairDeviceFailed(
                                    contingencies.get(harvested[s].index()).contingencyId());
                        }
                        if (config.stateSink() != null && (int) row[n + 2] == GpuAcSolverStatus.CONVERGED) {
                            // the state leaves the heap now, as its chunk completes; the result keeps the tail
                            // (iterations, norm, status) - see sunkTail / fullRow
                            config.stateSink().put(contingencies.get(harvested[s].index()).contingencyId(),
                                    java.util.Arrays.copyOf(row, n));
                            row = new double[] {row[n], row[n + 1], row[n + 2]};
                        }
                        resultsByIndex[harvested[s].index()] = row;
                        // OLF_GPU_CPU_ROUTED=1: name every scenario the device hands back to the CPU, with
                        // the status that decided it (-2 rejected, -1 diverged, 1 max-iter). The aggregate
                        // "precomputed on device: N / M" says how many, never which.
                        if (GpuTrace.CPU_ROUTED.isOn() && chunkResults[s] != null
                                && (int) chunkResults[s][n + 2] != GpuAcSolverStatus.CONVERGED) {
                            System.err.printf("GPU_CPU_ROUTED ctg=%s status=%d%n",
                                    contingencies.get(harvested[s].index()).contingencyId(),
                                    (int) chunkResults[s][n + 2]);
                        }
                    }
                }
            }
            long elapsed = System.currentTimeMillis() - startTime;
            LOGGER.info("Batched N-1: {} GPU scenarios in a bank of {} slots solved in {} ms ({} ms/scenario)",
                    nb, bankSize, elapsed, (double) elapsed / nb);
            if (GpuTrace.PHASE_TIME.isOn()) {
                System.err.printf("BATCH_PHASE_TIME total=%d ms | batched main solves=%d ms | "
                        + "outer-loop orchestrators=%d ms | %d scenarios in a bank of %d slots%n",
                        elapsed, tMainSolveNs / 1_000_000, tOrchestratorNs / 1_000_000, nb, bankSize);
                orchNs.forEach((k, v) -> System.err.printf("BATCH_PHASE_TIME    %-38s %6d ms%n", k, v / 1_000_000));
                GpuBatchPhaseStats.snapshot().forEach((k, v) ->
                        System.err.printf("BATCH_PHASE_TIME    %-38s %6d x%n", k, v));
                GpuBatchPhaseStats.snapshotMs().forEach((k, v) ->
                        System.err.printf("BATCH_PHASE_TIME    %-38s %6d ms%n", k, v));
            }
        } catch (RuntimeException e) {
            GpuBatchContextCache.clearContextCache();   // a failed solve may leave the context inconsistent — rebuild next time
            throw e;
        }
    }

    /**
     * The active power the device distributed for each scenario of a chunk — what OLF's security analysis
     * reports as {@code preDistributedActivePower + loadFlowResult.getDistributedActivePower()}: the
     * contingency's loss pre-distribution plus every distributed-slack pass of the post-contingency load flow.
     *
     * <p>Each device pass re-derives the WHOLE distribution from the anchor (OLF's resetToInitialState), so
     * after the last pass that fired the participants carry it in their per-generator cumulative shares and
     * the slack its retained share: Σ distributedSlackGeneratorMove + retain — the same two read-backs the base load flow writes
     * back as targets ({@code GpuDistributedSlackOuterLoop.applyToNetwork}). When no pass fired the
     * distribution is the pre-distribution alone. Slack-only participation keeps everything in the slack's
     * absorbed total. Only valid when ONE solve covered the chunk (the read-backs are per solve): the caller
     * checks; a scenario left NaN reports the pre-distribution as before.
     *
     * <p>A strategy pair reports what OLF's action simulation reports: the distribution of the strategy
     * load flow alone, which starts from the parent contingency's distributed targets and has no loss
     * pre-distribution. The pair phase re-anchors the water-fill and the slack target at the parent
     * (its occupant's {@code distributedSlackBusAnchorShift}) and seeds the slack-only accumulator with the parent's
     * total ({@code pairSlackAbsorbedSeed}); both are taken back out here.
     *
     * <p>{@code preSlackSeed}: a contingency that leaves only the slack bus's generators participating has
     * its loss pre-distribution on them, carried as a shift of the slack target the device's retain is
     * measured from; it is added back.
     *
     * <p>{@code occupants}: what was installed into each of the solve's first {@code realCount} rows.
     */
    private static void recordDistributedActivePower(long handle, GpuAcData gpuData, SlotOccupant[] occupants, int realCount, int n,
                                                     boolean slackOnlyDistribution,
                                                     double[] preDistributedByIndex, double[] distributedByIndex) {
        if (slackOnlyDistribution) {
            double[] absorbed = BatchReadbackNative.readBatchSlackAbsorbed(handle);
            for (int s = 0; s < realCount && absorbed != null; s++) {
                distributedByIndex[occupants[s].index()] = absorbed[s] - occupants[s].pairSlackAbsorbedSeed();
            }
            return;
        }
        double[] retain = BatchReadbackNative.readBatchSlackRetainedShare(handle);
        if (retain == null) {
            return;
        }
        int nDistributedSlackGenerators = gpuData.distributedSlack().generatorBus() == null ? 0 : gpuData.distributedSlack().generatorBus().length;
        double[] distributedSlackGeneratorMove = null;
        double[] targets = null;
        for (int s = 0; s < realCount; s++) {
            SlotOccupant occupant = occupants[s];
            int idx = occupant.index();
            if (Double.isNaN(retain[s])) {                   // no pass fired: the pre-distribution alone
                distributedByIndex[idx] = occupant.pairDistribution() ? 0.0 : preDistributedByIndex[idx];
                continue;
            }
            double participants = 0;
            // The kernel runs the per-generator water-fill unless a scenario is re-anchored WITHOUT per-generator
            // anchors: then the bus-level loop moves the row targets and leaves distributedSlackGeneratorMove at zero (distributeSlackBatch's
            // `anchored` test, on the same values). distributedSlackGeneratorMove is measured from each generator's own anchor.
            double[] anchorShift = occupant.distributedSlackBusAnchorShift();
            boolean anchored = false;
            for (int i = 0; anchorShift != null && i < gpuData.distributedSlack().busPhiRow().length && !anchored; i++) {
                anchored = anchorShift[i] != 0.0;
            }
            if (nDistributedSlackGenerators > 0 && (!anchored || occupant.generatorAnchored())) {
                if (distributedSlackGeneratorMove == null) {
                    distributedSlackGeneratorMove = BatchReadbackNative.readBatchDistributedSlackGeneratorMove(handle);
                }
                if (distributedSlackGeneratorMove == null) {
                    continue;
                }
                for (int g = 0; g < nDistributedSlackGenerators; g++) {
                    if (gpuData.distributedSlack().generatorBus()[g] >= 0) {           // the slack's own generators are in retain
                        participants += distributedSlackGeneratorMove[s * nDistributedSlackGenerators + g];
                    }
                }
            } else {                                         // bus-level participants: targets from the anchor
                if (targets == null) {
                    targets = BatchReadbackNative.readBatchTargets(handle);
                }
                int ov = occupant.distributedSlackOverrideBus();
                for (int i = 0; i < gpuData.distributedSlack().busPhiRow().length; i++) {
                    double anchor = gpuData.distributedSlack().busBaseTargetP()[i] - (ov == i ? occupant.distributedSlackOverrideBaseTargetPDelta() : 0.0)
                            + (anchorShift != null ? anchorShift[i] : 0.0);
                    participants += targets[s * n + gpuData.distributedSlack().busPhiRow()[i]] - anchor;
                }
            }
            // a slack-only pass measures the slack's take from the target the pre-distribution moved
            distributedByIndex[idx] = participants + retain[s] + occupant.preSlackSeed();
        }
    }

    /** The run's constants the chunk loop's steps read. */
    private record RunView(ChunkInputs in, int n, GpuAcData gpuData, double[] baseState, double[] baseTargets,
                           List<BatchContingency> contingencies, BatchConfig config, int nDistributedSlackBuses,
                           int nDistributedSlackGenerators, int nReactiveLimitEntries, boolean slackOnlyDistribution,
                           boolean reactiveLimitsOuterLoop, int[] baseReactiveLimitsState, java.util.Map<String, Integer> indexById,
                           ParentCaptures captures, int[][] frozenRowsByIndex, int[][] islandedDistributedSlackByIndex,
                           List<com.powsybl.openloadflow.network.LfHvdc>[] islandedHvdcsByIndex,
                           List<GpuHvdcAcEmulationLimitsOuterLoop.Controller> hvdcControllers,
                           java.util.Map<com.powsybl.openloadflow.network.LfHvdc, Integer> hvdcElemIndex, double[][] resultsByIndex,
                           double[] preDistributedByIndex) {
    }

    /** The scenarios one install puts into the bank: group[chunkStart .. chunkStart + chunkSize), realCount of them real. */
    private record ChunkView(List<Integer> group, int chunkStart, int chunkSize, int realCount, boolean pairPhase) {
    }

    /** What a chunk's seeding steps write for its solve (null: not used by this chunk). */
    private static final class ChunkSeeds {
        double[] distributedSlackBusAnchorShift;
        double[] distributedSlackGeneratorAnchor;       // pairs: each participating generator's anchor (s * nDistributedSlackGenerators + g)
        double[] distributedSlackGeneratorSeedAnchor;   // ...and the same without the strategy's GeneratorActions
        double[] slackBusAnchorShift;
        double[] pairSlackAbsorbedSeed;                 // slack-only pairs: the parent's absorbed total they start from
        boolean[] rejectAfterSolve;                     // pairs whose parent the device did not answer: CPU
        final boolean[] pairParentConverged;
        double[] reactiveLimitSeedTargetQ;              // pairs: each parent's final frozen target Q (NaN = none)
        double[] preSlackSeed;                          // the loss pre-distribution's share on the slack generators
        double[] x0Full;                                // pairs: every slot's starting state
        int[] reactiveLimitsSeedState;
        int[] x0SeedIdx;                                // an islanding chunk: the frozen rows' starting values
        double[] x0SeedVal;

        ChunkSeeds(int chunkSize) {
            pairParentConverged = new boolean[chunkSize];
        }
    }

    /** What each contingency's solve leaves its operator strategies, by contingency index (see captureForStrategies). */
    private static final class ParentCaptures {
        final java.util.Map<Integer, int[]> reactiveLimitsStateByIndex = new java.util.HashMap<>();
        final java.util.Map<Integer, double[]> reactiveLimitTargetQByIndex = new java.util.HashMap<>();
        final java.util.Map<Integer, double[]> distributedSlackTargetsByIndex = new java.util.HashMap<>();
        final java.util.Map<Integer, double[]> genMovesByIndex = new java.util.HashMap<>();
        final java.util.Map<Integer, double[]> exhaustedStartByIndex = new java.util.HashMap<>();
        final java.util.Map<Integer, Double> slackAbsorbedByIndex = new java.util.HashMap<>();
    }

    /** The reactive-limit state every scenario starts from: the base's, as the device seeds it (initReactiveLimitQLimitTypeBatch). */
    private static int[] baseReactiveLimitsState(GpuAcData gpuData, double[] baseTargets, int[] baseRowMode, double[] baseState,
                                                 boolean reactiveLimitsOuterLoop) {
        int nReactiveLimitEntries = gpuData.reactiveLimits().voltageRow().length;
        int[] baseReactiveLimitsState = new int[nReactiveLimitEntries];
        byte[] baseLimitTypes = gpuData.reactiveLimits().baseQLimitType();
        for (int i = 0; reactiveLimitsOuterLoop && i < nReactiveLimitEntries; i++) {
            int vrow = gpuData.reactiveLimits().voltageRow()[i];
            // remote entries (plan-B3): base mode 1 = DISTR_Q share (still PV); only mode 4 is switched to PQ.
            // The pilot's V equals the group target at the base (degenerate) — key the direction on
            // WHICH limit the extracted BUS_TARGET_Q target equals instead.
            boolean remote = gpuData.reactiveLimits().voltageControlRole().length > i && gpuData.reactiveLimits().voltageControlRole()[i] != GpuVoltageControlRole.LOCAL;
            byte olfKind = baseLimitTypes.length > i ? baseLimitTypes[i] : GpuQLimitType.NONE;
            if (remote) {
                double tgt = baseTargets[vrow];
                baseReactiveLimitsState[i] = baseRowMode[vrow] == GpuRowMode.BUS_TARGET_Q
                        ? (olfKind != GpuQLimitType.NONE ? olfKind
                           : (Math.abs(tgt - (gpuData.reactiveLimits().maxQ()[i] - gpuData.reactiveLimits().loadQ()[i]))
                              <= Math.abs(tgt - (gpuData.reactiveLimits().minQ()[i] - gpuData.reactiveLimits().loadQ()[i])) ? GpuQLimitType.MAX_Q : GpuQLimitType.MIN_Q))
                        : GpuQLimitType.NONE;
            } else {
                baseReactiveLimitsState[i] = baseRowMode[vrow] == GpuRowMode.EQUATION
                        ? (olfKind != GpuQLimitType.NONE ? olfKind
                           : (baseState[vrow] < gpuData.reactiveLimits().targetV()[i] ? GpuQLimitType.MAX_Q : GpuQLimitType.MIN_Q))
                        : GpuQLimitType.NONE;
            }
        }
        return baseReactiveLimitsState;
    }

    /**
     * The loss pre-distribution of a chunk of contingencies, as OLF's DefaultContingencyActivePowerLossDistribution distributes
     * the lost active power before the post-contingency load flow: participant target overrides, and the slack's own share
     * in seeds.
     */
    private static void preDistributeContingencyLosses(RunView run, ChunkView chunk, ChunkLists cl, RunScratch rs, ChunkSeeds seeds) {
        int n = run.n();
        GpuAcData gpuData = run.gpuData();
        double[] baseTargets = run.baseTargets();
        List<BatchContingency> contingencies = run.contingencies();
        int nDistributedSlackBuses = run.nDistributedSlackBuses();
        int[][] frozenRowsByIndex = run.frozenRowsByIndex();
        int[][] islandedDistributedSlackByIndex = run.islandedDistributedSlackByIndex();
        double[] preDistributedByIndex = run.preDistributedByIndex();
        List<Integer> group = chunk.group();
        int chunkStart = chunk.chunkStart();
        int chunkSize = chunk.chunkSize();
        int realCount = chunk.realCount();
        double[] slackBusAnchorShift = seeds.slackBusAnchorShift;
        double[] preSlackSeed = seeds.preSlackSeed;
        // CPU parity (DefaultContingencyActivePowerLossDistribution): with DS on, the
        // contingency's SCHEDULED lost active power is distributed ONCE over the
        // participating generators BEFORE the post-contingency solve — the CPU's DS loop
        // then only measures the loss-nonlinearity residual (usually inside the band, so
        // it never redistributes). Replay the same one-shot water-fill host-side
        // (GpuBatchLossPreDistribution) and hand it to the device as the strategy-pair
        // trio: participant target overrides + anchor shifts + the slack's own share.
        for (int s = 0; s < realCount; s++) {
            // The amount OLF's DefaultContingencyActivePowerLossDistribution returns for this
            // contingency (mismatch - remaining, pu), as the replay below distributes it.
            int preIdx = group.get(chunkStart + s);
            if (cl.preLossBySlot[s] == 0) {
                preDistributedByIndex[preIdx] = 0.0;
                continue;
            }
            if (!rs.hasDistributedSlack) {
                // the slack generators take what their rooms allow, as OLF's water-fill places it;
                // the rest stays for the DistributedSlack check, where the failure behaviour decides
                double placed = Math.min(Math.max(cl.preLossBySlot[s], gpuData.distributedSlack().slackRoomDown()),
                        gpuData.distributedSlack().slackRoomUp());
                preDistributedByIndex[preIdx] = placed;
                // SLACK-ONLY (the slack bus is the only participant): OLF distributes
                // the whole scheduled loss into the slack generator's own target, so
                // its target for this scenario is base + loss. The device has no
                // BUS_TARGET_P row at the slack — its P is free — so slackBusTargetP is
                // a RUN CONSTANT and that displacement has nowhere to live except
                // dSlackOnlyAbsorbed, which slackBusAnchorShift seeds.
                //
                // Without this the whole block was skipped (the gate was hasDistributedSlack), the
                // accumulator started at 0, and slackOnlyDistributedSlackCheck measured the mismatch
                // against the BASE slack target: on the deviceHvdc* networks that is
                // -3.006957565 pu where OLF sees -0.006957565, so the check fired on
                // the first pass and absorbed the lot. OLF reports STABLE there (0.0070
                // < the 0.01 band) and fires only after the HVDC unfreeze, on -0.013703031
                // — which our accumulator, 3.0 too low, then read as -0.006745466 and
                // let pass. Same single distribution, wrong pass, and the two arms end
                // their Newton sequences on different calls.
                if (slackBusAnchorShift == null) {
                    slackBusAnchorShift = new double[chunkSize];
                }
                slackBusAnchorShift[s] = placed;
                if (GpuTrace.PRELOSS_TRACE.isOn()) {
                    System.err.printf("PRELOSS_SLACKONLY ctg=%s loss=%s slackBase=%s -> slackTarget=%s%n",
                            contingencies.get(group.get(chunkStart + s)).contingencyId(),
                            Double.toString(cl.preLossBySlot[s]),
                            Double.toString(gpuData.distributedSlack().slackTargetP()),
                            Double.toString(gpuData.distributedSlack().slackTargetP() + placed));
                }
                continue;
            }
            // A FRAGMENTING contingency puts some DS participants in another
            // synchronous component. OLF distributes over
            // getSynchronousNetworks().getFirst(), so they take no share — mark them
            // and let the water-fill exclude them, rather than skipping the whole
            // pre-distribution (which left the device DS loop to water-fill the
            // mismatch during the solve, at a different point inside the slack band).
            int ci = group.get(chunkStart + s);
            int[] frozen = frozenRowsByIndex[ci];
            boolean[] islandedDistributedSlackMask = null;
            // The PLAN's mask first (islandedDistributedSlackParticipants): it matches a participant on its BUS
            // as well as on its row, which the row test below cannot do for a participant whose P
            // row the coupler-tree relocation moved onto its parent coupler's DUMMY_P — the frozen
            // rows are the bus's own BUS_PHI/BUS_V. Rebuilding the mask here from the row test
            // alone repeated exactly the defect that comment describes: measured on the
            // middle-section fixture, the pre-loss gave the ISLANDED generator's relocated row a
            // share of -0.894 pu, and the device's slack loop then chased its own tail for five
            // passes and retired 0.837 pu away from OLF.
            if (islandedDistributedSlackByIndex != null && islandedDistributedSlackByIndex[ci] != null) {
                for (int participant : islandedDistributedSlackByIndex[ci]) {
                    if (participant >= 0 && participant < nDistributedSlackBuses) {
                        if (islandedDistributedSlackMask == null) {
                            islandedDistributedSlackMask = new boolean[nDistributedSlackBuses];
                        }
                        islandedDistributedSlackMask[participant] = true;
                    }
                }
            }
            for (int i = 0; i < nDistributedSlackBuses; i++) {
                for (int fr : frozen) {
                    if (fr == gpuData.distributedSlack().busPhiRow()[i]) {
                        if (islandedDistributedSlackMask == null) {
                            islandedDistributedSlackMask = new boolean[nDistributedSlackBuses];
                        }
                        islandedDistributedSlackMask[i] = true;
                    }
                }
            }
            double[] shares = GpuBatchLossPreDistribution.distribute(gpuData, cl.preLossBySlot[s],
                    rs.distributedSlackOverrideBus.length > 0 ? rs.distributedSlackOverrideBus[s] : -1,
                    rs.distributedSlackOverrideFactorDelta.length > 0 ? rs.distributedSlackOverrideFactorDelta[s] : 0,
                    rs.distributedSlackOverrideBaseTargetPDelta.length > 0 ? rs.distributedSlackOverrideBaseTargetPDelta[s] : 0,
                    rs.distributedSlackOverrideMaxDelta.length > 0 ? rs.distributedSlackOverrideMaxDelta[s] : 0,
                    rs.distributedSlackOverrideMinDelta.length > 0 ? rs.distributedSlackOverrideMinDelta[s] : 0, islandedDistributedSlackMask,
                    ChunkArrays.lostSlotsOf(cl.distributedSlackLostGeneratorList, s),
                    cl.genPreMovesBySlot != null ? (cl.genPreMovesBySlot[s] = new double[gpuData.distributedSlack().generatorBus().length]) : null);
            // OLF_GPU_SEED_DUMP=<file>: the per-bus SHARE the host-side loss
            // pre-distribution seeds, against OLF's own pre-distribution section in
            // OLF_DS_SPLIT. The two totals can agree while the SPLIT does not, and the
            // split is what the first solve sees — so the mismatch the DS loop then
            // measures, and everything after it, inherits the difference.
            if (GpuTrace.SEED_DUMP.isOn() && shares != null) {
                try (java.io.PrintWriter w = new java.io.PrintWriter(
                        new java.io.FileWriter(GpuTrace.SEED_DUMP.value(), true))) {
                    w.printf("# SEED ctg=%s%n",
                            contingencies.get(group.get(chunkStart + s)).contingencyId());
                    for (int i = 0; i < nDistributedSlackBuses; i++) {
                        if (shares[i] != 0) {
                            w.printf("%d %.12f%n", i, shares[i]);
                        }
                    }
                } catch (java.io.IOException e) {
                    throw new java.io.UncheckedIOException(e);
                }
            }
            if (GpuTrace.PRELOSS_TRACE.isOn()) {
                double absorbed = 0;
                if (shares != null) {
                    for (int i = 0; i < nDistributedSlackBuses; i++) {
                        absorbed += shares[i];
                    }
                }
                double fsAll = 0;
                double fsIslanded = 0;
                for (int i = 0; i < nDistributedSlackBuses; i++) {
                    fsAll += gpuData.distributedSlack().busFactor()[i];
                    if (islandedDistributedSlackMask != null && islandedDistributedSlackMask[i]) {
                        fsIslanded += gpuData.distributedSlack().busFactor()[i];
                    }
                }
                System.err.printf("ISLANDED_DS ctg=%s count=%d%n",
                        contingencies.get(ci).contingencyId(), ChunkArrays.countTrue(islandedDistributedSlackMask));
                System.err.printf("PRELOSS_FACTORS ctg=%s factorSumAll=%.6f factorSumIslanded=%.6f "
                        + "(%.2f%% of the participation is in the island)%n",
                        contingencies.get(ci).contingencyId(), fsAll, fsIslanded,
                        fsAll > 0 ? 100.0 * fsIslanded / fsAll : 0.0);
                System.err.printf("PRELOSS ctg=%s mismatch=%.6f pu absorbed=%.6f pu slackShare=%.6f "
                        + "islandedDistributedSlackParticipants=%d%n",
                        contingencies.get(ci).contingencyId(), cl.preLossBySlot[s], absorbed,
                        shares == null ? 0.0 : shares[nDistributedSlackBuses],
                        ChunkArrays.countTrue(islandedDistributedSlackMask));
            }
            if (shares == null) {
                preDistributedByIndex[preIdx] = 0.0;
                continue;
            }
            double preDistributed = 0;
            for (double share : shares) {                 // participants + the slack's own share
                preDistributed += share;
            }
            preDistributedByIndex[preIdx] = preDistributed;
            if (preDistributed == shares[nDistributedSlackBuses] && shares[nDistributedSlackBuses] != 0
                    && !GpuBatchLossPreDistribution.slackInPack(gpuData)) {
                // Only the slack bus's generators participate: OLF moved their target by the whole
                // loss, and its slack check measures against that target. The device's slack-only
                // pass (distributeSlackBatch) measures against slackTargetP + slackBusAnchorShift.
                if (slackBusAnchorShift == null) {
                    slackBusAnchorShift = new double[chunkSize];
                }
                if (preSlackSeed == null) {
                    preSlackSeed = new double[chunkSize];
                }
                slackBusAnchorShift[s] = shares[nDistributedSlackBuses];
                preSlackSeed[s] = shares[nDistributedSlackBuses];
            }
            for (int i = 0; i < nDistributedSlackBuses; i++) {
                cl.preAbsorbedBySlot[s] += shares[i];        // participants only
            }
            // The pre-distribution is a DISPLACEMENT from the anchor, not a new anchor.
            // OLF's post-contingency DS calls resetToInitialState, which UNDOES it and
            // returns it as previousMismatch, then water-fills
            // (preLoss + slackMismatch) ONCE from the anchor. Moving the anchor here
            // instead made the device fill only the slack mismatch, and a clamped
            // water-fill does not compose: on rte6515/LINE-6180-6074 OLF fills +4.639661
            // in one pass and saturates 10 generators at max, while the device filled
            // +4.8 then -0.160339 and saturated none. Seeding the TARGETS (so the solve
            // starts at the same operating point) while leaving the anchor alone is what
            // makes the kernel's reset pick the shares back up as absorbedNow.
            for (int i = 0; i < nDistributedSlackBuses; i++) {
                if (shares[i] == 0) {
                    continue;
                }
                // The row this scenario actually solves the bus's P balance on: the
                // coupler-tree relocation's dummy row unless this scenario opened that
                // coupler, in which case the balance — and so the share — is on the bus's
                // own phi row. The ANCHOR stays the participant's own distributedSlackBusBaseTargetP, which
                // the device carried to the new row with the equation.
                int distributedSlackBusRow = cl.distributedSlackBusRowOverrideByScenario[s] != null && cl.distributedSlackBusRowOverrideByScenario[s][i] >= 0
                        ? cl.distributedSlackBusRowOverrideByScenario[s][i] : gpuData.distributedSlack().busPhiRow()[i];
                int flat = s * n + distributedSlackBusRow;
                // compose with a gen-outage override on the same row
                rs.overrides.compose(flat, baseTargets[gpuData.distributedSlack().busPhiRow()[i]], shares[i]);
            }
        }
        seeds.slackBusAnchorShift = slackBusAnchorShift;
        seeds.preSlackSeed = preSlackSeed;
    }

    /**
     * The starting point of a chunk's scenarios: the frozen rows of an islanding contingency, and for an operator strategy
     * its parent contingency's post-contingency state - its voltages, PV/PQ states and distributed-slack targets - as OLF
     * starts a strategy's load flow from the network the contingency's left.
     */
    private static void seedWarmStarts(RunView run, ChunkView chunk, ChunkLists cl, RunScratch rs, ChunkSeeds seeds) {
        int n = run.n();
        GpuAcData gpuData = run.gpuData();
        double[] baseState = run.baseState();
        double[] baseTargets = run.baseTargets();
        List<BatchContingency> contingencies = run.contingencies();
        BatchConfig config = run.config();
        int nDistributedSlackBuses = run.nDistributedSlackBuses();
        int nDistributedSlackGenerators = run.nDistributedSlackGenerators();
        int nReactiveLimitEntries = run.nReactiveLimitEntries();
        boolean slackOnlyDistribution = run.slackOnlyDistribution();
        boolean reactiveLimitsOuterLoop = run.reactiveLimitsOuterLoop();
        int[] baseReactiveLimitsState = run.baseReactiveLimitsState();
        java.util.Map<String, Integer> indexById = run.indexById();
        int[][] frozenRowsByIndex = run.frozenRowsByIndex();
        double[][] resultsByIndex = run.resultsByIndex();
        java.util.Map<Integer, int[]> reactiveLimitsStateByIndex = run.captures().reactiveLimitsStateByIndex;
        java.util.Map<Integer, double[]> reactiveLimitTargetQByIndex = run.captures().reactiveLimitTargetQByIndex;
        java.util.Map<Integer, double[]> distributedSlackTargetsByIndex = run.captures().distributedSlackTargetsByIndex;
        java.util.Map<Integer, double[]> genMovesByIndex = run.captures().genMovesByIndex;
        java.util.Map<Integer, double[]> exhaustedStartByIndex = run.captures().exhaustedStartByIndex;
        java.util.Map<Integer, Double> slackAbsorbedByIndex = run.captures().slackAbsorbedByIndex;
        List<Integer> group = chunk.group();
        int chunkStart = chunk.chunkStart();
        int chunkSize = chunk.chunkSize();
        int realCount = chunk.realCount();
        boolean pairPhase = chunk.pairPhase();
        double[] distributedSlackBusAnchorShift = seeds.distributedSlackBusAnchorShift;
        double[] distributedSlackGeneratorAnchor = seeds.distributedSlackGeneratorAnchor;
        double[] distributedSlackGeneratorSeedAnchor = seeds.distributedSlackGeneratorSeedAnchor;
        double[] slackBusAnchorShift = seeds.slackBusAnchorShift;
        double[] pairSlackAbsorbedSeed = seeds.pairSlackAbsorbedSeed;
        boolean[] rejectAfterSolve = seeds.rejectAfterSolve;
        boolean[] pairParentConverged = seeds.pairParentConverged;
        double[] reactiveLimitSeedTargetQ = seeds.reactiveLimitSeedTargetQ;
        double[] x0Full = seeds.x0Full;
        int[] reactiveLimitsSeedState = seeds.reactiveLimitsSeedState;
        int[] x0SeedIdx = seeds.x0SeedIdx;
        double[] x0SeedVal = seeds.x0SeedVal;
        // Pair phase: warm continuation from each parent contingency's state + its PV/PQ states.
        if (!cl.frozenSeedSlot.isEmpty()) {         // islanding in this chunk: seed the frozen rows
            // Only the frozen rows differ from the base state: send them as seeds the device writes over
            // its own tiling of the base, instead of chunkSize * n doubles (372 MB per chunk on rte6515,
            // 1.5 GB for a single 14,260-scenario chunk). The HVDC freeze reads its warm start from the
            // same seeds.
            x0SeedIdx = new int[cl.frozenSeedSlot.size()];
            x0SeedVal = new double[cl.frozenSeedSlot.size()];
            for (int k = 0; k < cl.frozenSeedSlot.size(); k++) {
                x0SeedIdx[k] = cl.frozenSeedSlot.get(k) * n + cl.frozenSeedRow.get(k);
                x0SeedVal[k] = cl.frozenSeedValue.get(k);
            }
        }
        // Strategy-leg DS re-anchoring: OLF's processContingency calls
        // setGeneratorsInitialTargetPToTargetP() before the strategy re-solve, so the CPU's strategy
        // leg distributes only the INCREMENT from the parent contingency's already-distributed
        // targets. The device mirrors this with three coupled pieces (each alone is wrong —
        // participant shifts without the slack shift leave the band check measured against the
        // pre-distribution slack target, which kills the distribution):
        //  - distributedSlackBusAnchorShift: participant water-fill anchors moved to the parent's final DS targets;
        //  - slackBusAnchorShift: the slack bus's own absorbed share of the parent's distribution
        //    ((1-fs)/fs x the participants' absorbed sum), so the mismatch/band is measured against
        //    the parent's slack target exactly as the CPU leg measures it;
        //  - DS-row target seeds at the parent values, so the first inner solve runs at the CPU
        //    leg's starting operating point (losses are nonlinear along the path).
        if (pairPhase) {
            x0Full = new double[chunkSize * n];
            reactiveLimitsSeedState = reactiveLimitsOuterLoop && nReactiveLimitEntries > 0 ? new int[chunkSize * nReactiveLimitEntries] : null;
            if (nDistributedSlackBuses > 0) {
                distributedSlackBusAnchorShift = new double[chunkSize * nDistributedSlackBuses];
                slackBusAnchorShift = new double[chunkSize];
                distributedSlackGeneratorAnchor = nDistributedSlackGenerators > 0 ? new double[chunkSize * nDistributedSlackGenerators] : null;
                distributedSlackGeneratorSeedAnchor = nDistributedSlackGenerators > 0 ? new double[chunkSize * nDistributedSlackGenerators] : null;
            } else if (slackOnlyDistribution) {
                slackBusAnchorShift = new double[chunkSize];    // absorbed-total seeds (no participant anchors)
                pairSlackAbsorbedSeed = new double[chunkSize];
            }
            double distributedSlackFactorSum = 0;
            for (double fv : gpuData.distributedSlack().busFactor()) {
                distributedSlackFactorSum += fv;
            }
            for (int s = 0; s < chunkSize; s++) {
                double[] parent = null;
                int[] parentReactiveLimitsState = null;
                double[] parentDistributedSlackTargets = null;
                Double parentSlackAbs = null;
                if (s < realCount) {
                    String pairId = contingencies.get(group.get(chunkStart + s)).contingencyId();
                    Integer pIdx = indexById.get(pairId.substring(0, pairId.indexOf(BatchContingency.STRATEGY_KEY_SEP)));
                    int parentStatus = pIdx != null && resultsByIndex[pIdx] != null
                            ? ChunkArrays.rowStatus(resultsByIndex[pIdx], n) : GpuAcSolverStatus.REJECTED;
                    if (parentStatus == GpuAcSolverStatus.CONVERGED) {
                        parent = ChunkArrays.fullRow(resultsByIndex[pIdx], n, config.stateSink(),
                                contingencies.get(pIdx).contingencyId());
                        pairParentConverged[s] = true;
                    } else if (parentStatus == GpuAcSolverStatus.MAX_ITERATION_REACHED) {
                        parent = exhaustedStartByIndex.get(pIdx);   // OLF's network after the failed run
                    }
                    if (parent != null) {
                        parentReactiveLimitsState = reactiveLimitsStateByIndex.get(pIdx);
                        parentDistributedSlackTargets = distributedSlackTargetsByIndex.get(pIdx);
                        parentSlackAbs = slackAbsorbedByIndex.get(pIdx);
                    } else {
                        // The device has no answer for the parent (the CPU replays it): OLF starts the strategy
                        // from the state that replay leaves, which only the CPU has. CPU.
                        if (rejectAfterSolve == null) {
                            rejectAfterSolve = new boolean[chunkSize];
                        }
                        rejectAfterSolve[s] = true;
                        if (s < realCount) {
                            GpuBatchScopeCensus.record("PAIR:PARENT_NOT_ON_DEVICE");
                        }
                    }
                }
                System.arraycopy(parent != null ? parent : baseState, 0, x0Full, s * n, n);
                int parentIdx = ChunkArrays.pIdxOf(contingencies, group, chunkStart, s, indexById, realCount);
                if (parent != null && parentIdx >= 0 && frozenRowsByIndex[parentIdx] != null) {
                    // A bus the parent islanded and this pair re-energizes (a reclose) starts where OLF starts it:
                    // its stored voltage, which a disabled bus keeps at the base solution - not the parent's frozen
                    // identity (1 pu at angle 0), from which Newton found the collapsed root (rte6515 LINE-1404-389
                    // reclosed: VL-1404 at 0.02 kV where OLF has 65.26 kV)
                    java.util.Set<Integer> stillFrozen = new java.util.HashSet<>();
                    int[] pf = s < realCount ? frozenRowsByIndex[group.get(chunkStart + s)] : null;
                    for (int r : pf == null ? new int[0] : pf) {
                        stillFrozen.add(r);
                    }
                    for (int r : frozenRowsByIndex[parentIdx]) {
                        if (!stillFrozen.contains(r)) {
                            x0Full[s * n + r] = baseState[r];
                        }
                    }
                }
                for (int k = 0; k < cl.frozenSeedSlot.size(); k++) {
                    if (cl.frozenSeedSlot.get(k) == s) {   // a frozen row stays AT its frozen state
                        x0Full[s * n + cl.frozenSeedRow.get(k)] = cl.frozenSeedValue.get(k);
                    }
                }
                if (reactiveLimitsSeedState != null) {
                    // the parent's PV/PQ states, not its switch counts: OLF runs the strategy as a new load flow,
                    // whose reactive-limit loop counts PV -> PQ switches afresh
                    int[] seedFrom = parentReactiveLimitsState != null ? parentReactiveLimitsState : baseReactiveLimitsState;
                    for (int i = 0; i < nReactiveLimitEntries; i++) {
                        reactiveLimitsSeedState[s * nReactiveLimitEntries + i] = GpuReactiveLimitsState.limitType(seedFrom[i]);
                    }
                    double[] parentTargetQ = parentReactiveLimitsState != null
                            ? reactiveLimitTargetQByIndex.get(ChunkArrays.pIdxOf(contingencies, group, chunkStart, s, indexById, realCount)) : null;
                    if (parentTargetQ != null) {
                        if (reactiveLimitSeedTargetQ == null) {
                            reactiveLimitSeedTargetQ = new double[chunkSize * nReactiveLimitEntries];
                            java.util.Arrays.fill(reactiveLimitSeedTargetQ, Double.NaN);
                        }
                        System.arraycopy(parentTargetQ, 0, reactiveLimitSeedTargetQ, s * nReactiveLimitEntries, nReactiveLimitEntries);
                    }
                }
                if (slackOnlyDistribution && slackBusAnchorShift != null && parentSlackAbs != null) {
                    slackBusAnchorShift[s] = parentSlackAbs;    // seed the pair with the parent's absorbed total
                    pairSlackAbsorbedSeed[s] = parentSlackAbs;
                }
                if (parentDistributedSlackTargets != null && distributedSlackBusAnchorShift != null && distributedSlackFactorSum > 0) {
                    // parentDistributedSlackTargets = [absorption per DS row..., parent factor sum] (captured after the
                    // parent's solve). The anchor shift is the parent's PURE ABSORPTION — the part
                    // of the parent's final DS target that its distribution added on top of the
                    // parent's own effective base — which composes with the pair's own distributedSlackOverrideBaseTargetPDelta
                    // in the kernel (base_eff = base - ovDelta + shift), so gen-outage parents and
                    // action deltas are both handled without double counting.
                    double absorbedSum = 0;
                    // An absolute GeneratorAction on a participant sets its target (and OLF's initial
                    // target) to the value: its bus is not re-anchored on the parent's share, the
                    // action's own delta (base-referenced, on the override bus) is the whole move.
                    // The precomputer only batches it when that generator is its bus's only participant.
                    int absoluteBus = -1;
                    if (rs.distributedSlackOverrideBus.length > 0 && rs.distributedSlackOverrideBus[s] >= 0) {
                        for (com.powsybl.openloadflow.network.LfGenerator g : contingencies.get(group.get(chunkStart + s)).absoluteTargetGenerators()) {
                            if (config.distributedSlackParameters().factor(g) != 0) {
                                absoluteBus = rs.distributedSlackOverrideBus[s];
                            }
                        }
                    }
                    for (int i = 0; i < nDistributedSlackBuses; i++) {
                        int row = gpuData.distributedSlack().busPhiRow()[i];
                        double shift = i == absoluteBus ? 0.0 : parentDistributedSlackTargets[i];
                        distributedSlackBusAnchorShift[s * nDistributedSlackBuses + i] = shift;
                        absorbedSum += parentDistributedSlackTargets[i];      // the slack's share of the parent counts it all
                        // Seed the DS-row target at the CPU strategy leg's starting value — the
                        // parent-absorbed target plus the pair's own composed delta — so the first
                        // inner solve runs at the CPU leg's starting operating point (losses are
                        // nonlinear along the path). A pair override on the row (gen action /
                        // gen-outage pTarget) is base-referenced: add the parent absorption to it. On the row
                        // this pair solves the balance on (its own coupler openings), from the participant's
                        // own base target, as the loss pre-distribution composes its shares.
                        int[] rowOverrides = cl.distributedSlackBusRowOverrideByScenario[s];
                        int pairRow = rowOverrides != null && rowOverrides[i] >= 0 ? rowOverrides[i] : row;
                        rs.overrides.compose(s * n + pairRow, baseTargets[row], shift);
                    }
                    // the slack's own absorption in the parent's distribution: its own factor's share
                    // against the participants that distribution renormalized over (the kernel's retain),
                    // not one minus their sum - a lost or islanded participant's factor is not the
                    // slack's (pilotSelfCurveNetwork losing g3, the slack out of the distribution:
                    // 0.2234 pu of slack target where OLF has none, 2.1 MW distributed against 20.2)
                    double parentFs = parentDistributedSlackTargets[nDistributedSlackBuses];
                    slackBusAnchorShift[s] = parentFs > 0 ? (1.0 - distributedSlackFactorSum) / parentFs * absorbedSum : 0.0;
                }
                if (distributedSlackGeneratorAnchor != null && s < realCount) {
                    // Each generator's own anchor: where the parent's distribution left it, and for a
                    // GeneratorAction the action's value (LfGeneratorAction sets the target AND the initial
                    // target): the parent's target plus the change, or the value itself.
                    String pairId = contingencies.get(group.get(chunkStart + s)).contingencyId();
                    Integer pIdx = indexById.get(pairId.substring(0, pairId.indexOf(BatchContingency.STRATEGY_KEY_SEP)));
                    double[] parentMoves = parentDistributedSlackTargets != null && pIdx != null ? genMovesByIndex.get(pIdx) : null;
                    if (parentMoves != null) {
                        System.arraycopy(parentMoves, 0, distributedSlackGeneratorAnchor, s * nDistributedSlackGenerators, nDistributedSlackGenerators);
                        System.arraycopy(parentMoves, 0, distributedSlackGeneratorSeedAnchor, s * nDistributedSlackGenerators, nDistributedSlackGenerators);
                        if (slackBusAnchorShift != null && GpuBatchLossPreDistribution.slackInPack(gpuData)) {
                            // the slack's generators are listed: the parent left them at their own moves,
                            // except one an absolute action restarts from its value (the action's own
                            // move of the slack target carries that)
                            java.util.Set<String> absolute = new java.util.HashSet<>();
                            for (BatchContingency.GeneratorMove m : contingencies.get(group.get(chunkStart + s)).generatorActions()) {
                                if (m.absolute()) {
                                    absolute.add(m.generator().getId());
                                }
                            }
                            double slackMoves = 0;
                            for (int g = 0; g < nDistributedSlackGenerators; g++) {
                                if (gpuData.distributedSlack().generatorBus()[g] < 0 && !absolute.contains(gpuData.distributedSlack().generatorId()[g])) {
                                    slackMoves += parentMoves[g];
                                }
                            }
                            slackBusAnchorShift[s] = slackMoves;
                        }
                    }
                    // OLF re-checks each moved generator's participation at its new target
                    // (reApplyActivePowerControlChecks); the last action on a generator decides
                    java.util.Map<Integer, Boolean> staysBySlot = new java.util.TreeMap<>();
                    for (BatchContingency.GeneratorMove m : contingencies.get(group.get(chunkStart + s)).generatorActions()) {
                        int g = ChunkArrays.distributedSlackGeneratorIndexOf(gpuData, m.generator().getId());
                        if (g >= 0) {
                            // its target before the action: its anchor plus where the parent left it
                            double before = config.distributedSlackParameters().anchorTargetP(m.generator()) + distributedSlackGeneratorAnchor[s * nDistributedSlackGenerators + g];
                            staysBySlot.put(g, m.participatesAt() == null || m.participatesAt().test(m.newTargetP(before)));
                            int k = s * nDistributedSlackGenerators + g;
                            distributedSlackGeneratorAnchor[k] = m.absolute() ? m.deltaP() : distributedSlackGeneratorAnchor[k] + m.deltaP();
                        }
                    }
                    for (var e : staysBySlot.entrySet()) {
                        if (!e.getValue()) {             // out of the distribution: it keeps the action's target
                            cl.distributedSlackRemovedGeneratorList.add(s);
                            cl.distributedSlackRemovedGeneratorList.add(e.getKey());
                        }
                    }
                }
            }
        }
        seeds.distributedSlackBusAnchorShift = distributedSlackBusAnchorShift;
        seeds.distributedSlackGeneratorAnchor = distributedSlackGeneratorAnchor;
        seeds.distributedSlackGeneratorSeedAnchor = distributedSlackGeneratorSeedAnchor;
        seeds.slackBusAnchorShift = slackBusAnchorShift;
        seeds.pairSlackAbsorbedSeed = pairSlackAbsorbedSeed;
        seeds.rejectAfterSolve = rejectAfterSolve;
        seeds.reactiveLimitSeedTargetQ = reactiveLimitSeedTargetQ;
        seeds.x0Full = x0Full;
        seeds.reactiveLimitsSeedState = reactiveLimitsSeedState;
        seeds.x0SeedIdx = x0SeedIdx;
        seeds.x0SeedVal = x0SeedVal;
    }

    /**
     * The HvdcActions of a chunk's operator strategies: each station's target at the link's flow in the state the strategy
     * starts from, the link out of its equations. Returns the scenarios the CPU must solve (a parent that did not converge).
     */
    private static boolean[] applyHvdcActions(RunView run, ChunkView chunk, RunScratch rs, double[] x0Full, boolean[] pairParentConverged,
                                              boolean[] rejectedBefore, List<double[]> hvdcActionApplied) {
        boolean[] rejectAfterSolve = rejectedBefore;
        int n = run.n();
        double[] baseTargets = run.baseTargets();
        List<BatchContingency> contingencies = run.contingencies();
        List<com.powsybl.openloadflow.network.LfHvdc>[] islandedHvdcsByIndex = run.islandedHvdcsByIndex();
        List<GpuHvdcAcEmulationLimitsOuterLoop.Controller> hvdcControllers = run.hvdcControllers();
        java.util.Map<com.powsybl.openloadflow.network.LfHvdc, Integer> hvdcElemIndex = run.hvdcElemIndex();
        List<Integer> group = chunk.group();
        int chunkStart = chunk.chunkStart();
        int chunkSize = chunk.chunkSize();
        int realCount = chunk.realCount();
        boolean pairPhase = chunk.pairPhase();
        for (int s = 0; pairPhase && s < realCount; s++) {
            List<com.powsybl.openloadflow.network.LfHvdc> off = new ArrayList<>(contingencies.get(group.get(chunkStart + s)).acEmulationOffHvdcs());
            // a link whose converter bus the contingency isolated carries no power: OLF's action changes nothing
            // (the device held the bus's frozen 1 pu / 0 rad and froze the link at the flow that gave)
            List<com.powsybl.openloadflow.network.LfHvdc> islandedLinks = islandedHvdcsByIndex != null
                    ? islandedHvdcsByIndex[group.get(chunkStart + s)] : null;
            if (islandedLinks != null) {
                off.removeAll(islandedLinks);
            }
            if (off.isEmpty()) {
                continue;
            }
            // OLF evaluates the flow on its equation system's state: a parent that did not converge left
            // its last iterate there, which the device does not keep
            boolean ok = pairParentConverged[s] && x0Full != null && hvdcControllers != null;
            for (com.powsybl.openloadflow.network.LfHvdc hvdc : off) {
                Integer elem = hvdcElemIndex.get(hvdc);
                GpuHvdcAcEmulationLimitsOuterLoop.Controller c = !ok || elem == null ? null : hvdcControllers.stream()
                        .filter(controller -> controller.hvdcElementIndex() == elem).findFirst().orElse(null);
                if (c == null) {
                    ok = false;
                    break;
                }
                double[] warm = java.util.Arrays.copyOfRange(x0Full, s * n, (s + 1) * n);
                double d1 = -c.p1At(warm);
                double d2 = -c.p2At(warm);
                int[] rm = rs.rowModePerScenario[s];
                rs.disabledMask[s][c.hvdcElementIndex()] = true;
                if (c.bus1PActive()) {
                    rs.overrides.merge(s * n + c.bus1WriteRow(rm), baseTargets[c.bus1WriteRow(rm)], c.baseD1(), d1);
                }
                if (c.bus2PActive()) {
                    rs.overrides.merge(s * n + c.bus2WriteRow(rm), baseTargets[c.bus2WriteRow(rm)], c.baseD2(), d2);
                }
                hvdcActionApplied.add(new double[] {s, hvdcControllers.indexOf(c), d1 - c.baseD1(), d2 - c.baseD2()});
            }
            if (!ok) {
                if (rejectAfterSolve == null) {
                    rejectAfterSolve = new boolean[chunkSize];
                }
                rejectAfterSolve[s] = true;
                GpuBatchScopeCensus.record("PAIR:HVDC_ACTION_PARENT_NOT_CONVERGED");
            }
        }
        return rejectAfterSolve;
    }

    /** What a solve's harvested scenarios report: test sinks, outer-loop statistics and the traces. */
    @SuppressWarnings("java:S107")
    private static void recordAfterSolve(RunView run, ChunkView chunk, long handle, double[][] chunkResults, SlotOccupant[] harvested,
                                         int harvestedCount, int[] reactiveLimitsFinalState, boolean hasHvdcDriverLoops,
                                         boolean monitoringVoltageInKernel) {
        int n = run.n();
        GpuAcData gpuData = run.gpuData();
        double[] baseTargets = run.baseTargets();
        List<BatchContingency> contingencies = run.contingencies();
        int nDistributedSlackBuses = run.nDistributedSlackBuses();
        int nReactiveLimitEntries = run.nReactiveLimitEntries();
        int[][] islandedDistributedSlackByIndex = run.islandedDistributedSlackByIndex();
        List<GpuHvdcAcEmulationLimitsOuterLoop.Controller> hvdcControllers = run.hvdcControllers();
        boolean pairPhase = chunk.pairPhase();
        java.util.Map<String, int[]> outerLoopStatsSink = outerLoopStatsForTests;
        if (OUTER_LOOP_STATS || outerLoopStatsSink != null) {
            // OLF_OL_STATS: the device's per-contingency Newton work, in the same shape as OLF's
            // CPU_OL_STATS lines, so the two arms can be matched contingency by contingency.
            int[] outerLoopStats = BatchReadbackNative.readBatchOuterLoopStats(handle);
            int newtonLoopIterations = BatchReadbackNative.readBatchNewtonLoopIterations(handle);
            if (outerLoopStats != null) {
                for (int s2 = 0; s2 < harvestedCount; s2++) {
                    String id = contingencies.get(harvested[s2].index()).contingencyId();
                    int[] stats = java.util.Arrays.copyOf(java.util.Arrays.copyOfRange(outerLoopStats, s2 * 4, s2 * 4 + 4), 5);
                    stats[4] = newtonLoopIterations;
                    if (outerLoopStatsSink != null) {
                        outerLoopStatsSink.put(id, stats);
                    }
                    if (OUTER_LOOP_STATS) {
                        int status = chunkResults[s2] == null ? Integer.MIN_VALUE : (int) chunkResults[s2][n + 2];
                        System.err.println("GPU_OL_STATS ctg=" + id + " passes=" + stats[0] + " firstNr=" + stats[1]
                                + " totalNr=" + stats[2] + " status=" + status + " checks=" + stats[3]);
                    }
                }
            }
        }
        if (monitoringVoltageInKernel) {
            // Keep ACTIVATIONS meaningful with the loop in the driver: it is the guard
            // that a case really exercises the automaton, not orchestrator bookkeeping.
            GpuMonitoringVoltageOuterLoop.ACTIVATIONS.add(
                    VoltageMonitoringNative.readBatchMonitoringActivations(handle));
        }
        if (hasHvdcDriverLoops && GpuTrace.HVDC_TRACE.isOn()) {
            // the status each scenario ENDED with (0 linear, 1/2 saturated, 3 frozen), as the driver phases left it
            int[] devStatus = HvdcNative.readBatchHvdcStatus(handle);
            int nh = hvdcControllers.size();
            for (int s2 = 0; devStatus != null && s2 < harvestedCount; s2++) {
                StringBuilder sb = new StringBuilder();
                for (int h = 0; h < nh; h++) {
                    sb.append(' ').append(devStatus[s2 * nh + h]);
                }
                System.err.println("HVDC_END s=" + s2 + " ctg="
                        + contingencies.get(harvested[s2].index()).contingencyId() + " status:" + sb);
            }
        }
        // OLF_GPU_ITER_TRACE_CTG=<id>[,<id>]: the iteration count of NAMED scenarios, at any
        // chunk size. Two implementations that solve the same system in the same number of
        // iterations agree to machine precision, so the count is the first thing to compare
        // against OLF's own (OLF_LF_STEPS_TRACE) when the answers differ.
        String iterCtg = GpuTrace.ITER_TRACE_CTG.value();
        if (iterCtg != null && !pairPhase) {
            for (int s = 0; s < harvestedCount; s++) {
                String id = contingencies.get(harvested[s].index()).contingencyId();
                boolean named = false;
                for (String w : iterCtg.split(",")) {
                    named = named || w.trim().equals(id);
                }
                if (named) {
                    System.err.printf("ITER ctg=%s iters=%d finalMismatch=%.6e status=%d%n",
                            id, (int) chunkResults[s][n], chunkResults[s][n + 1],
                            (int) chunkResults[s][n + 2]);
                }
            }
        }
        if (GpuTrace.PRELOSS_TRACE.isOn() && nDistributedSlackBuses > 0 && !pairPhase) {
            // Did the DEVICE's own distributed-slack pass move power AFTER the host-side
            // pre-distribution? The CPU's loop should see only the loss-nonlinearity
            // residual and stay inside the slack band; if ours redistributes again, the
            // two land at different points inside that band.
            double[] ft = BatchReadbackNative.readBatchTargets(handle);
            for (int s = 0; s < harvestedCount; s++) {
                if (harvested[s].preLoss() == 0) {
                    continue;
                }
                int ci = harvested[s].index();
                double moved = 0;
                for (int i = 0; i < nDistributedSlackBuses; i++) {
                    int row = harvested[s].distributedSlackBusRow(i, gpuData.distributedSlack().busPhiRow());
                    // A participant this contingency ISLANDED has its row frozen to a flat
                    // identity, so (final - base) there is minus its whole anchor and not a
                    // distribution at all: counting it read the middle-section fixture's
                    // participants as -1.669969 pu off OLF when the real gap was -0.0687.
                    if (islandedDistributedSlackByIndex != null && islandedDistributedSlackByIndex[ci] != null
                            && ChunkArrays.contains(islandedDistributedSlackByIndex[ci], i)) {
                        continue;
                    }
                    moved += ft[s * n + row] - baseTargets[gpuData.distributedSlack().busPhiRow()[i]];
                }
                // preLossBySlot is the mismatch the pre-distribution was asked to absorb;
                // `moved` is where the participants ended up relative to the SAME anchor the
                // device DS loop resets to, so the delta is what that loop added on top.
                System.err.printf("PRELOSS_AFTER ctg=%s preLossMismatch=%.6f seededOnParticipants=%.6f "
                        + "deviceFinal=%.6f deltaFromDeviceDsLoop=%.6f pu%n",
                        contingencies.get(harvested[s].index()).contingencyId(),
                        harvested[s].preLoss(), harvested[s].preAbsorbed(), moved, moved - harvested[s].preAbsorbed());
            }
        }
        if (GpuBatchPhaseStats.ENABLED) {
            int itMax = 0;
            for (int s = 0; s < harvestedCount; s++) {
                itMax = Math.max(itMax, (int) chunkResults[s][n]);
            }
            GpuBatchPhaseStats.add("main.iterMax", itMax);
            GpuBatchPhaseStats.bump("main.solves");
        }
        if (GpuTrace.RL_TRACE.isOn() && reactiveLimitsFinalState != null) {
            // reactiveLimitsFinalState packs a GpuReactiveLimitsState per (controller, scenario):
            // switchCount > 0 means the DEVICE switched that controller during THIS solve, which
            // is exactly the decision OLF's ReactiveLimitsOuterLoop makes (and logs) on the CPU.
            for (int s = 0; s < harvestedCount; s++) {
                int switched = 0;
                int switchedToPq = 0;
                StringBuilder ids = new StringBuilder();
                for (int i = 0; i < nReactiveLimitEntries; i++) {
                    int v = reactiveLimitsFinalState[s * nReactiveLimitEntries + i];
                    if (GpuReactiveLimitsState.limitType(v) != GpuQLimitType.NONE) {
                        switchedToPq++;
                        if (ids.length() < 4000) {
                            // vRow identifies WHICH controller: it is the bus's stable device
                            // row, the same number the DISTR_Q / target dumps print.
                            ids.append(" p").append(i).append("(vRow=").append(gpuData.reactiveLimits().voltageRow()[i])
                               .append(",lt=").append(GpuReactiveLimitsState.limitType(v)).append(')');
                        }
                    }
                    if (GpuReactiveLimitsState.pvPqSwitchCount(v) > 0) {
                        switched++;
                        if (ids.length() < 300) {
                            ids.append(" rl").append(i).append("(vRow=")
                               .append(gpuData.reactiveLimits().voltageRow()[i]).append(",lt=").append(GpuReactiveLimitsState.limitType(v)).append(')');
                        }
                    }
                }
                System.err.printf("GPU_RL ctg=%s switchedThisSolve=%d switchedToPqTotal=%d%s%n",
                        contingencies.get(harvested[s].index()).contingencyId(), switched, switchedToPq, ids);
            }
        }
    }

    /**
     * What each harvested contingency leaves its operator strategies, as OLF starts a strategy from the contingency's
     * post-contingency network: its PV/PQ states and frozen target Q, its distributed-slack absorption and generator moves,
     * and the start of one that ran out of iterations.
     */
    @SuppressWarnings("java:S107")
    private static void captureForStrategies(RunView run, ChunkView chunk, long handle, double[][] chunkResults, SlotOccupant[] harvested,
                                             int harvestedCount, int[] reactiveLimitsFinalState, List<Integer> pairIdxList) {
        int n = run.n();
        GpuAcData gpuData = run.gpuData();
        double[] baseTargets = run.baseTargets();
        int nDistributedSlackBuses = run.nDistributedSlackBuses();
        int nDistributedSlackGenerators = run.nDistributedSlackGenerators();
        int nReactiveLimitEntries = run.nReactiveLimitEntries();
        boolean slackOnlyDistribution = run.slackOnlyDistribution();
        int[][] islandedDistributedSlackByIndex = run.islandedDistributedSlackByIndex();
        java.util.Map<Integer, int[]> reactiveLimitsStateByIndex = run.captures().reactiveLimitsStateByIndex;
        java.util.Map<Integer, double[]> reactiveLimitTargetQByIndex = run.captures().reactiveLimitTargetQByIndex;
        java.util.Map<Integer, double[]> distributedSlackTargetsByIndex = run.captures().distributedSlackTargetsByIndex;
        java.util.Map<Integer, double[]> genMovesByIndex = run.captures().genMovesByIndex;
        java.util.Map<Integer, double[]> exhaustedStartByIndex = run.captures().exhaustedStartByIndex;
        java.util.Map<Integer, Double> slackAbsorbedByIndex = run.captures().slackAbsorbedByIndex;
        boolean pairPhase = chunk.pairPhase();
        if (reactiveLimitsFinalState != null && !pairPhase) {      // capture each ctg's final PV/PQ states for its pairs
            // ...and the frozen target Q its PQ buses ENDED with, which its pairs restore (OLF restores the parent's
            // generation target Q; a limit that moved by less than the re-freeze tolerance kept the old one)
            double[] finalTargetQ = ReactiveLimitsNative.readBatchReactiveLimitTargetQ(handle);
            for (int s = 0; s < harvestedCount; s++) {
                if (finalTargetQ != null) {
                    reactiveLimitTargetQByIndex.put(harvested[s].index(), java.util.Arrays.copyOfRange(finalTargetQ, s * nReactiveLimitEntries, (s + 1) * nReactiveLimitEntries));
                }
                int[] slice = new int[nReactiveLimitEntries];
                System.arraycopy(reactiveLimitsFinalState, s * nReactiveLimitEntries, slice, 0, nReactiveLimitEntries);
                reactiveLimitsStateByIndex.put(harvested[s].index(), slice);
            }
        }
        if (!pairPhase && slackOnlyDistribution && !pairIdxList.isEmpty()) {
            // capture each ctg's slack-only-DS absorbed total for its pairs' seeding
            double[] absorbed = BatchReadbackNative.readBatchSlackAbsorbed(handle);
            if (absorbed != null) {
                for (int s = 0; s < harvestedCount; s++) {
                    slackAbsorbedByIndex.put(harvested[s].index(), absorbed[s]);
                }
            }
        }
        if (!pairPhase && !pairIdxList.isEmpty()) {
            // A contingency that ran out of Newton iterations on the device (OLF's MAX_ITERATION_REACHED):
            // OLF still runs its strategies, from what the failed run left - the voltages of its last
            // converged solve (NewtonRaphson updates the network only then), and the targets and bus
            // types its outer loops had moved, which the captures below read as they read a converged one.
            for (int s = 0; s < harvestedCount; s++) {
                if ((int) chunkResults[s][n + 2] == GpuAcSolverStatus.MAX_ITERATION_REACHED) {
                    exhaustedStartByIndex.put(harvested[s].index(),
                            BatchReadbackNative.readBatchLastConvergedState(handle, s));
                }
            }
        }
        if (!pairPhase && nDistributedSlackBuses > 0 && !pairIdxList.isEmpty()) {
            // Capture each ctg's DS ABSORPTION (final distributed target minus the ctg's own
            // effective base per DS row) plus its factor sum, for its pairs' re-anchoring —
            // the device analogue of OLF's setGeneratorsInitialTargetPToTargetP before a
            // strategy re-solve.
            double[] finalTargets = BatchReadbackNative.readBatchTargets(handle);
            for (int s = 0; s < harvestedCount; s++) {
                double[] slice = new double[nDistributedSlackBuses + 1];
                double fs = 0;
                int[] islandedDistributedSlackMask = islandedDistributedSlackByIndex != null ? islandedDistributedSlackByIndex[harvested[s].index()] : null;
                for (int i = 0; i < nDistributedSlackBuses; i++) {
                    if (islandedDistributedSlackMask != null && ChunkArrays.contains(islandedDistributedSlackMask, i)) {
                        // A participant this contingency islanded absorbed nothing. Its row is a FROZEN
                        // identity whose target is an angle: read as a P it handed the pairs minus the bus's
                        // whole base injection, composed onto the pair's frozen row (rte6515 LINE-4468-670:
                        // -8.728 rad held at the islanded VL-6297, 0.007 MW) or, by a reclose, onto its live one.
                        continue;
                    }
                    // read where this scenario solved the participant's balance: a coupler it opened
                    // brought a relocated balance home to the bus's own phi row, and the coupler's dummy
                    // row left behind holds a fixed stub (rte6515 BUS-1359_BBS_2: read there, -0.1926 pu
                    // of "absorption" that the pairs' first distribution took back, 19.27 MW)
                    int row = harvested[s].distributedSlackBusRow(i, gpuData.distributedSlack().busPhiRow());
                    double effBase = baseTargets[gpuData.distributedSlack().busPhiRow()[i]]
                            - (harvested[s].distributedSlackOverrideBus() == i ? harvested[s].distributedSlackOverrideBaseTargetPDelta() : 0.0);
                    slice[i] = finalTargets[s * n + row] - effBase;
                    // the factors the parent's distribution renormalized over: an islanded participant
                    // is out of them, a lost generator's share leaves its bus
                    fs += gpuData.distributedSlack().busFactor()[i]
                            - (harvested[s].distributedSlackOverrideBus() == i ? harvested[s].distributedSlackOverrideFactorDelta() : 0.0);
                }
                slice[nDistributedSlackBuses] = fs;
                distributedSlackTargetsByIndex.put(harvested[s].index(), slice);
            }
            if (nDistributedSlackGenerators > 0) {
                // ...and each GENERATOR's final move, which is what OLF's setGeneratorsInitialTargetPToTargetP
                // really keeps: the bus absorption above does not say how a bus's generators split it once
                // one of them saturates. A slack pass re-derives the whole distribution from the base
                // targets, so distributedSlackGeneratorMove holds it all when one fired; otherwise the pre-distribution alone moved.
                double[] distributedSlackGeneratorMove = BatchReadbackNative.readBatchDistributedSlackGeneratorMove(handle);
                double[] retain = BatchReadbackNative.readBatchSlackRetainedShare(handle);
                for (int s = 0; s < harvestedCount; s++) {
                    double[] moves = new double[nDistributedSlackGenerators];
                    if (distributedSlackGeneratorMove != null && retain != null && !Double.isNaN(retain[s])) {
                        System.arraycopy(distributedSlackGeneratorMove, s * nDistributedSlackGenerators, moves, 0, nDistributedSlackGenerators);
                    } else if (harvested[s].genPreMoves() != null) {
                        System.arraycopy(harvested[s].genPreMoves(), 0, moves, 0, nDistributedSlackGenerators);
                    }
                    genMovesByIndex.put(harvested[s].index(), moves);
                }
            }
        }
    }
}
