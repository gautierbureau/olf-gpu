/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.outerloop.gpu.ContinuousPhaseControl;
import com.powsybl.openloadflow.ac.outerloop.gpu.DeviceLoopKind;
import com.powsybl.openloadflow.ac.outerloop.gpu.DeviceLoopTables;
import com.powsybl.openloadflow.ac.outerloop.gpu.DevicePhaseInstaller;
import com.powsybl.openloadflow.ac.outerloop.gpu.DevicePhases;
import com.powsybl.openloadflow.ac.outerloop.gpu.DiscreteTransformerVoltageControl;
import com.powsybl.openloadflow.ac.outerloop.gpu.ForcedTargetQ;
import com.powsybl.openloadflow.ac.outerloop.gpu.HvdcAcEmulation;
import com.powsybl.openloadflow.ac.outerloop.gpu.ReactiveCapabilityCurves;
import com.powsybl.openloadflow.ac.outerloop.gpu.ReactiveLimitsNative;
import com.powsybl.openloadflow.ac.outerloop.gpu.SecondaryVoltageControlLoop;
import com.powsybl.openloadflow.ac.outerloop.gpu.VoltageMonitoring;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfShunt;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Drives the batched GPU solve of the qualified N-1 scenarios in device-memory-bounded chunks:
 * one cached native context (sized to the chunk, cuDSS analyzed once) reused across all chunks and
 * two phases (plain contingencies, then warm-continued (ctg, strategy) pairs). Assembles each
 * scenario's per-slot element mask, row-mode overrides (islanding freeze / PV->PQ flip / zi switch),
 * target/shunt/tap/distributed-slack overrides, runs the batched Newton, then the per-scenario
 * transformer-control orchestration, and writes results back by original contingency index.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuBatchChunkSolver {

    private static final boolean OL_STATS = System.getenv("OLF_OL_STATS") != null;

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuBatchChunkSolver.class);

    private GpuBatchChunkSolver() {
    }

    /**
     * Solve the qualified scenarios on the GPU in chunks sized by {@link GpuBatchContextCache#resolveChunkSize}
     * (auto from device memory, or a manual cap). ONE native context, sized to the chunk, is created and reused across all
     * chunks: cuDSS's symbolic ANALYSIS (on the shared pattern, identical for every chunk) is
     * done once on the first chunk and reused, so chunking costs nothing in amortization — it
     * only bounds the resident per-scenario LU factors. The last (partial) chunk is padded
     * with base-case scenarios (no element disabled): they converge at iteration 0 and their
     * results are discarded.
     */
    static void solveQualifiedInChunks(ChunkInputs inputs) {
        GpuAcData gpuData = inputs.base().gpuData();
        ChunkInputs in = inputs.withBaseState(ZeroImpedanceClusters.zeroZiFlows(gpuData, inputs.base().baseState()));
        List<Integer> batchedIndices = in.plan().batchedIndices();
        int n = in.base().n();
        double[][] resultsByIndex = in.sinks().resultsByIndex();
        int nb = batchedIndices.size();
        // Chunk sizing is a device-memory ESTIMATE (a cuDSS probe on the shared structure). When it
        // is too optimistic the factorization fails to allocate mid-solve, and a thrown batch means
        // the WHOLE run un-accelerates — so an allocation failure HALVES the chunk and retries
        // instead (measured on a 6.5k-bus RTE-like network: the probe recommended a chunk whose LU
        // factors did not fit, and 9042 contingencies all fell back to the CPU).
        int chunkSize = GpuBatchContextCache.resolveChunkSize(nb, n, gpuData);
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

    /** One row of the OLF_GPU_TARGET_ROW probe: the row's base target, its per-scenario row modes and
     *  every per-scenario override landing on it. */
    private static void emitTargetRow(int probe, int n, int realCount, double[] baseTargets, int[] baseRowMode,
                                      int[][] rowModePerScenario, int[] targetOverrideIndex,
                                      double[] targetOverrideValue) {
        StringBuilder sb = new StringBuilder(String.format(
                "TARGET_ROW row=%d base=%.9f baseMode=%d", probe, baseTargets[probe], baseRowMode[probe]));
        if (rowModePerScenario != null) {
            for (int s2 = 0; s2 < realCount; s2++) {
                if (rowModePerScenario[s2] != null && rowModePerScenario[s2][probe] != baseRowMode[probe]) {
                    sb.append(String.format(" mode[s=%d]=%d", s2, rowModePerScenario[s2][probe]));
                }
            }
        }
        for (int k2 = 0; k2 < targetOverrideIndex.length; k2++) {
            if (targetOverrideIndex[k2] % n == probe) {
                sb.append(String.format(" ov[s=%d]=%.9f",
                        targetOverrideIndex[k2] / n, targetOverrideValue[k2]));
            }
        }
        System.err.println(sb);
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
        GpuBatchElementMapper elementMapper = in.base().elementMapper();
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
        int[][] dqOvContribByIndex = plan.dqOvContribByIndex();
        double[][] dqOvWByIndex = plan.dqOvWByIndex();
        int[][] dqKeyOvMemberByIndex = plan.dqKeyOvMemberByIndex();
        double[][] dqKeyOvByIndex = plan.dqKeyOvByIndex();
        int[][] drOvContribByIndex = plan.drOvContribByIndex();
        double[][] drOvCByIndex = plan.drOvCByIndex();
        int[][] islandedDsByIndex = plan.islandedDsByIndex();
        int[][] islandedRlByIndex = plan.islandedRlByIndex();
        List<com.powsybl.openloadflow.network.LfHvdc>[] islandedHvdcsByIndex = plan.islandedHvdcsByIndex();
        int[][] dqSubByIndex = plan.dqSubByIndex();
        GpuBatchQualification.SlackRelocations slackRelocations = plan.slackRelocations();
        NewtonRaphsonParameters parameters = in.settings().parameters();
        double tol = in.settings().tol();
        double reactiveLimitTol = in.settings().reactiveLimitTol();
        double slackDistTol = in.settings().slackDistTol();
        double convEpsPerEq = in.settings().convEpsPerEq();
        double[][] rowCriteria = in.settings().rowCriteria();
        DevicePhases.Built phases = in.phases().phases();
        List<HvdcAcEmulation.Ctl> hvdcControllers = in.phases().hvdcControllers();
        ReactiveCapabilityCurves.Channel rlCurves = in.phases().rlCurves();
        ForcedTargetQ.Channel forcedTargetQ = in.phases().forcedTargetQ();
        double[][] resultsByIndex = in.sinks().resultsByIndex();
        double[] preDistributedByIndex = in.sinks().preDistributedByIndex();
        double[] distributedByIndex = in.sinks().distributedByIndex();
        BatchConfig config = in.config();
        int nb = batchedIndices.size();
        int nChunks = (nb + chunkCap - 1) / chunkCap;
        // BALANCE the chunks over that count instead of filling each to the cap and leaving a ragged
        // tail. The device context is allocated at the chunk size and every solve is that wide, so a
        // tail chunk holding 72 of 642 slots still solves a 642-wide batch — 570 of them padding.
        // Splitting 1998 scenarios as 4 x 500 rather than 3 x 642 + 72 measured 13.7 s vs 15.6 s of
        // batched time on rte6515, at LOWER device memory (the size only ever goes down from the cap).
        int chunkSize = (nb + nChunks - 1) / nChunks;

        // Reuse a cached analyzed context when the pattern/admittances/chunk/mode match (across
        // SA runs on the same network) — the context is NOT destroyed here, it stays cached.
        // MODE: the run's configured batch mode, or the ROBUST single-matrix solve on the retry pass
        // (config.robustSolveMode(), set only by BatchConfig.withRobustSolveMode) — mode 6 is mode 0's
        // per-scenario loop with matching and global column pivoting, which is what rte6515's full
        // busbar structure needs to factor at all. Switching mode EVICTS the single-entry context cache,
        // so the retry pays one cuDSS analysis; that is the price of keeping the scenarios on device.
        long handle = GpuBatchContextCache.acquireBatchContext(gpuData, chunkSize,
                config.robustSolveMode() ? GpuBatchContextCache.MODE_ROBUST : GpuBatchContextCache.resolveBatchMode(), n);
        // PER_EQUATION_TYPE criterion: install the per-row thresholds on the (possibly cached)
        // context — or CLEAR a previous run's thresholds under UNIFORM_CRITERIA.
        GpuAcNewtonSolver.installRowCriteria(handle, rowCriteria);
        // LINE_SEARCH: OLF's backtracking budget and step fold, on the (possibly cached) context; read only in lsMode 1.
        GpuAcNewtonSolver.setBatchLineSearch(handle, parameters.getLineSearchStateVectorScalingMaxIteration(),
                parameters.getLineSearchStateVectorScalingStepFold());
        // The run-constant outer-loop tables, shared with the base load flow's batch-of-one.
        DeviceLoopTables.install(handle, gpuData, rlCurves, forcedTargetQ,
                config.rlRobustMode(), config.minRealisticVoltage(), config.maxRealisticVoltage(),
                config.minNominalVoltageRealisticVoltageCheck(), config.realisticFixingLoop());

        // Which reactive-capability CURVE entries hang off each bus P row. Run-constant, and the
        // index a generator outage needs to find the entries it disturbs at its own bus.
        java.util.Map<Integer, int[]> curveEntriesByPRow = new java.util.HashMap<>();
        if (rlCurves != null) {
            java.util.Map<Integer, List<Integer>> acc = new java.util.HashMap<>();
            for (int e = 0; e < rlCurves.cvGenId().length; e++) {
                acc.computeIfAbsent(rlCurves.cvIdx()[e * 5 + 1], k -> new ArrayList<>()).add(e);
            }
            acc.forEach((row, es) -> curveEntriesByPRow.put(row, es.stream().mapToInt(Integer::intValue).toArray()));
        }

        // A control loop the kernel REFUSES must never fall back to an after-the-sweep host orchestrator.
        // Such an orchestrator runs out of OLF's order — after the loops OLF puts at orders 7 to 10 — and
        // two of them in a chain discard each other's target moves (measured 2.4e-1 kV). Un-accelerated is
        // the only honest alternative to wrong, so the whole group goes to the CPU: leaving resultsByIndex
        // null is exactly how the caller routes a contingency to the stock per-contingency path.
        DevicePhaseInstaller.InstalledPhases installed = DevicePhaseInstaller.install(handle, new DevicePhaseInstaller.Request(
                phases.incrementalTvc(), config.incTvcMaxTapShift(), phases.flow(), phases.continuousPhase(),
                phases.incrementalShunt(), config.incShuntMaxSectionShift(),
                config.transformerControl() || config.simpleTransformerControl() ? phases.discreteTvc() : null,
                config.simpleTransformerControl(), config.transformerControl(), phases.freeze(),
                config.simpleShuntControl() ? phases.simpleShunt() : null, false));
        if (!installed.refused().isEmpty()) {
            LOGGER.warn("GPU batched N-1 disabled for this run: the device cannot own {} — every contingency "
                    + "runs on the CPU rather than through a host orchestrator out of OLF's loop order", installed.refused());
            return;
        }

        RunScratch rs = new RunScratch(chunkSize, nElements, gpuData, elementMapper, baseRowMode);
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
        java.util.Map<Integer, double[]> rlPinByIndex = new java.util.HashMap<>();   // ctg index -> its final pinned targets
        // ctg index -> its FINAL distributed-slack targets at the DS participant rows (the device DS mutates
        // targets in place; read back after each ctg chunk). The pair phase re-anchors its distribution here,
        // mirroring OLF's setGeneratorsInitialTargetPToTargetP before the strategy re-solve.
        int nds = gpuData.dsPhiRow().length;
        java.util.Map<Integer, double[]> dsTargetsByIndex = new java.util.HashMap<>();
        // ctg index -> each participating generator's final move from its base target (the device's genCum, or the
        // loss pre-distribution alone when no slack pass fired): its pairs' per-generator anchors
        int ngen = gpuData.dgBus() == null ? 0 : gpuData.dgBus().length;
        java.util.Map<Integer, double[]> genMovesByIndex = new java.util.HashMap<>();
        // ctg index -> the state its pairs start from when it ran out of Newton iterations (see the capture)
        java.util.Map<Integer, double[]> exhaustedStartByIndex = new java.util.HashMap<>();
        java.util.Map<Integer, Double> slackAbsorbedByIndex = new java.util.HashMap<>();   // slack-only DS parents
        boolean dsSlackOnly = gpuData.dsPhiRow().length == 0 && gpuData.dsSlackPhiRow() >= 0;
        // Host-side base RL seed (mirrors initRlLimitTypeBatch) for PADDING slots of pair chunks.
        int[] baseRlSeed = new int[nrl];
        for (int i = 0; i < nrl; i++) {
            int vrow = gpuData.rlVrow()[i];
            // remote entries (plan-B3): base mode 1 = UNPINNED DISTR_Q share; only mode 4 is a pin.
            // The pilot's V equals the group target at the base (degenerate) — key the direction on
            // WHICH limit the extracted BUS_TARGET_Q target equals instead.
            boolean remote = gpuData.rlRemote().length > i && gpuData.rlRemote()[i] != 0;
            byte olfKind = gpuData.rlBaseQLimitType().length > i ? gpuData.rlBaseQLimitType()[i] : 0;
            if (remote) {
                double tgt = baseTargets[vrow];
                baseRlSeed[i] = baseRowMode[vrow] == 4
                        ? (olfKind != 0 ? olfKind
                           : (Math.abs(tgt - (gpuData.rlMaxQ()[i] - gpuData.rlLoadQ()[i]))
                              <= Math.abs(tgt - (gpuData.rlMinQ()[i] - gpuData.rlLoadQ()[i])) ? 2 : 1))
                        : 0;
            } else {
                baseRlSeed[i] = baseRowMode[vrow] == 1
                        ? (olfKind != 0 ? olfKind : (baseState[vrow] < gpuData.rlVtarget()[i] ? 2 : 1)) : 0;
            }
        }
        try {
            long startTime = System.currentTimeMillis();
            // PHASE TIMING (OLF_GPU_BATCH_TRACE): the batched main solve vs the per-scenario host-side
            // outer-loop orchestrators — on a control-heavy network the orchestrators, not the solve,
            // are where the wall time goes, and that is invisible without splitting the two.
            long tMainSolveNs = 0;
            long tOrchestratorNs = 0;
            java.util.Map<String, Long> orchNs = new java.util.LinkedHashMap<>();
            // The zero-impedance elements OUT of the base tree, once: a scenario with sticky non-tree couplers puts
            // every one of them back except its still-out set, and scanning all ziBranchNums per scenario with a
            // boxed set was 1.4 s of host time on network.xiidm (2653 elements x every such scenario).
            int[] baseNonTreeZi;
            {
                int[] zIdx = gpuData.ziIdx();
                int nz = gpuData.ziBranchNums() == null ? 0 : gpuData.ziBranchNums().length;
                int cnt = 0;
                for (int zb = 0; zb < nz; zb++) {
                    if (zIdx[zb * 11 + 10] == 0) {
                        cnt++;
                    }
                }
                baseNonTreeZi = new int[cnt];
                cnt = 0;
                for (int zb = 0; zb < nz; zb++) {
                    if (zIdx[zb * 11 + 10] == 0) {
                        baseNonTreeZi[cnt++] = zb;
                    }
                }
            }
            for (List<Integer> group : List.of(ctgIdxList, pairIdxList)) {
                boolean pairPhase = group == pairIdxList;
                int nbg = group.size();
                for (int chunkStart = 0; chunkStart < nbg; chunkStart += chunkSize) {
                    int realCount = Math.min(chunkSize, nbg - chunkStart);
                    rs.overrides.clear();
                    // fixTransformerVoltageControls is a POST-CONTINGENCY check on the CPU, so its
                    // connectivity masks are per scenario and belong to the chunk, not to the run.
                    if (installed.discreteTvc() && config.transformerControl() && phases.freeze() != null && phases.freeze().hasFix()) {
                        List<BatchContingency> slotCtgs = new ArrayList<>(realCount);
                        for (int s = 0; s < realCount; s++) {
                            slotCtgs.add(contingencies.get(group.get(chunkStart + s)));
                        }
                        DiscreteTransformerVoltageControl.uploadFixMasks(handle, phases.freeze(), slotCtgs, chunkSize);
                    }
                    ChunkLists cl = new ChunkLists(chunkSize, n, gpuData, forcedTargetQ, ngen);
                    SlotSeeding seeding = new SlotSeeding(chunkSize, pairPhase, n, gpuData, elementMapper, contingencies,
                            plan, baseState, baseTargets, baseRowMode, hvdcElemIndex, baseNonTreeZi, nds, rlCurves,
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
                            GenOutageSeed.seed(seeding, s, ci, slotGos, pilotBalanceRows);
                        }
                        // else: padding scenario = base case (no element disabled), discarded below
                    }
                    double[] dsAnchorShift = null;
                    double[] dgAnchor = null;                // pairs: each participating generator's anchor (s * ngen + g)
                    double[] dgSeedAnchor = null;            // ...and the same without the strategy's GeneratorActions
                    double[] dsSlackShift = null;
                    double[] pairSlackAbsorbedSeed = null;   // slack-only pairs: the parent's absorbed total they start from
                    boolean[] rejectAfterSolve = null;       // pairs whose parent the device did not answer: CPU
                    boolean[] pairParentConverged = new boolean[chunkSize];
                    double[] rlPinIn = null;                 // pairs: each parent's final pinned targets (NaN = none)
                    double[] preSlackSeed = null;            // the loss pre-distribution's share on the slack generators,
                                                             // when nobody else is left to participate
                    if (!pairPhase && (rs.hasDs || dsSlackOnly) && config.contingencyLossPreDistribution()) {
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
                            if (!rs.hasDs) {
                                // the slack generators take what their rooms allow, as OLF's water-fill places it;
                                // the rest stays for the DistributedSlack check, where the failure behaviour decides
                                double placed = Math.min(Math.max(cl.preLossBySlot[s], gpuData.dsSlackRoomDown()),
                                        gpuData.dsSlackRoomUp());
                                preDistributedByIndex[preIdx] = placed;
                                // SLACK-ONLY (the slack bus is the only participant): OLF distributes
                                // the whole scheduled loss into the slack generator's own target, so
                                // its target for this scenario is base + loss. The device has no
                                // BUS_TARGET_P row at the slack — its P is free — so dsSlackTargetP is
                                // a RUN CONSTANT and that displacement has nowhere to live except
                                // dDsSlackAbsorbed, which dsSlackShift seeds.
                                //
                                // Without this the whole block was skipped (the gate was hasDs), the
                                // accumulator started at 0, and slackOnlyDsCheck measured the mismatch
                                // against the BASE slack target: on the deviceHvdc* networks that is
                                // -3.006957565 pu where OLF sees -0.006957565, so the check fired on
                                // the first pass and absorbed the lot. OLF reports STABLE there (0.0070
                                // < the 0.01 band) and fires only after the HVDC unfreeze, on -0.013703031
                                // — which our accumulator, 3.0 too low, then read as -0.006745466 and
                                // let pass. Same single distribution, wrong pass, and the two arms end
                                // their Newton sequences on different calls.
                                if (dsSlackShift == null) {
                                    dsSlackShift = new double[chunkSize];
                                }
                                dsSlackShift[s] = placed;
                                if (System.getenv("OLF_GPU_PRELOSS_TRACE") != null) {
                                    System.err.printf("PRELOSS_SLACKONLY ctg=%s loss=%s slackBase=%s -> slackTarget=%s%n",
                                            contingencies.get(group.get(chunkStart + s)).contingencyId(),
                                            Double.toString(cl.preLossBySlot[s]),
                                            Double.toString(gpuData.dsSlackTargetP()),
                                            Double.toString(gpuData.dsSlackTargetP() + placed));
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
                            boolean[] islandedDs = null;
                            // The PLAN's mask first (islandedDsParticipants): it matches a participant on its BUS
                            // as well as on its row, which the row test below cannot do for a participant whose P
                            // row the coupler-tree relocation moved onto its parent coupler's DUMMY_P — the frozen
                            // rows are the bus's own BUS_PHI/BUS_V. Rebuilding the mask here from the row test
                            // alone repeated exactly the defect that comment describes: measured on the
                            // middle-section fixture, the pre-loss gave the ISLANDED generator's relocated row a
                            // share of -0.894 pu, and the device's slack loop then chased its own tail for five
                            // passes and retired 0.837 pu away from OLF.
                            if (islandedDsByIndex != null && islandedDsByIndex[ci] != null) {
                                for (int participant : islandedDsByIndex[ci]) {
                                    if (participant >= 0 && participant < nds) {
                                        if (islandedDs == null) {
                                            islandedDs = new boolean[nds];
                                        }
                                        islandedDs[participant] = true;
                                    }
                                }
                            }
                            for (int i = 0; i < nds; i++) {
                                for (int fr : frozen) {
                                    if (fr == gpuData.dsPhiRow()[i]) {
                                        if (islandedDs == null) {
                                            islandedDs = new boolean[nds];
                                        }
                                        islandedDs[i] = true;
                                    }
                                }
                            }
                            double[] shares = GpuBatchLossPreDistribution.distribute(gpuData, cl.preLossBySlot[s],
                                    rs.dsOvBus.length > 0 ? rs.dsOvBus[s] : -1,
                                    rs.dsOvFactorDelta.length > 0 ? rs.dsOvFactorDelta[s] : 0,
                                    rs.dsOvBaseDelta.length > 0 ? rs.dsOvBaseDelta[s] : 0,
                                    rs.dsOvMaxDelta.length > 0 ? rs.dsOvMaxDelta[s] : 0,
                                    rs.dsOvMinDelta.length > 0 ? rs.dsOvMinDelta[s] : 0, islandedDs,
                                    ChunkArrays.lostSlotsOf(cl.dsLostGenList, s),
                                    cl.genPreMovesBySlot != null ? (cl.genPreMovesBySlot[s] = new double[gpuData.dgBus().length]) : null);
                            // OLF_GPU_SEED_DUMP=<file>: the per-bus SHARE the host-side loss
                            // pre-distribution seeds, against OLF's own pre-distribution section in
                            // OLF_DS_SPLIT. The two totals can agree while the SPLIT does not, and the
                            // split is what the first solve sees — so the mismatch the DS loop then
                            // measures, and everything after it, inherits the difference.
                            if (System.getenv("OLF_GPU_SEED_DUMP") != null && shares != null) {
                                try (java.io.PrintWriter w = new java.io.PrintWriter(
                                        new java.io.FileWriter(System.getenv("OLF_GPU_SEED_DUMP"), true))) {
                                    w.printf("# SEED ctg=%s%n",
                                            contingencies.get(group.get(chunkStart + s)).contingencyId());
                                    for (int i = 0; i < nds; i++) {
                                        if (shares[i] != 0) {
                                            w.printf("%d %.12f%n", i, shares[i]);
                                        }
                                    }
                                } catch (java.io.IOException e) {
                                    throw new java.io.UncheckedIOException(e);
                                }
                            }
                            if (System.getenv("OLF_GPU_PRELOSS_TRACE") != null) {
                                double absorbed = 0;
                                if (shares != null) {
                                    for (int i = 0; i < nds; i++) {
                                        absorbed += shares[i];
                                    }
                                }
                                double fsAll = 0;
                                double fsIslanded = 0;
                                for (int i = 0; i < nds; i++) {
                                    fsAll += gpuData.dsFactor()[i];
                                    if (islandedDs != null && islandedDs[i]) {
                                        fsIslanded += gpuData.dsFactor()[i];
                                    }
                                }
                                System.err.printf("ISLANDED_DS ctg=%s count=%d%n",
                                        contingencies.get(ci).contingencyId(), ChunkArrays.countTrue(islandedDs));
                                System.err.printf("PRELOSS_FACTORS ctg=%s factorSumAll=%.6f factorSumIslanded=%.6f "
                                        + "(%.2f%% of the participation is in the island)%n",
                                        contingencies.get(ci).contingencyId(), fsAll, fsIslanded,
                                        fsAll > 0 ? 100.0 * fsIslanded / fsAll : 0.0);
                                System.err.printf("PRELOSS ctg=%s mismatch=%.6f pu absorbed=%.6f pu slackShare=%.6f "
                                        + "islandedDsParticipants=%d%n",
                                        contingencies.get(ci).contingencyId(), cl.preLossBySlot[s], absorbed,
                                        shares == null ? 0.0 : shares[nds],
                                        ChunkArrays.countTrue(islandedDs));
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
                            if (preDistributed == shares[nds] && shares[nds] != 0
                                    && !GpuBatchLossPreDistribution.slackInPack(gpuData)) {
                                // Only the slack bus's generators participate: OLF moved their target by the whole
                                // loss, and its slack check measures against that target. The device's slack-only
                                // pass (distributeSlackBatch) measures against slackTargetP + dsSlackShift.
                                if (dsSlackShift == null) {
                                    dsSlackShift = new double[chunkSize];
                                }
                                if (preSlackSeed == null) {
                                    preSlackSeed = new double[chunkSize];
                                }
                                dsSlackShift[s] = shares[nds];
                                preSlackSeed[s] = shares[nds];
                            }
                            for (int i = 0; i < nds; i++) {
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
                            for (int i = 0; i < nds; i++) {
                                if (shares[i] == 0) {
                                    continue;
                                }
                                // The row this scenario actually solves the bus's P balance on: the
                                // coupler-tree relocation's dummy row unless this scenario opened that
                                // coupler, in which case the balance — and so the share — is on the bus's
                                // own phi row. The ANCHOR stays the participant's own dsBaseTargetP, which
                                // the device carried to the new row with the equation.
                                int dsRow = cl.dsRowOvBySlot[s] != null && cl.dsRowOvBySlot[s][i] >= 0
                                        ? cl.dsRowOvBySlot[s][i] : gpuData.dsPhiRow()[i];
                                int flat = s * n + dsRow;
                                // compose with a gen-outage override on the same row
                                rs.overrides.compose(flat, baseTargets[gpuData.dsPhiRow()[i]], shares[i]);
                            }
                        }
                    }
                    // CONTINUOUS phase control: OLF's initialize() enables the CONTROLLER-mode controls
                    // BEFORE the first solve. This is the loop's initialize(), not an orchestrator — the
                    // in-kernel phase expects the activation to already carry it.
                    if (phases.continuousPhase() != null && !phases.continuousPhase().isEmpty()) {
                        ContinuousPhaseControl.applyPhaseOn(realCount, n, phases.continuousPhase(),
                                rs.rowModePerScenario, baseRowMode, rs.disabledMask, rs.overrides);
                    }
                    int[] targetOverrideIndex = rs.overrides.indexArray();
                    double[] targetOverrideValue = rs.overrides.valueArray();
                    if (rs.overrides.conflictingAddCount() > 0) {
                        // Two producers disagreed on one row's target. GpuBatchOverrides resolved it to the
                        // later one (no row reaches the device twice), but a sound plan has NO such row:
                        // every one so far was a write that should not exist (a DISTR_Q seed on a mode-0/4
                        // row). Name it; OLF_GPU_OV_DUP_TRACE=1 lists the rows.
                        LOGGER.warn("GPU batched SA: {} conflicting target overrides in chunk at {} (resolved to the later"
                                + " writer; OLF_GPU_OV_DUP_TRACE=1 lists them)", rs.overrides.conflictingAddCount(), chunkStart);
                        com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("HOST:OVERRIDE_CONFLICT");
                    }
                    if (GpuBatchOverrides.dupTrace()) {
                        // The invariant GpuBatchOverrides states in its javadoc, measured on the entries
                        // actually handed to the device: a row with two entries is written by two threads
                        // of applyTargetOverrideBatch in an unspecified order. Same value = benign;
                        // different values = a target chosen by a race.
                        System.err.printf("OV_DUP chunk entries=%d dupRows=%d conflictRows=%d%n",
                                targetOverrideIndex.length, rs.overrides.duplicateRowCount(),
                                rs.overrides.conflictingRowCount());
                        java.util.List<String> sites = rs.overrides.conflictSites();
                        for (int ci2 = 0; ci2 < rs.overrides.conflictSamples().size(); ci2++) {
                            double[] c = rs.overrides.conflictSamples().get(ci2);
                            int flat = (int) c[0];
                            int slot = flat / n;
                            String id = slot < realCount
                                    ? contingencies.get(group.get(chunkStart + slot)).contingencyId() : "<padding>";
                            int row2 = flat % n;
                            // The row MODE decides who is right: updateDistrQTargetsBatch refuses to
                            // maintain a DISTR_Q target on a mode-0 identity or a mode-4 own-Q row, so a
                            // seeded DISTR_Q target on such a row is the write that should not exist.
                            int mode2 = rs.rowModePerScenario != null && slot < realCount
                                    && rs.rowModePerScenario[slot] != null
                                    ? rs.rowModePerScenario[slot][row2] : baseRowMode[row2];
                            System.err.printf("OV_CONFLICT ctg=%s row=%d mode=%d baseMode=%d"
                                            + " shadowed=%.9f new=%.9f by=%s%n",
                                    id, row2, mode2, baseRowMode[row2], c[1], c[2],
                                    ci2 < sites.size() ? sites.get(ci2) : "?");
                        }
                    }
                    // OLF_GPU_TARGET_ROW=<row>: the target the DEVICE actually solves that row against —
                    // the base value and every per-scenario override landing on it. A row whose
                    // converged residual equals minus its OLF target is a row solved against 0.
                    String tgtRow = System.getenv("OLF_GPU_TARGET_ROW");
                    for (String probeSpec : tgtRow == null ? new String[0] : tgtRow.split(",")) {
                        // A LIST (or a lo-hi range): finding which device row carries a given equation
                        // means scanning them, and one row per JVM run made that a five-minute question.
                        int first = Integer.parseInt(probeSpec.trim().split("-")[0]);
                        int last = probeSpec.contains("-")
                                ? Integer.parseInt(probeSpec.trim().split("-")[1]) : first;
                        for (int probe = first; probe <= last && probe < n; probe++) {
                            emitTargetRow(probe, n, realCount, baseTargets, baseRowMode, rs.rowModePerScenario,
                                    targetOverrideIndex, targetOverrideValue);
                        }
                    }
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
                        System.err.println("GPU-PAIR-SHOV slotElem=" + rs.shOvSlotElemList + " gb=" + rs.shOvGbList
                                + " shPackOffset=" + rs.shPackOffset);
                    }
                    // Pair phase: warm continuation from each parent contingency's state + RL pinning.
                    double[] x0Full = null;
                    int[] rlStateIn = null;
                    int[] x0SeedIdx = null;
                    double[] x0SeedVal = null;
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
                    //  - dsAnchorShift: participant water-fill anchors moved to the parent's final DS targets;
                    //  - dsSlackShift: the slack bus's own absorbed share of the parent's distribution
                    //    ((1-fs)/fs x the participants' absorbed sum), so the mismatch/band is measured against
                    //    the parent's slack target exactly as the CPU leg measures it;
                    //  - DS-row target seeds at the parent values, so the first inner solve runs at the CPU
                    //    leg's starting operating point (losses are nonlinear along the path).
                    if (pairPhase) {
                        x0Full = new double[chunkSize * n];
                        rlStateIn = nrl > 0 ? new int[chunkSize * nrl] : null;
                        if (nds > 0) {
                            dsAnchorShift = new double[chunkSize * nds];
                            dsSlackShift = new double[chunkSize];
                            dgAnchor = ngen > 0 ? new double[chunkSize * ngen] : null;
                            dgSeedAnchor = ngen > 0 ? new double[chunkSize * ngen] : null;
                        } else if (dsSlackOnly) {
                            dsSlackShift = new double[chunkSize];    // absorbed-total seeds (no participant anchors)
                            pairSlackAbsorbedSeed = new double[chunkSize];
                        }
                        double dsFactorSum = 0;
                        for (double fv : gpuData.dsFactor()) {
                            dsFactorSum += fv;
                        }
                        for (int s = 0; s < chunkSize; s++) {
                            double[] parent = null;
                            int[] parentRl = null;
                            double[] parentDs = null;
                            Double parentSlackAbs = null;
                            if (s < realCount) {
                                String pairId = contingencies.get(group.get(chunkStart + s)).contingencyId();
                                Integer pIdx = indexById.get(pairId.substring(0, pairId.indexOf(GpuBatchStatus.STRATEGY_KEY_SEP)));
                                int parentStatus = pIdx != null && resultsByIndex[pIdx] != null
                                        ? ChunkArrays.rowStatus(resultsByIndex[pIdx], n) : GpuBatchStatus.STATUS_REJECTED;
                                if (parentStatus == GpuBatchStatus.STATUS_CONVERGED) {
                                    parent = ChunkArrays.fullRow(resultsByIndex[pIdx], n, config.stateSink(),
                                            contingencies.get(pIdx).contingencyId());
                                    pairParentConverged[s] = true;
                                } else if (parentStatus == GpuBatchStatus.STATUS_MAX_ITERATION) {
                                    parent = exhaustedStartByIndex.get(pIdx);   // OLF's network after the failed run
                                }
                                if (parent != null) {
                                    parentRl = rlByIndex.get(pIdx);
                                    parentDs = dsTargetsByIndex.get(pIdx);
                                    parentSlackAbs = slackAbsorbedByIndex.get(pIdx);
                                } else {
                                    // The device has no answer for the parent (the CPU replays it): OLF starts the strategy
                                    // from the state that replay leaves, which only the CPU has. CPU.
                                    if (rejectAfterSolve == null) {
                                        rejectAfterSolve = new boolean[chunkSize];
                                    }
                                    rejectAfterSolve[s] = true;
                                    if (s < realCount) {
                                        com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PAIR:PARENT_NOT_ON_DEVICE");
                                    }
                                }
                            }
                            for (int i = 0; i < n; i++) {
                                x0Full[s * n + i] = parent != null ? parent[i] : baseState[i];
                            }
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
                            if (rlStateIn != null) {
                                // the parent's pins, not its switch counts: OLF runs the strategy as a new load flow,
                                // whose reactive-limit loop counts PV -> PQ switches afresh
                                for (int i = 0; i < nrl; i++) {
                                    rlStateIn[s * nrl + i] = (parentRl != null ? parentRl : baseRlSeed)[i] & 7;
                                }
                                double[] pins = parentRl != null ? rlPinByIndex.get(ChunkArrays.pIdxOf(contingencies, group, chunkStart, s, indexById, realCount)) : null;
                                if (pins != null) {
                                    if (rlPinIn == null) {
                                        rlPinIn = new double[chunkSize * nrl];
                                        java.util.Arrays.fill(rlPinIn, Double.NaN);
                                    }
                                    System.arraycopy(pins, 0, rlPinIn, s * nrl, nrl);
                                }
                            }
                            if (dsSlackOnly && dsSlackShift != null && parentSlackAbs != null) {
                                dsSlackShift[s] = parentSlackAbs;    // seed the pair with the parent's absorbed total
                                pairSlackAbsorbedSeed[s] = parentSlackAbs;
                            }
                            if (parentDs != null && dsAnchorShift != null && dsFactorSum > 0) {
                                // parentDs = [absorption per DS row..., parent factor sum] (captured after the
                                // parent's solve). The anchor shift is the parent's PURE ABSORPTION — the part
                                // of the parent's final DS target that its distribution added on top of the
                                // parent's own effective base — which composes with the pair's own dsOvBaseDelta
                                // in the kernel (base_eff = base - ovDelta + shift), so gen-outage parents and
                                // action deltas are both handled without double counting.
                                double absorbedSum = 0;
                                // An absolute GeneratorAction on a participant sets its target (and OLF's initial
                                // target) to the value: its bus is not re-anchored on the parent's share, the
                                // action's own delta (base-referenced, on the override bus) is the whole move.
                                // The precomputer only batches it when that generator is its bus's only participant.
                                int absoluteBus = -1;
                                if (rs.dsOvBus.length > 0 && rs.dsOvBus[s] >= 0) {
                                    for (com.powsybl.openloadflow.network.LfGenerator g : contingencies.get(group.get(chunkStart + s)).absoluteTargetGenerators()) {
                                        if (config.dsParams().factor(g) != 0) {
                                            absoluteBus = rs.dsOvBus[s];
                                        }
                                    }
                                }
                                for (int i = 0; i < nds; i++) {
                                    int row = gpuData.dsPhiRow()[i];
                                    double shift = i == absoluteBus ? 0.0 : parentDs[i];
                                    dsAnchorShift[s * nds + i] = shift;
                                    absorbedSum += parentDs[i];      // the slack's share of the parent counts it all
                                    // Seed the DS-row target at the CPU strategy leg's starting value — the
                                    // parent-absorbed target plus the pair's own composed delta — so the first
                                    // inner solve runs at the CPU leg's starting operating point (losses are
                                    // nonlinear along the path). A pair override on the row (gen action /
                                    // gen-outage pTarget) is base-referenced: add the parent absorption to it. On the row
                                    // this pair solves the balance on (its own coupler openings), from the participant's
                                    // own base target, as the loss pre-distribution composes its shares.
                                    int pairRow = cl.dsRowOvBySlot[s] != null && cl.dsRowOvBySlot[s][i] >= 0 ? cl.dsRowOvBySlot[s][i] : row;
                                    rs.overrides.compose(s * n + pairRow, baseTargets[row], shift);
                                }
                                // the slack's own absorption in the parent's distribution: its own factor's share
                                // against the participants that distribution renormalized over (the kernel's retain),
                                // not one minus their sum - a lost or islanded participant's factor is not the
                                // slack's (pilotSelfCurveNetwork losing g3, the slack out of the distribution:
                                // 0.2234 pu of slack target where OLF has none, 2.1 MW distributed against 20.2)
                                double parentFs = parentDs[nds];
                                dsSlackShift[s] = parentFs > 0 ? (1.0 - dsFactorSum) / parentFs * absorbedSum : 0.0;
                            }
                            if (dgAnchor != null && s < realCount) {
                                // Each generator's own anchor: where the parent's distribution left it, and for a
                                // GeneratorAction the action's value (LfGeneratorAction sets the target AND the initial
                                // target): the parent's target plus the change, or the value itself.
                                String pairId = contingencies.get(group.get(chunkStart + s)).contingencyId();
                                Integer pIdx = indexById.get(pairId.substring(0, pairId.indexOf(GpuBatchStatus.STRATEGY_KEY_SEP)));
                                double[] parentMoves = parentDs != null && pIdx != null ? genMovesByIndex.get(pIdx) : null;
                                if (parentMoves != null) {
                                    System.arraycopy(parentMoves, 0, dgAnchor, s * ngen, ngen);
                                    System.arraycopy(parentMoves, 0, dgSeedAnchor, s * ngen, ngen);
                                    if (dsSlackShift != null && GpuBatchLossPreDistribution.slackInPack(gpuData)) {
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
                                        for (int g = 0; g < ngen; g++) {
                                            if (gpuData.dgBus()[g] < 0 && !absolute.contains(gpuData.dgGenId()[g])) {
                                                slackMoves += parentMoves[g];
                                            }
                                        }
                                        dsSlackShift[s] = slackMoves;
                                    }
                                }
                                // OLF re-checks each moved generator's participation at its new target
                                // (reApplyActivePowerControlChecks); the last action on a generator decides
                                java.util.Map<Integer, Boolean> staysBySlot = new java.util.TreeMap<>();
                                for (BatchContingency.GeneratorMove m : contingencies.get(group.get(chunkStart + s)).generatorActions()) {
                                    int g = ChunkArrays.dgSlotOf(gpuData, m.generator().getId());
                                    if (g >= 0) {
                                        // its target before the action: its anchor plus where the parent left it
                                        double before = config.dsParams().anchorTargetP(m.generator()) + dgAnchor[s * ngen + g];
                                        staysBySlot.put(g, m.participatesAt() == null || m.participatesAt().test(m.newTargetP(before)));
                                        dgAnchor[s * ngen + g] = m.absolute() ? m.deltaP() : dgAnchor[s * ngen + g] + m.deltaP();
                                    }
                                }
                                for (var e : staysBySlot.entrySet()) {
                                    if (!e.getValue()) {             // out of the distribution: it keeps the action's target
                                        cl.dsOutList.add(s);
                                        cl.dsOutList.add(e.getKey());
                                    }
                                }
                            }
                        }
                        targetOverrideIndex = rs.overrides.indexArray();
                        targetOverrideValue = rs.overrides.valueArray();
                        // OLF_GPU_TARGET_ROW once more, AFTER the pair's re-anchoring composed onto the overrides: the
                        // print above runs before it, and showed a frozen row's clean 0 where the device got -8.728
                        for (String probeSpec : tgtRow == null ? new String[0] : tgtRow.split(",")) {
                            int probe = Integer.parseInt(probeSpec.trim().split("-")[0]);
                            System.err.print("PAIR-COMPOSED ");
                            emitTargetRow(probe, n, realCount, baseTargets, baseRowMode, rs.rowModePerScenario,
                                    targetOverrideIndex, targetOverrideValue);
                        }
                    }
                    // HVDC AC-emulation: snapshot the pre-freeze override lists (the round rebuild base),
                    // then OLF's FreezingHvdcACEmulation freezes every live hvdc at its scenario's
                    // WARM-START flows BEFORE the first solve.
                    GpuBatchOverrides.Snapshot preHvdcOverrides = null;
                    int[][] hvdcStatus = null;
                    double[] hvAppliedD1 = null;
                    double[] hvAppliedD2 = null;
                    // HvdcAction (AC emulation off): OLF sets each station's target to the link's flow at the state the
                    // strategy starts from - the parent's converged solution, which is this pair's warm start - unclamped,
                    // and takes the link out of its equations. On the device: the element masked, the two set points as
                    // converter-bus target overrides, the status left LINEAR so the HVDC loops treat the link as the
                    // scenario's own (never frozen, unfrozen or saturated by them). A station on the slack bus has no row:
                    // its set point reaches the slack mismatch through the applied delta (hvdcSlackBalanceBatch), which
                    // the VSC curve channel reads too - set below, once the HVDC arrays exist.
                    List<double[]> hvActionApplied = new ArrayList<>();   // {slot, ctl, d1 - baseD1, d2 - baseD2}
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
                        for (com.powsybl.openloadflow.network.LfHvdc hv : off) {
                            Integer elem = hvdcElemIndex.get(hv);
                            HvdcAcEmulation.Ctl c = !ok || elem == null ? null : hvdcControllers.stream()
                                    .filter(ctl -> ctl.hvElemIndex() == elem).findFirst().orElse(null);
                            if (c == null) {
                                ok = false;
                                break;
                            }
                            double[] warm = java.util.Arrays.copyOfRange(x0Full, s * n, (s + 1) * n);
                            double d1 = -c.p1At(warm);
                            double d2 = -c.p2At(warm);
                            int[] rm = rs.rowModePerScenario[s];
                            rs.disabledMask[s][c.hvElemIndex()] = true;
                            if (c.bus1PActive()) {
                                rs.overrides.merge(s * n + c.bus1WriteRow(rm), baseTargets[c.bus1WriteRow(rm)], c.baseD1(), d1);
                            }
                            if (c.bus2PActive()) {
                                rs.overrides.merge(s * n + c.bus2WriteRow(rm), baseTargets[c.bus2WriteRow(rm)], c.baseD2(), d2);
                            }
                            hvActionApplied.add(new double[] {s, hvdcControllers.indexOf(c), d1 - c.baseD1(), d2 - c.baseD2()});
                        }
                        if (!ok) {
                            if (rejectAfterSolve == null) {
                                rejectAfterSolve = new boolean[chunkSize];
                            }
                            rejectAfterSolve[s] = true;
                            com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("PAIR:HVDC_ACTION_PARENT_NOT_CONVERGED");
                        }
                    }
                    if (hvdcControllers != null && !hvdcControllers.isEmpty()) {
                        hvAppliedD1 = new double[chunkSize * hvdcControllers.size()];
                        hvAppliedD2 = new double[chunkSize * hvdcControllers.size()];
                        preHvdcOverrides = rs.overrides.snapshot();
                        if (System.getenv("OLF_GPU_HVDC_TRACE") != null) {
                            for (int s = 0; s < realCount; s++) {
                                System.err.println("HVDC_SLOT s=" + s + " ctg="
                                        + contingencies.get(group.get(chunkStart + s)).contingencyId());
                                // the PRE-freeze override entries on this slot's converter-bus P rows: the freeze
                                // composes on top of them, so a stale one here is a double count downstream
                                for (HvdcAcEmulation.Ctl c : hvdcControllers) {
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
                            hvdcStatus = HvdcAcEmulation.applyFreeze(chunkSize, realCount, n, hvdcControllers,
                                    baseState, x0Full, x0Full == null ? x0SeedIdx : null, x0Full == null ? x0SeedVal : null,
                                    rs.disabledMask, baseTargets, rs.overrides, hvAppliedD1, hvAppliedD2, rs.rowModePerScenario);
                            targetOverrideIndex = rs.overrides.indexArray();
                            targetOverrideValue = rs.overrides.valueArray();
                        } else {
                            hvdcStatus = new int[chunkSize][hvdcControllers.size()];
                        }
                        // A NON-LINEAR-at-base hvdc (saturated/frozen base — no longer a whole-run
                        // gate): entries the freeze did not claim keep the base status + hv bit
                        // (no override, baseTargets already carry the base injections); padding
                        // slots get the base bits so they converge at iteration 0.
                        HvdcAcEmulation.seedBaseStatuses(chunkSize, realCount, hvdcControllers,
                                hvdcStatus, rs.disabledMask);
                    }
                    boolean dbgPairRl = pairPhase && nrl > 0 && "1".equals(System.getenv("OLF_GPU_DBG_PAIR_RL"));
                    boolean svcActive = phases.secondaryVc() != null && !phases.secondaryVc().ctls().isEmpty();
                    boolean monActive = phases.monitoring() != null && !phases.monitoring().isEmpty();
                    int[] rlStateOut = (!pairPhase && !pairIdxList.isEmpty() || dbgPairRl || svcActive || monActive
                            || System.getenv("OLF_GPU_RL_TRACE") != null) && nrl > 0 ? new int[chunkSize * nrl] : null;
                    cl.upload(handle, gpuData);
                    // The two HVDC AC-emulation loops run as DRIVER PHASES, in OLF's own position
                    // between DistributedSlack and ReactiveLimits. Running them instead as host
                    // orchestrators after the whole solve put them out of OLF's order and left one
                    // differing violation line over 2000 rte6515 branch contingencies.
                    for (double[] e : hvActionApplied) {           // the HvdcAction set points, as applied deltas
                        int k = (int) e[0] * hvdcControllers.size() + (int) e[1];
                        hvAppliedD1[k] = e[2];
                        hvAppliedD2[k] = e[3];
                    }
                    byte[] ctgOutOf = null;
                    boolean hasHvdcDriverLoops = hvdcControllers != null && !hvdcControllers.isEmpty();
                    ctgOutOf = HvdcAcEmulation.installDriverLoops(handle,
                            hasHvdcDriverLoops ? hvdcControllers : null, chunkSize, hvdcStatus, rs.disabledMask,
                            hvAppliedD1, hvAppliedD2, config.hvdcLimits());
                    // MonitoringVoltage as a DRIVER PHASE, in OLF's own slot (order 6, between the
                    // HVDC loops and ReactiveLimits) instead of as a host orchestrator after the whole
                    // solve. Run-constant table, but the context is per chunk.
                    boolean monInKernel = monActive;
                    VoltageMonitoring.installDriverLoop(handle, monInKernel ? phases.monitoring() : null);
                    // SecondaryVoltageControl as a DRIVER PAUSE at its OLF slot (order 4). Run as a host
                    // orchestrator it executed AFTER the whole device sweep — i.e. after the control loops
                    // OLF puts at orders 7-10 — so each decided on a state the other had already moved
                    // (1.7e-3 kV on a zone whose pilot bus feeds an AFTER_GENERATOR transformer; the
                    // pre-2026-09-15 host chain was worse still at 2.4e-1 kV, its re-solves not composing).
                    // Pausing puts it back in OLF's order with no new maths: the same decision code runs on
                    // the paused states and hands its target moves back as flat deltas.
                    // SecondaryVoltageControl, in OLF's order (order 4). Preferred form: an IN-KERNEL phase,
                    // the whole check on the device. Where the kernel refuses the zone set (wider than its
                    // dense system, or an observable on an unpacked element) the driver PAUSES at the same
                    // slot and the same decision runs on the host instead.
                    boolean svcInKernel = svcActive && SecondaryVoltageControlLoop.uploadToDevice(handle, phases.secondaryVc());
                    if (!svcInKernel) {
                        SecondaryVoltageControlLoop.uploadToDevice(handle, null);   // clear a cached context's table
                    }
                    boolean svcAsDriverPause = svcActive && rlStateOut != null && !svcInKernel;
                    GpuAcNewtonSolver.setBatchHostLoop(handle, svcAsDriverPause ? 1 : 0);
                    long solveCountBefore = GpuAcNewtonSolver.readBatchSolveCount(handle);
                    // Seeds only where no full per-scenario state goes (the pair phase builds its own from the parents).
                    GpuAcNewtonSolver.setBatchX0Seeds(handle, x0Full == null ? x0SeedIdx : null, x0Full == null ? x0SeedVal : null);
                    long tSolve0 = System.nanoTime();
                    // OLF_GPU_LOOP_TRACE_CTG=<id>: trace that contingency's slot in THIS chunk (a fixed slot number
                    // follows the chunk composition, which moves with the resolved chunk size).
                    // OLF's tree for the scenarios of this chunk, where it has moved from the one the
                    // structure was extracted on (empty clears the previous chunk's list).
                    GpuAcNewtonSolver.setBatchZiTreeOverrides(handle,
                            rs.ziTreeSlotElemValList.stream().mapToInt(Integer::intValue).toArray());
                    {
                        String traceCtg = System.getenv("OLF_GPU_LOOP_TRACE_CTG");
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
                        GpuAcNewtonSolver.setBatchLoopTraceSlot(handle, traceSlot);
                    }
                    // OLF_GPU_CHUNK_DUMP=<file>: every chunk's members in slot order, appended one line per chunk
                    // ("<pairPhase> <chunkStart> <chunkSize> <id> <id> ..."). A scenario whose answer moves with
                    // its chunk mates can only be reproduced in the chunk it was solved in.
                    String chunkDump = System.getenv("OLF_GPU_CHUNK_DUMP");
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
                    if (System.getenv("OLF_GPU_ROW_PROBE") != null && realCount > 1) {
                        // OLF_GPU_ROW_PROBE=<row>[,<row>]: slot 1's row mode and target next to the base's,
                        // and every row whose mode slot 1 changed - what the scenario solves differently.
                        int[] m1 = rs.rowModePerScenario[1];
                        java.util.List<Integer> changed = new java.util.ArrayList<>();
                        for (int r = 0; r < n; r++) {
                            if (m1[r] != baseRowMode[r]) {
                                changed.add(r);
                            }
                        }
                        java.util.Map<Integer, Double> ov1 = new java.util.TreeMap<>();
                        for (int k = 0; k < targetOverrideIndex.length; k++) {
                            if (targetOverrideIndex[k] / n == 1) {
                                ov1.put(targetOverrideIndex[k] % n, targetOverrideValue[k]);
                            }
                        }
                        System.err.println("ROW_PROBE ctg=" + contingencies.get(group.get(chunkStart + 1)).contingencyId()
                                + " modeChanged=" + changed.subList(0, Math.min(40, changed.size())) + " n=" + changed.size()
                                + " overrides=" + ov1.size());
                        for (String t : System.getenv("OLF_GPU_ROW_PROBE").split(",")) {
                            int r = Integer.parseInt(t.trim());
                            System.err.printf("ROW_PROBE row=%d baseMode=%d mode1=%d baseTarget=%.9f ov1=%s x=%.9f%n",
                                    r, baseRowMode[r], m1[r], baseTargets[r], ov1.get(r), baseState[r]);
                        }
                        ov1.entrySet().stream().limit(60).forEach(e -> System.err.printf(
                                "ROW_PROBE ov row=%d value=%.9f base=%.9f%n", e.getKey(), e.getValue(), baseTargets[e.getKey()]));
                    }
                    if (forcedTargetQ != null) {                 // where each slot's pre-distribution put the clamping gens
                        // a pair starts each generator at its anchor, a contingency at its pre-distributed target
                        ReactiveLimitsNative.setBatchForcedTargetQPre(handle, forcedTargetQ.preMoves(
                                pairPhase && dgAnchor != null ? ChunkArrays.perSlot(dgAnchor, chunkSize, ngen) : cl.genPreMovesBySlot));
                    }
                    if (cl.outageSlackShift != null && rs.hasDs) {     // on top of whatever the pre-distribution or the pair's
                        if (dsSlackShift == null) {               // parent put there: both are slack-target moves
                            dsSlackShift = new double[chunkSize];
                        }
                        for (int s = 0; s < chunkSize; s++) {
                            dsSlackShift[s] += cl.outageSlackShift[s];
                        }
                    }
                    if (dsSlackOnly) {
                        GpuAcNewtonSolver.setBatchSlackOnlyAnchor(handle, pairPhase ? pairSlackAbsorbedSeed : null);
                    }
                    if (nrl > 0) {
                        ReactiveLimitsNative.setBatchRlPinTargets(handle, pairPhase ? rlPinIn : null);
                    }
                    if (ngen > 0) {
                        GpuAcNewtonSolver.setBatchGenAnchor(handle, pairPhase ? dgAnchor : null,
                                pairPhase ? null : ChunkArrays.flat(cl.genPreMovesBySlot, chunkSize, ngen), pairPhase ? dgSeedAnchor : null);
                    }
                    double[][] chunkResults = GpuAcNewtonSolver.solveBatchContext(handle, baseState, baseTargets,
                            rs.rowModePerScenario, baseRowMode, rs.disabledMask, targetOverrideIndex, targetOverrideValue,
                            rs.dsOvBus, rs.dsOvFactorDelta, rs.dsOvBaseDelta, rs.dsOvMaxDelta, rs.dsOvMinDelta, dsAnchorShift, dsSlackShift,
                            ChunkArrays.intsOrNull(cl.rlVtOvIdx),
                            cl.rlVtOvIdx.isEmpty() ? null : cl.rlVtOvVal.stream().mapToDouble(Double::doubleValue).toArray(),
                            cl.rlQOvSlot.stream().mapToInt(Integer::intValue).toArray(),
                            cl.rlQOvVal.stream().mapToDouble(Double::doubleValue).toArray(),
                            cl.cvOvSlot.stream().mapToInt(Integer::intValue).toArray(),
                            cl.cvOvVal.stream().mapToDouble(Double::doubleValue).toArray(),
                                                        parameters.getMaxIterations(), tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                            config.lsMode(), config.maxDv(), config.maxDphi(),
                            config.lsMode() == 2 ? varType : null, x0Full,
                            rs.shOvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                            rs.shOvGbList.stream().mapToDouble(Double::doubleValue).toArray(),
                            rlStateIn, rlStateOut,
                            rs.r1OvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                            rs.r1OvValList.stream().mapToDouble(Double::doubleValue).toArray(),
                            config.maxOuterIterations(), config.maxPqPvSwitch(), config.dsRejectResidualFlag(),
                            // monitoring active: OLF checks VoltageMonitoring BEFORE ReactiveLimits — the main
                            // solve surfaces the DS-stable / pre-RL state; the orchestrator then interleaves
                            // and finishes with the full RL-enabled solves.
                            0,
                            rs.nziBase, rs.ziOnSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                            rs.ziOffSlotElemList.stream().mapToInt(Integer::intValue).toArray());
                    if (System.getenv("OLF_GPU_REJECT_TRACE") != null) {
                        // slot -> contingency for every scenario the device rejected in this chunk: the native
                        // GPU_REJECT lines name slots, and a slot only means something with the chunk's composition.
                        for (int s = 0; s < realCount; s++) {
                            int st = chunkResults[s].length > n + 2 ? (int) chunkResults[s][n + 2] : 0;
                            if (st <= GpuBatchStatus.STATUS_REJECTED) {
                                System.err.println("CHUNK_REJECT slot=" + s + " ctg=" + contingencies.get(group.get(chunkStart + s)).contingencyId()
                                        + " status=" + st);
                            }
                        }
                    }
                    if (System.getenv("OLF_GPU_DUMMY_CENSUS") != null) {
                        // OLF_GPU_DUMMY_CENSUS=1: a scenario whose final state carries a coupler flow beyond any physical
                        // value (|dummy| > 1e3 pu, 100 GW) solved a SINGULAR system - a loop of zero-impedance couplers
                        // whose circulating flow no equation determines. The equations then differ from OLF's, and the
                        // rounding of that flow moves the answer from one run to the next (network.xiidm busbar HFU).
                        int nz = gpuData.ziBranchNums() == null ? 0 : gpuData.ziBranchNums().length;
                        for (int s = 0; s < realCount; s++) {
                            double worst = 0;
                            int worstZe = -1;
                            for (int z = 0; z < nz; z++) {
                                for (int k = 4; k <= 5; k++) {
                                    int r = gpuData.ziIdx()[z * 11 + k];
                                    if (r >= 0 && r < n && Math.abs(chunkResults[s][r]) > worst) {
                                        worst = Math.abs(chunkResults[s][r]);
                                        worstZe = z;
                                    }
                                }
                            }
                            if (worst > 1e3) {
                                System.err.printf("DUMMY_CENSUS ctg=%s status=%d worst=%.3e coupler=#%d%n",
                                        contingencies.get(group.get(chunkStart + s)).contingencyId(),
                                        chunkResults[s].length > n + 2 ? (int) chunkResults[s][n + 2] : 0, worst,
                                        gpuData.ziBranchNums()[worstZe]);
                            }
                        }
                    }
                    tMainSolveNs += System.nanoTime() - tSolve0;
                    if (svcAsDriverPause) {
                        long tSvc0 = System.nanoTime();
                        int nSvcCtl = phases.secondaryVc().ctls().size();
                        double[][] svcTargetV = new double[chunkSize][nSvcCtl];
                        for (int s2 = 0; s2 < chunkSize; s2++) {
                            for (int c = 0; c < nSvcCtl; c++) {
                                svcTargetV[s2][c] = phases.secondaryVc().ctls().get(c).baseTargetV();
                            }
                        }
                        // The driver re-pauses at the SVC slot on EVERY sweep restart, and a reactive-limit
                        // switch restarts the sweep — so the pause count is bounded by the whole outer-loop
                        // budget, not by the number of zone adjustments. The kernel retires a scenario whose
                        // budget a host-UNSTABLE verdict exhausts, so this cap is only a safety net; stopping
                        // early would leave the context PAUSED and the chunk holding mid-sweep states.
                        int maxSvcRounds = 16 * (config.maxOuterIterations() > 0 ? config.maxOuterIterations() : 20);
                        int svcRounds = 0;
                        while (GpuAcNewtonSolver.readBatchDriverPaused(handle) == DeviceLoopKind.HOST_LOOP_SVC
                                && svcRounds++ < maxSvcRounds) {
                            var dec = SecondaryVoltageControlLoop.decideAtPause(handle, chunkSize, realCount, n, nrl,
                                    chunkResults, phases.secondaryVc(), rlStateOut, svcTargetV, baseState);
                            if (dec.targetRows() != null && dec.targetRows().length > 0) {
                                GpuAcNewtonSolver.setBatchTargetDeltas(handle, dec.targetRows(), dec.targetValues());
                            }
                            if (dec.unpinPairs() != null && dec.unpinPairs().length > 0) {
                                ReactiveLimitsNative.setBatchRlUnpin(handle, dec.unpinPairs(), dec.unpinTargets());
                            }
                            GpuAcNewtonSolver.setBatchDriverResume(handle, dec.hostChanged());
                            chunkResults = GpuAcNewtonSolver.solveBatchContext(handle, baseState, baseTargets,
                                    rs.rowModePerScenario, baseRowMode, rs.disabledMask, targetOverrideIndex, targetOverrideValue,
                                    rs.dsOvBus, rs.dsOvFactorDelta, rs.dsOvBaseDelta, rs.dsOvMaxDelta, rs.dsOvMinDelta, dsAnchorShift, dsSlackShift,
                                    ChunkArrays.concatInts(cl.rlVtOvIdx, dec.rlVtSlotIdx()), ChunkArrays.concatDoubles(cl.rlVtOvVal, dec.rlVtValues()),
                                    cl.rlQOvSlot.stream().mapToInt(Integer::intValue).toArray(),
                                    cl.rlQOvVal.stream().mapToDouble(Double::doubleValue).toArray(),
                                    cl.cvOvSlot.stream().mapToInt(Integer::intValue).toArray(),
                                    cl.cvOvVal.stream().mapToDouble(Double::doubleValue).toArray(),
                                                                        parameters.getMaxIterations(), tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                                    config.lsMode(), config.maxDv(), config.maxDphi(),
                                    config.lsMode() == 2 ? varType : null, x0Full,
                                    rs.shOvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                    rs.shOvGbList.stream().mapToDouble(Double::doubleValue).toArray(),
                                    null, rlStateOut,
                                    rs.r1OvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                    rs.r1OvValList.stream().mapToDouble(Double::doubleValue).toArray(),
                                    config.maxOuterIterations(), config.maxPqPvSwitch(), config.dsRejectResidualFlag(), 0,
                                    rs.nziBase, rs.ziOnSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                    rs.ziOffSlotElemList.stream().mapToInt(Integer::intValue).toArray());
                        }
                        if (GpuAcNewtonSolver.readBatchDriverPaused(handle) != 0) {
                            // Safety net only: the budget bookkeeping above should always end the cycle.
                            // Mid-sweep states must never be reported — cancel and let the CPU replay them.
                            LOGGER.warn("Secondary VC driver pause did not settle in {} rounds — chunk routed to the CPU",
                                    maxSvcRounds);
                            GpuAcNewtonSolver.cancelBatchDriverPause(handle);
                            for (int s2 = 0; s2 < realCount; s2++) {
                                if (chunkResults[s2] != null) {
                                    chunkResults[s2][n + 2] = GpuBatchStatus.STATUS_REJECTED;
                                }
                            }
                        }
                        if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
                            System.err.printf("SVC_PAUSE rounds=%d stillPaused=%d%n", svcRounds,
                                    GpuAcNewtonSolver.readBatchDriverPaused(handle));
                        }
                        orchNs.merge("SecondaryVc.driverPause", System.nanoTime() - tSvc0, Long::sum);
                    }
                    if (OL_STATS) {
                        // OLF_OL_STATS: the device's per-contingency Newton work, in the same shape as OLF's
                        // CPU_OL_STATS lines, so the two arms can be matched contingency by contingency.
                        int[] ol = GpuAcNewtonSolver.readBatchOlStats(handle);
                        if (ol != null) {
                            for (int s2 = 0; s2 < realCount; s2++) {
                                int st = chunkResults[s2] == null ? Integer.MIN_VALUE : (int) chunkResults[s2][n + 2];
                                System.err.println("GPU_OL_STATS ctg=" + contingencies.get(group.get(chunkStart + s2)).contingencyId()
                                        + " passes=" + ol[s2 * 3] + " firstNr=" + ol[s2 * 3 + 1] + " totalNr=" + ol[s2 * 3 + 2]
                                        + " status=" + st);
                            }
                        }
                    }
                    if (monInKernel) {
                        // Keep ACTIVATIONS meaningful with the loop in the driver: it is the guard
                        // that a case really exercises the automaton, not orchestrator bookkeeping.
                        VoltageMonitoring.ACTIVATIONS.add(
                                GpuAcNewtonSolver.readBatchMonitoringActivations(handle));
                    }
                    if (hasHvdcDriverLoops) {
                        // The driver phases mutated the hv element bits and the converter-bus targets
                        // ON THE DEVICE. The host keeps mirrors of both, and the orchestrators that
                        // run next re-upload them — so without this sync they would clobber the loops'
                        // work with the pre-solve state.
                        int[] devStatus = GpuAcNewtonSolver.readBatchHvdcStatus(handle);
                        int nh = hvdcControllers.size();
                        if (devStatus != null) {
                            for (int s2 = 0; s2 < chunkSize; s2++) {
                                System.arraycopy(devStatus, s2 * nh, hvdcStatus[s2], 0, nh);
                            }
                            if (System.getenv("OLF_GPU_HVDC_TRACE") != null) {
                                // the status each scenario ENDED with (0 linear, 1/2 saturated, 3 frozen)
                                for (int s2 = 0; s2 < realCount; s2++) {
                                    StringBuilder sb = new StringBuilder();
                                    for (int h = 0; h < nh; h++) {
                                        sb.append(' ').append(hvdcStatus[s2][h]);
                                    }
                                    System.err.println("HVDC_END s=" + s2 + " ctg="
                                            + contingencies.get(group.get(chunkStart + s2)).contingencyId() + " status:" + sb);
                                }
                            }
                            // Re-derive the host's OVERRIDE LIST and hv bits from those statuses, on
                            // top of the PRE-FREEZE list. Syncing only the bits is not enough: every
                            // orchestrator that runs next rebuilds its targets from this list, so a
                            // stale post-freeze list re-applies a freeze the kernel already lifted
                            // (measured: 566 differing lines over 2000 with only the bits synced).
                            rs.overrides.restore(preHvdcOverrides);
                            HvdcAcEmulation.applyStatusOverrides(chunkSize, realCount, n,
                                    hvdcControllers, hvdcStatus, ctgOutOf, rs.disabledMask, baseTargets, rs.overrides,
                                    hvAppliedD1, hvAppliedD2, rs.rowModePerScenario);
                            targetOverrideIndex = rs.overrides.indexArray();
                            targetOverrideValue = rs.overrides.valueArray();
                        }
                    }
                    if (System.getenv("OLF_GPU_ITER_TRACE") != null && !pairPhase) {
                        java.util.Map<Integer, Integer> hist = new java.util.TreeMap<>();
                        for (int s = 0; s < realCount; s++) {
                            hist.merge((int) chunkResults[s][n], 1, Integer::sum);
                        }
                        System.err.println("ITER_HIST main solve, iterations -> scenarios: " + hist);
                        // OLF_GPU_ITER_TRACE_CTG=<id>[,<id>]: the iteration count of NAMED scenarios, at any
                        // chunk size. Two implementations that solve the same system in the same number of
                        // iterations agree to machine precision, so the count is the first thing to compare
                        // against OLF's own (OLF_LF_STEPS_TRACE) when the answers differ - and on a full
                        // scope the per-scenario print below never fires, because the chunk holds hundreds.
                        String iterCtg = System.getenv("OLF_GPU_ITER_TRACE_CTG");
                        for (int s = 0; s < realCount; s++) {
                            String id = contingencies.get(group.get(chunkStart + s)).contingencyId();
                            boolean named = false;
                            if (iterCtg != null) {
                                for (String w : iterCtg.split(",")) {
                                    named = named || w.trim().equals(id);
                                }
                            }
                            if (named || realCount <= 4) {
                                System.err.printf("ITER ctg=%s iters=%d finalMismatch=%.6e status=%d%n",
                                        id, (int) chunkResults[s][n], chunkResults[s][n + 1],
                                        (int) chunkResults[s][n + 2]);
                            }
                        }
                    }
                    if (System.getenv("OLF_GPU_DS_FINAL_DUMP") != null && nds > 0 && !pairPhase) {
                        // Every participant's FINAL distributed-slack delta for slot 0, against the same
                        // anchor the water-fill resets to. Two water-fills that move the same TOTAL can
                        // still split it differently, and only the per-participant deltas say so (the
                        // CPU counterpart is OLF's DS_GEN trace under OLF_DS_TRACE).
                        double[] ftd = GpuAcNewtonSolver.readBatchTargets(handle);
                        for (int i = 0; i < nds; i++) {
                            // slot 0's row, diverted like the kernels' — reading the structural row for a
                            // participant this scenario moved reports the PINNED dummy row's target, which
                            // is how a spurious -1.043213 "delta" got read as a real distribution.
                            int row = cl.dsRowOvBySlot[0] != null && cl.dsRowOvBySlot[0][i] >= 0
                                    ? cl.dsRowOvBySlot[0][i] : gpuData.dsPhiRow()[i];
                            double d = ftd[row] - gpuData.dsBaseTargetP()[i];
                            if (d != 0) {
                                System.err.printf("DS_FINAL_ROW ctg=%s i=%d row=%d base=%.12e final=%.12e delta=%.12e factor=%.12e%n",
                                        contingencies.get(group.get(chunkStart)).contingencyId(), i, row,
                                        gpuData.dsBaseTargetP()[i], ftd[row], d, gpuData.dsFactor()[i]);
                            }
                        }
                    }
                    if (System.getenv("OLF_GPU_PRELOSS_TRACE") != null && nds > 0 && !pairPhase) {
                        // Did the DEVICE's own distributed-slack pass move power AFTER the host-side
                        // pre-distribution? The CPU's loop should see only the loss-nonlinearity
                        // residual and stay inside the slack band; if ours redistributes again, the
                        // two land at different points inside that band.
                        double[] ft = GpuAcNewtonSolver.readBatchTargets(handle);
                        for (int s = 0; s < realCount; s++) {
                            if (cl.preLossBySlot[s] == 0) {
                                continue;
                            }
                            int ci = group.get(chunkStart + s);
                            double moved = 0;
                            for (int i = 0; i < nds; i++) {
                                int row = cl.dsRowOvBySlot[s] != null && cl.dsRowOvBySlot[s][i] >= 0
                                        ? cl.dsRowOvBySlot[s][i] : gpuData.dsPhiRow()[i];
                                // A participant this contingency ISLANDED has its row frozen to a flat
                                // identity, so (final - base) there is minus its whole anchor and not a
                                // distribution at all: counting it read the middle-section fixture's
                                // participants as -1.669969 pu off OLF when the real gap was -0.0687.
                                if (islandedDsByIndex != null && islandedDsByIndex[ci] != null
                                        && ChunkArrays.contains(islandedDsByIndex[ci], i)) {
                                    continue;
                                }
                                moved += ft[s * n + row] - baseTargets[gpuData.dsPhiRow()[i]];
                            }
                            // preLossBySlot is the mismatch the pre-distribution was asked to absorb;
                            // `moved` is where the participants ended up relative to the SAME anchor the
                            // device DS loop resets to, so the delta is what that loop added on top.
                            System.err.printf("PRELOSS_AFTER ctg=%s preLossMismatch=%.6f seededOnParticipants=%.6f "
                                    + "deviceFinal=%.6f deltaFromDeviceDsLoop=%.6f pu%n",
                                    contingencies.get(group.get(chunkStart + s)).contingencyId(),
                                    cl.preLossBySlot[s], cl.preAbsorbedBySlot[s], moved, moved - cl.preAbsorbedBySlot[s]);
                        }
                    }
                    if (GpuBatchPhaseStats.ENABLED) {
                        int itMax = 0;
                        for (int s = 0; s < realCount; s++) {
                            itMax = Math.max(itMax, (int) chunkResults[s][n]);
                        }
                        GpuBatchPhaseStats.add("main.iterMax", itMax);
                        GpuBatchPhaseStats.bump("main.solves");
                    }
                    if (System.getenv("OLF_GPU_DS_TARGET_DUMP") != null && nds > 0) {
                        // The FINAL distributed target on every participant row, after every
                        // orchestrator. Both arms are converged, so if their reported states differ
                        // their TARGETS must differ — this is the only direct way to see where.
                        double[] ft = GpuAcNewtonSolver.readBatchTargets(handle);
                        for (int s = 0; s < realCount; s++) {
                            String id = contingencies.get(group.get(chunkStart + s)).contingencyId();
                            for (int i = 0; i < nds; i++) {
                                System.err.printf("GPU_DS_TARGET ctg=%s i=%d row=%d target=%s%n",
                                        id, i, gpuData.dsPhiRow()[i],
                                        Double.toString(ft[s * n + gpuData.dsPhiRow()[i]]));
                            }
                        }
                    }
                    if (System.getenv("OLF_GPU_RL_TRACE") != null && rlStateOut != null) {
                        // rlStateOut packs (limitType & 7) | (switchCount << 3) per (controller, scenario):
                        // switchCount > 0 means the DEVICE switched that controller during THIS solve, which
                        // is exactly the decision OLF's ReactiveLimitsOuterLoop makes (and logs) on the CPU.
                        for (int s = 0; s < realCount; s++) {
                            int switched = 0;
                            int pinned = 0;
                            StringBuilder ids = new StringBuilder();
                            for (int i = 0; i < nrl; i++) {
                                int v = rlStateOut[s * nrl + i];
                                if ((v & 7) != 0) {
                                    pinned++;
                                    if (ids.length() < 4000) {
                                        // vRow identifies WHICH controller: it is the bus's stable device
                                        // row, the same number the DISTR_Q / target dumps print.
                                        ids.append(" p").append(i).append("(vRow=").append(gpuData.rlVrow()[i])
                                           .append(",lt=").append(v & 7).append(')');
                                    }
                                }
                                if ((v >> 3) > 0) {
                                    switched++;
                                    if (ids.length() < 300) {
                                        ids.append(" rl").append(i).append("(vRow=")
                                           .append(gpuData.rlVrow()[i]).append(",lt=").append(v & 7).append(')');
                                    }
                                }
                            }
                            System.err.printf("GPU_RL ctg=%s switchedThisSolve=%d pinnedTotal=%d%s%n",
                                    contingencies.get(group.get(chunkStart + s)).contingencyId(), switched, pinned, ids);
                        }
                    }
                    long tOrch0 = System.nanoTime();
                    // Which scenarios already hold a REACTIVE-LIMITS-stabilized state: none yet when the
                    // main solve skipped the RL phase for the monitoring interleave, all of them otherwise.
                    // An orchestrator that RL-solves a scenario sets its flag, and VoltageMonitoring uses
                    // it to skip a confirmation solve that could only re-derive what it was handed.
                    boolean[] rlSolvedPerScenario = new boolean[chunkSize];
                    // When the network has HVDC the driver's loop LIST ends with ReactiveLimits, so
                    // the main solve leaves every scenario RL-stable and OLF's rule applies: a loop
                    // that reports STABLE never re-solves. Reporting those scenarios as still "owing"
                    // an RL solve makes the monitoring orchestrator run one extra Newton step the CPU
                    // never runs — measured as the WHOLE divergence on LINE-1392-677 and
                    // LINE-2045-221 (1.1e-3 pu = 7.0e-2 kV at a 63 kV bus).
                    java.util.Arrays.fill(rlSolvedPerScenario, true);
                    if (dbgPairRl) {
                        for (int s2 = 0; s2 < realCount; s2++) {
                            StringBuilder sb = new StringBuilder("GPU-PAIR-RL " + contingencies.get(group.get(chunkStart + s2)).contingencyId().replace(GpuBatchStatus.STRATEGY_KEY_SEP, '|'));
                            for (int i2 = 0; i2 < nrl; i2++) {
                                int v = rlStateOut[s2 * nrl + i2];
                                if ((v & 7) != 0) {
                                    sb.append(" ctl").append(i2).append("=lt").append(v & 7).append("/sc").append(v >> 3);
                                }
                            }
                            System.err.println(sb);
                        }
                    }
                    if (rlStateOut != null && !pairPhase) {      // capture each ctg's final RL pinning for its pairs
                        // ...and the targets its pins ENDED with, which its pairs restore (OLF restores the parent's
                        // generation target Q; a limit that moved by less than the re-freeze tolerance kept the old one)
                        double[] pins = ReactiveLimitsNative.readBatchRlPinTargets(handle);
                        for (int s = 0; s < realCount; s++) {
                            if (pins != null) {
                                rlPinByIndex.put(group.get(chunkStart + s), java.util.Arrays.copyOfRange(pins, s * nrl, (s + 1) * nrl));
                            }
                            int[] slice = new int[nrl];
                            System.arraycopy(rlStateOut, s * nrl, slice, 0, nrl);
                            rlByIndex.put(group.get(chunkStart + s), slice);
                        }
                    }
                    if (!pairPhase && dsSlackOnly && !pairIdxList.isEmpty()) {
                        // capture each ctg's slack-only-DS absorbed total for its pairs' seeding
                        double[] absorbed = GpuAcNewtonSolver.readBatchSlackAbsorbed(handle);
                        if (absorbed != null) {
                            for (int s = 0; s < realCount; s++) {
                                slackAbsorbedByIndex.put(group.get(chunkStart + s), absorbed[s]);
                            }
                        }
                    }
                    // OLF_GPU_ROWMODE_DUMP_CTG=<contingency id>: the DEVICE's final per-scenario row modes
                    // and targets for that scenario, as the rows that DIFFER from the base. This is the other
                    // half of the comparison against OLF_GPU_EQ_DUMP_CTG (OLF's post-contingency equations):
                    // together they say, row by row, which equation each side has where.
                    String rmCtg = System.getenv("OLF_GPU_ROWMODE_DUMP_CTG");
                    if (rmCtg != null && !rmCtg.isEmpty()) {
                        int[] devModes = GpuAcNewtonSolver.readBatchRowModes(handle);
                        double[] devTargets = GpuAcNewtonSolver.readBatchTargets(handle);
                        for (int s = 0; s < realCount; s++) {
                            if (!rmCtg.equals(contingencies.get(group.get(chunkStart + s)).contingencyId())) {
                                continue;
                            }
                            // OLF_GPU_ROWMODE_DUMP_ALL=1: EVERY row, not only the ones differing from
                            // base. Needed to diff the device's TARGET VECTOR against OLF's F0 section
                            // (which prints f and target per ACTIVE equation): a target that is wrong
                            // while equal to the base's is invisible in the differing-rows view, and
                            // equation IDENTITY matching says nothing about the numbers (tasks #67).
                            boolean rmAll = "1".equals(System.getenv("OLF_GPU_ROWMODE_DUMP_ALL"));
                            StringBuilder sb = new StringBuilder();
                            for (int r = 0; r < n; r++) {
                                int m = devModes[s * n + r];
                                if (rmAll || m != baseRowMode[r]) {
                                    sb.append(String.format("  row=%d mode=%d<-%d target=%.17g%n",
                                            r, m, baseRowMode[r], devTargets[s * n + r]));
                                }
                            }
                            System.err.print("ROWMODE_DUMP ctg=" + rmCtg + " slot=" + s
                                    + " rows differing from base:\n" + sb);
                        }
                    }
                    if (!pairPhase && !pairIdxList.isEmpty()) {
                        // A contingency that ran out of Newton iterations on the device (OLF's MAX_ITERATION_REACHED):
                        // OLF still runs its strategies, from what the failed run left - the voltages of its last
                        // converged solve (NewtonRaphson updates the network only then), and the targets and bus
                        // types its outer loops had moved, which the captures below read as they read a converged one.
                        for (int s = 0; s < realCount; s++) {
                            if ((int) chunkResults[s][n + 2] == GpuBatchStatus.STATUS_MAX_ITERATION) {
                                exhaustedStartByIndex.put(group.get(chunkStart + s),
                                        GpuAcNewtonSolver.readBatchLastConvergedState(handle, s));
                            }
                        }
                    }
                    if (!pairPhase && nds > 0 && !pairIdxList.isEmpty()) {
                        // Capture each ctg's DS ABSORPTION (final distributed target minus the ctg's own
                        // effective base per DS row) plus its factor sum, for its pairs' re-anchoring —
                        // the device analogue of OLF's setGeneratorsInitialTargetPToTargetP before a
                        // strategy re-solve.
                        double[] finalTargets = GpuAcNewtonSolver.readBatchTargets(handle);
                        for (int s = 0; s < realCount; s++) {
                            double[] slice = new double[nds + 1];
                            double fs = 0;
                            int[] islandedDs = islandedDsByIndex != null ? islandedDsByIndex[group.get(chunkStart + s)] : null;
                            for (int i = 0; i < nds; i++) {
                                if (islandedDs != null && ChunkArrays.contains(islandedDs, i)) {
                                    // A participant this contingency islanded absorbed nothing. Its row is a FROZEN
                                    // identity whose target is an angle: read as a P it handed the pairs minus the bus's
                                    // whole base injection, composed onto the pair's frozen row (rte6515 LINE-4468-670:
                                    // -8.728 rad held at the islanded VL-6297, 0.007 MW) or, by a reclose, onto its live one.
                                    continue;
                                }
                                // read where this scenario solved the participant's balance: a coupler it opened
                                // brought a relocated balance home to the bus's own phi row, and the coupler's dummy
                                // row left behind holds a pinned stub (rte6515 BUS-1359_BBS_2: read there, -0.1926 pu
                                // of "absorption" that the pairs' first distribution took back, 19.27 MW)
                                int row = cl.dsRowOvBySlot[s] != null && cl.dsRowOvBySlot[s][i] >= 0
                                        ? cl.dsRowOvBySlot[s][i] : gpuData.dsPhiRow()[i];
                                double effBase = baseTargets[gpuData.dsPhiRow()[i]]
                                        - (rs.dsOvBus[s] == i ? rs.dsOvBaseDelta[s] : 0.0);
                                slice[i] = finalTargets[s * n + row] - effBase;
                                // the factors the parent's distribution renormalized over: an islanded participant
                                // is out of them, a lost generator's share leaves its bus
                                fs += gpuData.dsFactor()[i]
                                        - (rs.dsOvBus[s] == i ? rs.dsOvFactorDelta[s] : 0.0);
                            }
                            slice[nds] = fs;
                            dsTargetsByIndex.put(group.get(chunkStart + s), slice);
                        }
                        if (ngen > 0) {
                            // ...and each GENERATOR's final move, which is what OLF's setGeneratorsInitialTargetPToTargetP
                            // really keeps: the bus absorption above does not say how a bus's generators split it once
                            // one of them saturates. A slack pass re-derives the whole distribution from the base
                            // targets, so genCum holds it all when one fired; otherwise the pre-distribution alone moved.
                            double[] genCum = GpuAcNewtonSolver.readBatchDsGenCum(handle);
                            double[] retain = GpuAcNewtonSolver.readBatchDsRetain(handle);
                            for (int s = 0; s < realCount; s++) {
                                double[] moves = new double[ngen];
                                if (genCum != null && retain != null && !Double.isNaN(retain[s])) {
                                    System.arraycopy(genCum, s * ngen, moves, 0, ngen);
                                } else if (cl.genPreMovesBySlot != null && cl.genPreMovesBySlot[s] != null) {
                                    System.arraycopy(cl.genPreMovesBySlot[s], 0, moves, 0, ngen);
                                }
                                genMovesByIndex.put(group.get(chunkStart + s), moves);
                            }
                        }
                    }
                    // HOST ORCHESTRATORS — in OLF's DefaultAcOuterLoopConfig order.
                    // In-kernel: DS(0), HVDC(1,2), Monitoring(5), RL(6) already ran in the main solve.
                    // Post-RL host orchestrators below: SVC(4), Monitoring(5), Phase(7), TVC(8), TRPC(9), Shunt(10).

                    tOrchestratorNs += System.nanoTime() - tOrch0;
                    // OLF_GPU_FINAL_TARGET_ROW=<row>: the target the device ENDED UP solving that row
                    // against, after every per-scenario kernel and orchestrator. The base+override
                    // view (OLF_GPU_TARGET_ROW) shows what was UPLOADED; this shows what survived.
                    if (System.getenv("OLF_GPU_FINAL_TARGET_ROW") != null) {
                        // Comma-separated rows, so a shape that spans a group (pilot + its members)
                        // is read in ONE run instead of one run per row.
                        double[] ftp = GpuAcNewtonSolver.readBatchTargets(handle);
                        int[] fmode = GpuAcNewtonSolver.readBatchRowModes(handle);
                        for (String tok : System.getenv("OLF_GPU_FINAL_TARGET_ROW").split(",")) {
                            int probe = Integer.parseInt(tok.trim());
                            for (int s = 0; s < realCount; s++) {
                                // setupMode is what the HOST asked for; devMode is what the device
                                // ENDED with. They differ exactly when an outer-loop kernel rewrote the
                                // row -- which is the thing worth seeing.
                                int setupMode = rs.rowModePerScenario != null && rs.rowModePerScenario[s] != null
                                        ? rs.rowModePerScenario[s][probe] : baseRowMode[probe];
                                System.err.printf("FINAL_TARGET ctg=%s row=%d base=%.9f final=%.9f"
                                                + " setupMode=%d devMode=%d%n",
                                        contingencies.get(group.get(chunkStart + s)).contingencyId(), probe,
                                        baseTargets[probe], ftp[s * n + probe], setupMode,
                                        fmode == null ? -99 : fmode[s * n + probe]);
                            }
                        }
                    }
                    // OLF_GPU_GENCUM_DUMP=<file>: each GENERATOR's own cumulative P movement as the
                    // device computed it, which is what the curve channel now evaluates its capability
                    // at. Joined against OLF's per-generator movement (OLF_DS_SPLIT's G lines) it says
                    // whether the device and OLF put the same power on the same machine — the question
                    // the per-bus totals cannot answer.
                    if (System.getenv("OLF_GPU_GENCUM_DUMP") != null && gpuData.dgGenId() != null) {
                        double[] gc = GpuAcNewtonSolver.readBatchDsGenCum(handle);
                        String[] ids = gpuData.dgGenId();
                        if (gc != null && ids.length > 0) {
                            try (java.io.PrintWriter w = new java.io.PrintWriter(
                                    new java.io.FileWriter(System.getenv("OLF_GPU_GENCUM_DUMP"), true))) {
                                // OLF_TRACE_CTG=<id>: only that contingency, matching the OLF-side
                                // filter on OLF_DS_SPLIT. One line per moved generator over 6667
                                // contingencies is millions of lines on each side; the join that this
                                // dump exists for needs one section from each.
                                String gcWant = System.getenv("OLF_TRACE_CTG");
                                for (int s2 = 0; s2 < realCount; s2++) {
                                    String gcId = contingencies.get(group.get(chunkStart + s2)).contingencyId();
                                    if (gcWant != null && !gcWant.equals(gcId)) {
                                        continue;
                                    }
                                    w.printf("# GENCUM ctg=%s%n", gcId);
                                    for (int g = 0; g < ids.length; g++) {
                                        double v = gc[s2 * ids.length + g];
                                        if (v != 0) {
                                            w.printf("%s %.12f%n", ids[g], v);
                                        }
                                    }
                                }
                            } catch (java.io.IOException e) {
                                throw new java.io.UncheckedIOException(e);
                            }
                        }
                    }
                    if (System.getenv("OLF_GPU_DS_TARGET_DUMP") != null && nds > 0) {
                        // The FINAL distributed target on every participant row, after every
                        // orchestrator. Both arms are converged, so if their reported states differ
                        // their TARGETS must differ — this is the only direct way to see where.
                        double[] ft = GpuAcNewtonSolver.readBatchTargets(handle);
                        for (int s = 0; s < realCount; s++) {
                            String id = contingencies.get(group.get(chunkStart + s)).contingencyId();
                            double tot = 0;
                            for (int i = 0; i < nds; i++) {
                                int row = gpuData.dsPhiRow()[i];
                                double effBase = baseTargets[row] - (rs.dsOvBus[s] == i ? rs.dsOvBaseDelta[s] : 0.0);
                                tot += ft[s * n + row] - effBase;
                                System.err.printf("GPU_DS_TARGET ctg=%s i=%d target=%s delta=%s%n",
                                        id, i, Double.toString(ft[s * n + row]),
                                        Double.toString(ft[s * n + row] - effBase));
                            }
                            // The CUMULATIVE distributed active power. OLF's own DS re-distributes the
                            // cumulative amount from the INITIAL target P on every pass, so this total is
                            // directly comparable to the LAST `effective` OLF_DS_TRACE reports for the
                            // same contingency.
                            System.err.printf("GPU_DS_TOTAL ctg=%s distributed=%.9f%n", id, tot)
;
                        }
                    }
                    if (System.getenv("OLF_GPU_RL_TRACE") != null && rlStateOut != null) {
                        // The FINAL reactive-limits state, after every orchestrator re-solve. The
                        // per-solve trace above only sees the main solve, and OLF pins late in its
                        // outer-loop sequence (DS, HVDC freeze, monitoring, THEN ReactiveLimits), so
                        // comparing the main solve's count against the CPU's compares two different
                        // points in the trajectory.
                        for (int s = 0; s < realCount; s++) {
                            int pinned = 0;
                            for (int i = 0; i < nrl; i++) {
                                if ((rlStateOut[s * nrl + i] & 7) != 0) {
                                    pinned++;
                                }
                            }
                            StringBuilder ids = new StringBuilder();
                            for (int i = 0; i < nrl; i++) {
                                if ((rlStateOut[s * nrl + i] & 7) != 0) {
                                    ids.append(' ').append(i).append("(vRow=").append(gpuData.rlVrow()[i])
                                       .append(",lt=").append(rlStateOut[s * nrl + i] & 7).append(')');
                                }
                            }
                            System.err.printf("GPU_RL_FINAL ctg=%s pinnedTotal=%d%s%n",
                                    contingencies.get(group.get(chunkStart + s)).contingencyId(), pinned, ids);
                        }
                    }
                    if ((rs.hasDs || dsSlackOnly)
                            && GpuAcNewtonSolver.readBatchSolveCount(handle) - solveCountBefore == 1) {
                        recordDistributedActivePower(handle, gpuData, realCount, chunkSize, n, group, chunkStart,
                                dsSlackOnly, rs.dsOvBus, rs.dsOvBaseDelta, pairPhase ? dsAnchorShift : null,
                                pairPhase ? pairSlackAbsorbedSeed : null, pairPhase ? null : preSlackSeed,
                                pairPhase && dgAnchor != null, preDistributedByIndex, distributedByIndex);
                    }
                    for (int s = 0; s < realCount; s++) {
                        double[] row = chunkResults[s];
                        if (rejectAfterSolve != null && rejectAfterSolve[s]) {
                            row[n + 2] = GpuBatchStatus.STATUS_REJECTED;   // the census names the reason where it is decided
                        } else if (pairPhase && (int) row[n + 2] != GpuBatchStatus.STATUS_CONVERGED) {
                            // a pair the device solved and did not converge: a device failure wherever OLF converges it
                            com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.pairDeviceFailed(
                                    contingencies.get(group.get(chunkStart + s)).contingencyId());
                        }
                        if (config.stateSink() != null && (int) row[n + 2] == GpuBatchStatus.STATUS_CONVERGED) {
                            // the state leaves the heap now, as its chunk completes; the result keeps the tail
                            // (iterations, norm, status) - see sunkTail / fullRow
                            config.stateSink().put(contingencies.get(group.get(chunkStart + s)).contingencyId(),
                                    java.util.Arrays.copyOf(row, n));
                            row = new double[] {row[n], row[n + 1], row[n + 2]};
                        }
                        resultsByIndex[group.get(chunkStart + s)] = row;
                        // OLF_GPU_CPU_ROUTED=1: name every scenario the device hands back to the CPU, with
                        // the status that decided it (-2 rejected, -1 diverged, 1 max-iter). The aggregate
                        // "precomputed on device: N / M" says how many, never which.
                        if (System.getenv("OLF_GPU_CPU_ROUTED") != null && chunkResults[s] != null
                                && (int) chunkResults[s][n + 2] != GpuBatchStatus.STATUS_CONVERGED) {
                            System.err.printf("GPU_CPU_ROUTED ctg=%s status=%d%n",
                                    contingencies.get(group.get(chunkStart + s)).contingencyId(),
                                    (int) chunkResults[s][n + 2]);
                        }
                    }
                }
            }
            long elapsed = System.currentTimeMillis() - startTime;
            LOGGER.info("Batched N-1: {} GPU scenarios in {} chunk(s) of <= {} solved in {} ms ({} ms/scenario)",
                    nb, nChunks, chunkSize, elapsed, (double) elapsed / nb);
            if (System.getenv("OLF_GPU_PHASE_TIME") != null) {
                System.err.printf("BATCH_PHASE_TIME total=%d ms | batched main solves=%d ms | "
                        + "outer-loop orchestrators=%d ms | %d scenarios in %d chunk(s) of <= %d%n",
                        elapsed, tMainSolveNs / 1_000_000, tOrchestratorNs / 1_000_000, nb, nChunks, chunkSize);
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
     * the slack its retained share: Σ genCum + retain — the same two read-backs the base load flow writes
     * back as targets ({@code GpuNewtonRaphson.writeBackDistributedSlack}). When no pass fired the
     * distribution is the pre-distribution alone. Slack-only participation keeps everything in the slack's
     * absorbed total. Only valid when ONE solve covered the chunk (the read-backs are per solve): the caller
     * checks; a scenario left NaN reports the pre-distribution as before.
     *
     * <p>A strategy pair reports what OLF's action simulation reports: the distribution of the strategy
     * load flow alone, which starts from the parent contingency's distributed targets and has no loss
     * pre-distribution. The pair phase re-anchors the water-fill and the slack target at the parent
     * ({@code anchorShift}) and seeds the slack-only accumulator with the parent's total
     * ({@code slackAbsorbedSeed}); both are taken back out here. Null outside the pair phase.
     *
     * <p>{@code preSlackSeed}: a contingency that leaves only the slack bus's generators participating has
     * its loss pre-distribution on them, carried as a shift of the slack target the device's retain is
     * measured from; it is added back. Null when no scenario of the chunk has one.
     */
    private static void recordDistributedActivePower(long handle, GpuAcData gpuData, int realCount, int chunkSize, int n,
                                                     List<Integer> group, int chunkStart, boolean dsSlackOnly,
                                                     int[] dsOvBus, double[] dsOvBaseDelta,
                                                     double[] anchorShift, double[] slackAbsorbedSeed, double[] preSlackSeed,
                                                     boolean genAnchored,
                                                     double[] preDistributedByIndex, double[] distributedByIndex) {
        boolean pair = anchorShift != null || slackAbsorbedSeed != null;
        if (dsSlackOnly) {
            double[] absorbed = GpuAcNewtonSolver.readBatchSlackAbsorbed(handle);
            for (int s = 0; s < realCount && absorbed != null; s++) {
                distributedByIndex[group.get(chunkStart + s)] = absorbed[s] - (slackAbsorbedSeed != null ? slackAbsorbedSeed[s] : 0.0);
            }
            return;
        }
        double[] retain = GpuAcNewtonSolver.readBatchDsRetain(handle);
        if (retain == null) {
            return;
        }
        int ngen = gpuData.dgBus() == null ? 0 : gpuData.dgBus().length;
        double[] genCum = null;
        double[] targets = null;
        for (int s = 0; s < realCount; s++) {
            int idx = group.get(chunkStart + s);
            if (Double.isNaN(retain[s])) {                   // no pass fired: the pre-distribution alone
                distributedByIndex[idx] = pair ? 0.0 : preDistributedByIndex[idx];
                continue;
            }
            double participants = 0;
            // The kernel runs the per-generator water-fill unless a scenario is re-anchored WITHOUT per-generator
            // anchors: then the bus-level loop moves the row targets and leaves genCum at zero (distributeSlackBatch's
            // `anchored` test, on the same values). genCum is measured from each generator's own anchor.
            boolean anchored = false;
            for (int i = 0; anchorShift != null && i < gpuData.dsPhiRow().length && !anchored; i++) {
                anchored = anchorShift[s * gpuData.dsPhiRow().length + i] != 0.0;
            }
            if (ngen > 0 && (!anchored || genAnchored)) {
                if (genCum == null) {
                    genCum = GpuAcNewtonSolver.readBatchDsGenCum(handle);
                }
                if (genCum == null) {
                    continue;
                }
                for (int g = 0; g < ngen; g++) {
                    if (gpuData.dgBus()[g] >= 0) {           // the slack's own generators are in retain
                        participants += genCum[s * ngen + g];
                    }
                }
            } else {                                         // bus-level participants: targets from the anchor
                if (targets == null) {
                    targets = GpuAcNewtonSolver.readBatchTargets(handle);
                }
                int ov = dsOvBus != null && dsOvBus.length > 0 ? dsOvBus[s] : -1;
                for (int i = 0; i < gpuData.dsPhiRow().length; i++) {
                    double anchor = gpuData.dsBaseTargetP()[i] - (ov == i ? dsOvBaseDelta[s] : 0.0)
                            + (anchorShift != null ? anchorShift[s * gpuData.dsPhiRow().length + i] : 0.0);
                    participants += targets[s * n + gpuData.dsPhiRow()[i]] - anchor;
                }
            }
            // a slack-only pass measures the slack's take from the target the pre-distribution moved
            distributedByIndex[idx] = participants + retain[s] + (preSlackSeed != null ? preSlackSeed[s] : 0.0);
        }
    }

}
