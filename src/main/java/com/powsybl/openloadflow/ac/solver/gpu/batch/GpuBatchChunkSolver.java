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

    static void solveQualifiedInChunks(List<Integer> batchedIndices, List<BatchContingency> contingencies,
                                               GpuBatchElementMapper elementMapper, GpuAcData gpuData,
                                               double[] baseState, double[] baseTargets, int[] baseRowMode,
                                               int[][] frozenRowsByIndex, double[][] frozenTargetsByIndex,
                                               GenOutage[][] genOutagesByIndex, double[] activePowerLossByIndex,
                                               java.util.Map<com.powsybl.openloadflow.network.LfHvdc, Integer> hvdcElemIndex,
                                               double[][] rowCriteria, GpuBatchRlCurves.Channel rlCurves,
                                               int[][] dqOvContribByIndex, double[][] dqOvWByIndex,
                                               int[][] drOvContribByIndex, double[][] drOvCByIndex,
                                               int[][] islandedDsByIndex, int[][] islandedRlByIndex,
                                               List<com.powsybl.openloadflow.network.LfHvdc>[] islandedHvdcsByIndex,
                                               int n, int nElements, NewtonRaphsonParameters parameters,
                                               double tol, double reactiveLimitTol, double slackDistTol,
                                               double convEpsPerEq, double[][] resultsByIndex, int[] varType,
                                               GpuBatchPhases.Built phases,
                                               List<GpuBatchHvdcAcEmulation.Ctl> hvdcControllers,
                                               BatchConfig config) {
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
                solveQualifiedInChunksAt(chunkSize, remaining, contingencies, elementMapper, gpuData, baseState,
                        baseTargets, baseRowMode, frozenRowsByIndex, frozenTargetsByIndex, genOutagesByIndex,
                        activePowerLossByIndex, hvdcElemIndex, rowCriteria, rlCurves,
                        dqOvContribByIndex, dqOvWByIndex, drOvContribByIndex, drOvCByIndex, islandedDsByIndex,
                        islandedRlByIndex, islandedHvdcsByIndex,
                        n, nElements, parameters, tol, reactiveLimitTol, slackDistTol, convEpsPerEq, resultsByIndex,
                        varType, phases, hvdcControllers, config);
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

    private static int countTrue(boolean[] flags) {
        if (flags == null) {
            return 0;
        }
        int n = 0;
        for (boolean f : flags) {
            if (f) {
                n++;
            }
        }
        return n;
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
    private static void solveQualifiedInChunksAt(int chunkCap,
                                               List<Integer> batchedIndices, List<BatchContingency> contingencies,
                                               GpuBatchElementMapper elementMapper, GpuAcData gpuData,
                                               double[] baseState, double[] baseTargets, int[] baseRowMode,
                                               int[][] frozenRowsByIndex, double[][] frozenTargetsByIndex,
                                               GenOutage[][] genOutagesByIndex, double[] activePowerLossByIndex,
                                               java.util.Map<com.powsybl.openloadflow.network.LfHvdc, Integer> hvdcElemIndex,
                                               double[][] rowCriteria, GpuBatchRlCurves.Channel rlCurves,
                                               int[][] dqOvContribByIndex, double[][] dqOvWByIndex,
                                               int[][] drOvContribByIndex, double[][] drOvCByIndex,
                                               int[][] islandedDsByIndex, int[][] islandedRlByIndex,
                                               List<com.powsybl.openloadflow.network.LfHvdc>[] islandedHvdcsByIndex,
                                               int n, int nElements, NewtonRaphsonParameters parameters,
                                               double tol, double reactiveLimitTol, double slackDistTol,
                                               double convEpsPerEq, double[][] resultsByIndex, int[] varType,
                                               GpuBatchPhases.Built phases,
                                               List<GpuBatchHvdcAcEmulation.Ctl> hvdcControllers,
                                               BatchConfig config) {
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
        long handle = GpuBatchContextCache.acquireBatchContext(gpuData, chunkSize, GpuBatchContextCache.resolveBatchMode(), n);
        // PER_EQUATION_TYPE criterion: install the per-row thresholds on the (possibly cached)
        // context — or CLEAR a previous run's thresholds under UNIFORM_CRITERIA.
        GpuAcNewtonSolver.setBatchRowCriteria(handle,
                rowCriteria != null ? rowCriteria[0] : null, rowCriteria != null ? rowCriteria[1] : null);
        // The run-constant outer-loop tables, shared with the base load flow's batch-of-one.
        GpuBatchOuterLoopTables.install(handle, gpuData, rlCurves,
                config.rlRobustMode(), config.minRealisticVoltage(), config.maxRealisticVoltage(),
                config.minNominalVoltageRealisticVoltageCheck(), config.realisticFixingLoop());

        // Which reactive-capability CURVE entries hang off each bus P row. Run-constant, and the
        // index a generator outage needs to find the entries it disturbs at its own bus.
        java.util.Map<Integer, int[]> curveEntriesByPRow = new java.util.HashMap<>();
        if (rlCurves != null) {
            java.util.Map<Integer, List<Integer>> acc = new java.util.HashMap<>();
            for (int e = 0; e < rlCurves.cvGenId().length; e++) {
                acc.computeIfAbsent(rlCurves.cvIdx()[e * 4 + 1], k -> new ArrayList<>()).add(e);
            }
            acc.forEach((row, es) -> curveEntriesByPRow.put(row, es.stream().mapToInt(Integer::intValue).toArray()));
        }

        // Incremental TVC: upload controller data to device (run-constant, not per-chunk).
        // IncrementalTransformerVoltageControl as a DRIVER PHASE (OLF order 8), shared controlled
        // buses included (the one-tap-at-a-time round robin runs in the group kernel).
        // Every table upload also CLEARS when the run has none: the context is cached across runs on the
        // same network, and a stale table from a previous run selects the wrong driver phase (a Simple
        // TVC run after an INCREMENTAL one ran LOOP_TVC_KERNEL on its controls — 3.6e-7 kV).
        boolean tvcInKernel = false;
        if (phases.incrementalTvc() != null && !phases.incrementalTvc().isEmpty()) {
            tvcInKernel = GpuBatchIncrementalTvc.uploadToDevice(handle, phases.incrementalTvc(),
                    config.incTvcMaxTapShift());
        } else {
            GpuBatchIncrementalTvc.uploadToDevice(handle, null, 1);
        }
        // The flow-observed incremental loops (RPC at OLF order 9, incremental phase at order 7) as DRIVER PHASES.
        boolean flowInKernel = false;
        if (phases.flow() != null && !phases.flow().isEmpty()) {
            flowInKernel = GpuBatchIncrementalFlowControl.uploadToDevice(handle, phases.flow());
        } else {
            GpuBatchIncrementalFlowControl.uploadToDevice(handle, null);
        }
        // CONTINUOUS PhaseControl as a DRIVER PHASE (OLF order 7); the pre-solve control-ON activation stays host-side.
        boolean cphInKernel = false;
        if (phases.continuousPhase() != null && !phases.continuousPhase().isEmpty()) {
            cphInKernel = GpuBatchContinuousPhase.uploadToDevice(handle, phases.continuousPhase());
        } else {
            GpuBatchContinuousPhase.uploadToDevice(handle, null);
        }
        // IncrementalShuntVoltageControl as a DRIVER PHASE (OLF order 10).
        boolean shuntInKernel = false;
        if (phases.incrementalShunt() != null && !phases.incrementalShunt().isEmpty()) {
            shuntInKernel = GpuBatchIncrementalShuntVc.uploadToDevice(handle, phases.incrementalShunt(),
                    config.incShuntMaxSectionShift());
        } else {
            GpuBatchIncrementalShuntVc.uploadToDevice(handle, null, 3);
        }
        // The DISCRETE TransformerVoltageControl (AFTER_GENERATOR / WITH_GENERATOR, OLF order 8) and the
        // WITH_GENERATOR ShuntVoltageControl (order 10) as DRIVER PHASES: the control-ON entry (OLF's
        // initialize) and the rounding/dispatch checks run in-kernel; the host machines stay for a
        // control the upload refuses (a member without a tap table).
        boolean tvdInKernel = false;
        if ((config.transformerControl() || config.simpleTransformerControl())
                && phases.discreteTvc() != null && !phases.discreteTvc().isEmpty()) {
            tvdInKernel = GpuBatchTransformerControl.uploadToDevice(handle, phases.discreteTvc(),
                    config.simpleTransformerControl() ? 1 : 0, false);
            if (tvdInKernel) {
                // AFTER_GENERATOR: OLF's GeneratorVoltageControlManager freeze + fixTransformerVoltageControls
                GpuBatchTransformerControl.uploadFreeze(handle, config.transformerControl() ? phases.freeze() : null);
            } else if (phases.freeze() != null && phases.freeze().hasCandidates()) {
                throw new com.powsybl.commons.PowsyblException("GPU batch: the transformer voltage control needs the "
                        + "in-kernel phase (generator freeze under the controlled level) but a control has no tap table");
            }
        } else {
            GpuBatchTransformerControl.uploadToDevice(handle, null, 0, false);   // clears a cached context's table
        }
        boolean svdInKernel = false;
        if (config.simpleShuntControl() && phases.simpleShunt() != null && !phases.simpleShunt().isEmpty()) {
            svdInKernel = GpuBatchSimpleShuntVc.uploadToDevice(handle, phases.simpleShunt(), false);
        } else {
            GpuBatchSimpleShuntVc.uploadToDevice(handle, null, false);
        }

        // A control loop the kernel REFUSES must never fall back to an after-the-sweep host orchestrator.
        // Such an orchestrator runs out of OLF's order — after the loops OLF puts at orders 7 to 10 — and
        // two of them in a chain discard each other's target moves (measured 2.4e-1 kV). Un-accelerated is
        // the only honest alternative to wrong, so the whole group goes to the CPU: leaving resultsByIndex
        // null is exactly how the caller routes a contingency to the stock per-contingency path.
        List<String> refusedLoops = new ArrayList<>();
        if (phases.incrementalTvc() != null && !phases.incrementalTvc().isEmpty() && !tvcInKernel) {
            refusedLoops.add("IncrementalTransformerVoltageControl");
        }
        if (phases.flow() != null && !phases.flow().isEmpty() && !flowInKernel) {
            refusedLoops.add("IncrementalTransformerReactivePowerControl/IncrementalPhaseControl");
        }
        if (phases.continuousPhase() != null && !phases.continuousPhase().isEmpty() && !cphInKernel) {
            refusedLoops.add("PhaseControl");
        }
        if (phases.incrementalShunt() != null && !phases.incrementalShunt().isEmpty() && !shuntInKernel) {
            refusedLoops.add("IncrementalShuntVoltageControl");
        }
        if ((config.transformerControl() || config.simpleTransformerControl())
                && phases.discreteTvc() != null && !phases.discreteTvc().isEmpty() && !tvdInKernel) {
            refusedLoops.add("TransformerVoltageControl");
        }
        if (config.simpleShuntControl() && phases.simpleShunt() != null && !phases.simpleShunt().isEmpty() && !svdInKernel) {
            refusedLoops.add("ShuntVoltageControl");
        }
        if (!refusedLoops.isEmpty()) {
            LOGGER.warn("GPU batched N-1 disabled for this run: the device cannot own {} — every contingency "
                    + "runs on the CPU rather than through a host orchestrator out of OLF's loop order", refusedLoops);
            return;
        }

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
        // The per-chunk override list AND its row index in one object: a row may carry only one
        // override (the device scatter has no ordering), the lookup that enforces that must not be
        // a linear scan, and the index must never outlive the entries — see GpuBatchOverrides.
        GpuBatchOverrides overrides = new GpuBatchOverrides();
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
        // ctg index -> its FINAL distributed-slack targets at the DS participant rows (the device DS mutates
        // targets in place; read back after each ctg chunk). The pair phase re-anchors its distribution here,
        // mirroring OLF's setGeneratorsInitialTargetPToTargetP before the strategy re-solve.
        int nds = gpuData.dsPhiRow().length;
        java.util.Map<Integer, double[]> dsTargetsByIndex = new java.util.HashMap<>();
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
            if (remote) {
                double tgt = baseTargets[vrow];
                baseRlSeed[i] = baseRowMode[vrow] == 4
                        ? (Math.abs(tgt - (gpuData.rlMaxQ()[i] - gpuData.rlLoadQ()[i]))
                           <= Math.abs(tgt - (gpuData.rlMinQ()[i] - gpuData.rlLoadQ()[i])) ? 2 : 1)
                        : 0;
            } else {
                baseRlSeed[i] = baseRowMode[vrow] == 1 ? (baseState[vrow] < gpuData.rlVtarget()[i] ? 2 : 1) : 0;
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
            for (List<Integer> group : List.of(ctgIdxList, pairIdxList)) {
                boolean pairPhase = group == pairIdxList;
                int nbg = group.size();
                for (int chunkStart = 0; chunkStart < nbg; chunkStart += chunkSize) {
                    int realCount = Math.min(chunkSize, nbg - chunkStart);
                    overrides.clear();
                    // fixTransformerVoltageControls is a POST-CONTINGENCY check on the CPU, so its
                    // connectivity masks are per scenario and belong to the chunk, not to the run.
                    if (tvdInKernel && config.transformerControl() && phases.freeze() != null && phases.freeze().hasFix()) {
                        List<BatchContingency> slotCtgs = new ArrayList<>(realCount);
                        for (int s = 0; s < realCount; s++) {
                            slotCtgs.add(contingencies.get(group.get(chunkStart + s)));
                        }
                        GpuBatchTransformerControl.uploadFixMasks(handle, phases.freeze(), slotCtgs, chunkSize);
                    }
                    // {slot, participant} pairs: the distributed-slack participants this scenario
                    // strands in another synchronous component (see distributeSlackBatch).
                    List<Integer> dsIslandedList = new ArrayList<>();
                    List<Integer> rlIslandedList = new ArrayList<>();
                    // {slot, rlEntry} pairs + {dMinQ, dMaxQ}: the reactive capability a lost generator
                    // takes with it, at a bus that still controls voltage.
                    List<Integer> rlQOvSlot = new ArrayList<>();
                    List<Double> rlQOvVal = new ArrayList<>();
                    // {slot, curveEntry} pairs + {pStep, shareScale}: how a generator outage repairs
                    // the reactive-capability CURVE channel at its own bus (shareScale < 0 = the entry
                    // is the lost generator's own and stops evaluating).
                    List<Integer> cvOvSlot = new ArrayList<>();
                    List<Double> cvOvVal = new ArrayList<>();
                    // {slot, dqMember} pairs: a DISTR_Q controller that stops controlling in this
                    // scenario. Its key has to leave the renormalization denominator on the TARGET
                    // side, which the qPct override channel does not reach.
                    List<Integer> dqInactList = new ArrayList<>();
                    // slot -> the RL entries of the controllers still holding a deactivated pilot's
                    // voltage. All pinned = its Q balance lost its row; the CPU replays that scenario.
                    java.util.Map<Integer, int[]> pilotPinCheck = new java.util.HashMap<>();
                    List<Integer> dqOvIdxList = new ArrayList<>();   // {s, contrib} pairs (DISTR_Q reweight)
                    List<Double> dqOvWList = new ArrayList<>();
                    List<Integer> drOvIdxList = new ArrayList<>();   // {s, contrib} pairs (DISTR_RHO renorm, plan A5)
                    List<Double> drOvCList = new ArrayList<>();
                    // per-scenario ACTIVE equation counts (L2 threshold parity — the CPU removes an
                    // equation per frozen islanded row and per outaged element carrying a variable)
                    int[] nActive = new int[chunkSize];
                    java.util.Arrays.fill(nActive, n);
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
                    double[] preLossBySlot = new double[chunkSize];   // scheduled active power loss per slot (plain phase)
                    for (int s = 0; s < chunkSize; s++) {
                        // Clear the WHOLE slot, not just the outage this slot carried. The chunk
                        // loop reuses one disabledMask per slot, and it used to be reset by replaying
                        // disabledIdx — which records only the contingency's own outaged elements. But
                        // the HVDC machinery also sets bits here (seedBaseStatuses marks a non-LINEAR
                        // entry's hv element, and the freeze/status paths follow it), and those were
                        // never recorded. They therefore SURVIVED into the next chunk, leaving an hv
                        // element disabled for whatever unrelated scenario landed in that slot.
                        //
                        // The result was a scenario-dependent-on-its-predecessors batch: correct when
                        // run alone, wrong after any preceding chunk. Measured on rte6515 at 9042
                        // branch contingencies: 18 differing violation lines against the CPU, and the
                        // affected set changed with the chunk size (64 -> 140 lines, 1024 -> 10)
                        // because the chunking decides which scenario inherits which stale bit. With
                        // the full clear: ZERO. A slot is nElements booleans; the fill costs nothing
                        // measurable (9042 contingencies: 82.6 s either way).
                        java.util.Arrays.fill(disabledMask[s], false);
                        disabledIdx[s] = null;
                        for (int e : gpuData.baseOpenCbElements()) {
                            disabledMask[s][e] = true;           // base-open elements stay OUT unless this scenario recloses them
                        }
                        rowModePerScenario[s] = baseRowMode;     // default: base layout (padding + non-fragmenting)
                        if (s < realCount) {                 // real scenario: disable its outaged branch(es)
                            int ci = group.get(chunkStart + s);
                            if (!pairPhase && activePowerLossByIndex != null) {
                                preLossBySlot[s] = activePowerLossByIndex[ci];
                            }
                            if (dqOvContribByIndex != null && dqOvContribByIndex[ci] != null) {
                                for (int k = 0; k < dqOvContribByIndex[ci].length; k++) {
                                    dqOvIdxList.add(s);
                                    dqOvIdxList.add(dqOvContribByIndex[ci][k]);
                                    dqOvWList.add(dqOvWByIndex[ci][k]);
                                }
                            }
                            if (drOvContribByIndex != null && drOvContribByIndex[ci] != null) {
                                for (int k = 0; k < drOvContribByIndex[ci].length; k++) {
                                    drOvIdxList.add(s);
                                    drOvIdxList.add(drOvContribByIndex[ci][k]);
                                    drOvCList.add(drOvCByIndex[ci][k]);
                                }
                            }
                            if (islandedDsByIndex != null && islandedDsByIndex[ci] != null) {
                                for (int participant : islandedDsByIndex[ci]) {
                                    dsIslandedList.add(s);
                                    dsIslandedList.add(participant);
                                }
                            }
                            if (islandedRlByIndex != null && islandedRlByIndex[ci] != null) {
                                for (int controller : islandedRlByIndex[ci]) {
                                    rlIslandedList.add(s);
                                    rlIslandedList.add(controller);
                                }
                            }
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
                            int[] eis = new int[branches.size() + shunts.size() + hvdcs.size() + sideOpens.size()];
                            int ei = 0;
                            for (BatchContingency.SideOpen so : sideOpens) {
                                // OLF's AC path treats a side-opened branch as FULLY out (measured:
                                // a bus loss is bit-identical to the full outage of its incident
                                // branches — the dangling half carries no model); mask the element.
                                int cbIdx2 = elementMapper.getElementIndex(so.branch());   // qualification guarantees >= 0
                                disabledMask[s][cbIdx2] = true;
                                eis[ei++] = cbIdx2;
                            }
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
                            for (com.powsybl.openloadflow.network.LfHvdc hvdc : hvdcs) { // AC-emulation HVDC outage
                                int idx = hvdcElemIndex.get(hvdc);                       // composition guarantees packed
                                disabledMask[s][idx] = true;
                                eis[ei++] = idx;
                            }
                            disabledIdx[s] = eis;
                            nActive[s] -= frozenRowsByIndex[ci].length;  // islanded buses' equations leave the CPU index
                            for (LfBranch branch : branches) {           // var-carrying outaged branches: the CPU drops
                                int eIdx = elementMapper.getElementIndex(branch);   // the rho/a1 identity equations too
                                if (eIdx >= 0 && eIdx < elementMapper.getClosedBranchCount()) {
                                    if (gpuData.cbIdx()[eIdx * 12 + 9] >= 0) {
                                        nActive[s]--;        // BRANCH_RHO1 variable + its identity leave the index
                                    }
                                    if (gpuData.cbIdx()[eIdx * 12 + 8] >= 0) {
                                        nActive[s]--;        // BRANCH_ALPHA1 variable + its identity leave the index
                                    }
                                }
                            }
                            int[] frozen = frozenRowsByIndex[ci];
                            if (frozen.length > 0) {         // fragmenting outage: freeze the islanded buses' rows
                                int[] row = baseRowMode.clone();
                                double[] frozenTargets = frozenTargetsByIndex[ci];
                                for (int k = 0; k < frozen.length; k++) {
                                    row[frozen[k]] = 0;      // identity row — bus decoupled from the main component
                                    // ...held at a CLEAN flat state (V=1, phi=0) via the target override, NOT the base
                                    // equation target (a P/Q value that gives a nonsensical islanded voltage/angle).
                                    overrides.add(s * n + frozen[k], frozenTargets[k]);
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
                                overrides.add(s * n + dpRow, 0.0);
                                overrides.add(s * n + dqRow, 0.0);
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
                                    if (go.pRow() >= 0) {    // pRow < 0: a row-mode-only entry (a voltage
                                        overrides.add(s * n + go.pRow(), go.pTarget());   // control the island killed)
                                    }
                                    if (go.qOverride()) {    // Q target sheds the lost gens' Q: on the PV->PQ flip AND
                                        // on an ALREADY-PQ bus losing fixed-Q gens (#21)
                                        overrides.add(s * n + go.vRow(), go.qTarget());
                                    }
                                    if (go.toPq()) {
                                        int[] row = rowModePerScenario[s] == baseRowMode ? baseRowMode.clone() : rowModePerScenario[s];
                                        row[go.vRow()] = 1;  // BUS_V row becomes the reactive-power equation (PV→PQ)
                                        rowModePerScenario[s] = row;
                                    }
                                    if (go.silencesReactiveLimitEntry()) {
                                        // Nothing left to control voltage at that bus: mask its
                                        // reactive-limit entry for this scenario, the same channel a
                                        // contingency-stranded controller uses. A bus the BASE loadflow
                                        // pinned PQ otherwise keeps a seeded limit type and can switch
                                        // BACK to PV with its only generator gone.
                                        rlIslandedList.add(s);
                                        rlIslandedList.add(go.rlEntry());
                                    }
                                    if (go.deactivatesDqMember()) {
                                        dqInactList.add(s);
                                        dqInactList.add(go.inactiveDqMember());
                                    }
                                    if (go.needsPilotPinCheck()) {
                                        pilotPinCheck.put(s, go.pilotGroupSurvivorRlEntries());
                                    }
                                    if (go.shrinksReactiveLimits()) {
                                        // The bus KEEPS voltage control, so its reactive-limit entry is
                                        // still read — and the device's aggregate still holds the lost
                                        // generator's capability, which OLF's bus.getMinQ() does not.
                                        rlQOvSlot.add(s);
                                        rlQOvSlot.add(go.rlEntry());
                                        rlQOvVal.add(go.minQDelta());
                                        rlQOvVal.add(go.maxQDelta());
                                        rlQOvVal.add(go.loadQDelta());
                                    }
                                    if (go.pRow() >= 0 && curveEntriesByPRow.containsKey(go.pRow())) {
                                        // The CURVE channel reconstructs each generator's OWN active
                                        // power from the bus row target, to re-evaluate its capability
                                        // curve there. A generator loss skews that reconstruction three
                                        // ways, and all three are repaired here, per scenario:
                                        //  - the lost generator's entry must stop evaluating (its base
                                        //    capability is already taken off by minQDelta/maxQDelta);
                                        //    a pure LOAD loss loses nobody and only needs the second;
                                        //  - the step the outage itself puts in the row target is not
                                        //    distributed-slack power and must move nobody's curve;
                                        //  - the survivors now share the bus's slack over their own
                                        //    factors alone, so each one's share renormalizes.
                                        for (int e : curveEntriesByPRow.get(go.pRow())) {
                                            double busFactor = rlCurves.cvBusFactor()[e];
                                            double left = busFactor - go.lostFactorSum();
                                            boolean dead = go.lostGenIds().contains(rlCurves.cvGenId()[e]);
                                            cvOvSlot.add(s);
                                            cvOvSlot.add(e);
                                            cvOvVal.add(go.pStep());
                                            cvOvVal.add(dead || left <= 0 ? -1 : busFactor / left);
                                        }
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
                    double[] dsAnchorShift = null;
                    double[] dsSlackShift = null;
                    if (!pairPhase && (hasDs || dsSlackOnly) && config.contingencyLossPreDistribution()) {
                        // CPU parity (DefaultContingencyActivePowerLossDistribution): with DS on, the
                        // contingency's SCHEDULED lost active power is distributed ONCE over the
                        // participating generators BEFORE the post-contingency solve — the CPU's DS loop
                        // then only measures the loss-nonlinearity residual (usually inside the band, so
                        // it never redistributes). Replay the same one-shot water-fill host-side
                        // (GpuBatchLossPreDistribution) and hand it to the device as the strategy-pair
                        // trio: participant target overrides + anchor shifts + the slack's own share.
                        for (int s = 0; s < realCount; s++) {
                            if (preLossBySlot[s] == 0) {
                                continue;
                            }
                            if (!hasDs) {
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
                                dsSlackShift[s] = preLossBySlot[s];
                                if (System.getenv("OLF_GPU_PRELOSS_TRACE") != null) {
                                    System.err.printf("PRELOSS_SLACKONLY ctg=%s loss=%s slackBase=%s -> slackTarget=%s%n",
                                            contingencies.get(group.get(chunkStart + s)).contingencyId(),
                                            Double.toString(preLossBySlot[s]),
                                            Double.toString(gpuData.dsSlackTargetP()),
                                            Double.toString(gpuData.dsSlackTargetP() + preLossBySlot[s]));
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
                            double[] shares = GpuBatchLossPreDistribution.distribute(gpuData, preLossBySlot[s],
                                    dsOvBus.length > 0 ? dsOvBus[s] : -1,
                                    dsOvFactorDelta.length > 0 ? dsOvFactorDelta[s] : 0,
                                    dsOvBaseDelta.length > 0 ? dsOvBaseDelta[s] : 0,
                                    dsOvMaxDelta.length > 0 ? dsOvMaxDelta[s] : 0,
                                    dsOvMinDelta.length > 0 ? dsOvMinDelta[s] : 0, islandedDs);
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
                                        contingencies.get(ci).contingencyId(), countTrue(islandedDs));
                                System.err.printf("PRELOSS_FACTORS ctg=%s factorSumAll=%.6f factorSumIslanded=%.6f "
                                        + "(%.2f%% of the participation is in the island)%n",
                                        contingencies.get(ci).contingencyId(), fsAll, fsIslanded,
                                        fsAll > 0 ? 100.0 * fsIslanded / fsAll : 0.0);
                                System.err.printf("PRELOSS ctg=%s mismatch=%.6f pu absorbed=%.6f pu slackShare=%.6f "
                                        + "islandedDsParticipants=%d%n",
                                        contingencies.get(ci).contingencyId(), preLossBySlot[s], absorbed,
                                        shares == null ? 0.0 : shares[nds],
                                        countTrue(islandedDs));
                            }
                            if (shares == null) {
                                continue;
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
                                int flat = s * n + gpuData.dsPhiRow()[i];
                                // compose with a gen-outage override on the same row
                                overrides.compose(flat, baseTargets[gpuData.dsPhiRow()[i]], shares[i]);
                            }
                        }
                    }
                    // CONTINUOUS phase control: OLF's initialize() enables the CONTROLLER-mode controls
                    // BEFORE the first solve. This is the loop's initialize(), not an orchestrator — the
                    // in-kernel phase expects the activation to already carry it.
                    List<Integer> prePcOvIdx = null;
                    List<Double> prePcOvVal = null;
                    if (phases.continuousPhase() != null && !phases.continuousPhase().isEmpty()) {
                        prePcOvIdx = new ArrayList<>(overrides.indexList());
                        prePcOvVal = new ArrayList<>(overrides.valueList());
                        GpuBatchContinuousPhase.applyPhaseOn(realCount, n, phases.continuousPhase(),
                                rowModePerScenario, baseRowMode, disabledMask, overrides.mutableIndexList(), overrides.mutableValueList());
                    }
                    int[] targetOverrideIndex = overrides.indexArray();
                    double[] targetOverrideValue = overrides.valueArray();
                    // OLF_GPU_TARGET_ROW=<row>: the target the DEVICE actually solves that row against —
                    // the base value and every per-scenario override landing on it. A row whose
                    // converged residual equals minus its OLF target is a row solved against 0.
                    String tgtRow = System.getenv("OLF_GPU_TARGET_ROW");
                    if (tgtRow != null) {
                        int probe = Integer.parseInt(tgtRow);
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
                        } else if (dsSlackOnly) {
                            dsSlackShift = new double[chunkSize];    // absorbed-total seeds (no participant anchors)
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
                                if (pIdx != null && resultsByIndex[pIdx] != null
                                        && (int) resultsByIndex[pIdx][n + 2] == GpuBatchStatus.STATUS_CONVERGED) {
                                    parent = resultsByIndex[pIdx];
                                    parentRl = rlByIndex.get(pIdx);
                                    parentDs = dsTargetsByIndex.get(pIdx);
                                    parentSlackAbs = slackAbsorbedByIndex.get(pIdx);
                                }
                            }
                            for (int i = 0; i < n; i++) {
                                x0Full[s * n + i] = parent != null ? parent[i] : baseState[i];
                            }
                            if (rlStateIn != null) {
                                System.arraycopy(parentRl != null ? parentRl : baseRlSeed, 0, rlStateIn, s * nrl, nrl);
                            }
                            if (dsSlackOnly && dsSlackShift != null && parentSlackAbs != null) {
                                dsSlackShift[s] = parentSlackAbs;    // seed the pair with the parent's absorbed total
                            }
                            if (parentDs != null && dsAnchorShift != null && dsFactorSum > 0) {
                                // parentDs = [absorption per DS row..., parent factor sum] (captured after the
                                // parent's solve). The anchor shift is the parent's PURE ABSORPTION — the part
                                // of the parent's final DS target that its distribution added on top of the
                                // parent's own effective base — which composes with the pair's own dsOvBaseDelta
                                // in the kernel (base_eff = base - ovDelta + shift), so gen-outage parents and
                                // action deltas are both handled without double counting.
                                double absorbedSum = 0;
                                for (int i = 0; i < nds; i++) {
                                    int row = gpuData.dsPhiRow()[i];
                                    double shift = parentDs[i];
                                    dsAnchorShift[s * nds + i] = shift;
                                    absorbedSum += shift;
                                    // Seed the DS-row target at the CPU strategy leg's starting value — the
                                    // parent-absorbed target plus the pair's own composed delta — so the first
                                    // inner solve runs at the CPU leg's starting operating point (losses are
                                    // nonlinear along the path). A pair override on the row (gen action /
                                    // gen-outage pTarget) is base-referenced: add the parent absorption to it.
                                    int flat = s * n + row;
                                    overrides.compose(flat, baseTargets[row], shift);
                                }
                                // the slack's own absorption in the parent's distribution, proportional to the
                                // participants' share at the PARENT's factor sum (reweighted if the parent lost
                                // a participating generator)
                                double parentFs = parentDs[nds];
                                dsSlackShift[s] = parentFs > 0 ? (1.0 - parentFs) / parentFs * absorbedSum : 0.0;
                            }
                        }
                        targetOverrideIndex = overrides.indexArray();
                        targetOverrideValue = overrides.valueArray();
                    }
                    // HVDC AC-emulation: snapshot the pre-freeze override lists (the round rebuild base),
                    // then OLF's FreezingHvdcACEmulation freezes every live hvdc at its scenario's
                    // WARM-START flows BEFORE the first solve.
                    List<Integer> preHvdcOvIdx = null;
                    List<Double> preHvdcOvVal = null;
                    int[][] hvdcStatus = null;
                    double[] hvAppliedD1 = null;
                    double[] hvAppliedD2 = null;
                    if (hvdcControllers != null && !hvdcControllers.isEmpty()) {
                        hvAppliedD1 = new double[chunkSize * hvdcControllers.size()];
                        hvAppliedD2 = new double[chunkSize * hvdcControllers.size()];
                        preHvdcOvIdx = new ArrayList<>(overrides.indexList());
                        preHvdcOvVal = new ArrayList<>(overrides.valueList());
                        if (config.hvdcFreezing()) {
                            hvdcStatus = GpuBatchHvdcAcEmulation.applyFreeze(chunkSize, realCount, n, hvdcControllers,
                                    baseState, x0Full, disabledMask, baseTargets, overrides.mutableIndexList(),
                                    overrides.mutableValueList(), hvAppliedD1, hvAppliedD2);
                            targetOverrideIndex = overrides.indexArray();
                            targetOverrideValue = overrides.valueArray();
                        } else {
                            hvdcStatus = new int[chunkSize][hvdcControllers.size()];
                        }
                        // A NON-LINEAR-at-base hvdc (saturated/frozen base — no longer a whole-run
                        // gate): entries the freeze did not claim keep the base status + hv bit
                        // (no override, baseTargets already carry the base injections); padding
                        // slots get the base bits so they converge at iteration 0.
                        GpuBatchHvdcAcEmulation.seedBaseStatuses(chunkSize, realCount, hvdcControllers,
                                hvdcStatus, disabledMask);
                    }
                    boolean dbgPairRl = pairPhase && nrl > 0 && "1".equals(System.getenv("OLF_GPU_DBG_PAIR_RL"));
                    boolean svcActive = phases.secondaryVc() != null && !phases.secondaryVc().ctls().isEmpty();
                    boolean monActive = phases.monitoring() != null && !phases.monitoring().isEmpty();
                    int[] rlStateOut = (!pairPhase && !pairIdxList.isEmpty() || dbgPairRl || svcActive || monActive
                            || !pilotPinCheck.isEmpty()
                            || System.getenv("OLF_GPU_RL_TRACE") != null) && nrl > 0 ? new int[chunkSize * nrl] : null;
                    // DISTR_Q weight overrides: per-chunk context state (orchestrator re-solves keep
                    // the slot assignment); cleared when the chunk has none.
                    // Context state, like the qPct overrides beside it: a monitoring or secondary-VC
                    // re-solve on this context must not drop it (as a solve PARAMETER it did — the
                    // re-solve passed null and the DISTR_Q target was recomputed with the dead
                    // controller's key back in the denominator, clobbering the right target).
                    GpuAcNewtonSolver.setBatchDqInactiveMembers(handle,
                            dqInactList.isEmpty() ? null : dqInactList.stream().mapToInt(Integer::intValue).toArray());
                    GpuAcNewtonSolver.setBatchDqWeightOverrides(handle,
                            dqOvIdxList.isEmpty() ? null : dqOvIdxList.stream().mapToInt(Integer::intValue).toArray(),
                            dqOvWList.isEmpty() ? null : dqOvWList.stream().mapToDouble(Double::doubleValue).toArray());
                    GpuAcNewtonSolver.setBatchDrCoefOverrides(handle,
                            drOvIdxList.isEmpty() ? null : drOvIdxList.stream().mapToInt(Integer::intValue).toArray(),
                            drOvCList.isEmpty() ? null : drOvCList.stream().mapToDouble(Double::doubleValue).toArray());
                    GpuAcNewtonSolver.setBatchActiveRowCounts(handle, nActive);
                    // Cleared when the chunk strands nobody, so a chunk never inherits the previous
                    // chunk's participant mask through the reused context.
                    GpuAcNewtonSolver.setBatchDsIslanded(handle,
                            dsIslandedList.isEmpty() ? null : dsIslandedList.stream().mapToInt(Integer::intValue).toArray());
                    GpuAcNewtonSolver.setBatchRlIslanded(handle,
                            rlIslandedList.isEmpty() ? null : rlIslandedList.stream().mapToInt(Integer::intValue).toArray());
                    // The two HVDC AC-emulation loops run as DRIVER PHASES, in OLF's own position
                    // between DistributedSlack and ReactiveLimits. Running them instead as host
                    // orchestrators after the whole solve put them out of OLF's order and left one
                    // differing violation line over 2000 rte6515 branch contingencies.
                    byte[] ctgOutOf = null;
                    boolean hasHvdcDriverLoops = hvdcControllers != null && !hvdcControllers.isEmpty();
                    ctgOutOf = GpuBatchHvdcAcEmulation.installDriverLoops(handle,
                            hasHvdcDriverLoops ? hvdcControllers : null, chunkSize, hvdcStatus, disabledMask,
                            hvAppliedD1, hvAppliedD2, config.hvdcLimits());
                    // MonitoringVoltage as a DRIVER PHASE, in OLF's own slot (order 6, between the
                    // HVDC loops and ReactiveLimits) instead of as a host orchestrator after the whole
                    // solve. Run-constant table, but the context is per chunk.
                    boolean monInKernel = monActive;
                    GpuBatchVoltageMonitoring.installDriverLoop(handle, monInKernel ? phases.monitoring() : null);
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
                    // slot and the same decision runs on the host instead. OLF_GPU_SVC_HOST_ORCHESTRATOR=1
                    // falls all the way back to the pre-2026-09-15 after-the-sweep orchestrator for A/B.
                    boolean svcHostFallback = "1".equals(System.getenv("OLF_GPU_SVC_HOST_ORCHESTRATOR"));
                    // Three layers, most faithful first: the IN-KERNEL phase owns the loop; where it refuses
                    // the tables (a zone set wider than its dense system, or an observable on an element the
                    // batch does not pack) the driver PAUSES at the same slot and the same decision runs on
                    // the host; OLF_GPU_SVC_HOST_ORCHESTRATOR=1 falls all the way back to the
                    // pre-2026-09-15 after-the-sweep orchestrator, for A/B only.
                    boolean svcInKernel = svcActive && !svcHostFallback
                            && GpuBatchSecondaryVc.uploadToDevice(handle, phases.secondaryVc());
                    if (!svcInKernel) {
                        GpuBatchSecondaryVc.uploadToDevice(handle, null);   // clear a cached context's table
                    }
                    boolean svcAsDriverPause = svcActive && rlStateOut != null && !svcHostFallback && !svcInKernel;
                    GpuAcNewtonSolver.setBatchHostLoop(handle, svcAsDriverPause ? 1 : 0);
                    long tSolve0 = System.nanoTime();
                    double[][] chunkResults = GpuAcNewtonSolver.solveBatchContext(handle, baseState, baseTargets,
                            rowModePerScenario, baseRowMode, disabledMask, targetOverrideIndex, targetOverrideValue,
                            dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta, dsAnchorShift, dsSlackShift,
                            null, null,
                            rlQOvSlot.stream().mapToInt(Integer::intValue).toArray(),
                            rlQOvVal.stream().mapToDouble(Double::doubleValue).toArray(),
                            cvOvSlot.stream().mapToInt(Integer::intValue).toArray(),
                            cvOvVal.stream().mapToDouble(Double::doubleValue).toArray(),
                                                        parameters.getMaxIterations(), tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                            config.lsMode(), config.maxDv(), config.maxDphi(),
                            config.lsMode() == 2 ? varType : null, x0Full,
                            shOvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                            shOvGbList.stream().mapToDouble(Double::doubleValue).toArray(),
                            rlStateIn, rlStateOut,
                            r1OvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                            r1OvValList.stream().mapToDouble(Double::doubleValue).toArray(),
                            config.maxOuterIterations(), config.maxPqPvSwitch(), config.dsRejectResidualFlag(),
                            // monitoring active: OLF checks VoltageMonitoring BEFORE ReactiveLimits — the main
                            // solve surfaces the DS-stable / pre-RL state; the orchestrator then interleaves
                            // and finishes with the full RL-enabled solves.
                            0,
                            nziBase, ziOnSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                            ziOffSlotElemList.stream().mapToInt(Integer::intValue).toArray());
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
                        while (GpuAcNewtonSolver.readBatchDriverPaused(handle) == GpuAcNewtonSolver.HOST_LOOP_SVC
                                && svcRounds++ < maxSvcRounds) {
                            var dec = GpuBatchSecondaryVc.decideAtPause(handle, chunkSize, realCount, n, nrl,
                                    chunkResults, phases.secondaryVc(), rlStateOut, svcTargetV, baseState);
                            if (dec.targetRows() != null && dec.targetRows().length > 0) {
                                GpuAcNewtonSolver.setBatchTargetDeltas(handle, dec.targetRows(), dec.targetValues());
                            }
                            if (dec.unpinPairs() != null && dec.unpinPairs().length > 0) {
                                GpuAcNewtonSolver.setBatchRlUnpin(handle, dec.unpinPairs(), dec.unpinTargets());
                            }
                            GpuAcNewtonSolver.setBatchDriverResume(handle, dec.hostChanged());
                            chunkResults = GpuAcNewtonSolver.solveBatchContext(handle, baseState, baseTargets,
                                    rowModePerScenario, baseRowMode, disabledMask, targetOverrideIndex, targetOverrideValue,
                                    dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta, dsAnchorShift, dsSlackShift,
                                    dec.rlVtSlotIdx(), dec.rlVtValues(),
                                    rlQOvSlot.stream().mapToInt(Integer::intValue).toArray(),
                                    rlQOvVal.stream().mapToDouble(Double::doubleValue).toArray(),
                                    cvOvSlot.stream().mapToInt(Integer::intValue).toArray(),
                                    cvOvVal.stream().mapToDouble(Double::doubleValue).toArray(),
                                                                        parameters.getMaxIterations(), tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                                    config.lsMode(), config.maxDv(), config.maxDphi(),
                                    config.lsMode() == 2 ? varType : null, x0Full,
                                    shOvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                    shOvGbList.stream().mapToDouble(Double::doubleValue).toArray(),
                                    null, rlStateOut,
                                    r1OvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                    r1OvValList.stream().mapToDouble(Double::doubleValue).toArray(),
                                    config.maxOuterIterations(), config.maxPqPvSwitch(), config.dsRejectResidualFlag(), 0,
                                    nziBase, ziOnSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                    ziOffSlotElemList.stream().mapToInt(Integer::intValue).toArray());
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
                    if (monInKernel) {
                        // Keep ACTIVATIONS meaningful with the loop in the driver: it is the guard
                        // that a case really exercises the automaton, not orchestrator bookkeeping.
                        GpuBatchVoltageMonitoring.ACTIVATIONS.add(
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
                            // Re-derive the host's OVERRIDE LIST and hv bits from those statuses, on
                            // top of the PRE-FREEZE list. Syncing only the bits is not enough: every
                            // orchestrator that runs next rebuilds its targets from this list, so a
                            // stale post-freeze list re-applies a freeze the kernel already lifted
                            // (measured: 566 differing lines over 2000 with only the bits synced).
                            List<Integer> idx = new ArrayList<>(preHvdcOvIdx);
                            List<Double> val = new ArrayList<>(preHvdcOvVal);
                            GpuBatchHvdcAcEmulation.applyStatusOverrides(chunkSize, realCount, n,
                                    hvdcControllers, hvdcStatus, ctgOutOf, disabledMask, baseTargets, idx, val,
                                    hvAppliedD1, hvAppliedD2);
                            overrides.mutableIndexList().clear();
                            overrides.mutableIndexList().addAll(idx);
                            overrides.mutableValueList().clear();
                            overrides.mutableValueList().addAll(val);
                            targetOverrideIndex = overrides.indexArray();
                            targetOverrideValue = overrides.valueArray();
                        }
                    }
                    if (System.getenv("OLF_GPU_ITER_TRACE") != null && !pairPhase) {
                        java.util.Map<Integer, Integer> hist = new java.util.TreeMap<>();
                        for (int s = 0; s < realCount; s++) {
                            hist.merge((int) chunkResults[s][n], 1, Integer::sum);
                        }
                        System.err.println("ITER_HIST main solve, iterations -> scenarios: " + hist);
                        for (int s = 0; s < realCount && realCount <= 4; s++) {
                            System.err.printf("ITER ctg=%s iters=%d%n",
                                    contingencies.get(group.get(chunkStart + s)).contingencyId(),
                                    (int) chunkResults[s][n]);
                        }
                    }
                    if (System.getenv("OLF_GPU_PRELOSS_TRACE") != null && nds > 0 && !pairPhase) {
                        // Did the DEVICE's own distributed-slack pass move power AFTER the host-side
                        // pre-distribution? The CPU's loop should see only the loss-nonlinearity
                        // residual and stay inside the slack band; if ours redistributes again, the
                        // two land at different points inside that band.
                        double[] ft = GpuAcNewtonSolver.readBatchTargets(handle);
                        for (int s = 0; s < realCount; s++) {
                            if (preLossBySlot[s] == 0) {
                                continue;
                            }
                            double moved = 0;
                            for (int i = 0; i < nds; i++) {
                                int row = gpuData.dsPhiRow()[i];
                                moved += ft[s * n + row] - baseTargets[row];
                            }
                            // preLossBySlot is the mismatch the pre-distribution was asked to absorb;
                            // `moved` is where the participants ended up relative to the SAME anchor the
                            // device DS loop resets to, so the delta is what that loop added on top.
                            System.err.printf("PRELOSS_AFTER ctg=%s preLossMismatch=%.6f deviceFinal=%.6f "
                                    + "deltaFromDeviceDsLoop=%.6f pu%n",
                                    contingencies.get(group.get(chunkStart + s)).contingencyId(),
                                    preLossBySlot[s], moved, moved - preLossBySlot[s]);
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
                                System.err.printf("GPU_DS_TARGET ctg=%s i=%d target=%s%n",
                                        id, i, Double.toString(ft[s * n + gpuData.dsPhiRow()[i]]));
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
                        for (int s = 0; s < realCount; s++) {
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
                    if (!pairPhase && nds > 0 && !pairIdxList.isEmpty()) {
                        // Capture each ctg's DS ABSORPTION (final distributed target minus the ctg's own
                        // effective base per DS row) plus its factor sum, for its pairs' re-anchoring —
                        // the device analogue of OLF's setGeneratorsInitialTargetPToTargetP before a
                        // strategy re-solve.
                        double[] finalTargets = GpuAcNewtonSolver.readBatchTargets(handle);
                        for (int s = 0; s < realCount; s++) {
                            double[] slice = new double[nds + 1];
                            double fs = 0;
                            for (int i = 0; i < nds; i++) {
                                int row = gpuData.dsPhiRow()[i];
                                double effBase = baseTargets[row]
                                        - (dsOvBus[s] == i ? dsOvBaseDelta[s] : 0.0);
                                slice[i] = finalTargets[s * n + row] - effBase;
                                fs += gpuData.dsFactor()[i]
                                        - (dsOvBus[s] == i ? dsOvFactorDelta[s] : 0.0);
                            }
                            slice[nds] = fs;
                            dsTargetsByIndex.put(group.get(chunkStart + s), slice);
                        }
                    }
                    // HOST ORCHESTRATORS — in OLF's DefaultAcOuterLoopConfig order.
                    // In-kernel: DS(0), HVDC(1,2), Monitoring(5), RL(6) already ran in the main solve.
                    // Post-RL host orchestrators below: SVC(4), Monitoring(5), Phase(7), TVC(8), TRPC(9), Shunt(10).

                    tOrchestratorNs += System.nanoTime() - tOrch0;
                    if (System.getenv("OLF_GPU_DS_TARGET_DUMP") != null && nds > 0) {
                        // The FINAL distributed target on every participant row, after every
                        // orchestrator. Both arms are converged, so if their reported states differ
                        // their TARGETS must differ — this is the only direct way to see where.
                        double[] ft = GpuAcNewtonSolver.readBatchTargets(handle);
                        for (int s = 0; s < realCount; s++) {
                            String id = contingencies.get(group.get(chunkStart + s)).contingencyId();
                            for (int i = 0; i < nds; i++) {
                                System.err.printf("GPU_DS_TARGET ctg=%s i=%d target=%s%n",
                                        id, i, Double.toString(ft[s * n + gpuData.dsPhiRow()[i]]));
                            }
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
                    // A deactivated pilot keeps its Q balance on a SURVIVING controller's row. If every
                    // one of those controllers pinned, the reactive-limit machinery took their rows for
                    // their own pinned balances and the pilot's balance is no longer enforced anywhere —
                    // the solve still converges, on an equation system missing a row. Whether they pin is
                    // a runtime outcome, so it is caught here and the CPU replays the scenario.
                    if (!pilotPinCheck.isEmpty() && rlStateOut != null) {
                        for (var e : pilotPinCheck.entrySet()) {
                            int s = e.getKey();
                            if (s >= realCount || chunkResults[s] == null) {
                                continue;
                            }
                            boolean allPinned = true;
                            for (int entry : e.getValue()) {
                                if ((rlStateOut[s * nrl + entry] & 7) == 0) {
                                    allPinned = false;
                                    break;
                                }
                            }
                            if (allPinned) {
                                chunkResults[s][n + 2] = GpuBatchStatus.STATUS_REJECTED;
                                com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus
                                        .record("RUNTIME:DISTR_Q_PILOT_ALL_PINNED");
                            }
                        }
                    }
                    for (int s = 0; s < realCount; s++) {
                        resultsByIndex[group.get(chunkStart + s)] = chunkResults[s];
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
}
