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
import com.powsybl.openloadflow.network.GeneratorVoltageControl;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
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

    /** OLF_GPU_X0_SEEDS=0: send islanded scenarios' starting state as full chunkSize * n arrays again (A/B). */
    private static final boolean X0_SEEDS = !"0".equals(System.getenv("OLF_GPU_X0_SEEDS"));

    /** OLF starts EVERY solve with the zero-impedance branch flows at zero: AcSolverUtil.initStateVector
     *  has {@code case DUMMY_P, DUMMY_Q: x[v.getRow()] = 0}, and updateNetwork never writes them back
     *  (they have no network backing). So each post-contingency solve discards the coupler flows and
     *  rebuilds them in its first Newton step - on network.xiidm an entry residual of 14.988 pu, the same
     *  for every contingency, against 0.16 when they are carried. The device starts the same way, so both
     *  arms take the same first step from the same state. */
    private static double[] zeroZiFlows(GpuAcData gpuData, double[] baseState) {
        int[] zi = gpuData.ziIdx();
        if (zi == null || zi.length == 0) {
            return baseState;
        }
        double[] x0 = baseState.clone();
        for (int k = 0; k + 10 < zi.length; k += 11) {
            if (zi[k + 4] >= 0) {
                x0[zi[k + 4]] = 0;                           // DUMMY_P column
            }
            if (zi[k + 5] >= 0) {
                x0[zi[k + 5]] = 0;                           // DUMMY_Q column
            }
        }
        return x0;
    }

    static void solveQualifiedInChunks(List<Integer> batchedIndices, List<BatchContingency> contingencies,
                                               GpuBatchElementMapper elementMapper, GpuAcData gpuData,
                                               double[] baseStateIn, double[] baseTargets, int[] baseRowMode,
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
                                               double convEpsPerEq, double[][] resultsByIndex, double[] preDistributedByIndex,
                                               double[] distributedByIndex, int[] varType, GpuBatchPhases.Built phases,
                                               List<GpuBatchHvdcAcEmulation.Ctl> hvdcControllers,
                                               BatchConfig config) {
        double[] baseState = zeroZiFlows(gpuData, baseStateIn);
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
                        preDistributedByIndex, distributedByIndex, varType, phases, hvdcControllers, config);
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

    /** The DS generator slots one scenario's outage removes, out of the flat (scenario, slot) pairs.
     *  The host seed needs the same identity the kernel has, or the two model the outaged bus
     *  differently and the seed stops being the function it is meant to replay. */
    private static int[] lostSlotsOf(java.util.List<Integer> pairs, int scenario) {
        java.util.List<Integer> out = new java.util.ArrayList<>();
        for (int k = 0; k + 1 < pairs.size(); k += 2) {
            if (pairs.get(k) == scenario) {
                out.add(pairs.get(k + 1));
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
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

    /** Does this scenario's plan already carry a renormalized DISTR_Q target for {@code row}? */
    private static boolean seedsDqTargetRow(GenOutage[] gos, int row) {
        for (GenOutage go : gos == null ? new GenOutage[0] : gos) {
            if (go.seedsDqTargets() && contains(go.dqTargetRows(), row)) {
                return true;
            }
        }
        return false;
    }

    private static boolean contains(int[] values, int v) {
        for (int x : values) {
            if (x == v) {
                return true;
            }
        }
        return false;
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
                                               double convEpsPerEq, double[][] resultsByIndex, double[] preDistributedByIndex,
                                               double[] distributedByIndex, int[] varType, GpuBatchPhases.Built phases,
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
        // MODE: the run's configured batch mode, or the ROBUST single-matrix solve on the retry pass
        // (config.robustSolveMode(), set only by BatchConfig.withRobustSolveMode) — mode 6 is mode 0's
        // per-scenario loop with matching and global column pivoting, which is what rte6515's full
        // busbar structure needs to factor at all. Switching mode EVICTS the single-entry context cache,
        // so the retry pays one cuDSS analysis; that is the price of keeping the scenarios on device.
        long handle = GpuBatchContextCache.acquireBatchContext(gpuData, chunkSize,
                config.robustSolveMode() ? GpuBatchContextCache.MODE_ROBUST : GpuBatchContextCache.resolveBatchMode(), n);
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
                acc.computeIfAbsent(rlCurves.cvIdx()[e * 5 + 1], k -> new ArrayList<>()).add(e);
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
        List<Integer> ziOnSlotElemList = new ArrayList<>();
        List<Integer> ziTreeSlotElemValList = new ArrayList<>();   // {slot, ziElement, isTree} where OLF's tree moved // per-scenario zi-element enables (switch closes)
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
                    // {slot, participant, row} triples: a DS participant whose P-balance row this
                    // scenario MOVES. Under the coupler-tree relocation a child bus's P balance lives on
                    // its parent coupler's DUMMY_P row, and dsPhiRow — a STRUCTURAL array — points there.
                    // A contingency that opens that coupler pins the dummy row to a DUMMY_TARGET identity
                    // and the stock-placement fallback puts the balance back on the child's OWN phi row,
                    // so for THAT scenario the participant's row is the child's. Without this the whole DS
                    // machinery kept writing to the pinned row: measured on the middle-section fixture,
                    // g2's 0.173869 pu share landed on a dead row while both DS kernels read its
                    // (0 - anchor) as -1.043213 of "absorbed", so the slack loop fired four times where
                    // OLF fires once and over-placed 0.0687 pu on the rows that still worked (0.34 kV on
                    // vl3_b). Per chunk, like every other list here — the index is a chunk SLOT.
                    List<Integer> dsRowOvList = new ArrayList<>();
                    // ...and the same diversion host-side, for the loss pre-distribution below (it runs
                    // after this loop and composes its shares onto the participants' rows).
                    int[][] dsRowOvBySlot = new int[chunkSize][];
                    List<Integer> rlIslandedList = new ArrayList<>();
                    List<Integer> rlLocalList = new ArrayList<>();     // (slot, entry): re-homed PV buses checked as LOCAL
                    List<Double> rlLocalVt = new ArrayList<>();        // their own voltage target (pu)
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
                    // PER CHUNK, like dqInactList above: the scenario index in every one of these is a
                    // SLOT in this chunk, so a list that outlives the chunk applies chunk N's entries to
                    // chunk N+1's slots. That is invisible to any validation run in windows of one chunk.
                    List<Integer> dsLostGenList = new ArrayList<>();
                    List<Integer> frozenSeedSlot = new ArrayList<>();
                    List<Integer> frozenSeedRow = new ArrayList<>();
                    List<Double> frozenSeedValue = new ArrayList<>();
                    // slot -> the RL entries of the controllers still holding a deactivated pilot's
                    // voltage. All pinned = its Q balance lost its row; the CPU replays that scenario.
                    java.util.Map<Integer, int[]> pilotPinCheck = new java.util.HashMap<>();
                    // Pilot-row flip entries: the scenarios that used to be REJECTED by that check are
                    // FIXED in the solve instead. Per chunk, like every other list here — the scenario
                    // index is a chunk SLOT.
                    List<Integer> dqTqOvKey = new ArrayList<>();     // slot*nDqMembers + member
                    List<Double> dqTqOvVal = new ArrayList<>();
                    List<Integer> pfSlot = new ArrayList<>();
                    List<Integer> pfRow = new ArrayList<>();
                    List<Double> pfQTarget = new ArrayList<>();
                    List<Integer> pfRlCount = new ArrayList<>();
                    List<Integer> pfRlPool = new ArrayList<>();
                    List<Integer> dqOvIdxList = new ArrayList<>();   // {s, contrib} pairs (DISTR_Q reweight)
                    List<Double> dqOvWList = new ArrayList<>();
                    List<Integer> drOvIdxList = new ArrayList<>();   // {s, contrib} pairs (DISTR_RHO renorm, plan A5)
                    List<Double> drOvCList = new ArrayList<>();
                    // per-scenario ACTIVE equation counts (L2 threshold parity — the CPU removes an
                    // equation per frozen islanded row and per outaged element carrying a variable)
                    int[] nActive = new int[chunkSize];
                    java.util.Arrays.fill(nActive, n);
                    ziOffSlotElemList.clear();
                    ziTreeSlotElemValList.clear();
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
                    // ...and what the pre-distribution put on the PARTICIPANTS alone (the slack keeps the
                    // rest). PRELOSS_AFTER has to compare the device's final participant total against
                    // THIS, not against the whole mismatch: with a participating slack the two differ by
                    // the slack's retained share, which is not a device defect.
                    double[] preAbsorbedBySlot = new double[chunkSize];
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
                                // A side-opened branch is NOT fully out. OLF keeps it DANGLING, and the
                                // dangling half's charging still injects at the connected bus — OLF's own
                                // testLineOpenOneSideContingency measures -55.873 MVAr on exactly this
                                // shape. Masking the element deletes that injection: measured on rte6515
                                // BUS-1003_BBS as EXACTLY -2 x b_side x v^2 at every bus bordering the
                                // opened side (six buses, ratios -1.994..-2.004), and on the 9-bus
                                // reproducer as VL2 374.62 kV against the CPU's 379.85.
                                //
                                // A branch open at one end is, seen from the live bus, a PURE SHUNT: the
                                // far side's series+shunt reduces to one equivalent admittance. So keep
                                // the element LIVE and override its pi row to that shunt — y = 0 kills the
                                // series coupling (so the frozen far bus contributes nothing and receives
                                // nothing), and the connected side's g/b carry the equivalent. No new
                                // kernel: this is the per-scenario pi-model override the tap channel
                                // already uses, and the cb evaluation reads it through findTapOverride.
                                double[] cbRow = GpuBatchSideOpen.danglingCbRow(so.branch(), so.side());
                                if (cbRow == null) {         // not expressible -> keep the old behaviour
                                    int cbIdx2 = elementMapper.getElementIndex(so.branch());
                                    disabledMask[s][cbIdx2] = true;
                                    eis[ei++] = cbIdx2;
                                } else {
                                    int soElem = elementMapper.getElementIndex(so.branch());
                                    if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                                        // Does this branch also feed a DISTR_Q row? fillDistrQBatch reads
                                        // cbIn with NO r1Ov parameters, so it cannot see the per-scenario
                                        // pi override: a side-opened branch that contributes to a remote
                                        // voltage-control group's reactive distribution would be counted
                                        // there with its FULL series admittance instead of the dangling
                                        // equivalent.
                                        int hits = 0;
                                        for (int k = 0; k < gpuData.dqElem().length; k++) {
                                            if (gpuData.dqElem()[k] == soElem) {
                                                hits++;
                                            }
                                        }
                                        var sopi = so.branch().getPiModel();
                                        System.err.printf("SIDEOPEN_DQ ctg=%s branch=%s side=%d elem=%d"
                                                        + " dqContribs=%d base(g1=%.9e b1=%.9e g2=%.9e b2=%.9e)"
                                                        + " eq(g1=%.9e b1=%.9e g2=%.9e b2=%.9e)%n",
                                                contingencies.get(ci).contingencyId(), so.branch().getId(),
                                                so.side(), soElem, hits,
                                                sopi.getG1(), sopi.getB1(), sopi.getG2(), sopi.getB2(),
                                                cbRow[1], cbRow[3], cbRow[2], cbRow[4]);
                                    }
                                    r1OvSlotElemList.add(s);
                                    r1OvSlotElemList.add(soElem);
                                    for (double v : cbRow) {
                                        r1OvValList.add(v);
                                    }
                                }
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
                            if (System.getenv("OLF_GPU_SA_RESID") != null) {
                                System.err.printf("DEV_NACTIVE ctg=%s n=%d frozen=%d nActive=%d%n",
                                        contingencies.get(ci).contingencyId(), n,
                                        frozenRowsByIndex[ci].length, nActive[s]);
                            }
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
                            // Rows that host a PILOT'S relocated reactive balance this scenario (the carrier a frozen base
                            // carrier handed it to): a DISTR_Q target seed must not land on them, see the seed loop below.
                            gnu.trove.set.hash.TIntHashSet pilotBalanceRows = new gnu.trove.set.hash.TIntHashSet();
                            int[] frozen = frozenRowsByIndex[ci];
                            if (frozen.length > 0) {         // fragmenting outage: freeze the islanded buses' rows
                                int[] row = baseRowMode.clone();
                                double[] frozenTargets = frozenTargetsByIndex[ci];
                                for (int k = 0; k < frozen.length; k++) {
                                    row[frozen[k]] = 0;      // identity row — bus decoupled from the main component
                                    // ...held at a CLEAN flat state (V=1, phi=0) via the target override, NOT the base
                                    // equation target (a P/Q value that gives a nonsensical islanded voltage/angle).
                                    overrides.add(s * n + frozen[k], frozenTargets[k]);
                                    // ...and START that row AT its target. It is an identity row, so its
                                    // solution IS the target and seeding it costs nothing — but leaving it
                                    // at the base state hands the convergence test a residual OLF never
                                    // sees, because OLF REMOVES an islanded bus's equations rather than
                                    // freezing them. Measured on rte6515 LINE-4413-5: the islanded bus's
                                    // angle row sits at -1.050832654 rad against a frozen target of 0, and
                                    // that 1.050832654 IS the device's whole initial |F|inf for the solve,
                                    // against OLF's 0.1296557068. The device then needs an extra Newton
                                    // iteration, lands at 1.68e-06 where OLF stops at 8.04e-04, and the
                                    // distributed-slack loop reads a slack mismatch 7.43e-04 different —
                                    // which is the entire branch-family error (8.328e-05 kV).
                                    frozenSeedSlot.add(s);
                                    frozenSeedRow.add(frozen[k]);
                                    frozenSeedValue.add(frozenTargets[k]);
                                    if (System.getenv("OLF_GPU_SA_RESID") != null) {
                                        System.err.printf("FROZEN ctg=%s row=%d target=%.9f base=%.9f |F0|=%.9f%n",
                                                contingencies.get(ci).contingencyId(), frozen[k], frozenTargets[k],
                                                baseState[frozen[k]], Math.abs(baseState[frozen[k]] - frozenTargets[k]));
                                    }
                                }
                                // A frozen row that was a remote gen-VC group's BASE CARRIER (the free
                                // member's row hosting the pilot's reactive balance): the device's mode scan
                                // moves that balance to the next live member's row from the first Newton
                                // solve on, but that row's TARGET is its own DISTR_Q one (0) - the kernel
                                // that stamps the pilot's target on the carrier runs only inside the
                                // outer-loop passes, and the host re-applies its overrides at every pass.
                                // Stamp it here, by the kernel's own rule (relocChoose: first candidate
                                // whose mode is neither identity nor pinned). Measured on the remote-control
                                // fixture, b1 islanded: OLF's BUS_TARGET_Q at the pilot off by exactly its
                                // load, b2 0.45 kV; rte6515: five busbar sections up to 1.3 kV.
                                int[] qrPtr = gpuData.qrPtr();
                                int[] qrRows = gpuData.qrRows();
                                for (int g = 0; qrPtr != null && g + 1 < qrPtr.length; g++) {
                                    int baseCarrier = -1;
                                    int newCarrier = -1;
                                    for (int c = qrPtr[g] + 1; c < qrPtr[g + 1]; c++) {
                                        int r = qrRows[c];
                                        if (baseCarrier < 0 && baseRowMode[r] != 0 && baseRowMode[r] != 4) {
                                            baseCarrier = r;
                                        }
                                        if (newCarrier < 0 && row[r] != 0 && row[r] != 4) {
                                            newCarrier = r;
                                        }
                                    }
                                    // A PILOT-SELF group (rotation: the fallback row is itself a candidate) whose pilot is
                                    // UNPINNED in the base carries no pilot balance on its carrier: that row hosts its OWN
                                    // DISTR_Q, and the base carrier's target is that member's DISTR_Q target, not a pilot
                                    // Q — stamping it on the next member overwrote the renormalized target the outage
                                    // seeding had just put there (middle-section fixture: −0.1333 over −0.2, bus-mid
                                    // 0.32 kV off). Only a PINNED pilot's balance travels with the carrier.
                                    boolean unpinnedPilotSelf = false;
                                    int fallbackRow = qrRows[qrPtr[g]];
                                    for (int c = qrPtr[g] + 1; c < qrPtr[g + 1]; c++) {
                                        if (qrRows[c] == fallbackRow) {
                                            unpinnedPilotSelf = true;
                                            break;
                                        }
                                    }
                                    if (unpinnedPilotSelf && gpuData.rlVrow() != null && gpuData.rlRemote() != null) {
                                        for (int e = 0; e < gpuData.rlVrow().length; e++) {
                                            if (gpuData.rlVrow()[e] == fallbackRow && gpuData.rlRemote()[e] == 2) {
                                                byte lt = gpuData.rlBaseQLimitType() != null ? gpuData.rlBaseQLimitType()[e] : 0;
                                                unpinnedPilotSelf = lt < 1 || lt > 4;
                                                break;
                                            }
                                        }
                                    }
                                    if (baseCarrier >= 0 && newCarrier >= 0 && newCarrier != baseCarrier
                                            && row[baseCarrier] == 0 && baseRowMode[baseCarrier] != 0 && !unpinnedPilotSelf) {
                                        overrides.add(s * n + newCarrier, baseTargets[baseCarrier]);
                                        pilotBalanceRows.add(newCarrier);
                                        if (System.getenv("OLF_GPU_SA_RESID") != null) {
                                            System.err.printf("CARRIER_MOVED ctg=%s group=%d from=%d to=%d pilotQ=%.9f%n",
                                                    contingencies.get(ci).contingencyId(), g, baseCarrier, newCarrier,
                                                    baseTargets[baseCarrier]);
                                        }
                                    }
                                }
                                rowModePerScenario[s] = row;
                            }
                            for (LfBranch ziOut : contingencies.get(ci).outagedZiBranches()) {
                                Integer ze = ziElemByNumAll.get(ziOut.getNum());
                                // OLF_GPU_ZITREE_TRACE: does this scenario OPEN a coupler the tree relocation
                                // moved rows onto? Such a scenario deactivates the coupler's ZERO_V/ZERO_PHI,
                                // which the relocation put on the CHILD bus's rows, and pins its dummies with
                                // DUMMY_TARGET identities those rows no longer host — the per-scenario gap.
                                // childSide (ziIdx slot 10, bits 1-2) is nonzero exactly when rows moved.
                                if (System.getenv("OLF_GPU_ZITREE_TRACE") != null
                                        && ((gpuData.ziIdx()[ze * 11 + 10] >> 1) & 3) != 0) {
                                    System.err.println("ZITREE_OPEN ctg=" + contingencies.get(ci).contingencyId()
                                            + " coupler=" + ziOut.getId()
                                            + " childSide=" + ((gpuData.ziIdx()[ze * 11 + 10] >> 1) & 3));
                                }
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
                                // The RELOCATION gave this coupler's ZERO_V the CHILD bus's own V row. With the
                                // coupler open that equation is gone, and OLF's post-contingency system has
                                // BUS_TARGET_V active on that bus instead (read off the CPU dump for
                                // BUS-6014_BBS_2: bus_target_v6283 ACTIVE, zero_v9043 inactive). The device
                                // must make the same move, or the row is left EMPTY - the P balance comes back
                                // through the stock-placement fallback, but nothing restores the V identity.
                                // Only for a bus that is still VOLTAGE-CONTROLLED: a PQ child's reactive
                                // balance returns to this row through that same fallback instead.
                                int relocBits = gpuData.ziIdx()[ze * 11 + 10];
                                int childSide = (relocBits >> 1) & 3;
                                if ((relocBits & 8) != 0 && childSide != 0) {
                                    LfBus child = childSide == 1 ? ziOut.getBus1() : ziOut.getBus2();
                                    int childVRow = gpuData.ziIdx()[ze * 11 + (childSide == 1 ? 6 : 7)];
                                    var childVc = child == null ? java.util.Optional.<GeneratorVoltageControl>empty()
                                            : child.getGeneratorVoltageControl();
                                    // The TARGET must travel with the equation. The child's P balance lives on
                                    // this coupler's DUMMY_P while it is closed, so its target (the bus's net
                                    // scheduled injection, and whatever the distributed slack has shifted it
                                    // to) sits there too. The stock-placement fallback moves the EQUATION back
                                    // to the child's own PHI row when the coupler opens; without moving the
                                    // target with it the bus's whole injection is left on a row that is now a
                                    // DUMMY_TARGET identity, and the slack absorbs it — measured on
                                    // BUS-6014_BBS_2 as 7.386 MW of distributed active power against the CPU
                                    // and every bus shifted 2-4e-2 kV in one direction.
                                    int childPhiRow = childSide == 1 ? gpuData.ziIdx()[ze * 11 + 8]
                                            : gpuData.ziIdx()[ze * 11 + 9];
                                    // ...unless the child is in the ISLAND this outage created. Then the
                                    // freeze already holds its rows at a clean flat state (V=1, phi=0) and
                                    // OLF has removed the bus, so moving an injection target onto them would
                                    // overwrite the freeze - these overrides are applied after it.
                                    int[] frozenForCtg = frozenRowsByIndex[ci];
                                    boolean childFrozen = false;
                                    for (int fr : frozenForCtg) {
                                        if (fr == childPhiRow || fr == childVRow) {
                                            childFrozen = true;
                                            break;
                                        }
                                    }
                                    if (childFrozen) {
                                        continue;
                                    }
                                    // The child keeps controlling its own voltage once the coupler that carried
                                    // its ZERO_V is gone, so its own V row takes the BUS_TARGET_V identity.
                                    // This USED TO RUN ABOVE the frozen guard, so an ISLANDED child got the
                                    // identity too -- a second override on a row the island freeze holds flat,
                                    // and applyTargetOverrideBatch resolves two entries on one row by a RACE.
                                    // Measured on rte6515 BUS-6194_BBS_2: VL-6194_1 is islanded and its V row
                                    // carried both 1.000 (the freeze) and 1.123 (this identity). The guard's own
                                    // comment already said the freeze must not be overwritten -- it just sat
                                    // below the write it was guarding.
                                    boolean applied = childVRow >= 0 && child != null && !child.isDisabled()
                                            && child.isGeneratorVoltageControlled() && childVc.isPresent();
                                    if (applied) {
                                        rowZi[childVRow] = 0;
                                        overrides.add(s * n + childVRow, childVc.orElseThrow().getTargetValue());
                                    }
                                    if (childPhiRow >= 0 && dpRow >= 0) {
                                        overrides.add(s * n + childPhiRow, baseTargets[dpRow]);
                                        // The DS participant on that bus has to follow its equation. Its
                                        // structural row is this coupler's DUMMY_P, now an identity.
                                        for (int i = 0; i < nds; i++) {
                                            if (gpuData.dsPhiRow()[i] != dpRow) {
                                                continue;
                                            }
                                            dsRowOvList.add(s);
                                            dsRowOvList.add(i);
                                            dsRowOvList.add(childPhiRow);
                                            if (dsRowOvBySlot[s] == null) {
                                                dsRowOvBySlot[s] = new int[nds];
                                                java.util.Arrays.fill(dsRowOvBySlot[s], -1);
                                            }
                                            dsRowOvBySlot[s][i] = childPhiRow;
                                        }
                                    }
                                    // A child that is NOT voltage-controlled has its reactive balance on the
                                    // coupler's DUMMY_Q instead, so its Q target follows the same way.
                                    //
                                    // ...unless that row is a DISTR_Q row whose target the plan RENORMALIZES for
                                    // this scenario (an island took a controller out of the group). Then
                                    // baseTargets[dqRow] is the BASE-key target and the plan's entry is OLF's,
                                    // and writing both left TWO override entries on one row — which the device
                                    // resolves by a race (applyTargetOverrideBatch is one thread per entry
                                    // writing target[ovIndex[k]]). Measured on the middle-section fixture: the
                                    // base -0.133333333 won over the renormalized -0.2, so the row kept a target
                                    // built from three keys while its weights had been renormalized over two —
                                    // 13.33 MVAr on l2 and 0.32 kV on every bus behind it. Yielding here (rather
                                    // than letting the plan's entry overwrite) keeps every scenario whose plan
                                    // seeds no target for the row bit-identical.
                                    if (!applied && childVRow >= 0 && dqRow >= 0
                                            && !seedsDqTargetRow(genOutagesByIndex[ci], childVRow)) {
                                        overrides.add(s * n + childVRow, baseTargets[dqRow]);
                                    }
                                    if (System.getenv("OLF_GPU_ZITREE_TRACE") != null) {
                                        System.err.println("ZITREE_VID coupler=" + ziOut.getId()
                                                + " child=" + (child == null ? "-" : child.getId())
                                                + " vRow=" + childVRow
                                                + " disabled=" + (child != null && child.isDisabled())
                                                + " vc=" + (child != null && child.isGeneratorVoltageControlled())
                                                + " baseMode=" + (childVRow >= 0 ? baseRowMode[childVRow] : -1)
                                                + " target=" + (childVc.isPresent()
                                                        ? childVc.orElseThrow().getTargetValue() : Double.NaN)
                                                + " applied=" + applied);
                                    }
                                }
                                rowModePerScenario[s] = rowZi;
                            }
                            // OLF's spanning tree, replayed by GpuBatchPrecomputer. The device extracted
                            // its structure on the BASE tree; where OLF has moved on by the time it reaches
                            // this contingency, the coupler that is out of the tree there takes the identity
                            // rows (target 0) and the one it replaced takes the ZERO_V/ZERO_PHI coupling
                            // back. The split is physically equivalent either way - this is about solving
                            // the system OLF solves, from the state OLF solves it at.
                            for (LfBranch nonTree : contingencies.get(ci).stickyNonTreeZi()) {
                                Integer ze = ziElemByNumAll.get(nonTree.getNum());
                                if (ze == null || gpuData.ziIdx()[ze * 11 + 10] == 0) {
                                    continue;                // absent, or already out of the base tree
                                }
                                ziTreeSlotElemValList.add(s);
                                ziTreeSlotElemValList.add(ze);
                                ziTreeSlotElemValList.add(0);
                                int dpRow = gpuData.ziIdx()[ze * 11 + 4];
                                int dqRow = gpuData.ziIdx()[ze * 11 + 5];
                                int[] row = rowModePerScenario[s] == baseRowMode ? baseRowMode.clone() : rowModePerScenario[s];
                                row[dpRow] = 0;              // identity rows, target 0
                                row[dqRow] = 0;
                                overrides.add(s * n + dpRow, 0.0);
                                overrides.add(s * n + dqRow, 0.0);
                                rowModePerScenario[s] = row;
                            }
                            if (!contingencies.get(ci).stickyNonTreeZi().isEmpty()) {
                                java.util.Set<Integer> stillOut = contingencies.get(ci).stickyNonTreeZi().stream()
                                        .map(LfBranch::getNum).collect(java.util.stream.Collectors.toSet());
                                for (int zb = 0; zb < gpuData.ziBranchNums().length; zb++) {
                                    if (gpuData.ziIdx()[zb * 11 + 10] != 0 || stillOut.contains(gpuData.ziBranchNums()[zb])) {
                                        continue;            // in the base tree already, or still out
                                    }
                                    ziTreeSlotElemValList.add(s);
                                    ziTreeSlotElemValList.add(zb);
                                    ziTreeSlotElemValList.add(1);
                                    int bdp = gpuData.ziIdx()[zb * 11 + 4];
                                    int bdq = gpuData.ziIdx()[zb * 11 + 5];
                                    int[] r2 = rowModePerScenario[s] == baseRowMode ? baseRowMode.clone() : rowModePerScenario[s];
                                    r2[bdp] = 1;             // back into the coupling
                                    r2[bdq] = 1;
                                    rowModePerScenario[s] = r2;
                                }
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
                                    if (go.killsPilotVoltageIdentity()) {
                                        // The pilot's voltage identity goes NOW, with the member's
                                        // mode-4 row, so the two halves of the dead control land in
                                        // the same fill.
                                        int[] row = rowModePerScenario[s] == baseRowMode
                                                ? baseRowMode.clone() : rowModePerScenario[s];
                                        row[go.deadPilotVRow()] = 1;
                                        rowModePerScenario[s] = row;
                                        overrides.add(s * n + go.deadPilotVRow(), go.deadPilotQTarget());
                                    }
                                    if (go.toPq()) {
                                        int[] row = rowModePerScenario[s] == baseRowMode ? baseRowMode.clone() : rowModePerScenario[s];
                                        row[go.vRow()] = go.pqRowModeOrDefault();   // BUS_V row becomes the
                                                                                    // reactive-power equation (PV→PQ)
                                        rowModePerScenario[s] = row;
                                    }
                                    if (go.reHomesRlEntry()) {
                                        rlLocalList.add(s);
                                        rlLocalList.add(go.rlEntry());
                                        rlLocalVt.add(go.qTarget());
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
                                    if (go.overridesDqMemberTargetQ() && gpuData.dqMemberBusNum() != null) {
                                        dqTqOvKey.add(s * gpuData.dqMemberBusNum().length
                                                + go.dqTargetQOvMember());
                                        dqTqOvVal.add(go.dqTargetQOvValue());
                                    }
                                    if (go.seedsDqTargets()) {
                                        for (int t = 0; t < go.dqTargetRows().length; t++) {
                                            int dqRow2 = go.dqTargetRows()[t];
                                            int dqMode = rowModePerScenario[s] != null
                                                    ? rowModePerScenario[s][dqRow2] : baseRowMode[dqRow2];
                                            // Seed only where the row still HOSTS a DISTR_Q equation this
                                            // scenario. Mode 4 means it hosts its own reactive balance (an RL
                                            // pin, or a control the contingency killed and whose BUS_TARGET_Q
                                            // the injection override owns); mode 0 means it is an IDENTITY —
                                            // a re-homed control's voltage target (reHomedPvEntry, the remote
                                            // main of a split group) or a frozen/outaged row. updateDistrQ-
                                            // TargetsBatch refuses to maintain a DISTR_Q target on either ("A
                                            // DISTR_Q target has no business here"), so uploading one is a
                                            // write this code does not own: it lands as a SECOND override on a
                                            // row that already has one. Measured over the full busbar scope
                                            // (OLF_GPU_OV_DUP_TRACE): 26 mode-4 rows across 24 contingencies
                                            // disagreed, all from this site — BUS-1359_BBS_1 and BUS-1869_BBS_1
                                            // (the +-2 violation pair), BUS-1599_BBS_1. Then, once the split
                                            // re-homes batched (tasks #70), the mode-0 case: BUS-2640_BBS_1's
                                            // elected main VL-2640_1 got its 1.034 identity on row 5956 AND
                                            // this seed's 0.0 (its renormalized DISTR_Q target); the device
                                            // picked 0.0 in the full 6667-contingency batch (slot 1844: V
                                            // driven to 0, 193.85 MW of distributed power, +197 violations)
                                            // and 1.034 in a 3-contingency batch (slot 0, bit-identical to
                                            // OLF). The earlier "mode 0 is NOT skipped: skipping it moved
                                            // deviceMultiChunkMatchesCpu by 1.06e-12 kV" was the race's own
                                            // noise being guarded.
                                            if (dqMode == 4 || dqMode == 0) {
                                                continue;
                                            }
                                            // ...nor on the row that HOSTS A PILOT'S BALANCE this scenario: a frozen base
                                            // carrier hands the pilot's reactive balance to the next live member's row
                                            // (the block above stamps the pilot's target there), and that member's own
                                            // renormalized DISTR_Q target is not that row's equation any more. The two
                                            // writes raced on the device and the pilot's won by luck — measured once the
                                            // race was closed: deviceBusLossIslandingBaseCarrierMemberMatchesCpu, row 2
                                            // −2.0 (b4's load) shadowed by this seed's 0.0, b2 22.49 → 22.04 kV.
                                            if (pilotBalanceRows.contains(dqRow2)) {
                                                continue;
                                            }
                                            // Any OTHER earlier override on a seed row is a producer this code does not
                                            // know about: keep the more specific one, and count it so a composition gap
                                            // (two seeds for one row from two paths) is measurable, never a race.
                                            if (overrides.has(s * n + dqRow2)) {
                                                com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("HOST:DQSEED_SHADOWED");
                                                continue;
                                            }
                                            overrides.add(s * n + dqRow2, go.dqTargetValues()[t]);
                                        }
                                    }
                                    if (go.flipsPilotRowWhenAllPinned()) {
                                        pfSlot.add(s);                     // handled IN the solve
                                        pfRow.add(go.pinnedPilotVRow());
                                        pfQTarget.add(go.pinnedPilotQTarget());
                                        pfRlCount.add(go.pilotGroupSurvivorRlEntries().length);
                                        for (int rl : go.pilotGroupSurvivorRlEntries()) {
                                            pfRlPool.add(rl);
                                        }
                                    } else if (go.needsPilotPinCheck()) {
                                        // No pilot row to move the balance onto (the outaged bus is not
                                        // the pilot): still a runtime reject.
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
                                        // ...and WHICH generators left, as slots in the DS generator pack. With the
                                        // identity the outaged bus's SURVIVORS can take part in the per-generator
                                        // water-fill like any others, instead of the bus riding on one aggregate
                                        // entry — which is what leaves their own P unknown (genCum stays zero
                                        // there) and sends the curve channel back to the share model that
                                        // per-generator clamping breaks.
                                        String[] dgIds = gpuData.dgGenId();
                                        for (int g = 0; dgIds != null && g < dgIds.length; g++) {
                                            if (go.lostGenIds().contains(dgIds[g])) {
                                                dsLostGenList.add(s);
                                                dsLostGenList.add(g);
                                            }
                                        }
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
                            // The amount OLF's DefaultContingencyActivePowerLossDistribution returns for this
                            // contingency (mismatch - remaining, pu), as the replay below distributes it.
                            int preIdx = group.get(chunkStart + s);
                            if (preLossBySlot[s] == 0) {
                                preDistributedByIndex[preIdx] = 0.0;
                                continue;
                            }
                            if (!hasDs) {
                                preDistributedByIndex[preIdx] = preLossBySlot[s];   // the slack generator takes it all
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
                            double[] shares = GpuBatchLossPreDistribution.distribute(gpuData, preLossBySlot[s],
                                    dsOvBus.length > 0 ? dsOvBus[s] : -1,
                                    dsOvFactorDelta.length > 0 ? dsOvFactorDelta[s] : 0,
                                    dsOvBaseDelta.length > 0 ? dsOvBaseDelta[s] : 0,
                                    dsOvMaxDelta.length > 0 ? dsOvMaxDelta[s] : 0,
                                    dsOvMinDelta.length > 0 ? dsOvMinDelta[s] : 0, islandedDs,
                                    lostSlotsOf(dsLostGenList, s));
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
                                preDistributedByIndex[preIdx] = 0.0;
                                continue;
                            }
                            double preDistributed = 0;
                            for (double share : shares) {                 // participants + the slack's own share
                                preDistributed += share;
                            }
                            preDistributedByIndex[preIdx] = preDistributed;
                            for (int i = 0; i < nds; i++) {
                                preAbsorbedBySlot[s] += shares[i];        // participants only
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
                                int dsRow = dsRowOvBySlot[s] != null && dsRowOvBySlot[s][i] >= 0
                                        ? dsRowOvBySlot[s][i] : gpuData.dsPhiRow()[i];
                                int flat = s * n + dsRow;
                                // compose with a gen-outage override on the same row
                                overrides.compose(flat, baseTargets[gpuData.dsPhiRow()[i]], shares[i]);
                            }
                        }
                    }
                    // CONTINUOUS phase control: OLF's initialize() enables the CONTROLLER-mode controls
                    // BEFORE the first solve. This is the loop's initialize(), not an orchestrator — the
                    // in-kernel phase expects the activation to already carry it.
                    if (phases.continuousPhase() != null && !phases.continuousPhase().isEmpty()) {
                        GpuBatchContinuousPhase.applyPhaseOn(realCount, n, phases.continuousPhase(),
                                rowModePerScenario, baseRowMode, disabledMask, overrides);
                    }
                    int[] targetOverrideIndex = overrides.indexArray();
                    double[] targetOverrideValue = overrides.valueArray();
                    if (overrides.conflictingAddCount() > 0) {
                        // Two producers disagreed on one row's target. GpuBatchOverrides resolved it to the
                        // later one (no row reaches the device twice), but a sound plan has NO such row:
                        // every one so far was a write that should not exist (a DISTR_Q seed on a mode-0/4
                        // row). Name it; OLF_GPU_OV_DUP_TRACE=1 lists the rows.
                        LOGGER.warn("GPU batched SA: {} conflicting target overrides in chunk at {} (resolved to the later"
                                + " writer; OLF_GPU_OV_DUP_TRACE=1 lists them)", overrides.conflictingAddCount(), chunkStart);
                        com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("HOST:OVERRIDE_CONFLICT");
                    }
                    if (GpuBatchOverrides.dupTrace()) {
                        // The invariant GpuBatchOverrides states in its javadoc, measured on the entries
                        // actually handed to the device: a row with two entries is written by two threads
                        // of applyTargetOverrideBatch in an unspecified order. Same value = benign;
                        // different values = a target chosen by a race.
                        System.err.printf("OV_DUP chunk entries=%d dupRows=%d conflictRows=%d%n",
                                targetOverrideIndex.length, overrides.duplicateRowCount(),
                                overrides.conflictingRowCount());
                        java.util.List<String> sites = overrides.conflictSites();
                        for (int ci2 = 0; ci2 < overrides.conflictSamples().size(); ci2++) {
                            double[] c = overrides.conflictSamples().get(ci2);
                            int flat = (int) c[0];
                            int slot = flat / n;
                            String id = slot < realCount
                                    ? contingencies.get(group.get(chunkStart + slot)).contingencyId() : "<padding>";
                            int row2 = flat % n;
                            // The row MODE decides who is right: updateDistrQTargetsBatch refuses to
                            // maintain a DISTR_Q target on a mode-0 identity or a mode-4 own-Q row, so a
                            // seeded DISTR_Q target on such a row is the write that should not exist.
                            int mode2 = rowModePerScenario != null && slot < realCount
                                    && rowModePerScenario[slot] != null
                                    ? rowModePerScenario[slot][row2] : baseRowMode[row2];
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
                            emitTargetRow(probe, n, realCount, baseTargets, baseRowMode, rowModePerScenario,
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
                        System.err.println("GPU-PAIR-SHOV slotElem=" + shOvSlotElemList + " gb=" + shOvGbList
                                + " shPackOffset=" + shPackOffset);
                    }
                    // Pair phase: warm continuation from each parent contingency's state + RL pinning.
                    double[] x0Full = null;
                    int[] rlStateIn = null;
                    int[] x0SeedIdx = null;
                    double[] x0SeedVal = null;
                    if (!frozenSeedSlot.isEmpty()) {         // islanding in this chunk: seed the frozen rows
                        // Only the frozen rows differ from the base state: send them as seeds the device writes over
                        // its own tiling of the base, instead of chunkSize * n doubles (372 MB per chunk on rte6515,
                        // 1.5 GB for a single 14,260-scenario chunk). The HVDC freeze reads its warm start from the
                        // same seeds.
                        if (X0_SEEDS) {
                            x0SeedIdx = new int[frozenSeedSlot.size()];
                            x0SeedVal = new double[frozenSeedSlot.size()];
                            for (int k = 0; k < frozenSeedSlot.size(); k++) {
                                x0SeedIdx[k] = frozenSeedSlot.get(k) * n + frozenSeedRow.get(k);
                                x0SeedVal[k] = frozenSeedValue.get(k);
                            }
                        } else {
                            x0Full = new double[chunkSize * n];
                            for (int s = 0; s < chunkSize; s++) {
                                System.arraycopy(baseState, 0, x0Full, s * n, n);
                            }
                            for (int k = 0; k < frozenSeedSlot.size(); k++) {
                                x0Full[frozenSeedSlot.get(k) * n + frozenSeedRow.get(k)] = frozenSeedValue.get(k);
                            }
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
                            for (int k = 0; k < frozenSeedSlot.size(); k++) {
                                if (frozenSeedSlot.get(k) == s) {   // a frozen row stays AT its frozen state
                                    x0Full[s * n + frozenSeedRow.get(k)] = frozenSeedValue.get(k);
                                }
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
                    GpuBatchOverrides.Snapshot preHvdcOverrides = null;
                    int[][] hvdcStatus = null;
                    double[] hvAppliedD1 = null;
                    double[] hvAppliedD2 = null;
                    if (hvdcControllers != null && !hvdcControllers.isEmpty()) {
                        hvAppliedD1 = new double[chunkSize * hvdcControllers.size()];
                        hvAppliedD2 = new double[chunkSize * hvdcControllers.size()];
                        preHvdcOverrides = overrides.snapshot();
                        if (System.getenv("OLF_GPU_HVDC_TRACE") != null) {
                            for (int s = 0; s < realCount; s++) {
                                System.err.println("HVDC_SLOT s=" + s + " ctg="
                                        + contingencies.get(group.get(chunkStart + s)).contingencyId());
                                // the PRE-freeze override entries on this slot's converter-bus P rows: the freeze
                                // composes on top of them, so a stale one here is a double count downstream
                                for (GpuBatchHvdcAcEmulation.Ctl c : hvdcControllers) {
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
                            hvdcStatus = GpuBatchHvdcAcEmulation.applyFreeze(chunkSize, realCount, n, hvdcControllers,
                                    baseState, x0Full, x0Full == null ? x0SeedIdx : null, x0Full == null ? x0SeedVal : null,
                                    disabledMask, baseTargets, overrides, hvAppliedD1, hvAppliedD2, rowModePerScenario);
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
                    GpuAcNewtonSolver.setBatchFrozenRows(handle,
                            frozenSeedSlot.isEmpty() ? null : java.util.stream.IntStream
                                    .range(0, frozenSeedSlot.size())
                                    .flatMap(k -> java.util.stream.IntStream.of(
                                            frozenSeedSlot.get(k), frozenSeedRow.get(k)))
                                    .toArray());
                    GpuAcNewtonSolver.setBatchDsLostGenerators(handle,
                            dsLostGenList.isEmpty() ? null
                                    : dsLostGenList.stream().mapToInt(Integer::intValue).toArray());
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
                    GpuAcNewtonSolver.setBatchDsRowOverrides(handle,
                            dsRowOvList.isEmpty() ? null : dsRowOvList.stream().mapToInt(Integer::intValue).toArray());
                    GpuAcNewtonSolver.setBatchRlIslanded(handle,
                            rlIslandedList.isEmpty() ? null : rlIslandedList.stream().mapToInt(Integer::intValue).toArray());
                    GpuAcNewtonSolver.setBatchRlLocal(handle,
                            rlLocalList.isEmpty() ? null : rlLocalList.stream().mapToInt(Integer::intValue).toArray(),
                            rlLocalList.isEmpty() ? null : rlLocalVt.stream().mapToDouble(Double::doubleValue).toArray());
                    GpuAcNewtonSolver.setBatchDqMemberTargetQOverrides(handle,
                            dqTqOvKey.isEmpty() ? null : dqTqOvKey.stream().mapToInt(Integer::intValue).toArray(),
                            dqTqOvKey.isEmpty() ? null : dqTqOvVal.stream().mapToDouble(Double::doubleValue).toArray());
                    GpuAcNewtonSolver.setBatchPilotAllPinnedFlip(handle,
                            pfSlot.isEmpty() ? null : pfSlot.stream().mapToInt(Integer::intValue).toArray(),
                            pfSlot.isEmpty() ? null : pfRow.stream().mapToInt(Integer::intValue).toArray(),
                            pfSlot.isEmpty() ? null : pfQTarget.stream().mapToDouble(Double::doubleValue).toArray(),
                            pfSlot.isEmpty() ? null : pfRlCount.stream().mapToInt(Integer::intValue).toArray(),
                            pfSlot.isEmpty() ? null : pfRlPool.stream().mapToInt(Integer::intValue).toArray());
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
                    long solveCountBefore = GpuAcNewtonSolver.readBatchSolveCount(handle);
                    // Seeds only where no full per-scenario state goes (the pair phase builds its own from the parents).
                    GpuAcNewtonSolver.setBatchX0Seeds(handle, x0Full == null ? x0SeedIdx : null, x0Full == null ? x0SeedVal : null);
                    // -Dolf.gpu.sa.fixedInnerIters=K: K Newton steps per inner solve, tested once at the end (a
                    // performance comparison against the per-iteration test; unconverged scenarios go to the CPU).
                    GpuAcNewtonSolver.setBatchFixedIter(handle, Integer.getInteger("olf.gpu.sa.fixedInnerIters", 0));
                    long tSolve0 = System.nanoTime();
                    // OLF_GPU_LOOP_TRACE_CTG=<id>: trace that contingency's slot in THIS chunk (a fixed slot number
                    // follows the chunk composition, which moves with the resolved chunk size).
                    // OLF's tree for the scenarios of this chunk, where it has moved from the one the
                    // structure was extracted on (empty clears the previous chunk's list).
                    GpuAcNewtonSolver.setBatchZiTreeOverrides(handle,
                            ziTreeSlotElemValList.stream().mapToInt(Integer::intValue).toArray());
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
                            overrides.restore(preHvdcOverrides);
                            GpuBatchHvdcAcEmulation.applyStatusOverrides(chunkSize, realCount, n,
                                    hvdcControllers, hvdcStatus, ctgOutOf, disabledMask, baseTargets, overrides,
                                    hvAppliedD1, hvAppliedD2, rowModePerScenario);
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
                            int row = dsRowOvBySlot[0] != null && dsRowOvBySlot[0][i] >= 0
                                    ? dsRowOvBySlot[0][i] : gpuData.dsPhiRow()[i];
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
                            if (preLossBySlot[s] == 0) {
                                continue;
                            }
                            int ci = group.get(chunkStart + s);
                            double moved = 0;
                            for (int i = 0; i < nds; i++) {
                                int row = dsRowOvBySlot[s] != null && dsRowOvBySlot[s][i] >= 0
                                        ? dsRowOvBySlot[s][i] : gpuData.dsPhiRow()[i];
                                // A participant this contingency ISLANDED has its row frozen to a flat
                                // identity, so (final - base) there is minus its whole anchor and not a
                                // distribution at all: counting it read the middle-section fixture's
                                // participants as -1.669969 pu off OLF when the real gap was -0.0687.
                                if (islandedDsByIndex != null && islandedDsByIndex[ci] != null
                                        && contains(islandedDsByIndex[ci], i)) {
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
                                    preLossBySlot[s], preAbsorbedBySlot[s], moved, moved - preAbsorbedBySlot[s]);
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
                                int setupMode = rowModePerScenario != null && rowModePerScenario[s] != null
                                        ? rowModePerScenario[s][probe] : baseRowMode[probe];
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
                                double effBase = baseTargets[row] - (dsOvBus[s] == i ? dsOvBaseDelta[s] : 0.0);
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
                    if (!pairPhase && (hasDs || dsSlackOnly)
                            && GpuAcNewtonSolver.readBatchSolveCount(handle) - solveCountBefore == 1) {
                        recordDistributedActivePower(handle, gpuData, realCount, chunkSize, n, group, chunkStart,
                                dsSlackOnly, dsOvBus, dsOvBaseDelta, preDistributedByIndex, distributedByIndex);
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
     */
    private static void recordDistributedActivePower(long handle, GpuAcData gpuData, int realCount, int chunkSize, int n,
                                                     List<Integer> group, int chunkStart, boolean dsSlackOnly,
                                                     int[] dsOvBus, double[] dsOvBaseDelta,
                                                     double[] preDistributedByIndex, double[] distributedByIndex) {
        if (dsSlackOnly) {
            double[] absorbed = GpuAcNewtonSolver.readBatchSlackAbsorbed(handle);
            for (int s = 0; s < realCount && absorbed != null; s++) {
                distributedByIndex[group.get(chunkStart + s)] = absorbed[s];
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
                distributedByIndex[idx] = preDistributedByIndex[idx];
                continue;
            }
            double participants = 0;
            if (ngen > 0) {
                if (genCum == null) {
                    genCum = GpuAcNewtonSolver.readBatchDsGenCum(handle);
                }
                if (genCum == null) {
                    continue;
                }
                for (int g = 0; g < ngen; g++) {
                    participants += genCum[s * ngen + g];
                }
            } else {                                         // bus-level participants: targets from the anchor
                if (targets == null) {
                    targets = GpuAcNewtonSolver.readBatchTargets(handle);
                }
                int ov = dsOvBus != null && dsOvBus.length > 0 ? dsOvBus[s] : -1;
                for (int i = 0; i < gpuData.dsPhiRow().length; i++) {
                    double anchor = gpuData.dsBaseTargetP()[i] - (ov == i ? dsOvBaseDelta[s] : 0.0);
                    participants += targets[s * n + gpuData.dsPhiRow()[i]] - anchor;
                }
            }
            distributedByIndex[idx] = participants + retain[s];
        }
    }
}
