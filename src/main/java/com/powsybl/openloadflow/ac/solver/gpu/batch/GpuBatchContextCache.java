/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcStructures;
import com.powsybl.openloadflow.ac.equations.vector.gpu.StructureNative;
import com.powsybl.openloadflow.ac.outerloop.gpu.DeviceLoopTables;
import com.powsybl.openloadflow.ac.solver.gpu.BatchSolveNative;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuLinearSolveMode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The batched N-1 path's single-entry native-context cache and its sizing/mode policy. Owns the ONLY
 * mutable global state of the batch solver — the analyzed, allocated cuDSS context reused across
 * {@code solveBatchN1} calls whose extracted {@link GpuAcData}, chunk size and mode match — behind
 * synchronized accessors, plus the shutdown hook that frees the native handle at JVM exit. Also
 * resolves the chunk size (from free device memory or a manual cap) and the native batched-solve mode.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuBatchContextCache {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuBatchContextCache.class);

    /** Test hook: a chunk size every resolution of this JVM uses instead of the auto-sizing, or null. */
    private static volatile Integer chunkSizeForTests;

    /** TESTS ONLY: resolve every chunk to {@code size} (clamped to the scenario count) until reset with null. */
    public static void forceChunkSizeForTests(Integer size) {
        chunkSizeForTests = size;
    }

    /** Fallback chunk size used when auto-sizing is unavailable (no GPU / memory-query failure). */
    public static final int DEFAULT_MAX_BATCH = 256;
    static final int AUTO_MAX_CHUNK = 4096;

    private static CachedBatchContext cachedContext;
    private static final java.util.concurrent.atomic.AtomicLong BUILDS = new java.util.concurrent.atomic.AtomicLong();

    /** How many batched contexts this JVM has built (a reuse does not count). */
    public static long buildCount() {
        return BUILDS.get();
    }

    /** The auto-sized chunk RECOMMENDATION for the extraction it was computed on, remembered for the
     *  whole run. {@code recommendBatchSize} prices the chunk against FREE device memory, so asking it
     *  again once another partition has allocated its context returns a smaller answer: under
     *  threadCount &gt; 1 the partitions were resolving 498, then 250, then 500, then 250 for the same
     *  network — different sizes evict each other from the single-entry context cache, each eviction
     *  paying a fresh cuDSS analysis, and the size a run picks depends on who won a race for memory.
     *  Remembering the first answer makes the sizing reproducible, which matters more than the seconds:
     *  chunk boundaries decide which scenarios share a solve. */
    private static volatile int recommendedChunk;
    private static volatile GpuAcData recommendedFor;

    static {
        Runtime.getRuntime().addShutdownHook(new Thread(GpuBatchContextCache::clearContextCache,
                "olf-gpu-batch-context-cleanup"));
    }

    private GpuBatchContextCache() {
    }

    /**
     * Cached batched native context, reused across {@code solveBatchN1} calls whose extracted
     * {@link GpuAcData}, chunk size and mode are identical. The cuDSS symbolic ANALYSIS (and the
     * uploaded per-element static data) depend only on the network pattern + admittances — NOT on
     * the operating point (injections/targets are passed per solve) — so successive SA runs on the
     * same network reuse the analyzed, allocated context instead of rebuilding it (the per-run
     * extraction/build/analysis + device alloc/free was ~25% of a single-run wall time). The native
     * handle is freed when a non-matching context evicts it, by {@link #clearContextCache()}, or at
     * JVM shutdown. NOT thread-safe across different networks (single-entry cache); guarded so the
     * native handle is never corrupted.
     */
    private static final class CachedBatchContext {
        final long handle;
        final GpuAcData data;
        final int chunkSize;
        final int mode;
        /** The shared device structure the context was built on. The base load flow's VALUES path refreshes a
         *  structure's device values in place and re-keys it (GpuAcStructures.rebind), after which this context
         *  solves the refreshed values while {@code data} still describes the old ones: a request for the old
         *  extraction must then resolve to ANOTHER structure and rebuild, not be handed this context. */
        final long structure;

        CachedBatchContext(long handle, GpuAcData data, int chunkSize, int mode, long structure) {
            this.handle = handle;
            this.data = data;
            this.chunkSize = chunkSize;
            this.mode = mode;
            this.structure = structure;
        }
    }

    /** Test hook: a mode every context of this JVM is built in instead of {@link GpuLinearSolveMode#UBATCH}, or null. */
    private static volatile Integer modeForTests;

    /** The native batched-solve mode: {@link GpuLinearSolveMode#UBATCH}. */
    public static int resolveBatchMode() {
        Integer forced = modeForTests;
        return forced != null ? forced : GpuLinearSolveMode.UBATCH;
    }

    /** TESTS ONLY: build every context in {@code mode} (e.g. {@link GpuLinearSolveMode#ROBUST}) until reset with null. */
    public static void forceModeForTests(Integer mode) {
        modeForTests = mode;
    }

    /** Acquire a batched native context for this (data, chunkSize, mode), reusing the cache when it matches. */
    public static synchronized long acquireBatchContext(GpuAcData data, int chunkSize, int mode, int n) {
        // the structure this extraction resolves to today (a hit on the shared cache, or a fresh build that a
        // rebuilt context would need anyway); a cached context built on another one solves other values
        long structure = GpuAcStructures.handle(n, data);
        if (cachedContext != null && cachedContext.chunkSize == chunkSize && cachedContext.mode == mode
                && cachedContext.data.sameAs(data) && cachedContext.structure == structure) {
            LOGGER.debug("Batched N-1: reusing cached GPU context (chunk {}, mode {})", chunkSize, mode);
            return cachedContext.handle;
        }
        if (cachedContext != null) {                         // evict a stale context (different network/chunk/mode)
            if (System.getenv("OLF_GPU_PHASE_TIME") != null) {
                System.err.printf("CONTEXT_EVICT cached chunk=%d -> requested chunk=%d (a cuDSS ANALYSIS is "
                        + "paid again)%n", cachedContext.chunkSize, chunkSize);
            }
            BatchSolveNative.destroyBatchContext(cachedContext.handle);
            cachedContext = null;
        }
        // the shared structure comes from the per-extraction cache — when the base load flow ran
        // on the GPU with the same extraction (GpuAcDataCache hand-off), this is a device-side hit
        long handle = BatchSolveNative.createBatchContext(structure, chunkSize, mode,
                data.rlVrow(), data.rlMinQ(), data.rlMaxQ(), data.rlLoadQ(), data.rlVtarget(), data.rlMonitored(), data.voltageRows().slope(),
                data.dsPhiRow(), data.dsFactor(), data.dsSlackPhiRow(), data.dsSlackTargetP(),
                data.dsBaseTargetP(), data.dsMaxDelta(), data.dsMinDelta(),
                data.dsMode(), data.dsMarginUp(), data.dsMarginDown(),
                data.dsSlackMarginUp(), data.dsSlackMarginDown());
        DeviceLoopTables.installRlElection(handle, data);
        cachedContext = new CachedBatchContext(handle, data, chunkSize, mode, structure);
        BUILDS.incrementAndGet();
        return handle;
    }

    /** Free the cached batched native context, if any. Safe to call repeatedly. */
    public static synchronized void clearContextCache() {
        if (cachedContext != null) {
            BatchSolveNative.destroyBatchContext(cachedContext.handle);
            cachedContext = null;
        }
        recommendedFor = null;                               // a failed solve may have been the size's fault
        recommendedChunk = 0;
    }

    /** Chunk sizes this JVM has SEEN fail to allocate on the device: the auto-size probe is an
     *  estimate, and on a small card it can recommend a chunk whose LU factors do not fit. Each
     *  failed attempt costs a cuDSS analysis, so once the chunk solver has discovered a ceiling we
     *  keep every later run under it instead of rediscovering it. Monotonic, never raised. */
    private static volatile int maxSafeChunk = Integer.MAX_VALUE;

    /** Record that {@code chunk} could not be allocated — later sizing stays strictly below it. */
    public static void noteChunkAllocFailure(int chunk) {
        maxSafeChunk = Math.min(maxSafeChunk, Math.max(1, chunk / 2));
    }

    /** The run's chunk recommendation for this extraction, priced ONCE (see {@link #recommendedChunk}).
     *  {@code AUTO_MAX_CHUNK} rather than the caller's contingency count is deliberate: the answer must
     *  not depend on which partition asked first. */
    private static synchronized int rememberedRecommendation(int n, GpuAcData gpuData) {
        if (recommendedFor != null && recommendedFor.sameAs(gpuData) && recommendedChunk > 0) {
            return recommendedChunk;
        }
        // sized on the shared structure (its exact nnz + a cuDSS LU probe cached there, so
        // only the first query per extraction pays the probe analysis)
        recommendedChunk = StructureNative.recommendBatchSize(GpuAcStructures.handle(n, gpuData), AUTO_MAX_CHUNK);
        recommendedFor = gpuData;
        return recommendedChunk;
    }

    /**
     * Resolve the GPU batch chunk size, bounding the resident device memory (the uniform
     * batched factorization holds an LU per scenario, so the footprint scales with the chunk).
     * <ul>
     *   <li>AUTO-SIZED from free device memory via
     *       {@link GpuAcNewtonSolver#recommendBatchSize} (real Jacobian nnz + heuristic LU
     *       estimate + safety margin).</li>
     *   <li>if auto-sizing is unavailable (no GPU / query error) — {@value #DEFAULT_MAX_BATCH}.</li>
     * </ul>
     * The result is always clamped to {@code [1, nb]}.
     */
    public static int resolveChunkSize(int nb, int n, GpuAcData gpuData) {
        Integer explicit = chunkSizeForTests;
        if (System.getenv("OLF_GPU_PHASE_TIME") != null) {
            System.err.printf("CHUNK_SIZING nb=%d explicit=%s%n", nb, explicit);
        }
        if (explicit != null && explicit > 0) {
            return Math.min(nb, explicit);
        }
        try {
            int rec = rememberedRecommendation(n, gpuData);
            return Math.min(Math.min(nb, maxSafeChunk), Math.max(1, rec));
        } catch (Throwable t) {
            LOGGER.debug("GPU batch auto-size unavailable ({}); using default chunk {}", t.toString(), DEFAULT_MAX_BATCH);
            return Math.min(Math.min(nb, maxSafeChunk), DEFAULT_MAX_BATCH);
        }
    }
}
