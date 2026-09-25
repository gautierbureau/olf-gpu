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
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcStructures;
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

    public static final String MAX_BATCH_PROPERTY = "olf.gpu.sa.maxBatch";
    /** Fallback chunk size used when auto-sizing is unavailable (no GPU / memory-query failure). */
    public static final int DEFAULT_MAX_BATCH = 256;
    /**
     * System property: the native batched-solve mode (cuDSS >= 0.8, all compiled in, runtime-selected).
     * 0 = per-scenario refactor+solve loop (memory-light serial fallback; never benchmarked as best);
     * 1 = uniform batched cuDSS API (one batched factorization+solve per iteration — faster per-iteration
     * SOLVE, but its nS-block ANALYSIS costs ~1.6 s cold, the single-pass loser even with the context cache);
     * 2 = block-diagonal single matrix, factorize-once + refactorize-every (the refactor-every strategy;
     * measured ~parity with mode 1 since cuDSS refactorization is not cheaper than factorization);
     * 3 = UBATCH (single per-block matrix + CUDSS_CONFIG_UBATCH_SIZE), the DEFAULT
     * mechanism: a cheap one-block ANALYSIS (~150 ms vs mode 1's ~1.6 s) wins the PRIMARY single-cold-pass
     * use case. The CHUNK SIZE is decisive on cuDSS 0.8: its UBATCH phases carry a per-launch fixed cost
     * that amortizes with the batch (pegase9241, K=4, warm ms/scenario: chunk 32 = 17.8, 125 = 6.35,
     * 500 = 4.87 — the fastest measured configuration on any mode/version; cuDSS 0.7.x is
     * chunk-size-insensitive at ~8.3 instead). Keep mode 3 with LARGE chunks (the memory auto-sizing)
     * on 0.8; mode 1 is the small-chunk alternative there (~6.2 at chunk 32).
     * 4 = UBATCH + iter0-only (factor at each chunk's iter 0, SOLVE-only after) and
     * 5 = UBATCH + refactorize-after: EXPERIMENTAL factor-skip probes, kept for retesting on other
     * grids/formulations. Measured non-beneficial on case9241pegase (mode 5 = mode 3 exactly; mode 4 is
     * ~2x faster per iter but needs ~3x more iters to converge, a net wash) — do NOT use in production.
     */
    public static final String BATCH_MODE_PROPERTY = "olf.gpu.batchMode";
    /** 6 = the ROBUST single-matrix solve: mode 0's per-scenario loop with BTF-COLAMD ordering, global
     *  column pivoting and matching (cuDSS single-matrix API). What the batch-of-one retries in after a
     *  linear-solve reject: rte6515's full busbar structure does not factor in the uniform batch. */
    public static final int MODE_ROBUST = 6;

    static final int AUTO_MAX_CHUNK = 4096;

    private static CachedBatchContext cachedContext;

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

        CachedBatchContext(long handle, GpuAcData data, int chunkSize, int mode) {
            this.handle = handle;
            this.data = data;
            this.chunkSize = chunkSize;
            this.mode = mode;
        }
    }

    /** Resolve the native batched-solve mode from {@value #BATCH_MODE_PROPERTY} (default 3, UBATCH). */
    public static int resolveBatchMode() {
        return Integer.getInteger(BATCH_MODE_PROPERTY, 3);
    }

    /** Acquire a batched native context for this (data, chunkSize, mode), reusing the cache when it matches. */
    public static synchronized long acquireBatchContext(GpuAcData data, int chunkSize, int mode, int n) {
        if (cachedContext != null && cachedContext.chunkSize == chunkSize && cachedContext.mode == mode
                && cachedContext.data.sameAs(data)) {
            LOGGER.debug("Batched N-1: reusing cached GPU context (chunk {}, mode {})", chunkSize, mode);
            return cachedContext.handle;
        }
        if (cachedContext != null) {                         // evict a stale context (different network/chunk/mode)
            if (System.getenv("OLF_GPU_PHASE_TIME") != null) {
                System.err.printf("CONTEXT_EVICT cached chunk=%d -> requested chunk=%d (a cuDSS ANALYSIS is "
                        + "paid again)%n", cachedContext.chunkSize, chunkSize);
            }
            GpuAcNewtonSolver.destroyBatchContext(cachedContext.handle);
            cachedContext = null;
        }
        // the shared structure comes from the per-extraction cache — when the base load flow ran
        // on the GPU with the same extraction (GpuAcDataCache hand-off), this is a device-side hit
        long handle = GpuAcNewtonSolver.createBatchContext(GpuAcStructures.handle(n, data), chunkSize, mode,
                data.rlVrow(), data.rlMinQ(), data.rlMaxQ(), data.rlLoadQ(), data.rlVtarget(), data.rlMonitored(), data.rowSlope(),
                data.dsPhiRow(), data.dsFactor(), data.dsSlackPhiRow(), data.dsSlackTargetP(),
                data.dsBaseTargetP(), data.dsMaxDelta(), data.dsMinDelta(),
                data.dsMode(), data.dsMarginUp(), data.dsMarginDown(),
                data.dsSlackMarginUp(), data.dsSlackMarginDown());
        GpuBatchOuterLoopTables.installRlElection(handle, data);
        cachedContext = new CachedBatchContext(handle, data, chunkSize, mode);
        return handle;
    }

    /** Free the cached batched native context, if any. Safe to call repeatedly. */
    public static synchronized void clearContextCache() {
        if (cachedContext != null) {
            GpuAcNewtonSolver.destroyBatchContext(cachedContext.handle);
            cachedContext = null;
        }
        recommendedFor = null;                               // a failed solve may have been the size's fault
        recommendedChunk = 0;
    }

    /**
     * Resolve the GPU batch chunk size, bounding the resident device memory (the uniform
     * batched factorization holds an LU per scenario, so the footprint scales with the chunk).
     * <ul>
     *   <li>{@value #MAX_BATCH_PROPERTY} set to a positive value — an explicit manual cap.</li>
     *   <li>otherwise — AUTO-SIZED from free device memory via
     *       {@link GpuAcNewtonSolver#recommendBatchSize} (real Jacobian nnz + heuristic LU
     *       estimate + safety margin).</li>
     *   <li>if auto-sizing is unavailable (no GPU / query error) — {@value #DEFAULT_MAX_BATCH}.</li>
     * </ul>
     * The result is always clamped to {@code [1, nb]}.
     */
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
        recommendedChunk = GpuAcNewtonSolver.recommendBatchSize(GpuAcStructures.handle(n, gpuData), AUTO_MAX_CHUNK);
        recommendedFor = gpuData;
        return recommendedChunk;
    }

    public static int resolveChunkSize(int nb, int n, GpuAcData gpuData) {
        Integer explicit = Integer.getInteger(MAX_BATCH_PROPERTY);
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
