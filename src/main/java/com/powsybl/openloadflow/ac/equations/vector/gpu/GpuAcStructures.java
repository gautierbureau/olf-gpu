/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.ref.Cleaner;
import java.util.Map;
import java.util.WeakHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Per-extraction cache of the native SHARED STRUCTURE ({@link GpuAcNewtonSolver#createStructure}):
 * the device-resident pattern / families / destination arrays / static trig that BOTH solver
 * contexts derive from one {@link GpuAcData}. Keyed weakly by the extraction — and since
 * {@link GpuAcDataCache} hands the base-case load flow and the batched security analysis the
 * SAME {@code GpuAcData} instance for one equation-system state, the base builds the structure
 * and the batch's build becomes a cache hit (previously each rebuilt it: duplicated host pattern
 * work, device uploads and allocations).
 *
 * <p>Lifetime: the native handle is released by a {@link Cleaner} when the extraction is
 * garbage-collected. That only drops the Java-side reference — a context built from the handle
 * holds shared native ownership, so the structure lives until the last context is destroyed.
 *
 * <p>{@code GpuAcData} is a record whose array components compare by reference, so map lookups
 * are effectively identity-based: distinct extractions never collide, and the instance handed
 * out by {@link GpuAcDataCache} always hits.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuAcStructures {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuAcStructures.class);

    private static final Cleaner CLEANER = Cleaner.create();

    private static final Map<GpuAcData, Long> CACHE = new WeakHashMap<>();

    // Diagnostics (process-wide totals), mirroring GpuNewtonRaphson's context counters: a HIT is a
    // consumer reusing a structure another already built — in a base-on-GPU SA pass, the batch and
    // the chunk-size query hitting the base load flow's structure.
    private static final AtomicLong BUILDS = new AtomicLong();
    private static final AtomicLong HITS = new AtomicLong();

    private GpuAcStructures() {
    }

    public static long buildCount() {
        return BUILDS.get();
    }

    public static long hitCount() {
        return HITS.get();
    }

    /**
     * The native shared-structure handle for this extraction, building it on first use.
     *
     * @param n    number of variables / equations
     * @param data the extraction (the caller must keep a strong reference for as long as it uses
     *             the handle — the structure is Cleaner-freed when the extraction is collected)
     */
    public static synchronized long handle(int n, GpuAcData data) {
        Long handle = CACHE.get(data);
        if (handle == null) {
            long created = GpuAcNewtonSolver.createStructure(n,
                    data.cbIn(), data.cbIdx(), data.obIn(), data.obIdx(),
                    data.shIn(), data.shIdx(), data.hvIn(), data.hvIdx(),
                    data.dqElem(), data.dqSide(), data.dqWeight(), data.dqRow(), data.dqKind(),
                    data.ziIn(), data.ziIdx(),
                    data.drRow(), data.drCol(), data.drCoef());
            CLEANER.register(data, () -> GpuAcNewtonSolver.destroyStructure(created));
            CACHE.put(data, created);
            BUILDS.incrementAndGet();
            return created;
        }
        HITS.incrementAndGet();
        LOGGER.debug("GPU shared-structure cache hit");
        return handle;
    }
}
