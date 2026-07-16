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
 * <p>CROSS-FLAVOR sharing: the base load flow always extracts the DS_RL flavor while an SA run
 * extracts with ITS distributed-slack / reactive-limits flags — different {@code GpuAcData}
 * instances for the same equation-system state. The flavor flags only select the outer-loop
 * RL/DS packs, which are batch-CONTEXT inputs, not structure inputs — so on an instance miss the
 * cache falls back to CONTENT matching over the structure inputs
 * ({@link GpuAcData#sameStructureInputsAs}) and aliases the existing structure instead of
 * rebuilding it. The native handle is therefore held through a refcounting holder shared by all
 * aliased keys.
 *
 * <p>Lifetime: the holder releases the native handle through a {@link Cleaner} once every
 * extraction mapped to it is garbage-collected. That only drops the Java-side reference — a
 * context built from the handle holds shared native ownership, so the structure lives until the
 * last context is destroyed.
 *
 * <p>VALUES refresh: {@code refreshContextValues} mutates the shared structure's device values in
 * place (tap/shunt moves). The refresher must then call {@link #rebind} so the cache maps the NEW
 * extraction (whose arrays match the device state) and drops the stale one — otherwise a later
 * extraction content-equal to the OLD values could alias a structure that no longer holds them.
 *
 * <p>{@code GpuAcData} is a record whose array components compare by reference, so instance-level
 * map lookups are effectively identity-based: distinct extractions never collide, and the
 * instance handed out by {@link GpuAcDataCache} always hits without any array comparison.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuAcStructures {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuAcStructures.class);

    private static final Cleaner CLEANER = Cleaner.create();

    /**
     * The Java-side owner of one native structure handle, shared by every extraction key aliased
     * to it. The Cleaner action fires when the LAST key (hence this holder, referenced only by the
     * cache values) becomes unreachable; it must not capture the holder itself.
     */
    private static final class Structure {
        private final long handle;
        private final int n;

        private Structure(long handle, int n) {
            this.handle = handle;
            this.n = n;
            CLEANER.register(this, () -> GpuAcNewtonSolver.destroyStructure(handle));
        }
    }

    private static final Map<GpuAcData, Structure> CACHE = new WeakHashMap<>();

    // Diagnostics (process-wide totals), mirroring GpuNewtonRaphson's context counters: a HIT is a
    // consumer reusing a structure another already built — same-instance (the GpuAcDataCache
    // base->batch hand-off, the chunk-size query) or cross-flavor content match (DS/RL flags off).
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
     * The native shared-structure handle for this extraction, building it on first use. An
     * instance miss falls back to content matching over the structure inputs, so a different
     * FLAVOR of an already-cached equation-system state aliases the existing structure.
     *
     * @param n    number of variables / equations
     * @param data the extraction (the caller must keep a strong reference for as long as it uses
     *             the handle — the structure is Cleaner-freed when every extraction mapped to it
     *             is collected)
     */
    public static synchronized long handle(int n, GpuAcData data) {
        Structure structure = CACHE.get(data);
        if (structure == null) {
            for (Map.Entry<GpuAcData, Structure> entry : CACHE.entrySet()) {
                if (entry.getValue().n == n && entry.getKey().sameStructureInputsAs(data)) {
                    structure = entry.getValue();            // another flavor of the same state
                    LOGGER.debug("GPU shared-structure cross-flavor content match");
                    break;
                }
            }
            if (structure == null) {
                structure = new Structure(GpuAcNewtonSolver.createStructure(n,
                        data.cbIn(), data.cbIdx(), data.obIn(), data.obIdx(),
                        data.shIn(), data.shIdx(), data.hvIn(), data.hvIdx(),
                        data.dqElem(), data.dqSide(), data.dqWeight(), data.dqRow(), data.dqKind(),
                        data.ziIn(), data.ziIdx(),
                        data.drRow(), data.drCol(), data.drCoef()), n);
                CACHE.put(data, structure);
                BUILDS.incrementAndGet();
                return structure.handle;
            }
            CACHE.put(data, structure);
        }
        HITS.incrementAndGet();
        LOGGER.debug("GPU shared-structure cache hit");
        return structure.handle;
    }

    /**
     * Re-key after a VALUES-level device refresh: the structure mapped from {@code from} now holds
     * {@code to}'s element values, so EVERY key aliased to it is dropped (their arrays no longer
     * describe the device state — leaving any would let a later content-equal extraction alias
     * wrong values) and {@code to} becomes its key (unless it already has its own structure).
     */
    /**
     * Test/bench hook: drop every cached mapping, so the next lookups rebuild instead of
     * instance- or content-matching an earlier extraction. Java-side only — a structure still
     * referenced by a live context keeps its native ownership until that context is destroyed.
     * Mirrors {@code GpuBatchedSecurityAnalysisSolver#clearContextCache} for cold-path
     * measurements (without it, content matching would alias a previous pass's device structure
     * and silently skip the rebuild being measured).
     */
    public static synchronized void clear() {
        CACHE.clear();
    }

    public static synchronized void rebind(GpuAcData from, GpuAcData to) {
        Structure structure = CACHE.remove(from);
        if (structure != null) {
            CACHE.values().removeIf(v -> v == structure);    // stale cross-flavor aliases
            CACHE.putIfAbsent(to, structure);
        } else {
            // the refreshed structure is unknown here — conservatively drop any key holding the
            // old content (its structure may be the refreshed one); consumers just rebuild
            CACHE.keySet().removeIf(k -> k.sameStructureInputsAs(from));
        }
    }
}
