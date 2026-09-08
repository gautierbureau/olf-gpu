/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.atomic.LongAdder;

/**
 * Counters the phase timings alone cannot explain: how many batched device solves each host-side
 * orchestrator actually issued, and how many rounds it took to get there.
 *
 * <p>A wall-time split says "the HVDC orchestrator cost 10 s"; it does not say whether that was one
 * expensive solve or twenty cheap re-checks that each rebuilt a 300k-entry override list. Only the
 * second reading tells you which end to optimise, so the counters ship beside the timers and are
 * printed by the same {@code OLF_GPU_PHASE_TIME} switch.
 *
 * <p>Off by default and read once at class-init: when the switch is unset every call here is a
 * predictable branch on a static final, so the counters can sit on hot paths.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuBatchPhaseStats {

    static final boolean ENABLED = System.getenv("OLF_GPU_PHASE_TIME") != null;

    private static final Map<String, LongAdder> COUNTS = new ConcurrentSkipListMap<>();
    private static final Map<String, LongAdder> NANOS = new ConcurrentSkipListMap<>();

    private GpuBatchPhaseStats() {
    }

    /** Count one occurrence of {@code key} (a device solve, an orchestrator round, ...). */
    static void bump(String key) {
        if (ENABLED) {
            COUNTS.computeIfAbsent(key, k -> new LongAdder()).increment();
        }
    }

    /** Add {@code v} to the counter under {@code key} (iteration totals, entry counts, ...). */
    static void add(String key, long v) {
        if (ENABLED) {
            COUNTS.computeIfAbsent(key, k -> new LongAdder()).add(v);
        }
    }

    /** Accumulate nanoseconds under {@code key}; printed in ms beside the counts. */
    static void addNs(String key, long ns) {
        if (ENABLED) {
            NANOS.computeIfAbsent(key, k -> new LongAdder()).add(ns);
        }
    }

    static void reset() {
        COUNTS.clear();
        NANOS.clear();
    }

    static Map<String, Long> snapshot() {
        Map<String, Long> out = new TreeMap<>();
        COUNTS.forEach((k, v) -> out.put(k, v.sum()));
        return out;
    }

    /** The accumulated timings, in MILLISECONDS. */
    static Map<String, Long> snapshotMs() {
        Map<String, Long> out = new TreeMap<>();
        NANOS.forEach((k, v) -> out.put(k, v.sum() / 1_000_000));
        return out;
    }
}
