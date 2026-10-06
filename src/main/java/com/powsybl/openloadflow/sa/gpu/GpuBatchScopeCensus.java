/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

/**
 * WHY a contingency did not solve on the device, counted by reason. The batched SA routes anything
 * it cannot represent to OLF's own CPU path — always correct, just un-accelerated — which makes the
 * un-batched SHARE invisible unless something counts it. This census is that something: a
 * process-wide reason histogram an external harness (the gpu-validation project's SA mode) can
 * read after a run to turn "some contingencies went to the CPU" into "N went to the CPU, for these
 * reasons", which is what drives the remaining scope work on real networks.
 *
 * <p>Reasons are recorded at the three funnels a contingency can leave the batch through:
 * <ul>
 *   <li>{@code MAPPER:*} — the contingency's elements could not be COMPOSED into a batch scenario
 *       (an unresolvable or unmodeled element kind: see {@code GpuContingencyMapper});</li>
 *   <li>{@code VALIDATOR:*} — composed, but disqualified by shape (not in the GPU model, islands
 *       the slack, fragments the network with islanding off: see
 *       {@code GpuBatchContingencyValidator.Disqualification});</li>
 *   <li>{@code RUNTIME:*} — batched and solved, but the device result was rejected (diverged,
 *       straggler early-exit, saturated slack under a FAIL behavior…) so the CPU replays it.</li>
 * </ul>
 *
 * <p>Counters are cumulative until {@link #reset()}; they carry NO network data beyond the reason
 * labels themselves.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuBatchScopeCensus {

    /** Test hook: how many distributed-slack participants the last scenario plan masked out as
     *  stranded in another synchronous component (summed over the plan's contingencies). The
     *  bus-loss path once froze the island's rows without setting this mask, and the water-fill then
     *  read the frozen rows as live participants; a test that islands a participant asserts > 0. */
    public static final java.util.concurrent.atomic.AtomicInteger LAST_ISLANDED_DS_MASKED =
            new java.util.concurrent.atomic.AtomicInteger();

    private static final Map<String, LongAdder> COUNTS = new ConcurrentHashMap<>();
    /** The contingencies the scenario PLAN refused (never attempted on the device), by id. The counters
     *  above are per attempt — a cold and a warm run, or a robust retry, count a refusal twice — so a
     *  harness that subtracts them from a per-contingency list gets nonsense ("-20 of them tried and
     *  failed"). The split "refused at plan time" vs "tried on the device and failed" needs the IDS. */
    private static final java.util.Set<String> PLAN_REFUSED_IDS = ConcurrentHashMap.newKeySet();
    private static final Map<String, String> PLAN_REFUSAL_TAG = new ConcurrentHashMap<>();

    private GpuBatchScopeCensus() {
    }

    /** Count one contingency leaving the batched path for {@code reason}. */
    public static void record(String reason) {
        COUNTS.computeIfAbsent(reason, k -> new LongAdder()).increment();
    }

    /** Name a contingency the plan refused outright — it never reached the device. */
    public static void planRefused(String contingencyId) {
        PLAN_REFUSED_IDS.add(contingencyId);
    }

    /** ...with the refusal's reason tag, from the qualification site that decided it. */
    public static void planRefused(String contingencyId, String tag) {
        PLAN_REFUSED_IDS.add(contingencyId);
        PLAN_REFUSAL_TAG.putIfAbsent(contingencyId, tag);
    }

    /** The ids the plan refused since the last {@link #reset()}. */
    public static java.util.Set<String> planRefusedIds() {
        return java.util.Collections.unmodifiableSet(PLAN_REFUSED_IDS);
    }

    /** The reason tag recorded for a refused id, or "?" when only the never-batched marker was seen. */
    public static String planRefusalTag(String contingencyId) {
        return PLAN_REFUSAL_TAG.getOrDefault(contingencyId, "?");
    }

    /** The reason histogram since the last {@link #reset()}, ordered by reason. */
    public static Map<String, Long> snapshot() {
        Map<String, Long> out = new TreeMap<>();
        COUNTS.forEach((k, v) -> out.put(k, v.sum()));
        return out;
    }

    /** Clear the histogram — call before a run whose census you want to read. */
    public static void reset() {
        COUNTS.clear();
        PLAN_REFUSED_IDS.clear();
        PLAN_REFUSAL_TAG.clear();
    }
}
