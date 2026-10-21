/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import org.slf4j.Logger;

import java.util.Arrays;

/**
 * Per-thread account of the controllers a phase builder could NOT represent.
 *
 * <p>Every builder walks OLF's controllers and skips the ones it cannot express — a control whose
 * variable is not in the equation system, a branch the extraction does not pack. Skipping was only ever
 * logged, which hides the case that matters: when a builder skips SOME of a loop's controllers the list
 * is still non-empty, the kernel takes the loop, and the skipped controllers are silently dropped — the
 * device then runs a loop OLF runs differently.
 *
 * <p>So the rule a caller wants is all-or-nothing: drive a loop on the device only when its
 * representation is COMPLETE. This is what lets it ask.
 *
 * <p>Counts are per thread because the batched security analysis builds on OLF's partition threads.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuBatchSkips {

    /** The outer loop a skipped controller belongs to. */
    public enum Slot {
        TVC, SHUNT, PHASE, RPC, SVC, MONITORING
    }

    private static final ThreadLocal<int[]> COUNTS = ThreadLocal.withInitial(() -> new int[Slot.values().length]);

    private GpuBatchSkips() {
    }

    /** Record one controller this builder could not represent, and log it as before. */
    public static void record(Slot slot, Logger logger, String message, Object... args) {
        COUNTS.get()[slot.ordinal()]++;
        logger.info(message, args);
    }

    /** Forget what this thread has recorded — call before building a run's phase tables. */
    public static void reset() {
        Arrays.fill(COUNTS.get(), 0);
    }

    /** How many controllers of this slot the builders could not represent. */
    public static int count(Slot slot) {
        return COUNTS.get()[slot.ordinal()];
    }

    /** True when every controller of this slot was represented. */
    public static boolean complete(Slot slot) {
        return count(slot) == 0;
    }

    /** A one-line summary of everything skipped, or an empty string. */
    public static String summary() {
        StringBuilder sb = new StringBuilder();
        for (Slot s : Slot.values()) {
            if (count(s) > 0) {
                sb.append(sb.length() == 0 ? "" : " ").append(s).append('=').append(count(s));
            }
        }
        return sb.toString();
    }
}
