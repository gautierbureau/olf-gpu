/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.codes;

import com.powsybl.openloadflow.network.LfBus;

/**
 * A reactive-limit entry's limit type as the device holds it (the low bits of reactiveLimitsState, reactiveLimitBaseQLimitType): OLF's
 * {@link LfBus.QLimitType} (ordinal + 1), {@link #NONE} when the entry is not switched to PQ, and {@link #FROZEN}, the
 * device's mark for a voltage control OLF's GeneratorVoltageControlManager froze or whose generators are all lost.
 * Mirrored by native/cuda/common/codes/gpu_q_limit_type.cuh.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuQLimitType {

    /** Not switched to PQ: the bus controls its voltage (OLF: no QLimitType). */
    public static final int NONE = 0;
    public static final int MIN_Q = 1;
    public static final int MAX_Q = 2;
    public static final int MIN_REALISTIC_V = 3;
    public static final int MAX_REALISTIC_V = 4;
    /** Device only: voltage control frozen or all generators lost; the reactive limits pass leaves it alone. */
    public static final int FROZEN = 5;

    private GpuQLimitType() {
    }

    /** The bus's OLF QLimitType, NONE when it has none. */
    public static byte of(LfBus bus) {
        return bus.getQLimitType().map(t -> switch (t) {
            case MIN_Q -> MIN_Q;
            case MAX_Q -> MAX_Q;
            case MIN_REALISTIC_V -> MIN_REALISTIC_V;
            case MAX_REALISTIC_V -> MAX_REALISTIC_V;
        }).orElse(NONE).byteValue();
    }

    /** OLF's QLimitType of an entry switched to PQ (MIN_Q to MAX_REALISTIC_V). */
    public static LfBus.QLimitType toOlf(int limitType) {
        return switch (limitType) {
            case MIN_Q -> LfBus.QLimitType.MIN_Q;
            case MAX_Q -> LfBus.QLimitType.MAX_Q;
            case MIN_REALISTIC_V -> LfBus.QLimitType.MIN_REALISTIC_V;
            default -> LfBus.QLimitType.MAX_REALISTIC_V;
        };
    }
}
