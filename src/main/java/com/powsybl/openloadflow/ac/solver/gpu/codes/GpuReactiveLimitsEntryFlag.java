/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.codes;

/**
 * Bit flags of a reactive-limit entry (reactiveLimitFlags). Mirrored by
 * native/cuda/jni/common/codes/gpu_reactive_limits_entry_flag.cuh.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuReactiveLimitsEntryFlag {

    /** The bus holds a static var compensator standing by under OLF's voltage monitoring (OLF's StandByAutomaton):
     *  seeded unpinned, only the monitoring loop activates it. */
    public static final int STAND_BY_AUTOMATON = 1;
    /** The bus is not one of its voltage control's controller elements, so the degenerate-range guard may skip it
     *  (OLF's loop walks controller elements only). */
    public static final int NOT_CONTROLLER_ELEMENT = 2;

    private GpuReactiveLimitsEntryFlag() {
    }
}
