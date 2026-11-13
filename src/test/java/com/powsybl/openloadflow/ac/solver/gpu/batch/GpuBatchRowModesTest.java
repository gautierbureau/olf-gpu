/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuRowMode;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The row modes of the slots one solve call installs: the base row's, except the rows a slot sets, and the entries
 * the device takes where they differ from the base. CPU only.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuBatchRowModesTest {

    private static final int[] BASE = {GpuRowMode.EQUATION, GpuRowMode.FIXED_VARIABLE, GpuRowMode.EQUATION, GpuRowMode.EQUATION};

    @Test
    void aSlotReadsTheBaseRowExceptTheRowsItSets() {
        GpuBatchRowModes rowModes = new GpuBatchRowModes(2, BASE);
        rowModes.set(0, 2, GpuRowMode.BUS_TARGET_Q);
        rowModes.set(0, 2, GpuRowMode.FIXED_VARIABLE);
        assertEquals(GpuRowMode.FIXED_VARIABLE, rowModes.get(0, 2), "the last write wins");
        assertEquals(GpuRowMode.FIXED_VARIABLE, rowModes.get(0, 1), "a row the slot did not set is the base row's");
        assertEquals(GpuRowMode.EQUATION, rowModes.get(1, 2), "another slot keeps the base row");
        rowModes.reset(0);
        assertEquals(GpuRowMode.EQUATION, rowModes.get(0, 2), "a reset slot is back on the base row");
    }

    @Test
    void entriesAreTheRowsThatDifferFromTheBaseInSlotThenRowOrder() {
        GpuBatchRowModes rowModes = new GpuBatchRowModes(3, BASE);
        rowModes.set(1, 3, GpuRowMode.BUS_TARGET_Q);
        rowModes.set(1, 0, GpuRowMode.FIXED_VARIABLE);
        rowModes.set(1, 1, GpuRowMode.FIXED_VARIABLE);              // its base mode: no entry
        rowModes.set(0, 2, GpuRowMode.CROSS_FIXED_VARIABLE);
        rowModes.set(2, 0, GpuRowMode.FIXED_VARIABLE);              // past the slots asked for
        assertArrayEquals(new int[] {0, 2, GpuRowMode.CROSS_FIXED_VARIABLE, 1, 0, GpuRowMode.FIXED_VARIABLE, 1, 3, GpuRowMode.BUS_TARGET_Q},
                rowModes.entries(2));
        assertArrayEquals(new int[0], new GpuBatchRowModes(2, BASE).entries(2), "every slot on the base row: no entry");
    }
}
