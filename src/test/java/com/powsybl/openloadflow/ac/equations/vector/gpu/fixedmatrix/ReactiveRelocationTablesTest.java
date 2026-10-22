/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix;

import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuRowMode;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The Java carrier pick must choose the row the kernel chooses (native chooseCarrierRow,
 * fixedmatrix/reactive_relocation.cuh): the first member candidate whose row holds neither a fixed variable
 * ({@code FIXED_VARIABLE}, {@code CROSS_FIXED_VARIABLE}) nor a bus's own reactive balance ({@code BUS_TARGET_Q}). Otherwise the
 * host seeds the pilot's reactive target on a row the device does not read.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class ReactiveRelocationTablesTest {

    /** One group: candidate 0 (the fallback, row 0), then member rows 1, 2, 3. */
    private static final ReactiveRelocationCandidates ONE_GROUP = new ReactiveRelocationCandidates(
            new int[] {0, 4}, new int[] {0, 1, 2, 3}, new int[] {-1}, new int[] {-1}, new int[0]);

    private static int[] modes(int row1, int row2, int row3) {
        return new int[] {GpuRowMode.EQUATION, row1, row2, row3};
    }

    @Test
    void skipsEveryRowThatHoldsAFixedVariableOrABusTargetQ() {
        // a member whose own V row now holds a cross identity (a re-merged control under a live coupler) is not free
        assertEquals(3, ReactiveRelocationTables.chooseCarrierRow(ONE_GROUP, 0,
                modes(GpuRowMode.CROSS_FIXED_VARIABLE, GpuRowMode.BUS_TARGET_Q, GpuRowMode.EQUATION)));
        assertEquals(2, ReactiveRelocationTables.chooseCarrierRow(ONE_GROUP, 0,
                modes(GpuRowMode.FIXED_VARIABLE, GpuRowMode.EQUATION, GpuRowMode.EQUATION)));
    }

    @Test
    void noFreeMemberMeansNoCarrier() {
        assertEquals(-1, ReactiveRelocationTables.chooseCarrierRow(ONE_GROUP, 0,
                modes(GpuRowMode.CROSS_FIXED_VARIABLE, GpuRowMode.FIXED_VARIABLE, GpuRowMode.BUS_TARGET_Q)));
    }
}
