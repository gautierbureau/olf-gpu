/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix;

/**
 * docs/gpu-vs-olf.md section 2.4 (cross identity): a fixed variable whose own row is taken. A re-merged
 * voltage-controlled bus whose own {@code BUS_V} row carries a LIVE coupler's {@code ZERO_V} (the coupler-tree
 * relocation put it there, {@link CouplerTreeRelocation}) keeps its {@code BUS_TARGET_V} on the parent coupler's
 * {@code DUMMY_Q} row, on the bus's own V column: row mode 6, {@code GpuRowMode.CROSS_FIXED_VARIABLE}. The planners
 * choose the row here; the device writes it in four kernels (native {@code fixedmatrix/cross_identity.cuh}): the
 * fills' {@code applyRowModeBatch}, the reactive-limit pass and its seed (the pin row), and the pilot voltage and
 * remote voltage control kernels (a row holding a fixed variable).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class CrossIdentity {

    private CrossIdentity() {
    }

    /** The row a cross identity of the bus whose own {@code BUS_V} row is {@code vRow} lands on in THIS scenario:
     *  its parent coupler's {@code DUMMY_Q} row while that coupler stays closed
     *  ({@link CouplerTreeReHoming#scenarioReactiveRowOfVRow}), -1 when the bus's own row is free (not a relocated
     *  child, or its coupler is outaged). {@code relocatedChildRows} is {@link CouplerTreeReHoming#relocatedChildRows}. */
    public static int row(int vRow, java.util.Map<Integer, int[]> relocatedChildRows, java.util.Set<Integer> outagedZeroImpedanceBranchNums) {
        int crossRow = CouplerTreeReHoming.scenarioReactiveRowOfVRow(vRow, relocatedChildRows, outagedZeroImpedanceBranchNums);
        return crossRow != vRow ? crossRow : -1;
    }
}
