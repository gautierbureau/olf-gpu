/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import java.util.Map;
import java.util.Objects;

/**
 * The right-hand sides a batched sensitivity analysis has the device solve at every scenario's converged state, on that
 * state's Jacobian, and the rows of the solutions it reads back.
 *
 * <p>The basis columns are, in this order: one unit column per entry of {@code unitRows}; one weighted column per
 * multi-variable factor group ({@code weightedStart}/{@code weightedRows}/{@code weightedValues}, CSR by column); and,
 * when {@code participationByContingency} is not null, one column per scenario: its contingency's slack participation.
 * Every row is a STABLE equation row (the diagonal-paired variable row of the base extraction: a bus's P row is its
 * BUS_PHI variable row, its Q and voltage-target row its BUS_V variable row), which the device maps to its own rows.
 * {@code gatherRows} are base variable rows: the rows of each solution read back, in that order.
 *
 * <p>A scenario's result is then {@code columnCount() * gatherRows.length} values, column-major: the solution of basis
 * column {@code j} at gather row {@code f} is entry {@code j * gatherRows.length + f}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record SensitivityBasisRequest(int[] unitRows, int[] weightedStart, int[] weightedRows, double[] weightedValues,
                                      int[] gatherRows, Map<String, SparseColumn> participationByContingency) {

    /** A sparse column over stable equation rows. */
    public record SparseColumn(int[] rows, double[] values) {
        public SparseColumn {
            Objects.requireNonNull(rows);
            Objects.requireNonNull(values);
            if (rows.length != values.length) {
                throw new IllegalArgumentException("one value per row");
            }
        }

        public static final SparseColumn EMPTY = new SparseColumn(new int[0], new double[0]);
    }

    public SensitivityBasisRequest {
        Objects.requireNonNull(unitRows);
        Objects.requireNonNull(weightedStart);
        Objects.requireNonNull(weightedRows);
        Objects.requireNonNull(weightedValues);
        Objects.requireNonNull(gatherRows);
        if (weightedStart.length == 0 || weightedStart[0] != 0 || weightedStart[weightedStart.length - 1] != weightedRows.length
                || weightedRows.length != weightedValues.length) {
            throw new IllegalArgumentException("weighted columns: CSR start with a leading 0 and one value per row");
        }
    }

    public int weightedColumnCount() {
        return weightedStart.length - 1;
    }

    public boolean hasParticipation() {
        return participationByContingency != null;
    }

    /** The basis columns a scenario solves. */
    public int columnCount() {
        return unitRows.length + weightedColumnCount() + (hasParticipation() ? 1 : 0);
    }

    /** The index of the participation column, -1 without one. */
    public int participationColumn() {
        return hasParticipation() ? unitRows.length + weightedColumnCount() : -1;
    }

    /** The values a scenario's result holds. */
    public int resultLength() {
        return columnCount() * gatherRows.length;
    }

    /** The participation column of a contingency (empty when it has none). */
    public SparseColumn participation(String contingencyId) {
        if (participationByContingency == null) {
            return SparseColumn.EMPTY;
        }
        return participationByContingency.getOrDefault(contingencyId, SparseColumn.EMPTY);
    }
}
