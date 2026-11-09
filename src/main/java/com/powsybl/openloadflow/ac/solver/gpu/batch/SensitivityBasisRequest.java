/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuRowMode;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuSensitivityBasisKind;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuSensitivityFunction;

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
 * column {@code j} at gather row {@code f} is entry {@code j * gatherRows.length + f}. With a {@link Contraction}, the device
 * combines those solutions into each factor group's solved state and contracts it with each factor's function gradient at
 * the converged state, and the result is the factors' sensitivities instead ({@link #resultLength}): only those
 * values leave the device, with the row modes the combination was decided on.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record SensitivityBasisRequest(int[] unitRows, int[] weightedStart, int[] weightedRows, double[] weightedValues,
                                      int[] gatherRows, Map<String, SparseColumn> participationByContingency,
                                      Contraction contraction) {

    /**
     * The contraction of the basis solutions into the factors' sensitivities, on the device.
     *
     * <ul>
     *   <li>Per unit column {@code u}: the stable row of its bus's active power balance ({@code unitBusPRow}), whose row mode
     *       the coefficient reads with the unit row's own; {@code slackPRow} the slack bus's.</li>
     *   <li>Per group, CSR ({@code groupColumnStart}): the basis columns it combines ({@code groupColumns}) and how
     *       ({@code groupColumnKind}, a {@link GpuSensitivityBasisKind}): see {@link #coefficient}.</li>
     *   <li>Per factor, in OLF's order of the VALID factors: its group, its function (a {@link GpuSensitivityFunction}), its closed
     *       branch (the device pack's index, -1 for a bus voltage) and four gather positions: a branch's angle and voltage
     *       of side 1, then of side 2; a bus voltage's in the first, -1 in the others.</li>
     * </ul>
     *
     * <p>A scenario's result is its factors' sensitivities, then per unit column the modes of its row and of its bus's
     * active power row: {@code factorGroup.length + 2 * unitRows.length} values.
     */
    public record Contraction(int[] unitBusPRow, int slackPRow, int[] groupColumnStart, int[] groupColumns, int[] groupColumnKind,
                              int[] factorGroup, int[] factorFunction, int[] factorElement, int[] factorGather) {
        public Contraction {
            Objects.requireNonNull(unitBusPRow);
            Objects.requireNonNull(groupColumnStart);
            Objects.requireNonNull(groupColumns);
            Objects.requireNonNull(groupColumnKind);
            Objects.requireNonNull(factorGroup);
            Objects.requireNonNull(factorFunction);
            Objects.requireNonNull(factorElement);
            Objects.requireNonNull(factorGather);
            if (groupColumns.length != groupColumnKind.length || factorFunction.length != factorGroup.length
                    || factorElement.length != factorGroup.length || factorGather.length != 4 * factorGroup.length) {
                throw new IllegalArgumentException("contraction tables of inconsistent lengths");
            }
        }

        public int factorCount() {
            return factorGroup.length;
        }

        /**
         * The coefficient a group gives a basis column of {@code kind} in a scenario, from that scenario's row mode on the
         * column's row ({@code rowMode}) and on its bus's active power row ({@code busPRowMode}): the device's rule
         * (sensitivity_basis.cu), which this mirrors. A row mode {@link GpuRowMode#FIXED_VARIABLE} is an identity (a voltage
         * held, an angle reference, an islanded bus's frozen rows), any other a balance. A bus's active power balance (P) and reactive power balance (Q) are active on a balance
         * row; its voltage target (V) on an identity voltage row of a bus that is not islanded (its active power row a
         * balance, or the slack's).
         */
        public static double coefficient(int kind, int rowMode, int busPRowMode, boolean slackBus) {
            return switch (kind) {
                case GpuSensitivityBasisKind.P, GpuSensitivityBasisKind.Q -> rowMode != GpuRowMode.FIXED_VARIABLE ? 1 : 0;
                case GpuSensitivityBasisKind.V -> rowMode == GpuRowMode.FIXED_VARIABLE
                        && (busPRowMode != GpuRowMode.FIXED_VARIABLE || slackBus) ? 1 : 0;
                default -> 1;
            };
        }
    }

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

    /** The values a scenario's result holds: the basis solutions at the gather rows, or with a contraction the factors'
     *  sensitivities and the unit columns' row modes. */
    public int resultLength() {
        return contraction != null ? contraction.factorCount() + 2 * unitRows.length : columnCount() * gatherRows.length;
    }

    /** The participation column of a contingency (empty when it has none). */
    public SparseColumn participation(String contingencyId) {
        if (participationByContingency == null) {
            return SparseColumn.EMPTY;
        }
        return participationByContingency.getOrDefault(contingencyId, SparseColumn.EMPTY);
    }
}
