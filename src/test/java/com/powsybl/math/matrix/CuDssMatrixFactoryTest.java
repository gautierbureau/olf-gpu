/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.math.matrix;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/**
 * The column starts a cuDSS decomposition reads, as {@code SparseMatrix.fillLastEmptyColumns()} completes them.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class CuDssMatrixFactoryTest {

    private static int[] columnStartAfterDecomposition(SparseMatrix matrix) {
        try (LUDecomposition lu = matrix.decomposeLU()) {
            // cuDSS available: nothing more to check here
        } catch (MatrixException e) {
            // no cuDSS library: the column starts were completed before the decomposition was attempted
        }
        return matrix.getColumnStart();
    }

    @Test
    void trailingEmptyColumnsStartAtTheValueCount() {
        // 4 x 4, columns 0 and 1 filled, 2 and 3 empty
        SparseMatrix matrix = new CuDssMatrixFactory().create(4, 4, 3);
        matrix.set(0, 0, 1);
        matrix.set(1, 0, 2);
        matrix.set(1, 1, 3);
        assertArrayEquals(new int[] {0, 2, 3, 3, 3}, columnStartAfterDecomposition(matrix));
    }

    @Test
    void anEmptyColumnBetweenFilledOnesKeepsItsStart() {
        // column 1 empty between filled columns 0 and 2, column 3 empty at the end
        SparseMatrix matrix = new CuDssMatrixFactory().create(4, 4, 2);
        matrix.set(0, 0, 1);
        matrix.set(2, 2, 2);
        assertArrayEquals(new int[] {0, 1, 1, 2, 2}, columnStartAfterDecomposition(matrix));
    }

    @Test
    void anEmptyMatrixStartsEveryColumnAtZero() {
        SparseMatrix matrix = new CuDssMatrixFactory().create(3, 3, 0);
        assertArrayEquals(new int[] {0, 0, 0, 0}, columnStartAfterDecomposition(matrix));
    }
}
