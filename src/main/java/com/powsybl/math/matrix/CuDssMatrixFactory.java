/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.math.matrix;

/**
 * Sparse matrix factory whose matrices decompose with NVIDIA cuDSS (GPU) instead
 * of KLU. The matrix storage is the same {@link SparseMatrix} (CSC); only
 * {@link SparseMatrix#decomposeLU()} is overridden to return a
 * {@link CuDssLUDecomposition}.
 *
 * <p>cuDSS is optional: check {@link #isAvailable()} and fall back to
 * {@link SparseMatrixFactory} when it returns false (no cuDSS library, no GPU).
 *
 * @see SparseMatrix
 * @see CuDssLUDecomposition
 */
public class CuDssMatrixFactory implements MatrixFactory {

    /**
     * @return true if the cuDSS native library could be loaded.
     */
    public static boolean isAvailable() {
        return CuDssLUDecomposition.isAvailable();
    }

    @Override
    public SparseMatrix create(int rowCount, int columnCount, int estimatedValueCount) {
        return new CuDssSparseMatrix(rowCount, columnCount, estimatedValueCount);
    }

    /**
     * A {@link SparseMatrix} that decomposes with cuDSS.
     */
    private static final class CuDssSparseMatrix extends SparseMatrix {

        private CuDssSparseMatrix(int rowCount, int columnCount, int estimatedValueCount) {
            super(rowCount, columnCount, estimatedValueCount);
        }

        @Override
        public LUDecomposition decomposeLU() {
            fillTrailingEmptyColumns(this);
            return new CuDssLUDecomposition(this);
        }
    }

    /**
     * What {@code SparseMatrix.fillLastEmptyColumns()} (private) does before a decomposition: every column after the
     * last one holding a value starts at the value count, so that the CSC column starts are complete. {@code set}
     * counts each value it adds to its column, so the last column holding a value is the last one filled.
     */
    private static void fillTrailingEmptyColumns(SparseMatrix matrix) {
        int[] columnStart = matrix.getColumnStart();
        int[] columnValueCount = matrix.getColumnValueCount();
        int lastFilled = columnValueCount.length - 1;
        while (lastFilled >= 0 && columnValueCount[lastFilled] == 0) {
            lastFilled--;
        }
        int valueCount = matrix.getValueCount();
        for (int k = lastFilled + 1; k < columnValueCount.length; k++) {
            columnStart[k] = valueCount;
        }
    }
}
