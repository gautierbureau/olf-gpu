/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sensi;

import com.powsybl.math.matrix.LUDecomposition;
import com.powsybl.math.matrix.Matrix;
import com.powsybl.math.matrix.MatrixFactory;
import com.powsybl.math.matrix.DenseMatrix;
import com.powsybl.math.matrix.SparseMatrix;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

/**
 * A {@link Matrix} that records what is written into it instead of storing it, so that OLF's own
 * {@code SensitivityFactorGroup.fillRhs} can build the factor right-hand sides straight into a
 * sparse form the device can consume.
 *
 * <p>The point is fidelity: the right-hand sides are not re-derived here, they are the ones OLF
 * writes, produced by OLF's code. Only the storage is different — {@code nnz} entries instead of an
 * {@code equationCount x groupCount} dense matrix, which for a large network with many injection
 * groups is the difference between a few hundred kilobytes and several gigabytes.
 *
 * <p>Everything a right-hand side build does not need throws.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class SparseColumnRecorder implements Matrix {

    private final int rowCount;
    private final int columnCount;
    private final List<List<Entry>> columns;

    private record Entry(int row, double value) {
    }

    SparseColumnRecorder(int rowCount, int columnCount) {
        this.rowCount = rowCount;
        this.columnCount = columnCount;
        this.columns = new ArrayList<>(columnCount);
        for (int i = 0; i < columnCount; i++) {
            columns.add(new ArrayList<>());
        }
    }

    /** Column pointers of the recorded CSC ({@code columnCount + 1}). */
    int[] columnPointers() {
        int[] colPtr = new int[columnCount + 1];
        for (int j = 0; j < columnCount; j++) {
            colPtr[j + 1] = colPtr[j] + columns.get(j).size();
        }
        return colPtr;
    }

    int[] rows() {
        int[] rows = new int[nonZeroCount()];
        int i = 0;
        for (List<Entry> column : columns) {
            for (Entry e : column) {
                rows[i++] = e.row();
            }
        }
        return rows;
    }

    double[] values() {
        double[] values = new double[nonZeroCount()];
        int i = 0;
        for (List<Entry> column : columns) {
            for (Entry e : column) {
                values[i++] = e.value();
            }
        }
        return values;
    }

    int nonZeroCount() {
        int count = 0;
        for (List<Entry> column : columns) {
            count += column.size();
        }
        return count;
    }

    @Override
    public int getRowCount() {
        return rowCount;
    }

    @Override
    public int getColumnCount() {
        return columnCount;
    }

    @Override
    public void set(int i, int j, double value) {
        // fillRhs only ever sets a cell it has not touched (one entry per variable equation)
        add(i, j, value);
    }

    @Override
    public void add(int i, int j, double value) {
        if (i < 0 || i >= rowCount || j < 0 || j >= columnCount) {
            throw new IndexOutOfBoundsException("(" + i + ", " + j + ") outside " + rowCount + " x " + columnCount);
        }
        // the device fill kernel accumulates, so a repeated (row, column) is summed there as here
        columns.get(j).add(new Entry(i, value));
    }

    @Override
    public void reset() {
        columns.forEach(List::clear);
    }

    private static UnsupportedOperationException unsupported() {
        return new UnsupportedOperationException("SparseColumnRecorder only records a right-hand side build");
    }

    @Override
    public Element addAndGetElement(int i, int j, double value) {
        throw unsupported();
    }

    @Override
    public int addAndGetIndex(int i, int j, double value) {
        throw unsupported();
    }

    @Override
    public void setAtIndex(int index, double value) {
        throw unsupported();
    }

    @Override
    public void setQuickAtIndex(int index, double value) {
        throw unsupported();
    }

    @Override
    public void addAtIndex(int index, double value) {
        throw unsupported();
    }

    @Override
    public void addQuickAtIndex(int index, double value) {
        throw unsupported();
    }

    @Override
    public LUDecomposition decomposeLU() {
        throw unsupported();
    }

    @Override
    public Matrix times(Matrix other, double scalar) {
        throw unsupported();
    }

    @Override
    public Matrix times(Matrix other) {
        throw unsupported();
    }

    @Override
    public Matrix add(Matrix other, double alpha, double beta) {
        throw unsupported();
    }

    @Override
    public Matrix add(Matrix other) {
        throw unsupported();
    }

    @Override
    public void iterateNonZeroValue(ElementHandler handler) {
        throw unsupported();
    }

    @Override
    public void iterateNonZeroValueOfColumn(int j, ElementHandler handler) {
        throw unsupported();
    }

    @Override
    public DenseMatrix toDense() {
        throw unsupported();
    }

    @Override
    public SparseMatrix toSparse() {
        throw unsupported();
    }

    @Override
    public Matrix to(MatrixFactory factory) {
        throw unsupported();
    }

    @Override
    public Matrix copy(MatrixFactory factory) {
        throw unsupported();
    }

    @Override
    public Matrix transpose() {
        throw unsupported();
    }

    @Override
    public void print(PrintStream out, List<String> rowNames, List<String> columnNames) {
        throw unsupported();
    }

    @Override
    public void print(PrintStream out) {
        throw unsupported();
    }
}
