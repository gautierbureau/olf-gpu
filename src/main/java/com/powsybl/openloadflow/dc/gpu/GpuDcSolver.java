/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.dc.gpu;

import com.powsybl.math.matrix.SparseMatrix;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;

import java.util.Objects;

/**
 * The DC device context: one cuDSS factorization of OLF's DC matrix, reused against many
 * right-hand sides, with the per-column work (Woodbury combination, branch flows, limit screen)
 * done on the device — {@code native/cuda/jni/olf_gpu_dc_jni.cu}, in the same {@code libolfgpu}
 * as the AC Newton-Raphson.
 *
 * <p>The matrix is handed over as OLF's own {@link SparseMatrix} arrays: the column-compressed
 * storage of the transposed Jacobian OLF keeps IS the row-compressed storage of the true matrix
 * (rows = equations, columns = variables), so the device solves {@code J x = b} directly. Every
 * row index is the equation system's own {@code Variable.getRow()} / {@code Equation.getColumn()}.
 *
 * <p>Not thread-safe: one context serves one security analysis on one thread, like the AC batch
 * contexts. {@link #close()} frees the device memory; the {@link java.lang.ref.Cleaner} is a
 * safety net for a forgotten close.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuDcSolver implements AutoCloseable {

    /** Elements per contingency the alpha kernel solves in registers (MAX_K in the native). */
    public static final int MAX_ELEMENTS = 8;

    private static final java.lang.ref.Cleaner CLEANER = java.lang.ref.Cleaner.create();

    private final int n;
    private final State state;
    private final java.lang.ref.Cleaner.Cleanable cleanable;

    /** The native handle, owned by the cleaner action so a forgotten close still frees the device. */
    private static final class State implements Runnable {
        private long handle;

        private State(long handle) {
            this.handle = handle;
        }

        @Override
        public void run() {
            if (handle != 0) {
                destroyContext(handle);
                handle = 0;
            }
        }
    }

    private GpuDcSolver(int n, long handle) {
        this.n = n;
        this.state = new State(handle);
        this.cleanable = CLEANER.register(this, state);
    }

    /** True when {@code libolfgpu} (and its cuDSS/CUDA chain) loaded — the AC loader's verdict. */
    public static boolean isAvailable() {
        return GpuAcNewtonSolver.isAvailable();
    }

    /**
     * Factorize the DC matrix on the device.
     *
     * @param matrix OLF's Jacobian matrix (rows = variables, columns = equations, square)
     */
    public static GpuDcSolver create(SparseMatrix matrix) {
        Objects.requireNonNull(matrix);
        if (!isAvailable()) {
            throw new IllegalStateException("GPU native library not available");
        }
        int n = matrix.getRowCount();
        if (matrix.getColumnCount() != n) {
            throw new IllegalArgumentException("DC matrix is not square: " + n + " x " + matrix.getColumnCount());
        }
        int[] columnStart = matrix.getColumnStart();
        int nnz = columnStart[n];
        int[] rowIndices = matrix.getRowIndices();
        double[] values = matrix.getValues();
        if (rowIndices.length != nnz || values.length != nnz) {
            // the backing arrays may be larger than the filled part
            rowIndices = java.util.Arrays.copyOf(rowIndices, nnz);
            values = java.util.Arrays.copyOf(values, nnz);
        }
        long handle = createContext(n, columnStart, rowIndices, values);
        return new GpuDcSolver(n, handle);
    }

    public int size() {
        return n;
    }

    private long handle() {
        if (state.handle == 0) {
            throw new IllegalStateException("GpuDcSolver is closed");
        }
        return state.handle;
    }

    /** Solve {@code J x = rhs} for one right-hand side (the pre-contingency and modified-target solves). */
    public double[] solve(double[] rhs) {
        return solve(handle(), rhs);
    }

    /**
     * The contingency elements: {@code +1} at {@code rowPlus[e]}, {@code -1} at {@code rowMinus[e]}
     * ({@code -1} = no entry, the slack side), one device-resident S column per element.
     *
     * @return false when the device cannot hold S — route the run to the CPU
     */
    public boolean setElements(int[] rowPlus, int[] rowMinus, int[] elemBranch) {
        return setElements(handle(), rowPlus, rowMinus, elemBranch);
    }

    /** One S column back on the host (the LODF/PTDF oracle in tests). */
    public double[] readElementStates(int elem) {
        return readElementStates(handle(), elem);
    }

    /**
     * The branch model in {@code LfNetwork.getBranches()} order: the angle rows of the two ends
     * ({@code -1} when a side has no angle variable — open or zero-impedance branch: flow 0), the
     * phase-shift variable row or {@code -1} with the constant shift, and OLF's {@code power}.
     */
    public void setBranches(int[] ph1Row, int[] ph2Row, int[] a1Row, double[] a1Const, double[] power) {
        setBranches(handle(), ph1Row, ph2Row, a1Row, a1Const, power);
    }

    /** The pre-contingency state the Woodbury correction is applied to. */
    public void setBase(double[] x0) {
        setBase(handle(), x0);
    }

    /** The flattened limit table (GpuLimitTable's arrays) the screen compares against. */
    public void setLimitTable(int[] branchGroupStart, int[] branchGroupEnd, int[] groupQuantity,
                              int[] groupLimitStart, int[] groupLimitEnd, double[] limitThreshold, double dcPowerFactor) {
        setLimitTable(handle(), branchGroupStart, branchGroupEnd, groupQuantity, groupLimitStart, groupLimitEnd, limitThreshold, dcPowerFactor);
    }

    /**
     * One chunk of contingencies (CSR over element indices). Fills {@code outStates} with the
     * {@code n x nCtg} post-contingency states (column-major) and returns
     * {@code [counts(nCtg) | records(nCtg x capacity)]} — the limit-table candidates per
     * contingency; a count above {@code capacity} means "screen this one on the host".
     */
    public int[] woodburyChunk(int[] ctgPtr, int[] ctgElem, double band, int capacity, double[] outStates) {
        return woodburyChunk(handle(), ctgPtr, ctgElem, band, capacity, outStates);
    }

    @Override
    public void close() {
        cleanable.clean();
    }

    private static native long createContext(int n, int[] rowPtr, int[] colIdx, double[] val);

    private static native void destroyContext(long handle);

    private static native double[] solve(long handle, double[] rhs);

    private static native boolean setElements(long handle, int[] rowPlus, int[] rowMinus, int[] elemBranch);

    private static native double[] readElementStates(long handle, int elem);

    private static native void setBranches(long handle, int[] ph1Row, int[] ph2Row, int[] a1Row, double[] a1Const, double[] power);

    private static native void setBase(long handle, double[] x0);

    private static native void setLimitTable(long handle, int[] branchGroupStart, int[] branchGroupEnd, int[] groupQuantity,
                                             int[] groupLimitStart, int[] groupLimitEnd, double[] limitThreshold, double dcPowerFactor);

    private static native int[] woodburyChunk(long handle, int[] ctgPtr, int[] ctgElem, double band, int capacity, double[] outStates);
}
