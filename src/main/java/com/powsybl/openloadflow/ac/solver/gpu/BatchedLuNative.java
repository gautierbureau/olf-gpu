/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;

/**
 * Our batched sparse LU alone, on an explicit CSR pattern: what the device factorization is checked with before the
 * Newton loop uses it.
 *
 * <p>Every matrix of a batch shares one CSR pattern ({@code rowPtr}, {@code colIdx}, columns ascending in each row)
 * and is eliminated in one order: row {@code i} of the eliminated matrix is row {@code rowOrder[i]}, its column
 * {@code j} is column {@code colOrder[j]}. Two engines: a fast one without pivoting factorizes every matrix, and a
 * matrix it does not trust (a pivot below 1e-13 in magnitude, or a multiplier above 1e8) is factorized again by a
 * pivoting one, which pivots inside the supernodes of that order (postordered) on its own; a pivot still below 1e-13
 * there is clamped and counted.
 *
 * <p>JNI glue: {@code native/cuda/jni/batched_lu_jni.cu}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class BatchedLuNative {

    static {
        GpuAcNewtonSolver.isAvailable();   // loads libolfgpu once
    }

    private BatchedLuNative() {
    }

    /** {the factorizations our batched LU ran in this process, the lanes of them its pivoting engine factorized}. */
    public static native long[] factorizationCounts();

    /**
     * The fast engine's plan: {LU entries (the exact fill of the unsymmetric pattern), contributions (multiply-adds of
     * one factorization), factorization steps, forward-solve levels, backward-solve levels}.
     */
    public static native long[] scalarPlanStatistics(int n, int[] rowPtr, int[] colIdx, int[] rowOrder, int[] colOrder);

    /**
     * The pivoting engine's plan: {stored entries, multiply-adds of the gathers of one factorization, supernode levels,
     * supernodes, columns of the largest supernode, L + U entries of the symmetrized structure (the stored entries
     * less the zeros the supernodes are padded with)}.
     */
    public static native long[] supernodalPlanStatistics(int n, int[] rowPtr, int[] colIdx, int[] rowOrder, int[] colOrder);

    /**
     * Factorizes the scenarios of {@code laneMap} and solves their systems, every other scenario's solution left as
     * it is.
     *
     * @param capacity       scenarios in the arrays
     * @param values         CSR values, {@code capacity * nnz}, scenario-major
     * @param rightHandSides {@code capacity * n}, scenario-major
     * @param laneMap        the scenarios to factorize and solve
     * @param solutions      {@code capacity * n}, scenario-major: the solved scenarios' entries are overwritten
     * @param pivoted        per entry of {@code laneMap}: 1 when the pivoting engine factorized it (written)
     * @param pivotClamps    per entry of {@code laneMap}: the pivots the pivoting engine clamped (written)
     */
    public static native void factorizeAndSolve(int n, int[] rowPtr, int[] colIdx, int[] rowOrder, int[] colOrder,
                                                int capacity, double[] values, double[] rightHandSides, int[] laneMap,
                                                double[] solutions, int[] pivoted, int[] pivotClamps);

    /**
     * The orders cuDSS's analysis picks for the pattern, written into {@code rowOrder} and {@code colOrder} as cuDSS
     * returns them, and the LU fill it reports.
     */
    public static native long cudssReorder(int n, int[] rowPtr, int[] colIdx, int[] rowOrder, int[] colOrder);
}
