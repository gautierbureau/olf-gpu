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
 * there is clamped and counted. The fast engine learns from the matrices it did not trust: a pivot that cancelled
 * there is delayed after the row that gives it a value back, and the next factorization on the same plan runs on that
 * order ({@link #factorizeRounds}; every other entry point builds its plan for one factorization).
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

    /**
     * {the factorizations our batched LU ran in this process, the lanes of them its pivoting engine factorized, the
     * fast-engine orders built with learned pivot delays, the fast-engine factorization steps and solve levels launched
     * with their long rows split over several warps, the pivoting-engine forward solve levels launched with their long
     * rows split, the lanes sent straight to the pivoting engine, past the fast one, because the pivoting engine
     * factorized them last, the lanes the fast engine's triangular solves ran on, the pivoting-engine backward solve
     * levels launched with a supernode's rows on separate warps, its split forward solve levels launched with a
     * supernode's rows on separate blocks, its diagonal blocks factorized with their rows on separate warps}.
     */
    public static native long[] factorizationCounts();

    /** The batch contexts this process built with a cuDSS handle: every one but those that solve on our batched LU. */
    public static native long cudssContexts();

    /**
     * Tests: true makes the pivoting engine run its one-thread-per-supernode kernels where it would spread a supernode's
     * rows (backward and split forward solve levels, the factorization of the diagonal blocks of FACTOR_ROWS_MIN rows or
     * more) - the forms the others must match bit for bit. False (the default) restores the spread ones.
     */
    public static native void setPivotingEngineOneThreadKernelsForTests(boolean on);

    /**
     * The fast engine's plan: {LU entries (the exact fill of the unsymmetric pattern), contributions (multiply-adds of
     * one factorization), factorization steps, forward-solve levels, backward-solve levels}.
     */
    public static native long[] scalarPlanStatistics(int n, int[] rowPtr, int[] colIdx, int[] rowOrder, int[] colOrder);

    /** The exact minimum-degree elimination order of A + A^T (position -> original index): GpuBatchedLuOrdering.MINIMUM_DEGREE. */
    public static native int[] minimumDegreeOrder(int n, int[] rowPtr, int[] colIdx);

    /** The elimination-tree height of A + A^T eliminated in {@code order}: what GpuBatchedLuOrdering.FEWEST_LEVELS compares. */
    public static native int eliminationLevels(int n, int[] rowPtr, int[] colIdx, int[] order);

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
     * {@link #factorizeAndSolve} with the values handed to the factorization interleaved by scenario, as a batch
     * context holds them: the engines then read them in place instead of copying them. The arrays are the same,
     * scenario-major.
     */
    public static native void factorizeAndSolveInterleaved(int n, int[] rowPtr, int[] colIdx, int[] rowOrder,
                                                           int[] colOrder, int capacity, double[] values,
                                                           double[] rightHandSides, int[] laneMap, double[] solutions,
                                                           int[] pivoted, int[] pivotClamps);

    /**
     * Factorizes one system on one lane, then solves it against {@code cols} right-hand sides at once, on that lane's
     * factors (the DC path's single matrix with many right-hand sides).
     *
     * @param values         CSR values, {@code nnz}
     * @param rightHandSides {@code n * cols}, column-major
     * @param solutions      {@code n * cols}, column-major (written)
     */
    public static native void factorizeAndSolveColumns(int n, int[] rowPtr, int[] colIdx, int[] rowOrder, int[] colOrder,
                                                       double[] values, int cols, double[] rightHandSides, double[] solutions);

    /**
     * Factorizes the scenarios of {@code laneMap} as {@link #factorizeAndSolve} does, then solves the systems of those
     * in {@code selected} only, as the adaptive refinement solves its corrections on the lanes still refining. Every
     * other scenario's solution is left as it is.
     *
     * @param capacity       scenarios in the arrays
     * @param values         CSR values, {@code capacity * nnz}, scenario-major
     * @param rightHandSides {@code capacity * n}, scenario-major
     * @param laneMap        the scenarios to factorize
     * @param selected       the scenarios to solve, among them
     * @param solutions      {@code capacity * n}, scenario-major: the selected scenarios' entries are overwritten
     */
    public static native void factorizeAndSolveSelected(int n, int[] rowPtr, int[] colIdx, int[] rowOrder, int[] colOrder,
                                                        int capacity, double[] values, double[] rightHandSides,
                                                        int[] laneMap, int[] selected, double[] solutions);

    /**
     * The Newton step's linear solve of a structure with zero-impedance couplers on the scenarios of {@code laneMap}:
     * a factorization, the solve with cuDSS's in-solve refinement steps and the adaptive refinement, as a batch context
     * on our LU runs them. Every other scenario's solution is left as it is.
     *
     * @param capacity       scenarios in the arrays
     * @param values         CSR values, {@code capacity * nnz}, scenario-major
     * @param rightHandSides {@code capacity * n}, scenario-major
     * @param laneMap        the scenarios to factorize and solve
     * @param solutions      {@code capacity * n}, scenario-major: the solved scenarios' entries are overwritten
     */
    public static native void factorizeAndSolveNewtonStep(int n, int[] rowPtr, int[] colIdx, int[] rowOrder, int[] colOrder,
                                                          int capacity, double[] values, double[] rightHandSides,
                                                          int[] laneMap, double[] solutions);

    /**
     * Factorizes the scenarios of {@code laneMap} {@code rounds} times on one plan, the fast engine's order learning its
     * pivot delays between rounds, then solves their systems with the last factorization; every other scenario's
     * solution is left as it is.
     *
     * @param capacity       scenarios in the arrays
     * @param values         CSR values, {@code capacity * nnz}, scenario-major
     * @param rightHandSides {@code capacity * n}, scenario-major
     * @param laneMap        the scenarios to factorize and solve
     * @param installRound   the round before which every lane's scenario is installed anew, as a solve installs a
     *                       new scenario in a slot: it starts again on the fast engine; -1 for none
     * @param solutions      {@code capacity * n}, scenario-major: the solved scenarios' entries are overwritten
     * @param pivoted        {@code rounds * laneMap.length}, round-major: 1 when the pivoting engine factorized the lane
     *                       that round (written)
     * @param delays         per round: the pivot delays the fast engine's order had (written)
     */
    public static native void factorizeRounds(int n, int[] rowPtr, int[] colIdx, int[] rowOrder, int[] colOrder,
                                              int capacity, double[] values, double[] rightHandSides, int[] laneMap,
                                              int rounds, int installRound, double[] solutions, int[] pivoted,
                                              int[] delays);

    /**
     * {@link #factorizeRounds} with the values reaching the factorizations interleaved by scenario, as a batch context
     * holds them ({@code values} is given scenario-major and interleaved on the way).
     */
    public static native void factorizeRoundsInterleaved(int n, int[] rowPtr, int[] colIdx, int[] rowOrder,
                                                         int[] colOrder, int capacity, double[] values,
                                                         double[] rightHandSides, int[] laneMap, int rounds,
                                                         int installRound, double[] solutions, int[] pivoted,
                                                         int[] delays);

    /**
     * The orders cuDSS's analysis picks for the pattern, written into {@code rowOrder} and {@code colOrder} as cuDSS
     * returns them, and the LU fill it reports.
     */
    public static native long cudssReorder(int n, int[] rowPtr, int[] colIdx, int[] rowOrder, int[] colOrder);
}
