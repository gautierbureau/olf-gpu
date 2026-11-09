/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.openloadflow.ac.solver.gpu.BatchedLuNative;
import com.powsybl.openloadflow.ac.solver.gpu.GpuLoadFlowParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.TreeMap;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Our batched sparse LU alone, against references written here: boolean Gaussian eliminations for the symbolic plans
 * (the fast engine's exact fill and multiply-adds, the pivoting engine's Cholesky fill of the symmetrized pattern), a
 * dense solve with partial pivoting for the numbers.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuBatchedLuTest {

    private record Csr(int n, int[] rowPtr, int[] colIdx) {
        int nnz() {
            return rowPtr[n];
        }
    }

    /** The line group of {@link #lineGroupValues}: rows {phi1, phi2}, {phi1, phi2, p}, {phi2, phi3}, {phi3, p}. */
    private static final Csr LINE_GROUP = new Csr(4, new int[] {0, 2, 5, 7, 9}, new int[] {0, 1, 0, 1, 3, 1, 2, 2, 3});

    @BeforeEach
    void checkGpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(),
                "GPU full-NR native lib not available (build it with native/build-gpu.sh + cuDSS)");
    }

    /**
     * An unsymmetric pattern with a full diagonal, shaped like a Jacobian: random couplings, and every fifth row
     * reduced to its diagonal (a PV bus's voltage row) while its column keeps its entries.
     */
    private static boolean[][] jacobianLikePattern(int n, double density, long seed) {
        Random random = new Random(seed);
        boolean[][] pattern = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            pattern[i][i] = true;
            for (int j = 0; j < n; j++) {
                if (j != i && random.nextDouble() < density) {
                    pattern[i][j] = true;
                    pattern[j][i] = random.nextDouble() < 0.8;
                }
            }
        }
        for (int i = 0; i < n; i += 5) {
            Arrays.fill(pattern[i], false);
            pattern[i][i] = true;
        }
        return pattern;
    }

    private static Csr toCsr(boolean[][] pattern) {
        int n = pattern.length;
        int[] rowPtr = new int[n + 1];
        for (int i = 0; i < n; i++) {
            int count = 0;
            for (boolean b : pattern[i]) {
                count += b ? 1 : 0;
            }
            rowPtr[i + 1] = rowPtr[i] + count;
        }
        int[] colIdx = new int[rowPtr[n]];
        int k = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (pattern[i][j]) {
                    colIdx[k++] = j;
                }
            }
        }
        return new Csr(n, rowPtr, colIdx);
    }

    private static int[] identity(int n) {
        int[] order = new int[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        return order;
    }

    private static int[] reversed(int n) {
        int[] order = new int[n];
        for (int i = 0; i < n; i++) {
            order[i] = n - 1 - i;
        }
        return order;
    }

    private static int[] shuffled(int n, long seed) {
        int[] order = identity(n);
        Random random = new Random(seed);
        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int t = order[i];
            order[i] = order[j];
            order[j] = t;
        }
        return order;
    }

    /**
     * Gaussian elimination without pivoting on the pattern alone, in the given order: {LU entries, multiply-adds}. A
     * diagonal entry never filled counts as one, as the plan keeps it.
     */
    private static long[] booleanElimination(boolean[][] pattern, int[] rowOrder, int[] colOrder) {
        int n = pattern.length;
        boolean[][] b = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                b[i][j] = pattern[rowOrder[i]][colOrder[j]];
            }
            b[i][i] = true;
        }
        long contributions = 0;
        for (int k = 0; k < n; k++) {
            int lower = 0;
            int upper = 0;
            for (int i = k + 1; i < n; i++) {
                lower += b[i][k] ? 1 : 0;
                upper += b[k][i] ? 1 : 0;
            }
            contributions += (long) lower * upper;
            for (int i = k + 1; i < n; i++) {
                if (b[i][k]) {
                    for (int j = k + 1; j < n; j++) {
                        b[i][j] |= b[k][j];
                    }
                }
            }
        }
        long entries = 0;
        for (boolean[] row : b) {
            for (boolean e : row) {
                entries += e ? 1 : 0;
            }
        }
        return new long[] {entries, contributions};
    }

    /**
     * The Cholesky fill of the symmetrized pattern eliminated in the given order, by boolean Gaussian elimination: the
     * entries of the lower triangle, diagonal included.
     */
    private static long symmetrizedCholeskyFill(boolean[][] pattern, int[] order) {
        int n = pattern.length;
        boolean[][] b = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                b[i][j] = pattern[order[i]][order[j]] || pattern[order[j]][order[i]];
            }
            b[i][i] = true;
        }
        for (int k = 0; k < n; k++) {
            for (int i = k + 1; i < n; i++) {
                if (b[i][k]) {
                    for (int j = k + 1; j < n; j++) {
                        b[i][j] |= b[k][j];
                    }
                }
            }
        }
        long entries = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                entries += b[i][j] ? 1 : 0;
            }
        }
        return entries;
    }

    /** Dense solve of {@code m x = b} by Gaussian elimination with partial pivoting: the numeric reference. */
    private static double[] denseSolve(double[][] m, double[] b) {
        int n = b.length;
        double[][] a = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(m[i], 0, a[i], 0, n);
            a[i][n] = b[i];
        }
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int r = col + 1; r < n; r++) {
                if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) {
                    pivot = r;
                }
            }
            double[] t = a[col];
            a[col] = a[pivot];
            a[pivot] = t;
            for (int r = col + 1; r < n; r++) {
                double f = a[r][col] / a[col][col];
                for (int c = col; c <= n; c++) {
                    a[r][c] -= f * a[col][c];
                }
            }
        }
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double s = a[i][n];
            for (int j = i + 1; j < n; j++) {
                s -= a[i][j] * x[j];
            }
            x[i] = s / a[i][i];
        }
        return x;
    }

    /** Values on the pattern, a strong diagonal and signed couplings, different for every scenario. */
    private static double[] diagonallyDominantValues(Csr csr, int capacity, long seed) {
        Random random = new Random(seed);
        double[] values = new double[capacity * csr.nnz()];
        for (int s = 0; s < capacity; s++) {
            for (int i = 0; i < csr.n(); i++) {
                for (int k = csr.rowPtr()[i]; k < csr.rowPtr()[i + 1]; k++) {
                    double coupling = random.nextDouble() * 2 - 1;
                    values[s * csr.nnz() + k] = csr.colIdx()[k] == i ? 20 + 5 * random.nextDouble() : coupling;
                }
            }
        }
        return values;
    }

    private record CouplerSystem(Csr csr, double[] values, double[] rightHandSide) {
    }

    /**
     * A system shaped like a Jacobian with zero-impedance couplers as the zero-diagonal repair hands it to our LU: random
     * couplings on a strong diagonal, then coupler rows - +a and -a on two later columns, no diagonal - each repaired by
     * adding a donor row that carries the coupler's column, that column's only other entry. Eliminating the donor first
     * undoes the repair exactly, and the coupler's pivot is zero.
     */
    private static CouplerSystem couplerSystem(long seed, int n, int couplers, double density) {
        Random random = new Random(seed);
        List<TreeMap<Integer, Double>> rows = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            TreeMap<Integer, Double> row = new TreeMap<>();
            for (int j = 0; j < n; j++) {
                if (j != i && random.nextDouble() < density) {
                    row.put(j, random.nextDouble() * 2 - 1);
                }
            }
            double sum = row.values().stream().mapToDouble(Math::abs).sum();
            row.put(i, (sum + 1) * (0.5 + random.nextDouble()));
            rows.add(row);
        }
        for (int c = 0; c < couplers; c++) {
            int r = n / 4 + random.nextInt(n / 2);
            int d = random.nextInt(r);
            int c1 = r + 1 + random.nextInt(n - r - 1);
            int c2 = r + 1 + random.nextInt(n - r - 1);
            if (c1 == c2 || d == r) {
                continue;
            }
            for (int i = 0; i < n; i++) {
                if (i != d && i != r) {
                    rows.get(i).remove(r);
                }
            }
            rows.get(d).put(r, 0.5 + random.nextDouble());
            double a = 1 + random.nextDouble();
            TreeMap<Integer, Double> repaired = new TreeMap<>();
            repaired.put(c1, a);
            repaired.put(c2, -a);
            for (var e : rows.get(d).entrySet()) {
                repaired.merge(e.getKey(), e.getValue(), Double::sum);
            }
            rows.set(r, repaired);
        }
        int[] rowPtr = new int[n + 1];
        for (int i = 0; i < n; i++) {
            rows.get(i).putIfAbsent(i, 0.0);
            rowPtr[i + 1] = rowPtr[i] + rows.get(i).size();
        }
        int[] colIdx = new int[rowPtr[n]];
        double[] values = new double[rowPtr[n]];
        int p = 0;
        for (int i = 0; i < n; i++) {
            for (var e : rows.get(i).entrySet()) {
                colIdx[p] = e.getKey();
                values[p++] = e.getValue();
            }
        }
        double[] rightHandSide = new double[n];
        for (int i = 0; i < n; i++) {
            rightHandSide[i] = random.nextDouble() * 2 - 1;
        }
        return new CouplerSystem(new Csr(n, rowPtr, colIdx), values, rightHandSide);
    }

    private static double[][] dense(Csr csr, double[] values, int scenario) {
        double[][] m = new double[csr.n()][csr.n()];
        for (int i = 0; i < csr.n(); i++) {
            for (int k = csr.rowPtr()[i]; k < csr.rowPtr()[i + 1]; k++) {
                m[i][csr.colIdx()[k]] = values[scenario * csr.nnz() + k];
            }
        }
        return m;
    }

    @Test
    void fastPlanMatchesBooleanElimination() {
        for (int n : new int[] {7, 40, 120}) {
            boolean[][] pattern = jacobianLikePattern(n, 3.0 / n, n);
            Csr csr = toCsr(pattern);
            int[][][] orders = {
                {identity(n), identity(n)},
                {reversed(n), reversed(n)},
                {shuffled(n, 1), shuffled(n, 1)},
                {shuffled(n, 2), shuffled(n, 3)},                // unsymmetric: most pivots are structural zeros
            };
            for (int[][] order : orders) {
                long[] stats = BatchedLuNative.scalarPlanStatistics(n, csr.rowPtr(), csr.colIdx(), order[0], order[1]);
                long[] reference = booleanElimination(pattern, order[0], order[1]);
                assertEquals(reference[0], stats[0], "LU entries, n=" + n);
                assertEquals(reference[1], stats[1], "contributions, n=" + n);
                assertTrue(stats[2] >= 2 && stats[3] >= 1 && stats[4] >= 1, "steps and levels: " + Arrays.toString(stats));
            }
        }
    }

    /**
     * The pivoting plan's structure is the Cholesky fill of the symmetrized pattern (L and U, twice the lower triangle less the
     * diagonal); its supernodes store it with some padding, at most 16 columns each.
     */
    @Test
    void pivotingPlanHoldsTheSymmetrizedCholeskyFill() {
        for (int n : new int[] {7, 40, 120}) {
            boolean[][] pattern = jacobianLikePattern(n, 3.0 / n, n);
            Csr csr = toCsr(pattern);
            for (int[] order : new int[][] {identity(n), reversed(n), shuffled(n, 1)}) {
                long[] stats = BatchedLuNative.supernodalPlanStatistics(n, csr.rowPtr(), csr.colIdx(), order, order);
                assertEquals(2 * symmetrizedCholeskyFill(pattern, order) - n, stats[5], "structure, n=" + n);
                assertTrue(stats[0] >= stats[5], "stored entries hold the structure: " + Arrays.toString(stats));
                assertTrue(stats[2] >= 1 && stats[3] >= 1 && stats[3] <= n && stats[4] <= 16, "plan: " + Arrays.toString(stats));
            }
        }
    }

    private static void assertSolvesTheLanesOfTheMap(int n, int capacity, int[] laneMap, long seed) {
        Csr csr = toCsr(jacobianLikePattern(n, 4.0 / n, seed));
        double[] values = diagonallyDominantValues(csr, capacity, seed + 1);
        double[] rightHandSides = new double[capacity * n];
        Random random = new Random(seed + 2);
        for (int k = 0; k < rightHandSides.length; k++) {
            rightHandSides[k] = random.nextDouble() * 2 - 1;
        }
        double[] solutions = new double[capacity * n];
        Arrays.fill(solutions, 7.0);
        int[] pivoted = new int[laneMap.length];
        int[] clamps = new int[laneMap.length];
        int[] order = shuffled(n, seed + 3);
        BatchedLuNative.factorizeAndSolve(n, csr.rowPtr(), csr.colIdx(), order, order, capacity, values,
                rightHandSides, laneMap, solutions, pivoted, clamps);
        assertArrayEquals(new int[laneMap.length], pivoted, "a diagonally dominant matrix needs no pivoting");
        assertArrayEquals(new int[laneMap.length], clamps);
        boolean[] solved = new boolean[capacity];
        for (int s : laneMap) {
            solved[s] = true;
        }
        double worst = 0;
        for (int s = 0; s < capacity; s++) {
            double[] x = Arrays.copyOfRange(solutions, s * n, (s + 1) * n);
            if (!solved[s]) {
                double[] untouched = new double[n];
                Arrays.fill(untouched, 7.0);
                assertArrayEquals(untouched, x, "scenario " + s + " is outside the lane map");
                continue;
            }
            double[] reference = denseSolve(dense(csr, values, s), Arrays.copyOfRange(rightHandSides, s * n, (s + 1) * n));
            for (int i = 0; i < n; i++) {
                worst = Math.max(worst, Math.abs(x[i] - reference[i]));
            }
        }
        assertEquals(0, worst, 1e-12, "batched LU vs dense pivoted solve, n=" + n + " capacity=" + capacity);
    }

    @Test
    void solvesEveryScenarioOfTheLaneMapAndNoOther() {
        assertSolvesTheLanesOfTheMap(40, 6, new int[] {4, 1, 3}, 11);
        assertSolvesTheLanesOfTheMap(300, 70, java.util.stream.IntStream.range(0, 70).filter(s -> s % 3 != 1).toArray(), 12);
        assertSolvesTheLanesOfTheMap(257, 33, identity(33), 13);
    }

    /**
     * Each lane pivots on its own: [[0, 1], [1, 0]] has a zero pivot the fast engine cannot take, so that lane alone
     * goes to the pivoting engine and swaps its rows; [[4, 1], [1, 3]] stays on the fast engine. Both solve exactly in
     * one batch, nothing clamped. Without the pivoting engine the first lane's zero pivot is clamped and its solution
     * is wrong.
     */
    @Test
    void eachLanePivotsInsideASupernode() {
        Csr csr = toCsr(new boolean[][] {{true, true}, {true, true}});
        double[] values = {0, 1, 1, 0, 4, 1, 1, 3};
        double[] rightHandSides = {1, 2, 1, 2};
        double[] solutions = new double[4];
        int[] pivoted = new int[2];
        int[] clamps = new int[2];
        BatchedLuNative.factorizeAndSolve(2, csr.rowPtr(), csr.colIdx(), identity(2), identity(2), 2, values,
                rightHandSides, new int[] {0, 1}, solutions, pivoted, clamps);
        assertArrayEquals(new int[] {1, 0}, pivoted);
        assertArrayEquals(new int[] {0, 0}, clamps);
        assertArrayEquals(new double[] {2, 1, 1.0 / 11, 7.0 / 11}, solutions, 1e-15);
    }

    /** A zero diagonal on every lane of a dense block: the pivoting engine's complete pivoting inside the supernode
     *  solves it exactly. */
    @Test
    void zeroDiagonalsInsideASupernodeSolveExactly() {
        int n = 12;
        boolean[][] pattern = new boolean[n][n];
        for (boolean[] row : pattern) {
            Arrays.fill(row, true);
        }
        Csr csr = toCsr(pattern);
        int capacity = 3;
        double[] values = diagonallyDominantValues(csr, capacity, 21);
        for (int s = 0; s < capacity; s++) {
            for (int i = 0; i < n; i++) {
                values[s * csr.nnz() + i * n + i] = 0.0;                     // every diagonal zero
            }
        }
        double[] rightHandSides = new double[capacity * n];
        Random random = new Random(22);
        for (int k = 0; k < rightHandSides.length; k++) {
            rightHandSides[k] = random.nextDouble() * 2 - 1;
        }
        double[] solutions = new double[capacity * n];
        int[] pivoted = new int[capacity];
        int[] clamps = new int[capacity];
        BatchedLuNative.factorizeAndSolve(n, csr.rowPtr(), csr.colIdx(), identity(n), identity(n), capacity, values,
                rightHandSides, identity(capacity), solutions, pivoted, clamps);
        int[] all = new int[capacity];
        Arrays.fill(all, 1);
        assertArrayEquals(all, pivoted);
        assertArrayEquals(new int[capacity], clamps);
        for (int s = 0; s < capacity; s++) {
            double[] reference = denseSolve(dense(csr, values, s), Arrays.copyOfRange(rightHandSides, s * n, (s + 1) * n));
            assertArrayEquals(reference, Arrays.copyOfRange(solutions, s * n, (s + 1) * n), 1e-12, "scenario " + s);
        }
    }

    /**
     * A pivot that is not zero but tiny for its column: the fast engine clamps nothing, its multiplier 1e10 is the
     * trouble, and the lane goes to the pivoting engine, which solves it to machine precision. Unpivoted, the
     * elimination grows by 1e10 and the solution loses that many digits.
     */
    @Test
    void outsizedMultiplierSendsTheLaneToThePivotingEngine() {
        Csr csr = toCsr(new boolean[][] {{true, true}, {true, true}});
        double[] values = {1e-10, 1, 1, 1, 4, 1, 1, 3};
        double[] rightHandSides = {1, 2, 1, 2};
        double[] solutions = new double[4];
        int[] pivoted = new int[2];
        int[] clamps = new int[2];
        BatchedLuNative.factorizeAndSolve(2, csr.rowPtr(), csr.colIdx(), identity(2), identity(2), 2, values,
                rightHandSides, new int[] {0, 1}, solutions, pivoted, clamps);
        assertArrayEquals(new int[] {1, 0}, pivoted);
        assertArrayEquals(new int[] {0, 0}, clamps);
        double[] reference = denseSolve(new double[][] {{1e-10, 1}, {1, 1}}, new double[] {1, 2});
        assertArrayEquals(reference, Arrays.copyOfRange(solutions, 0, 2), 1e-15);
        assertArrayEquals(new double[] {1.0 / 11, 7.0 / 11}, Arrays.copyOfRange(solutions, 2, 4), 1e-15);
    }

    /** A singular lane: even pivoted, its last pivot is clamped and counted there; the other lane stays on the fast
     *  engine, its solution untouched. */
    @Test
    void singularLaneIsClampedOnItsLaneOnly() {
        Csr csr = toCsr(new boolean[][] {{true, true}, {true, true}});
        double[] values = {1, 1, 1, 1, 4, 1, 1, 3};
        double[] rightHandSides = {1, 2, 1, 2};
        double[] solutions = new double[4];
        int[] pivoted = new int[2];
        int[] clamps = new int[2];
        BatchedLuNative.factorizeAndSolve(2, csr.rowPtr(), csr.colIdx(), identity(2), identity(2), 2, values,
                rightHandSides, new int[] {0, 1}, solutions, pivoted, clamps);
        assertArrayEquals(new int[] {1, 0}, pivoted);
        assertArrayEquals(new int[] {1, 0}, clamps);
        assertEquals(1.0 / 11, solutions[2], 1e-15);
        assertEquals(7.0 / 11, solutions[3], 1e-15);
    }

    /** cuDSS returns its elimination order new -> old: as returned, it gives cuDSS's own LU fill. */
    /**
     * Our LU solves a Newton step as cuDSS's SOLVE phase does under the batch configuration: its internal refinement
     * step inside every solve, the adaptive refinement's corrections included. Here the repair's undoing clamps two
     * pivots; refined with bare corrections, a correction that did not lower the residual ended the refinement at a
     * backward ratio of 9e-9 (on a network.xiidm busbar contingency: 1.1, where cuDSS refines the same system to 3e-11).
     */
    @Test
    void newtonStepCorrectionsCarryTheInternalRefinementStep() {
        new GpuLoadFlowParameters().install();                   // one internal step, at most four solves, 1e-12
        CouplerSystem system = couplerSystem(32, 80, 3, 0.03);
        Csr csr = system.csr();
        int n = csr.n();
        int[] pivoted = new int[1];
        int[] clamps = new int[1];
        BatchedLuNative.factorizeAndSolve(n, csr.rowPtr(), csr.colIdx(), identity(n), identity(n), 1, system.values(),
                system.rightHandSide(), new int[] {0}, new double[n], pivoted, clamps);
        assertArrayEquals(new int[] {1}, pivoted);
        assertArrayEquals(new int[] {2}, clamps, "the repair's undoing leaves two pivots the pivoting engine cannot find");
        double[] x = new double[n];
        BatchedLuNative.factorizeAndSolveNewtonStep(n, csr.rowPtr(), csr.colIdx(), identity(n), identity(n), 1,
                system.values(), system.rightHandSide(), new int[] {0}, x);
        double[][] m = dense(csr, system.values(), 0);
        double residual = 0;
        double scale = 0;
        for (int i = 0; i < n; i++) {
            double acc = 0;
            for (int j = 0; j < n; j++) {
                acc += m[i][j] * x[j];
            }
            residual = Math.max(residual, Math.abs(system.rightHandSide()[i] - acc));
            scale = Math.max(scale, Math.abs(system.rightHandSide()[i]));
        }
        assertTrue(residual / scale < 1e-12, "backward ratio " + residual / scale);
        assertArrayEquals(denseSolve(m, system.rightHandSide()), x, 1e-10);
    }

    /**
     * Diagonally dominant values whose last pivot is {@code pivot} (the last diagonal entry less its Schur complement):
     * the fast engine keeps the lane - no multiplier divides by that pivot - and its solution is 1 / pivot times its
     * right-hand side along one direction, so a solve leaves its backward ratio far above the refinement's tolerance.
     */
    private static double[] nearlySingularValues(Csr csr, long seed, double pivot) {
        int n = csr.n();
        double[] values = diagonallyDominantValues(csr, 1, seed);
        double[][] m = dense(csr, values, 0);
        double[][] leading = new double[n - 1][n - 1];
        double[] lastColumn = new double[n - 1];
        for (int i = 0; i < n - 1; i++) {
            System.arraycopy(m[i], 0, leading[i], 0, n - 1);
            lastColumn[i] = m[i][n - 1];
        }
        double[] y = denseSolve(leading, lastColumn);
        double schurComplement = m[n - 1][n - 1];
        for (int j = 0; j < n - 1; j++) {
            schurComplement -= m[n - 1][j] * y[j];
        }
        for (int k = csr.rowPtr()[n - 1]; k < csr.rowPtr()[n]; k++) {
            if (csr.colIdx()[k] == n - 1) {
                values[k] += pivot - schurComplement;
            }
        }
        return values;
    }

    /**
     * Six scenarios on the coupler system's pattern: the coupler system itself on 1 and doubled on 3 (the pivoting
     * engine's lanes: the repair's undoing clamps two pivots) and a nearly singular lane on 0 (the fast engine's), all
     * three refining past the plain solve; diagonally dominant lanes, solved to the tolerance by it, on 2, 4 and 5.
     */
    private static double[] refinementBatch(CouplerSystem system) {
        Csr csr = system.csr();
        int nnz = csr.nnz();
        double[][] lanes = {nearlySingularValues(csr, 5, 1e-9), system.values(), diagonallyDominantValues(csr, 1, 6),
            Arrays.stream(system.values()).map(v -> 2 * v).toArray(), diagonallyDominantValues(csr, 1, 7),
            diagonallyDominantValues(csr, 1, 8)};
        double[] values = new double[lanes.length * nnz];
        for (int s = 0; s < lanes.length; s++) {
            System.arraycopy(lanes[s], 0, values, s * nnz, nnz);
        }
        return values;
    }

    /**
     * The refinement batch's lane map: no scenario at its own position, the nearly singular lane last (the fast engine's
     * lane 4) and the coupler system the pivoting engine's lane 1, behind its double; scenario 5 is not factorized.
     */
    private static final int[] REFINEMENT_LANE_MAP = {3, 2, 1, 4, 0};

    private static double[] refinementRightHandSides(int length) {
        double[] rightHandSides = new double[length];
        Random random = new Random(23);
        for (int k = 0; k < rightHandSides.length; k++) {
            rightHandSides[k] = random.nextDouble() * 2 - 1;
        }
        return rightHandSides;
    }

    private static double[] scenario(double[] batch, int s, int count) {
        return Arrays.copyOfRange(batch, s * count, (s + 1) * count);
    }

    private static double[] filled(int length, double value) {
        double[] a = new double[length];
        Arrays.fill(a, value);
        return a;
    }

    /**
     * Solving selected lanes - the adaptive refinement's corrections, on the lanes still refining - gives each of them
     * the solution the solve of every lane gives it, bit for bit, on either engine, and leaves every other scenario's
     * as it was: scenario 0 is the fast engine's lane 4 here, scenario 1 its lane 2 and the pivoting engine's lane 1.
     * Checked to fail with the selection taken as scenarios, with either engine solving its first lanes instead of the
     * selected ones, and without the pivoting engine's solve.
     */
    @Test
    void selectedLanesAreSolvedAsInTheWholeBatch() {
        CouplerSystem system = couplerSystem(32, 80, 3, 0.03);
        Csr csr = system.csr();
        int n = csr.n();
        double[] values = refinementBatch(system);
        double[] rightHandSides = refinementRightHandSides(6 * n);
        double[] whole = filled(6 * n, 42);
        int[] pivoted = new int[REFINEMENT_LANE_MAP.length];
        int[] clamps = new int[REFINEMENT_LANE_MAP.length];
        BatchedLuNative.factorizeAndSolve(n, csr.rowPtr(), csr.colIdx(), identity(n), identity(n), 6, values,
                rightHandSides, REFINEMENT_LANE_MAP, whole, pivoted, clamps);
        assertArrayEquals(new int[] {1, 0, 1, 0, 0}, pivoted);
        double[] selected = filled(6 * n, 42);
        BatchedLuNative.factorizeAndSolveSelected(n, csr.rowPtr(), csr.colIdx(), identity(n), identity(n), 6, values,
                rightHandSides, REFINEMENT_LANE_MAP, new int[] {1, 0}, selected);
        for (int s = 0; s < 6; s++) {
            double[] expected = s <= 1 ? scenario(whole, s, n) : filled(n, 42);
            assertArrayEquals(expected, scenario(selected, s, n), "scenario " + s);
        }
    }

    /** A hub joined to every other row, plus a chain through those rows; the hub is row n - 1 or row 0. */
    private static Csr hubAndChain(int n, boolean hubLast) {
        boolean[][] pattern = new boolean[n][n];
        int hub = hubLast ? n - 1 : 0;
        for (int i = 0; i < n; i++) {
            pattern[i][i] = true;
            pattern[i][hub] = true;
            pattern[hub][i] = true;
            if (i + 1 < n) {
                pattern[i][i + 1] = true;
                pattern[i + 1][i] = true;
            }
        }
        return toCsr(pattern);
    }

    private static void assertSolvesTheDenseWay(Csr csr, int capacity, long seed) {
        int n = csr.n();
        double[] values = diagonallyDominantValues(csr, capacity, seed);
        double[] rightHandSides = new double[capacity * n];
        Random random = new Random(seed + 1);
        for (int k = 0; k < rightHandSides.length; k++) {
            rightHandSides[k] = random.nextDouble() * 2 - 1;
        }
        double[] whole = filled(capacity * n, 7);
        int[] pivoted = new int[capacity];
        int[] clamps = new int[capacity];
        BatchedLuNative.factorizeAndSolve(n, csr.rowPtr(), csr.colIdx(), identity(n), identity(n), capacity, values,
                rightHandSides, identity(capacity), whole, pivoted, clamps);
        assertArrayEquals(new int[capacity], pivoted, "the fast engine factorizes every lane");
        assertArrayEquals(new int[capacity], clamps);
        int[] chosen = {capacity - 1, 0, 33, 5, 31};
        double[] selected = filled(capacity * n, 7);
        BatchedLuNative.factorizeAndSolveSelected(n, csr.rowPtr(), csr.colIdx(), identity(n), identity(n), capacity,
                values, rightHandSides, identity(capacity), chosen, selected);
        for (int s = 0; s < capacity; s++) {
            double[] reference = denseSolve(dense(csr, values, s), scenario(rightHandSides, s, n));
            assertArrayEquals(reference, scenario(whole, s, n), 1e-12, "n=" + n + " scenario " + s);
            final int scenario = s;
            boolean isChosen = Arrays.stream(chosen).anyMatch(c -> c == scenario);
            assertArrayEquals(isChosen ? scenario(whole, s, n) : filled(n, 7), scenario(selected, s, n),
                    "n=" + n + " selected scenario " + s);
        }
    }

    /**
     * A row with a long run of terms - the L row of a hub eliminated last, 299 terms and its pivot's 299 contributions;
     * every row of the full matrix a hub eliminated first leaves, on both sides of the diagonal - is summed by several
     * warps (batchedLuSplitLongRows) when the level leaves the GPU threads to spare, as these 37 lanes do: the warps'
     * partial sums are added in a fixed order, so the factors and solutions are the dense pivoted solve's to rounding,
     * on all 37 lanes (a full and a partial group of 32) and on 5 selected ones, which take the solve of every lane bit
     * for bit. With the parameter off no step or level is split. Checked to fail with the last warp's partial sum left
     * out of the factorization, or out of the forward or the backward solve.
     */
    @Test
    void longRowsSplitOverSeveralWarpsSolveAsTheDenseSolve() {
        try {
            for (boolean split : new boolean[] {true, false}) {
                new GpuLoadFlowParameters().setBatchedLuSplitLongRows(split).install();
                long before = BatchedLuNative.factorizationCounts()[3];
                assertSolvesTheDenseWay(hubAndChain(300, true), 37, 51);
                assertSolvesTheDenseWay(hubAndChain(64, false), 37, 52);
                long launches = BatchedLuNative.factorizationCounts()[3] - before;
                assertTrue(split ? launches > 0 : launches == 0, "split=" + split + ": " + launches + " split launches");
            }
        } finally {
            new GpuLoadFlowParameters().install();                   // the next tests run with the defaults
        }
    }

    /**
     * The minimum-degree elimination order (GpuBatchedLuOrdering.MINIMUM_DEGREE) on a hub joined to four leaves: three
     * leaves, then the hub and the last leaf tied at one neighbour each (the lower index first), an elimination tree
     * three levels deep; eliminated first, the hub makes its leaves a clique and a chain of five levels. Our fast
     * engine's plan follows: fewer entries and forward levels.
     */
    @Test
    void minimumDegreeEliminatesTheLeavesBeforeTheirHub() {
        int[] rowPtr = {0, 5, 7, 9, 11, 13};
        int[] colIdx = {0, 1, 2, 3, 4, 0, 1, 0, 2, 0, 3, 0, 4};
        int[] order = BatchedLuNative.minimumDegreeOrder(5, rowPtr, colIdx);
        int[] natural = {0, 1, 2, 3, 4};
        assertArrayEquals(new int[] {1, 2, 3, 0, 4}, order);
        assertEquals(3, BatchedLuNative.eliminationLevels(5, rowPtr, colIdx, order));
        assertEquals(5, BatchedLuNative.eliminationLevels(5, rowPtr, colIdx, natural));
        long[] minimumDegree = BatchedLuNative.scalarPlanStatistics(5, rowPtr, colIdx, order, order);
        long[] hubFirst = BatchedLuNative.scalarPlanStatistics(5, rowPtr, colIdx, natural, natural);
        assertTrue(minimumDegree[0] < hubFirst[0], "LU entries");
        assertTrue(minimumDegree[3] < hubFirst[3], "forward levels");
    }

    /** The minimum-degree order of a coupler system's pattern is a permutation of its rows. */
    @Test
    void minimumDegreeOrdersEveryRowOnce() {
        Csr csr = couplerSystem(32, 80, 3, 0.03).csr();
        int[] order = BatchedLuNative.minimumDegreeOrder(csr.n(), csr.rowPtr(), csr.colIdx());
        int[] sorted = order.clone();
        Arrays.sort(sorted);
        int[] identity = new int[csr.n()];
        for (int i = 0; i < identity.length; i++) {
            identity[i] = i;
        }
        assertArrayEquals(identity, sorted);
    }

    /**
     * The adaptive refinement solves its corrections on the lanes still refining only, and each lane's Newton step in a
     * batch is the one it takes alone, bit for bit: the coupler systems refine on the pivoting engine and the nearly
     * singular lane on the fast engine, beside lanes done after the plain solve and a scenario outside the lane map.
     * Checked to fail with the selection taken as scenarios and with the fast engine solving its first lanes instead
     * of the selected ones.
     */
    @Test
    void newtonStepOfEachLaneIsTheOneItTakesAlone() {
        CouplerSystem system = couplerSystem(32, 80, 3, 0.03);
        Csr csr = system.csr();
        int n = csr.n();
        int nnz = csr.nnz();
        double[] values = refinementBatch(system);
        double[] rightHandSides = refinementRightHandSides(6 * n);
        double[] plainSolve = new double[6 * n];
        try {
            new GpuLoadFlowParameters().setIterativeRefinementBaseSteps(0).setIterativeRefinementMaxSteps(0).install();   // no refinement
            BatchedLuNative.factorizeAndSolveNewtonStep(n, csr.rowPtr(), csr.colIdx(), identity(n), identity(n), 6,
                    values, rightHandSides, REFINEMENT_LANE_MAP, plainSolve);
        } finally {
            new GpuLoadFlowParameters().install();               // one internal step, at most four solves, 1e-12
        }
        double[] batch = filled(6 * n, 42);
        BatchedLuNative.factorizeAndSolveNewtonStep(n, csr.rowPtr(), csr.colIdx(), identity(n), identity(n), 6, values,
                rightHandSides, REFINEMENT_LANE_MAP, batch);
        for (int s : REFINEMENT_LANE_MAP) {
            double[] alone = new double[n];
            BatchedLuNative.factorizeAndSolveNewtonStep(n, csr.rowPtr(), csr.colIdx(), identity(n), identity(n), 1,
                    scenario(values, s, nnz), scenario(rightHandSides, s, n), new int[] {0}, alone);
            assertArrayEquals(alone, scenario(batch, s, n), "scenario " + s);
        }
        assertArrayEquals(filled(n, 42), scenario(batch, 5, n));
        for (int s : new int[] {0, 1, 3}) {
            assertFalse(Arrays.equals(scenario(plainSolve, s, n), scenario(batch, s, n)),
                    "scenario " + s + " refines past the plain solve");
        }
        for (int s : new int[] {2, 4}) {                     // solved to the tolerance by the plain solve: no refinement
            assertArrayEquals(scenario(plainSolve, s, n), scenario(batch, s, n), "scenario " + s + " takes no refinement step");
        }
    }

    /**
     * Two buses joined by a line, and to the rest of the network by a coupler only, as breaker mode builds them, in the
     * variables {phi1, phi2, phi3, p}: the line's balances g1 (phi1 - phi2) and g2 (phi2 - phi1) + p (p the coupler's
     * flow) depend on the angle difference only; the coupler's ZERO_PHI phi2 - phi3 sits on the third bus's angle row
     * and that bus's balance s phi3 - p on the coupler's dummy row, as the coupler-tree relocation places them. Per lane
     * {g1, g2, s}.
     */
    private static double[] lineGroupValues(double[][] lanes) {
        double[] values = new double[lanes.length * 9];
        for (int s = 0; s < lanes.length; s++) {
            double g1 = lanes[s][0];
            double g2 = lanes[s][1];
            System.arraycopy(new double[] {g1, -g1, -g2, g2, 1, 1, -1, lanes[s][2], -1}, 0, values, s * 9, 9);
        }
        return values;
    }

    /**
     * Factorizes the line group's lanes four times on one plan: asserts the rounds the pivoting engine took them in, the
     * delays each round ran on, and the last factorization's solutions against a dense pivoted solve.
     */
    private static void assertLineGroupLearns(double[][] lanes, int[] pivotedRounds, int[] delays) {
        int capacity = lanes.length;
        double[] values = lineGroupValues(lanes);
        Random random = new Random(7);
        double[] rightHandSides = new double[capacity * 4];
        for (int i = 0; i < rightHandSides.length; i++) {
            rightHandSides[i] = random.nextDouble() * 2 - 1;
        }
        int[] laneMap = identity(capacity);
        double[] solutions = new double[capacity * 4];
        int[] pivoted = new int[4 * capacity];
        int[] roundDelays = new int[4];
        BatchedLuNative.factorizeRounds(4, LINE_GROUP.rowPtr(), LINE_GROUP.colIdx(), identity(4), identity(4), capacity,
                values, rightHandSides, laneMap, 4, solutions, pivoted, roundDelays);
        int[] expected = new int[4 * capacity];
        for (int round = 0; round < 4; round++) {
            Arrays.fill(expected, round * capacity, (round + 1) * capacity, pivotedRounds[round]);
        }
        assertArrayEquals(expected, pivoted, "lanes the pivoting engine took, round by round");
        assertArrayEquals(delays, roundDelays, "pivot delays of each round's order");
        for (int s = 0; s < capacity; s++) {
            double[] expectedSolution = denseSolve(dense(LINE_GROUP, values, s), Arrays.copyOfRange(rightHandSides, s * 4, s * 4 + 4));
            double scale = Arrays.stream(expectedSolution).map(Math::abs).max().orElseThrow();
            for (int i = 0; i < 4; i++) {
                assertEquals(expectedSolution[i], solutions[s * 4 + i], 1e-12 * scale, "lane " + s + " unknown " + i);
            }
        }
    }

    /**
     * In the order given the line's balances come first, and the second one cancels on every lane: its row and the
     * first one's are opposite on the two angles. After the first factorization the fast engine delays that pivot after
     * the row with its column's largest entry below, the ZERO_PHI row, where it still cancels (the ZERO_PHI row ties the
     * third angle to the group, not the group to anything); after the second it moves it after the third bus's balance,
     * which brings that angle in. From the third factorization on, the fast engine keeps every lane and solves it
     * exactly. Checked to fail with the learning off (every round pivots), with a delay that does not move (the order
     * stays where the pivot still cancels), and with the batched LU keeping the plan it was created on.
     */
    @Test
    void cancelledPivotIsDelayedUntilARowGivesItAValue() {
        Random random = new Random(11);
        double[][] lanes = new double[3][];
        for (int s = 0; s < lanes.length; s++) {
            double g = 1 + 9 * random.nextDouble();
            lanes[s] = new double[] {g, g, 1 + 9 * random.nextDouble()};
        }
        assertLineGroupLearns(lanes, new int[] {1, 1, 0, 0}, new int[] {0, 1, 1, 1});
    }

    /**
     * A lossy line's balances are opposite on the two angles too, with different coefficients: the second pivot cancels
     * to rounding, here 1.7e-12 - above the clamp, so only its multipliers flag the lane - and the fast engine delays it
     * all the same: it is below 1e-8 of its row's and column's largest entry. Checked to fail with clamped pivots alone
     * learned (the lane keeps going to the pivoting engine).
     */
    @Test
    void pivotCancelledAboveTheClampIsDelayedToo() {
        double g1 = 31415.9265;
        double g2 = 27182.8183;
        double residue = Math.fma(g2 / g1, -g1, g2);       // the fast engine's pivot: the same IEEE operations
        assertTrue(Math.abs(residue) > 1e-13 && Math.abs(residue) < 1e-8 * g1, "the pivot left by rounding: " + residue);
        assertLineGroupLearns(new double[][] {{g1, g2, 3}}, new int[] {1, 1, 0, 0}, new int[] {0, 1, 1, 1});
    }

    /**
     * Pairs of rows {0 x_i + y_i, x_i + y_i + z}, the x of every pair tied to a hub row h (x_1 + ... + x_m) + z with
     * h > 1: in the order given each pair's first pivot is zero, and the largest entry of its column below is the hub's,
     * so the fast engine would delay every one of them after the hub - whose elimination then fills the pairs' first rows
     * with every x, a dense block, where the order given fills one entry per pair. That order costs more than a lane it
     * would spare costs today (the fast engine on the order given and the pivoting engine together), so it is not taken:
     * every round runs on the order given, the pivoting engine takes every lane, pivots inside each pair and solves it
     * exactly. Checked to fail with every learned order taken (the second round on runs on 16 delays).
     */
    @Test
    void orderCostlierThanTheLanesItSparesIsNotTaken() {
        int pairs = 16;
        int n = 2 * pairs + 1;
        int hub = 2 * pairs;
        int[] rowPtr = new int[n + 1];
        List<Integer> columns = new ArrayList<>();
        for (int i = 0; i < pairs; i++) {
            columns.addAll(List.of(2 * i, 2 * i + 1));
            rowPtr[2 * i + 1] = columns.size();
            columns.addAll(List.of(2 * i, 2 * i + 1, hub));
            rowPtr[2 * i + 2] = columns.size();
        }
        for (int i = 0; i < pairs; i++) {
            columns.add(2 * i);
        }
        columns.add(hub);
        rowPtr[n] = columns.size();
        Csr csr = new Csr(n, rowPtr, columns.stream().mapToInt(Integer::intValue).toArray());
        int capacity = 2;
        double[] values = new double[capacity * csr.nnz()];
        for (int s = 0; s < capacity; s++) {
            int k = s * csr.nnz();
            for (int i = 0; i < pairs; i++) {
                values[k++] = 0;                             // x_i's pivot
                values[k++] = 1;
                values[k++] = 1;
                values[k++] = 1;
                values[k++] = 1;
            }
            for (int i = 0; i < pairs; i++) {
                values[k++] = 10 + s;                        // the largest entry of every x column
            }
            values[k] = 1;
        }

        // the order the delays ask for - every pair's second row first, then the hub, then the first rows - against
        // what a lane costs today
        int[] delayed = new int[n];
        for (int i = 0; i < pairs; i++) {
            delayed[i] = 2 * i + 1;
            delayed[pairs + 1 + i] = 2 * i;
        }
        delayed[pairs] = hub;
        long[] given = BatchedLuNative.scalarPlanStatistics(n, csr.rowPtr(), csr.colIdx(), identity(n), identity(n));
        long[] pivoting = BatchedLuNative.supernodalPlanStatistics(n, csr.rowPtr(), csr.colIdx(), identity(n), identity(n));
        long[] asked = BatchedLuNative.scalarPlanStatistics(n, csr.rowPtr(), csr.colIdx(), delayed, delayed);
        assertTrue(asked[0] > given[0] + pivoting[0] && asked[1] > given[1] + pivoting[1],
                "LU entries and contributions: delayed " + asked[0] + ", " + asked[1] + "; given " + given[0] + ", "
                        + given[1] + "; pivoting engine " + pivoting[0] + ", " + pivoting[1]);

        Random random = new Random(5);
        double[] rightHandSides = new double[capacity * n];
        for (int i = 0; i < rightHandSides.length; i++) {
            rightHandSides[i] = random.nextDouble() * 2 - 1;
        }
        double[] solutions = new double[capacity * n];
        int[] pivoted = new int[4 * capacity];
        int[] roundDelays = new int[4];
        BatchedLuNative.factorizeRounds(n, csr.rowPtr(), csr.colIdx(), identity(n), identity(n), capacity, values,
                rightHandSides, identity(capacity), 4, solutions, pivoted, roundDelays);
        assertArrayEquals(new int[4], roundDelays, "pivot delays of each round's order");
        int[] everyLane = new int[4 * capacity];
        Arrays.fill(everyLane, 1);
        assertArrayEquals(everyLane, pivoted, "lanes the pivoting engine took, round by round");
        for (int s = 0; s < capacity; s++) {
            double[] expected = denseSolve(dense(csr, values, s), Arrays.copyOfRange(rightHandSides, s * n, (s + 1) * n));
            double scale = Arrays.stream(expected).map(Math::abs).max().orElseThrow();
            for (int i = 0; i < n; i++) {
                assertEquals(expected[i], solutions[s * n + i], 1e-12 * scale, "lane " + s + " unknown " + i);
            }
        }
    }

    @Test
    void cudssOrderConvention() {
        int side = 12;
        int n = 2 * side * side;                            // two variables per node of a grid, coupled to the four neighbours
        boolean[][] pattern = new boolean[n][n];
        for (int r = 0; r < side; r++) {
            for (int c = 0; c < side; c++) {
                int node = r * side + c;
                int[] neighbours = {node, r > 0 ? node - side : -1, r < side - 1 ? node + side : -1,
                    c > 0 ? node - 1 : -1, c < side - 1 ? node + 1 : -1};
                for (int other : neighbours) {
                    if (other >= 0) {
                        for (int a = 0; a < 2; a++) {
                            for (int b = 0; b < 2; b++) {
                                pattern[2 * node + a][2 * other + b] = true;
                            }
                        }
                    }
                }
            }
        }
        Csr csr = toCsr(pattern);
        int[] rowOrder = new int[n];
        int[] colOrder = new int[n];
        long cudssLuNnz = BatchedLuNative.cudssReorder(n, csr.rowPtr(), csr.colIdx(), rowOrder, colOrder);
        int[] rowInverse = new int[n];
        int[] colInverse = new int[n];
        for (int i = 0; i < n; i++) {
            rowInverse[rowOrder[i]] = i;
            colInverse[colOrder[i]] = i;
        }
        long[] asReturned = BatchedLuNative.supernodalPlanStatistics(n, csr.rowPtr(), csr.colIdx(), rowOrder, colOrder);
        long[] inverted = BatchedLuNative.supernodalPlanStatistics(n, csr.rowPtr(), csr.colIdx(), rowInverse, colInverse);
        assertTrue(Arrays.equals(rowOrder, colOrder), "cuDSS orders a general matrix symmetrically under AMD");
        // the convention: the orders as cuDSS returns them are new -> old, so they give cuDSS's own fill, on this
        // symmetric pattern the exact fill of the fast engine as well
        assertEquals(cudssLuNnz, asReturned[5], "the structure in cuDSS's order is cuDSS's fill");
        assertEquals(cudssLuNnz, BatchedLuNative.scalarPlanStatistics(n, csr.rowPtr(), csr.colIdx(), rowOrder, colOrder)[0]);
        assertTrue(inverted[5] > asReturned[5], "the inverse permutation fills more: " + inverted[5]);
    }

    /** A batch's systems with their rows moved: row i of every scenario becomes row pi[i]. */
    private record MovedRows(Csr csr, double[] values) {
    }

    private static MovedRows moveRows(Csr csr, double[] values, int capacity, int[] pi) {
        int n = csr.n();
        int nnz = csr.nnz();
        int[] from = new int[n];                                     // row of the moved system -> row of csr
        for (int i = 0; i < n; i++) {
            from[pi[i]] = i;
        }
        int[] rowPtr = new int[n + 1];
        for (int r = 0; r < n; r++) {
            rowPtr[r + 1] = rowPtr[r] + csr.rowPtr()[from[r] + 1] - csr.rowPtr()[from[r]];
        }
        int[] colIdx = new int[nnz];
        double[] moved = new double[capacity * nnz];
        for (int r = 0; r < n; r++) {
            int start = csr.rowPtr()[from[r]];
            int length = csr.rowPtr()[from[r] + 1] - start;
            System.arraycopy(csr.colIdx(), start, colIdx, rowPtr[r], length);
            for (int s = 0; s < capacity; s++) {
                System.arraycopy(values, s * nnz + start, moved, s * nnz + rowPtr[r], length);
            }
        }
        return new MovedRows(new Csr(n, rowPtr, colIdx), moved);
    }

    /** rowOrder for the moved rows' system to eliminate as the original did on order: pi o order. */
    private static int[] movedRowOrder(int[] pi, int[] order) {
        int[] rowOrder = new int[order.length];
        for (int k = 0; k < order.length; k++) {
            rowOrder[k] = pi[order[k]];
        }
        return rowOrder;
    }

    private static double[][] transpose(double[][] m) {
        double[][] t = new double[m.length][m.length];
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m.length; j++) {
                t[j][i] = m[i][j];
            }
        }
        return t;
    }

    private static void assertClose(double[] expected, double[] actual, String message) {
        double scale = Math.max(1, Arrays.stream(expected).map(Math::abs).max().orElse(0));
        assertArrayEquals(expected, actual, 1e-12 * scale, message);
    }

    /**
     * One factorization of the scenarios of {@code laneMap}, then both solves on it (factorizeAndSolveTransposed):
     * {@code A x = b} and {@code Aᵀ y = c} against the dense pivoted solve of each, every other scenario's solutions left
     * as they were, nothing clamped. Returns, per lane, whether the pivoting engine took it.
     */
    private static int[] assertSolvesBothWays(Csr csr, double[] values, int capacity, int[] laneMap, int[] rowOrder,
                                              int[] colOrder, long seed) {
        int n = csr.n();
        Random random = new Random(seed);
        double[] rightHandSides = new double[capacity * n];
        double[] transposedRightHandSides = new double[capacity * n];
        for (int k = 0; k < rightHandSides.length; k++) {
            rightHandSides[k] = random.nextDouble() * 2 - 1;
            transposedRightHandSides[k] = random.nextDouble() * 2 - 1;
        }
        double[] solutions = filled(capacity * n, 7);
        double[] transposedSolutions = filled(capacity * n, 9);
        int[] pivoted = new int[laneMap.length];
        int[] clamps = new int[laneMap.length];
        BatchedLuNative.factorizeAndSolveTransposed(n, csr.rowPtr(), csr.colIdx(), rowOrder, colOrder, capacity, values,
                rightHandSides, transposedRightHandSides, laneMap, solutions, transposedSolutions, pivoted, clamps);
        assertArrayEquals(new int[laneMap.length], clamps, "nothing clamped");
        boolean[] solved = new boolean[capacity];
        for (int s : laneMap) {
            solved[s] = true;
        }
        for (int s = 0; s < capacity; s++) {
            if (!solved[s]) {
                assertArrayEquals(filled(n, 7), scenario(solutions, s, n), "scenario " + s + " is outside the lane map");
                assertArrayEquals(filled(n, 9), scenario(transposedSolutions, s, n),
                        "scenario " + s + " is outside the lane map");
                continue;
            }
            double[][] a = dense(csr, values, s);
            assertClose(denseSolve(a, scenario(rightHandSides, s, n)), scenario(solutions, s, n), "A x = b, scenario " + s);
            assertClose(denseSolve(transpose(a), scenario(transposedRightHandSides, s, n)),
                    scenario(transposedSolutions, s, n), "Aᵀ y = c, scenario " + s);
        }
        return pivoted;
    }

    /**
     * The transposed solve on the plain solve's factors (the adjoint's solve, which cuDSS does not offer): one
     * factorization, then {@code A x = b} and {@code Aᵀ y = c}, both the dense pivoted solve's to rounding, on the fast
     * engine - shuffled orders, and every scenario's rows moved so the row and column orders differ: the plain solve
     * takes its right-hand side in through the row positions and its solution out through the column ones, the
     * transposed solve the other way round, and only differing orders tell the two apart (the host check,
     * tools/run_lu_host_check.sh, fails with them swapped on such systems only). Lanes outside the lane map keep both
     * their solutions.
     */
    @Test
    void transposedSolveOnTheFastEngineFactors() {
        for (int[] size : new int[][] {{40, 6}, {300, 70}, {257, 33}}) {
            int n = size[0];
            int capacity = size[1];
            Csr csr = toCsr(jacobianLikePattern(n, 4.0 / n, n));
            double[] values = diagonallyDominantValues(csr, capacity, n + 1L);
            int[] laneMap = java.util.stream.IntStream.range(0, capacity).filter(s -> s % 3 != 1).toArray();
            int[] order = shuffled(n, n + 2L);
            assertArrayEquals(new int[laneMap.length],
                    assertSolvesBothWays(csr, values, capacity, laneMap, order, order, n + 3L),
                    "a diagonally dominant matrix needs no pivoting, n=" + n);
            int[] pi = shuffled(n, n + 4L);
            MovedRows moved = moveRows(csr, values, capacity, pi);
            assertArrayEquals(new int[laneMap.length], assertSolvesBothWays(moved.csr(), moved.values(), capacity, laneMap,
                    movedRowOrder(pi, order), order, n + 5L), "rows moved: the same elimination, n=" + n);
        }
    }

    /**
     * The pivoting engine's transposed solve: every diagonal of a dense block zero, so each lane pivots inside its
     * supernodes (12 columns: one supernode, 40: three) with its own row and column orders, which the transposed solve
     * takes the other way round - the column order in, U_sᵀ up from the leaves, L_sᵀ down from the root, the row order
     * out. Rows moved too. Then the two lanes of {@link #eachLanePivotsInsideASupernode}, unsymmetric so that A and Aᵀ
     * differ: [[0, 2], [1, 0]] on the pivoting engine (a row swap), [[4, 1], [2, 3]] on the fast one, both directions
     * exact.
     */
    @Test
    void transposedSolveOnThePivotingEngineFactors() {
        for (int n : new int[] {12, 40}) {
            boolean[][] pattern = new boolean[n][n];
            for (boolean[] row : pattern) {
                Arrays.fill(row, true);
            }
            Csr csr = toCsr(pattern);
            int capacity = 3;
            double[] values = diagonallyDominantValues(csr, capacity, 21L + n);
            for (int s = 0; s < capacity; s++) {
                for (int i = 0; i < n; i++) {
                    values[s * csr.nnz() + i * n + i] = 0.0;
                }
            }
            int[] everyLane = new int[capacity];
            Arrays.fill(everyLane, 1);
            assertArrayEquals(everyLane, assertSolvesBothWays(csr, values, capacity, identity(capacity), identity(n),
                    identity(n), 23L + n), "every lane pivots, n=" + n);
            int[] pi = shuffled(n, 24L + n);
            MovedRows moved = moveRows(csr, values, capacity, pi);
            assertArrayEquals(everyLane, assertSolvesBothWays(moved.csr(), moved.values(), capacity, identity(capacity),
                    movedRowOrder(pi, identity(n)), identity(n), 25L + n), "rows moved, n=" + n);
        }
        Csr csr = toCsr(new boolean[][] {{true, true}, {true, true}});
        double[] values = {0, 2, 1, 0, 4, 1, 2, 3};
        double[] solutions = new double[4];
        double[] transposedSolutions = new double[4];
        int[] pivoted = new int[2];
        int[] clamps = new int[2];
        BatchedLuNative.factorizeAndSolveTransposed(2, csr.rowPtr(), csr.colIdx(), identity(2), identity(2), 2, values,
                new double[] {1, 2, 1, 2}, new double[] {1, 2, 1, 2}, new int[] {0, 1}, solutions, transposedSolutions,
                pivoted, clamps);
        assertArrayEquals(new int[] {1, 0}, pivoted);
        assertArrayEquals(new int[] {0, 0}, clamps);
        assertArrayEquals(new double[] {2, 0.5, 0.1, 0.6}, solutions, 1e-15);
        assertArrayEquals(new double[] {1, 1, -0.1, 0.7}, transposedSolutions, 1e-15);
    }

    /**
     * A long column is summed by several warps in the transposed solve, as a long row is in the plain one
     * (batchedLuSplitLongRows): a hub eliminated last has a U column of 299 terms (its row of Uᵀ), one eliminated first
     * an L column as long, and the transposed solve launches split levels of its own - more split launches than the
     * factorization and plain solve alone - and gives the dense solve's answer. With the parameter off, none.
     */
    @Test
    void longColumnsSplitOverSeveralWarpsInTheTransposedSolve() {
        try {
            for (boolean split : new boolean[] {true, false}) {
                new GpuLoadFlowParameters().setBatchedLuSplitLongRows(split).install();
                for (boolean hubLast : new boolean[] {true, false}) {
                    Csr csr = hubAndChain(hubLast ? 300 : 64, hubLast);
                    int n = csr.n();
                    int capacity = 37;
                    double[] values = diagonallyDominantValues(csr, capacity, 61);
                    long before = BatchedLuNative.factorizationCounts()[3];
                    BatchedLuNative.factorizeAndSolve(n, csr.rowPtr(), csr.colIdx(), identity(n), identity(n), capacity,
                            values, filled(capacity * n, 1), identity(capacity), new double[capacity * n],
                            new int[capacity], new int[capacity]);
                    long plain = BatchedLuNative.factorizationCounts()[3] - before;
                    before = BatchedLuNative.factorizationCounts()[3];
                    assertArrayEquals(new int[capacity], assertSolvesBothWays(csr, values, capacity, identity(capacity),
                            identity(n), identity(n), 62), "the fast engine factorizes every lane");
                    long both = BatchedLuNative.factorizationCounts()[3] - before;
                    assertTrue(split ? both > plain : both == 0,
                            "split=" + split + " hubLast=" + hubLast + ": " + plain + " then " + both + " split launches");
                }
            }
        } finally {
            new GpuLoadFlowParameters().install();                   // the next tests run with the defaults
        }
    }
}
