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

    /**
     * The lanes the fast engine flags go to the pivoting engine, which solves them: the fast engine's triangular solves
     * run on the lanes it trusts only, in the solve of every lane and in the solve of selected lanes, and every lane's
     * solution stays the one the whole batch gives it. Here the fast engine flags 2 of its 5 lanes (scenario 1 among
     * them), so the whole solve runs it on 3 lanes and the selected solve of scenarios 1 and 0 on 1. Checked to fail with
     * the flagged lanes solved by the fast engine too, in the whole solve and in the selection.
     */
    @Test
    void flaggedLanesAreSolvedByThePivotingEngineAlone() {
        CouplerSystem system = couplerSystem(32, 80, 3, 0.03);
        Csr csr = system.csr();
        int n = csr.n();
        double[] values = refinementBatch(system);
        double[] rightHandSides = refinementRightHandSides(6 * n);
        double[] whole = filled(6 * n, 42);
        int[] pivoted = new int[REFINEMENT_LANE_MAP.length];
        int[] clamps = new int[REFINEMENT_LANE_MAP.length];
        long before = BatchedLuNative.factorizationCounts()[6];
        BatchedLuNative.factorizeAndSolve(n, csr.rowPtr(), csr.colIdx(), identity(n), identity(n), 6, values,
                rightHandSides, REFINEMENT_LANE_MAP, whole, pivoted, clamps);
        long wholeSolves = BatchedLuNative.factorizationCounts()[6] - before;
        assertArrayEquals(new int[] {1, 0, 1, 0, 0}, pivoted);
        assertEquals(3, wholeSolves, "lanes the fast engine solved in the whole solve: the 3 it trusts");
        double[] selected = filled(6 * n, 42);
        before = BatchedLuNative.factorizationCounts()[6];
        BatchedLuNative.factorizeAndSolveSelected(n, csr.rowPtr(), csr.colIdx(), identity(n), identity(n), 6, values,
                rightHandSides, REFINEMENT_LANE_MAP, new int[] {1, 0}, selected);
        long selectedSolves = BatchedLuNative.factorizationCounts()[6] - before;
        assertEquals(1, selectedSolves, "lanes the fast engine solved in the selection: scenario 0's, not flagged 1's");
        for (int s = 0; s < 6; s++) {
            double[] expected = s <= 1 ? scenario(whole, s, n) : filled(n, 42);
            assertArrayEquals(expected, scenario(selected, s, n), "scenario " + s);
        }
    }

    /**
     * The fast engine learns its pivot delays from the values of the lanes it flagged; interleaved by scenario, as a
     * batch context holds them, they are gathered on the device by scenario. The line group of
     * cancelledPivotIsDelayedUntilARowGivesItAValue on scenarios 2, 3 and 4 of 5, interleaved, learns as it does there:
     * scenarios 0 and 1, outside the lane map, hold NaN values, which teach nothing, and keep their solutions. Checked to
     * fail with the gather taking a lane's position in the lane map as its scenario (it reads scenarios 0 and 1).
     */
    @Test
    void pivotDelaysAreLearnedFromInterleavedValues() {
        Random random = new Random(11);
        int capacity = 5;
        double[][] lanes = new double[capacity][];
        for (int s = 0; s < capacity; s++) {
            double g = 1 + 9 * random.nextDouble();
            lanes[s] = s < 2 ? new double[] {Double.NaN, Double.NaN, Double.NaN} : new double[] {g, g, 1 + 9 * random.nextDouble()};
        }
        double[] values = lineGroupValues(lanes);
        double[] rightHandSides = new double[capacity * 4];
        for (int i = 0; i < rightHandSides.length; i++) {
            rightHandSides[i] = random.nextDouble() * 2 - 1;
        }
        int[] laneMap = {2, 3, 4};
        double[] solutions = filled(capacity * 4, 42);
        int[] pivoted = new int[4 * laneMap.length];
        int[] roundDelays = new int[4];
        BatchedLuNative.factorizeRoundsInterleaved(4, LINE_GROUP.rowPtr(), LINE_GROUP.colIdx(), identity(4), identity(4),
                capacity, values, rightHandSides, laneMap, 4, -1, solutions, pivoted, roundDelays);
        int[] expected = new int[4 * laneMap.length];
        Arrays.fill(expected, 0, 2 * laneMap.length, 1);
        assertArrayEquals(expected, pivoted, "lanes the pivoting engine took, round by round");
        assertArrayEquals(new int[] {0, 1, 1, 1}, roundDelays, "pivot delays of each round's order");
        for (int s = 0; s < capacity; s++) {
            double[] solution = Arrays.copyOfRange(solutions, s * 4, s * 4 + 4);
            if (s < 2) {
                assertArrayEquals(filled(4, 42), solution, "scenario " + s + " is outside the lane map");
                continue;
            }
            double[] expectedSolution = denseSolve(dense(LINE_GROUP, values, s), Arrays.copyOfRange(rightHandSides, s * 4, s * 4 + 4));
            double scale = Arrays.stream(expectedSolution).map(Math::abs).max().orElseThrow();
            for (int i = 0; i < 4; i++) {
                assertEquals(expectedSolution[i], solution[i], 1e-12 * scale, "scenario " + s + " unknown " + i);
            }
        }
    }

    /**
     * Values interleaved by scenario, as a batch context holds them, are read in place by both engines - the fast
     * one's factorization steps and the pivoting one's gathers, a fill entry starting from 0 - and give every lane the
     * solution, the engine and the clamps the values copied from the scenario-major layout give it, bit for bit: lanes
     * on both engines, a lane map neither in order nor holding every scenario, and a stride (8) above the 6 scenarios.
     * Checked to fail with either engine's in-place read taking lane a's scenario as a, or a fill entry starting from
     * 1 instead of 0.
     */
    @Test
    void interleavedValuesAreReadInPlaceAsTheCopiedOnes() {
        CouplerSystem system = couplerSystem(32, 80, 3, 0.03);
        Csr csr = system.csr();
        int n = csr.n();
        double[] values = refinementBatch(system);
        double[] rightHandSides = refinementRightHandSides(6 * n);
        double[] copied = filled(6 * n, 42);
        int[] copiedPivoted = new int[REFINEMENT_LANE_MAP.length];
        int[] copiedClamps = new int[REFINEMENT_LANE_MAP.length];
        BatchedLuNative.factorizeAndSolve(n, csr.rowPtr(), csr.colIdx(), identity(n), identity(n), 6, values,
                rightHandSides, REFINEMENT_LANE_MAP, copied, copiedPivoted, copiedClamps);
        double[] inPlace = filled(6 * n, 42);
        int[] inPlacePivoted = new int[REFINEMENT_LANE_MAP.length];
        int[] inPlaceClamps = new int[REFINEMENT_LANE_MAP.length];
        BatchedLuNative.factorizeAndSolveInterleaved(n, csr.rowPtr(), csr.colIdx(), identity(n), identity(n), 6, values,
                rightHandSides, REFINEMENT_LANE_MAP, inPlace, inPlacePivoted, inPlaceClamps);
        assertArrayEquals(new int[] {1, 0, 1, 0, 0}, copiedPivoted);
        assertArrayEquals(copiedPivoted, inPlacePivoted);
        assertArrayEquals(copiedClamps, inPlaceClamps);
        for (int s = 0; s < 6; s++) {
            assertArrayEquals(scenario(copied, s, n), scenario(inPlace, s, n), "scenario " + s);
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
     * The pivoting engine's forward solve splits its long rows over several warps as the fast engine does: the hub of
     * hubAndChain(300, true), eliminated last, holds 299 L terms in a level of one supernode. A dense corner of four
     * rows whose first three diagonals are zero on every lane, its couplings strong, sends every lane to the pivoting engine, which pivots
     * inside that corner's supernode. The warps' partial sums are added in a fixed order, so the solutions are the dense
     * pivoted solve's to rounding, on all 37 lanes and on 5 selected ones, which take the solve of every lane bit for
     * bit. With the parameter off no pivoting-engine level is split. Checked to fail with the last warp's partial sum
     * left out of the forward solve.
     */
    @Test
    void pivotingEngineSplitsItsLongForwardRows() {
        int n = 300;
        int capacity = 37;
        boolean[][] pattern = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            pattern[i][i] = true;
            pattern[i][n - 1] = true;
            pattern[n - 1][i] = true;
            if (i + 1 < n) {
                pattern[i][i + 1] = true;
                pattern[i + 1][i] = true;
            }
        }
        for (int i = 0; i < 4; i++) {
            Arrays.fill(pattern[i], 0, 4, true);
        }
        Csr csr = toCsr(pattern);
        double[] values = diagonallyDominantValues(csr, capacity, 61);
        Random corner = new Random(63);
        for (int s = 0; s < capacity; s++) {
            for (int i = 0; i < 4; i++) {
                for (int k = csr.rowPtr()[i]; k < csr.rowPtr()[i + 1]; k++) {
                    int j = csr.colIdx()[k];
                    if (j < 4) {                                     // zero diagonals, strong couplings in the corner
                        values[s * csr.nnz() + k] = j == i && i < 3 ? 0.0 : 20 + 5 * corner.nextDouble();
                    }
                }
            }
        }
        double[] rightHandSides = new double[capacity * n];
        Random random = new Random(62);
        for (int k = 0; k < rightHandSides.length; k++) {
            rightHandSides[k] = random.nextDouble() * 2 - 1;
        }
        int[] all = new int[capacity];
        Arrays.fill(all, 1);
        int[] chosen = {capacity - 1, 0, 33, 5, 31};
        try {
            for (boolean split : new boolean[] {true, false}) {
                new GpuLoadFlowParameters().setBatchedLuSplitLongRows(split).install();
                long before = BatchedLuNative.factorizationCounts()[4];
                double[] whole = filled(capacity * n, 7);
                int[] pivoted = new int[capacity];
                int[] clamps = new int[capacity];
                BatchedLuNative.factorizeAndSolve(n, csr.rowPtr(), csr.colIdx(), identity(n), identity(n), capacity,
                        values, rightHandSides, identity(capacity), whole, pivoted, clamps);
                assertArrayEquals(all, pivoted, "every lane goes to the pivoting engine");
                assertArrayEquals(new int[capacity], clamps);
                double[] selected = filled(capacity * n, 7);
                BatchedLuNative.factorizeAndSolveSelected(n, csr.rowPtr(), csr.colIdx(), identity(n), identity(n),
                        capacity, values, rightHandSides, identity(capacity), chosen, selected);
                for (int s = 0; s < capacity; s++) {
                    double[] reference = denseSolve(dense(csr, values, s), scenario(rightHandSides, s, n));
                    assertArrayEquals(reference, scenario(whole, s, n), 1e-12, "split=" + split + " scenario " + s);
                    final int scenario = s;
                    boolean isChosen = Arrays.stream(chosen).anyMatch(c -> c == scenario);
                    assertArrayEquals(isChosen ? scenario(whole, s, n) : filled(n, 7), scenario(selected, s, n),
                            "split=" + split + " selected scenario " + s);
                }
                long launches = BatchedLuNative.factorizationCounts()[4] - before;
                assertTrue(split ? launches > 0 : launches == 0, "split=" + split + ": " + launches + " split launches");
            }
        } finally {
            new GpuLoadFlowParameters().install();                   // the next tests run with the defaults
        }
    }

    /**
     * The pivoting engine spreads a supernode's rows where a level leaves the GPU threads to spare - over warps in a
     * backward level, over blocks in a split forward level - and factorizes a diagonal block of 8 rows or more a row per
     * warp: each form solves as the one-thread-per-supernode kernels do, bit for bit, on all 37 lanes and on 5 selected
     * ones (setPivotingEngineOneThreadKernelsForTests). Four systems:
     * <ul>
     * <li>40 units of two dense 3-row blocks and 2 border rows joined to both (each block a supernode with the border rows
     * in its U panel) and a dense block of 16 rows on its own: narrow backward levels, no forward split;</li>
     * <li>the hub and chain of pivotingEngineSplitsItsLongForwardRows: the hub row's 299 L terms, a split forward level;</li>
     * <li>an 8-row supernode with a 2-row U panel (it and a 2-row block are the children of the last 2 rows): an identity
     * row (empty panel) after a row of a single nonzero whose panel is not empty, and 100 in the identity row's column
     * on another row - complete pivoting would take it first.
     * The singleton pivot keeps the identity row's value at its right-hand side to the last bit;</li>
     * <li>a dense 16-row block of zero diagonals and off-diagonal entries in -3..3: complete pivoting meets ties, broken as
     * the one-thread kernel breaks them.</li>
     * </ul>
     * Every lane goes to the pivoting engine (zero diagonals). Checked to fail with a row's last U panel term left out of
     * backwardLevelRows, with forwardLevelSplitRows' partial sums added from the last warp, with factorBlocksRows' ties
     * going to the last row and with its singleton rows ignored.
     */
    @Test
    void pivotingEngineSpreadKernelsSolveAsTheirOneThreadForms() {
        int capacity = 37;
        int[] chosen = {capacity - 1, 0, 33, 5, 31};
        Random random = new Random(81);
        // narrow backward levels
        int units = 40;
        int n = units * 8 + 16;
        boolean[][] pattern = new boolean[n][n];
        for (int u = 0; u < units; u++) {
            int first = u * 8;
            for (int i = first; i < first + 8; i++) {
                for (int j = first; j < first + 8; j++) {
                    boolean blockA = i < first + 3 && j < first + 3;
                    boolean blockB = i >= first + 3 && i < first + 6 && j >= first + 3 && j < first + 6;
                    pattern[i][j] = blockA || blockB || i >= first + 6 || j >= first + 6;
                }
            }
        }
        for (int i = units * 8; i < n; i++) {
            Arrays.fill(pattern[i], units * 8, n, true);
        }
        Csr units8 = toCsr(pattern);
        double[] unitValues = withZeroDiagonalCorner(units8, diagonallyDominantValues(units8, capacity, 71), capacity, 3, 2);
        long[] counts = assertSpreadSolvesAsOneThread(units8, unitValues, capacity, chosen, random);
        assertTrue(counts[7] > 0, "backward levels solved a row per warp: " + counts[7]);
        assertTrue(counts[9] > 0, "diagonal blocks factorized a row per warp: " + counts[9]);
        assertEquals(0, counts[4], "no forward level is split");
        // a split forward level
        int hubN = 300;
        boolean[][] hub = new boolean[hubN][hubN];
        for (int i = 0; i < hubN; i++) {
            hub[i][i] = true;
            hub[i][hubN - 1] = true;
            hub[hubN - 1][i] = true;
            if (i + 1 < hubN) {
                hub[i][i + 1] = true;
                hub[i + 1][i] = true;
            }
        }
        for (int i = 0; i < 4; i++) {
            Arrays.fill(hub[i], 0, 4, true);
        }
        Csr hubCsr = toCsr(hub);
        double[] hubValues = withZeroDiagonalCorner(hubCsr, diagonallyDominantValues(hubCsr, capacity, 61), capacity, 4, 3);
        counts = assertSpreadSolvesAsOneThread(hubCsr, hubValues, capacity, chosen, random);
        assertTrue(counts[8] > 0, "split forward levels summed a row per block: " + counts[8]);
        // a singleton row in a supernode with a U panel: block A (rows 0-7) and block B (rows 8-9) are the two children
        // of rows 10-11, so A is a supernode of its own with rows 10-11 in its U panel; NaN: no entry, 0: an entry that is 0
        double x = Double.NaN;
        int sn = 12;
        double[][] m = new double[sn][sn];
        for (double[] row : m) {
            Arrays.fill(row, x);
        }
        for (int i = 0; i < 8; i++) {
            Arrays.fill(m[i], 0, 8, 0.0);                    // A dense
            m[i][10] = 0.0;
            m[i][11] = 0.0;
            m[10][i] = 0.0;
            m[11][i] = 0.0;
        }
        m[0][0] = 3;                                         // a single nonzero in A, its panel not empty
        m[0][10] = 1;
        m[1][1] = 1;                                         // the identity row, its panel empty
        m[2][0] = 2;                                         // 100 in the identity row's column, a zero diagonal
        m[2][1] = 100;
        for (int c = 3; c < 8; c++) {
            m[2][c] = 1;
        }
        m[2][11] = 1;
        Random blockValues = new Random(83);
        for (int i = 3; i < 8; i++) {
            for (int c = 0; c < 8; c++) {
                m[i][c] = c == i ? 40 : 1 + 4 * blockValues.nextDouble();
            }
            m[i][10 + i % 2] = 2;
            m[10 + i % 2][i] = 1;
        }
        m[8][8] = 10;                                        // block B
        m[8][9] = 1;
        m[9][8] = 1;
        m[9][9] = 10;
        m[8][10] = 1;
        m[9][11] = 1;
        m[10][8] = 1;
        m[11][9] = 1;
        m[10][10] = 10;                                      // the two last rows
        m[10][11] = 1;
        m[11][10] = 1;
        m[11][11] = 10;
        m[10][0] = 1;
        m[11][2] = 1;
        boolean[][] singleton = new boolean[sn][sn];
        for (int i = 0; i < sn; i++) {
            for (int j = 0; j < sn; j++) {
                singleton[i][j] = !Double.isNaN(m[i][j]);
            }
        }
        Csr singletonCsr = toCsr(singleton);
        double[] singletonValues = new double[capacity * singletonCsr.nnz()];
        for (int sc = 0; sc < capacity; sc++) {
            for (int i = 0; i < sn; i++) {
                for (int k = singletonCsr.rowPtr()[i]; k < singletonCsr.rowPtr()[i + 1]; k++) {
                    int j = singletonCsr.colIdx()[k];
                    singletonValues[sc * singletonCsr.nnz() + k] = i == 1 && j == 1 ? 1.0 : m[i][j] * (1 + sc / 64.0);
                }
            }
        }
        double[] singletonRightHandSides = new double[capacity * sn];
        for (int k = 0; k < singletonRightHandSides.length; k++) {
            singletonRightHandSides[k] = random.nextDouble() * 2 - 1;
        }
        double[][] solved = new double[2][];
        counts = assertSpreadSolvesAsOneThread(singletonCsr, singletonValues, capacity, chosen, singletonRightHandSides, solved);
        assertTrue(counts[9] > 0, "the singleton block factorized a row per warp: " + counts[9]);
        for (int s = 0; s < capacity; s++) {
            assertEquals(singletonRightHandSides[s * sn + 1], solved[0][s * sn + 1], 0.0, "identity row of scenario " + s);
        }
        // ties: a dense 16-row block, zero diagonals, off-diagonal entries in -3..3 without 0
        boolean[][] dense16 = new boolean[16][16];
        for (boolean[] row : dense16) {
            Arrays.fill(row, true);
        }
        Csr tiesCsr = toCsr(dense16);
        double[] tiesValues = new double[capacity * tiesCsr.nnz()];
        Random ties = new Random(87);
        for (int sc = 0; sc < capacity; sc++) {
            for (int i = 0; i < 16; i++) {
                for (int k = tiesCsr.rowPtr()[i]; k < tiesCsr.rowPtr()[i + 1]; k++) {
                    int v = 1 + ties.nextInt(3);
                    tiesValues[sc * tiesCsr.nnz() + k] = tiesCsr.colIdx()[k] == i ? 0.0 : (ties.nextBoolean() ? v : -v);
                }
            }
        }
        counts = assertSpreadSolvesAsOneThread(tiesCsr, tiesValues, capacity, chosen, random);
        assertTrue(counts[9] > 0, "the tied block factorized a row per warp: " + counts[9]);
    }

    /** values with, on every lane, the first `size` rows' block made of zero diagonals (the first `zeros` rows) and
     *  strong couplings: every lane goes to the pivoting engine. */
    private static double[] withZeroDiagonalCorner(Csr csr, double[] values, int capacity, int size, int zeros) {
        Random corner = new Random(63);
        for (int s = 0; s < capacity; s++) {
            for (int i = 0; i < size; i++) {
                for (int k = csr.rowPtr()[i]; k < csr.rowPtr()[i + 1]; k++) {
                    int j = csr.colIdx()[k];
                    if (j < size) {
                        values[s * csr.nnz() + k] = j == i && i < zeros ? 0.0 : 20 + 5 * corner.nextDouble();
                    }
                }
            }
        }
        return values;
    }

    private static long[] assertSpreadSolvesAsOneThread(Csr csr, double[] values, int capacity, int[] chosen, Random random) {
        double[] rightHandSides = new double[capacity * csr.n()];
        for (int k = 0; k < rightHandSides.length; k++) {
            rightHandSides[k] = random.nextDouble() * 2 - 1;
        }
        return assertSpreadSolvesAsOneThread(csr, values, capacity, chosen, rightHandSides, new double[2][]);
    }

    /** Solves every lane and the chosen ones with the pivoting engine's spread kernels and with their one-thread forms:
     *  every lane pivots, the solutions are the dense solve's to 1e-12 of their scale and equal bit for bit between the two forms
     *  (solved: the whole batch's solutions of each, spread first). Returns the counts the spread solve moved; the
     *  one-thread solve must move none of the three it replaces. */
    private static long[] assertSpreadSolvesAsOneThread(Csr csr, double[] values, int capacity, int[] chosen,
                                                        double[] rightHandSides, double[][] solved) {
        int n = csr.n();
        int[] all = new int[capacity];
        Arrays.fill(all, 1);
        double[][] whole = new double[2][];
        double[][] selected = new double[2][];
        long[] spreadCounts = null;
        try {
            for (int form = 0; form < 2; form++) {
                BatchedLuNative.setPivotingEngineOneThreadKernelsForTests(form == 1);
                long[] before = BatchedLuNative.factorizationCounts();
                whole[form] = filled(capacity * n, 7);
                int[] pivoted = new int[capacity];
                int[] clamps = new int[capacity];
                BatchedLuNative.factorizeAndSolve(n, csr.rowPtr(), csr.colIdx(), identity(n), identity(n), capacity,
                        values, rightHandSides, identity(capacity), whole[form], pivoted, clamps);
                assertArrayEquals(all, pivoted, "every lane goes to the pivoting engine");
                assertArrayEquals(new int[capacity], clamps);
                selected[form] = filled(capacity * n, 7);
                BatchedLuNative.factorizeAndSolveSelected(n, csr.rowPtr(), csr.colIdx(), identity(n), identity(n),
                        capacity, values, rightHandSides, identity(capacity), chosen, selected[form]);
                long[] after = BatchedLuNative.factorizationCounts();
                long[] moved = new long[after.length];
                for (int k = 0; k < after.length; k++) {
                    moved[k] = after[k] - before[k];
                }
                if (form == 0) {
                    spreadCounts = moved;
                } else {
                    assertEquals(0, moved[7] + moved[8] + moved[9], "the one-thread forms spread nothing");
                }
                for (int s = 0; s < capacity; s++) {
                    double[] reference = denseSolve(dense(csr, values, s), scenario(rightHandSides, s, n));
                    double scale = Math.max(1, Arrays.stream(reference).map(Math::abs).max().orElseThrow());
                    assertArrayEquals(reference, scenario(whole[form], s, n), 1e-12 * scale, "form " + form + " scenario " + s);
                }
            }
        } finally {
            BatchedLuNative.setPivotingEngineOneThreadKernelsForTests(false);
        }
        assertArrayEquals(whole[1], whole[0], "every lane: spread as one thread, bit for bit");
        assertArrayEquals(selected[1], selected[0], "the selected lanes: spread as one thread, bit for bit");
        solved[0] = whole[0];
        solved[1] = whole[1];
        return spreadCounts;
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
                values, rightHandSides, laneMap, 4, -1, solutions, pivoted, roundDelays);
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

    /** The pattern of {@link #orderCostlierThanTheLanesItSparesIsNotTaken}: pairs of rows then the hub row. */
    private static Csr hubPairs(int pairs) {
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
        return new Csr(n, rowPtr, columns.stream().mapToInt(Integer::intValue).toArray());
    }

    /** Its values, per lane s: every x_i's pivot xPivot[s], the hub's x entries 10 + s. */
    private static double[] hubPairsValues(Csr csr, int pairs, double[] xPivot) {
        double[] values = new double[xPivot.length * csr.nnz()];
        for (int s = 0; s < xPivot.length; s++) {
            int k = s * csr.nnz();
            for (int i = 0; i < pairs; i++) {
                values[k++] = xPivot[s];                     // x_i's pivot
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
        return values;
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
        Csr csr = hubPairs(pairs);
        int capacity = 2;
        double[] values = hubPairsValues(csr, pairs, new double[] {0, 0});

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
                rightHandSides, identity(capacity), 4, -1, solutions, pivoted, roundDelays);
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

    /**
     * A lane keeps the engine it ended on. The hub pairs on three lanes: the first two cancel every x pivot, the third
     * one's pivots are 2, so the fast engine keeps it; the order never changes (see
     * {@link #orderCostlierThanTheLanesItSparesIsNotTaken}). The first factorization flags lanes 0 and 1, and from the
     * second on they go straight to the pivoting engine, the fast engine factorizing lane 2 alone - until every lane's
     * scenario is installed anew before the third round, which sends them through the fast engine once more. Every
     * round's lanes are the same, and the last factorization solves each lane exactly. Checked to fail with the
     * routing off (no lane routed) and with installs that forget nothing (6 lanes routed).
     */
    @Test
    void pivotedLaneGoesStraightBackToThePivotingEngine() {
        int pairs = 16;
        int n = 2 * pairs + 1;
        Csr csr = hubPairs(pairs);
        int capacity = 3;
        double[] values = hubPairsValues(csr, pairs, new double[] {0, 0, 2});
        Random random = new Random(17);
        double[] rightHandSides = new double[capacity * n];
        for (int i = 0; i < rightHandSides.length; i++) {
            rightHandSides[i] = random.nextDouble() * 2 - 1;
        }
        double[] solutions = new double[capacity * n];
        int[] pivoted = new int[4 * capacity];
        int[] roundDelays = new int[4];
        long routedBefore = BatchedLuNative.factorizationCounts()[5];
        BatchedLuNative.factorizeRounds(n, csr.rowPtr(), csr.colIdx(), identity(n), identity(n), capacity, values,
                rightHandSides, identity(capacity), 4, 2, solutions, pivoted, roundDelays);
        long routed = BatchedLuNative.factorizationCounts()[5] - routedBefore;
        assertArrayEquals(new int[4], roundDelays, "pivot delays of each round's order");
        assertArrayEquals(new int[] {1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0}, pivoted,
                "lanes the pivoting engine took, round by round");
        assertEquals(4, routed, "lanes sent straight to the pivoting engine: rounds 1 and 3, lanes 0 and 1");
        for (int s = 0; s < capacity; s++) {
            double[] rightHandSide = Arrays.copyOfRange(rightHandSides, s * n, (s + 1) * n);
            double[] expected = denseSolve(dense(csr, values, s), rightHandSide);
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
}
