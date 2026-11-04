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
