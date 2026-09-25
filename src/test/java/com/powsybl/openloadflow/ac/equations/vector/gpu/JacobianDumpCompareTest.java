/*
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.math.matrix.LUDecomposition;
import com.powsybl.math.matrix.SparseMatrix;
import com.powsybl.math.matrix.SparseMatrixFactory;
import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * PURE NUMERICS, no GPU and no network — reads the two first-Newton-step dumps written by
 * {@code OLF_NR_DX_DUMP} (CPU arm) and {@code OLF_GPU_DX_DUMP} (device arm) and answers the
 * question the dumps were taken for: the two arms assemble the SAME Jacobian and the SAME
 * mismatch to 1-2 ulp, yet their steps differ by ~1e-10. Is that difference what ulp-level
 * assembly noise is WORTH on this matrix, or is it too big and therefore a defect?
 *
 * <p>The experiment is the only way to settle it: re-solve the CPU's own system with its own
 * entries nudged by +/-1 ulp and see how far the step moves. If a random 1-ulp nudge moves the
 * step as far as the device does, the two arms are as close as double precision allows on this
 * matrix. If it moves it far less, the device is doing something systematically different and
 * the gap is a bug.
 *
 * <p>Set {@code OLF_DUMP_DIR} to the directory holding {@code cpu.step0} / {@code gpu.step0}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class JacobianDumpCompareTest {

    /** One dumped Newton step: J in (equation row, variable column) orientation, F, dx. */
    private record Step(int n, Map<Long, Double> j, double[] f, double[] dx) { }

    private static Step read(Path p, boolean transpose) throws IOException {
        Map<Long, Double> j = new HashMap<>();
        Map<Integer, Double> f = new HashMap<>();
        Map<Integer, Double> dx = new HashMap<>();
        try (BufferedReader r = Files.newBufferedReader(p)) {
            String line;
            while ((line = r.readLine()) != null) {
                String[] t = line.split(" ");
                switch (t[0]) {
                    case "F" -> f.put(Integer.parseInt(t[1]), Double.parseDouble(t[2]));
                    case "DX" -> dx.put(Integer.parseInt(t[1]), Double.parseDouble(t[2]));
                    case "J" -> {
                        int a = Integer.parseInt(t[1]);
                        int b = Integer.parseInt(t[2]);
                        // CPU stores (varRow, eqCol) — OLF's transposed storage; the device stores
                        // (eqRow, varCol). Normalise both to (eqRow, varCol).
                        int row = transpose ? b : a;
                        int col = transpose ? a : b;
                        j.merge(((long) row << 32) | (col & 0xffffffffL), Double.parseDouble(t[3]), Double::sum);
                    }
                    default -> { }
                }
            }
        }
        int n = f.size();
        double[] fa = new double[n];
        double[] da = new double[n];
        for (int i = 0; i < n; i++) {
            fa[i] = f.get(i);
            da[i] = dx.get(i);
        }
        return new Step(n, j, fa, da);
    }

    private static SparseMatrix build(Step s, double[] perturbedValues) {
        // CSC: fill column by column in ascending order.
        List<List<int[]>> byCol = new ArrayList<>(s.n());
        for (int c = 0; c < s.n(); c++) {
            byCol.add(new ArrayList<>());
        }
        List<Long> keys = new ArrayList<>(s.j().keySet());
        keys.sort(null);
        double[] vals = new double[keys.size()];
        int[] rows = new int[keys.size()];
        int[] cols = new int[keys.size()];
        for (int k = 0; k < keys.size(); k++) {
            long key = keys.get(k);
            rows[k] = (int) (key >> 32);
            cols[k] = (int) key;
            vals[k] = perturbedValues == null ? s.j().get(key) : perturbedValues[k];
        }
        Integer[] order = new Integer[keys.size()];
        for (int k = 0; k < order.length; k++) {
            order[k] = k;
        }
        final int[] fcols = cols;
        java.util.Arrays.sort(order, (a, b) -> Integer.compare(fcols[a], fcols[b]));
        SparseMatrix m = (SparseMatrix) new SparseMatrixFactory().create(s.n(), s.n(), keys.size());
        for (int idx : order) {
            m.set(rows[idx], cols[idx], vals[idx]);
        }
        return m;
    }

    /** The values in the same order build() uses, so a perturbation lines up with the entries. */
    private static double[] valuesInBuildOrder(Step s) {
        List<Long> keys = new ArrayList<>(s.j().keySet());
        keys.sort(null);
        double[] v = new double[keys.size()];
        for (int k = 0; k < keys.size(); k++) {
            v[k] = s.j().get(keys.get(k));
        }
        return v;
    }

    private static double[] solve(SparseMatrix m, double[] rhs) {
        double[] b = rhs.clone();
        try (LUDecomposition lu = m.decomposeLU()) {
            lu.solve(b);
        }
        return b;
    }

    /**
     * Hager / Higham 1-norm estimator for {@code ||A^-1||_1}, the quantity that turns an input
     * difference into a solution difference. "Same J and same F to machine precision, so the
     * solution must match to machine precision" is only true when this is O(1): the actual bound is
     * {@code rel(x) <= kappa * rel(input)}, and on a power-flow Jacobian with retained switches
     * kappa is nowhere near 1. Needs both a solve and a transposed solve, which the LU gives.
     */
    private static double invNorm1(SparseMatrix m, int n) {
        double[] x = new double[n];
        java.util.Arrays.fill(x, 1.0 / n);
        double est = 0;
        try (LUDecomposition lu = m.decomposeLU()) {
            for (int iter = 0; iter < 8; iter++) {
                double[] y = x.clone();
                lu.solve(y);
                est = 0;
                for (double v : y) {
                    est += Math.abs(v);
                }
                double[] xi = new double[n];
                for (int i = 0; i < n; i++) {
                    xi[i] = y[i] >= 0 ? 1 : -1;
                }
                lu.solveTransposed(xi);
                int j = 0;
                for (int i = 1; i < n; i++) {
                    if (Math.abs(xi[i]) > Math.abs(xi[j])) {
                        j = i;
                    }
                }
                double zx = 0;
                for (int i = 0; i < n; i++) {
                    zx += xi[i] * x[i];
                }
                if (Math.abs(xi[j]) <= zx) {
                    break;
                }
                java.util.Arrays.fill(x, 0.0);
                x[j] = 1.0;
            }
        }
        return est;
    }

    private static double norm1(Step s, int n) {
        double[] col = new double[n];
        for (Map.Entry<Long, Double> e : s.j().entrySet()) {
            col[(int) (long) e.getKey().longValue()] += Math.abs(e.getValue());
        }
        double w = 0;
        for (double v : col) {
            w = Math.max(w, v);
        }
        return w;
    }

    private static double maxAbsDiff(double[] a, double[] b) {
        double w = 0;
        for (int i = 0; i < a.length; i++) {
            w = Math.max(w, Math.abs(a[i] - b[i]));
        }
        return w;
    }

    private static double maxAbs(double[] a) {
        double w = 0;
        for (double v : a) {
            w = Math.max(w, Math.abs(v));
        }
        return w;
    }

    @Test
    void ulpNoiseAccountsForTheStepDifference() throws IOException {
        String dir = System.getenv("OLF_DUMP_DIR");
        assumeTrue(dir != null, "OLF_DUMP_DIR not set");
        Path cpuPath = Path.of(dir, "cpu.step0");
        Path gpuPath = Path.of(dir, "gpu.step0");
        assumeTrue(Files.exists(cpuPath) && Files.exists(gpuPath), "dumps not found");

        Step cpu = read(cpuPath, true);
        Step gpu = read(gpuPath, false);
        System.out.printf("n=%d  cpu nnz=%d  gpu nnz=%d%n", cpu.n(), cpu.j().size(), gpu.j().size());

        SparseMatrix mc = build(cpu, null);
        double[] xc = solve(mc, cpu.f());
        System.out.printf("re-solved CPU system: |x - dumped dx|inf = %.3e   (||dx||inf=%.6e)%n",
                maxAbsDiff(xc, cpu.dx()), maxAbs(cpu.dx()));

        // Both systems are built as J x = F, so x is OLF's dx on both arms; the device's own dDx is
        // the solution of J dx = -F (it applies x += dx where OLF applies x -= dx), hence the negate.
        SparseMatrix mg = build(gpu, null);
        double[] xg = solve(mg, gpu.f());
        System.out.printf("re-solved GPU system: |x - (-dumped dx)|inf = %.3e%n",
                maxAbsDiff(xg, negate(gpu.dx())));
        double armGap = maxAbsDiff(xc, xg);
        System.out.printf("CPU step vs GPU step (both re-solved with KLU): |diff|inf = %.6e%n", armGap);

        // ---- the conditioning, which is what decides whether machine-precision inputs give a
        //      machine-precision solution at all ----
        double a1 = norm1(cpu, cpu.n());
        double ai1 = invNorm1(mc, cpu.n());
        System.out.printf("||J||1=%.4e  ||J^-1||1~%.4e  kappa_1~%.4e  -> eps*kappa = %.3e"
                + " (the floor on the RELATIVE step agreement)%n", a1, ai1, a1 * ai1, Math.ulp(1.0) * a1 * ai1);

        // ---- the experiment: how far does a RANDOM +/-1 ulp nudge of J move the step? ----
        double[] base = valuesInBuildOrder(cpu);
        Random rng = new Random(20260918L);
        double worstJ = 0;
        for (int trial = 0; trial < 5; trial++) {
            double[] pert = new double[base.length];
            for (int k = 0; k < base.length; k++) {
                pert[k] = base[k] + Math.ulp(base[k]) * (rng.nextInt(3) - 1);
            }
            double[] xp = solve(build(cpu, pert), cpu.f());
            double d = maxAbsDiff(xp, xc);
            worstJ = Math.max(worstJ, d);
            System.out.printf("  trial %d: +/-1 ulp on J -> |step change|inf = %.6e%n", trial, d);
        }

        // The same experiment sized to the perturbation the two arms ACTUALLY carry, measured
        // entry by entry: 64% bit-identical, 26% one ulp, 6% two, the rest a few more. An
        // independent +/-1-ulp nudge everywhere is a harsher perturbation than that, so without
        // this trial the comparison flatters the arms.
        double worstMatched = 0;
        for (int trial = 0; trial < 5; trial++) {
            double[] pert = new double[base.length];
            for (int k = 0; k < base.length; k++) {
                double r = rng.nextDouble();
                int nUlp = r < 0.6414 ? 0 : r < 0.9007 ? 1 : r < 0.9598 ? 2 : r < 0.9737 ? 3 : 4;
                pert[k] = base[k] + Math.ulp(base[k]) * nUlp * (rng.nextBoolean() ? 1 : -1);
            }
            double[] xp = solve(build(cpu, pert), cpu.f());
            double d = maxAbsDiff(xp, xc);
            worstMatched = Math.max(worstMatched, d);
            System.out.printf("  trial %d: MEASURED-distribution ulp noise on J -> |step change|inf = %.6e%n",
                    trial, d);
        }

        double worstF = 0;
        for (int trial = 0; trial < 3; trial++) {
            double[] fp = cpu.f().clone();
            for (int i = 0; i < fp.length; i++) {
                fp[i] += Math.ulp(fp[i]) * (rng.nextInt(3) - 1);
            }
            double[] xp = solve(mc, fp);
            double d = maxAbsDiff(xp, xc);
            worstF = Math.max(worstF, d);
            System.out.printf("  trial %d: +/-1 ulp on F -> |step change|inf = %.6e%n", trial, d);
        }

        System.out.printf("VERDICT arm gap=%.3e | 1-ulp J noise %.3e | measured-distribution J noise %.3e"
                + " | 1-ulp F noise %.3e%n", armGap, worstJ, worstMatched, worstF);
    }

    private static double[] negate(double[] a) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            r[i] = -a[i];
        }
        return r;
    }
}
