/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.math.matrix.LUDecomposition;
import com.powsybl.math.matrix.SparseMatrix;
import com.powsybl.math.matrix.SparseMatrixFactory;
import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Factors a Jacobian the DEVICE could not solve, with the CPU's own solver (KLU), to decide the only question
 * that matters about it: is it SINGULAR, or merely hard?
 *
 * <p>The device writes the matrix with {@code OLF_GPU_VALS_DUMP=<path>} (see {@code scanValsFinite} in
 * {@code olf_gpu_nr_jni.cu}), which fires on the first fill carrying a mode-4 row — the RL-pinned remote
 * gen-VC member where rte6515's switch-heavy base stops factoring. Point this test at that file with
 * {@code OLF_GPU_MATRIX_FILE}; with the variable unset it skips.
 *
 * <p>Why the CPU solver and not a better GPU configuration: OLF solves this network on the CPU, so a matrix
 * KLU calls singular is OUR assembly's defect — an equation missing or scattered twice — and no ordering,
 * pivot type or matching setting can rescue rank. A matrix KLU factors cleanly puts the fault on the device
 * solve instead. Measured on the device for this matrix: finite values, maxAbs 4.07e+04, every diagonal
 * present and above 1e-12, no duplicate row, no dead column, and a linear residual ratio of 8e+08 under
 * every ordering (AMD / ND / BTF-COLAMD) and pivot strategy (auto / global column / global row + threshold).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuDumpedMatrixRankTest {

    private record Dump(int n, int[] rowPtr, int[] colIdx, double[] vals, double[] f, int[] mode) { }

    private static Dump read(Path path) throws Exception {
        try (BufferedReader r = Files.newBufferedReader(path)) {
            String[] head = r.readLine().trim().split("\\s+");
            int n = Integer.parseInt(head[1]);
            int nnz = Integer.parseInt(head[3]);
            int[] rowPtr = ints(r.readLine(), n + 1);
            int[] colIdx = ints(r.readLine(), nnz);
            double[] vals = doubles(r.readLine(), nnz);
            double[] f = doubles(r.readLine(), n);
            int[] mode = ints(r.readLine(), n);
            return new Dump(n, rowPtr, colIdx, vals, f, mode);
        }
    }

    private static int[] ints(String line, int expected) {
        String[] t = line.trim().split("\\s+");
        if (t.length != expected) {
            throw new IllegalStateException("expected " + expected + " ints, got " + t.length);
        }
        int[] out = new int[expected];
        for (int i = 0; i < expected; i++) {
            out[i] = Integer.parseInt(t[i]);
        }
        return out;
    }

    private static double[] doubles(String line, int expected) {
        String[] t = line.trim().split("\\s+");
        if (t.length != expected) {
            throw new IllegalStateException("expected " + expected + " doubles, got " + t.length);
        }
        double[] out = new double[expected];
        for (int i = 0; i < expected; i++) {
            out[i] = Double.parseDouble(t[i]);
        }
        return out;
    }

    @Test
    void factorsTheDeviceRejectedMatrixWithKlu() throws Exception {
        String file = System.getenv("OLF_GPU_MATRIX_FILE");
        assumeTrue(file != null && !file.isEmpty(), "OLF_GPU_MATRIX_FILE required (a device OLF_GPU_VALS_DUMP)");
        Dump d = read(Path.of(file));
        int n = d.n();
        StringBuilder pinned = new StringBuilder();
        for (int r = 0; r < n; r++) {
            if (d.mode()[r] == 4) {
                pinned.append(' ').append(r);
            }
        }
        System.out.printf("DUMP n=%d nnz=%d mode4Rows=[%s]%n", n, d.vals().length, pinned.toString().trim());

        // SparseMatrix wants its entries COLUMN by column, so walk the CSR once per column.
        int nnz = d.vals().length;
        int[] rowOf = new int[nnz];
        for (int r = 0; r < n; r++) {
            for (int k = d.rowPtr()[r]; k < d.rowPtr()[r + 1]; k++) {
                rowOf[k] = r;
            }
        }
        Integer[] order = new Integer[nnz];
        for (int k = 0; k < nnz; k++) {
            order[k] = k;
        }
        java.util.Arrays.sort(order, (a, b) -> Integer.compare(d.colIdx()[a], d.colIdx()[b]));
        SparseMatrix m = (SparseMatrix) new SparseMatrixFactory().create(n, n, nnz);
        for (int k : order) {
            m.set(rowOf[k], d.colIdx()[k], d.vals()[k]);
        }

        double[] b = new double[n];
        for (int r = 0; r < n; r++) {
            b[r] = -d.f()[r];                                // the device solves J dx = -F
        }
        try (LUDecomposition lu = m.decomposeLU()) {
            lu.solve(b);
        } catch (RuntimeException e) {
            System.out.println("KLU REFUSED to factor this matrix: " + e);
            throw e;                                         // singular on the CPU solver too — our assembly
        }
        double dxInf = 0;
        for (double v : b) {
            dxInf = Math.max(dxInf, Math.abs(v));
        }
        // Residual of the CPU solution, in the device's own measure: ||J.dx - (-F)||inf / ||F||inf.
        double[] res = new double[n];
        for (int r = 0; r < n; r++) {
            double acc = 0;
            for (int k = d.rowPtr()[r]; k < d.rowPtr()[r + 1]; k++) {
                acc += d.vals()[k] * b[d.colIdx()[k]];
            }
            res[r] = acc + d.f()[r];
        }
        double rInf = 0;
        double fInf = 0;
        for (int r = 0; r < n; r++) {
            rInf = Math.max(rInf, Math.abs(res[r]));
            fInf = Math.max(fInf, Math.abs(d.f()[r]));
        }
        System.out.printf("KLU factored it. ||dx||inf=%.6e ||J.dx-B||inf=%.6e ||F||inf=%.6e ratio=%.3e%n",
                dxInf, rInf, fInf, fInf > 0 ? rInf / fInf : 0.0);
        // A solve this wrong has run away along a NULL direction, so the largest components of dx name the
        // columns the rank deficiency lives in — the fastest pointer there is to the variables whose
        // equations went missing or doubled. Print them with their row mode and the row's own entry count.
        Integer[] byMag = new Integer[n];
        for (int r = 0; r < n; r++) {
            byMag[r] = r;
        }
        final double[] sol = b;
        java.util.Arrays.sort(byMag, (x, y) -> Double.compare(Math.abs(sol[y]), Math.abs(sol[x])));
        System.out.println("largest |dx| columns (the null direction):");
        for (int i = 0; i < Math.min(12, n); i++) {
            int r = byMag[i];
            int entries = d.rowPtr()[r + 1] - d.rowPtr()[r];
            double diag = 0;
            for (int k = d.rowPtr()[r]; k < d.rowPtr()[r + 1]; k++) {
                if (d.colIdx()[k] == r) {
                    diag = d.vals()[k];
                }
            }
            System.out.printf("  row/col=%d dx=%.6e mode=%d rowEntries=%d diag=%.6e F=%.6e%n",
                    r, sol[r], d.mode()[r], entries, diag, d.f()[r]);
        }
        // WHICH equations touch those columns, and with what values: a variable whose only appearances
        // CANCEL is in no equation at all, which is the documented DISTR_Q-against-DUMMY_Q failure. Print
        // each suspect column as the list of (row, mode, value) that write into it, plus the column's sum.
        System.out.println("columns of the null direction, by the rows that touch them:");
        for (int i = 0; i < Math.min(4, n); i++) {
            int c = byMag[i];
            StringBuilder sb = new StringBuilder();
            double sum = 0;
            int touching = 0;
            for (int r = 0; r < n; r++) {
                for (int k = d.rowPtr()[r]; k < d.rowPtr()[r + 1]; k++) {
                    if (d.colIdx()[k] == c && d.vals()[k] != 0.0) {
                        sb.append(String.format(" (row=%d mode=%d v=%.6g)", r, d.mode()[r], d.vals()[k]));
                        sum += d.vals()[k];
                        touching++;
                    }
                }
            }
            System.out.printf("  col=%d touchedBy=%d colSum=%.6e%s%n", c, touching, sum, sb);
        }
    }
}
