package com.powsybl.openloadflow.sa.gpu.dc;

import com.powsybl.math.matrix.CuDssMatrixFactory;
import com.powsybl.math.matrix.DenseMatrix;
import com.powsybl.math.matrix.LUDecomposition;
import com.powsybl.math.matrix.SparseMatrix;
import com.powsybl.math.matrix.SparseMatrixFactory;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Scratch probe (not to be committed): cuDSS multi-RHS transposed solve vs KLU on a 1-D Laplacian. */
class CuDssMultiRhsProbe {

    @Test
    void multiRhs() {
        assumeTrue(CuDssMatrixFactory.isAvailable());
        int n = Integer.parseInt(System.getenv().getOrDefault("OLF_N", "5000"));
        int nrhs = Integer.parseInt(System.getenv().getOrDefault("OLF_NRHS", "1"));
        SparseMatrix cu;
        SparseMatrix kl;
        String matrixFile = System.getenv("OLF_MATRIX");
        if (matrixFile != null) {
            try (java.io.DataInputStream in = new java.io.DataInputStream(new java.io.BufferedInputStream(new java.io.FileInputStream(matrixFile)))) {
                n = in.readInt();
                int apLen = in.readInt();
                int[] ap = new int[apLen];
                for (int i = 0; i < apLen; i++) {
                    ap[i] = in.readInt();
                }
                int nnz = in.readInt();
                int[] ai = new int[nnz];
                for (int k = 0; k < nnz; k++) {
                    ai[k] = in.readInt();
                }
                double[] ax = new double[nnz];
                for (int k = 0; k < nnz; k++) {
                    ax[k] = in.readDouble();
                }
                long nanAx = 0;
                long zeroAx = 0;
                double minAbsDiag = Double.MAX_VALUE;
                int[] diagCount = new int[n];
                for (int j = 0; j + 1 < ap.length; j++) {
                    for (int k = ap[j]; k < ap[j + 1]; k++) {
                        if (Double.isNaN(ax[k])) {
                            nanAx++;
                        }
                        if (ax[k] == 0) {
                            zeroAx++;
                        }
                        if (ai[k] == j) {
                            diagCount[j]++;
                            minAbsDiag = Math.min(minAbsDiag, Math.abs(ax[k]));
                        }
                    }
                }
                long missingDiag = java.util.Arrays.stream(diagCount).filter(c -> c == 0).count();
                System.out.printf("CUDSS-PROBE matrix stats: NaN values %d, explicit zeros %d, columns without diagonal %d, min |diag| %.3e%n",
                        nanAx, zeroAx, missingDiag, minAbsDiag);
                cu = fill(new CuDssMatrixFactory().create(n, n, nnz), ap, ai, ax);
                kl = fill(new SparseMatrixFactory().create(n, n, nnz), ap, ai, ax);
                System.out.printf("CUDSS-PROBE matrix %s n=%d nnz=%d%n", matrixFile, n, nnz);
            } catch (java.io.IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
        } else {
            cu = build(new CuDssMatrixFactory().create(n, n, 3 * n), n);
            kl = build(new SparseMatrixFactory().create(n, n, 3 * n), n);
        }
        DenseMatrix bCu = new DenseMatrix(n, nrhs);
        DenseMatrix bKl = new DenseMatrix(n, nrhs);
        for (int j = 0; j < nrhs; j++) {
            for (int i = 0; i < n; i++) {
                double v = "1".equals(System.getenv("OLF_SPARSE_RHS"))
                        ? (i == (2 * j) % n ? 1.0 : i == (2 * j + 1) % n ? -1.0 : 0.0)   // the Woodbury +1/-1 pattern
                        : Math.sin(i * 0.01 + j);
                bCu.set(i, j, v);
                bKl.set(i, j, v);
            }
        }
        long t0 = System.nanoTime();
        try (LUDecomposition lu = cu.decomposeLU()) {
            if ("1".equals(System.getenv("OLF_VEC_FIRST"))) {
                double[] v = new double[n];
                lu.solveTransposed(v);                          // a double[] solve first, like the pre-contingency solve
            }
            if ("1".equals(System.getenv("OLF_UPDATE"))) {
                lu.update(false);                               // a refactorization before the dense solve
            }
            lu.solveTransposed(bCu);
        }
        long tCu = System.nanoTime() - t0;
        t0 = System.nanoTime();
        try (LUDecomposition lu = kl.decomposeLU()) {
            lu.solveTransposed(bKl);
        }
        long tKl = System.nanoTime() - t0;
        long nanCu = 0;
        long nanKl = 0;
        for (int j = 0; j < nrhs; j++) {
            for (int i = 0; i < n; i++) {
                if (Double.isNaN(bCu.get(i, j))) {
                    nanCu++;
                }
                if (Double.isNaN(bKl.get(i, j))) {
                    nanKl++;
                }
            }
        }
        System.out.printf("CUDSS-PROBE NaN in cuDSS solution: %d, in KLU solution: %d%n", nanCu, nanKl);
        double maxDiff = 0;
        for (int j = 0; j < nrhs; j++) {
            for (int i = 0; i < n; i++) {
                maxDiff = Math.max(maxDiff, Math.abs(bCu.get(i, j) - bKl.get(i, j)));
            }
        }
        System.out.printf("CUDSS-PROBE n=%d nrhs=%d maxDiff=%.3e cuDSS %.1f ms KLU %.1f ms%n", n, nrhs, maxDiff, tCu / 1e6, tKl / 1e6);
        assertTrue(maxDiff < 1e-8, "diff " + maxDiff);
    }

    private static SparseMatrix fill(com.powsybl.math.matrix.Matrix m, int[] ap, int[] ai, double[] ax) {
        for (int j = 0; j + 1 < ap.length; j++) {
            for (int k = ap[j]; k < ap[j + 1]; k++) {
                m.add(ai[k], j, ax[k]);
            }
        }
        return (SparseMatrix) m;
    }

    private static SparseMatrix build(com.powsybl.math.matrix.Matrix m, int n) {
        // column-compressed fill order: column by column, ascending row
        for (int j = 0; j < n; j++) {
            if (j > 0) {
                m.add(j - 1, j, -1.0);
            }
            m.add(j, j, 2.5 + 0.001 * j);
            if (j < n - 1) {
                m.add(j + 1, j, -1.0);
            }
        }
        return (SparseMatrix) m;
    }
}
