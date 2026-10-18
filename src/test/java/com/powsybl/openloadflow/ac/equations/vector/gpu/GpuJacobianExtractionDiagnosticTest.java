/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory;
import com.powsybl.iidm.network.Network;
import com.powsybl.math.matrix.SparseMatrix;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.ac.AcTargetVector;
import com.powsybl.openloadflow.ac.equations.AcEquationSystemCreator;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.solver.AcSolverUtil;
import com.powsybl.openloadflow.ac.solver.gpu.TransposedSolveNative;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.EquationVector;
import com.powsybl.openloadflow.equations.JacobianMatrix;
import com.powsybl.openloadflow.equations.StateVector;
import com.powsybl.openloadflow.network.FirstSlackBusSelector;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfNetworkParameters;
import com.powsybl.openloadflow.network.impl.Networks;
import com.powsybl.openloadflow.network.util.UniformValueVoltageInitializer;
import org.junit.jupiter.api.Test;

/**
 * PURE DIAGNOSTIC, no GPU involved — narrowing down the #40 step 1b failure recorded in
 * {@link GpuAdjointOnRealJacobianTest}. The GPU transposed solve is already proven at 1e-16 on
 * synthetic matrices, so the fault must be in how we read OLF's Jacobian out of powsybl-math.
 *
 * <p>Reading {@code SparseMatrix} raised two concrete suspects that theory cannot settle:
 * <ol>
 *   <li>{@code columnStart} is initialised to -1 and only written for NON-EMPTY columns, so an empty
 *       column leaves a -1 in the middle of the pointer array — which read as CSR is a catastrophic
 *       row range, not a small error.</li>
 *   <li>{@code add()} coalesces a repeated {@code (row,col)} only against the IMMEDIATELY PRECEDING
 *       entry, so a column visited as rows 3,5,3 stores row 3 TWICE. Duplicates must be summed by
 *       whoever consumes the arrays; KLU and cuDSS need not agree on that.</li>
 * </ol>
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuJacobianExtractionDiagnosticTest {


    private record Sys(EquationSystem<AcVariableType, AcEquationType> es, LfNetwork network, int n) { }

    private static Sys build() {
        Network iidm = IeeeCdfNetworkFactory.create14();
        LfNetworkParameters params = new LfNetworkParameters().setSlackBusSelector(new FirstSlackBusSelector());
        LfNetwork network = Networks.load(iidm, params).get(0);
        EquationSystem<AcVariableType, AcEquationType> es = new AcEquationSystemCreator(network).create();
        AcSolverUtil.initStateVector(network, es, new UniformValueVoltageInitializer());
        return new Sys(es, network, es.getIndex().getSortedVariablesToFind().size());
    }

    private static void converge(Sys s) {
        AcTargetVector target = new AcTargetVector(s.network(), s.es());
        EquationVector<AcVariableType, AcEquationType> eq = new EquationVector<>(s.es());
        StateVector sv = s.es().getStateVector();
        try (JacobianMatrix<AcVariableType, AcEquationType> j =
                     new JacobianMatrix<>(s.es(), new SparseMatrixFactory())) {
            eq.minus(target);
            for (int it = 0; it < 30 && maxAbs(eq.getArray()) >= 1e-12; it++) {
                double[] dx = eq.getArray().clone();
                j.solveTransposed(dx);
                sv.minus(dx);
                eq.minus(target);
            }
            System.out.printf("converged residual = %.3e%n", maxAbs(eq.getArray()));
        }
    }

    /** Q1: does the extracted CSC actually contain -1 pointers or duplicate (row,col) entries? */
    @Test
    void inspectRawCscStructure() {
        Sys s = build();
        converge(s);
        try (JacobianMatrix<AcVariableType, AcEquationType> jac = new JacobianMatrix<>(s.es(), new SparseMatrixFactory())) {
            SparseMatrix sm = (SparseMatrix) jac.getMatrix();
            int[] cs = sm.getColumnStart();
            int n = s.n();
            int nnz = cs[n];
            System.out.printf("n=%d  columnStart.length=%d  nnz(cs[n])=%d  values.length=%d  rowIndices.length=%d%n",
                    n, cs.length, nnz, sm.getValues().length, sm.getRowIndices().length);

            int negatives = 0;
            int nonMonotonic = 0;
            for (int c = 0; c <= n; c++) {
                if (cs[c] < 0) {
                    negatives++;
                    System.out.printf("  columnStart[%d] = %d  <-- EMPTY COLUMN SENTINEL%n", c, cs[c]);
                }
            }
            for (int c = 0; c < n; c++) {
                if (cs[c] >= 0 && cs[c + 1] >= 0 && cs[c + 1] < cs[c]) {
                    nonMonotonic++;
                }
            }
            System.out.printf("columnStart: %d negative, %d non-monotonic%n", negatives, nonMonotonic);

            // duplicates within a column
            int[] ri = sm.getRowIndices();
            int dupPairs = 0;
            int colsWithDup = 0;
            for (int c = 0; c < n; c++) {
                if (cs[c] < 0 || cs[c + 1] < 0) {
                    continue;
                }
                boolean any = false;
                for (int a = cs[c]; a < cs[c + 1]; a++) {
                    for (int b = a + 1; b < cs[c + 1]; b++) {
                        if (ri[a] == ri[b]) {
                            dupPairs++;
                            any = true;
                        }
                    }
                }
                if (any) {
                    colsWithDup++;
                }
            }
            System.out.printf("DUPLICATES: %d duplicate (row,col) pairs across %d columns%n", dupPairs, colsWithDup);

            // sortedness within a column (harmless for us, but tells us what KLU is handed)
            int unsortedCols = 0;
            for (int c = 0; c < n; c++) {
                if (cs[c] < 0 || cs[c + 1] < 0) {
                    continue;
                }
                for (int a = cs[c] + 1; a < cs[c + 1]; a++) {
                    if (ri[a] < ri[a - 1]) {
                        unsortedCols++;
                        break;
                    }
                }
            }
            System.out.printf("UNSORTED: %d columns have row indices out of order%n", unsortedCols);
        }
    }

    /**
     * Q2: does the extracted matrix CHANGE between a flat start and convergence? The #40 note records
     * bit-identical GPU results across both, which would mean a stale extraction. This checks the
     * SOURCE arrays directly, so it separates "extraction is stale" from "GPU consumes it wrongly".
     */
    @Test
    void jacobianValuesMustChangeWithState() {
        Sys flat = build();
        double[] atFlat = extractValues(flat);
        Sys conv = build();
        converge(conv);
        double[] atConv = extractValues(conv);

        double worst = 0;
        int differing = 0;
        for (int i = 0; i < Math.min(atFlat.length, atConv.length); i++) {
            double d = Math.abs(atFlat[i] - atConv[i]);
            if (d > 0) {
                differing++;
            }
            worst = Math.max(worst, d);
        }
        System.out.printf("JACOBIAN VS STATE: len flat=%d conv=%d, %d entries differ, max |dv| = %.3e%n",
                atFlat.length, atConv.length, differing, worst);
    }

    private static double[] extractValues(Sys s) {
        try (JacobianMatrix<AcVariableType, AcEquationType> jac = new JacobianMatrix<>(s.es(), new SparseMatrixFactory())) {
            SparseMatrix sm = (SparseMatrix) jac.getMatrix();
            return java.util.Arrays.copyOf(sm.getValues(), sm.getColumnStart()[s.n()]);
        }
    }

    /**
     * Q3: the decisive one. Build a DENSE matrix from the CSC arrays by SUMMING duplicates, solve
     * {@code M x = b} densely, and compare with OLF's own {@code solve()}. If they agree, our reading
     * of the arrays is right and KLU sums duplicates too — which would move the blame back to the GPU
     * consumer. If they disagree, the extraction itself is what is broken.
     */
    @Test
    void denseReferenceFromCscMustMatchOlfSolve() {
        Sys s = build();
        converge(s);
        int n = s.n();
        double[] b = new double[n];
        for (int i = 0; i < n; i++) {
            b[i] = Math.sin(i * 0.7) + 0.5;
        }
        try (JacobianMatrix<AcVariableType, AcEquationType> jac = new JacobianMatrix<>(s.es(), new SparseMatrixFactory())) {
            SparseMatrix sm = (SparseMatrix) jac.getMatrix();
            int[] cs = sm.getColumnStart();
            int[] ri = sm.getRowIndices();
            double[] vs = sm.getValues();
            int nnz = cs[n];

            double[][] summed = new double[n][n];
            double[][] lastWins = new double[n][n];
            for (int c = 0; c < n; c++) {
                for (int k = cs[c]; k < cs[c + 1]; k++) {
                    summed[ri[k]][c] += vs[k];
                    lastWins[ri[k]][c] = vs[k];
                }
            }
            double dupEffect = 0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    dupEffect = Math.max(dupEffect, Math.abs(summed[i][j] - lastWins[i][j]));
                }
            }

            double[] cpu = b.clone();
            jac.solve(cpu);                                  // M⁻¹ b
            double[] refSummed = denseSolve(n, summed, b);
            double[] refLast = denseSolve(n, lastWins, b);

            System.out.printf("DENSE-VS-KLU n=%d nnz=%d ||x||inf=%.3e%n", n, nnz, maxAbs(cpu));
            System.out.printf("  duplicate effect on entries : %.3e%n", dupEffect);
            System.out.printf("  |dense(summed)   - solve()| : %.3e%n", maxAbsDiff(refSummed, cpu));
            System.out.printf("  |dense(last-wins)- solve()| : %.3e%n", maxAbsDiff(refLast, cpu));
        }
    }

    /**
     * Q4: with extraction cleared, feed the GPU two ROUTES to the SAME mathematical solve
     * {@code M x = b} and see whether they agree with each other and with the dense reference:
     * <ul>
     *   <li>RAW  — hand it powsybl's CSC arrays read as CSR (that reading is {@code Mᵀ}), let its
     *       internal transpose bring it back to {@code M}.</li>
     *   <li>DENSE — build true CSR of {@code Mᵀ} myself from the dense matrix, same call.</li>
     * </ul>
     * Both must equal {@code refSummed}. If RAW and DENSE agree with each other but both miss the
     * reference, cuDSS is mis-solving this matrix (pivoting) and the extraction debate is over. If
     * they disagree, the raw arrays are not the matrix we believe they are.
     */
    @Test
    void gpuRoutesToTheSameSolveMustAgree() {
        org.junit.jupiter.api.Assumptions.assumeTrue(GpuAcNewtonSolver.isAvailable(), "no GPU lib");
        Sys s = build();
        converge(s);
        int n = s.n();
        double[] b = new double[n];
        for (int i = 0; i < n; i++) {
            b[i] = Math.sin(i * 0.7) + 0.5;
        }
        try (JacobianMatrix<AcVariableType, AcEquationType> jac = new JacobianMatrix<>(s.es(), new SparseMatrixFactory())) {
            SparseMatrix sm = (SparseMatrix) jac.getMatrix();
            int[] cs = sm.getColumnStart();
            int[] ri = sm.getRowIndices();
            double[] vs = sm.getValues();
            int nnz = cs[n];

            double[][] mDense = new double[n][n];
            for (int c = 0; c < n; c++) {
                for (int k = cs[c]; k < cs[c + 1]; k++) {
                    mDense[ri[k]][c] += vs[k];
                }
            }
            double[] ref = denseSolve(n, mDense, b);

            // Why a fill-reducing reordering can wreck this solve: cuDSS is given no pivoting, so it
            // relies on the diagonal. Count how weak that diagonal is BEFORE any reordering.
            int zeroDiag = 0;
            double minAbsDiag = Double.MAX_VALUE;
            int notDominant = 0;
            for (int i = 0; i < n; i++) {
                double d = Math.abs(mDense[i][i]);
                double off = 0;
                for (int j = 0; j < n; j++) {
                    if (j != i) {
                        off += Math.abs(mDense[i][j]);
                    }
                }
                if (d == 0) {
                    zeroDiag++;
                }
                if (d < off) {
                    notDominant++;
                }
                minAbsDiag = Math.min(minAbsDiag, d);
            }
            System.out.printf("DIAGONAL of the factorized matrix: %d exactly zero, min|diag|=%.3e, "
                    + "%d/%d rows NOT diagonally dominant%n", zeroDiag, minAbsDiag, notDominant, n);

            // RAW route: powsybl's own arrays, untouched.
            double[] gpuRaw = TransposedSolveNative.solveTransposedExplicit(n,
                    java.util.Arrays.copyOf(cs, n + 1),
                    java.util.Arrays.copyOf(ri, nnz),
                    java.util.Arrays.copyOf(vs, nnz), b.clone());

            // DENSE route: true CSR of Mᵀ, built here with no powsybl involvement.
            int[] tRowPtr = new int[n + 1];
            int cnt = 0;
            for (int r = 0; r < n; r++) {
                for (int c = 0; c < n; c++) {
                    if (mDense[c][r] != 0) {
                        cnt++;
                    }
                }
                tRowPtr[r + 1] = cnt;
            }
            int[] tColIdx = new int[cnt];
            double[] tVals = new double[cnt];
            int k = 0;
            for (int r = 0; r < n; r++) {
                for (int c = 0; c < n; c++) {
                    if (mDense[c][r] != 0) {
                        tColIdx[k] = c;
                        tVals[k] = mDense[c][r];
                        k++;
                    }
                }
            }
            double[] gpuDense = TransposedSolveNative.solveTransposedExplicit(n, tRowPtr, tColIdx, tVals, b.clone());

            System.out.printf("GPU ROUTES n=%d nnz(raw)=%d nnz(dense)=%d ||ref||inf=%.3e%n",
                    n, nnz, cnt, maxAbs(ref));
            System.out.printf("  |gpuRaw   - ref|   = %.3e   residual ||M x - b|| = %.3e%n",
                    maxAbsDiff(gpuRaw, ref), residual(n, mDense, gpuRaw, b));
            System.out.printf("  |gpuDense - ref|   = %.3e   residual ||M x - b|| = %.3e%n",
                    maxAbsDiff(gpuDense, ref), residual(n, mDense, gpuDense, b));
            System.out.printf("  |gpuRaw   - gpuDense| = %.3e%n", maxAbsDiff(gpuRaw, gpuDense));
        }
    }

    private static double residual(int n, double[][] m, double[] x, double[] b) {
        double worst = 0;
        for (int i = 0; i < n; i++) {
            double acc = 0;
            for (int j = 0; j < n; j++) {
                acc += m[i][j] * x[j];
            }
            worst = Math.max(worst, Math.abs(acc - b[i]));
        }
        return worst;
    }

    /** Dense Gaussian elimination with partial pivoting: solves {@code M x = b}. */
    private static double[] denseSolve(int n, double[][] m, double[] b) {
        double[][] a = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(m[i], 0, a[i], 0, n);
            a[i][n] = b[i];
        }
        for (int col = 0; col < n; col++) {
            int piv = col;
            for (int r = col + 1; r < n; r++) {
                if (Math.abs(a[r][col]) > Math.abs(a[piv][col])) {
                    piv = r;
                }
            }
            double[] t = a[col];
            a[col] = a[piv];
            a[piv] = t;
            for (int r = col + 1; r < n; r++) {
                double f = a[r][col] / a[col][col];
                for (int c = col; c <= n; c++) {
                    a[r][c] -= f * a[col][c];
                }
            }
        }
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double sum = a[i][n];
            for (int j = i + 1; j < n; j++) {
                sum -= a[i][j] * x[j];
            }
            x[i] = sum / a[i][i];
        }
        return x;
    }

    private static double maxAbs(double[] a) {
        double m = 0;
        for (double v : a) {
            m = Math.max(m, Math.abs(v));
        }
        return m;
    }

    private static double maxAbsDiff(double[] a, double[] b) {
        double m = 0;
        for (int i = 0; i < a.length; i++) {
            m = Math.max(m, Math.abs(a[i] - b[i]));
        }
        return m;
    }
}
