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
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.EquationVector;
import com.powsybl.openloadflow.equations.JacobianMatrix;
import com.powsybl.openloadflow.network.FirstSlackBusSelector;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfNetworkParameters;
import com.powsybl.openloadflow.network.impl.Networks;
import com.powsybl.openloadflow.network.util.UniformValueVoltageInitializer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Step 1b (first half) of the batched-adjoint work (docs/tasks.md #40): take the GPU transposed
 * solve proven on synthetic matrices in {@link GpuTransposedSolveTest} and exercise it on a REAL
 * OLF Jacobian, against OLF's own KLU solve.
 *
 * <p>This is the test that pins the CPU/GPU orientation claim empirically rather than by reasoning.
 * The claim under test: powsybl-math stores {@code M = Jᵀ} (its CSC-as-CSR convention), so its
 * plain {@code solve()} IS the adjoint direction {@code J⁻ᵀ}; our GPU builds true CSR of {@code J},
 * so it needs an explicit transpose to reach the same direction. If that is right, feeding the
 * stored arrays READ AS CSR into {@code solveTransposedExplicit} must reproduce ONE of
 * {@code solve()} / {@code solveTransposed()} — and which one it reproduces settles the convention
 * as a measurement, not an argument.
 *
 * <p>Deliberately reports both comparisons before asserting, so a future convention change shows up
 * as a readable diagnostic instead of a bare failure.
 *
 * <p>NOTE this does NOT yet cover the adjoint's RHS assembly {@code (∂y/∂x)ᵀȳ} or the contraction
 * {@code θ̄ = (∂y/∂p)ᵀȳ − (∂F/∂p)ᵀλ} — those are the second half of 1b and amount to porting
 * {@code analyseAdjoint}. What is retired here is the solve+orientation on real data.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuAdjointOnRealJacobianTest {

    static {
        GpuTestPaths.init();
    }

    @BeforeEach
    void checkGpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(),
                "GPU full-NR native lib not available (build it with native/build-gpu.sh + cuDSS)");
    }

    @Test
    void gpuTransposedSolveMatchesOlfSolveOnARealJacobian() {
        // RESOLVED (was failing at 7.164e-03). The cause was NOT the extraction and NOT the transpose:
        // it was cuDSS PIVOTING. GpuJacobianExtractionDiagnosticTest measured every step —
        //   * the CSC arrays are clean: no -1 pointers, no duplicate (row,col), no unsorted columns;
        //   * the matrix DOES track the state (170 of 180 entries move between flat start and
        //     convergence), so the earlier "stale extraction" reading was wrong — the bit-identical
        //     numbers came from the reordering being state-independent, not the matrix;
        //   * a dense solve built from those very arrays reproduces OLF's solve() to 6.217e-15,
        //     which proves the extraction AND the orientation claim outright.
        // What was actually broken: the factorized matrix has 5 EXACTLY ZERO diagonal entries and
        // 0 of 28 diagonally dominant rows, while applyReorderAlg forces AMD with no pivoting. cuDSS
        // substitutes a static perturbation for the zero pivot and returns a wrong x WITH A MATCHING
        // RESIDUAL — which is why this masqueraded as a correctness bug. Measured on this matrix:
        //   AMD 7.164e-03 | nested dissection 7.179e-02 | natural order 6.217e-15
        //   iterative refinement changes NOTHING (IR=0 and IR=3 are bit-identical)
        //   AMD + MC64-style matching (CUDSS_CONFIG_MATCHING_ALG) 3.553e-15   <-- the fix
        // The batched NR never hit this because it builds a superset pattern with a GUARANTEED
        // nonzero diagonal; a raw Jacobian handed over unpermuted has no such guarantee. See
        // applyAdjointMatching in olf_gpu_nr_jni.cu.
        // ieee14 with default (no transformer/phase control) so the system is plain AC power flow —
        // the shape #40 step 1 asks for: no binding limits, nothing to confuse a mismatch with.
        Network iidm = IeeeCdfNetworkFactory.create14();
        LfNetworkParameters params = new LfNetworkParameters().setSlackBusSelector(new FirstSlackBusSelector());
        LfNetwork network = Networks.load(iidm, params).get(0);
        EquationSystem<AcVariableType, AcEquationType> es = new AcEquationSystemCreator(network).create();
        AcSolverUtil.initStateVector(network, es, new UniformValueVoltageInitializer());
        AcTargetVector target = new AcTargetVector(network, es);
        int n = es.getIndex().getSortedVariablesToFind().size();

        // CONVERGE FIRST. The adjoint is only ever taken at a solution, and the AC Jacobian at a FLAT
        // start is badly conditioned — evaluated there, cuDSS and KLU legitimately disagree at ~1e-3
        // with a matching residual, which looks like a solver bug and is not one. (Measured: residual
        // 1.9e-3 at flat start vs ~1e-16 after convergence.) Plain Newton, as the other GPU tests do.
        EquationVector<AcVariableType, AcEquationType> eq = new EquationVector<>(es);
        com.powsybl.openloadflow.equations.StateVector sv = es.getStateVector();
        try (JacobianMatrix<AcVariableType, AcEquationType> jNewton =
                     new JacobianMatrix<>(es, new SparseMatrixFactory())) {
            eq.minus(target);
            for (int it = 0; it < 30 && maxAbs(eq.getArray()) >= 1e-12; it++) {
                double[] dx = eq.getArray().clone();
                jNewton.solveTransposed(dx);                 // forward Newton (stored matrix is Jᵀ)
                sv.minus(dx);                                // StateVector.minus, NOT get()/set():
                eq.minus(target);                            // get() copies, so manual writeback is inert
            }
            assertTrue(maxAbs(eq.getArray()) < 1e-8, "the Newton must actually converge before the "
                    + "adjoint is taken — an unconverged (flat-start) AC Jacobian is badly conditioned "
                    + "and makes cuDSS and KLU disagree at ~1e-3 for reasons that are not a solver bug");
        }

        double[] b = new double[n];
        for (int i = 0; i < n; i++) {
            b[i] = Math.sin(i * 0.7) + 0.5;                  // deterministic, non-trivial RHS
        }

        try (JacobianMatrix<AcVariableType, AcEquationType> jac = new JacobianMatrix<>(es, new SparseMatrixFactory())) {
            SparseMatrix sm = (SparseMatrix) jac.getMatrix();
            // powsybl-math SparseMatrix exposes CSC arrays. Read them AS CSR (rowPtr = columnStart,
            // colIdx = rowIndices): that reinterpretation is itself the transpose, which is exactly
            // the "CSC-as-CSR" convention the GPU JNI header contrasts our true-CSR layout against.
            // TWO traps here, both measured the hard way:
            //
            // 1. The arrays are OVER-ALLOCATED: values.length was 336 while the real nnz is
            //    cscPtr[n] = 180. Taking nnz from the array length feeds 156 stale entries into the
            //    matrix — that, not any transpose error, was the 7.2e-3 residual first observed.
            //    Always take nnz from the pointer array, never from values.length.
            // 2. powsybl-math stores CSC, so reading (columnStart, rowIndices, values) AS CSR is
            //    ALREADY one transpose. That reinterpretation is the "CSC-as-CSR" convention, and it
            //    is what KLU factorizes — which is exactly why OLF's plain solve() is the ADJOINT
            //    direction. So we hand these arrays to the GPU unchanged and let its single transpose
            //    land on the same matrix KLU has. Converting CSC->CSR first would transpose twice and
            //    silently give the FORWARD direction instead (measured: it flips the match from
            //    solve() to solveTransposed()).
            int[] rowPtr = sm.getColumnStart();
            assertEquals(n + 1, rowPtr.length, "square system: one pointer per column plus the tail");
            int nnz = rowPtr[n];
            int[] colIdx = java.util.Arrays.copyOf(sm.getRowIndices(), nnz);
            double[] vals = java.util.Arrays.copyOf(sm.getValues(), nnz);

            double[] gpu = GpuAcNewtonSolver.solveTransposedExplicit(n, rowPtr, colIdx, vals, b);

            double[] cpuSolve = b.clone();
            jac.solve(cpuSolve);                             // M⁻¹ b, and M = Jᵀ ⇒ the ADJOINT direction
            double[] cpuSolveT = b.clone();
            jac.solveTransposed(cpuSolveT);                  // M⁻ᵀ b ⇒ the FORWARD direction

            double dSolve = maxAbsDiff(gpu, cpuSolve);
            double dSolveT = maxAbsDiff(gpu, cpuSolveT);
            double scale = Math.max(maxAbs(cpuSolve), 1e-300);
            // RESIDUAL is the real correctness measure: does the GPU x actually satisfy Aᵀx = b?
            // Comparing two solvers' outputs only bounds their agreement, which at large ||x|| says
            // little — the same "compare at the right scale" lesson as the stopping-criterion work.
            double residual = 0;
            for (int i = 0; i < n; i++) {
                double acc = 0;
                for (int k = 0; k < n; k++) {                // (Aᵀ x)_i = Σ over entries in ROW k of A with col i
                    for (int t = rowPtr[k]; t < rowPtr[k + 1]; t++) {
                        if (colIdx[t] == i) {
                            acc += vals[t] * gpu[k];
                        }
                    }
                }
                residual = Math.max(residual, Math.abs(acc - b[i]));
            }
            System.out.printf("REAL-JACOBIAN n=%d nnz=%d ||x||inf=%.3e | vs solve abs=%.3e rel=%.3e | vs solveT abs=%.3e | residual=%.3e%n",
                    n, nnz, scale, dSolve, dSolve / scale, dSolveT, residual);

            // Exactly one of the two must match, and it must be a clean match: if BOTH matched the
            // matrix would be symmetric and this test would prove nothing about orientation.
            assertTrue(dSolve / scale < 1e-9 || dSolveT / scale < 1e-9,
                    "the GPU transposed solve must reproduce one of OLF's two directions "
                            + "(solve=" + dSolve + ", solveTransposed=" + dSolveT + ")");
            assertTrue(Math.abs(dSolve - dSolveT) > 1e-6,
                    "OLF's two directions must DIFFER on this Jacobian, else the comparison is vacuous "
                            + "(solve=" + dSolve + ", solveTransposed=" + dSolveT + ")");
            // The orientation claim recorded in docs/tasks.md #40: reading the stored CSC arrays as
            // CSR yields J, so a transposed solve of it is J⁻ᵀ — OLF's plain solve().
            assertEquals(0, dSolve / scale, 1e-9,
                    "GPU Aᵀ-solve should equal OLF's PLAIN solve (the stored matrix is Jᵀ, so plain "
                            + "solve is already the adjoint direction). If this fails while "
                            + "solveTransposed matches instead, the CPU/GPU orientation note in "
                            + "docs/tasks.md #40 is inverted and must be corrected.");
        }
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
