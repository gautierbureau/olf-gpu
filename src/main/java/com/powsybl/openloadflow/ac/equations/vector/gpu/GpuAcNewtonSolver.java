/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.openloadflow.ac.outerloop.gpu.ReactiveLimitsNative;
import com.powsybl.openloadflow.ac.solver.gpu.BatchSolveNative;
import com.powsybl.openloadflow.ac.solver.gpu.NewtonSettingsNative;
import com.powsybl.openloadflow.ac.solver.gpu.batch.ControlReHoming;

/**
 * JNI binding: a FULL Newton-Raphson AC power flow run on the GPU behind one native call.
 * The native library libolfgpu (native/cuda/jni/olf_gpu_nr_jni.cu, built by the root
 * CMakeLists — see native/build-gpu.sh) keeps the fixed-pattern CSR Jacobian, the
 * mismatch and the state resident on the device: the four generated fused kernels
 * (closed-branch / open-branch / shunt / HVDC AC emulation) fill J and F each
 * iteration, cuDSS refactorizes and solves, a kernel applies the state update.
 *
 * <p>Distribution: libolfgpu is OUR wrapper (MPL) and is bundled in the jar, built
 * location-independent (no RPATH); what is NOT redistributable is NVIDIA's cuDSS, which
 * the wrapper only references dynamically ({@code NEEDED libcudss.so.0}): the system loader
 * resolves it ({@code LD_LIBRARY_PATH} or ldconfig, see GpuNativeLoader). {@link #isAvailable()}
 * is false when the library or its cuDSS dependency cannot be loaded, so CUDA-free builds are
 * unaffected.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuAcNewtonSolver {

    private static final boolean LOADED = GpuNativeLoader.load();

    private GpuAcNewtonSolver() {
    }

    public static boolean isAvailable() {
        return LOADED;
    }

    /**
     * Build the SHARED STRUCTURE for one equation-system extraction: the diagonal-pairing row
     * permutation, the fixed SUPERSET CSR pattern (every row carries its full power-equation
     * entries + a guaranteed diagonal), the per-family device data with role-folded destination
     * arrays, the DISTR_Q / DISTR_RHO / zero-impedance packs and the per-element static trig.
     * This is everything a batched context derives from one {@link GpuAcData} — build it once
     * and pass the handle to {@link #createBatchContext}, so the base load flow and the batched
     * security analysis sharing one extraction share one device-resident structure (see {@link GpuAcStructures}
     * for the per-extraction cache). Rows use the STABLE indexing (the diagonal-paired variable
     * row, see GpuAcDataExtractor), so the structure survives activation switching; element
     * counts are implied by the index array lengths and -1 marks an absent branch side.
     *
     * @param n     number of variables / equations
     * @param cbIn  closed branches, {@code ncb*8}: (y g1 g2 b1 b2 ksi r1 a1)
     * @param cbIdx closed branches, {@code ncb*12}: (Prow1 Qrow1 Prow2 Qrow2 ph1col v1col ph2col v2col a1col r1col qAlt1 qAlt2)
     * @param obIn  open branches, {@code nob*7}: (y ksi g1 b1 g2 b2 r1)
     * @param obIdx open branches, {@code nob*6}: (Prow1 Qrow1 Prow2 Qrow2 v1col v2col)
     * @param shIn  shunts, {@code nsh*2}: (g b)
     * @param shIdx shunts, {@code nsh*3}: (Prow Qrow vcol)
     * @param hvIn  HVDC AC emulation, {@code nhv*5}: (p0 k lossFactor1 lossFactor2 r)
     * @param hvIdx HVDC AC emulation, {@code nhv*4}: (Prow1 Prow2 ph1col ph2col)
     * @return an opaque native handle; release it with {@link #destroyStructure}
     */
    /** No dependent-pilot group for any candidate group (the raw-row entry points have no merged controls). */
    private static int[] noDepGroups(int[] qrPtr) {
        int[] a = new int[Math.max(0, qrPtr.length - 1)];
        java.util.Arrays.fill(a, -1);
        return a;
    }

    /** No dependent-pilot group for any device row. */
    private static int[] noDepRows(int n) {
        int[] a = new int[n];
        java.util.Arrays.fill(a, -1);
        return a;
    }

    private static int[] noQAlt(int ne) {
        int[] a = new int[ne * 2];
        java.util.Arrays.fill(a, -1);
        return a;
    }

    private static int[] noPcSide(int ncb) {
        int[] a = new int[ncb];
        java.util.Arrays.fill(a, -1);
        return a;
    }

    /**
     * One-shot undamped pure Newton-Raphson: {@link #createStructure} + {@link #solveOnStructure}, then the
     * structure is destroyed.
     *
     * @return {@code double[n + 3]}: the state, then iterations, the final mismatch infinity norm and the
     *         convergence status (see {@link #solveBatchContext})
     */
    public static double[] solve(double[] x0, double[] target, int[] rowMode, int maxIter, double tol,
                                 double[] cbIn, int[] cbIdx, double[] obIn, int[] obIdx,
                                 double[] shIn, int[] shIdx, double[] hvIn, int[] hvIdx,
                                 int[] dqElem, int[] dqSide, double[] dqWeight, int[] dqRow, int[] dqKind,
                                 double[] ziIn, int[] ziIdx,
                                 int[] drRow, int[] drCol, double[] drCoef,
                                 int[] dbRow, int[] dbCol, double[] dbCoef) {
        return solve(x0, target, rowMode, maxIter, tol, cbIn, cbIdx, obIn, obIdx, shIn, shIdx, hvIn, hvIdx,
                dqElem, dqSide, dqWeight, dqRow, dqKind, ziIn, ziIdx, drRow, drCol, drCoef, dbRow, dbCol, dbCoef,
                new int[] {0}, new int[0], noQAlt(obIdx.length / 6), noQAlt(ziIdx.length / 11));
    }

    /** {@link #solve} with the GROUP pilot-Q relocation tables (plan-A6) and the ob/zi alternate
     *  Q rows — required whenever the index packs carry qAlt encodings (shared groups / remote
     *  members). Pass all-(-1) ob/zi alt arrays when only cb/sh relocate. */
    @SuppressWarnings("java:S107")
    public static double[] solve(double[] x0, double[] target, int[] rowMode, int maxIter, double tol,
                                 double[] cbIn, int[] cbIdx, double[] obIn, int[] obIdx,
                                 double[] shIn, int[] shIdx, double[] hvIn, int[] hvIdx,
                                 int[] dqElem, int[] dqSide, double[] dqWeight, int[] dqRow, int[] dqKind,
                                 double[] ziIn, int[] ziIdx,
                                 int[] drRow, int[] drCol, double[] drCoef,
                                 int[] dbRow, int[] dbCol, double[] dbCoef, int[] qrPtr, int[] qrRows,
                                 int[] obQAlt, int[] ziQAlt) {
        long structure = StructureNative.createStructure(x0.length, cbIn, cbIdx, obIn, obIdx, shIn, shIdx, hvIn, hvIdx,
                dqElem, dqSide, dqWeight, dqRow, dqKind, ziIn, ziIdx, drRow, drCol, drCoef, dbRow, dbCol, dbCoef,
                noPcSide(cbIdx.length / 12), new double[0], new int[0], qrPtr, qrRows, obQAlt, ziQAlt,
                noDepGroups(qrPtr), noDepGroups(qrPtr), noDepRows(x0.length));
        try {
            return solveOnStructure(structure, x0, target, rowMode, maxIter, tol, 0, 0.0, 0.0, new int[x0.length]);
        } finally {
            StructureNative.destroyStructure(structure);
        }
    }

    /**
     * One-shot pure Newton-Raphson on an existing structure: a batch-of-one {@link #createBatchContext}
     * without any outer-loop pack, one {@link #solveBatchContext}, then the context is destroyed.
     *
     * @param lsMode  device step scaling: 0 = none, 1 = line search, 2 = max voltage change
     * @param maxDv   MAX_VOLTAGE_CHANGE (lsMode 2): clamp on |dv| per iteration (BUS_V rows)
     * @param maxDphi MAX_VOLTAGE_CHANGE (lsMode 2): clamp on |dphi| per iteration (BUS_PHI rows, radians)
     * @param varType per row, the variable kind for the clamp, see GpuAcActivationExtractor#extractVarType
     * @return {@code double[n + 3]}: the state, then iterations, the final mismatch infinity norm and the
     *         convergence status (see {@link #solveBatchContext})
     */
    @SuppressWarnings("java:S107")
    public static double[] solveOnStructure(long structure, double[] x0, double[] target, int[] rowMode,
                                            int maxIter, double tol, int lsMode, double maxDv, double maxDphi,
                                            int[] varType) {
        long handle = BatchSolveNative.createBatchContext(structure, 1, com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchContextCache.MODE_UBATCH,
                new int[0], new double[0], new double[0], new double[0], new double[0], new byte[0], new double[x0.length],
                new int[0], new double[0], -1, 0.0, new double[0], new double[0], new double[0],
                0, new double[0], new double[0], 0, 0);
        try {
            return BatchSolveNative.solveBatchContext(handle, x0, target, new int[][] {rowMode}, rowMode, new boolean[1][0],
                    new int[0], new double[0], new int[0], new double[0], new double[0], new double[0], new double[0], null, null, null, null, null, null, null, null,
                    maxIter, tol, 0.0, 0.0, 0.0, lsMode, maxDv, maxDphi, varType, null, null, null, null, null, null, null, 0,
                    com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop.MAX_SWITCH_PQ_PV_DEFAULT_VALUE,
                    0, 0, 0, null, null)[0];
        } finally {
            BatchSolveNative.destroyBatchContext(handle);
        }
    }

    /**
     * Upload the run's limit thresholds and the branch model they are compared against, once per
     * run: a contingency cannot change a threshold, so this outlives every chunk. Returns 0 when the
     * upload failed (the caller then screens on the host).
     *
     * <p>Groups and limits are flat, in the host table's order, so a record is simply the limit's
     * index. See {@code GpuViolationScreen} for the layout and {@code olf_gpu_violations.cu} for
     * why the kernel screens rather than decides.
     */
    public static native long uploadViolationTable(
            int[] branchFamily, int[] branchElem, int[] branchGroupStart, int[] branchGroupEnd,
            int[] groupSide, int[] groupQuantity, int[] groupLimitStart, int[] groupLimitEnd,
            double[] limitThreshold,
            int[] busVRow, double[] busHighLimit, double[] busLowLimit,
            double[] cbIn, int[] cbIdx, double[] obIn, int[] obIdx);

    /**
     * Screen a chunk of post-contingency states against an uploaded table.
     *
     * @param states {@code nScenarios * n} states, scenario-major
     * @param band relative slack on every threshold, so the screen can only over-emit
     * @param capacity records per scenario
     * @return {@code nScenarios} counts, then {@code nScenarios * capacity} records; a count ABOVE
     *         the capacity means that scenario overflowed and must be scanned on the host
     */
    public static native int[] screenViolationsBatch(long tableHandle, double[] states, int nScenarios,
                                                     int n, double band, int capacity);

    /** Release a violation table (idempotent for handle 0). */
    public static native void destroyViolationTable(long tableHandle);

    /** Installs the four arrays of GpuAcActivationExtractor.extractRowCriteria, or clears every threshold. */
    public static void installRowCriteria(long handle, double[][] rowCriteria) {
        NewtonSettingsNative.setBatchRowCriteria(handle, rowCriteria != null ? rowCriteria[0] : null, rowCriteria != null ? rowCriteria[1] : null);
        if (rowCriteria != null && rowCriteria.length >= 4) {
            int[] sel = new int[rowCriteria[3].length];
            for (int i = 0; i < sel.length; i++) {
                sel[i] = (int) rowCriteria[3][i];
            }
            NewtonSettingsNative.setBatchRowCriteriaAlt(handle, rowCriteria[2], sel);
        }
    }

    /**
     * Solves {@code Aᵀ x = b} for an explicit CSR matrix {@code A}, by building the transpose and
     * factorizing it. Validation / measurement entry for the batched-adjoint work (docs/tasks.md #40
     * step 1a) — NOT the production path.
     *
     * <p>cuDSS through 0.8.0 has no transpose solve (verified: its header carries no transpose/OP_T
     * symbol), so one GPU factorization yields ONE direction, where KLU's {@code klu_solve} /
     * {@code klu_tsolve} pair yields two from the same factors. Our Jacobian is stored as TRUE CSR
     * (row = equation, col = variable) so the plain solve serves the forward Newton; the adjoint
     * ({@code Jᵀλ = …}) is therefore the direction the forward's factors cannot provide, and until
     * cuDSS ships a transpose solve we buy it with a second factorization of an explicit transpose.
     *
     * <p>Deliberately self-contained (own handle/config/data, no Newton context) so a failure here
     * can only mean the CSR transpose or the cuDSS factorize/solve of {@code Aᵀ} is wrong — never the
     * adjoint's RHS assembly or contraction.
     *
     * @param n       matrix dimension
     * @param rowPtr  CSR row pointers of {@code A}, length {@code n+1}
     * @param colIdx  CSR column indices of {@code A}, length {@code rowPtr[n]}
     * @param vals    CSR values of {@code A}, length {@code rowPtr[n]}
     * @param b       right-hand side, length {@code n}
     * @return {@code x} such that {@code Aᵀ x = b}
     */
    public static native double[] solveTransposedExplicit(int n, int[] rowPtr, int[] colIdx,
                                                          double[] vals, double[] b);

    /**
     * One-shot batch convenience: {@link #createStructure} + {@link #createBatchContext} +
     * {@link #solveBatchContext} + destroy both. For repeated batches on the same structure,
     * cache the context instead. The base state/targets/row-mode are shared by every scenario
     * (tiled on-device); {@code nScenarios} is {@code disabledElementMask.length}.
     */
    public static double[][] solveBatch(double[] x0Base, double[] targetBase, int[][] rowMode,
                                         boolean[][] disabledElementMask, int mode, int maxIter, double tol,
                                         double reactiveLimitTol, double slackDistTol,
                                         double[] cbIn, int[] cbIdx, double[] obIn, int[] obIdx,
                                         double[] shIn, int[] shIdx, double[] hvIn, int[] hvIdx,
                                         int[] rlVrow, double[] rlMinQ, double[] rlMaxQ, double[] rlLoadQ,
                                         double[] rlVtarget, int[] dsPhiRow, double[] dsFactor,
                                         int dsSlackPhiRow, double dsSlackTargetP,
                                         double[] dsBaseTargetP, double[] dsMaxDelta, double[] dsMinDelta,
                                         int[] dqElem, int[] dqSide, double[] dqWeight, int[] dqRow, int[] dqKind,
                                         int[] drRow, int[] drCol, double[] drCoef,
                                         int[] dbRow, int[] dbCol, double[] dbCoef, double[] ziIn, int[] ziIdx) {
        int n = x0Base.length;
        int nScenarios = disabledElementMask.length;
        long structure = StructureNative.createStructure(n, cbIn, cbIdx, obIn, obIdx, shIn, shIdx, hvIn, hvIdx,
                dqElem, dqSide, dqWeight, dqRow, dqKind, ziIn, ziIdx, drRow, drCol, drCoef, dbRow, dbCol, dbCoef,
                noPcSide(cbIdx.length / 12), new double[0], new int[0], new int[] {0}, new int[0],
                noQAlt(obIdx.length / 6), noQAlt(ziIdx.length / 11), new int[0], new int[0], noDepRows(n));
        try {
            long handle = BatchSolveNative.createBatchContext(structure, nScenarios, mode,
                    rlVrow, rlMinQ, rlMaxQ, rlLoadQ, rlVtarget, new byte[rlVrow.length], new double[x0Base.length],
                    dsPhiRow, dsFactor, dsSlackPhiRow, dsSlackTargetP,
                    dsBaseTargetP, dsMaxDelta, dsMinDelta, 0, new double[0], new double[0], 0, 0);
            try {
                // OLF's forced-PV election keys. This one-shot entry point has no network behind it
                // (the caller hands raw rows), so the keys are NEUTRAL: equal nominal voltage, no
                // live target-P row, and the entry order as the id rank — which reduces the rule to
                // OLF's last tie-break, "lowest id stays PV". A context that reaches a solve without
                // any keys is refused, and the rule can only fire at all when every PV bus is
                // pinnable, so this keeps the low-level entry usable without inventing a ranking.
                int[] idRank = new int[rlVrow.length];
                int[] pRow = new int[rlVrow.length];
                for (int i = 0; i < idRank.length; i++) {
                    idRank[i] = i;
                    pRow[i] = -1;
                }
                ReactiveLimitsNative.setBatchRlElection(handle, new double[rlVrow.length], pRow, new double[rlVrow.length], idRank);
                // one-shot: no per-scenario gen-override flips, so any scenario's row mode IS the base row mode
                return BatchSolveNative.solveBatchContext(handle, x0Base, targetBase, rowMode, rowMode[0], disabledElementMask,
                        new int[0], new double[0], new int[0], new double[0], new double[0], new double[0], new double[0], null, null, null, null, null, null, null, null,
                        maxIter, tol, reactiveLimitTol, slackDistTol, 0.0, 0, 0.0, 0.0, null, null, null, null, null, null, null, null, 0,
                        com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop.MAX_SWITCH_PQ_PV_DEFAULT_VALUE,   // OLF's cap: 0 means never switch back
                        0, 0, 0, null, null);
            } finally {
                BatchSolveNative.destroyBatchContext(handle);
            }
        } finally {
            StructureNative.destroyStructure(structure);
        }
    }
}
