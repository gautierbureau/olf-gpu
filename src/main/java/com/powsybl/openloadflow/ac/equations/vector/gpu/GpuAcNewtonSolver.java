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
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuDistributedSlackFactorMode;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuLinearSolveMode;

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

    private static int[] noPcSide(int nClosedBranch) {
        int[] a = new int[nClosedBranch];
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
                                 double[] closedBranchParameters, int[] closedBranchIndices, double[] openBranchParameters, int[] openBranchIndices,
                                 double[] shuntParameters, int[] shuntIndices, double[] hvdcAcEmulationParameters, int[] hvdcAcEmulationIndices,
                                 int[] dqElem, int[] dqSide, double[] dqWeight, int[] dqRow, int[] dqKind,
                                 double[] ziIn, int[] ziIdx,
                                 int[] distrRhoRow, int[] distrRhoColumn, double[] distrRhoCoefficient,
                                 int[] distrShuntBRow, int[] distrShuntBColumn, double[] distrShuntBCoefficient) {
        return solve(x0, target, rowMode, maxIter, tol, closedBranchParameters, closedBranchIndices,
                openBranchParameters, openBranchIndices, shuntParameters, shuntIndices, hvdcAcEmulationParameters,
                hvdcAcEmulationIndices,
                dqElem, dqSide, dqWeight, dqRow, dqKind, ziIn, ziIdx, distrRhoRow, distrRhoColumn, distrRhoCoefficient, distrShuntBRow, distrShuntBColumn, distrShuntBCoefficient,
                new int[] {0}, new int[0], noQAlt(openBranchIndices.length / 6), noQAlt(ziIdx.length / 11));
    }

    /** {@link #solve} with the GROUP pilot-Q relocation tables (plan-A6) and the ob/zi alternate
     *  Q rows — required whenever the index packs carry qAlt encodings (shared groups / remote
     *  members). Pass all-(-1) ob/zi alt arrays when only cb/sh relocate. */
    @SuppressWarnings("java:S107")
    public static double[] solve(double[] x0, double[] target, int[] rowMode, int maxIter, double tol,
                                 double[] closedBranchParameters, int[] closedBranchIndices, double[] openBranchParameters, int[] openBranchIndices,
                                 double[] shuntParameters, int[] shuntIndices, double[] hvdcAcEmulationParameters, int[] hvdcAcEmulationIndices,
                                 int[] dqElem, int[] dqSide, double[] dqWeight, int[] dqRow, int[] dqKind,
                                 double[] ziIn, int[] ziIdx,
                                 int[] distrRhoRow, int[] distrRhoColumn, double[] distrRhoCoefficient,
                                 int[] distrShuntBRow, int[] distrShuntBColumn, double[] distrShuntBCoefficient, int[] qrPtr, int[] qrRows,
                                 int[] openBranchAlternateReactiveDestination, int[] ziQAlt) {
        long structure = StructureNative.createStructure(x0.length, closedBranchParameters, closedBranchIndices,
                openBranchParameters, openBranchIndices, shuntParameters, shuntIndices, hvdcAcEmulationParameters,
                hvdcAcEmulationIndices,
                dqElem, dqSide, dqWeight, dqRow, dqKind, ziIn, ziIdx, distrRhoRow, distrRhoColumn, distrRhoCoefficient, distrShuntBRow, distrShuntBColumn, distrShuntBCoefficient,
                noPcSide(closedBranchIndices.length / 12), new double[0], new int[0], qrPtr, qrRows, openBranchAlternateReactiveDestination, ziQAlt,
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
        long handle = BatchSolveNative.createBatchContext(structure, 1, GpuLinearSolveMode.UBATCH,
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
     * One-shot batch convenience: {@link #createStructure} + {@link #createBatchContext} +
     * {@link #solveBatchContext} + destroy both. For repeated batches on the same structure,
     * cache the context instead. The base state/targets/row-mode are shared by every scenario
     * (tiled on-device); {@code nScenarios} is {@code disabledElementMask.length}.
     */
    public static double[][] solveBatch(double[] x0Base, double[] targetBase, int[][] rowMode,
                                         boolean[][] disabledElementMask, int mode, int maxIter, double tol,
                                         double reactiveLimitTol, double slackDistTol,
                                         double[] closedBranchParameters, int[] closedBranchIndices, double[] openBranchParameters, int[] openBranchIndices,
                                         double[] shuntParameters, int[] shuntIndices, double[] hvdcAcEmulationParameters, int[] hvdcAcEmulationIndices,
                                         int[] rlVrow, double[] rlMinQ, double[] rlMaxQ, double[] rlLoadQ,
                                         double[] rlVtarget, int[] dsPhiRow, double[] dsFactor,
                                         int dsSlackPhiRow, double dsSlackTargetP,
                                         double[] dsBaseTargetP, double[] dsMaxDelta, double[] dsMinDelta,
                                         int[] dqElem, int[] dqSide, double[] dqWeight, int[] dqRow, int[] dqKind,
                                         int[] distrRhoRow, int[] distrRhoColumn, double[] distrRhoCoefficient,
                                         int[] distrShuntBRow, int[] distrShuntBColumn, double[] distrShuntBCoefficient, double[] ziIn, int[] ziIdx) {
        int n = x0Base.length;
        int nScenarios = disabledElementMask.length;
        long structure = StructureNative.createStructure(n, closedBranchParameters, closedBranchIndices,
                openBranchParameters, openBranchIndices, shuntParameters, shuntIndices, hvdcAcEmulationParameters,
                hvdcAcEmulationIndices,
                dqElem, dqSide, dqWeight, dqRow, dqKind, ziIn, ziIdx, distrRhoRow, distrRhoColumn, distrRhoCoefficient, distrShuntBRow, distrShuntBColumn, distrShuntBCoefficient,
                noPcSide(closedBranchIndices.length / 12), new double[0], new int[0], new int[] {0}, new int[0],
                noQAlt(openBranchIndices.length / 6), noQAlt(ziIdx.length / 11), new int[0], new int[0], noDepRows(n));
        try {
            long handle = BatchSolveNative.createBatchContext(structure, nScenarios, mode,
                    rlVrow, rlMinQ, rlMaxQ, rlLoadQ, rlVtarget, new byte[rlVrow.length], new double[x0Base.length],
                    dsPhiRow, dsFactor, dsSlackPhiRow, dsSlackTargetP,
                    dsBaseTargetP, dsMaxDelta, dsMinDelta, GpuDistributedSlackFactorMode.STATIC, new double[0], new double[0], 0, 0);
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
