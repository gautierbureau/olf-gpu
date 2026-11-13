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
import com.powsybl.openloadflow.ac.solver.gpu.DriverNative;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuDistributedSlackFactorMode;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuLinearSolveMode;

/**
 * Test support: a pure Newton-Raphson on hand-packed arrays, with no network and no equation system behind
 * them. Each call builds a structure and a batch context from the raw packs, solves once and destroys both,
 * so a test can check the device equations and the batched solve on a system it wrote by hand.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class PackedNewtonRaphson {

    private PackedNewtonRaphson() {
    }

    /** No dependent-pilot group for any candidate group (the raw-row entry points have no merged controls). */
    private static int[] noDependentBaseGroups(int[] reactiveRelocationGroupStart) {
        int[] a = new int[Math.max(0, reactiveRelocationGroupStart.length - 1)];
        java.util.Arrays.fill(a, -1);
        return a;
    }

    /** No dependent-pilot group for any device row. */
    private static int[] noDependentGroupOfRows(int n) {
        int[] a = new int[n];
        java.util.Arrays.fill(a, -1);
        return a;
    }

    private static int[] noAlternateQRows(int ne) {
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
     * One-shot undamped pure Newton-Raphson: {@link StructureNative#createStructure} + {@link #solveOnStructure}, then the
     * structure is destroyed.
     *
     * @return {@code double[n + 3]}: the state, then iterations, the final mismatch infinity norm and the
     *         convergence status (see {@link BatchSolveNative#solveBatchContext})
     */
    public static double[] solve(double[] x0, double[] target, int[] rowMode, int maxIter, double tol,
                                 double[] closedBranchParameters, int[] closedBranchIndices, double[] openBranchParameters, int[] openBranchIndices,
                                 double[] shuntParameters, int[] shuntIndices, double[] hvdcAcEmulationParameters, int[] hvdcAcEmulationIndices,
                                 int[] distrQElement, int[] distrQSide, double[] distrQWeight, int[] distrQRow, int[] distrQKind,
                                 double[] zeroImpedanceBranchParameters, int[] zeroImpedanceBranchIndices,
                                 int[] distrRhoRow, int[] distrRhoColumn, double[] distrRhoCoefficient,
                                 int[] distrShuntBRow, int[] distrShuntBColumn, double[] distrShuntBCoefficient) {
        return solve(x0, target, rowMode, maxIter, tol, closedBranchParameters, closedBranchIndices,
                openBranchParameters, openBranchIndices, shuntParameters, shuntIndices, hvdcAcEmulationParameters,
                hvdcAcEmulationIndices,
                distrQElement, distrQSide, distrQWeight, distrQRow, distrQKind, zeroImpedanceBranchParameters, zeroImpedanceBranchIndices,
                distrRhoRow, distrRhoColumn, distrRhoCoefficient, distrShuntBRow, distrShuntBColumn,
                distrShuntBCoefficient,
                new int[] {0}, new int[0], noAlternateQRows(openBranchIndices.length / 6), noAlternateQRows(zeroImpedanceBranchIndices.length / 11));
    }

    /** {@link #solve} with the GROUP pilot-Q relocation tables (plan-A6) and the ob/zero-impedance alternate
     *  Q rows — required whenever the index packs carry alternateQRows encodings (shared groups / remote
     *  members). Pass all-(-1) ob/zero-impedance alt arrays when only cb/sh relocate. */
    @SuppressWarnings("java:S107")
    public static double[] solve(double[] x0, double[] target, int[] rowMode, int maxIter, double tol,
                                 double[] closedBranchParameters, int[] closedBranchIndices, double[] openBranchParameters, int[] openBranchIndices,
                                 double[] shuntParameters, int[] shuntIndices, double[] hvdcAcEmulationParameters, int[] hvdcAcEmulationIndices,
                                 int[] distrQElement, int[] distrQSide, double[] distrQWeight, int[] distrQRow, int[] distrQKind,
                                 double[] zeroImpedanceBranchParameters, int[] zeroImpedanceBranchIndices,
                                 int[] distrRhoRow, int[] distrRhoColumn, double[] distrRhoCoefficient,
                                 int[] distrShuntBRow, int[] distrShuntBColumn, double[] distrShuntBCoefficient, int[] reactiveRelocationGroupStart, int[] reactiveRelocationCandidateRows,
                                 int[] openBranchAlternateReactiveDestination, int[] zeroImpedanceBranchAlternateReactiveDestination) {
        long structure = StructureNative.createStructure(x0.length, closedBranchParameters, closedBranchIndices,
                openBranchParameters, openBranchIndices, shuntParameters, shuntIndices, hvdcAcEmulationParameters,
                hvdcAcEmulationIndices,
                distrQElement, distrQSide, distrQWeight, distrQRow, distrQKind, zeroImpedanceBranchParameters, zeroImpedanceBranchIndices,
                distrRhoRow, distrRhoColumn, distrRhoCoefficient, distrShuntBRow, distrShuntBColumn,
                distrShuntBCoefficient,
                noPcSide(closedBranchIndices.length / 12), new double[0], new int[0], reactiveRelocationGroupStart,
                reactiveRelocationCandidateRows, openBranchAlternateReactiveDestination,
                zeroImpedanceBranchAlternateReactiveDestination,
                noDependentBaseGroups(reactiveRelocationGroupStart), noDependentBaseGroups(reactiveRelocationGroupStart), noDependentGroupOfRows(x0.length));
        try {
            return solveOnStructure(structure, x0, target, rowMode, maxIter, tol, 0, 0.0, 0.0, new int[x0.length]);
        } finally {
            StructureNative.destroyStructure(structure);
        }
    }

    /**
     * One-shot pure Newton-Raphson on an existing structure: a batch-of-one {@link BatchSolveNative#createBatchContext}
     * without any outer-loop pack, one {@link BatchSolveNative#solveBatchContext}, then the context is destroyed.
     *
     * @param stateVectorScalingMode  device step scaling: 0 = none, 1 = line search, 2 = max voltage change
     * @param maxDv   MAX_VOLTAGE_CHANGE (stateVectorScalingMode 2): clamp on |dv| per iteration (BUS_V rows)
     * @param maxDphi MAX_VOLTAGE_CHANGE (stateVectorScalingMode 2): clamp on |dphi| per iteration (BUS_PHI rows, radians)
     * @param varType per row, the variable kind for the clamp, see GpuAcActivationExtractor#extractVarType
     * @return {@code double[n + 3]}: the state, then iterations, the final mismatch infinity norm and the
     *         convergence status (see {@link BatchSolveNative#solveBatchContext})
     */
    @SuppressWarnings("java:S107")
    public static double[] solveOnStructure(long structure, double[] x0, double[] target, int[] rowMode,
                                            int maxIter, double tol, int stateVectorScalingMode, double maxDv, double maxDphi,
                                            int[] varType) {
        long handle = BatchSolveNative.createBatchContext(structure, 1, GpuLinearSolveMode.UBATCH,
                new int[0], new double[0], new double[0], new double[0], new double[0], new byte[0], new double[x0.length],
                new int[0], new double[0], -1, 0.0, new double[0], new double[0], new double[0],
                0, new double[0], new double[0], 0, 0);
        try {
            DriverNative.setBatchOuterLoops(handle, false, false);           // pure Newton-Raphson: no outer loop
            return BatchSolveNative.solveBatchContext(handle, x0, target, new int[0], rowMode, new boolean[1][0],
                    new int[0], new double[0], new int[0], new double[0], new double[0], new double[0], new double[0], null, null, null, null, null, null, null, null,
                    maxIter, tol, 0.0, 0.0, 0.0, stateVectorScalingMode, maxDv, maxDphi, varType, null, null, null, null, null, null, 0,
                    com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop.MAX_SWITCH_PQ_PV_DEFAULT_VALUE,
                    0, 0, 0, null, null, 0)[0];
        } finally {
            BatchSolveNative.destroyBatchContext(handle);
        }
    }

    /**
     * One-shot batch convenience: {@link StructureNative#createStructure} + {@link BatchSolveNative#createBatchContext} +
     * {@link BatchSolveNative#solveBatchContext} + destroy both. For repeated batches on the same structure,
     * cache the context instead. The base state/targets/row-mode are shared by every scenario
     * (tiled on-device); {@code nScenarios} is {@code disabledElementMask.length}.
     */
    public static double[][] solveBatch(double[] x0Base, double[] targetBase, int[][] rowMode,
                                         boolean[][] disabledElementMask, int mode, int maxIter, double tol,
                                         double reactiveLimitTol, double slackDistTol,
                                         double[] closedBranchParameters, int[] closedBranchIndices, double[] openBranchParameters, int[] openBranchIndices,
                                         double[] shuntParameters, int[] shuntIndices, double[] hvdcAcEmulationParameters, int[] hvdcAcEmulationIndices,
                                         int[] reactiveLimitVoltageRow, double[] reactiveLimitMinQ, double[] reactiveLimitMaxQ, double[] reactiveLimitLoadQ,
                                         double[] reactiveLimitTargetV, int[] distributedSlackBusPhiRow, double[] distributedSlackBusFactor,
                                         int slackBusPhiRow, double slackBusTargetP,
                                         double[] distributedSlackBusBaseTargetP, double[] distributedSlackBusMaxDelta, double[] distributedSlackBusMinDelta,
                                         int[] distrQElement, int[] distrQSide, double[] distrQWeight, int[] distrQRow, int[] distrQKind,
                                         int[] distrRhoRow, int[] distrRhoColumn, double[] distrRhoCoefficient,
                                         int[] distrShuntBRow, int[] distrShuntBColumn, double[] distrShuntBCoefficient, double[] zeroImpedanceBranchParameters, int[] zeroImpedanceBranchIndices) {
        int n = x0Base.length;
        int nScenarios = disabledElementMask.length;
        long structure = StructureNative.createStructure(n, closedBranchParameters, closedBranchIndices,
                openBranchParameters, openBranchIndices, shuntParameters, shuntIndices, hvdcAcEmulationParameters,
                hvdcAcEmulationIndices,
                distrQElement, distrQSide, distrQWeight, distrQRow, distrQKind, zeroImpedanceBranchParameters, zeroImpedanceBranchIndices,
                distrRhoRow, distrRhoColumn, distrRhoCoefficient, distrShuntBRow, distrShuntBColumn,
                distrShuntBCoefficient,
                noPcSide(closedBranchIndices.length / 12), new double[0], new int[0], new int[] {0}, new int[0],
                noAlternateQRows(openBranchIndices.length / 6), noAlternateQRows(zeroImpedanceBranchIndices.length / 11), new int[0], new int[0], noDependentGroupOfRows(n));
        try {
            long handle = BatchSolveNative.createBatchContext(structure, nScenarios, mode,
                    reactiveLimitVoltageRow, reactiveLimitMinQ, reactiveLimitMaxQ, reactiveLimitLoadQ, reactiveLimitTargetV, new byte[reactiveLimitVoltageRow.length], new double[x0Base.length],
                    distributedSlackBusPhiRow, distributedSlackBusFactor, slackBusPhiRow, slackBusTargetP,
                    distributedSlackBusBaseTargetP, distributedSlackBusMaxDelta, distributedSlackBusMinDelta, GpuDistributedSlackFactorMode.STATIC, new double[0], new double[0], 0, 0);
            try {
                // OLF's forced-PV election keys. This one-shot entry point has no network behind it
                // (the caller hands raw rows), so the keys are NEUTRAL: equal nominal voltage, no
                // live target-P row, and the entry order as the id rank — which reduces the rule to
                // OLF's last tie-break, "lowest id stays PV". A context that reaches a solve without
                // any keys is refused, and the rule only fires when every bus a check finds PV
                // switches to PQ, so this keeps the low-level entry usable without inventing a ranking.
                int[] idRank = new int[reactiveLimitVoltageRow.length];
                int[] pRow = new int[reactiveLimitVoltageRow.length];
                for (int i = 0; i < idRank.length; i++) {
                    idRank[i] = i;
                    pRow[i] = -1;
                }
                ReactiveLimitsNative.setBatchReactiveLimitStrongestPvKeys(handle, new double[reactiveLimitVoltageRow.length], pRow, new double[reactiveLimitVoltageRow.length], idRank);
                // no loop list behind this entry point either: the caller asks for the two loops by handing their tables
                DriverNative.setBatchOuterLoops(handle, true, true);
                // one-shot: no per-scenario gen-override flips, so the first scenario's row mode is the base row mode
                return BatchSolveNative.solveBatchContext(handle, x0Base, targetBase, rowModeEntries(rowMode, rowMode[0]), rowMode[0],
                        disabledElementMask,
                        new int[0], new double[0], new int[0], new double[0], new double[0], new double[0], new double[0], null, null, null, null, null, null, null, null,
                        maxIter, tol, reactiveLimitTol, slackDistTol, 0.0, 0, 0.0, 0.0, null, null, null, null, null, null, null, 0,
                        com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop.MAX_SWITCH_PQ_PV_DEFAULT_VALUE,   // OLF's cap: 0 means never switch back
                        0, 0, 0, null, null, 0);
            } finally {
                BatchSolveNative.destroyBatchContext(handle);
            }
        } finally {
            StructureNative.destroyStructure(structure);
        }
    }

    /** The (scenario, row, mode) entries where the scenarios' rows differ from {@code base}, scenario then row ascending:
     *  what the solve takes in place of whole rows. */
    private static int[] rowModeEntries(int[][] rowModes, int[] base) {
        gnu.trove.list.array.TIntArrayList entries = new gnu.trove.list.array.TIntArrayList();
        for (int s = 0; s < rowModes.length; s++) {
            for (int r = 0; r < base.length; r++) {
                if (rowModes[s][r] != base[r]) {
                    entries.add(s);
                    entries.add(r);
                    entries.add(rowModes[s][r]);
                }
            }
        }
        return entries.toArray();
    }
}
