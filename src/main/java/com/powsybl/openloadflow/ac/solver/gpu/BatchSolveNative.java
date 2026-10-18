/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;

/**
 * The batched solve entry points.
 *
 * <p>JNI glue: {@code native/cuda/jni/jni/batch_solve_jni.cu}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class BatchSolveNative {

    static {
        GpuAcNewtonSolver.isAvailable();   // loads libolfgpu once
    }

    private BatchSolveNative() {
    }

    /**
     * Build a CACHED solver context for BATCHED N-1 security analysis on a shared structure:
     * multiple scenarios (post-contingency states) against the same network STRUCTURE and
     * superset pattern. Per-scenario element disablement is encoded as device masks; converged
     * scenarios can stop early. cuDSS's symbolic analysis is done once over the superset
     * pattern. The pattern / families / destination arrays are NOT duplicated: they live in
     * the structure, built once per extraction — only the per-scenario state and the reactive-limits /
     * distributed-slack outer-loop packs are owned here.
     *
     * @param structureHandle a structure from {@link #createStructure}
     * @param nScenarios      number of contingency scenarios to solve in parallel
     * @return an opaque native handle for batched solves; release with {@link #destroyBatchContext}
     */
    public static native long createBatchContext(long structureHandle, int nScenarios, int mode,
                                                  int[] rlVrow, double[] rlMinQ, double[] rlMaxQ, double[] rlLoadQ,
                                                  double[] rlVtarget, byte[] rlMonitored, double[] rowSlope,
                                                  int[] dsPhiRow, double[] dsFactor,
                                                  int dsSlackPhiRow, double dsSlackTargetP,
                                                  double[] dsBaseTargetP, double[] dsMaxDelta, double[] dsMinDelta,
                                                  int dsMode, double[] dsMarginUp, double[] dsMarginDown,
                                                  double dsSlackMarginUp, double dsSlackMarginDown);

    public static native void destroyBatchContext(long handle);

    public static native double[][] solveBatchContext(long handle, double[] x0Base, double[] targetBase,
                                                       int[][] rowMode, int[] baseRowMode, boolean[][] disabledElementMask,
                                                       int[] targetOverrideIndex, double[] targetOverrideValue,
                                                       int[] dsOverrideBus, double[] dsOverrideFactorDelta,
                                                       double[] dsOverrideBaseDelta, double[] dsOverrideMaxDelta,
                                                       double[] dsOverrideMinDelta, double[] dsAnchorShift,
                                                       double[] dsSlackShift,
                                                       int[] rlVtOvSlotIdx, double[] rlVtOvVal,
                                                       int[] rlQOvSlotEntry, double[] rlQOvMinMax,
                                                       int[] cvOvSlotEntry, double[] cvOvStepScale,
                                                       int maxIter, double tol,
                                                       double reactiveLimitTol, double slackDistTol, double convEpsPerEq,
                                                       int lsMode, double maxDv, double maxDphi, int[] varType,
                                                       double[] x0PerScenario,
                                                       int[] shuntOvSlotElem, double[] shuntOvGb,
                                                       int[] rlStateIn, int[] rlStateOut,
                                                       int[] r1OvSlotElem, double[] r1OvVal,
                                                       int maxOuterIterations, int maxPqPvSwitch, int dsRejectResidual, int skipRlPhase,
                                                       int nziBase, int[] ziOnSlotElem, int[] ziOffSlotElem);
}
