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
 * <p>JNI glue: {@code native/cuda/jni/batch_solve_jni.cu}.
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
     * @param structureHandle a structure from {@code StructureNative.createStructure}
     * @param nScenarios      number of contingency scenarios to solve in parallel
     * @return an opaque native handle for batched solves; release with {@link #destroyBatchContext}
     */
    public static native long createBatchContext(long structureHandle, int nScenarios, int mode,
                                                  int[] reactiveLimitVoltageRow, double[] reactiveLimitMinQ, double[] reactiveLimitMaxQ, double[] reactiveLimitLoadQ,
                                                  double[] reactiveLimitTargetV, byte[] reactiveLimitFlags, double[] rowSlope,
                                                  int[] distributedSlackBusPhiRow, double[] distributedSlackBusFactor,
                                                  int slackBusPhiRow, double slackBusTargetP,
                                                  double[] distributedSlackBusBaseTargetP, double[] distributedSlackBusMaxDelta, double[] distributedSlackBusMinDelta,
                                                  int distributedSlackFactorMode, double[] distributedSlackBusMarginUp, double[] distributedSlackBusMarginDown,
                                                  double slackBusMarginUp, double slackBusMarginDown);

    /** Release a batch context from {@link #createBatchContext} (idempotent for handle 0). */
    public static native void destroyBatchContext(long handle);

    /**
     * Choose the bank slots the next scenarios go into: local slot {@code j} of every per-scenario setter, of the solve
     * and of the readbacks after it is bank slot {@code bankSlots[j]}, until the next call. Each bank slot appears at
     * most once; the other slots keep what they hold. Null or empty: every slot of the bank, local slot {@code j} into
     * slot {@code j}.
     */
    public static native void beginBatchInstall(long handle, int[] bankSlots);

    /**
     * Run batched Newton-Raphson AC power flow on the GPU: one iteration loop over all
     * scenarios, with per-scenario convergence retirement (stop filling/solving once
     * converged). Scenarios remain in the batch until they converge or hit maxIter.
     *
     * <p>The base state/targets are SHARED by every scenario (a single-branch N-1 starts from the same
     * converged base operating point): they are passed as single length-{@code n} rows and tiled across
     * the batch on-device. The row-mode is PER-SCENARIO ({@code [nScenarios][n]}) so a contingency can
     * freeze islanded buses or flip PV↔PQ (reactive limits) independently; it is uploaded as-is and
     * mutated in place on-device by the outer-loop kernels. The disabled-element mask is also
     * per-scenario. {@code nScenarios} is {@code disabledElementMask.length}.
     *
     * @param handle              a batch context from {@link #createBatchContext}
     * @param x0Base              shared initial state, {@code [n]}: tiled to every scenario on-device
     * @param targetBase          shared targets, {@code [n]}: in STABLE row order, tiled per scenario
     * @param rowMode             per-scenario row modes, {@code [nScenarios][n]}: 1 = equation, 0 = identity
     *                            (islanded buses frozen, slack/PV identity rows)
     * @param disabledElementMask per-scenario disabled elements, {@code [nScenarios][nElements]}: bit-packed
     *                            (true if that element is outaged in this scenario; elements enumerated
     *                            as: first nClosedBranch closed branches, then nOpenBranch open branches, then nShunt shunts, then nHvdcAcEmulation HVDCs)
     * @param maxIter             maximum Newton iterations per scenario
     * @param tol                 convergence tolerance on mismatch infinity norm (the device floor)
     * @param convEpsPerEq        if {@code > 0}, the inner-Newton convergence test uses OLF's EXACT criterion
     *                            ({@code DefaultNewtonRaphsonStoppingCriteria}: {@code ‖F‖₂ < sqrt(convEpsPerEq² · n)})
     *                            instead of the infinity-norm floor — so a batched SA can stop on the SAME
     *                            decision as the CPU loadflow (apples-to-apples timing). {@code <= 0} keeps the
     *                            {@code tol} infinity-norm floor.
     * @param pauseAtEnded        0: return when every scenario has ended; k &gt; 0: return as soon as k of them have, the
     *                            others still solving, for the caller to install new scenarios into the slots handed back
     *                            and call again (which goes on with those still solving)
     * @return per scenario the solve handed back (the selection the readbacks then read, in the order the scenarios were
     *         installed): [state(n), iterations, finalMismatch, convergenceStatus] where convergenceStatus is:
     *         0 = converged, 1 = max_iter, -1 = diverged/NaN
     */
    /** The bank slots of the scenarios the last solve handed back, in the order of its result rows (every slot, in order,
     *  when it handed back the whole bank). */
    public static native int[] readBatchHandedBackSlots(long handle);

    public static native double[][] solveBatchContext(long handle, double[] x0Base, double[] targetBase,
                                                       int[][] rowMode, int[] baseRowMode, boolean[][] disabledElementMask,
                                                       int[] targetOverrideIndex, double[] targetOverrideValue,
                                                       int[] distributedSlackOverrideBus, double[] distributedSlackOverrideFactorDelta,
                                                       double[] distributedSlackOverrideBaseTargetPDelta, double[] distributedSlackOverrideMaxDelta,
                                                       double[] distributedSlackOverrideMinDelta, double[] distributedSlackBusAnchorShift,
                                                       double[] slackBusAnchorShift,
                                                       int[] reactiveLimitTargetVOverrideScenarioEntry, double[] reactiveLimitTargetVOverrideValue,
                                                       int[] reactiveLimitDeltaOverrideScenarioEntry, double[] reactiveLimitDeltaOverrideValues,
                                                       int[] reactiveCapabilityCurveOverrideScenarioCurve, double[] reactiveCapabilityCurveOverrideStepScale,
                                                       int maxIter, double tol,
                                                       double reactiveLimitTol, double slackDistTol, double convEpsPerEq,
                                                       int stateVectorScalingMode, double maxDv, double maxDphi, int[] varType,
                                                       double[] x0PerScenario,
                                                       int[] shuntOvSlotElem, double[] shuntOvGb,
                                                       int[] reactiveLimitsSeedState,
                                                       int[] closedBranchParameterOverrideScenarioElement, double[] closedBranchParameterOverrideValues,
                                                       int maxOuterIterations, int maxPqPvSwitch, int rejectSlackDistributionFailure, int skipReactiveLimitsPhase,
                                                       int nBaseClosedZeroImpedanceBranches, int[] zeroImpedanceBranchCloseScenarioElement, int[] zeroImpedanceBranchOpenScenarioElement,
                                                       int pauseAtEnded);
}
