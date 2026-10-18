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
 * The batched Newton-Raphson settings: stopping criteria, line search, iteration budgets.
 *
 * <p>JNI glue: {@code native/cuda/jni/jni/newton_settings_jni.cu}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class NewtonSettingsNative {

    static {
        GpuAcNewtonSolver.isAvailable();   // loads libolfgpu once
    }

    private NewtonSettingsNative() {
    }

    /** The installed linear-solver settings, in {@link #setLinearSolverSettings}'s order. */
    public static native double[] readLinearSolverSettings();

    /** Install (or clear, with null) per-scenario ACTIVE equation counts for the L2 stopping
     *  threshold sqrt(eps^2 * nActive) — the CPU removes equations for outaged variable-carrying
     *  elements (rho/a1 identities) and islanded buses; the padded device n over-counts those rows
     *  and skews the stop decision by the sqrt ratio. Per-chunk state. */
    public static native void setBatchActiveRowCounts(long handle, int[] nActive);

    /** Rows an islanding outage froze to a flat identity state, as (scenario, row) pairs. Their targets
     *  are not the outer loops' to write — OLF removes an islanded bus's equations, so nothing there
     *  re-derives a target for them. */
    public static native void setBatchFrozenRows(long handle, int[] slotRow);

    /** LINE_SEARCH backtracking parameters for a batch context (OLF LineSearchStateVectorScaling's
     *  maxIteration and stepFold) — consumed by solves called with {@code lsMode == 1}. */
    public static native void setBatchLineSearch(long handle, int lsMaxIter, double lsStepFold);

    /** OLF_GPU_LOOP_TRACE by contingency: the slot to trace in the next solve of this context, -1 = none. */
    public static native void setBatchLoopTraceSlot(long handle, int slot);

    /** Release a batch context from {@link #createBatchContext} (idempotent for handle 0). */
    /** Install (or clear, with nulls) the PER_EQUATION_TYPE per-row stopping thresholds on a batch
     *  context: {@code thrIdent[r]} for identity/slope rows, {@code thrPower[r]} for power rows —
     *  the convergence test then requires EVERY row's |F| under its threshold (OLF's
     *  PerEquationTypeStoppingCriteria) instead of the 2-norm/infnorm rule. Contexts are CACHED, so
     *  a run without the criterion must clear (null, null). */
    public static native void setBatchRowCriteria(long handle, double[] thrIdent, double[] thrPower);

    /** The alternate per-row threshold and its selector row (-1: none) of the PER_EQUATION_TYPE criterion (see
     *  GpuAcActivationExtractor.extractRowCriteria); nulls clear them. Call after {@link #setBatchRowCriteria}. */
    public static native void setBatchRowCriteriaAlt(long handle, double[] thrAlt, int[] thrSel);

    /** Absolute target writes at flat {@code (scenario * n + row)} positions, for a host-owned loop. */
    public static native void setBatchTargetDeltas(long handle, int[] rows, double[] values);

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
     *                            as: first ncb closed branches, then nob open branches, then nsh shunts, then nhv HVDCs)
     * @param maxIter             maximum Newton iterations per scenario
     * @param tol                 convergence tolerance on mismatch infinity norm (the device floor)
     * @param convEpsPerEq        if {@code > 0}, the inner-Newton convergence test uses OLF's EXACT criterion
     *                            ({@code DefaultNewtonRaphsonStoppingCriteria}: {@code ‖F‖₂ < sqrt(convEpsPerEq² · n)})
     *                            instead of the infinity-norm floor — so a batched SA can stop on the SAME
     *                            decision as the CPU loadflow (apples-to-apples timing). {@code <= 0} keeps the
     *                            {@code tol} infinity-norm floor.
     * @return {@code [nScenarios][n + 3]}: per-scenario [state(n), iterations, finalMismatch, convergenceStatus]
     *         where convergenceStatus is: 0 = converged, 1 = max_iter, -1 = diverged/NaN
     */
    /** Starting values for the next batch solve that is given no full per-scenario state: {@code s * n + row} ->
     *  value, written over the base state the device tiles; consumed by that solve. Null clears. */
    public static native void setBatchX0Seeds(long handle, int[] flatIndex, double[] value);

    /** The run's linear-solver settings (GpuLoadFlowParameters.install): -1 reorder / pivot type = AUTO, NaN threshold =
     *  cuDSS's own. Read by every context built afterwards and by every solve's iterative refinement. */
    public static native void setLinearSolverSettings(int reorder, int matching, int pivotType, double pivotThreshold,
                                                      int irBase, int irMax, double irTol);
}
