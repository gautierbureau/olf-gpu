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
     *  maxIteration and stepFold) — consumed by solves called with {@code stateVectorScalingMode == GpuStateVectorScalingMode.LINE_SEARCH}. */
    public static native void setBatchLineSearch(long handle, int lineSearchStateVectorScalingMaxIteration, double lineSearchStateVectorScalingStepFold);

    /** OLF_GPU_LOOP_TRACE by contingency: the slot to trace in the next solve of this context, -1 = none. */
    public static native void setBatchLoopTraceSlot(long handle, int slot);

    /** Install (or clear, with nulls) the PER_EQUATION_TYPE per-row stopping thresholds on a batch
     *  context: {@code fixedVariableRowMaxMismatch[r]} for identity/slope rows, {@code equationRowMaxMismatch[r]} for power rows —
     *  the convergence test then requires EVERY row's |F| under its threshold (OLF's
     *  PerEquationTypeStoppingCriteria) instead of the 2-norm/infnorm rule. Contexts are CACHED, so
     *  a run without the criterion must clear (null, null). */
    public static native void setBatchRowCriteria(long handle, double[] fixedVariableRowMaxMismatch, double[] equationRowMaxMismatch);

    /** The alternate per-row threshold and its selector row (-1: none) of the PER_EQUATION_TYPE criterion (see
     *  GpuAcActivationExtractor.extractRowCriteria); nulls clear them. Call after {@link #setBatchRowCriteria}. */
    public static native void setBatchRowCriteriaAlternate(long handle, double[] alternateRowMaxMismatch, int[] alternateRowMaxMismatchSelector);

    /** Absolute target writes at flat {@code (scenario * n + row)} positions, for a host-owned loop. */
    public static native void setBatchTargetDeltas(long handle, int[] rows, double[] values);

    /** Starting values for the next batch solve that is given no full per-scenario state: {@code s * n + row} ->
     *  value, written over the base state the device tiles; consumed by that solve. Null clears. */
    public static native void setBatchX0Seeds(long handle, int[] flatIndex, double[] value);

    /** The run's linear-solver settings (GpuLoadFlowParameters.install): -1 reorder / pivot type = AUTO, NaN threshold =
     *  cuDSS's own. Read by every context built afterwards and by every solve's iterative refinement. */
    public static native void setLinearSolverSettings(int reorder, int matching, int pivotType, double pivotThreshold,
                                                      int iterativeRefinementBaseSteps, int iterativeRefinementMaxSteps, double iterativeRefinementTolerance);
}
