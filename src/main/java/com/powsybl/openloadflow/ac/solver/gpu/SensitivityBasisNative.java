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
 * The sensitivity basis of a batched sensitivity analysis on a batch context: what every converged scenario solves on its
 * Jacobian at its converged state, once the driver's Newton solve of the call has ended (the native
 * {@code sensitivityBasisStep}), and the solutions read back with the handed-back scenarios' rows
 * ({@link com.powsybl.openloadflow.ac.solver.gpu.batch.SensitivityBasisRequest} for the layout).
 *
 * <p>JNI glue: {@code native/cuda/jni/sensitivity_basis_jni.cu}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class SensitivityBasisNative {

    static {
        GpuAcNewtonSolver.isAvailable();   // loads libolfgpu once
    }

    /** Whether the loaded library has these entry points: a library built before them solves no basis, and its batches
     *  run as before (the sensitivities of their contingencies are solved on the CPU). */
    private static final boolean IN_LIBRARY = probe();

    private SensitivityBasisNative() {
    }

    private static boolean probe() {
        try {
            setBatchSensitivityBasis(0, null, null, null, null, null, false);   // a null context: returns at once
            return true;
        } catch (UnsatisfiedLinkError e) {
            return false;
        }
    }

    /** True when the device is there and the library can solve a sensitivity basis. */
    public static boolean isAvailable() {
        return GpuAcNewtonSolver.isAvailable() && IN_LIBRARY;
    }

    /**
     * The basis every converged scenario of the context solves, for the calls that follow: unit columns and weighted
     * columns (CSR by column) on stable equation rows, the gather rows (base variable rows), and whether a per-scenario
     * participation column follows ({@link #setBatchSensitivityParticipation}). Null unit or gather rows clear it.
     */
    public static native void setBatchSensitivityBasis(long handle, int[] unitRows, int[] weightedStart, int[] weightedRows,
                                                       double[] weightedValues, int[] gatherRows, boolean participation);

    /** Per installed scenario, in the install's order: its participation column, {@code n} values over stable equation
     *  rows. Null: none. */
    public static native void setBatchSensitivityParticipation(long handle, double[] rows);

    /** The handed-back scenarios' basis solutions, in the install's order ({@code columns x gather rows} each, column-major);
     *  null without a basis. NaN where the device solved none (the robust mode, a call that paused). */
    public static native double[] readBatchSensitivityBasis(long handle);
}
