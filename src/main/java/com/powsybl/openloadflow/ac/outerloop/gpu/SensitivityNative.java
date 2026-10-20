/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;

/**
 * The batched sensitivities the incremental control loops read.
 *
 * <p>JNI glue: {@code native/cuda/jni/sensitivity_jni.cu}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class SensitivityNative {

    static {
        GpuAcNewtonSolver.isAvailable();   // loads libolfgpu once
    }

    private SensitivityNative() {
    }

    /** Batched SENSITIVITY solve for the incremental outer-loop family: J at the given per-scenario
     *  states (same forward system as the Newton), unit RHS at each {@code unitRows} pinned-variable
     *  identity row, the {@code observeCols} variable entries of each solution gathered per scenario.
     *  Returns {@code [nScenarios * unitRows.length * observeCols.length]} laid out scenario-major.
     *  MUST follow a solveBatchContext call on the same handle with the same per-scenario overrides. */
    public static native double[] solveBatchSensitivities(long handle, double[] states,
                                                          int[] unitRows, int[] observeCols);
}
