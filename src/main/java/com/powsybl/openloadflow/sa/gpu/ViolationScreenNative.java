/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;

/**
 * The device's violation screen: a limit table uploaded once per run, screened against a batch of states.
 *
 * <p>JNI glue: {@code native/cuda/sa/olf_gpu_violations.cu}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class ViolationScreenNative {

    static {
        GpuAcNewtonSolver.isAvailable();   // loads libolfgpu once
    }

    private ViolationScreenNative() {
    }

    /** Release a violation table (idempotent for handle 0). */
    public static native void destroyViolationTable(long tableHandle);

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
            double[] closedBranchParameters, int[] closedBranchIndices, double[] openBranchParameters, int[] openBranchIndices);
}
