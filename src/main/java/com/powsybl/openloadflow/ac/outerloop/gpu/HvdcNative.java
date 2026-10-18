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
 * The device's HVDC AC-emulation loops (FreezingHvdcACEmulationOuterloop, AcHvdcAcEmulationLimitsOuterLoop).
 *
 * <p>JNI glue: {@code native/cuda/jni/jni/hvdc_jni.cu}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class HvdcNative {

    static {
        GpuAcNewtonSolver.isAvailable();   // loads libolfgpu once
    }

    private HvdcNative() {
    }

    /** The HVDC statuses after a solve, {@code nScenarios * nHvdc}, or null when none are installed. */
    public static native int[] readBatchHvdcStatus(long handle);

    /**
     * Install the HVDC AC-emulation outer loops' tables and their per-scenario starting state, so
     * FreezingHvdcACEmulation and AcHvdcAcEmulationLimits run as DRIVER PHASES in OLF's order —
     * between DistributedSlack and ReactiveLimits — instead of as host orchestrators after the whole
     * solve. Pass a null/empty {@code elem} to leave them to the host orchestrator.
     *
     * @param in per hvdc x9: {@code p0 k pMax12 pMax21 loss1 loss2 r baseD1 baseD2}
     * @param ctgOut per (scenario, hvdc): the contingency itself outaged it, so the loops leave it alone
     * @param status per (scenario, hvdc) starting status, and {@code appliedD1/D2} the converter-row
     *        deltas the pre-solve freeze already applied — transitions are applied incrementally
     */
    public static native void setBatchHvdcLoops(long handle, int[] elem, int[] bus1Row, int[] bus2Row,
                                                int[] bus1Phi, int[] bus2Phi,
                                                double[] in, byte[] ctgOut, int[] status,
                                                double[] appliedD1, double[] appliedD2, int limitsActive);
}
