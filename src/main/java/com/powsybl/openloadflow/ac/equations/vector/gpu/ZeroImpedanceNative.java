/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

/**
 * The zero-impedance (coupler) tables of the device structure.
 *
 * <p>JNI glue: {@code native/cuda/jni/jni/zero_impedance_jni.cu}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class ZeroImpedanceNative {

    static {
        GpuAcNewtonSolver.isAvailable();   // loads libolfgpu once
    }

    private ZeroImpedanceNative() {
    }

    /** Per (scenario, coupler) spanning-tree flag, as flat {slot, zeroImpedanceElement, isTree} triples, for the
     *  scenarios where OLF's tree has moved away from the one the structure was extracted on. Empty
     *  clears. */
    public static native void setBatchSpanningTreeOverrides(long handle, int[] slotElemVal);
}
