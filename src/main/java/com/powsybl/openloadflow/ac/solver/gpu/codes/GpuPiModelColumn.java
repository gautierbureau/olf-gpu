/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.codes;

/**
 * A pi model packed as {@link #COUNT} doubles, in the order of OLF's {@code PiModel} getters: one row per tap in the
 * tap tables of the transformer and phase control loops (tapData), and one per closed branch (closedBranchParameters). Mirrored by
 * native/cuda/jni/common/codes/gpu_pi_model_column.cuh.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuPiModelColumn {

    public static final int Y = 0;
    public static final int G1 = 1;
    public static final int G2 = 2;
    public static final int B1 = 3;
    public static final int B2 = 4;
    public static final int KSI = 5;
    /** The ratio, the voltage control loops' tap value. */
    public static final int R1 = 6;
    /** The phase shift, the phase control loops' tap value. */
    public static final int A1 = 7;
    public static final int COUNT = 8;

    private GpuPiModelColumn() {
    }
}
