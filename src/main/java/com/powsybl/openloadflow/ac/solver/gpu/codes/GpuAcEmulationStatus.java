/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.codes;

import com.powsybl.openloadflow.network.LfHvdc;

/**
 * OLF's {@code LfHvdc.AcEmulationControl.AcEmulationStatus} as the device holds it, per (scenario, hvdc). Mirrored by
 * native/cuda/common/codes/gpu_ac_emulation_status.cuh.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuAcEmulationStatus {

    public static final int LINEAR_MODE = 0;
    public static final int SATURATION_MODE_FROM_CS1_TO_CS2 = 1;
    public static final int SATURATION_MODE_FROM_CS2_TO_CS1 = 2;
    public static final int FROZEN = 3;

    private GpuAcEmulationStatus() {
    }

    public static int of(LfHvdc.AcEmulationControl.AcEmulationStatus status) {
        return switch (status) {
            case LINEAR_MODE -> LINEAR_MODE;
            case SATURATION_MODE_FROM_CS1_TO_CS2 -> SATURATION_MODE_FROM_CS1_TO_CS2;
            case SATURATION_MODE_FROM_CS2_TO_CS1 -> SATURATION_MODE_FROM_CS2_TO_CS1;
            case FROZEN -> FROZEN;
        };
    }
}
