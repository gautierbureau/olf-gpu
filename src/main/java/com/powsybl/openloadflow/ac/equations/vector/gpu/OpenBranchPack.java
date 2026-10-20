/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import java.util.Arrays;

/**
 * The half-open branches packed for the device ({@link ElementPack} layout documented on {@link GpuAcData}), plus,
 * per element, the alternate destinations of its reactive terms ({@code alternateReactiveDestination}, two per
 * element) when the bus's reactive balance is relocated (docs/gpu-vs-olf.md section 2.3).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record OpenBranchPack(double[] parameters, int[] indices, int[] alternateReactiveDestination) {

    public static OpenBranchPack empty() {
        return new OpenBranchPack(new double[0], new int[0], new int[0]);
    }

    public boolean sameParametersAs(OpenBranchPack other) {
        return Arrays.equals(parameters, other.parameters);
    }

    public boolean sameIndicesAs(OpenBranchPack other) {
        return Arrays.equals(indices, other.indices)
                && Arrays.equals(alternateReactiveDestination, other.alternateReactiveDestination);
    }
}
