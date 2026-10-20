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
 * One element family packed for the device: per element, its {@code parameters} (doubles, a fixed count per element)
 * and its {@code indices} (the rows and columns it touches, a fixed count per element; -1 for an absent side). The
 * layouts are documented on {@link GpuAcData}. Used as is for shunts, HVDC AC emulation and load models.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record ElementPack(double[] parameters, int[] indices) {

    public static ElementPack empty() {
        return new ElementPack(new double[0], new int[0]);
    }

    public boolean sameParametersAs(ElementPack other) {
        return Arrays.equals(parameters, other.parameters);
    }

    public boolean sameIndicesAs(ElementPack other) {
        return Arrays.equals(indices, other.indices);
    }
}
