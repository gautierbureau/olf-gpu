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
 * The closed branches packed for the device ({@link ElementPack} layout documented on {@link GpuAcData}), plus, per
 * phase-control alternate destination, the side packed with its a1 column ({@code phaseControlSide}): the phase
 * control alternate-destination tables are built from it, so it is an index array.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record ClosedBranchPack(double[] parameters, int[] indices, int[] phaseControlSide) {

    public static ClosedBranchPack empty() {
        return new ClosedBranchPack(new double[0], new int[0], new int[0]);
    }

    public boolean sameParametersAs(ClosedBranchPack other) {
        return Arrays.equals(parameters, other.parameters);
    }

    public boolean sameIndicesAs(ClosedBranchPack other) {
        return Arrays.equals(indices, other.indices) && Arrays.equals(phaseControlSide, other.phaseControlSide);
    }
}
