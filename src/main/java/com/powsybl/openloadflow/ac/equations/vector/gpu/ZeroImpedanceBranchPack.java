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
 * The zero-impedance branches (bus couplers) packed for the device (layout documented on {@link GpuAcData}), plus
 * each element's LfBranch number ({@code branchNum}) and the alternate destinations of its reactive terms
 * ({@code alternateReactiveDestination}, two per element) when a bus's reactive balance is relocated.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record ZeroImpedanceBranchPack(double[] parameters, int[] indices, int[] branchNum,
                                      int[] alternateReactiveDestination) {

    public static ZeroImpedanceBranchPack empty() {
        return new ZeroImpedanceBranchPack(new double[0], new int[0], new int[0], new int[0]);
    }

    public boolean sameParametersAs(ZeroImpedanceBranchPack other) {
        return Arrays.equals(parameters, other.parameters);
    }

    /** Same indices and alternate destinations; the branch numbers are not compared (as before the grouping). */
    public boolean sameIndicesAs(ZeroImpedanceBranchPack other) {
        return Arrays.equals(indices, other.indices)
                && Arrays.equals(alternateReactiveDestination, other.alternateReactiveDestination);
    }
}
