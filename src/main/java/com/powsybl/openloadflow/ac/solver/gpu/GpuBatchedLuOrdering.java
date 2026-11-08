/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

/**
 * The elimination order the batched LU ({@link GpuLinearSolver#BATCHED_LU}) is planned on.
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public enum GpuBatchedLuOrdering {
    /** cuDSS's analysis order ({@link GpuReorderingAlgorithm}): the coupler chains of a breaker-mode network stay short. */
    CUDSS(0),
    /** The exact minimum degree of the Jacobian's symmetrized pattern, computed on the host: a shallower elimination
     *  tree on a network with few couplers. */
    MINIMUM_DEGREE(1),
    /** Whichever of the two has the fewer elimination levels on the system's pattern. */
    FEWEST_LEVELS(2);

    private final int nativeCode;

    GpuBatchedLuOrdering(int nativeCode) {
        this.nativeCode = nativeCode;
    }

    int nativeCode() {
        return nativeCode;
    }
}
