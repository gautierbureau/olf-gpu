/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

/**
 * The pivoting of the robust single-matrix solve.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public enum GpuPivotType {
    /** Global column pivoting under the COLAMD orderings, cuDSS's automatic choice otherwise. */
    BY_ORDERING(-1),
    AUTO(0),
    NONE(1),
    GLOBAL_COL(2),
    GLOBAL_ROW(3),
    DIAGONAL(4);

    private final int nativeCode;

    GpuPivotType(int nativeCode) {
        this.nativeCode = nativeCode;
    }

    int nativeCode() {
        return nativeCode;
    }
}
