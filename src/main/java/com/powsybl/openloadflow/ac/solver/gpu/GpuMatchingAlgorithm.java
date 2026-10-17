/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

/**
 * The matching cuDSS applies before factoring (robust single-matrix solve and adjoint). There is no "none": a
 * bus fed only through a zero-impedance branch leaves rows with a zero diagonal, which need matching.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public enum GpuMatchingAlgorithm {
    AUTO(6),
    MAX_DIAG_COUNT(1),
    MAX_MIN_DIAG(2),
    MAX_MIN_DIAG_ALT(3),
    MAX_DIAG_SUM(4),
    MAX_DIAG_PRODUCT(5);

    private final int nativeCode;

    GpuMatchingAlgorithm(int nativeCode) {
        this.nativeCode = nativeCode;
    }

    int nativeCode() {
        return nativeCode;
    }
}
