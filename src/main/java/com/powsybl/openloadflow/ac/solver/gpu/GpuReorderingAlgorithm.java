/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

/**
 * The fill-reducing ordering cuDSS analyzes the Jacobian with.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public enum GpuReorderingAlgorithm {
    /** AMD on cuDSS 0.8 (the measured winner), the library default on 0.7. */
    AUTO(-1),
    LIBRARY_DEFAULT(0),
    BTF_COLAMD(1),
    COLAMD(2),
    AMD(3),
    NESTED_DISSECTION(4),
    NONE(5);

    private final int nativeCode;

    GpuReorderingAlgorithm(int nativeCode) {
        this.nativeCode = nativeCode;
    }

    int nativeCode() {
        return nativeCode;
    }
}
