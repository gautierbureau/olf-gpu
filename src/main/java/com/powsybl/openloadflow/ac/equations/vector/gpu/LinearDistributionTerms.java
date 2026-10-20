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
 * A linear distribution equation as constant terms (OLF's DISTR_RHO, ratio distribution across several transformers
 * controlling one bus, and DISTR_SHUNT_B, its shunt-susceptance analogue): per term, the stable row of the equation
 * (a controller's BRANCH_RHO1 / SHUNT_B row), a controller's column, and the constant coefficient ({@code 1/count}
 * for a foreign controller, {@code 1/count - 1} for the row's own). Empty unless several controllers share one bus.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record LinearDistributionTerms(int[] row, int[] column, double[] coefficient) {

    public static LinearDistributionTerms empty() {
        return new LinearDistributionTerms(new int[0], new int[0], new double[0]);
    }

    public boolean sameAs(LinearDistributionTerms other) {
        return Arrays.equals(row, other.row) && Arrays.equals(column, other.column)
                && Arrays.equals(coefficient, other.coefficient);
    }
}
