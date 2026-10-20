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
 * Per-row parameters of the bus voltage rows, indexed by row (0 on every other row):
 * <ul>
 *   <li>{@code slope}: the slope of a static var compensator's BUS_TARGET_V_WITH_SLOPE equation
 *       ({@code V + slope * sum Q = target});</li>
 *   <li>{@code nominalV}: the bus's nominal voltage, for OLF's realistic-voltage check
 *       (AcloadFlowEngine.isStateUnrealistic walks the BUS_V of non-fictitious buses; 0 = a row it never checks).</li>
 * </ul>
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record VoltageRowParameters(double[] slope, double[] nominalV) {

    public static VoltageRowParameters empty() {
        return new VoltageRowParameters(new double[0], new double[0]);
    }

    public boolean sameAs(VoltageRowParameters other) {
        return Arrays.equals(slope, other.slope) && Arrays.equals(nominalV, other.nominalV);
    }
}
