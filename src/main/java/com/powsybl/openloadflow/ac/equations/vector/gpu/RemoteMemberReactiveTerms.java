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
 * A remote voltage control member's OWN reactive injection, term by term (plan-B3): per term, the member's own row,
 * the family element, its side, its {@code GpuReactiveTermKind} (as in the DISTR_Q pack) and a weight (1, or +-1
 * for a zero-impedance DUMMY_Q term). A twin of the DISTR_Q assembly accumulates each member's own bus q into the
 * diagnostic Q buffer at its own row (free, because the main fill suppresses member rows), so the reactive limits
 * kernel's {@code q = qBuf[row] + loadQ} works verbatim for remote entries.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record RemoteMemberReactiveTerms(int[] row, int[] element, int[] side, int[] kind, double[] weight) {

    public static RemoteMemberReactiveTerms empty() {
        return new RemoteMemberReactiveTerms(new int[0], new int[0], new int[0], new int[0], new double[0]);
    }

    public boolean sameAs(RemoteMemberReactiveTerms other) {
        return Arrays.equals(row, other.row) && Arrays.equals(element, other.element)
                && Arrays.equals(side, other.side) && Arrays.equals(kind, other.kind)
                && Arrays.equals(weight, other.weight);
    }
}
