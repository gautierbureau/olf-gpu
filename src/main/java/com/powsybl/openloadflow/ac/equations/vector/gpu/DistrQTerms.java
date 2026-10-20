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
 * The DISTR_Q rows of remote generator voltage control as terms (OLF's DistributionReactivePowerEquationTerm,
 * {@code DISTR_Q_k = qPct_k * sum_j q_Gj - q_Gk}): per term, the family {@code element} and {@code side} of a
 * reactive injection of controller {@code injectionMember} (j), its {@link com.powsybl.openloadflow.ac.solver.gpu.codes.GpuReactiveTermKind},
 * the stable DISTR_Q {@code row} of controller {@code rowMember} (k, a controller's BUS_V row) and the {@code weight}
 * {@code qPct_k - (j == k ? 1 : 0)}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record DistrQTerms(int[] element, int[] side, double[] weight, int[] row, int[] kind,
                          int[] rowMember, int[] injectionMember) {

    public static DistrQTerms empty() {
        return new DistrQTerms(new int[0], new int[0], new double[0], new int[0], new int[0], new int[0], new int[0]);
    }

    public int size() {
        return element.length;
    }

    /** Same terms, the member indices aside (what the shared device structure is built from). */
    public boolean sameTermsAs(DistrQTerms other) {
        return Arrays.equals(element, other.element) && Arrays.equals(side, other.side)
                && Arrays.equals(weight, other.weight) && Arrays.equals(row, other.row) && Arrays.equals(kind, other.kind);
    }

    public boolean sameAs(DistrQTerms other) {
        return sameTermsAs(other) && Arrays.equals(rowMember, other.rowMember)
                && Arrays.equals(injectionMember, other.injectionMember);
    }
}
