/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix;

import java.util.Arrays;

/**
 * Where a bus's reactive balance may be relocated per scenario (docs/gpu-vs-olf.md section 2.3: the carrier): the
 * candidate groups, each a run of candidate rows (CSR: group g holds {@code candidateRows[groupStart[g]]} to
 * {@code candidateRows[groupStart[g + 1] - 1]}), and the dependent-pilot links (ExtractionColumns): per group its
 * base group or -1, per group the base row of the pilot's reactive target or -1, per device row the dependent group of
 * the bus owning that V row or -1.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record ReactiveRelocationCandidates(int[] groupStart, int[] candidateRows, int[] dependentBaseGroup,
                                           int[] baseTargetRow, int[] dependentGroupOfRow) {

    public static ReactiveRelocationCandidates empty() {
        return new ReactiveRelocationCandidates(new int[] {0}, new int[0], new int[0], new int[0], new int[0]);
    }

    public boolean sameAs(ReactiveRelocationCandidates other) {
        return sameGroupsAs(other) && Arrays.equals(dependentBaseGroup, other.dependentBaseGroup)
                && Arrays.equals(baseTargetRow, other.baseTargetRow)
                && Arrays.equals(dependentGroupOfRow, other.dependentGroupOfRow);
    }

    /** Same candidate groups (the dependent-pilot links aside). */
    public boolean sameGroupsAs(ReactiveRelocationCandidates other) {
        return Arrays.equals(groupStart, other.groupStart) && Arrays.equals(candidateRows, other.candidateRows);
    }
}
