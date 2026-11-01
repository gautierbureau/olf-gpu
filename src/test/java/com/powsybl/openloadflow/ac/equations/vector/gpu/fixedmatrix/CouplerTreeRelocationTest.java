/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/**
 * Which buses' V rows host another bus's reactive balance, the input of the coupler-tree relocation's choice of the
 * buses that cannot be a tree's child ({@link CouplerTreeRelocation#spanningTreeRelocation}): counted per row, it must
 * answer what the pairwise definition answers.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class CouplerTreeRelocationTest {

    /** The definition: some other bus has its reactive balance on this bus's V row. */
    private static boolean[] pairwise(int nBuses, int[] qRow, int[] vCol) {
        boolean[] hosts = new boolean[nBuses];
        for (int other = 0; other < nBuses; other++) {
            for (int bn = 0; bn < nBuses; bn++) {
                if (bn != other && qRow[other] >= 0 && qRow[other] == vCol[bn]) {
                    hosts[bn] = true;
                }
            }
        }
        return hosts;
    }

    @Test
    void ownBalanceOnItsVRowIsNotAnothers() {
        // bus 0 keeps its balance on its own V row; bus 1's balance sits on bus 2's V row (a group's free controller
        // carrying the pilot's); bus 2 has no reactive row; bus 3 has no V column, and its balance's row is nobody's V row
        int[] qRow = {1, 5, -1, 7};
        int[] vCol = {1, 3, 5, -1};
        assertArrayEquals(new boolean[] {false, false, true, false}, CouplerTreeRelocation.hostsAnotherReactiveRow(4, qRow, vCol));
    }

    @Test
    void sharedRowsCountEveryOtherBus() {
        // two buses' balances on bus 0's V row, bus 0's own among them: still another's
        int[] qRow = {2, 2, -1};
        int[] vCol = {2, 4, 6};
        assertArrayEquals(new boolean[] {true, false, false}, CouplerTreeRelocation.hostsAnotherReactiveRow(3, qRow, vCol));
    }

    @Test
    void matchesThePairwiseDefinition() {
        Random random = new Random(20261101L);
        for (int trial = 0; trial < 500; trial++) {
            int nBuses = 1 + random.nextInt(30);
            int[] qRow = new int[nBuses];
            int[] vCol = new int[nBuses];
            for (int bn = 0; bn < nBuses; bn++) {
                // small ranges: unset rows (-1), collisions, V columns no balance reaches
                qRow[bn] = random.nextInt(nBuses + 2) - 1;
                vCol[bn] = random.nextInt(nBuses + 4) - 1;
            }
            assertArrayEquals(pairwise(nBuses, qRow, vCol), CouplerTreeRelocation.hostsAnotherReactiveRow(nBuses, qRow, vCol),
                    "trial " + trial);
        }
    }
}
