/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * The branches the base case leaves open but the packs still hold, so a contingency or an action can close them:
 * the closed-branch family elements, and the zero-impedance family elements with their LfBranch numbers. A base-open
 * zero-impedance branch with an end bus out of the network is not packed: {@code identityRows} are its DUMMY_P / DUMMY_Q
 * rows, which only hold OLF's DUMMY_TARGET identities and which the structure pairs with themselves.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record BaseOpenBranches(int[] closedBranchElements, int[] zeroImpedanceElements, int[] zeroImpedanceBranchNums,
                               int[] identityRows) {

    public static BaseOpenBranches empty() {
        return new BaseOpenBranches(new int[0], new int[0], new int[0], new int[0]);
    }

    /** Same packed elements (the shared device structure depends on them; the branch numbers do not). */
    public boolean sameElementsAs(BaseOpenBranches other) {
        return Arrays.equals(closedBranchElements, other.closedBranchElements)
                && Arrays.equals(zeroImpedanceElements, other.zeroImpedanceElements)
                && Arrays.equals(identityRows, other.identityRows);
    }

    /** Each base-open zero-impedance branch's element, by LfBranch number. */
    public Map<Integer, Integer> zeroImpedanceElementByBranchNum() {
        Map<Integer, Integer> out = new HashMap<>();
        for (int z = 0; z < zeroImpedanceElements.length; z++) {
            out.put(zeroImpedanceBranchNums[z], zeroImpedanceElements[z]);
        }
        return out;
    }
}
