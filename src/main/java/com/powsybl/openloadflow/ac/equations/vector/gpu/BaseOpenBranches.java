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
 * the closed-branch family elements, and the zero-impedance family elements with their LfBranch numbers.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record BaseOpenBranches(int[] closedBranchElements, int[] zeroImpedanceElements, int[] zeroImpedanceBranchNums) {

    public static BaseOpenBranches empty() {
        return new BaseOpenBranches(new int[0], new int[0], new int[0]);
    }

    /** Same packed elements (the shared device structure depends on them; the branch numbers do not). */
    public boolean sameElementsAs(BaseOpenBranches other) {
        return Arrays.equals(closedBranchElements, other.closedBranchElements)
                && Arrays.equals(zeroImpedanceElements, other.zeroImpedanceElements);
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
