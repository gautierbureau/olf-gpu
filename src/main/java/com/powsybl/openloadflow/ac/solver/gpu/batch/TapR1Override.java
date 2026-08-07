/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.network.LfBranch;

/**
 * The target tap's full static branch row [y, g1, g2, b1, b2, ksi, r1, a1] (cbIn packing),
 * replacing the element's row for one scenario — per-tap impedance changes included. The
 * branch's ratio and phase must be FIXED (not state variables): the batch qualification
 * rejects controlled transformers.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record TapR1Override(LfBranch branch, double[] cbRow) {
}
