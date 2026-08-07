/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.network.LfShunt;

/**
 * A per-scenario shunt admittance change (pu deltas ADDED to the packed g/b): section shifts
 * in either direction; full disconnection = minus the section admittance; reconnection = plus.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record ShuntShift(LfShunt shunt, double dg, double db) {
}
