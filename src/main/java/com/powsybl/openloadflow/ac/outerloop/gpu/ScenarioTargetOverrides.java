/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

/**
 * The per-scenario equation-target overrides a batched outer loop seeds before the device solve, as the loop sees
 * them. A row is addressed flat, {@code scenario * rowCount + row}, and carries at most one override.
 *
 * <p>The batch solver owns the list (its index, snapshots and reset); a loop only writes into it, so it gets this
 * view and not the list itself.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public interface ScenarioTargetOverrides {

    /** Set the override of {@code flat} to {@code v}. A row that already has one gets its value replaced: the
     *  last write wins. */
    void add(int flat, double v);

    /** The row's current override (or {@code baseTarget} when it has none) minus {@code baseD} plus {@code d}. */
    void merge(int flat, double baseTarget, double baseD, double d);
}
