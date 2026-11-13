/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

/**
 * The per-scenario row modes a batched outer loop or a slot's seeding reads and sets before the device solve: the base
 * row's modes, except the rows a scenario sets.
 *
 * <p>The batch solver owns the store (its reset and the entries it hands to the device); a loop only reads and sets
 * rows, so it gets this view and not the store itself.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public interface ScenarioRowModes {

    /** The mode of {@code row} in {@code scenario}: the last one set, or the base row's. */
    int get(int scenario, int row);

    /** Set the mode of {@code row} in {@code scenario}. A row set again gets the later mode: the last write wins. */
    void set(int scenario, int row, int mode);
}
