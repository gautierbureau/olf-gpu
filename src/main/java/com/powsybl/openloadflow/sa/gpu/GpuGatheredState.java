/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

/**
 * The state rows of a scenario the host answers from them alone (P4 of {@code docs/device-final-results-plan.md}): those
 * its answer reads, gathered on the device as (row, value) pairs ({@link GpuStateHandback}). {@link #fill} writes them into
 * a scratch state whose every other row is NaN, which no comparison of the detection or of the realistic-voltage check
 * reports; {@link #clear} puts the NaNs back.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
record GpuGatheredState(int[] rows, double[] values) {

    void fill(double[] scratch) {
        for (int k = 0; k < rows.length; k++) {
            scratch[rows[k]] = values[k];
        }
    }

    void clear(double[] scratch) {
        for (int row : rows) {
            scratch[row] = Double.NaN;
        }
    }
}
