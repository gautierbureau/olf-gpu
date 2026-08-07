/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import java.util.Objects;

/**
 * The per-scenario outcome: the converged state, iteration count, final mismatch and
 * the convergence status (see the {@code STATUS_*} constants in {@link GpuBatchStatus}).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record BatchScenarioResult(String contingencyId, double[] state, int iterations,
                                  double finalMismatch, int convergenceStatus, int[] islandedBusNums) {
    public BatchScenarioResult {
        Objects.requireNonNull(contingencyId);
        Objects.requireNonNull(state);
    }

    public boolean hasConverged() {
        return convergenceStatus == GpuBatchStatus.STATUS_CONVERGED;
    }

    /** True if this contingency must fall back to the sequential CPU path. */
    public boolean needsCpuFallback() {
        return convergenceStatus == GpuBatchStatus.STATUS_REJECTED;
    }
}
