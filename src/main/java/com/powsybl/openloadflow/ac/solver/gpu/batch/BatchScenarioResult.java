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
 * the convergence status (see the {@code STATUS_*} constants in {@link GpuBatchStatus}), and the
 * contingency's active power loss pre-distribution as the batch replayed it ({@code preDistributedActivePower},
 * pu, what OLF's {@code DefaultContingencyActivePowerLossDistribution} returns; NaN when not replayed), and the
 * active power the device distributed in all ({@code distributedActivePower}, pu: that pre-distribution plus its own
 * distributed-slack passes — what OLF reports on a post-contingency result; NaN when not known).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record BatchScenarioResult(String contingencyId, double[] state, int iterations,
                                  double finalMismatch, int convergenceStatus, int[] islandedBusNums,
                                  double preDistributedActivePower, double distributedActivePower) {
    public BatchScenarioResult {
        Objects.requireNonNull(contingencyId);
        Objects.requireNonNull(state);
    }

    /** A result whose contingency loss pre-distribution was not replayed ({@code preDistributedActivePower}
     *  NaN: whoever needs it must ask OLF's own distribution). */
    public BatchScenarioResult(String contingencyId, double[] state, int iterations,
                               double finalMismatch, int convergenceStatus, int[] islandedBusNums) {
        this(contingencyId, state, iterations, finalMismatch, convergenceStatus, islandedBusNums, Double.NaN, Double.NaN);
    }

    public boolean hasConverged() {
        return convergenceStatus == GpuBatchStatus.STATUS_CONVERGED;
    }

    /** True if this contingency must fall back to the sequential CPU path. */
    public boolean needsCpuFallback() {
        return convergenceStatus == GpuBatchStatus.STATUS_REJECTED
                || convergenceStatus == GpuBatchStatus.STATUS_LINEAR_SOLVE_FAILED;
    }
}
