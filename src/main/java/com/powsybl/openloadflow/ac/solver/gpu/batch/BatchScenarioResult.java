/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuAcSolverStatus;
import java.util.Objects;

/**
 * The per-scenario outcome: the converged state, iteration count, final mismatch and
 * the convergence status (see the constants in {@link com.powsybl.openloadflow.ac.solver.gpu.codes.GpuAcSolverStatus}), and the
 * contingency's active power loss pre-distribution as the batch replayed it ({@code preDistributedActivePower},
 * pu, what OLF's {@code DefaultContingencyActivePowerLossDistribution} returns; NaN when not replayed), and the
 * active power the device distributed in all ({@code distributedActivePower}, pu: that pre-distribution plus its own
 * distributed-slack passes — what OLF reports on a post-contingency result; NaN when not known).
 *
 * <p>{@code state} holds the {@code n} variable values in its FIRST {@code n} entries and may be longer: a device
 * scenario's is the solver's own result row, with the iteration count, final mismatch and status after the state,
 * handed over as is instead of copied (on rte6515 a copy is 14,260 x 13,040 doubles, ~0.5 s of allocation). Read
 * it by row, or take {@code Arrays.copyOf(state, n)} where a length-{@code n} array is required.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record BatchScenarioResult(String contingencyId, double[] state, int iterations,
                                  double finalMismatch, int convergenceStatus, int[] islandedBusNums,
                                  double preDistributedActivePower, double distributedActivePower,
                                  /** The nums of the branches the scenario's AutomationSystem phase switched from their
                                   *  pre-contingency state (opened or closed): its systems' doing, not the contingency's. */
                                  int[] automationSystemSwitchedBranchNums) {
    public BatchScenarioResult {
        Objects.requireNonNull(contingencyId);
        Objects.requireNonNull(automationSystemSwitchedBranchNums);
        // state may be null: the chunk solver put it in the run's state sink (BatchConfig.stateSink) as its chunk completed
    }

    /** A result whose contingency loss pre-distribution was not replayed ({@code preDistributedActivePower}
     *  NaN: whoever needs it must ask OLF's own distribution). */
    public BatchScenarioResult(String contingencyId, double[] state, int iterations,
                               double finalMismatch, int convergenceStatus, int[] islandedBusNums) {
        this(contingencyId, state, iterations, finalMismatch, convergenceStatus, islandedBusNums, Double.NaN, Double.NaN, new int[0]);
    }

    public boolean hasConverged() {
        return convergenceStatus == GpuAcSolverStatus.CONVERGED;
    }

    /** True if this contingency must fall back to the sequential CPU path. */
    public boolean needsCpuFallback() {
        return convergenceStatus == GpuAcSolverStatus.REJECTED
                || convergenceStatus == GpuAcSolverStatus.LINEAR_SOLVE_FAILED;
    }
}
