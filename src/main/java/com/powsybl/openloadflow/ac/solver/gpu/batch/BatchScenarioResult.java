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
 * <p>{@code reactiveLimitDecisions}: the scenario's PV/PQ decisions as its solve ended, when the request asked for them
 * ({@link BatchSolveRequest#reactiveLimitDecisions()}) and the network has reactive-limit entries; null otherwise.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record BatchScenarioResult(String contingencyId, double[] state, int iterations,
                                  double finalMismatch, int convergenceStatus, int[] islandedBusNums,
                                  double preDistributedActivePower, double distributedActivePower,
                                  /** The nums of the branches the scenario's AutomationSystem phase switched from their
                                   *  pre-contingency state (opened or closed): its systems' doing, not the contingency's. */
                                  int[] automationSystemSwitchedBranchNums,
                                  /** The taps the scenario's outer loops moved from the base, {branchNum, tap index,
                                   *  rounding r1} per transformer (DeviceTapMoves; empty where none moved). */
                                  double[] deviceTapMoves,
                                  /** The target P the scenario's distribution left each participating generator (null
                                   *  where not read: read for a contingency that has strategy pairs, under distributed
                                   *  slack). */
                                  GeneratorTargets generatorTargets,
                                  /** The scenario's PV/PQ decisions as its solve ended (see the class comment). */
                                  ReactiveLimitDecisions reactiveLimitDecisions,
                                  /** The solutions of the request's sensitivity basis at the scenario's converged state
                                   *  (SensitivityBasisRequest#resultLength values), null when not asked or not solved. */
                                  double[] sensitivityBasis) {

    /**
     * A scenario's reactive-limit decisions as its solve ended, per reactive-limit entry (the order of
     * {@code GpuAcData.reactiveLimits()}): its {@link com.powsybl.openloadflow.ac.solver.gpu.codes.GpuReactiveLimitsState}
     * ({@code finalState}), and the target its switch to PQ froze on its row ({@code frozenRowTargetQ}, the limit minus
     * the bus's load Q; NaN for an entry still PV or islanded, the whole array null when the device exported none). What
     * a caller hands to OLF's network so that its equation system is the one the scenario converged on.
     */
    public record ReactiveLimitDecisions(int[] finalState, double[] frozenRowTargetQ) {
        public ReactiveLimitDecisions {
            Objects.requireNonNull(finalState);
        }
    }

    public BatchScenarioResult {
        Objects.requireNonNull(contingencyId);
        Objects.requireNonNull(automationSystemSwitchedBranchNums);
        Objects.requireNonNull(deviceTapMoves);
        // state may be null: the chunk solver put it in the run's state sink (BatchConfig.stateSink) as its chunk completed
    }

    /** A result whose contingency loss pre-distribution was not replayed ({@code preDistributedActivePower}
     *  NaN: whoever needs it must ask OLF's own distribution). */
    public BatchScenarioResult(String contingencyId, double[] state, int iterations,
                               double finalMismatch, int convergenceStatus, int[] islandedBusNums) {
        this(contingencyId, state, iterations, finalMismatch, convergenceStatus, islandedBusNums, Double.NaN, Double.NaN, new int[0],
                new double[0], null, null, null);
    }

    /** The target P each participating generator ends a scenario with, by generator id (its distribution anchor plus
     *  the device's final move): what OLF's load flow leaves on the generators. */
    public record GeneratorTargets(String[] generatorIds, double[] targetP) {
        public GeneratorTargets {
            Objects.requireNonNull(generatorIds);
            Objects.requireNonNull(targetP);
        }
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
