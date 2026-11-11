/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcElementMapper;
import com.powsybl.openloadflow.ac.solver.gpu.batch.BatchScenarioResult;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.BatchContingency;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.Variable;

import java.util.Map;

/**
 * The output of {@link GpuBatchPrecomputer#precompute}: the ONLY coupling between the batch precompute
 * and the per-contingency injection. Holds the converged device state per scenario id
 * ({@code states}), the islanded LfBus nums to exclude from each scenario's results
 * ({@code islandedBuses}), the BASE variable-row snapshot the injection remaps through
 * ({@code baseVariableRows}), and the engine's equation system the states were solved against
 * ({@code equationSystem} — {@code null} when nothing was precomputed, the CPU-fallback signal), and
 * per converged scenario id the contingency loss pre-distribution the batch replayed
 * ({@code preDistributedActivePower}, pu — absent where it was not replayed), the total the device distributed
 * ({@code distributedActivePower}, pu — pre-distribution plus its own distributed-slack passes; absent where unknown)
 * the nums of the branches its AutomationSystem phase switched ({@code automationSystemSwitched} — absent where none),
 * the taps its outer loops moved from the base ({@code deviceTapMoves}, DeviceTapMoves' {branchNum, tap index,
 * rounding r1} triples — absent where none moved) and the target P its distribution left each participating generator
 * ({@code generatorTargets} — present for a contingency that has strategy pairs, under distributed slack).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
record PrecomputedBatch(Map<String, double[]> states, Map<String, int[]> islandedBuses,
                        /** Contingencies the device carried through the whole Newton budget without
                         *  converging: reported as MAX_ITERATION_REACHED with an empty network result,
                         *  exactly what OLF's own post-contingency path reports (it builds the network
                         *  result and detects violations only for CONVERGED). A REJECTED scenario
                         *  (status -2: an extractor gate or the straggler exit) is not in here and
                         *  still replays on the CPU. */
                        java.util.Set<String> exhausted,
                        Map<Variable<AcVariableType>, Integer> baseVariableRows,
                        EquationSystem<AcVariableType, AcEquationType> equationSystem,
                        com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData gpuData,
                        GpuAcElementMapper elementMapper,
                        Map<String, Double> preDistributedActivePower, Map<String, Double> distributedActivePower,
                        Map<String, int[]> automationSystemSwitched,
                        Map<String, double[]> deviceTapMoves,
                        Map<String, BatchScenarioResult.GeneratorTargets> generatorTargets,
                        Map<String, BatchContingency> scenarios,
                        /** Per screened scenario, its candidates, screened on the device as it was handed back
                         *  (GpuStateHandback); absent for one screened on the host (an overflow, no device table). */
                        Map<String, int[]> candidates,
                        /** The scenarios the host answers from their state rows alone, with those rows: no full state for
                         *  them in {@code states} (except while the comparison is on). */
                        Map<String, GpuGatheredState> gathered,
                        /** The limit table and the screen the candidates index, null without the handback. */
                        GpuLimitTable limitTable, GpuViolationScreen violationScreen) {

    /** Nothing precomputed — every contingency falls back to the sequential CPU path. */
    static PrecomputedBatch empty() {
        return new PrecomputedBatch(Map.of(), Map.of(), java.util.Set.of(), Map.of(), null, null, null, Map.of(), Map.of(), Map.of(), Map.of(),
                Map.of(), Map.of(), Map.of(), Map.of(), null, null);
    }

    /** True when the device converged scenario {@code id}: its full state, or the rows the host answers it with. */
    boolean answered(String id) {
        return states.containsKey(id) || gathered.containsKey(id);
    }

    /** The scenarios the device converged. */
    java.util.Set<String> answeredIds() {
        java.util.Set<String> ids = new java.util.HashSet<>(states.keySet());
        ids.addAll(gathered.keySet());
        return ids;
    }

    /** True when the batch precompute reached the device solve (an equation system is captured). */
    boolean isReady() {
        return equationSystem != null;
    }
}
