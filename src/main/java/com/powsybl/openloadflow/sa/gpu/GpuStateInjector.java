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
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchStatus;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.security.results.NetworkResult;

import java.util.Map;

/**
 * Injects a precomputed device state into the (possibly contingency-mutated) OLF equation system and
 * filters islanded buses out of the reported results — the OLF-only half of the GPU SA integration,
 * with no solver dependency. State injection remaps row-by-row through the BASE variable-row snapshot
 * so a disabled islanded bus's deactivated variables cannot shift its neighbours; result filtering
 * mirrors the CPU solve's connectivity-driven bus exclusion purely in result space.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuStateInjector {

    private GpuStateInjector() {
    }

    /** Injects a device state (BASE row layout) into the possibly contingency-mutated equation
     *  system: rows are remapped variable-by-variable through the base snapshot, so a disabled
     *  islanded bus's deactivated variables cannot shift its neighbors. */
    static void injectState(EquationSystem<AcVariableType, AcEquationType> es,
                            Map<Variable<AcVariableType>, Integer> baseVariableRows, double[] deviceState) {
        var vars = es.getIndex().getSortedVariablesToFind();
        double[] mapped = new double[vars.size()];
        for (var v : vars) {
            Integer baseRow = baseVariableRows.get(v);
            mapped[v.getRow()] = baseRow != null ? deviceState[baseRow] : 0;
        }
        es.getStateVector().set(mapped);
    }

    /** Mirrors the CPU solve's connectivity processing for an injected state: islanded buses (frozen
     *  to a clean flat state on the device) must be EXCLUDED from the results the way the CPU's
     *  connectivity-driven disabling excludes them. NEVER by LfBus.setDisabled on the live network —
     *  that fires the equation-system listeners, deactivates rows and shifts the state mapping for
     *  every subsequent injection (measured: the NEXT scenario's bus read its neighbor's slot).
     *  Pure result-space filtering instead. */
    static NetworkResult filterIslanded(LfNetwork network, int[] nums, String scenarioId, NetworkResult result) {
        if (nums == null) {
            return result;
        }
        java.util.Set<String> islandedBusIds = new java.util.HashSet<>();
        boolean dbg = "1".equals(System.getenv("OLF_GPU_DBG_BUSDUMP"));
        for (int num : nums) {
            LfBus b = network.getBuses().get(num);
            b.createBusResults().forEach(br -> islandedBusIds.add(br.getBusId()));
        }
        var filtered = result.getBusResults().stream().filter(br -> !islandedBusIds.contains(br.getBusId())).toList();
        if (dbg) {
            System.err.println("FILTER-ISLANDED '" + scenarioId.replace(GpuBatchStatus.STRATEGY_KEY_SEP, '|')
                    + "' ids=" + islandedBusIds + " " + result.getBusResults().size() + "->" + filtered.size());
        }
        return new NetworkResult(result.getBranchResults(), filtered,
                result.getThreeWindingsTransformerResults());
    }
}
