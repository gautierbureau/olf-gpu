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

import java.util.List;
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

    /**
     * The row remap, kept across contingencies instead of rebuilt for each one.
     *
     * <p>The mapping itself cannot be hoisted out of the loop: the equation system is
     * CONTINGENCY-MUTATED, and an outage that islands a bus deactivates that bus's variables, which
     * renumbers every row after them — remapping variable-by-variable through the base snapshot is
     * what stops an islanded bus shifting its neighbours' states. But it only CHANGES when the
     * variable set does, which is the rare case: a plain branch outage removes no variable.
     *
     * <p>{@code EquationSystemIndex} caches {@code sortedVariablesToFind} and rebuilds it only when
     * a variable is added or removed, so the list's IDENTITY is an exact, free invalidation signal —
     * same instance means same variables in the same rows, and the permutation still holds. Without
     * this the injector did one hash lookup per variable per contingency (13040 x 2000 = 26 million
     * on rte6515) and allocated a fresh state array each time (2000 x 104 KB of garbage).
     *
     * <p>Owned by the partition's RunState, never static: partitions run concurrently on their own
     * equation systems, and a shared cache would hand one partition another's permutation.
     */
    static final class Remap {
        private List<Variable<AcVariableType>> vars;         // identity of the list the permutation was built from
        private int[] baseRowOf;                             // current row -> base row, -1 = variable absent at base
        private boolean identity;                            // no variable was removed: row == base row throughout
        private double[] buf;                                // the state array handed to the equation system
        /** base row per (elementNum, variable type), -1 = absent — the run-constant base snapshot in
         *  dense form. Applying a contingency invalidates the equation system's variable index about
         *  half the time (measured: 943 rebuilds over 2000 branch outages on rte6515), so the rebuild
         *  is not rare enough to leave on hash lookups. */
        private int[] baseRowKeyed;
        private int types;
    }

    private static int[] buildKeyedBaseRows(Map<Variable<AcVariableType>, Integer> baseVariableRows, int types) {
        int maxElem = 0;
        for (Variable<AcVariableType> v : baseVariableRows.keySet()) {
            maxElem = Math.max(maxElem, v.getElementNum());
        }
        int[] keyed = new int[(maxElem + 1) * types];
        java.util.Arrays.fill(keyed, -1);
        baseVariableRows.forEach((v, row) -> keyed[v.getElementNum() * types + v.getType().ordinal()] = row);
        return keyed;
    }

    /** Injects a device state (BASE row layout) into the possibly contingency-mutated equation
     *  system: rows are remapped variable-by-variable through the base snapshot, so a disabled
     *  islanded bus's deactivated variables cannot shift its neighbors. */
    static void injectState(EquationSystem<AcVariableType, AcEquationType> es,
                            Map<Variable<AcVariableType>, Integer> baseVariableRows, double[] deviceState,
                            Remap remap) {
        injectState(es, baseVariableRows, deviceState, remap, null);
    }

    /** As above; with {@code closedBranchNums} non-null only those branches' flows are refreshed — for a caller that
     *  reads no other branch before the next state set (StateVector.setValuesOnly(double[], int[])). */
    static void injectState(EquationSystem<AcVariableType, AcEquationType> es,
                            Map<Variable<AcVariableType>, Integer> baseVariableRows, double[] deviceState,
                            Remap remap, int[] closedBranchNums) {
        List<Variable<AcVariableType>> vars = es.getIndex().getSortedVariablesToFind();
        if (remap.vars != vars) {                            // identity: the variable set changed, rebuild
            if (System.getenv("OLF_GPU_PHASE_TIME") != null) {
                GpuBatchedAcSecurityAnalysis.PHASE_NS.computeIfAbsent("    inject: remap rebuilds (count)",
                        k -> new java.util.concurrent.atomic.LongAdder()).add(1_000_000);   // printed in "ms" = count
            }
            if (remap.baseRowKeyed == null) {
                remap.types = AcVariableType.values().length;
                remap.baseRowKeyed = buildKeyedBaseRows(baseVariableRows, remap.types);
            }
            int[] keyed = remap.baseRowKeyed;
            int types = remap.types;
            int[] rows = new int[vars.size()];
            for (Variable<AcVariableType> v : vars) {
                int key = v.getElementNum() * types + v.getType().ordinal();
                rows[v.getRow()] = key < keyed.length ? keyed[key] : -1;
            }
            boolean identity = true;
            for (int row = 0; row < rows.length && identity; row++) {
                identity = rows[row] == row;
            }
            remap.baseRowOf = rows;
            remap.identity = identity;
            remap.buf = new double[vars.size()];
            remap.vars = vars;
        }
        if (remap.identity) {                                // the common case: the outage islanded nothing,
            System.arraycopy(deviceState, 0, remap.buf, 0, remap.buf.length);   // so the rows still line up
            setValues(es, remap.buf, closedBranchNums);
            return;
        }
        int[] baseRowOf = remap.baseRowOf;
        double[] mapped = remap.buf;
        for (int row = 0; row < baseRowOf.length; row++) {
            int baseRow = baseRowOf[row];
            mapped[row] = baseRow >= 0 ? deviceState[baseRow] : 0;
        }
        // VALUES ONLY: the injected state feeds result building (flows, currents, voltages) and never a
        // Jacobian — the next solve on this equation system starts with its own full state set. That
        // skips refreshing every branch's partial derivatives, which measured 993 ms of the 1119 ms
        // this notification cost over 2000 contingencies.
        setValues(es, mapped, closedBranchNums);
    }

    private static void setValues(EquationSystem<AcVariableType, AcEquationType> es, double[] values, int[] closedBranchNums) {
        if (closedBranchNums != null) {
            es.getStateVector().setValuesOnly(values, closedBranchNums);
        } else {
            es.getStateVector().setValuesOnly(values);
        }
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
