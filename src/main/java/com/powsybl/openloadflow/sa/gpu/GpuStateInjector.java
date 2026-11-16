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
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.BatchContingency;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.EquationSystemIndex;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.util.gpu.GpuTrace;
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
     *
     * <p>Kept up to date from the index's published delta rather than rebuilt: on network.xiidm's 14,266-contingency
     * list the variable set changed for 9,464 of 12,726 injections, by 3.8 variables on average (a disabled bus's V
     * and PHI), and each change paid a full rebuild - a two-pointer merge over all ~21k variables, a fresh row array,
     * a fresh state array (about 2 GB of garbage a run), then a per-row gather. While the current list is the base
     * minus a few rows ({@link #subsequence}), the absent base rows are all the mapping needs: the injection copies
     * the segments between them. What the injection still costs is OLF's own index update for the same change.
     */
    static final class Remap {
        private List<Variable<AcVariableType>> vars;         // identity of the list the permutation was built from
        private int[] baseRowOf;                             // current row -> base row, -1 = variable absent at base (only when !subsequence)
        private double[] buf;                                // the state array handed to the equation system
        /** The current list is exactly the base list without {@link #absentRows}: no variable the base lacks. */
        private boolean subsequence;
        /** The base rows absent from the current list, ascending, and their variables (first {@link #absentCount}). */
        private int[] absentRows = new int[8];
        @SuppressWarnings("unchecked")
        private Variable<AcVariableType>[] absentVars = new Variable[8];
        private int absentCount;
        /** The index's variables-update generation the mapping matches, -1 before the first one. */
        private long generation = -1;
        /** State arrays by length: the list length only takes a few values (the base minus 2 per disabled bus), and the
         *  equation system keeps the array it is handed only until the next injection sets another. */
        private final Map<Integer, double[]> bufByLength = new java.util.HashMap<>();
        /** How many mapping changes took the delta path and how many the full rebuild (read by the tests). */
        int deltaUpdates;
        int fullRebuilds;
        /** base row per (elementNum, variable type), -1 = absent — the run-constant base snapshot in
         *  dense form. Applying a contingency invalidates the equation system's variable index about
         *  half the time (measured: 943 rebuilds over 2000 branch outages on rte6515), so the rebuild
         *  is not rare enough to leave on hash lookups. */
        private int[] baseRowKeyed;
        private int types;
        /** The base variables in base-row order: the current list is always a SUBSEQUENCE of it (a contingency only
         *  ever removes variables - a disabled bus's V and PHI - and the index sorts both lists the same way), so the
         *  row mapping is a two-pointer merge on REFERENCES, without reading a field of any of the 32k variables. */
        private Variable<AcVariableType>[] baseByRow;
    }

    /** Current row -> base row by merging the current sorted list into the base one on reference equality, or null
     *  when the current list is not a subsequence of the base (then the caller falls back to the keyed walk). Rows are
     *  list positions: the variables index assigns {@code row = position} (no separated ordering in this solver), which
     *  is checked on the two ends. */
    private static int[] mergeRows(List<Variable<AcVariableType>> vars, Variable<AcVariableType>[] baseByRow) {
        int n = vars.size();
        if (n > baseByRow.length || n == 0 || vars.get(0).getRow() != 0 || vars.get(n - 1).getRow() != n - 1) {
            return null;
        }
        int[] rows = new int[n];
        int i = 0;
        for (int j = 0; j < n; j++) {
            Variable<AcVariableType> v = vars.get(j);
            while (i < baseByRow.length && baseByRow[i] != v) {
                i++;
            }
            if (i == baseByRow.length) {
                return null;
            }
            rows[j] = i++;
        }
        return rows;
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
        // VARIABLES-ONLY index update: the injected state is read back through variable rows (flows, voltages) and
        // never solved, so rebuilding the equations-to-solve list here is wasted - the contingency's apply and the
        // restore invalidate it every time, and on network.xiidm that rebuild was ~2.3 s of the 8.3 s this call
        // spent inside OLF's index. The equation side stays lazy for the next solver that needs it.
        EquationSystemIndex<AcVariableType, AcEquationType> index = es.getIndex();
        List<Variable<AcVariableType>> vars = index.getSortedVariablesToFindOnly();
        if (remap.vars != vars) {                            // identity: the variable set changed
            if (remap.baseRowKeyed == null) {
                remap.types = AcVariableType.values().length;
                remap.baseRowKeyed = buildKeyedBaseRows(baseVariableRows, remap.types);
            }
            if (remap.baseByRow == null) {
                @SuppressWarnings("unchecked")
                Variable<AcVariableType>[] byRow = new Variable[baseVariableRows.size()];
                baseVariableRows.forEach((v, row) -> byRow[row] = v);
                remap.baseByRow = byRow;
            }
            long generation = index.getVariablesUpdateGeneration();
            // the delta describes the change from the list the mapping was made on only when exactly one update ran
            // since then, and an incremental one; anything else (a CPU replay's solve in between, a full rebuild)
            // goes through the full merge
            boolean delta = remap.subsequence && remap.generation >= 0 && generation == remap.generation + 1
                    && index.isLastVariablesUpdateIncremental()
                    && applyRemovedDelta(remap, index.getLastRemovedVariables(), vars.size());
            if (delta) {
                remap.deltaUpdates++;
            } else {
                rebuildFully(remap, vars);
                remap.fullRebuilds++;
            }
            if (GpuTrace.PHASE_TIME.isOn()) {
                GpuBatchedAcSecurityAnalysis.PHASE_NS.computeIfAbsent(delta ? "    inject: remap delta updates (count)"
                                : "    inject: remap full rebuilds (count)",
                        k -> new java.util.concurrent.atomic.LongAdder()).add(1_000_000);   // printed in "ms" = count
            }
            remap.generation = generation;
            remap.buf = remap.bufByLength.computeIfAbsent(vars.size(), double[]::new);
            remap.vars = vars;
        }
        double[] mapped = remap.buf;
        if (remap.subsequence) {
            // the current rows are the base rows without the absent ones, in the same order: copy the segments
            // between them (one copy when nothing is absent, the outage islanded nothing)
            int src = 0;
            int dst = 0;
            for (int k = 0; k < remap.absentCount; k++) {
                int absentRow = remap.absentRows[k];
                System.arraycopy(deviceState, src, mapped, dst, absentRow - src);
                dst += absentRow - src;
                src = absentRow + 1;
            }
            System.arraycopy(deviceState, src, mapped, dst, mapped.length - dst);
        } else {
            int[] baseRowOf = remap.baseRowOf;
            for (int row = 0; row < baseRowOf.length; row++) {
                int baseRow = baseRowOf[row];
                mapped[row] = baseRow >= 0 ? deviceState[baseRow] : 0;
            }
        }
        // VALUES ONLY: the injected state feeds result building (flows, currents, voltages) and never a
        // Jacobian — the next solve on this equation system starts with its own full state set. That
        // skips refreshing every branch's partial derivatives, which measured 993 ms of the 1119 ms
        // this notification cost over 2000 contingencies.
        setValues(es, mapped, closedBranchNums);
    }

    /** Brings the absent base rows up to date with one incremental index update, or returns false when the delta does
     *  not describe the new list (the caller then rebuilds fully). A tracked variable that has a row again was re-added;
     *  every variable the update removed is reported, and its base row joins the absent set. The tracked set is then
     *  exact for the base variables, so the list is the base without it iff its size is the base's minus the set's - a
     *  variable the base lacks (a strategy closing a coupler) makes the sizes differ. */
    private static boolean applyRemovedDelta(Remap remap, List<Variable<AcVariableType>> removed, int listSize) {
        int count = 0;
        for (int k = 0; k < remap.absentCount; k++) {
            if (remap.absentVars[k].getRow() < 0) {          // still absent: keep, in order
                remap.absentVars[count] = remap.absentVars[k];
                remap.absentRows[count++] = remap.absentRows[k];
            }
        }
        for (Variable<AcVariableType> v : removed) {
            int key = v.getElementNum() * remap.types + v.getType().ordinal();
            int baseRow = key < remap.baseRowKeyed.length ? remap.baseRowKeyed[key] : -1;
            if (baseRow < 0 || remap.baseByRow[baseRow] != v) {
                remap.absentCount = 0;
                return false;                                // not a base variable: the bookkeeping cannot follow
            }
            int pos = count;                                 // insertion into the ascending absent rows
            while (pos > 0 && remap.absentRows[pos - 1] > baseRow) {
                pos--;
            }
            if (pos > 0 && remap.absentRows[pos - 1] == baseRow) {
                continue;                                    // already absent
            }
            if (count == remap.absentRows.length) {
                remap.absentRows = java.util.Arrays.copyOf(remap.absentRows, 2 * count);
                remap.absentVars = java.util.Arrays.copyOf(remap.absentVars, 2 * count);
            }
            System.arraycopy(remap.absentRows, pos, remap.absentRows, pos + 1, count - pos);
            System.arraycopy(remap.absentVars, pos, remap.absentVars, pos + 1, count - pos);
            remap.absentRows[pos] = baseRow;
            remap.absentVars[pos] = v;
            count++;
        }
        remap.absentCount = count;
        if (listSize != remap.baseByRow.length - count) {
            remap.absentCount = 0;
            return false;
        }
        return true;
    }

    /** The full mapping: a two-pointer merge into the base list, which also yields the absent base rows when the
     *  current list is a subsequence of it, else a keyed walk (a variable the base lacks). */
    private static void rebuildFully(Remap remap, List<Variable<AcVariableType>> vars) {
        int[] rows = mergeRows(vars, remap.baseByRow);
        remap.absentCount = 0;
        if (rows != null) {
            remap.subsequence = true;
            remap.baseRowOf = null;
            int next = 0;                                    // base rows the merge skipped are the absent ones
            for (int row : rows) {
                for (; next < row; next++) {
                    addAbsent(remap, next);
                }
                next = row + 1;
            }
            for (; next < remap.baseByRow.length; next++) {
                addAbsent(remap, next);
            }
            return;
        }
        int[] keyed = remap.baseRowKeyed;
        int types = remap.types;
        rows = new int[vars.size()];
        for (Variable<AcVariableType> v : vars) {
            int key = v.getElementNum() * types + v.getType().ordinal();
            rows[v.getRow()] = key < keyed.length ? keyed[key] : -1;
        }
        remap.subsequence = false;
        remap.baseRowOf = rows;
    }

    private static void addAbsent(Remap remap, int baseRow) {
        if (remap.absentCount == remap.absentRows.length) {
            remap.absentRows = java.util.Arrays.copyOf(remap.absentRows, 2 * remap.absentCount);
            remap.absentVars = java.util.Arrays.copyOf(remap.absentVars, 2 * remap.absentCount);
        }
        remap.absentRows[remap.absentCount] = baseRow;
        remap.absentVars[remap.absentCount++] = remap.baseByRow[baseRow];
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
        boolean busDumpTrace = "1".equals(GpuTrace.DBG_BUSDUMP.value());
        for (int num : nums) {
            LfBus b = network.getBuses().get(num);
            b.createBusResults().forEach(busResult -> islandedBusIds.add(busResult.getBusId()));
        }
        var filtered = result.getBusResults().stream().filter(busResult -> !islandedBusIds.contains(busResult.getBusId())).toList();
        if (busDumpTrace) {
            System.err.println("FILTER-ISLANDED '" + scenarioId.replace(BatchContingency.STRATEGY_KEY_SEP, '|')
                    + "' ids=" + islandedBusIds + " " + result.getBusResults().size() + "->" + filtered.size());
        }
        return new NetworkResult(result.getBranchResults(), filtered,
                result.getThreeWindingsTransformerResults());
    }
}
