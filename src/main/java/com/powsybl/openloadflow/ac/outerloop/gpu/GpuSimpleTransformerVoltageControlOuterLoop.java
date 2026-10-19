/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import java.util.List;

/**
 * DEVICE port of OLF's {@code SimpleTransformerVoltageControlOuterLoop} (the WITH_GENERATOR mode):
 * a fixed two-phase flow with no sensitivities —
 *
 * <ol>
 *   <li>{@code initialize()} enables CONTINUOUS voltage control on every connected transformer
 *       controller BEFORE the first solve: the scenario's first batch pass therefore runs with each
 *       controller's VC-ON activation deltas applied ({@link #applyVcOn} mutates the per-scenario
 *       row modes and targets, reusing the activation-diff rows captured by
 *       {@code GpuTransformerVoltageControlOuterLoop.buildTransformerControllers});</li>
 *   <li>the loop's first check ({@code roundVoltageRatios}) disables VC and rounds each continuous
 *       rho to the CLOSEST TAP → one warm re-solve with the VC-OFF (base) activation and the rho
 *       identity rows pinned at the rounded taps ({@link #roundAndResolve}); further checks are
 *       STABLE by construction.</li>
 * </ol>
 *
 * <p>OLF's {@code fixTransformerVoltageControls} (the no-PV-component connectivity guard) is applied
 * implicitly at its BASE-topology verdict: a controller it would disable never reaches the builder's
 * output; per-contingency divergence of that verdict is out of stage-1 scope (non-fragmenting
 * outages keep the main component intact, and fragmenting ones freeze their islands).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuSimpleTransformerVoltageControlOuterLoop {

    private GpuSimpleTransformerVoltageControlOuterLoop() {
    }

    /** Pre-solve mutation: apply every live controller's VC-ON activation deltas to each real
     *  scenario's row modes and target overrides (OLF's initialize() enables VC before the first
     *  solve). Returns the extended override lists to hand to the first batch pass. */
    static void applyVcOn(int realCount, int n, List<GpuTransformerVoltageControlOuterLoop.TransformerController> tcs,
                          int[][] rowModePerScenario, int[] baseRowMode, boolean[][] disabledMask,
                          double[] baseState, List<Integer> ovIndexList, List<Double> ovValueList) {
        for (int slot = 0; slot < realCount; slot++) {
            for (GpuTransformerVoltageControlOuterLoop.TransformerController tc : tcs) {
                boolean anySurvivor = false;                 // plan-A6: free-member outages re-pick on-device
                for (GpuTransformerVoltageControlOuterLoop.Member m : tc.members()) {
                    anySurvivor |= !disabledMask[slot][m.elemIndex()];
                }
                if (!anySurvivor) {
                    continue;
                }
                int[] row = rowModePerScenario[slot] == baseRowMode
                        ? baseRowMode.clone() : rowModePerScenario[slot];
                // rows owned by an OUTAGED member get their own override below — the same row must
                // NEVER carry two target overrides (parallel scatter, no ordering)
                java.util.Set<Integer> ownRows = new java.util.HashSet<>();
                for (GpuTransformerVoltageControlOuterLoop.Member m : tc.members()) {
                    if (disabledMask[slot][m.elemIndex()]) {
                        ownRows.add(m.rhoRow());
                    }
                }
                for (int k = 0; k < tc.deltaRows().length; k++) {
                    if (ownRows.contains(tc.deltaRows()[k])) {
                        continue;
                    }
                    row[tc.deltaRows()[k]] = tc.deltaModesOn()[k];
                    ovIndexList.add(slot * n + tc.deltaRows()[k]);
                    ovValueList.add(tc.deltaTargetsOn()[k]);
                }
                for (GpuTransformerVoltageControlOuterLoop.Member m : tc.members()) {
                    if (!disabledMask[slot][m.elemIndex()]) {
                        continue;
                    }
                    for (int k = 0; k < m.offDeltaRows().length; k++) {   // outaged NON-free member
                        row[m.offDeltaRows()[k]] = m.offDeltaModes()[k];
                        if (m.offDeltaRows()[k] != m.rhoRow()) {
                            ovIndexList.add(slot * n + m.offDeltaRows()[k]);
                            ovValueList.add(m.offDeltaTargets()[k]);
                        }
                    }
                    // pin the re-activated rho identity at the BASE-STATE rho — matching the warm
                    // start, so the row's residual is exactly 0 (the CPU REMOVES the equation; the
                    // captured stale target injected a UNIT residual and an extra Newton iterate)
                    ovIndexList.add(slot * n + m.rhoRow());
                    ovValueList.add(baseState[m.rhoRow()]);
                }
                rowModePerScenario[slot] = row;
            }
        }
    }

}
