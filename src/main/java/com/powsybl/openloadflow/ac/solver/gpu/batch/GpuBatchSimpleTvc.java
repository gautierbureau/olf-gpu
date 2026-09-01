/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;

import java.util.ArrayList;
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
 *       {@code GpuBatchTransformerControl.buildTransformerControllers});</li>
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
final class GpuBatchSimpleTvc {

    private GpuBatchSimpleTvc() {
    }

    /** Pre-solve mutation: apply every live controller's VC-ON activation deltas to each real
     *  scenario's row modes and target overrides (OLF's initialize() enables VC before the first
     *  solve). Returns the extended override lists to hand to the first batch pass. */
    static void applyVcOn(int realCount, int n, List<GpuBatchTransformerControl.TransformerController> tcs,
                          int[][] rowModePerScenario, int[] baseRowMode, boolean[][] disabledMask,
                          List<Integer> ovIndexList, List<Double> ovValueList) {
        for (int slot = 0; slot < realCount; slot++) {
            for (GpuBatchTransformerControl.TransformerController tc : tcs) {
                if (disabledMask[slot][tc.elemIndex()]) {
                    continue;                                // the controller transformer itself is outaged
                }
                int[] row = rowModePerScenario[slot] == baseRowMode
                        ? baseRowMode.clone() : rowModePerScenario[slot];
                for (int k = 0; k < tc.deltaRows().length; k++) {
                    row[tc.deltaRows()[k]] = tc.deltaModesOn()[k];
                    ovIndexList.add(slot * n + tc.deltaRows()[k]);
                    ovValueList.add(tc.deltaTargetsOn()[k]);
                }
                rowModePerScenario[slot] = row;
            }
        }
    }

    /** Post-solve rounding: read each scenario's continuous rho, round to the closest tap, restore
     *  the VC-OFF (base) activation and re-solve warm with the rho identity rows pinned at the
     *  rounded values and the rho states seeded there (the CPU re-reads r1 from the pi model). */
    @SuppressWarnings("java:S107")
    static double[][] roundAndResolve(long handle, int chunkSize, int realCount, int n,
            double[][] chunkResults, List<GpuBatchTransformerControl.TransformerController> tcs,
            int[][] rowModePerScenario, int[] baseRowMode, boolean[][] disabledMask,
            List<Integer> baseOvIndexList, List<Double> baseOvValueList,
            int[] dsOvBus, double[] dsOvFactorDelta, double[] dsOvBaseDelta, double[] dsOvMaxDelta, double[] dsOvMinDelta,
            double[] dsAnchorShift, double[] dsSlackShift,
            int[] r1OvSlotElem, double[] r1OvVal, int[] shOvSlotElem, double[] shOvGb,
            int nziBase, int[] ziOnSlotElem, int[] ziOffSlotElem,
            double[] baseState, double[] baseTargets, NewtonRaphsonParameters parameters,
            double tol, double reactiveLimitTol, double slackDistTol, double convEpsPerEq, int[] varType,
            BatchConfig config) {
        // rebuild the VC-OFF overrides: the ORIGINAL (pre-VC-on) list + one rho pin per live controller
        List<Integer> ovIdx = new ArrayList<>(baseOvIndexList);
        List<Double> ovVal = new ArrayList<>(baseOvValueList);
        double[] x0Full = new double[chunkSize * n];
        int[][] rowModeOff = new int[chunkSize][];
        boolean[] affected = new boolean[chunkSize];
        for (int s = 0; s < chunkSize; s++) {
            double[] src = s < realCount && chunkResults[s] != null ? chunkResults[s] : baseState;
            System.arraycopy(src, 0, x0Full, s * n, n);
            rowModeOff[s] = baseRowMode;                     // VC disabled again — the base activation
        }
        // scenarios with frozen rows / gen-outage row flips carried their OWN row array: rebuild it
        // from the base by re-applying every non-TVC difference (the VC-on deltas are dropped)
        for (int s = 0; s < realCount; s++) {
            if (chunkResults[s] == null || (int) chunkResults[s][n + 2] != GpuBatchStatus.STATUS_CONVERGED) {
                continue;
            }
            int[] row = null;
            for (GpuBatchTransformerControl.TransformerController tc : tcs) {
                if (disabledMask[s][tc.elemIndex()]) {
                    continue;
                }
                if (row == null) {
                    row = rowModePerScenario[s].clone();
                }
                for (int k = 0; k < tc.deltaRows().length; k++) {
                    row[tc.deltaRows()[k]] = baseRowMode[tc.deltaRows()[k]];   // undo the VC-on flip
                }
                double rounded = tc.closestTap(chunkResults[s][tc.rhoRow()]);  // OLF roundR1ToClosestTap
                ovIdx.add(s * n + tc.rhoRow());
                ovVal.add(rounded);                          // the identity row pins rho at the rounded tap
                x0Full[s * n + tc.rhoRow()] = rounded;
                affected[s] = true;
            }
            if (row != null) {
                rowModeOff[s] = row;
            }
        }
        boolean any = false;
        for (boolean a : affected) {
            any |= a;
        }
        if (!any) {
            return chunkResults;
        }
        double[][] r2 = GpuAcNewtonSolver.solveBatchContext(handle, baseState, baseTargets,
                rowModeOff, baseRowMode, disabledMask,
                ovIdx.stream().mapToInt(Integer::intValue).toArray(),
                ovVal.stream().mapToDouble(Double::doubleValue).toArray(),
                dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta, dsAnchorShift, dsSlackShift,
                parameters.getMaxIterations(), tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                config.lsMode(), config.maxDv(), config.maxDphi(),
                config.lsMode() == 2 ? varType : null, x0Full,
                shOvSlotElem, shOvGb, null, null,
                r1OvSlotElem, r1OvVal,
                config.maxOuterIterations(), config.maxPqPvSwitch(), config.dsRejectResidualFlag(),
                nziBase, ziOnSlotElem, ziOffSlotElem);
        for (int s = 0; s < realCount; s++) {
            if (affected[s]) {
                chunkResults[s] = r2[s];
            }
        }
        return chunkResults;
    }
}
