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
                          double[] baseState, List<Integer> ovIndexList, List<Double> ovValueList) {
        for (int slot = 0; slot < realCount; slot++) {
            for (GpuBatchTransformerControl.TransformerController tc : tcs) {
                boolean anySurvivor = false;                 // plan-A6: free-member outages re-pick on-device
                for (GpuBatchTransformerControl.Member m : tc.members()) {
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
                for (GpuBatchTransformerControl.Member m : tc.members()) {
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
                for (GpuBatchTransformerControl.Member m : tc.members()) {
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

    /** Post-solve rounding: read each scenario's continuous rho, round to the closest tap, restore
     *  the VC-OFF (base) activation and re-solve warm with the rho identity rows pinned at the
     *  rounded values and the rho states seeded there (the CPU re-reads r1 from the pi model). */
    @SuppressWarnings("java:S107")
    static double[][] roundAndResolve(long handle, int chunkSize, int realCount, int n,
            double[][] chunkResults, List<GpuBatchTransformerControl.TransformerController> tcs,
            int[][] rowModePerScenario, int[] baseRowMode, boolean[][] disabledMask,
            List<Integer> baseOvIndexList, List<Double> baseOvValueList,
            int[] dsOvBus, double[] dsOvFactorDelta, double[] dsOvBaseDelta, double[] dsOvMaxDelta, double[] dsOvMinDelta,
            double[] dsAnchorShift, double[] dsSlackShift, int[] dsPhiRow, double[] dsFactor,
            int[] r1OvSlotElem, double[] r1OvVal, int[] shOvSlotElem, double[] shOvGb,
            int nziBase, int[] ziOnSlotElem, int[] ziOffSlotElem,
            double[] baseState, double[] baseTargets, NewtonRaphsonParameters parameters,
            double tol, double reactiveLimitTol, double slackDistTol, double convEpsPerEq, int[] varType,
            BatchConfig config) {
        // rebuild the VC-OFF overrides: the ORIGINAL (pre-VC-on) list + one rho pin per live controller
        List<Integer> ovIdx = new ArrayList<>(baseOvIndexList);
        List<Double> ovVal = new ArrayList<>(baseOvValueList);
        // DS target carry (same three coupled pieces as the AFTER_GENERATOR machine): the main
        // solve's slack distribution must be carried into this re-solve, or the kernel re-derives
        // it from the re-uploaded baseTargets and takes Newton steps the CPU never takes.
        int nds = dsPhiRow != null ? dsPhiRow.length : 0;
        double[] dsAnchorUse = dsAnchorShift;
        double[] dsSlackUse = dsSlackShift;
        if (nds > 0) {
            double[] anchor = new double[chunkSize * nds];
            double[] slackC = new double[chunkSize];
            double[] seed = new double[chunkSize * nds];
            GpuBatchTransformerControl.captureDsCarry(handle, chunkSize, realCount, n, dsPhiRow, dsFactor,
                    baseTargets, dsOvBus, dsOvFactorDelta, dsOvBaseDelta, chunkResults, anchor, slackC, seed);
            dsAnchorUse = anchor;
            dsSlackUse = slackC;
            // absolute DS-row target seeds (the shared, index-backed merge — an inline
            // lastIndexOf scan here was quadratic in realCount * nds)
            GpuBatchTransformerControl.seedDsTargets(nds, dsPhiRow, seed, chunkResults, realCount, n, ovIdx, ovVal);
        }
        // rounding to a non-base tap switches the CPU branch's FULL per-step R/X/G/B, not just rho:
        // extend the base r1 overrides with the rounded tap's whole pi row per (scenario, member).
        // Rounded entries are collected FIRST — the device scan takes the first (slot, elem) match,
        // and a rounded tap must win over a base tap-action override on the same transformer.
        List<Integer> r1Se = new ArrayList<>();
        List<Double> r1Vv = new ArrayList<>();
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
                boolean anySurvivor = false;                 // plan-A6: free-member outages re-pick on-device
                for (GpuBatchTransformerControl.Member m : tc.members()) {
                    anySurvivor |= !disabledMask[s][m.elemIndex()];
                }
                if (!anySurvivor) {
                    continue;
                }
                if (row == null) {
                    row = rowModePerScenario[s].clone();
                }
                for (int k = 0; k < tc.deltaRows().length; k++) {
                    row[tc.deltaRows()[k]] = baseRowMode[tc.deltaRows()[k]];   // undo the VC-on flip
                }
                for (GpuBatchTransformerControl.Member m : tc.members()) {     // OLF roundR1ToClosestTap, per member
                    if (disabledMask[s][m.elemIndex()]) {
                        continue;                            // outaged: stays at its base identity
                    }
                    double rounded = m.closestTap(chunkResults[s][m.rhoRow()]);
                    ovIdx.add(s * n + m.rhoRow());
                    ovVal.add(rounded);                      // the identity row pins rho at the rounded tap
                    x0Full[s * n + m.rhoRow()] = rounded;
                    double[] tapRow = m.closestTapRow(rounded);
                    if (tapRow != null) {                    // the rounded tap's FULL pi row
                        r1Se.add(s);
                        r1Se.add(m.elemIndex());
                        for (double d : tapRow) {
                            r1Vv.add(d);
                        }
                    }
                }
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
        if (r1OvSlotElem != null) {                          // base overrides AFTER the rounded ones
            for (int i : r1OvSlotElem) {
                r1Se.add(i);
            }
            for (double d : r1OvVal) {
                r1Vv.add(d);
            }
        }
        double[][] r2 = GpuAcNewtonSolver.solveBatchContext(handle, baseState, baseTargets,
                rowModeOff, baseRowMode, disabledMask,
                ovIdx.stream().mapToInt(Integer::intValue).toArray(),
                ovVal.stream().mapToDouble(Double::doubleValue).toArray(),
                dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta, dsAnchorUse, dsSlackUse, null, null,
                parameters.getMaxIterations(), tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                config.lsMode(), config.maxDv(), config.maxDphi(),
                config.lsMode() == 2 ? varType : null, x0Full,
                shOvSlotElem, shOvGb, null, null,
                r1Se.isEmpty() ? null : r1Se.stream().mapToInt(Integer::intValue).toArray(),
                r1Vv.isEmpty() ? null : r1Vv.stream().mapToDouble(Double::doubleValue).toArray(),
                config.maxOuterIterations(), config.maxPqPvSwitch(), config.dsRejectResidualFlag(), 0,
                nziBase, ziOnSlotElem, ziOffSlotElem);
        for (int s = 0; s < realCount; s++) {
            if (affected[s]) {
                chunkResults[s] = r2[s];
            }
        }
        return chunkResults;
    }
}
