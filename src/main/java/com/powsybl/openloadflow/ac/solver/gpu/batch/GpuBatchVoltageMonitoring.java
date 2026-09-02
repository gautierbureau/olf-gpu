/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivation;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfStaticVarCompensator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * DEVICE port of OLF's {@code MonitoringVoltageOuterLoop} (SVC stand-by automatons): a monitoring
 * SVC's bus stands by as PQ (its base-PQ RL entry seeds UNPINNED via {@code rlMonitored} — the CPU's
 * null qLimitType, so the reactive-limits loop never touches it spontaneously) until the bus voltage
 * leaves the automaton's [low, high] threshold band; then the bus switches PQ → PV at the automaton's
 * LOW/HIGH target voltage (OLF's {@code switchPqPv}) and stays a normal voltage controller for the
 * rest of the run (the RL loop may then pin it at its Q limits like any PV bus — the moved target
 * reaches the RL kernel through the {@code rlVtOv} channel).
 *
 * <p>Per-scenario activation = the bus's VC-ON activation deltas (captured by enable-toggle diff,
 * like the Simple-TVC port) with the V-row target REPLACED by the chosen automaton setpoint.
 * Stage-1 scope: LOCAL monitoring only (the SVC controls its own bus).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuBatchVoltageMonitoring {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuBatchVoltageMonitoring.class);

    private GpuBatchVoltageMonitoring() {
    }

    /** One monitored bus: its V row, RL pack index, the automaton band + setpoints, and the VC-ON
     *  activation deltas (row modes + targets; the V-row target is replaced per direction). */
    record Ctl(int vRow, int rlIdx, double lowThreshold, double highThreshold,
               double lowTargetV, double highTargetV,
               int[] deltaRows, int[] deltaModesOn, double[] deltaTargetsOn) {
    }

    /** True when every MONITORING_VOLTAGE SVC is local (controls its own bus) — else CPU. */
    public static boolean supported(LfNetwork network) {
        for (LfBus bus : network.getBuses()) {
            for (LfGenerator g : bus.getGenerators()) {
                if (g.getGeneratorControlType() == LfGenerator.GeneratorControlType.MONITORING_VOLTAGE) {
                    var vc = bus.getGeneratorVoltageControl().orElse(null);
                    if (vc == null || vc.getControlledBus() != bus
                            || ((LfStaticVarCompensator) g).getStandByAutomaton().isEmpty()) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    static List<Ctl> buildControllers(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                      com.powsybl.openloadflow.equations.TargetVector<AcVariableType, AcEquationType> targetVector,
                                      GpuAcActivation baseActivation, double[] baseTargets, int[] rlVrow) {
        Map<Integer, Integer> busVRows = new LinkedHashMap<>();
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            if (v.getType() == AcVariableType.BUS_V) {
                busVRows.put(v.getElementNum(), v.getRow());
            }
        }
        Map<Integer, Integer> rlIdxOfVRow = new LinkedHashMap<>();
        for (int i = 0; i < rlVrow.length; i++) {
            rlIdxOfVRow.put(rlVrow[i], i);
        }
        List<Ctl> out = new ArrayList<>();
        for (LfBus bus : network.getBuses()) {
            LfStaticVarCompensator svc = null;
            for (LfGenerator g : bus.getGenerators()) {
                if (g.getGeneratorControlType() == LfGenerator.GeneratorControlType.MONITORING_VOLTAGE) {
                    svc = (LfStaticVarCompensator) g;
                    break;
                }
            }
            if (svc == null) {
                continue;
            }
            var automatonOpt = svc.getStandByAutomaton();
            Integer vRow = busVRows.get(bus.getNum());
            Integer rlIdx = vRow != null ? rlIdxOfVRow.get(vRow) : null;
            if (automatonOpt.isEmpty() || vRow == null || rlIdx == null || bus.isGeneratorVoltageControlEnabled()) {
                LOGGER.info("Monitoring SVC bus {} not representable on the GPU batch — skipped", bus.getId());
                continue;
            }
            var automaton = automatonOpt.orElseThrow();
            // VC-ON activation diff (like the Simple-TVC port): what enabling voltage control on this
            // bus changes in row modes and targets
            boolean was = bus.isGeneratorVoltageControlEnabled();
            bus.setGeneratorVoltageControlEnabledAndRecomputeTargetQ(true);
            GpuAcActivation onAct = GpuAcActivation.fromEquationSystem(network, es);
            double[] onTargets = onAct.applyTargets(targetVector.getArray());
            bus.setGeneratorVoltageControlEnabledAndRecomputeTargetQ(was);
            List<Integer> dRows = new ArrayList<>();
            for (int r = 0; r < baseTargets.length; r++) {
                if (onAct.rowMode()[r] != baseActivation.rowMode()[r] || onTargets[r] != baseTargets[r]) {
                    dRows.add(r);
                }
            }
            int[] deltaRows = dRows.stream().mapToInt(Integer::intValue).toArray();
            int[] deltaModesOn = new int[deltaRows.length];
            double[] deltaTargetsOn = new double[deltaRows.length];
            for (int k = 0; k < deltaRows.length; k++) {
                deltaModesOn[k] = onAct.rowMode()[deltaRows[k]];
                deltaTargetsOn[k] = onTargets[deltaRows[k]];
            }
            out.add(new Ctl(vRow, rlIdx, automaton.getLowVoltageThreshold(), automaton.getHighVoltageThreshold(),
                    automaton.getLowTargetV(), automaton.getHighTargetV(),
                    deltaRows, deltaModesOn, deltaTargetsOn));
        }
        return out;
    }

    @SuppressWarnings({"java:S107", "java:S3776"})
    static double[][] orchestrate(long handle, int chunkSize, int realCount, int n, int nrl,
            double[][] chunkResults, List<Ctl> ctls, int[] rlState,
            int[][] rowModePerScenario, int[] baseRowMode, boolean[][] disabledMask,
            int[] targetOverrideIndex, double[] targetOverrideValue,
            int[] dsOvBus, double[] dsOvFactorDelta, double[] dsOvBaseDelta, double[] dsOvMaxDelta, double[] dsOvMinDelta,
            double[] dsAnchorShift, double[] dsSlackShift,
            int[] r1OvSlotElem, double[] r1OvVal, int[] shOvSlotElem, double[] shOvGb,
            int nziBase, int[] ziOnSlotElem, int[] ziOffSlotElem,
            double[] baseState, double[] baseTargets, NewtonRaphsonParameters parameters,
            double tol, double reactiveLimitTol, double slackDistTol, double convEpsPerEq, int[] varType,
            BatchConfig config) {
        int nc = ctls.size();
        if (nc == 0 || rlState == null) {
            return chunkResults;
        }
        double[][] activatedTargetV = new double[chunkSize][nc];   // NaN = still standing by
        for (double[] row : activatedTargetV) {
            java.util.Arrays.fill(row, Double.NaN);
        }
        int maxRounds = config.maxOuterIterations() > 0 ? config.maxOuterIterations() : 20;

        // ---- PHASE A: OLF's loop ORDER puts VoltageMonitoring BEFORE ReactiveLimits — the checks run
        // at DS-stable / PRE-RL states (the chunk's main solve was made with skipRlPhase=1). Iterate
        // activations at DS-only states until monitoring is stable there. ----
        for (int round = 0; round < maxRounds; round++) {
            // ---- OLF check(): standing-by monitored buses whose V left the automaton band ----
            boolean[] affected = new boolean[chunkSize];
            boolean any = false;
            for (int s = 0; s < realCount; s++) {
                double[] r = chunkResults[s];
                if (r == null || (int) r[n + 2] != GpuBatchStatus.STATUS_CONVERGED) {
                    continue;
                }
                for (int c = 0; c < nc; c++) {
                    Ctl ctl = ctls.get(c);
                    if (!Double.isNaN(activatedTargetV[s][c])) {
                        continue;                            // already switched — a normal PV bus now
                    }
                    double v = r[ctl.vRow()];
                    if (v > ctl.highThreshold()) {
                        activatedTargetV[s][c] = ctl.highTargetV();
                    } else if (v < ctl.lowThreshold()) {
                        activatedTargetV[s][c] = ctl.lowTargetV();
                    } else {
                        continue;
                    }
                    affected[s] = true;
                    any = true;
                }
            }
            if (!any) {
                break;
            }
            // ---- apply the activations: VC-ON deltas with the chosen setpoint, rlVtOv, re-solve ----
            List<Integer> ovIdx = new ArrayList<>();
            List<Double> ovVal = new ArrayList<>();
            for (int i = 0; i < targetOverrideIndex.length; i++) {
                ovIdx.add(targetOverrideIndex[i]);
                ovVal.add(targetOverrideValue[i]);
            }
            double[] x0Full = new double[chunkSize * n];
            for (int s = 0; s < chunkSize; s++) {
                double[] src = s < realCount && chunkResults[s] != null ? chunkResults[s] : baseState;
                System.arraycopy(src, 0, x0Full, s * n, n);
            }
            applyActivations(realCount, n, ctls, activatedTargetV, rowModePerScenario, ovIdx, ovVal);
            double[][] r2 = resolve(handle, chunkSize, n, nrl, ctls, activatedTargetV, rlState, 1,
                    rowModePerScenario, baseRowMode, disabledMask, ovIdx, ovVal, x0Full,
                    dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta, dsAnchorShift, dsSlackShift,
                    r1OvSlotElem, r1OvVal, shOvSlotElem, shOvGb, nziBase, ziOnSlotElem, ziOffSlotElem,
                    baseState, baseTargets, parameters, tol, reactiveLimitTol, slackDistTol, convEpsPerEq, varType,
                    config, realCount);
            for (int s = 0; s < realCount; s++) {
                if (affected[s]) {                           // untouched scenarios KEEP their DS-only state
                    chunkResults[s] = r2[s];
                }
            }
        }
        // ---- PHASE B: monitoring stable at DS-only states — run the FULL solve (RL enabled) for every
        // scenario (the CPU's ReactiveLimits loop runs only after monitoring stabilizes), then keep
        // re-checking monitoring at the post-RL states (the engine re-checks it after every RL re-solve). ----
        for (int round = 0; round < maxRounds; round++) {
            List<Integer> ovIdx = new ArrayList<>();
            List<Double> ovVal = new ArrayList<>();
            for (int i = 0; i < targetOverrideIndex.length; i++) {
                ovIdx.add(targetOverrideIndex[i]);
                ovVal.add(targetOverrideValue[i]);
            }
            double[] x0Full = new double[chunkSize * n];
            for (int s = 0; s < chunkSize; s++) {
                double[] src = s < realCount && chunkResults[s] != null ? chunkResults[s] : baseState;
                System.arraycopy(src, 0, x0Full, s * n, n);
            }
            applyActivations(realCount, n, ctls, activatedTargetV, rowModePerScenario, ovIdx, ovVal);
            double[][] r2 = resolve(handle, chunkSize, n, nrl, ctls, activatedTargetV, rlState, 0,
                    rowModePerScenario, baseRowMode, disabledMask, ovIdx, ovVal, x0Full,
                    dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta, dsAnchorShift, dsSlackShift,
                    r1OvSlotElem, r1OvVal, shOvSlotElem, shOvGb, nziBase, ziOnSlotElem, ziOffSlotElem,
                    baseState, baseTargets, parameters, tol, reactiveLimitTol, slackDistTol, convEpsPerEq, varType,
                    config, realCount);
            for (int s = 0; s < realCount; s++) {
                chunkResults[s] = r2[s];                     // the RL-stabilized result for EVERY scenario
            }
            // post-RL monitoring re-check
            boolean any = false;
            for (int s = 0; s < realCount; s++) {
                double[] r = chunkResults[s];
                if (r == null || (int) r[n + 2] != GpuBatchStatus.STATUS_CONVERGED) {
                    continue;
                }
                for (int c = 0; c < nc; c++) {
                    Ctl ctl = ctls.get(c);
                    if (!Double.isNaN(activatedTargetV[s][c])) {
                        continue;
                    }
                    double v = r[ctl.vRow()];
                    if (v > ctl.highThreshold()) {
                        activatedTargetV[s][c] = ctl.highTargetV();
                        any = true;
                    } else if (v < ctl.lowThreshold()) {
                        activatedTargetV[s][c] = ctl.lowTargetV();
                        any = true;
                    }
                }
            }
            if (!any) {
                break;
            }
        }
        return chunkResults;
    }

    /** Fold every activated (scenario, bus) into the row modes and override lists. */
    private static void applyActivations(int realCount, int n, List<Ctl> ctls, double[][] activatedTargetV,
                                         int[][] rowModePerScenario, List<Integer> ovIdx, List<Double> ovVal) {
        for (int s = 0; s < realCount; s++) {
            int[] row = null;
            for (int c = 0; c < ctls.size(); c++) {
                Ctl ctl = ctls.get(c);
                double target = activatedTargetV[s][c];
                if (Double.isNaN(target)) {
                    continue;
                }
                if (row == null) {
                    row = rowModePerScenario[s].clone();
                }
                for (int k = 0; k < ctl.deltaRows().length; k++) {
                    row[ctl.deltaRows()[k]] = ctl.deltaModesOn()[k];
                    ovIdx.add(s * n + ctl.deltaRows()[k]);
                    ovVal.add(ctl.deltaRows()[k] == ctl.vRow() ? target : ctl.deltaTargetsOn()[k]);
                }
            }
            if (row != null) {
                rowModePerScenario[s] = row;
            }
        }
    }

    @SuppressWarnings("java:S107")
    private static double[][] resolve(long handle, int chunkSize, int n, int nrl, List<Ctl> ctls,
            double[][] activatedTargetV, int[] rlState, int skipRl,
            int[][] rowModePerScenario, int[] baseRowMode, boolean[][] disabledMask,
            List<Integer> ovIdx, List<Double> ovVal, double[] x0Full,
            int[] dsOvBus, double[] dsOvFactorDelta, double[] dsOvBaseDelta, double[] dsOvMaxDelta, double[] dsOvMinDelta,
            double[] dsAnchorShift, double[] dsSlackShift,
            int[] r1OvSlotElem, double[] r1OvVal, int[] shOvSlotElem, double[] shOvGb,
            int nziBase, int[] ziOnSlotElem, int[] ziOffSlotElem,
            double[] baseState, double[] baseTargets, NewtonRaphsonParameters parameters,
            double tol, double reactiveLimitTol, double slackDistTol, double convEpsPerEq, int[] varType,
            BatchConfig config, int realCount) {
        List<Integer> vtSlotIdx = new ArrayList<>();
        List<Double> vtVal = new ArrayList<>();
        for (int s = 0; s < realCount; s++) {
            for (int c = 0; c < ctls.size(); c++) {
                if (!Double.isNaN(activatedTargetV[s][c])) {
                    vtSlotIdx.add(s);
                    vtSlotIdx.add(ctls.get(c).rlIdx());
                    vtVal.add(activatedTargetV[s][c]);       // the RL kernel's checks use the moved target
                }
            }
        }
        int[] rlStateOut = new int[chunkSize * nrl];
        double[][] r2 = GpuAcNewtonSolver.solveBatchContext(handle, baseState, baseTargets,
                rowModePerScenario, baseRowMode, disabledMask,
                ovIdx.stream().mapToInt(Integer::intValue).toArray(),
                ovVal.stream().mapToDouble(Double::doubleValue).toArray(),
                dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta, dsAnchorShift, dsSlackShift,
                vtSlotIdx.stream().mapToInt(Integer::intValue).toArray(),
                vtVal.stream().mapToDouble(Double::doubleValue).toArray(),
                parameters.getMaxIterations(), tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                config.lsMode(), config.maxDv(), config.maxDphi(),
                config.lsMode() == 2 ? varType : null, x0Full,
                shOvSlotElem, shOvGb, rlState, rlStateOut,
                r1OvSlotElem, r1OvVal,
                config.maxOuterIterations(), config.maxPqPvSwitch(), config.dsRejectResidualFlag(), skipRl,
                nziBase, ziOnSlotElem, ziOffSlotElem);
        System.arraycopy(rlStateOut, 0, rlState, 0, rlStateOut.length);
        return r2;
    }
}
