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

    /** Phase-B confirmation solves this JVM has skipped, and monitored buses it has switched PQ -&gt; PV.
     *  Test-visible: the skip is only reachable when a monitoring SVC and an orchestrator that
     *  RL-solves (the HVDC AC-emulation loops) are in the SAME network, so a test that does not
     *  check these counters cannot tell whether it exercised the skip or merely passed around it. */
    public static final java.util.concurrent.atomic.LongAdder SKIPPED_CONFIRM_SOLVES = new java.util.concurrent.atomic.LongAdder();
    public static final java.util.concurrent.atomic.LongAdder ACTIVATIONS = new java.util.concurrent.atomic.LongAdder();

    /** {@code OLF_GPU_MON_TRACE}: print each monitored bus's per-unit voltage against its automaton
     *  band. Thresholds are per-unit inside OLF, so a kV band picked by eye is easy to get wrong —
     *  and an SVC whose BASE voltage is out of band is switched to plain voltage control for good by
     *  OLF's monitoring loop, leaving nothing for the batch to monitor. */
    private static final boolean MON_TRACE = System.getenv("OLF_GPU_MON_TRACE") != null;

    private GpuBatchVoltageMonitoring() {
    }

    /** One monitored bus: its V row, RL pack index, the automaton band + setpoints, and the VC-ON
     *  activation deltas (row modes + targets; the V-row target is replaced per direction). */
    public record Ctl(int vRow, int rlIdx, double lowThreshold, double highThreshold,
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

    public static List<Ctl> buildControllers(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
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
            double[][] chunkResults, List<Ctl> ctls, int[] rlState, boolean[] rlSolved,
            int[][] rowModePerScenario, int[] baseRowMode, boolean[][] disabledMask,
            int[] targetOverrideIndex, double[] targetOverrideValue,
            int[] dsOvBus, double[] dsOvFactorDelta, double[] dsOvBaseDelta, double[] dsOvMaxDelta, double[] dsOvMinDelta,
            int[] dsPhiRow, double[] dsFactor,
            int[] r1OvSlotElem, double[] r1OvVal, int[] shOvSlotElem, double[] shOvGb,
            int nziBase, int[] ziOnSlotElem, int[] ziOffSlotElem,
            double[] baseState, double[] baseTargets, NewtonRaphsonParameters parameters,
            double tol, double reactiveLimitTol, double slackDistTol, double convEpsPerEq, int[] varType,
            BatchConfig config) {
        int nc = ctls.size();
        if (nc == 0 || rlState == null) {
            return chunkResults;
        }
        // DS target carry (same three coupled pieces as GpuBatchTransformerControl): the MAIN
        // solve's slack distribution must ride into the activation/RL re-solves, or the kernel
        // re-derives it from the re-uploaded baseTargets along a trajectory the CPU never takes.
        int nds = dsPhiRow != null ? dsPhiRow.length : 0;
        double[] dsAnchorCarry = nds > 0 ? new double[chunkSize * nds] : null;
        double[] dsSlackCarry = nds > 0 ? new double[chunkSize] : null;
        double[] dsSeed = nds > 0 ? new double[chunkSize * nds] : null;
        if (nds > 0) {
            GpuBatchTransformerControl.captureDsCarry(handle, chunkSize, realCount, n, dsPhiRow, dsFactor, baseTargets,
                    dsOvBus, dsOvFactorDelta, dsOvBaseDelta, chunkResults, dsAnchorCarry, dsSlackCarry, dsSeed);
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
                    if (MON_TRACE) {                     // OLF_GPU_MON_TRACE: the per-unit band check
                        System.err.printf("MON s=%d c=%d v=%.6f band=[%.6f,%.6f]%n",
                                s, c, v, ctl.lowThreshold(), ctl.highThreshold());
                    }
                    if (v > ctl.highThreshold()) {
                        activatedTargetV[s][c] = ctl.highTargetV();
                    } else if (v < ctl.lowThreshold()) {
                        activatedTargetV[s][c] = ctl.lowTargetV();
                    } else {
                        continue;
                    }
                    ACTIVATIONS.increment();
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
            GpuBatchTransformerControl.seedDsTargets(nds, dsPhiRow, dsSeed, chunkResults, realCount, n, ovIdx, ovVal);
            double[] x0Full = new double[chunkSize * n];
            for (int s = 0; s < chunkSize; s++) {
                double[] src = s < realCount && chunkResults[s] != null ? chunkResults[s] : baseState;
                System.arraycopy(src, 0, x0Full, s * n, n);
            }
            applyActivations(realCount, n, ctls, activatedTargetV, rowModePerScenario, ovIdx, ovVal);
            GpuBatchPhaseStats.bump("VoltageMonitoring.solvesPreRl");
            double[][] r2 = resolve(handle, chunkSize, n, nrl, ctls, activatedTargetV, rlState, 1,
                    rowModePerScenario, baseRowMode, disabledMask, ovIdx, ovVal, x0Full,
                    dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta, dsAnchorCarry, dsSlackCarry,
                    r1OvSlotElem, r1OvVal, shOvSlotElem, shOvGb, nziBase, ziOnSlotElem, ziOffSlotElem,
                    baseState, baseTargets, parameters, tol, reactiveLimitTol, slackDistTol, convEpsPerEq, varType,
                    config, realCount);
            for (int s = 0; s < realCount; s++) {
                if (affected[s]) {                           // untouched scenarios KEEP their DS-only state
                    chunkResults[s] = r2[s];
                }
            }
            if (nds > 0) {                                   // this solve may have distributed further
                GpuBatchTransformerControl.captureDsCarry(handle, chunkSize, realCount, n, dsPhiRow, dsFactor, baseTargets,
                        dsOvBus, dsOvFactorDelta, dsOvBaseDelta, chunkResults, dsAnchorCarry, dsSlackCarry, dsSeed, affected);
            }
        }
        // ---- PHASE B: monitoring stable at DS-only states — run the FULL solve (RL enabled) for the
        // scenarios that still OWE one, then keep re-checking monitoring at the post-RL states (the
        // engine re-checks it after every RL re-solve).
        //
        // A scenario owes the solve when its state has not been through the reactive-limits phase (the
        // main solve skips it while monitoring is interleaved) or when its automaton just switched
        // PQ -> PV. A scenario that owes NEITHER is one both loops would report STABLE on, and the CPU
        // runs no Newton for a stable scenario: re-solving it and keeping the result drifts it off the
        // CPU by the solver's own convergence slack (measured 1e-3 kV when another scenario in the same
        // chunk activated), and when NO scenario owes anything the whole batched solve is pure waste
        // (measured 4.5 s of a 20 s batch on rte6515 at 2000 contingencies). So phase B follows the same
        // "untouched scenarios KEEP their converged state" rule as every other orchestrator. ----
        boolean[] owed = new boolean[chunkSize];
        for (int s = 0; s < realCount; s++) {
            owed[s] = rlSolved == null || !rlSolved[s];
            for (int c = 0; c < nc && !owed[s]; c++) {
                owed[s] = !Double.isNaN(activatedTargetV[s][c]);
            }
        }
        for (int round = 0; round < maxRounds; round++) {
            if (noneOwed(realCount, owed)) {
                if (round == 0) {
                    GpuBatchPhaseStats.bump("VoltageMonitoring.skippedConfirmSolves");
                    SKIPPED_CONFIRM_SOLVES.increment();
                }
                break;
            }
            List<Integer> ovIdx = new ArrayList<>();
            List<Double> ovVal = new ArrayList<>();
            for (int i = 0; i < targetOverrideIndex.length; i++) {
                ovIdx.add(targetOverrideIndex[i]);
                ovVal.add(targetOverrideValue[i]);
            }
            GpuBatchTransformerControl.seedDsTargets(nds, dsPhiRow, dsSeed, chunkResults, realCount, n, ovIdx, ovVal);
            double[] x0Full = new double[chunkSize * n];
            for (int s = 0; s < chunkSize; s++) {
                double[] src = s < realCount && chunkResults[s] != null ? chunkResults[s] : baseState;
                System.arraycopy(src, 0, x0Full, s * n, n);
            }
            applyActivations(realCount, n, ctls, activatedTargetV, rowModePerScenario, ovIdx, ovVal);
            GpuBatchPhaseStats.bump("VoltageMonitoring.solvesRl");
            double[][] r2 = resolve(handle, chunkSize, n, nrl, ctls, activatedTargetV, rlState, 0,
                    rowModePerScenario, baseRowMode, disabledMask, ovIdx, ovVal, x0Full,
                    dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta, dsAnchorCarry, dsSlackCarry,
                    r1OvSlotElem, r1OvVal, shOvSlotElem, shOvGb, nziBase, ziOnSlotElem, ziOffSlotElem,
                    baseState, baseTargets, parameters, tol, reactiveLimitTol, slackDistTol, convEpsPerEq, varType,
                    config, realCount);
            if (GpuBatchPhaseStats.ENABLED) {
                int itMax = 0;
                for (int s = 0; s < realCount; s++) {
                    itMax = Math.max(itMax, (int) r2[s][n]);
                }
                GpuBatchPhaseStats.add("VoltageMonitoring.iterMax", itMax);
            }
            for (int s = 0; s < realCount; s++) {
                if (owed[s]) {                               // untouched scenarios KEEP their converged state
                    chunkResults[s] = r2[s];
                }
            }
            if (nds > 0) {                                   // this solve may have distributed further
                GpuBatchTransformerControl.captureDsCarry(handle, chunkSize, realCount, n, dsPhiRow, dsFactor, baseTargets,
                        dsOvBus, dsOvFactorDelta, dsOvBaseDelta, chunkResults, dsAnchorCarry, dsSlackCarry, dsSeed, owed);
            }
            // post-RL monitoring re-check: only a scenario that switches now owes the NEXT solve
            owed = new boolean[chunkSize];
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
                        ACTIVATIONS.increment();
                        owed[s] = true;
                        any = true;
                    } else if (v < ctl.lowThreshold()) {
                        activatedTargetV[s][c] = ctl.lowTargetV();
                        ACTIVATIONS.increment();
                        owed[s] = true;
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

    /** True when no scenario still owes a reactive-limits solve — every outer loop would report STABLE. */
    private static boolean noneOwed(int realCount, boolean[] owed) {
        for (int s = 0; s < realCount; s++) {
            if (owed[s]) {
                return false;
            }
        }
        return true;
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
            double[] dsAnchorCarry, double[] dsSlackCarry,
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
                dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta, dsAnchorCarry, dsSlackCarry,
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

    /** Install the MonitoringVoltage driver phase's table on a batch context (null/empty clears it).
     *  Shared by the batched SA's chunk contexts and the base load flow's batch-of-one. */
    public static void installDriverLoop(long handle, List<Ctl> ctls) {
        if (ctls == null || ctls.isEmpty()) {
            GpuAcNewtonSolver.setBatchMonitoringLoops(handle, null, null, null, null, null, null);
            return;
        }
        int nc = ctls.size();
        int[] monVRow = new int[nc];
        double[] monBand = new double[nc * 4];
        int[] deltaPtr = new int[nc + 1];
        List<Integer> dRow = new ArrayList<>();
        List<Integer> dMode = new ArrayList<>();
        List<Double> dTgt = new ArrayList<>();
        for (int c = 0; c < nc; c++) {
            Ctl ctl = ctls.get(c);
            monVRow[c] = ctl.vRow();
            monBand[c * 4] = ctl.lowThreshold();
            monBand[c * 4 + 1] = ctl.highThreshold();
            monBand[c * 4 + 2] = ctl.lowTargetV();
            monBand[c * 4 + 3] = ctl.highTargetV();
            deltaPtr[c] = dRow.size();
            for (int k = 0; k < ctl.deltaRows().length; k++) {
                dRow.add(ctl.deltaRows()[k]);
                dMode.add(ctl.deltaModesOn()[k]);
                dTgt.add(ctl.deltaTargetsOn()[k]);
            }
        }
        deltaPtr[nc] = dRow.size();
        GpuAcNewtonSolver.setBatchMonitoringLoops(handle, monVRow, monBand, deltaPtr,
                dRow.stream().mapToInt(Integer::intValue).toArray(),
                dMode.stream().mapToInt(Integer::intValue).toArray(),
                dTgt.stream().mapToDouble(Double::doubleValue).toArray());
    }

    /** Give the driver's activations (scenario 0: the chosen target per controller, NaN = standing by)
     *  back to the network exactly as OLF's MonitoringVoltageOuterLoop.switchPqPv does: voltage control
     *  on with the recomputed Q target, the SVC in VOLTAGE control, the automaton's target as setpoint. */
    public static void writeBack(LfNetwork network, List<Ctl> ctls, double[] activated, int[] rlBusNum) {
        if (ctls == null || ctls.isEmpty() || activated == null || rlBusNum == null) {
            return;
        }
        for (int c = 0; c < ctls.size() && c < activated.length; c++) {
            double t = activated[c];
            Ctl ctl = ctls.get(c);
            if (Double.isNaN(t) || ctl.rlIdx() < 0 || ctl.rlIdx() >= rlBusNum.length) {
                continue;
            }
            LfBus bus = network.getBus(rlBusNum[ctl.rlIdx()]);
            if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
                System.err.println("WB_MON bus=" + (bus == null ? "?" : bus.getId()) + " activatedTarget=" + t
                        + " vcEnabled=" + (bus != null && bus.isGeneratorVoltageControlEnabled()));
            }
            if (bus == null || bus.isGeneratorVoltageControlEnabled()) {
                continue;
            }
            bus.setGeneratorVoltageControlEnabledAndRecomputeTargetQ(true);
            for (LfGenerator g : bus.getGenerators()) {
                if (g.getGeneratorControlType() == LfGenerator.GeneratorControlType.MONITORING_VOLTAGE) {
                    g.setGeneratorControlType(LfGenerator.GeneratorControlType.VOLTAGE);
                    break;
                }
            }
            bus.getGeneratorVoltageControl().ifPresent(vc -> vc.setTargetValue(t));
        }
    }
}
