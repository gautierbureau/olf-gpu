/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivation;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.ac.solver.gpu.BatchSolveNative;
import com.powsybl.openloadflow.ac.solver.gpu.batch.BatchConfig;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchSkips;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuStateVectorScalingMode;
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
public final class GpuMonitoringVoltageOuterLoop {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuMonitoringVoltageOuterLoop.class);

    /** Phase-B confirmation solves this JVM has skipped, and monitored buses it has switched PQ -&gt; PV.
     *  Test-visible: the skip is only reachable when a monitoring SVC and an orchestrator that
     *  RL-solves (the HVDC AC-emulation loops) are in the SAME network, so a test that does not
     *  check these counters cannot tell whether it exercised the skip or merely passed around it. */
    public static final java.util.concurrent.atomic.LongAdder SKIPPED_CONFIRM_SOLVES = new java.util.concurrent.atomic.LongAdder();
    public static final java.util.concurrent.atomic.LongAdder ACTIVATIONS = new java.util.concurrent.atomic.LongAdder();

    private GpuMonitoringVoltageOuterLoop() {
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
                GpuBatchSkips.record(GpuBatchSkips.Slot.MONITORING, LOGGER,
                        "Monitoring SVC bus {} not representable on the GPU batch — skipped", bus.getId());
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

    @SuppressWarnings("java:S107")
    private static double[][] resolve(long handle, int chunkSize, int n, int nrl, List<Ctl> ctls,
            double[][] activatedTargetV, int[] rlState, int skipRl,
            int[][] rowModePerScenario, int[] baseRowMode, boolean[][] disabledMask,
            List<Integer> ovIdx, List<Double> ovVal, double[] x0Full,
            int[] dsOvBus, double[] dsOvFactorDelta, double[] dsOvBaseDelta, double[] dsOvMaxDelta, double[] dsOvMinDelta,
            double[] dsAnchorCarry, double[] dsSlackCarry,
            int[] r1OvSlotElem, double[] r1OvVal, int[] shuntAdmittanceOverrideScenarioElement, double[] shuntAdmittanceOverrideDelta,
            int nBaseClosedZeroImpedanceBranches, int[] zeroImpedanceBranchCloseScenarioElement, int[] zeroImpedanceBranchOpenScenarioElement,
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
        double[][] r2 = BatchSolveNative.solveBatchContext(handle, baseState, baseTargets,
                rowModePerScenario, baseRowMode, disabledMask,
                ovIdx.stream().mapToInt(Integer::intValue).toArray(),
                ovVal.stream().mapToDouble(Double::doubleValue).toArray(),
                dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta, dsAnchorCarry, dsSlackCarry,
                vtSlotIdx.stream().mapToInt(Integer::intValue).toArray(),
                vtVal.stream().mapToDouble(Double::doubleValue).toArray(),
                null, null,
                null, null,
                parameters.getMaxIterations(), tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                config.lsMode(), config.maxDv(), config.maxDphi(),
                config.lsMode() == GpuStateVectorScalingMode.MAX_VOLTAGE_CHANGE ? varType : null, x0Full,
                shuntAdmittanceOverrideScenarioElement, shuntAdmittanceOverrideDelta, rlState, rlStateOut,
                r1OvSlotElem, r1OvVal,
                config.maxOuterIterations(), config.maxPqPvSwitch(), config.dsRejectResidualFlag(), skipRl,
                nBaseClosedZeroImpedanceBranches, zeroImpedanceBranchCloseScenarioElement, zeroImpedanceBranchOpenScenarioElement);
        System.arraycopy(rlStateOut, 0, rlState, 0, rlStateOut.length);
        return r2;
    }

    /** Install the MonitoringVoltage driver phase's table on a batch context (null/empty clears it).
     *  Shared by the batched SA's chunk contexts and the base load flow's batch-of-one. */
    public static void installDriverLoop(long handle, List<Ctl> ctls) {
        if (ctls == null || ctls.isEmpty()) {
            VoltageMonitoringNative.setBatchMonitoringLoops(handle, null, null, null, null, null, null);
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
        VoltageMonitoringNative.setBatchMonitoringLoops(handle, monVRow, monBand, deltaPtr,
                dRow.stream().mapToInt(Integer::intValue).toArray(),
                dMode.stream().mapToInt(Integer::intValue).toArray(),
                dTgt.stream().mapToDouble(Double::doubleValue).toArray());
    }

    /** Give the driver's activations (scenario 0: the chosen target per controller, NaN = standing by)
     *  back to the network exactly as OLF's MonitoringVoltageOuterLoop.switchPqPv does: voltage control
     *  on with the recomputed Q target, the SVC in VOLTAGE control, the automaton's target as setpoint. */
    public static void applyToNetwork(LfNetwork network, List<Ctl> ctls, double[] activated, int[] rlBusNum) {
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
