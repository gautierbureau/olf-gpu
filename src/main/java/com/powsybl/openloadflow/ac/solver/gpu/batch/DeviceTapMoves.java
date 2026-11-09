/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.outerloop.gpu.DeviceFlowControllers;
import com.powsybl.openloadflow.ac.outerloop.gpu.DevicePhases;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuIncrementalTransformerVoltageControlOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuPhaseControlOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuTransformerVoltageControlOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.gpu.PhaseControlNative;
import com.powsybl.openloadflow.ac.outerloop.gpu.TransformerVoltageControlNative;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuDiscreteControlMemberState;

import java.util.ArrayList;
import java.util.List;

/**
 * The taps the device's outer loops left on each harvested scenario's transformers, as moves from the base tap: what
 * OLF's own loops leave on the network's branch models after a post-contingency load flow, and what its results are
 * then evaluated with. The device solves with each moved tap's parameters (the per-scenario tap overrides), but nothing
 * gave the move back to the host, so the reported flows and the confirmed violations of a retapped transformer came
 * from its base tap's admittance (measured: 1.4 % of a transformer's flow after a one-step move).
 *
 * <p>A move is {@code {branchNum, tap index, rounding r1}}: the tap index is relative to the branch's tap range minimum,
 * and the rounding r1 is the continuous ratio OLF's discrete transformer voltage control rounded from (NaN for the
 * incremental loops, which move whole taps). Read from the four phases that can move a tap — incremental transformer
 * voltage control, the flow controllers (reactive power and incremental phase control), continuous phase control and
 * discrete transformer voltage control — after a solve call, for the scenarios it handed back.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class DeviceTapMoves {

    /** Values per move in a scenario's array. */
    static final int STRIDE = 3;

    private DeviceTapMoves() {
    }

    /**
     * The moves of every scenario the last solve call handed back, in its row order (null for a scenario without any),
     * or null when the run has no phase that moves a tap. The per-slot read-backs list the handed-back slots in that
     * same order (readInstalledRows), one row per controller (per member for the discrete loop) in the order the
     * phase's controllers were installed.
     */
    static double[][] read(DevicePhases.Built phases, long handle, int harvestedCount) {
        if (phases == null) {
            return null;
        }
        List<GpuIncrementalTransformerVoltageControlOuterLoop.Controller> incremental = phases.incrementalTransformerVoltageControllers();
        List<DeviceFlowControllers.Controller> flow = phases.flow();
        List<GpuPhaseControlOuterLoop.Controller> continuousPhase = phases.continuousPhase();
        List<GpuTransformerVoltageControlOuterLoop.TransformerController> discrete = phases.discreteTransformerVoltageControl();
        if (isEmpty(incremental) && isEmpty(flow) && isEmpty(continuousPhase) && isEmpty(discrete)) {
            return null;
        }
        @SuppressWarnings("unchecked")
        List<double[]>[] moves = new List[harvestedCount];
        if (!isEmpty(incremental)) {
            int[] taps = TransformerVoltageControlNative.readBatchIncrementalTransformerVoltageControlTaps(handle);
            int[] branchNum = incremental.stream().mapToInt(GpuIncrementalTransformerVoltageControlOuterLoop.Controller::branchNum).toArray();
            int[] baseTap = incremental.stream().mapToInt(GpuIncrementalTransformerVoltageControlOuterLoop.Controller::baseTapIndex).toArray();
            addWholeTapMoves(moves, taps, branchNum, baseTap);
        }
        if (!isEmpty(flow)) {
            int[] taps = PhaseControlNative.readBatchFlowTaps(handle);
            int[] branchNum = flow.stream().mapToInt(DeviceFlowControllers.Controller::branchNum).toArray();
            int[] baseTap = flow.stream().mapToInt(DeviceFlowControllers.Controller::baseTapIndex).toArray();
            addWholeTapMoves(moves, taps, branchNum, baseTap);
        }
        if (!isEmpty(continuousPhase)) {
            int[] taps = PhaseControlNative.readBatchContinuousPhaseTaps(handle);
            int[] branchNum = continuousPhase.stream().mapToInt(GpuPhaseControlOuterLoop.Controller::branchNum).toArray();
            int[] baseTap = continuousPhase.stream().mapToInt(GpuPhaseControlOuterLoop.Controller::baseTapIndex).toArray();
            addWholeTapMoves(moves, taps, branchNum, baseTap);
        }
        if (!isEmpty(discrete)) {
            addRoundedMoves(moves, TransformerVoltageControlNative.readBatchTransformerVoltageControlState(handle), discrete);
        }
        double[][] bySlot = new double[harvestedCount][];
        for (int s = 0; s < harvestedCount; s++) {
            if (moves[s] != null) {
                double[] flat = new double[moves[s].size() * STRIDE];
                for (int k = 0; k < moves[s].size(); k++) {
                    System.arraycopy(moves[s].get(k), 0, flat, k * STRIDE, STRIDE);
                }
                bySlot[s] = flat;
            }
        }
        return bySlot;
    }

    private static boolean isEmpty(List<?> list) {
        return list == null || list.isEmpty();
    }

    /** One row of {@code branchNum.length} tap indices per handed-back scenario; a move wherever the tap left its base. */
    private static void addWholeTapMoves(List<double[]>[] moves, int[] taps, int[] branchNum, int[] baseTap) {
        int nc = branchNum.length;
        if (taps == null || nc == 0) {
            return;                                          // the phase's tables are not installed on this context
        }
        if (taps.length != moves.length * nc) {
            throw new IllegalStateException("tap read-back of " + taps.length + " values for " + moves.length
                    + " scenarios x " + nc + " controllers");
        }
        for (int s = 0; s < moves.length; s++) {
            for (int c = 0; c < nc; c++) {
                int tap = taps[s * nc + c];
                if (tap != baseTap[c]) {
                    add(moves, s, branchNum[c], tap, Double.NaN);
                }
            }
        }
    }

    /** {state, tap index, rounding r1} per (scenario, member): a move where a member was ROUNDED off its base tap. */
    private static void addRoundedMoves(List<double[]>[] moves, double[] state,
                                        List<GpuTransformerVoltageControlOuterLoop.TransformerController> controllers) {
        if (state == null) {
            return;
        }
        List<GpuTransformerVoltageControlOuterLoop.Member> members = new ArrayList<>();
        for (GpuTransformerVoltageControlOuterLoop.TransformerController controller : controllers) {
            members.addAll(List.of(controller.members()));
        }
        int nm = members.size();
        if (state.length != moves.length * nm * 3) {
            throw new IllegalStateException("transformer voltage control read-back of " + state.length + " values for "
                    + moves.length + " scenarios x " + nm + " members");
        }
        for (int s = 0; s < moves.length; s++) {
            for (int m = 0; m < nm; m++) {
                int k = (s * nm + m) * 3;
                int tap = (int) state[k + 1];
                GpuTransformerVoltageControlOuterLoop.Member member = members.get(m);
                if ((int) state[k] == GpuDiscreteControlMemberState.ROUNDED && tap != member.baseTapIndex()) {
                    add(moves, s, member.branchNum(), tap, state[k + 2]);
                }
            }
        }
    }

    private static void add(List<double[]>[] moves, int s, int branchNum, int tap, double roundingR1) {
        if (moves[s] == null) {
            moves[s] = new ArrayList<>();
        }
        moves[s].add(new double[] {branchNum, tap, roundingR1});
    }
}
