/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;

/**
 * The device's transformer voltage control loops, incremental and discrete, and the generator freeze.
 *
 * <p>JNI glue: {@code native/cuda/jni/transformer_voltage_control_jni.cu}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class TransformerVoltageControlNative {

    static {
        GpuAcNewtonSolver.isAvailable();   // loads libolfgpu once
    }

    private TransformerVoltageControlNative() {
    }

    /** Per (scenario, freeze candidate): {frozen (0/1), the row target while frozen (generation − load Q)},
     *  layout {@code (s * nGeneratorVoltageControlFreezeCandidates + c) * 2} — null without candidates. */
    public static native double[] readBatchGeneratorVoltageControlFreeze(long handle);

    /** Per (scenario, member): {state (0 off, 1 continuous, 2 fixed), tap index, the r1 fed to the rounding},
     *  layout {@code (s * nMembers + m) * 3} — null without the table. */
    public static native double[] readBatchTransformerVoltageControlState(long handle);

    /** Per (scenario, incremental TVC controller): the tap index the driver's TransformerVoltageControl
     *  phase left — null without the table. Layout {@code s * nIncrementalTransformerVoltageControlControllerBranches + c}. */
    public static native int[] readBatchIncrementalTransformerVoltageControlTaps(long handle);

    /** Install the DISCRETE transformer voltage-control tables for the in-kernel TransformerVoltageControl
     *  phase (OLF order 8): {@code mode} 0 = AFTER_GENERATOR (TransformerVoltageControlOuterLoop), 1 =
     *  WITH_GENERATOR (SimpleTransformerVoltageControlOuterLoop); {@code entryOn} = the uploaded activation
     *  already carries the control ON (a base load flow whose network OLF's initialize switched on). Groups
     *  are controlled buses (CSR over members) with their toggle-diffed ON/OFF row deltas (CSR); see
     *  GpuTransformerVoltageControlOuterLoop.uploadToDevice. Returns 1 when the kernel loop owns them. */
    @SuppressWarnings("java:S107")
    public static native int setBatchTransformerVoltageControl(long handle, int mode, int entryOn,
            int[] groupMemberStart, int[] groupControlledBusVRow, double[] groupTargetV, double[] groupHalfTargetDeadband, int[] groupUseInitialTapPosition,
            int[] deltaStart, int[] deltaRow, int[] deltaModeOn, int[] deltaModeOff, double[] deltaTargetOn, double[] deltaTargetOff,
            int[] memberRhoRow, int[] memberElement, int[] memberBaseTap, int[] memberTapStart, double[] memberMinR1, double[] memberMaxR1, double[] memberInitialR1,
            double[] tapRows);

    /** fixTransformerVoltageControls' per-(scenario, member) connectivity masks over the
     *  {@code fixPvRow} universe, {@code words} ints each: {@code minMask} cuts every controller branch
     *  (a subset of OLF's component), {@code maxMask} cuts none (a superset). A PV bus in min means keep
     *  the control, no PV bus in max means disable it, anything between routes the scenario to the CPU.
     *  Null clears the channel. */
    public static native void setBatchTransformerVoltageControlFixMasks(long handle, int[] minMask, int[] maxMask, int words);

    /** GeneratorVoltageControlManager tables for the AFTER_GENERATOR phase (after setBatchTransformerVoltageControl): the
     *  freeze candidates (controller-bus V row, load Q, and the reactive-limit entry of a REMOTE
     *  controller or −1) and the UNIVERSE of
     *  fixTransformerVoltageControls — the {@code BUS_V} row of every generator-voltage-controlled bus,
     *  which the per-scenario masks index. Returns 1 when installed. */
    public static native int setBatchGeneratorVoltageControlFreeze(long handle, int[] generatorVoltageControlFreezeVRow,
            double[] generatorVoltageControlFreezeLoadTargetQ, int[] generatorVoltageControlFreezeReactiveLimitEntry,
                                                      int[] fixPvRow);

    /** Upload incremental TVC controller data for the in-kernel outer loop. rhoRow/vRow/elemIndex: per-controller
     *  int arrays; targetV/halfTargetDeadband: per-controller doubles; tapStart: CSR pointers [nControllers+1] into tapRows; tapRows:
     *  flat [totalTaps*8] (y,g1,g2,b1,b2,ksi,r1,a1); baseTap: per-controller initial tap index; groupControllerStart/groupController:
     *  CSR grouping of controllers sharing a controlled bus (for the one-tap-at-a-time round-robin).
     *  Returns 1 when the in-kernel TVC loop owns these controllers, 0 when it refused them (a
     *  controlled bus shared by several transformers stays on the host orchestrator). */
    public static native int setBatchIncrementalTransformerVoltageControl(long handle,
            int[] rhoRow, int[] vRow, double[] targetV, double[] halfTargetDeadband, int[] elemIndex,
            int[] tapStart, double[] tapRows, int[] baseTap, int[] groupControllerStart, int[] groupController,
            int maxTapShift);
}
