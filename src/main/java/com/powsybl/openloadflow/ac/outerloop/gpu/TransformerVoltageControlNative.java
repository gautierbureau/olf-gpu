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
 * <p>JNI glue: {@code native/cuda/jni/jni/transformer_voltage_control_jni.cu}.
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
     *  layout {@code (s * nFz + c) * 2} — null without candidates. */
    public static native double[] readBatchDiscreteTvcFreeze(long handle);

    /** Per (scenario, member): {state (0 off, 1 continuous, 2 fixed), tap index, the r1 fed to the rounding},
     *  layout {@code (s * nMembers + m) * 3} — null without the table. */
    public static native double[] readBatchDiscreteTvcState(long handle);

    /** Per (scenario, incremental TVC controller): the tap index the driver's TransformerVoltageControl
     *  phase left — null without the table. Layout {@code s * nTvc + c}. */
    public static native int[] readBatchTvcTaps(long handle);

    /** Install the DISCRETE transformer voltage-control tables for the in-kernel TransformerVoltageControl
     *  phase (OLF order 8): {@code mode} 0 = AFTER_GENERATOR (TransformerVoltageControlOuterLoop), 1 =
     *  WITH_GENERATOR (SimpleTransformerVoltageControlOuterLoop); {@code entryOn} = the uploaded activation
     *  already carries the control ON (a base load flow whose network OLF's initialize switched on). Groups
     *  are controlled buses (CSR over members) with their toggle-diffed ON/OFF row deltas (CSR); see
     *  DiscreteTransformerVoltageControl.uploadToDevice. Returns 1 when the kernel loop owns them. */
    @SuppressWarnings("java:S107")
    public static native int setBatchDiscreteTvc(long handle, int mode, int entryOn,
            int[] grpPtr, int[] grpVRow, double[] grpTargetV, double[] grpHalfDb, int[] grpUseInit,
            int[] deltaPtr, int[] deltaRow, int[] deltaModeOn, int[] deltaModeOff, double[] deltaTargetOn, double[] deltaTargetOff,
            int[] memRhoRow, int[] memElem, int[] memBaseTap, int[] memTapPtr, double[] memMin, double[] memMax, double[] memInit,
            double[] tapData);

    /** fixTransformerVoltageControls' per-(scenario, member) connectivity masks over the
     *  {@code fixPvRow} universe, {@code words} ints each: {@code minMask} cuts every controller branch
     *  (a subset of OLF's component), {@code maxMask} cuts none (a superset). A PV bus in min means keep
     *  the control, no PV bus in max means disable it, anything between routes the scenario to the CPU.
     *  Null clears the channel. */
    public static native void setBatchDiscreteTvcFixMasks(long handle, int[] minMask, int[] maxMask, int words);

    /** GeneratorVoltageControlManager tables for the AFTER_GENERATOR phase (after setBatchDiscreteTvc): the
     *  freeze candidates (controller-bus V row, load Q, and the reactive-limit entry of a REMOTE
     *  controller or −1) and the UNIVERSE of
     *  fixTransformerVoltageControls — the {@code BUS_V} row of every generator-voltage-controlled bus,
     *  which the per-scenario masks index. Returns 1 when installed. */
    public static native int setBatchDiscreteTvcFreeze(long handle, int[] fzVRow, double[] fzLoadQ, int[] fzRlEntry,
                                                      int[] fixPvRow);

    /** Upload incremental TVC controller data for the in-kernel outer loop. rhoRow/vRow/elemIndex: per-controller
     *  int arrays; targetV/halfDb: per-controller doubles; tapPtr: CSR pointers [nCtrl+1] into tapData; tapData:
     *  flat [totalTaps*8] (y,g1,g2,b1,b2,ksi,r1,a1); baseTap: per-controller initial tap index; groupPtr/groupCtrl:
     *  CSR grouping of controllers sharing a controlled bus (for the one-tap-at-a-time round-robin).
     *  Returns 1 when the in-kernel TVC loop owns these controllers, 0 when it refused them (a
     *  controlled bus shared by several transformers stays on the host orchestrator). */
    public static native int setBatchTvcControllers(long handle,
            int[] rhoRow, int[] vRow, double[] targetV, double[] halfDb, int[] elemIndex,
            int[] tapPtr, double[] tapData, int[] baseTap, int[] groupPtr, int[] groupCtrl,
            int maxTapShift);
}
