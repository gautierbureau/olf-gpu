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
 * The device's reactive-limit tables and per-scenario overrides (ReactiveLimitsOuterLoop): the pin entries,
 * the capability curves, forced target Q, the realistic-voltage check and the remote member pins.
 *
 * <p>JNI glue: {@code native/cuda/jni/jni/reactive_limits_jni.cu}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class ReactiveLimitsNative {

    static {
        GpuAcNewtonSolver.isAvailable();   // loads libolfgpu once
    }

    private ReactiveLimitsNative() {
    }

    /** Per (scenario, reactive-limit entry): the pinned target the last solve ended with, NaN when not pinned. */
    public static native double[] readBatchRlPinTargets(long handle);

    /** Install (or clear, with nulls) the forceTargetQInReactiveLimits channel on a batch context: the Q
     *  targets of generators OLF clamps into their curve limits at their current P (see ForceTargetQInReactiveLimits). */
    public static native void setBatchForcedTargetQ(long handle, int[] fqIdx, double[] fqIn, double[] fqPts,
                                                    int extrapolate);

    /** Per-chunk loss pre-distribution moves of the forced-target-Q generators ([nS * nfq] pu), or null. */
    public static native void setBatchForcedTargetQPre(long handle, double[] fqPre);

    /** {@code AcloadFlowEngine.isStateUnrealistic} on the driver's INTERMEDIATE states: per row, 1 = a
     *  non-fictitious bus's {@code BUS_V} at or above {@code minNominalVoltageRealisticVoltageCheck},
     *  plus the realistic band. A scenario that ever leaves the band goes to the CPU replay.
     *  {@code enabled = 0} clears the channel on a cached context. */
    public static native void setBatchRealisticCheck(long handle, byte[] rowChecked, double minV, double maxV,
                                                     int enabled, int fixingLoop);

    /** OLF's own base pin kind per RL entry (0 none, 1 MIN_Q, 2 MAX_Q, 3 MIN_REALISTIC_V,
     *  4 MAX_REALISTIC_V); null clears it on a cached context and the device falls back to its
     *  nearest-limit guess. */
    public static native void setBatchRlBaseLimitType(long handle, byte[] rlBaseQLimitType);

    /** Remote controllers OLF re-merges onto a NEW pilot after a zero-impedance split (GpuBatchQualification
     *  .reHomedControls): per (slot, entry) pair the V row the entry's reactive-limit check compares against the
     *  group target instead of the base pilot's. The target itself rides the per-scenario V-target overrides of
     *  the solve. Null clears the context's list. */
    public static native void setBatchRlCmpOverrides(long handle, int[] slotEntryPairs, int[] cmpVRows);

    /** Install (or clear, with nulls) the reactive-capability-CURVE channel on a batch context —
     *  per-scenario dynamic RL limits evaluated at the current targets (see ReactiveCapabilityCurves). */
    public static native void setBatchRlCurves(long handle, int[] cvIdx, double[] cvIn, double[] cvPts,
                                               int[] cvHv, int extrapolate);

    /** OLF's GLOBAL forced-PV election keys per RL entry (ReactiveLimitsOuterLoop.switchPvPq: when every
     *  PV bus would switch, the strongest stays PV — highest controlled-bus nominal V, then highest bus
     *  target P, then lowest id): the nominal V, the bus's BUS_TARGET_P row (−1 = none, use staticP)
     *  and the id rank. Installed on every batch context; the RL pass then runs the election in-kernel. */
    public static native void setBatchRlElection(long handle, double[] nominalV, int[] pRow, double[] staticTargetP, int[] idRank);

    /** (scenario, reactive-limit entry) pairs whose bus lost EVERY generator this scenario: no controller there
     *  (OLF has no voltage control at the bus), so the entry is seeded FROZEN — skipped by the limit check, read
     *  as "not enabled" by the pilot-self layout, and its reactive balance emitted through the pilot-self gate. */
    public static native void setBatchRlGenLost(long handle, int[] slotEntryPairs);

    /**
     * Install (or clear, with null) the per-scenario reactive-limits controllers a fragmenting
     * contingency strands in another synchronous component, as {@code {slot, controller}} pairs.
     *
     * <p>OLF disables those buses, so its ReactiveLimits loop never sees them. The device holds their
     * rows frozen at a flat 1&#8736;0, so the q it would read is a placeholder — switching on it pins a
     * PV bus OLF leaves alone and perturbs the main component through the controller's group.
     */
    public static native void setBatchRlIslanded(long handle, int[] slotControllerPairs);

    /** Re-homed voltage controls (ControlReHoming.reHomedControls): per (slot, entry) pair the entry is
     *  checked by the reactive-limits pass as a LOCAL controller at {@code vTargets[k]} — the bus is a plain PV
     *  bus this scenario, whatever its base packing. Null clears the context's list. */
    public static native void setBatchRlLocal(long handle, int[] slotEntryPairs, double[] vTargets);

    /** (scenario, reactive-limit entry) pairs whose UNPIN the device cannot host (a pinned member of a re-homed
     *  cluster with no enabled controller): the kernel rejects the scenario instead of switching (CPU replay). */
    public static native void setBatchRlNoUnpin(long handle, int[] slotEntryPairs);

    /** Per (scenario, reactive-limit entry): the pinned target a continued solve restores (a strategy pair takes its
     *  parent's, as OLF restores the parent's target Q), NaN = stamp the pin at its limit. Null clears it. */
    public static native void setBatchRlPinTargets(long handle, double[] targets);

    /** Remote-member RL entries (plan-B1): per-entry remote kinds + compare rows (PILOT v column for
     *  remote group members — both RL switch conditions test the pilot's V against the GROUP target).
     *  {@code dqMemberQrGroup} (plan-B4) links each group member to its pilot's relocation candidate
     *  group (−1 = none) so the kernels suppress the CURRENT carrier's DISTR_Q row and hand its
     *  target the pilot's Q when the free role re-picks. Nulls clear. */
    public static native void setBatchRlRemote(long handle, byte[] rlRemote, int[] rlCmpVrow,
                                               int[] rmqRow, int[] rmqElem, int[] rmqSide, int[] rmqKind,
                                               double[] rmqWeight,
                                               int[] dqKMember, int[] dqJMember, double[] dqMemberKey,
                                               int[] dqMemberRlEntry, int[] dqMemberGroupStart,
                                               int[] dqMemberGroupCount, int[] dqMemberRow, byte[] dqMemberHasRow,
                                               int[] dqMemberQrGroup, double[] dqMemberTargetQ,
                                               int[] dqMemberTgtPtr, int[] dqMemberTgtPeer);

    /** {@code voltageRemoteControlRobustMode} (OLF's default): per RL entry, the initial generation
     *  target Q a realistic-V freeze pins at, plus the two thresholds the controller's OWN voltage is
     *  tested against ({@code minRealisticVoltage * 1.02} and {@code maxRealisticVoltage / 1.02}).
     *  {@code robust = 0} clears the channel on a cached context. */
    public static native void setBatchRlRobust(long handle, double[] rlInitTargetQ, double vLo, double vHi, int robust);

    /** Install the voltage-dependent SVC reactive-limit coefficients on a batch context:
     *  per RL entry, limit(v) = packed base limit + svcB*(v^2 - v2Base) — exactly OLF's live
     *  Bmin/Bmax*(v*nomV)^2 evaluation (and the pinned re-freeze via busesWithUpdatedQLimits). */
    public static native void setBatchRlSvcLimits(long handle, double[] svcBmin, double[] svcBmax,
                                                  double[] v2Base);

    /** Un-pin reactive-limit entries at a driver pause (SecondaryVoltageControl's re-enable step):
     *  {@code pairs} is (scenario, RL entry) flattened, {@code newTargets} the voltage target each row
     *  returns to. A resume skips the RL seeding, so this is the only way a host-side un-pin reaches
     *  the device. No-op for an entry that is not pinned. */
    public static native void setBatchRlUnpin(long handle, int[] pairs, double[] newTargets);
}
