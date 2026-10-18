/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

/**
 * The remote voltage control (DISTR_Q) tables and per-scenario overrides.
 *
 * <p>JNI glue: {@code native/cuda/jni/jni/remote_voltage_control_jni.cu}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class RemoteVoltageControlNative {

    static {
        GpuAcNewtonSolver.isAvailable();   // loads libolfgpu once
    }

    private RemoteVoltageControlNative() {
    }

    /** Install (or clear, with nulls) per-scenario DISTR_Q weight overrides on a batch context —
     *  {@code idx} = flat {scenario, contribution} pairs, {@code w} = the renormalized weight (a gen
     *  outage at a remote-VC controller bus re-runs updateReactiveKeys on the CPU). Per-chunk state. */
    /** {@code {slot, dqMember}} pairs: DISTR_Q controllers a scenario DEACTIVATES (their bus lost its
     *  last voltage generator while the group survives). Context state, not a solve parameter — a
     *  monitoring or secondary-VC re-solve on the same context must not silently drop it, which is
     *  exactly what happened while this was a parameter: the re-solve passed null and
     *  updateDistrQTargetsBatch recomputed the DISTR_Q target with the dead controller's key back in
     *  the denominator, clobbering the right target with the base one. Cleared when the chunk has none. */
    public static native void setBatchDqInactiveMembers(long handle, int[] slotMember);

    /** Per-scenario member reactive keys: {@code idx} = flat {scenario, member} pairs, {@code key} = OLF's
     *  qPct after the scenario's generator losses (updateReactiveKeys); the DISTR_Q fill and the target kernel
     *  renormalize them over the members a pin leaves enabled. Per-chunk state; nulls clear it. */
    public static native void setBatchDqKeyOverrides(long handle, int[] idx, double[] key);

    /** Per-scenario member targetQ overrides for the DISTR_Q target kernel, as {@code key = slot *
     *  nDqMembers + memberIndex} with the member's post-contingency targetQ. Needed because that kernel
     *  RECOMPUTES the group's row targets from the base member targetQ on every reactive-limits pass, so
     *  a host-side row seed alone is put straight back. Nulls clear. */
    public static native void setBatchDqMemberTargetQOverrides(long handle, int[] keys, double[] values);

    /** tasks #71: per (scenario, dq member) SUBGROUP labels of a group a contingency split behind its surviving
     *  main (dense {@code chunkSize x nDqMembers} bytes, 0 = the base pilot's group) and the qr group -> first
     *  dq member map the kernels use to pair candidates with members. Null clears the channel. */
    public static native void setBatchDqSubgroups(long handle, byte[] dense, int[] qrMemberStart);

    public static native void setBatchDqWeightOverrides(long handle, int[] idx, double[] w);

    /** Install (or clear, with nulls) per-scenario DISTR_RHO coefficient overrides on a batch
     *  context — {@code idx} = flat {scenario, contribution} pairs, {@code c} = the renormalized
     *  1/count' coefficient (a shared-TVC group member outaged in that scenario). Per-chunk state. */
    public static native void setBatchDrCoefOverrides(long handle, int[] idx, double[] c);

    /** Pilot-row flip entries: one per (scenario, pilot) where the outage made the pilot stop being one
     *  of its OWN controllers. While the survivors hold its voltage the pilot keeps BUS_TARGET_V and its
     *  acquired reactive balance rides one of THEIR rows; when every survivor pins, those rows take their
     *  own pinned balances and the pilot's is evicted. The kernel then puts it on the pilot's freed row,
     *  as OLF does. {@code rlCounts} sizes each entry's slice of {@code rlPool} (the surviving
     *  controllers' reactive-limit entries). Nulls clear. */
    public static native void setBatchPilotAllPinnedFlip(long handle, int[] slots, int[] rows,
                                                         double[] qTargets, int[] rlCounts, int[] rlPool);

    /** Per-scenario moves of a remote-control pilot's own reactive target ({key = slot * nQr + group, delta pu}, one
     *  entry per key): added to the value captured from the base carrier and to the first fill's carrier target.
     *  Nulls clear. */
    public static native void setBatchPilotQDeltas(long handle, int[] keys, double[] deltas);
}
