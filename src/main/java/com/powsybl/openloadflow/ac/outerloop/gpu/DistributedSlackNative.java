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
 * The device's distributed-slack tables and per-scenario overrides (DistributedSlackOuterLoop).
 *
 * <p>JNI glue: {@code native/cuda/jni/jni/distributed_slack_jni.cu}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class DistributedSlackNative {

    static {
        GpuAcNewtonSolver.isAvailable();   // loads libolfgpu once
    }

    private DistributedSlackNative() {
    }

    /**
     * Install the PER-GENERATOR distributed-slack participants: the factors and limits OLF's
     * water-fill actually saturates on, grouped by bus ({@code distributedSlackBusGeneratorStart}/{@code distributedSlackBusGeneratorEnd} slice
     * the generator arrays, so a bus's row value is the sum over its slice).
     *
     * <p>OLF's {@code ParticipatingElement} wraps an {@code LfGenerator} and {@code step.run} clamps
     * each at its own {@code minTargetP}/{@code maxTargetP} and removes it, so a bus-level aggregate
     * reproduces OLF only while nothing saturates.
     */
    public static native void setBatchDistributedSlackGenerators(long handle, int[] genBus, double[] genFactor,
                                                   double[] genMaxDelta, double[] genMinDelta,
                                                   double[] genMarginUp, double[] genMarginDown,
                                                   int[] distributedSlackBusGeneratorStart, int[] distributedSlackBusGeneratorEnd);

    /**
     * Install (or clear, with null) the per-scenario distributed-slack participants that a
     * fragmenting contingency strands in another synchronous component, as {@code {slot, participant}}
     * pairs indexing {@code distributedSlackBusPhiRow}.
     *
     * <p>OLF distributes over {@code getSynchronousNetworks().getFirst()}, so a stranded participant
     * is not a participant: the device drops it from the distribution AND renormalizes the survivors'
     * shares — including the slack's retained share and the stopping tolerance — over the set that
     * remains. Doing only the first half leaves the pass retiring with most of the mismatch
     * undistributed.
     */
    public static native void setBatchDistributedSlackIslanded(long handle, int[] slotParticipantPairs);

    /** The generators an outage REMOVES, as (scenario, DS generator slot) pairs. With the identity the
     *  outaged bus's survivors take part in the per-generator slack water-fill like any others, instead
     *  of the bus riding on one aggregate entry — which leaves their own P unknown to everything that
     *  reads it, the reactive-capability curve channel above all. */
    public static native void setBatchDistributedSlackLostGenerators(long handle, int[] slotGen);

    /** The participants a strategy's GeneratorAction takes out of the slack distribution (OLF's
     *  reApplyActivePowerControlChecks), as (scenario, DS generator slot) pairs: they keep the action's target and
     *  take no share. Null clears the channel. */
    public static native void setBatchDistributedSlackRemovedGenerators(long handle, int[] slotGen);

    /**
     * Install (or clear, with null) the per-scenario P-balance row of a distributed-slack participant
     * whose row THIS scenario moves, as {@code {slot, participant, row}} triples ({@code participant}
     * indexes {@code distributedSlackBusPhiRow}, {@code row} is an equation row in the same space as {@code distributedSlackBusPhiRow}).
     *
     * <p>{@code distributedSlackBusPhiRow} is structural, and under the coupler-tree relocation a child bus's P balance
     * sits on its parent coupler's {@code DUMMY_P} row. A contingency that OPENS that coupler pins the
     * dummy row to a {@code DUMMY_TARGET} identity and the stock-placement fallback returns the balance
     * to the child's own {@code BUS_PHI} row — so for that scenario the participant's row is the
     * child's, and the distribution must read its anchor and write its share there. Writing to the
     * pinned row instead loses the share AND feeds the retain/water-fill totals a spurious
     * {@code (0 - anchor)} of absorbed power, which makes the slack loop redistribute what OLF placed
     * once.
     */
    public static native void setBatchDistributedSlackBusRowOverrides(long handle, int[] slotParticipantRowTriples);

    /** The slack bus's participating generators' total rooms (GpuAcData.slackBusRoomUp/Down): the bound of a
     *  slack-only distribution, beyond which the device flags OLF's remaining mismatch. */
    public static native void setBatchSlackBusRooms(long handle, double roomUp, double roomDown);

    /** Per slot, the row the distributed-slack kernels read the slack's balance on: the angle row of the bus OLF
     *  re-picks as the slack when a contingency isolates the base one, -1 = the base slack row. Null clears. */
    public static native void setBatchSlackBusRows(long handle, int[] rows);

    /** Per scenario and participating generator ({@code s * nDistributedSlackGenerators + g}, pu) for the following solves: {@code anchor},
     *  its move from its base target P to where the slack's water-fill starts it (a strategy pair: the parent's final
     *  target, or a GeneratorAction's value; null = the base); {@code start}, its move from that anchor when the solve
     *  starts (a contingency's loss pre-distribution; null = none). The curve and forced-target-Q channels evaluate a
     *  generator at anchor + its water-fill move once a slack pass ran, at anchor + start before. {@code seedAnchor}
     *  (strategy pairs): the anchor without the strategy's GeneratorActions, the P the pins continued from the parent
     *  are stamped at (OLF keeps their restored target Q until its reactive-limit check); null = the anchor. */
    public static native void setBatchDistributedSlackGeneratorAnchors(long handle, double[] anchor, double[] start, double[] seedAnchor);

    /** Slack-only distribution: per scenario, the slack generators' initial target relative to the slack's target P
     *  for the following solves - a strategy pair's parent total, which OLF re-anchors on; null = 0 (a contingency,
     *  whose loss pre-distribution OLF undoes at its first pass). */
    public static native void setBatchSlackOnlyAnchor(long handle, double[] anchor);
}
