/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import java.util.Arrays;

/**
 * What the device's distributed slack passes distribute over, as OLF's ActivePowerDistribution would:
 * <ul>
 *   <li>per participating bus ({@code bus*}): its BUS_PHI row, raw factor, base target P, the room left up and down
 *       ({@code busMaxDelta} = maxTargetP - targetP under OLF's no-sign-change rule, {@code busMinDelta}), the
 *       REMAINING_MARGIN margins, its LfBus number, and the slice of its generators ({@code busGeneratorStart} /
 *       {@code busGeneratorEnd}: generators are grouped by bus and contiguous);</li>
 *   <li>the slack bus's own share ({@code slack*}): its BUS_PHI row, target P, REMAINING_MARGIN margins, and its
 *       participating generators' total rooms from their anchor targets (what a slack-only distribution can place
 *       before OLF's water-fill leaves a remaining mismatch; infinite for a load-based slack);</li>
 *   <li>per participating generator ({@code generator*}): the index of its bus, raw factor, rooms, margins, id;</li>
 *   <li>{@code factorMode}: a {@link com.powsybl.openloadflow.ac.solver.gpu.codes.GpuDistributedSlackFactorMode}.</li>
 * </ul>
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record DistributedSlackParticipants(int[] busPhiRow, double[] busFactor, int slackPhiRow, double slackTargetP,
                                           double[] busBaseTargetP, double[] busMaxDelta, double[] busMinDelta,
                                           int factorMode, double[] busMarginUp, double[] busMarginDown,
                                           double slackMarginUp, double slackMarginDown,
                                           double slackRoomUp, double slackRoomDown,
                                           int[] generatorBus, double[] generatorFactor, double[] generatorMaxDelta,
                                           double[] generatorMinDelta, double[] generatorMarginUp,
                                           double[] generatorMarginDown, int[] busGeneratorStart, int[] busGeneratorEnd,
                                           int[] busNum, String[] generatorId) {

    public static DistributedSlackParticipants empty() {
        return new DistributedSlackParticipants(new int[0], new double[0], -1, 0.0, new double[0], new double[0],
                new double[0], 0, new double[0], new double[0], 0.0, 0.0, Double.POSITIVE_INFINITY,
                Double.NEGATIVE_INFINITY, new int[0], new double[0], new double[0], new double[0], new double[0],
                new double[0], new int[0], new int[0], new int[0], new String[0]);
    }

    public int size() {
        return busPhiRow.length;
    }

    /**
     * What a cached batched context depends on: the bus and slack arrays and scalars it is created from (and the
     * slack rooms). The generator arrays and busGeneratorStart / End are installed again from the current extraction
     * before every chunk ({@code DeviceLoopTables.install}); busNum and generatorId stay on the host.
     */
    public boolean sameAs(DistributedSlackParticipants other) {
        return Arrays.equals(busPhiRow, other.busPhiRow) && Arrays.equals(busFactor, other.busFactor)
                && slackPhiRow == other.slackPhiRow && slackTargetP == other.slackTargetP
                && Arrays.equals(busBaseTargetP, other.busBaseTargetP)
                && Arrays.equals(busMaxDelta, other.busMaxDelta) && Arrays.equals(busMinDelta, other.busMinDelta)
                && factorMode == other.factorMode
                && Arrays.equals(busMarginUp, other.busMarginUp) && Arrays.equals(busMarginDown, other.busMarginDown)
                && slackMarginUp == other.slackMarginUp && slackMarginDown == other.slackMarginDown
                && slackRoomUp == other.slackRoomUp && slackRoomDown == other.slackRoomDown;
    }
}
