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
 * The controllers of the remote generator voltage controls the DISTR_Q terms refer to, grouped by control: per
 * member, its reactive {@code key} (OLF's qPct numerator), its reactive limits {@code reactiveLimitEntry}, its group's
 * {@code groupStart} / {@code groupCount}, its DISTR_Q {@code row} and whether it has one ({@code hasRow}), its
 * relocation candidate group ({@code relocationGroup}), its {@code targetQ} and {@code busNum}; and, as CSR
 * ({@code targetPeerStart} / {@code targetPeer}), the peers whose targetQ (frozen once switched to PQ) enters its DISTR_Q
 * target.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record DistrQMembers(double[] key, int[] reactiveLimitEntry, int[] groupStart, int[] groupCount, int[] row,
                            byte[] hasRow, int[] relocationGroup, double[] targetQ, int[] busNum,
                            int[] targetPeerStart, int[] targetPeer) {

    public static DistrQMembers empty() {
        return new DistrQMembers(new double[0], new int[0], new int[0], new int[0], new int[0], new byte[0], new int[0],
                new double[0], new int[0], new int[0], new int[0]);
    }

    /**
     * What a cached device context depends on. targetPeerStart and targetPeer are not compared (as before the grouping;
     * recorded in the cleanup plan, step 6).
     */
    public boolean sameAs(DistrQMembers other) {
        return Arrays.equals(key, other.key) && Arrays.equals(reactiveLimitEntry, other.reactiveLimitEntry)
                && Arrays.equals(groupStart, other.groupStart) && Arrays.equals(groupCount, other.groupCount)
                && Arrays.equals(row, other.row) && Arrays.equals(hasRow, other.hasRow)
                && Arrays.equals(relocationGroup, other.relocationGroup) && Arrays.equals(targetQ, other.targetQ)
                && Arrays.equals(busNum, other.busNum);
    }
}
