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
 * The reactive limits loop's entries: one per generator voltage controller bus (a local controller, a remote member,
 * a pilot-self pilot) that OLF's ReactiveLimitsOuterLoop may switch PV -> PQ. Per entry:
 * <ul>
 *   <li>{@code voltageRow}: its stable BUS_V row, where the switch writes (GpuRowMode FIXED_VARIABLE <-> BUS_TARGET_Q,
 *       target {@code targetV} <-> the limit - {@code loadQ}); {@code comparedVoltageRow}: the row whose voltage the
 *       switch conditions compare (the pilot's, for a remote member);</li>
 *   <li>{@code minQ} / {@code maxQ}: its reactive limits (pu); {@code loadQ}: its load target Q; {@code targetV}: its
 *       controlled voltage target; {@code initialTargetQ}: its generation target Q before any pin;</li>
 *   <li>{@code svcMinB} / {@code svcMaxB} / {@code baseVSquared}: a static var compensator's susceptance limits and
 *       base voltage squared, for its live limits {@code B * V^2};</li>
 *   <li>{@code flags}: GpuReactiveLimitsEntryFlag bits; {@code voltageControlRole}: a GpuVoltageControlRole;</li>
 *   <li>{@code busNum}, {@code nominalV}, {@code activePowerRow}, {@code targetP}: the bus, its nominal voltage, the row
 *       of its BUS_TARGET_P and its target P (reactive capability curves are read at the live P);
 *       {@code idRank}: the bus id's rank, OLF's BY_ID_COMPARATOR tie-break;</li>
 *   <li>{@code baseQLimitType}: OLF's LfBus.getQLimitType() after the base load flow, as a GpuQLimitType.</li>
 * </ul>
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record ReactiveLimitEntries(int[] voltageRow, double[] minQ, double[] maxQ, double[] loadQ, double[] targetV,
                                   double[] svcMinB, double[] svcMaxB, double[] baseVSquared,
                                   byte[] flags, byte[] voltageControlRole, int[] comparedVoltageRow, int[] busNum,
                                   double[] nominalV, int[] activePowerRow, double[] targetP, int[] idRank,
                                   double[] initialTargetQ, byte[] baseQLimitType) {

    public static ReactiveLimitEntries empty() {
        return new ReactiveLimitEntries(new int[0], new double[0], new double[0], new double[0], new double[0],
                new double[0], new double[0], new double[0], new byte[0], new byte[0], new int[0], new int[0],
                new double[0], new int[0], new double[0], new int[0], new double[0], new byte[0]);
    }

    public int size() {
        return voltageRow.length;
    }

    /**
     * What a cached device context depends on. busNum, nominalV, activePowerRow, targetP and idRank are not compared
     * (as before the grouping; whether they should be is recorded in the cleanup plan, step 6).
     */
    public boolean sameAs(ReactiveLimitEntries other) {
        return Arrays.equals(voltageRow, other.voltageRow) && Arrays.equals(minQ, other.minQ)
                && Arrays.equals(maxQ, other.maxQ) && Arrays.equals(loadQ, other.loadQ)
                && Arrays.equals(targetV, other.targetV)
                && Arrays.equals(svcMinB, other.svcMinB) && Arrays.equals(svcMaxB, other.svcMaxB)
                && Arrays.equals(baseVSquared, other.baseVSquared)
                && Arrays.equals(flags, other.flags) && Arrays.equals(voltageControlRole, other.voltageControlRole)
                && Arrays.equals(baseQLimitType, other.baseQLimitType)
                && Arrays.equals(comparedVoltageRow, other.comparedVoltageRow)
                && Arrays.equals(initialTargetQ, other.initialTargetQ);
    }
}
