/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.openloadflow.network.LfBranch;

import java.util.Set;

/**
 * How one scenario differs, for the violation screen, from the base its limit table was built on: the branches it takes
 * out, the base-open branches it closes, the branches it opens at one side, and the pi rows it evaluates closed branches
 * with. It is the scenario the device solved: the slot seeding's outages, reclosings, side openings and tap actions, then
 * the taps the device's outer loops moved and the branches its automation system switched.
 *
 * <p>{@code branches} are limit-table branch indices, ascending, one entry each; {@code kinds} their {@link #OUT},
 * {@link #IN}, {@link #ROW}, {@link #OPEN_1} and {@link #OPEN_2} bits; {@code rows} eight doubles per entry
 * ({@code y g1 g2 b1 b2 ksi r1 a1}, zero unless {@link #ROW}). {@code covered} are the branches whose scenario the entries
 * account for: the host need not scan them in full.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
record GpuScreenScenario(int[] branches, byte[] kinds, double[] rows, Set<LfBranch> covered) {

    /** The branch is out of service in this scenario: no candidate. */
    static final byte OUT = 1;
    /** A base-open branch this scenario closes: screened like a closed one. */
    static final byte IN = 2;
    /** Screened with this entry's pi row instead of the pack's (a ratio or phase shift that is a state variable still
     *  comes from the state, as in the solve's fill). */
    static final byte ROW = 4;
    /** Opened at side 1 (resp. 2): screened at the other side with OLF's open-branch formula, as OLF's open terms
     *  evaluate it, from the pack's row or the entry's. The device solves the opening with a dangling row (its
     *  equivalent shunt), whose active power loses relative precision to a cancellation: measured 3.9e-6 against the
     *  open terms on network.xiidm, above the band. */
    static final byte OPEN_1 = 8;
    static final byte OPEN_2 = 16;

    /** A scenario that changes nothing: the base topology with the base parameters. */
    static final GpuScreenScenario BASE = new GpuScreenScenario(new int[0], new byte[0], new double[0], Set.of());

    /** The entry of table branch {@code b}, or -1. */
    int entryOf(int b) {
        int k = java.util.Arrays.binarySearch(branches, b);
        return k >= 0 ? k : -1;
    }
}
