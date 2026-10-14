/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import org.junit.jupiter.api.Test;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The two gates of the coupler-opening path. A plan entry that only writes a TARGET at the killed coupler's child's
 * row (a merged group's DISTR_Q seed brought home onto the elected main's own row, a Q override, an injection target)
 * makes the structural default targets yield, and nothing else; only an entry that changes the row's MODE (a flip to
 * the bus's own reactive balance, a killed pilot's identity) also withholds the voltage identity the child holds for
 * its cluster. Measured when the seed withheld it: rte6515 BUS-6066_BBS_1 6.7 kV and 1.27 MW against OLF,
 * BUS-2341_BBS_1 2.3 kV, BUS-246_BBS_2 singular.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuBatchChunkSolverGateTest {

    private static final int ROW = 13328;

    private static GenOutage entry(int pRow, int vRow, boolean toPq, boolean qOverride, int[] dqRows, int deadPilotVRow) {
        return new GenOutage(pRow, 0, vRow, 0, toPq, qOverride, -1, 0, 0, 0, 0, -1, 0, 0, false, Set.of(), 0, 0, 0,
                dqRows.length > 0 ? 0 : -1, new int[0], dqRows, new double[dqRows.length], 0,
                deadPilotVRow, 0, -1, 0, -1, 0, -1, -1, 0, false, -1, 0, 0);
    }

    @Test
    void aDistrQSeedWritesTheRowButDoesNotFlipIt() {
        GenOutage[] seed = {entry(-1, -1, false, false, new int[] {ROW, 14433}, -1)};
        assertTrue(seed[0].seedsDqTargets());
        assertTrue(GpuBatchChunkSolver.planSeedsRow(seed, ROW), "the default Q target yields to the seed");
        assertFalse(GpuBatchChunkSolver.planFlipsRow(seed, ROW), "the voltage identity stays");
        assertFalse(GpuBatchChunkSolver.planSeedsRow(seed, ROW + 1));
    }

    @Test
    void aFlipOrAKilledPilotWithholdsTheIdentity() {
        GenOutage[] flip = {entry(-1, ROW, true, true, new int[0], -1)};
        assertTrue(GpuBatchChunkSolver.planFlipsRow(flip, ROW));
        assertTrue(GpuBatchChunkSolver.planSeedsRow(flip, ROW));
        GenOutage[] dead = {entry(-1, -1, false, false, new int[0], ROW)};
        assertTrue(dead[0].killsPilotVoltageIdentity());
        assertTrue(GpuBatchChunkSolver.planFlipsRow(dead, ROW));
        assertFalse(GpuBatchChunkSolver.planFlipsRow(dead, ROW + 1));
    }

    @Test
    void anInjectionTargetYieldsThePDefaultOnly() {
        GenOutage[] load = {entry(7945, 7944, false, true, new int[0], -1)};   // a lost load brought home (rowsHome)
        assertTrue(GpuBatchChunkSolver.planSeedsRow(load, 7945), "the P default yields");
        assertTrue(GpuBatchChunkSolver.planSeedsRow(load, 7944), "the Q default yields to the Q override");
        assertFalse(GpuBatchChunkSolver.planFlipsRow(load, 7944), "a PQ bus's Q override is not a flip");
        assertFalse(GpuBatchChunkSolver.planSeedsRow(null, 7945));
    }
}
