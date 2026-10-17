/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

/**
 * The seeding of a slot whose contingency isolates the base slack: OLF re-picks a slack in the main component, whose
 * angle becomes the reference and whose scheduled P the distributed slack reads from its shifted anchor.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class SlackRelocationSeed {

    private SlackRelocationSeed() {
    }

    /** Seed slot {@code s} (contingency index {@code ci}) with the slack OLF re-picked, if any. */
    static void seed(SlotSeeding ctx, int s, int ci) {
        final int[] baseRowMode = ctx.baseRowMode();
        final int chunkSize = ctx.chunkSize();
        final ChunkLists cl = ctx.cl();
        final int n = ctx.n();
        final RunScratch rs = ctx.rs();
        final GpuBatchQualification.SlackRelocations slackRelocations = ctx.plan().slackRelocations();
        // The outage isolates the base slack and OLF re-picked one in the main component: its P
        // balance leaves the system and its angle becomes the reference (BUS_TARGET_PHI, target 0).
        // The state starts from the base angles, as OLF's does - the first step shifts them all.
        int reference = slackRelocations.row(ci);
        if (reference >= 0) {
            int[] row = rs.rowModePerScenario[s] == baseRowMode ? baseRowMode.clone() : rs.rowModePerScenario[s];
            row[reference] = 0;
            rs.rowModePerScenario[s] = row;
            rs.overrides.add(s * n + reference, 0.0);
            if (cl.dsSlackRows == null) {
                cl.dsSlackRows = new int[chunkSize];
                java.util.Arrays.fill(cl.dsSlackRows, -1);
            }
            cl.dsSlackRows[s] = reference;
            if (cl.outageSlackShift == null) {
                cl.outageSlackShift = new double[chunkSize];
            }
            cl.outageSlackShift[s] += slackRelocations.shiftByIndex()[ci];
        }
    }
}
